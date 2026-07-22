import asyncio
import parsl
from academy.agent import Agent, action
from pathlib import Path
from parsl.config import Config

@parsl.python_app
def code_execution_app(
    code_snippet: str,
    workdir: str | None,
    extra_paths: list[str] | None,
    timeout: int=300,
) -> dict[str, str]:
    # A parsl @python_app body runs synchronously inside a worker process, so
    # this must NOT be a coroutine: use the blocking subprocess API and let
    # parsl's executor provide the concurrency (one app per worker). All imports
    # are function-local because parsl ships the body to the worker.
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    # Build PYTHONPATH: agentic_execution package root (parent of the
    # `agentic_execution` package
    # dir, i.e. the `src` root) first, then any skill script dirs, then
    # whatever the agent already had, so generated code can import the
    # project and the skill helpers.
    pkg_root = Path(__file__).resolve().parents[2]  # .../src
    path_parts = [str(pkg_root)]
    if extra_paths:
        path_parts.extend(str(Path(p)) for p in extra_paths)
    existing = os.environ.get('PYTHONPATH')
    if existing:
        path_parts.append(existing)

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(path_parts)

    # Write to a real temp file rather than passing via ``-c`` so tracebacks
    # carry a stable filename and correct line numbers for the agent to read.
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', prefix='qm_exec_', delete=False,
    ) as f:
        f.write(code_snippet)
        script_path = f.name

    try:
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(workdir) if workdir is not None else None,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired carries whatever was captured before the kill.
            stdout = (exc.stdout or b'').decode('utf-8', errors='replace')
            stderr = (exc.stderr or b'').decode('utf-8', errors='replace')
            timeout_msg = f'Execution timed out after {timeout:g}s and was killed.'
            stderr = f'{stderr}\n{timeout_msg}'.strip() if stderr else timeout_msg
            return {'stdout': stdout, 'stderr': stderr, 'returncode': '-1'}

        stdout = proc.stdout.decode('utf-8', errors='replace')
        stderr = proc.stderr.decode('utf-8', errors='replace')
        return {
            'stdout': stdout,
            'stderr': stderr,
            'returncode': str(proc.returncode),
        }
    finally:
        Path(script_path).unlink(missing_ok=True)

class AgenticExecution(Agent):
    def __init__(self,
                 num_threads: int,
                 max_memory: int=160000,
                 parsl_config: Config | None=None):
        super().__init__()
        self.num_threads = num_threads
        self.max_memory = max_memory
        # Loaded in agent_on_startup; defaults to a local CPU config.
        self._parsl_config = parsl_config

    async def agent_on_startup(self) -> None:
        """Load a parsl DataFlowKernel so the workflow apps can execute.

        Without this, the first @python_app call raises NoDataFlowKernelError.
        """
        self.dfk = parsl.load(self._parsl_config)

    async def agent_on_shutdown(self) -> None:
        """Tear the parsl runtime down cleanly on agent shutdown."""
        try:
            self.dfk.cleanup()
        except Exception:
            pass
        parsl.clear()

    @action
    async def execute_code(self,
                           code_snippet: str,
                           workdir: Path | None = None,
                           extra_paths: list[Path] | None = None,
                           timeout: float = 300.0) -> dict[str, str]:
        """Code execution action. Used for performing bespoke analysis or simulation,
        entirely LLM agent driven. Initially here to allow the reasoning to sidestep
        the above curated actions if need be.

        The snippet is written to a temporary file and executed in a fresh Python
        subprocess (the same interpreter running the agent, so it shares the
        agent's environment: rdkit, numpy, pyscf, ambertools bindings, etc.).
        Running out-of-process isolates the agent from hard crashes (segfaults,
        ``sys.exit``, C-extension aborts) in the generated code and lets us reap
        the whole thing on timeout.

        The subprocess runs with ``workdir`` as its current directory (so relative
        paths the model writes land in the run's output directory, and it can read
        the artifacts produced by earlier steps), and with the agentic_execution package root
        plus any ``extra_paths`` (e.g. a skill's ``scripts/`` directory) prepended
        to ``PYTHONPATH``. That lets generated code import ``agentic_execution`` and
        import/execute the helper scripts shipped with the project skills.

        stdout and stderr are captured separately. On any failure -- a raised
        exception, a non-zero exit, or a timeout -- the reason (Python traceback,
        exit status, or timeout notice) lands in ``stderr`` so the LLM agent can
        read why the code failed and revise it. On success ``stdout`` carries the
        printed output (possibly empty) and ``stderr`` is whatever the snippet
        wrote there (usually empty).

        Arguments:
            code_snippet (str): A multi-line string containing the python code to be
                executed by the QMAgent.
            workdir (Path | None): Defaults to None. Directory to run the snippet
                in (typically the run's output directory). Created if missing;
                falls back to the process cwd when None.
            extra_paths (list[Path] | None): Defaults to None. Additional directories
                to prepend to the subprocess ``PYTHONPATH`` (e.g. skill ``scripts/``
                dirs) so their modules are importable from the snippet.
            timeout (float): Defaults to 300.0. Wall-clock seconds before the
                subprocess is killed and reported as a timeout.

        Returns:
            (dict[str, str]): ``{'stdout': ..., 'stderr': ..., 'returncode': ...}``.
                A non-zero ``returncode`` signals failure; any traceback is placed
                in ``stderr``.
        """
        # code_execution_app is a parsl @python_app: calling it returns an
        # AppFuture (a concurrent.futures.Future). Wrap it so we await the
        # result without blocking the agent's event loop, letting parsl's
        # executor run the snippet on a worker while other actions proceed.
        future = asyncio.wrap_future(
            code_execution_app(
                code_snippet,
                str(workdir) if workdir is not None else None,
                [str(p) for p in extra_paths] if extra_paths else None,
                timeout,
            )
        )
        return await future
