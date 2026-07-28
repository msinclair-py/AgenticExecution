"""Academy agents that run ON the HPC endpoint (or in a local thread pool).

These are launched by the local MCP server through
`Manager.from_exchange_factory(..., executors={'aurora-gpu': GCExecutor(...)})`
and addressed by handle.  They are STATEFUL: the agent process lives for the
life of the session, so anything expensive it loads -- a Chai-1 model in GPU
memory, an OpenMM `Simulation`, a parsed trajectory -- stays loaded between
calls.  That is the whole reason to run an agent instead of submitting bare
functions: stateless submission re-imports and re-loads on every call, which for
a fold battery is most of the runtime.

THIS MODULE MUST BE IMPORTABLE ON THE REMOTE HOST.  Academy pickles the agent
class by reference, so the endpoint's `worker_init` must put this directory on
PYTHONPATH (`cd` alone does not make code importable) and the file must be
staged there.  See service/gc_endpoint_config*.yaml.

PATH RULES (learned from the deepdrivewe-academy reference implementation):
  * Every agent has a `base_dir` and chdirs to it on startup, so RELATIVE paths
    resolve on the host that actually runs the work.
  * Never let an absolute orchestrator path leak into a remote call. Config
    objects are pickled across the network without `.resolve()` normalisation,
    so a laptop path arrives intact on the HPC side and silently means nothing.
  * The orchestrator never opens remote files. Results travel back as objects
    through the exchange.
"""

from __future__ import annotations

import os
from typing import Any

from academy.agent import Agent, action


class HpcWorkerAgent(Agent):
    """A long-lived Python execution agent on one endpoint.

    Two execution modes, deliberately different:

    `run_code`   -- fresh SUBPROCESS per call. A segfault, `sys.exit`, or a
                    C-extension abort costs one call, not the agent, and the
                    timeout is enforceable. Use for anything untrusted or new.

    `run_stateful` -- runs in the AGENT'S OWN process against a persistent
                    namespace that survives across calls. Use to load a model
                    once and reuse it. The tradeoff is real and is the reason
                    both exist: a crash here kills the agent and everything it
                    was holding.
    """

    def __init__(self, *, base_dir: str | None = None,
                 output_dir: str | None = None,
                 env: dict[str, str] | None = None,
                 label: str = 'hpc-worker') -> None:
        super().__init__()
        # Plain str, not Path: these are pickled across a network boundary and
        # a Path from the laptop means nothing on the HPC filesystem anyway.
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.env_extra = dict(env or {})
        self.label = label
        self._ns: dict[str, Any] = {}
        self._startup: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------

    async def agent_on_startup(self) -> None:
        """chdir to base_dir so RELATIVE paths resolve on THIS host."""
        import os
        import socket
        import sys
        from pathlib import Path

        if self.base_dir:
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            os.chdir(self.base_dir)
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        os.environ.update(self.env_extra)
        self._ns = {'__name__': '__hpc_session__'}
        self._startup = {
            'label': self.label,
            'hostname': socket.gethostname(),
            'pid': str(os.getpid()),
            'cwd': os.getcwd(),
            'python': sys.version.split()[0],
            'python_full': f'{sys.version_info.major}.{sys.version_info.minor}.'
                           f'{sys.version_info.micro}',
            'executable': sys.executable,
            'TMPDIR': os.environ.get('TMPDIR', ''),
            'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK', ''),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        }

    async def agent_on_shutdown(self) -> None:
        self._ns.clear()

    # -- actions -----------------------------------------------------------

    @action
    async def probe(self) -> dict[str, Any]:
        """Report this agent's host, python version, cwd and accelerator pinning.

        The `python_full` field is load-bearing: Globus Compute serializes tasks
        and results with `dill`, and a Python version skew between the
        orchestrator and this host does NOT raise -- it SEGFAULTS the
        interpreter (verified: exit 139, no traceback). Compare it against the
        orchestrator's version before trusting anything else.
        """
        import os
        import sys
        info = dict(self._startup)
        info.update({
            'base_dir': self.base_dir or '',
            'output_dir': self.output_dir or '',
            # os.chdir is process-global. When agents share a process the last
            # startup wins, so report the ACTUAL cwd alongside the intended
            # base_dir instead of assuming they agree.
            'cwd_now': os.getcwd(),
            'cwd_matches_base_dir': str(
                (not self.base_dir) or os.path.realpath(os.getcwd())
                == os.path.realpath(self.base_dir)),
            'PBS_JOBID': os.environ.get('PBS_JOBID', ''),
            'session_names': sorted(k for k in self._ns if not k.startswith('__')),
            'dill': _version('dill'),
            'python_full_now': (f'{sys.version_info.major}.{sys.version_info.minor}.'
                                f'{sys.version_info.micro}'),
        })
        return info

    @action
    async def env_report(self, modules: list[str] | None = None) -> dict[str, Any]:
        """Report versions of scientific packages available to this agent."""
        mods = modules or ['numpy', 'scipy', 'MDAnalysis', 'openmm', 'torch',
                           'parsl', 'globus_compute_sdk', 'academy', 'chai_lab']
        out: dict[str, Any] = {m: _version(m) for m in mods}
        out['_accelerators'] = _accelerator_report()
        return out

    @action
    async def run_code(self, code: str, timeout: float = 300.0,
                       workdir: str | None = None,
                       extra_paths: list[str] | None = None,
                       env: dict[str, str] | None = None) -> dict[str, str]:
        """Run a snippet in a FRESH SUBPROCESS on this host.

        Returns {stdout, stderr, returncode, ...} rather than raising, so a
        failing snippet is a readable result the caller can fix and retry.
        Exceptions are reserved for infrastructure faults.
        """
        import asyncio
        import os
        import socket
        import subprocess
        import sys
        import tempfile
        import time
        from pathlib import Path

        t0 = time.time()
        # Default to THIS agent's own base_dir, never os.getcwd(). os.chdir is
        # process-global, so when several agents share a process (which is what
        # --exchange local does) the last one to start wins and every other
        # agent's relative paths silently resolve in the wrong directory.
        # Verified: three agents in local mode all reported the third's cwd.
        wd = workdir or self.output_dir or self.base_dir or os.getcwd()
        Path(wd).mkdir(parents=True, exist_ok=True)

        proc_env = os.environ.copy()
        proc_env.update(self.env_extra)
        if env:
            proc_env.update({str(k): str(v) for k, v in env.items()})
        if extra_paths:
            parts = [str(p) for p in extra_paths]
            if existing := proc_env.get('PYTHONPATH'):
                parts.append(existing)
            proc_env['PYTHONPATH'] = os.pathsep.join(parts)

        # A REAL temp file, not `python -c`: tracebacks then carry a stable
        # filename and correct line numbers for the caller to read and fix.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                         dir=wd) as fh:
            fh.write(code)
            script = fh.name
        try:
            # asyncio.to_thread, NOT a bare subprocess.run. This @action is
            # async and runs on the agent's event loop; a blocking call here
            # stalls the whole loop, so every concurrent action on this agent
            # serializes. Verified: 8 tasks x 5 s took 40.5 s (i.e. serial)
            # before this fix, on a pool sized for 4.
            def _run() -> dict[str, str]:
                try:
                    p = subprocess.run([sys.executable, script],
                                       capture_output=True, cwd=wd,
                                       env=proc_env, timeout=timeout)
                    return {'stdout': p.stdout.decode(errors='replace'),
                            'stderr': p.stderr.decode(errors='replace'),
                            'returncode': str(p.returncode)}
                except subprocess.TimeoutExpired as exc:
                    return {'stdout': (exc.stdout or b'').decode(errors='replace'),
                            'stderr': ((exc.stderr or b'').decode(errors='replace')
                                       + f'\nTimed out after {timeout}s and was '
                                         'killed.').strip(),
                            'returncode': '-1'}
            res = await asyncio.to_thread(_run)
        finally:
            Path(script).unlink(missing_ok=True)

        res.update({
            'hostname': socket.gethostname(),
            'pid': str(os.getpid()),
            'cwd': wd,
            'ZE_AFFINITY_MASK': proc_env.get('ZE_AFFINITY_MASK', ''),
            'CUDA_VISIBLE_DEVICES': proc_env.get('CUDA_VISIBLE_DEVICES', ''),
            'TMPDIR': proc_env.get('TMPDIR', ''),
            'elapsed_s': f'{time.time() - t0:.2f}',
            'mode': 'subprocess',
        })
        return res

    @action
    async def run_stateful(self, code: str) -> dict[str, str]:
        """Run a snippet IN THIS AGENT'S PROCESS against a persistent namespace.

        Names bound here survive to the next call. That is the point: load a
        model once, reuse it for every fold. The cost is isolation -- a segfault
        takes the agent down with everything it holds, and there is no timeout.
        Use `run_code` for anything you are not confident in.
        """
        import contextlib
        import io
        import os
        import socket
        import time
        import traceback

        t0 = time.time()
        out, err = io.StringIO(), io.StringIO()
        rc = '0'
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(code, self._ns)          # noqa: S102 -- that IS the feature
        except BaseException:                 # noqa: BLE001 -- report, never raise
            err.write(traceback.format_exc())
            rc = '1'
        return {
            'stdout': out.getvalue(),
            'stderr': err.getvalue(),
            'returncode': rc,
            'hostname': socket.gethostname(),
            'pid': str(os.getpid()),
            'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK', ''),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'elapsed_s': f'{time.time() - t0:.2f}',
            'mode': 'stateful',
            'session_names': ','.join(sorted(k for k in self._ns
                                             if not k.startswith('__'))),
        }

    @action
    async def clear_session(self) -> dict[str, str]:
        """Drop everything held in the stateful namespace (frees GPU memory)."""
        n = len([k for k in self._ns if not k.startswith('__')])
        self._ns.clear()
        self._ns['__name__'] = '__hpc_session__'
        return {'cleared': str(n)}


# --- helpers (module level so both the agent and verify.py can use them) ----

def _version(mod: str) -> str:
    import importlib
    try:
        m = importlib.import_module(mod)
        return str(getattr(m, '__version__', 'present'))
    except Exception as exc:                          # noqa: BLE001
        return f'MISSING ({type(exc).__name__})'


def _accelerator_report() -> dict[str, Any]:
    import os
    rep: dict[str, Any] = {
        'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK', ''),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'ZE_FLAT_DEVICE_HIERARCHY': os.environ.get('ZE_FLAT_DEVICE_HIERARCHY', ''),
        'visible_devices': None,
        'backend': None,
    }
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            rep['backend'] = 'xpu'
            rep['visible_devices'] = torch.xpu.device_count()
        elif torch.cuda.is_available():
            rep['backend'] = 'cuda'
            rep['visible_devices'] = torch.cuda.device_count()
        else:
            rep['backend'] = 'none'
    except Exception as exc:                          # noqa: BLE001
        rep['backend'] = f'torch unavailable ({type(exc).__name__})'
    return rep
