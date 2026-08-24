"""Template: the ONLY agent this pattern ever deploys -- a pure code-execution
relay. From the hpc-deployment skill.

This agent exposes exactly one capability: run an arbitrary code payload in a
subprocess and return what happened. It is not a framework for turning
arbitrary Python into MCP tools -- if a campaign needs a new capability, that
capability is a library the payload imports, never a new @action added here.
Adding domain-specific actions defeats the reason this pattern is safe to
reuse across campaigns: the LLM harness is solely responsible for generating
the code that runs; this agent's only job is relaying it to compute the
harness doesn't otherwise have.

Launched via Globus Compute (see scripts/mcp_server.py), never via qsub.
Getting the underlying allocation -- nodes, accelerators, queue, walltime,
worker environment -- is the Globus Compute endpoint's job, configured once,
out of band, on the HPC side. There is no Parsl or PBS/Slurm code in this
file because it doesn't need any: the endpoint places this process on one
pinned accelerator and sets that device's env var (CUDA_VISIBLE_DEVICES /
ZE_AFFINITY_MASK -- see references/site-profiles.md) before this code starts,
exactly the same pinning contract Parsl workers use. subprocess.run() below
inherits it for free via env=os.environ.copy().
"""
import asyncio
import os

from academy.agent import Agent, action


class HpcAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._inflight = 0
        self._watchdog: asyncio.Task | None = None

    async def agent_on_startup(self) -> None:
        self._watchdog = asyncio.create_task(self._lifetime_watchdog())

    async def agent_on_shutdown(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()

    async def _lifetime_watchdog(self) -> None:
        """The Globus exchange credential this agent was launched under never
        refreshes -- HttpExchangeFactory resolves it once, in __init__, and
        holds it for the transport's life. Retire before it expires, or this
        agent detaches silently and keeps holding its worker slot forever.
        See references/long-running-pools.md. Pick the TTL against the
        credential's lifetime, not intuition, and read it on the worker (this
        process's env, not the client's)."""
        ttl = float(os.environ.get('ACADEMY_AGENT_TTL_S', 30 * 3600))
        if ttl <= 0:                              # 0 disables, for debugging
            return
        await asyncio.sleep(ttl)
        while self._inflight > 0:                 # don't kill a running task;
            await asyncio.sleep(30)                # a slow finish beats a lost result
        self.agent_shutdown()

    @action
    async def probe(self) -> dict[str, float]:
        """Cheap liveness/load check. The pool manager uses this to prune
        dead handles, and it's the ground-truth diagnostic for a wedged pool:
        every other signal (endpoint status, agent count, worker logs) can
        report healthy while nothing is actually running -- a node reporting
        load1 == 0.0 while a task is 'running' is wedged, full stop."""
        return {'pid': float(os.getpid()), 'load1': os.getloadavg()[0]}

    @action
    async def run_code(self, payload: str, workdir: str | None = None,
                       extra_paths: list[str] | None = None,
                       timeout: float = 300.0) -> dict[str, str]:
        """Run an LLM-authored Python snippet in a fresh subprocess on this
        agent's pinned accelerator. Returns {stdout, stderr, returncode}. A
        nonzero returncode is not raised as an error -- the harness reads
        stderr, fixes its snippet, and retries. This is the ONLY action this
        agent exposes; anything else the snippet needs -- another library, a
        call out to some other tool -- it must do itself, in code."""
        self._inflight += 1
        try:
            # Off the event loop: subprocess.run() blocks, and the agent must
            # keep answering `probe` (and other in-flight run_code calls)
            # while a long snippet runs.
            return await asyncio.get_running_loop().run_in_executor(
                None, _run_subprocess, payload, workdir, extra_paths, timeout)
        finally:
            self._inflight -= 1


def _run_subprocess(payload: str, workdir: str | None,
                    extra_paths: list[str] | None, timeout: float) -> dict[str, str]:
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()   # carries the endpoint's device-pinning var through
    parts = [str(p) for p in (extra_paths or [])]
    if existing := env.get('PYTHONPATH'):
        parts.append(existing)
    if parts:
        env['PYTHONPATH'] = os.pathsep.join(parts)

    # Write a REAL temp file rather than passing `-c`, so tracebacks carry a
    # stable filename and correct line numbers for the LLM to read and fix.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(payload)
        script = f.name
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              cwd=workdir, env=env, timeout=timeout)
        return {'stdout': proc.stdout.decode(errors='replace'),
                'stderr': proc.stderr.decode(errors='replace'),
                'returncode': str(proc.returncode)}
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired carries whatever was captured before the kill --
        # return it, so the harness sees how far the snippet got.
        out = (exc.stdout or b'').decode(errors='replace')
        err = (exc.stderr or b'').decode(errors='replace')
        return {'stdout': out,
                'stderr': f'{err}\nTimed out after {timeout}s and was killed.'.strip(),
                'returncode': '-1'}
    finally:
        Path(script).unlink(missing_ok=True)
