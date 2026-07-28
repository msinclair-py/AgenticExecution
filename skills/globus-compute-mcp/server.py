#!/usr/bin/env python
"""Local MCP server driving STATEFUL academy agents on HPC via Globus Compute.

    LAPTOP                                          HPC (Aurora / Polaris)
    +----------------------------+                 +--------------------------+
    | MCP server (THIS PROCESS)  |                 | globus-compute-endpoint  |
    |  FastMCP, stdio            |                 |  (persistent daemon on   |
    |     |                      |                 |   a login node)          |
    |  academy Manager           |  -- HTTPS -->   |        |                 |
    |   executors={              |  Globus svc +   |  PBSProProvider -> PBS   |
    |     'aurora-gpu': GCExec,  |  Academy        |        |                 |
    |     'aurora-cpu': GCExec}  |  Exchange Cloud |  HpcWorkerAgent (STATEFUL|
    |     |                      |  <-- results -- |   process, 1 per worker) |
    |  agent handles             |                 +--------------------------+
    +----------------------------+

THIS PROCESS RUNS ON THE LAPTOP, not in a batch allocation.  That inversion is
the point:

  * the endpoint dials OUTBOUND to the Globus service, so no inbound port and
    no facility-internal IP has to be routable;
  * the MCP server is local, so it is always up and has NO WALLTIME -- a PBS
    job is a resource the endpoint acquires, not the thing hosting the service;
  * auth is a refreshable OAuth token, so there is no MFA prompt per call.

The agents are STATEFUL and long-lived: a model loaded once stays in GPU memory
across calls.  Stateless function submission would re-import and re-load every
time, which for a fold battery is most of the runtime.

Two modes:
    --exchange local    everything in-process on ThreadPoolExecutors. No Globus,
                        no endpoint, no SSH. Full tool path is testable here.
    --exchange globus   agents launched onto Globus Compute endpoints through
                        the Academy Exchange Cloud.

!! PYTHON VERSION MUST MATCH ACROSS ALL HOSTS !!  Globus Compute serializes with
`dill`.  A version skew does not raise a clear error -- it SEGFAULTS the
interpreter (verified locally: exit 139, no traceback, no message).  See
`verify.py` rung 1/4 and SKILL.md.

Run:
    ./.venv-gc311/bin/python skills/globus-compute-mcp/server.py --exchange local
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid as _uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# stdout isolation -- defined first, CALLED before server.run(). On stdio the
# JSON-RPC channel IS stdout; academy, parsl and globus-compute all print
# banners, and one stray byte on fd 1 breaks the framing for the whole session.
# ---------------------------------------------------------------------------


class _StdoutShim:
    """`.buffer` stays the real client stream (the MCP transport wraps it);
    every `write` -- prints, banners -- is diverted to stderr."""

    def __init__(self, real_buffer: Any) -> None:
        self.buffer = real_buffer

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(sys.stderr, name)


def isolate_stdout_for_stdio() -> None:
    """Preserve the true client stream, then point fd 1 at stderr.

    Order is load-bearing: dup(1) FIRST (that fd is the only remaining handle on
    the real stdout), then dup2 so native/subprocess writers to fd 1 land on
    stderr, then swap sys.stdout for a shim whose `.buffer` is the preserved
    stream. Idempotent.
    """
    if isinstance(sys.stdout, _StdoutShim):
        return
    real = os.fdopen(os.dup(1), 'wb')
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = _StdoutShim(real)


# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))     # so `hpc_agents` imports here AND on workers

ENDPOINTS_FILE = Path(os.environ.get('GCMCP_ENDPOINTS', HERE / 'endpoints.yaml'))


class ConfigError(RuntimeError):
    """endpoints.yaml is missing, malformed, or lacks a UUID."""


def load_registry(path: Path | None = None) -> dict[str, Any]:
    import yaml
    p = path or ENDPOINTS_FILE
    if not p.exists():
        raise ConfigError(f'endpoint registry not found: {p}')
    doc = yaml.safe_load(p.read_text()) or {}
    eps = doc.get('endpoints') or {}
    if not isinstance(eps, dict):
        raise ConfigError(f'{p}: "endpoints" must be a mapping')
    return {'default': doc.get('default'), 'endpoints': eps, 'path': str(p)}


# ---------------------------------------------------------------------------
# Session: manager + handles + the task registry
# ---------------------------------------------------------------------------

class Session:
    """Everything the tools need, built once in the FastMCP lifespan.

    Each endpoint gets a POOL of agents, not one. That is where parallelism
    comes from: an academy agent is one process handling one action stream, so a
    single agent can never occupy more than one worker slot -- and therefore
    never more than one pinned accelerator. The reference implementation
    launches one simulation agent per walker for exactly this reason. Sizing the
    pool to the endpoint's worker count is what lets `submit_batch` fan a sweep
    across every GPU in the allocation.
    """

    def __init__(self, exchange: str, registry: dict[str, Any]) -> None:
        self.exchange = exchange
        self.registry = registry
        self.pools: dict[str, list[Any]] = {}      # endpoint -> [Handle, ...]
        self.executors: dict[str, Executor | None] = {}
        self._rr: dict[str, int] = {}
        self.manager: Any = None
        self.agent_class: Any = None
        self._locks: dict[str, asyncio.Lock] = {}
        # task_id -> asyncio.Task. Submit-then-poll lives here, in this process:
        # academy actions return awaitables, not service-side task ids.
        self.tasks: dict[str, asyncio.Task] = {}
        self.task_meta: dict[str, dict[str, Any]] = {}

    def pool_size(self, name: str) -> int:
        rec = self.registry['endpoints'][name]
        if self.exchange == 'local':
            return int(rec.get('local_workers', 4))
        # One agent per worker slot the endpoint can offer.
        explicit = rec.get('agent_pool')
        if explicit:
            return int(explicit)
        return max(1, int(rec.get('accelerators_per_node') or 1)
                   * int(rec.get('nodes_per_block', 1) or 1))

    async def ensure_pool(self, name: str) -> list[Any]:
        """Launch this endpoint's agent pool if it is not up yet (idempotent).

        Called on first use rather than at startup. Each agent holds a worker
        slot for its whole life, and the endpoints idle at zero blocks, so an
        eager launch would stall behind the PBS queue during the MCP handshake.
        """
        from academy.handle import Handle

        if self.pools.get(name):
            return self.pools[name]
        if self.manager is None or name not in self.executors:
            return []
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if self.pools.get(name):                  # won the race elsewhere
                return self.pools[name]
            rec = self.registry['endpoints'][name]
            local = self.exchange == 'local'
            base_dir = rec.get('local_base_dir') if local else rec.get('base_dir')
            out_dir = rec.get('local_output_dir') if local else rec.get('output_dir')
            if local and base_dir:
                # Only safe in local mode: orchestrator and agent are one host.
                # NEVER mkdir a remote path from here -- it would silently create
                # a laptop directory named /flare/... instead.
                base_dir = str(Path(base_dir).expanduser())
                Path(base_dir).mkdir(parents=True, exist_ok=True)
                out_dir = str(Path(out_dir).expanduser()) if out_dir else None
            pool: list[Any] = []
            for k in range(self.pool_size(name)):
                try:
                    handle = await self.manager.launch(
                        self.agent_class,
                        kwargs={'base_dir': base_dir, 'output_dir': out_dir,
                                'env': rec.get('env') or {},
                                'label': f'{name}[{k}]'},
                        executor=name, name=f'hpc-worker-{name}-{k}')
                except Exception as exc:              # noqa: BLE001
                    print(f'[gcmcp] FAILED to launch agent {k} on {name}: '
                          f'{type(exc).__name__}: {exc}', file=sys.stderr)
                    continue
                # REBIND, do not store `handle` directly. A Handle resolves its
                # ExchangeClient through a ContextVar, and contextvars propagate
                # parent->child only at task creation. An MCP tool call runs in a
                # DIFFERENT asyncio task than the lifespan that created this
                # handle, so the lookup fails and every call dies with
                # ExchangeClientNotFoundError -- AFTER list_tools has already
                # succeeded, which makes it look like a tool bug, not a wiring bug.
                pool.append(Handle(handle.agent_id,
                                   exchange=self.manager.exchange_client))
            if pool:
                self.pools[name] = pool
                print(f'[gcmcp] {len(pool)} agent(s) live on {name!r}',
                      file=sys.stderr)
            return pool

    def resolve(self, name: str | None) -> str:
        name = name or self.registry['default']
        if not name:
            raise ConfigError(f'no endpoint given and no `default:` in '
                              f'{self.registry["path"]}')
        if name not in self.registry['endpoints']:
            raise ConfigError(f'unknown endpoint {name!r}. '
                              f'Known: {sorted(self.registry["endpoints"])}')
        if not self.pools.get(name):
            raise ConfigError(
                f'endpoint {name!r} is configured but has no live agents. '
                f'Live: {sorted(k for k, v in self.pools.items() if v)}. In '
                '--exchange globus this means its uuid is missing from the '
                'registry or the agents failed to launch; check stderr.')
        return name

    def pick(self, name: str, agent_index: int | None = None) -> Any:
        """Round-robin, or a PINNED agent when the caller needs a stable one."""
        pool = self.pools[name]
        if agent_index is not None:
            if not 0 <= agent_index < len(pool):
                raise ConfigError(
                    f'agent_index {agent_index} out of range for {name!r} '
                    f'(pool size {len(pool)})')
            return pool[agent_index]
        i = self._rr.get(name, 0)
        self._rr[name] = (i + 1) % len(pool)
        return pool[i]


def build_executors(exchange: str, registry: dict[str, Any]
                    ) -> tuple[dict[str, Executor | None], list[str]]:
    """{endpoint_name: Executor} for the academy Manager, plus skip reasons.

    local  -> ThreadPoolExecutor per endpoint (no Globus, no endpoint, no SSH).
    globus -> one globus_compute_sdk.Executor per CONFIGURED endpoint uuid.
    """
    execs: dict[str, Executor | None] = {}
    skipped: list[str] = []
    for name, rec in registry['endpoints'].items():
        workers = int(rec.get('local_workers', 4))
        if exchange == 'local':
            execs[name] = ThreadPoolExecutor(max_workers=workers)
            continue
        uuid = rec.get('uuid')
        if not uuid:
            skipped.append(f'{name}: no uuid in the registry (bootstrap not done)')
            continue
        # Imported lazily so --exchange local needs no globus-compute at all.
        from globus_compute_sdk import Executor as GCExecutor
        execs[name] = GCExecutor(endpoint_id=str(uuid))
    return execs, skipped


def create_exchange_factory(exchange: str):
    from academy.exchange.local import LocalExchangeFactory
    if exchange == 'local':
        return LocalExchangeFactory()
    from academy.exchange.cloud.client import HttpExchangeFactory
    return HttpExchangeFactory(auth_method='globus')


# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
These tools run Python on an ALCF supercomputer (Aurora / Polaris) through
long-lived academy agents on Globus Compute endpoints. This MCP server is LOCAL;
the compute is remote and reached over HTTPS -- no SSH, no tunnel, and no
walltime on the service itself.

Four rules govern effective use:

1. THROUGHPUT COMES FROM CONCURRENCY AT THIS CLIENT. One serial `run_python`
   call uses one worker no matter how many nodes are allocated. Use
   `submit_batch` with many independent specs so they are all in flight.

2. SUBMIT THEN POLL. `submit_batch` returns task IDs immediately; `poll` and
   `get_result` never block. Use `run_python` only for work finishing in well
   under a minute -- anything longer must be submitted and polled, or the MCP
   call times out while the job is still queued.

3. THE AGENTS ARE STATEFUL. `run_python(..., stateful=True)` executes in the
   agent's own process against a namespace that PERSISTS between calls. Load a
   model once and reuse it, instead of paying the import on every fold. The
   default (stateful=False) runs in a fresh subprocess and is isolated: use it
   for anything that might crash.

4. PATHS ARE REMOTE. Every path you pass is interpreted ON THE HPC FILESYSTEM.
   Never hand these tools a laptop path, and never expect to open a returned
   path locally -- print what you need and read it from stdout.

A snippet that fails returns returncode != 0 with the traceback in `stderr` --
read it, fix the code, resubmit. An `error` field instead means the
infrastructure failed.
"""


def build_server(exchange: str, registry_path: Path | None = None):
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    registry = load_registry(registry_path)
    session = Session(exchange, registry)

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        from academy.handle import Handle
        from academy.manager import Manager

        import hpc_agents

        execs, skipped = build_executors(exchange, registry)
        session.executors = execs
        for reason in skipped:
            print(f'[gcmcp] SKIP {reason}', file=sys.stderr)
        if not execs:
            print('[gcmcp] WARNING: no usable endpoints. In --exchange globus '
                  'every endpoint needs a uuid; run the bootstrap or use '
                  '--exchange local.', file=sys.stderr)

        default_exec = registry['default'] if registry['default'] in execs \
            else (next(iter(execs)) if execs else 'event_loop')

        try:
            async with await Manager.from_exchange_factory(
                    factory=create_exchange_factory(exchange),
                    executors=execs or None,
                    default_executor=default_exec) as manager:
                session.manager = manager
                session.agent_class = hpc_agents.HpcWorkerAgent
                # LAZY: agents are launched on FIRST USE of an endpoint, not
                # here. An academy agent occupies a worker slot for its whole
                # life, and the endpoints run init_blocks: 0 / min_blocks: 0 so
                # that idling costs nothing -- which means at startup there are
                # ZERO workers and PBS has not been asked for any. Launching a
                # 12-agent pool eagerly would block server startup until the
                # scheduler granted nodes, i.e. the MCP client would time out
                # during its handshake. Deferring the launch keeps startup
                # instant and lets the first real call pay (and report) the
                # queue wait.
                if exchange == 'local':
                    # Local mode has no scheduler, so warm up immediately: it
                    # makes the smoke test deterministic.
                    for name in list(execs):
                        await session.ensure_pool(name)
                yield session
        finally:
            for ex in session.executors.values():
                if ex is not None:
                    ex.shutdown(wait=False)

    mcp = FastMCP(name='globus-compute-hpc', instructions=INSTRUCTIONS,
                  lifespan=lifespan, on_duplicate='replace')

    def _fail(exc: Exception) -> ToolError:
        return ToolError(f'{type(exc).__name__}: {exc}')

    async def _ep(endpoint: str | None) -> str:
        """Resolve a name and make sure its agent pool is up (lazy launch)."""
        name = endpoint or registry['default']
        if name not in registry['endpoints']:
            raise ConfigError(f'unknown endpoint {name!r}. '
                              f'Known: {sorted(registry["endpoints"])}')
        await session.ensure_pool(name)
        return session.resolve(name)

    async def _act(name: str, act: str, *args, agent_index: int | None = None,
                   **kwargs) -> Any:
        return await session.pick(name, agent_index).action(act, *args, **kwargs)

    def _register(name: str, kind: str, meta: dict[str, Any]) -> str:
        tid = str(_uuid.uuid4())
        session.task_meta[tid] = {'endpoint': name, 'kind': kind,
                                  'submitted_t': time.time(), **meta}
        return tid

    def _status(tid: str) -> dict[str, Any]:
        task = session.tasks.get(tid)
        meta = session.task_meta.get(tid)
        if task is None or meta is None:
            return {'task_id': tid, 'pending': True, 'status': 'unknown',
                    'note': 'no such task in this server process. Task handles '
                            'live in memory, so a server restart loses them.'}
        base = {'task_id': tid, 'endpoint': meta['endpoint'],
                'label': meta.get('label'),
                'elapsed_s': f'{time.time() - meta["submitted_t"]:.1f}'}
        if not task.done():
            return {**base, 'pending': False if False else True, 'status': 'running'}
        if task.cancelled():
            return {**base, 'pending': False, 'status': 'cancelled'}
        exc = task.exception()
        if exc is not None:
            # Infrastructure fault: agent died, exchange dropped, serialization
            # failed. NOT the same as a snippet with a non-zero returncode.
            return {**base, 'pending': False, 'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}'}
        res = task.result()
        out = {**base, 'pending': False, 'status': 'success'}
        out.update(res if isinstance(res, dict) else {'result': res})
        return out

    # -- discovery ---------------------------------------------------------

    @mcp.tool
    def list_endpoints() -> dict[str, Any]:
        """List the HPC endpoints this server can dispatch to.

        Endpoints are named by TASK CLASS, not just machine -- e.g. 'aurora-gpu'
        for folding/MD and 'aurora-cpu' for trajectory analysis and statistics.
        Send GPU work to a gpu endpoint and single-threaded analysis to a cpu
        one; they are separate allocations with different worker counts.

        `live: true` means an agent is running there right now and it can accept
        work. `live: false` with `configured: false` means the one-time bootstrap
        has not been done for that endpoint.
        """
        eps = []
        for name, rec in registry['endpoints'].items():
            eps.append({
                'endpoint': name,
                'live': bool(session.pools.get(name)),
                'agents': len(session.pools.get(name, [])),
                'configured': bool(rec.get('uuid')) or exchange == 'local',
                'task_class': rec.get('task_class'),
                'machine': rec.get('machine'),
                'accelerators_per_node': rec.get('accelerators_per_node'),
                'pinning_var': rec.get('pinning_var'),
                'queue': rec.get('queue'),
                'base_dir': (rec.get('local_base_dir') if exchange == 'local'
                             else rec.get('base_dir')),
                'notes': rec.get('notes'),
            })
        return {'exchange_mode': exchange, 'default': registry['default'],
                'registry_file': registry['path'], 'endpoints': eps}

    @mcp.tool
    async def endpoint_status(endpoint: str | None = None) -> dict[str, Any]:
        """Is the endpoint's agent alive, on what host, with what pinning?

        Asks the live agent directly (so it proves the whole path works), and in
        --exchange globus also reports the Globus Compute service's view: online
        status and worker counts.

        Zero workers with tasks pending is NORMAL right after the first
        submission: init_blocks/min_blocks are 0, so the endpoint queues a PBS
        job only when work arrives. If it stays that way, check `qstat` over SSH.
        """
        try:
            name = await _ep(endpoint)
            info = await _act(name, 'probe')
        except Exception as exc:                      # noqa: BLE001
            raise _fail(exc) from exc
        rec = registry['endpoints'][name]
        out: dict[str, Any] = {'endpoint': name, 'exchange_mode': exchange,
                               'agent': info,
                               'accelerators_per_node': rec.get('accelerators_per_node')}
        # local python, for the version-skew check that matters most
        out['orchestrator_python'] = (f'{sys.version_info.major}.'
                                      f'{sys.version_info.minor}.{sys.version_info.micro}')
        remote_py = info.get('python_full', '')
        if remote_py and remote_py.split('.')[:2] != out['orchestrator_python'].split('.')[:2]:
            out['PYTHON_VERSION_MISMATCH'] = (
                f'orchestrator {out["orchestrator_python"]} vs agent {remote_py}. '
                'dill serialization across a minor-version skew SEGFAULTS the '
                'interpreter (exit 139, no traceback). Rebuild one side to match.')
        if exchange == 'globus' and rec.get('uuid'):
            try:
                from globus_compute_sdk import Client
                st = Client().get_endpoint_status(str(rec['uuid']))
                out['service_status'] = st.get('status')
                out['service_details'] = st.get('details', {})
            except Exception as exc:                  # noqa: BLE001
                out['service_status_error'] = f'{type(exc).__name__}: {exc}'
        return out

    # -- execution ---------------------------------------------------------

    @mcp.tool
    async def run_python(code: str,
                         endpoint: str | None = None,
                         timeout: float = 300.0,
                         wait: float = 60.0,
                         stateful: bool = False,
                         agent_index: int | None = None,
                         workdir: str | None = None,
                         extra_paths: list[str] | None = None,
                         env: dict[str, str] | None = None) -> dict[str, Any]:
        """Run ONE Python snippet on an HPC agent and wait briefly for it.

        For short interactive work: probing the environment, a quick
        calculation, loading a model into a stateful session. For anything
        longer or for more than one job use `submit_batch` + `poll` --
        ONE SERIAL CALL USES ONE WORKER no matter how big the allocation is.

        Anything you want back must be PRINTED; the snippet's value is discarded.
        All paths are on the HPC filesystem, not the laptop.

        Args:
            code: the snippet. Print results to stdout.
            endpoint: name from `list_endpoints` (e.g. 'aurora-gpu').
            timeout: seconds before the subprocess is killed. Ignored when
                stateful=True (there is no subprocess to kill).
            wait: seconds THIS CALL blocks. If it elapses you get
                `pending: true` plus a `task_id`; the work keeps running and is
                retrievable with `get_result`.
            stateful: run in the AGENT'S OWN process against a namespace that
                PERSISTS across calls -- load a model once, reuse it for every
                subsequent call. Not isolated: a crash kills the agent and
                everything it holds, and `timeout` does not apply. Default False
                runs in a fresh subprocess, which is isolated and interruptible.
            agent_index: which agent in the endpoint's pool to use. Omit to
                round-robin (the right choice for independent work). Stateful
                calls pin to agent 0 unless you say otherwise -- the namespace
                lives in ONE process, so `run_python(stateful=True)` on a
                different index sees a DIFFERENT namespace. Use distinct indices
                deliberately to keep several models resident at once.
            workdir: directory ON THE HPC FILESYSTEM (subprocess mode only).
            extra_paths: prepended to the worker's PYTHONPATH.
            env: extra environment variables for the subprocess.

        Returns:
            {stdout, stderr, returncode, hostname, pid, ZE_AFFINITY_MASK,
             CUDA_VISIBLE_DEVICES, elapsed_s, mode, task_id}. A non-zero
            `returncode` is a TASK failure: read `stderr`, which carries a real
            traceback with filename and line number. An `error` field instead
            means the infrastructure failed.
        """
        try:
            name = await _ep(endpoint)
            if stateful:
                # stateful work MUST pin to one agent: the namespace lives in
                # that process and round-robin would scatter it across the pool.
                coro = _act(name, 'run_stateful', code,
                            agent_index=agent_index if agent_index is not None else 0)
            else:
                coro = _act(name, 'run_code', code, timeout=timeout,
                            workdir=workdir, extra_paths=extra_paths, env=env,
                            agent_index=agent_index)
            tid = _register(name, 'stateful' if stateful else 'subprocess',
                            {'label': 'run_python'})
            session.tasks[tid] = asyncio.ensure_future(coro)
        except Exception as exc:                      # noqa: BLE001
            raise _fail(exc) from exc
        try:
            await asyncio.wait_for(asyncio.shield(session.tasks[tid]),
                                   timeout=max(0.0, wait))
        except asyncio.TimeoutError:
            return {'task_id': tid, 'endpoint': name, 'pending': True,
                    'status': 'running',
                    'hint': (f'still running after wait={wait}s. NOT cancelled -- '
                             f'call get_result("{tid}") later.')}
        except Exception:                             # noqa: BLE001
            pass                                      # surfaced by _status
        return _status(tid)

    @mcp.tool
    async def submit_batch(specs: list[dict[str, Any]],
                           endpoint: str | None = None,
                           timeout: float = 300.0,
                           stateful: bool = False,
                           workdir: str | None = None,
                           extra_paths: list[str] | None = None,
                           env: dict[str, str] | None = None) -> dict[str, Any]:
        """Submit MANY independent snippets at once. Returns task IDs immediately.

        THIS IS HOW YOU GET THROUGHPUT. The allocation has
        nodes x accelerators_per_node workers (Aurora GPU: 12/node, Polaris GPU:
        4/node), and they only work in parallel if many tasks are IN FLIGHT AT
        ONCE. Calling `run_python` in a loop uses exactly one worker and wastes
        the rest of the allocation. Submit the whole sweep here in one call.

        Never blocks on execution: it returns as soon as the work is dispatched.
        Then `poll(task_ids)` until they finish. That separation is deliberate --
        an HPC job can take hours and an MCP call cannot.

        Args:
            specs: one dict per job, required key `code`. Optional per-job
                overrides: `label` (yours, echoed back), `timeout`, `workdir`,
                `extra_paths`, `env`, `stateful`. A bare string is shorthand for
                {'code': ...}.
            endpoint: name from `list_endpoints`.
            Other args are defaults for specs that do not override them. Specs
                are round-robined across the endpoint's agent pool, which is what
                spreads the sweep over every accelerator; set a spec's
                `agent_index` only to pin it deliberately.

        Returns:
            {endpoint, count, tasks: [{label, task_id}], task_ids: [...]}.
            Feed `task_ids` straight into `poll`.

        Note: a batch sent `stateful=True` shares ONE namespace on ONE agent and
        the calls interleave -- fine for reads, dangerous for writes. Batches
        should normally be stateless.
        """
        if not specs:
            raise ToolError('specs is empty -- nothing to submit')
        try:
            name = await _ep(endpoint)
            tasks = []
            for i, spec in enumerate(specs):
                if isinstance(spec, str):
                    spec = {'code': spec}
                if not isinstance(spec, dict) or 'code' not in spec:
                    raise ValueError(
                        f'specs[{i}] must be a dict with a "code" key (or a bare '
                        f'string); got {type(spec).__name__}')
                label = str(spec.get('label', f'task{i}'))
                if bool(spec.get('stateful', stateful)):
                    coro = _act(name, 'run_stateful', spec['code'],
                                agent_index=int(spec.get('agent_index', 0)))
                    kind = 'stateful'
                else:
                    # no agent_index -> round-robin across the pool, which is
                    # what fans the sweep over every accelerator.
                    coro = _act(name, 'run_code', spec['code'],
                                timeout=float(spec.get('timeout', timeout)),
                                workdir=spec.get('workdir', workdir),
                                extra_paths=spec.get('extra_paths', extra_paths),
                                env=spec.get('env', env),
                                agent_index=spec.get('agent_index'))
                    kind = 'subprocess'
                tid = _register(name, kind, {'label': label})
                session.tasks[tid] = asyncio.ensure_future(coro)
                tasks.append({'label': label, 'task_id': tid})
        except Exception as exc:                      # noqa: BLE001
            raise _fail(exc) from exc
        return {'endpoint': name, 'count': len(tasks), 'tasks': tasks,
                'task_ids': [t['task_id'] for t in tasks],
                'next': 'call poll(task_ids) -- it does not block'}

    # -- retrieval (never blocks) ------------------------------------------

    @mcp.tool
    def get_result(task_id: str) -> dict[str, Any]:
        """Fetch ONE task's result. Returns immediately, finished or not.

        If the task is still running you get {pending: true, status: 'running'}
        -- a normal answer, not an error. Call again later. Nothing here waits on
        a job, so a six-hour run cannot stall the MCP session.

        Non-zero `returncode` = the snippet failed; read `stderr`. An `error`
        field = the infrastructure failed.
        """
        return _status(str(task_id))

    @mcp.tool
    def poll(task_ids: list[str], summary_only: bool = False) -> dict[str, Any]:
        """Check MANY tasks at once. Never blocks.

        The companion to `submit_batch`: submit a sweep, then poll the returned
        `task_ids` until `pending` reaches 0. This is the ONLY safe pattern for
        long HPC work -- a tool that blocked on a 6-hour MD run would time out
        the MCP client and lose the handle to the job.

        Args:
            task_ids: ids from `submit_batch` / `run_python`.
            summary_only: return just counts and the ids in each bucket, no
                stdout/stderr. Use it while waiting on a large sweep so polling
                does not flood the context.

        Returns:
            {total, pending, succeeded, failed_task, failed_infra, results}
            where `failed_task` counts non-zero returncodes (your snippet) and
            `failed_infra` counts agent/exchange failures.
        """
        results, buckets = [], {'pending': [], 'succeeded': [],
                                'failed_task': [], 'failed_infra': []}
        for tid in map(str, task_ids):
            s = _status(tid)
            if s.get('pending'):
                buckets['pending'].append(tid)
            elif 'error' in s:
                buckets['failed_infra'].append(tid)
            elif str(s.get('returncode', '0')) != '0':
                buckets['failed_task'].append(tid)
            else:
                buckets['succeeded'].append(tid)
            results.append(s)
        out: dict[str, Any] = {
            'total': len(results), 'pending': len(buckets['pending']),
            'succeeded': len(buckets['succeeded']),
            'failed_task': len(buckets['failed_task']),
            'failed_infra': len(buckets['failed_infra'])}
        out['buckets' if summary_only else 'results'] = \
            buckets if summary_only else results
        return out

    @mcp.tool
    async def session_info(endpoint: str | None = None,
                           agent_index: int = 0) -> dict[str, Any]:
        """What is currently held in an agent's STATEFUL namespace, and its env.

        Use this to see whether an expensive object (a loaded model, an OpenMM
        Simulation) is still resident before paying to load it again, and to
        read the scientific package versions actually available on that host.
        """
        try:
            name = await _ep(endpoint)
            probe = await _act(name, 'probe', agent_index=agent_index)
            envr = await _act(name, 'env_report', agent_index=agent_index)
        except Exception as exc:                      # noqa: BLE001
            raise _fail(exc) from exc
        return {'endpoint': name, 'session_names': probe.get('session_names', []),
                'host': probe.get('hostname'), 'cwd': probe.get('cwd_now'),
                'python': probe.get('python_full'), 'packages': envr}

    @mcp.tool
    async def clear_session(endpoint: str | None = None,
                            agent_index: int = 0) -> dict[str, Any]:
        """Drop everything in an agent's stateful namespace, freeing GPU memory."""
        try:
            name = await _ep(endpoint)
            return {'endpoint': name, 'agent_index': agent_index,
                    **await _act(name, 'clear_session', agent_index=agent_index)}
        except Exception as exc:                      # noqa: BLE001
            raise _fail(exc) from exc

    return mcp, session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exchange', choices=['local', 'globus'],
                    default=os.environ.get('GCMCP_EXCHANGE', 'globus'),
                    help="'globus' launches agents onto Globus Compute endpoints "
                         "via the Academy Exchange Cloud; 'local' runs everything "
                         'in-process on thread pools for smoke testing (no Globus, '
                         'no endpoint, no SSH).')
    ap.add_argument('--endpoints', type=Path, default=None,
                    help='path to endpoints.yaml (default: alongside this file)')
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # BEFORE server.run(): startup is the noisiest moment and on stdio a single
    # banner byte on fd 1 breaks JSON-RPC framing for the whole session.
    isolate_stdout_for_stdio()
    mcp, _ = build_server(args.exchange, args.endpoints)
    mcp.run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    main()
