---
name: academy-mcp-hpc
description: >-
  Deploy an academy Agent as a FastMCP server inside an HPC batch allocation, so
  an external MCP harness (Claude Code, Codex, ...) drives the agent's actions
  and a Parsl HighThroughputExecutor fans each call across every GPU on every
  node of the job. Use when you want to expose supercomputer compute to an LLM
  harness as MCP tools, scale a single MCP tool over a multi-node PBS/Slurm
  allocation, or turn an academy `@action` into a network-reachable tool. Covers
  the agent skeleton, the Parsl scaling config, the FastMCP lifespan wiring, the
  run-once head-node launcher, the PBS script, the inverted variant for
  multi-day campaigns (local MCP server driving an agent pool on a Globus
  Compute endpoint, with agent self-retirement, adopt-before-launch and pool
  top-up), and the failure modes that kill this stack on a real machine (AF_UNIX
  TMPDIR, PATH shadowing, single-GPU trap, mpiexec double-launch, HSN
  reachability, the ALCF proxy 503, and the non-refreshing exchange token that
  sterilises an allocation).
license: Apache-2.0
metadata:
  verified-against: >-
    End-to-end deployed and driven on ALCF Aurora 2026-07-24 (2 nodes, debug
    queue, job 8698638): academy-py 0.5.0, parsl 2026.07.20, fastmcp 3.4.4,
    frameworks/2025.3.1. 96-call fan-out over 24 GPU-tile workers, 0 errors.
    Slurm profile is ported, NOT run. Step 7 (the inverted Globus Compute
    variant and the pool-lifecycle patches) verified on ALCF Polaris
    2026-07-30..08-02: capacity queue, 2 nodes x 4 A100, a 96 h and a 48 h
    reservation, four multi-day blind campaigns driven from a local MCP server.
    Every failure mode described there was observed, not anticipated.
---

# Deploying an academy agent + MCP server on HPC

The pattern: one **academy `Agent`** owns a **Parsl** runtime; its `@action`
methods submit `@python_app` tasks that Parsl spreads across the allocation's
GPUs; a **FastMCP** server exposes those actions as MCP tools over HTTP; an
external harness connects and calls them. The batch job is a *long-lived compute
service*, not a script — the harness supplies the reasoning, the allocation
supplies the throughput.

This is instruction-only and self-contained: nothing here imports a project
package. Substitute your own agent and actions.

## When to use it

- You want Claude Code / Codex / any MCP client to run work on a supercomputer,
  with the LLM outside the allocation and the compute inside it.
- You have (or want) an academy agent with `@action` methods and need it reachable.
- You need **one tool call to land on one GPU**, with many calls in flight — the
  throughput comes from fanning independent calls, not from one big MPI job.
- Site is PBS Pro (ALCF Polaris/Aurora shape) or Slurm; nodes have N accelerators.

Do **not** use it for a single tightly-coupled MPI simulation — that is a plain
`mpiexec` job. This pattern is for embarrassingly-parallel, agent-issued work.

## Setup (gate before installing)

Three packages carry the whole stack; everything else is your own science code.

```bash
python -c "import academy, parsl, fastmcp" 2>/dev/null && echo ready || \
  pip install 'academy-py>=0.5' 'parsl>=2024.1' 'fastmcp>=3.4' pydantic pyyaml
```

`academy` is imported from the **`academy-py`** distribution — the import name and
the package name differ. On a login node this install is fine; on Aurora build the
venv with `--system-site-packages` on top of `frameworks` first (see
[references/site-profiles.md](references/site-profiles.md)) so `torch.xpu`
survives.

Expect pip to report dependency conflicts on Aurora: `fastmcp` pulls a newer
`starlette` than the frameworks image's `fastapi` / `prometheus-fastapi-instrumentator`
pin, and `vllm` wants a different torch. Harmless for this stack — the venv's
copies win and none of those packages are used — but do not "fix" it by
downgrading `fastmcp`.

## What you assemble

The blocks below compose into two files. Each block shows only its own imports —
collect them at the top of the file you paste into.

| File | Blocks | Runs |
|---|---|---|
| `hpc_agent.py` | Steps 1–3: `@python_app`, `Agent`, `hpc_config`, `register`, `create_managed_server` | imported |
| `launch_scaling_mcp.py` | Step 4: `count_nodes`, `routable_ip`, `default_worker_init`, `main` | once, head node |

The batch script (Step 5) invokes only `launch_scaling_mcp.py`.

Step 7 is a **variant, not a third file**: it inverts where the server lives so
the harness outlives the PBS job, and it patches Step 1's agent and Step 3's
launch path. Skip it for anything shorter than a day; adopt it before anything
longer.

## Architecture

```
external MCP harness ──HTTP──► FastMCP server ──► academy Agent ──► parsl.load(config)
   (Claude Code)                (your @actions)     (one process,          │
                                                     head node)           │
                                       HighThroughputExecutor 'gpu' (1 worker/accelerator)
                                       HighThroughputExecutor 'cpu' (M workers/node)
                                       LocalProvider(nodes_per_block=N) + MpiExecLauncher(--ppn 1)
                                                                          │
                                            @python_app fanned over N nodes × A accelerators
```

Four layers, each with one job:

| Layer | Owns |
|---|---|
| PBS/Slurm job | the allocation, the modules, the venv, one head-node process |
| Parsl | fan-out: one HTEx manager per node, one worker per accelerator |
| academy Agent | the runtime lifecycle (`parsl.load` / `cleanup`) and the actions |
| FastMCP | the wire protocol the harness speaks |

## Step 1 — the agent

The agent loads Parsl in `agent_on_startup` and tears it down in
`agent_on_shutdown`. Actions are `async`; they call a `@python_app` (which
returns a `concurrent.futures.Future`) and await it via `asyncio.wrap_future`, so
the agent's event loop keeps serving other calls while the app runs on a worker.

```python
import asyncio, parsl
from academy.agent import Agent, action
from parsl.config import Config
from pathlib import Path

# executors=['gpu'] is REQUIRED, not decoration. Parsl's default is
# executors='all', so without it this app also lands on the 'cpu' executor's
# workers, which have no accelerator pinning — verified on Aurora, where a
# 96-call fan-out hit 26 distinct workers (2 nodes x 12 tiles + 2 unpinned CPU
# workers) and those two reported ZE_AFFINITY_MASK unset.
@parsl.python_app(executors=['gpu'])
def run_task(payload: str, workdir: str | None, extra_paths: list[str] | None,
             timeout: int = 300) -> dict[str, str]:
    """Run an LLM-authored snippet in a fresh subprocess on a worker."""
    # A @python_app body runs SYNCHRONOUSLY in a worker process -> must NOT be
    # a coroutine. Use blocking APIs; parsl's executor provides the concurrency
    # (one app per worker). Every import is function-local: parsl ships the body
    # to the worker, where module-level names of the defining file don't exist.
    import os, subprocess, sys, tempfile
    from pathlib import Path

    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)

    # Let the snippet import your package and any helper dirs (e.g. a skill's
    # scripts/). The worker inherits worker_init's env; extend, don't replace.
    env = os.environ.copy()
    # `__file__` here is the DEFINING module's path, captured when parsl
    # serializes the app — not a path on the worker. Fine when the venv is on a
    # shared filesystem (the normal HPC case); pass the root in as an argument
    # if yours is node-local.
    parts = [str(Path(__file__).resolve().parents[1])]          # your src root
    parts += [str(Path(p)) for p in (extra_paths or [])]
    if existing := env.get('PYTHONPATH'):
        parts.append(existing)
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
        # TimeoutExpired carries whatever was captured before the kill — return
        # it, so the harness sees how far the snippet got.
        out = (exc.stdout or b'').decode(errors='replace')
        err = (exc.stderr or b'').decode(errors='replace')
        return {'stdout': out,
                'stderr': f'{err}\nTimed out after {timeout}s and was killed.'.strip(),
                'returncode': '-1'}
    finally:
        Path(script).unlink(missing_ok=True)

class HpcAgent(Agent):
    def __init__(self, num_threads: int, parsl_config: Config | None = None):
        super().__init__()
        self.num_threads = num_threads
        self._parsl_config = parsl_config      # None -> parsl's local default

    async def agent_on_startup(self) -> None:
        # Without this the first @python_app call raises NoDataFlowKernelError.
        self.dfk = parsl.load(self._parsl_config)

    async def agent_on_shutdown(self) -> None:
        try:
            self.dfk.cleanup()
        finally:
            parsl.clear()

    @action
    async def do_work(self, payload: str, workdir: Path | None = None,
                      extra_paths: list[Path] | None = None,
                      timeout: float = 300.0) -> dict[str, str]:
        """Docstring matters: it becomes the MCP tool description the LLM reads."""
        # Calling a @python_app returns an AppFuture (a concurrent Future).
        # wrap_future lets us await it without blocking the agent's event loop,
        # so other actions proceed while parsl runs this on a worker.
        return await asyncio.wrap_future(
            run_task(payload,
                     str(workdir) if workdir else None,
                     [str(p) for p in extra_paths] if extra_paths else None,
                     timeout)
        )
```

Design notes that pay off later:
- **Run untrusted / LLM-authored work in a subprocess**, not in the worker
  process — a segfault, `sys.exit`, or C-extension abort then costs one task, not
  one worker. It also makes the timeout enforceable.
- **Return `{stdout, stderr, returncode}` rather than raising** for task-level
  failure. The harness reads stderr, fixes its input, and retries. Reserve
  exceptions for infrastructure faults.
- Keep the action's **docstring** precise — it is the tool schema description.

## Step 2 — the Parsl scaling config

This is the whole scaling story. One HTEx **manager per node** (`--ppn 1`), each
running `available_accelerators` workers, one pinned per device.

```python
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider

def hpc_config(run_dir, *, nodes: int, accelerators: int | list[str],
               worker_init: str, cpu_workers_per_node: int = 8) -> Config:
    n_gpu = accelerators if isinstance(accelerators, int) else len(accelerators)
    provider = lambda: LocalProvider(
        nodes_per_block=nodes, init_blocks=1, max_blocks=1,
        # LocalProvider + MpiExecLauncher is the canonical ALCF recipe: the job
        # already HAS the nodes, so parsl just mpiexec's one manager onto each.
        launcher=MpiExecLauncher(bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1'),
        worker_init=worker_init,
    )
    return Config(
        run_dir=str(run_dir / 'runinfo'),
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='gpu', provider=provider(),
                max_workers_per_node=n_gpu,
                available_accelerators=accelerators,  # pins one worker per device
                cpu_affinity='block',
                worker_port_range=(10000, 20000),
                worker_debug=True,
            ),
            HighThroughputExecutor(
                label='cpu', provider=provider(),
                max_workers_per_node=cpu_workers_per_node,
                cores_per_worker=1.0,
                worker_port_range=(10000, 20000),
            ),
        ],
    )
```

- `available_accelerators` as an **int** `A` expands to `['0'..'A-1']` and Parsl
  exports the right pinning var per worker — `CUDA_VISIBLE_DEVICES` on NVIDIA,
  `ZE_AFFINITY_MASK` on Intel. Pass an explicit list when the device names are
  not plain integers.
- **GPU workers = nodes × A.** That number, not the node count, is your
  concurrency. 8 Polaris nodes × 4 A100 = 32; 8 Aurora nodes × 12 tiles = 96.
- Two executors let you route: heavy device work to `'gpu'`, cheap CPU work to
  `'cpu'`, without the CPU tasks occupying GPU slots. (Set an app's
  `executors=['cpu']` to route it.)
- `worker_init` is shell run on **each node** before its workers start. It is the
  single most common source of "works on the head node, fails everywhere else" —
  see Step 4.

## Step 3 — the MCP server

Register tools **inside the FastMCP lifespan**, because a tool is bound to an
agent handle that does not exist until the agent is launched. Clients list tools
after startup, so they still see them.

```python
from academy.exchange import HttpExchangeFactory, LocalExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from academy.handle import Handle
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pathlib import Path

# Your domain framing, served as the server's `instructions` so an external
# harness adopts the same persona your own orchestrator would. Keep it in one
# importable place and feed it to BOTH entry points, or they drift into two
# different personas. Replace with your domain:
SYSTEM_PROMPT = (
    'You are a <domain expert> with access to <the tools/libraries in this '
    'environment>. <What the actions are for, and the standards you hold.>'
)

def register(server: FastMCP, handle, output_path):
    async def do_work(payload: str, workdir: str | None = None,
                      timeout: float = 300.0) -> dict[str, str]:
        """<- this docstring is the tool description the LLM sees. Be specific:
        what runs, where, what the return fields mean, what a nonzero
        returncode implies."""
        try:
            return await handle.action('do_work', payload,
                                       workdir=Path(workdir) if workdir else output_path,
                                       timeout=timeout)
        except Exception as exc:
            raise ToolError(str(exc)) from exc   # correctable error, not a crash
    server.add_tool(do_work)

def create_managed_server(*, parsl_config, output_path, num_threads, exchange='local'):
    @asynccontextmanager
    async def lifespan(server: FastMCP):
        factory = (LocalExchangeFactory() if exchange == 'local'
                   else HttpExchangeFactory(exchange, auth_method='globus'))
        async with await Manager.from_exchange_factory(
            factory=factory, executors=ThreadPoolExecutor(),
        ) as manager:
            handle = await manager.launch(
                HpcAgent(num_threads=num_threads, parsl_config=parsl_config))
            # REBIND, do not register `handle` directly. A Handle resolves its
            # ExchangeClient through a ContextVar, and contextvars propagate
            # parent->child only at task creation. An MCP tool call is handled
            # in a DIFFERENT asyncio task than this lifespan, so the lookup
            # raises and every call dies with ExchangeClientNotFoundError.
            # Passing `exchange=` gives _get_client() an explicit fallback.
            bound = Handle(handle.agent_id, exchange=manager.exchange_client)
            register(server, bound, output_path)
            yield bound
    return FastMCP(name='hpc-agent', instructions=SYSTEM_PROMPT,
                   lifespan=lifespan, on_duplicate='replace')
```

- `instructions=` ships your domain framing to the harness, so an external client
  adopts the same persona your own orchestrator would use. Serve the same string
  both places so the two entry points can't drift.
- **Exchange choice:** `LocalExchangeFactory` runs the agent in this process (the
  normal case). Point `exchange` at an academy HTTP exchange URL instead when the
  agent should live elsewhere / be federated (e.g. via Globus Compute) — the
  server then becomes a thin MCP front-end.
- **Transport:** use `http` on HPC. `stdio` only works when the harness itself
  spawns the process, which is not the case across a batch job.

### If you do support stdio, isolate stdout

On stdio the JSON-RPC channel *is* stdout, but the transport only touches
`sys.stdout.buffer`; every stray `print`, Parsl banner, and native writer goes
through `write` or fd 1 and corrupts the protocol. Fix it once, before the
server runs (so the lifespan's noisy startup is covered too):

```python
import os, sys
from typing import Any

class _StdoutShim:
    """`.buffer` stays the real client stream (the MCP transport wraps it);
    every `write` — prints, banners — is diverted to stderr."""
    def __init__(self, real_buffer: Any) -> None:
        self.buffer = real_buffer
    def write(self, s: str) -> int:
        return sys.stderr.write(s)
    def flush(self) -> None:
        sys.stderr.flush()
    def __getattr__(self, name: str) -> Any:
        return getattr(sys.stderr, name)

def isolate_stdout_for_stdio() -> None:
    real = os.fdopen(os.dup(1), 'wb')   # preserve the true client stream FIRST
    os.dup2(sys.stderr.fileno(), 1)     # fd-level writers (native, subprocess) -> stderr
    sys.stdout = _StdoutShim(real)      # .buffer = real stream; .write -> stderr
```

Call it **before** `server.run()`, not per-request — the lifespan's Parsl/academy
startup is the noisiest part and must be covered too.

## Step 4 — the launcher (run ONCE on the head node)

The launcher is the piece that makes the deployment scale. It resolves the
allocation's shape, builds the config, and serves.

```python
import os, socket, subprocess
from pathlib import Path

def count_nodes() -> int:
    nf = os.environ.get('PBS_NODEFILE')                    # Slurm: SLURM_JOB_NUM_NODES
    if nf and Path(nf).exists():
        return len({l.strip() for l in open(nf) if l.strip()}) or 1
    return 1

def routable_ip() -> str:
    # Bind to the node's HSN address, NOT 127.0.0.1, or nothing else can reach it.
    out = subprocess.run(['hostname', '-i'], capture_output=True, text=True).stdout.split()
    return out[0] if out else socket.gethostbyname(socket.gethostname())

def default_worker_init() -> str:
    """Reproduce THIS environment on every remote node's HTEx manager.
    MpiExecLauncher starts each manager in a fresh login shell — modules, venv,
    env vars and cwd do not carry over."""
    lines = [
        'export TMPDIR=/tmp',                 # MUST be first — see pitfalls
        'module use /soft/modulefiles',
        f'module load {os.environ.get("HPC_MODULE", "cudatoolkit-standalone/12.4.1")}',
    ]
    if venv := os.environ.get('VIRTUAL_ENV'):
        lines.append(f'source {venv}/bin/activate')   # AFTER module load, so its python wins
    if amber := os.environ.get('AMBERHOME'):
        lines += [f'export AMBERHOME={amber}', 'export PATH="$PATH:$AMBERHOME/bin"']  # APPEND
    cwd = os.getcwd()
    lines.append(f'cd {cwd}')
    # `cd` does NOT make your code importable. cloudpickle serializes a
    # @python_app defined in an importable module BY REFERENCE, so each worker
    # must `import` that module or every task fails with ModuleNotFoundError.
    # Only skip this if your package is pip-installed into the venv.
    lines.append(f'export PYTHONPATH={cwd}:$PYTHONPATH')
    return '\n'.join(lines)

def main():
    nodes  = int(os.environ.get('HPC_NODES', count_nodes()))
    accel  = int(os.environ.get('HPC_ACCEL', 4))
    threads= int(os.environ.get('HPC_THREADS', 8))
    out    = Path(os.environ.get('HPC_OUTPUT', './execution_output')); out.mkdir(parents=True, exist_ok=True)
    host   = os.environ.get('HPC_HOST') or routable_ip()
    port   = int(os.environ.get('HPC_PORT', 8000))
    cfg = hpc_config(out, nodes=nodes, accelerators=accel,
                     worker_init=os.environ.get('HPC_WORKER_INIT') or default_worker_init(),
                     cpu_workers_per_node=threads)
    print(f'[launcher] {nodes} node(s) x {accel} = {nodes*accel} GPU workers')
    print(f'[launcher] MCP -> http://{host}:{port}/mcp')     # grep this in the log
    create_managed_server(parsl_config=cfg, output_path=out,
                          num_threads=threads).run(transport='http', host=host, port=port)
```

Make every knob an **environment variable with a sane default**: an MCP client
launching a server, and a PBS script, can both only reach the process through
argv/env.

## Step 5 — the batch script

```bash
#!/bin/bash -l
#PBS -A <allocation>
#PBS -N agent-mcp
#PBS -l select=8                     # N nodes  (Polaris: select=8:system=polaris)
#PBS -l walltime=01:00:00
#PBS -l filesystems=<home:grand|flare>
#PBS -l place=scatter
#PBS -q <debug|debug-scaling|prod>
#PBS -j oe
# NOT `set -u`: Lmod's bash init dereferences ZSH_EVAL_CONTEXT unbound, so the
# FIRST `module` call aborts the whole script. Verified on Aurora.
set -o pipefail

PROJECT_DIR=/lus/<fs>/projects/<proj>/<you>/<repo>
cd "$PROJECT_DIR"
module use /soft/modulefiles
module load <cudatoolkit-standalone/12.4.1 | frameworks>
export TMPDIR=/tmp
source "$PROJECT_DIR/.venv/bin/activate"

export HPC_ACCEL=4          # devices per node: Polaris 4, Aurora 12
export HPC_THREADS=8        # OMP threads per worker / cpu-executor workers per node
export HPC_OUTPUT=./execution_output

# ONCE, on the head node. Parsl owns the mpiexec fan-out over $PBS_NODEFILE.
python launch_scaling_mcp.py > agent-mcp.log 2>&1
```

Then `qsub`, `qstat -u $USER`, and `tail -f agent-mcp.log` until you see the
`MCP -> http://<hsn-ip>:8000/mcp` line and Uvicorn's `Application startup
complete`.

## Step 6 — connect the harness

```json
{"mcpServers": {"hpc-agent": {"transport": "http", "url": "http://<hsn-ip>:8000/mcp"}}}
```

Drive it from **another job on the interconnect**, not from a login node. There
is no auth — keep it inside the facility network, and treat the tool as
arbitrary code execution under your allocation.

Throughput comes from **concurrency at the harness**: issue many independent tool
calls in flight and let Parsl place them. One serial call at a time uses one GPU
regardless of how many you allocated.

## Step 7 — keep the pool alive (long runs only)

**Applies to the inverted deployment**, the one worth knowing about for
multi-day work. Steps 1–6 put the MCP server *inside* the allocation, so the
server dies with the job. Invert it: run the MCP server **locally** (on your
laptop, under the harness), and have it launch a *pool* of `HpcAgent`s onto a
**Globus Compute endpoint** whose own Parsl template owns the PBS job. Use
`HttpExchangeFactory(url, auth_method='globus')` instead of the local exchange,
and hold `self.pools[name] -> list[Handle]`, one handle per pinned accelerator,
dispatching round-robin. The harness now survives job death, walltime is no
longer service lifetime, and the endpoint re-queues on its own.

The price is that the pool is a long-lived distributed object you now have to
maintain — which is what this step is. **Nothing above keeps a pool alive.** On
a multi-day campaign it decays, and it decays *silently*: the endpoint reports
online, the agent count looks right, and work simply stops being done. Three
mechanisms, in the order they bite. All three were observed on a 96 h Polaris
run; the first cost ~73 h of a reservation.

**The root cause is that the exchange credential never refreshes.**
`HttpExchangeFactory.__init__` materialises a bearer token *once*:

```python
additional_headers |= get_auth_headers(auth_method)   # resolved ONCE, held forever
```

There is no refresh, even with a valid refresh token sitting in the local store.
When it lapses the client starts failing with `ForbiddenError: Exchange entity
UserId<…> is not authorized to access this resource` — wording that points at
*permissions*, so the natural first hypothesis is a misconfigured identity. It
is not. It is expiry. Worse, **every agent carries the same credential**, so the
entire pool detaches at once, and each agent keeps holding its Parsl worker slot
forever. The allocation is then sterile: nothing can be placed, no orphan can be
asked to exit (asking needs the exchange it just fell off), and only killing the
PBS job frees the nodes.

### 7a. Agents retire themselves

The agent cannot renew its credential, so give it a lifetime shorter than the
credential's and let it stand down cleanly. `agent_shutdown()` signals the
runtime's shutdown event, the task completes, and **Parsl reclaims the slot**.

```python
    async def agent_on_startup(self) -> None:
        ...                                       # existing setup
        self._inflight = 0
        self._watchdog = asyncio.create_task(self._lifetime_watchdog())

    async def _lifetime_watchdog(self) -> None:
        import asyncio, os
        ttl = float(os.environ.get('ACADEMY_AGENT_TTL_S', 30 * 3600))
        if ttl <= 0:                              # 0 disables, for debugging
            return
        await asyncio.sleep(ttl)
        # Wait for quiescence. Recycling that kills a running calculation trades
        # one failure mode for another; a long job finishing is worth more than
        # punctual retirement.
        while self._inflight > 0:
            await asyncio.sleep(30)
        self.agent_shutdown()

    async def agent_on_shutdown(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        ...
```

Increment `self._inflight` at the top of every `@action` that does real work and
decrement it in a `finally`.

**Pick the TTL against the credential, not against intuition.** A 6 h default
looked prudent and was badly wrong: with a ~48 h token it retired agents eight
times per allocation instead of once. **Self-retirement without 7b and 7c is
just a scheduled outage** — capacity halves on a timer whether or not anything
is wrong.

### 7b. Adopt before launching

Handles live in the client's process memory; agents live on the compute nodes.
A client restart loses the handles but not the agents — which keep running and
keep holding every slot. Launch unconditionally and the new agents queue behind
orphans forever, and **each restart makes it worse**.

But a `Handle` is only `(agent_id, exchange_client)`, and `discover()` returns
the ids. **A living agent is never lost, only unaddressed.**

```python
async def adopt_live_agents(self, name, base_dir, want):
    from academy.handle import Handle
    client = self.manager.exchange_client
    adopted = getattr(self, '_adopted', set()); self._adopted = adopted
    ids = await client.discover(self.agent_class, allow_subclasses=True)

    async def check(aid):
        try:                                      # probe CONCURRENTLY: after a few
            h = Handle(aid, exchange=client)      # restarts most candidates are
            await asyncio.wait_for(h.action('probe'), timeout=20)
            return aid, h                         # stale and will time out
        except Exception:
            return None

    live = [r for r in await asyncio.gather(
        *(check(a) for a in ids if str(a) not in adopted)) if r]
    # Filter by working directory, or one campaign adopts another's agents.
    chosen = [(a, h) for a, h in live][:max(0, want)]
    for aid, _ in chosen:
        adopted.add(str(aid))
    return [h for _a, h in chosen]
```

**Distinguishing endpoints by `base_dir` is not enough if two endpoints share
one.** That happened: a queued endpoint with *zero nodes* adopted a running
endpoint's agents and reported `live: true, agents: 8`. Real agents, wrong
endpoint, and a completely wrong capacity decision downstream.

### 7c. Re-validate and top up

A pool is not durable. Cache it at first use and never check it again and every
retirement becomes permanent lost capacity.

```python
POOL_REVALIDATE_S = float(os.environ.get('GCMCP_REVALIDATE', '600'))

# in ensure_pool, before the fast-path return:
if self.pools.get(name) and (time.monotonic() - checked.get(name, 0)) < POOL_REVALIDATE_S:
    return self.pools[name]
...
# under the lock: prune dead handles, THEN top up
pool = [h for h in await asyncio.gather(*(alive(h) for h in prior)) if h]
if len(pool) < want:
    pool = pool + await self.adopt_live_agents(name, base_dir, want - len(pool))
for k in range(len(pool), want):
    ...                                           # launch the remainder
if pool:
    self.pools[name] = pool
    checked[name] = time.monotonic()
else:
    self.pools.pop(name, None)                    # do NOT stamp an empty pool:
                                                  # it suppresses revalidation and
                                                  # makes "slow to start" look dead
```

Two mistakes worth avoiding, both made while writing this: **extend the pruned
survivors, never overwrite them** (overwriting discards live agents and then
tries to launch into the slots they still hold), and **ask adoption only for the
shortfall**.

### Knobs

| Variable | Default | Set it by |
|---|---|---|
| `ACADEMY_AGENT_TTL_S` | `108000` (30 h) | the credential's life, minus a margin. `0` disables. Read **on the worker**, so it must be in the endpoint's env, not the client's. |
| `GCMCP_REVALIDATE` | `600` s | how fast you need a lost agent noticed, against the cost of a probe round. |
| `GCMCP_ADOPT_TIMEOUT` | `20` s | round-trip to the exchange plus agent scheduling latency. Too low and you orphan live agents; too high and a restart stalls on the stale ones. |

### Telling "wedged" from "busy" — the diagnostic that actually works

This is the hardest part and every convenient signal lies:

| Signal | What it looks like when wedged |
|---|---|
| endpoint status | `online` |
| pool agent count | full |
| worker logs | quiet — they only log task *receipt* |
| PBS `cput` | tiny; it counts the mother process, not remote workers |
| task state | `pending` forever, indistinguishable from queued |

**Only two things are reliable.** `failed_infra` versus `pending` in a poll
result: dead agents return the former *immediately*, busy ones return the
latter. And a **direct load measurement on the node** — run
`open('/proc/loadavg').read()` through a live agent. A node at `load 0.00` with
"running" tasks is wedged, full stop. If even that is inconclusive, find the
process: `ps` for the script name and check `%CPU` and state.

## Pitfalls (each of these was a real, silent failure)

The first four were found by deploying this skill on Aurora; each one killed a
real job. They are listed first because they fire before anything else can.

- **`set -u` kills `module`.** Lmod's `/usr/share/lmod/lmod/init/bash`
  dereferences `ZSH_EVAL_CONTEXT` without a default, so under `set -u` the FIRST
  `module` call aborts the entire batch script — exit 1 at ~4 seconds, with only
  `ZSH_EVAL_CONTEXT: unbound variable` in the log. Use `set -o pipefail` alone
  (or `set +u` around the module block).
- **A lifespan-created Handle cannot be called from a tool.** `Handle` resolves
  its `ExchangeClient` from a **ContextVar**, and contextvars propagate to child
  tasks only at creation time. The FastMCP tool coroutine runs in a different
  task, so the lookup fails and every call raises
  `ExchangeClientNotFoundError` — *after* `list_tools` has already succeeded,
  which makes it look like a tool bug rather than a wiring bug. Rebind with
  `Handle(handle.agent_id, exchange=manager.exchange_client)` before registering.
- **`cd` in `worker_init` does not make code importable.** cloudpickle pickles a
  `@python_app` living in an importable module **by reference**, so workers must
  `import` it; without `PYTHONPATH` every task returns
  `ModuleNotFoundError: No module named '<your module>'`. Add
  `export PYTHONPATH=<projdir>:$PYTHONPATH`, or pip-install your package into
  the venv (which is why this never bites an installed package).
- **An app with no `executors=` runs on the wrong executor.** Parsl defaults to
  `executors='all'`. With a two-executor config your GPU app also lands on `cpu`
  workers that have no accelerator pinning, so a fraction of "GPU" tasks
  silently run unpinned — on Aurora a 96-call fan-out hit 26 distinct workers
  (24 pinned tiles + 2 CPU workers reporting `ZE_AFFINITY_MASK` unset). Always
  pass `executors=['gpu']`.

- **`TMPDIR` overruns the AF_UNIX limit.** PBS sets
  `TMPDIR=/var/tmp/pbs.<jobid>.<fqdn>` (~60+ chars). The HTEx worker pool's
  multiprocessing `SyncManager` socket lives under it and blows past the ~108-char
  AF_UNIX path limit: every worker dies before registering with
  `OSError: AF_UNIX path too long`, surfacing at the client as a bare `EOFError`.
  **`export TMPDIR=/tmp` first in `worker_init`** (and in the batch script).
- **Never *prepend* a bundled toolchain to `PATH`.** `$AMBERHOME/bin` (and
  friends) ship their own `python` *and* their own OpenMPI `mpiexec`. Prepending
  shadows (1) the venv python → `ModuleNotFoundError` for your package on the
  workers, and (2) the Cray **PALS** `mpiexec` that `MpiExecLauncher` needs —
  OpenMPI rejects `--cpu-bind`, every block exits 213, and Parsl reports
  `Job is marked as MISSING since the workers failed to register`. **Append.**
- **The single-GPU trap.** A default/local Parsl config caps the GPU executor at
  one worker. `qsub`-ing your plain server module therefore burns a whole node to
  use one device. The launcher exists solely to attach the scaling config —
  if a run's throughput looks like 1/A of expectation, this is why.
- **Do not wrap the launcher in `mpiexec`.** Parsl already fans out over
  `$PBS_NODEFILE`. Wrapping gives you N duplicate servers fighting over the port.
- **`worker_init` must rebuild the environment.** Managers start in fresh login
  shells; if the module / venv / cwd / device-hierarchy vars aren't re-established
  there, imports fail **only on the non-head nodes** — a partial failure that
  reads like a flaky machine. If you override `worker_init` wholesale, re-apply
  the `TMPDIR` and append-PATH fixes yourself.
- **The HSN address is not reachable from a login node.** `hostname -i` gives a
  routable compute-network IP (`10.x.x.x`). A login-node client just times out.
- **The site proxy answers `503` for internal IPs.** ALCF compute nodes export
  `http_proxy`/`https_proxy=http://proxy.alcf.anl.gov:3128` with a `no_proxy`
  covering *hostnames* but not raw HSN IPs, so an httpx/MCP client routes through
  the proxy and gets `503 Service Unavailable`. It looks like a server bug; the
  server log shows a clean startup. Fix in the **client** shell:
  `export no_proxy='*'`.
- **Thread oversubscription.** accelerators/node × threads-per-worker must stay
  under the node's hardware threads (Polaris 64, Aurora 208). 4×8=32 and 12×8=96
  leave headroom; 12×32 does not.
- **Walltime is the service lifetime.** When the job ends the MCP endpoint
  vanishes mid-call. Size the walltime to the campaign, keep artifacts on the
  parallel filesystem (not `TMPDIR`), and make the harness resilient to a dead
  endpoint.
- **The exchange token never refreshes, and its expiry sterilises the
  allocation.** Resolved once in `HttpExchangeFactory.__init__` and held for the
  transport's life. Every agent inherits it, so the whole pool detaches
  simultaneously and each orphan keeps its worker slot forever. Surfaces as
  `ForbiddenError: … is not authorized`, which reads like a permissions problem
  and is not. See Step 7.
- **A restart leaks worker slots unless the client adopts first.** Orphaned
  agents keep running and keep their slots; launching unconditionally queues the
  replacements behind them permanently, and every restart compounds it. Observed
  going 8 → 4 → 2 reachable workers across three restarts. See Step 7b.
- **Self-retirement without replacement is a scheduled outage.** If agents stand
  down on a TTL but nothing refills the pool, capacity halves on a timer. Both
  halves are required. See Step 7a and 7c.
- **`submit_batch` round-robins into dead handles.** With *k* dead of *n*, it
  silently loses *k/n* of everything submitted — no error, no timeout, tasks
  pending forever. At 6 dead of 8 that is three quarters of the work. Until the
  pool is healthy, pin explicitly to verified `agent_index` values.
- **Everything a harness can see reports health while the pool is wedged** —
  endpoint online, agent count full, worker logs quiet, tasks `pending`. Trust
  `failed_infra`-vs-`pending`, or a direct `/proc/loadavg` read through a live
  agent. Nothing else.

## Verification ladder — climb it before trusting a big job

1. **Config shape**, on a login node:
   ```python
   cfg = hpc_config(Path('/tmp/t'), nodes=8, accelerators=4, worker_init='x')
   for ex in cfg.executors:
       print(ex.label, ex.max_workers_per_node,
             getattr(ex, 'available_accelerators', None), ex.provider.nodes_per_block)
   # gpu 4 ['0','1','2','3'] 8
   ```
2. **Synthesized `worker_init`** — print it and read the ordering (TMPDIR first,
   module before venv, PATH appended).
3. **Server end-to-end, local config**: run with `--transport http --host
   127.0.0.1`, then from an MCP client `list_tools()` → your tool, and a trivial
   call → expected stdout.
4. **Single node, real machine** (`debug` queue, 1 node): proves modules, venv,
   device visibility, and the HTTP bind.
5. **Two-plus nodes** (`debug-scaling`): the only test that exercises
   `worker_init` on a remote node. Confirm the log's worker count equals
   nodes × accelerators before scaling to production.

## Site profiles and porting

Per-machine node facts, modules, queues, and the Slurm variant are in
[references/site-profiles.md](references/site-profiles.md) — read it when
targeting Polaris, Aurora, or a non-ALCF cluster.
