---
name: globus-compute-mcp
description: >-
  Drive HPC compute (ALCF Aurora / Polaris) from an LLM harness running on a
  laptop, with NO SSH per call, no tunnel, no inbound port and no MFA prompt.
  A FastMCP server runs LOCALLY and dispatches work to stateful academy agents
  living on Globus Compute endpoints, which dial outbound to the Globus service
  and acquire PBS allocations on demand. Use when you want Claude Code / Codex /
  any MCP client to run folding, MD or analysis on a supercomputer while the
  harness stays outside the facility network; when a previous in-allocation MCP
  server was unreachable or died at walltime; or when you need a model loaded
  once in GPU memory and reused across many tool calls. Covers the inversion
  rationale, the local/globus exchange modes, the endpoint YAMLs, the
  submit-then-poll pattern, a nine-rung verification ladder, and the failure
  modes that kill this stack silently (python/dill version skew SEGFAULT, a
  guessed interface name that hangs forever, AF_UNIX TMPDIR, PATH shadowing).
license: Apache-2.0
metadata:
  verified: >-
    Rungs 1 and 3 PASS on the laptop 2026-07-27 (python 3.11.11, academy 0.5.0,
    globus-compute-sdk 4.15.0, fastmcp 3.4.4, dill 0.3.9): 18 assertions
    including full MCP tool path, stateful sessions, 8-way concurrency
    (5.6 s vs 40 s serial), stdio isolation under PYTHONUNBUFFERED=1. Rung 2
    FAILS on expired Globus tokens; rungs 4-9 BLOCKED on the one-time bootstrap
    (ALCF SSH down). Design ported from ramanathanlab/deepdrivewe-academy.
---

# Driving HPC from a local MCP server via Globus Compute

## The architecture — the inversion is the whole point

```
LAPTOP                                          HPC (Aurora / Polaris)
┌────────────────────────────┐                 ┌──────────────────────────┐
│ MCP server (LOCAL process) │                 │ globus-compute-endpoint  │
│  FastMCP, transport=stdio  │                 │  (persistent daemon on   │
│  tools: run_python,        │  ── HTTPS ───►  │   a LOGIN node)          │
│         submit_batch,      │   Globus svc +  │        ↓                 │
│         poll, get_result   │   Academy       │  PBSProProvider → PBS    │
│         ↓                  │   Exchange      │        ↓                 │
│  academy Manager           │  ◄── results ─  │  HpcWorkerAgent pool     │
│   executors={              │                 │  (STATEFUL, 1 process    │
│     'aurora-gpu': GCExec,  │                 │   per worker, 1 worker   │
│     'aurora-cpu': GCExec}  │                 │   per accelerator)       │
└────────────────────────────┘                 └──────────────────────────┘
```

**Why this and not an MCP server inside the allocation.** The `academy-mcp-hpc`
skill runs FastMCP *inside* a batch job. That assumes the MCP client is also
inside the facility network — its own docs say to "drive it from another job on
the interconnect, not from a login node," because the HSN IP is unroutable
externally. With the harness on a laptop that design fails three ways:

1. **the HSN address is unreachable from outside** the facility;
2. **walltime becomes the service lifetime** — the endpoint vanishes mid-call
   when the job ends;
3. **the ALCF proxy returns 503** for raw internal IPs.

**Inverting it fixes all three.** A Globus Compute endpoint dials *outbound* to
the Globus service; the laptop client talks to that service over HTTPS with
refreshable OAuth tokens. No inbound port, no tunnel, no MFA per call. The MCP
server is local, so it is always up and has **no walltime**. PBS jobs become
*a resource the endpoint acquires*, not the thing hosting the service.

`service/launch_scaling_mcp.py` and `service/run_mcp_service.pbs` implement the
older in-allocation design. They are **kept, not deleted** — both carry a LEGACY
header. They remain usable from another job on the interconnect. They are not
the path a laptop uses.

## Stateful agents, not bare function submission

Work is done by **academy `Agent`s launched onto Globus Compute executors**,
addressed by handle — the pattern from
[`ramanathanlab/deepdrivewe-academy`](https://github.com/ramanathanlab/deepdrivewe-academy)
(`examples/openmm_ntl9_hk_multisite/`):

```python
from globus_compute_sdk import Executor as GCExecutor

executors = {'aurora-gpu': GCExecutor(endpoint_id=gpu_uuid),
             'aurora-cpu': GCExecutor(endpoint_id=cpu_uuid)}

async with await Manager.from_exchange_factory(
        factory=create_exchange_factory(args.exchange),   # Local | HttpExchange
        executors=executors,
        default_executor='aurora-gpu') as manager:
    handle = await manager.launch(HpcWorkerAgent, executor='aurora-gpu', ...)
```

The agent is a **long-lived process**, so anything expensive it loads stays
loaded: a Chai-1 model in GPU memory, an OpenMM `Simulation`, a parsed
trajectory. Stateless function submission re-imports and re-loads on *every*
call — for a fold battery that is most of the runtime. `run_python(...,
stateful=True)` executes against a namespace that persists between calls.

**A pool, not one agent.** One academy agent is one action stream on one worker
slot, so a single agent can never use more than one accelerator no matter how
big the allocation. The reference launches one simulation agent per walker for
exactly this reason. Here each endpoint gets a pool sized to its worker count,
and `submit_batch` round-robins across it — that is what fans a sweep over every
GPU. This was found the hard way: see *Pitfalls*.

## ⚠ Python version must match on every host — a skew SEGFAULTS

Globus Compute serializes tasks and results with `dill`. The reference README
says mismatched versions "cause deserialization failures". **It is worse than
that.** Measured on this laptop, `globus_compute_sdk.serialize.ComputeSerializer`:

| serialize | deserialize | result |
|---|---|---|
| 3.13.7 | 3.11.11 | **exit 139 (SIGSEGV)** — no exception, no traceback, no message |
| 3.11.11 | 3.13.7 | **exit 139 (SIGSEGV)** — no exception, no traceback, no message |
| 3.11.11 | 3.11.11 | works |

A segfault with no Python-level error is attributed to a bad node, a flaky
filesystem, anything but a version skew. **Pin one minor version everywhere:**
orchestrator, every endpoint venv, every worker. This stack pins **3.11**
(`REQUIRED_PY` in `verify.py`; asserted by rung 1, by `endpoint_status`, by rung
4, and by a `python -c 'assert sys.version_info[:2]==(3,11)'` at the end of every
endpoint's `worker_init`).

The laptop venv is `.venv-gc311/` (python 3.11.11). The pre-existing `.venv-gc/`
is **3.13.7 and must not be used for this stack** — it is left in place because
earlier campaign artifacts reference it.

## Two modes — build and test in `local` first

```bash
# no Globus, no endpoint, no SSH: orchestrator, exchange and agents in-process
./.venv-gc311/bin/python skills/globus-compute-mcp/server.py --exchange local

# the real thing: agents launched onto Globus Compute endpoints
./.venv-gc311/bin/python skills/globus-compute-mcp/server.py --exchange globus
```

`--exchange local` uses `LocalExchangeFactory` + `ThreadPoolExecutor`s and
exercises **the entire tool path** — Manager wiring, handle rebinding, `base_dir`
chdir, stateful sessions, submit-then-poll, concurrency. Use it for every change
before touching HPC; it is rung 3 of the ladder and needs nothing but the laptop.

## Files

| File | Role |
|---|---|
| `skills/globus-compute-mcp/server.py` | the LOCAL FastMCP server (stdio) |
| `skills/globus-compute-mcp/hpc_agents.py` | `HpcWorkerAgent` — runs ON the endpoint |
| `skills/globus-compute-mcp/endpoints.yaml` | task-class registry: name → UUID + paths |
| `skills/globus-compute-mcp/verify.py` | the nine-rung ladder |
| `service/gc_endpoint_config.yaml` | Aurora **GPU** endpoint (12 tiles/node) |
| `service/gc_endpoint_config_aurora_cpu.yaml` | Aurora **CPU** endpoint (analysis) |
| `service/gc_endpoint_config_polaris.yaml` | Polaris **GPU** endpoint (4 A100/node) |
| `service/bootstrap_aurora.sh` | one-time, idempotent, per-endpoint bootstrap |
| `service/stage_to_aurora.sh` | laptop → HPC staging (includes `hpc_agents.py`) |
| `service/launch_scaling_mcp.py`, `run_mcp_service.pbs` | **LEGACY**, in-allocation design |

Endpoints are named by **task class**, not just machine — `aurora-gpu`,
`aurora-cpu`, `polaris-gpu` — mirroring the reference's split between a GPU
simulation endpoint and a CPU inference endpoint. Send folding and MD to `*-gpu`;
send trajectory analysis, SASA/patch computation and statistics to `*-cpu`, which
holds no GPU allocation. Tool calls say `endpoint='aurora-gpu'`, never a UUID.

## One-time bootstrap — the only human steps

Everything else is scripted. These two need a person:

```bash
# [HUMAN 1] authenticate SSH with an MFA token
ssh aurora

# stage + bootstrap, from the LAPTOP (scripted; stops before the browser step)
bash service/stage_to_aurora.sh

# [HUMAN 2] authenticate the endpoint in a browser, ON the login node
ssh aurora
globus-compute-endpoint start aurora-gpu-s-elongatus
#   prints https://auth.globus.org/v2/oauth2/authorize?client_id=...
#   The URL is PKCE-bound to THAT process, so it cannot be pre-generated and
#   handed over — the user must run the command and use the URL it prints.
globus-compute-endpoint list          # copy the UUID
```

Paste the UUID into `skills/globus-compute-mcp/endpoints.yaml` under
`endpoints.aurora-gpu.uuid`. Repeat for `aurora-cpu` / `polaris-gpu` as needed
(`bash bootstrap_aurora.sh aurora-cpu`).

Then, from the laptop with **no SSH**:

```bash
./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --endpoint aurora-gpu
```

After this, no SSH is needed to run compute again — the endpoint holds its own
refreshable tokens and reconnects by itself.

## Registering the server

Project-scoped `.mcp.json` (already written to this project):

```json
{
  "mcpServers": {
    "hpc": {
      "command": "/Users/msinclair/agent/s_elongatus/.venv-gc311/bin/python",
      "args": ["/Users/msinclair/agent/s_elongatus/skills/globus-compute-mcp/server.py",
               "--exchange", "globus"],
      "env": {"PYTHONUNBUFFERED": "1"}
    },
    "hpc-local": {
      "command": "/Users/msinclair/agent/s_elongatus/.venv-gc311/bin/python",
      "args": ["/Users/msinclair/agent/s_elongatus/skills/globus-compute-mcp/server.py",
               "--exchange", "local"],
      "env": {"PYTHONUNBUFFERED": "1"}
    }
  }
}
```

Use the **absolute** path to `.venv-gc311/bin/python`; the venv is not activated
when an MCP client spawns the process.

## Tool reference

| Tool | Signature |
|---|---|
| `list_endpoints` | `() → {exchange_mode, default, endpoints:[{endpoint, live, agents, task_class, machine, accelerators_per_node, …}]}` |
| `endpoint_status` | `(endpoint=None) → {agent{hostname,pid,cwd,python_full,…}, service_status, PYTHON_VERSION_MISMATCH?}` |
| `run_python` | `(code, endpoint=None, timeout=300, wait=60, stateful=False, agent_index=None, workdir=None, extra_paths=None, env=None) → {stdout, stderr, returncode, hostname, pid, ZE_AFFINITY_MASK, CUDA_VISIBLE_DEVICES, elapsed_s, mode, task_id}` |
| `submit_batch` | `(specs, endpoint=None, timeout=300, stateful=False, workdir=None, extra_paths=None, env=None) → {endpoint, count, tasks:[{label,task_id}], task_ids}` |
| `get_result` | `(task_id) → {pending, status, stdout, stderr, returncode, …}` |
| `poll` | `(task_ids, summary_only=False) → {total, pending, succeeded, failed_task, failed_infra, results|buckets}` |
| `session_info` | `(endpoint=None, agent_index=0) → {session_names, host, cwd, python, packages}` |
| `clear_session` | `(endpoint=None, agent_index=0) → {cleared}` |

**Task failure vs infrastructure failure** is the key distinction, and the tools
keep it: a snippet that raises returns `returncode != 0` with a real traceback in
`stderr` (filename and line number intact, because the snippet is written to a
real temp file, not passed to `python -c`). That is a *result* to read and fix.
An `error` field, or a `ToolError`, means the endpoint/agent/serialization
failed. `poll` counts them separately as `failed_task` and `failed_infra`.

## The submit-then-poll pattern

**A tool that blocks on a 6-hour MD run is useless** — MCP clients time out, HPC
jobs take hours. Design for submit-then-poll from the start.

```python
# 1. fan out. ONE call, many jobs in flight -> uses the whole allocation.
b = submit_batch(
    specs=[{'code': f'...fold({seq!r})...', 'label': seq_id}
           for seq_id, seq in candidates.items()],
    endpoint='aurora-gpu', timeout=3600)

# 2. poll. Never blocks; summary_only keeps large sweeps out of context.
s = poll(b['task_ids'], summary_only=True)
# {'total': 96, 'pending': 96, 'succeeded': 0, 'failed_task': 0, 'failed_infra': 0}

# ... do other work, come back later ...
s = poll(b['task_ids'], summary_only=True)
# {'total': 96, 'pending': 0, 'succeeded': 94, 'failed_task': 2, 'failed_infra': 0}

# 3. read only what you need
for tid in poll(b['task_ids'], summary_only=True)['buckets']['failed_task']:
    print(get_result(tid)['stderr'])      # real traceback, fix, resubmit
```

**Throughput comes from concurrency at the client.** One serial `run_python`
call uses one worker no matter how many nodes are allocated. Aurora GPU: 12
workers/node. Polaris GPU: 4/node. `submit_batch` round-robins across the agent
pool; a loop of `run_python` calls does not.

## The stateful pattern

```python
# pay the load ONCE, pinned to one agent
run_python(code='import chai_lab; MODEL = load_model(); print("loaded")',
           endpoint='aurora-gpu', stateful=True, agent_index=0, wait=600)

# every later call reuses the resident model
run_python(code='print(MODEL.fold(SEQ))',
           endpoint='aurora-gpu', stateful=True, agent_index=0)

session_info(endpoint='aurora-gpu')   # -> session_names: ['MODEL', ...]
clear_session(endpoint='aurora-gpu')  # frees the GPU memory
```

Stateful calls **pin to one agent** (`agent_index`, default 0) because the
namespace lives in that one process; a round-robined stateful call would land on
a different namespace. Use distinct indices to keep several models resident.
Stateful is **not isolated**: a segfault kills the agent and everything it holds,
and `timeout` does not apply. Default (`stateful=False`) runs in a fresh
subprocess and is safe.

## Path rules — a whole class of silent bugs

From the reference implementation, and all four apply here:

- Each agent has a **`base_dir`** and **chdirs to it on startup**, so relative
  paths resolve on *that* host.
- **Avoid absolute paths leaking between hosts.** Config objects are pickled
  across network boundaries without `.resolve()` normalisation — a laptop path
  arrives intact on the HPC side and silently means nothing.
- Files needed by more than one host must resolve to the **same path on every
  host**, or be pre-staged.
- **The orchestrator never opens remote files.** Print what you need and read it
  from `stdout`; results travel back as objects through the exchange.

The MCP server therefore never `mkdir`s a remote `base_dir` — only the local-mode
one, where orchestrator and agent are the same host.

## Verification ladder

`skills/globus-compute-mcp/verify.py`. Every rung prints real values.

| # | Rung | Needs |
|---|---|---|
| 1 | imports, versions, **python 3.11 pin** | laptop |
| 2 | Globus Auth token validity | laptop + network |
| 3 | **`--exchange local`: the full MCP tool path** | laptop |
| 4 | `endpoint_status` + orchestrator/agent version match | endpoint |
| 5 | trivial call → compute-node hostname | a PBS block |
| 6 | environment probe (numpy / MDAnalysis / OpenMM), worker `TMPDIR` | " |
| 7 | accelerator pinning + visible device count | " |
| 8 | concurrency: N tasks → N distinct workers, wall ≈ serial/N | " |
| 9 | stateful session survives between calls | " |

```bash
./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --max-rung 3   # today
./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --endpoint aurora-gpu
```

⚠ **Rungs 3 and 8 sleep 5 s per probe, deliberately.** The previous campaign got
a **false negative** here: the probe returned in microseconds, so every task
finished before the next was dispatched and 8 concurrent calls reported ONE
worker on a perfectly healthy system. **An instantaneous task cannot demonstrate
parallelism.** `--sleep 0` is rejected outright.

## Runbook

```bash
# health, from the LAPTOP, no SSH
endpoint_status(endpoint='aurora-gpu')       # via the MCP tool
./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --endpoint aurora-gpu --max-rung 5

# health, on the login node (needs SSH)
globus-compute-endpoint list                 # Running / Stopped
qstat -u $USER                               # the PBS blocks it queued
tail -f ~/.globus_compute/aurora-gpu-s-elongatus/endpoint.log

# restart (the UUID is STABLE across restarts — no client change needed)
globus-compute-endpoint stop  aurora-gpu-s-elongatus
globus-compute-endpoint start aurora-gpu-s-elongatus

# survive a login-node drain
setsid nohup globus-compute-endpoint start aurora-gpu-s-elongatus \
    > ~/gce-start.log 2>&1 < /dev/null &

# scale up for a campaign: edit ~/.globus_compute/<ep>/config.yaml, then restart
#   queue: capacity          # Aurora: 1-16 nodes, up to 168 h, 2 running/user
#   walltime: 24:00:00
#   nodes_per_block: 8       # 8 x 12 = 96 GPU workers
#   max_blocks: 2
# and raise `agent_pool` for that endpoint in endpoints.yaml to match.
```

**Shutdown semantics.** Ctrl+C on the orchestrator closes the `GCExecutor`s, and
**remote agents terminate when the exchange connection drops.** That is the clean
shutdown path — and also a hazard: a dropped laptop connection kills the remote
agents and everything resident in their memory. Long work should write artifacts
to the parallel filesystem, not hold them only in a stateful session.

**Open risk — checkpoint resume across the exchange.** The reference implementation
documents that *checkpoint resume in multi-site workflows is not fully supported*;
single-host mode resumes correctly, but multi-site resume dispatches fresh work
rather than restoring state. Our MD runs are long and checkpointed, so **do not
assume resume works across the exchange** — verify it explicitly before relying
on it.

## Pitfalls

Each of these was a real failure in this campaign, not a hypothetical.

**Version, serialization and startup**

- **A python minor-version skew SEGFAULTS instead of erroring.** Measured: a
  `ComputeSerializer` blob written under 3.13.7 and read under 3.11.11 exits
  **139 with no traceback at all**, and vice versa. Pin one version everywhere.
- **An academy agent occupies a worker slot for its entire life.** With
  `init_blocks: 0` / `min_blocks: 0` there are **zero workers at startup**, so
  launching a 12-agent pool eagerly would block server startup behind the PBS
  queue and time out the MCP handshake. Pools are therefore launched **lazily,
  on first use** of an endpoint. Academy also warns `Executor overload:
  submitted agents exceeds worker count` when the pool outruns the executor.
- **One agent cannot use more than one accelerator.** A single agent is one
  action stream; sending a sweep to it serializes the sweep. Verified in rung 3:
  8 × 5 s tasks took **40.5 s** (perfectly serial) before the pool existed and
  **5.6 s** after. Size the pool to the worker count.
- **A blocking call inside an `async @action` stalls the agent's event loop**, so
  every concurrent action on that agent serializes — the other half of the 40.5 s
  result above. `run_code` uses `asyncio.to_thread(subprocess.run)`, not a bare
  `subprocess.run`.
- **`os.chdir` is process-global.** In `--exchange local` all agents share one
  process, so the last `base_dir` chdir wins — three agents all reported the
  third's cwd. Never default a working directory to `os.getcwd()`; default it to
  *this agent's* `base_dir`, and pass `cwd=` explicitly to subprocesses.
- **`hpc_agents.py` must be importable on the remote host.** Academy pickles the
  agent class **by reference**, so a missing `PYTHONPATH` entry means every launch
  fails with `ModuleNotFoundError`. `cd` alone does not make code importable.

**stdio transport**

- **On stdio the JSON-RPC channel IS stdout.** Install the `_StdoutShim` (dup fd
  1 first, redirect fd 1 to stderr, keep `.buffer` as the true client stream)
  **before** `server.run()`. Verified: with the shim installed, prints, raw
  `os.write(1, ...)`, and subprocess stdout all land on stderr and the session
  survives. A print *before* the shim reaches the wire and corrupts the protocol
  — reproduced with `PYTHONUNBUFFERED=1`, which MCP launchers commonly set.
- **A lifespan-created `Handle` cannot be called from a tool.** `Handle` resolves
  its `ExchangeClient` from a **ContextVar**, and contextvars propagate to child
  tasks only at creation. The tool coroutine runs in a different task, so every
  call raises `ExchangeClientNotFoundError` — *after* `list_tools` succeeded,
  which makes it look like a tool bug. Rebind:
  `Handle(handle.agent_id, exchange=manager.exchange_client)`.

**Endpoint configuration**

- ✅ **`engine.address.ifname` — RESOLVED 2026-07-27. Aurora is `hsn0`, not `bond0`.**
  Source: ALCF's own user-guide source (`github.com/argonne-lcf/user-guides`,
  `docs/aurora/workflows/parsl.md`), which uses `address_by_interface('hsn0')` —
  the Slingshot HSN interface. **`bond0` is the POLARIS value**; shipping it on
  Aurora would have made every task hang forever with no error, because workers
  could not reach the interchange. This was carried as the deployment's highest
  open risk on the strength of "ALCF documents bond0" — true, but for the wrong
  machine. Both Aurora configs now ship `ifname: hsn0`; Polaris keeps `bond0`.
  `bootstrap_aurora.sh` STEP 3 still prints `ip -br addr` — confirm `hsn0` is UP
  on the UAN rather than trusting the doc blindly.
  **General lesson: a value documented for one machine at a site is not a value
  documented for the site.** The public docs answered this in one fetch; it did
  not need a login-node round trip.
- ✅ **Aurora accelerator naming — RESOLVED. Flat integers `0`–`11`. Ignore the user guide here.**
  ALCF's Parsl doc builds `gpu.tile` names (`'0.0' … '5.1'`) via
  `[f'{gid}.{tid}' for gid in range(6) for tid in range(2)]`. **That form does not
  apply to this site's current configuration.** Aurora runs with the *flat* device
  hierarchy (`ZE_FLAT_DEVICE_HIERARCHY`), under which `ZE_AFFINITY_MASK` is a
  single integer `0`–`11`. Confirmed by the machine owner, and independently by
  this campaign's Aurora recon, which observed the flat default on the UAN.
  So: **`available_accelerators: 12`** (the integer form, which Parsl expands to
  `'0'…'11'`) is correct — do NOT substitute tile-name strings.
  Note this failure mode is nastier than a wrong `ifname`: a wrong ifname *hangs*,
  which you notice, whereas wrong accelerator naming **silently under-pins** —
  the fan-out looks successful while several tasks quietly share a tile. rung 7 of
  `verify.py` still probes a real worker for `ZE_AFFINITY_MASK` and the visible
  device count, which is how you would catch a future hierarchy change.

- **`TMPDIR` overruns the AF_UNIX ~108-char limit.** PBS sets
  `TMPDIR=/var/tmp/pbs.<jobid>.<fqdn>`; the HTEx worker pool's multiprocessing
  socket lives under it, so every worker dies with `OSError: AF_UNIX path too
  long`, surfacing as a bare `EOFError`. **`export TMPDIR=/tmp` FIRST** in
  `worker_init`.
- **Parsl's venv `bin/` must be on `PATH`** or HTEx fails with
  `FileNotFoundError: [Errno 2] No such file or directory: 'interchange.py'`
  while the server still logs a clean startup. Found in the prior run; not in the
  academy skill.
- **Never *prepend* a bundled toolchain to `PATH`.** `$AMBERHOME/bin` ships its
  own python *and* its own OpenMPI `mpiexec`; prepending shadows the venv python
  and the Cray PALS `mpiexec` that `MpiExecLauncher` needs — blocks exit 213 and
  parsl reports `Job is marked as MISSING since the workers failed to register`.
  **Append.**
- **`set -u` kills `module`.** Lmod's bash init dereferences `ZSH_EVAL_CONTEXT`
  unbound, so the FIRST `module` call aborts the whole script. Use
  `set -o pipefail` alone.
- **An app without `executors=[...]` runs on the wrong executor** — Parsl
  defaults to `'all'`, so "GPU" tasks silently land on unpinned CPU workers.
- **Polaris `prod` is a Route queue with `resources_min.nodect=10`** and rejects
  small jobs at submit time. Use `debug` / `preemptable` / `capacity`. On Aurora,
  `tiny` is disabled and `prod` needs ≥256 nodes.

**Operational**

- **A job ID means the submission parsed, not that the job will run.**
  `qsub -v 'VAR=a,b,c'` silently drops the variable, and multiple `-v` flags do
  not concatenate — the last wins. Four queued jobs would have died on startup.
  **Put structured input in files, never in `-v`.**
- **A stale SSH ControlMaster hangs forever** while `ssh -O check` still reports
  "Master running" — worse than a dead one. Every command, even `echo`, returns
  nothing and blocks in session startup. `ssh -O exit`, then re-auth. Add
  `ServerAliveInterval 60` / `ServerAliveCountMax 3`.
- **A global `ControlPath` earlier in `~/.ssh/config` shadows per-host ones.**
  SSH is first-match-wins, so sockets land somewhere other than the `Host` block
  says. Verify with `ssh -G <host> | grep controlpath`.
- **Uniform failure across a batch is a route problem, not absence.** A 404 on
  every accession meant AlphaFold had migrated to a new URL scheme; "not open
  access" on every PMC id meant a deposit-type issue, not wrong IDs.
- **Task handles live in the MCP server's memory.** Restarting the server loses
  the mapping from `task_id` to in-flight work (the work itself keeps running on
  the endpoint). For anything long, have the snippet write its artifact to the
  parallel filesystem rather than relying on retrieving it later.
