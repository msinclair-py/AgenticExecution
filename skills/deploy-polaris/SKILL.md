---
name: deploy-polaris
description: Deploy the AgenticExecution framework on ALCF Polaris as a scaling MCP server for high-throughput code-execution campaigns. Creates and submits a PBS job that launches the academy AgenticExecution agent as a FastMCP HTTP server whose Parsl LocalProvider + HighThroughputExecutor fans the execute_code @python_app across every A100 GPU on every node in the allocation. An external MCP harness (Claude Code, Codex, ...) then drives the single execute_code tool to run LLM-authored snippets against that server. Use for deploying/scaling AgenticExecution on Polaris (NVIDIA A100, PBS Pro); the sibling deploy-aurora skill does the same for Aurora (Intel Max GPUs, 12 tiles/node). For the science the snippets do see the pyscf / rdkit / chai-folding / chroma-diffusion / proteinmpnn / md-simulation / rag skills.
license: Apache-2.0
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: fastmcp 3.4.4, parsl LocalProvider+MpiExecLauncher, ALCF Polaris (4x A100/node), cudatoolkit-standalone/12.4.1
---

# Deploy AgenticExecution on ALCF Polaris (scaling MCP server)

## When to use this

Use this to run AgenticExecution on **Polaris** for a high-throughput campaign:
one MCP server, backed by an academy `AgenticExecution` agent, whose Parsl
runtime spreads `execute_code` snippets across **all GPUs on all nodes** of a PBS
allocation. An external harness connects over HTTP and drives the single
`execute_code` tool, authoring Python that uses the project environment (pyscf,
gpu4pyscf, rdkit, ambertools, torch) to do QM / force-field / ML work.

Polaris is the **NVIDIA-A100** twin of `deploy-aurora` (Intel Max). Same package,
same launcher shape — the differences are the node facts (4 GPUs vs 12 tiles),
the module system (`cudatoolkit` vs `frameworks`/oneAPI), and the
queues/walltimes. Read `deploy-aurora` too; the AF_UNIX, run-the-launcher-once,
and HSN-proxy pitfalls are shared.

## Architecture

```
external MCP harness  --HTTP-->  FastMCP server  -->  AgenticExecution (academy)
                                  (execute_code)              |
                                                   parsl.load(HeterogeneousSettings config)
                                                              |
                          HighThroughputExecutor 'gpu' (1 worker per A100)  +  'cpu'
                          LocalProvider(nodes_per_block=N) + MpiExecLauncher(--ppn 1)
                                                              |
                                fans the execute_code @python_app over N nodes x 4 GPUs
```

The agent exposes exactly **one** action, `execute_code`: it writes an
LLM-authored snippet to a temp file and runs it in a fresh subprocess on a Parsl
worker (out-of-process, crash-isolated, timeout-enforced), returning
`{stdout, stderr, returncode}`. The MCP server surfaces only that tool.

**The scaling seam:** `python -m agentic_execution.mcp_server` constructs
`ServerConfig` with **no** `parsl_config`, so the agent loads a local config with
the `gpu` executor capped at ONE worker. Running the module directly on a GPU
node therefore uses a **single GPU**, no matter how many the node has.

**The workaround (no repo change to the entrypoint):**
`scripts/polaris/launch_scaling_mcp.py` composes the package's public pieces —
`HeterogeneousSettings.config_factory` → `ServerConfig(parsl_config=...)` →
`mcp_server.create_managed_server` — and serves the **same** `execute_code` tool
over the **same** FastMCP server, but with the scaling Parsl config attached.

## Polaris facts that shape the config

- 1 node = **4× A100** + 1 EPYC Milan (32 physical cores / 64 HW threads).
- Scheduler = **PBS Pro**; jobs run under Cray **PALS** `mpiexec` against
  `$PBS_NODEFILE`.
- The `MpiExecLauncher(bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1')` +
  `available_accelerators` pattern in `parsl_settings.py` **is** the canonical
  ALCF Polaris Parsl recipe. `--ppn 1` = one HTEx manager per node; each manager
  runs `available_accelerators` workers, one pinned per GPU
  (Parsl sets `CUDA_VISIBLE_DEVICES` per worker).
- **GPU workers = nodes × 4.** 8 nodes → 32 concurrent GPU tasks. The package
  default `available_accelerators=12` is Aurora-shaped, so on Polaris the
  launcher passes **`AGENTIC_EXECUTION_ACCEL=4`**.
- **Queues / walltimes** (verify against the
  [running-jobs docs](https://docs.alcf.anl.gov/polaris/running-jobs/) — they change):

  | Queue | Nodes | Max walltime | Notes |
  |-------|-------|--------------|-------|
  | `debug` | 1–2 | 1 hr | use for the single-node baseline |
  | `debug-scaling` | 1–10 | 1 hr | use to prove multi-node scaling |
  | `prod` (routing) | 10–496 | 24 hr | the real campaign |
  | `preemptable` | 1–10 | 72 hr | long, may be preempted |

- **Filesystem** = `home:grand` (or `eagle`). Polaris does **not** mount
  `/lus/flare` (Aurora-only), so the package must be installed on a
  Polaris-mounted path; set `#PBS -l filesystems=home:grand` and `PROJECT_DIR`
  accordingly.

## Files this skill provides

| File | Role |
|------|------|
| `scripts/polaris/run_mcp_singlenode.pbs` | Baseline: 1 node, 1 GPU, HTTP server. Proves the stack boots. Does **not** scale. |
| `scripts/polaris/launch_scaling_mcp.py` | Standalone launcher that attaches the Parsl scaling config to the agent and serves it. |
| `scripts/polaris/run_mcp_scaling.pbs` | High-throughput: N nodes, all GPUs, submits the launcher. |

## Deploy — step by step

### 1. One-time: confirm the environment on Polaris

```bash
cd /lus/grand/projects/FRAME-IDP/msinclair/software/AgenticExecution
module use /soft/modulefiles
module load cudatoolkit-standalone/12.4.1   # verify this module exists
uv sync --extra gpu                          # installs gpu4pyscf-cuda12x + cupy
source .venv/bin/activate
# verify on a GPU node (interactive: qsub -I -q debug -l select=1 ...):
python -c "import gpu4pyscf, cupy; print('gpu stack OK')"
python -c "import agentic_execution, pyscf, rdkit; print('stack OK')"
```

Confirm / fill placeholders in both `.pbs` files before first submit:
- `#PBS -A FRAME-IDP` — the charging allocation.
- `PROJECT_DIR=...` — where the package is installed on a Polaris filesystem.
- `AMBERHOME` (in `run_mcp_scaling.pbs`) — real AmberTools root, only if snippets
  shell out to antechamber/parmchk2/tleap/paramfit.

### 2. Prove the stack with the single-node baseline

```bash
qsub skills/deploy-polaris/scripts/polaris/run_mcp_singlenode.pbs
qstat -u $USER
tail -f agentic-exec-mcp.log   # look for "-> http://<hsn-ip>:8000/mcp"
```

### 3. Launch the scaling campaign

Set node count in `run_mcp_scaling.pbs` (`#PBS -l select=<N>:system=polaris`) and
pick a queue that admits it (`debug-scaling` ≤10 nodes/1 hr; `prod` for 10+),
then:

```bash
qsub skills/deploy-polaris/scripts/polaris/run_mcp_scaling.pbs
tail -f agentic-exec-mcp-scale.log
# expect: "[launcher] Parsl scaling: N node(s) x 4 GPU worker(s) = 4N GPU workers"
```

### 4. Connect a harness and send work

The server speaks MCP over HTTP at `http://<head-node-hsn-ip>:8000/mcp` (printed
in the log). From another job on the Polaris network, point an MCP client at it
(no auth — keep it inside the ALCF network). Example `.mcp.json`:

```json
{"mcpServers": {"agentic-execution": {"transport": "http", "url": "http://<head-node-hsn-ip>:8000/mcp"}}}
```

The harness calls the single `execute_code` tool, authoring Python that imports
pyscf/gpu4pyscf/rdkit/ambertools to build compounds, run geometry optimizations,
ESP/RESP fits, torsion scans, etc. High throughput comes from **fanning
independent snippets** — many `execute_code` calls in flight — which Parsl
spreads across the GPU workers.

## Environment knobs (launcher reads these)

| Var | Default | Meaning |
|-----|---------|---------|
| `AGENTIC_EXECUTION_NODES` | count `$PBS_NODEFILE` | nodes → `LocalProvider(nodes_per_block=N)` |
| `AGENTIC_EXECUTION_ACCEL` | `4` | GPUs/node → GPU workers/node |
| `AGENTIC_EXECUTION_THREADS` | `8` | OMP threads per worker + cpu-executor workers/node |
| `AGENTIC_EXECUTION_WORKER_INIT` | synthesized | shell run on each worker node before start |
| `AGENTIC_EXECUTION_CUDA_MODULE` | `cudatoolkit-standalone/12.4.1` | cuda module for synthesized worker_init |
| `AGENTIC_EXECUTION_OUTPUT` | `./execution_output` | default working dir for snippets |
| `AGENTIC_EXECUTION_EXCHANGE` | `local` | academy exchange: `local` or an http(s) URL |
| `AGENTIC_EXECUTION_HOST` / `_PORT` | `hostname -i` / 8000 | HTTP bind |
| `AMBERHOME` | — | AmberTools root; if set, its bin is APPENDED to worker PATH |

## Pitfalls

The first three were found and fixed by running the launcher on a real Polaris
A100 node — the synthesized `worker_init` handles all three, but they bite
instantly if you hand-roll `AGENTIC_EXECUTION_WORKER_INIT` or a PBS script and get
the ordering wrong.

- **`$AMBERHOME/bin` must be APPENDED to PATH, never prepended.** AmberTools
  bundles its own `python` (an unrelated 3.12) *and* its own OpenMPI `mpiexec`
  in `$AMBERHOME/bin`. Prepending it shadows two things at once:
  (1) the venv `python`, so workers launch under the wrong interpreter and die
  with `ModuleNotFoundError: No module named 'agentic_execution'`; and
  (2) the Cray **PALS** `mpiexec` that Parsl's `MpiExecLauncher` shells out to.
  PALS mpiexec is the only one that accepts `--cpu-bind`; AmberTools' OpenMPI
  rejects it and every block fails with
  `An unrecognized option ... Option: --cpu-bind` → `EXIT CODE: 213` →
  `Job is marked as MISSING since the workers failed to register`.
- **Pin `TMPDIR` to a short path (`/tmp`) on the workers.** PBS sets
  `TMPDIR=/var/tmp/pbs.<jobid>.<fqdn>` (~60+ chars). Parsl's HTEx worker pool
  opens a multiprocessing `SyncManager` whose **AF_UNIX** socket lives under
  `TMPDIR`; the full socket path then overruns the ~108-char AF_UNIX limit and
  every GPU worker dies before registering with
  `OSError: AF_UNIX path too long` (surfacing to the client as an `EOFError`).
  The synthesized worker_init exports `TMPDIR=/tmp` first.
- **worker_init must reproduce the env on remote nodes.** `MpiExecLauncher`
  starts each node's HTEx manager in a fresh login shell. If CUDA / the venv /
  `AMBERHOME` / the cwd are not re-established there, `import gpu4pyscf` (or the
  AMBER steps) fail only on the non-head nodes — a confusing partial failure. The
  launcher synthesizes a worker_init from `VIRTUAL_ENV` + `AMBERHOME` +
  `AGENTIC_EXECUTION_CUDA_MODULE` (+ the `TMPDIR`/append-PATH fixes above);
  override the whole string with `AGENTIC_EXECUTION_WORKER_INIT` only if your
  setup differs — and if you do, re-apply the append-PATH and `TMPDIR=/tmp`
  fixes yourself.
- **Single-GPU trap.** If you `qsub` a plain `python -m agentic_execution.mcp_server`,
  you get ONE GPU. You must use `launch_scaling_mcp.py` to scale — this is the
  whole reason the launcher exists.
- **Do NOT wrap the launcher in mpiexec.** Run it once on the head node; Parsl
  owns the mpiexec fan-out via `$PBS_NODEFILE`. Wrapping it launches N duplicate
  servers that fight over the port.
- **Thread oversubscription.** GPU workers/node × `AGENTIC_EXECUTION_THREADS`
  should stay ≤ the node's HW threads (64). Default 4×8 = 32 is safe headroom.
- **Port reachability + the ALCF proxy.** The server binds to the node's routable
  HSN IP (`hostname -i` → `10.201.x.x` / `10.140.x.x`). Two consequences, both
  verified on Polaris:
  - That HSN address is reachable from other **compute** nodes but **not from a
    login node** — a login-node client just times out. Drive the server from
    another job on the HSN (or the same node), not from a login shell.
  - Compute nodes export `http_proxy`/`https_proxy=http://proxy.alcf.anl.gov:3128`
    and a `no_proxy` list that covers ALCF **hostnames** but **not** the raw HSN
    IP. An MCP/httpx client therefore routes the request through the proxy, which
    answers **`503 Service Unavailable`** for the internal address. Bypass it:
    `export no_proxy='*'` (or add the IP to `no_proxy`) in the client's shell.
    This 503 looks like a server bug but is pure proxy interference — the server
    log shows a clean `Application startup complete`.

## Verification (before trusting a job)

```bash
# config shape (run in the venv):
python - <<'PY'
from pathlib import Path
from agentic_execution.utils.parsl_settings import HeterogeneousSettings
cfg = HeterogeneousSettings(available_accelerators=4, nodes=8, worker_init='x').config_factory(Path('/tmp/t'))
for ex in cfg.executors:
    print(ex.label, 'workers/node=', ex.max_workers_per_node,
          'accel=', getattr(ex, 'available_accelerators', None),
          'nodes/block=', ex.provider.nodes_per_block)
# expect: gpu workers/node= 4 accel= ['0','1','2','3'] nodes/block= 8
PY

# launcher imports + synthesized worker_init (see the TMPDIR + append-PATH lines):
VIRTUAL_ENV=$VIRTUAL_ENV AMBERHOME=$AMBERHOME python -c \
  "import importlib.util as u; s=u.spec_from_file_location('l','skills/deploy-polaris/scripts/polaris/launch_scaling_mcp.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m._default_worker_init())"

# end-to-end over HTTP (login node, local config):
AGENTIC_EXECUTION_THREADS=2 python -m agentic_execution.mcp_server --transport http --host 127.0.0.1 --port 8877 &
# then from an MCP client: list_tools() -> ['execute_code']; execute_code(code_snippet="print(6*7)") -> stdout '42'
```

## Adapting to another cluster

`HeterogeneousSettings` is generic. For a non-Polaris PBS/MPI GPU cluster:
change `AGENTIC_EXECUTION_ACCEL` to that node's GPU count, adjust
`AGENTIC_EXECUTION_CUDA_MODULE` / `worker_init` for its module system, and update
the `#PBS` directives (account, queue, `select=...:system=...`, filesystems). For
Intel-GPU nodes see `deploy-aurora`. For a Slurm site the launcher's
config-building logic ports directly but `HeterogeneousSettings` would need a
`SlurmProvider`/`SrunLauncher` variant.
