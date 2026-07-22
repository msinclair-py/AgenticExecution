---
name: deploy-aurora
description: Deploy the AgenticExecution framework on ALCF Aurora as a scaling MCP server for high-throughput code-execution campaigns. Creates and submits a PBS job that launches the academy AgenticExecution agent as a FastMCP HTTP server whose Parsl LocalProvider + HighThroughputExecutor fans the execute_code @python_app across all 12 GPU tiles on every node in the allocation. An external MCP harness (Claude Code, Codex, ...) then drives the single execute_code tool to run LLM-authored snippets against that server. Use for deploying/scaling AgenticExecution on Aurora (Intel Max GPUs, PBS Pro); the sibling deploy-polaris skill does the same for Polaris (NVIDIA A100). For the science the snippets do see the pyscf / rdkit / chai-folding / chroma-diffusion / proteinmpnn / md-simulation / rag skills.
license: Apache-2.0
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: fastmcp 3.4.4, parsl LocalProvider+MpiExecLauncher, ALCF Aurora (6x Intel Max 1550 / 12 tiles per node), frameworks module
---

# Deploy AgenticExecution on ALCF Aurora (scaling MCP server)

## When to use this

Use this to run AgenticExecution on **Aurora** for a high-throughput campaign:
one MCP server, backed by an academy `AgenticExecution` agent, whose Parsl
runtime spreads `execute_code` snippets across **all 12 GPU tiles on all nodes**
of a PBS allocation. An external harness connects over HTTP and drives the single
`execute_code` tool, authoring Python that uses the project environment.

Aurora is the **Intel-GPU** twin of `deploy-polaris` (NVIDIA A100). Same package,
same launcher shape — the differences are all in the node facts (12 tiles vs 4
GPUs), the module system (`frameworks`/oneAPI vs `cudatoolkit`), the tile
hierarchy, and the queues/walltimes. Read `deploy-polaris` too; the AF_UNIX,
append-PATH-not-prepend, and "run the launcher once, not under mpiexec" pitfalls
are shared.

## Architecture

```
external MCP harness  --HTTP-->  FastMCP server  -->  AgenticExecution (academy)
                                  (execute_code)              |
                                                   parsl.load(HeterogeneousSettings config)
                                                              |
                          HighThroughputExecutor 'gpu' (1 worker per tile)  +  'cpu'
                          LocalProvider(nodes_per_block=N) + MpiExecLauncher(--ppn 1)
                                                              |
                              fans the execute_code @python_app over N nodes x 12 tiles
```

The agent exposes exactly **one** action, `execute_code`: it writes an
LLM-authored snippet to a temp file and runs it in a fresh subprocess on a Parsl
worker (out-of-process, crash-isolated, timeout-enforced), returning
`{stdout, stderr, returncode}`. The MCP server surfaces only that tool.

**The scaling seam:** `python -m agentic_execution.mcp_server` constructs
`ServerConfig` with **no** `parsl_config`, so the agent loads a local config with
the `gpu` executor capped at ONE worker. Running the module directly on an Aurora
node therefore uses a **single tile**, no matter that the node has 12.

**The workaround (no repo change to the entrypoint):**
`scripts/aurora/launch_scaling_mcp.py` composes the package's public pieces —
`HeterogeneousSettings.config_factory` → `ServerConfig(parsl_config=...)` →
`mcp_server.create_managed_server` — and serves the **same** `execute_code` tool
over the **same** FastMCP server, but with the scaling Parsl config attached.

## Aurora facts that shape the config

- 1 node = **6× Intel Data Center GPU Max 1550**, each with **2 tiles** =
  **12 tiles** that behave as 12 devices. + 2× Xeon CPU Max (Sapphire Rapids),
  **52 physical cores each = 104 cores / 208 HW threads**. (Cores 0, 52, 104, 156
  are reserved for system services.)
- Scheduler = **PBS Pro**; jobs run under Cray **PALS** `mpiexec` against
  `$PBS_NODEFILE` — same launcher family as Polaris, so the
  `MpiExecLauncher(bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1')` recipe
  in `parsl_settings.py` carries over unchanged. `--ppn 1` = one HTEx manager per
  node; each manager runs `available_accelerators` workers, one pinned per tile.
- **Tile pinning = `ZE_AFFINITY_MASK`.** Parsl's `available_accelerators` sets it
  per worker (alongside `CUDA_VISIBLE_DEVICES` etc.). With
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (the `frameworks` default) the 12 tiles are
  addressed as devices `'0'..'11'`, so the package default
  `available_accelerators=12` (int → those strings) pins one worker per tile and
  `torch.xpu.device_count() == 1` inside each worker. **This is why the package
  already defaults to 12** — it was written Aurora-shaped.
- **GPU workers = nodes × 12.** 8 nodes → 96 concurrent tile tasks.
- **Queues / walltimes** (verify against the
  [running-jobs docs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/) — they change):

  | Queue | Nodes | Max walltime | Notes |
  |-------|-------|--------------|-------|
  | `debug` | 1–2 | 1 hr | 1 job/user; use for the single-node baseline |
  | `debug-scaling` | 2–256 | 1 hr | 1 job/user; use to prove multi-node scaling |
  | `prod` (routing) | 256–10624 | 24 hr | routes to small/medium/large |
  | `capacity` | 1–16 | 168 hr | long-running small jobs; ≤512 nodes aggregate |

  All queues: ≤10 jobs running/accruing and ≤100 queued per project.
- **Filesystem** = `flare` (this project lives on `/lus/flare`). Set
  `#PBS -l filesystems=flare` (add `:daos_user` only if you use DAOS).

## The Intel-GPU caveat (read before assuming GPU QM works)

Aurora GPUs are **Intel**, not NVIDIA. The `gpu` extra
(`gpu4pyscf-cuda12x`, `cupy-cuda12x`) is **CUDA-12 only** and will not install or
run on Aurora. Consequence:

- **QM (pyscf) runs CPU-bound** on Aurora — use `use_gpu=False`; the 104 cores
  and HBM carry it. Do **not** `uv sync --extra gpu` here.
- **The 12 tiles accelerate the torch/`xpu` skills** — `chai-folding`,
  `chroma-diffusion`, `proteinmpnn`, and the OpenMM (`OpenCL`) path of
  `md-simulation`. Those skills are device-agnostic via
  `_device.resolve_device()` (xpu → cuda → cpu), so the same snippet that uses
  `cuda` on Polaris uses `xpu` on Aurora with no change.

So on Aurora the scaling story is: Parsl fans independent snippets across the 12
tiles/node for the ML/folding/diffusion/MD work, while QM snippets run on the
cores. `available_accelerators=12` is still correct — it governs tile fan-out.

## Files this skill provides

| File | Role |
|------|------|
| `scripts/aurora/run_mcp_singlenode.pbs` | Baseline: 1 node, 1 tile, HTTP server. Proves the stack boots. Does **not** scale. |
| `scripts/aurora/launch_scaling_mcp.py` | Standalone launcher that attaches the Parsl scaling config to the agent and serves it. |
| `scripts/aurora/run_mcp_scaling.pbs` | High-throughput: N nodes, all 12 tiles, submits the launcher. |

## Deploy — step by step

### 1. One-time: build the environment on Aurora

The `frameworks` module supplies Python + torch with `xpu` support. Build the
venv **on top of it** with `--system-site-packages` so torch/IPEX stay
importable, then install the package **without** the CUDA `gpu` extra:

```bash
cd /lus/flare/projects/FRAME-IDP/msinclair/software/AgenticExecution
module use /soft/modulefiles
module load frameworks
python -m venv --system-site-packages .venv    # inherits frameworks' torch(xpu)
source .venv/bin/activate
pip install -e .                               # base only — NOT '.[gpu]'
# verify on a compute node (interactive: qsub -I -q debug -l select=1 ...):
python -c "import torch; print('xpu', torch.xpu.is_available(), torch.xpu.device_count())"
python -c "import agentic_execution, pyscf, rdkit; print('stack OK')"
```

Confirm / fill placeholders in both `.pbs` files before first submit:
- `#PBS -A FRAME-IDP` — the charging allocation.
- `PROJECT_DIR=...` — must match where you cloned/installed the package.
- `#PBS -l filesystems=flare` — add `:daos_user` only if snippets touch DAOS.

### 2. Prove the stack with the single-node baseline

```bash
qsub skills/deploy-aurora/scripts/aurora/run_mcp_singlenode.pbs
qstat -u $USER
tail -f agentic-exec-mcp.log   # look for "-> http://<hsn-ip>:8000/mcp"
```

### 3. Launch the scaling campaign

Set node count in `run_mcp_scaling.pbs` (`#PBS -l select=<N>`) and pick a queue
that admits it (`debug-scaling` ≤256 nodes/1 hr; `prod` for 256+), then:

```bash
qsub skills/deploy-aurora/scripts/aurora/run_mcp_scaling.pbs
tail -f agentic-exec-mcp-scale.log
# expect: "[launcher] Parsl scaling: N node(s) x 12 GPU worker(s) = 12N GPU workers"
```

### 4. Connect a harness and send work

The server speaks MCP over HTTP at `http://<head-node-hsn-ip>:8000/mcp` (printed
in the log). From another job on the Aurora network, point an MCP client at it
(no auth — keep it inside the ALCF network). Example `.mcp.json`:

```json
{"mcpServers": {"agentic-execution": {"transport": "http", "url": "http://<head-node-hsn-ip>:8000/mcp"}}}
```

The harness calls the single `execute_code` tool. High throughput comes from
**fanning independent snippets** — many `execute_code` calls in flight — which
Parsl spreads across the tile workers.

## Environment knobs (launcher reads these)

| Var | Default | Meaning |
|-----|---------|---------|
| `AGENTIC_EXECUTION_NODES` | count `$PBS_NODEFILE` | nodes → `LocalProvider(nodes_per_block=N)` |
| `AGENTIC_EXECUTION_ACCEL` | `12` | tiles/node → GPU workers/node |
| `AGENTIC_EXECUTION_THREADS` | `8` | OMP threads per worker + cpu-executor workers/node |
| `AGENTIC_EXECUTION_WORKER_INIT` | synthesized | shell run on each worker node before start |
| `AGENTIC_EXECUTION_FRAMEWORKS` | `frameworks` | module loaded in the synthesized worker_init |
| `AGENTIC_EXECUTION_OUTPUT` | `./execution_output` | default working dir for snippets |
| `AGENTIC_EXECUTION_EXCHANGE` | `local` | academy exchange: `local` or an http(s) URL |
| `AGENTIC_EXECUTION_HOST` / `_PORT` | `hostname -i` / 8000 | HTTP bind |

## Pitfalls

Most of these are shared with `deploy-polaris`; the tile/hierarchy ones are
Aurora-specific.

- **Tile hierarchy must match the accelerator strings.** With
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (frameworks default) tiles are `'0'..'11'`, so
  `available_accelerators=12` (int) is correct. If you instead switch to
  `COMPOSITE`, `ZE_AFFINITY_MASK='7'` is invalid and workers won't pin — you'd
  then have to pass composite names
  `['0.0','0.1',...,'5.0','5.1']` as `available_accelerators` (the form the ALCF
  Parsl docs show). The launcher exports `FLAT` explicitly so it can't drift.
- **Do NOT install the `gpu` extra on Aurora.** `gpu4pyscf-cuda12x`/`cupy-cuda12x`
  are CUDA-only; installing them fails or (worse) shadows a working CPU pyscf.
  Base install + `use_gpu=False` for QM; the tiles serve the torch(xpu) skills.
- **Build the venv with `--system-site-packages` against the frameworks Python.**
  A fresh `uv sync`/`python -m venv` (no system packages) creates an isolated env
  where `import torch` fails or lacks `xpu`. Activate the venv AFTER
  `module load frameworks` so its `python` wins on PATH.
- **Pin `TMPDIR=/tmp` on the workers.** PBS sets a ~60-char `TMPDIR`; the HTEx
  worker pool's **AF_UNIX** `SyncManager` socket then overruns the ~108-char
  limit and every worker dies before registering
  (`OSError: AF_UNIX path too long`, surfacing to the client as `EOFError`). The
  synthesized worker_init exports `TMPDIR=/tmp` first.
- **worker_init must reproduce the env on remote nodes.** `MpiExecLauncher`
  starts each node's HTEx manager in a fresh login shell; if `frameworks` / the
  venv / the cwd / the hierarchy vars are not re-established there, `torch.xpu`
  fails only on the non-head nodes — a confusing partial failure. The launcher
  synthesizes worker_init from `VIRTUAL_ENV` + cwd + the module/hierarchy/TMPDIR
  lines; override the whole string with `AGENTIC_EXECUTION_WORKER_INIT` only if
  your setup differs (and re-apply the `TMPDIR=/tmp` and activate-after-module
  fixes yourself).
- **Single-tile trap.** `qsub`-ing a plain `python -m agentic_execution.mcp_server`
  gives ONE tile. Use `launch_scaling_mcp.py` to scale — this is why it exists.
- **Do NOT wrap the launcher in mpiexec.** Run it once on the head node; Parsl
  owns the mpiexec fan-out via `$PBS_NODEFILE`. Wrapping it launches N duplicate
  servers fighting over the port.
- **Thread oversubscription.** tiles/node × `AGENTIC_EXECUTION_THREADS` should
  stay ≤ the node's usable HW threads. 12 × 8 = 96 leaves headroom under 208.
- **Port reachability + the ALCF proxy** (same as Polaris). The server binds to
  the routable HSN IP (`hostname -i`), reachable from other **compute** nodes but
  **not from a login node**. Compute nodes also export
  `http_proxy`/`https_proxy` whose `no_proxy` covers ALCF hostnames but not the
  raw HSN IP, so an httpx/MCP client routes through the proxy and gets a
  **`503`** for the internal address. Bypass it in the client shell:
  `export no_proxy='*'`. The 503 looks like a server bug but is pure proxy
  interference — the server log shows a clean `Application startup complete`.

## Verification (before trusting a job)

```bash
# config shape (run in the venv):
python - <<'PY'
from pathlib import Path
from agentic_execution.utils.parsl_settings import HeterogeneousSettings
cfg = HeterogeneousSettings(available_accelerators=12, nodes=8, worker_init='x').config_factory(Path('/tmp/t'))
for ex in cfg.executors:
    print(ex.label, 'workers/node=', ex.max_workers_per_node,
          'accel=', getattr(ex, 'available_accelerators', None),
          'nodes/block=', ex.provider.nodes_per_block)
# expect: gpu workers/node= 12 accel= ['0','1',...,'11'] nodes/block= 8
PY

# launcher imports + synthesized worker_init (see the FLAT + TMPDIR lines):
VIRTUAL_ENV=$VIRTUAL_ENV python -c \
  "import importlib.util as u; s=u.spec_from_file_location('l','skills/deploy-aurora/scripts/aurora/launch_scaling_mcp.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m._default_worker_init())"

# end-to-end over HTTP (interactive compute node, local config):
AGENTIC_EXECUTION_THREADS=2 python -m agentic_execution.mcp_server --transport http --host 127.0.0.1 --port 8877 &
# then from an MCP client: list_tools() -> ['execute_code']; execute_code(code_snippet="print(6*7)") -> stdout '42'
```

## Adapting to another cluster

`HeterogeneousSettings` is generic. For a non-Aurora PBS/MPI cluster: set
`available_accelerators` to that node's device count, swap the module/hierarchy
lines in the launcher's `_default_worker_init` for its system, and update the
`#PBS` directives (account, queue, `select=`, `filesystems=`). For NVIDIA nodes
see `deploy-polaris`. For a Slurm site the config-building logic ports directly
but `HeterogeneousSettings` would need a `SlurmProvider`/`SrunLauncher` variant.
