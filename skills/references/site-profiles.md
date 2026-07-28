# Site profiles for the academy + MCP + Parsl execution layer

Per-machine facts that change between clusters. The agent, server, and launcher
code are identical everywhere; only these knobs move.

Consumed by `globus-compute-mcp` (the endpoint YAMLs and `worker_init` need the
device count, pinning var, module lines, filesystems, and queue limits) and by
`polaris-jobs` (node hardware and the `mpiexec` affinity lines).

## ALCF Polaris (NVIDIA A100, PBS Pro)

| Fact | Value |
|---|---|
| Node | 4× A100 + 1× AMD EPYC Milan (32 physical cores / 64 HW threads) |
| Accelerators | `HPC_ACCEL=4` → GPU workers = nodes × 4 |
| Pinning var | `CUDA_VISIBLE_DEVICES` (Parsl sets it from `available_accelerators`) |
| Module | `module use /soft/modulefiles; module load cudatoolkit-standalone/12.4.1` |
| Filesystems | `#PBS -l filesystems=home:grand` (or `eagle`). **No `/lus/flare`** |
| Select | `#PBS -l select=<N>:system=polaris` |
| Env build | Plain venv; CUDA-12 wheels (`gpu4pyscf-cuda12x`, `cupy-cuda12x`) |

Queues (verify at https://docs.alcf.anl.gov/polaris/running-jobs/ — they change):

| Queue | Nodes | Walltime | Use |
|---|---|---|---|
| `debug` | 1–2 | 1 hr | single-node baseline |
| `debug-scaling` | 1–10 | 1 hr | prove multi-node fan-out |
| `prod` (routing) | 10–496 | 24 hr | the campaign |
| `preemptable` | 1–10 | 72 hr | long, may be preempted |

`worker_init` skeleton:

```bash
export TMPDIR=/tmp
module use /soft/modulefiles
module load cudatoolkit-standalone/12.4.1
source $VIRTUAL_ENV/bin/activate
export PATH="$PATH:$AMBERHOME/bin"    # only if needed, APPENDED
cd <project dir>
```

## ALCF Aurora (Intel Data Center GPU Max 1550, PBS Pro)

| Fact | Value |
|---|---|
| Node | 6× Max 1550 = **12 tiles** (12 addressable devices) + 2× Xeon CPU Max, 104 cores / 208 HW threads (cores 0, 52, 104, 156 reserved) |
| Accelerators | `HPC_ACCEL=12` → GPU workers = nodes × 12 |
| Pinning var | `ZE_AFFINITY_MASK` |
| Hierarchy | `export ZE_FLAT_DEVICE_HIERARCHY=FLAT` (+ `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`) |
| Module | `module use /soft/modulefiles; module load frameworks` (oneAPI + torch(xpu) + Python) |
| Filesystems | `#PBS -l filesystems=flare` (add `:daos_user` only for DAOS) |
| Select | `#PBS -l select=<N>` |
| Env build | `python -m venv --system-site-packages .venv` **on top of** `frameworks` — base only, no CUDA extras |

Queues (verify at https://docs.alcf.anl.gov/aurora/running-jobs-aurora/):

| Queue | Nodes | Walltime | Use |
|---|---|---|---|
| `debug` | 1–2 | 1 hr | baseline; 1 job/user |
| `debug-scaling` | 2–256 | 1 hr | multi-node; 1 job/user |
| `prod` (routing) | 256–10624 | 24 hr | routes to small/medium/large |
| `capacity` | 1–16 | 168 hr | long-running small jobs |

All queues: ≤10 jobs running/accruing, ≤100 queued per project.

### Aurora-specific traps

- **`ZE_FLAT_DEVICE_HIERARCHY` must be `FLAT`** for `available_accelerators=12`
  to mean tiles `'0'..'11'`. Under `COMPOSITE`, `ZE_AFFINITY_MASK='7'` is invalid
  and workers silently fail to pin — you would instead need explicit composite
  names `['0.0','0.1',...,'5.0','5.1']`. Export `FLAT` in both the batch script
  and `worker_init` so it cannot drift.
- **Never install CUDA extras here.** `gpu4pyscf-cuda12x` / `cupy-cuda12x` are
  CUDA-only; they fail to install or shadow a working CPU build. On Aurora, QM
  runs CPU-bound (104 cores + HBM) while the 12 tiles serve torch/`xpu` work.
- **Build the venv with `--system-site-packages` against the frameworks Python**,
  and activate it *after* `module load frameworks`. An isolated venv loses
  `torch.xpu` entirely.
- Device-agnostic code (`xpu → cuda → cpu` resolution) lets the same tool payload
  run unchanged on both machines.

`worker_init` skeleton:

```bash
export TMPDIR=/tmp
module use /soft/modulefiles
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
source $VIRTUAL_ENV/bin/activate
cd <project dir>
```

## Porting to another PBS/MPI cluster

Change four things:

1. `HPC_ACCEL` → that node's device count.
2. The module lines in the endpoint's `worker_init` → its module system.
3. The `#PBS` directives → account, queue, `select=`, `filesystems=`.
4. Verify `mpiexec` accepts `--cpu-bind` (Cray PALS does; OpenMPI does not). If
   it is OpenMPI, replace `MpiExecLauncher(bind_cmd='--cpu-bind', ...)` with the
   launcher/flags that MPI accepts, or use `SimpleLauncher` + `nodes_per_block=1`
   and one block per node.

## Porting to Slurm

The agent, server, and launcher are unchanged; only the provider/launcher swap.

```python
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher

provider = SlurmProvider(
    partition='<gpu-partition>', account='<acct>',
    nodes_per_block=nodes, init_blocks=1, max_blocks=1,
    scheduler_options='#SBATCH --gpus-per-node=4',
    launcher=SrunLauncher(overrides='-c 8'),
    worker_init=worker_init, walltime='01:00:00',
)
```

Adjust the launcher's node count to read `SLURM_JOB_NUM_NODES` (or
`scontrol show hostnames $SLURM_JOB_NODELIST`) instead of `$PBS_NODEFILE`. Two
usage modes:

- **Inside an existing allocation** (`salloc`/`sbatch`, the PBS-equivalent shape):
  keep `LocalProvider` + `SrunLauncher` so Parsl just spreads managers over the
  nodes you already hold.
- **Letting Parsl submit**: use `SlurmProvider` as above; then the MCP server
  runs wherever you started it and Parsl queues its own allocation — the server's
  bind address must be reachable from the login/submit host in that case.

Everything in the `globus-compute-mcp` pitfall list still applies: `TMPDIR=/tmp`, appended
PATH, `worker_init` rebuilding the environment, one launcher process, bind to a
routable address, and mind the site proxy.
