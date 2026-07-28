# Building tool environments (HPC-aware)

The GPU tool skills (folding, inverse-folding, protein-language-explore,
md-simulation, peptide-diffusion) each need a **pre-provisioned environment**
with heavy dependencies, model weights, and CUDA runtime libraries that match
the node. Their inline `pip install` gates are a fallback, not a promise: on a
real HPC cluster you cannot bootstrap them at run time. **Build every tool's
environment once, at the start of a project, on a login node**, then point the
tools at shared storage.

This guide collects the cross-cutting, hard-won lessons. It is generic — nothing
here hardcodes a user, project, or key. Substitute your own values for the
placeholders:

| Placeholder | Meaning |
|---|---|
| `<env-name>` | the conda env / venv name for one tool (e.g. `<proj>-folding`) |
| `$PROJECT_ENVS` | shared project dir that holds all the tool envs |
| `$SHARED_WEIGHTS` | shared project dir for downloaded model weights / caches |
| `$PROTEINMPNN_PATH` | a ProteinMPNN repo checkout on disk (inverse-folding) |

Put `$PROJECT_ENVS` and `$SHARED_WEIGHTS` on **shared parallel project storage**
that both login and compute nodes can read (e.g. a project filesystem), **not**
in `$HOME/.cache`, node-local `/tmp`, or `/dev/shm` — those are not visible to
the compute node that later runs the job.

## The core HPC reality: compute nodes have no internet

On most clusters **only login nodes reach the internet; compute nodes are
offline.** Every step that downloads something — pip installs, HuggingFace
snapshots, Chai-1/ESM weights — must happen on a login node **and land in shared
storage** so the offline compute node can reuse it. Consequences:

- **Install envs on a login node**, into `$PROJECT_ENVS`.
- **Pre-download all model weights on a login node**, into `$SHARED_WEIGHTS`,
  before submitting any job. Point each tool's cache at shared storage:
  - `HF_HOME=$SHARED_WEIGHTS/hf` — transformers / ESM-2 (protein-language-explore,
    and Chai-1's ESM embeddings).
  - `CHAI_DOWNLOADS_DIR=$SHARED_WEIGHTS/chai` — Chai-1 model weights (folding,
    interactant-folding).
  - `TMPDIR=$SHARED_WEIGHTS/tmp` — for any tool that caches under Python's
    `gettempdir()`; keep it off node-local `/tmp` so a login-node pre-fetch is
    reused.
- **On the offline compute node, force offline mode** so libraries don't stall
  trying to reach the hub: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
  (Do NOT set these on the login node while you are still pre-downloading.)

## Match CUDA runtime libs to torch's build

The most common GPU-node failure is a mismatch between the CUDA runtime libraries
on the node and the CUDA version torch was built against. Check what torch
expects first:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Symptoms and the fix:

- `failed to open libnvrtc-builtins.so.X.Y` (a JIT/kernel compile step) or
  `libcudnn.so.N: cannot open shared object file` (a missing cuDNN) →
  **pip install the matching NVIDIA lib package into the same env**, e.g.
  `pip install nvidia-cuda-nvrtc-cu12==X.Y.*` or `pip install nvidia-cudnn-cu12`,
  where `X.Y` matches `torch.version.cuda`.
- Then append **only that package's** lib dir to `LD_LIBRARY_PATH`:

  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
  ```

**Never prepend a whole other env's `lib/` directory to `LD_LIBRARY_PATH`.** Doing
so shadows `libtorch`/`libc10` from that other env and breaks the torch import
with confusing errors like `module 'torch._C' has no attribute
'_OutOfMemoryError'`. Add the single `site-packages/nvidia/<pkg>/lib` directory
you actually need and nothing more.

## Login-node OpenMP cap

CPU-heavy imports and builds (numpy, torch, compiling extensions) on a shared
login node can exhaust the per-user thread/process limit and fail with
`libgomp: Thread creation failed: Resource temporarily unavailable`. Cap the
thread pools before running such steps on a login node:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

(Raise these back up inside a compute-node job where you own the cores.)

## Packages that pin broken/old deps

Some tool repos pin ancient, unbuildable dependency versions (e.g.
`scikit_learn==1.1.2`, which does not build on Python 3.11). Install the repo
without letting pip resolve those pins, and rely on a modern base scientific
stack instead:

```bash
pip install -e <repo> --no-deps       # skip the broken pins
pip install <the few light real deps> # add back only what's actually imported
```

This works when numpy / scipy / pandas / scikit-learn are already present and
recent — the pinned old versions were the only real blocker.

## Reuse a shared base scientific env

Rebuilding a heavy torch/numpy stack per tool is slow and error-prone. If you
have a known-good base env (correct torch for the node, numpy, scipy), create the
tool env as a venv that **inherits** it:

```bash
python -m venv --system-site-packages $PROJECT_ENVS/<env-name>
source $PROJECT_ENVS/<env-name>/bin/activate
pip install <only the tool-specific extras>   # torch/numpy come from the base
```

This keeps the working torch (with its matched CUDA libs) and only layers the
tool's own light dependencies on top.

## Bootstrap all envs at project start

Do this once, on a login node, before any job. Skeleton — replace the
placeholder `module load` / cluster commands with your site's:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- shared locations (edit these; keep on shared project storage) ---
export PROJECT_ENVS=/path/to/shared/project/envs
export SHARED_WEIGHTS=/path/to/shared/project/weights
export HF_HOME=$SHARED_WEIGHTS/hf
export CHAI_DOWNLOADS_DIR=$SHARED_WEIGHTS/chai
export TMPDIR=$SHARED_WEIGHTS/tmp
mkdir -p "$PROJECT_ENVS" "$HF_HOME" "$CHAI_DOWNLOADS_DIR" "$TMPDIR"

# login-node safety caps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# site module system (PLACEHOLDER — use your cluster's)
module load conda            # or: source <miniconda>/etc/profile.d/conda.sh

PROJ=<proj>                  # short project prefix for env names

# one env per tool; all pin Python 3.11
create_conda () {            # create_conda <suffix> <extra pip/conda spec...>
  local name="${PROJ}-$1"; shift
  conda create -y -p "$PROJECT_ENVS/$name" python=3.11
  conda activate "$PROJECT_ENVS/$name"
  "$@"                       # tool-specific install commands
  conda deactivate
}

# folding (Chai-1)
create_conda folding  pip install chai_lab gemmi
# inverse-folding (ProteinMPNN) — repo checkout, not a pip package
create_conda mpnn     bash -c 'pip install torch numpy && \
  git clone https://github.com/dauparas/ProteinMPNN.git "$PROJECT_ENVS/ProteinMPNN"'
# protein-language-explore (ESM-2)
create_conda esm      pip install torch transformers
# md-simulation (OpenMM + AmberTools) — conda-forge, not pip
conda create -y -p "$PROJECT_ENVS/${PROJ}-md" -c conda-forge \
  python=3.11 openmm ambertools mdanalysis pdbfixer
# peptide-diffusion — provisioned separately (see its SKILL.md)

# --- pre-download weights on the login node (compute nodes are offline) ---
conda activate "$PROJECT_ENVS/${PROJ}-esm"
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t33_650M_UR50D')"
conda deactivate
# Chai-1 weights + ESM embeddings: run one tiny fold once so CHAI_DOWNLOADS_DIR fills.
```

### Per-tool env matrix

| Tool skill | Base library | Create command | Weights to pre-cache |
|---|---|---|---|
| `folding` / `interactant-folding` | chai_lab (Chai-1) | `conda create -n <proj>-folding python=3.11 && pip install chai_lab gemmi` | Chai-1 weights + ESM embeddings → `$CHAI_DOWNLOADS_DIR` (first run) |
| `inverse-folding` | ProteinMPNN (repo) | `conda create -n <proj>-mpnn python=3.11 && pip install torch numpy` + `git clone .../ProteinMPNN` | `.pt` weights ship in the repo — no separate download |
| `protein-language-explore` | transformers (ESM-2) | `conda create -n <proj>-esm python=3.11 && pip install torch transformers` | `facebook/esm2_t33_650M_UR50D` → `$HF_HOME` |
| `md-simulation` | OpenMM + AmberTools | `conda create -n <proj>-md -c conda-forge python=3.11 openmm ambertools mdanalysis pdbfixer` | none (force fields ship with AmberTools) |
| `peptide-diffusion` | (provisioned separately) | see its `SKILL.md` | see its `SKILL.md` |

## HPC job checklist

Before submitting a GPU job that uses these skills:

1. Env built on a login node into `$PROJECT_ENVS` (Python 3.11), and it imports
   cleanly there.
2. `torch.version.cuda` matches the node's CUDA; any missing NVIDIA lib package
   pip-installed and **only its** `nvidia/<pkg>/lib` added to `LD_LIBRARY_PATH`.
3. All weights pre-downloaded on a login node into `$SHARED_WEIGHTS`;
   `HF_HOME` / `CHAI_DOWNLOADS_DIR` / `TMPDIR` point there.
4. In the job script (offline compute node): `export HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1` and re-export the same cache-dir vars.
5. GPU visible in the job (`python -c "import torch; print(torch.cuda.is_available())"`).
6. Raise thread caps for the compute node; the login-node `OMP_NUM_THREADS=1` was
   only to survive the shared login node.

## Per-tool setup details

Each tool skill carries its own `## Setup — build the environment (once per
project)` section with the exact recipe and tool-specific gotchas. Cross-refs:
`folding/SKILL.md`, `interactant-folding/SKILL.md`, `inverse-folding/SKILL.md`,
`protein-language-explore/SKILL.md`, `md-simulation/SKILL.md`,
`peptide-diffusion/SKILL.md`.
