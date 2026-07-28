---
name: peptide-diffusion
description: >-
  De novo generate binder backbones conditioned on a target receptor and a
  hotspot, using Generate's Chroma diffusion model. Given a receptor PDB, one or
  more hotspot selections, and desired binder lengths, produce novel binder
  backbone PDBs (which you then sequence with the inverse-folding skill and
  validate with the folding skill). Use when you want NOVEL folds rather than
  redesigning an existing backbone. GPU-heavy; needs large local Chroma weights
  and device-specific setup (the repo code hardcodes Intel .xpu()).
---

# De novo binder backbone generation (Chroma diffusion)

Generate new binder **backbones** from scratch, conditioned to sit against a
fixed target receptor and reach toward a chosen hotspot, using Generate
Biomedicines' **Chroma** diffusion model. This is the "novel fold" entry point of
a campaign: it invents backbone geometry (no sequence), which you then hand to
the **inverse-folding** skill for sequences and to the **folding** skill for
validation.

This is a **self-contained, instruction-only skill** — no wrapper script and no
`bindify` dependency. You (the agent) build the conditioners and run Chroma
sampling directly.

## When to use it

- You want **novel binder folds** against a target, not a redesign of an existing
  backbone.
- You have a **receptor PDB**, a **hotspot** to aim at, and target **binder
  lengths**, and want backbone candidates.

Pipeline: generate backbones here → **inverse-folding** to get sequences → 
**folding** to validate the complex.

## Setup — build the environment (once per project)

Chroma runs **in-process** on a GPU. Build a dedicated env and pre-download the
weights at project start. This follows the cross-cutting HPC conventions in
[ENVIRONMENTS.md](../ENVIRONMENTS.md) (offline compute nodes, shared weight
caches, CUDA-lib matching) — read that first if you're on a cluster.

```bash
# 1. Env — inherit a working base torch, or make a fresh conda env with a torch
#    whose CUDA (or XPU) matches your node.
python -m venv --system-site-packages <proj>-chroma && source <proj>-chroma/bin/activate

# 2. Install Chroma. Upstream:
pip install generate-chroma
#    OR a fork:  git clone <fork-url> chroma && pip install -e chroma --no-deps
#    Use --no-deps if the fork's requirements pin an unbuildable version
#    (e.g. scikit_learn==1.1.2 fails on Python 3.11); then add the light deps:
pip install nglview      # imported at module load even though only needed for viz
```

**Weights** download on first use, gated by a **free API key** you register at
`https://chroma-weights.generatebiomedicines.com` (academic license). In code:
`from chroma import api; api.register_key("<YOUR_KEY>")`, then `Chroma()` fetches
the `named:public` backbone + design weights. It caches them under
`gettempdir()/chroma_weights` — on HPC **set `TMPDIR` to SHARED project storage
and pre-run `Chroma(device="cpu")` once on a login node** (compute nodes are
usually offline), so the GPU job loads from the shared cache.

**GPU device / branch:**
- **NVIDIA (CUDA):** if sampling dies with `failed to open
  libnvrtc-builtins.so.X.Y`, your torch is a `cuX.Y` build missing its nvrtc
  runtime — `pip install nvidia-cuda-nvrtc-cu12==X.Y.*` (match
  `python -c "import torch; print(torch.version.cuda)"`) and append **only** that
  package's `.../site-packages/nvidia/cuda_nvrtc/lib` to `LD_LIBRARY_PATH` (never
  a whole other env's `lib/` — it shadows libtorch and breaks the import).
- **Intel XPU (Aurora):** use an XPU torch + `intel_extension_for_pytorch`; some
  forks ship a separate XPU branch (with `.xpu()` calls) vs a CUDA branch — pick
  the branch matching your hardware.
- Login-node weight pre-download can hit `libgomp: Thread creation failed` → set
  `OMP_NUM_THREADS=1` for that step.

**Fork caveat:** some forks have a `self.centered_pdb_file` AttributeError in
`Chroma.__init__` / `graph_backbone.py` / `graph_design.py` — the attribute is
read but never assigned (it should be the local `centered_pdb_file` param). Patch
those reads if you hit it.

Gate before running (adapt `cuda`→`xpu` on Intel):

```bash
python -c "from chroma import Chroma, Protein, conditioners; import torch; \
print('ready' if torch.cuda.is_available() else 'no-cuda-gpu')"
```

Also needed: **Biopython** (`Bio.PDB`) for CIF→PDB and **MDAnalysis** for hotspot
vectors (both usually present via the base env).

## Procedure

Inputs you supply: receptor PDB (`input_dir`), one or more `hotspots`,
`binder_lengths`, `n_rounds`, `n_backbones` (samples per round). Weights come in
as file paths.

### 1. Load the model

```python
import torch
from chroma import Chroma, Protein, conditioners, api

api.register_key("<YOUR_KEY>")                     # from chroma-weights.generatebiomedicines.com
device = "cuda" if torch.cuda.is_available() else "xpu"
model = Chroma(device=device)                      # loads cached named:public weights
# or pass explicit checkpoints: Chroma(weights_backbone=<path>, weights_design=<path>, device=device)
```

### 2. Build the conditioners

Compose two conditioners so the receptor stays fixed while a new binder region is
pushed toward the hotspot:

- **`SubstructureConditioner`** — holds the receptor fixed. Select it with
  `'namesel receptor'` (the receptor substructure).
- **`InflateConditioner(vector, magnitude)`** — pushes the newly added binder
  region toward the hotspot. Compute `vector` from the hotspot **center-of-mass**
  (use MDAnalysis to select the hotspot atoms on the receptor and take their COM
  direction) and pick a `magnitude` that reaches the binder region.

Then compose them into one conditioner (Chroma composes conditioners), and
**extend the protein's X / C / S tensors with a length-N binder region** (N =
the chosen `binder_length`) so the model has residues to diffuse into a binder.

### 3. Sample

```python
out = model.sample(
    protein_init=protein,          # receptor + appended length-N binder region
    conditioner=conditioner,       # composed Substructure + Inflate
    design_selection=mask_aa,      # which positions are designed
    langevin_factor=2,
    langevin_isothermal=True,
    inverse_temperature=8.0,
    sde_func="langevin",
    full_output=True,
    steps=steps,
    samples=n_backbones,           # backbones per round
    num_designs=N,
)
```

### 4. Save

Write the sampled structure to **CIF**, then convert CIF→**PDB** with Biopython
(`Bio.PDB`). Loop over `binder_lengths` × `n_rounds` to build a pool of backbone
PDBs.

## Interpreting the result

- Output is a set of **binder backbone PDBs** (geometry only — no meaningful
  sequence yet). Diversity grows with more rounds/lengths.
- These are candidates, not validated designs: run **inverse-folding** on each
  backbone to get sequences, then **folding** to fold and score the complex, and
  rank by the folding interface metrics.

## Cost / fidelity

High cost: diffusion sampling on a GPU with large weights, multiplied over
lengths × rounds × samples. Output is a hypothesis pool; real fidelity comes only
after sequencing and re-folding.

## Notes

- Deciding hotspots, binder lengths, how many rounds/backbones, and which
  candidates to promote is your job as the agent.
- You are responsible for adapting the hardcoded Intel `.xpu()` device calls to
  your hardware and for supplying correct weight paths — the repo does neither
  for you.
- Cross-references: **inverse-folding** (sequence the backbones),
  **folding** (fold and score the resulting complexes).
