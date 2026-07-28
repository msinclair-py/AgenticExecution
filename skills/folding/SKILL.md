---
name: folding
description: >-
  Predict the 3D structure of a target+binder protein complex (co-folding) from
  sequences using the Chai-1 model, and score the interface (ipSAE, ipTM,
  pDockQ, pDockQ2, LIS). Use when you have a target sequence and a designed
  binder sequence (optionally glycan/PTM chains and residue constraints) and need
  a complex structure plus interface-confidence metrics — e.g. to rank binder
  designs or to produce a structure you then feed to the contact-eval skill.
  GPU-heavy; needs a pre-provisioned Chai-1 environment with local model weights.
---

# Co-folding a target+binder complex (chai_lab / Chai-1)

Fold a two-or-more-chain complex (campaign target + designed binder, optionally
glycan/PTM chains) with the Chai-1 model, run five diffusion models, score each
interface with ipSAE, and return the best model as a PDB alongside its interface
metrics. The headline confidence is **ipSAE** (fall back to ipTM, then pDockQ).
This is the expensive, high-fidelity structural step of a design campaign:
minutes-to-tens-of-minutes on a GPU per complex, gated by local model weights.

This is a **self-contained, instruction-only skill** — no wrapper script and no
`bindify` dependency. You (the agent) write and run the Chai-1 inference and the
ipSAE post-processing directly over the base libraries.

## When to use it

- You have a target sequence and a binder sequence and want their **3D complex**
  structure predicted from scratch (co-folding).
- You want **interface-confidence metrics** to rank or filter binder designs.
- You need a complex structure to hand to the **contact-eval** skill (interface
  residues / hotspot footprint), or you have sequences produced by the
  **inverse-folding** skill and want to validate them by re-folding.

For co-folding the target with a *known natural interactant* instead of a
designed binder, use the **interactant-folding** skill (same pipeline; it just
substitutes the interactant sequence for the binder).

Do **not** use it if you only need contacts on an existing structure — go
straight to contact-eval.

## Setup — build the environment (once per project)

This skill runs the Chai-1 model **in-process** (no HTTP, no subprocess) and
requires a machine with a working accelerator (CUDA GPU or Intel XPU) and the
**local Chai-1 + ESM embedding weights already downloaded**. Build the env once,
on an internet-capable login node, into shared project storage — see
`../ENVIRONMENTS.md` for the cross-cutting HPC rules (offline compute nodes,
matching CUDA libs, shared caches).

```bash
conda create -n <proj>-folding python=3.11 -y && conda activate <proj>-folding
pip install chai_lab gemmi      # gemmi for CIF->PDB; a pinned fork may be used instead
export CHAI_DOWNLOADS_DIR=$SHARED_WEIGHTS/chai   # persistent, shared cache for weights
```

- `chai_lab` — the Chai-1 model. Import: `from chai_lab.chai1 import run_inference`.
  A pinned fork (e.g. `git+https://github.com/msinclair-py/chai-lab.git`) may be
  used instead of the PyPI package.
- `gemmi` — CIF→PDB conversion.
- `molecular_simulations` — provides `molecular_simulations.analysis.ipSAE` for
  interface scoring (install if your env doesn't already have an ipSAE impl).

**Pre-download the weights on a login node.** Chai-1 fetches large model + ESM
embedding weights on first run; run one tiny fold on an internet-capable node so
`$CHAI_DOWNLOADS_DIR` fills, and the offline compute nodes reuse that shared dir.
Chai-1 also works out of shared memory (`/dev/shm`) at run time, so the compute
node needs enough `/dev/shm`. On offline compute nodes also
`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (the ESM embeddings come via
HuggingFace).

Gate on an import check that fails gracefully with a clear message:

```bash
python - <<'PY'
import sys
try:
    from chai_lab.chai1 import run_inference          # the fork
    from molecular_simulations.analysis.ipSAE import ipSAE
    import gemmi, torch
except Exception as e:
    sys.exit(
        "folding skill unavailable: needs a pre-provisioned GPU env with the "
        "msinclair-py chai_lab fork, molecular_simulations, gemmi and local "
        "Chai-1/ESM weights. Missing: %r" % (e,)
    )
print("ready")
PY
```

If this prints anything other than `ready`, stop and report that the environment
is not provisioned — do not attempt to `pip install` your way to the weights.

## Procedure

### 1. Build the FASTA

Chai-1 reads a FASTA where each record header is `>type|name`. Order matters for
downstream steps: **write the target first and the binder LAST** (the
inverse-folding skill relies on the binder being the last chain).

```
>protein|target
<TARGET_SEQUENCE>
>glycan|<chain>        # optional PTM / glycan lines, one per modification
<GLYCAN_SPEC>
>protein|binder
<BINDER_SEQUENCE>
```

Include the `>glycan|...` records only if the target carries PTMs/glycans you
want modeled; omit otherwise.

### 2. (Optional) Constraints

To bias the fold toward a known contact, write a CSV named `constraints.txt` with
exactly these columns:

```
chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance,max_distance,comment,restraint_id
```

Pass its path as `constraint_path`; pass `None` when unconstrained.

### 3. Run inference (in-process)

**GOTCHA:** `run_inference` asserts the `output_dir` is **EMPTY**. Keep the input
FASTA (and constraints CSV) OUTSIDE `output_dir`, and pass a fresh empty
directory as `output_dir` — otherwise the assertion fails before any compute.

```python
import tempfile, torch
from pathlib import Path
from chai_lab.chai1 import run_inference

device = "cuda" if torch.cuda.is_available() else "xpu"   # or the env's accelerator
work = Path(tempfile.mkdtemp(dir="/dev/shm"))            # Chai-1 works out of /dev/shm
fasta = work / "input.fasta"                             # input lives OUTSIDE output_dir
fasta.write_text(FASTA_TEXT)
out = work / "out"; out.mkdir()                          # fresh EMPTY output dir

run_inference(
    fasta_file=fasta,
    output_dir=out,
    device=device,
    use_esm_embeddings=True,
    num_diffn_timesteps=100,
    constraint_path=CONSTRAINTS_PATH_OR_NONE,   # Path to constraints.txt, or None
    full_data=True,
)
# Output: 5 models  pred.model_idx_{i}.cif  +  scores.model_idx_{i}.npz  (i = 0..4)
```

### 4. Post-process: score, pick best, convert

Run ipSAE on each of the 5 models, pick the model with the **highest mean ipSAE**,
convert its CIF to PDB with gemmi, and collect the interface metrics.

```python
import numpy as np, gemmi
from molecular_simulations.analysis.ipSAE import ipSAE

best = None
for i in range(5):
    cif = out / f"pred.model_idx_{i}.cif"
    scores = ipSAE(str(cif))                    # interface scoring for this model
    mean_ipsae = float(np.mean(scores["ipSAE"]))
    if best is None or mean_ipsae > best["mean_ipsae"]:
        best = {"idx": i, "cif": cif, "mean_ipsae": mean_ipsae, "scores": scores}

# best CIF -> PDB
st = gemmi.read_structure(str(best["cif"]))
pdb_path = out / f"best_model_{best['idx']}.pdb"
st.write_pdb(str(pdb_path))

metrics = {
    "ipSAE":  best["mean_ipsae"],
    "ipTM":   float(best["scores"].get("ipTM",  float("nan"))),
    "pDockQ": float(best["scores"].get("pDockQ", float("nan"))),
    "pDockQ2":float(best["scores"].get("pDockQ2",float("nan"))),
    "LIS":    float(best["scores"].get("LIS",    float("nan"))),
}
```

## Interpreting the result

- **Headline confidence = ipSAE.** If ipSAE is unavailable for a model, fall back
  to **ipTM**, then **pDockQ**. Higher is better for all of these.
- Rank/filter binder designs by the headline scalar of their best model.
- The returned PDB is the best-scoring of the 5 diffusion models; feed it to the
  **contact-eval** skill to get the interface residue footprint.

## Cost / fidelity

High fidelity, high cost: five diffusion trajectories on a GPU per complex, plus
ESM embeddings. Fidelity of everything downstream (contacts, metrics) is bounded
by this fold. Budget accordingly and prefer batching many complexes on the
provisioned node.

## Notes

- Deciding *which* target/binder pairs to fold, how to sweep constraints, and
  what to do with the metrics (rank, threshold, promote to contact-eval) is your
  job as the agent — this skill only computes one complex's structure and scores.
- Cross-references: **contact-eval** (interface residues from the output PDB),
  **inverse-folding** (redesign the binder on this backbone, then re-fold here),
  **interactant-folding** (same pipeline with a natural partner).
