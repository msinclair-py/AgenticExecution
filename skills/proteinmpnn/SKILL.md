---
name: proteinmpnn
description: Inverse folding (structure → sequence) with ProteinMPNN. Given a backbone (PDB), design new amino-acid sequences that fold to it. Covers the in-process API (import protein_mpnn_utils, no subprocess) and the CLI/subprocess path; designing only the binder chain while keeping a multi-chain target fixed; fixing interface positions (fixed_positions); sampling temperature; the noise-level model variants (v_48_002…030) and soluble vs vanilla weights; and batch sampling to hit a target sequence count. Use to redesign a de novo backbone, diversify a hit's sequence, or recover native-like sequences. Feeds the sequences back to chai-folding for validation.
license: MIT (ProteinMPNN is MIT; this skill text is project-local)
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: ProteinMPNN protein_mpnn_utils (in-process); protein_mpnn_run.py (CLI)
---

# ProteinMPNN Inverse Folding

## Overview

ProteinMPNN designs **sequences** for a fixed **backbone**. You give it a PDB;
it samples amino-acid sequences predicted to fold to that geometry. It is fast
(seconds–minute per backbone, batched), which makes it the workhorse for
diversifying binder sequences before the slower folding/MD validation steps.

Two ways to run it, both covered here:

1. **In-process** (recommended) — import `protein_mpnn_utils` and run inference
   in the current process. Avoids a `torch.xpu`/CUDA fork+exec conflict that can
   occur when a worker already initialized the GPU (e.g. from a prior Chai fold)
   and then spawns a subprocess that re-inits the GPU. See
   `scripts/mpnn_runner.py`.
2. **CLI / subprocess** — shell out to the repo's `protein_mpnn_run.py` with the
   helper scripts. Kept as a fallback. See `scripts/inverse_fold_cli.py`.

### Prerequisite: `PROTEINMPNN_PATH`

ProteinMPNN ships `protein_mpnn_utils.py`, `protein_mpnn_run.py`,
`helper_scripts/`, and the weight directories (`soluble_model_weights/`,
`vanilla_model_weights/`) **inside its GitHub clone**, not as a pip package.
Point `PROTEINMPNN_PATH` at that clone; the runner prepends it to `sys.path`
before importing, and loads weights from `<PROTEINMPNN_PATH>/<model_weights>/<model_name>.pt`.

```bash
export PROTEINMPNN_PATH=/path/to/ProteinMPNN
```

---

## 1. The design contract: which chain, which positions

**Design only the binder chain; keep the target fixed.** In a Chai co-fold the
target chain(s) come first and the binder is last, so after stripping
non-protein chains the **binder is the last chain**. Designing only that chain
and leaving every target chain fixed is exactly ProteinMPNN's native multi-chain
case — a multi-chain target needs no special handling. `mpnn_runner.py`
identifies the binder as the last chain in the (cleaned) PDB.

**`fixed_positions` preserves chemistry you must keep.** Pass 1-indexed residue
IDs (within the designed chain) to hold constant — everything else is
re-sampled. Use it to lock a binding interface, a catalytic residue, or a motif.

> ⚠️ **Naming footgun:** the project's config field is called
> `remodel_positions`, but it is passed straight into ProteinMPNN's
> `fixed_positions` — i.e. it lists the residues to **KEEP FIXED**, *not* the
> ones to remodel. Everything **not** listed is what gets redesigned. Double-check
> the sense before wiring a campaign.

Non-canonical residues (glycans, PTMs, cofactors, solvent) are dropped before
design — ProteinMPNN can only see the 20 standard amino acids (MSE is kept and
treated as MET). `mpnn_runner.py` does this cleaning pass automatically.

---

## 2. In-process design (the common path)

```python
import os
from pathlib import Path
from mpnn_runner import design_sequences   # scripts/mpnn_runner.py

seqs = design_sequences(
    pdb_path=Path('complex.pdb'),          # a folded PDB (target + binder)
    proteinmpnn_path=Path(os.environ['PROTEINMPNN_PATH']),
    num_seq=64,                            # how many sequences to return (deduped)
    batch_size=64,                         # per-batch sample count (VRAM-bound; <=225)
    sampling_temp=0.1,                     # 0.1 conservative … 0.3 diverse
    model_name='v_48_020',                 # noise level (see §4)
    model_weights='soluble_model_weights', # 'soluble' | 'vanilla' (see §4)
    fixed_positions=[],                    # 1-indexed binder residues to KEEP
)
for s in seqs:
    print(s)
```

Returns a deduplicated `list[str]` of designed binder sequences (multi-chain
outputs are '/'-joined, but since only the binder is designed you get one
segment). Each is a candidate to fold back with `chai-folding` and score.

---

## 3. Conditions (which script)

| Condition | Script |
|-----------|--------|
| Redesign the binder of a complex (in-process) | `scripts/inverse_fold_single.py` |
| Preserve interface / motif residues | `scripts/inverse_fold_fixed_positions.py` |
| Multi-chain target (native case) | `scripts/inverse_fold_multichain.py` |
| Sweep temperature × model variant | `scripts/inverse_fold_temperature_sweep.py` |
| CLI / subprocess fallback | `scripts/inverse_fold_cli.py` |

---

## 4. Choosing the model

**`model_name` — backbone-noise level** (all use 48 message-passing layers; the
suffix is the Å of Gaussian noise added to backbone coords during training):

| Model | Noise | Use when the backbone is… |
|-------|-------|---------------------------|
| `v_48_002` | 0.02 Å | a near-exact high-res crystal (≤1.5 Å); most structure-faithful, brittle to error. |
| `v_48_010` | 0.10 Å | well-resolved experimental (1.5–2.0 Å). |
| `v_48_020` | 0.20 Å | **default.** Tolerates typical X-ray (≤2.5 Å) / high-confidence AF2 (pLDDT ≥ 85). |
| `v_48_030` | 0.30 Å | low-res / cryo-EM / homology / **predicted** backbones (pLDDT 70–85). |

Rule of thumb: **Chai/AF2 predicted backbones → `v_48_020` or `v_48_030`**, since
they carry coordinate uncertainty. When in doubt, `v_48_020`.

**`model_weights` — training-set bias:**
- `soluble_model_weights` (**default**) — biased toward soluble, monomeric,
  globular proteins: fewer exposed hydrophobics, lower aggregation. Best for most
  cytoplasmic/secreted design.
- `vanilla_model_weights` — full PDB, no solubility filter. Use for
  membrane-associated, disordered, or large-complex targets, or when the soluble
  bias over-penalizes needed hydrophobics.

**`sampling_temp`** — 0.1 recovers native-like/conservative sequences; 0.2–0.3
increase diversity (and the risk of exploring unfavourable sequence space). Start
at 0.1; raise to diversify a stalled campaign.

**`batch_size` / `num_seq`** — the runner samples in batches until it has
`num_seq` unique sequences (or `max_retries` batches elapse; some are duplicates
or fail QC). Keep `batch_size ≤ ~225` on most GPUs (VRAM-bound).

---

## Best Practices

- **Set `PROTEINMPNN_PATH`** to the repo clone (has utils + weights). Missing →
  hard error.
- **Prefer the in-process runner** to dodge the GPU fork+exec conflict after a
  same-worker fold.
- **Design the binder (last chain); fix the target.** Don't redesign target
  chains you want to preserve.
- **`fixed_positions` = keep, not remodel.** Mind the `remodel_positions` name.
- **Predicted backbones → higher-noise model** (`v_48_020`/`v_48_030`).
- **Validate every design by folding it back** (chai-folding) and scoring ipSAE —
  ProteinMPNN gives sequences, not a guarantee they fold to the target.
- **Device-agnostic:** the runner resolves `xpu`/`cuda`/`cpu`.

## Common Pitfalls

1. Confusing `remodel_positions`/`fixed_positions` sense → you freeze the wrong
   residues (or redesign the interface you meant to keep).
2. Forgetting `PROTEINMPNN_PATH` → `ModuleNotFoundError: protein_mpnn_utils`.
3. Subprocess path after a same-worker GPU fold → torch device re-init conflict;
   use the in-process runner.
4. Feeding a glycosylated/ligand-bound PDB without cleaning → non-canonical
   residues confuse parsing (the runner strips them, the raw CLI may not).
5. Using a low-noise model (`v_48_002`) on a predicted backbone → brittle,
   over-fit sequences.
6. Treating ProteinMPNN output as validated — always fold + score it.
7. `batch_size` too large for VRAM → OOM.

## Resources

### references/
- `api_reference.md` — full argument reference for the in-process runner and the
  CLI, plus the output format.

### scripts/
- `_device.py` — `resolve_device()` (xpu/cuda/cpu).
- `mpnn_runner.py` — self-contained in-process design (load model, pick binder
  chain, fix positions, batch-sample, dedup).
- `inverse_fold_single.py` — redesign a complex's binder chain.
- `inverse_fold_fixed_positions.py` — keep interface/motif residues fixed.
- `inverse_fold_multichain.py` — multi-chain target, binder-only design.
- `inverse_fold_temperature_sweep.py` — temperature × model grid.
- `inverse_fold_cli.py` — subprocess fallback via protein_mpnn_run.py.
