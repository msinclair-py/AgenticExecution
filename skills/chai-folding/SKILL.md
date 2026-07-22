---
name: chai-folding
description: Predict 3D structure of proteins and complexes from sequence with Chai-1 (chai_lab). Covers the full range of folding conditions — single chain, multi-chain / homo- & hetero-multimeric complexes, co-folding a binder against a target, ligands (SMILES), glycans and other PTMs, and restraint-guided folding via the constraint file (pocket / contact / covalent / disulfide). Also covers reading the 5 model outputs and scoring interfaces with ipSAE / ipTM / pDockQ. Use for any sequence→structure task. For designing the sequence that gets folded see proteinmpnn (inverse folding) or chroma-diffusion (backbone design); to simulate the folded complex see md-simulation.
license: Apache-2.0 (Chai-1 / chai_lab is Apache-2.0; this skill text is project-local)
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: chai_lab run_inference; molecular_simulations.analysis.ipSAE
---

# Chai-1 Structure Prediction

## Overview

Chai-1 (`chai_lab`) is an AlphaFold3-class folding model. It takes a FASTA
where each record is one **chain** of a **complex** — proteins, nucleic acids,
ligands (as SMILES), and glycans — optionally guided by a **constraint file**,
and emits 5 predicted models (`.cif`) plus per-model confidence arrays
(`.npz`). This skill emphasizes the workflow used in this project:

1. **Build the FASTA** — one record per chain, laid out in a deterministic
   chain order, named by the letter each chain will receive.
2. **Run `run_inference`** — one call folds the whole complex.
3. **(Optional) constraints** — a CSV that pins pockets, contacts, covalent
   links (glycans, non-canonical bonds) or disulfides.
4. **Postprocess** — convert the best `.cif` to PDB and score the interface
   (ipSAE / ipTM / pDockQ) with `molecular_simulations.analysis.ipSAE`.

The single most important rule: **Chai assigns chain letters strictly by order
of appearance in the FASTA.** Everything downstream — constraint chain letters,
which chain is the binder, where a glycan attaches — depends on this order.
Lay the FASTA out as **target chain(s) → glycans/ligands → binder**, and name
each record by the letter it will get (`>protein|A`, `>glycan|B`,
`>protein|C`). See `references/constraints_reference.md` for the FASTA entity
grammar and the constraint CSV schema.

---

## 1. The core call: `run_inference`

```python
from pathlib import Path
from chai_lab.chai1 import run_inference

run_inference(
    fasta_file=Path('complex.fa'),   # one record per chain (see §2)
    output_dir=Path('out/'),         # 5 models land here as pred.model_idx_{0..4}.cif
    device=resolve_device(),         # 'cuda' | 'xpu' | 'cpu' — see scripts/_device.py
    use_esm_embeddings=True,         # single-sequence ESM embeddings (no MSA needed)
    num_diffn_timesteps=100,         # diffusion steps; ~100–200. More = slower, marginally better
    constraint_path=None,            # Path to a constraint CSV, or None (see §4)
    full_data=True,                  # also write the score .npz needed for ipSAE/postprocess
)
```

Outputs written to `output_dir`:
- `pred.model_idx_{0..4}.cif` — 5 predicted structures (mmCIF).
- `scores.model_idx_{0..4}.npz` — per-model confidence: pLDDT and PAE arrays,
  plus aggregate/ptm/iptm fields. Required by the ipSAE scorer and postprocess.

`run_inference` is heavy (loads the model + ESM). In a Parsl/worker setting,
instantiate it **once per worker** and reuse it — do not reload per fold.

---

## 2. Building the FASTA (chain order is the contract)

Each record is `>ENTITY|CHAINLETTER` then the sequence on the next line. Entity
types: `protein`, `ligand` (value is a **SMILES** string), `glycan` (value is a
glycan tree string), `rna`, `dna`.

```
>protein|A
MKTAYIAKQR...        # target chain A
>protein|B
GSHMEDLLK...         # target chain B  (multi-chain target)
>glycan|C
MAN(6-1 MAN)(3-1 MAN)... # a glycan, its own chain C
>ligand|D
CC(=O)Oc1ccccc1C(=O)O    # a ligand (SMILES), chain D
>protein|E
DKINDESIGNEDBINDER...    # binder, folded last -> chain E
```

**Why name each record by its letter:** it keeps the FASTA aligned with the
constraint chain letters and forces every chain to be a distinct entity — so a
homo-dimeric target actually folds as two chains instead of collapsing into one.
A single-chain input still lands on chain A unchanged.

A multi-chain target is often carried as one delimited string
(`"SEQ_A:SEQ_B"`, AF-multimer convention); split on `:` and emit one record per
chain. `scripts/fold_multichain_target.py` shows this.

---

## 3. Folding conditions (which script to use)

| Condition | Script | Notes |
|-----------|--------|-------|
| Single chain (monomer) | `scripts/fold_single_chain.py` | Simplest case; chain A only. |
| Target + binder co-fold | `scripts/fold_complex_binder.py` | The core binder-design case: target → binder, binder is last. |
| Multi-chain target complex | `scripts/fold_multichain_target.py` | Hetero-/homo-multimer; split on `:`, one record per chain. |
| With a ligand | `scripts/fold_with_ligand.py` | `>ligand|X` + SMILES; optional pocket constraint to place it. |
| With glycans / PTMs | `scripts/fold_with_glycans.py` | `>glycan|X` chains + `covalent` constraints linking C1 to the acceptor atom. |
| Restraint-guided | `scripts/fold_with_constraints.py` | pocket / contact hotspots, disulfides. |
| Score the result | `scripts/score_ipsae.py` | Best-model selection + ipSAE/ipTM/pDockQ. |

Each script imports `resolve_device` from `scripts/_device.py` so it runs on
NVIDIA (`cuda`), Intel/Aurora (`xpu`), or CPU without edits.

---

## 4. Constraints (the CSV, and the dict shorthand)

Constraints guide folding toward hotspots or enforce covalent geometry. The
file is a CSV with this header:

```
chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id
```

This project builds it from a compact dict `{int: {...}}` where each entry has
`chainA, resA, chainB, resB, const_type, distance`; the writer fills
`confidence=1.0`, `min_distance=0.0`, `max_distance=distance`. Two canonical
forms:

**Pocket / hotspot** — tell the binder to bind near a target residue,
non-specifically:
```python
{0: {'chainA': 'B', 'resA': '', 'chainB': 'A', 'resB': 'N9',
     'const_type': 'pocket', 'distance': 5.0}}
```
Chain B (binder) is constrained to sit within 5 Å of residue N9 on chain A.

**Covalent (glycan / disulfide / non-canonical bond)** — link two specific
atoms. The `resX@atom` notation names the atom; a glycan is always linked from
its `C1` and its residue index is empty (`@C1`):
```python
# glycan C1 <-> Asn N of residue 343 on chain A
{0: {'chainA': 'A', 'resA': 'N343@ND2', 'chainB': 'C', 'resB': '@C1',
     'const_type': 'covalent', 'distance': 1.5}}
# disulfide: SG of Cys55 (chain A) <-> SG of Cys90 (chain A)
{1: {'chainA': 'A', 'resA': 'C55@SG', 'chainB': 'A', 'resB': 'C90@SG',
     'const_type': 'covalent', 'distance': 2.05}}
```

`connection_type`/`const_type` values: `pocket`, `contact`, `covalent`.
`scripts/write_constraints.py` turns the dict into a valid CSV; the fold scripts
call it. Full grammar and the N-/O-linked glycan acceptor-atom rules are in
`references/constraints_reference.md`.

**Rule:** every `covalent` link to a glycan needs a matching `>glycan|X` record
in the FASTA — the constraint references a chain that must exist. A covalent
glycan link with no glycan chain is an error.

---

## 5. Postprocessing & interface scoring

`run_inference` writes 5 models. Pick the best and score it. This project ranks
by **ipSAE** (interface score from aligned errors) via
`molecular_simulations.analysis.ipSAE`, which also returns ipTM, pDockQ,
pDockQ2 and LIS per chain pair.

```python
import gemmi
from molecular_simulations.analysis import ipSAE

best_score, best = -float('inf'), 0
for i in range(5):
    scorer = ipSAE(
        structure_file=out / f'pred.model_idx_{i}.cif',
        plddt_file=out / f'scores.model_idx_{i}.npz',   # pLDDT lives in the npz
        pae_file=out / f'scores.model_idx_{i}.npz',     # PAE too
    )
    scorer.run()
    s = scorer.scores['ipSAE'].mean()      # mean over chain pairs
    if s > best_score:
        best_score, best = s, i

model = out / f'pred.model_idx_{best}.cif'
gemmi.read_structure(str(model)).write_pdb('best.pdb')   # cif -> pdb for MD / MPNN
```

Interpreting the metrics (all higher = more confident interface): **ipSAE**
(headline), **ipTM**, **pDockQ / pDockQ2**, **LIS**. `scores` is a Polars
DataFrame (one row per ordered chain pair). See the `analyze-ipsae` skill in the
`molecular-simulations` repo and `scripts/score_ipsae.py`.

> A poorly-folded de novo binder legitimately produces **ipSAE = 0**. Use an
> explicit `is None` / `> best` comparison when picking a headline scalar — never
> `x or fallback` — so a real 0.0 isn't silently replaced.

---

## Best Practices

- **Chain order = the contract.** target → glycans/ligands → binder, each record
  named by its letter. Everything downstream depends on it.
- **`full_data=True`** or you won't get the `.npz` scores ipSAE/postprocess need.
- **Instantiate the model once per worker**; reloading per fold dominates runtime.
- **cif → pdb** with `gemmi` (or `Bio.PDB`) before handing structures to MD /
  ProteinMPNN, which expect PDB.
- **Every covalent-to-glycan constraint needs its `>glycan|X` FASTA record.**
- **Score with ipSAE, not raw pLDDT**, when ranking interfaces / binders.
- **Device-agnostic:** use `resolve_device()`; Chai runs on `cuda`, `xpu`, `cpu`.
- Chai writes into a scratch/`/dev/shm` dir in the toolbelt for speed; large
  campaigns benefit from folding into node-local fast storage then copying out.

## Common Pitfalls

1. FASTA records out of order (binder not last) → the binder isn't the final
   chain, breaking downstream binder-chain identification and MM-PBSA splits.
2. A homo-multimer written as one record → collapses to a single chain.
3. Covalent glycan constraint without a `>glycan|X` record → dangling chain ref.
4. Using `x or 0.0` to pick a headline score → a real ipSAE 0.0 gets hidden.
5. `full_data=False` → no `.npz`, so ipSAE/postprocess can't run.
6. Passing a `.cif` straight to ProteinMPNN/AmberTools → convert to PDB first.
7. Multi-chain target string not split on `:` → target folds as one giant chain.
8. Reloading `run_inference`'s model per call in a loop → needless minutes each.

## Resources

### references/
- `constraints_reference.md` — FASTA entity grammar (protein/ligand/glycan/na),
  the constraint CSV schema, the dict shorthand, and N-/O-linked glycan
  acceptor-atom rules.

### scripts/
- `_device.py` — `resolve_device()` helper (cuda/xpu/cpu).
- `write_constraints.py` — dict → Chai constraint CSV.
- `fold_single_chain.py` — monomer.
- `fold_complex_binder.py` — target + binder co-fold (core case).
- `fold_multichain_target.py` — multi-chain target complex.
- `fold_with_ligand.py` — protein + ligand (SMILES) + pocket constraint.
- `fold_with_glycans.py` — glycan chains + covalent linkages.
- `fold_with_constraints.py` — pocket/contact hotspots and disulfides.
- `score_ipsae.py` — best-model selection + interface scoring.

Run scripts directly (`python scripts/fold_complex_binder.py ...`) or import
them as templates.
