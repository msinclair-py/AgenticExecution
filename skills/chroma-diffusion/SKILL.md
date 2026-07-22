---
name: chroma-diffusion
description: De novo binder backbone design with the Chroma generative diffusion model. Given a target structure (PDB) and a hotspot patch, diffuse new binder backbones docked against that epitope by composing conditioners — SubstructureConditioner (hold the receptor fixed), InflateConditioner (place the binder at a hotspot vector), and optionally SubsequenceConditioner (graft a fixed motif/subsequence into the binder). Covers protein tensor prep (X/C/S), the design mask, sampling parameters (steps, samples, num_designs, Langevin temperature), and saving CIF→PDB. Use to generate binder scaffolds against a target; feed the backbones to proteinmpnn for sequences, then chai-folding + md-simulation to validate.
license: Apache-2.0 (Chroma / generate; this skill text is project-local)
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: chroma (Chroma, Protein, conditioners) — ChromaDesigner pattern
---

# Chroma Binder Diffusion

## Overview

Chroma is a protein backbone diffusion model. For **binder design** you load a
target structure, append an empty binder segment, and run reverse diffusion that
only remodels the binder while **conditioners** keep the target fixed and steer
the binder against a chosen epitope. The output is a set of backbones (CIF→PDB)
to sequence with `proteinmpnn` and validate with `chai-folding` + `md-simulation`.

Pipeline position: **chroma (backbone) → proteinmpnn (sequence) → chai-folding
(structure/ipSAE) → md-simulation (stability/ΔG).**

Three moving parts:
1. **Model** — `Chroma(weights_backbone, weights_design, device)`.
2. **Protein tensors** — a `Protein` decomposes to `X, C, S` (coords, chain
   labels, sequence). You extend them to add the binder segment and build a
   **design mask** so only the binder is sampled.
3. **Conditioners** — composed constraints that fix the receptor and place the
   binder at a hotspot (and optionally graft a subsequence).

> The reference `chroma.py` (ChromaDesigner) is Aurora/XPU-specific (`.xpu()`
> hardcoded). These scripts use device-agnostic `.to(device)` so they run on
> CUDA / XPU / CPU — see `scripts/_device.py`.

---

## 1. Load the model

```python
from chroma import Chroma
model = Chroma(
    weights_backbone=str(backbone_weights),   # backbone network weights
    weights_design=str(design_weights),       # design (sequence) network weights
    device=resolve_device(),                  # 'cuda' | 'xpu' | 'cpu'
)
```
Weights are local paths (or Chroma's registry). Load the model **once** and reuse
across rounds; it is expensive to construct.

---

## 2. Protein tensors and the design mask

A `Protein` built from the target PDB gives `X` (coords, shape `[1, L, 4, 3]`),
`C` (chain id per residue: **1 = receptor, 2 = binder**), and `S` (sequence
indices). To add a binder of length `len_binder` you concatenate empty tensors
for it and mark those residues as chain 2:

```python
import torch
from chroma import Protein

protein = Protein(str(pdb), device=device)
X, C, S = protein.to_XCS()

X = torch.cat([X, torch.zeros(1, len_binder, 4, 3).to(device)], dim=1)
C = torch.cat([C, torch.full((1, len_binder), 2).to(device)], dim=1)   # 2 = binder
S = torch.cat([S, torch.full((1, len_binder), 0).to(device)], dim=1)
protein = Protein(X, C, S, device=device)

# save the receptor residues as a named selection for the SubstructureConditioner
L_receptor = int((C == 1).sum())
protein.sys.save_selection(gti=list(range(L_receptor)), selname='receptor')
```

The **design mask** `mask_aa` (shape `[1, L, 20]`) tells `sample` which residues
to design: receptor positions are locked to their native amino acid; binder
positions are free (all 20 allowed). `scripts/chroma_designer.py` builds it.

---

## 3. Conditioners (compose them)

```python
from chroma import conditioners

substructure = conditioners.SubstructureConditioner(
    protein=protein,
    backbone_model=model.backbone_network,
    selection='namesel receptor',      # keep the receptor rigid
).to(device)

displacement = conditioners.InflateConditioner(vector, magnitude).to(device)  # place binder at hotspot

conditioner = conditioners.ComposedConditioner([displacement, substructure])
```

- **SubstructureConditioner** — pins the receptor substructure so diffusion only
  remodels the binder.
- **InflateConditioner(vector, magnitude)** — translates the binder toward a
  hotspot. Compute the vector as the displacement from the whole-target center of
  mass to the hotspot center of mass; scale its magnitude by `vector_scaling`:
  ```python
  import MDAnalysis as mda, numpy as np
  u = mda.Universe(str(pdb))
  hs = u.select_atoms(f'protein and resid {hotspot_resids}')   # e.g. '47 48 49'
  vector = hs.center_of_mass() - u.atoms.center_of_mass()
  magnitude = np.linalg.norm(vector) * vector_scaling
  vector = torch.tensor(vector, dtype=torch.float)
  ```
- **SubsequenceConditioner** — (subsequence design) grafts a fixed motif into the
  binder; see §5 and `scripts/design_subsequence.py`.

---

## 4. Sample

```python
proteins, trajectories = model.sample(
    protein_init=protein,
    conditioner=conditioner,
    design_selection=mask_aa,      # only the binder is designed
    langevin_factor=2,
    langevin_isothermal=True,
    inverse_temperature=8.0,       # higher = lower temperature = more conservative
    sde_func='langevin',
    full_output=True,
    steps=diff_steps,              # diffusion steps (e.g. 200–500)
    samples=n_backbones,           # independent backbones per call
    num_designs=n_designs,         # sequence designs per backbone
)
```
Save each `Protein` to CIF and convert to PDB:
```python
protein.to('out.cif')
# Bio.PDB MMCIFParser -> PDBIO, or gemmi.read_structure(cif).write_pdb(...)
```

**Sampling knobs:**
- `binder_lengths` — sample a length per round; 30–60 (fast, simple topologies),
  60–90 (typical mini-binder range), 90–120 (wraps complex epitopes, harder to
  design). Chroma degrades > ~120.
- `hotspots` — 1-indexed target residues defining the epitope patch. Choose
  solvent-exposed, rigid (low B-factor / high pLDDT), functionally relevant
  residues. 3–15 is a good range; too few → off-target contacts, too many →
  over-constrained, low diversity.
- `n_backbones` — 5–10 for exploration, 20–100 to sample a highly designable
  scaffold (scales through every downstream stage). 10–50 per length per round
  is a practical start.
- `inverse_temperature` / `langevin_factor` — sampling temperature; higher
  inverse_temperature = more conservative/native-like geometry.

---

## 5. Conditions (which script)

| Condition | Script |
|-----------|--------|
| Diffuse binder backbones at a hotspot | `scripts/design_binder_hotspot.py` |
| Graft a fixed subsequence/motif into the binder | `scripts/design_subsequence.py` |
| Config-driven multi-round campaign | `scripts/config_example.toml` + `chroma_designer.py` |

`scripts/chroma_designer.py` is a device-agnostic distillation of the project's
`ChromaDesigner` (round loop, input sampling, save + CIF→PDB).

---

## Best Practices

- **Load `Chroma` once**, reuse across rounds/backbones.
- **Chain labels are the contract:** receptor = 1, binder = 2. The design mask
  and `namesel receptor` depend on it.
- **Hotspot vector = target COM → hotspot COM**, scaled — this is what aims the
  binder at the epitope.
- **Pick rigid, exposed, functional hotspot residues**; 3–15 of them.
- **Keep binder length ≤ ~120**; Chroma degrades beyond its training range.
- **Save CIF then convert to PDB** for the downstream ProteinMPNN/Chai steps.
- **Device-agnostic:** `.to(resolve_device())`, not hardcoded `.xpu()`.
- Wrap the round loop in `try/except RuntimeError` to skip transient oneMKL/GPU
  hiccups (the reference `run_rounds` does this).

## Common Pitfalls

1. Not saving a `namesel receptor` selection → SubstructureConditioner can't pin
   the receptor.
2. Mislabeling binder residues (must be chain `2`) → design mask remodels the
   wrong residues or nothing.
3. Omitting the InflateConditioner → binder diffuses anywhere, not at the epitope.
4. Binder length > 120 → poor, misfolding-prone backbones.
5. Too many hotspot residues → over-constrained, low-diversity output.
6. Leaving CIF for downstream tools → convert to PDB first.
7. Hardcoding `.xpu()` on a CUDA/CPU host → device mismatch errors.

## Resources

### references/
- `conditioners_reference.md` — Substructure / Inflate / Subsequence /
  Composed conditioners, tensor shapes, and sampling arguments.

### scripts/
- `_device.py` — `resolve_device()` (xpu/cuda/cpu).
- `chroma_designer.py` — device-agnostic designer (model load, tensor prep, mask,
  conditioners, sample, save CIF→PDB).
- `design_binder_hotspot.py` — diffuse binder backbones against a hotspot.
- `design_subsequence.py` — graft a fixed motif into the binder.
- `config_example.toml` — config for a multi-round campaign.
