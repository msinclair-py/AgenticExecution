---
name: md-simulation
description: Run molecular dynamics and binding free-energy analysis on protein / protein-complex / protein-ligand systems with the molecular_simulations package (AmberTools build + OpenMM simulate + MM-PBSA). Covers building AMBER topologies (explicit OPC / implicit GB / protein-ligand GAFF2 / glycoprotein), running the minimize→heat→equilibrate→NPT-production pipeline with automatic restart, RMSD stability assessment of a binder against its target, and MM-P(G)BSA ΔG_binding with frame-parallelism. Self-contained: the full fold→MD→ΔG binder pipeline including glycan prep lives here. Use to assess whether a designed binder stays bound and estimate its affinity. Upstream: chai-folding produces the PDB; for the folding confidence score see chai-folding's ipSAE.
license: MIT-style (molecular_simulations; this skill text is project-local)
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: molecular_simulations.build / .simulate / .analysis; AmberTools; OpenMM
---

# Molecular Dynamics & Binding Free Energy

## Overview

`molecular_simulations` turns a structure into an MD trajectory and a binding
free energy in three stages:

1. **Build** (`.build`) — AmberTools (`tleap`/`cpptraj`/`antechamber`) parameterize
   the system → `system.prmtop` + `system.inpcrd` (+ `system.pdb`).
2. **Simulate** (`.simulate`) — OpenMM runs minimize → heat → restrained
   equilibration → NPT production, with checkpoint/restart.
3. **Analyze** — RMSD stability (MDAnalysis) gates whether the binder stayed
   bound; if so, **MM-P(G)BSA** estimates ΔG_binding.

**Requires AmberTools with `$AMBERHOME` set** (builders and MM-PBSA resolve
`tleap`/`cpptraj`/`mmpbsa_py_energy` under `$AMBERHOME/bin`; unset → `ValueError`).
Simulation runs on GPU (`CUDA`/`OpenCL`) or `CPU` — device-agnostic via
`scripts/_device.py`.

The binder pipeline in one line: **fold (chai) → build → simulate → RMSD gate →
MM-PBSA**. `scripts/fold_to_dG_pipeline.py` is the whole thing end to end.

---

## 1. Build the system (`molecular_simulations.build`)

Builders write `system.prmtop`/`system.inpcrd`/`system.pdb` into `path` and move
intermediates into `path/build/`. Point the simulator at `path`.

**Explicit solvent (most common)** — cubic OPC box, neutralized, ~150 mM NaCl:
```python
from pathlib import Path
from molecular_simulations.build import ExplicitSolvent

ExplicitSolvent(
    path=Path('out/'),           # where system.prmtop/.inpcrd land
    pdb=Path('complex.pdb'),
    padding=10.0,                # Å water padding on the longest axis
).build()
```
Options: `disulfide_residues=[23, 88]` (force CYX + let tleap bond — good for
binder design where distance-based detection is unwanted); `glycans=True` (see
§2); `rna`/`dna`/`phos_protein`/`mod_protein` toggle extra `leaprc`;
`polarizable=True` (ff15ipq/SPC-Eb); `debug=True` (write `tleap.in` to disk to
inspect a failed build).

**Implicit solvent (GB)** — no water box, `mbondi3` radii (pair with
`ImplicitSimulator`):
```python
from molecular_simulations.build import ImplicitSolvent
ImplicitSolvent(path='out/', pdb='protein.pdb', protein=True).build()
```
Note: **implicit solvent cannot handle glycans** (no cpptraj sugar prep → tleap
can't form glycosidic bonds). Glycoproteins must use explicit solvent.

**Protein–ligand complex** — GAFF2 + AM1-BCC charges via antechamber:
```python
from molecular_simulations.build import ComplexBuilder
ComplexBuilder(path='out/', pdb='protein.pdb', lig='ligand.sdf', padding=12.0).build()
```
Ligand residues become `LG0`, `LG1`, …; reuse params with
`lig_param_prefix='/path/params/ligand'`; a `LigandError` means antechamber/SQM
failed (check `build/*_sqm.out` for "Calculation Completed").

Helpers: `convert_cif_with_gemmi(cif)` / `convert_cif_with_biopython(cif)`
(CIF→PDB, since Chai emits CIF but you converted to PDB during scoring);
`add_chains(pdb, first_res, last_res)` (assign A/B by residue range).

---

## 2. Glycoproteins (two paths)

A Chai glycoprotein fold carries sugar residues (NAG, MAN, …) in PDB naming.
AMBER needs GLYCAM naming + explicit glycosidic/disulfide bonds. Two options:

**(a) Let `ExplicitSolvent(glycans=True)` do it (recommended).** Keep the
structure in Chai's PDB sugar names; the explicit builder's `cpptraj
prepareforleap` renames to GLYCAM and forms the bonds itself. **Pre-renaming to
GLYCAM would hide the sugars from cpptraj** — don't. For the MM-PBSA fragment
split to be clean, move the binder chain to *last* first (see
`scripts/prep_glycoprotein.py`). Also force `glycans=True`.

**(b) `GlycanConverter` (manual / inspectable).** Detects glycosidic + disulfide
bonds by geometry, assigns GLYCAM names, and writes a renamed PDB + a `tleap`
script with the bond commands:
```python
from molecular_simulations.build.glycam_mapper import GlycanConverter  # project material
result = GlycanConverter().convert_pdb('fold.pdb', output_dir='prep/', mol_name='PROT')
# result['glycam_pdb'], result['leap_script'], result['bond_commands'], ...
```
Use `mol_name='PROT'` for ExplicitSolvent, `'prot'` for ImplicitSolvent (must
match the tleap variable). See `references/glycan_reference.md`.

> The `GlycanConverter` module lives in the `bindify` toolbelt materials
> (`materials/glycam_mapper.py`); path (a) needs only `molecular_simulations`.

---

## 3. Simulate (`molecular_simulations.simulate`)

`Simulator` reads `system.prmtop`/`.inpcrd` from `path` and runs the full
pipeline; **re-invoking `run()` after a crash resumes** from the last checkpoint
(built for preemptible HPC queues).

```python
from pathlib import Path
from molecular_simulations.simulate import Simulator

ns, prod_dt = 100, 0.004                    # production ns; 4 fs timestep (HMR)
prod_steps = int(ns / prod_dt * 1000)       # ns -> steps  (1 ns = 250_000 steps)

Simulator(
    path=Path('out/'),
    equil_steps=1_250_000,                  # 2.5 ns @ 2 fs (default)
    prod_steps=prod_steps,
    temperature=300.0,
    platform=resolve_md_platform(),         # 'CUDA' | 'OpenCL' | 'CPU'
    device_ids=[0],
).run()
```
`run()`: skips equilibration if `eq.*` exist; else minimize → slow-heat 5 K→T →
relax backbone restraints in 5 stages → NPT for `n_equil_cycles`; then production
from the eq checkpoint (resumes from `prod.rst.chk`, reading `prod.log` for
remaining steps). Outputs: `prod.dcd` (frames every `prod_reporter_frequency`),
`prod.log`, checkpoints. Key args: `heat_steps`, `prod_dt_in_ps=0.004`,
`hydrogen_mass_repartitioning=1.5`, `force_constant=10.0`, `membrane=True`
(anisotropic barostat), `ff='charmm'`.

**GPU pinning:** on `CUDA` with `CUDA_VISIBLE_DEVICES` set (by Parsl/scheduler),
the in-process index is forced to `0` — leave `device_ids=[0]` and let the
scheduler mask GPUs.

**Implicit (GB):**
```python
from molecular_simulations.simulate import ImplicitSimulator
from openmm.app import GBn2
ImplicitSimulator(path='out/', implicit_solvent=GBn2, prod_steps=100_000_000,
                  solvent_dielectric=78.5).run()
```
Minimization only: `Minimizer(topology, coordinates, out='min.pdb', platform, device_ids).minimize()`.

**Simulation length guidance:** quick stability screen 1–10 ns; committed
evaluation 25–50 ns; gold-standard ranking 100+ ns (or multi-replica). MM-PBSA
estimates stabilize with more sampling.

---

## 4. RMSD stability gate

Before spending MM-PBSA compute, check the binder actually stayed bound. Align
each frame on the **target backbone**, then measure the **binder heavy-atom
RMSD** vs the start. Because alignment is on the target only, this RMSD captures
both conformational distortion *and* rigid-body drift away from the site.

```python
import numpy as np, MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD

u = mda.Universe('out/system.prmtop', 'out/prod.dcd')
solute = u.atoms[~np.isin(u.atoms.resnames, SOLVENT_RESNAMES)]   # strip water/ions
*target_frags, binder = solute.fragments      # binder is the LAST fragment
target = solute - binder
first, last = target.residues.resids[0], target.residues.resids[-1]
binder_sel = 'not type H and resid ' + ' '.join(map(str, binder.residues.resids))

R = RMSD(solute, select=f'backbone and resid {first} to {last}',
         groupselections=[binder_sel])
R.run()
binder_rmsd = R.rmsd[:, -1]                    # per-frame binder heavy-atom RMSD
mean_rmsd = binder_rmsd.mean()
```
If `mean_rmsd < rmsd_cutoff` (default 5.0 Å; tighten to 2–3 Å for small/rigid
binders, loosen to 5–7 Å for large/flexible ones) the complex is "stably bound"
→ proceed to MM-PBSA; otherwise flag it dissociated and skip. This is why the
**binder must be the last fragment** — hence the binder-last reordering in
glycan prep. Full helper: `scripts/rmsd_stability.py`.

---

## 5. MM-P(G)BSA binding free energy

`MMPBSA` slices the solvated complex into complex/receptor/ligand topologies,
runs GB **and** PB energies per frame (frame-parallel), and reports ΔG. The
`selections` are two cpptraj residue ranges (receptor, ligand) — build them from
the target/binder resids.

```python
from molecular_simulations.simulate import MMPBSA

mmpbsa = MMPBSA(
    top='out/system.prmtop',
    dcd='out/prod.dcd',
    selections=[target_sel, binder_sel],   # cpptraj ':1-120', ':121-165'
    n_cpus=8,                              # frame-level parallelism
    parallel_mode='frame',                 # 'frame' (fast) | 'serial'
    # amberhome=None -> $AMBERHOME
)
mmpbsa.run()
print(mmpbsa.free_energy)                  # [mean, std] ΔG_binding (PB), kcal/mol
```
Format resid lists for cpptraj with `MMPBSA.format_for_cpptraj(resids)` (`:1-120`
style, collapsing contiguous ranges). If `selections=None`, MM-PBSA infers
receptor vs ligand from `OXT` chain termini (last chain = ligand). Writes
`statistics.json` (per-term GB/PB means) and `deltaG.txt`; `free_energy` is the
PB **∆G Binding** `[mean, std]`. Note this is a solute-only MM-PBSA (bonded terms
cancel; it verifies that as a sanity check). See `references/mmpbsa_reference.md`.

---

## Best Practices

- **`$AMBERHOME` must be set** for builds and MM-PBSA.
- **Explicit solvent for glycoproteins**; implicit can't form glycosidic bonds.
- **Keep glycans in Chai PDB naming into `ExplicitSolvent(glycans=True)`** — let
  cpptraj rename to GLYCAM; don't pre-rename.
- **Binder must be the last chain/fragment** — the RMSD split and MM-PBSA
  ligand selection depend on it. Reorder before building glycoproteins.
- **RMSD-gate before MM-PBSA** — don't spend the compute on a dissociated binder.
- **`run()` is restartable** — re-invoke to resume; ideal for preempt queues.
- **Steps from time:** 1 ns = 250 000 steps at the default 4 fs production dt.
- **One `Simulator` = one replica/dir**; fan replicas out with Parsl (see the
  `parsl-hpc` skill in the molecular-simulations repo).
- **Device-agnostic:** `resolve_md_platform()` maps xpu→OpenCL, cuda→CUDA, else CPU.

## Common Pitfalls

1. `$AMBERHOME` unset → builder/MM-PBSA `ValueError`.
2. Glycoprotein in implicit solvent → disconnected sugars ("faulty Connection
   atoms"). Use explicit + `glycans=True`.
3. Pre-renaming sugars to GLYCAM before `ExplicitSolvent(glycans=True)` → cpptraj
   can't see them, bonds never form.
4. Binder not the last fragment → RMSD/MM-PBSA split grabs the wrong molecule.
5. Skipping the RMSD gate → MM-PBSA on a drifted binder returns meaningless ΔG.
6. Passing a CIF to the builder → convert to PDB first (`convert_cif_with_gemmi`).
7. Fighting GPU pinning under Parsl → leave `device_ids=[0]` when
   `CUDA_VISIBLE_DEVICES` is set.
8. Too-short production for MM-PBSA → unconverged, high-variance ΔG.

## Resources

### references/
- `mmpbsa_reference.md` — selections, GB/PB terms, outputs, ΔG interpretation.
- `glycan_reference.md` — GLYCAM naming, bond detection, the two prep paths.
- `analysis_reference.md` — RMSD, ipSAE, SASA, interaction-energy pointers.

### scripts/
- `_device.py` — `resolve_device()` / `resolve_md_platform()`.
- `build_explicit.py` — explicit OPC solvent build.
- `build_implicit.py` — implicit GB build.
- `build_complex_ligand.py` — protein-ligand (GAFF2) build.
- `prep_glycoprotein.py` — reorder binder last + build with `glycans=True`.
- `run_production.py` — minimize→equil→production (restartable).
- `rmsd_stability.py` — binder RMSD stability gate.
- `run_mmpbsa.py` — MM-P(G)BSA ΔG from a trajectory.
- `fold_to_dG_pipeline.py` — end-to-end: PDB → build → simulate → RMSD → MM-PBSA.
