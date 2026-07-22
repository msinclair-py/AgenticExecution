# MM-P(G)BSA reference

`molecular_simulations.simulate.MMPBSA` is a reconstructed AMBER MM-P(G)BSA with
frame-level parallelism. It slices the solvated complex into complex / receptor /
ligand sub-topologies, computes GB **and** PB energies per frame, and reports
ΔG_binding.

## Constructor

```python
MMPBSA(
    top,                       # solvated system prmtop
    dcd,                       # trajectory (DCD/MDCRD)
    selections,                # [receptor_mask, ligand_mask] cpptraj strings, or None
    first_frame=0, last_frame=-1, stride=1,
    n_cpus=1,                  # frame-parallel workers
    out='mmpbsa',              # output prefix/dir
    solvent_probe=1.4, offset=0, gb_surften=0.0072, gb_surfoff=0.0,
    amberhome=None,            # None -> $AMBERHOME
    parallel_mode='frame',     # 'frame' (recommended) | 'serial'
    max_retries=3,
)
mmpbsa.run()
mmpbsa.free_energy             # [mean, std] of PB ΔG Binding, kcal/mol
```

## Selections

Two cpptraj residue masks: `[receptor, ligand]`. Build them from resid arrays
with `MMPBSA.format_for_cpptraj(resids)` → `:1-120` / `:1-10,25-43`.
- Receptor = the target chain(s) resids.
- Ligand = the binder resids (binder is the last solute fragment).

If `selections=None`, MM-PBSA infers them from `OXT` chain termini — everything
up to the last chain is receptor, the last chain is ligand. Explicit selections
are safer.

## What run() does

1. `FileHandler` strips water/ions, autoimages, RMS-fits, and writes
   complex/receptor/ligand `.prmtop` + `.crd` (+ a reference `.pdb`); splits the
   trajectory into `n_cpus` chunks for frame parallelism.
2. Writes GB (`igb=2`, `saltcon=0.10`, `extdiel=78.3`) and PB (`inp=2`, …) mdin
   files; runs `mmpbsa_py_energy` per chunk + `cpptraj molsurf` SASA per chunk.
3. `OutputAnalyzer` parses energy terms, checks bonded terms cancel
   (receptor+ligand ≈ complex), writes `statistics.json` (per-term GB & PB means
   ± std/err) and `deltaG.txt`, and sets `free_energy = [mean, std]` from the PB
   **∆G Binding** row.

## Energy terms

- **GB:** BOND, ANGLE, DIHED, VDWAALS, EEL, EGB, 1-4 VDW, 1-4 EEL, ESURF.
- **PB:** …, EPB, ECAVITY, EDISPER.
- Grouped as `G gas` (bonded + vdW + electrostatics) and `G solv` (EGB/ESURF or
  EPB/ECAVITY). ΔG = ΔG_gas + ΔG_solv over (complex − receptor − ligand).

## Interpretation

- `free_energy[0]` is the PB ΔG_binding **mean** (kcal/mol); `[1]` is its std.
- **More negative = tighter predicted binding.** MM-PBSA is a *relative* ranker,
  not an absolute affinity; compare candidates run identically.
- Variance shrinks with more frames / longer production. A high std relative to
  the mean means undersampled — extend the run before trusting the number.
- A bonded-term mismatch raises `ValueError` (receptor+ligand should equal
  complex for bonded energies) — usually a broken selection or topology.
