# Glycoprotein prep reference

Chai writes glycans as PDB sugar residues (NAG, MAN, BMA, GAL, FUC, SIA, GLC,
XYS) in ordinary chains. AMBER needs GLYCAM 3-letter names + explicit
glycosidic/disulfide bond commands. Two prep paths.

## Path A — ExplicitSolvent(glycans=True) (recommended)

Keep the fold in **Chai PDB sugar naming** and pass it to
`ExplicitSolvent(path, pdb, glycans=True)`. Its `cpptraj prepareforleap` step
does the GLYCAM renaming and detects the glycosidic/disulfide bonds itself.

- **Do NOT pre-rename to GLYCAM** — that hides the sugars from cpptraj and the
  bonds never form ("faulty Connection atoms" in tleap).
- **Move the binder chain last first** (so the MM-PBSA fragment split
  target+glycan | binder is clean). See `scripts/prep_glycoprotein.py`.
- **Explicit solvent only.** `ImplicitSolvent` uses a plain `loadpdb` with no
  cpptraj sugar prep, so GLYCAM residues load without glycosidic bonds. Route
  glycan binders to explicit solvent (or fail loudly).

## Path B — GlycanConverter (manual / inspectable)

`bindify` material `materials/glycam_mapper.py` `GlycanConverter().convert_pdb()`
does the conversion explicitly and emits a tleap script you can read:

```python
from molecular_simulations.build.glycam_mapper import GlycanConverter  # or bindify material path
result = GlycanConverter(ff='ff14SB', glycam_ff='GLYCAM_06j-1').convert_pdb(
    'fold.pdb', output_dir='prep/', mol_name='PROT',
)
result['glycam_pdb']      # renamed PDB
result['leap_script']     # tleap .in with source/bond/solvate commands
result['bond_commands']   # list of tleap 'bond ...' strings (disulfide + glycan)
result['disulfide_bonds'] # detected SG-SG pairs
result['nln_residues']    # ASN -> NLN (glycosylated)
result['cyx_residues']    # CYS -> CYX (disulfide)
```

- `mol_name` must match the tleap variable that loads the PDB: **`'PROT'` for
  ExplicitSolvent, `'prot'` for ImplicitSolvent.**
- Detection is geometric: glycosidic bond cutoff 1.85 Å (C1–O), protein–sugar
  2.0 Å (C1–ND2), disulfide 2.1 Å (SG–SG). Anomers are assigned from glycan-tree
  topology (more robust than geometry on predicted structures).
- N-acetyl atom renames (C7→C2N, C8→CME, O7→O2N) and ASN→NLN are applied.

## GLYCAM naming quick key

3-char code = `<linkage><sugar-letter><anomer>`, e.g. `4YB` = 4-linked GlcNAc β.
- Sugar letters: NAG/NDG/A2G→Y (GlcNAc), MAN/BMA→M, GAL→L, GLC→G, FUC→F
  (L-config → lowercase), SIA→S.
- Linkage char encodes which hydroxyls are substituted (0 terminal, 2/3/4/6
  single, Z/D/U/E/V/H/S/T/Q/W multi).
- Anomer: A (α) / B (β).
