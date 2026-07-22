# Method Selection Reference

Practical guidance for choosing functional / basis / dispersion / grid / solvent in PySCF for small-molecule parameterization (geometry optimization, torsion scans, ESP→RESP charges). These are pragmatic defaults, not the only valid choices.

## Exchange-correlation functionals (`mf.xc`)

| Functional | Character | Typical use |
| --- | --- | --- |
| `b3lyp` | Hybrid GGA, ubiquitous | General-purpose geometries/energies; pair with dispersion (`b3lyp` + `d3bj`). |
| `pbe0` | Hybrid GGA | Robust geometries, good cost/accuracy. |
| `wb97x-d3bj` | Range-separated hybrid w/ built-in D3(BJ) | Strong all-round choice for organics; **dispersion already included** — do not also set `mf.disp`. |
| `wb97x-v` / `wb97m-v` | Range-separated, VV10 nonlocal dispersion | High accuracy; pricier, needs NLC grid. |
| `pbe`, `blyp` | Pure GGA (no exact exchange) | Cheap; weaker for thermochemistry/torsions. |
| `m06-2x` | Meta-GGA hybrid | Good for main-group thermochemistry/noncovalent. |

**RESP convention note:** classic RESP charges (Amber/GAFF) were derived at **HF/6-31G\*** specifically because that level over-polarizes in a way that mimics condensed-phase charges. If you are reproducing canonical GAFF-style charges, `scf.RHF` + `6-31g*` is the historically matched level; modern RESP2 workflows instead use a DFT functional with gas+solvent ESP and interpolate. Match the level to the force field you are targeting.

## Basis sets (`basis`)

| Basis | Notes |
| --- | --- |
| `6-31g*` (a.k.a. `6-31g(d)`) | Classic RESP/GAFF charge derivation basis (with HF). Small, fast. |
| `def2-svp` | Small split-valence; quick optimizations / scans. |
| `def2-tzvp` | Triple-zeta; good default for final energies and ESP. |
| `def2-tzvpd` / `aug-cc-pvtz` | Add diffuse functions for anions, lone pairs, accurate ESP far from nuclei. |
| `cc-pvdz` / `cc-pvtz` | Dunning correlation-consistent series. |

For ESP/RESP, diffuse functions (`def2-tzvpd`, `aug-cc-pVTZ`) improve the potential in the valence region sampled by the MK grid. Balance against cost — torsion scans run dozens of optimizations.

## Dispersion (`mf.disp` or `dftd3`)

- `mf.disp = 'd3bj'` — DFT-D3 with Becke–Johnson damping; the common default.
- `mf.disp = 'd3zero'` — D3 with zero damping.
- `mf.disp = 'd4'` — newer DFT-D4 (charge-dependent).
- **Don't double-count:** `wb97x-d3bj`, `wb97x-v`, `b97-d3`, etc. already include dispersion. Only set `mf.disp` for functionals that lack it (e.g. plain `b3lyp`, `pbe0`).
- Legacy wrapper `from pyscf import dftd3; mf = dftd3.dftd3(mf)` exists; prefer the `mf.disp` attribute when available.

## Integration grid (`mf.grids.atom_grid`)

| Setting | Meaning |
| --- | --- |
| integer `3` | Coarse — drafts only. |
| integer `5` | ≈ Gaussian "ultrafine"; good default. |
| `(99, 590)` | (radial, angular Lebedev) ≈ ultrafine, explicit. |
| `(150, 974)` | Very fine; for smooth torsion profiles / tight energetics. |

Torsion scans are sensitive to grid noise: a relative-energy curve built from a too-coarse grid will wobble and corrupt the dihedral fit. Keep the grid **fine and identical** across all frames of a scan.

## Solvent dielectrics (`mf.with_solvent.eps`)

| Solvent | ε |
| --- | --- |
| Water | 78.36 |
| DMSO | 46.8 |
| Methanol | 32.6 |
| Acetonitrile | 35.7 |
| Octanol | 9.86 |
| Chloroform | 4.71 |
| Vacuum / gas | (no PCM wrap) |

`C-PCM` is a solid default for charge/ESP work. For **RESP2**, compute ESP in vacuum and in water, then interpolate the fitted charges `q = δ·q_solv + (1−δ)·q_gas` (δ commonly 0.5).

## A reasonable default recipe (organic small molecule)

- **Geometry / torsion scans:** `wb97x-d3bj` / `def2-svp`, grid `(99, 590)`, `symmetry=False`.
- **Final single point & ESP:** `wb97x-d3bj` / `def2-tzvp(d)`, grid `(99, 590)`.
- **RESP2:** run ESP gas-phase and `C-PCM` (ε=78.36), interpolate δ=0.5.

Tune up (bigger basis, finer grid, `wb97m-v`) when accuracy matters more than throughput; tune down for large scans.
