---
name: pyscf
description: Quantum chemistry with PySCF (and the gpu4pyscf GPU drop-in). Emphasis on the workflow this project relies on — building molecules (gto.M), running DFT/SCF (dft.RKS with xc/dispersion/grids), geometry optimization & constrained torsion scans via geomeTRIC, electrostatic potential (ESP) evaluation on a grid for RESP charge fitting, and implicit solvent (PCM/C-PCM). Covers unit conventions (Bohr vs Angstrom), spin/charge setup, threading & memory, and CPU↔GPU switching. Use for any PySCF/gpu4pyscf task; for force-field integration of the resulting charges see AmberTools, for 3D structure generation see rdkit.
license: Apache-2.0 (PySCF is licensed Apache-2.0; this skill text is project-local)
metadata:
    skill-author: local project skill (helicon/agentic)
    verified-against: pyscf 2.13.1
---

# PySCF Quantum Chemistry Toolkit

## Overview

PySCF is a Python-native quantum chemistry package. This skill emphasizes the pipeline used in this project for small-molecule / PTM force-field parameterization:

1. **Molecule construction** — `gto.M`, charge/spin/symmetry, units, memory & threads.
2. **DFT / SCF** — `dft.RKS`, functional / dispersion / integration grid, running the kernel, density matrices.
3. **Geometry optimization & torsion scans** — `pyscf.geomopt.geometric_solver.optimize` with geomeTRIC constraint dictionaries.
4. **ESP on a grid** — nuclear + electronic electrostatic potential for RESP charge fitting, via 3-center integrals and `fakemol_for_charges`.
5. **Implicit solvent** — PCM / C-PCM for solvated calculations.
6. **CPU ↔ GPU** — the `gpu4pyscf` drop-in and how it differs.

The single most important rule, threaded throughout: **PySCF works in atomic units (Bohr) internally.** Coordinates you pass into `gto.M` are read as Angstrom by default, but everything that comes *back out* of a built `Mole` — `mol.atom_coords()`, integral geometries, ESP grids you build yourself — must be reconciled in Bohr. Mixing the two is the most common source of silently wrong numbers.

---

## 1. Building a Molecule (`gto.M`)

```python
from pyscf import gto

mol = gto.M(
    atom='''O  0.000  0.000  0.000
            H  0.000  0.000  0.957
            H  0.927  0.000 -0.240''',   # element + xyz, one atom per line
    basis='def2-tzvp',                    # basis set name (or a dict per element)
    charge=0,                             # net molecular charge
    spin=0,                               # spin = 2S = (multiplicity - 1), NOT multiplicity
    symmetry=False,                       # point-group symmetry detection
    max_memory=4000,                      # MB; raise it for big systems / many grid points
    verbose=4,                            # 0 silent ... 4 normal ... 9 debug
    unit='Angstrom',                      # default; use 'Bohr' if coords are already a.u.
)
```

The `atom` field accepts a multiline string (as above) or a list of `[symbol, (x, y, z)]`. A geometry string can be built from element/coordinate arrays:

```python
geom_str = '\n'.join(
    f'{e}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}'
    for e, c in zip(elements, coords, strict=True)
)
mol = gto.M(atom=geom_str, basis='def2-svp', charge=charge, spin=multiplicity - 1)
```

**Critical gotchas**
- **`spin` is `2S` (number of unpaired electrons), i.e. `multiplicity - 1`.** A singlet is `spin=0`, a doublet radical is `spin=1`. Passing the multiplicity directly is a frequent bug.
- `charge` + `spin` must be electron-count consistent; an impossible combination raises at build time.
- `symmetry=True` can speed SCF but **breaks constrained geometry optimization** (the constraint can lower the symmetry mid-optimization). Use `symmetry=False` for geomeTRIC optimizations/scans.
- `mol.atom_coords()` returns **Bohr** by default; pass `unit='Angstrom'` to get Å.

```python
mol.natm                      # number of atoms
mol.nao                       # number of atomic orbitals (basis functions)
mol.atom_charges()            # nuclear charges Z_A, ndarray (natm,)
mol.atom_coords()             # Bohr  (default)
mol.atom_coords(unit='Angstrom')
```

### Threads and memory

```python
from pyscf import lib
lib.num_threads(8)            # OpenMP threads for integrals / linear algebra
```

`max_memory` (MB, set on `gto.M`) caps in-core memory; integral routines fall back to slower on-disk/batched paths above it. Set both `lib.num_threads(...)` and `max_memory` deliberately on worker nodes.

---

## 2. DFT / SCF

```python
from pyscf import dft

mf = dft.RKS(mol)             # restricted Kohn-Sham; use UKS for open-shell, ROKS for restricted-open
mf.xc = 'wb97x-d3bj'          # exchange-correlation functional
mf.disp = 'd3bj'              # empirical dispersion correction (see note below)
mf.grids.atom_grid = (99, 590)  # (radial, angular) Lebedev grid; or an int preset level
e_tot = mf.kernel()           # runs SCF, returns total energy in Hartree

print(mf.converged)           # ALWAYS check this
dm = mf.make_rdm1()           # 1-particle density matrix (nao, nao) — needed for ESP, properties
```

**Always check `mf.converged`.** A non-converged SCF still returns a number; downstream ESP/optimization will be garbage. Wrap critical runs:

```python
e_tot = mf.kernel()
if not mf.converged:
    mf = mf.newton()          # second-order (Newton) solver often rescues hard cases
    e_tot = mf.kernel()
```

### Functional, dispersion, and grids

- `mf.xc` accepts standard names (`'b3lyp'`, `'pbe0'`, `'wb97x-v'`, `'wb97x-d3bj'`, …). See `references/methods_reference.md`.
- `mf.disp` selects an empirical dispersion model (`'d3bj'`, `'d3zero'`, `'d4'`). Some functionals (e.g. `wb97x-d3bj`) already imply a correction — don't double-count. If a functional has no built-in dispersion and you need it, set `mf.disp`. An alternative older pattern wraps the mean-field object:
  ```python
  from pyscf import dftd3
  mf = dftd3.dftd3(mf)        # legacy add-on; prefer mf.disp = 'd3bj' when available
  ```
- `mf.grids.atom_grid` controls DFT integration accuracy. A tuple `(nrad, nang)` like `(99, 590)` is "ultrafine"-class; integer presets `0..9` also work (`5` ≈ Gaussian ultrafine). Too-coarse grids cause noisy torsion-scan energies.

### Properties off a converged `mf`

```python
mf.e_tot                      # total energy (Hartree)
mf.mo_energy                  # orbital energies
mf.mo_coeff                   # MO coefficients
dm = mf.make_rdm1()           # density matrix
```

---

## 3. Geometry Optimization and Torsion Scans

PySCF delegates optimization to **geomeTRIC** (an optional dependency — `pip install geometric`). The entry point:

```python
from pyscf.geomopt.geometric_solver import optimize

mol_eq = optimize(mf, maxsteps=200)          # returns a NEW, optimized Mole
opt_coords = mol_eq.atom_coords(unit='Angstrom')
```

`optimize` returns an optimized **`Mole`**, not an energy. To get the final energy, build a fresh mean-field on the optimized geometry and run it:

```python
mf_final = dft.RKS(mol_eq)
mf_final.xc = 'wb97x-d3bj'
mf_final.disp = 'd3bj'
mf_final.grids.atom_grid = (99, 590)
e_final = mf_final.kernel()
```

### Constrained optimization (the basis of torsion scans)

Constraints are passed as a **geomeTRIC constraint dictionary**. Atom indices in this dict are **1-indexed** (geomeTRIC convention), unlike PySCF's 0-indexed atoms:

```python
i, j, k, l = (idx + 1 for idx in torsion)    # convert 0-indexed -> 1-indexed
constraints = {
    'set': [
        {'type': 'dihedral', 'indices': [i, j, k, l], 'value': angle_degrees}
    ]
}
mol_opt = optimize(mf, maxsteps=150, constraints=constraints)
```

A relaxed torsion scan = loop over target angles, re-optimizing with the dihedral frozen at each, recording the final single-point energy per frame. Constraint `type` may be `'distance'`, `'angle'`, `'dihedral'`, `'xyz'`, etc.; `'freeze'` (vs `'set'`) holds a coordinate at its current value.

**Scan tips**
- Use `symmetry=False` (see §1) — a frozen dihedral can break symmetry.
- Re-feed the previous frame's optimized geometry as the next frame's start for smooth, continuous scans.
- Keep the grid fine and the functional consistent across frames; relative energies are what matter (convert Hartree → kcal/mol with `* 627.5095`).

---

## 4. Electrostatic Potential (ESP) on a Grid

The ESP at a point **r** is the sum of a nuclear term and an electronic term:

```
V(r) = Σ_A Z_A / |r - R_A|  −  ∫ ρ(r') / |r - r'| dr'
```

The electronic term is evaluated from the density matrix and 3-center Coulomb integrals between the AO basis and **point charges placed at the grid locations** (via `fakemol_for_charges`). **All coordinates here must be in Bohr.**

```python
import numpy as np
from pyscf import df, gto

dm = mf.make_rdm1()

# grid_pts is YOUR array of evaluation points in Angstrom -> convert to Bohr
BOHR_PER_ANGSTROM = 1.8897259886
grid_bohr = grid_pts * BOHR_PER_ANGSTROM
coords_bohr = mol.atom_coords()              # already Bohr

# --- nuclear contribution ---
Z = mol.atom_charges()
esp_nuc = np.zeros(len(grid_bohr))
for A in range(mol.natm):
    r = np.linalg.norm(grid_bohr - coords_bohr[A], axis=1)
    esp_nuc += Z[A] / r

# --- electronic contribution (batched over grid points) ---
esp_elec = np.zeros(len(grid_bohr))
batch = 500
for start in range(0, len(grid_bohr), batch):
    pts = grid_bohr[start:start + batch]
    fakemol = gto.fakemol_for_charges(pts)              # point "charges" at grid pts
    ints = df.incore.aux_e2(mol, fakemol, intor='int3c2e')  # shape (nao, nao, npts_batch)
    for k in range(ints.shape[2]):
        esp_elec[start + k] = -np.einsum('ij,ij->', dm, ints[:, :, k])

esp_total = esp_nuc + esp_elec               # Hartree / e (atomic units)
```

**Why it's done this way:** `int3c2e` gives `⟨μ| 1/|r−R| |ν⟩` for each grid point `R`; contracting with the density matrix yields the electron contribution. Batching keeps the `(nao, nao, npts)` integral tensor in memory. The electronic term carries a **minus sign** (electrons are negative); forgetting it flips the potential.

`references/api_reference.md` documents the integral routines; `scripts/compute_esp.py` is a ready-to-use, batched implementation.

### Generating the MK (Merz–Kollman) grid

RESP fitting samples the ESP on nested Connolly shells at 1.4/1.6/1.8/2.0× the vdW radius, excluding points inside neighboring atoms' radii. This project generates that grid itself (see `QMAgent.generate_mk_grid`) and passes it to the ESP routine. The grid is built and consumed in **Angstrom**, then converted to Bohr inside the ESP step — keep that boundary clear.

---

## 5. Implicit Solvent (PCM / C-PCM)

Wrap the mean-field object **before** calling `kernel()`:

```python
mf = dft.RKS(mol)
mf.xc = 'wb97x-d3bj'

mf = mf.PCM()                       # attach a polarizable continuum model
mf.with_solvent.method = 'C-PCM'    # or 'IEF-PCM', 'COSMO', 'SS(V)PE'
mf.with_solvent.eps = 78.3553       # solvent dielectric (78.36 = water)

e_solv = mf.kernel()
dm_solv = mf.make_rdm1()            # density is solvent-polarized -> use for solvated ESP
```

For **RESP2** you run the ESP twice — gas phase and solvated — and interpolate the fitted charges `q = δ·q_solv + (1−δ)·q_gas` (commonly δ=0.5). The only QM-side difference is the `mf.PCM()` wrap; the ESP evaluation in §4 is identical, just using the solvated density matrix.

Common dielectrics: water 78.36, DMSO 46.8, methanol 32.6, octanol 9.86, chloroform 4.71.

---

## 6. CPU ↔ GPU (`gpu4pyscf`)

`gpu4pyscf` is a near drop-in replacement that runs SCF/DFT and gradients on NVIDIA GPUs. Swap the import:

```python
if gpu:
    from gpu4pyscf import dft     # GPU
else:
    from pyscf import dft         # CPU

mf = dft.RKS(mol)
mf.xc = 'wb97x-d3bj'
mf.disp = 'd3bj'
mf.grids.atom_grid = (99, 590)
e = mf.kernel()                   # runs on GPU when imported from gpu4pyscf
```

**Notes**
- The `Mole` (`gto.M`) is always built with CPU PySCF; only the mean-field/post-SCF objects come from `gpu4pyscf`.
- Most results live on GPU; some return NumPy, some return CuPy arrays. Call `.get()` on a CuPy array (or `cupy.asnumpy(...)`) before handing data to NumPy-only code (e.g. the ESP/RESP routines).
- Not every CPU feature is mirrored. ESP via `df.incore.aux_e2` and RESP fitting in this project run on the **CPU executor** even when SCF runs on GPU — keep that split in mind.
- `geomeTRIC` optimization works with a `gpu4pyscf` `mf`; the optimizer is CPU-side, energies/gradients are GPU-side.

---

## Best Practices

- **Check `mf.converged`** after every `kernel()`; retry with `mf.newton()` on failure.
- **`spin = multiplicity − 1`**, always.
- **Track units.** Coordinates in to `gto.M` are Angstrom; everything out of `mol` is Bohr; ESP math is Bohr. Convert at one clearly marked boundary.
- **`symmetry=False` for optimizations/scans**; `symmetry=True` only for fixed-geometry single points where it helps.
- **`optimize()` returns a `Mole`, not an energy** — rebuild a mean-field for the final energy.
- **Don't double-count dispersion** (functional name vs `mf.disp`).
- **Keep the grid fine and consistent** across a scan; relative energies are meaningless if the grid changes.
- Set `lib.num_threads(...)` and `max_memory` per worker; large ESP grids and big bases are memory-hungry.

## Common Pitfalls

1. Passing multiplicity where `spin` (= mult−1) is expected.
2. Forgetting the **minus sign** on the electronic ESP term, or mixing Bohr/Angstrom in the grid.
3. Using the energy "returned" by `optimize()` — it returns a `Mole`; rebuild `mf` for the energy.
4. Leaving `symmetry=True` during a constrained scan (constraint breaks symmetry → crashes/garbage).
5. Not checking `mf.converged`.
6. 0-indexed vs 1-indexed atoms: PySCF atoms are 0-indexed, geomeTRIC constraint dicts are **1-indexed**.
7. Double-counting dispersion via both the functional and `mf.disp`/`dftd3`.
8. Handing a CuPy array from `gpu4pyscf` to NumPy-only code without `.get()`.
9. Too-coarse `grids.atom_grid` → noisy torsion energies that ruin dihedral fits.

## Resources

### references/
- `api_reference.md` — PySCF modules and the key functions/objects used here (`gto`, `dft`, `df`, `geomopt`, `solvent`, `lib`).
- `methods_reference.md` — choosing functionals, basis sets, dispersion, grids, and solvent dielectrics.

### scripts/
- `compute_esp.py` — batched ESP-on-a-grid evaluation from a converged mean-field (§4), CPU/GPU aware.
- `run_dft.py` — build a molecule, run a single-point or geometry optimization, print energy (§2–§3).

Run scripts directly or use them as templates. Load the references when you need specific API details or method-selection guidance.
