# PySCF API Reference (workflow-focused)

Scoped to the modules this project uses. Verified against **pyscf 2.13.1**.

## `pyscf.gto` — molecule / basis / integrals

| Symbol | Purpose |
| --- | --- |
| `gto.M(atom, basis, charge, spin, symmetry, max_memory, verbose, unit)` | Build a `Mole`. `spin = 2S = multiplicity − 1`. `unit` defaults to `'Angstrom'`. |
| `gto.Mole()` | Lower-level builder; `mol.build()` after setting attributes. `gto.M` is the convenience wrapper. |
| `mol.natm` | Number of atoms. |
| `mol.nao` / `mol.nao_nr()` | Number of atomic orbitals (basis functions). |
| `mol.atom_charges()` | Nuclear charges `Z_A`, ndarray `(natm,)`. |
| `mol.atom_coords(unit='Bohr')` | Atom coordinates. **Default is Bohr**; pass `unit='Angstrom'` for Å. |
| `mol.atom_symbol(i)` / `mol.elements` | Element labels. |
| `gto.fakemol_for_charges(coords)` | Build a "molecule" of point charges at `coords` (in Bohr) — used to evaluate integrals at arbitrary grid points (ESP). |

## `pyscf.dft` — Kohn–Sham DFT

| Symbol | Purpose |
| --- | --- |
| `dft.RKS(mol)` | Restricted KS mean-field (closed shell). |
| `dft.UKS(mol)` | Unrestricted KS (open shell / radicals). |
| `dft.ROKS(mol)` | Restricted-open KS. |
| `mf.xc` | Exchange-correlation functional name (e.g. `'b3lyp'`, `'wb97x-d3bj'`). |
| `mf.disp` | Empirical dispersion model (`'d3bj'`, `'d3zero'`, `'d4'`). |
| `mf.grids.atom_grid` | Integration grid: `(nrad, nang)` tuple or integer preset `0..9`. |
| `mf.kernel()` | Run SCF; returns total energy (Hartree). Sets `mf.e_tot`, `mf.mo_*`. |
| `mf.converged` | Bool — **always check** after `kernel()`. |
| `mf.newton()` | Return a second-order (Newton) solver variant; re-run `.kernel()` to rescue hard SCF. |
| `mf.make_rdm1()` | 1-particle density matrix `(nao, nao)` (or `(2, nao, nao)` for UKS). |
| `mf.mo_energy`, `mf.mo_coeff`, `mf.mo_occ` | Orbital energies / coefficients / occupations. |
| `mf.nuc_grad_method()` | Gradient (force) method object; `.kernel()` for the nuclear gradient. |

`pyscf.scf.RHF/UHF/ROHF` are the Hartree–Fock analogues with the same `kernel()`/`converged`/`make_rdm1()` interface.

## `pyscf.df` — density fitting / 3-center integrals

| Symbol | Purpose |
| --- | --- |
| `df.incore.aux_e2(mol, fakemol, intor='int3c2e')` | 3-center 2-electron integrals between `mol`'s AOs and `fakemol`'s points. Shape `(nao, nao, npts)`. Core of ESP evaluation. |
| `intor='int3c2e'` | The Coulomb 3-center integral `⟨μ| 1/|r−R| |ν⟩`. |

ESP electronic term at point k: `-np.einsum('ij,ij->', dm, ints[:, :, k])` (note the minus sign).

## `pyscf.geomopt.geometric_solver` — geometry optimization

Requires the optional **`geometric`** package (`pip install geometric`).

| Symbol | Purpose |
| --- | --- |
| `optimize(mf, maxsteps=N, constraints=...)` | Optimize geometry; returns a **new optimized `Mole`** (not an energy). |
| `constraints` | geomeTRIC constraint dict. Atom indices are **1-indexed**. |

Constraint dict shape:
```python
{'set':    [{'type': 'dihedral', 'indices': [i, j, k, l], 'value': angle}]}   # hold at value
{'freeze': [{'type': 'distance', 'indices': [i, j]}]}                          # freeze at current
```
Constraint `type` ∈ `distance`, `angle`, `dihedral`, `xyz`. Top-level keys: `set` (target a value) and `freeze` (hold current).

## `pyscf.solvent` — implicit solvation

| Symbol | Purpose |
| --- | --- |
| `mf.PCM()` | Wrap a mean-field with a polarizable continuum model; returns the solvated `mf`. |
| `mf.with_solvent.method` | `'C-PCM'`, `'IEF-PCM'`, `'COSMO'`, `'SS(V)PE'`. |
| `mf.with_solvent.eps` | Solvent dielectric constant (water ≈ 78.36). |
| `mf.DDCOSMO()` / `mf.ddCOSMO()` | Domain-decomposition COSMO alternative. |

Wrap **before** `kernel()`. The resulting `make_rdm1()` density is solvent-polarized.

## `pyscf.lib` — runtime

| Symbol | Purpose |
| --- | --- |
| `lib.num_threads(n)` | Set OpenMP thread count for integrals / BLAS. |
| `lib.logger` | Verbosity levels mirror `mol.verbose` (0 silent … 9 debug). |

`max_memory` (MB) is set on the `Mole` (`gto.M(..., max_memory=...)`) and inherited by mean-field objects.

## `gpu4pyscf` — GPU drop-in

Mirror of `pyscf.dft` / `pyscf.scf`:
```python
from gpu4pyscf import dft        # dft.RKS(mol), same .xc/.disp/.grids/.kernel() API
```
- Build the `Mole` with CPU `pyscf.gto` as usual.
- Outputs may be CuPy arrays; call `.get()` / `cupy.asnumpy(...)` before NumPy-only code.
- Coverage is a subset of CPU PySCF — verify a feature exists before assuming parity.

## Units cheat-sheet

| Quantity | Unit |
| --- | --- |
| `gto.M(atom=...)` input | Angstrom (default; `unit='Bohr'` to override) |
| `mol.atom_coords()` | **Bohr** (default) |
| Energies (`mf.kernel()`, `e_tot`) | Hartree |
| ESP values `V(r)` | Hartree / e (a.u.) |
| Bohr ↔ Angstrom | `1 Bohr = 0.5291772109 Å`; `BOHR_PER_ANGSTROM = 1.8897259886` |
| Hartree → kcal/mol | `× 627.5095` |
