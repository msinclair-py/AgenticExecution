---
name: md-simulation
description: >-
  Gold-standard stability + binding-free-energy assessment of a folded
  protein-binder complex via molecular dynamics: build a solvated system with
  AmberTools, propagate with OpenMM on GPU (heat -> equilibrate -> production),
  measure backbone RMSD of the binder relative to the aligned target with
  MDAnalysis, and — only if the complex stays stable — run MMPBSA/MM-GBSA
  end-point free energy over the trajectory. Use as the expensive final rank on
  top candidates after cheap folding + contact filters. Hours-scale, needs a
  pre-provisioned GPU env with OpenMM + AmberTools ($AMBERHOME) + MDAnalysis.
---

# Molecular dynamics stability + binding free energy (OpenMM + AmberTools + MDAnalysis)

Assess a folded complex the expensive, high-fidelity way: build a solvated
Amber system from the complex PDB, run an OpenMM MD trajectory (heat, multi-cycle
equilibration, microsecond-scale production), then judge stability by the binder's
backbone RMSD with the target aligned. If the complex holds together (mean
backbone RMSD below a cutoff) you run MMPBSA/MM-GBSA over the trajectory to obtain
a binding free energy in kcal/mol; if it drifts apart, the binding energy is
reported as zero. This is the gold-standard rank — hours on a GPU per complex —
so reserve it for the top handful of candidates that already survived cheap
folding and contact filters. **Fold the complex first** (this skill consumes the
complex PDB); downstream, filter-binders buckets on the stability/energy it emits.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) drive OpenMM, AmberTools, and
MDAnalysis directly. The procedure below is everything you need.

## When to use it

- You have a small set of top folded complexes and want a high-fidelity
  stability + binding-energy verdict before committing wet-lab or compute budget.
- You want the expensive, trajectory-based tiebreaker after folding and
  contact-based filters have already narrowed the pool.

Do **not** use it for triage over many candidates — it is hours-scale per
complex. Cheap filters (folding confidence, contact-eval, peptide-descriptors)
belong earlier. Do not use it if you have no folded complex PDB yet.

## Setup — build the environment (once per project)

This tool cannot be pip-bootstrapped inline; build a conda env once, on a login
node, into shared project storage — see `../ENVIRONMENTS.md` for the
cross-cutting HPC rules.

```bash
conda create -n <proj>-md -c conda-forge python=3.11 openmm ambertools mdanalysis pdbfixer -y
conda activate <proj>-md
export AMBERHOME=$CONDA_PREFIX        # MMPBSA / MM-GBSA needs AmberTools
```

- **OpenMM** — pick the platform for your accelerator: `CUDA` for NVIDIA GPUs,
  `OpenCL` for Intel XPUs.
- **AmberTools** — provides `tleap`, `sander`/`pmemd`, and `MMPBSA.py` for the
  end-point free energy; needs `$AMBERHOME` exported. Its force fields ship with
  the conda package, so **no separate weights download** — this runs fine on an
  offline compute node once built.
- **pdbfixer** — prep raw PDBs before `tleap` (add missing atoms / hydrogens).
- **MDAnalysis** — the RMSD analysis.

Gate on all of it before doing anything; if the check fails, stop and rebuild the
env rather than attempting to install AmberTools or a CUDA OpenMM on the fly.

```bash
python -c "import openmm, MDAnalysis, os; assert os.environ.get('AMBERHOME'), 'AMBERHOME unset'" \
  2>/dev/null && echo ready || echo "NOT READY: need pre-provisioned OpenMM + AmberTools (\$AMBERHOME) + MDAnalysis env"
```

## Procedure

Reproduce the pipeline directly on OpenMM + Amber. Four stages.

### 1. Build the solvated system (AmberTools `tleap`)

From the complex PDB, build an Amber topology + coordinates. Choose the solvent
model by the speed/fidelity you need:

- **ExplicitSolvent** — solvate in a TIP3P water box (+ neutralizing ions),
  highest fidelity, slowest. Produces `system.prmtop` + `system.inpcrd`.
- **ImplicitSolvent** — GB continuum (no explicit waters), much faster, lower
  fidelity. Also produces a `system.prmtop`.

Drive `tleap` with a leap script that loads the protein force field
(e.g. `ff14SB`), reads the PDB, (for explicit) solvates + adds ions, and
`saveAmberParm` to write `system.prmtop`. Keep the prmtop — it is a durable
artifact and the input to MMPBSA later.

### 2. Simulate with OpenMM: build -> heat -> equilibrate -> production

Load the prmtop/inpcrd into OpenMM, pick the platform, then run the staged
schedule. Key default parameters to document and carry:

- **heat_steps = 100_000** (~200 ps at a 2 fs timestep) — warm from low T to 300 K.
- **equil_steps = 1_250_000** (~2.5 ns) per cycle, with **n_equil_cycles = 3**
  (restrain -> relax over cycles).
- **prod_steps = 250_000_000** (~1 µs) at **prod_dt = 0.004 ps** (4 fs, requires
  hydrogen-mass repartitioning) — the production run, saved to `prod.dcd`.

```python
import openmm as mm
import openmm.app as app
from openmm import unit

PLATFORM = "CUDA"   # "CUDA" for NVIDIA, "OpenCL" for Intel XPU
prmtop = app.AmberPrmtopFile("system.prmtop")
inpcrd = app.AmberInpcrdFile("system.inpcrd")

system = prmtop.createSystem(
    nonbondedMethod=app.PME,           # use app.GBn2/implicit for ImplicitSolvent
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=app.HBonds,
    hydrogenMass=1.5*unit.amu,         # HMR enables the 4 fs production step
)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond,
                                         0.004*unit.picoseconds)
sim = app.Simulation(prmtop.topology, system, integrator,
                     mm.Platform.getPlatformByName(PLATFORM))
sim.context.setPositions(inpcrd.positions)
sim.minimizeEnergy()

# heat: ~200 ps @ 2 fs (swap to a 0.002 ps integrator for the heat/equil phases)
sim.step(100_000)
# equilibrate: 3 cycles of ~2.5 ns each, relaxing restraints between cycles
for _ in range(3):
    sim.step(1_250_000)
# production: ~1 us @ 4 fs, write the trajectory
sim.reporters.append(app.DCDReporter("prod.dcd", 25_000))
sim.step(250_000_000)
```

### 3. Stability via backbone RMSD (MDAnalysis)

Load `system.prmtop` + `prod.dcd`, strip solvent, split target vs binder by
fragments (**the binder is the last fragment**), and compute the binder's
backbone RMSD **with the target aligned**:

```python
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD

u = mda.Universe("system.prmtop", "prod.dcd")
solute = u.select_atoms("not resname WAT HOH Na+ Cl- K+")   # strip solvent by resname
fragments = solute.fragments
target = fragments[0]
binder = fragments[-1]                                       # binder = last fragment

R = RMSD(u, select="backbone",                              # align on the target backbone
         groupselections=[f"backbone and group binder"],
         binder=binder)                                      # report binder RMSD in the aligned frame
R.run()
binder_rmsd = R.results.rmsd[:, -1]                          # per-frame binder backbone RMSD (Å)
rmsd_mean, rmsd_std = float(binder_rmsd.mean()), float(binder_rmsd.std())
```

### 4. Conditional binding free energy (MMPBSA / MM-GBSA)

- If **rmsd_mean < rmsd_cutoff** (default **5.0 Å**) the complex is stable — run
  MMPBSA/MM-GBSA over `prod.dcd` (via AmberTools `MMPBSA.py` with the complex,
  receptor, and ligand prmtops) to get the binding free energy and its error, in
  kcal/mol.
- Otherwise the complex is judged unstable and `binding_free_energy = [0, 0]`.

```python
if rmsd_mean < 5.0:
    # AmberTools MMPBSA.py over prod.dcd -> parse FINAL DELTA G + stderr
    binding_free_energy = [dG_value, dG_error]   # kcal/mol
else:
    binding_free_energy = [0.0, 0.0]
```

## Interpreting the result

- **rmsd = [mean, std]** — the stability score. Low mean backbone RMSD (below the
  cutoff) means the binder stayed docked; a large mean means it drifted / unbound.
- **binding_free_energy = [value, error]** kcal/mol — more negative is stronger
  predicted binding; only meaningful when the complex was stable. `[0, 0]` is the
  sentinel for "too unstable to score", not "zero affinity".
- **Artifacts** — keep `system.prmtop` (needed to re-analyze or re-run MMPBSA) and
  `prod.dcd` (the trajectory).
- Rank candidates first by stability (RMSD), then by binding free energy among the
  stable ones.

## Notes

- Inputs you supply: the complex PDB, a workdir, the OpenMM platform, `$AMBERHOME`,
  the solvent model (explicit vs implicit — the main speed/fidelity dial), and the
  `rmsd_cutoff`.
- **Deciding which complexes earn an MD run is your job as the agent.** This is the
  expensive gold-standard step: run it only on the top candidates that already
  passed folding + contact filters, and hand the resulting stability/energy to a
  filter-binders bucketing step. This skill only computes the MD verdict for one
  complex.
