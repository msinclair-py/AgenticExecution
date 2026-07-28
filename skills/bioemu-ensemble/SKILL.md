---
name: bioemu-ensemble
description: Generate a protein conformational ensemble (equilibrium-like) with Microsoft BioEmu — fast ML alternative to MD for sampling flexible/disordered regions. Use when you need many conformations of a protein (e.g. to characterize an IDR, cluster loop states, or design against an ensemble rather than one static structure). Covers install, the CUDA/torch pitfall, MSA, proper structure reconstruction, and per-region flexibility analysis.
---

# BioEmu conformational ensembles

BioEmu emulates a protein's equilibrium conformational distribution from sequence in minutes on
one GPU — a cheap stand-in for long MD when you need an *ensemble* (disordered tails, flexible
loops, alternative states) rather than a single fold. Output is a set of backbone conformations
(CA + backbone-frame orientations) you can convert to PDB/XTC and analyze.

## When to use
- Characterizing intrinsic disorder / flexibility of a region (RMSF, secondary-structure propensity).
- Producing conformations to design binders *against an ensemble* (see `peptide-clamp-design`).
- Any time a single AlphaFold model would mislead because the region is an ensemble, not one state.

## Install (do once, per cluster)
```bash
python -m venv bioemu_env && source bioemu_env/bin/activate
pip install --upgrade pip
pip install bioemu           # pulls tensorflow-cpu, torch, mdtraj, etc. (~few min)
```
**CRITICAL torch/CUDA fix.** BioEmu pins the newest torch, which may be a CUDA build *newer than the
cluster driver* (e.g. torch `2.13+cu130` while the node supports only CUDA 12.8) → `torch.cuda.is_available()`
is False and it silently runs on CPU (unusably slow). Check and reinstall a matching build:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # e.g. 2.13.0+cu130
# driver's max CUDA: torch's warning "found version 12080" == CUDA 12.8. Install <= that:
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
python -c "import bioemu; print('ok')"   # confirm bioemu still imports
```
On ALCF (Polaris/Aurora) login nodes `nvidia-smi` may error (NVML) — verify GPU on a *compute* node.

## Run (on a GPU compute node, via the scheduler)
```bash
source bioemu_env/bin/activate
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=$HTTP_PROXY   # colabfold MSA needs egress
export http_proxy=$HTTP_PROXY https_proxy=$HTTP_PROXY
python -m bioemu.sample SEQ_OR_FASTA NUM_SAMPLES OUTDIR --batch_size_100 8
```
- `SEQUENCE` may be a raw sequence, a `.fasta`, or a `.a3m` (precomputed MSA). Without an a3m it calls
  **colabfold** to build the MSA/embedding → needs internet (works through the ALCF proxy).
- Each sample lands as `batch_*.npz` (`pos` in **nanometers** `(1,N,3)`, `node_orientations` `(1,N,3,3)`,
  `sequence`). At the end it writes `topology.pdb` + `samples.xtc`.
- `torch cuda avail: True` must print, and one A100 does ~100 samples of a ~400-mer in ~20 min.

## Convert npz → proper structures
Do NOT hand-roll backbone reconstruction (frame convention + nm-vs-Å will look "janky"). Use BioEmu's
own converter:
```python
import glob, numpy as np, torch
from bioemu.convert_chemgraph import save_pdb_and_xtc
files = sorted(glob.glob("OUTDIR/batch_*.npz"))
seq = np.load(files[0])["sequence"].item()
pos = torch.tensor(np.concatenate([np.load(f)["pos"] for f in files]))
ori = torch.tensor(np.concatenate([np.load(f)["node_orientations"] for f in files]))
save_pdb_and_xtc(pos_nm=pos, node_orientations=ori, sequence=seq,
                 topology_path="OUTDIR/topology.pdb", xtc_path="OUTDIR/samples.xtc")
```
Run this with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on a login node, or it hits
`libgomp: Thread creation failed`. Load as a trajectory: `mdtraj.load("samples.xtc", top="topology.pdb")`.

## Analyze flexibility (find the disordered/flexible regions)
Superpose on the ordered core, then per-residue RMSF + DSSP:
```python
import mdtraj as md, numpy as np
t = md.load("OUTDIR/samples.xtc", top="OUTDIR/topology.pdb")
ca = {r.resSeq: t.topology.select("name CA and resSeq %d" % r.resSeq)[0] for r in t.topology.residues}
core = [ca[r] for r in range(CORE_LO, CORE_HI+1)]
t.superpose(t, 0, atom_indices=core)
dssp = md.compute_dssp(t, simplified=True)          # H/E/C per residue per frame
# RMSF(region) = sqrt(mean_frames((xyz-mean)^2)) * 10 (nm->Angstrom); high RMSF + ~100% coil == IDR
```
Cluster loop/tail conformations (scipy `linkage`/`fcluster` on pairwise region-RMSD) to get K
representative states; but if intra-cluster RMSD is tens of Å and DSSP is ~100% coil, the region is a
featureless random coil — there are no meaningful discrete states, and binder design must read the
*sequence* (extended-motif/β-augmentation), not clamp a 3D conformation.

## Gotchas (all learned the hard way)
- torch cu130 vs driver → CPU fallback. Reinstall cu124. (biggest one)
- positions are in **nm**, not Å.
- login-node thread limits → set `OMP_NUM_THREADS=1` for conversion/analysis.
- MSA needs egress → export the proxy; or pass a precomputed `.a3m`.
- `save_pdb_and_xtc` for geometry — never reconstruct frames by hand.
