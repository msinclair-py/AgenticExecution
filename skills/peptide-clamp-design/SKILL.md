---
name: peptide-clamp-design
description: Design mini-protein binders that clamp a short peptide / disordered-region motif, using Chroma conditioned on the ISOLATED target peptide (not the whole protein). Use when the binding target is a linear/disordered epitope (an IDR segment, a SLiM, a flexible tail) rather than a folded pocket — the standard full-protein binder-design setup fails there because binders drift to the large ordered surface. Covers target extraction, the isolated-peptide trick, generation, and engagement validation.
---

# Peptide-clamp binder design (for disordered / linear targets)

To design a binder against a **short or disordered epitope**, do NOT condition generation on the full
protein: with a big folded domain next to a thin flexible target, binders overwhelmingly stick to the
ordered core and ignore the epitope (observed: ~0 real engagement with a disordered tail on a
full-length receptor). Instead present the **target as an isolated peptide** — with no competing
surface, the binder is forced to clamp it. This gives ~90%+ engagement.

## When to use vs alternatives
- Use this for IDR segments, SLiMs, flexible tails, linear epitopes.
- For a **structureless random coil** (DSSP ~100% coil, RMSF tens of Å — see `bioemu-ensemble`),
  clamping one 3D conformation won't transfer across the ensemble; design instead against the
  *sequence* in an extended/β conformation (still an isolated-peptide run, but pick an extended frame).
- For a folded pocket, use normal full-receptor hotspot conditioning.

## 1. Extract the isolated target peptide
Pick a representative conformation (e.g. from a `bioemu-ensemble` medoid, or an extended frame for a
coil) and slice out just the target residues, keeping backbone atoms (N,CA,C,O):
```python
import mdtraj as md
t = md.load("ensemble/samples.xtc", top="ensemble/topology.pdb")
sel = t.topology.select("resSeq %d to %d" % (LO, HI))
t[REP_FRAME].atom_slice(sel).save_pdb("target_pep.pdb")   # e.g. 18-residue segment
```

## 2. Generate clamp binders (Chroma, isolated peptide as receptor)
Use a Chroma substructure-conditioned generator whose `RECEPTOR` is **env-configurable** and point it
at the peptide; hotspot = central peptide residues; binder lengths large enough to wrap the epitope
(~2-3× the target length). Run multi-GPU (one process per GPU, `CUDA_VISIBLE_DEVICES`), shared
`TMPDIR` for the Chroma weight cache.
```bash
CUDA_VISIBLE_DEVICES=$g RECEPTOR=target_pep.pdb HOTSPOT="345,348,351,354" \
  LENGTHS="35,45,55" OUT=clamp_out python gen_chroma_disordered.py
```
**Deploy check:** after editing the generator, re-`scp` it before running — a stale hardcoded
`RECEPTOR` silently reverts to the full protein (the failure mode above). Confirm the log prints
`receptor_residues == <peptide length>`, and directly inspect one output structure (chain A should be
the short peptide, not the whole protein).

## 3. Design binder sequences (ProteinMPNN)
The clamp complex is chain A = fixed target peptide, chain B = binder. Fix A, **design B**:
`assign_fixed_chains.py --chain_list "B"`. (Same as receptor binder design, but note the target may be
*shorter* than the binder.)

## 4. Validate engagement — with the two traps
Measure the fraction of target residues the binder contacts (< 5 Å). Two things break naive analysis:
- **Chroma renumbers the fixed peptide to 1..N** — don't match on original residue numbers.
- The target peptide can be **shorter than the binder**, so "shortest chain == binder" is wrong.
Identify the target chain by its **known length**; the binder is the other chain:
```python
tchain = [c for c in chains if len(c) == TARGET_LEN][0]
bchain = [c for c in chains if len(c) != TARGET_LEN]
cov = fraction of tchain residues within 5 Å of any bchain atom
# healthy: median coverage ~0.6-0.7, >90% of binders engage >=50% of the target
```
A near-zero coverage almost always means a bug (stale full-protein run, mixed old output files, or the
chain-identification traps above) — inspect an actual structure before concluding the method failed.

## 5. Confirm in context
Isolated-peptide clamp proves the *interface* is designable, but validate the winners by co-folding
`binder + full-length protein (+ cofactors)` and checking the binder still engages the target region
in the presence of the rest of the protein. Rank by interface confidence (Chai iptm) at the target.

## Lessons baked in
- Isolated peptide beats full-protein for disordered/linear targets.
- Redeploy edited scripts; inspect a structure before trusting aggregates; purge stale outputs.
- Random-coil targets → design against sequence (extended conformation), not one 3D state.
