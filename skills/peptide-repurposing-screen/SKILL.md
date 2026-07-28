---
name: peptide-repurposing-screen
description: Screen a large peptide database (e.g. known drugs/anticancer peptides) against a protein target by co-folding, to find existing peptides that bind — a repurposing/baseline screen. Also use it to benchmark de-novo designs against real therapeutics and to calibrate a developability scorer. Covers cleaning the DB, building a fold library, scalable co-folding, and the two-phase iptm+region analysis with its interpretation traps.
---

# Peptide-library repurposing / baseline screen

Given a database of peptides (existing drugs, anticancer peptides, natural products) and a protein
target, co-fold each peptide against the target and rank by predicted interface confidence to find
binders. Two honest uses: (1) *repurposing* — do any existing peptides bind your target; (2)
*baseline* — how do your de-novo designs compare to real therapeutics (usually design wins).

## 1. Clean the database
Real peptide DBs are messy — chemical modifications, D-/non-natural residues, SMILES-like strings.
Keep only natural-AA sequences of a foldable length; dedupe:
```python
aa=set("ACDEFGHIKLMNPQRSTVWY"); clean=[]; seen=set()
for s in sequences:
    s=s.strip().upper()
    if 5<=len(s)<=60 and all(c in aa for c in s) and s not in seen:
        seen.add(s); clean.append(s)
```
An 800k-entry DB may still yield ~800k clean, or far fewer — check. For interface-forming binders,
prefer 12–40 aa; sample diversely (`clean[::step]`) if you can't fold all.

## 2. Developability calibration (cheap, do it)
Score the clean DB with your developability scorer (see `developability`). Real therapeutic peptides
give you the reference distribution — e.g. anticancer peptides median dev ~0.80, ~30% ≥0.85. If your
designed binders sit in the same band, the scorer is calibrated and your designs are drug-like. Make a
2-panel overlay figure (DB vs your designs).

## 3. Build the fold library + co-fold
Write each peptide as a record `{id, seq, len, dev_score}` in a `library.jsonl`. Co-fold `target +
peptide (+ cofactors)` with a device-agnostic folding runner (see the folding pipeline / Chai holo
setup). For a **full-length** or non-default target, make the runner's `TARGET_FASTA` env-configurable.
Scale on the cluster: `mpiexec` one rank per GPU tile, checkpoint by peptide id. ~600 folds/node-hr for
a ~400-residue target; 15k folds ≈ 25 node-hr → a 32-node debug-scaling job in ~45 min.

## 4. Two-phase analysis (avoid the small-file timeout)
Do NOT parse all N structures inline — reading tens of thousands of small files over a shared FS times
out. Instead:
- **Phase 1 (fast):** read only the `results/*.json` (has iptm), rank all N by iptm.
- **Phase 2:** parse structures only for the **top ~600** to classify *where* each binds (define target
  region footprints — e.g. catalytic site, an ordered element, a disordered region — and count binder
  contacts <5 Å).
Run detached (`nohup ... &`) writing to a log; don't rely on a 120 s SSH call.

## 5. Interpretation traps
- **iptm on a disordered region is unreliable.** A structureless tail is "sticky" — many peptides
  loosely drape on it and score a modest iptm without specific binding. Discount disordered-region
  "hits"; trust hits at ordered pockets/sites. (In one screen, 499/600 top hits fell on the disordered
  tail — almost all non-specific.)
- **No hit reaching design-quality iptm is a real result**, not a failure: it means de-novo design >
  repurposing for this target, and the DB's value is as a developability benchmark + scaffold source.
- Report the iptm ceiling of the DB vs your designs side by side.

## Output
`{db}_ranked.json` (top hits with iptm + region), a region-distribution + iptm-comparison figure, and a
one-line conclusion (repurposing hits vs design-wins).
