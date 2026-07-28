---
name: developability
description: >-
  Assess whether a designed protein/peptide binder is DEVELOPABLE — a realistic
  lab/therapeutic candidate — not just a good in-silico binder. Fast sequence-level
  scan (chemical liabilities, charge/pI, hydrophobicity, aggregation/solubility
  proxy, low-complexity, immunogenicity flags) that scales to millions of
  sequences as a funnel filter; plus a structure-level check (hydrophobic patches,
  charge patches) for the folded shortlist. Use to filter/rank a designed library
  toward manufacturable, soluble, low-immunogenicity, low-liability binders.
---

# Binder developability assessment

A binder that aggregates, is insoluble, is immunogenic, or carries chemical
liabilities won't survive expression, purification, or the clinic — no matter how
good its predicted affinity. Developability screens for those risks. The field's
tools are CamSol/Aggrescan (solubility/aggregation), the Therapeutic Antibody
Profiler / TAP (hydrophobicity + charge patches), sequence-liability scans, and
MHC-II immunogenicity prediction; good pipelines optimize a Pareto front across
affinity, specificity, stability, solubility, expression, and immunogenicity.

Two tiers: a **fast sequence scan** (cheap → run on the whole library as a filter)
and a **structure-level** check (for candidates that have been folded).

This is a **self-contained, instruction-only skill**. The sequence tier needs only
`peptides` + stdlib; the structure tier needs MDAnalysis/biopython; heavier
external tools (CamSol, Aggrescan, NetMHCIIpan) are noted for the shortlist.

## When to use it

- Filtering a large designed library (e.g. 1M sequences) down to developable
  candidates before spending fold/MD compute.
- Ranking a folded shortlist by manufacturability + safety, not just affinity.

## Tier 1 — sequence scan (fast, scales to millions)

Flag chemical/physical liabilities and score solubility/aggregation from sequence:

- **Chemical liabilities** (motifs; higher risk if in flexible/exposed loops):
  deamidation `N[GSTNH]`; isomerization/cleavage `D[GSTPH]`; N-glycosylation
  sequon `N[^P][ST]`; oxidation-prone exposed `M`/`W`; unpaired/odd `C` count;
  N-terminal `Q`/`E` (pyroglutamate); `DP` acid-cleavage; free-cysteine risk.
- **Charge / pI:** net charge at pH7 and isoelectric point (via `peptides`);
  avoid pI ≈ physiological (7.0±1) — low solubility — and avoid extreme net charge.
- **Hydrophobicity / aggregation:** GRAVY (Kyte–Doolittle mean) and the maximum
  hydrophobic-window score; a CamSol-like intrinsic-solubility proxy combining
  per-residue hydrophobicity, charge, and β-aggregation propensity.
- **Low-complexity / repeats:** reject homopolymer runs (≥4 same residue) and
  very low sequence entropy (expression/aggregation risk).

```python
import re, math
import peptides

LIABILITY = {
    "deamidation": r"N[GSTNH]", "isomerization": r"D[GSTPH]",
    "n_glycosylation": r"N[^P][ST]", "acid_cleavage_DP": r"DP",
}
KD = dict(zip("ACDEFGHIKLMNPQRSTVWY",
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))

def developability(seq):
    seq = seq.upper()
    flags = {k: [m.start() for m in re.finditer(p, seq)] for k, p in LIABILITY.items()}
    flags["oxidation_MW"] = [i for i, a in enumerate(seq) if a in "MW"]
    flags["free_cys"] = seq.count("C")
    flags["nterm_pyroE"] = seq[0] in "QE"
    # low complexity: any residue >4 in a row, or Shannon entropy low
    runs = max((len(r.group()) for r in re.finditer(r"(.)\1*", seq)), default=1)
    from collections import Counter
    p = [c/len(seq) for c in Counter(seq).values()]
    entropy = -sum(x*math.log2(x) for x in p)
    pep = peptides.Peptide(seq)
    charge = pep.charge(pH=7.0); pI = pep.isoelectric_point()
    gravy = sum(KD.get(a,0) for a in seq)/len(seq)
    win = max(sum(KD.get(a,0) for a in seq[i:i+7])/7 for i in range(max(1,len(seq)-6)))
    # simple pass/flag: penalize liabilities, near-neutral pI, high hydrophobic window
    n_liab = sum(len(v) if isinstance(v,list) else 0 for v in flags.values())
    risky = (n_liab>3) or (runs>=4) or (entropy<2.5) or (6.0<pI<8.0) or (win>1.5) or (abs(charge)>9)
    return {"seq_len": len(seq), "net_charge_pH7": round(charge,2), "pI": round(pI,2),
            "gravy": round(gravy,2), "max_hydrophobic_window": round(win,2),
            "max_run": runs, "entropy": round(entropy,2), "n_liabilities": n_liab,
            "flags": {k:(v if isinstance(v,int) or isinstance(v,bool) else len(v)) for k,v in flags.items()},
            "developable": not risky}
```

Use `developable` (and the numeric fields) as a **funnel filter + rank key**: keep
developable sequences, and among binders of similar predicted affinity prefer
lower liabilities, GRAVY, and hydrophobic-window, and pI away from 7.

## Tier 2 — structure-level (folded shortlist)

On the folded binder (or complex), add spatial developability:

- **Hydrophobic patches / SAP** (spatial aggregation propensity): sum exposed
  hydrophobic surface in local patches (MDAnalysis SASA + KD weighting); large
  patches → aggregation/poor expression.
- **Charge patches / dipole** (TAP-like): clustered same-charge surface residues.
- **Exposed unpaired Cys / free thiols.**
- **Predicted stability:** use the fold confidence (pLDDT/ipTM) as a proxy, or a
  ΔΔG/stability predictor; unstable binders express and behave poorly.

## Tier 3 — external tools (top candidates only)

- **CamSol** intrinsic/structurally-corrected solubility; **Aggrescan3D** for
  structure-aware aggregation; **TAP** for antibody-like profiles.
- **Immunogenicity:** MHC-II epitope density via **NetMHCIIpan** (T-cell epitope
  scan) — important for therapeutic candidates; minimize predicted binders across
  common HLA-II alleles.

## Notes

- Run Tier 1 across the whole library (cheap, `peptides`+stdlib); reserve Tiers
  2–3 for candidates that pass affinity/fold gates.
- Deciding thresholds (how strict) and which liabilities are acceptable for the
  intended modality is your job as the agent / the supervisor's call.
- Downstream: combine with `filter-binders` (bucket on a developability score
  alongside affinity), and feed the labels into the active-learning surrogate so
  the generator learns to produce developable binders.
