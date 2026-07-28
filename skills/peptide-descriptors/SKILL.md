---
name: peptide-descriptors
description: >-
  Compute physicochemical descriptors for a peptide/binder sequence with the
  `peptides` (peptides.py) package — molecular weight, net charge, isoelectric
  point, instability index, aliphatic index, Boman index, hydrophobicity and
  hydrophobic moment, plus amino-acid composition, a predicted structural class,
  and a membrane-position profile. Use to score and rank a binder sequence's
  properties for filtering. Pure CPU, deterministic, sub-second, no weights or
  network — needs only the `peptides` package.
---

# Peptide descriptors (peptides.py)

Score a peptide sequence's physicochemical properties with the `peptides`
package. Everything runs on CPU in well under a second, is deterministic, and
needs no model weights or network access. The output is a plain dict of scalar
metrics plus composition and profile data — exactly the kind of per-binder
metric the `filter-binders` skill buckets on.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run a few lines of
`peptides` directly.

## When to use it

- You have a peptide/binder amino-acid sequence and want physicochemical scores
  (MW, net charge, pI, hydrophobicity, instability, membrane profile) to filter
  or rank candidates.

## Setup (gate before installing)

The only dependency is `peptides`. Check first; install only if missing.

```bash
python -c "import peptides" 2>/dev/null && echo ready || pip install peptides
```

## Procedure

Build one `peptides.Peptide(sequence)` and read three descriptor families off
it. Key parameters and their defaults are inlined below.

- **Physicochemical** —
  `p.molecular_weight()`;
  `p.charge(pH=7.0, pKscale="Lehninger")`;
  `p.isoelectric_point()`;
  `p.instability_index()`;
  `p.aliphatic_index()`;
  `p.boman()`;
  `p.hydrophobicity(scale="KyteDoolittle")`;
  `p.hydrophobic_moment()`.
- **Composition** — `p.counts()` and `p.frequencies()` over the 20 canonical AAs.
- **Structural** —
  `p.structural_class(frequencies="Nakashima", distance="mahalanobis")`;
  `p.membrane_position_profile(window=11, angle=100)`.

Defaults to carry forward: `pH=7.0`, `pK_scale="Lehninger"`,
`hydrophobicity_scale="KyteDoolittle"`, `membrane_window=11`,
`membrane_angle=100`, `structural_frequencies="Nakashima"`,
`structural_distance="mahalanobis"`.

```bash
python - <<'PY'
import json
import peptides

SEQUENCE = "KLAKLAKKLAKLAK"  # edit me

p = peptides.Peptide(SEQUENCE)

result = {
    "sequence": SEQUENCE,
    "physchem": {
        "molecular_weight": p.molecular_weight(),
        "charge": p.charge(pH=7.0, pKscale="Lehninger"),
        "isoelectric_point": p.isoelectric_point(),
        "instability_index": p.instability_index(),
        "aliphatic_index": p.aliphatic_index(),
        "boman": p.boman(),
        "hydrophobicity": p.hydrophobicity(scale="KyteDoolittle"),
        "hydrophobic_moment": p.hydrophobic_moment(),
    },
    "counts": p.counts(),
    "frequencies": p.frequencies(),
    "structural_class": p.structural_class(
        frequencies="Nakashima", distance="mahalanobis"
    ),
    "membrane_position_profile": p.membrane_position_profile(window=11, angle=100),
}

print(json.dumps(result, indent=2))
PY
```

## Interpreting the result

- **charge / isoelectric_point** — net charge at pH 7 and the pH of zero net
  charge; relevant for solubility and electrostatic complementarity.
- **hydrophobicity / hydrophobic_moment** — mean hydrophobicity and its
  amphipathic periodicity; a high moment suggests a facially amphipathic helix.
- **instability_index** — a common cutoff is ~40 (above suggests an unstable
  peptide).
- **boman / membrane_position_profile** — potential protein-binding /
  membrane-interaction propensity.
- These are sequence-only estimates, not structure-based measurements — use them
  for triage, not as ground truth.

## Notes

- Deciding which descriptor to key on and gathering the values across a pool of
  candidates is your job as the agent — this skill computes the metrics for one
  sequence.
- To triage a pool of binders on any of these scalars into keep/cull priority
  tiers, pass the values to the `filter-binders` skill.
