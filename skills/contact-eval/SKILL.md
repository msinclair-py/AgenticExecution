---
name: contact-eval
description: >-
  Identify which residues on a target protein chain are in direct contact with a
  binding partner chain (a designed binder or a natural interactant) from a
  folded complex structure (CIF or PDB, e.g. Chai-1 output), using MDAnalysis.
  Use to get structure-based evidence of a binding interface / target hotspots,
  to validate hotspots inferred from literature, or to annotate a designed
  binder's contact footprint. Needs only MDAnalysis and a complex structure file
  on disk.
---

# Contact evaluation (MDAnalysis)

Report the residues on a **target** chain that are in direct contact with a
**partner** chain in a folded complex, via a heavy-atom distance scan. A target
residue counts as a contact if any of its non-hydrogen atoms lies within the
cutoff (default 5 Å) of any non-hydrogen atom on the partner chain. Heavy-atom
contacts at 5 Å are a well-established proxy for the binding interface; fidelity
is bounded by the quality of the upstream fold. Cost is negligible — seconds on
CPU for one structure.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run a few lines of
MDAnalysis directly. The procedure below is everything you need.

## When to use it

- You have a folded complex (CIF/PDB) and want the list of interface residues.
- You want structure-based evidence to confirm or refute hotspots inferred from
  literature.
- You want to annotate a designed binder with the residues it contacts.

Two situations, same procedure:
- **binder–target complex** — the contacts are the binder's interface footprint
  on the target.
- **interactant–target complex** — the contacts are structure-validated hotspot
  evidence for the target.

Do **not** use it if no complex structure exists yet — fold the complex first.

## Setup (gate before installing)

The only dependency is MDAnalysis. Check first; install only if missing.

```bash
python -c "import MDAnalysis" 2>/dev/null && echo ready || pip install MDAnalysis
```

## Procedure

Load the complex, select non-hydrogen atoms of each chain, run a distance scan,
and report the unique target residues within the cutoff. Key details that make
the result correct and portable:

- **Match chains against both `segid` and `chainID`.** CIF (mmCIF, Chai-1
  default) and PDB label chains differently; selecting on both covers both.
- **Exclude hydrogens** before the scan — reduces noise and cost. Use the
  portable selection token `not name H*` (the bare `hydrogen` keyword is not
  recognized by MDAnalysis 2.x and raises `SelectionError`).
- **Report residues as single-letter code + residue number** (e.g. `R47`),
  de-duplicated and sorted by residue number. Code an unrecognized residue as
  `X`.
- **Chai-1 defaults:** target chain `A`, partner (binder/interactant) chain `B`.

```bash
python - <<'PY'
import json
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# --- edit these ---
STRUCTURE_PATH = "/path/to/complex.cif"  # folded complex, CIF or PDB
TARGET_CHAIN   = "A"                      # chain to report contacts FOR
PARTNER_CHAIN  = "B"                      # chain measured AGAINST
CUTOFF         = 5.0                      # heavy-atom threshold, Å
# ------------------

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

u = mda.Universe(STRUCTURE_PATH)
target = u.select_atoms(f"(segid {TARGET_CHAIN} or chainID {TARGET_CHAIN}) and not name H*")
partner = u.select_atoms(f"(segid {PARTNER_CHAIN} or chainID {PARTNER_CHAIN}) and not name H*")

residues = []
if len(target) and len(partner):
    dist = distance_array(target.positions, partner.positions)
    contact_atoms = target[np.any(dist <= CUTOFF, axis=1)]
    seen = set()
    tmp = []
    for a in contact_atoms:
        key = (a.resname, int(a.resid))
        if key in seen:
            continue
        seen.add(key)
        tmp.append((int(a.resid), f"{THREE_TO_ONE.get(a.resname.upper(), 'X')}{a.resid}"))
    tmp.sort(key=lambda x: x[0])
    residues = [label for _, label in tmp]

print(json.dumps({"contact_residues": residues, "n": len(residues)}))
PY
```

## Interpreting the result

- A longer list means a larger predicted interface footprint.
- An **empty list** means the two chains were not found in contact — or, most
  commonly, a **chain ID did not match** anything in the file. If you get an
  empty list, inspect the structure's actual chain labels
  (`sorted(set(u.atoms.chainIDs))` / `sorted(set(u.atoms.segids))`) before
  concluding there is no interface, and re-run with the correct labels.
- Larger cutoff → more, looser contacts; smaller → fewer, tighter. 5.0 Å is the
  standard interface proxy.

## Notes

- To evaluate several complexes, loop over the paths and run the scan per file —
  it is cheap enough to run serially.
- Deciding *which* complexes to evaluate and what to do with the residues
  afterward (validate a hotspot, annotate a binder) is your job as the agent —
  this skill only computes the contacts for one structure.
