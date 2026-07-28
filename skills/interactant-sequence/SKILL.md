---
name: interactant-sequence
description: >-
  Given an interacting partner's UniProt accession, fetch just its amino-acid
  sequence from the UniProt REST API using raw `requests`. Use when you have a
  list of interacting-partner UniProt IDs (e.g. the INTERACTION-comment partners
  from the `uniprot` skill) and need their sequences to fold or analyze them.
  Needs only `requests`.
---

# Interactant sequence (requests + UniProt REST API)

A focused subset of the `uniprot` skill: turn an interactant's UniProt accession
into its amino-acid sequence. One HTTP GET, seconds per call, no SDK or GPU.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run one `requests` call
directly.

## When to use it

- You have one or more UniProt IDs for a target's interacting partners (for
  example, the `uniProtKBAccession` values from the `uniprot` skill's INTERACTION
  comments) and you need their sequences to fold a complex or score them.

## Setup (gate before installing)

The only dependency is `requests`. Check first; install only if missing.

```bash
python -c "import requests" 2>/dev/null && echo ready || pip install requests
```

## Procedure

Fetch the entry JSON and read the sequence fields:

```bash
python - <<'PY'
import requests

BASE = "https://rest.uniprot.org"
UNIPROT_ID = "P01112"  # edit me

entry = requests.get(f"{BASE}/uniprotkb/{UNIPROT_ID}.json", timeout=30).json()
seq = entry["sequence"]["value"]
length = entry["sequence"]["length"]
print(UNIPROT_ID, length)
print(seq)
PY
```

For several partners, loop the accessions and collect `{id: sequence}`.

## Interpreting the result

- `entry["sequence"]["value"]` is the canonical isoform sequence;
  `entry["sequence"]["length"]` should match `len(seq)`.
- A failed lookup (KeyError / non-200) usually means an invalid or obsolete
  accession — verify the ID, or use the `uniprot` skill's ID-mapping to resolve
  it.

## Notes

- This skill only retrieves sequences. For full metadata, search, batch
  retrieval, ID mapping, or enumerating a target's interactors, use the
  `uniprot` skill.
- Deciding which partners to fetch and what to do with their sequences (fold,
  descriptor-score, filter) is your job as the agent.
