---
name: uniprot
description: >-
  Fetch a target protein's sequence, metadata, and known interactors directly
  from the UniProt REST API using raw `requests` — get one entry by accession
  (JSON or FASTA), run a search query, stream large result sets, batch-retrieve
  many accessions, or run the async ID-mapping service. Parse a UniProtKB entry
  for sequence, recommended name, gene names, organism, and INTERACTION comments
  (interaction partners with IntAct IDs and experiment counts) plus interaction
  cross-references (IntAct, STRING, BioGRID, DIP, MINT, ComplexPortal, ELM,
  CORUM). Use when you need a target's sequence/metadata or its list of known
  interacting partners by accession or search query. Needs only `requests`.
---

# UniProt REST access (requests + UniProt REST API)

Query the public UniProt REST API (`https://rest.uniprot.org`) with plain HTTP
via `requests`. There is no SDK, no model weights, and no GPU — each call is a
network round-trip taking seconds. Use it to pull a protein's sequence and
metadata, search for candidates, or enumerate a target's known interactors.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run a few `requests` calls
directly. The procedures below are everything you need.

## When to use it

- You have a UniProt accession (e.g. `P04637`) and need its amino-acid sequence,
  name, genes, organism, or known interactors.
- You have a search query (gene name, protein name, keyword) and need to find
  matching entries.
- You want a target's known interaction partners as a starting pool of
  interactants to fold or analyze downstream.

## Setup (gate before installing)

The only dependency is `requests`. Check first; install only if missing.

```bash
python -c "import requests" 2>/dev/null && echo ready || pip install requests
```

## Procedure

All requests go to `BASE = "https://rest.uniprot.org"`. The core endpoints:

- **Get one entry** — `GET {BASE}/uniprotkb/{accession}.json` (or `.fasta` for
  raw FASTA). No params.
- **Search** — `GET {BASE}/uniprotkb/search` with params
  `{query, format: "json", size, fields}` where `fields` is a comma-joined list
  of result fields.
- **Stream (large result sets)** — `GET {BASE}/uniprotkb/stream` with params
  `{query, format, fields}`, `stream=True`, consumed via `iter_content`.
- **Batch retrieve** — build a query
  `"accession:X OR accession:Y OR ..."` and hit `search` with `size=len(ids)`.
- **ID mapping (async, 3 steps, max 100k ids)** —
  1. `POST {BASE}/idmapping/run` with data `{from, to, ids}` (ids comma-joined)
     → returns a `jobId`.
  2. Poll `GET {BASE}/idmapping/status/{jobId}` every ~3 s until the response
     contains `results` or `failedIds`.
  3. `GET {BASE}/idmapping/results/{jobId}?format=json` for the mapped results.
- **Field discovery** — `GET {BASE}/configure/uniprotkb/result-fields` lists the
  valid `fields` values; `GET {BASE}/configure/idmapping/fields` lists the
  ID-mapping database names for `from`/`to`.

### Parsing a UniProtKB JSON entry

- `entry["sequence"]["value"]` / `entry["sequence"]["length"]` — the sequence.
- `entry["proteinDescription"]["recommendedName"]["fullName"]["value"]` — name.
- `entry["genes"]` — each has `geneName.value` plus `synonyms`.
- `entry["organism"]["scientificName"]` / `entry["organism"]["taxonId"]`.
- `entry["comments"]` where `commentType == "INTERACTION"` — interaction
  partners: each interaction has `interactantOne` and `interactantTwo`, and each
  interactant carries `uniProtKBAccession`, `geneName`, `intActId`,
  `numberOfExperiments`, and `organismDiffer`.
- `entry["comments"]` where `commentType == "SUBUNIT"` — free-text complex
  description.
- `entry["uniProtKBCrossReferences"]` filtered to interaction databases
  (`IntAct`, `STRING`, `BioGRID`, `DIP`, `MINT`, `ComplexPortal`, `ELM`,
  `CORUM`).

### Compact example — get + parse

```bash
python - <<'PY'
import requests

BASE = "https://rest.uniprot.org"
ACCESSION = "P04637"  # edit me

entry = requests.get(f"{BASE}/uniprotkb/{ACCESSION}.json", timeout=30).json()

# sequence
seq = entry["sequence"]["value"]
length = entry["sequence"]["length"]

# name
name = (entry.get("proteinDescription", {})
             .get("recommendedName", {})
             .get("fullName", {})
             .get("value"))

# genes
genes = [g.get("geneName", {}).get("value") for g in entry.get("genes", [])]

# known interactors from INTERACTION comments
interactors = []
for c in entry.get("comments", []):
    if c.get("commentType") != "INTERACTION":
        continue
    for inter in c.get("interactions", []):
        for side in ("interactantOne", "interactantTwo"):
            p = inter.get(side, {})
            acc = p.get("uniProtKBAccession")
            if acc and acc != ACCESSION:
                interactors.append({
                    "accession": acc,
                    "gene": p.get("geneName"),
                    "intActId": p.get("intActId"),
                    "numberOfExperiments": inter.get("numberOfExperiments"),
                    "organismDiffer": inter.get("organismDiffer"),
                })

print(name, "|", "/".join(str(g) for g in genes), "| len", length)
print(seq[:60], "...")
for i in interactors[:10]:
    print(i)
PY
```

## Interpreting the result

- The INTERACTION comments give experimentally observed partners; a high
  `numberOfExperiments` is stronger evidence. `organismDiffer` true means the
  partner is from a different organism (often a pathogen/host pair).
- An empty interactor list does not mean no interactions exist — cross-check the
  interaction cross-references (IntAct/STRING/etc.) for additional partners.
- A search returning no hits usually means the query syntax is off; use the
  field-discovery endpoint to confirm valid field names and refine the query.

## Notes

- Deciding *which* accession or query to run, and what to do with the results
  (fold a partner, seed a design campaign, validate a hotspot) is your job as the
  agent — this skill only fetches and parses.
- To pull sequences for a list of interactor accessions you found here, see the
  `interactant-sequence` skill.
