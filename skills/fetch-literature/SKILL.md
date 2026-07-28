---
name: fetch-literature
description: >-
  Gather a literature corpus about a protein/target from PubMed E-utilities and
  Europe PMC (raw HTTP, no API key needed) and write it as JSONL records
  (text + pmid/title/year) ready for the rag-build-index skill. Use to collect
  papers/abstracts about a target before building a RAG store to mine hotspots,
  PTM sites, or known interactions. Needs only `requests`; remote HTTP,
  seconds-to-minutes, no GPU.
---

# Fetch literature (requests → PubMed E-utilities + Europe PMC)

Given a protein/target name or a search query, pull relevant literature from
public biomedical APIs and assemble it into JSONL records suitable for the
rag-build-index skill. Two complementary sources: **PubMed E-utilities**
(abstracts + metadata for almost everything) and **Europe PMC** (adds
open-access full text where available). No API key is required for basic use.
Cost/fidelity: plain remote HTTP — seconds to a couple of minutes depending on
result count; no GPU.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) make the HTTP calls yourself with
`requests`. NEVER import bindify. (A `fetch_pubmed` helper existed in the repo,
but this skill stands alone on `requests`.)

## When to use it

- You want a corpus of papers/abstracts about a target before building a RAG
  store to mine hotspots / PTM sites / known interactions.
- You need JSONL literature records to feed the rag-build-index skill.

Do **not** use it to answer questions — this only collects text; querying is the
rag-build-index → rag-query pair.

## Setup (gate before installing)

Only dependency is `requests`. Check first; install only if missing.

```bash
python -c "import requests" 2>/dev/null && echo ready || pip install requests
```

## Procedure

For the query, get IDs then fetch content, and emit one JSON record per line:

- **PubMed (a).** `esearch` for PMIDs:
  `GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=<query>&retmax=N&retmode=json`.
  Then `efetch` (`db=pubmed&id=<csv pmids>&rettype=abstract&retmode=xml`) or
  `esummary` for abstracts + metadata (title, year, journal).
- **Europe PMC (b).** For open-access full text:
  `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=<query>&format=json&pageSize=N`.
- **Assemble** each result into a record with at least a `text` field (abstract
  or full text) plus metadata (`pmid`, `title`, `year`, `source`). Write JSONL —
  exactly the shape rag-build-index expects.

**Be polite to NCBI:** rate-limit your requests (≈3/sec without a key), and
include `tool=` and `email=` params on E-utilities calls.

Inputs: protein/target name or search query, max results `N`. Output: a JSONL
file of literature records.

```python
import json, time, requests

QUERY   = "MDM2 p53 binding hotspot"
MAX     = 30
OUT     = "/path/to/corpus.jsonl"
EMAIL   = "you@example.org"          # your email — be a polite NCBI citizen
TOOL    = "bindify-fetch-literature"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
records = []

# (a) PubMed: esearch -> PMIDs -> esummary metadata.
ids = requests.get(f"{EUTILS}/esearch.fcgi", params={
    "db": "pubmed", "term": QUERY, "retmax": MAX, "retmode": "json",
    "tool": TOOL, "email": EMAIL}, timeout=60).json()["esearchresult"]["idlist"]
if ids:
    summ = requests.get(f"{EUTILS}/esummary.fcgi", params={
        "db": "pubmed", "id": ",".join(ids), "retmode": "json",
        "tool": TOOL, "email": EMAIL}, timeout=60).json()["result"]
    for pmid in ids:
        d = summ.get(pmid, {})
        records.append({"pmid": pmid, "title": d.get("title", ""),
                        "year": (d.get("pubdate", "") or "")[:4],
                        "source": "pubmed", "text": d.get("title", "")})
        time.sleep(0.34)             # ~3 req/sec
# For real abstracts, follow up with efetch (rettype=abstract, retmode=xml)
# and parse the AbstractText into each record's "text".

# (b) Europe PMC: open-access full text where available.
epmc = requests.get(
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
    params={"query": QUERY, "format": "json", "pageSize": MAX},
    timeout=60).json().get("resultList", {}).get("result", [])
for r in epmc:
    records.append({"pmid": r.get("pmid", ""), "title": r.get("title", ""),
                    "year": str(r.get("pubYear", "")), "source": "europepmc",
                    "text": r.get("abstractText") or r.get("title", "")})

with open(OUT, "w") as fh:
    for rec in records:
        fh.write(json.dumps(rec) + "\n")
print(f"wrote {len(records)} records -> {OUT}")
```

## Interpreting the result

- Success = a JSONL file where each line has a non-empty `text` plus metadata.
  Records with empty `text` (title-only) contribute little to a RAG store —
  follow up with `efetch` for the actual abstract when you need substance.
- PubMed covers breadth (abstracts); Europe PMC adds open-access full text.
  De-duplicate on `pmid` if you merge both sources.

## Notes

- Rate-limit and include `tool`/`email` on E-utilities calls; register for an
  NCBI API key if you need higher throughput.
- Deciding the query terms, how many results to pull, and which sources to merge
  is your job as the agent. This skill only fetches and formats the corpus.
- Downstream: feed the JSONL to rag-build-index, then ask questions with
  rag-query.
