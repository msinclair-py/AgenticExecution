"""Fetch a target's PubMed abstracts and build a FAISS store on the fly.

Also materializes the fetched corpus to curated_dataset.jsonl beside the store
so it can be re-embedded deterministically later.

Usage:
    NCBI_API_KEY=... python fetch_literature_and_build.py <QUERY> <output_dir> [max_pmids]

QUERY is a PubMed term, e.g. 'NF1[Gene]' or a free-text phrase. To fetch by
UniProt ID with curated PMIDs + Europe PMC + Semantic Scholar + full text, use
the bindify toolbelt's fetch_pubmed.fetch_for_protein instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fetch_pubmed_min import fetch_pubmed_abstracts
from rag_build import build_store, write_jsonl


def main(query: str, out_dir: str, max_pmids: int = 200) -> None:
    out = Path(out_dir)
    records = fetch_pubmed_abstracts(query, max_pmids=max_pmids)
    if not records:
        raise SystemExit(f'No records fetched for {query!r}')
    write_jsonl(records, out / 'curated_dataset.jsonl')
    result = build_store(records, out)
    print(f"Fetched {len(records)} papers -> {result['ntotal']} chunks in {out}")


if __name__ == '__main__':
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    main(sys.argv[1], sys.argv[2], n)
