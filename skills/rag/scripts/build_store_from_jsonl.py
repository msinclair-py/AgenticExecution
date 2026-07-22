"""Build a FAISS RAG store from an existing JSONL of {text, ...} records.

Usage:
    python build_store_from_jsonl.py <corpus.jsonl> <output_dir> [chunk_size] [overlap]
"""
from __future__ import annotations

import sys

from rag_build import build_store, read_jsonl


def main(jsonl: str, out_dir: str, chunk_size: int = 1500, overlap: int = 200) -> None:
    records = read_jsonl(jsonl)
    if not records:
        raise SystemExit(f'No records with non-empty text in {jsonl}')
    result = build_store(records, out_dir, chunk_size=chunk_size, chunk_overlap=overlap)
    print(f"Built store: {result['ntotal']} chunks, dim {result['dim']}")
    print(' ', result['faiss_index_path'])
    print(' ', result['dataset_dir'])


if __name__ == '__main__':
    cs = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    ov = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    main(sys.argv[1], sys.argv[2], cs, ov)
