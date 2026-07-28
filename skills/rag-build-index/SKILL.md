---
name: rag-build-index
description: >-
  Build a searchable vector store over a text corpus (papers, binder records)
  from a JSONL of records — embed each text with all-MiniLM-L6-v2, then persist
  a FAISS inner-product index alongside an on-disk HuggingFace Arrow dataset.
  Use before you can semantically search or RAG-query a corpus; produces the
  faiss.index + dataset/ store the rag-query skill reads. Needs faiss, datasets,
  and sentence-transformers; CPU is fine.
---

# RAG: build index (faiss + datasets + sentence-transformers)

Turn a JSONL corpus into a persistent semantic search store. Each record's text
is embedded with `sentence-transformers/all-MiniLM-L6-v2` (mean-pooled,
L2-normalized), the vectors go into a `faiss.IndexFlatIP` (cosine similarity via
inner product on normalized vectors), and the texts+metadata are saved as an
on-disk HuggingFace Arrow dataset so retrieval can map a hit back to its record.
Cost/fidelity: the MiniLM encoder is small and CPU-friendly; a flat index is
exact (no approximation) and fine up to hundreds of thousands of records.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run the indexer yourself
over faiss + sentence-transformers. NEVER import bindify.

## When to use it

- You have a JSONL of text records (e.g. paper abstracts from the
  fetch-literature skill, or binder records) and want a store you can later
  query semantically.
- You are setting up a RAG pipeline and need the retrieval index before any
  question can be answered.

Do **not** use it to answer questions — that is the rag-query skill, which reads
the store this skill writes.

## Setup (gate before installing)

Needs faiss, HuggingFace datasets, and sentence-transformers. Check first;
install only if missing.

```bash
python -c "import faiss, datasets, sentence_transformers" 2>/dev/null && echo ready || pip install faiss-cpu datasets sentence-transformers
```

CPU is fine — the MiniLM encoder is small. First run downloads the encoder to
the local HuggingFace cache.

## Procedure

Read the JSONL (one record per line, each with at least a `text` field plus any
metadata), embed all texts, build a normalized inner-product index, and persist
both the index and the records:

- **Embed** with `all-MiniLM-L6-v2`. `SentenceTransformer.encode(...,
  normalize_embeddings=True)` already mean-pools token embeddings and
  L2-normalizes, so inner product == cosine similarity.
- **Index** with `faiss.IndexFlatIP(dim)` where `dim` is the embedding width
  (384 for MiniLM). `index.add(vectors)` — row `i` of the index corresponds to
  record `i` of the dataset, so keep their order identical.
- **Persist** the index with `faiss.write_index(index, "<out>/faiss.index")` and
  the records as a HuggingFace `Dataset.save_to_disk("<out>/dataset")`. The
  rag-query skill reloads both and relies on this positional alignment.

Inputs: input JSONL path, output directory. Output: `<out>/faiss.index` +
`<out>/dataset/`.

```python
import json, faiss
from datasets import Dataset
from sentence_transformers import SentenceTransformer

IN_JSONL = "/path/to/corpus.jsonl"   # one JSON record per line, needs "text"
OUT_DIR  = "/path/to/store"          # writes faiss.index + dataset/

records = [json.loads(line) for line in open(IN_JSONL) if line.strip()]
texts = [r["text"] for r in records]

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = encoder.encode(texts, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)

index = faiss.IndexFlatIP(vectors.shape[1])   # cosine via IP on normalized vecs
index.add(vectors)
faiss.write_index(index, f"{OUT_DIR}/faiss.index")

# Persist records in the SAME order as the vectors (row i <-> record i).
Dataset.from_list(records).save_to_disk(f"{OUT_DIR}/dataset")
print(f"indexed {len(records)} records -> {OUT_DIR}")
```

## Interpreting the result

- Success = a `faiss.index` file and a `dataset/` directory in the output dir,
  with the same number of vectors as records (`index.ntotal == len(records)`).
- The store is now ready for the rag-query skill. Nothing about relevance is
  decided here — this step only makes the corpus searchable.

## Notes

- Keep vector order and record order identical; retrieval maps a FAISS row index
  straight back to a dataset row.
- Deciding what corpus to index, how to clean/chunk the text, and which metadata
  fields to keep is your job as the agent. This skill only builds the store.
- Upstream: gather the corpus with the fetch-literature skill. Downstream: query
  it with the rag-query skill.
