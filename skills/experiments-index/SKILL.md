---
name: experiments-index
description: >-
  Incrementally append campaign result records (designed binders + their
  analyses rendered to text + metadata) to a growing on-disk vector store —
  embed each record, L2-normalize, add the vectors to an existing faiss index and
  concatenate the records into an existing HuggingFace Arrow dataset, all under an
  fcntl file lock so concurrent writers don't corrupt the store. Use to maintain a
  queryable memory of everything a campaign has produced so later rounds can
  retrieve prior results (rag-query style). This APPENDS to an existing store
  (unlike rag-build-index, which builds fresh). CPU is fine; needs faiss +
  datasets + a sentence encoder.
---

# Experiments index — incremental vector-store append (faiss + HuggingFace datasets)

Keep a persistent, searchable record of a campaign's output. Each result record —
a designed binder plus its computed analyses rendered to text, plus metadata — is
embedded with a sentence encoder, L2-normalized, and **appended** to an existing
faiss index while the record itself is **concatenated** into an existing
HuggingFace Arrow dataset. The whole append happens under an `fcntl` file lock so
multiple campaign workers writing at once cannot corrupt the store. This is the
**incremental appender** — use rag-build-index to create a store from scratch, and
rag-query to search one. Cost is negligible: CPU embedding + on-disk I/O.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) run faiss + datasets directly. The
procedure below is everything you need.

## When to use it

- You just produced a batch of binders + analyses in a campaign round and want to
  fold them into a long-lived, queryable memory.
- You want later design rounds to retrieve prior results via a rag-query search
  ("what did we already try / what worked").
- The store already exists (or should be created on first write) and multiple
  workers may append concurrently.

Use **rag-build-index** instead when starting a store fresh, and **rag-query** to
read one back.

## Setup (gate before installing)

Needs faiss, HuggingFace `datasets`, and a sentence-embedding encoder. CPU is
fine. Check first; install only if missing.

```bash
python -c "import faiss, datasets" 2>/dev/null && echo ready || pip install faiss-cpu datasets
```

The encoder can be any mean-pooled, normalized sentence embedder (e.g. a
`sentence-transformers` / HF model). Load it once and reuse it.

## Procedure

Under a file lock: read (or create) the index, embed the new texts, add them,
write the index back, then load + concatenate + save the dataset, then release the
lock. Key details that keep the store correct:

- **Hold an `fcntl` lock the whole time** — acquire an exclusive lock on a lock
  file next to the index so concurrent appenders serialize; corruption otherwise.
- **Use inner-product with normalized vectors** — `IndexFlatIP` + L2-normalized
  embeddings makes cosine similarity, matching rag-query.
- **Create on first write** — if the index file is absent, make a fresh
  `IndexFlatIP(dim)`; if the dataset dir is absent, the new records become the
  dataset.
- **Never `save_to_disk` over the dir you loaded from** — HuggingFace `datasets`
  memory-maps the dataset, so overwriting it in place raises "a dataset can't
  overwrite itself." Save the merged dataset to a temp dir, drop the mmap
  (`del existing, merged`), then swap the temp dir into place.
- **Vectors and records must stay row-aligned** — add embeddings and concatenate
  records in the same order so index row `i` corresponds to dataset row `i`. Write
  the faiss index only after the dataset commit so a failure can't desync them.

```python
import os, fcntl, json, shutil
import numpy as np
import faiss
from datasets import Dataset, load_from_disk, concatenate_datasets

# --- edit these ---
DATASET_DIR       = "/path/to/experiments_dataset"   # HuggingFace Arrow dataset on disk
FAISS_INDEX_PATH  = "/path/to/experiments.faiss"     # faiss index on disk
LOCK_PATH         = FAISS_INDEX_PATH + ".lock"
# records: list of dicts you selected for this round, each with a "text" field
# (binder + rendered analyses) plus any metadata columns.
records = [ {"text": "...", "binder_id": "...", "iteration": 3, "priority": 0.9}, ]
# ------------------

def encode(texts):
    # any mean-pooled sentence encoder; L2-normalize for cosine via inner product
    vecs = MODEL.encode(texts)                       # -> (n, dim) float32
    vecs = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

texts = [r["text"] for r in records]

lock = open(LOCK_PATH, "w")
fcntl.flock(lock, fcntl.LOCK_EX)                     # serialize concurrent appenders
try:
    vecs = encode(texts)
    dim = vecs.shape[1]

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(dim)               # create fresh on first write
    index.add(vecs)

    # Commit the dataset FIRST. A dataset loaded from DATASET_DIR is memory-mapped
    # from it, so concatenate + save_to_disk(DATASET_DIR) raises "a dataset can't
    # overwrite itself" — write to a temp dir, drop the mmap, then swap it in.
    new_ds = Dataset.from_list(records)
    if os.path.isdir(DATASET_DIR) and os.listdir(DATASET_DIR):
        existing = load_from_disk(DATASET_DIR)
        merged = concatenate_datasets([existing, new_ds])
        tmp = DATASET_DIR + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        merged.save_to_disk(tmp)
        del existing, merged                         # release the mmap on DATASET_DIR
        shutil.rmtree(DATASET_DIR)
        shutil.move(tmp, DATASET_DIR)
    else:
        new_ds.save_to_disk(DATASET_DIR)             # first write

    # Write the index only AFTER the dataset commit, so a mid-append failure can't
    # leave the index ahead of the dataset (row i must still map to record i).
    faiss.write_index(index, FAISS_INDEX_PATH)

    result = {
        "appended": len(records),
        "index_size": index.ntotal,
        "dataset_dir": DATASET_DIR,
        "faiss_index_path": FAISS_INDEX_PATH,
    }
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()

print(json.dumps(result))
```

## Interpreting the result

- **appended** — how many records this call added.
- **index_size** — the total vectors now in the store (`index.ntotal`); should
  equal the dataset row count. A mismatch means a partial/failed append — inspect
  before trusting subsequent queries.
- **dataset_dir / faiss_index_path** — the store paths to hand to rag-query.

## Notes

- Inputs you supply: the records to append, `dataset_dir`, `faiss_index_path`, the
  encoder/pooler config, and selection params — `max_records`, `min_priority`,
  `include_prior_iterations`.
- **Which binders and analyses to include is your orchestration job.** This skill
  performs the real vector-store I/O over faiss + datasets; deciding what text to
  render for each record and which records clear your selection thresholds is up
  to you as the agent. Cross-reference rag-query (to search this store) and
  rag-build-index (to create one from scratch).
