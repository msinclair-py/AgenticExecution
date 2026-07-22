---
name: rag
description: Retrieval-augmented generation over a scientific-literature corpus — build a FAISS vector store from text records (JSONL or fetched PubMed/Europe PMC/Semantic Scholar), then query it for free-text synthesis or structured extraction (protein interactions, PTM sites, known binders). Covers chunking + embedding with a distllm encoder/pooler into a FAISS IndexFlatIP + HuggingFace dataset, the retriever+LLM generation glue, choosing a generator backend (OpenAI-compatible / vLLM / ALCF), and the structured-extraction JSON schemas. Use to mine literature for a target protein — hotspots, modifications, existing binders — to seed a design campaign.
license: project-local skill text; distllm/faiss/datasets per their own licenses
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: build_faiss (distllm+faiss), rag_chat (distllm Retriever + LLM), fetch_pubmed
---

# Retrieval-Augmented Generation (literature RAG)

## Overview

Two phases:

1. **Build a store** — turn text records into a FAISS vector index. Each record
   needs a `text` field; records are chunked, embedded with a distllm
   encoder+pooler, and written as `faiss.index` (cosine / `IndexFlatIP`) plus a
   HuggingFace `dataset/` (Arrow) that carries the chunk text + metadata.
2. **Query the store** — a retriever pulls top-k chunks for a query; an LLM
   generates over them. Two modes: **freetext** (open synthesis → a string) and
   **structured extraction** (context-first prompt → a validated JSON array of
   typed records).

The store can be built from a curated **JSONL** you already have, or fetched
**on the fly** from the literature for a UniProt target. The store layout is
compatible with `distllm`'s `RetrieverConfig` (`faiss_index_v2`).

Pipeline role: RAG mines a target's literature for **hotspots, PTM sites, and
known binders**, which seed folding/diffusion campaigns (`chai-folding`,
`chroma-diffusion`).

---

## 1. Build a store

Records are dicts with at least `{'text': ...}` (plus optional metadata:
`pmid`, `title`, `doi`, `year`, `has_fulltext`, …).

```python
from rag_build import build_store, read_jsonl   # scripts/rag_build.py

records = read_jsonl('corpus.jsonl')             # [{'text': ...}, ...]
result = build_store(
    records,
    output_dir='rag_stores/literature',
    chunk_size=1500,      # chars per chunk
    chunk_overlap=200,    # carry-over between chunks
    batch_size=32,        # embedding batch
)
# result -> {'faiss_index_path', 'dataset_dir', 'ntotal', 'dim'}
```
On disk:
```
rag_stores/literature/
    faiss.index            # IndexFlatIP, L2-normalized (cosine)
    dataset/               # HF Dataset (Arrow): chunk text + metadata + embeddings
```
The default encoder is `sentence-transformers/all-MiniLM-L6-v2` with mean
pooling + normalization (384-dim). Override via `encoder_cfg`/`pooler_cfg` (see
`references/generators_reference.md`). Chunking breaks at paragraph/sentence
boundaries within the trailing window so chunks don't slice mid-sentence.

### Build from fetched literature

`scripts/fetch_literature_and_build.py` fetches PubMed abstracts (and, in the
full toolbelt, Europe PMC + Semantic Scholar + PMC full text) for a UniProt ID,
writes a curated JSONL, and builds the store:
```python
from fetch_pubmed_min import fetch_pubmed_abstracts   # scripts/fetch_pubmed_min.py
records = fetch_pubmed_abstracts('P21359', max_pmids=200)   # NF1, e.g.
build_store(records, 'rag_stores/NF1')
```
Set `NCBI_API_KEY` to raise the NCBI rate limit (3→10 req/s). The project's
richer `fetch_for_protein` (Europe PMC, Semantic Scholar, Unpaywall PDFs,
full-text) lives in the `bindify` toolbelt — see the reference.

---

## 2. Query — free-text synthesis

Assemble a config dict, build the RAG model, and generate. The generator backend
is OpenAI-compatible (`OpenAIAPIGeneratorConfig`), a vLLM server, or the ALCF
gateway — selected by the `_target_` key.

```python
from rag_query import build_rag, freetext        # scripts/rag_query.py

rag = build_rag(
    faiss_index_path='rag_stores/literature/faiss.index',
    dataset_dir='rag_stores/literature/dataset',
    generator={'_target_': 'OpenAIAPIGeneratorConfig',
               'model': 'gpt-4.1', 'api_key': '${env:OPENAI_API_KEY}'},
)
answer = freetext(rag, 'What residues on the target are implicated in binding?',
                  retrieval_top_k=200, retrieval_score_threshold=0.1)
print(answer)
```

---

## 3. Query — structured extraction

Structured modes return a **validated JSON array** of typed records. The prompt
puts retrieved context first, then the schema. Three query types:

| query_type | Extracts | Key fields |
|------------|----------|-----------|
| `protein_interactions` | interacting proteins + hotspots | `interactant_name`, `target_hotspot` (e.g. `["R47","K136"]`), `interaction_type`, `evidence` |
| `ptm_sites` | PTM sites + effects | `mod_residue` (e.g. `S127`), `post_translation_mod_type`, `mod_sequence` (glycan/ubiquitin linkage), `functional_effect` |
| `relevant_binders` | known binders/inhibitors | `binder_name`, `binder_type`, `sequence_or_smiles`, `binding_site`, `affinity`, `evidence` |

```python
from rag_query import structured

hits = structured(rag, 'Glycosylation sites on this protein', query_type='ptm_sites')
for h in hits:                    # list[dict], schema-validated where possible
    print(h['mod_residue'], h['post_translation_mod_type'], h.get('functional_effect'))
```
The extractor tolerates markdown fences, parses the JSON array, and keeps items
that fail validation as raw dicts (never drops evidence). Sequences found via
`relevant_binders` can seed a design campaign directly.

---

## Best Practices

- **Every record needs a non-empty `text`.** `read_jsonl` drops the rest.
- **Chunk size trades recall vs precision:** ~1500 chars / 200 overlap is a good
  default for abstracts+full text; smaller chunks sharpen retrieval, larger keep
  context.
- **Normalize embeddings** (default) so FAISS `IndexFlatIP` = cosine similarity.
- **Match the encoder at query time to the one used at build time** — different
  encoders/dims can't share an index.
- **Freetext for reading, structured for machine-consumable extraction.** Feed
  `relevant_binders` sequences / `protein_interactions` hotspots downstream.
- **Set `NCBI_API_KEY`** (and `S2_API_KEY` for the full fetcher) to avoid rate
  limits on large literature pulls.
- **Retrieval knobs:** raise `retrieval_top_k` (e.g. 200) for synthesis breadth;
  `retrieval_score_threshold` (~0.1) filters weak matches.
- **In-allocation vLLM endpoints** on an HSN address must bypass the ALCF proxy
  (`trust_env=False` / `no_proxy`) — the glue does this for internal hosts.

## Common Pitfalls

1. Records without `text` → silently skipped; empty store.
2. Building with one encoder and querying with another → dimension mismatch /
   garbage retrieval.
3. Forgetting to normalize → `IndexFlatIP` scores are dot products, not cosine.
4. No generator endpoint configured (missing `OPENAI_API_KEY`/base_url) → the
   store retrieves but generation errors.
5. Structured mode: assuming the LLM always returns clean JSON — the extractor
   strips fences and keeps unparseable items as raw dicts; check both.
6. Huge `max_pmids` with no NCBI key → throttled/slow; full-text upgrade is the
   dominant cost (one eutils round-trip per PMID).
7. Pointing the retriever at the `faiss.index` file but the wrong `dataset/` dir
   (or vice versa) → indices and texts misalign.

## Resources

### references/
- `generators_reference.md` — encoder/pooler configs, the four generator
  backends and their config dicts, the ChatAppConfig shape, and the structured
  extraction schemas.

### scripts/
- `rag_build.py` — self-contained store builder (chunk → embed → FAISS + HF ds).
- `fetch_pubmed_min.py` — minimal PubMed esearch+efetch → records (self-contained).
- `fetch_literature_and_build.py` — fetch a UniProt target's abstracts + build.
- `build_store_from_jsonl.py` — build a store from an existing JSONL.
- `rag_query.py` — build the RAG model; `freetext` + `structured` queries.
- `query_examples.py` — end-to-end freetext + all three structured query types.
