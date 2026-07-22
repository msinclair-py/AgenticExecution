# RAG stack reference

## Encoder / pooler config

The embedding model is a distllm encoder + pooler. Defaults (build & query MUST
match):

```python
ENCODER = {
    'name': 'auto',
    'pretrained_model_name_or_path': 'sentence-transformers/all-MiniLM-L6-v2',
    'tokenizer_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'half_precision': False, 'eval_mode': True,
    'compile_model': False, 'quantization': False,
}
POOLER = {'name': 'mean', 'normalize': True}     # normalize -> cosine via IndexFlatIP
```
Swap `pretrained_model_name_or_path` for a domain encoder (e.g. a PubMedBERT /
BioBERT sentence model) to improve biomedical retrieval — but re-build the store,
since the dimension/space changes. `half_precision`/`quantization` trade accuracy
for memory/speed on large corpora.

## Retriever config (distllm)

```python
retriever_config = {
    'encoder_config': ENCODER,
    'pooler_config': POOLER,
    'faiss_config': {
        'name': 'faiss_index_v2',
        'faiss_index_path': '<store>/faiss.index',
        'dataset_dir': '<store>/dataset',
        'search_algorithm': 'exact',       # IndexFlatIP exact search
    },
    'batch_size': 4,
}
# distllm.rag.search.RetrieverConfig(**retriever_config).get_retriever()
# retriever.search([query], top_k=..., score_threshold=...) -> (results, _)
#   results.total_indices[0]  -> chunk indices;  retriever.get_texts(idx) -> texts
```

## Generator backends (`_target_` selects one)

The project's `rag_chat` glue supports four generator configs, chosen by the
`_target_` key in the generator dict. `${env:VAR}` placeholders resolve from the
environment.

| `_target_` | Backend | Key fields |
|-----------|---------|-----------|
| `OpenAIAPIGeneratorConfig` | OpenAI / any OpenAI-compatible server | `model`, `api_key`, `base_url` (opt), `max_tokens` |
| `VLLMGeneratorConfig` | a vLLM server (`http://<server>.cels.anl.gov:<port>`) | `server`, `port`, `api_key`, `model` |
| `ArgoGeneratorConfig` | Argo proxy (OpenAI client to a local base_url) | `model`, `base_url`, `api_key` |
| `AlcfGeneratorConfig` | ALCF Sophia inference gateway (Globus token) | `model`, `base_url` |

`rag_query.py` here implements the OpenAI-compatible path (covers OpenAI, vLLM,
Argo via `base_url`). Example config:
```python
{'_target_': 'OpenAIAPIGeneratorConfig', 'model': 'gpt-4.1',
 'base_url': '${env:OPENAI_BASE_URL}', 'api_key': '${env:OPENAI_API_KEY}'}
```

**ALCF proxy note:** an in-allocation vLLM endpoint on an HSN address
(`*.hsn.cm.aurora.alcf.anl.gov`, or `localhost`) is unreachable through the ALCF
Squid proxy (→ HTTP 503). Bypass it for internal hosts (`httpx.Client(trust_env=
False)` / `no_proxy`); external endpoints keep the proxy.

## ChatAppConfig (project glue shape)

The bindify `RagTool` assembles this dict and hands it to `ChatAppConfig.from_dict`:
```python
{
  'rag_configs': {
      'generator_config': {...with _target_...},
      'retriever_config': {...as above...},
      'verbose': False,
  },
  'save_conversation_path': '/tmp/bindify_rag_conversation.json',
}
```
`ChatAppConfig.from_dict(...).rag_configs.get_rag_model()` returns a
`RagGenerator` whose `.generate(texts, prompt_template, retrieval_top_k,
retrieval_score_threshold, max_tokens, temperature, retrieval_texts)` runs
retrieve+generate. `rag_query.py` reproduces this without the bindify dependency.

## Structured extraction schemas

Three `query_type`s append a JSON-array schema to the query and validate the
result:
- **protein_interactions** → `interactant_name`, `interactant_uniprotid`,
  `target_hotspot` (residue list), `interaction_type`, `evidence`.
- **ptm_sites** → `mod_residue`, `post_translation_mod_type`, `mod_sequence`
  (glycan composition / ubiquitin linkage / SUMO isoform), `functional_effect`,
  `evidence`.
- **relevant_binders** → `binder_name`, `binder_type`, `sequence_or_smiles`,
  `binding_site`, `affinity`, `evidence`.

The extractor strips markdown fences, slices the `[ ... ]` array, and (in the
full toolbelt) validates each item with a Pydantic model, keeping unparseable
items as raw dicts so no evidence is lost.

## Fetching literature

`scripts/fetch_pubmed_min.py` — minimal esearch+efetch abstracts (self-contained).
The bindify toolbelt's `fetch_pubmed.fetch_for_protein(uniprot_id, max_pmids,
skip_esearch, query, fetch_fulltext, europe_pmc, semantic_scholar, s2_api_key,
pdf_dir, email)` is the richer path: UniProt-curated PMIDs + NCBI elink + gene
esearch, Europe PMC (preprints + OA full text), Semantic Scholar bulk search,
Unpaywall OA PDF download, and PMC full-text upgrade. Env: `NCBI_API_KEY`
(3→10 req/s), `S2_API_KEY`. Full-text upgrade dominates wall-clock (one eutils
round-trip per PMID) — use abstracts-only for breadth, full-text for a committed
target.
