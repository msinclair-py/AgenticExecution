---
name: rag-query
description: >-
  Ask questions of a vector store built by rag-build-index — retrieval-augmented
  answers over a corpus. Embed the query with all-MiniLM-L6-v2, search the FAISS
  index, and pass the top passages to an LLM chat-completion. Use to mine a
  target's hotspots / PTM sites / known interactions from a literature store, or
  for free-text Q&A. Supports FREETEXT answers and STRUCTURED JSON extraction.
  Needs faiss + datasets + sentence-transformers AND a configured
  OpenAI-compatible chat endpoint.
---

# RAG: query (faiss retrieval + OpenAI-compatible LLM)

Answer a question against a persisted store built by the rag-build-index skill.
The query is embedded with the **same** `all-MiniLM-L6-v2` encoder, searched
against the `faiss.IndexFlatIP`, the top-k hit texts are pulled from the on-disk
HuggingFace dataset, assembled into a context-first prompt, and sent to an
OpenAI-compatible chat-completions endpoint for a single grounded answer.
Cost/fidelity: retrieval is exact and cheap; answer quality depends on the LLM
and on whether the corpus actually contains the answer.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run the retrieval + LLM
call yourself. NEVER import bindify.

## When to use it

- You have a store from rag-build-index and want to **mine facts** from it —
  e.g. a target's binding hotspots, PTM sites, or known protein interactions.
- You want free-text Q&A grounded in the corpus, or structured records extracted
  from it (`protein_interactions`, `ptm_sites`, `relevant_binders`, …).

Do **not** use it before a store exists — build one first with rag-build-index.

## Setup (gate before installing)

Needs the same retrieval stack **and** a reachable LLM endpoint.

```bash
python -c "import faiss, datasets, sentence_transformers" 2>/dev/null && echo ready || pip install faiss-cpu datasets sentence-transformers
```

**Be honest about the endpoint: it must be configured, this skill cannot invent
one.** Read the base URL from the env var `BINDIFY_LLM_BASE_URL` (default model
e.g. `Qwen/Qwen2.5-32B-Instruct`). It may be a hosted OpenAI-compatible API, a
local vLLM server (`requests.post` to `http://<server>/v1/chat/completions`), or
an ALCF endpoint. If `BINDIFY_LLM_BASE_URL` is unset, stop and ask for it rather
than guessing. Confirm reachability:

```bash
echo "${BINDIFY_LLM_BASE_URL:?set BINDIFY_LLM_BASE_URL to your /v1 chat endpoint}"
```

## Procedure

Retrieve, assemble context, ask once:

- **Embed** the query with `all-MiniLM-L6-v2`, `normalize_embeddings=True` (must
  match the encoder used at index time).
- **Search** `index.search(qvec, top_k)`; optionally drop hits below a
  `score_threshold` (inner-product / cosine score). Map each returned row index
  back to its dataset record to recover the text + metadata.
- **Assemble** a context-first prompt: the retrieved passages, then the question.
- **Ask** with a single `requests.post` to the endpoint's
  `/v1/chat/completions`.
- **FREETEXT** → return the answer string.
- **STRUCTURED** (`query_type` like `protein_interactions`, `ptm_sites`,
  `relevant_binders`) → append a JSON-schema instruction to the prompt, then
  regex-extract the first JSON array from the response and validate each item;
  return the list of validated records.

Inputs: `query`, store paths (the `faiss.index` + `dataset/` from
rag-build-index), `top_k`, `score_threshold`, `query_type`. Output: an answer
string OR a list of validated structured records.

```python
import os, re, json, requests, faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

STORE   = "/path/to/store"          # dir holding faiss.index + dataset/
QUERY   = "What residues are known hotspots on <target>?"
TOP_K   = 5
BASE    = os.environ["BINDIFY_LLM_BASE_URL"].rstrip("/")   # .../v1
MODEL   = os.environ.get("BINDIFY_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index   = faiss.read_index(f"{STORE}/faiss.index")
ds      = load_from_disk(f"{STORE}/dataset")

qvec = encoder.encode([QUERY], normalize_embeddings=True, convert_to_numpy=True)
scores, idxs = index.search(qvec, TOP_K)
passages = [ds[int(i)]["text"] for i in idxs[0] if i >= 0]

context = "\n\n".join(f"[{n+1}] {p}" for n, p in enumerate(passages))
prompt = (f"Use only the context to answer.\n\nContext:\n{context}\n\n"
          f"Question: {QUERY}\nAnswer:")
# For STRUCTURED extraction, append e.g.:
#   "Return a JSON array of objects with keys ... and nothing else."

resp = requests.post(
    f"{BASE}/chat/completions",
    json={"model": MODEL,
          "messages": [{"role": "user", "content": prompt}],
          "temperature": 0.0},
    timeout=120,
)
answer = resp.json()["choices"][0]["message"]["content"]

# FREETEXT: use `answer` as-is.
# STRUCTURED: pull the first JSON array and validate each item.
m = re.search(r"\[.*\]", answer, re.S)
records = json.loads(m.group(0)) if m else []
print(json.dumps({"answer": answer, "records": records}))
```

## Interpreting the result

- If retrieved `scores` are all low (below your `score_threshold`), the corpus
  likely does not cover the question — treat the answer with suspicion or widen
  the corpus rather than trusting a hallucinated reply.
- STRUCTURED extraction can return an empty list if the model did not emit a
  clean JSON array; loosen the prompt or re-ask. Always validate the extracted
  records before using them downstream.
- Answers are grounded only in the retrieved passages, not the whole corpus —
  raise `top_k` if the answer needs broader context.

## Notes

- Use the same encoder and normalization as the index; a mismatch silently
  degrades retrieval.
- Deciding which questions to ask, what `query_type`/schema to extract, and how
  to act on the mined hotspots/interactions is your job as the agent. This skill
  only runs one retrieval-augmented query.
- Upstream: rag-build-index (builds the store) and fetch-literature (gathers the
  corpus).
