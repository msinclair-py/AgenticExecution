"""Query a FAISS RAG store: retrieve top-k chunks, generate over them.

Self-contained: distllm Retriever for retrieval + an OpenAI-compatible client
for generation. Two modes: freetext (synthesis string) and structured
(validated JSON array). Mirrors the project's rag_chat glue without depending on
bindify.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from rag_build import DEFAULT_ENCODER_CONFIG, DEFAULT_POOLER_CONFIG

# ── Structured extraction schemas (query_type -> instruction) ───────────
STRUCTURED_SCHEMAS: dict[str, str] = {
    'protein_interactions': (
        'Extract all reported protein-protein interactions. Each JSON element:\n'
        '{"interactant_name": "...", "interactant_uniprotid": "... or null", '
        '"target_hotspot": ["R47","K136"], '
        '"interaction_type": "direct|scaffold|substrate|inhibitor|activator|complex|other", '
        '"evidence": "1-2 sentence quote"}\nReturn [] if none.'
    ),
    'ptm_sites': (
        'Extract all reported post-translational modification sites. Each JSON element:\n'
        '{"mod_residue": "S127", '
        '"post_translation_mod_type": "phosphorylation|ubiquitination|acetylation|glycosylation|sumoylation|methylation|other", '
        '"mod_sequence": "glycan/ubiquitin-linkage/isoform or null", '
        '"functional_effect": "activating|inhibitory|stability|localization|null", '
        '"evidence": "1-2 sentence quote"}\nReturn [] if none.'
    ),
    'relevant_binders': (
        'Extract all reported binders, inhibitors, or binding agents. Each JSON element:\n'
        '{"binder_name": "...", '
        '"binder_type": "peptide|small_molecule|antibody|protein|nanobody|aptamer|other", '
        '"sequence_or_smiles": "... or null", "binding_site": "... or null", '
        '"affinity": "Kd=100nM or null", "evidence": "1-2 sentence quote"}\nReturn [] if none.'
    ),
}


def _structured_prompt(task: str, contexts: list[str]) -> str:
    ctx = '\n---\n'.join(contexts) if contexts else '(no context retrieved)'
    return (
        f'[Retrieved Literature]\n{ctx}\n\n[Extraction Task]\n{task}\n\n'
        'Use the retrieved literature above as your primary source. '
        'Return ONLY a valid JSON array matching the schema above. '
        'No text or markdown outside the JSON.'
    )


def _conversation_prompt(query: str, contexts: list[str]) -> str:
    p = f'User: {query}\nAssistant:'
    if contexts:
        p += '\n\n[Context from retrieval]\n' + '\n'.join(contexts)
    return p


# ── Generator (OpenAI-compatible) ───────────────────────────────────────
def _resolve_env(v: Any) -> Any:
    if isinstance(v, str) and v.startswith('${env:') and v.endswith('}'):
        return os.getenv(v[6:-1], '')
    return v


def _make_client(generator: dict):
    import openai
    g = {k: _resolve_env(v) for k, v in generator.items() if k != '_target_'}
    model = g.get('model', os.getenv('OPENAI_MODEL', 'gpt-4.1'))
    kwargs: dict = {'api_key': g.get('api_key') or os.getenv('OPENAI_API_KEY')}
    base_url = g.get('base_url') or os.getenv('OPENAI_BASE_URL')
    if base_url:
        kwargs['base_url'] = base_url
    return openai.OpenAI(**kwargs), model


@dataclass
class Rag:
    retriever: Any
    client: Any
    model: str

    def _retrieve(self, query: str, top_k: int, threshold: float) -> list[str]:
        results, _ = self.retriever.search([query], top_k=top_k,
                                           score_threshold=threshold)
        idx = results.total_indices[0]
        return list(self.retriever.get_texts(idx))

    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                      {'role': 'user', 'content': prompt}],
            temperature=temperature, max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ''


def build_rag(faiss_index_path: str, dataset_dir: str, generator: dict,
              encoder_cfg: dict | None = None, pooler_cfg: dict | None = None) -> Rag:
    from distllm.rag.search import RetrieverConfig

    retriever_cfg = RetrieverConfig(
        encoder_config=encoder_cfg or DEFAULT_ENCODER_CONFIG,
        pooler_config=pooler_cfg or DEFAULT_POOLER_CONFIG,
        faiss_config={'name': 'faiss_index_v2', 'faiss_index_path': faiss_index_path,
                      'dataset_dir': dataset_dir, 'search_algorithm': 'exact'},
        batch_size=4,
    )
    retriever = retriever_cfg.get_retriever()
    client, model = _make_client(generator)
    return Rag(retriever=retriever, client=client, model=model)


def freetext(rag: Rag, query: str, retrieval_top_k: int = 200,
             retrieval_score_threshold: float = 0.1, max_tokens: int = 16384,
             temperature: float = 0.0) -> str:
    contexts = rag._retrieve(query, retrieval_top_k, retrieval_score_threshold)
    return rag._generate(_conversation_prompt(query, contexts), max_tokens, temperature)


def structured(rag: Rag, query: str, query_type: str, retrieval_top_k: int = 200,
               retrieval_score_threshold: float = 0.1, max_tokens: int = 16384) -> list[dict]:
    if query_type not in STRUCTURED_SCHEMAS:
        raise ValueError(f'Unknown query_type {query_type!r}. '
                         f'Choices: {list(STRUCTURED_SCHEMAS)}')
    contexts = rag._retrieve(query, retrieval_top_k, retrieval_score_threshold)
    task = f'{query}\n\n{STRUCTURED_SCHEMAS[query_type]}'
    response = rag._generate(_structured_prompt(task, contexts), max_tokens, 0.0)

    text = re.sub(r'```(?:json)?\n?', '', response).strip().rstrip('`')
    start, end = text.find('['), text.rfind(']') + 1
    if start == -1 or end <= 0:
        return []
    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    return [item for item in raw if isinstance(item, dict)]
