"""End-to-end RAG queries: freetext synthesis + all three structured extractions.

Usage:
    OPENAI_API_KEY=... python query_examples.py <faiss.index> <dataset_dir>
Optionally set OPENAI_BASE_URL / OPENAI_MODEL for a non-default endpoint.
"""
from __future__ import annotations

import sys

from rag_query import build_rag, freetext, structured


def main(faiss_index: str, dataset_dir: str) -> None:
    rag = build_rag(
        faiss_index_path=faiss_index,
        dataset_dir=dataset_dir,
        generator={'_target_': 'OpenAIAPIGeneratorConfig',
                   'api_key': '${env:OPENAI_API_KEY}'},
    )

    print('=== freetext ===')
    print(freetext(rag, 'Summarize what is known about this protein and its binding partners.'))

    print('\n=== protein_interactions ===')
    for h in structured(rag, 'Interacting proteins and their hotspot residues',
                        query_type='protein_interactions'):
        print(h.get('interactant_name'), h.get('target_hotspot'))

    print('\n=== ptm_sites ===')
    for h in structured(rag, 'Post-translational modification sites',
                        query_type='ptm_sites'):
        print(h.get('mod_residue'), h.get('post_translation_mod_type'),
              h.get('functional_effect'))

    print('\n=== relevant_binders ===')
    for h in structured(rag, 'Known binders, inhibitors, or antibodies',
                        query_type='relevant_binders'):
        print(h.get('binder_name'), h.get('binder_type'), h.get('sequence_or_smiles'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
