"""Self-contained FAISS store builder (chunk -> embed -> FAISS + HF dataset).

Store layout (distllm RetrieverConfig 'faiss_index_v2' compatible):
    <output_dir>/faiss.index   IndexFlatIP, L2-normalized (cosine)
    <output_dir>/dataset/      HuggingFace Dataset (Arrow) with chunk text + metadata

Records are dicts with at least a 'text' field.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from datasets import Dataset

DEFAULT_ENCODER_CONFIG: dict[str, Any] = {
    'name': 'auto',
    'pretrained_model_name_or_path': 'sentence-transformers/all-MiniLM-L6-v2',
    'tokenizer_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'half_precision': False,
    'eval_mode': True,
    'compile_model': False,
    'quantization': False,
}
DEFAULT_POOLER_CONFIG: dict[str, Any] = {'name': 'mean', 'normalize': True}


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if overlap >= max_chars:
        raise ValueError('chunk overlap must be < chunk size')
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:                                    # break on a boundary
            window_start = max(end - 300, start + 1)
            for sep in ('\n\n', '\n', '. ', '! ', '? ', ' '):
                idx = text.rfind(sep, window_start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def expand_into_chunks(records, max_chars, overlap):
    out = []
    for r in records:
        chunks = chunk_text(r.get('text', ''), max_chars, overlap)
        if not chunks:
            continue
        base = {k: v for k, v in r.items() if k != 'text'}
        base.setdefault('pmcid', None)
        base.setdefault('has_fulltext', False)
        base.setdefault('doi', None)
        base.setdefault('source_db', 'MED')
        for i, chunk in enumerate(chunks):
            cr = dict(base)
            cr.update(text=chunk, chunk_index=i, num_chunks=len(chunks))
            out.append(cr)
    return out


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get('text'):
                    out.append(r)
    return out


def write_jsonl(records, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        for r in records:
            r = {k: v for k, v in r.items() if k != 'embeddings'}
            fh.write(json.dumps(r) + '\n')
    return str(path)


def _get_models(encoder_cfg, pooler_cfg):
    try:
        from distllm.embed.encoders import get_encoder
        from distllm.embed.poolers import get_pooler
    except ImportError:
        from distllm.embed import get_encoder, get_pooler  # type: ignore
    return get_encoder(encoder_cfg), get_pooler(pooler_cfg)


def _embed_batch(encoder, pooler, texts) -> np.ndarray:
    import torch
    batch = encoder.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    batch = {k: v.to(encoder.device) for k, v in batch.items()}
    with torch.no_grad():
        token_emb = encoder.encode(batch)
        pooled = pooler.pool(token_emb, batch['attention_mask'])
    return pooled.detach().cpu().numpy().astype('float32')


def build_store(records, output_dir, encoder_cfg=None, pooler_cfg=None,
                batch_size=32, chunk_size=1500, chunk_overlap=200) -> dict:
    output_dir = Path(output_dir)
    encoder_cfg = encoder_cfg or DEFAULT_ENCODER_CONFIG
    pooler_cfg = pooler_cfg or DEFAULT_POOLER_CONFIG

    records = expand_into_chunks(records, chunk_size, chunk_overlap)
    if not records:
        raise ValueError('No non-empty text records to embed.')

    encoder, pooler = _get_models(encoder_cfg, pooler_cfg)
    vectors = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        vecs = _embed_batch(encoder, pooler, [r['text'] for r in batch])
        if pooler_cfg.get('normalize'):
            faiss.normalize_L2(vecs)
        vectors.append(vecs)

    matrix = np.vstack(vectors).astype('float32')
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = output_dir / 'faiss.index'
    faiss.write_index(index, str(faiss_path))

    for r, v in zip(records, matrix):
        r['embeddings'] = v.tolist()
    ds = Dataset.from_list(records)
    dataset_dir = output_dir / 'dataset'
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    ds.save_to_disk(str(dataset_dir))

    return {'faiss_index_path': str(faiss_path), 'dataset_dir': str(dataset_dir),
            'ntotal': index.ntotal, 'dim': dim}
