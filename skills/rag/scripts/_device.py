"""Device note for the rag scripts.

Embedding (distllm encoder) runs on whatever device the encoder config selects;
generation is a remote LLM call, so no local GPU is needed for querying. This
helper is provided for consistency with the other skills.
"""
from __future__ import annotations


def resolve_device() -> str:
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
