"""Device resolution for the proteinmpnn scripts (xpu -> cuda -> cpu)."""
from __future__ import annotations


def resolve_device() -> str:
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
