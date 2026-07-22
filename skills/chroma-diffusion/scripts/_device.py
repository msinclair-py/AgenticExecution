"""Device resolution for the chroma-diffusion scripts (xpu -> cuda -> cpu).

The reference ChromaDesigner hardcodes .xpu(); these scripts use .to(device)
with the string returned here so they run on any backend.
"""
from __future__ import annotations


def resolve_device() -> str:
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
