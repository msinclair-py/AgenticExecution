"""Device resolution for the protein-language-model-design scripts.

resolve_device() -> torch device string ('xpu' | 'cuda' | 'cpu')

Intel/Aurora XPU is preferred, then NVIDIA CUDA (Polaris/Lambda), else CPU.
ESM-2 inference runs on any of the three; a 650M model is comfortable on one
GPU/tile and workable (slower) on CPU.
"""
from __future__ import annotations


def resolve_device() -> str:
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
