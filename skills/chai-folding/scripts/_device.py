"""Device resolution shared by the chai-folding scripts.

Device-agnostic: prefers an Intel XPU (Aurora), then CUDA (NVIDIA), then CPU.
Chai-1's run_inference takes the resulting string directly as its `device` arg.
"""
from __future__ import annotations


def resolve_device() -> str:
    """Return the best available torch device string: 'xpu' | 'cuda' | 'cpu'."""
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
