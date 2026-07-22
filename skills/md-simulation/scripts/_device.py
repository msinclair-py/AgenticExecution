"""Device / OpenMM-platform resolution for the md-simulation scripts.

resolve_device()      -> torch device string ('xpu' | 'cuda' | 'cpu')
resolve_md_platform() -> OpenMM platform name ('OpenCL' | 'CUDA' | 'CPU')

The mapping: Intel/Aurora XPU -> OpenMM 'OpenCL'; NVIDIA -> 'CUDA'; else 'CPU'.
"""
from __future__ import annotations


def resolve_device() -> str:
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def resolve_md_platform() -> str:
    return {'xpu': 'OpenCL', 'cuda': 'CUDA'}.get(resolve_device(), 'CPU')
