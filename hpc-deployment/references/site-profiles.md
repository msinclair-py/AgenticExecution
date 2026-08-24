# Site facts for HpcAgent payloads

Everything about *getting* an allocation -- node count, accelerators/node,
queue, walltime, worker environment -- is Globus Compute endpoint
configuration, set up once out of band on the HPC side
(`globus-compute-endpoint configure` / `start`). That's out of scope for this
skill: `scripts/mcp_server.py` only needs a running endpoint's exchange URL.

What's still relevant here is narrower: which environment variable a
`run_code` payload should read to find *which* accelerator this particular
agent process was pinned to, since that pinning is set by the endpoint before
the agent starts and the payload may need to act on it explicitly (e.g. to
pick a `torch` device rather than rely on there being exactly one visible).

## ALCF Polaris (NVIDIA A100)

| Fact | Value |
|---|---|
| Node | 4x A100 |
| Pinning var | `CUDA_VISIBLE_DEVICES` |

## ALCF Aurora (Intel Data Center GPU Max 1550)

| Fact | Value |
|---|---|
| Node | 6x Max 1550 = 12 tiles (12 addressable devices) |
| Pinning var | `ZE_AFFINITY_MASK` |
| Hierarchy | Endpoint must run with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` for `ZE_AFFINITY_MASK` to mean a plain tile index (`'0'..'11'`); under `COMPOSITE` it needs `'x.y'` pairs instead. This is an endpoint-config concern, but it changes what a payload should expect to parse out of the env var. |

## Writing a device-agnostic payload

A payload that resolves its device with an `xpu -> cuda -> cpu` fallback
(rather than hardcoding one backend) runs unchanged regardless of which site
the pool was launched on:

```python
import torch
if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch, 'xpu', None) and torch.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'
```

Do not install CUDA-only extras (`cupy-cuda12x`, `gpu4pyscf-cuda12x`, ...) into
an environment that might also run on Aurora -- they fail to install or shadow
a working CPU/XPU build. That environment is itself endpoint configuration,
not something this skill's scripts manage.
