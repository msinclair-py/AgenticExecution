"""Preflight validation for hpc-deployment local and worker bootstrap inputs.

Run locally before mcp_server.py. It does not submit a scheduler job.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REQUIRED = ["HPC_GLOBUS_ENDPOINT_ID", "HPC_POOL_SIZE", "HPC_ENDPOINT_CONFIG_JSON", "HPC_BASE_DIR", "HPC_CAMPAIGN_ID", "ACADEMY_EXCHANGE_URL"]


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        print("missing:", ", ".join(missing), file=sys.stderr)
        return 2
    try:
        config = json.loads(os.environ["HPC_ENDPOINT_CONFIG_JSON"])
        assert isinstance(config, dict)
        assert int(os.environ["HPC_POOL_SIZE"]) > 0
    except Exception as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    execution = os.environ.get("HPC_EXECUTION", "direct")
    if execution not in ("direct", "cluster"):
        print(f"HPC_EXECUTION must be 'direct' or 'cluster': {execution!r}", file=sys.stderr)
        return 2
    if execution == "cluster":
        # Cluster mode has two couplings the server cannot discover for itself:
        # the EL policy's leaf count must match the nodes PBS actually
        # allocates, and the system name selects the GPU-pinning variable.
        system = os.environ.get("HPC_SYSTEM", "")
        if system not in ("aurora", "polaris"):
            print(f"cluster mode needs HPC_SYSTEM aurora|polaris: {system!r}", file=sys.stderr)
            return 2
        try:
            nnodes = int(os.environ["HPC_NNODES"])
            assert nnodes > 0
        except Exception as exc:
            print(f"cluster mode needs a positive HPC_NNODES: {exc}", file=sys.stderr)
            return 2
        requested = config.get("nodes_per_block")
        if requested is not None and int(requested) != nnodes:
            print(f"HPC_NNODES={nnodes} disagrees with endpoint nodes_per_block={requested}: "
                  "EL's leaf count would exceed the allocated nodes", file=sys.stderr)
            return 2
    packages = {name: bool(importlib.util.find_spec(name)) for name in ("academy", "fastmcp", "globus_compute_sdk")}
    if not all(packages.values()):
        print(f"missing local packages: {packages}", file=sys.stderr)
        return 2
    base = Path(os.environ["HPC_BASE_DIR"])
    print(json.dumps({
        "execution": execution,
        "local_dependencies": packages,
        "endpoint_id": os.environ["HPC_GLOBUS_ENDPOINT_ID"],
        "campaign_id": os.environ["HPC_CAMPAIGN_ID"],
        "pool_size": int(os.environ["HPC_POOL_SIZE"]),
        "base_dir": str(base),
        "endpoint_config": config,
        "remote_requirements": [
            "worker environment installs academy-py and compatible dependencies",
            "worker can import hpc_agent and pool_manager, via staged module/PYTHONPATH or approved serializer",
            "worker sets HPC_BASE_DIR and HPC_CAMPAIGN_ID",
        ] + ([
            # ensemble_launcher is imported on the WORKER, not here: a local
            # find_spec would pass on a laptop and still fail at agent startup.
            "worker environment installs ensemble_launcher (cluster mode only)",
            "one endpoint worker slot per client agent PLUS one for the long-lived "
            "HpcClusterAgent, which holds its slot for the life of the allocation",
            "no per-agent accelerator pin: EL owns GPU placement in cluster mode",
        ] if execution == "cluster" else [
            "worker sets one unique accelerator pin per agent",
        ]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
