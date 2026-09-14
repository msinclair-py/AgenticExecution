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
    packages = {name: bool(importlib.util.find_spec(name)) for name in ("academy", "fastmcp", "globus_compute_sdk")}
    if not all(packages.values()):
        print(f"missing local packages: {packages}", file=sys.stderr)
        return 2
    base = Path(os.environ["HPC_BASE_DIR"])
    print(json.dumps({
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
            "worker sets one unique accelerator pin per agent",
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
