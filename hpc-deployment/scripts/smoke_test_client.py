"""Smoke-test a running hpc-deployment MCP server.

This client deliberately validates one normal code result and one bounded
filesystem result. Run it only after the endpoint allocation/agent pool is up.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from fastmcp import Client


async def main() -> None:
    url = os.environ.get("HPC_MCP_URL")
    if not url:
        raise RuntimeError("set HPC_MCP_URL, for example http://127.0.0.1:8000/mcp")
    async with Client(url, timeout=180) as client:
        names = {tool.name for tool in await client.list_tools()}
        expected = {"run_code", "inspect_filesystem"}
        if names != expected:
            raise RuntimeError(f"unexpected tools: {names}; expected {expected}")
        probe = """import json, os, socket\nprint(json.dumps({'host': socket.gethostname(), 'campaign': os.environ.get('HPC_CAMPAIGN_ID'), 'pin': os.environ.get('ZE_AFFINITY_MASK', os.environ.get('CUDA_VISIBLE_DEVICES'))}))"""
        code = await client.call_tool("run_code", {"payload": probe})
        print("RUN_CODE", code.content[0].text)
        root = os.environ["HPC_BASE_DIR"]
        listing = await client.call_tool("inspect_filesystem", {"operation": "ls", "path": root})
        print("INSPECT", listing.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
