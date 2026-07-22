"""MCP server: exposes exactly one tool (execute_code) bound to an agent handle.

Driven in-process against a stub handle via FastMCP's in-memory Client, so no
HTTP transport, academy runtime or parsl kernel is needed.
"""
import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from agentic_execution.mcp_server import create_server


class _StubHandle:
    """Records the action call and returns a canned execute_code result."""
    def __init__(self):
        self.calls = []

    async def action(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return {'stdout': '42\n', 'stderr': '', 'returncode': '0'}


def _run(coro):
    return asyncio.run(coro)


def test_exposes_only_execute_code():
    server = create_server(_StubHandle(), output_path=Path('/tmp/hv'))

    async def go():
        async with Client(server) as c:
            return [t.name for t in await c.list_tools()]

    assert _run(go()) == ['execute_code']


def test_execute_code_forwards_to_handle():
    handle = _StubHandle()
    server = create_server(handle, output_path=Path('/tmp/hv'))

    async def go():
        async with Client(server) as c:
            return await c.call_tool('execute_code', {'code_snippet': 'print(6*7)'})

    result = _run(go())
    assert result.structured_content == {'stdout': '42\n', 'stderr': '', 'returncode': '0'}
    name, args, kwargs = handle.calls[0]
    assert name == 'execute_code'
    assert args[0] == 'print(6*7)'


def test_workdir_defaults_to_output_path():
    handle = _StubHandle()
    out = Path('/tmp/hv_out')
    server = create_server(handle, output_path=out)

    async def go():
        async with Client(server) as c:
            await c.call_tool('execute_code', {'code_snippet': 'x=1'})

    _run(go())
    assert handle.calls[0][2]['workdir'] == out
