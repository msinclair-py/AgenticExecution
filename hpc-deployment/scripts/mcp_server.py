"""Template: the ONLY entry point this pattern has.

Run this LOCALLY, alongside the harness -- on your laptop, not on the HPC and
not inside any allocation. It talks to a Globus Compute-backed academy
exchange to launch and maintain a pool of HpcAgent processes
(hpc_agent.py) and exposes exactly one MCP tool: run_code. There is no
head-node launcher and no batch script in this pattern -- getting compute is
a Globus Compute endpoint call over HTTPS, not an ssh session, and there is
never a `qsub` anywhere in this flow.

Precondition: a running Globus Compute endpoint. Its accelerators/node,
queue, walltime, and worker environment are configured once, out of band, on
the HPC side (`globus-compute-endpoint configure` / `start`) -- that's out of
scope for this skill. You need only its exchange URL and enough of an
accelerator count to size HPC_POOL_SIZE.
"""
import itertools
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from academy.exchange import HttpExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from hpc_agent import HpcAgent
from pool_manager import PoolManager

SYSTEM_PROMPT = (
    'You have a run_code tool that executes a Python snippet on a GPU-pinned '
    'HPC worker and returns its stdout/stderr/returncode. It is the only '
    'tool this server exposes -- anything else the work needs (another '
    'library, a call out to some other program) must happen inside the '
    'snippet itself.'
)

EXCHANGE_URL = os.environ['ACADEMY_EXCHANGE_URL']      # Globus Compute-backed academy exchange
POOL_NAME = os.environ.get('HPC_POOL_NAME', 'default')
POOL_SIZE = int(os.environ.get('HPC_POOL_SIZE', '4'))  # match the endpoint's accelerator count
BASE_DIR = os.environ.get('HPC_BASE_DIR', os.getcwd()) # distinguishes this pool from another
                                                        # campaign's, if they'd otherwise share
                                                        # an endpoint -- see adopt_live_agents


def create_server() -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP):
        factory = HttpExchangeFactory(EXCHANGE_URL, auth_method='globus')
        async with await Manager.from_exchange_factory(
            factory=factory, executors=ThreadPoolExecutor(),
        ) as manager:
            pool = PoolManager(manager, HpcAgent)
            dispatch = itertools.count()

            async def launch_one() -> Handle:
                handle = await manager.launch(HpcAgent())
                # REBIND, do not return `handle` directly. A Handle resolves
                # its ExchangeClient through a ContextVar, and contextvars
                # propagate parent->child only at task creation. The FastMCP
                # tool call below runs in a DIFFERENT asyncio task than this
                # lifespan, so an unrebound handle dies with
                # ExchangeClientNotFoundError on its first call.
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def run_code(payload: str, workdir: str | None = None,
                               timeout: float = 300.0) -> dict[str, str]:
                """Run a Python snippet on a GPU-pinned HPC worker. Returns
                {stdout, stderr, returncode}. A nonzero returncode means the
                snippet failed -- read stderr and retry with a fix; it is not
                a tool error."""
                handles = await pool.ensure_pool(POOL_NAME, BASE_DIR, POOL_SIZE, launch_one)
                if not handles:
                    raise ToolError('no live HPC workers in the pool')
                handle = handles[next(dispatch) % len(handles)]
                try:
                    return await handle.action('run_code', payload,
                                               workdir=workdir, timeout=timeout)
                except Exception as exc:
                    raise ToolError(str(exc)) from exc

            server.add_tool(run_code)
            yield

    return FastMCP(name='hpc-agent', instructions=SYSTEM_PROMPT,
                   lifespan=lifespan, on_duplicate='replace')


if __name__ == '__main__':
    create_server().run(transport='http', host='127.0.0.1',
                        port=int(os.environ.get('HPC_PORT', 8000)))
