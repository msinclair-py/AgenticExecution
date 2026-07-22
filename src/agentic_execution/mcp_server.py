"""FastMCP server exposing the AgenticExecution code-execution layer to an MCP harness.

The academy ``AgenticExecution`` has exactly one action -- ``execute_code`` -- which runs
an LLM-authored Python snippet in a fresh subprocess on a Parsl worker. This
server exposes that single capability over MCP so any client (Claude Code,
Codex, ...) can drive bespoke QM/analysis code against the project environment
(rdkit, numpy, pyscf, gpu4pyscf, ambertools) with Parsl spreading the work
across the allocation's workers.

Nothing here imports pydantic-ai: an externally managed run never constructs an
orchestrator, needs no model configuration and no API key.

Configuration comes from the environment or flags, because an MCP client launches
the server as a subprocess and can only reach it through argv/env:

    AGENTIC_EXECUTION_OUTPUT   default working dir for snippets (default ./execution_output)
    AGENTIC_EXECUTION_THREADS  agent worker threads (default os.cpu_count())
    AGENTIC_EXECUTION_EXCHANGE 'local' or an academy exchange http(s) URL

The Parsl runtime the agent loads is chosen at construction: with no
``parsl_config`` the agent falls back to a local config, which caps the 'gpu'
executor at a single worker. To fan snippets across every GPU on every node of a
PBS allocation, build a scaling config from
``agentic_execution.utils.parsl_settings.HeterogeneousSettings`` and pass it in -- see
``scripts/polaris/launch_scaling_mcp.py``.

Examples
--------
    uv run python -m agentic_execution.mcp_server                 # stdio
    uv run python -m agentic_execution.mcp_server --transport http --port 8000

Registering with an MCP client (Claude Code's ``.mcp.json``)::

    {"mcpServers": {"agentic-execution": {
      "command": "uv",
      "args": ["run", "python", "-m", "agentic_execution.mcp_server"],
      "env": {"AGENTIC_EXECUTION_OUTPUT": "./execution_output"}}}}
"""

import argparse
import os
import sys
from academy.exchange import HttpExchangeFactory, LocalExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from parsl.config import Config
from pathlib import Path
from typing import Any

from .execution_layer import AgenticExecution
from .system_prompt import SYSTEM_PROMPT

SERVER_NAME = 'agentic-execution'


@dataclass(frozen=True)
class ServerConfig:
    """Everything fixed for the lifetime of one server process."""
    output_path: Path = Path('./execution_output')
    num_threads: int = 8
    exchange: str = 'local'
    # When None the agent loads a local Parsl config (gpu executor capped at one
    # worker). The scaling launcher passes a HeterogeneousSettings config here.
    parsl_config: Config | None = None


def _register(server: FastMCP, handle: Handle, output_path: Path) -> None:
    """Register ``execute_code`` as the server's sole tool.

    The tool forwards to the academy agent's ``execute_code`` action, so the
    snippet runs on a Parsl worker (out-of-process, isolated from crashes) and
    the result comes back as ``{stdout, stderr, returncode}``.
    """

    async def execute_code(
        code_snippet: str,
        workdir: str | None = None,
        extra_paths: list[str] | None = None,
        timeout: float = 300.0,
    ) -> dict[str, str]:
        """Execute an LLM-authored Python snippet on a Parsl worker.

        The snippet is written to a temp file and run in a fresh subprocess that
        shares the project environment (rdkit, numpy, pyscf, gpu4pyscf,
        ambertools). Running out-of-process isolates the server from hard
        crashes in the generated code and enforces the timeout. stdout/stderr are
        captured separately; a non-zero ``returncode`` (with the traceback on
        stderr) signals failure so the caller can revise and retry.

        Arguments:
            code_snippet: The Python source to execute.
            workdir: Directory to run in (created if missing). Defaults to the
                server's output directory so artifacts land in the run scope.
            extra_paths: Extra directories prepended to the subprocess
                PYTHONPATH (e.g. a skill's scripts/ dir) so the snippet can
                import them.
            timeout: Wall-clock seconds before the subprocess is killed.

        Returns:
            ``{'stdout': ..., 'stderr': ..., 'returncode': ...}``.
        """
        try:
            return await handle.action(
                'execute_code',
                code_snippet,
                workdir=Path(workdir) if workdir else output_path,
                extra_paths=[Path(p) for p in extra_paths] if extra_paths else None,
                timeout=timeout,
            )
        except Exception as exc:  # surface a correctable error to the client
            raise ToolError(str(exc)) from exc

    server.add_tool(execute_code)


def create_server(handle: Handle, *, output_path: Path = Path('./execution_output')) -> FastMCP:
    """A server bound to an already-running AgenticExecution handle.

    For embedding the code-execution tool in a process that owns its own
    ``AgenticExecution`` (the scaling launcher does this so it can attach a Parsl config),
    and for tests driving the server in-process against a stub handle.
    """
    server = FastMCP(name=SERVER_NAME, instructions=SYSTEM_PROMPT)
    _register(server, handle, output_path)
    return server


@asynccontextmanager
async def _managed_handle(config: ServerConfig) -> AsyncIterator[Handle]:
    """Launch an AgenticExecution instance for the life of the server and yield its handle."""
    config.output_path.mkdir(parents=True, exist_ok=True)
    factory = (
        LocalExchangeFactory() if config.exchange == 'local'
        else HttpExchangeFactory(config.exchange, auth_method='globus')
    )
    async with await Manager.from_exchange_factory(
        factory=factory,
        executors=ThreadPoolExecutor(),
    ) as manager:
        handle = await manager.launch(
            AgenticExecution(
                num_threads=config.num_threads,
                parsl_config=config.parsl_config,
            )
        )
        yield handle


def create_managed_server(config: ServerConfig) -> FastMCP:
    """A server that owns its AgenticExecution instance: the lifespan launches it and tears it down.

    The tool is registered inside the lifespan because it is bound to the agent
    handle, which does not exist until it is launched. Clients only list tools
    after startup, so they see it.
    """
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[Handle]:
        async with _managed_handle(config) as handle:
            _register(server, handle, config.output_path)
            yield handle

    return FastMCP(
        name=SERVER_NAME,
        instructions=SYSTEM_PROMPT,
        lifespan=lifespan,
        on_duplicate='replace',
    )


class _StdoutShim:
    """A ``sys.stdout`` replacement that splits the two ways stdout is used.

    On stdio transport the JSON-RPC channel *is* stdout, but the MCP transport
    reaches it only through ``sys.stdout.buffer`` (it wraps the binary buffer),
    whereas stray text output -- ``print``, a library banner -- goes through
    ``write``. So ``buffer`` stays the real client stream while every ``write``
    is diverted to stderr. Combined with an fd-level ``dup2`` of stderr onto
    fd 1 (for native/subprocess writers), nothing but framed protocol reaches
    the client.
    """

    def __init__(self, real_buffer: Any) -> None:
        self.buffer = real_buffer  # the MCP transport wraps this -> reaches client

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(sys.stderr, name)


def _isolate_stdout_for_stdio() -> None:
    """Keep every non-protocol writer in this process off the JSON-RPC channel.

    Snippets run out-of-process and academy/parsl are noisy at startup, so done
    once before the server runs (covering the lifespan launch, no per-call race):

    * ``dup(1)`` preserves the real client stdout for the transport;
    * ``dup2(2, 1)`` repoints fd 1 at stderr, so native code and subprocesses
      that write to the file descriptor land on stderr;
    * ``sys.stdout`` becomes a :class:`_StdoutShim` whose ``buffer`` is the
      preserved client stream but whose ``write`` goes to stderr.
    """
    real_stdout = os.fdopen(os.dup(1), 'wb')  # preserved BEFORE the dup2 below
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = _StdoutShim(real_stdout)


def _build_parser() -> argparse.ArgumentParser:
    """Flags default to the environment, so a client can configure the server
    either through ``env`` in its MCP config or through ``args``."""
    parser = argparse.ArgumentParser(
        prog='agentic-execution-mcp',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('AGENTIC_EXECUTION_OUTPUT', './execution_output')),
                        help='Default working directory for snippets (env AGENTIC_EXECUTION_OUTPUT).')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('AGENTIC_EXECUTION_THREADS', os.cpu_count() or 8)),
                        help='Agent worker threads (env AGENTIC_EXECUTION_THREADS).')
    parser.add_argument('--exchange', default=os.environ.get('AGENTIC_EXECUTION_EXCHANGE', 'local'),
                        help="academy exchange: 'local' or an http(s) URL (env AGENTIC_EXECUTION_EXCHANGE).")
    parser.add_argument('--transport', choices=('stdio', 'http'), default=os.environ.get('AGENTIC_EXECUTION_TRANSPORT', 'stdio'),
                        help='stdio for a client-spawned server; http to serve a remote one.')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host for --transport http.')
    parser.add_argument('--port', type=int, default=8000, help='Bind port for --transport http.')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = ServerConfig(
        output_path=args.output,
        num_threads=args.threads,
        exchange=args.exchange,
    )
    if args.transport == 'stdio':
        # Must happen before the server (and its lifespan) runs, so no parsl/
        # academy output can reach the protocol channel. http has a real stdout.
        _isolate_stdout_for_stdio()

    server = create_managed_server(config)
    if args.transport == 'stdio':
        server.run(transport='stdio', show_banner=False)
    else:
        server.run(transport='http', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
