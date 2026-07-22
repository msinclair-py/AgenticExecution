#!/usr/bin/env python
"""Launch AgenticExecution as a scaling FastMCP HTTP server on ALCF Polaris.

`python -m agentic_execution.mcp_server` builds a *local* Parsl config whose 'gpu'
executor is capped at ONE worker -- so on a Polaris node with 4 A100 GPUs it
would use a single GPU. This launcher composes the package's public pieces with
a scaling Parsl config attached, then serves the SAME single `execute_code` tool:

    HeterogeneousSettings.config_factory  ->  ServerConfig(parsl_config=...)
                                          ->  create_managed_server  ->  http

`HeterogeneousSettings` pins one HTEx 'gpu' worker per accelerator via Parsl's
`available_accelerators` (which sets CUDA_VISIBLE_DEVICES per worker), so with
`AGENTIC_EXECUTION_ACCEL=4` the runtime fans `execute_code` across all 4 GPUs on
every node in the allocation. GPU workers = nodes x 4.

It is intended to run ONCE on the head node inside a PBS job (see
`run_mcp_scaling.pbs`). Parsl owns the mpiexec fan-out over `$PBS_NODEFILE`; do
NOT wrap this in mpiexec yourself.

Env knobs (all optional; sensible Polaris defaults):

    AGENTIC_EXECUTION_NODES        nodes -> LocalProvider(nodes_per_block)   [count $PBS_NODEFILE]
    AGENTIC_EXECUTION_ACCEL        GPUs/node -> gpu workers/node             [4]
    AGENTIC_EXECUTION_THREADS      OMP threads/worker + cpu-executor workers [8]
    AGENTIC_EXECUTION_WORKER_INIT  shell run on each node before workers     [synthesized below]
    AGENTIC_EXECUTION_CUDA_MODULE  cuda module for synthesized worker_init   [cudatoolkit-standalone/12.4.1]
    AGENTIC_EXECUTION_OUTPUT       default workdir for snippets              [./execution_output]
    AGENTIC_EXECUTION_EXCHANGE     'local' or an academy exchange URL        [local]
    AGENTIC_EXECUTION_HOST         HTTP bind host                            [`hostname -i`]
    AGENTIC_EXECUTION_PORT         HTTP bind port                            [8000]
    AMBERHOME                      AmberTools root; if set, its bin is APPENDED to worker PATH
"""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from agentic_execution.mcp_server import ServerConfig, create_managed_server
from agentic_execution.utils.parsl_settings import HeterogeneousSettings


def _count_nodes() -> int:
    """Nodes in this allocation, from $PBS_NODEFILE (falls back to 1)."""
    nodefile = os.environ.get('PBS_NODEFILE')
    if nodefile and Path(nodefile).exists():
        with open(nodefile) as f:
            hosts = {line.strip() for line in f if line.strip()}
        if hosts:
            return len(hosts)
    return 1


def _routable_ip() -> str:
    """The node's routable HSN address, matching `hostname -i` on Polaris.

    The server must bind to this (not 127.0.0.1) to be reachable from other
    compute nodes in the job.
    """
    try:
        out = subprocess.run(
            ['hostname', '-i'], capture_output=True, text=True, check=True
        ).stdout.split()
        if out:
            return out[0]
    except Exception:
        pass
    return socket.gethostbyname(socket.gethostname())


def _default_worker_init() -> str:
    """Reproduce this login environment on every remote node's HTEx manager.

    `MpiExecLauncher` starts each node's manager in a fresh shell, so CUDA, the
    venv, AMBERHOME and the cwd must be re-established there or `import gpu4pyscf`
    (or the AMBER steps) fail only on the non-head nodes -- a confusing partial
    failure. Ordering matters; see the pitfalls in SKILL.md.
    """
    cuda_module = os.environ.get(
        'AGENTIC_EXECUTION_CUDA_MODULE', 'cudatoolkit-standalone/12.4.1'
    )
    venv = os.environ.get('VIRTUAL_ENV')
    amberhome = os.environ.get('AMBERHOME')
    cwd = os.getcwd()

    lines = [
        # AF_UNIX socket for the HTEx worker pool lives under TMPDIR; PBS sets a
        # ~60-char TMPDIR that overruns the 108-char AF_UNIX limit -> workers die
        # with "OSError: AF_UNIX path too long". Pin it short, FIRST.
        'export TMPDIR=/tmp',
        'module use /soft/modulefiles',
        f'module load {cuda_module}',
    ]
    if venv:
        # Activate the venv AFTER `module load` so its `python` wins on PATH.
        lines.append(f'source {venv}/bin/activate')
    if amberhome:
        # APPEND $AMBERHOME/bin, never prepend: AmberTools bundles its own python
        # (an unrelated 3.12) AND its own OpenMPI mpiexec. Prepending shadows the
        # venv python (-> ModuleNotFoundError: agentic_execution) and the Cray
        # PALS mpiexec that MpiExecLauncher needs for --cpu-bind. See SKILL.md.
        lines.append(f'export AMBERHOME={amberhome}')
        lines.append('export PATH="$PATH:$AMBERHOME/bin"')
    lines.append(f'cd {cwd}')
    return '\n'.join(lines)


def main() -> None:
    nodes = int(os.environ.get('AGENTIC_EXECUTION_NODES', _count_nodes()))
    accel = int(os.environ.get('AGENTIC_EXECUTION_ACCEL', 4))
    threads = int(os.environ.get('AGENTIC_EXECUTION_THREADS', 8))
    worker_init = os.environ.get('AGENTIC_EXECUTION_WORKER_INIT') or _default_worker_init()
    output = Path(os.environ.get('AGENTIC_EXECUTION_OUTPUT', './execution_output'))
    exchange = os.environ.get('AGENTIC_EXECUTION_EXCHANGE', 'local')
    host = os.environ.get('AGENTIC_EXECUTION_HOST') or _routable_ip()
    port = int(os.environ.get('AGENTIC_EXECUTION_PORT', 8000))

    output.mkdir(parents=True, exist_ok=True)

    parsl_config = HeterogeneousSettings(
        available_accelerators=accel,   # int 4 -> workers pinned to GPUs '0'..'3'
        nodes=nodes,
        worker_init=worker_init,
        max_workers_per_node=threads,   # cpu executor: one worker per OMP thread
        cores_per_worker=1.0,
    ).config_factory(output)

    print(
        f'[launcher] Parsl scaling: {nodes} node(s) x {accel} GPU worker(s) '
        f'= {nodes * accel} GPU workers'
    )
    print(f'[launcher] AgenticExecution MCP server -> http://{host}:{port}/mcp')

    server = create_managed_server(
        ServerConfig(
            output_path=output,
            num_threads=threads,
            exchange=exchange,
            parsl_config=parsl_config,
        )
    )
    server.run(transport='http', host=host, port=port)


if __name__ == '__main__':
    main()
