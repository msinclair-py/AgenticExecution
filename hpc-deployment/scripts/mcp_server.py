"""Local FastMCP server: GCExecutor → PBSPro allocation → pinned Academy agents.

Required environment:
  HPC_GLOBUS_ENDPOINT_ID, HPC_POOL_SIZE, HPC_ENDPOINT_CONFIG_JSON,
  HPC_BASE_DIR, HPC_CAMPAIGN_ID, ACADEMY_EXCHANGE_URL

Run locally with the directory containing this file on PYTHONPATH. The endpoint
must make this directory/module importable on workers, or use an endpoint-
approved Globus Compute code-shipping serializer.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

from academy.exchange import HttpExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from globus_compute_sdk import Executor
from hpc_agent import HpcAgent, HpcClientAgent, HpcClusterAgent
from pool_manager import CampaignPool, ClusterPool


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


ENDPOINT_ID = required("HPC_GLOBUS_ENDPOINT_ID")
POOL_SIZE = int(required("HPC_POOL_SIZE"))
BASE_DIR = required("HPC_BASE_DIR")
CAMPAIGN_ID = required("HPC_CAMPAIGN_ID")
EXCHANGE_URL = required("ACADEMY_EXCHANGE_URL")
ENDPOINT_CONFIG = json.loads(required("HPC_ENDPOINT_CONFIG_JSON"))
if not isinstance(ENDPOINT_CONFIG, dict):
    raise RuntimeError("HPC_ENDPOINT_CONFIG_JSON must encode an object")
if POOL_SIZE < 1:
    raise RuntimeError("HPC_POOL_SIZE must be positive")
INSPECT_ROOTS = [item for item in os.environ.get("HPC_INSPECT_ROOTS", "").split(":") if item]
# Which execution style this server serves. `direct` keeps one HpcAgent per
# pinned GPU/tile, each running its own payload: no Ensemble Launcher, so
# ensemble_launcher need not be installed on the worker. `cluster` runs one
# HpcClusterAgent holding an EL cluster plus HPC_POOL_SIZE HpcClientAgents that
# submit into it, with EL owning placement and run_batch available. Default
# `direct`: it is the mode this skill shipped with, and it has the smaller
# worker-side dependency surface.
EXECUTION = os.environ.get("HPC_EXECUTION", "direct")
if EXECUTION not in ("direct", "cluster"):
    raise RuntimeError("HPC_EXECUTION must be 'direct' or 'cluster'")
# Selects the EL SystemConfig and the GPU-pinning variable the cluster agent
# hands to tasks: ZE_AFFINITY_MASK on Aurora, CUDA_VISIBLE_DEVICES on Polaris.
SYSTEM = os.environ.get("HPC_SYSTEM", "polaris")
if SYSTEM not in ("aurora", "polaris"):
    raise RuntimeError("HPC_SYSTEM must be 'aurora' or 'polaris'")


# The GC executor is the only allocation launcher. Passing the user endpoint
# config here causes the endpoint's Parsl PBSProProvider to render the PBS job.
def make_executor() -> Executor:
    return Executor(endpoint_id=ENDPOINT_ID, user_endpoint_config=ENDPOINT_CONFIG)


def create_server(cluster:bool=False,ckpt_dir=None, nnodes:int | None = None) -> FastMCP:
    # ckpt_dir only seeds the first cluster generation; ClusterPool mints a
    # fresh one for every rebuild so a new ClusterClient can never bind to the
    # dead master's checkpoint record.
    if cluster:
        assert nnodes is not None, "create server needs nnodes when using a EL cluster"

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        # Values are serialized into every agent constructor; worker-side setup
        # must make BASE_DIR available. Do not mkdir it locally: it may be a
        # remote-only HPC filesystem path.
        factory = HttpExchangeFactory(EXCHANGE_URL, auth_method="globus")
        gc_executor = make_executor()
        served = False
        try:
            async with _serve(factory, gc_executor, server):
                served = True
                yield
        except Exception:  # noqa: BLE001 - shutdown must finish even if an agent died badly
            if not served:
                raise  # a startup failure must still fail the server loudly
            # Manager.close() asks every agent to shut down first, then awaits
            # each one and re-raises the first failure. After a cluster rebuild
            # the previous generation's agents always fail (ManagerLost at
            # walltime), so without this the exit skipped the executor shutdown
            # below and its non-daemon threads kept the process alive forever.
            logging.getLogger(__name__).exception("agent failures during shutdown (agents were already asked to stop)")
        finally:
            gc_executor.shutdown(wait=False, cancel_futures=True)

    @asynccontextmanager
    async def _serve(factory, gc_executor, server: FastMCP):
        async with await Manager.from_exchange_factory(factory=factory, executors={"globus": gc_executor}, default_executor="globus") as manager:
            # In cluster mode EL owns placement and fanout, so the pool is a
            # fixed set of interchangeable submitters built once at startup;
            # the client agents are unpinned by design. Direct mode keeps
            # CampaignPool, which pins one accelerator per agent.
            pool = ClusterPool(CAMPAIGN_ID, POOL_SIZE, ckpt_dir) if cluster else CampaignPool(
                manager, HpcAgent, CAMPAIGN_ID, POOL_SIZE)

            # Handles are rebound to the exchange client: the pool probes them
            # for the life of the server, past the launch call that made them.
            async def launch_one() -> Handle:
                # The agent process is submitted through GCExecutor. This is
                # deliberately not a local executor and has no PBS commands.
                handle = await manager.launch(
                    HpcAgent(CAMPAIGN_ID, BASE_DIR, INSPECT_ROOTS),
                    executor="globus", name=f"{CAMPAIGN_ID}-agent"
                )
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def launch_client(generation_ckpt_dir: str) -> Handle:
                handle = await manager.launch(
                    HpcClientAgent(CAMPAIGN_ID, BASE_DIR, generation_ckpt_dir, INSPECT_ROOTS),
                    executor="globus", name=f"{CAMPAIGN_ID}-agent"
                )
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def launch_cluster(generation_ckpt_dir: str) -> Handle:
                handle = await manager.launch(HpcClusterAgent(
                    BASE_DIR, generation_ckpt_dir, nnodes, SYSTEM,
                ))
                return Handle(handle.agent_id, exchange=manager.exchange_client)

            async def run_code(payload: str, workdir: str | None = None, timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
                """Run Python on one campaign-owned, GPU/tile-pinned Academy agent."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("run_code", payload, workdir=workdir or BASE_DIR, timeout=timeout, venv_path = venv_path)
                except Exception as exc:
                    raise ToolError(f"run_code failed: {type(exc).__name__}: {exc}") from exc

            async def inspect_filesystem(operation: str, path: str, offset: int = 1, limit: int = 100) -> dict:
                """Bounded read-only ls/cat/tail inspection on a pinned campaign worker."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("inspect_filesystem", operation, path, offset=offset, limit=limit)
                except Exception as exc:
                    raise ToolError(f"inspect_filesystem failed: {type(exc).__name__}: {exc}") from exc

            async def run_batch(script_path: str | None = None, inputs_path: str | None = None,
                                timeout_each: float = 900.0, gpus: float = 1,
                                venv_path: str | None = None, batch_id: str | None = None,
                                labels: dict | None = None, tool: str | None = None,
                                tool_version: int | None = None) -> dict:
                """Run one Python script once per row of a JSONL file, in parallel across the cluster's GPUs.

                Give either script_path or tool (a name from list_tools; newest version
                unless tool_version). Each row runs as `python script.py input.json` in its
                own folder, where input.json is that row (an "id" field is optional); a tool
                registered with call="argv" runs as `python script.py --name value ...`.
                For a tool, every row is checked against its params before anything runs.
                Write results to files in that folder and exit 0 on success. The script and
                inputs file are copied when the batch is created, so later edits do not
                affect it. gpus is the number of GPUs/tiles EACH ROW gets, not the batch:
                rows run side by side on all free tiles, so a script that uses one device
                (e.g. xpu:0 or cuda:0) needs gpus=1 and 12 rows then fill a 12-tile node at
                once. Raise it only when a single row itself spans several devices.
                Returns at once with a batch_id: poll batch_status, never wait.
                To resume, pass only batch_id; finished runs are skipped. Use this rather
                than run_code for more than a handful of runs.
                """
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action(
                        "run_batch", script_path=script_path, inputs_path=inputs_path,
                        timeout_each=timeout_each, gpus=gpus, venv_path=venv_path,
                        batch_id=batch_id, labels=labels, tool=tool, tool_version=tool_version,
                    )
                except Exception as exc:
                    raise ToolError(f"run_batch failed: {type(exc).__name__}: {exc}") from exc

            async def batch_status(batch_id: str) -> dict:
                """Progress of one run_batch batch: total, succeeded, failed, unfinished, and the first failures' stderr."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("batch_status", batch_id)
                except Exception as exc:
                    raise ToolError(f"batch_status failed: {type(exc).__name__}: {exc}") from exc

            async def list_batches(plan_id: str | None = None, hypothesis_id: str | None = None,
                                   limit: int = 50) -> list[dict]:
                """Batches in the registry, newest first, with their last known status."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("list_batches", plan_id=plan_id,
                                                     hypothesis_id=hypothesis_id, limit=limit)
                except Exception as exc:
                    raise ToolError(f"list_batches failed: {type(exc).__name__}: {exc}") from exc

            async def register_tool(name: str, script_path: str, params: dict, description: str,
                                    example: dict, defaults: dict | None = None,
                                    call: str = "json", venv_path: str | None = None,
                                    skip_smoke: bool = False, labels: dict | None = None,
                                    expects: list[str] | None = None) -> dict:
                """Save a reusable script as a named, versioned tool that run_batch(tool=name) can run.

                params maps each input name to one of int, float, str, bool, list, dict;
                defaults fills params a row leaves out. example is one input row: it is run
                once as a smoke test and the tool is registered only if it exits 0 (use
                skip_smoke for a script that needs a GPU; it is then marked tested=false).
                call="json" runs `python script.py input.json`; call="argv" runs
                `python script.py --name value ...`. The script (at most 1 MB) is copied, so
                later edits need a new registration, which becomes the next version.
                Registering an unchanged script returns the existing version. Tools cannot
                be deleted. Call list_tools first and reuse a tool that fits.
                expects lists output filenames/globs (relative to each run's folder) the
                script must write; a run that exits 0 without them counts as failed.
                """
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action(
                        "register_tool", name, script_path, params, description, example,
                        defaults=defaults, call=call, venv_path=venv_path,
                        skip_smoke=skip_smoke, labels=labels, expects=expects,
                    )
                except Exception as exc:
                    raise ToolError(f"register_tool failed: {type(exc).__name__}: {exc}") from exc

            async def list_tools(query: str | None = None, name: str | None = None,
                                 all_versions: bool = False, limit: int = 50) -> list[dict]:
                """Registered tools: params, description, example, how often batches used them, last clean batch."""
                try:
                    agent = await pool.next_agent(launch_one)
                    return await agent.handle.action("list_tools", query=query, name=name,
                                                     all_versions=all_versions, limit=limit)
                except Exception as exc:
                    raise ToolError(f"list_tools failed: {type(exc).__name__}: {exc}") from exc

            # Build the pool before serving: a tool call must never have to
            # launch or verify an agent, or concurrent calls serialize. The
            # pool brings the EL cluster up first, then the clients that
            # handshake with its checkpoint dir.
            if cluster:
                await pool.start(launch_cluster, launch_client)

            server.add_tool(run_code)
            server.add_tool(inspect_filesystem)
            # run_batch needs EL; status/listing only read disk and register_tool's
            # smoke run is a local subprocess, so any agent serves those.
            if cluster:
                server.add_tool(run_batch)
            server.add_tool(batch_status)
            server.add_tool(list_batches)
            server.add_tool(register_tool)
            server.add_tool(list_tools)
            try:
                yield
            finally:
                # Stop health checking before Manager.close() shuts the agents
                # down, or the reaper sees them die and relaunches into a
                # cluster that is itself on the way out.
                if cluster:
                    await pool.stop()

    return FastMCP(name="hpc-deployment", instructions=(
        "run_code executes Python on a pinned accelerator. inspect_filesystem is bounded read-only ls/cat/tail; it is not a shell. "
        "For many runs of one script use run_batch, then poll batch_status; list_batches shows past batches. "
        "Before writing a new script, call list_tools and reuse a tool that fits; save a reusable script "
        "with register_tool and run it with run_batch(tool=...)."
    ), lifespan=lifespan, on_duplicate="replace")


if __name__ == "__main__":
    # Cluster mode only, but harmless in direct mode:
    # uvicorn's capture_signals restores the original SIGTERM handler and then
    # re-raises the signal once serve() returns. Python's default SIGTERM
    # handler kills the process on the spot, so the FastMCP lifespan never
    # resumes past its yield: Manager.close() is skipped, no agent is asked to
    # shut down, and HpcClusterAgent.agent_on_shutdown never calls el.stop() --
    # orphaning a live EL cluster that keeps its allocation until walltime.
    # Raising KeyboardInterrupt instead lets that re-raise unwind the lifespan
    # stack, so the cluster still dies with its agent. SIGINT already does this.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if EXECUTION == "cluster":
        # Must match the node count requested in HPC_ENDPOINT_CONFIG_JSON: it
        # sets the EL policy's leaf count, not the allocation size.
        nnodes = int(os.environ.get("HPC_NNODES", "1"))
        if nnodes < 1:
            raise RuntimeError("HPC_NNODES must be positive")
        server = create_server(cluster=True, nnodes=nnodes)
    else:
        server = create_server()
    server.run(transport=os.environ.get("HPC_MCP_TRANSPORT", "stdio"))
