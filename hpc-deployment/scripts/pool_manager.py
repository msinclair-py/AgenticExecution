"""Campaign-scoped Academy-agent lifecycle for hpc-deployment."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass

from academy.handle import Handle

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    handle: Handle
    pin: str


class CampaignPool:
    """Discover/adopt only this campaign's agents; then launch a shortfall."""

    def __init__(self, manager, agent_class, campaign_id: str, want: int, timeout_s: float = 60.0,
                 startup_timeout_s: float | None = None, require_pin: bool = True):
        self.manager = manager
        self.agent_class = agent_class
        self.campaign_id = campaign_id
        self.want = want
        # Direct mode: Parsl pins one accelerator per worker and the agent runs
        # the payload itself, so a unique non-empty pin is required. Cluster
        # mode: EL owns placement (gpu_selector/ngpus_per_process) and the
        # client agent only submits tasks, so it holds no accelerator at all.
        self.require_pin = require_pin
        # Liveness probes of pool members must fail fast. A freshly launched
        # agent instead has to clear scheduler wait and agent_on_startup (the
        # EL ClusterClient checkpoint handshake), so it gets a longer deadline.
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s if startup_timeout_s is not None else float(
            os.environ.get("HPC_AGENT_STARTUP_TIMEOUT", "600")
        )
        self._agents: list[AgentInfo] = []
        self._lock = asyncio.Lock()
        self._cursor = 0

    async def _probe(self, handle: Handle, timeout_s: float | None = None) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(handle.action("probe"), timeout=timeout_s or self.timeout_s)
            if result.get("campaign_id") != self.campaign_id:
                logger.warning("agent %s rejected: campaign_id=%r (want %r)",
                               handle.agent_id, result.get("campaign_id"), self.campaign_id)
                return None
            if self.require_pin and not result.get("device_pin"):
                logger.warning("agent %s rejected: empty device_pin; endpoint config must set "
                               "available_accelerators", handle.agent_id)
                return None
            return AgentInfo(handle, result.get("device_pin", ""))
        except Exception as exc:  # noqa: BLE001 - an unhealthy agent must never propagate
            logger.warning("probe of agent %s failed: %s: %s", handle.agent_id, type(exc).__name__, exc)
            return None

    async def _discover(self) -> list[AgentInfo]:
        ids = await self.manager.exchange_client.discover(self.agent_class, allow_subclasses=True)
        checks = [self._probe(Handle(agent_id, exchange=self.manager.exchange_client)) for agent_id in ids]
        return [item for item in await asyncio.gather(*checks) if item is not None]

    async def ensure(self, launch_one) -> list[AgentInfo]:
        async with self._lock:
            survivors = [item for item in await asyncio.gather(*(self._probe(a.handle) for a in self._agents)) if item]
            seen = {str(item.handle.agent_id) for item in survivors}
            for item in await self._discover():
                if str(item.handle.agent_id) not in seen:
                    survivors.append(item)
                    seen.add(str(item.handle.agent_id))
                if len(survivors) >= self.want:
                    break
            while len(survivors) < self.want:
                handle = await launch_one()
                checked = await self._probe(handle, timeout_s=self.startup_timeout_s)
                if checked is None:
                    raise RuntimeError(
                        "new Academy agent did not become healthy within "
                        f"{self.startup_timeout_s}s (see probe warnings above)"
                    )
                survivors.append(checked)
            pins = [agent.pin for agent in survivors]
            if self.require_pin and len(set(pins)) != len(pins):
                raise RuntimeError(f"agent device pins are not unique: {pins}")
            self._agents = survivors
            return survivors

    async def next_agent(self, launch_one) -> AgentInfo:
        pool = await self.ensure(launch_one)
        agent = pool[self._cursor % len(pool)]
        self._cursor += 1
        return agent


class ClusterUnrecoverable(RuntimeError):
    """The EL cluster could not be rebuilt within the allowed attempts."""


class ClusterPool:
    """Owns the EL cluster agent and the client agents that submit into it.

    In cluster mode EL owns placement and fanout, so the client agents are
    interchangeable submitters and hold no accelerator. The pool is built once
    by start() and handed out by next_agent() with no work per call; a
    background task re-probes it on an interval and repairs what died.

    The cluster agent belongs here because it defines what "dead" means for
    everything else. HpcClusterAgent.agent_on_shutdown stops the EL cluster,
    so losing that agent means the cluster is gone, and every client agent is
    then holding a ClusterClient bound to a cluster that no longer exists --
    whether or not it still answers a probe. A lost cluster therefore replaces
    the whole pool, while a lost client is replaced on its own. Because the
    repair happens here rather than in the server, the MCP server and its
    client sessions stay up across a cluster restart.

    This is deliberately not CampaignPool. That class reconciles the pool
    inside next_agent() -- probe, then exchange discovery -- under a single
    lock, so N concurrent run_code calls each wait out the previous caller's
    round trip and dispatch one at a time instead of overlapping. Moving the
    same repair work onto a timer keeps it off the request path. Adoption is
    also dropped: Manager.close() shuts down every agent it launched, and a
    client that did survive a hard kill is bound to the previous cluster.

    next_agent() keeps CampaignPool's async signature so the MCP tool bodies
    work with either pool; launch_one is accepted and ignored.
    """

    def __init__(self, campaign_id: str, want: int, ckpt_dir: str | None = None,
                 startup_timeout_s: float | None = None, probe_interval_s: float | None = None,
                 probe_timeout_s: float | None = None, cluster_timeout_s: float | None = None,
                 cluster_grace: int | None = None, max_restarts: int | None = None,
                 wait_timeout_s: float | None = None):
        self.campaign_id = campaign_id
        self.want = want
        # A new client has to clear scheduler wait and agent_on_startup (the EL
        # ClusterClient checkpoint handshake) before it answers a probe.
        self.startup_timeout_s = startup_timeout_s if startup_timeout_s is not None else float(
            os.environ.get("HPC_AGENT_STARTUP_TIMEOUT", "600")
        )
        # The cluster agent's first probe waits out the PBS queue, not just
        # process startup, so it gets a much larger budget than anything else.
        self.cluster_timeout_s = cluster_timeout_s if cluster_timeout_s is not None else float(
            os.environ.get("HPC_CLUSTER_STARTUP_TIMEOUT", "3000")
        )
        self.probe_interval_s = probe_interval_s if probe_interval_s is not None else float(
            os.environ.get("HPC_POOL_PROBE_INTERVAL", "60")
        )
        # Liveness probes must fail fast; only a cold launch gets the long one.
        self.probe_timeout_s = probe_timeout_s if probe_timeout_s is not None else float(
            os.environ.get("HPC_POOL_PROBE_TIMEOUT", "30")
        )
        # Tearing down a live cluster over one dropped probe costs a restart
        # and possibly a fresh allocation, so the cluster gets consecutive
        # strikes where a client agent gets one.
        self.cluster_grace = cluster_grace if cluster_grace is not None else int(
            os.environ.get("HPC_CLUSTER_PROBE_STRIKES", "3")
        )
        # Consecutive failed rebuilds before self-healing gives up for good.
        self.max_restarts = max_restarts if max_restarts is not None else int(
            os.environ.get("HPC_CLUSTER_MAX_RESTARTS", "3")
        )
        # How long a tool call waits for a restart. Unset means wait as long as
        # it takes and let the MCP client's own timeout be the bound.
        if wait_timeout_s is None:
            raw = os.environ.get("HPC_POOL_WAIT_TIMEOUT", "")
            wait_timeout_s = float(raw) if raw else None
        self.wait_timeout_s = wait_timeout_s

        self._cluster: AgentInfo | None = None
        self._agents: list[AgentInfo] = []
        self._cursor = 0
        self._started = False
        self._failed = False
        self._cluster_misses = 0
        self._restart_failures = 0
        # Caller-supplied dir seeds generation 0; every later generation mints
        # its own so a new ClusterClient cannot bind to the dead master's
        # record. Nothing here cleans the old dir up: it is remote HPC storage
        # and its contents are the evidence for why the cluster died.
        self._seed_ckpt_dir = ckpt_dir
        self._ckpt_dir: str | None = None
        self._launch_cluster = None
        self._launch_one = None
        self._reaper: asyncio.Task | None = None
        # Clear while the cluster is being rebuilt; tool calls wait on it
        # rather than failing, so a restart is invisible to a patient caller.
        self._ready = asyncio.Event()

    # -- probing ---------------------------------------------------------

    async def _probe(self, agent: AgentInfo, check_campaign: bool = True) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(
                agent.handle.action("probe"), timeout=self.probe_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - an unhealthy agent must never propagate
            logger.warning("agent %s failed liveness probe: %s: %s",
                           agent.handle.agent_id, type(exc).__name__, exc)
            return None
        # HpcClusterAgent.probe reports pid/device_pin only; the campaign check
        # applies to client agents.
        if check_campaign and result.get("campaign_id") != self.campaign_id:
            logger.warning("agent %s rejected: campaign_id=%r (want %r)",
                           agent.handle.agent_id, result.get("campaign_id"), self.campaign_id)
            return None
        return agent

    # -- launching -------------------------------------------------------

    def _next_ckpt_dir(self) -> str:
        if self._seed_ckpt_dir is not None:
            seeded, self._seed_ckpt_dir = self._seed_ckpt_dir, None
            return seeded
        return f"ckpt_{uuid.uuid4().hex}"

    async def _start_cluster(self) -> None:
        self._ckpt_dir = self._next_ckpt_dir()
        handle = await self._launch_cluster(self._ckpt_dir)
        result = await asyncio.wait_for(handle.action("probe"), timeout=self.cluster_timeout_s)
        self._cluster = AgentInfo(handle, result.get("device_pin", ""))
        self._cluster_misses = 0
        logger.info("EL cluster agent %s up (pid %s, ckpt_dir=%s)",
                    handle.agent_id, result.get("pid"), self._ckpt_dir)

    async def _launch_client(self) -> AgentInfo:
        handle = await self._launch_one(self._ckpt_dir)
        result = await asyncio.wait_for(handle.action("probe"), timeout=self.startup_timeout_s)
        if result.get("campaign_id") != self.campaign_id:
            raise RuntimeError(
                f"agent {handle.agent_id} reports campaign_id="
                f"{result.get('campaign_id')!r} (want {self.campaign_id!r})"
            )
        # No device_pin check: EL assigns GPUs to tasks, so the client agent
        # itself is unpinned by design.
        return AgentInfo(handle, result.get("device_pin", ""))

    async def _launch_clients(self, count: int) -> list[AgentInfo]:
        launched = await asyncio.gather(
            *(self._launch_client() for _ in range(count)), return_exceptions=True,
        )
        for item in launched:
            # gather(return_exceptions=True) swallows cancellation; re-raise so
            # stop() can actually stop us.
            if isinstance(item, asyncio.CancelledError):
                raise item
            if isinstance(item, BaseException):
                logger.error("client agent launch failed: %s: %s", type(item).__name__, item)
        return [item for item in launched if isinstance(item, AgentInfo)]

    async def _shutdown_all(self, agents: list[AgentInfo]) -> None:
        """Release the worker slots the replacements are about to need.

        The endpoint is sized at one slot per client plus one for the cluster
        agent, so leaving the old agents running would leave the new ones with
        nowhere to land. Best effort throughout: the agent being replaced is
        usually the one that stopped answering in the first place.
        """
        if not agents:
            return
        logger.info("shutting down %d stale agent(s) to free worker slots", len(agents))

        async def one(agent: AgentInfo) -> None:
            try:
                await asyncio.wait_for(agent.handle.shutdown(), timeout=self.probe_timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - it is probably already gone
                logger.debug("shutdown of %s failed: %s: %s",
                             agent.handle.agent_id, type(exc).__name__, exc)

        for item in await asyncio.gather(*(one(a) for a in agents), return_exceptions=True):
            if isinstance(item, asyncio.CancelledError):
                raise item

    # -- repair ----------------------------------------------------------

    async def _restart_cluster(self) -> None:
        """Rebuild the cluster and every client that was bound to the old one."""
        logger.error("EL cluster agent lost after %d probes; rebuilding cluster and "
                     "%d client agent(s)", self.cluster_grace, len(self._agents))
        self._ready.clear()
        stale = ([self._cluster] if self._cluster is not None else []) + self._agents
        # Stop serving agents wired to the dead cluster before anything slow.
        self._cluster = None
        self._agents = []
        await self._shutdown_all(stale)
        try:
            await self._start_cluster()
            self._agents = await self._launch_clients(self.want)
            if not self._agents:
                raise RuntimeError("no client agent came up against the new EL cluster")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - decide whether to keep healing
            self._restart_failures += 1
            logger.error("cluster restart attempt %d/%d failed: %s: %s",
                         self._restart_failures, self.max_restarts, type(exc).__name__, exc)
            if self._restart_failures >= self.max_restarts:
                self._failed = True
                # Release anyone waiting; they get the terminal error instead.
                self._ready.set()
                raise ClusterUnrecoverable(
                    f"EL cluster unrecoverable after {self.max_restarts} restart attempts"
                ) from exc
            return  # try again on the next sweep
        self._restart_failures = 0
        self._ready.set()
        logger.info("cluster rebuilt: %d client agent(s), ckpt_dir=%s",
                    len(self._agents), self._ckpt_dir)

    async def _repair(self) -> None:
        if self._cluster is None or await self._probe(self._cluster, check_campaign=False) is None:
            self._cluster_misses += 1
            logger.warning("EL cluster agent probe failed [%d/%d]",
                           self._cluster_misses, self.cluster_grace)
            if self._cluster_misses >= self.cluster_grace:
                await self._restart_cluster()
            return
        self._cluster_misses = 0
        try:
            await self._repair_clients()
        finally:
            # A rebuild that brought the cluster up but no clients returns with
            # _ready cleared; once a sweep restores clients, release waiters.
            if self._agents and not self._ready.is_set():
                self._restart_failures = 0
                self._ready.set()

    async def _repair_clients(self) -> None:
        probed = await asyncio.gather(*(self._probe(agent) for agent in self._agents))
        survivors = [agent for agent in probed if agent is not None]
        dropped = [agent for agent, ok in zip(self._agents, probed) if ok is None]
        if len(survivors) < len(self._agents):
            # Publish survivors before relaunching: a dead agent must stop
            # being handed out even if its replacement cannot be launched.
            logger.warning("cluster pool lost %d client agent(s)",
                           len(self._agents) - len(survivors))
            self._agents = survivors
        shortfall = self.want - len(survivors)
        if shortfall <= 0:
            return
        # M8: free the dropped agents' worker slots BEFORE launching replacements.
        # A client that fails its probe but is still alive (hung) keeps holding
        # its slot; the endpoint is sized at one slot per client (+1 for the
        # cluster), so without this the replacement has nowhere to land and
        # _launch_client blocks the whole repair sweep for startup_timeout_s
        # (~600s) -- freezing cluster liveness for ~10 min. Best effort: the
        # dropped agent is usually already gone, and _shutdown_all is bounded.
        if dropped:
            await self._shutdown_all(dropped)
        fresh = await self._launch_clients(shortfall)
        if fresh:
            self._agents = survivors + fresh
            logger.info("cluster pool restored to %d client agent(s)", len(self._agents))

    async def _repair_loop(self) -> None:
        while True:
            await asyncio.sleep(self.probe_interval_s)
            try:
                await self._repair()
            except asyncio.CancelledError:
                raise
            except ClusterUnrecoverable as exc:
                logger.error("%s; self-healing stopped. The MCP server stays up and every "
                             "tool call will report this.", exc)
                return
            except Exception:  # noqa: BLE001 - one bad sweep must not end health checking
                logger.exception("cluster pool repair sweep failed; retrying next interval")

    # -- lifecycle -------------------------------------------------------

    async def start(self, launch_cluster, launch_client) -> list[AgentInfo]:
        """Bring up the cluster, then the clients, then begin health checking.

        Call once before serving. The cluster must be up first: a client's
        agent_on_startup handshakes with the EL checkpoint dir. Both callbacks
        take the generation's ckpt_dir.
        """
        self._launch_cluster = launch_cluster
        self._launch_one = launch_client
        await self._start_cluster()
        self._agents = await self._launch_clients(self.want)
        if not self._agents:
            raise RuntimeError("no client agent could be launched against the EL cluster")
        self._started = True
        self._ready.set()
        self._reaper = asyncio.create_task(self._repair_loop(), name="cluster-pool-repair")
        logger.info("campaign %s cluster pool ready: %d client agent(s), probing every %ss",
                    self.campaign_id, len(self._agents), self.probe_interval_s)
        return self._agents

    async def stop(self) -> None:
        """Stop health checking. Must run before the Manager shuts agents down,
        or the reaper races the shutdown and rebuilds what it just lost."""
        if self._reaper is None:
            return
        self._reaper.cancel()
        try:
            await self._reaper
        except asyncio.CancelledError:
            pass
        self._reaper = None

    async def next_agent(self, launch_one=None) -> AgentInfo:
        if not self._started:
            raise RuntimeError("ClusterPool.start() has not run")
        if not self._ready.is_set():
            logger.info("tool call waiting for EL cluster rebuild")
            if self.wait_timeout_s is None:
                await self._ready.wait()
            else:
                await asyncio.wait_for(self._ready.wait(), timeout=self.wait_timeout_s)
        if self._failed:
            raise RuntimeError(
                f"EL cluster unrecoverable after {self.max_restarts} restart attempts; "
                "self-healing has stopped and the server must be restarted"
            )
        if not self._agents:
            raise RuntimeError("cluster pool has no healthy client agents")
        agent = self._agents[self._cursor % len(self._agents)]
        self._cursor += 1
        return agent
