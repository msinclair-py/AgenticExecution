"""Template: client-side pool lifecycle for the HpcAgent pool.

From the hpc-deployment skill. Runs on the SAME side as the local MCP server
(scripts/mcp_server.py) -- never on the HPC -- and holds Handles to
HpcAgents (hpc_agent.py) launched onto a Globus Compute endpoint. This is not
an optional add-on for long campaigns: every deployment goes through a Globus
exchange, whose credential never refreshes (see
references/long-running-pools.md), so every pool -- a five-minute session
included -- needs adoption and revalidation to survive a client restart.

`manager` is an academy Manager built from an HttpExchangeFactory pointed at
the Globus Compute endpoint; `agent_class` is HpcAgent (or a subclass).
"""
import asyncio
import os
import time

from academy.handle import Handle

POOL_REVALIDATE_S = float(os.environ.get('GCMCP_REVALIDATE', '600'))
ADOPT_TIMEOUT_S = float(os.environ.get('GCMCP_ADOPT_TIMEOUT', '20'))


class PoolManager:
    def __init__(self, manager, agent_class):
        self.manager = manager
        self.agent_class = agent_class
        self.pools: dict[str, list[Handle]] = {}
        self._checked: dict[str, float] = {}
        self._adopted: set[str] = set()
        self._lock = asyncio.Lock()

    async def adopt_live_agents(self, name: str, base_dir: str, want: int) -> list[Handle]:
        """A living agent is never lost, only unaddressed -- a Handle is just
        (agent_id, exchange_client), and discover() returns the ids. Filter by
        base_dir, or one campaign silently adopts another's agents (two
        endpoints sharing one base_dir has happened: a queued endpoint with
        ZERO nodes adopted a running endpoint's agents and reported
        `live: true, agents: 8`)."""
        client = self.manager.exchange_client
        ids = await client.discover(self.agent_class, allow_subclasses=True)

        async def check(aid):
            try:                                      # probe CONCURRENTLY: after a few
                h = Handle(aid, exchange=client)      # restarts most candidates are
                await asyncio.wait_for(h.action('probe'), timeout=ADOPT_TIMEOUT_S)
                return aid, h                         # stale and will time out
            except Exception:
                return None

        live = [r for r in await asyncio.gather(
            *(check(a) for a in ids if str(a) not in self._adopted)) if r]
        chosen = live[:max(0, want)]
        for aid, _ in chosen:
            self._adopted.add(str(aid))
        return [h for _aid, h in chosen]

    async def ensure_pool(self, name: str, base_dir: str, want: int, launch_one) -> list[Handle]:
        """launch_one() -> Awaitable[Handle] launches one fresh agent onto the
        endpoint; supply it from wherever you call manager.launch(...)."""
        fresh = time.monotonic() - self._checked.get(name, 0) < POOL_REVALIDATE_S
        if self.pools.get(name) and fresh:
            return self.pools[name]

        async with self._lock:
            prior = self.pools.get(name, [])

            async def alive(h: Handle):
                try:
                    await asyncio.wait_for(h.action('probe'), timeout=ADOPT_TIMEOUT_S)
                    return h
                except Exception:
                    return None

            # Extend the pruned survivors, never overwrite them -- overwriting
            # discards live agents and then tries to launch into the slots
            # they still hold.
            pool = [h for h in await asyncio.gather(*(alive(h) for h in prior)) if h]
            if len(pool) < want:
                # Ask adoption only for the shortfall.
                pool = pool + await self.adopt_live_agents(name, base_dir, want - len(pool))
            for _ in range(len(pool), want):
                pool.append(await launch_one())

            if pool:
                self.pools[name] = pool
                self._checked[name] = time.monotonic()
            else:
                # Do NOT stamp an empty pool: it suppresses revalidation and
                # makes "slow to start" look permanently dead.
                self.pools.pop(name, None)
            return pool

    async def submit_batch(self, name: str, tasks: list, base_dir: str, want: int, launch_one):
        """Round-robins into pool handles. If the pool is unhealthy this
        silently loses work to dead handles -- k dead of n loses k/n of
        everything submitted, with no error and no timeout, just tasks
        pending forever. Only rely on round-robin once a poll has confirmed
        every handle in the pool is alive; pin explicitly to verified
        agent_index values otherwise."""
        pool = await self.ensure_pool(name, base_dir, want, launch_one)
        if not pool:
            raise RuntimeError(f'pool {name!r} has no live agents')
        results = []
        for i, task in enumerate(tasks):
            handle = pool[i % len(pool)]
            results.append(await handle.action('run_code', task))
        return results
