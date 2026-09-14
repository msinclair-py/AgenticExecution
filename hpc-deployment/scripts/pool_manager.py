"""Campaign-scoped Academy-agent lifecycle for hpc-deployment."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from academy.handle import Handle


@dataclass
class AgentInfo:
    handle: Handle
    pin: str


class CampaignPool:
    """Discover/adopt only this campaign's agents; then launch a shortfall."""

    def __init__(self, manager, agent_class, campaign_id: str, want: int, timeout_s: float = 60.0):
        self.manager = manager
        self.agent_class = agent_class
        self.campaign_id = campaign_id
        self.want = want
        self.timeout_s = timeout_s
        self._agents: list[AgentInfo] = []
        self._lock = asyncio.Lock()
        self._cursor = 0

    async def _probe(self, handle: Handle) -> AgentInfo | None:
        try:
            result = await asyncio.wait_for(handle.action("probe"), timeout=self.timeout_s)
            if result.get("campaign_id") != self.campaign_id or not result.get("device_pin"):
                return None
            return AgentInfo(handle, result["device_pin"])
        except Exception:
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
                checked = await self._probe(handle)
                if checked is None:
                    raise RuntimeError("new Academy agent did not become healthy")
                survivors.append(checked)
            pins = [agent.pin for agent in survivors]
            if len(set(pins)) != len(pins):
                raise RuntimeError(f"agent device pins are not unique: {pins}")
            self._agents = survivors
            return survivors

    async def next_agent(self, launch_one) -> AgentInfo:
        pool = await self.ensure(launch_one)
        agent = pool[self._cursor % len(pool)]
        self._cursor += 1
        return agent
