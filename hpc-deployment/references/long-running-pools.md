# Keeping the agent pool alive

This is not an optional add-on for multi-day campaigns -- it's how the whole
pattern works. The MCP server runs **locally** (laptop, under the harness)
and launches a *pool* of `HpcAgent`s onto a **Globus Compute endpoint**, over
`HttpExchangeFactory(url, auth_method='globus')`, holding
`self.pools[name] -> list[Handle]` and dispatching round-robin. There is no
in-allocation server and no batch job to give it a lifetime, so nothing here
is scoped to "long runs" -- a five-minute session goes through the exact same
exchange, with the exact same credential problem described below.

Code: `scripts/hpc_agent.py` (the agent side -- the TTL watchdog is built in,
not a separate variant) and `scripts/pool_manager.py` (the client side, run
alongside `scripts/mcp_server.py`).

The pool is a long-lived distributed object you have to maintain even for a
short session, and it decays *silently*: the endpoint reports online, the
agent count looks right, and work simply stops being done. Three mechanisms,
in the order they bite. All three were observed on a 96h Polaris run; the
first cost ~73h of a reservation.

## Root cause: the exchange credential never refreshes

`HttpExchangeFactory.__init__` materialises a bearer token **once**:

```python
additional_headers |= get_auth_headers(auth_method)   # resolved ONCE, held forever
```

There is no refresh, even with a valid refresh token sitting in the local
store. When it lapses the client starts failing with `ForbiddenError:
Exchange entity UserId<...> is not authorized to access this resource` --
wording that points at *permissions*, so the natural first hypothesis is a
misconfigured identity. It is not. It is expiry. Worse, **every agent carries
the same credential**, so the entire pool detaches at once, and each agent
keeps holding its worker slot forever. The endpoint is then sterile: nothing
can be placed, no orphan can be asked to exit (asking needs the exchange it
just fell off), and only the endpoint's own admin tooling can reclaim the
slot -- this skill has no way to force it from the outside.

## The three mechanisms, and why all three are required

1. **Agents retire themselves** (`hpc_agent.py`'s `_lifetime_watchdog`) --
   give the agent a lifetime shorter than the credential's TTL and let it
   stand down cleanly once quiescent, so the slot is reclaimed. **Pick the
   TTL against the credential, not against intuition**: a 6h default against
   a ~48h token
   retired agents eight times per allocation instead of once.
   Self-retirement alone, with nothing refilling the pool, is just a
   scheduled outage -- capacity halves on a timer whether or not anything is
   wrong.

2. **Adopt before launching** (`pool_manager.adopt_live_agents`) -- Handles
   live in the client's process memory, agents live on the endpoint's
   compute. A client restart loses the handles but not the agents, which keep running
   and keep holding every slot. Launch unconditionally and the new agents
   queue behind orphans forever, and each restart makes it worse (observed
   going 8 -> 4 -> 2 reachable workers across three restarts). A `Handle` is
   only `(agent_id, exchange_client)` and `discover()` returns the ids, so a
   living agent is never lost, only unaddressed -- probe candidates
   concurrently and adopt the survivors.

   Distinguishing endpoints by `base_dir` is not enough if two endpoints
   share one: a queued endpoint with zero nodes adopted a running endpoint's
   agents and reported `live: true, agents: 8`. Real agents, wrong endpoint,
   wrong capacity decision downstream.

3. **Re-validate and top up** (`pool_manager.ensure_pool`) -- a pool is not
   durable. Cache it at first use and never check it again, and every
   retirement becomes permanent lost capacity. Two mistakes worth avoiding,
   both made while writing this: **extend the pruned survivors, never
   overwrite them** (overwriting discards live agents and then tries to
   launch into the slots they still hold), and **ask adoption only for the
   shortfall**. Never stamp an empty pool as checked -- that suppresses
   revalidation and makes "slow to start" look permanently dead.

## Knobs

| Variable | Default | Set it by |
|---|---|---|
| `ACADEMY_AGENT_TTL_S` | `108000` (30h) | the credential's life, minus a margin. `0` disables. Read **on the worker**, so it must be in the endpoint's env, not the client's. |
| `GCMCP_REVALIDATE` | `600` s | how fast you need a lost agent noticed, against the cost of a probe round. |
| `GCMCP_ADOPT_TIMEOUT` | `20` s | round-trip to the exchange plus agent scheduling latency. Too low and you orphan live agents; too high and a restart stalls on the stale ones. |

## Telling "wedged" from "busy"

This is the hardest part and every convenient signal lies:

| Signal | What it looks like when wedged |
|---|---|
| endpoint status | `online` |
| pool agent count | full |
| worker logs | quiet -- they only log task *receipt* |
| task state | `pending` forever, indistinguishable from queued |

**Only two things are reliable.** `failed_infra` versus `pending` in a poll
result: dead agents return the former *immediately*, busy ones return the
latter. And a **direct load measurement on the node** -- `HpcAgent.probe()`
(`hpc_agent.py`) returns `os.getloadavg()[0]` for exactly this. A node at
`load1 == 0.0` with "running" tasks is wedged, full stop. If even that is
inconclusive, find the process: `ps` for the script name and check `%CPU` and
state.

`pool_manager.submit_batch` round-robins into whatever handles are in the
pool -- with *k* dead of *n* it silently loses *k/n* of everything submitted,
no error, no timeout. Only trust round-robin once a poll has confirmed every
handle is alive; pin explicitly to verified `agent_index` values otherwise.
