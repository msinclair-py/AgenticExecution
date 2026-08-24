---
name: hpc-deployment
description: >-
  Expose HPC compute to an external MCP harness (Claude Code, Codex, ...) as a
  single code-execution tool, with zero ssh and zero batch-script submission.
  A local FastMCP server launches and maintains a pool of academy Agents onto
  a Globus Compute endpoint, one agent per pinned accelerator; each agent's
  only capability is running an LLM-authored code payload in a subprocess and
  returning stdout/stderr/returncode. Use when you want an LLM harness to run
  work on a supercomputer without the harness (or you) ever touching qsub,
  sbatch, or an interactive shell on the machine, or when a prior deployment
  based on submitting a PBS/Slurm batch job is the wrong shape for the
  problem. Covers the agent (deliberately the only tool it will ever expose),
  the local MCP server, the pool lifecycle that keeps agents alive across
  client restarts and non-refreshing Globus credentials, and the diagnostic
  ladder for telling a wedged pool from a busy one.
license: Apache-2.0
metadata:
  verified-against: >-
    Prior PBS/Slurm-submission variant of this pattern deployed end-to-end on
    ALCF Aurora 2026-07-24 and driven via a directly-submitted batch job. The
    Globus Compute pool mechanics (self-retirement, adopt-before-launch,
    revalidate-and-top-up) verified on ALCF Polaris 2026-07-30..08-02 across
    four multi-day blind campaigns. This all-Globus-Compute, no-batch-script
    shape supersedes the earlier PBS-submission variant; re-verify the
    end-to-end HTTP path (endpoint -> local MCP server -> run_code) against a
    live endpoint before trusting a big campaign on it.
---

# Deploying a code-execution agent on HPC via MCP + Globus Compute

The pattern: one **academy `Agent`** exposes exactly one capability -- run
this code payload in a subprocess, return what happened. A **FastMCP** server
running **locally** (on your laptop, under the harness) launches a *pool* of
these agents onto a **Globus Compute endpoint** and exposes that single
capability as one MCP tool. An external harness connects to the local server;
the harness generates code, the tool relays it to a pinned accelerator, and
that's the entire interface.

There is no batch script anywhere in this pattern and no direct interaction
with a scheduler. Getting the underlying allocation is the Globus Compute
endpoint's job -- configured once, out of band, on the HPC side. This skill
never runs `qsub`, `sbatch`, or opens an interactive session on the machine;
everything happens over the Globus Compute SDK's authenticated HTTPS API from
wherever you run the local MCP server.

This is instruction-only and self-contained: nothing here imports a project
package. `scripts/` holds ready-to-copy files.

## When to use it

- You want Claude Code / Codex / any MCP client to run work on a
  supercomputer, with the LLM outside the allocation and the compute inside
  it, and you cannot or do not want to give the harness (or yourself, per
  campaign) ssh access to submit anything.
- You have, or are willing to write, a Globus Compute endpoint on the target
  machine -- someone with HPC access has already configured and started it.
- You need **one tool call to land on one pinned accelerator**, with many
  calls in flight -- the throughput comes from fanning independent calls
  across a pool, not from one big MPI job.

Do **not** use it for a single tightly-coupled MPI simulation -- that's a
plain `mpiexec` job under whatever the endpoint's environment provides, not
something this pattern's one-call-per-accelerator model fits.

## The one rule this whole pattern exists to enforce

**The agent has exactly one action: run a code payload.** Never add a
domain-specific `@action` to `HpcAgent`. If a campaign needs a new
capability -- a library, a call out to some other tool, a different kind of
analysis -- that capability is something the *payload* imports or invokes,
written by the LLM that's driving the harness. The agent is a relay, not a
framework: its only job is giving the harness compute it doesn't otherwise
have. The moment you're tempted to add a second `@action`, that logic belongs
in the code the harness sends instead.

This is also why there's no per-domain "replace this section" framing in the
scripts below, unlike a typical template -- there's nothing domain-specific
to replace.

## Setup (gate before installing)

```bash
python -c "import academy, fastmcp" 2>/dev/null && echo ready || \
  pip install 'academy-py>=0.5' 'fastmcp>=3.4' pydantic
```

`academy` is imported from the **`academy-py`** distribution -- the import
name and the package name differ. The Globus auth path
(`HttpExchangeFactory(url, auth_method='globus')`) needs whatever extra
academy-py ships for Globus auth; check its docs for the current extra name
if `pip install` fails to resolve it. Parsl is not a direct dependency of
anything in `scripts/` -- it lives inside the Globus Compute endpoint's own
engine, which is configured independently of this skill.

## Architecture

```
external MCP harness ──HTTP──► local FastMCP server ──Globus Compute──► HpcAgent (pinned accelerator)
   (Claude Code)                (scripts/mcp_server.py)   (HTTPS,        (scripts/hpc_agent.py)
                                        │                  no ssh)              │
                                        │                                run_code(payload)
                                 PoolManager: list[Handle],                     │
                                 one per pinned accelerator,             subprocess.run(...)
                                 dispatched round-robin
```

| Layer | Owns | Runs |
|---|---|---|
| Globus Compute endpoint | the allocation, the modules, the venv, per-agent accelerator pinning | on the HPC, configured out of band -- **not this skill** |
| academy `HpcAgent` | one action (`run_code`), the TTL self-retirement watchdog | one process per pinned accelerator, launched by the endpoint |
| `PoolManager` | adopt/revalidate/top-up so the pool survives client restarts and credential expiry | inside the local MCP server process |
| FastMCP | the wire protocol the harness speaks; exposes exactly one tool | locally, alongside the harness |

## What you deploy

| File | Runs | Purpose |
|---|---|---|
| [`scripts/hpc_agent.py`](scripts/hpc_agent.py) | one process per pinned accelerator, launched via Globus Compute | the agent: `run_code`, `probe`, and the TTL watchdog -- nothing else, ever |
| [`scripts/pool_manager.py`](scripts/pool_manager.py) | imported by `mcp_server.py` | pool lifecycle: adopt live agents, revalidate, top up |
| [`scripts/mcp_server.py`](scripts/mcp_server.py) | locally, alongside the harness | the only entry point -- opens the Globus exchange, keeps the pool alive, exposes `run_code` as the sole MCP tool |

Every file is copy-paste ready with inline comments explaining the non-obvious
parts at the point they matter.

## The flow

1. **Precondition**: a Globus Compute endpoint is already running on the
   target HPC, configured (accelerator count, queue, walltime, worker
   environment) out of band via `globus-compute-endpoint configure` /
   `start`. That configuration is not part of this skill -- get its exchange
   URL from whoever set it up.
2. **Agent** (`hpc_agent.py`): `run_code` runs the payload in a subprocess off
   the event loop, so the agent keeps answering `probe` (and other in-flight
   calls) while a long snippet runs. Run untrusted/LLM-authored work in a
   subprocess rather than the agent process itself -- a segfault or `sys.exit`
   then costs one task, not the whole agent -- and return
   `{stdout, stderr, returncode}` rather than raising for task-level failure;
   reserve exceptions for infrastructure faults. `subprocess.run`'s
   `env=os.environ.copy()` inherits the accelerator-pinning var
   (`CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK`, see
   [references/site-profiles.md](references/site-profiles.md)) that the
   endpoint already set on this process before it started.
3. **Pool** (`pool_manager.py`): every deployment goes through a Globus
   exchange credential that never refreshes, so the pool needs
   adopt-before-launch and periodic revalidation even for a short session --
   see [references/long-running-pools.md](references/long-running-pools.md)
   for the failure modes this exists to survive, and read it before trusting
   a real campaign to this pattern.
4. **Server** (`mcp_server.py`): the only entry point. Set
   `ACADEMY_EXCHANGE_URL` (the endpoint's exchange) and `HPC_POOL_SIZE`
   (match the endpoint's accelerator count), then run it locally. It rebinds
   every launched `Handle` with an explicit `exchange=` before use -- a
   `Handle` created inside the lifespan and called from a different asyncio
   task (every FastMCP tool call is one) otherwise dies with
   `ExchangeClientNotFoundError`, because it resolves its `ExchangeClient`
   through a ContextVar that doesn't propagate across that task boundary.
5. **Connect the harness** to the local server over HTTP
   (`http://127.0.0.1:<HPC_PORT>/mcp` by default). Throughput comes from
   concurrency *at the harness*: issue many independent `run_code` calls and
   let the pool's round-robin place them across accelerators.

## Verification ladder -- climb it before trusting a big job

1. **Endpoint reachable**: from the machine that will run `mcp_server.py`,
   confirm the Globus Compute exchange URL answers and your Globus auth is
   valid, before writing any campaign code against it.
2. **Single agent, single call**: run `mcp_server.py` with `HPC_POOL_SIZE=1`,
   connect an MCP client, `list_tools()` → `run_code`, and a trivial payload
   (`print("hi")`) → expected stdout. Confirms the launch, the `Handle`
   rebind, and the round trip.
3. **Pool of N**: raise `HPC_POOL_SIZE` to match the endpoint's real
   accelerator count, fire N+ concurrent calls, and confirm they land on
   distinct workers (have the payload print its pinning env var).
4. **Restart survival**: kill and restart `mcp_server.py` mid-campaign;
   confirm the next `run_code` call adopts the still-live agents rather than
   launching duplicates on top of them (watch the pool size, not just that a
   call succeeds).
5. **Credential-expiry dry run**: set `ACADEMY_AGENT_TTL_S` low (minutes) in
   the endpoint's env for a throwaway test, and confirm an agent retires and
   the pool tops itself back up without manual intervention, before relying
   on the real (hours-to-days) TTL for a production campaign.

## Reference material

- [references/site-profiles.md](references/site-profiles.md) -- which env
  var a payload should read for its pinned accelerator, per site, and how to
  write device-agnostic payload code.
- [references/long-running-pools.md](references/long-running-pools.md) -- why
  the exchange credential never refreshes, the three mechanisms that keep the
  pool alive despite it, and the diagnostic ladder for telling a wedged pool
  from a busy one. Read this before any real campaign, not just long ones.
