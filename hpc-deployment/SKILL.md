---
name: hpc-deployment
description: >-
  Deploy a local one-tool MCP code-execution server backed by a user-supplied
  Globus Compute endpoint. The endpoint's Parsl PBSProProvider provisions
  user-specified queue, walltime, node and accelerator resources; one Academy
  code-execution agent is pinned to each allocated GPU/tile.
license: Apache-2.0
---

# Globus Compute → PBSPro → pinned Academy-agent MCP deployment

## Sole responsibility

This skill does exactly this, and nothing else:

1. Start a **local FastMCP server** with two public tools: `run_code` and bounded, read-only `inspect_filesystem`. The second tool exists solely to provide `ls`/`cat`/`tail`-equivalent worker visibility without SSH.
2. The server uses a **Globus Compute `Executor`** targeting the **endpoint UUID supplied by the user**.
3. The submitted endpoint job uses its **Parsl `PBSProProvider`** to request the **user-supplied allocation** (`account`, `queue`, `walltime`, `nodes_per_block`, block count, filesystem/place settings, and worker environment).
4. Within that allocation, start Academy agents in one of two modes. **Direct mode** starts one `HpcAgent` per pinned GPU/tile, and the agent runs the payload itself. **Cluster mode** (`HPC_EXECUTION=cluster`) starts one `HpcClusterAgent` that runs an Ensemble Launcher cluster, plus `HPC_POOL_SIZE` `HpcClientAgent`s that submit into it; EL owns GPU placement, so the client agents are deliberately unpinned. Every agent exposes `run_code` plus one constrained, read-only `inspect_filesystem` action. `run_code` executes submitted Python in a subprocess and returns `{stdout, stderr, returncode}`; `inspect_filesystem` supports `ls`, `cat`, and `tail` without giving the harness SSH.

The harness never uses SSH, `qsub`, `qstat`, `sbatch`, or a campaign-local scheduler wrapper to execute campaign work. The endpoint and its Parsl provider own scheduling. The agent has no domain-specific actions and no arbitrary shell action.

## Required inputs

Never guess, hard-code, or inherit these from an earlier campaign:

| Input | Meaning |
|---|---|
| `HPC_EXECUTION` | `direct` (default) or `cluster`. Picks the execution style; see "Two execution styles" below. |
| `HPC_GLOBUS_ENDPOINT_ID` | Globus Compute endpoint/manager UUID addressed by the local server. |
| `HPC_POOL_SIZE` | Agents to launch. Direct mode: equal to requested addressable GPU/tile count. Cluster mode: number of client agents submitting into the EL cluster. |
| `HPC_ENDPOINT_CONFIG_JSON` | JSON object accepted by the endpoint user-config schema; defines PBSPro resources. |
| `HPC_BASE_DIR` | Campaign directory visible to all workers. |
| `HPC_CAMPAIGN_ID` | Immutable ID used to isolate agent discovery/adoption. |
| `ACADEMY_EXCHANGE_URL` | Academy exchange the manager and agents rendezvous through. |
| `HPC_SYSTEM` | `aurora` or `polaris`. Selects the EL `SystemConfig` and the GPU-pinning variable (`ZE_AFFINITY_MASK` / `CUDA_VISIBLE_DEVICES`). Cluster mode only. |
| `HPC_NNODES` | Nodes in the allocation. Must match `nodes` in `HPC_ENDPOINT_CONFIG_JSON`: it sets the EL policy's leaf count, not the allocation size. Cluster mode only. |

Optional: `HPC_INSPECT_ROOTS` (colon-separated extra `inspect_filesystem` roots)
and `HPC_MCP_TRANSPORT` (`http` or `stdio`, default `stdio`).

Cluster-mode health checking (see "Self-healing" below) reads `HPC_POOL_PROBE_INTERVAL`
(60s), `HPC_POOL_PROBE_TIMEOUT` (30s), `HPC_CLUSTER_PROBE_STRIKES` (3),
`HPC_CLUSTER_MAX_RESTARTS` (3), `HPC_CLUSTER_STARTUP_TIMEOUT` (3000s),
`HPC_AGENT_STARTUP_TIMEOUT` (600s), and `HPC_POOL_WAIT_TIMEOUT` (unset).

### Two execution styles

This skill serves one of two styles, chosen by `HPC_EXECUTION` and nothing else:

| | `direct` (default) | `cluster` |
|---|---|---|
| Agents | one `HpcAgent` per pinned GPU/tile | one `HpcClusterAgent` holding an EL cluster, plus `HPC_POOL_SIZE` `HpcClientAgent`s |
| Who places work on GPUs | the endpoint, via one worker per pin | Ensemble Launcher |
| Tools | `run_code`, `inspect_filesystem`, `batch_status`, `list_batches`, `register_tool`, `list_tools` | the same, plus `run_batch` |
| Worker needs `ensemble_launcher` | no | **yes** |
| Extra required inputs | none | `HPC_SYSTEM`, `HPC_NNODES` |
| Acceptance signal | unique `(hostname, device pin)` per agent | overlapping concurrent calls; pins are empty by design |

Direct mode is the default because it has the smaller worker-side dependency
surface: `hpc_agent.py` imports `ensemble_launcher` inside a `try`, so a worker
without it serves direct mode fine and only the cluster agents fail. Cluster
mode is what you want for many runs of one script across every tile of a
multi-node allocation, since that is what `run_batch` plus EL placement buys.

### Setting them: `set_env.sh`

Define all of them in a `set_env.sh` and `source` it into the shell that runs
every later step (`set_env.example.sh` is a starting point for both styles):

```bash
cp hpc-deployment/set_env.example.sh hpc-deployment/set_env.sh
# edit it, then:
source hpc-deployment/set_env.sh
```

`mcp_server.py` reads the process environment and fails fast via `required()` on
anything missing, so make this file the single definition point rather than
exporting values by hand. A variable set in one shell and forgotten in the next
is the usual cause of a campaign that starts against the wrong endpoint or
adopts another campaign's agents.

`set_env.sh` is campaign-specific — edit it per deployment and keep the endpoint
UUID, queue, walltime, and `HPC_CAMPAIGN_ID` together in it so the whole
configuration is reviewable in one place.

Two couplings belong in the file itself, not in the operator's head:

- **Worker count.** The endpoint needs one worker slot per client agent *plus*
  one for the long-running `HpcClusterAgent`, which holds its slot for the life
  of the allocation. Derive it rather than typing a literal:
  `"workers": '"$((HPC_POOL_SIZE + 1))"'`.
- **Worker environment.** `worker_init` must activate the environment and put
  `HPC_BASE_DIR` on `PYTHONPATH`, or the agent modules will not import on the
  worker.

Example only—pass keys only when the supplied endpoint's user-config schema permits them:

```json
{
  "account": "PROJECT",
  "queue": "debug-scaling",
  "walltime": "01:00:00",
  "nodes_per_block": 1,
  "init_blocks": 1,
  "min_blocks": 1,
  "max_blocks": 1,
  "available_accelerators": 12,
  "max_workers_per_node": 12,
  "filesystems": "flare",
  "place": "scatter"
}
```

For Aurora, request 12 addressable tiles per node. Endpoint worker initialization must set `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and a distinct `ZE_AFFINITY_MASK` for every pinned worker. `HPC_POOL_SIZE` is `requested nodes × 12` unless the endpoint explicitly exposes another count.

## Architecture

```text
MCP harness
  → local FastMCP server (one public tool: run_code)
  → Globus Compute Executor(endpoint_id=HPC_GLOBUS_ENDPOINT_ID,
                            user_endpoint_config=HPC_ENDPOINT_CONFIG_JSON)
  → endpoint manager / user endpoint
  → Parsl PBSProProvider allocation
  → direct mode:  one worker per pinned GPU/tile
                  → one Academy HpcAgent per worker
  → cluster mode: HpcClusterAgent → EL cluster (owns GPU placement)
                  → HPC_POOL_SIZE HpcClientAgents submit into it
  → run_code(payload)
```

The endpoint UUID routes the request; it does not select the queue. The endpoint's accepted user configuration is rendered by its Parsl `PBSProProvider` into the PBS request. In multi-user deployments, record both the manager UUID and the spawned user-endpoint UUID: they are different by design.

## Deployment and verification

1. On the endpoint host, inspect the live endpoint user-config schema/template. Confirm that it accepts the requested allocation keys, activates the intended environment, makes `HPC_BASE_DIR` importable, and sets accelerator pinning.
2. Create/reuse a local uv environment containing `fastmcp`, `globus-compute-sdk`, and `academy-py`.
3. Edit `set_env.sh` for this campaign and `source` it. Run every later step in that same shell — the server inherits its configuration from the process environment and nowhere else.
4. Run `scripts/preflight.py`. It validates local dependencies/configuration and emits the exact worker-side import/pinning prerequisites without submitting a job.
5. Start `scripts/mcp_server.py`. Its Academy launch manager uses the Globus Compute `Executor` created with the supplied UUID and endpoint configuration. A local `ThreadPoolExecutor` is invalid: it launches agents locally.
6. Launch the agents. Direct mode launches exactly `HPC_POOL_SIZE` `HpcAgent`s, each reporting hostname, `HPC_CAMPAIGN_ID`, and a unique device pin. Cluster mode launches the `HpcClusterAgent` first, then `HPC_POOL_SIZE` `HpcClientAgent`s against that cluster's checkpoint dir; clients report an empty device pin by design. All handles are rebound to their exchange client before serving MCP calls — the pool probes them for the life of the server.
7. Run `scripts/smoke_test_client.py` against the server. It requires both `run_code` and `inspect_filesystem`, a normal code result, and bounded worker filesystem output.
8. Submit concurrent cheap calls and confirm they **overlap** rather than running one at a time. In direct mode also verify unique `(hostname, device pin)` pairs; if pins duplicate or an agent is absent, stop and repair placement. Pins are empty and identical in cluster mode, so they are not a valid acceptance signal there — EL owns placement.
9. Agents do not outlive the server. `Manager.close()` shuts down everything it launched, so a restarted server builds a fresh pool rather than adopting one. Within a run, cluster mode repairs itself (below).

The skill supplies `scripts/hpc_agent.py`, `scripts/pool_manager.py`, `scripts/mcp_server.py`, `scripts/preflight.py`, `scripts/smoke_test_client.py`, and `scripts/hpc_control.py` (cluster mode: resize or restart the allocation), plus the campaign's `set_env.sh`. A deployment is not considered ready until steps 7–8 succeed against the live endpoint.

### Shutting the server down

Stop the server with `SIGINT` (`Ctrl-C`, or `kill -INT`). A clean exit is what
runs `Manager.close()`, which asks every launched agent to shut down; for
`HpcClusterAgent` that is what calls `el.stop()` and releases the cluster.
`mcp_server.py` installs `signal.default_int_handler` for `SIGTERM` so that
signal unwinds the same way — without it uvicorn re-raises `SIGTERM` after
`serve()` returns, the default handler kills the process before the FastMCP
lifespan resumes past its `yield`, and the EL cluster is orphaned until
walltime. Never stop the server with `kill -9`: no agent is told to shut down
and the allocation leaks.

In cluster mode the lifespan calls `pool.stop()` before `Manager.close()`. That
ordering is load-bearing: the health-check task must stop before the manager
starts shutting agents down, or it sees them die during teardown and rebuilds
the cluster on the way out.

## Self-healing (cluster mode)

`ClusterPool` owns the cluster agent and the client agents, and re-probes them
every `HPC_POOL_PROBE_INTERVAL`. Repair runs on that timer, never inside a tool
call — reconciling per call is what serializes concurrent `run_code` requests.

- **A client agent dies** → it is dropped from rotation immediately and replaced
  on its own. The cluster is untouched.
- **The cluster agent dies** → the EL cluster died with it, so every client is
  holding a `ClusterClient` bound to nothing. After `HPC_CLUSTER_PROBE_STRIKES`
  consecutive failed probes the pool shuts the stale agents down (freeing their
  worker slots), mints a **fresh `ckpt_dir`**, and rebuilds the cluster and all
  clients. A new generation never reuses the dead master's checkpoint record.
- **During a rebuild** a tool call waits rather than failing, bounded by the MCP
  client's own timeout unless `HPC_POOL_WAIT_TIMEOUT` is set.
- **After `HPC_CLUSTER_MAX_RESTARTS` consecutive failed rebuilds** healing stops
  for good. The MCP server stays up and every tool call reports the terminal
  error; restart the server to try again.

The MCP server, its client sessions, and the Globus/Academy plumbing survive all
of this — a failed cluster is no longer a restart-everything event.

Known gap: `HpcClusterAgent.probe` reports only that the *agent* is alive. An EL
master that dies under a live agent reads as healthy, and tasks hang until their
own timeout instead of triggering a rebuild.

## Timeouts and failures

Distinguish:

- **Submit rejection:** no endpoint task was accepted—inspect UUID, auth, and endpoint configuration.
- **Scheduler wait:** PBS block is queued—do not blindly resubmit. Scheduler-wait allowance is independent of payload timeout.
- **Payload failure:** an agent returned a normal result with nonzero `returncode`—use stderr to fix/retry only that payload.
- **Cluster loss (cluster mode):** the cluster agent stopped answering—the pool rebuilds it automatically; see "Self-healing". Do not restart the server unless healing has reported itself unrecoverable.

Never alter queue or resource values outside the endpoint user-config schema. Never call scheduler commands from the harness.

## Agent capabilities and filesystem inspection

The Academy agent's public capabilities:

- `run_code(payload, workdir, timeout)` — execute a Python payload on the agent's pinned accelerator.
- `inspect_filesystem(operation, path, offset, limit)` — read-only visibility into the worker filesystem without SSH.
- `run_batch` / `batch_status` / `list_batches` (cluster mode) — run one script once per row of a JSONL file through EL; returns a `batch_id` at once, resumable by `batch_id`.
- `register_tool` / `list_tools` — a per-campaign library of frozen, versioned, smoke-tested scripts under `HPC_BASE_DIR/tools/`, runnable with `run_batch(tool=...)`. Tools are never deleted or deprecated.

**`run_batch` `gpus` is tiles per row, not per batch — use `gpus=1` for any script that touches one device (`xpu:0` / `cuda:0`).** EL then places rows side by side on every free tile: on Aurora (FLAT hierarchy) each row gets its own `ZE_AFFINITY_MASK` tile with 64 GB, and 12 rows fill a node at once. Verified 2026-10-20: 12 rows at `gpus=1` ran concurrently on masks 0–11 in 15 s. `gpus=12` instead reserves the whole node for every row, so rows run one at a time and a one-device script leaves 11 tiles idle (observed: 12 Chai folds took ~8.5 min instead of ~45 s). Raise `gpus` only when a single row itself spans several devices. EL tracks CPUs (`ppn=1`, one of 102 per Aurora node) and tiles, not host memory or thread counts; if many concurrent rows are CPU-heavy, cap `OMP_NUM_THREADS` in the script.

`inspect_filesystem` permits only:

| Operation | Semantics | Required bounds |
|---|---|---|
| `ls` | Sorted immediate directory entries with type and size metadata; no recursive traversal. | At most 200 entries. |
| `cat` | UTF-8 text read, with line numbers. | `offset ≥ 1`, at most 200 lines and 64 KiB. |
| `tail` | UTF-8 text read from end, with line numbers. | At most 200 lines and 64 KiB. |

The action resolves paths and restricts them to the explicit allowed roots: `HPC_BASE_DIR`, `HPC_INSPECT_ROOTS` (colon-separated optional additional roots), and `/tmp`. It rejects symlink escapes, binary data, directories passed to `cat`/`tail`, device files, and every write/mutation operation. Do not add a generic `shell`, `exec`, `find`, `grep`, environment dump, scheduler query, or recursive listing action. For programmatic analysis beyond bounded inspection, write it as a `run_code` payload.

## Non-negotiable constraints

- `HpcAgent` exposes only `run_code` and bounded, read-only `inspect_filesystem`; an internal health probe is permitted but is never a public MCP tool.
- One agent is pinned to one addressable accelerator; subprocesses inherit its pinning environment.
- Preserve endpoint UUID, endpoint config, campaign ID, agent IDs, pins, seeds, commands and outputs.
- This skill contains no molecular-design logic and exposes no additional MCP capabilities.
