#!/bin/bash
# Template for a campaign's set_env.sh. Copy it, fill in the site values, and
# `source` it into the shell that runs preflight.py and mcp_server.py -- the
# server reads the process environment and nowhere else.
#
#     cp hpc-deployment/set_env.example.sh hpc-deployment/set_env.sh
#     $EDITOR hpc-deployment/set_env.sh
#     source hpc-deployment/set_env.sh
#
# Keep the real set_env.sh out of git if it carries anything site-private.

# ---- Execution style -------------------------------------------------------
# direct  : one HpcAgent per pinned GPU/tile; the agent runs the payload. No
#           Ensemble Launcher, so the worker does not need ensemble_launcher.
# cluster : one HpcClusterAgent holds an EL cluster and HPC_POOL_SIZE
#           HpcClientAgents submit into it; EL owns GPU placement and run_batch
#           becomes available. The worker MUST have ensemble_launcher installed.
export HPC_EXECUTION="direct"

# ---- Required in both styles -----------------------------------------------
# Globus Compute endpoint/manager UUID. The UUID routes the request; it does
# not select the queue -- that comes from HPC_ENDPOINT_CONFIG_JSON below.
export HPC_GLOBUS_ENDPOINT_ID="REPLACE-WITH-ENDPOINT-UUID"
# Agents to launch. direct: one per addressable GPU/tile. cluster: how many
# client agents may submit at once (each is cheap -- it holds no accelerator).
export HPC_POOL_SIZE=1
# Campaign directory visible to every worker. batches/ and tools/ are created
# here, so it must be on a shared filesystem, never node-local.
export HPC_BASE_DIR="/REPLACE/WITH/SHARED/CAMPAIGN/DIR"
# Immutable; isolates agent discovery so two campaigns cannot adopt each other.
export HPC_CAMPAIGN_ID="REPLACE-WITH-CAMPAIGN-ID"
export ACADEMY_EXCHANGE_URL="https://exchange.academy-agents.org/v1"
# http keeps the server alive across harness restarts; stdio is the default.
export HPC_MCP_TRANSPORT="http"

# Extra read-only roots for inspect_filesystem (colon-separated). Unset, the
# agent can only look inside HPC_BASE_DIR -- which means it cannot read results
# it wrote into a project tree elsewhere.
# export HPC_INSPECT_ROOTS="/path/one:/path/two"

# ---- Required in cluster mode only -----------------------------------------
# Selects the EL SystemConfig and the GPU-pinning variable:
# aurora -> ZE_AFFINITY_MASK, polaris -> CUDA_VISIBLE_DEVICES.
export HPC_SYSTEM="aurora"
# Nodes in the allocation. This sets EL's policy leaf count, NOT the allocation
# size, so it must equal nodes_per_block below -- preflight.py refuses a
# mismatch, because a leaf count above the allocated nodes hangs the cluster.
export HPC_NNODES=1

# ---- PBS request -----------------------------------------------------------
# Rendered by the endpoint's Parsl PBSProProvider. Only keys the endpoint's
# user-config schema accepts are honoured; anything else is silently dropped,
# so check the endpoint's user_config_template.yaml.j2 before adding one.
#
# max_workers_per_node: one slot per client agent, PLUS one for the long-lived
# HpcClusterAgent, which holds its slot for the life of the allocation. Derive
# it rather than typing a literal. In direct mode, drop the +1.
#
# worker_init must be ONE line (the template renders it as a single quoted YAML
# string) and must both activate the environment and put the scripts directory
# on PYTHONPATH, or hpc_agent will not import on the worker.
export HPC_ENDPOINT_CONFIG_JSON='{
  "account": "REPLACE-WITH-ALLOCATION",
  "queue": "debug",
  "walltime": "00:30:00",
  "nodes_per_block": '"$HPC_NNODES"',
  "max_workers_per_node": '"$((HPC_POOL_SIZE + 1))"',
  "init_blocks": 1,
  "min_blocks": 1,
  "max_blocks": 1,
  "max_idletime": 3600,
  "worker_init": "export TMPDIR=/tmp; module reset; module load frameworks; source /REPLACE/WITH/VENV/bin/activate; export PYTHONPATH=/REPLACE/WITH/REPO/hpc-deployment/scripts:$PYTHONPATH"
}'
