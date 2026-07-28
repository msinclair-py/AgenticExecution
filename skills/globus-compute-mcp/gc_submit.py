"""Robust submission helper for the s_elongatus Globus Compute endpoint.

Two hazards this exists to absorb, both observed for real:

1. COLD-START RACE. In the 4.x manager model the first submit is what causes the
   manager to fork a user endpoint (uep). On Lustre that took ~56 s to reach
   "Starting Engine", while the submit API gives up sooner and returns
       422 ENDPOINT_NOT_ONLINE "User endpoint did not signal readiness"
   whose text blames "a misconfiguration in the request" -- misleading; it is a
   cold start. Retrying is the fix.

2. EACH DISTINCT user_endpoint_config GETS ITS OWN uep. Changing any value (even
   adding nodes_per_block: 1) mints a new uep and pays the cold start again. So
   pin ONE canonical config and reuse it; only vary it when you genuinely want a
   different resource shape, and expect a cold start when you do.

Also seen and absorbed here: transient GlobusConnectionError and
"SDK thread quit unexpectedly" on the first attempts of a session.

OBSERVABILITY STARVATION (architectural, found the hard way):
polling is itself a Globus Compute task. Saturate every worker with science and
you can no longer observe your own runs -- the monitoring task queues behind the
work it is meant to watch. This is the concrete reason the reference
implementation (ramanathanlab/deepdrivewe-academy) runs a SEPARATE CPU endpoint
alongside the GPU one: analysis and status checks must not compete with the
science for the same slots. Fix: a second endpoint (`*-cpu`, few workers, no
accelerators) for polling/analysis, or reserve a worker. Until then, checkpoint
aggressively -- state you cannot observe must still be state you cannot lose.

MODULE GLOBALS DO NOT REACH THE WORKER:
a remote function referencing a module-level constant (`K_REST`) raises
`NameError: name 'K_REST' is not defined` ON THE WORKER, silently killing those
tasks while others succeed. Same root cause as the define-in-__main__ rule: only
the function body travels. Close over values as default arguments or pass them
explicitly. Partial batch loss is the dangerous shape here -- the batch "works".

DEFINE-IN-__main__ RULE (hit for real, costs a whole batch):
dill pickles a function IMPORTED from a module BY REFERENCE, so
    from gc_production import run_md   # then submit run_md
makes every worker die with `ModuleNotFoundError: No module named 'gc_production'`
even though the code is right there on the shared filesystem. The remote function
must be DEFINED IN the __main__ script you are executing (or the module must be
importable on the worker via PYTHONPATH). Same failure family as the return-type
rule below: the task never runs, and the error names a module rather than the
real cause.

RETURN-TYPE RULE (not enforceable here, but the most common own-goal):
remote functions must return PLAIN JSON-able types. `torch.__version__` is a
TorchVersion object that pickles BY REFERENCE to torch -- the task succeeds and
the *client* dies with `ModuleNotFoundError: No module named 'torch'`, which
reads like a remote failure and is not. Coerce with str() at the boundary.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from globus_compute_sdk import Executor
from globus_sdk.services.compute.errors import ComputeAPIError

ENDPOINT_ID = "c4456e74-942a-4410-a814-9be3e5e8a4fc"

# The canonical shape. Keep it stable so submissions reuse one warm uep.
# 2 nodes x 12 tiles = 24 pinned workers in ONE PBS job -- deliberately nodes
# rather than blocks, so we make a single queue entry instead of N competing
# against the per-user limit on jobs in 'Q' state (which is what refused our
# very first qsub).
CANONICAL_UEC: dict[str, Any] = {
    "queue": "capacity",
    "walltime": "12:00:00",
    "nodes_per_block": 2,
    "max_blocks": 1,
}

# ---------------------------------------------------------------------------
# THE WORKBENCH PROFILE  (adopt at the next natural restart, not mid-campaign)
#
# CANONICAL_UEC above uses min_blocks: 0, which makes idling free -- and is also
# what GIVES THE BLOCK AWAY. From parsl/jobs/strategy.py:
#
#     if active_tasks == 0:
#         if active_blocks <= min_blocks:   # min_blocks 0 -> False
#             ...take no action
#         else:                             # -> scale in after max_idletime
#
# So the moment a batch drains, the block is torn down and the next submission
# pays a fresh queue wait. That is correct for occasional batch jobs and WRONG
# for the intended model here: a continuously-available computational workbench
# that expensive code can be thrown at on demand, with no queue latency between
# pieces of work.
#
# WORKBENCH_UEC pins one block for the whole walltime. Cost is the idle
# node-hours (2 nodes x 12 h = 24) against a 367,205 node-hour allocation --
# negligible, and bought back the first time it saves a queue wait.
#
# ⚠ Switching profiles mints a NEW user endpoint (any UEC change does) and pays
# a ~60 s cold start, and does NOT rescue a block already running under the old
# profile. So switch at a natural boundary, never mid-batch.
WORKBENCH_UEC: dict[str, Any] = {
    "queue": "capacity",
    "walltime": "12:00:00",
    "nodes_per_block": 2,
    "min_blocks": 1,      # <-- the whole difference: capacity stays warm
    "max_blocks": 1,
}

_TRANSIENT = ("ENDPOINT_NOT_ONLINE", "ConnectionError", "SDK thread quit")


def submit_many(
    fn: Callable,
    args_list: Sequence[tuple],
    *,
    uec: dict[str, Any] | None = None,
    endpoint_id: str = ENDPOINT_ID,
    result_timeout: float = 3600,
    max_wait: float = 3000,
    verbose: bool = True,
) -> list[Any]:
    """Submit fn once per arg-tuple, tolerating cold starts. Returns results in order."""
    uec = CANONICAL_UEC if uec is None else uec
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with Executor(endpoint_id=endpoint_id, user_endpoint_config=uec) as gce:
                futs = [gce.submit(fn, *a) for a in args_list]
                if verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: "
                          f"{len(futs)} task(s) accepted", flush=True)
                return [f.result(timeout=result_timeout) for f in futs]
        except ComputeAPIError as e:
            if any(t in str(e) for t in _TRANSIENT):
                if verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: uep cold; "
                          f"retry in 30s", flush=True)
                time.sleep(30)
                continue
            raise
        except Exception as e:  # transient SDK/connection faults
            if any(t in str(e) for t in _TRANSIENT):
                if verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: "
                          f"{type(e).__name__}; retry in 30s", flush=True)
                time.sleep(30)
                continue
            raise
    raise TimeoutError(f"endpoint never became ready within {max_wait}s")


def submit_one(fn: Callable, *args, **kw) -> Any:
    return submit_many(fn, [args], **kw)[0]
