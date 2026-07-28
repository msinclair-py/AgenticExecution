"""Submit with retry across the user-endpoint readiness race.

In the 4.x manager model the FIRST submit is what causes the manager to fork a
user endpoint. On a filesystem as slow as Lustre that uep took ~56 s to reach
'Starting Engine', while the submit API gives up much sooner and returns
    422 ENDPOINT_NOT_ONLINE "User endpoint did not signal readiness"
which reads like a misconfiguration but is really a cold start. Retrying is the
correct response: the first submit warms the uep, a later one lands.
"""
import time, sys
from globus_compute_sdk import Executor
from globus_sdk.services.compute.errors import ComputeAPIError

EP = "c4456e74-942a-4410-a814-9be3e5e8a4fc"
# Passed explicitly rather than relying on template defaults, so the log
# records exactly what shape was requested.
UEC = {"queue": "capacity", "walltime": "12:00:00", "max_blocks": 1}

def probe():
    import os, socket, sys
    return {"hostname": socket.gethostname(), "python": sys.version.split()[0],
            "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK", "<unset>"),
            "PBS_JOBID": os.environ.get("PBS_JOBID", "<unset>"),
            "TMPDIR": os.environ.get("TMPDIR", "<unset>"),
            "cwd": os.getcwd()}

deadline = time.time() + 3000
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        with Executor(endpoint_id=EP, user_endpoint_config=UEC) as gce:
            fut = gce.submit(probe)
            print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: accepted; "
                  f"awaiting PBS block (task pends until a Q slot frees)", flush=True)
            res = fut.result(timeout=max(60, deadline - time.time()))
        print("\n*** TASK RETURNED ***")
        for k, v in res.items():
            print(f"  {k:18} {v}")
        sys.exit(0)
    except ComputeAPIError as e:
        if "ENDPOINT_NOT_ONLINE" in str(e):
            print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: uep still cold; retry in 30s", flush=True)
            time.sleep(30); continue
        print("API error:", e); sys.exit(1)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        time.sleep(30)
print("TIMEOUT"); sys.exit(3)
