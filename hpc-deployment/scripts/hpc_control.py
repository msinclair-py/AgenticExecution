"""Stdio MCP server that lets the harness change the HPC allocation and restart mcp_server.py.

mcp_server.py reads its allocation (queue, walltime, nodes) once at startup and
holds one EL cluster on it, so changing the allocation means restarting that
process. This controller does it the safe way: SIGINT (never SIGKILL, which
leaks the allocation), wait for exit, then start a detached replacement with
the new settings. The harness launches this over stdio; mcp_server.py itself
stays an http server that survives harness restarts.

Environment:
  HPC_ENV_FILE        set_env script to source (required), e.g. set_env_akv.sh
  HPC_ALLOWED_QUEUES  comma list (default debug,debug-scaling,capacity)
  HPC_MAX_NODES       cap on nodes per allocation (default 4)
  HPC_MAX_WALLTIME    cap on walltime, HH:MM:SS (default 12:00:00)
  HPC_CONTROL_DIR     state, pid and log directory (default ~/.agentic-hpc)
  HPC_MCP_URL         where mcp_server.py listens (default http://127.0.0.1:8000/mcp)
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

SCRIPTS = Path(__file__).resolve().parent
SERVER = SCRIPTS / "mcp_server.py"
_WALLTIME = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)$")
# Aurora queue limits (qstat -Qf, 2026-10-20). prod needs >= 256 nodes, so it
# is not offered. Operator caps below apply on top of these.
QUEUE_LIMITS = {
    "debug": {"max_nodes": 2, "max_walltime": "01:00:00"},
    "debug-scaling": {"max_nodes": 256, "max_walltime": "01:00:00"},
    "capacity": {"max_nodes": 16, "max_walltime": "168:00:00"},
}
_STOP_WAIT_S = 180


def _seconds(walltime: str) -> int:
    match = _WALLTIME.match(walltime)
    if not match:
        raise ValueError(f"walltime must be HH:MM:SS, got {walltime!r}")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def limits() -> dict:
    allowed = [q.strip() for q in os.environ.get(
        "HPC_ALLOWED_QUEUES", "debug,debug-scaling,capacity").split(",") if q.strip()]
    return {
        "allowed_queues": allowed,
        "max_nodes": int(os.environ.get("HPC_MAX_NODES", "4")),
        "max_walltime": os.environ.get("HPC_MAX_WALLTIME", "12:00:00"),
        "queue_limits": {q: QUEUE_LIMITS[q] for q in allowed if q in QUEUE_LIMITS},
    }


def validate(queue: str, walltime: str, nodes: int, caps: dict | None = None) -> None:
    """Refuse an allocation outside the operator caps or the queue's own limits."""

    caps = caps or limits()
    if queue not in caps["allowed_queues"]:
        raise ValueError(f"queue {queue!r} not allowed; choose one of {caps['allowed_queues']}")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1:
        raise ValueError("nodes must be a positive integer")
    queue_caps = QUEUE_LIMITS.get(queue, {})
    max_nodes = min(caps["max_nodes"], queue_caps.get("max_nodes", caps["max_nodes"]))
    if nodes > max_nodes:
        raise ValueError(f"nodes={nodes} exceeds the cap of {max_nodes} for queue {queue!r}")
    max_wall = min(_seconds(caps["max_walltime"]),
                   _seconds(queue_caps.get("max_walltime", caps["max_walltime"])))
    if _seconds(walltime) > max_wall:
        cap = f"{max_wall // 3600:02d}:{max_wall % 3600 // 60:02d}:{max_wall % 60:02d}"
        raise ValueError(f"walltime {walltime} exceeds the cap of {cap} for queue {queue!r}")


def _control_dir() -> Path:
    # Not ~/.loam-hpc: the pid and allocation state here identify the server
    # THIS controller restarts. Sharing a directory with another checkout's
    # controller would let one SIGINT the other's server and its allocation.
    path = Path(os.environ.get("HPC_CONTROL_DIR") or Path.home() / ".agentic-hpc").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sourced_env(env_file: str) -> dict[str, str]:
    """The environment after sourcing the operator's set_env script."""

    out = subprocess.run(["bash", "-c", 'set -a; source "$1" >/dev/null; env -0', "_", env_file],
                         capture_output=True, check=True, timeout=30).stdout
    return dict(item.split("=", 1) for item in out.decode().split("\0") if "=" in item)


def current_settings(env: dict[str, str], saved: dict | None) -> dict:
    """Settings from the env file, overridden by the last hpc_restart choice."""

    config = json.loads(env["HPC_ENDPOINT_CONFIG_JSON"])
    settings = {
        "queue": config.get("queue", "debug"),
        "walltime": config.get("walltime", "01:00:00"),
        "nodes": int(env.get("HPC_NNODES") or config.get("nodes_per_block") or 1),
    }
    settings.update({k: v for k, v in (saved or {}).items() if k in settings})
    return settings


def child_env(env: dict[str, str], settings: dict) -> dict[str, str]:
    """Patch queue/walltime/nodes into the endpoint config; nodes must agree in both places."""

    env = dict(env)
    config = json.loads(env["HPC_ENDPOINT_CONFIG_JSON"])
    config.update(queue=settings["queue"], walltime=settings["walltime"],
                  nodes_per_block=settings["nodes"])
    env["HPC_ENDPOINT_CONFIG_JSON"] = json.dumps(config)
    env["HPC_NNODES"] = str(settings["nodes"])
    # This controller exists to resize the EL cluster's allocation, so the
    # server it starts must be in cluster mode. Without this it would inherit
    # HPC_EXECUTION from the env file -- or the "direct" default -- and start a
    # server that silently ignores both HPC_NNODES and the allocation change.
    env["HPC_EXECUTION"] = "cluster"
    env["HPC_MCP_TRANSPORT"] = "http"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SCRIPTS), env.get("PYTHONPATH", "")) if p)
    return env


def _server_pids() -> list[int]:
    # Match the repo-relative tail so a server started by hand from the repo
    # root (`python skills/hpc-deployment/scripts/mcp_server.py`) is found too.
    out = subprocess.run(["pgrep", "-f", "hpc-deployment/scripts/mcp_server.py"],
                         capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def _listening(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=2):
            return True
    except OSError:
        return False


def _tail(path: Path, lines: int = 15) -> str:
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    keep = [line for line in text if "POST /mcp" not in line and "GET /mcp" not in line]
    return "\n".join(keep[-lines:])[-3000:]


class Controller:
    def __init__(self) -> None:
        env_file = os.environ.get("HPC_ENV_FILE")
        if not env_file or not Path(env_file).expanduser().is_file():
            raise RuntimeError("HPC_ENV_FILE must name the set_env script to source")
        self.env_file = str(Path(env_file).expanduser())
        self.dir = _control_dir()
        self.state_path = self.dir / "allocation.json"
        self.log_path = self.dir / "mcp_server.log"
        self.url = os.environ.get("HPC_MCP_URL", "http://127.0.0.1:8000/mcp")

    def _saved(self) -> dict | None:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return None

    def status(self) -> dict:
        settings = current_settings(sourced_env(self.env_file), self._saved())
        pids = _server_pids()
        state = "listening" if pids and _listening(self.url) else (
            "starting (waiting for the PBS job before it listens)" if pids else "stopped")
        return {"state": state, "pids": pids, "settings": settings, "limits": limits(),
                "url": self.url, "log": str(self.log_path), "log_tail": _tail(self.log_path)}

    def _stop(self) -> None:
        pids = _server_pids()
        for pid in pids:
            os.kill(pid, signal.SIGINT)
        deadline = time.time() + _STOP_WAIT_S
        while _server_pids() and time.time() < deadline:
            time.sleep(2)
        if _server_pids():
            # Never escalate to SIGKILL: it skips the lifespan shutdown and
            # orphans the EL cluster on its allocation until walltime.
            raise RuntimeError(f"mcp_server.py did not exit within {_STOP_WAIT_S}s of SIGINT; "
                               "not force-killing it (that would leak the allocation)")

    def restart(self, queue: str | None, walltime: str | None, nodes: int | None) -> dict:
        env = sourced_env(self.env_file)
        settings = current_settings(env, self._saved())
        settings.update({k: v for k, v in (("queue", queue), ("walltime", walltime),
                                           ("nodes", nodes)) if v is not None})
        validate(settings["queue"], settings["walltime"], settings["nodes"])
        previous = _server_pids()
        self._stop()
        self.state_path.write_text(json.dumps(settings))
        log = open(self.log_path, "ab")
        log.write(f"\n=== hpc_restart {time.strftime('%Y-%m-%d %H:%M:%S')} {json.dumps(settings)}\n".encode())
        log.flush()
        # Own session: the server and its allocation outlive this controller
        # and the harness that spawned it.
        proc = subprocess.Popen([sys.executable, str(SERVER)], env=child_env(env, settings),
                                cwd=str(SCRIPTS), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        return {"restarted": True, "stopped_pids": previous, "pid": proc.pid, "settings": settings,
                "next": "The server listens only once its PBS job runs: poll hpc_allocation_status "
                        "until state is 'listening' (queue waits can be long), then /reload-mcp or "
                        "just retry the hpc tools. Rows of batches that were running are lost; "
                        "resume them with run_batch(batch_id=...)."}


def build_server():
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    controller = Controller()
    server = FastMCP(name="hpc-control", instructions=(
        "Controls the HPC allocation behind the hpc MCP server. hpc_allocation_status shows whether "
        "it is up and its queue/walltime/nodes; hpc_restart changes them and restarts it. A restart "
        "drops the current allocation and waits in the PBS queue again, so restart only when the job "
        "shape must change or the server is stuck, never while batches you need are running."))

    @server.tool
    def hpc_allocation_status() -> dict:
        """Is the hpc server listening, starting (waiting for PBS) or stopped; its queue, walltime and nodes; the allowed limits; the recent log."""
        try:
            return controller.status()
        except Exception as exc:
            raise ToolError(f"hpc_allocation_status failed: {type(exc).__name__}: {exc}") from exc

    @server.tool
    def hpc_restart(queue: str | None = None, walltime: str | None = None,
                    nodes: int | None = None) -> dict:
        """Restart the hpc server, optionally on a new allocation; unset fields keep their current value.

        queue: one of the allowed queues (see hpc_allocation_status). Aurora: debug = 1-2 nodes
        up to 1 h for quick checks; debug-scaling = up to 1 h; capacity = 1-16 nodes up to 168 h
        for campaigns. walltime: HH:MM:SS. nodes: whole nodes (12 tiles each on Aurora); every
        node multiplies how many run_batch rows run at once. The choice persists across later
        restarts. Requests over the operator caps are refused. Stops the old server with SIGINT
        (its allocation is released), starts the new one detached, and returns at once.
        """
        try:
            return controller.restart(queue, walltime, nodes)
        except Exception as exc:
            raise ToolError(f"hpc_restart failed: {type(exc).__name__}: {exc}") from exc

    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
