"""Pinned Academy agent for hpc-deployment.

Public actions are deliberately narrow:
- run_code: run an LLM-authored Python payload in a subprocess.
- inspect_filesystem: bounded read-only ls/cat/tail under explicit roots.
- run_batch (cluster mode): run one script once per row of a JSONL file as
  one EL task per row; returns a batch_id at once.
- batch_status / list_batches: read a batch's progress and the batch
  registry from disk.
- register_tool / list_tools: a library of frozen, versioned, smoke-tested
  scripts with typed params that run_batch(tool=...) can run by name.

Scheduler operations, generic shell execution, environment dumps, and file
mutation are intentionally absent. The endpoint/Parsl provider owns allocation
and accelerator pinning before this process starts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiohttp
from academy.agent import Agent, action

logger = logging.getLogger(__name__)

try:
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import (
        LauncherConfig,
        PolicyConfig,
        aurora_config,
        polaris_config,
    )
    from ensemble_launcher.ensemble import Task
    from ensemble_launcher.orchestrator import ClusterClient
    EL_AVAILABLE=True
except ModuleNotFoundError:
    EL_AVAILABLE=False

_MAX_ENTRIES = 200
_MAX_LINES = 200
_MAX_BYTES = 64 * 1024

# A dropped exchange stream must not read as "the agent is gone".
_TRANSIENT_LISTEN_ERRORS = (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError)
# The Academy exchange is reached over the ALCF HTTP proxy, which severs the
# long-poll stream routinely (observed in production: repeated ClientPayloadError
# / TransferEncodingError 400 against exchange.academy-agents.org via
# proxy.alcf.anl.gov, one every few seconds). Those drops are NORMAL; they must
# never accumulate into a false "agent lost" that tears the whole EL cluster --
# and its PBS allocation -- down. A fixed count of consecutive retries is the
# wrong budget: the proxy drops faster than any "healthy for N seconds" reset, so
# the count is reached in under a minute even while the exchange is perfectly
# alive (this is what zombied a live capacity block). So the budget is TIME-based
# and resets on every delivered message: keep reconnecting indefinitely as long
# as messages still flow, and give up only after the exchange has produced
# nothing at all for this long. Env-tunable.
_LISTEN_GIVEUP_S = float(os.environ.get("LOAM_EXCHANGE_GIVEUP_S", "1200"))  # 20 min of silence
_LISTEN_BACKOFF_CAP = 8.0


def _listen_should_give_up(first_failure: float | None, now: float, giveup_s: float = _LISTEN_GIVEUP_S) -> bool:
    """True once the exchange has delivered NOTHING for longer than the budget.

    `first_failure` is the monotonic time of the first drop since the last
    delivered message, or None if a message has arrived since (so a stream that
    delivers at least one message per `giveup_s` never gives up -- it is flaky,
    not dead).
    """
    return first_failure is not None and (now - first_failure) >= giveup_s


def _listen_backoff(failures: int, cap: float = _LISTEN_BACKOFF_CAP) -> float:
    """Exponential backoff for consecutive reconnects (1,2,4,8...), capped."""
    return min(2.0 ** min(max(failures - 1, 0), 3), cap)


def _harden_exchange_listener() -> None:
    """Reconnect the exchange stream instead of dying on a transient drop.

    HttpExchangeTransport.listen already reconnects after each clean stream
    expiry (the server closes it every request_timeout_s), but it reads the
    response body *inside* the request context manager, so a truncated stream
    raises ClientPayloadError past the reconnect loop. That kills the listener
    task, which shuts down the agent runtime, which runs agent_on_shutdown.
    For HpcClusterAgent that calls el.stop(), so a momentary network blip
    tears down a live EL cluster and its allocation.

    Cluster lifetime is deliberately tied to agent lifetime, so the fix is to
    make agent death mean something: ride out blips, but still propagate once
    the exchange is persistently unreachable and the agent really is lost.
    """
    from academy.exchange.cloud.client import HttpExchangeTransport

    original = HttpExchangeTransport.listen
    if getattr(original, '_hpc_hardened', False):
        return

    async def listen(self, timeout: float | None = None):
        first_failure: float | None = None  # monotonic time of first drop since last message
        failures = 0
        while True:
            try:
                # aclosing: the inner generator owns an open HTTP response, so
                # it must be finalized if our consumer abandons us mid-stream.
                async with aclosing(original(self, timeout=timeout)) as stream:
                    async for message in stream:
                        # A delivered message proves the exchange is alive, so
                        # clear the failure budget: routine proxy drops between
                        # messages never accumulate toward a false "agent lost".
                        first_failure = None
                        failures = 0
                        yield message
                return
            except _TRANSIENT_LISTEN_ERRORS as exc:
                now = time.monotonic()
                if first_failure is None:
                    first_failure = now
                failures += 1
                if _listen_should_give_up(first_failure, now):
                    logger.error(
                        'exchange stream unrecoverable: no message for %.0fs across %d reconnects: %s: %s',
                        now - first_failure, failures, type(exc).__name__, exc,
                    )
                    raise
                backoff = _listen_backoff(failures)
                logger.warning(
                    'exchange stream dropped (%s: %s); reconnecting in %.0fs [silent %.0fs/%.0fs]',
                    type(exc).__name__, exc, backoff, now - first_failure, _LISTEN_GIVEUP_S,
                )
                await asyncio.sleep(backoff)

    listen._hpc_hardened = True
    HttpExchangeTransport.listen = listen


_harden_exchange_listener()


class HpcAgent(Agent):
    def __init__(self, campaign_id: str, base_dir: str, inspect_roots: list[str] | None = None) -> None:
        super().__init__()
        self._inflight = 0
        self.campaign_id = campaign_id
        self.base_dir = str(Path(base_dir).resolve())
        self.inspect_roots = inspect_roots or []

    @action
    async def probe(self) -> dict[str, str]:
        """Internal pool-health check; do not expose as an MCP tool."""
        return {
            "pid": str(os.getpid()),
            "campaign_id": self.campaign_id,
            "device_pin": os.environ.get("ZE_AFFINITY_MASK", os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        }

    @action
    async def run_code(self, payload: str, workdir: str | None = None,
                       timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
        self._inflight += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _run_subprocess, payload, workdir or self.base_dir, timeout, venv_path
            )
        finally:
            self._inflight -= 1

    @action
    async def inspect_filesystem(self, operation: str, path: str,
                                 offset: int = 1, limit: int = 100) -> dict:
        """Bounded read-only worker filesystem inspection: ls, cat, or tail."""
        return await asyncio.get_running_loop().run_in_executor(
            None, _inspect, operation, path, offset, limit, self.base_dir, self.inspect_roots
        )

    @action
    async def batch_status(self, batch_id: str) -> dict:
        """Progress of one run_batch batch, counted from its run folders."""
        return await asyncio.get_running_loop().run_in_executor(
            None, _batch_status, batch_id, self.base_dir
        )

    @action
    async def list_batches(self, plan_id: str | None = None, hypothesis_id: str | None = None,
                           limit: int = 50) -> list[dict]:
        """Registry entries, newest first, filtered by Loam labels."""
        return await asyncio.get_running_loop().run_in_executor(
            None, _list_batches, self.base_dir, plan_id, hypothesis_id, limit
        )

    @action
    async def register_tool(self, name: str, script_path: str, params: dict, description: str,
                            example: dict, defaults: dict | None = None,
                            call: Literal["json", "argv"] = "json", venv_path: str | None = None,
                            skip_smoke: bool = False, labels: dict | None = None,
                            expects: list[str] | None = None) -> dict:
        """Freeze a script as the next version of a named tool, after one smoke run.

        skip_smoke is for scripts that need a GPU the client node may lack: the
        tool is registered with tested=False. `expects` is a list of output
        filenames/globs (relative to each run's folder) the script must produce;
        a run that exits 0 without them is counted as failed (L3), so a fold
        that exits clean but writes nothing never travels downstream as a
        datapoint.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, _register_tool, name, script_path, params, description, example, defaults,
            call, venv_path, skip_smoke, labels, expects, self.base_dir, self.inspect_roots,
        )

    @action
    async def list_tools(self, query: str | None = None, name: str | None = None,
                         all_versions: bool = False, limit: int = 50) -> list[dict]:
        """Registered tools, newest version of each unless all_versions."""
        return await asyncio.get_running_loop().run_in_executor(
            None, _list_tools, self.base_dir, query, name, all_versions, limit
        )


class HpcClusterAgent(Agent):
    """
    Agent to manage the lifecyle of EL cluster
    """
    def __init__(self, 
                 base_dir: str, 
                 ckpt_dir: str,
                 nnodes: int,
                 system: Literal["aurora", "polaris"],
                 task_executor_name : list[str] = ["async_processpool"]
                  
                 ):
        super().__init__()
        self.ckpt_dir = str(Path(base_dir, ckpt_dir))
        if system == "aurora":
            self.system_config = aurora_config
            gpu_selector = "ZE_AFFINITY_MASK"
        elif system == "polaris":
            self.system_config = polaris_config
            gpu_selector = "CUDA_VISIBLE_DEVICES"
        else:
            raise ValueError(f"unsupported system {system!r}; expected 'aurora' or 'polaris'")

        self.launcher_config = LauncherConfig(
            child_executor_name="async_mpi",
            task_executor_name=task_executor_name,
            cluster=True,
            checkpoint_dir=self.ckpt_dir,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
            gpu_selector=gpu_selector,
            master_logs=True,
            children_scheduler_policy="fixed_leafs_children_policy",
            policy_config=PolicyConfig(nlevels=2, leaf_nodes=nnodes),
            log_dir=str(Path(base_dir,"logs")),
        )
        self._el = None

    async def agent_on_startup(self):
        self._el = EnsembleLauncher(ensemble_file={},
                                    system_config=self.system_config, 
                                    launcher_config=self.launcher_config)
        self._el.start()

    @action
    async def probe(self) -> dict[str, str]:
        """Health check"""
        return {
            "pid": str(os.getpid()),
            "device_pin": os.environ.get("ZE_AFFINITY_MASK", os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        }
        
    async def agent_on_shutdown(self):
        if self._el is not None:
            self._el.stop()

class HpcClientAgent(HpcAgent):
    def __init__(self, campaign_id: str, base_dir: str, ckpt_dir: str, inspect_roots: list[str] | None = None) -> None:
        super().__init__(campaign_id, base_dir, inspect_roots)
        self.ckpt_dir = str(Path(self.base_dir, ckpt_dir))
        self._client = None
        # Holds each batch's futures so they stay referenced. Progress is never
        # read from here: batch_status reads disk, since round-robin dispatch
        # and pool repair mean the agent asking is rarely the one that submitted.
        self._batches: dict[str, list] = {}

    async def agent_on_startup(self):
        assert EL_AVAILABLE, "EL is not available"
        # ClusterClient takes no log_dir; EL resolves one from EL_LOGDIR, else
        # <cwd>/logs. A Globus Compute worker's cwd is the endpoint's
        # tasks_working_dir, so without this the client log lands there instead
        # of beside the master's, which HpcClusterAgent places under base_dir.
        os.environ.setdefault("EL_LOGDIR", str(Path(self.base_dir, "logs")))
        self._client = ClusterClient(checkpoint_dir=self.ckpt_dir, checkpoint_timeout=300)
        self._client.start()

    async def agent_on_shutdown(self):
        if self._client is not None:
            self._client.stop()


    def _create_task(self, payload: str, workdir: str, timeout: float, venv_path: str | None = None):
        return Task(
            task_id=uuid.uuid4().hex,
            nnodes=1,
            ppn=1,
            ngpus_per_process=1,
            executable=_run_subprocess,
            args=(payload, workdir or self.base_dir, timeout, venv_path)
        )

    @action
    async def run_code(self, payload: str, workdir: str | None = None,
                       timeout: float = 300.0, venv_path: str | None = None) -> dict[str, str]:
        self._inflight += 1
        try:
            return await asyncio.wrap_future(
                self._client.submit(self._create_task(payload, workdir, timeout, venv_path))
            )
        finally:
            self._inflight -= 1

    @action
    async def run_batch(self, script_path: str | None = None, inputs_path: str | None = None,
                        timeout_each: float = 900.0, gpus: float = 1,
                        venv_path: str | None = None, batch_id: str | None = None,
                        labels: dict | None = None, tool: str | None = None,
                        tool_version: int | None = None) -> dict:
        """Submit one EL task per inputs row and return without waiting.

        A new batch runs either script_path or a registered tool, and freezes
        copies of the script and inputs and registers itself; passing an
        existing batch_id resumes it from those copies, skipping runs that
        already finished cleanly.

        H4: gpus, timeout_each and venv_path are frozen on the first submission
        and read back from the registry entry here, so a bare resume keeps the
        batch's real accelerator pinning, walltime and interpreter instead of
        silently reverting to the defaults (gpus=1, timeout=900s, system python)
        -- which mis-pinned multi-tile folds, killed long folds at 900s, and ran
        resumed rows under the wrong environment. The entry is the single source
        of truth: on a new batch it holds exactly the values passed; on a resume
        it wins over whatever this call passes.
        """
        entry, script, todo = await asyncio.get_running_loop().run_in_executor(
            None, _prepare_batch, script_path, inputs_path, batch_id, labels,
            timeout_each, gpus, venv_path, self.base_dir, self.inspect_roots, tool, tool_version,
        )
        batch_id = entry["batch_id"]
        # Use the frozen run parameters, not this call's args (H4). .get keeps
        # pre-H4 batches (which have no frozen field) resumable on the passed args.
        timeout_each = float(entry.get("timeout_each", timeout_each))
        gpus = entry.get("gpus", gpus)
        venv_path = entry.get("venv_path", venv_path)
        expects = entry.get("expects")  # L3: frozen output contract, or None
        # The call style belongs to the frozen script, so resume reuses it.
        call, order = entry.get("call", "json"), entry.get("param_order")
        tasks = [
            Task(
                task_id=f"{batch_id}-{run_id}",
                nnodes=1,
                ppn=1,
                ngpus_per_process=gpus,
                executable=_run_one,
                args=(script, row, run_dir, timeout_each, venv_path, call, order,
                      entry["script_sha256"], expects),
            )
            for run_id, row, run_dir in todo
        ]
        # submit_batch only buffers the tasks and hands back futures; EL's
        # master spreads them over the nodes and GPUs.
        if tasks:
            self._batches[batch_id] = self._client.submit_batch(tasks)
        return {
            "batch_id": batch_id,
            "total": entry["total"],
            "submitted": len(tasks),
            "already_done": entry["total"] - len(tasks),
            "labels": entry["labels"],
        }


def _roots(base_dir: str, inspect_roots: list[str]) -> list[Path]:
    configured = [base_dir, *inspect_roots, "/tmp"]
    return [Path(item).resolve() for item in configured if item]


def _safe_path(raw: str, base_dir: str, inspect_roots: list[str]) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    roots = _roots(base_dir, inspect_roots)
    if not any(path == root or root in path.parents for root in roots):
        # Name the roots. A caller that only learns it guessed wrong guesses
        # again: an agent asked for `/workspace` (a container convention that
        # does not exist here) and two other outside paths in eight seconds
        # (2026-10-21). Each refusal was correct, but blind, so it produced
        # another attempt -- and a run of refusals used to trip the MCP
        # connectivity breaker, which told every agent in the process that this
        # healthy server was unreachable. Listing the roots turns a guessing
        # loop into one corrected call.
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(
            f"path {str(path)!r} is outside allowed inspection roots; "
            f"inspect only within: {allowed}"
        )
    if path.is_symlink():  # resolve() normally removes this; retain explicit policy.
        raise ValueError("symlink paths are not inspectable")
    if not (path.is_file() or path.is_dir()):
        raise ValueError("only regular files and directories are inspectable")
    return path


def _clamp(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, _MAX_LINES)


def _read_text_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError("cat/tail require a regular file")
    if path.stat().st_size > _MAX_BYTES:
        # Read only bounded bytes so inspection cannot pull an arbitrary log.
        with path.open("rb") as handle:
            data = handle.read(_MAX_BYTES)
    else:
        data = path.read_bytes()
    if b"\0" in data:
        raise ValueError("binary files are not inspectable")
    return data.decode("utf-8", errors="replace").splitlines()


def _tail_lines(path: Path, count: int) -> list[str]:
    """Last `count` lines of a text file, reading at most _MAX_BYTES from its end."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _MAX_BYTES))
            data = handle.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-count:]


def _inspect(operation: str, raw_path: str, offset: int, limit: int,
             base_dir: str, inspect_roots: list[str]) -> dict:
    path = _safe_path(raw_path, base_dir, inspect_roots)
    limit = _clamp(limit)
    if operation == "ls":
        if not path.is_dir():
            raise ValueError("ls requires a directory")
        entries = []
        for item in sorted(path.iterdir(), key=lambda x: x.name)[:_MAX_ENTRIES]:
            stat = item.stat(follow_symlinks=False)
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file" if item.is_file() else "other",
                "size_bytes": stat.st_size,
            })
        return {"operation": "ls", "path": str(path), "entries": entries,
                "truncated": len(list(path.iterdir())) > _MAX_ENTRIES}
    lines = _read_text_lines(path)
    if operation == "cat":
        if offset < 1:
            raise ValueError("offset must be at least 1")
        start = offset - 1
        selected = lines[start:start + limit]
        first = start + 1
    elif operation == "tail":
        selected = lines[-limit:]
        first = max(1, len(lines) - len(selected) + 1)
    else:
        raise ValueError("operation must be one of: ls, cat, tail")
    return {"operation": operation, "path": str(path), "total_lines": len(lines),
            "lines": [{"number": first + i, "text": text} for i, text in enumerate(selected)]}


def _run_subprocess(payload: str, workdir: str | None, timeout: float, venv_path: str | None = None) -> dict[str, str]:
    import subprocess
    import sys
    import tempfile

    cwd = Path(workdir or os.environ.get("HPC_BASE_DIR", ".")).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=cwd) as handle:
        handle.write(payload)
        script = Path(handle.name)
    if venv_path is not None:
        exec = str(Path(venv_path, "bin/python"))
    else:
        exec = sys.executable

    try:
        proc = subprocess.run([exec, str(script)], cwd=cwd, env=os.environ.copy(),
                              capture_output=True, timeout=timeout)
        return {"stdout": proc.stdout.decode(errors="replace"),
                "stderr": proc.stderr.decode(errors="replace"),
                "returncode": str(proc.returncode)}
    except subprocess.TimeoutExpired as exc:
        return {"stdout": (exc.stdout or b"").decode(errors="replace"),
                "stderr": ((exc.stderr or b"").decode(errors="replace") + f"\nTimed out after {timeout}s.").strip(),
                "returncode": "-1"}
    finally:
        script.unlink(missing_ok=True)


# -- run_batch -------------------------------------------------------------
#
# Layout under HPC_BATCH_ROOT (default <HPC_BASE_DIR>/batches):
#   registry/<batch_id>.json   one entry per batch: labels, hashes, last status
#   <batch_id>/script.py       frozen copy of the submitted script
#   <batch_id>/inputs.jsonl    frozen copy of the submitted inputs
#   <batch_id>/batch.json      same content as the registry entry
#   <batch_id>/run_<id>/       input.json, stdout.txt, stderr.txt,
#                              status.json, then done (written last)
#
# One file per batch rather than a shared database: many nodes write here,
# and file locking on Lustre depends on mount options. Every JSON write is
# tmp-file + os.replace so readers never see a partial file.

_MAX_BATCH_ROWS = int(os.environ.get("HPC_BATCH_MAX_ROWS", "100000"))
# Leading alphanumeric: rejects "." and "..", which would escape the batch root.
_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _batch_root(base_dir: str) -> Path:
    return Path(os.environ.get("HPC_BATCH_ROOT") or Path(base_dir, "batches")).resolve()


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_id(value: str, what: str) -> str:
    if not _BATCH_ID.fullmatch(value):
        raise ValueError(f"bad {what}: {value!r}")
    return value


def _read_rows(path: Path) -> list[tuple[str, dict]]:
    """Parse a JSONL inputs file into (run_id, row) pairs; `id` is optional."""
    rows: list[tuple[str, dict]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"inputs line {number}: not a JSON object")
            run_id = _check_id(str(row.get("id", number)), f"id on inputs line {number}")
            if run_id in seen:
                raise ValueError(f"inputs line {number}: duplicate id {run_id!r}")
            seen.add(run_id)
            rows.append((run_id, row))
    if not rows:
        raise ValueError("inputs file has no rows")
    if len(rows) > _MAX_BATCH_ROWS:
        raise ValueError(f"{len(rows)} rows exceeds HPC_BATCH_MAX_ROWS={_MAX_BATCH_ROWS}")
    return rows


def _clean_labels(labels: dict | None) -> dict[str, str]:
    labels = labels or {}
    if not isinstance(labels, dict) or len(labels) > 16:
        raise ValueError("labels must be a JSON object with at most 16 keys")
    return {str(key)[:64]: str(value)[:200] for key, value in labels.items()}


def _run_succeeded(status: dict) -> bool:
    """A run counts as a success only if it exited 0 AND produced every output
    its tool declared (L3). rc0 with a missing declared output is a failure: a
    fold that exits clean but writes nothing is not a datapoint.
    """
    return status.get("returncode") == 0 and not status.get("missing_outputs")


def _finished_ok(run_dir: Path) -> bool:
    if not (run_dir / "done").exists():
        return False
    try:
        return _run_succeeded(json.loads((run_dir / "status.json").read_text()))
    except (OSError, ValueError):
        return False


# A run folder can only be legitimately owned while its task is alive, and a
# task cannot outlive its own `timeout` (subprocess.run kills it). So a lease
# older than timeout + this margin means the owner died and the row is safe to
# re-dispatch. The margin absorbs clock skew and the gap between the subprocess
# being killed and its wrapper returning.
LEASE_STALE_MARGIN = 300.0


def _lease_age(run_dir: Path) -> float | None:
    """Seconds since this run folder was leased, or None if unleased/unreadable."""
    try:
        info = json.loads((run_dir / "lease.json").read_text())
        started = float(info.get("started_at"))
    except (OSError, ValueError, TypeError):
        return None
    return time.time() - started


def _active_lease(run_dir: Path, max_age: float) -> bool:
    """Whether a live task currently owns this run folder (C3).

    `done` is written last, so its presence means the task is terminal and the
    lease is irrelevant. Otherwise a lease younger than `max_age` marks a task
    still in flight -- resume must not dispatch a second task onto its folder.
    """
    if (run_dir / "done").exists():
        return False
    age = _lease_age(run_dir)
    return age is not None and age < max_age


def _acquire_lease(folder: Path, max_age: float) -> dict | None:
    """Atomically claim a run folder. Returns our lease, or None if a live task
    already owns it (the caller must then NOT run, to avoid clobbering it).

    O_CREAT|O_EXCL makes the claim atomic between two tasks that were both
    dispatched for the same row (e.g. a resume overlapping an in-flight run):
    the second to start loses the race, sees a fresh lease, and backs off. A
    clearly-stale lease (dead owner) is stolen so a crashed row can still rerun.
    """
    import socket

    lease = folder / "lease.json"
    me = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time()}
    payload = json.dumps(me).encode()
    for _ in range(2):
        try:
            fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # A terminal run (`done` written) no longer owns its folder, so its
            # lease is reclaimable at once -- otherwise a just-failed row could
            # not be re-run until its own lease aged past max_age. A live run
            # (no done) is honored until its lease goes stale.
            if not (folder / "done").exists():
                age = _lease_age(folder)
                if age is not None and age < max_age:
                    return None  # a live owner holds it; back off
            try:
                lease.unlink()  # stale, terminal, or unreadable; drop it and retry
            except OSError:
                return None
            continue
        else:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            return me
    return None


def _stage_local(script: str, sha: str | None) -> str:
    """Return a node-local copy of a frozen script, copying it on first use.

    Thousands of runs starting python on one Lustre file is a metadata storm;
    /tmp is per node, so each node reads the shared copy once. The sha in the
    path means a stale copy can't be picked up. Any failure falls back to the
    shared copy.
    """
    if not sha:
        return script
    local = Path(os.environ.get("HPC_LOCAL_STAGE") or "/tmp/loam-tools", sha, "script.py")
    try:
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_name(f".script.{uuid.uuid4().hex}")
            shutil.copyfile(script, tmp)
            os.replace(tmp, local)  # racing runs copy the same bytes; last rename wins
        return str(local)
    except OSError:
        return script


def _argv(row: dict, order: list[str] | None) -> list[str]:
    """`--name value` pairs in params order; lists, dicts, bools, null as JSON."""
    argv = []
    for name in order or [key for key in row if key != "id"]:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, (list, dict, bool)) or value is None:
            value = json.dumps(value)
        argv += [f"--{name}", str(value)]
    return argv


# Files the run bookkeeping owns at the top of a run folder; a promoted output
# that collides with one of these names is left in the attempt folder instead.
_RUN_RESERVED = {"input.json", "stdout.txt", "stderr.txt", "status.json", "done", "lease.json"}


def _promote_outputs(attempt: Path, folder: Path) -> None:
    """Move a successful attempt's output files up to the run-folder top level,
    where harvesters look, without clobbering the run bookkeeping (M6)."""
    for item in attempt.iterdir():
        if item.name in _RUN_RESERVED or item.name.startswith("attempt_"):
            continue
        dest = folder / item.name
        try:
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            shutil.move(str(item), str(dest))
        except OSError:
            continue


def _run_one(script: str, row: dict, run_dir: str, timeout: float,
             venv_path: str | None = None, call: str = "json",
             order: list[str] | None = None, sha: str | None = None,
             expects: list[str] | None = None) -> dict:
    """Run one batch row inside an EL task; only a small status dict returns.

    The script is invoked as `python script input.json` (call="json") or
    `python script --name value ...` (call="argv") with a fresh per-attempt
    subfolder as cwd. stdout/stderr and the status live at the run-folder top
    level so batch_status can read them; the script's own output files are
    written into the attempt subfolder and PROMOTED to the top level only on
    success (M6), so a failed attempt can never leave a result artifact at the
    top level that a globbing harvester would read as this run's output. If the
    tool declared expected outputs (`expects`), a run that exits 0 without
    producing them is recorded as a failure (L3). `done` is written last, so it
    means "finished".
    """
    import socket
    import subprocess
    import sys

    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # C3: claim the folder before touching anything. If a live task already owns
    # it (a resume that overlapped an in-flight run), back off WITHOUT clearing
    # its done/outputs, rather than running a second process into the same dir.
    if _acquire_lease(folder, timeout + LEASE_STALE_MARGIN) is None:
        return {"returncode": None, "skipped": "run folder actively leased by another task",
                "host": socket.gethostname()}
    (folder / "done").unlink(missing_ok=True)  # a rerun after a failure
    # M6: give each attempt its own cwd. The top level only ever holds promoted
    # (successful) outputs plus the bookkeeping; a failed attempt's files stay
    # under attempt_<n>/ for debugging and are never seen by a harvester.
    attempt_n = 1 + sum(1 for p in folder.glob("attempt_*") if p.is_dir())
    attempt = folder / f"attempt_{attempt_n}"
    attempt.mkdir(parents=True, exist_ok=True)
    (folder / "input.json").write_text(json.dumps(row))
    (attempt / "input.json").write_text(json.dumps(row))
    python = str(Path(venv_path, "bin/python")) if venv_path else sys.executable
    args = _argv(row, order) if call == "argv" else ["input.json"]
    started = time.time()
    with open(folder / "stdout.txt", "wb") as out, open(folder / "stderr.txt", "wb") as err:
        try:
            returncode = subprocess.run([python, _stage_local(script, sha), *args], cwd=attempt,
                                        stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            err.write(f"\nTimed out after {timeout}s.".encode())
            returncode = -1
    status = {
        "returncode": returncode,
        "attempt": attempt_n,
        "seconds": round(time.time() - started, 1),
        "host": socket.gethostname(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("ZE_AFFINITY_MASK", "")),
    }
    # L3: enforce the tool's declared output contract on an otherwise-clean exit.
    if returncode == 0 and expects:
        missing = [pattern for pattern in expects if not any(True for _ in attempt.glob(pattern))]
        if missing:
            status["missing_outputs"] = missing
    if _run_succeeded(status):
        _promote_outputs(attempt, folder)
    _write_json(folder / "status.json", status)  # M5: atomic, so readers never tear it
    (folder / "done").touch()
    return status


def _prepare_batch(script_path: str | None, inputs_path: str | None, batch_id: str | None,
                   labels: dict | None, timeout_each: float, gpus: float,
                   venv_path: str | None, base_dir: str, inspect_roots: list[str],
                   tool: str | None = None,
                   tool_version: int | None = None) -> tuple[dict, str, list]:
    """Create or resume a batch on disk and list the runs still to submit.

    New: validate, freeze copies of script (or the tool's script) and inputs,
    register. With a tool, every row is checked against its params first.
    Resume (registry entry exists): reuse the frozen copies, ignore new paths
    and tool args, and refuse labels that name a different plan or hypothesis.
    """
    root = _batch_root(base_dir)
    registry = root / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    batch_id = _check_id(batch_id or f"b-{uuid.uuid4().hex[:12]}", "batch_id")
    labels = _clean_labels(labels)
    entry_path = registry / f"{batch_id}.json"
    batch_dir = root / batch_id

    if entry_path.exists():
        entry = json.loads(entry_path.read_text())
        for key in ("plan_id", "hypothesis_id"):
            if labels.get(key) and labels[key] != entry["labels"].get(key):
                raise ValueError(f"batch {batch_id} belongs to another {key}")
    else:
        if bool(script_path) == bool(tool) or not inputs_path:
            raise ValueError("a new batch needs inputs_path and exactly one of script_path or tool")
        spec = _load_tool(base_dir, tool, tool_version) if tool else None
        script = (Path(spec["dir"], "script.py") if spec
                  else _safe_path(script_path, base_dir, inspect_roots))
        inputs = _safe_path(inputs_path, base_dir, inspect_roots)
        if not (script.is_file() and inputs.is_file()):
            raise ValueError("script_path and inputs_path must be files")
        # Reject bad inputs before creating anything.
        for run_id, row in _read_rows(inputs):
            if spec:
                _check_row(row, spec, f"inputs row {run_id!r}")
        batch_dir.mkdir(parents=True)  # a fresh id must not collide
        shutil.copy2(script, batch_dir / "script.py")
        shutil.copy2(inputs, batch_dir / "inputs.jsonl")
        entry = {
            "batch_id": batch_id,
            "created_at": _now(),
            "labels": labels,
            "source_script": str(script),
            "source_inputs": str(inputs),
            "script_sha256": _sha256(batch_dir / "script.py"),
            "inputs_sha256": _sha256(batch_dir / "inputs.jsonl"),
            "timeout_each": timeout_each,
            "gpus": gpus,
            "venv_path": venv_path,  # H4: frozen so a resume keeps the interpreter
        }
        if spec:
            entry.update(tool=spec["tool"], tool_version=spec["version"],
                         tool_sha256=spec["sha256"], call=spec["call"],
                         param_order=list(spec["params"]),
                         expects=spec.get("expects"),  # L3: frozen output contract
                         tool_spec={key: spec[key] for key in ("params", "defaults")})

    rows = _read_rows(batch_dir / "inputs.jsonl")
    if entry.get("tool_spec"):
        rows = [(run_id, _check_row(row, entry["tool_spec"], f"inputs row {run_id!r}"))
                for run_id, row in rows]
    # C3: skip rows that finished cleanly AND rows a live task still owns, so a
    # resume that overlaps an in-flight submission does not dispatch a second
    # task onto the same run folder. Staleness uses the batch's frozen timeout.
    lease_max_age = float(entry.get("timeout_each", timeout_each)) + LEASE_STALE_MARGIN
    todo = [(run_id, row, str(batch_dir / f"run_{run_id}"))
            for run_id, row in rows
            if not _finished_ok(batch_dir / f"run_{run_id}")
            and not _active_lease(batch_dir / f"run_{run_id}", lease_max_age)]
    entry["total"] = len(rows)
    entry.setdefault("submissions", []).append({"at": _now(), "submitted": len(todo)})
    _write_json(batch_dir / "batch.json", entry)
    _write_json(entry_path, entry)  # registered last: an entry implies a complete batch dir
    return entry, str(batch_dir / "script.py"), todo


def _batch_status(batch_id: str, base_dir: str, max_failures: int = 5) -> dict:
    """Count finished runs on disk and cache the counts in the registry.

    `complete` means every run reached a terminal state (none unfinished); it
    does NOT mean every run produced a result -- a batch where most folds
    crashed is still `complete`. `ok` is the harvest signal: complete AND zero
    failures (H3). A consumer must key on `ok`, not `complete`.
    """
    root = _batch_root(base_dir)
    entry_path = root / "registry" / f"{_check_id(batch_id, 'batch_id')}.json"
    if not entry_path.exists():
        raise ValueError(f"unknown batch_id {batch_id!r}")
    entry = json.loads(entry_path.read_text())
    succeeded = failed = 0
    failures = []
    with os.scandir(root / batch_id) as entries:
        for item in entries:
            if not item.name.startswith("run_"):
                continue
            folder = Path(item.path)
            if not (folder / "done").exists():
                continue
            try:  # M5: one torn status.json must not abort the whole batch's status
                run_status = json.loads((folder / "status.json").read_text())
            except (OSError, ValueError):
                continue  # leave this run counted as unfinished until it settles
            if _run_succeeded(run_status):
                succeeded += 1
                continue
            failed += 1
            if len(failures) < max_failures:
                tail = _tail_lines(folder / "stderr.txt", 5)
                record = {"run": item.name, "returncode": run_status.get("returncode"),
                          "stderr_tail": tail}
                if run_status.get("missing_outputs"):
                    record["missing_outputs"] = run_status["missing_outputs"]
                failures.append(record)
    unfinished = entry["total"] - succeeded - failed
    status = {"succeeded": succeeded, "failed": failed, "unfinished": unfinished, "at": _now()}
    entry["last_status"] = status
    _write_json(entry_path, entry)
    return {"batch_id": batch_id, "labels": entry["labels"], "total": entry["total"],
            **status, "complete": unfinished == 0, "ok": unfinished == 0 and failed == 0,
            "dir": str(root / batch_id), "failures": failures}


def _list_batches(base_dir: str, plan_id: str | None = None,
                  hypothesis_id: str | None = None, limit: int = 50) -> list[dict]:
    """Registry entries matching the labels, newest first, using cached status."""
    registry = _batch_root(base_dir) / "registry"
    if not registry.is_dir():
        return []
    found = []
    for path in registry.glob("*.json"):
        try:  # one torn or hand-edited entry must not break every listing
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        entry_labels = entry.get("labels", {})
        if plan_id and entry_labels.get("plan_id") != plan_id:
            continue
        if hypothesis_id and entry_labels.get("hypothesis_id") != hypothesis_id:
            continue
        found.append({key: entry.get(key) for key in
                      ("batch_id", "created_at", "labels", "total", "last_status", "source_inputs",
                       "tool", "tool_version")})
    found.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return found[:max(1, min(limit, 500))]


# -- tool library ----------------------------------------------------------
#
# Layout under HPC_TOOL_ROOT (default <HPC_BASE_DIR>/tools):
#   <name>/v<N>/script.py   frozen copy of the registered script
#   <name>/v<N>/smoke/      the smoke run's input.json, stdout, stderr, status
#   <name>/v<N>/tool.json   params, description, hashes; written LAST, so a
#                           version folder without it is not registered (a
#                           failed smoke test leaves one, and N skips past it)
#
# Versions never change. Usage is not stored here (many nodes would rewrite
# it); list_tools counts it from the batch registry, which records the tool.

_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{1,40}")
_PARAM_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_PARAM_TYPES = ("int", "float", "str", "bool", "list", "dict")
_MAX_TOOL_PARAMS = 32
_MAX_TOOL_BYTES = 1 << 20
_SMOKE_TIMEOUT = float(os.environ.get("HPC_TOOL_SMOKE_TIMEOUT", "120"))


def _tool_root(base_dir: str) -> Path:
    return Path(os.environ.get("HPC_TOOL_ROOT") or Path(base_dir, "tools")).resolve()


def _type_ok(value, kind: str) -> bool:
    # bool is an int in Python; a param typed int or float must not take True.
    if kind in ("int", "float") and isinstance(value, bool):
        return False
    return isinstance(value, {"int": int, "float": (int, float), "str": str,
                              "bool": bool, "list": list, "dict": dict}[kind])


def _check_params(params: dict, defaults: dict | None) -> tuple[dict, dict]:
    """Validate a tool's {name: type} params and its defaults."""
    if not isinstance(params, dict) or not 1 <= len(params) <= _MAX_TOOL_PARAMS:
        raise ValueError(f"params must be an object with 1 to {_MAX_TOOL_PARAMS} entries")
    for name, kind in params.items():
        if name == "id" or not _PARAM_NAME.fullmatch(str(name)):
            raise ValueError(f"bad param name {name!r}")
        if kind not in _PARAM_TYPES:
            raise ValueError(f"param {name!r}: type must be one of {', '.join(_PARAM_TYPES)}")
    defaults = defaults or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    for name, value in defaults.items():
        if name not in params:
            raise ValueError(f"default for unknown param {name!r}")
        if not _type_ok(value, params[name]):
            raise ValueError(f"default for {name!r} must be {params[name]}")
    return dict(params), dict(defaults)


def _check_row(row: dict, spec: dict, what: str) -> dict:
    """Fill in defaults and check one input row against a tool's params."""
    params, defaults = spec["params"], spec.get("defaults") or {}
    unknown = sorted(set(row) - set(params) - {"id"})
    if unknown:
        raise ValueError(f"{what}: unknown params {unknown}")
    filled = {**defaults, **row}
    missing = [name for name in params if name not in filled]
    if missing:
        raise ValueError(f"{what}: missing params {missing}")
    for name, kind in params.items():
        if not _type_ok(filled[name], kind):
            raise ValueError(f"{what}: {name} must be {kind}, got {type(filled[name]).__name__}")
    return filled


_MAX_EXPECTS = 16


def _check_expects(expects: list[str] | None) -> list[str]:
    """Validate a tool's declared output patterns (L3): relative filenames or
    globs, resolved against each run's folder. Absolute paths and `..` escapes
    are refused so the contract can only name files the run itself produces."""
    if expects is None:
        return []
    if not isinstance(expects, list) or len(expects) > _MAX_EXPECTS:
        raise ValueError(f"expects must be a list of at most {_MAX_EXPECTS} output patterns")
    cleaned = []
    for pattern in expects:
        if (not isinstance(pattern, str) or not pattern.strip()
                or pattern.startswith("/") or ".." in pattern):
            raise ValueError(f"bad expected-output pattern {pattern!r}: use a relative filename or glob")
        cleaned.append(pattern.strip())
    return cleaned


def _version_dirs(tool_dir: Path) -> list[tuple[int, Path]]:
    """Every v<N> folder of a tool, registered or not, lowest N first."""
    if not tool_dir.is_dir():
        return []
    found = [(int(path.name[1:]), path) for path in tool_dir.iterdir()
             if path.is_dir() and path.name[:1] == "v" and path.name[1:].isdigit()]
    return sorted(found)


def _registered(tool_dir: Path) -> list[dict]:
    """tool.json of each registered version, lowest first, with its folder.

    A version folder with no tool.json is simply not registered (a failed smoke
    leaves one). But a tool.json that EXISTS and will not parse is corruption of
    a frozen, immutable spec, so raise loudly rather than silently skip it and
    fall back to an older version that runs different science (M15).
    """
    specs = []
    for _, path in _version_dirs(tool_dir):
        tool_json = path / "tool.json"
        if not tool_json.exists():
            continue
        try:
            specs.append({**json.loads(tool_json.read_text()), "dir": str(path)})
        except (OSError, ValueError) as exc:
            raise ValueError(f"tool version {path.name} in {tool_dir.name!r} has an "
                             f"unreadable tool.json: {exc}") from exc
    return specs


def _load_tool(base_dir: str, name: str, version: int | None = None) -> dict:
    """One registered tool version (newest by default), or ValueError."""
    if not _TOOL_NAME.fullmatch(name or ""):
        raise ValueError(f"bad tool name {name!r}")
    specs = _registered(_tool_root(base_dir) / name)
    if version is not None:
        specs = [spec for spec in specs if spec["version"] == version]
    if not specs:
        raise ValueError(f"no registered tool {name!r}" + (f" version {version}" if version else ""))
    return specs[-1]


def _register_tool(name: str, script_path: str, params: dict, description: str,
                   example: dict, defaults: dict | None, call: str, venv_path: str | None,
                   skip_smoke: bool, labels: dict | None, expects: list[str] | None,
                   base_dir: str, inspect_roots: list[str]) -> dict:
    """Freeze a script as the next version of a tool and smoke-test it.

    Registering the same script with the same params, defaults and call style
    again returns the existing version. The smoke test runs `example` through
    _run_one, the runner batches use; on failure nothing is registered and
    the error carries the stderr tail. skip_smoke registers without it, as
    tested=False. Tools are never deleted or deprecated.
    """
    if not _TOOL_NAME.fullmatch(name or ""):
        raise ValueError(f"tool name must match {_TOOL_NAME.pattern}")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 2000:
        raise ValueError("description must be 1 to 2000 characters")
    if call not in ("json", "argv"):
        raise ValueError("call must be 'json' or 'argv'")
    params, defaults = _check_params(params, defaults)
    expects = _check_expects(expects)
    if not isinstance(example, dict):
        raise ValueError("example must be one input row (an object)")
    example = _check_row(example, {"params": params, "defaults": defaults}, "example")
    script = _safe_path(script_path, base_dir, inspect_roots)
    if not script.is_file() or script.stat().st_size > _MAX_TOOL_BYTES:
        raise ValueError(f"script_path must be a file of at most {_MAX_TOOL_BYTES} bytes")
    labels = _clean_labels(labels)
    sha = _sha256(script)
    tool_dir = _tool_root(base_dir) / name

    for spec in _registered(tool_dir):
        if (spec["sha256"], spec["params"], spec.get("defaults"), spec["call"],
                spec.get("expects") or []) == (sha, params, defaults, call, expects):
            return {"tool": name, "version": spec["version"], "sha256": sha,
                    "tested": spec.get("tested"), "existing": True, "dir": spec["dir"]}

    tool_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        version = max((number for number, _ in _version_dirs(tool_dir)), default=0) + 1
        version_dir = tool_dir / f"v{version}"
        try:
            version_dir.mkdir()  # atomic: a racing registration takes the next N
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"could not claim a version folder for tool {name!r}")
    shutil.copy2(script, version_dir / "script.py")
    sha = _sha256(version_dir / "script.py")  # hash what was frozen, not the source

    smoke_status = None
    if not skip_smoke:
        smoke_status = _run_one(str(version_dir / "script.py"), example,
                                str(version_dir / "smoke"), _SMOKE_TIMEOUT, venv_path,
                                call, list(params), expects=expects)
        if not _run_succeeded(smoke_status):
            tail = (version_dir / "smoke" / "stderr.txt").read_text(errors="replace")
            missing = smoke_status.get("missing_outputs")
            why = (f"did not produce declared outputs {missing}" if missing
                   else f"returncode {smoke_status['returncode']}")
            raise ValueError(
                f"smoke test failed ({why}); "
                f"{name!r} not registered (run left in {version_dir / 'smoke'}). stderr tail:\n"
                + "\n".join(tail.splitlines()[-20:]))
    spec = {
        "tool": name,
        "version": version,
        "description": description.strip(),
        "params": params,
        "defaults": defaults,
        "call": call,
        "example": example,
        "expects": expects,
        "sha256": sha,
        "source_script": str(script),
        "venv_path": venv_path,
        "labels": labels,
        "registered_at": _now(),
        "tested": not skip_smoke,
        "smoke": smoke_status,
    }
    _write_json(version_dir / "tool.json", spec)
    return {"tool": name, "version": version, "sha256": sha, "tested": not skip_smoke,
            "smoke": smoke_status, "existing": False, "dir": str(version_dir)}


def _tool_usage(base_dir: str) -> tuple[dict, dict]:
    """Batch counts and newest fully-successful batch per (tool, version)."""
    uses: dict[tuple, int] = {}
    last_ok: dict[tuple, tuple[str, str]] = {}
    registry = _batch_root(base_dir) / "registry"
    if not registry.is_dir():
        return uses, last_ok
    for path in registry.glob("*.json"):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not entry.get("tool"):
            continue
        key = (entry["tool"], entry.get("tool_version"))
        uses[key] = uses.get(key, 0) + 1
        status = entry.get("last_status") or {}
        if status.get("unfinished") == 0 and status.get("failed") == 0:
            if entry["created_at"] > last_ok.get(key, ("", ""))[0]:
                last_ok[key] = (entry["created_at"], entry["batch_id"])
    return uses, last_ok


def _list_tools(base_dir: str, query: str | None = None, name: str | None = None,
                all_versions: bool = False, limit: int = 50) -> list[dict]:
    """Registered tools by name, newest version first; the script is not included."""
    root = _tool_root(base_dir)
    if not root.is_dir():
        return []
    uses, last_ok = _tool_usage(base_dir)
    needle = (query or "").casefold()
    found = []
    for tool_dir in sorted(root.iterdir()):
        if name and tool_dir.name != name:
            continue
        specs = _registered(tool_dir)
        for spec in reversed(specs if all_versions else specs[-1:]):
            if needle and needle not in f"{spec['tool']} {spec['description']}".casefold():
                continue
            key = (spec["tool"], spec["version"])
            found.append({
                **{field: spec.get(field) for field in (
                    "tool", "version", "description", "params", "defaults", "call",
                    "example", "expects", "tested", "labels", "registered_at", "sha256")},
                "uses": uses.get(key, 0),
                "last_ok_batch": last_ok.get(key, ("", None))[1],
            })
    return found[:max(1, min(limit, 500))]
