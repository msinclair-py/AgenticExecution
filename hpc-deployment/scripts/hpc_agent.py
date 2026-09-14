"""Pinned Academy agent for hpc-deployment.

Public actions are deliberately narrow:
- run_code: run an LLM-authored Python payload in a subprocess.
- inspect_filesystem: bounded read-only ls/cat/tail under explicit roots.

Scheduler operations, generic shell execution, environment dumps, and file
mutation are intentionally absent. The endpoint/Parsl provider owns allocation
and accelerator pinning before this process starts.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from academy.agent import Agent, action

_MAX_ENTRIES = 200
_MAX_LINES = 200
_MAX_BYTES = 64 * 1024


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
                       timeout: float = 300.0) -> dict[str, str]:
        self._inflight += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _run_subprocess, payload, workdir or self.base_dir, timeout
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


def _roots(base_dir: str, inspect_roots: list[str]) -> list[Path]:
    configured = [base_dir, *inspect_roots, "/tmp"]
    return [Path(item).resolve() for item in configured if item]


def _safe_path(raw: str, base_dir: str, inspect_roots: list[str]) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    if not any(path == root or root in path.parents for root in _roots(base_dir, inspect_roots)):
        raise ValueError("path is outside allowed inspection roots")
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


def _run_subprocess(payload: str, workdir: str | None, timeout: float) -> dict[str, str]:
    import subprocess
    import sys
    import tempfile

    cwd = Path(workdir or os.environ.get("HPC_BASE_DIR", ".")).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=cwd) as handle:
        handle.write(payload)
        script = Path(handle.name)
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=cwd, env=os.environ.copy(),
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
