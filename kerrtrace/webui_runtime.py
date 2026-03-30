from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any

FORBIDDEN_WORKSPACE_PREFIXES: tuple[Path, ...] = (
    Path("/etc"),
    Path("/private/etc"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr/bin"),
    Path("/usr/sbin"),
    Path("/System"),
    Path("/Windows/System32"),
)


def _is_under_path(candidate: Path, parent: Path) -> bool:
    try:
        return candidate == parent or candidate.is_relative_to(parent)
    except ValueError:
        return False


def validate_workspace_path(raw: str) -> Path:
    """Resolve and validate a user-supplied workspace path."""
    try:
        resolved = Path(raw).expanduser().resolve()
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid workspace path '{raw}': {exc}") from exc

    root_path = Path(resolved.anchor) if resolved.anchor else None
    if root_path is not None and resolved == root_path:
        raise ValueError(f"Workspace path '{resolved}' cannot be the filesystem root.")

    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Workspace path '{resolved}' must be a directory.")

    for prefix in FORBIDDEN_WORKSPACE_PREFIXES:
        if _is_under_path(resolved, prefix):
            raise ValueError(f"Workspace path '{resolved}' points to a system directory and is not allowed.")

    return resolved


def launch_background_process(
    *,
    cmd: list[str],
    workspace_path: Path,
    log_path: Path,
    cfg_path: Path,
    output_hint: str,
    stamp: str,
    job_id: str,
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace_path),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    finally:
        log_file.close()
    meta = {
        "job_id": job_id,
        "started_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "queued_stamp": stamp,
        "cfg_path": str(cfg_path),
        "log_path": str(log_path),
        "workspace": str(workspace_path),
        "output_hint": str(output_hint),
        "cmd": " ".join(cmd),
        "history_recorded": False,
    }
    return proc, meta
