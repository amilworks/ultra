from __future__ import annotations

import shutil
import time
from pathlib import Path


def cleanup_expired_code_workspaces(
    *,
    root_dir: str | Path,
    retention_seconds: int,
    now_seconds: float | None = None,
) -> list[Path]:
    if retention_seconds < 0:
        return []
    root = Path(root_dir)
    if not root.exists():
        return []

    now = time.time() if now_seconds is None else now_seconds
    expires_before = now - retention_seconds
    removed: list[Path] = []
    for workspace in root.glob("*/*/*/workspace"):
        try:
            if not workspace.is_dir() or workspace.stat().st_mtime >= expires_before:
                continue
            run_dir = workspace.parent
            shutil.rmtree(run_dir)
            removed.append(run_dir)
            _remove_empty_parents(run_dir.parent, stop_at=root)
        except OSError:
            continue
    return removed


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    stop = stop_at.resolve()
    current = path
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
