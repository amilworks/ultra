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
    """Remove per-run scratch workspaces older than the retention window.

    The live layout is flat: each direct child of ``root_dir`` is a per-run
    workspace (``<workspace_root>/<run_id>``) holding scratch files — matplotlib
    output, staged uploads, and staged repos. Durable deliverables live separately
    under the artifact root, so reclaiming an expired run dir never deletes user
    outputs. Best-effort and mtime-based: an active run keeps a recent mtime and
    sits far inside any sane retention window. ``retention_seconds <= 0`` disables
    the sweep.
    """
    if retention_seconds <= 0:
        return []
    root = Path(root_dir)
    if not root.exists():
        return []

    now = time.time() if now_seconds is None else now_seconds
    expires_before = now - retention_seconds
    removed: list[Path] = []
    for run_dir in sorted(root.iterdir()):
        try:
            if not run_dir.is_dir():
                continue
            if run_dir.stat().st_mtime >= expires_before:
                continue
            shutil.rmtree(run_dir)
            removed.append(run_dir)
        except OSError:
            continue
    return removed
