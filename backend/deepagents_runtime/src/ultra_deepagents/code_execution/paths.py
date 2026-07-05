from __future__ import annotations

from pathlib import Path

SANDBOX_ROOT = "/workspace"


def resolve_workspace_file(workspace_dir: str | Path, requested_path: str) -> Path:
    raw_path = requested_path.strip()
    if not raw_path:
        raise ValueError("Sandbox path is required.")

    if raw_path == SANDBOX_ROOT:
        relative = Path()
    elif raw_path.startswith(f"{SANDBOX_ROOT}/"):
        relative = Path(raw_path.removeprefix(f"{SANDBOX_ROOT}/"))
    else:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("Path is outside /workspace.")
        relative = candidate

    workspace = Path(workspace_dir).resolve()
    resolved = (workspace / relative).resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError("Path is outside /workspace.")
    return resolved
