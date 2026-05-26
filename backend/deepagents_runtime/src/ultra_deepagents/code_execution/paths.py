from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SAFE_PART_RE = re.compile(r"[^A-Za-z0-9._-]")
SANDBOX_ROOT = "/workspace"


def safe_path_part(value: str | None, fallback: str) -> str:
    cleaned = SAFE_PART_RE.sub("_", value or fallback).strip("_")
    return cleaned or fallback


def code_workspace_path(*, root_dir: str | Path, context: Any) -> Path:
    org_id = safe_path_part(_context_value(context, "org_id"), "default")
    user_id = safe_path_part(_context_value(context, "user_id"), "anonymous")
    run_id = safe_path_part(_context_value(context, "run_id"), "default")
    return Path(root_dir) / org_id / user_id / run_id / "workspace"


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


def _context_value(context: Any, key: str) -> str | None:
    if isinstance(context, dict):
        value = context.get(key)
    else:
        value = getattr(context, key, None)
    return value if isinstance(value, str) else None
