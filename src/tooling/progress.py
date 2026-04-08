from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable

ProgressCallback = Callable[[dict[str, Any]], None]

_PROGRESS_CB: ContextVar[ProgressCallback | None] = ContextVar("_PROGRESS_CB", default=None)

PROGRESS_CHUNK_PREFIX = "\x00tool_progress\x00"


def set_progress_callback(callback: ProgressCallback | None) -> object:
    """Set a per-thread progress callback used by long-running tools.

    Returns an opaque token that should be passed to `reset_progress_callback`.
    """
    return _PROGRESS_CB.set(callback)


def reset_progress_callback(token: object) -> None:
    _PROGRESS_CB.reset(token)


def emit_progress(
    message: str,
    *,
    tool: str | None = None,
    event: str = "log",
    level: str = "info",
    **extra: Any,
) -> None:
    """Emit a progress event to the active callback (if any)."""
    callback = _PROGRESS_CB.get()
    if callback is None:
        return
    payload: dict[str, Any] = {
        "event": str(event),
        "level": str(level),
        "message": str(message),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    if tool:
        payload["tool"] = str(tool)
    payload.update(extra)
    try:
        callback(payload)
    except Exception:
        # Never let progress reporting break the tool run.
        return


def encode_progress_chunk(payload: dict[str, Any]) -> str:
    """Encode a progress payload as an out-of-band stream chunk."""
    return PROGRESS_CHUNK_PREFIX + json.dumps(payload, ensure_ascii=False)


def decode_progress_chunk(chunk: str) -> dict[str, Any] | None:
    """Decode an out-of-band progress chunk if present."""
    if not isinstance(chunk, str) or not chunk.startswith(PROGRESS_CHUNK_PREFIX):
        return None
    raw = chunk[len(PROGRESS_CHUNK_PREFIX) :]
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None
