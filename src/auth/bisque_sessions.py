from __future__ import annotations

import time
from threading import Lock
from typing import Any
from uuid import uuid4


_BISQUE_SESSIONS: dict[str, dict[str, Any]] = {}
_BISQUE_SESSIONS_LOCK = Lock()


def cleanup_expired_bisque_sessions(*, now: float | None = None) -> None:
    ts = time.time() if now is None else float(now)
    with _BISQUE_SESSIONS_LOCK:
        expired = [
            session_id
            for session_id, session in _BISQUE_SESSIONS.items()
            if float(session.get("expires_at", 0.0)) <= ts
        ]
        for session_id in expired:
            _BISQUE_SESSIONS.pop(session_id, None)


def create_bisque_session(
    *,
    username: str,
    password: str,
    bisque_root: str,
    ttl_seconds: int,
    mode: str = "bisque",
    guest_profile: dict[str, Any] | None = None,
    auth_provider: str = "local",
    access_token: str | None = None,
    id_token: str | None = None,
    bisque_cookie_header: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    created_at = time.time() if now is None else float(now)
    expires_at = created_at + max(1, int(ttl_seconds))
    session_id = uuid4().hex
    payload = {
        "session_id": session_id,
        "username": username,
        "password": password,
        "bisque_root": str(bisque_root or "").rstrip("/"),
        "created_at": created_at,
        "expires_at": expires_at,
        "mode": mode,
        "guest_profile": dict(guest_profile or {}) if mode == "guest" else None,
        "auth_provider": str(auth_provider or "local").strip().lower(),
        "access_token": str(access_token or "").strip() or None,
        "id_token": str(id_token or "").strip() or None,
        "bisque_cookie_header": str(bisque_cookie_header or "").strip() or None,
    }
    with _BISQUE_SESSIONS_LOCK:
        _BISQUE_SESSIONS[session_id] = payload
    return dict(payload)


def delete_bisque_session(session_id: str | None) -> None:
    raw_session_id = str(session_id or "").strip()
    if not raw_session_id:
        return
    with _BISQUE_SESSIONS_LOCK:
        _BISQUE_SESSIONS.pop(raw_session_id, None)


def touch_bisque_session(
    session: dict[str, Any],
    *,
    ttl_seconds: int,
    now: float | None = None,
) -> dict[str, Any]:
    payload = dict(session)
    current_time = time.time() if now is None else float(now)
    payload["expires_at"] = current_time + max(1, int(ttl_seconds))
    session_id = str(payload.get("session_id") or "").strip()
    if session_id:
        with _BISQUE_SESSIONS_LOCK:
            _BISQUE_SESSIONS[session_id] = dict(payload)
    return payload


def get_bisque_session(
    session_id: str | None,
    *,
    ttl_seconds: int | None = None,
    touch: bool = False,
    now: float | None = None,
) -> dict[str, Any] | None:
    raw_session_id = str(session_id or "").strip()
    if not raw_session_id:
        return None
    cleanup_expired_bisque_sessions(now=now)
    with _BISQUE_SESSIONS_LOCK:
        session = _BISQUE_SESSIONS.get(raw_session_id)
        if session is None:
            return None
        payload = dict(session)
    if touch:
        if ttl_seconds is None:
            raise ValueError("ttl_seconds is required when touch=True")
        payload = touch_bisque_session(payload, ttl_seconds=ttl_seconds, now=now)
    return payload
