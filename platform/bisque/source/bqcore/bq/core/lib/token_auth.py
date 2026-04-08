"""Token helpers for dual-stack API authentication.

This module issues and validates signed access tokens used by the
``Authorization: Bearer <token>`` path. It is intentionally self-contained
so auth_service endpoints and repoze.who plugins share one implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from paste.deploy.converters import asbool
from tg import config

from .oidc_auth import decode_oidc_token, local_tokens_enabled, oidc_enabled

log = logging.getLogger("bq.auth.token")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def token_algorithm() -> str:
    return str(config.get("bisque.auth.token_algorithm", "HS256"))


def token_issuer() -> str:
    return str(config.get("bisque.auth.token_issuer", "bisque"))


def token_audience() -> str:
    return str(config.get("bisque.auth.token_audience", "bisque-api"))


def token_expiry_seconds() -> int:
    return max(60, _as_int(config.get("bisque.auth.token_expiry_seconds", 3600), 3600))


def token_clock_skew_seconds() -> int:
    return max(0, _as_int(config.get("bisque.auth.token_clock_skew_seconds", 15), 15))


def verify_audience() -> bool:
    return asbool(config.get("bisque.auth.token_verify_audience", False))


def token_secret() -> str:
    # Keep a backwards-compatible fallback so deployments continue to work
    # without immediate config changes.
    secret = config.get("bisque.auth.token_secret") or config.get("sa_auth.cookie_secret")
    if not secret:
        secret = "images"
    return str(secret)


def _decode_local_access_token(token: str, verify_exp: bool = True) -> dict[str, Any]:
    options = {
        "verify_signature": True,
        "verify_exp": verify_exp,
        "verify_nbf": True,
        "verify_iat": True,
    }
    decode_kwargs: dict[str, Any] = {
        "algorithms": [token_algorithm()],
        "issuer": token_issuer(),
        "options": options,
        "leeway": token_clock_skew_seconds(),
    }
    if verify_audience():
        decode_kwargs["audience"] = token_audience()
    else:
        options["verify_aud"] = False
    claims = jwt.decode(token, token_secret(), **decode_kwargs)
    if "sub" not in claims and "preferred_username" not in claims:
        raise jwt.InvalidTokenError("Token missing required subject claim")
    claims.setdefault("token_source", "local")
    return claims


def issue_access_token(
    username: str,
    groups: list[str] | None = None,
    scopes: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=token_expiry_seconds())
    claims: dict[str, Any] = {
        "sub": username,
        "preferred_username": username,
        "scope": " ".join(scopes or ["bisque:api"]),
        "groups": groups or [],
        "iss": token_issuer(),
        "aud": token_audience(),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "typ": "access",
    }
    if extra_claims:
        for key, value in extra_claims.items():
            if value is not None:
                claims[key] = value
    encoded = jwt.encode(claims, token_secret(), algorithm=token_algorithm())
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    return encoded, claims


def decode_access_token(token: str, verify_exp: bool = True) -> dict[str, Any]:
    errors = []

    if local_tokens_enabled():
        try:
            return _decode_local_access_token(token, verify_exp=verify_exp)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local:{exc}")

    if oidc_enabled():
        try:
            claims = decode_oidc_token(token, verify_exp=verify_exp)
            claims.setdefault("token_source", "oidc")
            return claims
        except Exception as exc:  # noqa: BLE001
            errors.append(f"oidc:{exc}")

    detail = "; ".join(errors) if errors else "no token validators enabled"
    raise jwt.InvalidTokenError(f"Unable to validate token ({detail})")
