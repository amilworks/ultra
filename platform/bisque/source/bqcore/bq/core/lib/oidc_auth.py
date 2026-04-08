"""OIDC helpers for auth mode selection, metadata discovery, and token validation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import jwt
import requests
from paste.deploy.converters import asbool
from tg import config

log = logging.getLogger("bq.auth.oidc")

_meta_cache: dict[str, Any] = {"value": None, "ts": 0.0}
_jwks_clients: dict[str, jwt.PyJWKClient] = {}
_cache_lock = threading.Lock()


def auth_mode() -> str:
    mode = str(config.get("bisque.auth.mode", "legacy")).strip().lower()
    if mode not in {"legacy", "dual", "oidc"}:
        log.warning("Unknown bisque.auth.mode=%s, falling back to legacy", mode)
        return "legacy"
    return mode


def oidc_enabled() -> bool:
    return auth_mode() in {"dual", "oidc"}


def oidc_required() -> bool:
    return auth_mode() == "oidc"


def local_tokens_enabled() -> bool:
    raw = config.get("bisque.auth.local_token.enabled")
    if raw is None:
        return auth_mode() in {"legacy", "dual"}
    return asbool(raw)


def oidc_issuer() -> str:
    return str(config.get("bisque.oidc.issuer", "")).strip().rstrip("/")


def oidc_client_id() -> str:
    return str(config.get("bisque.oidc.client_id", "")).strip()


def oidc_client_secret() -> str:
    return str(config.get("bisque.oidc.client_secret", "")).strip()


def oidc_redirect_uri() -> str:
    return str(config.get("bisque.oidc.redirect_uri", "")).strip()


def oidc_scopes() -> str:
    scopes = str(config.get("bisque.oidc.scopes", "openid profile email")).strip()
    return scopes or "openid profile email"


def oidc_username_claim() -> str:
    claim = str(config.get("bisque.oidc.username_claim", "preferred_username")).strip()
    return claim or "preferred_username"


def oidc_groups_claim() -> str:
    claim = str(config.get("bisque.oidc.groups_claim", "groups")).strip()
    return claim or "groups"


def oidc_verify_audience() -> bool:
    return asbool(config.get("bisque.oidc.verify_audience", False))


def oidc_audience() -> str:
    audience = str(config.get("bisque.oidc.audience", "")).strip()
    if audience:
        return audience
    return oidc_client_id()


def oidc_timeout_seconds() -> int:
    try:
        value = int(config.get("bisque.oidc.http_timeout_seconds", 10))
    except (TypeError, ValueError):
        return 10
    return max(2, value)


def oidc_metadata_ttl_seconds() -> int:
    try:
        value = int(config.get("bisque.oidc.metadata_ttl_seconds", 300))
    except (TypeError, ValueError):
        return 300
    return max(30, value)


def oidc_clock_skew_seconds() -> int:
    try:
        value = int(config.get("bisque.oidc.clock_skew_seconds", config.get("bisque.auth.token_clock_skew_seconds", 15)))
    except (TypeError, ValueError):
        return 15
    return max(0, value)


def oidc_algorithms() -> list[str]:
    raw = str(config.get("bisque.oidc.algorithms", "RS256")).strip()
    algs = [item.strip() for item in raw.split(",") if item.strip()]
    return algs or ["RS256"]


def oidc_metadata_url() -> str:
    explicit = str(config.get("bisque.oidc.metadata_url", "")).strip()
    if explicit:
        return explicit
    issuer = oidc_issuer()
    if not issuer:
        return ""
    return issuer + "/.well-known/openid-configuration"


def oidc_metadata(force_refresh: bool = False) -> dict[str, Any]:
    url = oidc_metadata_url()
    if not url:
        raise ValueError("OIDC metadata URL is not configured")
    ttl = oidc_metadata_ttl_seconds()
    now = time.time()
    with _cache_lock:
        cached = _meta_cache.get("value")
        ts = float(_meta_cache.get("ts", 0.0) or 0.0)
        if (not force_refresh) and cached and (now - ts) < ttl:
            return dict(cached)

    response = requests.get(url, timeout=oidc_timeout_seconds())
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OIDC metadata payload is not a JSON object")
    with _cache_lock:
        _meta_cache["value"] = dict(payload)
        _meta_cache["ts"] = now
    return dict(payload)


def oidc_jwks_uri() -> str:
    explicit = str(config.get("bisque.oidc.jwks_uri", "")).strip()
    if explicit:
        return explicit
    return str(oidc_metadata().get("jwks_uri", "")).strip()


def _oidc_endpoint(config_key: str, metadata_key: str) -> str:
    explicit = str(config.get(config_key, "")).strip()
    if explicit:
        return explicit
    return str(oidc_metadata().get(metadata_key, "")).strip()


def oidc_authorization_endpoint() -> str:
    return _oidc_endpoint("bisque.oidc.authorization_endpoint", "authorization_endpoint")


def oidc_token_endpoint() -> str:
    return _oidc_endpoint("bisque.oidc.token_endpoint", "token_endpoint")


def oidc_end_session_endpoint() -> str:
    return _oidc_endpoint("bisque.oidc.end_session_endpoint", "end_session_endpoint")


def oidc_userinfo_endpoint() -> str:
    return _oidc_endpoint("bisque.oidc.userinfo_endpoint", "userinfo_endpoint")


def oidc_username_from_claims(claims: dict[str, Any]) -> str | None:
    priority = [oidc_username_claim(), "preferred_username", "email", "sub"]
    seen = set()
    for claim_name in priority:
        if not claim_name or claim_name in seen:
            continue
        seen.add(claim_name)
        value = claims.get(claim_name)
        if value:
            return str(value)
    return None


def oidc_groups_from_claims(claims: dict[str, Any]) -> list[str]:
    groups_claim = oidc_groups_claim()
    value = claims.get(groups_claim)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _jwks_client() -> jwt.PyJWKClient:
    uri = oidc_jwks_uri()
    if not uri:
        raise ValueError("OIDC jwks_uri is not configured")
    with _cache_lock:
        client = _jwks_clients.get(uri)
        if client is None:
            client = jwt.PyJWKClient(uri, cache_keys=True)
            _jwks_clients[uri] = client
        return client


def decode_oidc_token(token: str, verify_exp: bool = True) -> dict[str, Any]:
    if not token:
        raise jwt.InvalidTokenError("Missing token")
    signing_key = _jwks_client().get_signing_key_from_jwt(token).key
    options = {
        "verify_signature": True,
        "verify_exp": verify_exp,
        "verify_nbf": True,
        "verify_iat": False,
    }
    kwargs: dict[str, Any] = {
        "algorithms": oidc_algorithms(),
        "issuer": oidc_issuer() or None,
        "options": options,
        "leeway": oidc_clock_skew_seconds(),
    }
    audience = oidc_audience()
    if oidc_verify_audience() and audience:
        kwargs["audience"] = audience
    else:
        options["verify_aud"] = False

    claims = jwt.decode(token, signing_key, **kwargs)
    username = oidc_username_from_claims(claims)
    if not username:
        raise jwt.InvalidTokenError("OIDC token missing username claim")
    return claims
