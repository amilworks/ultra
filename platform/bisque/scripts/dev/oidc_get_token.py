#!/usr/bin/env python3
"""Fetch an OIDC access token for local/dev verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def _default(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    value = value.strip()
    return value or fallback


def _discover_metadata(issuer: str, timeout: int) -> dict[str, Any]:
    issuer = issuer.rstrip("/")
    url = issuer + "/.well-known/openid-configuration"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("OIDC metadata response is not a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Get OIDC access token")
    parser.add_argument("--provider", default="local", choices=["local", "custom"])
    parser.add_argument("--issuer", default=None, help="Issuer URL (defaults from provider/env)")
    parser.add_argument("--token-endpoint", default=None, help="Explicit token endpoint")
    parser.add_argument("--client-id", default=None, help="OIDC client id")
    parser.add_argument("--client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--user", default=None, help="Username for password grant")
    parser.add_argument("--password", default=None, help="Password for password grant")
    parser.add_argument("--scope", default=None, help="Scope override")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds")
    parser.add_argument("--print-access-token", action="store_true", help="Print access token only")
    parser.add_argument("--print-id-token", action="store_true", help="Print id token only")
    parser.add_argument("--json", action="store_true", help="Print full JSON token payload")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.provider == "local":
        issuer = _default(args.issuer, os.environ.get("BISQUE_OIDC_ISSUER", "http://127.0.0.1:18080/realms/bisque"))
        client_id = _default(args.client_id, os.environ.get("BISQUE_OIDC_CLIENT_ID", "bisque-dev"))
        client_secret = _default(args.client_secret, os.environ.get("BISQUE_OIDC_CLIENT_SECRET", "bisque-dev-secret"))
        user = _default(args.user, os.environ.get("OIDC_USERNAME", "admin"))
        password = _default(args.password, os.environ.get("OIDC_PASSWORD", "admin"))
        scope = _default(args.scope, os.environ.get("BISQUE_OIDC_SCOPES", "openid profile email"))
    else:
        if not args.issuer and not args.token_endpoint:
            parser.error("--issuer or --token-endpoint is required for provider=custom")
        issuer = (args.issuer or "").strip()
        client_id = _default(args.client_id, os.environ.get("BISQUE_OIDC_CLIENT_ID", ""))
        client_secret = _default(args.client_secret, os.environ.get("BISQUE_OIDC_CLIENT_SECRET", ""))
        user = _default(args.user, os.environ.get("OIDC_USERNAME", ""))
        password = _default(args.password, os.environ.get("OIDC_PASSWORD", ""))
        scope = _default(args.scope, os.environ.get("BISQUE_OIDC_SCOPES", "openid profile email"))

    if not user or not password:
        parser.error("username/password are required")
    if not client_id:
        parser.error("client id is required")

    try:
        if args.token_endpoint:
            token_endpoint = args.token_endpoint.strip()
        else:
            metadata = _discover_metadata(issuer, timeout=args.timeout)
            token_endpoint = str(metadata.get("token_endpoint", "")).strip()
        if not token_endpoint:
            raise RuntimeError("OIDC token endpoint is missing")

        payload = {
            "grant_type": "password",
            "client_id": client_id,
            "username": user,
            "password": password,
            "scope": scope,
        }
        if client_secret:
            payload["client_secret"] = client_secret

        response = requests.post(
            token_endpoint,
            data=payload,
            timeout=args.timeout,
            headers={"Accept": "application/json"},
        )
        if response.status_code >= 400:
            detail = response.text[:400]
            raise RuntimeError(f"Token request failed status={response.status_code}: {detail}")
        token_payload = response.json()
        if not isinstance(token_payload, dict):
            raise RuntimeError("OIDC token response is not a JSON object")

        if args.print_access_token:
            token = token_payload.get("access_token", "")
            if not token:
                raise RuntimeError("OIDC token response missing access_token")
            print(token)
            return 0
        if args.print_id_token:
            token = token_payload.get("id_token", "")
            if not token:
                raise RuntimeError("OIDC token response missing id_token")
            print(token)
            return 0
        if args.json:
            print(json.dumps(token_payload, indent=2, sort_keys=True))
            return 0

        summary = {
            "token_endpoint": token_endpoint,
            "token_type": token_payload.get("token_type"),
            "expires_in": token_payload.get("expires_in"),
            "scope": token_payload.get("scope"),
            "has_access_token": bool(token_payload.get("access_token")),
            "has_id_token": bool(token_payload.get("id_token")),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
