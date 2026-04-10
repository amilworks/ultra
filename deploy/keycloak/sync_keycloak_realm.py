#!/usr/bin/env python3
"""Render and reconcile the production Keycloak realm/client configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _require(name: str) -> str:
    value = _env(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit(f"Invalid URL for origin resolution: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _client_redirect_uris(public_root: str) -> list[str]:
    explicit = _split_csv(_env("KEYCLOAK_CLIENT_REDIRECT_URIS"))
    if explicit:
        return explicit
    return [
        f"{public_root}/auth_service/oidc_callback",
        f"{public_root}/client_service/*",
    ]


def _client_web_origins(public_origin: str) -> list[str]:
    explicit = _split_csv(_env("KEYCLOAK_CLIENT_WEB_ORIGINS"))
    if explicit:
        return explicit
    return [public_origin]


def _post_logout_redirect_uris(public_origin: str) -> str:
    explicit = _env("KEYCLOAK_CLIENT_POST_LOGOUT_REDIRECT_URIS")
    if explicit:
        return explicit
    return f"{public_origin}/*"


def _build_client(public_root: str) -> dict[str, object]:
    public_origin = _origin(public_root)
    client_id = _require("BISQUE_OIDC_CLIENT_ID")
    client_secret = _require("BISQUE_OIDC_CLIENT_SECRET")
    return {
        "clientId": client_id,
        "name": "BisQue Web Client",
        "enabled": True,
        "protocol": "openid-connect",
        "publicClient": False,
        "secret": client_secret,
        "standardFlowEnabled": _env("KEYCLOAK_CLIENT_STANDARD_FLOW", "true").lower()
        in {"1", "true", "yes", "on"},
        "directAccessGrantsEnabled": _env(
            "KEYCLOAK_CLIENT_DIRECT_ACCESS_GRANTS", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        "serviceAccountsEnabled": False,
        "rootUrl": public_origin,
        "baseUrl": f"{public_root}/client_service/",
        "redirectUris": _client_redirect_uris(public_root),
        "webOrigins": _client_web_origins(public_origin),
        "attributes": {
            "access.token.lifespan": _env("KEYCLOAK_ACCESS_TOKEN_LIFESPAN", "900"),
            "post.logout.redirect.uris": _post_logout_redirect_uris(public_origin),
        },
    }


def _build_groups_mapper() -> dict[str, object]:
    return {
        "name": "groups",
        "protocol": "openid-connect",
        "protocolMapper": "oidc-group-membership-mapper",
        "consentRequired": False,
        "config": {
            "full.path": "false",
            "id.token.claim": "true",
            "access.token.claim": "true",
            "userinfo.token.claim": "true",
            "claim.name": _env("BISQUE_OIDC_GROUPS_CLAIM", "groups"),
        },
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_realm() -> dict[str, object]:
    base_path = _repo_root() / "platform" / "bisque" / "docker" / "keycloak" / "realm-bisque-dev.json"
    realm = json.loads(base_path.read_text(encoding="utf-8"))
    public_root = _env("BISQUE_SERVER") or _env("BISQUE_ROOT") or f"https://{_require('BISQUE_PUBLIC_HOST')}"
    public_root = public_root.rstrip("/")
    realm["displayName"] = _env("BISQUE_TITLE", "BisQue")
    realm["clients"] = [_build_client(public_root)]
    return realm


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


def _docker_exec(container: str, *args: str) -> str:
    return _run(["docker", "exec", container, *args])


def _copy_into_container(container: str, source: Path, target: str) -> None:
    _run(["docker", "cp", str(source), f"{container}:{target}"])


def _auth_keycloak(container: str, server: str) -> None:
    _docker_exec(
        container,
        "/opt/keycloak/bin/kcadm.sh",
        "config",
        "credentials",
        "--server",
        server,
        "--realm",
        "master",
        "--user",
        _require("KEYCLOAK_ADMIN"),
        "--password",
        _require("KEYCLOAK_ADMIN_PASSWORD"),
    )


def _client_id(container: str, realm: str, client_id: str) -> str | None:
    payload = _docker_exec(
        container,
        "/opt/keycloak/bin/kcadm.sh",
        "get",
        "clients",
        "-r",
        realm,
        "-q",
        f"clientId={client_id}",
    )
    clients = json.loads(payload or "[]")
    if not clients:
        return None
    return str(clients[0]["id"])


def _upsert_client(container: str, realm: str, client_payload: dict[str, object]) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        payload_path = Path(tmpdir) / "client.json"
        payload_path.write_text(json.dumps(client_payload), encoding="utf-8")
        _copy_into_container(container, payload_path, "/tmp/bisque-client.json")
        existing_id = _client_id(container, realm, str(client_payload["clientId"]))
        if existing_id:
            _docker_exec(
                container,
                "/opt/keycloak/bin/kcadm.sh",
                "update",
                f"clients/{existing_id}",
                "-r",
                realm,
                "-f",
                "/tmp/bisque-client.json",
            )
            return existing_id
        _docker_exec(
            container,
            "/opt/keycloak/bin/kcadm.sh",
            "create",
            "clients",
            "-r",
            realm,
            "-f",
            "/tmp/bisque-client.json",
        )
    created_id = _client_id(container, realm, str(client_payload["clientId"]))
    if not created_id:
        raise SystemExit(f"Failed to create Keycloak client {client_payload['clientId']}")
    return created_id


def _ensure_groups_mapper(container: str, realm: str, client_uuid: str) -> None:
    payload = _docker_exec(
        container,
        "/opt/keycloak/bin/kcadm.sh",
        "get",
        f"clients/{client_uuid}/protocol-mappers/models",
        "-r",
        realm,
    )
    mappers = json.loads(payload or "[]")
    for mapper in mappers:
        if str(mapper.get("name") or "").strip() == "groups":
            return
    with tempfile.TemporaryDirectory() as tmpdir:
        payload_path = Path(tmpdir) / "groups-mapper.json"
        payload_path.write_text(json.dumps(_build_groups_mapper()), encoding="utf-8")
        _copy_into_container(container, payload_path, "/tmp/groups-mapper.json")
        _docker_exec(
            container,
            "/opt/keycloak/bin/kcadm.sh",
            "create",
            f"clients/{client_uuid}/protocol-mappers/models",
            "-r",
            realm,
            "-f",
            "/tmp/groups-mapper.json",
        )


def render_realm(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_build_realm(), indent=2) + "\n", encoding="utf-8")
    print(f"Rendered Keycloak realm import to {output}")


def reconcile(container: str, server: str, realm: str) -> None:
    public_root = (
        _env("BISQUE_SERVER") or _env("BISQUE_ROOT") or f"https://{_require('BISQUE_PUBLIC_HOST')}"
    ).rstrip("/")
    client_payload = _build_client(public_root)
    _auth_keycloak(container, server)
    client_uuid = _upsert_client(container, realm, client_payload)
    _ensure_groups_mapper(container, realm, client_uuid)
    print(f"Reconciled Keycloak client {client_payload['clientId']} in realm {realm}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser("render", help="Render the Keycloak realm import JSON")
    render_parser.add_argument("--output", required=True, help="Target JSON file path")

    reconcile_parser = sub.add_parser("reconcile", help="Reconcile the live Keycloak client")
    reconcile_parser.add_argument("--container", default="bisque-keycloak")
    reconcile_parser.add_argument("--server", default="http://localhost:8080/auth")
    reconcile_parser.add_argument("--realm", default="bisque")

    args = parser.parse_args()
    if args.command == "render":
        render_realm(Path(args.output).expanduser().resolve())
        return 0
    if args.command == "reconcile":
        reconcile(args.container, args.server, args.realm)
        return 0
    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
