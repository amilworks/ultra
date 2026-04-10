#!/usr/bin/env python3
"""Render and reconcile the production Keycloak realm/client configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
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


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _bool_env(name: str, default: str) -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


def _normalized_public_root() -> str:
    public_root = (
        _env("BISQUE_SERVER")
        or _env("BISQUE_ROOT")
        or _env("BISQUE_AUTH_OIDC_FRONTEND_REDIRECT_URL")
        or f"https://{_require('BISQUE_PUBLIC_HOST')}"
    )
    return public_root.rstrip("/")


def _relative_path(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return "/auth"
    if not token.startswith("/"):
        token = f"/{token}"
    return token.rstrip("/") or "/"


def _client_redirect_uris(public_root: str, *, env_name: str, defaults: list[str]) -> list[str]:
    explicit = _split_csv(_env(env_name))
    if explicit:
        return explicit
    return _unique(defaults)


def _client_web_origins(public_origin: str, *, env_name: str, defaults: list[str] | None = None) -> list[str]:
    explicit = _split_csv(_env(env_name))
    if explicit:
        return explicit
    return _unique(defaults or [public_origin])


def _post_logout_redirect_uris(*, env_name: str, default: str) -> str:
    explicit = _env(env_name)
    if explicit:
        return explicit
    return default


def _build_client(
    *,
    client_id: str,
    client_secret: str,
    name: str,
    root_url: str,
    base_url: str,
    redirect_uris: list[str],
    web_origins: list[str],
    post_logout_redirect_uris: str,
) -> dict[str, object]:
    public_origin = _origin(root_url)
    return {
        "clientId": client_id,
        "name": name,
        "enabled": True,
        "protocol": "openid-connect",
        "publicClient": False,
        "secret": client_secret,
        "standardFlowEnabled": _bool_env("KEYCLOAK_CLIENT_STANDARD_FLOW", "true"),
        "directAccessGrantsEnabled": _bool_env("KEYCLOAK_CLIENT_DIRECT_ACCESS_GRANTS", "false"),
        "serviceAccountsEnabled": False,
        "rootUrl": public_origin,
        "baseUrl": base_url,
        "redirectUris": _unique(redirect_uris),
        "webOrigins": _unique(web_origins),
        "attributes": {
            "access.token.lifespan": _env("KEYCLOAK_ACCESS_TOKEN_LIFESPAN", "900"),
            "post.logout.redirect.uris": post_logout_redirect_uris,
        },
    }


def _build_bisque_client(public_root: str) -> dict[str, object]:
    public_origin = _origin(public_root)
    return _build_client(
        client_id=_require("BISQUE_OIDC_CLIENT_ID"),
        client_secret=_require("BISQUE_OIDC_CLIENT_SECRET"),
        name="BisQue Web Client",
        root_url=public_origin,
        base_url=f"{public_root}/client_service/",
        redirect_uris=_client_redirect_uris(
            public_root,
            env_name="KEYCLOAK_BISQUE_CLIENT_REDIRECT_URIS",
            defaults=[
                _env("BISQUE_OIDC_REDIRECT_URI") or f"{public_root}/auth_service/oidc_callback",
                f"{public_root}/client_service/*",
            ],
        ),
        web_origins=_client_web_origins(
            public_origin,
            env_name="KEYCLOAK_BISQUE_CLIENT_WEB_ORIGINS",
        ),
        post_logout_redirect_uris=_post_logout_redirect_uris(
            env_name="KEYCLOAK_BISQUE_CLIENT_POST_LOGOUT_REDIRECT_URIS",
            default=_env("BISQUE_OIDC_POST_LOGOUT_REDIRECT_URI") or f"{public_root}/client_service/*",
        ),
    )


def _build_ultra_client(public_root: str) -> dict[str, object] | None:
    client_id = _env("BISQUE_AUTH_OIDC_CLIENT_ID")
    client_secret = _env("BISQUE_AUTH_OIDC_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    frontend_root = (
        _env("BISQUE_AUTH_OIDC_FRONTEND_REDIRECT_URL")
        or _env("ORCHESTRATOR_API_URL")
        or public_root
    ).rstrip("/")
    frontend_origin = _origin(frontend_root)
    redirect_uri = _env("BISQUE_AUTH_OIDC_REDIRECT_URI") or f"{public_root}/v1/auth/oidc/callback"
    return _build_client(
        client_id=client_id,
        client_secret=client_secret,
        name="BisQue Ultra Web Client",
        root_url=frontend_origin,
        base_url=frontend_root or frontend_origin,
        redirect_uris=_client_redirect_uris(
            public_root,
            env_name="KEYCLOAK_ULTRA_CLIENT_REDIRECT_URIS",
            defaults=[redirect_uri],
        ),
        web_origins=_client_web_origins(
            frontend_origin,
            env_name="KEYCLOAK_ULTRA_CLIENT_WEB_ORIGINS",
            defaults=[frontend_origin, _origin(public_root)],
        ),
        post_logout_redirect_uris=_post_logout_redirect_uris(
            env_name="KEYCLOAK_ULTRA_CLIENT_POST_LOGOUT_REDIRECT_URIS",
            default=_env("BISQUE_AUTH_LOGOUT_REDIRECT_URL") or f"{frontend_origin}/*",
        ),
    )


def _build_clients(public_root: str) -> list[dict[str, object]]:
    clients = [_build_bisque_client(public_root)]
    ultra_client = _build_ultra_client(public_root)
    if ultra_client is not None:
        clients.append(ultra_client)
    merged_by_id: dict[str, dict[str, object]] = {}
    for client in clients:
        client_id = str(client["clientId"])
        existing = merged_by_id.get(client_id)
        if existing is None:
            merged_by_id[client_id] = client
            continue
        if str(existing.get("secret") or "") != str(client.get("secret") or ""):
            raise SystemExit(
                f"Conflicting secrets configured for Keycloak client {client_id!r}. "
                "Use distinct client IDs or align the secrets."
            )
        existing["redirectUris"] = _unique(
            list(existing.get("redirectUris") or []) + list(client.get("redirectUris") or [])
        )
        existing["webOrigins"] = _unique(
            list(existing.get("webOrigins") or []) + list(client.get("webOrigins") or [])
        )
        existing["attributes"] = dict(existing.get("attributes") or {})
        merged_post_logout = _unique(
            _split_csv(str(existing["attributes"].get("post.logout.redirect.uris") or ""))
            + _split_csv(str(dict(client.get("attributes") or {}).get("post.logout.redirect.uris") or ""))
        )
        if merged_post_logout:
            existing["attributes"]["post.logout.redirect.uris"] = ",".join(merged_post_logout)
    return list(merged_by_id.values())


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
    public_root = _normalized_public_root()
    realm["realm"] = _env("KEYCLOAK_REALM_NAME", str(realm.get("realm") or "bisque"))
    realm["displayName"] = _env("BISQUE_TITLE", "BisQue")
    realm["users"] = []
    realm["clients"] = _build_clients(public_root)
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


def _auth_keycloak(container: str, server: str, *, retries: int = 30, delay_seconds: float = 2.0) -> None:
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "docker",
                "exec",
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
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout or "").strip()
        if attempt < retries:
            time.sleep(delay_seconds)
    raise SystemExit(
        f"Failed to authenticate to Keycloak admin API at {server} after {retries} attempts.\n{last_error}"
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


def _keycloak_admin_server_url() -> str:
    explicit = _env("KEYCLOAK_ADMIN_SERVER_URL")
    if explicit:
        return explicit.rstrip("/")
    relative_path = _relative_path(_env("KEYCLOAK_HTTP_RELATIVE_PATH", "/auth"))
    return f"http://127.0.0.1:8080{relative_path}".rstrip("/")


def render_realm(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_build_realm(), indent=2) + "\n", encoding="utf-8")
    print(f"Rendered Keycloak realm import to {output}")


def reconcile(container: str, server: str, realm: str) -> None:
    public_root = _normalized_public_root()
    client_payloads = _build_clients(public_root)
    _auth_keycloak(container, server)
    reconciled: list[str] = []
    for client_payload in client_payloads:
        client_uuid = _upsert_client(container, realm, client_payload)
        _ensure_groups_mapper(container, realm, client_uuid)
        reconciled.append(str(client_payload["clientId"]))
    print(f"Reconciled Keycloak client(s) {', '.join(reconciled)} in realm {realm}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser("render", help="Render the Keycloak realm import JSON")
    render_parser.add_argument("--output", required=True, help="Target JSON file path")

    reconcile_parser = sub.add_parser("reconcile", help="Reconcile the live Keycloak client(s)")
    reconcile_parser.add_argument("--container", default="bisque-keycloak")
    reconcile_parser.add_argument("--server", default=_keycloak_admin_server_url())
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
