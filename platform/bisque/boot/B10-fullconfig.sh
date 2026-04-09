#!/bin/sh
# Updated by Wahid Sadique Koly on 2025-07-29 to align with the new upgraded codebase.

set -x

source /usr/lib/bisque/bin/activate
echo "bq-admin location: $(which bq-admin)"

bq-admin setup -y fullconfig

# Normalize runtime URLs and optionally enable engine server for docker-compose.with-engine.
BISQUE_SERVER="${BISQUE_SERVER:-http://localhost:8080}"
BISQUE_ENGINE="${BISQUE_ENGINE:-http://localhost:27000}"
BISQUE_ENABLE_ENGINE="${BISQUE_ENABLE_ENGINE:-0}"
BISQUE_BIND_HOST="${BISQUE_BIND_HOST:-0.0.0.0}"

python - <<'PY'
import configparser
import os
from pathlib import Path
from urllib.parse import urlparse


def _section_bounds(lines, section_name):
    section_header = f"[{section_name}]"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == section_header:
            start = idx
            break
    if start is None:
        return None, None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].lstrip().startswith("["):
            end = idx
            break
    return start, end


def _has_plugin(lines, start, end, plugin):
    for idx in range(start + 1, end):
        if lines[idx].strip() == plugin:
            return True
    return False


def _plugin_indent(lines, start, end):
    for idx in range(start + 1, end):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("#") and stripped != "plugins =":
            return lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())] or "    "
    return "    "


def _ensure_plugin(lines, section, plugin, anchor=None, mode="after"):
    start, end = _section_bounds(lines, section)
    if start is None:
        return False
    if _has_plugin(lines, start, end, plugin):
        return False

    indent = _plugin_indent(lines, start, end)
    insert_at = end

    if anchor:
        for idx in range(start + 1, end):
            if lines[idx].strip() == anchor:
                insert_at = idx + (1 if mode == "after" else 0)
                break

    lines.insert(insert_at, f"{indent}{plugin}")
    return True


def _ensure_bearer_plugin_config(who_path):
    if not who_path.exists():
        return False

    lines = who_path.read_text(encoding="utf-8").splitlines()
    changed = False

    changed |= _ensure_plugin(
        lines,
        section="identifiers",
        plugin="bearerauth",
        anchor="auth_tkt",
        mode="after",
    )
    changed |= _ensure_plugin(
        lines,
        section="authenticators",
        plugin="bearerauth",
        anchor="sqlauth",
        mode="before",
    )

    if not any(line.strip() == "[plugin:bearerauth]" for line in lines):
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(
            [
                "[plugin:bearerauth]",
                "use = bq.core.lib.bearer_auth:make_plugin",
                "",
            ]
        )
        changed = True

    if changed:
        who_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed

cfg_path = Path("/source/config/site.cfg")
if not cfg_path.exists():
    raise SystemExit("site.cfg missing after fullconfig")

cfg = configparser.RawConfigParser()
cfg.optionxform = str
cfg.read(cfg_path)

if not cfg.has_section("main"):
    cfg.add_section("main")


def _set_option(name, value):
    cfg.set("main", name, value)
    if cfg.has_section("app:main"):
        cfg.set("app:main", name, value)


server_url = os.environ.get("BISQUE_SERVER", "http://localhost:8080")
engine_url = os.environ.get("BISQUE_ENGINE", "http://localhost:27000")
bind_host = os.environ.get("BISQUE_BIND_HOST", "0.0.0.0")

server_port = urlparse(server_url).port or 8080
engine_port = urlparse(engine_url).port or 27000
h1_bind_url = f"http://{bind_host}:{server_port}"
e1_bind_url = f"http://{bind_host}:{engine_port}"

_set_option("bisque.server", server_url)
_set_option("bisque.engine", engine_url)

auth_mode = os.environ.get("BISQUE_AUTH_MODE", "legacy").strip().lower() or "legacy"
if auth_mode not in {"legacy", "dual", "oidc"}:
    print(f"invalid BISQUE_AUTH_MODE={auth_mode!r}; falling back to legacy")
    auth_mode = "legacy"

local_token_enabled = os.environ.get("BISQUE_AUTH_LOCAL_TOKEN_ENABLED", "").strip().lower()
if not local_token_enabled:
    local_token_enabled = "true" if auth_mode in {"legacy", "dual"} else "false"

_set_option("bisque.auth.mode", auth_mode)
_set_option("bisque.auth.local_token.enabled", local_token_enabled)
_set_option("bisque.auth.token_algorithm", os.environ.get("BISQUE_AUTH_TOKEN_ALGORITHM", "HS256"))
_set_option("bisque.auth.token_issuer", os.environ.get("BISQUE_AUTH_TOKEN_ISSUER", "bisque"))
_set_option("bisque.auth.token_audience", os.environ.get("BISQUE_AUTH_TOKEN_AUDIENCE", "bisque-api"))
_set_option("bisque.auth.token_expiry_seconds", os.environ.get("BISQUE_AUTH_TOKEN_EXPIRY_SECONDS", "3600"))
_set_option("bisque.auth.token_clock_skew_seconds", os.environ.get("BISQUE_AUTH_TOKEN_CLOCK_SKEW_SECONDS", "15"))
_set_option("bisque.auth.token_verify_audience", os.environ.get("BISQUE_AUTH_TOKEN_VERIFY_AUDIENCE", "false"))
token_secret = os.environ.get("BISQUE_AUTH_TOKEN_SECRET", "").strip()
if token_secret:
    _set_option("bisque.auth.token_secret", token_secret)

_set_option("bisque.oidc.issuer", os.environ.get("BISQUE_OIDC_ISSUER", ""))
_set_option("bisque.oidc.metadata_url", os.environ.get("BISQUE_OIDC_METADATA_URL", ""))
_set_option("bisque.oidc.jwks_uri", os.environ.get("BISQUE_OIDC_JWKS_URI", ""))
_set_option("bisque.oidc.authorization_endpoint", os.environ.get("BISQUE_OIDC_AUTHORIZATION_ENDPOINT", ""))
_set_option("bisque.oidc.token_endpoint", os.environ.get("BISQUE_OIDC_TOKEN_ENDPOINT", ""))
_set_option("bisque.oidc.end_session_endpoint", os.environ.get("BISQUE_OIDC_END_SESSION_ENDPOINT", ""))
_set_option("bisque.oidc.userinfo_endpoint", os.environ.get("BISQUE_OIDC_USERINFO_ENDPOINT", ""))
_set_option("bisque.oidc.client_id", os.environ.get("BISQUE_OIDC_CLIENT_ID", ""))
_set_option("bisque.oidc.client_secret", os.environ.get("BISQUE_OIDC_CLIENT_SECRET", ""))
_set_option("bisque.oidc.redirect_uri", os.environ.get("BISQUE_OIDC_REDIRECT_URI", ""))
_set_option("bisque.oidc.post_logout_redirect_uri", os.environ.get("BISQUE_OIDC_POST_LOGOUT_REDIRECT_URI", ""))
_set_option("bisque.oidc.scopes", os.environ.get("BISQUE_OIDC_SCOPES", "openid profile email"))
_set_option("bisque.oidc.username_claim", os.environ.get("BISQUE_OIDC_USERNAME_CLAIM", "preferred_username"))
_set_option("bisque.oidc.groups_claim", os.environ.get("BISQUE_OIDC_GROUPS_CLAIM", "groups"))
_set_option("bisque.oidc.audience", os.environ.get("BISQUE_OIDC_AUDIENCE", ""))
_set_option("bisque.oidc.verify_audience", os.environ.get("BISQUE_OIDC_VERIFY_AUDIENCE", "false"))
_set_option("bisque.oidc.algorithms", os.environ.get("BISQUE_OIDC_ALGORITHMS", "RS256"))
_set_option("bisque.oidc.http_timeout_seconds", os.environ.get("BISQUE_OIDC_HTTP_TIMEOUT_SECONDS", "10"))
_set_option("bisque.oidc.metadata_ttl_seconds", os.environ.get("BISQUE_OIDC_METADATA_TTL_SECONDS", "300"))
_set_option("bisque.oidc.clock_skew_seconds", os.environ.get("BISQUE_OIDC_CLOCK_SKEW_SECONDS", "15"))
_set_option("bisque.oidc.auto_approve_users", os.environ.get("BISQUE_OIDC_AUTO_APPROVE_USERS", "false"))
_set_option("bisque.oidc.login_page_enabled", os.environ.get("BISQUE_OIDC_LOGIN_PAGE_ENABLED", "false"))
_set_option("bisque.oidc.provider_name", os.environ.get("BISQUE_OIDC_PROVIDER_NAME", "Keycloak"))
_set_option("bisque.oidc.login_button_text", os.environ.get("BISQUE_OIDC_LOGIN_BUTTON_TEXT", "Continue with Keycloak"))
_set_option("sqlalchemy.pool_recycle", os.environ.get("BISQUE_SQLALCHEMY_POOL_RECYCLE", "3600"))
_set_option("sqlalchemy.pool_pre_ping", os.environ.get("BISQUE_SQLALCHEMY_POOL_PRE_PING", "true"))
_set_option("sqlalchemy.pool_size", os.environ.get("BISQUE_SQLALCHEMY_POOL_SIZE", "25"))
_set_option("sqlalchemy.max_overflow", os.environ.get("BISQUE_SQLALCHEMY_MAX_OVERFLOW", "25"))
_set_option("sqlalchemy.pool_timeout", os.environ.get("BISQUE_SQLALCHEMY_POOL_TIMEOUT", "60"))

if not cfg.has_section("servers"):
    cfg.add_section("servers")

enable_engine = os.environ.get("BISQUE_ENABLE_ENGINE", "0").strip().lower() in {"1", "true", "yes", "on"}
if enable_engine:
    cfg.set("servers", "servers", "h1,e1")
    cfg.set("servers", "h1.url", h1_bind_url)
    cfg.set("servers", "h1.services_enabled", "")
    cfg.set("servers", "h1.services_disabled", "")
    cfg.set("servers", "h1.bisque.static_files", "true")
    cfg.set("servers", "e1.url", e1_bind_url)
    cfg.set("servers", "e1.services_enabled", "engine_service")
    cfg.set("servers", "e1.bisque.has_database", "false")
    cfg.set("servers", "e1.bisque.static_files", "false")
else:
    cfg.set("servers", "servers", "h1")
    cfg.set("servers", "h1.url", h1_bind_url)

with cfg_path.open("w") as fh:
    cfg.write(fh)

print(f"updated site.cfg engine_enabled={enable_engine} auth_mode={auth_mode}")
who_cfg_path = Path("/source/config/who.ini")
if _ensure_bearer_plugin_config(who_cfg_path):
    print("updated who.ini with bearerauth plugin migration")
PY

if [ "${BISQUE_ENABLE_ENGINE}" = "1" ] || [ "${BISQUE_ENABLE_ENGINE}" = "true" ] || [ "${BISQUE_ENABLE_ENGINE}" = "yes" ] || [ "${BISQUE_ENABLE_ENGINE}" = "on" ]; then
  # Ensure both h1/e1 paster configs exist and are synced with site.cfg.
  bq-admin setup --inscript -y webservers
fi
