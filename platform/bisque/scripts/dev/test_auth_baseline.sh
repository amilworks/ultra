#!/usr/bin/env bash
set -euo pipefail

EXPECT_MODE="legacy"
for arg in "$@"; do
  case "${arg}" in
    --expect-legacy) EXPECT_MODE="legacy" ;;
    --expect-dual) EXPECT_MODE="dual" ;;
    --expect-oidc) EXPECT_MODE="oidc" ;;
    *)
      echo "Unknown argument: ${arg}"
      echo "Usage: $0 [--expect-legacy|--expect-dual|--expect-oidc]"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BISQUE_URL="${BISQUE_URL:-http://127.0.0.1:8080}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

require_cmd curl
require_cmd rg
require_cmd "${PYTHON_BIN}"

echo "[1/7] Waiting for BisQue readiness at ${BISQUE_URL} ..."
for i in $(seq 1 60); do
  if curl -fsS "${BISQUE_URL}/services" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "BisQue did not become ready in time."
    exit 1
  fi
done

echo "[2/7] Checking login entrypoint behavior for expected mode=${EXPECT_MODE} ..."
curl -sS -D "${TMP_DIR}/login.headers" -o "${TMP_DIR}/login.body" "${BISQUE_URL}/auth_service/login" >/dev/null
LOGIN_STATUS="$(awk 'toupper($1) ~ /^HTTP/ {code=$2} END{print code}' "${TMP_DIR}/login.headers")"
LOGIN_LOCATION="$(awk 'BEGIN{IGNORECASE=1} /^Location:/ {print $2}' "${TMP_DIR}/login.headers" | tr -d '\r')"
if [ "${EXPECT_MODE}" = "legacy" ]; then
  [ "${LOGIN_STATUS}" = "200" ] || { echo "Expected /auth_service/login HTTP 200 in legacy mode, got ${LOGIN_STATUS}"; exit 1; }
else
  [ "${LOGIN_STATUS}" = "302" ] || { echo "Expected /auth_service/login HTTP 302 in ${EXPECT_MODE} mode, got ${LOGIN_STATUS}"; exit 1; }
  echo "${LOGIN_LOCATION}" | rg -q "/auth_service/oidc_login" || {
    echo "Expected login redirect to /auth_service/oidc_login in ${EXPECT_MODE} mode, got: ${LOGIN_LOCATION}"
    exit 1
  }
fi

echo "[3/7] Checking token issuance behavior ..."
TOKEN_STATUS="$(curl -sS -o "${TMP_DIR}/token.json" -w "%{http_code}" \
  -X POST "${BISQUE_URL}/auth_service/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASSWORD}\",\"grant_type\":\"password\"}")"

LOCAL_TOKEN=""
if [ "${EXPECT_MODE}" = "oidc" ]; then
  [ "${TOKEN_STATUS}" = "400" ] || { echo "Expected token endpoint 400 in oidc mode, got ${TOKEN_STATUS}"; exit 1; }
  "${PYTHON_BIN}" - "${TMP_DIR}/token.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("error") != "unsupported_grant_type":
    raise SystemExit(f"Expected unsupported_grant_type, got: {payload}")
print("local token endpoint correctly disabled in oidc mode")
PY
else
  [ "${TOKEN_STATUS}" = "200" ] || { echo "Expected token endpoint 200 in ${EXPECT_MODE} mode, got ${TOKEN_STATUS}"; exit 1; }
  LOCAL_TOKEN="$("${PYTHON_BIN}" - "${TMP_DIR}/token.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("access_token", ""))
PY
)"
  [ -n "${LOCAL_TOKEN}" ] || { echo "Token endpoint returned 200 but no access_token"; exit 1; }
fi

echo "[4/7] Checking Basic auth behavior ..."
BASIC_STATUS="$(curl -sS -u "${ADMIN_USER}:${ADMIN_PASSWORD}" -o "${TMP_DIR}/whoami_basic.xml" -w "%{http_code}" "${BISQUE_URL}/auth_service/whoami")"
if [ "${EXPECT_MODE}" = "oidc" ]; then
  [ "${BASIC_STATUS}" = "401" ] || { echo "Expected Basic auth HTTP 401 in oidc mode, got ${BASIC_STATUS}"; exit 1; }
else
  [ "${BASIC_STATUS}" = "200" ] || { echo "Expected Basic auth HTTP 200 in ${EXPECT_MODE} mode, got ${BASIC_STATUS}"; exit 1; }
  rg -q 'name="name" value="admin"' "${TMP_DIR}/whoami_basic.xml" || {
    echo "Basic whoami response did not contain admin user"
    exit 1
  }
fi

echo "[5/7] Checking Bearer auth behavior ..."
BEARER_TOKEN="${LOCAL_TOKEN}"
if [ -z "${BEARER_TOKEN}" ]; then
  BEARER_TOKEN="$("${PYTHON_BIN}" "${ROOT_DIR}/scripts/dev/oidc_get_token.py" --provider local --user "${ADMIN_USER}" --password "${ADMIN_PASSWORD}" --print-access-token)"
  [ -n "${BEARER_TOKEN}" ] || { echo "Failed to obtain OIDC bearer token in oidc mode"; exit 1; }
fi
curl -sS -H "Authorization: Bearer ${BEARER_TOKEN}" "${BISQUE_URL}/auth_service/whoami" > "${TMP_DIR}/whoami_bearer.xml"
rg -q 'name="name" value="admin"' "${TMP_DIR}/whoami_bearer.xml" || {
  echo "Bearer whoami response did not contain admin user"
  exit 1
}

echo "[6/7] Checking session endpoint with bearer token ..."
curl -sS -H "Authorization: Bearer ${BEARER_TOKEN}" "${BISQUE_URL}/auth_service/session" > "${TMP_DIR}/session_bearer.xml"
rg -q 'name="user"' "${TMP_DIR}/session_bearer.xml" || {
  echo "Bearer session response missing user tag"
  exit 1
}

echo "[7/7] Checking /auth_service/token_info ..."
TOKEN_INFO_STATUS="$(curl -sS -H "Authorization: Bearer ${BEARER_TOKEN}" -o "${TMP_DIR}/token_info.json" -w "%{http_code}" "${BISQUE_URL}/auth_service/token_info")"
[ "${TOKEN_INFO_STATUS}" = "200" ] || { echo "token_info expected HTTP 200, got ${TOKEN_INFO_STATUS}"; exit 1; }
"${PYTHON_BIN}" - "${TMP_DIR}/token_info.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("active"):
    raise SystemExit(f"Expected active token, got: {payload}")
print("token_info reports active token")
PY

echo "Auth baseline checks passed for expected mode=${EXPECT_MODE}."
