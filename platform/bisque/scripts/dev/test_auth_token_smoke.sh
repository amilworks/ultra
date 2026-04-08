#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
BISQUE_URL="${BISQUE_URL:-http://127.0.0.1:8080}"
MANAGE_SERVER="${MANAGE_SERVER:-1}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

cleanup() {
  cd "${SOURCE_DIR}"
  bq-admin server stop >/dev/null 2>&1 || true
}
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [ "${MANAGE_SERVER}" = "1" ]; then
  "${ROOT_DIR}/scripts/dev/init_test_config.sh"
  trap cleanup EXIT
  cd "${SOURCE_DIR}"
  echo "Starting BisQue server for token smoke tests..."
  bq-admin server start
fi

for i in $(seq 1 45); do
  if curl -fsS "${BISQUE_URL}/services" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "${i}" -eq 45 ]; then
    echo "BisQue failed to become ready on ${BISQUE_URL}"
    exit 1
  fi
done

TOKEN_JSON="$(curl -fsS -X POST "${BISQUE_URL}/auth_service/token" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin","grant_type":"password"}')"

TOKEN="$(TOKEN_JSON="${TOKEN_JSON}" python3 - <<'PY'
import json
import os
payload = json.loads(os.environ["TOKEN_JSON"])
print(payload.get("access_token", ""))
PY
)"
export TOKEN

if [ -z "${TOKEN}" ]; then
  echo "Token endpoint did not return an access token"
  exit 1
fi

echo "Validating bearer whoami/session endpoints..."
curl -fsS -H "Authorization: Bearer ${TOKEN}" "${BISQUE_URL}/auth_service/whoami" | rg 'value="admin"'
curl -fsS -H "Authorization: Bearer ${TOKEN}" "${BISQUE_URL}/auth_service/session" | rg 'name="user"'

echo "Validating bqapi token session..."
PYTHONPATH="${SOURCE_DIR}/bqapi:${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore" BISQUE_URL="${BISQUE_URL}" python3 - <<'PY'
from bqapi import BQSession
import os

token = os.environ["TOKEN"]
bisque_url = os.environ["BISQUE_URL"]
session = BQSession().init_token(token, bisque_root=bisque_url, create_mex=False)
whoami = session.fetchxml("/auth_service/whoami")
name = whoami.find("./tag[@name='name']")
assert name is not None and name.get("value") == "admin"
print("bqapi token session ok")
session.close()
PY

echo "Token auth smoke tests passed."
