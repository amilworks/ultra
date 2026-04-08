#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "source directory not found at ${SOURCE_DIR}"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

cd "${SOURCE_DIR}"

if [ ! -f "config/site.cfg" ]; then
  echo "Creating base config via bq-admin setup -y fullconfig..."
  bq-admin setup -y fullconfig
fi

if [ ! -f "config/test.ini" ]; then
  echo "Creating test config from config-defaults/test.ini.default..."
  cp "config-defaults/test.ini.default" "config/test.ini"
fi

needs_db_setup="$(python - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path("data/bisque.db")
needs = True
if db_path.exists():
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='taggable'"
            )
            needs = cur.fetchone() is None
        finally:
            conn.close()
    except Exception:
        needs = True

print("1" if needs else "0")
PY
)"

if [ "${needs_db_setup}" = "1" ]; then
  echo "Initializing database schema via bq-admin setup -y database..."
  bq-admin setup -y database
fi

core_static_marker="public/core/extjs/ext-all.js"
client_static_marker="public/client_service/css/welcome.css"

if [ ! -f "${core_static_marker}" ] || [ ! -f "${client_static_marker}" ]; then
  echo "Deploying static web assets via bq-admin setup -y statics..."
  bq-admin setup -y statics || true
fi

if [ ! -f "${core_static_marker}" ] || [ ! -f "${client_static_marker}" ]; then
  echo "Static asset deployment incomplete."
  echo "Missing expected files:"
  [ ! -f "${core_static_marker}" ] && echo "  - ${SOURCE_DIR}/${core_static_marker}"
  [ ! -f "${client_static_marker}" ] && echo "  - ${SOURCE_DIR}/${client_static_marker}"
  echo "Try: cd source && source ../.venv/bin/activate && bq-admin setup -y statics"
  exit 1
fi

python - <<'PY'
import configparser
from pathlib import Path

cfg_path = Path("config/test.ini")
site_cfg_path = Path("config/site.cfg")

cfg = configparser.ConfigParser()
cfg.read(cfg_path)

if not cfg.has_section("test"):
    cfg.add_section("test")

host_root = "http://127.0.0.1:8080"
if site_cfg_path.exists():
    site = configparser.ConfigParser()
    site.read(site_cfg_path)
    if site.has_option("app:main", "bisque.server"):
        host_root = site.get("app:main", "bisque.server").strip()
        host_root = host_root.replace("0.0.0.0", "127.0.0.1")

cfg.set("test", "host.root", host_root)
cfg.set("test", "host.user", cfg.get("test", "host.user", fallback="admin") or "admin")
cfg.set("test", "host.password", cfg.get("test", "host.password", fallback="admin") or "admin")

if not cfg.has_section("store"):
    cfg.add_section("store")

with cfg_path.open("w") as fp:
    cfg.write(fp)
PY

mkdir -p tests

if [ -d "${ROOT_DIR}/test_images" ] && [ ! -e "${SOURCE_DIR}/tests/test_images" ]; then
  echo "Linking test images from ${ROOT_DIR}/test_images"
  ln -s "${ROOT_DIR}/test_images" "${SOURCE_DIR}/tests/test_images"
fi

echo "Test config ready: ${SOURCE_DIR}/config/test.ini"
if [ -e "${SOURCE_DIR}/tests/test_images" ]; then
  echo "Test images ready: ${SOURCE_DIR}/tests/test_images"
else
  echo "No test_images directory found. Add images at ${ROOT_DIR}/test_images or ${SOURCE_DIR}/tests/test_images"
fi
