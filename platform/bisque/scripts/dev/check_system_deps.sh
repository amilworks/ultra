#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OS_NAME="$(uname -s)"

missing_required=0

ok() {
  echo "[OK] $1"
}

warn() {
  echo "[WARN] $1"
}

fail() {
  echo "[FAIL] $1"
  missing_required=1
}

echo "Checking backend system dependencies..."
echo "repo: ${ROOT_DIR}"
echo "os: ${OS_NAME}"

if command -v uv >/dev/null 2>&1; then
  ok "uv found ($(uv --version))"
else
  fail "uv is not installed"
fi

if command -v python3 >/dev/null 2>&1; then
  ok "python3 found ($(python3 --version 2>/dev/null))"
else
  fail "python3 is not installed"
fi

if command -v docker >/dev/null 2>&1; then
  ok "docker found ($(docker --version 2>/dev/null))"
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable"
  else
    warn "docker daemon not reachable (module docker runtime checks will fail)"
  fi
else
  warn "docker not found (required to run dockerized module workflows)"
fi

if [ ! -d "${ROOT_DIR}/source" ]; then
  fail "source/ directory not found from repo root"
else
  ok "source/ directory found"
fi

graphviz_header=""
if [ "${OS_NAME}" = "Darwin" ]; then
  if command -v brew >/dev/null 2>&1; then
    graphviz_prefix="$(brew --prefix graphviz 2>/dev/null || true)"
    if [ -n "${graphviz_prefix}" ] && [ -f "${graphviz_prefix}/include/graphviz/cgraph.h" ]; then
      graphviz_header="${graphviz_prefix}/include/graphviz/cgraph.h"
    fi
  fi
else
  if [ -f "/usr/include/graphviz/cgraph.h" ]; then
    graphviz_header="/usr/include/graphviz/cgraph.h"
  elif [ -f "/usr/local/include/graphviz/cgraph.h" ]; then
    graphviz_header="/usr/local/include/graphviz/cgraph.h"
  fi
fi

if [ -n "${graphviz_header}" ]; then
  ok "Graphviz headers found (${graphviz_header})"
else
  warn "Graphviz headers not found (pygraphviz builds may fail)"
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openslide; then
  ok "OpenSlide development files detected via pkg-config"
else
  warn "OpenSlide development files not detected"
fi

if command -v pg_config >/dev/null 2>&1; then
  ok "pg_config found"
else
  warn "pg_config not found (source builds using libpq may fail)"
fi

if command -v mysql_config >/dev/null 2>&1; then
  ok "mysql_config found"
else
  warn "mysql_config not found (mysqlclient source builds may fail)"
fi

for runtime_tool in imgcnv showinf bfconvert ImarisConvert; do
  if command -v "${runtime_tool}" >/dev/null 2>&1; then
    ok "${runtime_tool} found"
  else
    warn "${runtime_tool} not found (image/module runtime capability may be limited)"
  fi
done

echo
echo "Suggested install commands:"
echo "  macOS:  brew install graphviz openslide mysql-client postgresql"
echo "          install Docker Desktop for module engine support"
echo "          install imgcnv/showinf/bfconvert/ImarisConvert if you need full Dockerfile parity"
echo "  Ubuntu: sudo apt-get install -y graphviz libgraphviz-dev libopenslide-dev libpq-dev libmysqlclient-dev pkg-config"
echo

if [ "${missing_required}" -ne 0 ]; then
  exit 1
fi
