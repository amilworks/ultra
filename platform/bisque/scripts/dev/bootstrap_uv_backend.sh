#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RECREATE_VENV="${RECREATE_VENV:-1}"

echo "Bootstrapping BisQue backend with uv..."
echo "repo: ${ROOT_DIR}"
echo "venv: ${VENV_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed"
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}"
  exit 1
fi

if [ "${RECREATE_VENV}" = "1" ] && [ -d "${VENV_DIR}" ]; then
  echo "Removing existing venv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

uv venv "${VENV_DIR}" --python "${PYTHON_BIN}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Keep legacy stack compatible with pkg_resources users.
uv pip install -c "${ROOT_DIR}/source/constraints.uv.txt" "setuptools<81" pip wheel

# Install core third-party dependencies used by local packages and CLI tools.
uv pip install -c "${ROOT_DIR}/source/constraints.uv.txt" -r "${ROOT_DIR}/source/requirements.uv.txt"

# Install local legacy compatibility packages in deterministic order.
uv pip install -e "${ROOT_DIR}/source/legacy_upgraded/WebHelpers-2.0"
uv pip install -e "${ROOT_DIR}/source/legacy_upgraded/WebError-2.0"
uv pip install -e "${ROOT_DIR}/source/legacy_upgraded/paste-103.10.1"
uv pip install -e "${ROOT_DIR}/source/legacy_upgraded/Pylons-2.0"
uv pip install -e "${ROOT_DIR}/source/legacy_upgraded/Minimatic-2.0"

# Build/install pygraphviz before linesman to satisfy bqcore requirements.
# On macOS Homebrew Graphviz is outside default include/library paths.
if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  graphviz_prefix="$(brew --prefix graphviz 2>/dev/null || true)"
  if [ -n "${graphviz_prefix}" ] && [ -d "${graphviz_prefix}/include" ] && [ -d "${graphviz_prefix}/lib" ]; then
    CFLAGS="-I${graphviz_prefix}/include" \
    LDFLAGS="-L${graphviz_prefix}/lib" \
    uv pip install "pygraphviz==1.14"
  else
    uv pip install "pygraphviz==1.14"
  fi
else
  uv pip install "pygraphviz==1.14"
fi

uv pip install "linesman==0.3.2"

# Install the path-based CLI used for zero-copy registration.
uv pip install -e "${ROOT_DIR}/source/contrib/bisque_paths" --no-deps

# Install local BisQue packages without resolver side-effects.
uv pip install -e "${ROOT_DIR}/source/bqcore" --no-deps
uv pip install -e "${ROOT_DIR}/source/bqapi" --no-deps
uv pip install -e "${ROOT_DIR}/source/bqengine" --no-deps
uv pip install -e "${ROOT_DIR}/source/bqfeature" --no-deps
uv pip install -e "${ROOT_DIR}/source/bqserver" --no-deps
uv pip install -e "${ROOT_DIR}/source/pytest-bisque" --no-deps
uv pip install -e "${ROOT_DIR}/source/contrib/bisque_paths" --no-deps

echo "Running bootstrap sanity checks..."
python -c "import pkg_resources, tg, bqapi; print('import-ok')"
bq-admin --help >/dev/null
bq-path --help >/dev/null

echo "Bootstrap complete."
