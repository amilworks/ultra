#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_SHA="${RELEASE_SHA:-$(git rev-parse HEAD)}"
RELEASE_NAME="${RELEASE_NAME:-$RELEASE_SHA}"
RELEASE_OUT_DIR="${RELEASE_OUT_DIR:-$ROOT_DIR/.tmp/releases}"
RELEASE_WORK_DIR="${RELEASE_WORK_DIR:-$ROOT_DIR/.tmp/release-build}"

case "$RELEASE_OUT_DIR" in
  /*) ;;
  *) RELEASE_OUT_DIR="$ROOT_DIR/$RELEASE_OUT_DIR" ;;
esac
case "$RELEASE_WORK_DIR" in
  /*) ;;
  *) RELEASE_WORK_DIR="$ROOT_DIR/$RELEASE_WORK_DIR" ;;
esac

RELEASE_PARENT="$RELEASE_WORK_DIR/root"
RELEASE_DIR="$RELEASE_PARENT/$RELEASE_NAME"
CONTROL_BINARY="$RELEASE_WORK_DIR/bin/ultra-control"
TARBALL="$RELEASE_OUT_DIR/ultra-release-$RELEASE_SHA.tar.gz"
MANIFEST_IN_RELEASE="$RELEASE_DIR/release-manifest.json"
MANIFEST_OUT="$RELEASE_OUT_DIR/release-manifest-$RELEASE_SHA.json"
GOOS_VALUE="${GOOS:-linux}"
GOARCH_VALUE="${GOARCH:-amd64}"
CGO_ENABLED_VALUE="${CGO_ENABLED:-0}"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Required command is missing: $name" >&2
    exit 1
  fi
}

require_clean_head() {
  local head_sha
  head_sha="$(git rev-parse HEAD)"
  if [ "$RELEASE_SHA" != "$head_sha" ]; then
    echo "Refusing to mix release source and build outputs from different revisions." >&2
    echo "RELEASE_SHA must equal the checked-out HEAD: $RELEASE_SHA != $head_sha" >&2
    exit 1
  fi
  if [ -n "$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" ]; then
    echo "Refusing to build an immutable release from a dirty or untracked worktree." >&2
    echo "Commit, remove, or stash changes before building the release." >&2
    exit 1
  fi
}

require_cmd git
require_cmd go
require_cmd tar
require_cmd python3

if [ ! -f frontend/dist/index.html ]; then
  echo "Missing frontend/dist/index.html. Run pnpm --dir frontend build before building a release artifact." >&2
  exit 1
fi

require_clean_head

if [ -d "$RELEASE_WORK_DIR" ]; then
  chmod -R u+w "$RELEASE_WORK_DIR" 2>/dev/null || true
fi
rm -rf "$RELEASE_WORK_DIR"
mkdir -p "$RELEASE_DIR" "$RELEASE_WORK_DIR/bin" "$RELEASE_OUT_DIR"

echo "Archiving tracked source at $RELEASE_SHA"
git archive --format=tar "$RELEASE_SHA" | tar -xf - -C "$RELEASE_DIR"

echo "Building ultra-control for $GOOS_VALUE/$GOARCH_VALUE"
(
  cd backend/controlplane
  env \
    CGO_ENABLED="$CGO_ENABLED_VALUE" \
    GOOS="$GOOS_VALUE" \
    GOARCH="$GOARCH_VALUE" \
    GOCACHE="$RELEASE_WORK_DIR/go-cache" \
    GOMODCACHE="$RELEASE_WORK_DIR/go-mod-cache" \
    go build -trimpath -ldflags="-s -w" -o "$CONTROL_BINARY" ./cmd/ultra-control
)
mkdir -p "$RELEASE_DIR/bin"
install -m 0755 "$CONTROL_BINARY" "$RELEASE_DIR/bin/ultra-control"

echo "Embedding frontend/dist"
rm -rf "$RELEASE_DIR/frontend/dist"
mkdir -p "$RELEASE_DIR/frontend/dist"
tar -C frontend/dist -cf - . | tar -C "$RELEASE_DIR/frontend/dist" -xf -

echo "Writing release manifest"
export RELEASE_SHA RELEASE_NAME RELEASE_DIR GOOS_VALUE GOARCH_VALUE CGO_ENABLED_VALUE MANIFEST_IN_RELEASE MANIFEST_OUT
python3 <<'PY'
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def run_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def frontend_versions() -> dict[str, str]:
    package = json.loads(Path("frontend/package.json").read_text(encoding="utf-8"))
    packages = package.get("dependencies", {}) | package.get("devDependencies", {})
    selected = [
        "@react-three/fiber",
        "react",
        "react-dom",
        "shadcn",
        "three",
        "typescript",
        "vite",
    ]
    return {name: packages[name] for name in selected if name in packages}


def python_versions() -> dict[str, str]:
    lock_path = Path("backend/deepagents_runtime/uv.lock")
    if not lock_path.exists():
        return {}
    packages: dict[str, str] = {}
    current_name: str | None = None
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            current_name = None
            continue
        if stripped.startswith("name = "):
            current_name = json.loads(stripped.split("=", 1)[1].strip())
            continue
        if current_name and stripped.startswith("version = "):
            packages[current_name] = json.loads(stripped.split("=", 1)[1].strip())
            current_name = None
    selected = [
        "deepagents",
        "langchain",
        "langgraph",
        "langgraph-sdk",
        "nats-py",
        "openai",
        "pydantic",
        "websockets",
    ]
    return {name: packages[name] for name in selected if name in packages}


def go_versions() -> dict[str, str]:
    go_mod = Path("backend/controlplane/go.mod").read_text(encoding="utf-8").splitlines()
    selected = {
        "github.com/jackc/pgx/v5",
        "github.com/nats-io/nats.go",
        "github.com/workos/workos-go/v8",
    }
    versions: dict[str, str] = {}
    for line in go_mod:
        stripped = line.strip()
        if not stripped or stripped.startswith(("module ", "go ", "require (", ")")):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0] in selected:
            versions[parts[0]] = parts[1]
    return versions


def load_materials_verifier():
    release_root = Path(os.environ["RELEASE_DIR"])
    verifier_path = release_root / "scripts/verify_production_materials_sandbox.py"
    spec = importlib.util.spec_from_file_location(
        "ultra_release_materials_verifier", verifier_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load materials verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def materials_release_status() -> dict[str, object]:
    """A source bundle is never itself a production-image readiness claim."""

    release_root = Path(os.environ["RELEASE_DIR"])
    gate_path = release_root / "scripts/materials_readiness_gate.py"
    spec = importlib.util.spec_from_file_location("ultra_release_materials_readiness", gate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load materials readiness policy: {gate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {
        "full_materials_production_ready": False,
        "qualification_stage": "release-source-artifact",
        "product_label": "materials science research preview",
        "required_post_image_gate": "materials-production-readiness",
        "readiness_result_is_candidate_only": True,
        "required_release_envelope_schema": (
            "ultra.materials.release-envelope.v1"
        ),
        "required_final_verification_schema": (
            "ultra.materials.production-attestation-verification.v1"
        ),
        "requires_github_sigstore_attestation": True,
        "raw_mattools_evidence_may_be_public_artifact": False,
        "required_evidence": {
            "production_parity_scope": module.PRODUCTION_PARITY_SCOPE,
            "production_parity_gate": module.PRODUCTION_PARITY_GATE,
            "production_parity_evidence_bundle_schema_version": (
                materials_verifier.EVIDENCE_BUNDLE_SCHEMA_VERSION
            ),
            "sandbox_output_capture_mode": (
                "docker_sandbox_backend_combined_stdout_stderr"
            ),
            "calphad_runtime_test_count": (
                materials_verifier.REQUIRED_CALPHAD_RUNTIME_TEST_COUNT
            ),
            "calphad_tool_test_count": (
                materials_verifier.REQUIRED_CALPHAD_TOOLS_TEST_COUNT
            ),
            "calphad_cross_language_schema": (
                "ultra.calphad.cross-language-gate.v1"
            ),
            "calphad_cross_language_requires_production_runtime_image": True,
            "mattools_trial_count": module.TRIAL_COUNT,
            "mattools_runnable_denominator": module.RUNNABLE_DENOMINATOR,
            "mattools_runnable_minimum": module.RUNNABLE_MINIMUM,
            "mattools_scientific_denominator": module.SCIENTIFIC_DENOMINATOR,
            "mattools_scientific_minimum": module.SCIENTIFIC_MINIMUM,
        },
    }


materials_verifier = load_materials_verifier()
required_materials = materials_verifier.build_required_source_manifest(
    Path(os.environ["RELEASE_DIR"])
)
release_artifacts = materials_verifier.build_release_artifact_identities(
    Path(os.environ["RELEASE_DIR"])
)
if not isinstance(required_materials, dict) or not isinstance(release_artifacts, dict):
    raise RuntimeError("materials verifier returned malformed release integrity evidence")

manifest = {
    "schema_version": 1,
    "release_sha": os.environ["RELEASE_SHA"],
    "release_name": os.environ["RELEASE_NAME"],
    "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "source": {
        "repository": os.environ.get("GITHUB_REPOSITORY") or run_text("git", "config", "--get", "remote.origin.url"),
        "ref": os.environ.get("GITHUB_REF_NAME") or run_text("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "required_materials": required_materials,
    },
    "targets": {
        "goos": os.environ["GOOS_VALUE"],
        "goarch": os.environ["GOARCH_VALUE"],
        "cgo_enabled": os.environ["CGO_ENABLED_VALUE"],
        "control_binary": "bin/ultra-control",
        "control_binary_identity": release_artifacts["control_binary"],
        "frontend_dist": "frontend/dist",
        "frontend_dist_identity": release_artifacts["frontend_dist"],
        "deploy_scripts": [
            "scripts/deploy_ultra_control_stack.sh",
            "scripts/deploy_ultra_frontend.sh",
        ],
    },
    "toolchain": {
        "go": run_text("go", "version"),
        "go_env_version": run_text("go", "env", "GOVERSION"),
        "node": run_text("node", "--version"),
        "pnpm": run_text("pnpm", "--version"),
        "python": run_text("python3", "--version"),
        "uv": run_text("uv", "--version"),
    },
    "dependencies": {
        "controlplane_go": go_versions(),
        "deepagents_python": python_versions(),
        "frontend": frontend_versions(),
    },
    "materials": materials_release_status(),
}

payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
Path(os.environ["MANIFEST_IN_RELEASE"]).write_text(payload, encoding="utf-8")
Path(os.environ["MANIFEST_OUT"]).write_text(payload, encoding="utf-8")
PY

echo "Creating release tarball"
rm -f "$TARBALL" "$TARBALL.sha256"
(
  cd "$RELEASE_PARENT"
  tar -czf "$TARBALL" "$RELEASE_NAME"
)

(
  cd "$RELEASE_OUT_DIR"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$(basename "$TARBALL")" >"$(basename "$TARBALL").sha256"
  else
    shasum -a 256 "$(basename "$TARBALL")" >"$(basename "$TARBALL").sha256"
  fi
)

echo "Release artifact: $TARBALL"
echo "Release checksum: $TARBALL.sha256"
echo "Release manifest: $MANIFEST_OUT"
