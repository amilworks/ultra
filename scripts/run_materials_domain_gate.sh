#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/deploy/docker/materials-domain-gate.Dockerfile"
REQUIREMENTS="${REPO_ROOT}/deploy/docker/materials-requirements.txt"
IMAGE_REF="${MATERIALS_DOMAIN_GATE_IMAGE:-bisque-ultra-materials-domain-gate:py311}"
REPORT_DIR="${MATERIALS_DOMAIN_GATE_REPORT_DIR:-${REPO_ROOT}/.tmp/materials-domain-gate}"

mkdir -p "${REPORT_DIR}"
REPORT_DIR="$(cd "${REPORT_DIR}" && pwd)"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f1
  else
    shasum -a 256 "$1" | cut -d ' ' -f1
  fi
}

git_sha="${GITHUB_SHA:-}"
if [[ -z "${git_sha}" ]]; then
  git_sha="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
git_ref="${GITHUB_REF:-}"
if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${REPO_ROOT}" symbolic-ref --quiet --short HEAD 2>/dev/null || echo detached)"
fi
git_dirty=unknown
if git_status="$(
  git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal 2>/dev/null
)"; then
  if [[ -n "${git_status}" ]]; then
    git_dirty=true
  else
    git_dirty=false
  fi
fi
require_clean_provenance="${MATERIALS_DOMAIN_GATE_REQUIRE_CLEAN_PROVENANCE:-0}"

case "${require_clean_provenance}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "Materials promotion provenance policy: required"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "Materials promotion provenance policy: not enforced (local/WIP evidence only)"
    ;;
  *)
    echo "Materials promotion provenance policy: invalid value; gate will fail closed"
    ;;
esac

if [[ "${MATERIALS_DOMAIN_GATE_SKIP_BUILD:-0}" != "1" ]]; then
  docker build \
    --pull \
    --build-arg "VCS_REF=${git_sha}" \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_REF}" \
    "${REPO_ROOT}"
fi

image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_REF}")"
image_digest="${MATERIALS_DOMAIN_GATE_IMAGE_DIGEST:-}"
dockerfile_sha256="$(sha256_file "${DOCKERFILE}")"
requirements_sha256="$(sha256_file "${REQUIREMENTS}")"

# Never allow stale evidence from an earlier pass to survive a failed launch.
rm -f \
  "${REPORT_DIR}/materials-domain-gate.json" \
  "${REPORT_DIR}/materials-domain-gate.md" \
  "${REPORT_DIR}/materials-junit.xml" \
  "${REPORT_DIR}/materials-capabilities-junit.xml" \
  "${REPORT_DIR}/calphad-experimental-benchmark.json" \
  "${REPORT_DIR}/calphad-runtime-junit.xml" \
  "${REPORT_DIR}/materials-pip-freeze.txt" \
  "${REPORT_DIR}/materials-pytest.stdout.txt" \
  "${REPORT_DIR}/materials-pytest.stderr.txt"

# Runtime-contract tests are separate from the 13 scientific invariants so the
# deterministic report keeps an exact validator count. They still execute in the
# same pinned, offline image and must pass without skips before the domain gate.
echo "Running bounded CALPHAD runtime contract preflight"
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=1g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace,readonly" \
  --mount "type=bind,src=${REPORT_DIR},dst=/reports" \
  --entrypoint python \
  "${IMAGE_REF}" \
  -m pytest -q \
  /workspace/backend/deepagents_runtime/tests/test_calphad_runtime.py \
  /workspace/backend/deepagents_runtime/tests/test_calphad_cli.py \
  --junitxml=/reports/calphad-runtime-junit.xml

# Capability tests exercise the newly exposed research primitives separately
# from the exact, promotion-scored invariant count below. Every optional
# reference backend is installed in this pinned image, so this lane also fails
# if a capability test is skipped.
echo "Running non-skipping materials capability preflight"
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=1g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace,readonly" \
  --mount "type=bind,src=${REPORT_DIR},dst=/reports" \
  --entrypoint python \
  "${IMAGE_REF}" \
  -m pytest -q \
  /workspace/backend/deepagents_runtime/tests/domain_correctness/test_processing_kinetics.py \
  /workspace/backend/deepagents_runtime/tests/domain_correctness/test_crystal_plasticity_invariants.py \
  /workspace/backend/deepagents_runtime/tests/domain_correctness/test_advanced_characterization.py \
  /workspace/backend/deepagents_runtime/tests/domain_correctness/test_degradation_primitives.py \
  /workspace/backend/deepagents_runtime/tests/test_sensor_series.py \
  --junitxml=/reports/materials-capabilities-junit.xml

python3 - "${REPORT_DIR}/materials-capabilities-junit.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
testcases = list(root.iter("testcase"))
if not testcases:
    raise SystemExit("materials capability preflight contains no testcases")
skipped = sum(testcase.find("skipped") is not None for testcase in testcases)
if skipped:
    raise SystemExit(f"materials capability preflight skipped {skipped} test(s)")

damask_prefix = "test_each_builtin_slip_family_matches_optional_damask_reference["
expected_damask_cases = {
    damask_prefix + "fcc-fcc-{111}<110>-None]",
    damask_prefix + "fcc-fcc-{110}<110>-None]",
    damask_prefix + "bcc-bcc-{110}<111>-None]",
    damask_prefix + "bcc-bcc-{112}<111>-None]",
    damask_prefix + "bcc-bcc-{123}<111>-None]",
    damask_prefix + "hcp-hcp-basal-{0001}<11-20>-1.632993161855452]",
    damask_prefix + "hcp-hcp-prismatic-{10-10}<11-20>-1.632993161855452]",
    damask_prefix + "hcp-hcp-pyramidal-{10-11}<11-20>-1.632993161855452]",
    damask_prefix + "hcp-hcp-pyramidal-{10-11}<11-23>-1.632993161855452]",
    damask_prefix + "hcp-hcp-pyramidal-{11-22}<11-23>-1.632993161855452]",
}
observed_damask_cases = {
    testcase.attrib.get("name", "")
    for testcase in testcases
    if testcase.attrib.get("name", "").startswith(damask_prefix)
}
if observed_damask_cases != expected_damask_cases:
    missing = sorted(expected_damask_cases - observed_damask_cases)
    unexpected = sorted(observed_damask_cases - expected_damask_cases)
    raise SystemExit(
        "materials capability preflight did not execute the exact 10 DAMASK 3.1.0 "
        f"reference comparisons; missing={missing!r}, unexpected={unexpected!r}"
    )
PY

set +e
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=1g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace,readonly" \
  --mount "type=bind,src=${REPORT_DIR},dst=/reports" \
  --env "ULTRA_MATERIALS_GATE_DOCKERFILE_SHA256=${dockerfile_sha256}" \
  --env "ULTRA_MATERIALS_GATE_GIT_DIRTY=${git_dirty}" \
  --env "ULTRA_MATERIALS_GATE_GIT_REF=${git_ref}" \
  --env "ULTRA_MATERIALS_GATE_GIT_SHA=${git_sha}" \
  --env "ULTRA_MATERIALS_GATE_IMAGE_DIGEST=${image_digest}" \
  --env "ULTRA_MATERIALS_GATE_IMAGE_ID=${image_id}" \
  --env "ULTRA_MATERIALS_GATE_IMAGE_REF=${IMAGE_REF}" \
  --env "ULTRA_MATERIALS_GATE_REQUIRE_CALPHAD_RUNTIME_JUNIT=1" \
  --env "ULTRA_MATERIALS_GATE_REQUIREMENTS_SHA256=${requirements_sha256}" \
  --env "MATERIALS_DOMAIN_GATE_REQUIRE_CLEAN_PROVENANCE=${require_clean_provenance}" \
  "${IMAGE_REF}" \
  --repo-root /workspace \
  --requirements /opt/ultra/materials-requirements.txt \
  --test-path backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py \
  --calphad-runtime-junit /reports/calphad-runtime-junit.xml \
  --output-dir /reports
gate_status=$?
set -e

if [[ -f "${REPORT_DIR}/materials-domain-gate.json" ]]; then
  echo "Materials domain-gate evidence: ${REPORT_DIR}"
  echo "This deterministic lane is not a full production-image or MatTools readiness claim."
fi
exit "${gate_status}"
