#!/usr/bin/env python3
"""Build and verify the reviewed MatTools evaluator reconstruction variant.

This utility performs build/provenance probes only. Snapshot validation reads
the pinned repository metadata, but it never submits or executes a benchmark
question, generated function, or upstream verifier.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_RELATIVE = Path("scripts/build_mattools_evaluator.py")
BUILDER_PATH = REPOSITORY_ROOT / BUILDER_RELATIVE
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "mattools_promotion_gate.py"
DOCKERFILE_RELATIVE = Path("deploy/docker/mattools-evaluator.Dockerfile")
SUPPLEMENTAL_RELATIVE = Path("deploy/docker/mattools-evaluator-supplemental-requirements.txt")
STRICT_SHADOW_RELATIVE = Path("scripts/mattools_strict_shadow.py")
SAFE_PARSER_RELATIVE = Path("scripts/mattools_safe_parser.py")
RUNNER_WRAPPER_RELATIVE = Path("scripts/mattools_runner_wrapper.py")
SEMANTIC_REPAIRS_RELATIVE = Path("scripts/mattools_semantic_repairs.py")
DOCKERFILE_PATH = REPOSITORY_ROOT / DOCKERFILE_RELATIVE
SUPPLEMENTAL_PATH = REPOSITORY_ROOT / SUPPLEMENTAL_RELATIVE
STRICT_SHADOW_PATH = REPOSITORY_ROOT / STRICT_SHADOW_RELATIVE
SAFE_PARSER_PATH = REPOSITORY_ROOT / SAFE_PARSER_RELATIVE
RUNNER_WRAPPER_PATH = REPOSITORY_ROOT / RUNNER_WRAPPER_RELATIVE
SEMANTIC_REPAIRS_PATH = REPOSITORY_ROOT / SEMANTIC_REPAIRS_RELATIVE

EVALUATOR_IMAGE_TAG = "mat-tool-ben"
ENVIRONMENT_KIND = "reviewed-reconstruction-variant"
DEFAULT_PLATFORM = "linux/arm64"
PYTHON_BASE_IMAGE = (
    "python:3.11.8@sha256:61d662f6d52206ab2290af4258257b5369573b6a4bbd904896699cc909221334"
)
UPSTREAM_REVISION = "1803a6abfe23a9da56c894076c59117873b758ff"
UPSTREAM_MANIFEST_SHA256 = "c70c9c5b1d085643372728e4017c28282e190cd452afa2f5e7fd3366e1a9528e"
UPSTREAM_REQUIREMENTS_SHA256 = "2c33bd9d99fedaf24bc99aebe60a2e64fa1bcc01d62eb6ae6e013790d3d60122"
ADAPTED_REQUIREMENTS_SHA256 = "d21b1a54095b8e60fc6631fc50c3a345bbbfdb0066cdd0a75513ba040aa3ee91"
TOOL_SOURCE_MANIFEST_SHA256 = "26c1b9224e58bdf7d8aeafbeca74a059d5551a66c85d0fa05b203dff410389d6"
TOOL_SOURCE_FILE_COUNT = 2756
CANDIDATE_FIXTURE_PREFIX = (
    "src/tool_source_code/pymatgen-analysis-defects/tests/test_files/"
)
CANDIDATE_FIXTURE_FILE_COUNT = 141
CANDIDATE_FIXTURE_MANIFEST_SHA256 = (
    "296b5b55a5c1640999dd46556c2cd1a1487ae9de3e0f050fa601d3c5236bf308"
)
REQUIRED_PACKAGES = {
    "pymatgen": "2024.8.9",
    "pymatgen-analysis-defects": "2024.7.19",
}

PROBE_SCRIPT = r"""
import hashlib
import json
import os
import platform
from importlib.metadata import distribution as metadata_distribution
from importlib.metadata import distributions, version
from pathlib import Path

def hash_path(path):
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = Path("/app/tool_source_code")
file_hashes = {}
for path in sorted(root.rglob("*")):
    if not path.is_symlink() and not path.is_file():
        continue
    relative = "src/tool_source_code/" + path.relative_to(root).as_posix()
    file_hashes[relative] = hash_path(path)
lines = "".join(
    f"{file_hashes[name]}  {name}\n" for name in sorted(file_hashes)
)
packages = {
    distribution.metadata.get("Name", "").lower(): distribution.version
    for distribution in distributions()
    if distribution.metadata.get("Name")
}
dependency_test_paths = {}
for package_name in ("pymatgen", "pymatgen-analysis-defects"):
    paths = []
    for entry in metadata_distribution(package_name).files or ():
        if any(
            part.casefold() in {"test", "tests"}
            or part.casefold().startswith("test_")
            for part in entry.parts
        ):
            paths.append(str(entry))
    dependency_test_paths[package_name] = sorted(paths)
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "machine": platform.machine(),
    "system": platform.system(),
    "packages": dict(sorted(packages.items())),
    "required_packages": {name: version(name) for name in (
        "pymatgen", "pymatgen-analysis-defects"
    )},
    "candidate_fixture_file_count": len(file_hashes),
    "candidate_fixture_manifest_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
    "candidate_visible_non_fixture_paths": sorted(
        name for name in file_hashes
        if not name.startswith(
            "src/tool_source_code/pymatgen-analysis-defects/tests/test_files/"
        )
    ),
    "candidate_visible_executable_source_paths": sorted(
        name for name in file_hashes
        if Path(name).suffix.lower() in {".py", ".pyc", ".pyo", ".ipynb"}
    ),
    "candidate_visible_dependency_test_paths": dependency_test_paths,
    "upstream_requirements_sha256": file_sha256("/app/upstream-requirements.txt"),
    "adapted_requirements_sha256": file_sha256("/app/evaluator-requirements.txt"),
    "supplemental_requirements_sha256": file_sha256(
        "/app/supplemental-requirements.txt"
    ),
    "task_execution_performed": False,
}, sort_keys=True))
"""


class BuildError(RuntimeError):
    """Raised when reconstruction provenance does not match the reviewed inputs."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def adapt_requirements(source: bytes) -> bytes:
    if sha256_bytes(source) != UPSTREAM_REQUIREMENTS_SHA256:
        raise BuildError("upstream requirements hash differs from the pinned MatTools commit")
    text = source.decode("utf-8")
    adapted = text.replace('python_version >= "3.13"', 'python_version >= "3.11"')
    adapted = adapted.replace('python_version == "3.13"', 'python_version == "3.11"')
    result = adapted.encode("utf-8")
    if sha256_bytes(result) != ADAPTED_REQUIREMENTS_SHA256:
        raise BuildError("the reviewed interpreter-marker adaptation is not reproducible")
    return result


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("mattools_promotion_gate_for_build", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise BuildError("could not load the MatTools promotion harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_snapshot(benchmark_root: Path) -> dict[str, Any]:
    gate = _load_harness()
    snapshot = gate.load_benchmark_snapshot(benchmark_root, strict_official=True)
    requirements_path = benchmark_root / "requirements.txt"
    pyproject_path = benchmark_root / "pyproject.toml"
    upstream_dockerfile = benchmark_root / "Dockerfile"
    requirements = requirements_path.read_bytes()
    adapt_requirements(requirements)
    pyproject = pyproject_path.read_text(encoding="utf-8")
    dockerfile = upstream_dockerfile.read_text(encoding="utf-8")
    if 'requires-python = ">=3.13,<4.0"' not in pyproject:
        raise BuildError("upstream pyproject no longer declares the reviewed Python range")
    if not re.search(r"^FROM python:3\.11\.8\s*$", dockerfile, flags=re.MULTILINE):
        raise BuildError("upstream Dockerfile no longer uses the reviewed Python 3.11.8 base")
    tool_hashes = {
        name: digest
        for name, digest in snapshot.file_hashes.items()
        if name.startswith("src/tool_source_code/")
    }
    tool_manifest = gate._manifest_hash(tool_hashes)
    if len(tool_hashes) != TOOL_SOURCE_FILE_COUNT or tool_manifest != TOOL_SOURCE_MANIFEST_SHA256:
        raise BuildError("upstream tool_source_code manifest differs from the reviewed snapshot")
    fixture_hashes = {
        name: digest
        for name, digest in snapshot.file_hashes.items()
        if name.startswith(CANDIDATE_FIXTURE_PREFIX)
    }
    fixture_manifest = gate._manifest_hash(fixture_hashes)
    if (
        len(fixture_hashes) != CANDIDATE_FIXTURE_FILE_COUNT
        or fixture_manifest != CANDIDATE_FIXTURE_MANIFEST_SHA256
    ):
        raise BuildError("candidate-visible input fixture manifest differs from the snapshot")
    if any(
        Path(name).suffix.lower() in {".py", ".pyc", ".pyo", ".ipynb"}
        for name in fixture_hashes
    ):
        raise BuildError("candidate-visible fixture bundle contains executable/test source")
    return {
        "revision": snapshot.revision,
        "manifest_sha256": snapshot.manifest_sha256,
        "upstream_requirements_sha256": sha256_bytes(requirements),
        "adapted_requirements_sha256": sha256_bytes(adapt_requirements(requirements)),
        "tool_source_file_count": len(tool_hashes),
        "tool_source_manifest_sha256": tool_manifest,
        "candidate_fixture_file_count": len(fixture_hashes),
        "candidate_fixture_manifest_sha256": fixture_manifest,
        "candidate_visible_source_policy": "input-fixtures-only",
        "upstream_docker_python": "3.11.8",
        "upstream_project_python": ">=3.13,<4.0",
    }


def validate_strict_shadow() -> dict[str, Any]:
    source = STRICT_SHADOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(STRICT_SHADOW_PATH))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if not {"execute_code", "execute_file"}.issubset(called_attributes):
        raise BuildError("strict shadow no longer captures code and raw verifier execution")
    if "run_test(" in source:
        raise BuildError("strict shadow must capture output before upstream loose normalization")
    if 'parsed == "ok"' not in source:
        raise BuildError("strict shadow no longer requires an exact raw JSON ok string")
    if "SafeComplexDictParser" not in source or "from utils import" in source:
        raise BuildError("strict shadow must use the reviewed safe parser, never upstream utils")
    return {
        "path": STRICT_SHADOW_RELATIVE.as_posix(),
        "sha256": sha256_file(STRICT_SHADOW_PATH),
        "capture_method": "raw DockerSandbox.execute_file return before run_test normalization",
        "task_execution_performed": False,
    }


def validate_safe_parser_boundary() -> dict[str, Any]:
    parser_source = SAFE_PARSER_PATH.read_text(encoding="utf-8")
    parser_tree = ast.parse(parser_source, filename=str(SAFE_PARSER_PATH))
    forbidden_calls = {
        node.func.id
        for node in ast.walk(parser_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"compile", "eval", "exec", "open", "__import__"}
    }
    if forbidden_calls:
        raise BuildError("safe parser contains a forbidden execution primitive")
    wrapper_source = RUNNER_WRAPPER_PATH.read_text(encoding="utf-8")
    if (
        'types.ModuleType("utils")' not in wrapper_source
        or "safe_utils.ComplexDictParser = SafeComplexDictParser" not in wrapper_source
        or "sys.modules[\"utils\"] = safe_utils" not in wrapper_source
    ):
        raise BuildError("runner wrapper no longer installs the synthetic safe utils module")
    return {
        "safe_parser_path": SAFE_PARSER_RELATIVE.as_posix(),
        "safe_parser_sha256": sha256_file(SAFE_PARSER_PATH),
        "runner_wrapper_path": RUNNER_WRAPPER_RELATIVE.as_posix(),
        "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_PATH),
        "semantic_repairs_path": SEMANTIC_REPAIRS_RELATIVE.as_posix(),
        "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_PATH),
        "candidate_host_eval_removed": True,
        "task_execution_performed": False,
    }


def expected_labels(snapshot: dict[str, Any], strict_shadow: dict[str, Any]) -> dict[str, str]:
    safe_boundary = validate_safe_parser_boundary()
    return {
        "io.ultra.mattools.adapted-requirements-sha256": ADAPTED_REQUIREMENTS_SHA256,
        "io.ultra.mattools.base-image": PYTHON_BASE_IMAGE,
        "io.ultra.mattools.environment-kind": ENVIRONMENT_KIND,
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": snapshot["manifest_sha256"],
        "io.ultra.mattools.safe-parser-sha256": safe_boundary["safe_parser_sha256"],
        "io.ultra.mattools.runner-wrapper-sha256": safe_boundary["runner_wrapper_sha256"],
        "io.ultra.mattools.semantic-repairs-sha256": safe_boundary[
            "semantic_repairs_sha256"
        ],
        "io.ultra.mattools.strict-shadow-sha256": strict_shadow["sha256"],
        "io.ultra.mattools.supplemental-requirements-sha256": sha256_file(SUPPLEMENTAL_PATH),
        "io.ultra.mattools.target-platform": DEFAULT_PLATFORM,
        "io.ultra.mattools.tool-source-manifest-sha256": TOOL_SOURCE_MANIFEST_SHA256,
        "io.ultra.mattools.candidate-fixture-file-count": str(CANDIDATE_FIXTURE_FILE_COUNT),
        "io.ultra.mattools.candidate-fixture-manifest-sha256": (
            CANDIDATE_FIXTURE_MANIFEST_SHA256
        ),
        "io.ultra.mattools.candidate-visible-source-policy": "input-fixtures-only",
        "io.ultra.mattools.upstream-requirements-sha256": UPSTREAM_REQUIREMENTS_SHA256,
        "org.opencontainers.image.revision": snapshot["revision"],
    }


def _run_capture(command: list[str], *, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def inspect_image(image_tag: str) -> dict[str, Any]:
    inspected = _run_capture(["docker", "image", "inspect", image_tag], timeout=60)
    if inspected.returncode != 0:
        raise BuildError(f"evaluator image {image_tag!r} is unavailable")
    try:
        image = json.loads(inspected.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise BuildError("could not parse Docker image inspection") from exc
    image_id = str(image.get("Id") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise BuildError("evaluator image has no immutable SHA-256 image ID")
    return image


def probe_image(image_tag: str) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        image_tag,
        "python",
        "-c",
        PROBE_SCRIPT,
    ]
    probed = _run_capture(command, timeout=600)
    if probed.returncode != 0:
        raise BuildError(f"evaluator metadata probe failed: {probed.stderr.strip()}")
    try:
        payload = json.loads(probed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BuildError("could not parse evaluator metadata probe") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), dict):
        raise BuildError("evaluator metadata probe returned an invalid package map")
    return payload


def verify_image(
    benchmark_root: Path,
    *,
    image_tag: str = EVALUATOR_IMAGE_TAG,
    lock_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = validate_snapshot(benchmark_root)
    strict_shadow = validate_strict_shadow()
    image = inspect_image(image_tag)
    labels = image.get("Config", {}).get("Labels") or {}
    expected = expected_labels(snapshot, strict_shadow)
    label_mismatches = {
        name: {"expected": value, "observed": labels.get(name)}
        for name, value in expected.items()
        if labels.get(name) != value
    }
    if label_mismatches:
        raise BuildError("image provenance labels differ: " + json.dumps(label_mismatches))
    if image.get("Os") != "linux" or image.get("Architecture") != "arm64":
        raise BuildError("reviewed lock is specific to linux/arm64")
    probe = probe_image(image_tag)
    checks = {
        "python_version": probe.get("python_version") == "3.11.8",
        "python_implementation": probe.get("python_implementation") == "CPython",
        "platform": probe.get("system") == "Linux" and probe.get("machine") == "aarch64",
        "required_packages": probe.get("required_packages") == REQUIRED_PACKAGES,
        "candidate_fixture_file_count": probe.get("candidate_fixture_file_count")
        == CANDIDATE_FIXTURE_FILE_COUNT,
        "candidate_fixture_manifest": probe.get("candidate_fixture_manifest_sha256")
        == CANDIDATE_FIXTURE_MANIFEST_SHA256,
        "candidate_visible_paths": probe.get("candidate_visible_non_fixture_paths") == [],
        "candidate_visible_executable_source": (
            probe.get("candidate_visible_executable_source_paths") == []
        ),
        "candidate_visible_dependency_tests": probe.get(
            "candidate_visible_dependency_test_paths"
        )
        == {"pymatgen": [], "pymatgen-analysis-defects": []},
        "upstream_requirements": probe.get("upstream_requirements_sha256")
        == UPSTREAM_REQUIREMENTS_SHA256,
        "adapted_requirements": probe.get("adapted_requirements_sha256")
        == ADAPTED_REQUIREMENTS_SHA256,
        "supplemental_requirements": probe.get("supplemental_requirements_sha256")
        == sha256_file(SUPPLEMENTAL_PATH),
        "no_task_execution": probe.get("task_execution_performed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise BuildError("evaluator metadata checks failed: " + ", ".join(failed))
    packages = {str(name).lower(): str(version) for name, version in probe["packages"].items()}
    if lock_payload is not None:
        validate_lock(lock_payload, packages=packages, probe=probe, snapshot=snapshot)
    return {
        "schema_version": "1",
        "environment_kind": ENVIRONMENT_KIND,
        "official_artifact": False,
        "image_tag": image_tag,
        "image_id": image["Id"],
        "image_size_bytes": image.get("Size"),
        "platform": DEFAULT_PLATFORM,
        "python_version": probe["python_version"],
        "package_count": len(packages),
        "package_map_sha256": canonical_sha256(packages),
        "required_packages": probe["required_packages"],
        "snapshot": snapshot,
        "strict_shadow": strict_shadow,
        "checks": checks,
        "task_execution_performed": False,
    }


def lock_payload(benchmark_root: Path, *, image_tag: str = EVALUATOR_IMAGE_TAG) -> dict[str, Any]:
    verify_image(benchmark_root, image_tag=image_tag)
    probe = probe_image(image_tag)
    packages = dict(sorted(probe["packages"].items()))
    return {
        "schema_version": "1",
        "environment_kind": ENVIRONMENT_KIND,
        "official_artifact": False,
        "variant_reason": (
            "Upstream Dockerfile Python 3.11.8 ignores its Python >=3.13 export; "
            "Python 3.13 has no NumPy 1.26.4 binary. This reconstruction retains "
            "Python 3.11.8, adapts only the export's interpreter markers, and adds "
            "the then-current ruamel.yaml.clib conditional dependency."
        ),
        "python_version": probe["python_version"],
        "platform": {
            "docker": DEFAULT_PLATFORM,
            "machine": probe["machine"],
            "python_implementation": probe["python_implementation"],
            "system": probe["system"],
        },
        "upstream": {
            "revision": UPSTREAM_REVISION,
            "manifest_sha256": UPSTREAM_MANIFEST_SHA256,
            "dockerfile_python": "3.11.8",
            "project_python": ">=3.13,<4.0",
            "requirements_sha256": UPSTREAM_REQUIREMENTS_SHA256,
        },
        "build": {
            "base_image": PYTHON_BASE_IMAGE,
            "builder_path": BUILDER_RELATIVE.as_posix(),
            "builder_sha256": sha256_file(BUILDER_PATH),
            "dockerfile_path": DOCKERFILE_RELATIVE.as_posix(),
            "dockerfile_sha256": sha256_file(DOCKERFILE_PATH),
            "adapted_requirements_sha256": ADAPTED_REQUIREMENTS_SHA256,
            "supplemental_requirements_path": SUPPLEMENTAL_RELATIVE.as_posix(),
            "supplemental_requirements_sha256": sha256_file(SUPPLEMENTAL_PATH),
            "tool_source_file_count": TOOL_SOURCE_FILE_COUNT,
            "tool_source_manifest_sha256": TOOL_SOURCE_MANIFEST_SHA256,
            "candidate_fixture_file_count": CANDIDATE_FIXTURE_FILE_COUNT,
            "candidate_fixture_manifest_sha256": CANDIDATE_FIXTURE_MANIFEST_SHA256,
            "candidate_visible_source_policy": "input-fixtures-only",
            "safe_parser_path": SAFE_PARSER_RELATIVE.as_posix(),
            "safe_parser_sha256": sha256_file(SAFE_PARSER_PATH),
            "runner_wrapper_path": RUNNER_WRAPPER_RELATIVE.as_posix(),
            "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_PATH),
            "strict_shadow_path": STRICT_SHADOW_RELATIVE.as_posix(),
            "strict_shadow_sha256": sha256_file(STRICT_SHADOW_PATH),
            "semantic_repairs_path": SEMANTIC_REPAIRS_RELATIVE.as_posix(),
            "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_PATH),
        },
        "package_map_sha256": canonical_sha256(packages),
        "packages": packages,
    }


def validate_lock(
    payload: dict[str, Any],
    *,
    packages: dict[str, str],
    probe: dict[str, Any],
    snapshot: dict[str, Any],
) -> None:
    expected = lock_payload_shape(packages=packages, probe=probe, snapshot=snapshot)
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise BuildError("environment lock differs from the image: " + ", ".join(mismatches))


def lock_payload_shape(
    *, packages: dict[str, str], probe: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "environment_kind": ENVIRONMENT_KIND,
        "official_artifact": False,
        "python_version": probe["python_version"],
        "platform": {
            "docker": DEFAULT_PLATFORM,
            "machine": probe["machine"],
            "python_implementation": probe["python_implementation"],
            "system": probe["system"],
        },
        "upstream": {
            "revision": snapshot["revision"],
            "manifest_sha256": snapshot["manifest_sha256"],
            "dockerfile_python": "3.11.8",
            "project_python": ">=3.13,<4.0",
            "requirements_sha256": UPSTREAM_REQUIREMENTS_SHA256,
        },
        "build": {
            "base_image": PYTHON_BASE_IMAGE,
            "builder_path": BUILDER_RELATIVE.as_posix(),
            "builder_sha256": sha256_file(BUILDER_PATH),
            "dockerfile_path": DOCKERFILE_RELATIVE.as_posix(),
            "dockerfile_sha256": sha256_file(DOCKERFILE_PATH),
            "adapted_requirements_sha256": ADAPTED_REQUIREMENTS_SHA256,
            "supplemental_requirements_path": SUPPLEMENTAL_RELATIVE.as_posix(),
            "supplemental_requirements_sha256": sha256_file(SUPPLEMENTAL_PATH),
            "tool_source_file_count": TOOL_SOURCE_FILE_COUNT,
            "tool_source_manifest_sha256": TOOL_SOURCE_MANIFEST_SHA256,
            "candidate_fixture_file_count": CANDIDATE_FIXTURE_FILE_COUNT,
            "candidate_fixture_manifest_sha256": CANDIDATE_FIXTURE_MANIFEST_SHA256,
            "candidate_visible_source_policy": "input-fixtures-only",
            "safe_parser_path": SAFE_PARSER_RELATIVE.as_posix(),
            "safe_parser_sha256": sha256_file(SAFE_PARSER_PATH),
            "runner_wrapper_path": RUNNER_WRAPPER_RELATIVE.as_posix(),
            "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_PATH),
            "strict_shadow_path": STRICT_SHADOW_RELATIVE.as_posix(),
            "strict_shadow_sha256": sha256_file(STRICT_SHADOW_PATH),
            "semantic_repairs_path": SEMANTIC_REPAIRS_RELATIVE.as_posix(),
            "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_PATH),
        },
        "package_map_sha256": canonical_sha256(packages),
        "packages": dict(sorted(packages.items())),
    }


def build_image(benchmark_root: Path, *, image_tag: str, target_platform: str) -> None:
    if target_platform != DEFAULT_PLATFORM:
        raise BuildError(f"only the reviewed {DEFAULT_PLATFORM} variant is supported")
    snapshot = validate_snapshot(benchmark_root)
    strict_shadow = validate_strict_shadow()
    safe_boundary = validate_safe_parser_boundary()
    command = [
        "docker",
        "build",
        "--progress=plain",
        "--platform",
        target_platform,
        "--tag",
        image_tag,
        "--file",
        str(DOCKERFILE_PATH),
        "--build-context",
        f"mattools={benchmark_root}",
        "--build-arg",
        f"PYTHON_BASE_IMAGE={PYTHON_BASE_IMAGE}",
        "--build-arg",
        f"MATTOOLS_REVISION={snapshot['revision']}",
        "--build-arg",
        f"MATTOOLS_MANIFEST_SHA256={snapshot['manifest_sha256']}",
        "--build-arg",
        f"UPSTREAM_REQUIREMENTS_SHA256={UPSTREAM_REQUIREMENTS_SHA256}",
        "--build-arg",
        f"ADAPTED_REQUIREMENTS_SHA256={ADAPTED_REQUIREMENTS_SHA256}",
        "--build-arg",
        f"SUPPLEMENTAL_REQUIREMENTS_SHA256={sha256_file(SUPPLEMENTAL_PATH)}",
        "--build-arg",
        f"TOOL_SOURCE_MANIFEST_SHA256={TOOL_SOURCE_MANIFEST_SHA256}",
        "--build-arg",
        f"CANDIDATE_FIXTURE_FILE_COUNT={CANDIDATE_FIXTURE_FILE_COUNT}",
        "--build-arg",
        f"CANDIDATE_FIXTURE_MANIFEST_SHA256={CANDIDATE_FIXTURE_MANIFEST_SHA256}",
        "--build-arg",
        f"SAFE_PARSER_SHA256={safe_boundary['safe_parser_sha256']}",
        "--build-arg",
        f"RUNNER_WRAPPER_SHA256={safe_boundary['runner_wrapper_sha256']}",
        "--build-arg",
        f"STRICT_SHADOW_SHA256={strict_shadow['sha256']}",
        "--build-arg",
        f"SEMANTIC_REPAIRS_SHA256={safe_boundary['semantic_repairs_sha256']}",
        str(REPOSITORY_ROOT),
    ]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if result.returncode != 0:
        raise BuildError("Docker build failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "lock"))
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--image-tag", default=EVALUATOR_IMAGE_TAG)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--lock", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    try:
        if args.command == "build":
            build_image(benchmark_root, image_tag=args.image_tag, target_platform=args.platform)
            report = verify_image(benchmark_root, image_tag=args.image_tag)
        elif args.command == "verify":
            supplied_lock = None
            if args.lock:
                supplied_lock = json.loads(args.lock.read_text(encoding="utf-8"))
            report = verify_image(
                benchmark_root,
                image_tag=args.image_tag,
                lock_payload=supplied_lock,
            )
        else:
            report = lock_payload(benchmark_root, image_tag=args.image_tag)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (BuildError, OSError, json.JSONDecodeError) as exc:
        print(f"mattools evaluator reconstruction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
