from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "build_mattools_evaluator.py"
LOCK_PATH = REPOSITORY_ROOT / "deploy/docker/mattools-evaluator-linux-arm64-lock.json"
PUBLISHED_AUDIT_PATH = (
    REPOSITORY_ROOT / "deploy/docker/mattools-upstream-published-linux-arm64-audit.json"
)
SPEC = importlib.util.spec_from_file_location("build_mattools_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_marker_adaptation_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b'one==1 ; python_version >= "3.13"\ntwo==2 ; python_version == "3.13"\n'
    expected = b'one==1 ; python_version >= "3.11"\ntwo==2 ; python_version == "3.11"\n'
    monkeypatch.setattr(builder, "UPSTREAM_REQUIREMENTS_SHA256", builder.sha256_bytes(source))
    monkeypatch.setattr(builder, "ADAPTED_REQUIREMENTS_SHA256", builder.sha256_bytes(expected))

    assert builder.adapt_requirements(source) == expected
    with pytest.raises(builder.BuildError, match="upstream requirements hash"):
        builder.adapt_requirements(source + b"tampered")


def test_reviewed_lock_is_complete_variant_and_self_consistent() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    packages = lock["packages"]

    assert lock["environment_kind"] == "reviewed-reconstruction-variant"
    assert lock["official_artifact"] is False
    assert lock["python_version"] == "3.11.8"
    assert lock["platform"] == {
        "docker": "linux/arm64",
        "machine": "aarch64",
        "python_implementation": "CPython",
        "system": "Linux",
    }
    assert len(packages) == 290
    assert packages["pymatgen"] == "2024.8.9"
    assert packages["pymatgen-analysis-defects"] == "2024.7.19"
    assert lock["package_map_sha256"] == builder.canonical_sha256(packages)
    assert lock["build"]["builder_path"] == builder.BUILDER_RELATIVE.as_posix()
    assert lock["build"]["builder_sha256"] == builder.sha256_file(builder.BUILDER_PATH)
    assert lock["build"]["dockerfile_sha256"] == builder.sha256_file(builder.DOCKERFILE_PATH)
    assert lock["build"]["supplemental_requirements_sha256"] == builder.sha256_file(
        builder.SUPPLEMENTAL_PATH
    )
    assert lock["build"]["strict_shadow_sha256"] == builder.sha256_file(builder.STRICT_SHADOW_PATH)
    assert lock["build"]["safe_parser_sha256"] == builder.sha256_file(builder.SAFE_PARSER_PATH)
    assert lock["build"]["runner_wrapper_sha256"] == builder.sha256_file(
        builder.RUNNER_WRAPPER_PATH
    )
    assert lock["build"]["semantic_repairs_sha256"] == builder.sha256_file(
        builder.SEMANTIC_REPAIRS_PATH
    )
    assert lock["build"]["candidate_fixture_file_count"] == 141
    assert lock["build"]["candidate_fixture_manifest_sha256"] == (
        builder.CANDIDATE_FIXTURE_MANIFEST_SHA256
    )
    assert lock["build"]["candidate_visible_source_policy"] == "input-fixtures-only"


def test_lock_validation_rejects_package_mutation() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    packages = lock["packages"]
    probe = {
        "python_version": "3.11.8",
        "machine": "aarch64",
        "python_implementation": "CPython",
        "system": "Linux",
    }
    snapshot = {
        "revision": builder.UPSTREAM_REVISION,
        "manifest_sha256": builder.UPSTREAM_MANIFEST_SHA256,
    }

    builder.validate_lock(lock, packages=packages, probe=probe, snapshot=snapshot)
    mutated = copy.deepcopy(lock)
    mutated["packages"]["pymatgen"] = "2026.5.4"
    with pytest.raises(builder.BuildError, match="environment lock differs"):
        builder.validate_lock(
            mutated,
            packages=mutated["packages"],
            probe=probe,
            snapshot=snapshot,
        )
    mutated_builder = copy.deepcopy(lock)
    mutated_builder["build"]["builder_sha256"] = "0" * 64
    with pytest.raises(builder.BuildError, match="environment lock differs"):
        builder.validate_lock(
            mutated_builder,
            packages=packages,
            probe=probe,
            snapshot=snapshot,
        )


def test_strict_shadow_capture_is_hashed_and_pre_normalization() -> None:
    record = builder.validate_strict_shadow()

    assert record["sha256"] == builder.sha256_file(builder.STRICT_SHADOW_PATH)
    assert "execute_file" in record["capture_method"]
    assert "before run_test" in record["capture_method"]
    assert record["task_execution_performed"] is False


def test_candidate_output_parser_boundary_is_hashed_and_eval_free() -> None:
    record = builder.validate_safe_parser_boundary()

    assert record["candidate_host_eval_removed"] is True
    assert record["safe_parser_sha256"] == builder.sha256_file(builder.SAFE_PARSER_PATH)
    assert record["runner_wrapper_sha256"] == builder.sha256_file(builder.RUNNER_WRAPPER_PATH)
    assert record["task_execution_performed"] is False


def test_reconstruction_dockerfile_is_digest_and_hash_locked() -> None:
    dockerfile = builder.DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert builder.PYTHON_BASE_IMAGE in dockerfile
    assert 'io.ultra.mattools.official-artifact="false"' in dockerfile
    assert "--require-hashes" in dockerfile
    assert 'python_version >= "3.13"' in dockerfile
    assert 'python_version >= "3.11"' in dockerfile
    assert "COPY --from=mattools src/tool_source_code /app/tool_source_code" not in dockerfile
    assert (
        "COPY --from=mattools src/tool_source_code/pymatgen-analysis-defects/tests/test_files "
        "/app/tool_source_code/pymatgen-analysis-defects/tests/test_files"
    ) in dockerfile
    assert 'io.ultra.mattools.candidate-visible-source-policy="input-fixtures-only"' in dockerfile


def test_evaluator_probe_rejects_dependency_shipped_test_modules() -> None:
    assert '"candidate_visible_dependency_test_paths": dependency_test_paths' in (
        builder.PROBE_SCRIPT
    )
    assert 'part.casefold() in {"test", "tests"}' in builder.PROBE_SCRIPT


def test_published_image_audit_is_pinned_but_not_auto_approved() -> None:
    audit = json.loads(PUBLISHED_AUDIT_PATH.read_text(encoding="utf-8"))
    environment = audit["environment"]
    comparison = audit["embedded_inputs"]["comparison_to_pinned_revision"]

    assert audit["registry"]["manifest_digest"] == (
        "sha256:f17faff921a093d7ea2bba508a907b348a19035f64b6087d7b62658eac813556"
    )
    assert audit["registry"]["labels"] is None
    assert environment["package_count"] == len(environment["packages"]) == 297
    assert environment["required_packages"] == builder.REQUIRED_PACKAGES
    assert audit["comparability"]["immutable_digest_available"] is True
    assert audit["comparability"]["automatically_approved"] is False
    assert comparison["unchanged_file_count"] == 2755
    assert comparison["changed"] == [
        "src/tool_source_code/pymatgen-analysis-defects/tests/ttt.ipynb"
    ]
    assert audit["task_execution_performed"] is False
