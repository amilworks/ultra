from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_production_materials_sandbox.py"
_REPO_ROOT = _SCRIPT.parents[1]
_SPEC = importlib.util.spec_from_file_location("verify_production_materials_sandbox", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_VERIFY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VERIFY
_SPEC.loader.exec_module(_VERIFY)

ImageInspection = _VERIFY.ImageInspection
SandboxPolicy = _VERIFY.SandboxPolicy
VerificationError = _VERIFY.VerificationError
build_required_source_manifest = _VERIFY.build_required_source_manifest
inspect_image = _VERIFY.inspect_image
load_required_domain_validator_ids = _VERIFY.load_required_domain_validator_ids
policy_for_scope = _VERIFY.policy_for_scope
run_verification = _VERIFY.run_verification
validate_backend_command = _VERIFY.validate_backend_command
validate_calphad_report = _VERIFY.validate_calphad_report
validate_calphad_runtime_junit = _VERIFY.validate_calphad_runtime_junit
validate_calphad_tools_junit = _VERIFY.validate_calphad_tools_junit
validate_domain_report = _VERIFY.validate_domain_report
validate_image = _VERIFY.validate_image
validate_retained_evidence_bundle = _VERIFY.validate_retained_evidence_bundle

GIT_SHA = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64
REQUIRED_VALIDATORS = (
    "materials.atomistics.ase_emt_cu_eos_smoke.v1",
    "materials.calphad.input_domain_axes_fixture_rejection.v1",
    "materials.calphad.nist_al_co_w_phase_field_checkpoints.v2",
    "materials.defects.nacl_generator_stoichiometry.v1",
    "materials.dream3d.geometry_feature_sentinel.v1",
    "materials.ebsd.ipf_cubic_color_coverage.v1",
    "materials.ebsd.ipf_cubic_tsl_corners.v1",
    "materials.ebsd.mackenzie_cubic_distribution.v1",
    "materials.informatics.magpie_ni3al_schema.v1",
    "materials.microstructure.anisotropic_stereology_volume.v1",
    "materials.porosity.porespy_true_void_local_radius.v1",
    "materials.structure.ordering_sensitive_space_group.v1",
    "materials.xrd.fcc_ni_cuka_peak_and_extinctions.v1",
)
PRODUCTION_ENV = {
    "ULTRA_DEEPAGENTS_SANDBOX_IMAGE": "production:test",
    "ULTRA_DEEPAGENTS_SANDBOX_NETWORK": "none",
    "ULTRA_DEEPAGENTS_SANDBOX_CPUS": "8",
    "ULTRA_DEEPAGENTS_SANDBOX_MEMORY": "32g",
    "ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT": "4096",
    "ULTRA_DEEPAGENTS_SANDBOX_SHM_SIZE": "8g",
    "ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS": "21600",
    "ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES": "52428800",
    "ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY": "8",
    "ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES": "true",
}


def _image_inspection(ref: str, title: str) -> ImageInspection:
    labels = {
        "org.opencontainers.image.revision": GIT_SHA,
        "org.opencontainers.image.title": title,
    }
    raw = {
        "Id": IMAGE_ID,
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {
            "Entrypoint": [],
            "Env": ["PYTHONPATH=/opt/ultra-runtime"],
            "Labels": labels,
        },
    }
    return ImageInspection(
        ref=ref,
        image_id=IMAGE_ID,
        revision=GIT_SHA,
        title=title,
        entrypoint=(),
        labels=labels,
        os="linux",
        architecture="amd64",
        raw_inspect=raw,
    )


def _write(path: Path, content: str = "# test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_release_tree(root: Path) -> None:
    for relative in _VERIFY.RELEASE_CRITICAL_FIXED_FILES:
        _write(root / relative)
    for relative in _VERIFY.RELEASE_CRITICAL_TREES:
        _write(root / relative / "release-integrity-fixture.txt")
    _write(
        root / "scripts/materials_readiness_gate.py",
        f"REQUIRED_DOMAIN_VALIDATORS = {REQUIRED_VALIDATORS!r}\n",
    )
    _write(root / "deploy/docker/deepagents-sandbox.Dockerfile", "FROM scratch\n")
    _write(
        root / "backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py",
        'MATPLOTLIBRC = "backend: Agg\\nfigure.dpi: 300\\n"\n',
    )
    _write(root / "deploy/docker/materials-requirements.txt", "pycalphad==0.11.2\n")
    _write(root / "backend/deepagents_runtime/tests/domain_correctness/conftest.py")
    _write(
        root / "backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"
    )
    _write(root / "tests/test_materials_domain_gate_runner.py")
    _write(root / "tests/test_mattools_evaluator_image.py")
    _write(root / "backend/deepagents_runtime/src/ultra_deepagents/materials/__init__.py")
    _write(root / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py")
    _write(root / "backend/deepagents_runtime/src/ultra_deepagents/crystal_plasticity_tools.py")
    _write(
        root
        / "backend/deepagents_runtime/src/ultra_deepagents/degradation_characterization_tools.py"
    )
    _write(root / "backend/deepagents_runtime/tests/test_crystal_plasticity_tools.py")
    _write(root / "backend/deepagents_runtime/tests/test_crystal_plasticity_agent_registration.py")
    _write(root / "backend/deepagents_runtime/tests/test_degradation_characterization_tools.py")
    _write(
        root
        / "backend/deepagents_runtime/tests/test_degradation_characterization_agent_registration.py"
    )
    _write(root / "backend/deepagents_runtime/tests/test_degradation_prompt_routing.py")
    _write(root / "backend/deepagents_runtime/tests/test_kinetics_tools.py")
    _write(root / "backend/deepagents_runtime/tests/test_materials_natural_prompt_fixtures.py")
    _write(root / "backend/deepagents_runtime/tests/test_ngff.py")
    _write(root / "backend/deepagents_runtime/tests/test_paper_tools.py")
    _write(root / "backend/deepagents_runtime/tests/test_runner_paper_preload.py")
    _write(root / "backend/deepagents_runtime/tests/test_sensor_tools.py")
    _write(root / "backend/deepagents_runtime/tests/test_sensor_worker_core_smoke.py")
    _write(root / "backend/deepagents_runtime/tests/test_vision_subagent.py")
    _write(root / "backend/deepagents_runtime/tests/test_zarr_tree_identity_contract.py")
    calphad_data = root / "backend/deepagents_runtime/materials_data/calphad"
    shutil.rmtree(calphad_data)
    shutil.copytree(
        _REPO_ROOT / "backend/deepagents_runtime/materials_data/calphad",
        calphad_data,
    )
    for raw_path in _VERIFY.REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S:
        relative = Path(raw_path)
        source = _REPO_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write(root / "bin/ultra-control", "control-binary-fixture\n")
    _write(root / "frontend/dist/index.html", "<main>release</main>\n")
    _write(root / "frontend/dist/assets/app.js", "console.log('release')\n")
    release_artifacts = _VERIFY.build_release_artifact_identities(root)
    _write(
        root / "release-manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "release_sha": GIT_SHA,
                "source": {
                    "required_materials": build_required_source_manifest(root),
                },
                "targets": {
                    "control_binary": "bin/ultra-control",
                    "control_binary_identity": release_artifacts["control_binary"],
                    "frontend_dist": "frontend/dist",
                    "frontend_dist_identity": release_artifacts["frontend_dist"],
                },
            }
        ),
    )


def test_release_critical_sources_include_protected_promotion_boundary() -> None:
    required = {path.as_posix() for path in _VERIFY.RELEASE_CRITICAL_FIXED_FILES}

    assert ".github/workflows/materials-production-qualification.yml" in required
    assert "scripts/materials_promotion_envelope.py" in required


def _calphad_experimental_benchmark_wrapper() -> dict[str, object]:
    report = {
        "schema_version": "ultra.calphad.experimental_benchmark.v1",
        "benchmark_id": "materials.calphad.al_co_w_experimental_two_lane.v1",
        "status": "passed",
        "required_independent_invariant": True,
        "production_promotion_blocked": False,
        "blocking_reasons": [],
        "lanes": {
            "calibration": {
                "classification": "calibration",
                "independent_validation": False,
                "required": True,
                "status": "passed",
                "observation_count": 6,
                "metrics": {
                    "weighted_rms_z": 0.49,
                    "weighted_rms_z_max": 1.0,
                    "max_abs_z": 0.79,
                    "max_abs_z_max": 2.0,
                },
            },
            "held_out": {
                "classification": "held_out",
                "independent_validation": True,
                "required": True,
                "status": "passed",
                "observation_count": 4,
                "metrics": {
                    "mae_K": 12.34,
                    "mae_K_max": 20.0,
                    "max_abs_error_K": 20.42,
                    "max_abs_error_K_max": 30.0,
                },
                "observations": [
                    {
                        "reported_uncertainty_K": None,
                        "uncertainty_status": "not_reported_numerically",
                    }
                    for _ in range(4)
                ],
            },
        },
    }
    encoded = json.dumps(report, sort_keys=True).encode("utf-8")
    return {
        "relative_path": "calphad-experimental-benchmark.json",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "report": report,
    }


def _domain_report(
    *,
    skipped: int = 0,
    image_id: str = IMAGE_ID,
    validators: tuple[str, ...] = REQUIRED_VALIDATORS,
) -> dict[str, object]:
    invariants = [
        {
            "validator_id": validator_id,
            "test_id": f"test_invariant_{index}",
            "required": True,
            "outcome": "pass",
        }
        for index, validator_id in enumerate(validators)
    ]
    return {
        "gate": "materials-domain-gate",
        "scope": "deterministic-domain-invariants",
        "status": "passed" if skipped == 0 else "failed",
        "junit": {
            "tests": len(invariants),
            "failures": 0,
            "errors": 0,
            "skipped": skipped,
        },
        "pytest": {"exit_code": 0},
        "version_drift": [],
        "expected_pins": {"pycalphad": "0.11.2"},
        "installed_direct": {"pycalphad": "0.11.2"},
        "installed_packages": [
            {
                "name": "pycalphad",
                "normalized_name": "pycalphad",
                "version": "0.11.2",
            },
            {
                "name": "pytest",
                "normalized_name": "pytest",
                "version": "8.4.2",
            },
        ],
        "invariant_evidence": {
            "complete": skipped == 0,
            "record_count": len(invariants),
            "passed": len(invariants),
            "failed": 0,
            "errors": [],
        },
        "invariants": invariants,
        "image": {"id": image_id},
        "provenance_policy": {"status": "enforced"},
        "calphad_experimental_benchmark": _calphad_experimental_benchmark_wrapper(),
        "runtime": {
            "python": "3.11.13",
            "python_implementation": "CPython",
            "platform": "linux-test",
        },
    }


def _write_runtime_junit(
    path: Path,
    *,
    skipped: int = 0,
    tests: int = _VERIFY.REQUIRED_CALPHAD_RUNTIME_TEST_COUNT,
    rename_first: bool = False,
) -> None:
    cases = []
    identities = [
        ("tests.test_calphad_runtime", name) for name in _VERIFY.REQUIRED_CALPHAD_CORE_TEST_NAMES
    ] + [("tests.test_calphad_cli", name) for name in _VERIFY.REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES]
    identities = identities[:tests]
    if rename_first and identities:
        identities[0] = (identities[0][0], "test_renamed_but_padded_runtime_case")
    for index, (classname, name) in enumerate(identities):
        child = "<skipped />" if index < skipped else ""
        cases.append(f'<testcase classname="{classname}" name="{name}">{child}</testcase>')
    path.write_text(
        (
            "<testsuites><testsuite "
            f' tests="{len(identities)}" failures="0" errors="0" skipped="{skipped}">'
            + "".join(cases)
            + "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _write_tools_junit(
    path: Path,
    *,
    skipped: int = 0,
    tests: int = _VERIFY.REQUIRED_CALPHAD_TOOLS_TEST_COUNT,
    rename_first: bool = False,
) -> None:
    cases = []
    names = list(_VERIFY.REQUIRED_CALPHAD_TOOL_TEST_NAMES[:tests])
    if rename_first and names:
        names[0] = "test_renamed_but_padded_tool_case"
    for index, name in enumerate(names):
        child = "<skipped />" if index < skipped else ""
        cases.append(
            f'<testcase classname="tests.test_calphad_tools" name="{name}">{child}</testcase>'
        )
    path.write_text(
        (
            "<testsuites><testsuite "
            f' tests="{len(names)}" failures="0" errors="0" skipped="{skipped}">'
            + "".join(cases)
            + "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _write_domain_junit(path: Path, report: dict[str, object]) -> None:
    invariants = report["invariants"]
    assert isinstance(invariants, list)
    suites = ET.Element("testsuites")
    suite = ET.SubElement(
        suites,
        "testsuite",
        tests=str(len(invariants)),
        failures="0",
        errors="0",
        skipped="0",
    )
    for record in invariants:
        assert isinstance(record, dict)
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname="tests.domain_correctness.test_materials_invariants",
            name=str(record["test_id"]),
        )
        properties = ET.SubElement(testcase, "properties")
        ET.SubElement(
            properties,
            "property",
            name="materials_invariant_evidence",
            value=json.dumps(record, sort_keys=True),
        )
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)


def _fake_host_suite(_repo_root: Path, junit_path: Path) -> dict[str, object]:
    _write_tools_junit(junit_path)
    stdout = f"{_VERIFY.REQUIRED_CALPHAD_TOOLS_TEST_COUNT} passed\n"
    stderr = ""
    return {
        "runner": "fixture",
        "exit_code": 0,
        "source_isolation": {
            "config": "/dev/null",
            "conftest_loading": False,
            "plugin_autoload": False,
            "pytest_plugins": "",
            "pythonpath": "retained-host-source/backend/deepagents_runtime/src",
            "uv_sync": False,
        },
        "stdout_text": stdout,
        "stderr_text": stderr,
        "stdout_size_bytes": len(stdout.encode()),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_size_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _calphad_report(
    workspace: Path,
    *,
    baked_materials_path: str = "/opt/ultra-runtime/ultra_deepagents/materials",
) -> dict[str, object]:
    calphad_root = workspace / "backend/deepagents_runtime/materials_data/calphad"
    manifest_path = calphad_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    entry = manifest["databases"][0]
    materials_root = workspace / "backend/deepagents_runtime/src/ultra_deepagents/materials"
    material_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(materials_root.glob("*.py"))
    }
    return {
        "status": "passed",
        "equilibrium_schema_version": "ultra.calphad.equilibrium.v2",
        "baked_materials_path": baked_materials_path,
        "materials_source_hashes": material_hashes,
        "materials_baked_hashes": material_hashes,
        "source_manifest_sha256": manifest_hash,
        "embedded_manifest_sha256": manifest_hash,
        "database_count": 1,
        "databases": [
            {
                "database_id": entry["database_id"],
                "filename": entry["filename"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
                "format": entry["format"],
                "assessment_pressure_limits_Pa": entry["assessment_pressure_limits_Pa"],
                "elements": entry["elements"],
                "phases": entry["phases"],
                "pycalphad_parse_supported": True,
                "ultra_inspection_supported": True,
            }
        ],
    }


@dataclass
class _Response:
    output: str = "materials parity passed\n"
    exit_code: int = 0
    truncated: bool = False


class _FakeBackend:
    def __init__(
        self,
        workspace: Path,
        outputs: Path,
        image_id: str,
        policy: SandboxPolicy,
    ) -> None:
        self.workspace = workspace
        self.outputs = outputs
        self.image_id = image_id
        self.config = policy
        docker_contract = (
            workspace / "backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py"
        )
        if docker_contract.is_file():
            matplotlibrc = _VERIFY.load_staged_matplotlibrc(workspace)
            (workspace / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)
            (workspace / ".cache/numba").mkdir(parents=True, exist_ok=True)
            (workspace / ".tmp").mkdir(parents=True, exist_ok=True)
            (workspace / "matplotlibrc").write_text(matplotlibrc, encoding="utf-8")
            (workspace / ".cache/matplotlib/matplotlibrc").write_text(
                matplotlibrc,
                encoding="utf-8",
            )

    def build_docker_command(self, command: str) -> list[str]:
        result = [
            "docker",
            "run",
            "--rm",
            "--label",
            "ultra.sandbox=1",
            "--label",
            f"ultra.sandbox.cap={self.config.timeout_seconds}",
            "--network",
            self.config.network,
            "--cap-drop",
            "ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--volume",
            f"{self.workspace.resolve()}:/workspace:rw",
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "MPLCONFIGDIR=/workspace/.cache/matplotlib",
            "--env",
            "NUMBA_CACHE_DIR=/workspace/.cache/numba",
            "--env",
            "XDG_CACHE_HOME=/workspace/.cache",
            "--env",
            "HOME=/workspace",
            "--env",
            "TMPDIR=/workspace/.tmp",
            "--security-opt",
            "no-new-privileges",
            "--volume",
            f"{self.outputs.resolve()}:/outputs:rw",
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory,
            "--pids-limit",
            str(self.config.pids_limit),
            "--shm-size",
            self.config.shm_size,
        ]
        if self.config.gpus:
            result.extend(["--gpus", self.config.gpus])
        result.extend([self.image_id, "bash", "-lc", command])
        return result

    def execute(self, command: str, *, timeout: int | None = None) -> _Response:
        _ = command, timeout
        domain = self.outputs / "domain"
        domain.mkdir(parents=True)
        domain_report = _domain_report()
        requirements_path = self.workspace / "deploy/docker/materials-requirements.txt"
        invariant_path = (
            self.workspace
            / "backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"
        )
        validation_path = (
            self.workspace
            / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"
        )
        domain_report["requirements"] = {
            "path": "/workspace/deploy/docker/materials-requirements.txt",
            "sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        }
        domain_report["test_source"] = {
            "path": (
                "/workspace/backend/deepagents_runtime/tests/domain_correctness/"
                "test_materials_invariants.py"
            ),
            "sha256": hashlib.sha256(invariant_path.read_bytes()).hexdigest(),
        }
        domain_report["git"] = {"sha": GIT_SHA, "ref": GIT_SHA, "dirty": False}
        domain_report["image"] = {
            "id": self.image_id,
            "ref": (
                "materials:test" if self.config.source == "ci_fixed_limits" else "production:test"
            ),
        }
        assert isinstance(domain_report["runtime"], dict)
        domain_report["runtime"]["materials_validation"] = {
            "module": "ultra_deepagents.materials.validation",
            "path": (
                "/opt/ultra/src/ultra_deepagents/materials/validation.py"
                if self.config.source == "ci_fixed_limits"
                else "/opt/ultra-runtime/ultra_deepagents/materials/validation.py"
            ),
            "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        }
        domain_report["runtime"]["calphad_runtime_preflight"] = {
            "path": "/outputs/calphad-runtime-junit.xml",
            "required": True,
            "validated": True,
            "junit": {
                "tests": _VERIFY.REQUIRED_CALPHAD_RUNTIME_TEST_COUNT,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "time_seconds": 1.0,
            },
            "core_tests": _VERIFY.REQUIRED_CALPHAD_CORE_TEST_COUNT,
            "typed_cli_tests": _VERIFY.REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT,
            "required_adversarial_test_names": sorted(
                _VERIFY.REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES
            ),
        }
        domain_report["pytest"] = {
            "exit_code": 0,
            "command": [
                "/usr/local/bin/python",
                "-m",
                "pytest",
                (
                    "/workspace/backend/deepagents_runtime/tests/domain_correctness/"
                    "test_materials_invariants.py"
                ),
                "-q",
                "-ra",
                "--color=no",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "-o",
                "junit_family=legacy",
                "--junitxml=/outputs/domain/materials-junit.xml",
            ],
        }
        benchmark_wrapper = domain_report["calphad_experimental_benchmark"]
        benchmark_report = benchmark_wrapper["report"]
        benchmark_validator_path = self.workspace / "scripts/calphad_experimental_benchmark.py"
        benchmark_manifest_path = (
            self.workspace / "backend/deepagents_runtime/materials_data/calphad/"
            "experimental_benchmark_manifest.json"
        )
        benchmark_report["source_manifest"] = {
            "relative_path": (
                "backend/deepagents_runtime/materials_data/calphad/"
                "experimental_benchmark_manifest.json"
            ),
            "sha256": hashlib.sha256(benchmark_manifest_path.read_bytes()).hexdigest(),
            "size_bytes": benchmark_manifest_path.stat().st_size,
        }
        benchmark_wrapper["validator"] = {
            "path": "/workspace/scripts/calphad_experimental_benchmark.py",
            "sha256": hashlib.sha256(benchmark_validator_path.read_bytes()).hexdigest(),
        }
        benchmark_bytes = json.dumps(benchmark_report, sort_keys=True).encode("utf-8")
        benchmark_wrapper["sha256"] = hashlib.sha256(benchmark_bytes).hexdigest()
        benchmark_wrapper["size_bytes"] = len(benchmark_bytes)
        (domain / "calphad-experimental-benchmark.json").write_bytes(benchmark_bytes)
        (domain / "materials-domain-gate.json").write_text(
            json.dumps(domain_report), encoding="utf-8"
        )
        (domain / "materials-domain-gate.md").write_text("passed\n", encoding="utf-8")
        (domain / "materials-pip-freeze.txt").write_text(
            "pycalphad==0.11.2\npytest==8.4.2\n",
            encoding="utf-8",
        )
        _write_domain_junit(domain / "materials-junit.xml", domain_report)
        (domain / "materials-pytest.stdout.txt").write_text("13 passed\n", encoding="utf-8")
        (domain / "materials-pytest.stderr.txt").write_text("", encoding="utf-8")
        (self.outputs / "calphad-embedded-probe.json").write_text(
            json.dumps(
                _calphad_report(
                    self.workspace,
                    baked_materials_path=(
                        "/opt/ultra/src/ultra_deepagents/materials"
                        if self.config.source == "ci_fixed_limits"
                        else "/opt/ultra-runtime/ultra_deepagents/materials"
                    ),
                )
            ),
            encoding="utf-8",
        )
        _write_runtime_junit(self.outputs / "calphad-runtime-junit.xml")
        return _Response()


def _factory(workspace: Path, outputs: Path, image_id: str, policy: SandboxPolicy) -> _FakeBackend:
    assert workspace.is_dir()
    return _FakeBackend(workspace, outputs, image_id, policy)


def test_inspect_image_reads_immutable_id_revision_and_entrypoint(monkeypatch) -> None:
    payload = [
        {
            "Id": IMAGE_ID,
            "Architecture": "arm64",
            "Os": "linux",
            "Config": {
                "Entrypoint": ["python", "/opt/gate.py"],
                "Labels": {
                    "org.opencontainers.image.revision": GIT_SHA,
                    "org.opencontainers.image.title": "Ultra deterministic materials domain gate",
                },
            },
        }
    ]

    def fake_run(*args, **kwargs):
        _ = args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(_VERIFY.subprocess, "run", fake_run)
    inspected = inspect_image("materials:test")

    assert inspected.image_id == IMAGE_ID
    assert inspected.revision == GIT_SHA
    assert inspected.entrypoint == ("python", "/opt/gate.py")
    assert inspected.os == "linux"
    assert inspected.architecture == "arm64"
    assert inspected.raw_inspect["Id"] == IMAGE_ID


def test_raw_docker_inspect_must_match_reported_label_identity(tmp_path: Path) -> None:
    inspected = _image_inspection("production:test", "Ultra Deep Agents scientific sandbox")
    raw = dict(inspected.raw_inspect)
    raw["Config"] = {
        **raw["Config"],
        "Labels": {
            **raw["Config"]["Labels"],
            "org.opencontainers.image.revision": "f" * 40,
        },
    }
    forged = ImageInspection(
        **{
            **inspected.__dict__,
            "raw_inspect": raw,
        }
    )
    with pytest.raises(VerificationError, match="labels disagree"):
        _VERIFY._retain_image_inspection(forged, tmp_path, role="executed")


def test_image_contract_rejects_wrong_revision_title_and_mutable_id() -> None:
    failures = validate_image(
        ImageInspection(
            ref="wrong:test",
            image_id="wrong:test",
            revision="f" * 40,
            title="Wrong image",
            entrypoint=(),
        ),
        expected_git_sha=GIT_SHA,
        scope="production-full",
        allow_entrypoint=False,
    )

    assert any("immutable configuration ID" in item for item in failures)
    assert any("OCI revision" in item for item in failures)
    assert any("wrong image title" in item for item in failures)


def _args(*, scope: str = "production-full", image: str = "production:test", **updates):
    values = {
        "repo_root": Path("."),
        "output_dir": Path(".tmp/test-materials-parity"),
        "image": image,
        "expected_git_sha": GIT_SHA,
        "scope": scope,
        "prepare_entrypoint_adapter": False,
        "cpus": 2.0,
        "memory": "8g",
        "pids_limit": 512,
        "shm_size": "1g",
        "timeout_seconds": 1200,
        "output_limit_bytes": 8 * 1024 * 1024,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _set_production_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PRODUCTION_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_GPUS", "")


def _run_full_fixture(
    repo: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_factory=_factory,
) -> tuple[int, Path]:
    _minimal_release_tree(repo)
    _set_production_environment(monkeypatch)
    args = _args(repo_root=repo, output_dir=output)

    def inspector(ref: str) -> ImageInspection:
        return _image_inspection(ref, "Ultra Deep Agents scientific sandbox")

    return run_verification(
        args,
        inspector=inspector,
        backend_factory=backend_factory,
        host_suite_runner=_fake_host_suite,
    )


def test_backend_contract_requires_every_resource_and_security_bound() -> None:
    policy = SandboxPolicy()
    workspace = Path("/tmp/workspace")
    outputs = Path("/tmp/outputs")
    valid = _FakeBackend(workspace, outputs, IMAGE_ID, policy).build_docker_command("true")
    validation = {
        "image_id": IMAGE_ID,
        "policy": policy,
        "workspace": workspace,
        "outputs": outputs,
        "expected_command": "true",
    }
    assert validate_backend_command(valid, **validation) == []

    invalid = [token for token in valid if token not in {"--read-only", "no-new-privileges"}]
    failures = validate_backend_command(invalid, **validation)
    assert any("--read-only" in item for item in failures)
    assert any("--security-opt" in item for item in failures)

    wrong_workspace = list(valid)
    volume_index = wrong_workspace.index(f"{workspace.resolve()}:/workspace:rw")
    wrong_workspace[volume_index] = "/tmp/unverified:/workspace:rw"
    failures = validate_backend_command(wrong_workspace, **validation)
    assert any("exact verified workspace" in item for item in failures)

    wrong_payload = list(valid)
    wrong_payload[-1] = "false"
    failures = validate_backend_command(wrong_payload, **validation)
    assert any("shell payload differs" in item for item in failures)


def test_parity_execution_keeps_pytest_temps_off_the_staged_source_tree() -> None:
    command = _VERIFY._execution_command(
        expected_git_sha=GIT_SHA,
        image=_image_inspection("production:test", "Ultra Deep Agents scientific sandbox"),
        requirements_sha256="a" * 64,
        scope="production-full",
    )

    assert f"export TMPDIR={_VERIFY.PARITY_TMPDIR}" in command
    assert 'mkdir -p "$TMPDIR" /outputs/domain' in command
    assert "/workspace/.tmp" not in command


@pytest.mark.parametrize(
    "bypass",
    [
        ["--network", "host"],
        ["--network=host"],
        ["--cap-add", "SYS_ADMIN"],
        ["--privileged"],
        ["--device", "/dev/kvm"],
        ["--cpus", "99"],
        ["--memory=0"],
        ["--read-only=false"],
        ["--security-opt", "seccomp=unconfined"],
        ["--volume", "/:/host:rw"],
    ],
)
def test_backend_contract_rejects_contradictory_or_override_flags(
    bypass: list[str],
) -> None:
    policy = SandboxPolicy()
    command = _FakeBackend(
        Path("/tmp/workspace"), Path("/tmp/outputs"), IMAGE_ID, policy
    ).build_docker_command("true")
    image_index = command.index(IMAGE_ID)
    command[image_index:image_index] = bypass

    assert validate_backend_command(
        command,
        image_id=IMAGE_ID,
        policy=policy,
        workspace=Path("/tmp/workspace"),
        outputs=Path("/tmp/outputs"),
        expected_command="true",
    )


def test_report_validators_fail_on_skip_image_drift_and_missing_calphad_parse(
    tmp_path: Path,
) -> None:
    domain_failures = validate_domain_report(
        _domain_report(skipped=1, image_id="sha256:" + "e" * 64),
        image_id=IMAGE_ID,
        required_validator_ids=REQUIRED_VALIDATORS,
    )
    assert any("skipped" in item for item in domain_failures)
    assert any("image ID" in item for item in domain_failures)

    _minimal_release_tree(tmp_path)
    calphad = _calphad_report(tmp_path)
    calphad["databases"][0]["pycalphad_parse_supported"] = False  # type: ignore[index]
    assert any("did not parse with pycalphad" in item for item in validate_calphad_report(calphad))
    wrong_schema = {
        **_calphad_report(tmp_path),
        "equilibrium_schema_version": "ultra.calphad.equilibrium.v1",
    }
    assert any("schema v2" in item for item in validate_calphad_report(wrong_schema))


def test_domain_report_requires_passing_calphad_independent_holdout() -> None:
    missing = _domain_report()
    del missing["calphad_experimental_benchmark"]
    failures = validate_domain_report(
        missing,
        image_id=IMAGE_ID,
        required_validator_ids=REQUIRED_VALIDATORS,
    )
    assert any("experimental benchmark evidence is missing" in item for item in failures)

    failed = _domain_report()
    benchmark = failed["calphad_experimental_benchmark"]["report"]
    benchmark["status"] = "failed"
    benchmark["production_promotion_blocked"] = True
    benchmark["lanes"]["held_out"]["status"] = "failed"
    benchmark["lanes"]["held_out"]["metrics"]["max_abs_error_K"] = 31.0
    failures = validate_domain_report(
        failed,
        image_id=IMAGE_ID,
        required_validator_ids=REQUIRED_VALIDATORS,
    )
    assert any("locked policy" in item for item in failures)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest_hash", "reviewed release contract"),
        ("source_hash", "source hashes"),
        ("format", "format is unsupported or mismatched"),
        ("pressure", "pressure scope is invalid"),
    ],
)
def test_calphad_report_rejects_stale_hash_format_and_pressure_contracts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    _minimal_release_tree(tmp_path)
    report = _calphad_report(tmp_path)
    if mutation == "manifest_hash":
        report["source_manifest_sha256"] = "0" * 64
        report["embedded_manifest_sha256"] = "0" * 64
    elif mutation == "source_hash":
        report["materials_source_hashes"]["calphad.py"] = "0" * 64  # type: ignore[index]
    elif mutation == "format":
        report["databases"][0]["format"] = "db"  # type: ignore[index]
    else:
        report["databases"][0]["assessment_pressure_limits_Pa"] = [0.0, 101325.0]  # type: ignore[index]

    assert any(message in failure for failure in validate_calphad_report(report))


def test_domain_report_requires_exact_13_readiness_validator_ids() -> None:
    assert (
        validate_domain_report(
            _domain_report(),
            image_id=IMAGE_ID,
            required_validator_ids=REQUIRED_VALIDATORS,
        )
        == []
    )
    one_test = _domain_report(validators=(REQUIRED_VALIDATORS[0],))
    assert validate_domain_report(
        one_test,
        image_id=IMAGE_ID,
        required_validator_ids=REQUIRED_VALIDATORS,
    )
    wrong = _domain_report(validators=(*REQUIRED_VALIDATORS[:-1], "materials.fake.bypass.v1"))
    wrong_failures = validate_domain_report(
        wrong,
        image_id=IMAGE_ID,
        required_validator_ids=REQUIRED_VALIDATORS,
    )
    assert any("exact readiness validator set" in failure for failure in wrong_failures)


def test_readiness_validator_contract_is_read_via_ast(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    assert load_required_domain_validator_ids(tmp_path) == REQUIRED_VALIDATORS
    _write(
        tmp_path / "scripts/materials_readiness_gate.py",
        "raise RuntimeError('must not execute')\n"
        f"REQUIRED_DOMAIN_VALIDATORS = {REQUIRED_VALIDATORS!r}\n",
    )
    assert load_required_domain_validator_ids(tmp_path) == REQUIRED_VALIDATORS


def test_calphad_runtime_junit_requires_all_39_tests_without_skips(tmp_path: Path) -> None:
    path = tmp_path / "calphad-runtime-junit.xml"
    _write_runtime_junit(path)
    summary, failures = validate_calphad_runtime_junit(path)
    assert summary == {"tests": 39, "failures": 0, "errors": 0, "skipped": 0}
    assert failures == []

    _write_runtime_junit(path, skipped=1)
    _, failures = validate_calphad_runtime_junit(path)
    assert any("skipped" in failure for failure in failures)
    _write_runtime_junit(path, tests=25)
    _, failures = validate_calphad_runtime_junit(path)
    assert any("exactly 39" in failure for failure in failures)

    # A padded core suite cannot substitute for the real typed CLI smoke.
    _write_runtime_junit(path)
    path.write_text(
        path.read_text().replace("tests.test_calphad_cli", "tests.test_calphad_runtime"),
        encoding="utf-8",
    )
    _, failures = validate_calphad_runtime_junit(path)
    assert any("exactly 3 real typed CLI" in failure for failure in failures)

    # Correct counts and classnames cannot hide a renamed/removed scientific case.
    _write_runtime_junit(path, rename_first=True)
    _, failures = validate_calphad_runtime_junit(path)
    assert any("exact required core test identities" in failure for failure in failures)


def test_calphad_tools_junit_requires_exact_56_tests_without_skips(tmp_path: Path) -> None:
    path = tmp_path / "calphad-tools-junit.xml"
    _write_tools_junit(path)
    summary, failures = validate_calphad_tools_junit(path)
    assert summary == {"tests": 56, "failures": 0, "errors": 0, "skipped": 0}
    assert failures == []

    _write_tools_junit(path, skipped=1)
    _, failures = validate_calphad_tools_junit(path)
    assert any("skipped" in failure for failure in failures)
    _write_tools_junit(path, tests=21)
    _, failures = validate_calphad_tools_junit(path)
    assert any("exactly 56" in failure for failure in failures)
    _write_tools_junit(path, rename_first=True)
    _, failures = validate_calphad_tools_junit(path)
    assert any("exact required test identities" in failure for failure in failures)


def test_release_calphad_test_contract_includes_format_pressure_scheil_and_mqmqa_adversaries() -> (
    None
):
    assert len(_VERIFY.REQUIRED_CALPHAD_CORE_TEST_NAMES) == 36
    assert len(_VERIFY.REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES) == 3
    assert len(_VERIFY.REQUIRED_CALPHAD_TOOL_TEST_NAMES) == 56
    assert {
        "test_parser_uses_the_validated_database_format",
        "test_database_input_rejects_unregistered_db_suffix",
        "test_pinned_pycalphad_database_corpus_parses_all_registered_text_formats",
        "test_dat_inspection_records_the_actual_parser_format",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits0]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits3]",
    } <= set(_VERIFY.REQUIRED_CALPHAD_CORE_TEST_NAMES)
    assert {
        "test_resource_missing_owner_pressure_scope_never_executes",
        "test_equilibrium_rejects_pressure_outside_owner_scope_before_execution",
        "test_cli_requires_fixed_or_bounded_resource_pressure_scope",
        "test_scheil_typed_tool_retains_va_and_returns_mass_closed_bounded_summary",
        "test_cli_scheil_uses_fixed_kernel_limits_and_retains_inspection_lineage",
    } <= set(_VERIFY.REQUIRED_CALPHAD_TOOL_TEST_NAMES)
    assert {
        "backend/deepagents_runtime/tests/test_calphad_runtime.py",
        "backend/deepagents_runtime/tests/test_calphad_tools.py",
        "backend/deepagents_runtime/tests/test_materials_live_trace.py",
        "scripts/calphad_cross_language_gate.py",
        "tests/test_calphad_cross_language_gate.py",
    } <= set(_VERIFY.REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S)


def test_materials_images_and_domain_gate_bind_the_reviewed_pressure_format_contract() -> None:
    for relative in (
        "deploy/docker/materials-domain-gate.Dockerfile",
        "deploy/docker/deepagents-sandbox.Dockerfile",
    ):
        dockerfile = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        assert _VERIFY.REQUIRED_CALPHAD_MANIFEST_SHA256 in dockerfile
        assert "Database.from_file(str(path), fmt=database_format)" in dockerfile
        assert 'record.get("assessment_pressure_limits_Pa")' in dockerfile
        assert 'inspection["assessment_pressure_limits_Pa"]' in dockerfile
    runner = (_REPO_ROOT / "scripts/run_materials_domain_gate.sh").read_text(encoding="utf-8")
    assert "ULTRA_MATERIALS_GATE_REQUIRE_CALPHAD_RUNTIME_JUNIT=1" in runner
    assert "--calphad-runtime-junit /reports/calphad-runtime-junit.xml" in runner
    assert 'testcase.find("skipped") is not None' in runner
    assert "expected_damask_cases" in runner
    assert "observed_damask_cases != expected_damask_cases" in runner
    assert "exact 10 DAMASK 3.1.0" in runner
    assert "reference comparisons" in runner


@pytest.mark.parametrize(
    "relative",
    [
        "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py",
        "backend/deepagents_runtime/materials_data/calphad/manifest.json",
        "backend/deepagents_runtime/tests/test_calphad_runtime.py",
        "backend/deepagents_runtime/tests/test_materials_live_trace.py",
    ],
)
def test_release_manifest_rejects_stale_calphad_input_hashes(
    tmp_path: Path,
    relative: str,
) -> None:
    _minimal_release_tree(tmp_path)
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b"\nstale\n")

    with pytest.raises(VerificationError, match="CALPHAD input hash drift"):
        build_required_source_manifest(tmp_path)


def test_host_calphad_tools_runner_executes_the_complete_test_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _minimal_release_tree(tmp_path)
    junit_path = tmp_path / "tools.xml"
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        _write_tools_junit(junit_path)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{_VERIFY.REQUIRED_CALPHAD_TOOLS_TEST_COUNT} passed\n",
            stderr="",
        )

    monkeypatch.setattr(_VERIFY.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(_VERIFY.subprocess, "run", fake_run)

    result = _VERIFY.run_host_calphad_tools_suite(tmp_path, junit_path)

    command = captured["command"]
    assert isinstance(command, list)
    assert "-k" not in command
    assert (
        command.count(str(tmp_path / "backend/deepagents_runtime/tests/test_calphad_tools.py")) == 1
    )
    assert result["exit_code"] == 0


def test_release_tree_requires_matching_per_file_content_hash_manifest(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    verified = _VERIFY._verify_source_revision(tmp_path, GIT_SHA)
    assert verified["required_materials"]["file_count"] > 0

    _write(tmp_path / "deploy/docker/materials-requirements.txt", "pycalphad==999\n")
    with pytest.raises(VerificationError, match="content hashes"):
        _VERIFY._verify_source_revision(tmp_path, GIT_SHA)


def test_release_manifest_covers_calphad_mattools_and_runtime_control_chain() -> None:
    manifest = build_required_source_manifest(Path(__file__).resolve().parents[1])
    paths = {record["path"] for record in manifest["files"]}
    required = {
        "backend/controlplane/api/openapi.yaml",
        "backend/controlplane/internal/domain/calphad.go",
        "backend/controlplane/internal/httpapi/handlers_calphad.go",
        "backend/controlplane/internal/httpapi/calphad_scientific_evidence.go",
        "backend/controlplane/internal/openapi/generated.gen.go",
        "backend/controlplane/internal/store/calphad_ledger.go",
        "backend/controlplane/internal/store/schema.sql",
        "backend/controlplane/migrations/000008_calphad_revision_ledger.up.sql",
        "backend/deepagents_runtime/src/ultra_deepagents/context.py",
        "backend/deepagents_runtime/src/ultra_deepagents/crystal_plasticity_tools.py",
        "backend/deepagents_runtime/src/ultra_deepagents/degradation_characterization_tools.py",
        "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_tools.py",
        "backend/deepagents_runtime/src/ultra_deepagents/nats_worker.py",
        "backend/deepagents_runtime/src/ultra_deepagents/runner.py",
        "backend/deepagents_runtime/tests/test_worker_transport.py",
        "backend/deepagents_runtime/materials_data/calphad/manifest.json",
        "backend/deepagents_runtime/skills/materials-characterization-advanced/SKILL.md",
        "backend/deepagents_runtime/skills/materials-crystal-plasticity/SKILL.md",
        "backend/deepagents_runtime/skills/materials-mechanics-degradation/SKILL.md",
        "backend/deepagents_runtime/skills/materials-processing-kinetics/SKILL.md",
        "backend/deepagents_runtime/skills/materials-sensor-data/SKILL.md",
        "backend/deepagents_runtime/skills/materials-structure-thermo/SKILL.md",
        "backend/deepagents_runtime/tests/kinetics_runtime/test_kawin_runtime.py",
        "backend/deepagents_runtime/tests/test_kinetics_tools.py",
        "backend/deepagents_runtime/tests/test_crystal_plasticity_agent_registration.py",
        "backend/deepagents_runtime/tests/test_crystal_plasticity_tools.py",
        "backend/deepagents_runtime/tests/test_degradation_characterization_agent_registration.py",
        "backend/deepagents_runtime/tests/test_degradation_characterization_tools.py",
        "backend/deepagents_runtime/tests/test_materials_natural_prompt_fixtures.py",
        "backend/deepagents_runtime/tests/test_ngff.py",
        "backend/deepagents_runtime/tests/test_paper_table_evidence.py",
        "backend/deepagents_runtime/tests/test_sensor_tools.py",
        "backend/deepagents_runtime/tests/test_sensor_worker_core_smoke.py",
        "backend/deepagents_runtime/tests/test_vision_subagent.py",
        "backend/deepagents_runtime/tests/test_zarr_tree_identity_contract.py",
        "deploy/docker/mattools-evaluator-linux-arm64-lock.json",
        "backend/deepagents_runtime/Dockerfile.worker",
        "backend/deepagents_runtime/.dockerignore",
        "scripts/build_mattools_evaluator.py",
        "scripts/calphad_ledger_gate.py",
        "scripts/mattools-validator-requirements.lock.txt",
        "scripts/mattools-validator-requirements.txt",
        "tests/test_calphad_ledger_gate.py",
    }
    assert required <= paths


def test_materials_domain_workflow_runs_cp_tools_in_zero_skip_lane() -> None:
    source = (_REPO_ROOT / ".github/workflows/materials-domain-gate.yml").read_text(
        encoding="utf-8"
    )

    for relative in (
        "backend/deepagents_runtime/src/ultra_deepagents/crystal_plasticity_tools.py",
        "backend/deepagents_runtime/tests/test_crystal_plasticity_tools.py",
        "backend/deepagents_runtime/tests/test_crystal_plasticity_agent_registration.py",
    ):
        assert f'- "{relative}"' in source
    assert "Test first-class crystal-plasticity tools and registration" in source
    assert "materials-crystal-plasticity-tools-junit.xml" in source
    assert source.count('"materials-crystal-plasticity-tools-junit.xml"') == 1


def test_materials_domain_workflow_runs_degradation_characterization_zero_skip_lane() -> None:
    source = (_REPO_ROOT / ".github/workflows/materials-domain-gate.yml").read_text(
        encoding="utf-8"
    )

    for relative in (
        "backend/deepagents_runtime/src/ultra_deepagents/degradation_characterization_tools.py",
        "backend/deepagents_runtime/tests/test_degradation_characterization_tools.py",
        "backend/deepagents_runtime/tests/test_degradation_characterization_agent_registration.py",
        "backend/deepagents_runtime/Dockerfile.worker",
        "backend/deepagents_runtime/.dockerignore",
    ):
        assert f'- "{relative}"' in source
    assert "Test bounded degradation and characterization tools and registration" in source
    report = '"materials-degradation-characterization-tools-junit.xml"'
    assert report in source
    assert source.count(report) == 1


def test_worker_core_dependency_and_image_pin_close_materials_import_surface() -> None:
    pyproject = (_REPO_ROOT / "backend/deepagents_runtime/pyproject.toml").read_text(
        encoding="utf-8"
    )
    dockerfile = (_REPO_ROOT / "backend/deepagents_runtime/Dockerfile.worker").read_text(
        encoding="utf-8"
    )
    dockerignore = (_REPO_ROOT / "backend/deepagents_runtime/.dockerignore").read_text(
        encoding="utf-8"
    )

    assert '"numpy>=1.26.0,<3"' in pyproject
    assert '"numcodecs==0.16.5"' in pyproject
    assert '"zarr==3.1.5"' in pyproject
    assert "ARG ULTRA_WORKER_NUMPY_VERSION=1.26.4" in dockerfile
    assert "ARG ULTRA_WORKER_NUMCODECS_VERSION=0.16.5" in dockerfile
    assert "ARG ULTRA_WORKER_ZARR_VERSION=3.1.5" in dockerfile
    assert '"numpy==${ULTRA_WORKER_NUMPY_VERSION}"' in dockerfile
    assert '"numcodecs==${ULTRA_WORKER_NUMCODECS_VERSION}"' in dockerfile
    assert '"zarr==${ULTRA_WORKER_ZARR_VERSION}"' in dockerfile
    assert "import numpy, numcodecs, zarr" in dockerfile
    assert "assert numcodecs.__version__ == '${ULTRA_WORKER_NUMCODECS_VERSION}'" in dockerfile
    assert "assert zarr.__version__ == '${ULTRA_WORKER_ZARR_VERSION}'" in dockerfile
    assert "RUN python /app/deepagents_runtime/tests/test_sensor_worker_core_smoke.py" in dockerfile
    assert "!tests/test_sensor_worker_core_smoke.py" in dockerignore
    for module in (
        "ultra_deepagents.agent",
        "ultra_deepagents.crystal_plasticity_tools",
        "ultra_deepagents.degradation_characterization_tools",
        "ultra_deepagents.sensors.tools",
    ):
        assert f"import {module}" in dockerfile


def test_release_workflow_requalifies_first_class_mechanics_and_characterization() -> None:
    source = (_REPO_ROOT / ".github/workflows/release-artifacts.yml").read_text(encoding="utf-8")

    assert "Test first-class mechanics and characterization tool surfaces" in source
    for relative in (
        "tests/test_crystal_plasticity_tools.py",
        "tests/test_crystal_plasticity_agent_registration.py",
        "tests/test_degradation_characterization_tools.py",
        "tests/test_degradation_characterization_agent_registration.py",
        "tests/test_sensor_worker_core_smoke.py",
    ):
        assert relative in source


def test_release_tree_rehashes_control_binary_and_frontend_dist(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    verified = _VERIFY._verify_source_revision(tmp_path, GIT_SHA)
    assert verified["release_artifacts"]["control_binary"]["size_bytes"] > 0
    assert verified["release_artifacts"]["frontend_dist"]["file_count"] == 2

    _write(tmp_path / "bin/ultra-control", "tampered-control\n")
    with pytest.raises(VerificationError, match="control binary or frontend"):
        _VERIFY._verify_source_revision(tmp_path, GIT_SHA)

    _minimal_release_tree(tmp_path)
    _write(tmp_path / "frontend/dist/assets/app.js", "tampered-frontend\n")
    with pytest.raises(VerificationError, match="control binary or frontend"):
        _VERIFY._verify_source_revision(tmp_path, GIT_SHA)


def test_production_full_retention_rejects_git_checkout_source(tmp_path: Path) -> None:
    with pytest.raises(VerificationError, match="extracted immutable release"):
        _VERIFY._retain_release_bundle(
            tmp_path,
            tmp_path / "evidence",
            {"kind": "clean_git_checkout"},
            scope="production-full",
        )


def test_content_addressed_retention_rejects_symlinked_parent(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "bundle").symlink_to(outside, target_is_directory=True)

    with pytest.raises(VerificationError, match="symlink"):
        _VERIFY._retain_bytes(
            b"evidence\n",
            output,
            directory=Path("bundle/logs"),
            stem="sandbox",
            suffix=".log",
        )
    assert list(outside.iterdir()) == []


def test_git_checkout_rejects_untracked_source(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    _write(tmp_path / "tracked.txt")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert _VERIFY._verify_source_revision(tmp_path, head)["untracked_files_clean"] is True
    _write(tmp_path / "untracked.txt")
    with pytest.raises(VerificationError, match="untracked"):
        _VERIFY._verify_source_revision(tmp_path, head)


def test_production_policy_uses_exported_worker_limits_and_rejects_unbounded_settings() -> None:
    policy = policy_for_scope(_args(), environ=PRODUCTION_ENV)
    assert policy.source == "exported_worker_environment"
    assert policy.cpus == 8
    assert policy.memory == "32g"
    assert policy.max_concurrency == 8

    for name, value in (
        ("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", "host"),
        ("ULTRA_DEEPAGENTS_SANDBOX_CPUS", "0"),
        ("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", ""),
        ("ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT", "0"),
        ("ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS", "0"),
        ("ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES", "0"),
        ("ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY", "0"),
        ("ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "false"),
    ):
        environment = {**PRODUCTION_ENV, name: value}
        with pytest.raises(VerificationError):
            policy_for_scope(_args(), environ=environment)


def test_staged_source_is_bound_after_release_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"
    _minimal_release_tree(repo)
    _set_production_environment(monkeypatch)
    args = _args(repo_root=repo, output_dir=output)

    def inspector(ref: str) -> ImageInspection:
        _write(
            repo / "backend/deepagents_runtime/tests/test_calphad_runtime.py",
            "# changed after release verification\n",
        )
        return _image_inspection(ref, "Ultra Deep Agents scientific sandbox")

    status, report_path = run_verification(
        args,
        inspector=inspector,
        backend_factory=_factory,
        host_suite_runner=_fake_host_suite,
    )
    assert status == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert any("staged source differs" in failure for failure in report["failures"])


@pytest.mark.parametrize("mutation", ["added", "deleted", "symlink"])
def test_post_execution_workspace_requires_exact_non_scratch_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"

    class MutatingBackend(_FakeBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> _Response:
            response = super().execute(command, timeout=timeout)
            target = self.workspace / "backend/deepagents_runtime/tests/test_calphad_runtime.py"
            if mutation == "added":
                (self.workspace / "undeclared-after-execution.txt").write_text(
                    "injected\n", encoding="utf-8"
                )
            elif mutation == "deleted":
                target.unlink()
            else:
                target.unlink()
                target.symlink_to("test_calphad_cli.py")
            return response

    def mutating_factory(
        workspace: Path,
        outputs: Path,
        image_id: str,
        policy: SandboxPolicy,
    ) -> _FakeBackend:
        return MutatingBackend(workspace, outputs, image_id, policy)

    status, report_path = _run_full_fixture(
        repo,
        output,
        monkeypatch,
        backend_factory=mutating_factory,
    )
    assert status == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["full_production_image_parity"] is False


def test_backend_factory_cannot_inject_an_undeclared_pre_execution_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def injecting_factory(
        workspace: Path,
        outputs: Path,
        image_id: str,
        policy: SandboxPolicy,
    ) -> _FakeBackend:
        backend = _FakeBackend(workspace, outputs, image_id, policy)
        (workspace / ".ultra-parity/sitecustomize.py").write_text(
            "raise RuntimeError('injected')\n",
            encoding="utf-8",
        )
        return backend

    status, report_path = _run_full_fixture(
        tmp_path / "release",
        tmp_path / "evidence",
        monkeypatch,
        backend_factory=injecting_factory,
    )
    assert status == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert any("undeclared input path" in failure for failure in report["failures"])


def test_production_release_and_evidence_paths_must_be_disjoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "release"
    _minimal_release_tree(repo)
    _set_production_environment(monkeypatch)
    with pytest.raises(VerificationError, match="must be disjoint"):
        run_verification(
            _args(repo_root=repo, output_dir=repo / "evidence"),
            inspector=lambda ref: _image_inspection(ref, "Ultra Deep Agents scientific sandbox"),
            backend_factory=_factory,
            host_suite_runner=_fake_host_suite,
        )


def test_full_orchestration_uses_mocked_backend_and_writes_content_addressed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"
    status, report_path = _run_full_fixture(repo, output, monkeypatch)

    assert status == 0
    payload = report_path.read_bytes()
    assert report_path.stem.endswith(hashlib.sha256(payload).hexdigest())
    report = json.loads(payload)
    assert report["status"] == "passed"
    assert report["scope"] == "production-full"
    assert report["full_production_image_parity"] is True
    assert report["sandbox"]["backend"] == "DockerSandboxBackend"
    assert report["sandbox"]["policy_source"] == "exported_worker_environment"
    assert report["sandbox"]["pytest_isolation"]["tmpdir"] == _VERIFY.PARITY_TMPDIR
    assert report["domain_gate"]["report"]["junit"]["tests"] == 13
    assert report["calphad_runtime"]["junit"]["tests"] == 39
    assert report["calphad_tool_orchestration"]["junit"]["tests"] == 56
    assert report["calphad_release_contract"]["manifest_sha256"] == (
        _VERIFY.REQUIRED_CALPHAD_MANIFEST_SHA256
    )
    assert report["calphad_tool_orchestration"]["binding"]["runtime_image_id"] == IMAGE_ID
    assert report["verified_release_artifacts"] == report["source"]["release_artifacts"]
    assert report["evidence_bundle"]["promotable"] is True
    assert validate_retained_evidence_bundle(report, output) == []
    assert (output / "domain/materials-domain-gate.json").is_file()
    assert (output / "calphad-embedded-probe.json").is_file()
    assert (output / "calphad-runtime-junit.xml").is_file()
    assert (output / "calphad-tools-junit.xml").is_file()

    # The retained bundle remains independently rehashable after the extracted
    # release used to execute the gate is no longer present.
    shutil.rmtree(repo)
    assert validate_retained_evidence_bundle(report, output) == []
    stale_contract = json.loads(json.dumps(report))
    stale_contract["calphad_release_contract"]["runtime_test_count"] = 29
    assert any(
        "CALPHAD release contract" in failure
        for failure in validate_retained_evidence_bundle(stale_contract, output)
    )


def test_retained_bundle_rejects_file_and_tree_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"
    status, report_path = _run_full_fixture(repo, output, monkeypatch)
    assert status == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = report["evidence_bundle"]

    file_records = [
        bundle["release"]["manifest"],
        bundle["release"]["control_binary"],
        bundle["image_identity"]["executed"]["docker_inspect"],
        bundle["execution_output"]["combined"],
        bundle["host_output"]["stdout"],
        bundle["environment"]["installed_packages_file"],
    ]
    for record in file_records:
        path = output / record["relative_path"]
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        assert validate_retained_evidence_bundle(report, output)
        path.write_bytes(original)
        assert validate_retained_evidence_bundle(report, output) == []

    for record in (
        bundle["release"]["frontend_dist"],
        bundle["staged_source"],
        bundle["domain_tree"],
    ):
        injected = output / record["relative_path"] / "undeclared-evidence.txt"
        injected.write_text("tamper\n", encoding="utf-8")
        failures = validate_retained_evidence_bundle(report, output)
        assert any("closure or aggregate" in failure for failure in failures)
        injected.unlink()
        assert validate_retained_evidence_bundle(report, output) == []

    staged_root = output / bundle["staged_source"]["relative_path"]
    symlink = staged_root / "undeclared-symlink"
    symlink.symlink_to(".ultra-parity/calphad_probe.py")
    failures = validate_retained_evidence_bundle(report, output)
    assert any("non-regular" in failure for failure in failures)
    symlink.unlink()
    assert validate_retained_evidence_bundle(report, output) == []

    bundle_path = output / "bundle"
    real_bundle_path = output / "real-bundle"
    bundle_path.rename(real_bundle_path)
    bundle_path.symlink_to(real_bundle_path, target_is_directory=True)
    failures = validate_retained_evidence_bundle(report, output)
    assert any("symlinked report-relative component" in failure for failure in failures)
    bundle_path.unlink()
    real_bundle_path.rename(bundle_path)
    assert validate_retained_evidence_bundle(report, output) == []


def test_retained_bundle_rejects_coherent_semantic_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"
    status, report_path = _run_full_fixture(repo, output, monkeypatch)
    assert status == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = report["evidence_bundle"]

    executed_identity = bundle["image_identity"]["executed"]
    old_inspect = executed_identity["docker_inspect"]
    old_summary_labels = dict(executed_identity["summary"]["labels"])
    old_report_labels = dict(report["executed_image"]["labels"])
    raw = json.loads((output / old_inspect["relative_path"]).read_text(encoding="utf-8"))
    raw["Config"]["Labels"]["org.opencontainers.image.revision"] = "f" * 40
    raw["Config"]["Labels"]["org.opencontainers.image.title"] = "Wrong raw title"
    forged_labels = dict(raw["Config"]["Labels"])
    forged_inspect = _VERIFY._retain_bytes(
        _VERIFY._canonical_json_bytes(raw, newline=True),
        output,
        directory=Path("bundle/adversarial"),
        stem="forged-docker-inspect",
        suffix=".json",
    )
    executed_identity["docker_inspect"] = forged_inspect
    executed_identity["summary"]["labels"] = forged_labels
    report["executed_image"]["labels"] = forged_labels
    failures = validate_retained_evidence_bundle(report, output)
    assert any("OCI labels differ" in failure for failure in failures)
    executed_identity["docker_inspect"] = old_inspect
    executed_identity["summary"]["labels"] = old_summary_labels
    report["executed_image"]["labels"] = old_report_labels
    assert validate_retained_evidence_bundle(report, output) == []

    environment = bundle["environment"]
    old_freeze = environment["pip_freeze"]
    old_freeze_packages = dict(environment["pip_freeze_packages"])
    forged_freeze = _VERIFY._retain_bytes(
        b"evil==9\n",
        output,
        directory=Path("bundle/adversarial"),
        stem="forged-pip-freeze",
        suffix=".txt",
    )
    environment["pip_freeze"] = forged_freeze
    environment["pip_freeze_packages"] = {"evil": "9"}
    failures = validate_retained_evidence_bundle(report, output)
    assert any("standalone and domain-tree pip freeze" in failure for failure in failures)
    environment["pip_freeze"] = old_freeze
    environment["pip_freeze_packages"] = old_freeze_packages
    assert validate_retained_evidence_bundle(report, output) == []

    old_probe_record = bundle["results"]["calphad_probe"]
    old_calphad_report = json.loads(json.dumps(report["calphad"]))
    forged_calphad = json.loads(json.dumps(report["calphad"]["report"]))
    forged_calphad["databases"][0]["sha256"] = "e" * 64
    forged_probe_payload = (
        json.dumps(forged_calphad, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    forged_probe_record = _VERIFY._retain_bytes(
        forged_probe_payload,
        output,
        directory=Path("bundle/adversarial"),
        stem="forged-calphad-probe",
        suffix=".json",
    )
    bundle["results"]["calphad_probe"] = forged_probe_record
    report["calphad"]["report"] = forged_calphad
    report["calphad"]["sha256"] = forged_probe_record["sha256"]
    failures = validate_retained_evidence_bundle(report, output)
    assert any("staged registry bytes" in failure for failure in failures)
    bundle["results"]["calphad_probe"] = old_probe_record
    report["calphad"] = old_calphad_report
    assert validate_retained_evidence_bundle(report, output) == []

    old_domain_tree = bundle["domain_tree"]
    forged_domain_root = tmp_path / "forged-domain"
    forged_domain_root.mkdir()
    (forged_domain_root / "materials-domain-gate.json").write_text(
        json.dumps(report["domain_gate"]["report"]),
        encoding="utf-8",
    )
    bundle["domain_tree"] = _VERIFY._retain_tree(
        forged_domain_root,
        output,
        directory=Path("bundle/adversarial"),
        stem="forged-domain-tree",
        label="forged domain tree",
    )
    failures = validate_retained_evidence_bundle(report, output)
    assert any("exact artifact closure" in failure for failure in failures)
    bundle["domain_tree"] = old_domain_tree
    assert validate_retained_evidence_bundle(report, output) == []


def test_lean_ci_source_contract_is_never_full_production_parity(tmp_path: Path) -> None:
    repo = tmp_path / "release"
    output = tmp_path / "evidence"
    _minimal_release_tree(repo)
    args = _args(
        repo_root=repo,
        output_dir=output,
        scope="ci-pinned-materials",
        image="materials:test",
    )

    def inspector(ref: str) -> ImageInspection:
        return _image_inspection(ref, "Ultra deterministic materials domain gate")

    status, report_path = run_verification(
        args,
        inspector=inspector,
        backend_factory=_factory,
        host_suite_runner=_fake_host_suite,
    )

    assert status == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"] == "ci-pinned-materials"
    assert report["full_production_image_parity"] is False


@pytest.mark.skipif(
    os.getenv("ULTRA_RUN_PRODUCTION_MATERIALS_SANDBOX_INTEGRATION") != "1",
    reason="set ULTRA_RUN_PRODUCTION_MATERIALS_SANDBOX_INTEGRATION=1 for the local full-image run",
)
def test_optional_local_production_image_integration(tmp_path: Path) -> None:
    repo = Path(os.environ["ULTRA_PRODUCTION_MATERIALS_RELEASE_ROOT"])
    image = os.environ["ULTRA_PRODUCTION_MATERIALS_SANDBOX_IMAGE"]
    expected_sha = (
        os.getenv("ULTRA_PRODUCTION_MATERIALS_EXPECTED_SHA")
        or subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    )
    status = _VERIFY.main(
        [
            "--repo-root",
            str(repo),
            "--image",
            image,
            "--expected-git-sha",
            expected_sha,
            "--scope",
            "production-full",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert status == 0
