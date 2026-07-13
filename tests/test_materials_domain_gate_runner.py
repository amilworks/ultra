from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materials_domain_gate.py"
_SPEC = importlib.util.spec_from_file_location("materials_domain_gate", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GATE
_SPEC.loader.exec_module(_GATE)

GateConfigurationError = _GATE.GateConfigurationError
JunitSummary = _GATE.JunitSummary
build_report = _GATE.build_report
compare_pins = _GATE.compare_pins
parse_exact_pins = _GATE.parse_exact_pins
parse_junit_invariant_evidence = _GATE.parse_junit_invariant_evidence
parse_junit_summary = _GATE.parse_junit_summary
render_markdown = _GATE.render_markdown
validate_calphad_runtime_junit = _GATE.validate_calphad_runtime_junit
validate_calphad_experimental_benchmark_report = (
    _GATE.validate_calphad_experimental_benchmark_report
)

_VALID_GIT_SHA = "a" * 40
_VALID_IMAGE_ID = "sha256:" + "b" * 64
_VALID_IMAGE_DIGEST = "sha256:" + "c" * 64


def _valid_evidence(index: int, *, outcome: str = "pass") -> dict[str, object]:
    return {
        "schema_version": "1",
        "validator_id": f"materials.test.invariant_{index}.v1",
        "test_id": f"test_invariant_{index}",
        "required": True,
        "outcome": outcome,
        "observed": {"value": index},
        "expected": {"value": index},
        "tolerance_rationale": "Exact synthetic runner contract value",
        "units": "unitless",
        "convention": "runner unit-test convention",
        "library_versions": {"test-library": "1.0"},
    }


def _valid_calphad_experimental_report() -> dict[str, object]:
    return {
        "schema_version": "ultra.calphad.experimental_benchmark.v1",
        "benchmark_id": "materials.calphad.al_co_w_experimental_two_lane.v1",
        "status": "passed",
        "required_independent_invariant": True,
        "production_promotion_blocked": False,
        "database_binding": {
            "database_id": "nist-al-co-w-wang-2017",
            "sha256": "d" * 64,
        },
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
                        "observation_id": f"holdout_{index}",
                        "reported_uncertainty_K": None,
                        "uncertainty_status": "not_reported_numerically",
                    }
                    for index in range(4)
                ],
            },
        },
    }


def _write_junit_with_properties(
    path: Path,
    cases: list[tuple[str, str | None, str | None]],
) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="materials",
        tests=str(len(cases)),
        failures="0",
        errors="0",
        skipped="0",
        time="0.1",
    )
    for test_name, property_name, property_value in cases:
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname="tests.domain_correctness.test_materials_invariants",
            name=test_name,
            time="0.01",
        )
        if property_name is not None:
            properties = ET.SubElement(testcase, "properties")
            ET.SubElement(
                properties,
                "property",
                name=property_name,
                value=property_value or "",
            )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_calphad_runtime_junit(
    path: Path,
    *,
    remove_name: str = "",
    skipped: int = 0,
) -> None:
    typed_names = list(_GATE.REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES)
    runtime_names = sorted(_GATE.REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES - set(typed_names))
    padding = [
        f"test_reviewed_runtime_case_{index}"
        for index in range(_GATE.REQUIRED_CALPHAD_CORE_TEST_COUNT - len(runtime_names))
    ]
    cases = [("tests.test_calphad_runtime", name) for name in [*runtime_names, *padding]] + [
        ("tests.test_calphad_cli", name) for name in typed_names
    ]
    if remove_name:
        cases = [case for case in cases if case[1] != remove_name]
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests=str(len(cases)),
        failures="0",
        errors="0",
        skipped=str(skipped),
        time="0.1",
    )
    for index, (classname, name) in enumerate(cases):
        testcase = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if index < skipped:
            ET.SubElement(testcase, "skipped")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _passing_report(
    *,
    require_clean_provenance: bool,
    git: dict[str, object],
    image: dict[str, object],
    invariant_evidence: list[dict[str, object]] | None = None,
    invariant_evidence_errors: list[str] | None = None,
) -> dict[str, object]:
    return build_report(
        generated_at_utc="2026-07-09T12:00:00Z",
        junit=JunitSummary(tests=12, failures=0, errors=0, skipped=0, time_seconds=3.5),
        pytest_exit_code=0,
        pytest_command=["python", "-m", "pytest", "test_materials_invariants.py"],
        version_drift=[],
        configuration_errors=[],
        expected_pins={"orix": "0.14.3"},
        installed_direct={"orix": "0.14.3"},
        installed_packages=[{"name": "orix", "version": "0.14.3"}],
        requirements={"path": "materials-requirements.txt", "sha256": "req-sha"},
        test_source={"path": "test_materials_invariants.py", "sha256": "test-sha"},
        git=git,
        image=image,
        runtime={"python": "3.11.13", "platform": "linux"},
        require_clean_provenance=require_clean_provenance,
        invariant_evidence=(
            [_valid_evidence(index) for index in range(12)]
            if invariant_evidence is None
            else invariant_evidence
        ),
        invariant_evidence_errors=(
            [] if invariant_evidence_errors is None else invariant_evidence_errors
        ),
    )


def test_exact_pin_parser_normalizes_names_and_rejects_unpinned_specs(tmp_path: Path) -> None:
    requirements = tmp_path / "materials-requirements.txt"
    requirements.write_text(
        "# deterministic direct dependencies\npymatgen-analysis-defects==2025.1.18\norix==0.14.3\n",
        encoding="utf-8",
    )

    assert parse_exact_pins(requirements) == {
        "orix": "0.14.3",
        "pymatgen-analysis-defects": "2025.1.18",
    }
    requirements.write_text("orix>=0.14\n", encoding="utf-8")
    with pytest.raises(GateConfigurationError, match="exact package==version pin"):
        parse_exact_pins(requirements)


def test_calphad_runtime_preflight_requires_39_non_skipping_pressure_format_scheil_cases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calphad-runtime.xml"
    _write_calphad_runtime_junit(path)

    evidence, errors = validate_calphad_runtime_junit(path)

    assert errors == ()
    assert evidence["validated"] is True
    assert evidence["junit"]["tests"] == 39
    assert evidence["core_tests"] == 36
    assert evidence["typed_cli_tests"] == 3

    _write_calphad_runtime_junit(
        path,
        remove_name="test_dat_inspection_records_the_actual_parser_format",
    )
    _, errors = validate_calphad_runtime_junit(path)
    assert any("exactly 39" in error for error in errors)
    assert any("adversarial or typed scientific" in error for error in errors)

    _write_calphad_runtime_junit(
        path,
        remove_name="test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va",
    )
    _, errors = validate_calphad_runtime_junit(path)
    assert any("exactly 39" in error for error in errors)
    assert any("adversarial or typed scientific" in error for error in errors)

    _write_calphad_runtime_junit(path, skipped=1)
    _, errors = validate_calphad_runtime_junit(path)
    assert any("skipped is nonzero" in error for error in errors)


def test_calphad_experimental_benchmark_is_required_and_thresholds_fail_closed() -> None:
    report = _valid_calphad_experimental_report()

    assert validate_calphad_experimental_benchmark_report(report) == ()

    failed = json.loads(json.dumps(report))
    failed["status"] = "failed"
    failed["production_promotion_blocked"] = True
    failed["lanes"]["held_out"]["status"] = "failed"
    failed["lanes"]["held_out"]["metrics"]["max_abs_error_K"] = 31.0
    errors = validate_calphad_experimental_benchmark_report(failed)
    assert any("did not pass" in error for error in errors)
    assert any("blocks production promotion" in error for error in errors)
    assert any("residuals exceed" in error for error in errors)

    relaxed = json.loads(json.dumps(report))
    relaxed["lanes"]["held_out"]["metrics"]["mae_K_max"] = 21.0
    errors = validate_calphad_experimental_benchmark_report(relaxed)
    assert any("locked 20/30 K policy" in error for error in errors)


def test_junit_parser_sums_suites_and_preserves_skip_count(tmp_path: Path) -> None:
    junit = tmp_path / "materials-junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="crystallography" tests="3" failures="0" errors="0" skipped="1" time="1.25" />
  <testsuite name="microstructure" tests="2" failures="1" errors="0" skipped="0" time="0.75" />
</testsuites>
""",
        encoding="utf-8",
    )

    assert parse_junit_summary(junit) == JunitSummary(
        tests=5,
        failures=1,
        errors=0,
        skipped=1,
        time_seconds=2.0,
    )


def test_junit_invariant_evidence_parser_returns_complete_record(tmp_path: Path) -> None:
    junit = tmp_path / "materials-junit.xml"
    evidence = _valid_evidence(0)
    _write_junit_with_properties(
        junit,
        [
            (
                str(evidence["test_id"]),
                "materials_invariant_evidence",
                json.dumps(evidence, sort_keys=True),
            )
        ],
    )

    parsed = parse_junit_invariant_evidence(junit)

    assert list(parsed.records) == [evidence]
    assert parsed.errors == ()


def test_junit_invariant_evidence_parser_reports_missing_property(tmp_path: Path) -> None:
    junit = tmp_path / "materials-junit.xml"
    _write_junit_with_properties(junit, [("test_invariant_0", None, None)])

    parsed = parse_junit_invariant_evidence(junit)

    assert parsed.records == ()
    assert any("missing 'materials_invariant_evidence'" in error for error in parsed.errors)


@pytest.mark.parametrize(
    ("property_value", "expected_error"),
    [
        ("{not-json", "malformed invariant evidence JSON"),
        (
            json.dumps(
                {
                    key: value
                    for key, value in _valid_evidence(0).items()
                    if key != "tolerance_rationale"
                }
            ),
            "missing required fields: tolerance_rationale",
        ),
        (
            json.dumps({**_valid_evidence(0), "library_versions": {}}),
            "library_versions must be a non-empty JSON object",
        ),
    ],
)
def test_junit_invariant_evidence_parser_reports_malformed_records(
    tmp_path: Path,
    property_value: str,
    expected_error: str,
) -> None:
    junit = tmp_path / "materials-junit.xml"
    _write_junit_with_properties(
        junit,
        [("test_invariant_0", "materials_invariant_evidence", property_value)],
    )

    parsed = parse_junit_invariant_evidence(junit)

    assert parsed.records == ()
    assert any(expected_error in error for error in parsed.errors)


def test_report_fails_closed_when_required_invariant_evidence_is_incomplete() -> None:
    report = _passing_report(
        require_clean_provenance=False,
        git={"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
        image={"ref": "materials:test", "id": _VALID_IMAGE_ID, "digest": ""},
        invariant_evidence=[_valid_evidence(index) for index in range(11)],
        invariant_evidence_errors=[
            "test_invariant_11: missing 'materials_invariant_evidence' property"
        ],
    )

    assert report["status"] == "failed"
    assert report["invariant_evidence"]["complete"] is False
    assert report["invariant_evidence"]["record_count"] == 11
    assert any("evidence count does not match" in failure for failure in report["failures"])
    assert any(
        "missing 'materials_invariant_evidence'" in failure for failure in report["failures"]
    )


def test_report_fails_closed_when_required_invariant_reports_failure() -> None:
    records = [_valid_evidence(index) for index in range(12)]
    records[7] = _valid_evidence(7, outcome="fail")
    report = _passing_report(
        require_clean_provenance=False,
        git={"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
        image={"ref": "materials:test", "id": _VALID_IMAGE_ID, "digest": ""},
        invariant_evidence=records,
    )

    assert report["status"] == "failed"
    assert report["invariant_evidence"]["failed"] == 1
    assert report["invariant_evidence"]["complete"] is False
    assert any(
        "required scientific invariants reported failure" in failure
        and "materials.test.invariant_7.v1" in failure
        for failure in report["failures"]
    )


def test_report_fails_closed_on_skip_and_pin_drift_and_records_provenance() -> None:
    drift = compare_pins(
        {"orix": "0.14.3", "pymatgen": "2026.5.4"},
        {"orix": "0.14.2", "pymatgen": "2026.5.4"},
    )
    report = build_report(
        generated_at_utc="2026-07-09T12:00:00Z",
        junit=JunitSummary(tests=9, failures=0, errors=0, skipped=1, time_seconds=3.5),
        pytest_exit_code=0,
        pytest_command=["python", "-m", "pytest", "test_materials_invariants.py"],
        version_drift=drift,
        configuration_errors=[],
        expected_pins={"orix": "0.14.3", "pymatgen": "2026.5.4"},
        installed_direct={"orix": "0.14.2", "pymatgen": "2026.5.4"},
        installed_packages=[{"name": "orix", "version": "0.14.2"}],
        requirements={"path": "deploy/docker/materials-requirements.txt", "sha256": "req-sha"},
        test_source={"path": "test_materials_invariants.py", "sha256": "test-sha"},
        git={"sha": "abc123", "ref": "refs/heads/main", "dirty": False},
        image={
            "ref": "materials-domain-gate:test",
            "id": "sha256:image",
            "digest": "sha256:digest",
        },
        runtime={"python": "3.11.13", "platform": "linux"},
        require_clean_provenance=False,
        invariant_evidence=[_valid_evidence(index) for index in range(9)],
        invariant_evidence_errors=[],
    )

    assert report["status"] == "failed"
    assert report["junit"]["skipped"] == 1
    assert report["version_drift"] == [
        {"package": "orix", "expected": "0.14.3", "actual": "0.14.2", "reason": "mismatch"}
    ]
    assert any("skipped" in failure.lower() for failure in report["failures"])
    assert any("version drift" in failure.lower() for failure in report["failures"])
    assert json.dumps(report, allow_nan=False)

    markdown = render_markdown(report)
    assert "abc123" in markdown
    assert "sha256:image" in markdown
    assert "orix" in markdown
    assert "FAILED" in markdown
    assert "Promotion provenance: NOT ENFORCED" in markdown


def test_clean_provenance_policy_accepts_clean_git_and_immutable_image_id() -> None:
    report = _passing_report(
        require_clean_provenance=True,
        git={"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
        image={"ref": "materials:test", "id": _VALID_IMAGE_ID, "digest": ""},
    )

    assert report["status"] == "passed"
    assert report["provenance_policy"] == {
        "required": True,
        "status": "enforced",
        "promotion_provenance_enforced": True,
        "would_pass_if_enforced": True,
        "immutable_image_identifiers": [
            {
                "source": "image.id",
                "value": _VALID_IMAGE_ID,
                "digest": _VALID_IMAGE_ID,
            }
        ],
        "issues": [],
    }
    assert "Promotion provenance: ENFORCED" in render_markdown(report)


def test_clean_provenance_policy_accepts_manifest_digest_without_local_image_id() -> None:
    report = _passing_report(
        require_clean_provenance=True,
        git={"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
        image={"ref": "materials:test", "id": "unknown", "digest": _VALID_IMAGE_DIGEST},
    )

    assert report["status"] == "passed"
    identifiers = report["provenance_policy"]["immutable_image_identifiers"]
    assert identifiers == [
        {
            "source": "image.digest",
            "value": _VALID_IMAGE_DIGEST,
            "digest": _VALID_IMAGE_DIGEST,
        }
    ]


@pytest.mark.parametrize(
    ("git", "image", "expected_issue"),
    [
        (
            {"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": True},
            {"id": _VALID_IMAGE_ID, "digest": ""},
            "git worktree is dirty",
        ),
        (
            {"sha": "unknown", "ref": "unknown", "dirty": False},
            {"id": _VALID_IMAGE_ID, "digest": ""},
            "git commit SHA is missing or invalid",
        ),
        (
            {"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": None},
            {"id": _VALID_IMAGE_ID, "digest": ""},
            "git worktree cleanliness is unknown",
        ),
        (
            {"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
            {"id": "unknown", "digest": ""},
            "immutable image digest or image ID is missing",
        ),
        (
            {"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
            {"id": "sha256:not-a-digest", "digest": ""},
            "image.id is not an immutable sha256 digest",
        ),
        (
            {"sha": _VALID_GIT_SHA, "ref": "refs/heads/main", "dirty": False},
            {"id": _VALID_IMAGE_ID, "digest": "sha256:not-a-digest"},
            "image.digest is not an immutable sha256 digest",
        ),
    ],
)
def test_required_clean_provenance_fails_closed(
    git: dict[str, object],
    image: dict[str, object],
    expected_issue: str,
) -> None:
    report = _passing_report(
        require_clean_provenance=True,
        git=git,
        image=image,
    )

    assert report["status"] == "failed"
    assert report["provenance_policy"]["status"] == "failed"
    assert expected_issue in report["provenance_policy"]["issues"]
    assert any(expected_issue in failure for failure in report["failures"])
    assert "Promotion provenance: FAILED" in render_markdown(report)


def test_local_wip_report_passes_but_cannot_claim_promotion_provenance() -> None:
    report = _passing_report(
        require_clean_provenance=False,
        git={"sha": "unknown", "ref": "unknown", "dirty": True},
        image={"id": "unknown", "digest": ""},
    )

    assert report["status"] == "passed"
    assert report["provenance_policy"]["status"] == "not_enforced"
    assert report["provenance_policy"]["promotion_provenance_enforced"] is False
    markdown = render_markdown(report)
    assert "Promotion provenance: NOT ENFORCED" in markdown
    assert "not eligible as promotion evidence" in markdown
