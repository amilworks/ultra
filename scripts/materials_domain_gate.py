#!/usr/bin/env python3
"""Run the deterministic materials invariants and emit promotion evidence.

This gate is deliberately independent of the MatTools agent benchmark.  It checks
field-standard deterministic invariants in a pinned scientific environment, fails
closed on skips or direct-dependency drift, and records enough provenance to audit
the exact image, source revision, packages, and test result later.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GateConfigurationError(ValueError):
    """The deterministic gate cannot prove its configured dependency/test scope."""


@dataclass(frozen=True)
class JunitSummary:
    tests: int
    failures: int
    errors: int
    skipped: int
    time_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class InvariantEvidenceParseResult:
    """Validated per-test scientific evidence extracted from JUnit properties."""

    records: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]


MATERIALS_INVARIANT_EVIDENCE_PROPERTY = "materials_invariant_evidence"
INVARIANT_EVIDENCE_SCHEMA_VERSION = "1"
_INVARIANT_VALIDATOR_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_INVARIANT_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "validator_id",
        "test_id",
        "required",
        "outcome",
        "observed",
        "expected",
        "tolerance_rationale",
        "units",
        "convention",
        "library_versions",
    }
)


_EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s*]+)$")
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNKNOWN_PROVENANCE_VALUES = {"", "none", "null", "unknown"}
REQUIRED_CALPHAD_RUNTIME_TEST_COUNT = 39
REQUIRED_CALPHAD_CORE_TEST_COUNT = 36
REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT = 3
REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES = (
    "test_typed_cli_real_pycalphad_0_11_2_inspection_and_equilibrium",
    "test_typed_cli_real_dat_resource_format_binding",
    "test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va",
)
REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES = frozenset(
    {
        "test_parser_uses_the_validated_database_format",
        "test_database_input_rejects_unregistered_db_suffix",
        "test_pinned_pycalphad_database_corpus_parses_all_registered_text_formats",
        "test_dat_inspection_records_the_actual_parser_format",
        "test_embedded_nist_manifest_directory_json_and_tdb_are_verified",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits0]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits1]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits2]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits3]",
        "test_composition_closure_domain_subset_grid_and_temperature_bounds",
        "test_typed_cli_real_pycalphad_0_11_2_inspection_and_equilibrium",
        "test_typed_cli_real_dat_resource_format_binding",
        "test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va",
    }
)
CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_SCHEMA = (
    "ultra.calphad.experimental_benchmark.v1"
)
CALPHAD_EXPERIMENTAL_BENCHMARK_ID = (
    "materials.calphad.al_co_w_experimental_two_lane.v1"
)
CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_NAME = (
    "calphad-experimental-benchmark.json"
)
CALPHAD_EXPERIMENTAL_BENCHMARK_SCRIPT_NAME = (
    "calphad_experimental_benchmark.py"
)


def canonical_distribution_name(name: str) -> str:
    """Return the PEP 503 comparison form used for pin/install matching."""

    return re.sub(r"[-_.]+", "-", str(name).strip()).lower()


def parse_exact_pins(path: Path) -> dict[str, str]:
    """Parse a direct requirements file and reject every non-exact specification."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GateConfigurationError(f"cannot read requirements file {path}: {exc}") from exc

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        match = _EXACT_PIN.fullmatch(line)
        if match is None:
            raise GateConfigurationError(
                f"{path}:{line_number} must be an exact package==version pin: {raw_line!r}"
            )
        package = canonical_distribution_name(match.group(1))
        version = match.group(2)
        if package in pins:
            raise GateConfigurationError(
                f"{path}:{line_number} duplicates normalized package pin {package!r}"
            )
        pins[package] = version
    if not pins:
        raise GateConfigurationError(f"requirements file {path} contains no exact pins")
    return dict(sorted(pins.items()))


def installed_package_versions() -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return normalized lookup and stable complete distribution inventory."""

    versions: dict[str, str] = {}
    records: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name") or getattr(distribution, "name", "")
        name = str(raw_name).strip()
        version = str(distribution.version).strip()
        if not name:
            continue
        normalized = canonical_distribution_name(name)
        versions.setdefault(normalized, version)
        records.append({"name": name, "normalized_name": normalized, "version": version})
    records.sort(key=lambda record: (record["normalized_name"], record["name"].lower()))
    return versions, records


def compare_pins(
    expected: Mapping[str, str], installed: Mapping[str, str]
) -> list[dict[str, str | None]]:
    """Return stable missing/mismatched direct pins; an empty list means exact parity."""

    normalized_installed = {
        canonical_distribution_name(name): str(version) for name, version in installed.items()
    }
    drift: list[dict[str, str | None]] = []
    for raw_name, expected_version in sorted(expected.items()):
        package = canonical_distribution_name(raw_name)
        actual = normalized_installed.get(package)
        if actual is None:
            drift.append(
                {
                    "package": package,
                    "expected": str(expected_version),
                    "actual": None,
                    "reason": "missing",
                }
            )
        elif actual != str(expected_version):
            drift.append(
                {
                    "package": package,
                    "expected": str(expected_version),
                    "actual": actual,
                    "reason": "mismatch",
                }
            )
    return drift


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _leaf_test_suites(element: ET.Element) -> list[ET.Element]:
    child_suites = [child for child in element if _xml_local_name(child.tag) == "testsuite"]
    if _xml_local_name(element.tag) == "testsuite" and not child_suites:
        return [element]
    leaves: list[ET.Element] = []
    for child in child_suites:
        leaves.extend(_leaf_test_suites(child))
    return leaves


def _junit_int(suite: ET.Element, attribute: str) -> int:
    raw = suite.attrib.get(attribute, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise GateConfigurationError(f"invalid JUnit {attribute}={raw!r}") from exc
    if value < 0:
        raise GateConfigurationError(f"invalid negative JUnit {attribute}={raw!r}")
    return value


def parse_junit_summary(path: Path) -> JunitSummary:
    """Parse pytest/xUnit XML without double-counting nested aggregate suites."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateConfigurationError(f"cannot parse JUnit report {path}: {exc}") from exc
    suites = _leaf_test_suites(root)
    if not suites:
        raise GateConfigurationError(f"JUnit report {path} contains no testsuite")

    tests = failures = errors = skipped = 0
    elapsed = 0.0
    for suite in suites:
        tests += _junit_int(suite, "tests")
        failures += _junit_int(suite, "failures")
        errors += _junit_int(suite, "errors")
        skipped += _junit_int(suite, "skipped")
        raw_time = suite.attrib.get("time", "0")
        try:
            suite_time = float(raw_time)
        except ValueError as exc:
            raise GateConfigurationError(f"invalid JUnit time={raw_time!r}") from exc
        if not math.isfinite(suite_time) or suite_time < 0:
            raise GateConfigurationError(f"invalid JUnit time={raw_time!r}")
        elapsed += suite_time
    return JunitSummary(
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        time_seconds=elapsed,
    )


def validate_calphad_runtime_junit(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate the exact non-skipping CALPHAD runtime/CLI preflight count."""

    errors: list[str] = []
    try:
        summary = parse_junit_summary(path)
        root = ET.parse(path).getroot()
    except (ET.ParseError, GateConfigurationError, OSError) as exc:
        return (
            {
                "path": str(path),
                "validated": False,
                "junit": JunitSummary(0, 0, 0, 0, 0.0).to_dict(),
                "core_tests": 0,
                "typed_cli_tests": 0,
                "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
            },
            (str(exc),),
        )
    testcases = [element for element in root.iter() if _xml_local_name(element.tag) == "testcase"]
    identities: list[str] = []
    names: list[str] = []
    core_tests = 0
    typed_cli_tests = 0
    for testcase in testcases:
        classname = str(testcase.attrib.get("classname") or "").strip()
        name = str(testcase.attrib.get("name") or "").strip()
        identities.append(f"{classname}::{name}")
        names.append(name)
        if classname.endswith("test_calphad_runtime"):
            core_tests += 1
        elif classname.endswith("test_calphad_cli"):
            typed_cli_tests += 1
        else:
            errors.append("CALPHAD runtime JUnit contains an unrelated testcase")
    if len(testcases) != summary.tests:
        errors.append("CALPHAD runtime JUnit testcase count disagrees with suite counters")
    if summary.tests != REQUIRED_CALPHAD_RUNTIME_TEST_COUNT:
        errors.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_CALPHAD_RUNTIME_TEST_COUNT} tests; found {summary.tests}"
        )
    if core_tests != REQUIRED_CALPHAD_CORE_TEST_COUNT:
        errors.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_CALPHAD_CORE_TEST_COUNT} core tests; found {core_tests}"
        )
    if typed_cli_tests != REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT:
        errors.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT} typed CLI tests; "
            f"found {typed_cli_tests}"
        )
    missing_adversaries = sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES - set(names))
    if missing_adversaries:
        errors.append(
            "CALPHAD runtime JUnit is missing required adversarial or typed scientific tests: "
            + ", ".join(missing_adversaries)
        )
    if len(set(identities)) != len(identities):
        errors.append("CALPHAD runtime JUnit contains duplicate testcase identities")
    for field in ("failures", "errors", "skipped"):
        if getattr(summary, field):
            errors.append(f"CALPHAD runtime JUnit {field} is nonzero")
    return (
        {
            "path": str(path),
            "validated": not errors,
            "junit": summary.to_dict(),
            "core_tests": core_tests,
            "typed_cli_tests": typed_cli_tests,
            "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
        },
        tuple(errors),
    )


def _testcase_label(testcase: ET.Element) -> str:
    classname = str(testcase.attrib.get("classname") or "").strip()
    name = str(testcase.attrib.get("name") or "<unnamed>").strip()
    return f"{classname}::{name}" if classname else name


def _junit_testcase_outcome(testcase: ET.Element) -> str:
    terminal_children = {
        _xml_local_name(child.tag) for child in testcase if _xml_local_name(child.tag)
    }
    return "fail" if terminal_children.intersection({"failure", "error", "skipped"}) else "pass"


def _property_values(testcase: ET.Element, property_name: str) -> list[str]:
    values: list[str] = []
    for child in testcase:
        if _xml_local_name(child.tag) != "properties":
            continue
        for prop in child:
            if _xml_local_name(prop.tag) == "property" and prop.attrib.get("name") == property_name:
                values.append(str(prop.attrib.get("value") or prop.text or ""))
    return values


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_invariant_evidence(
    payload: Any,
    *,
    testcase_name: str,
    junit_outcome: str,
) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["evidence payload must be a JSON object"]
    errors: list[str] = []
    missing = sorted(_INVARIANT_EVIDENCE_FIELDS - set(payload))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if payload.get("schema_version") != INVARIANT_EVIDENCE_SCHEMA_VERSION:
        errors.append("schema_version must be " + repr(INVARIANT_EVIDENCE_SCHEMA_VERSION))
    validator_id = str(payload.get("validator_id") or "").strip()
    if _INVARIANT_VALIDATOR_ID.fullmatch(validator_id) is None:
        errors.append("validator_id is blank or invalid")
    if payload.get("test_id") != testcase_name:
        errors.append(
            f"test_id must match JUnit testcase name {testcase_name!r}; "
            f"got {payload.get('test_id')!r}"
        )
    if payload.get("required") is not True:
        errors.append("required must be true for every promotion invariant")
    outcome = payload.get("outcome")
    if outcome not in {"pass", "fail"}:
        errors.append("outcome must be 'pass' or 'fail'")
    elif outcome != junit_outcome:
        errors.append(
            f"evidence outcome {outcome!r} disagrees with JUnit outcome {junit_outcome!r}"
        )
    for field_name in ("observed", "expected"):
        value = payload.get(field_name)
        if not isinstance(value, Mapping) or not value:
            errors.append(f"{field_name} must be a non-empty JSON object")
    for field_name in ("tolerance_rationale", "units", "convention"):
        if not str(payload.get(field_name) or "").strip():
            errors.append(f"{field_name} must be a non-blank string")
    versions = payload.get("library_versions")
    if not isinstance(versions, Mapping) or not versions:
        errors.append("library_versions must be a non-empty JSON object")
    elif any(not str(name).strip() or not str(value).strip() for name, value in versions.items()):
        errors.append("library_versions cannot contain blank names or versions")
    return errors


def parse_junit_invariant_evidence(path: Path) -> InvariantEvidenceParseResult:
    """Extract and validate exactly one scientific record per JUnit testcase."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateConfigurationError(f"cannot parse JUnit report {path}: {exc}") from exc
    suites = _leaf_test_suites(root)
    if not suites:
        raise GateConfigurationError(f"JUnit report {path} contains no testsuite")

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for suite in suites:
        for testcase in suite.iter():
            if _xml_local_name(testcase.tag) != "testcase":
                continue
            label = _testcase_label(testcase)
            testcase_name = str(testcase.attrib.get("name") or "").strip()
            property_values = _property_values(testcase, MATERIALS_INVARIANT_EVIDENCE_PROPERTY)
            if not property_values:
                errors.append(
                    f"{label}: missing {MATERIALS_INVARIANT_EVIDENCE_PROPERTY!r} property"
                )
                continue
            if len(property_values) != 1:
                errors.append(
                    f"{label}: expected one {MATERIALS_INVARIANT_EVIDENCE_PROPERTY!r} "
                    f"property, found {len(property_values)}"
                )
                continue
            try:
                payload = json.loads(
                    property_values[0],
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label}: malformed invariant evidence JSON: {exc}")
                continue
            payload_errors = _validate_invariant_evidence(
                payload,
                testcase_name=testcase_name,
                junit_outcome=_junit_testcase_outcome(testcase),
            )
            if payload_errors:
                errors.extend(f"{label}: {message}" for message in payload_errors)
                continue
            records.append(dict(payload))

    validator_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    for record in records:
        validator_id = str(record["validator_id"])
        test_id = str(record["test_id"])
        validator_counts[validator_id] = validator_counts.get(validator_id, 0) + 1
        test_counts[test_id] = test_counts.get(test_id, 0) + 1
    for validator_id, count in sorted(validator_counts.items()):
        if count > 1:
            errors.append(f"duplicate validator_id {validator_id!r} appears {count} times")
    for test_id, count in sorted(test_counts.items()):
        if count > 1:
            errors.append(f"duplicate test_id {test_id!r} appears {count} times")

    records.sort(key=lambda record: (str(record["test_id"]), str(record["validator_id"])))
    return InvariantEvidenceParseResult(records=tuple(records), errors=tuple(errors))


def _gate_failures(
    *,
    junit: JunitSummary,
    pytest_exit_code: int,
    version_drift: Sequence[Mapping[str, Any]],
    configuration_errors: Sequence[str],
    invariant_evidence: Sequence[Mapping[str, Any]],
    invariant_evidence_errors: Sequence[str],
) -> list[str]:
    failures = [f"configuration: {message}" for message in configuration_errors]
    if pytest_exit_code != 0:
        failures.append(f"pytest exited with code {pytest_exit_code}")
    if junit.tests <= 0:
        failures.append("JUnit reported zero tests; the gate would be vacuous")
    if junit.failures:
        failures.append(f"JUnit reported {junit.failures} failed test(s)")
    if junit.errors:
        failures.append(f"JUnit reported {junit.errors} error(s)")
    if junit.skipped:
        failures.append(f"JUnit reported {junit.skipped} skipped test(s); skips are forbidden")
    if version_drift:
        failures.append(
            f"direct dependency version drift detected for {len(version_drift)} package(s)"
        )
    if len(invariant_evidence) != junit.tests:
        failures.append(
            "scientific invariant evidence count does not match JUnit: "
            f"records={len(invariant_evidence)}, tests={junit.tests}"
        )
    failures.extend(
        f"scientific invariant evidence: {message}" for message in invariant_evidence_errors
    )
    failed_validator_ids = sorted(
        str(record.get("validator_id") or "<unknown>")
        for record in invariant_evidence
        if record.get("outcome") != "pass"
    )
    if failed_validator_ids:
        failures.append(
            "required scientific invariants reported failure: " + ", ".join(failed_validator_ids)
        )
    return failures


def evaluate_provenance_policy(
    *,
    required: bool,
    git: Mapping[str, Any],
    image: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate whether evidence can be treated as immutable promotion evidence.

    A Docker image ID and an OCI manifest digest are both immutable SHA-256
    identifiers.  At least one must be present and valid.  If either field is
    populated with a malformed value, fail the policy instead of silently
    preferring the other field and recording contradictory provenance.
    """

    issues: list[str] = []
    git_sha = str(git.get("sha") or "").strip().lower()
    git_dirty = git.get("dirty")
    if _GIT_OBJECT_ID.fullmatch(git_sha) is None:
        issues.append("git commit SHA is missing or invalid")
    if git_dirty is True:
        issues.append("git worktree is dirty")
    elif git_dirty is not False:
        issues.append("git worktree cleanliness is unknown")

    immutable_identifiers: list[dict[str, str]] = []
    for source, raw_value in (
        ("image.digest", image.get("digest")),
        ("image.id", image.get("id")),
    ):
        value = str(raw_value or "").strip().lower()
        if value in _UNKNOWN_PROVENANCE_VALUES:
            continue
        digest = value.rsplit("@", 1)[-1]
        if _SHA256_DIGEST.fullmatch(digest) is None:
            issues.append(f"{source} is not an immutable sha256 digest")
            continue
        immutable_identifiers.append({"source": source, "value": value, "digest": digest})
    if not immutable_identifiers:
        issues.append("immutable image digest or image ID is missing")

    would_pass = not issues
    if not required:
        status = "not_enforced"
    elif would_pass:
        status = "enforced"
    else:
        status = "failed"
    return {
        "required": bool(required),
        "status": status,
        "promotion_provenance_enforced": bool(required and would_pass),
        "would_pass_if_enforced": would_pass,
        "immutable_image_identifiers": immutable_identifiers,
        "issues": issues,
    }


def build_report(
    *,
    generated_at_utc: str,
    junit: JunitSummary,
    pytest_exit_code: int,
    pytest_command: Sequence[str],
    version_drift: Sequence[Mapping[str, Any]],
    configuration_errors: Sequence[str],
    expected_pins: Mapping[str, str],
    installed_direct: Mapping[str, str | None],
    installed_packages: Sequence[Mapping[str, str]],
    requirements: Mapping[str, Any],
    test_source: Mapping[str, Any],
    git: Mapping[str, Any],
    image: Mapping[str, Any],
    runtime: Mapping[str, Any],
    require_clean_provenance: bool,
    invariant_evidence: Sequence[Mapping[str, Any]],
    invariant_evidence_errors: Sequence[str],
) -> dict[str, Any]:
    """Assemble the machine-readable, fail-closed deterministic gate report."""

    drift = [dict(item) for item in version_drift]
    provenance_policy = evaluate_provenance_policy(
        required=require_clean_provenance,
        git=git,
        image=image,
    )
    effective_configuration_errors = [str(message) for message in configuration_errors]
    if require_clean_provenance:
        effective_configuration_errors.extend(
            f"clean provenance policy: {issue}" for issue in provenance_policy["issues"]
        )
    failures = _gate_failures(
        junit=junit,
        pytest_exit_code=int(pytest_exit_code),
        version_drift=drift,
        configuration_errors=effective_configuration_errors,
        invariant_evidence=invariant_evidence,
        invariant_evidence_errors=invariant_evidence_errors,
    )
    invariant_records = [dict(record) for record in invariant_evidence]
    invariant_passed = sum(record.get("outcome") == "pass" for record in invariant_records)
    invariant_failed = sum(record.get("outcome") == "fail" for record in invariant_records)
    return {
        "schema_version": 1,
        "gate": "materials-domain-gate",
        "scope": "deterministic-domain-invariants",
        "status": "passed" if not failures else "failed",
        "generated_at_utc": generated_at_utc,
        "failures": failures,
        "junit": junit.to_dict(),
        "invariants": invariant_records,
        "invariant_evidence": {
            "schema_version": INVARIANT_EVIDENCE_SCHEMA_VERSION,
            "junit_property": MATERIALS_INVARIANT_EVIDENCE_PROPERTY,
            "record_count": len(invariant_records),
            "passed": invariant_passed,
            "failed": invariant_failed,
            "errors": [str(message) for message in invariant_evidence_errors],
            "complete": (
                len(invariant_records) == junit.tests
                and not invariant_evidence_errors
                and invariant_failed == 0
            ),
        },
        "pytest": {
            "exit_code": int(pytest_exit_code),
            "command": [str(part) for part in pytest_command],
        },
        "version_drift": drift,
        "expected_pins": dict(
            sorted((str(key), str(value)) for key, value in expected_pins.items())
        ),
        "installed_direct": dict(
            sorted(
                (str(key), None if value is None else str(value))
                for key, value in installed_direct.items()
            )
        ),
        "installed_packages": [dict(record) for record in installed_packages],
        "requirements": dict(requirements),
        "test_source": dict(test_source),
        "git": dict(git),
        "image": dict(image),
        "provenance_policy": provenance_policy,
        "runtime": dict(runtime),
    }


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ") or "—"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human audit alongside the canonical JSON report."""

    status = str(report.get("status", "failed")).upper()
    junit = dict(report.get("junit", {}))
    git = dict(report.get("git", {}))
    image = dict(report.get("image", {}))
    provenance_policy = dict(report.get("provenance_policy", {}))
    requirements = dict(report.get("requirements", {}))
    test_source = dict(report.get("test_source", {}))
    invariant_summary = dict(report.get("invariant_evidence", {}))
    invariant_records = [
        dict(record) for record in report.get("invariants", []) if isinstance(record, Mapping)
    ]
    benchmark_wrapper = (
        dict(report.get("calphad_experimental_benchmark", {}))
        if isinstance(report.get("calphad_experimental_benchmark"), Mapping)
        else {}
    )
    benchmark_report = (
        dict(benchmark_wrapper.get("report", {}))
        if isinstance(benchmark_wrapper.get("report"), Mapping)
        else {}
    )
    benchmark_lanes = (
        dict(benchmark_report.get("lanes", {}))
        if isinstance(benchmark_report.get("lanes"), Mapping)
        else {}
    )
    expected = dict(report.get("expected_pins", {}))
    installed = dict(report.get("installed_direct", {}))
    drift_by_package = {str(item.get("package")): item for item in report.get("version_drift", [])}
    provenance_status = str(provenance_policy.get("status", "not_enforced"))
    provenance_banner = {
        "enforced": "**Promotion provenance: ENFORCED**",
        "failed": "**Promotion provenance: FAILED**",
    }.get(
        provenance_status,
        (
            "**Promotion provenance: NOT ENFORCED — local/WIP evidence only; "
            "not eligible for promotion.**"
        ),
    )

    lines = [
        "# Materials Domain Gate",
        "",
        f"**Status: {status}**",
        "",
        provenance_banner,
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Git SHA | `{_md(git.get('sha'))}` |",
        f"| Git ref | `{_md(git.get('ref'))}` |",
        f"| Dirty worktree | `{_md(git.get('dirty'))}` |",
        f"| Image ref | `{_md(image.get('ref'))}` |",
        f"| Image ID | `{_md(image.get('id'))}` |",
        f"| Image digest | `{_md(image.get('digest'))}` |",
        f"| Promotion provenance policy | `{_md(provenance_status)}` |",
        (
            "| Promotion provenance enforced | "
            f"`{_md(provenance_policy.get('promotion_provenance_enforced'))}` |"
        ),
        (
            "| Would pass clean-provenance checks | "
            f"`{_md(provenance_policy.get('would_pass_if_enforced'))}` |"
        ),
        f"| Requirements SHA-256 | `{_md(requirements.get('sha256'))}` |",
        f"| Checked-out requirements SHA-256 | `{_md(requirements.get('source_sha256'))}` |",
        f"| Test source SHA-256 | `{_md(test_source.get('sha256'))}` |",
        f"| Generated | `{_md(report.get('generated_at_utc'))}` |",
        "",
        "## Test result",
        "",
        "| Tests | Failures | Errors | Skipped | Seconds |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {_md(junit.get('tests'))} | {_md(junit.get('failures'))} | "
            f"{_md(junit.get('errors'))} | {_md(junit.get('skipped'))} | "
            f"{_md(junit.get('time_seconds'))} |"
        ),
        "",
        "## Scientific invariant evidence",
        "",
        (
            f"Records: **{_md(invariant_summary.get('record_count'))}**; "
            f"passed: **{_md(invariant_summary.get('passed'))}**; "
            f"failed: **{_md(invariant_summary.get('failed'))}**; "
            f"complete: **{_md(invariant_summary.get('complete'))}**."
        ),
        "",
        "| Test ID | Validator ID | Outcome | Units / convention |",
        "|---|---|---|---|",
    ]
    for record in invariant_records:
        units_and_convention = f"{_md(record.get('units'))}; {_md(record.get('convention'))}"
        lines.append(
            f"| `{_md(record.get('test_id'))}` | "
            f"`{_md(record.get('validator_id'))}` | "
            f"{_md(record.get('outcome'))} | {units_and_convention} |"
        )

    calibration_lane = (
        dict(benchmark_lanes.get("calibration", {}))
        if isinstance(benchmark_lanes.get("calibration"), Mapping)
        else {}
    )
    held_out_lane = (
        dict(benchmark_lanes.get("held_out", {}))
        if isinstance(benchmark_lanes.get("held_out"), Mapping)
        else {}
    )
    calibration_metrics = (
        dict(calibration_lane.get("metrics", {}))
        if isinstance(calibration_lane.get("metrics"), Mapping)
        else {}
    )
    held_out_metrics = (
        dict(held_out_lane.get("metrics", {}))
        if isinstance(held_out_lane.get("metrics"), Mapping)
        else {}
    )
    lines.extend(
        [
            "",
            "## CALPHAD experimental benchmark",
            "",
            (
                "The NIST phase-composition lane is assessment-basis calibration, not "
                "independent validation. The post-assessment DTA lane is the required "
                "independent engineering holdout; its sources report no numerical "
                "measurement uncertainty."
            ),
            "",
            "| Lane | Classification | Result | Metric | Limit |",
            "|---|---|---|---:|---:|",
            (
                "| NIST 2014 phase vertices | calibration | "
                f"{_md(calibration_lane.get('status'))} | RMS z "
                f"{_md(calibration_metrics.get('weighted_rms_z'))}; max z "
                f"{_md(calibration_metrics.get('max_abs_z'))} | 1.0 / 2.0 |"
            ),
            (
                "| 2018 + 2020 DTA transitions | held-out independent | "
                f"{_md(held_out_lane.get('status'))} | MAE "
                f"{_md(held_out_metrics.get('mae_K'))} K; max "
                f"{_md(held_out_metrics.get('max_abs_error_K'))} K | 20 / 30 K |"
            ),
            (
                "| Retained evidence | provenance | "
                f"{_md(benchmark_report.get('status'))} | SHA-256 "
                f"`{_md(benchmark_wrapper.get('sha256'))}` | required |"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Direct dependency pins",
            "",
            "| Package | Expected | Installed | Result |",
            "|---|---|---|---|",
        ]
    )
    for package, expected_version in sorted(expected.items()):
        actual = installed.get(package)
        outcome = "DRIFT" if package in drift_by_package else "ok"
        lines.append(
            f"| `{_md(package)}` | `{_md(expected_version)}` | `{_md(actual)}` | {outcome} |"
        )

    failures = [str(item) for item in report.get("failures", [])]
    lines.extend(["", "## Gate findings", ""])
    if provenance_status == "not_enforced":
        lines.append(
            "- Promotion provenance was not enforced; this local/WIP report is not "
            "eligible as promotion evidence."
        )
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        finding = "- All deterministic invariants ran without skips and every direct pin matched."
        if provenance_status == "enforced":
            finding = finding[:-1] + ", with clean immutable provenance enforced."
        lines.append(finding)
    lines.append("")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_calphad_experimental_benchmark_module() -> Any:
    script_path = Path(__file__).resolve().with_name(
        CALPHAD_EXPERIMENTAL_BENCHMARK_SCRIPT_NAME
    )
    if not script_path.is_file() or script_path.is_symlink():
        raise GateConfigurationError(
            f"CALPHAD experimental benchmark script is missing or nonregular: {script_path}"
        )
    module_name = "_ultra_calphad_experimental_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise GateConfigurationError(
            f"cannot load CALPHAD experimental benchmark script: {script_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_benchmark", None)):
        raise GateConfigurationError(
            "CALPHAD experimental benchmark script has no callable run_benchmark"
        )
    return module


def validate_calphad_experimental_benchmark_report(
    report: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recheck the required benchmark result before accepting it into the gate."""

    errors: list[str] = []
    if report.get("schema_version") != CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_SCHEMA:
        errors.append("report schema is missing or stale")
    if report.get("benchmark_id") != CALPHAD_EXPERIMENTAL_BENCHMARK_ID:
        errors.append("benchmark identity is missing or stale")
    if report.get("required_independent_invariant") is not True:
        errors.append("independent thermometric holdout is not required")
    if report.get("status") != "passed":
        errors.append("two-lane benchmark did not pass")
    if report.get("production_promotion_blocked") is not False:
        errors.append("benchmark explicitly blocks production promotion")

    binding = report.get("database_binding")
    if not isinstance(binding, Mapping):
        errors.append("database binding is missing")
    else:
        if binding.get("database_id") != "nist-al-co-w-wang-2017":
            errors.append("database identity differs from the bundled assessment")
        database_sha = str(binding.get("sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", database_sha) is None:
            errors.append("database SHA-256 is missing or malformed")

    lanes = report.get("lanes")
    if not isinstance(lanes, Mapping):
        errors.append("calibration/held-out lane evidence is missing")
        return tuple(errors)
    calibration = lanes.get("calibration")
    held_out = lanes.get("held_out")
    if not isinstance(calibration, Mapping):
        errors.append("calibration lane is missing")
    else:
        metrics = calibration.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append("calibration metrics are missing")
        else:
            try:
                rms = float(metrics.get("weighted_rms_z"))
                rms_limit = float(metrics.get("weighted_rms_z_max"))
                maximum = float(metrics.get("max_abs_z"))
                maximum_limit = float(metrics.get("max_abs_z_max"))
            except (TypeError, ValueError):
                errors.append("calibration metrics are nonnumeric")
            else:
                if not all(math.isfinite(value) for value in (rms, maximum)):
                    errors.append("calibration metrics are nonfinite")
                if rms_limit != 1.0 or maximum_limit != 2.0:
                    errors.append("calibration thresholds differ from the locked 1/2 z policy")
                if rms > rms_limit or maximum > maximum_limit:
                    errors.append("calibration metrics exceed the locked limits")
        if (
            calibration.get("classification") != "calibration"
            or calibration.get("independent_validation") is not False
            or calibration.get("required") is not True
            or calibration.get("status") != "passed"
            or calibration.get("observation_count") != 6
        ):
            errors.append("calibration lane classification/count/status is invalid")

    if not isinstance(held_out, Mapping):
        errors.append("independent held-out lane is missing")
    else:
        metrics = held_out.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append("held-out metrics are missing")
        else:
            try:
                mae = float(metrics.get("mae_K"))
                mae_limit = float(metrics.get("mae_K_max"))
                maximum = float(metrics.get("max_abs_error_K"))
                maximum_limit = float(metrics.get("max_abs_error_K_max"))
            except (TypeError, ValueError):
                errors.append("held-out metrics are nonnumeric")
            else:
                if not all(math.isfinite(value) for value in (mae, maximum)):
                    errors.append("held-out metrics are nonfinite")
                if mae_limit != 20.0 or maximum_limit != 30.0:
                    errors.append("held-out thresholds differ from the locked 20/30 K policy")
                if mae > mae_limit or maximum > maximum_limit:
                    errors.append("independent held-out residuals exceed the locked limits")
        observations = held_out.get("observations")
        if not isinstance(observations, list) or len(observations) != 4:
            errors.append("held-out lane does not retain exactly four observations")
        else:
            for observation in observations:
                if not isinstance(observation, Mapping):
                    errors.append("held-out observation is malformed")
                    continue
                if (
                    observation.get("reported_uncertainty_K") is not None
                    or observation.get("uncertainty_status")
                    != "not_reported_numerically"
                ):
                    errors.append(
                        "held-out observation does not explicitly preserve absent uncertainty"
                    )
        if (
            held_out.get("classification") != "held_out"
            or held_out.get("independent_validation") is not True
            or held_out.get("required") is not True
            or held_out.get("status") != "passed"
            or held_out.get("observation_count") != 4
        ):
            errors.append("held-out lane independence/count/status is invalid")
    return tuple(errors)


def _boolean_environment(name: str, *, default: bool | None) -> bool | None:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise GateConfigurationError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {raw_value!r}"
    )


def _run_pip_freeze(output_path: Path) -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        return f"pip freeze exited with code {result.returncode}: {result.stderr.strip()}"
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _probe_materials_validation() -> dict[str, str]:
    from ultra_deepagents.materials import validation
    from ultra_deepagents.materials.validation import assess_scientific_status

    if not callable(assess_scientific_status):
        raise GateConfigurationError(
            "ultra_deepagents.materials.validation.assess_scientific_status is not callable"
        )
    return {
        "module": "ultra_deepagents.materials.validation",
        "path": str(Path(validation.__file__).resolve()),
        "sha256": sha256_file(Path(validation.__file__).resolve()),
    }


def run_gate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    requirements_path = Path(args.requirements).resolve()
    test_path = Path(args.test_path)
    if not test_path.is_absolute():
        test_path = repo_root / test_path
    test_path = test_path.resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration_errors: list[str] = []
    try:
        require_clean_provenance = bool(
            _boolean_environment(
                "MATERIALS_DOMAIN_GATE_REQUIRE_CLEAN_PROVENANCE",
                default=False,
            )
        )
    except GateConfigurationError as exc:
        configuration_errors.append(str(exc))
        require_clean_provenance = False
    try:
        require_calphad_runtime_junit = bool(
            _boolean_environment(
                "ULTRA_MATERIALS_GATE_REQUIRE_CALPHAD_RUNTIME_JUNIT",
                default=False,
            )
        )
    except GateConfigurationError as exc:
        configuration_errors.append(str(exc))
        require_calphad_runtime_junit = False
    try:
        git_dirty = _boolean_environment(
            "ULTRA_MATERIALS_GATE_GIT_DIRTY",
            default=None,
        )
    except GateConfigurationError as exc:
        configuration_errors.append(str(exc))
        git_dirty = None
    validation_contract: dict[str, str] = {}
    calphad_runtime_contract: dict[str, Any] = {
        "path": None,
        "required": require_calphad_runtime_junit,
        "validated": False,
        "junit": JunitSummary(0, 0, 0, 0, 0.0).to_dict(),
        "core_tests": 0,
        "typed_cli_tests": 0,
        "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
    }
    expected_pins: dict[str, str] = {}
    requirements_sha256: str | None = None
    test_sha256: str | None = None
    try:
        expected_pins = parse_exact_pins(requirements_path)
        requirements_sha256 = sha256_file(requirements_path)
    except (GateConfigurationError, OSError) as exc:
        configuration_errors.append(str(exc))
    try:
        test_sha256 = sha256_file(test_path)
    except OSError as exc:
        configuration_errors.append(f"cannot hash test source {test_path}: {exc}")
    try:
        validation_contract = _probe_materials_validation()
    except (GateConfigurationError, ImportError, OSError) as exc:
        configuration_errors.append(f"materials validation contract import failed: {exc}")
    raw_calphad_junit = str(getattr(args, "calphad_runtime_junit", "") or "").strip()
    if raw_calphad_junit:
        calphad_junit_path = Path(raw_calphad_junit)
        if not calphad_junit_path.is_absolute():
            calphad_junit_path = repo_root / calphad_junit_path
        calphad_runtime_contract, calphad_contract_errors = validate_calphad_runtime_junit(
            calphad_junit_path.resolve()
        )
        calphad_runtime_contract["required"] = require_calphad_runtime_junit
        configuration_errors.extend(
            f"CALPHAD runtime preflight: {message}" for message in calphad_contract_errors
        )
    elif require_calphad_runtime_junit:
        configuration_errors.append(
            "CALPHAD runtime preflight JUnit is required but --calphad-runtime-junit is missing"
        )

    calphad_experimental_path = output_dir / CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_NAME
    calphad_experimental_report: dict[str, Any]
    calphad_experimental_validator_path = Path(__file__).resolve().with_name(
        CALPHAD_EXPERIMENTAL_BENCHMARK_SCRIPT_NAME
    )
    try:
        benchmark_module = _load_calphad_experimental_benchmark_module()
        raw_benchmark_report = benchmark_module.run_benchmark(repository_root=repo_root)
        if not isinstance(raw_benchmark_report, Mapping):
            raise GateConfigurationError("CALPHAD experimental benchmark returned a non-object")
        calphad_experimental_report = dict(raw_benchmark_report)
    except Exception as exc:  # fail closed while retaining a bounded diagnostic report
        calphad_experimental_report = {
            "schema_version": CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_SCHEMA,
            "benchmark_id": CALPHAD_EXPERIMENTAL_BENCHMARK_ID,
            "status": "failed",
            "required_independent_invariant": True,
            "production_promotion_blocked": True,
            "blocking_reasons": [f"benchmark execution failed: {str(exc)[:1000]}"],
        }
    benchmark_errors = validate_calphad_experimental_benchmark_report(
        calphad_experimental_report
    )
    configuration_errors.extend(
        f"CALPHAD experimental benchmark: {message}" for message in benchmark_errors
    )
    try:
        calphad_experimental_path.write_text(
            json.dumps(
                calphad_experimental_report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        calphad_experimental_evidence = {
            "relative_path": CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_NAME,
            "sha256": sha256_file(calphad_experimental_path),
            "size_bytes": calphad_experimental_path.stat().st_size,
            "validator": {
                "path": str(calphad_experimental_validator_path),
                "sha256": sha256_file(calphad_experimental_validator_path),
            },
            "report": calphad_experimental_report,
        }
    except (OSError, TypeError, ValueError) as exc:
        configuration_errors.append(
            f"CALPHAD experimental benchmark evidence could not be retained: {exc}"
        )
        calphad_experimental_evidence = {
            "relative_path": CALPHAD_EXPERIMENTAL_BENCHMARK_REPORT_NAME,
            "sha256": None,
            "size_bytes": None,
            "validator": {
                "path": str(calphad_experimental_validator_path),
                "sha256": None,
            },
            "report": calphad_experimental_report,
        }

    installed_versions, installed_packages = installed_package_versions()
    installed_direct = {package: installed_versions.get(package) for package in expected_pins}
    version_drift = compare_pins(expected_pins, installed_versions)

    source_requirements_sha256 = os.getenv("ULTRA_MATERIALS_GATE_REQUIREMENTS_SHA256", "").strip()
    if (
        source_requirements_sha256
        and requirements_sha256
        and source_requirements_sha256 != requirements_sha256
    ):
        configuration_errors.append(
            "requirements hash drift: the image copy does not match the checked-out pin file"
        )

    freeze_error = _run_pip_freeze(output_dir / "materials-pip-freeze.txt")
    if freeze_error:
        configuration_errors.append(freeze_error)

    junit_path = output_dir / "materials-junit.xml"
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        "-q",
        "-ra",
        "--color=no",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "-o",
        "junit_family=legacy",
        f"--junitxml={junit_path}",
    ]
    pytest_environment = os.environ.copy()
    checked_out_runtime = repo_root / "backend" / "deepagents_runtime" / "src"
    existing_pythonpath = pytest_environment.get("PYTHONPATH", "")
    pytest_environment.update(
        {
            "HOME": "/tmp",
            "LC_ALL": "C.UTF-8",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "NUMBA_CACHE_DIR": "/tmp/numba",
            "NUMBA_CPU_FEATURES": "",
            "NUMBA_CPU_NAME": "generic",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": ":".join(
                part for part in (str(checked_out_runtime), existing_pythonpath) if part
            ),
            "TZ": "UTC",
            "ULTRA_FAIL_ON_DOMAIN_SKIP": "1",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    if test_path.is_file():
        completed = subprocess.run(
            pytest_command,
            cwd=repo_root,
            env=pytest_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        pytest_exit_code = int(completed.returncode)
        (output_dir / "materials-pytest.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "materials-pytest.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
    else:
        pytest_exit_code = 4
        configuration_errors.append(f"materials invariant file is missing: {test_path}")

    try:
        junit = parse_junit_summary(junit_path)
    except GateConfigurationError as exc:
        configuration_errors.append(str(exc))
        junit = JunitSummary(tests=0, failures=0, errors=0, skipped=0, time_seconds=0.0)
    try:
        invariant_evidence_result = parse_junit_invariant_evidence(junit_path)
    except GateConfigurationError as exc:
        invariant_evidence_result = InvariantEvidenceParseResult(
            records=(),
            errors=(str(exc),),
        )

    report = build_report(
        generated_at_utc=_utc_now(),
        junit=junit,
        pytest_exit_code=pytest_exit_code,
        pytest_command=pytest_command,
        version_drift=version_drift,
        configuration_errors=configuration_errors,
        expected_pins=expected_pins,
        installed_direct=installed_direct,
        installed_packages=installed_packages,
        requirements={
            "path": str(requirements_path),
            "sha256": requirements_sha256,
            "source_sha256": source_requirements_sha256,
        },
        test_source={
            "path": str(test_path),
            "sha256": test_sha256,
        },
        git={
            "sha": os.getenv("ULTRA_MATERIALS_GATE_GIT_SHA", "unknown"),
            "ref": os.getenv("ULTRA_MATERIALS_GATE_GIT_REF", "unknown"),
            "dirty": git_dirty,
        },
        image={
            "ref": os.getenv("ULTRA_MATERIALS_GATE_IMAGE_REF", "unknown"),
            "id": os.getenv("ULTRA_MATERIALS_GATE_IMAGE_ID", "unknown"),
            "digest": os.getenv("ULTRA_MATERIALS_GATE_IMAGE_DIGEST", ""),
            "dockerfile_sha256": os.getenv("ULTRA_MATERIALS_GATE_DOCKERFILE_SHA256", ""),
        },
        runtime={
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "materials_validation": validation_contract,
            "calphad_runtime_preflight": calphad_runtime_contract,
            "determinism_environment": {
                key: pytest_environment[key]
                for key in (
                    "LC_ALL",
                    "NUMBA_CPU_FEATURES",
                    "NUMBA_CPU_NAME",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "PYTHONHASHSEED",
                    "TZ",
                )
            },
        },
        require_clean_provenance=require_clean_provenance,
        invariant_evidence=invariant_evidence_result.records,
        invariant_evidence_errors=invariant_evidence_result.errors,
    )
    report["calphad_experimental_benchmark"] = calphad_experimental_evidence
    report_path = output_dir / "materials-domain-gate.json"
    markdown_path = output_dir / "materials-domain-gate.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/workspace")
    parser.add_argument("--requirements", default="/opt/ultra/materials-requirements.txt")
    parser.add_argument(
        "--test-path",
        default="backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py",
    )
    parser.add_argument("--calphad-runtime-junit", default="")
    parser.add_argument("--output-dir", default="/reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_gate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
