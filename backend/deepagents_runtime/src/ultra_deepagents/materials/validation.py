"""Fail-closed scientific-status records for materials analyses.

An orchestration run can complete while its scientific result is wrong. This
module makes that distinction explicit and machine-readable. It deliberately
contains no domain algorithms; field-standard libraries and independent
benchmark verifiers produce :class:`ValidationCheck` records, while this module
decides whether the available evidence is sufficient to call a result verified.

The record is designed to be written as ``/outputs/materials_validation.json``
by sandbox jobs and linked to the corresponding Ultra run and artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ValidationOutcome(str, Enum):
    """Outcome emitted by one deterministic validator."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class ScientificStatus(str, Enum):
    """Scientific status, intentionally independent of the run status."""

    VERIFIED = "verified"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


def _clean_identifier(value: str, *, field_name: str) -> str:
    cleaned = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(cleaned):
        raise ValueError(f"{field_name} must match {_IDENTIFIER.pattern!r}; got {value!r}")
    return cleaned


def _clean_versions(values: Mapping[str, str] | None) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_name, raw_version in (values or {}).items():
        name = str(raw_name or "").strip()
        version = str(raw_version or "").strip()
        if not name or not version:
            raise ValueError("library_versions cannot contain blank names or versions")
        versions[name] = version
    return dict(sorted(versions.items()))


def _assert_finite_json_value(value: Any, *, field_name: str) -> None:
    """Reject NaN/Infinity and non-JSON values before verdict computation."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} contains a non-string object key")
            _assert_finite_json_value(item, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_json_value(item, field_name=f"{field_name}[{index}]")
        return
    raise ValueError(f"{field_name} contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True)
class EvidenceArtifact:
    """Content-addressed evidence used by a validation check."""

    name: str
    sha256: str
    artifact_id: str | None = None
    path: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        digest = str(self.sha256 or "").strip().lower()
        artifact_id = str(self.artifact_id or "").strip() or None
        path = str(self.path or "").strip() or None
        if not name:
            raise ValueError("evidence artifact name is required")
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid SHA-256 for evidence artifact {name!r}")
        if artifact_id is None and path is None:
            raise ValueError("evidence artifact requires artifact_id or path")
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("evidence artifact size_bytes cannot be negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "path", path)
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", int(self.size_bytes))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvidenceArtifact:
        return cls(
            name=str(value.get("name") or ""),
            sha256=str(value.get("sha256") or ""),
            artifact_id=value.get("artifact_id"),
            path=value.get("path"),
            size_bytes=value.get("size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "sha256": self.sha256}
        if self.artifact_id is not None:
            payload["artifact_id"] = self.artifact_id
        if self.path is not None:
            payload["path"] = self.path
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload


@dataclass(frozen=True)
class ValidationCheck:
    """One scientific invariant with its provenance and observed evidence."""

    validator_id: str
    outcome: ValidationOutcome
    observed: Any = None
    expected: Any = None
    units: str | None = None
    tolerance_rationale: str = ""
    required: bool = True
    critical: bool = False
    library_versions: Mapping[str, str] = field(default_factory=dict)
    evidence: tuple[EvidenceArtifact, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        validator_id = _clean_identifier(self.validator_id, field_name="validator_id")
        outcome = (
            self.outcome
            if isinstance(self.outcome, ValidationOutcome)
            else ValidationOutcome(str(self.outcome))
        )
        units = str(self.units or "").strip() or None
        tolerance_rationale = str(self.tolerance_rationale or "").strip()
        message = str(self.message or "").strip()
        evidence = tuple(
            item if isinstance(item, EvidenceArtifact) else EvidenceArtifact.from_mapping(item)
            for item in self.evidence
        )
        _assert_finite_json_value(self.observed, field_name="observed")
        _assert_finite_json_value(self.expected, field_name="expected")
        object.__setattr__(self, "validator_id", validator_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "tolerance_rationale", tolerance_rationale)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "library_versions", _clean_versions(self.library_versions))
        object.__setattr__(self, "evidence", evidence)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ValidationCheck:
        raw_evidence = value.get("evidence") or []
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            raise ValueError("validation evidence must be a list")
        versions = value.get("library_versions") or {}
        if not isinstance(versions, Mapping):
            raise ValueError("library_versions must be an object")
        return cls(
            validator_id=str(value.get("validator_id") or ""),
            outcome=ValidationOutcome(str(value.get("outcome") or "")),
            observed=value.get("observed"),
            expected=value.get("expected"),
            units=value.get("units"),
            tolerance_rationale=str(value.get("tolerance_rationale") or ""),
            required=bool(value.get("required", True)),
            critical=bool(value.get("critical", False)),
            library_versions={str(k): str(v) for k, v in versions.items()},
            evidence=tuple(EvidenceArtifact.from_mapping(item) for item in raw_evidence),
            message=str(value.get("message") or ""),
        )

    def evidence_complete(self) -> bool:
        """Return whether a passing required check has promotion-grade evidence."""

        if self.outcome is not ValidationOutcome.PASS:
            return True
        return (
            self.observed is not None
            and self.expected is not None
            and bool(self.units)
            and bool(self.tolerance_rationale)
            and bool(self.library_versions)
            and bool(self.evidence)
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "validator_id": self.validator_id,
            "outcome": self.outcome.value,
            "observed": self.observed,
            "expected": self.expected,
            "required": self.required,
            "critical": self.critical,
            "tolerance_rationale": self.tolerance_rationale,
            "library_versions": dict(self.library_versions),
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.units is not None:
            payload["units"] = self.units
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class ScientificAssessment:
    """Fail-closed scientific verdict for one Ultra materials result."""

    run_status: str
    scientific_status: ScientificStatus
    checks: tuple[ValidationCheck, ...]
    required_validator_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    missing_validator_ids: tuple[str, ...]
    critical_failures: tuple[str, ...]
    contradiction_failures: tuple[str, ...]
    capability_supported: bool
    silent_success: bool

    @property
    def verified(self) -> bool:
        return self.scientific_status is ScientificStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_status": self.run_status,
            "scientific_status": self.scientific_status.value,
            "verified": self.verified,
            "silent_success": self.silent_success,
            "capability_supported": self.capability_supported,
            "required_validator_ids": list(self.required_validator_ids),
            "missing_validator_ids": list(self.missing_validator_ids),
            "critical_failures": list(self.critical_failures),
            "contradiction_failures": list(self.contradiction_failures),
            "reasons": list(self.reasons),
            "checks": [check.to_dict() for check in self.checks],
        }


def assess_scientific_status(
    *,
    run_status: str,
    checks: Iterable[ValidationCheck | Mapping[str, Any]],
    required_validator_ids: Iterable[str] = (),
    capability_supported: bool = True,
    contradiction_failures: Iterable[str] = (),
) -> ScientificAssessment:
    """Assess scientific status without conflating it with orchestration success.

    ``required_validator_ids`` is supplied by the task contract, not by the
    generated analysis. Missing or skipped required validators fail closed to
    ``unverified``. A deterministic failure or prose/artifact contradiction is
    ``failed``. Unsupported capabilities are reported explicitly.
    """

    normalized_run_status = str(run_status or "").strip().lower() or "unknown"
    parsed_checks = tuple(
        item if isinstance(item, ValidationCheck) else ValidationCheck.from_mapping(item)
        for item in checks
    )
    ids = [check.validator_id for check in parsed_checks]
    duplicates = sorted({validator_id for validator_id in ids if ids.count(validator_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate validator_id values: {', '.join(duplicates)}")

    required_ids = tuple(
        sorted(
            {
                _clean_identifier(value, field_name="required_validator_id")
                for value in required_validator_ids
            }
        )
    )
    by_id = {check.validator_id: check for check in parsed_checks}
    missing = tuple(validator_id for validator_id in required_ids if validator_id not in by_id)
    critical_failures = tuple(
        sorted(
            check.validator_id
            for check in parsed_checks
            if check.critical and check.outcome is ValidationOutcome.FAIL
        )
    )
    required_failures = tuple(
        sorted(
            check.validator_id
            for check in parsed_checks
            if (check.required or check.validator_id in required_ids)
            and check.outcome is ValidationOutcome.FAIL
        )
    )
    skipped_required = tuple(
        sorted(
            check.validator_id
            for check in parsed_checks
            if (check.required or check.validator_id in required_ids)
            and check.outcome is ValidationOutcome.SKIP
        )
    )
    incomplete_evidence = tuple(
        sorted(
            check.validator_id
            for check in parsed_checks
            if (check.required or check.validator_id in required_ids)
            and check.outcome is ValidationOutcome.PASS
            and not check.evidence_complete()
        )
    )
    contradictions = tuple(
        sorted(
            {str(item or "").strip() for item in contradiction_failures if str(item or "").strip()}
        )
    )

    reasons: list[str] = []
    if not capability_supported:
        status = ScientificStatus.UNSUPPORTED
        reasons.append("requested materials capability is not supported by the release sandbox")
    elif critical_failures or required_failures or contradictions:
        status = ScientificStatus.FAILED
        if critical_failures:
            reasons.append(f"critical validators failed: {', '.join(critical_failures)}")
        noncritical_required = sorted(set(required_failures) - set(critical_failures))
        if noncritical_required:
            reasons.append(f"required validators failed: {', '.join(noncritical_required)}")
        if contradictions:
            reasons.append(f"prose/artifact contradictions: {'; '.join(contradictions)}")
    elif normalized_run_status != "succeeded":
        status = ScientificStatus.UNVERIFIED
        reasons.append(f"orchestration run status is {normalized_run_status}, not succeeded")
    elif missing or skipped_required or incomplete_evidence or not required_ids:
        status = ScientificStatus.UNVERIFIED
        if not required_ids:
            reasons.append("task contract declared no required validators")
        if missing:
            reasons.append(f"required validators missing: {', '.join(missing)}")
        if skipped_required:
            reasons.append(f"required validators skipped: {', '.join(skipped_required)}")
        if incomplete_evidence:
            reasons.append(
                "passing validators lack observed/expected values, tolerance rationale, "
                "explicit units, library versions, or hashed evidence: "
                f"{', '.join(incomplete_evidence)}"
            )
    else:
        status = ScientificStatus.VERIFIED

    silent_success = normalized_run_status == "succeeded" and status is ScientificStatus.FAILED
    return ScientificAssessment(
        run_status=normalized_run_status,
        scientific_status=status,
        checks=tuple(sorted(parsed_checks, key=lambda check: check.validator_id)),
        required_validator_ids=required_ids,
        reasons=tuple(reasons),
        missing_validator_ids=missing,
        critical_failures=critical_failures,
        contradiction_failures=contradictions,
        capability_supported=capability_supported,
        silent_success=silent_success,
    )


def parse_assessment_record(value: Mapping[str, Any]) -> ScientificAssessment:
    """Parse and recompute a serialized record, rejecting self-inconsistency.

    The generated analysis is not trusted to set its own top-level verdict. The
    parser reconstructs the verdict from checks and the task contract, then
    compares all decision-bearing fields with the serialized values.
    """

    _assert_finite_json_value(value, field_name="materials_validation")
    required_fields = {
        "schema_version",
        "run_status",
        "scientific_status",
        "verified",
        "silent_success",
        "capability_supported",
        "required_validator_ids",
        "missing_validator_ids",
        "critical_failures",
        "contradiction_failures",
        "reasons",
        "checks",
    }
    missing_fields = sorted(required_fields - set(value))
    if missing_fields:
        raise ValueError(
            "materials validation record is missing required fields: " + ", ".join(missing_fields)
        )
    if str(value.get("schema_version") or "") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported materials validation schema_version {value.get('schema_version')!r}"
        )
    raw_checks = value.get("checks")
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raise ValueError("materials validation checks must be a list")
    if any(not isinstance(item, Mapping) for item in raw_checks):
        raise ValueError("each materials validation check must be an object")
    raw_required_ids = value.get("required_validator_ids") or []
    if not isinstance(raw_required_ids, Sequence) or isinstance(raw_required_ids, (str, bytes)):
        raise ValueError("required_validator_ids must be a list")
    raw_contradictions = value.get("contradiction_failures") or []
    if not isinstance(raw_contradictions, Sequence) or isinstance(raw_contradictions, (str, bytes)):
        raise ValueError("contradiction_failures must be a list")
    if not isinstance(value.get("capability_supported"), bool):
        raise ValueError("capability_supported must be a boolean")
    if not isinstance(value.get("verified"), bool):
        raise ValueError("verified must be a boolean")
    if not isinstance(value.get("silent_success"), bool):
        raise ValueError("silent_success must be a boolean")

    assessment = assess_scientific_status(
        run_status=str(value.get("run_status") or ""),
        checks=[ValidationCheck.from_mapping(item) for item in raw_checks],
        required_validator_ids=[str(item) for item in raw_required_ids],
        capability_supported=value["capability_supported"],
        contradiction_failures=[str(item) for item in raw_contradictions],
    )
    expected = assessment.to_dict()
    decision_fields = (
        "run_status",
        "scientific_status",
        "verified",
        "silent_success",
        "capability_supported",
        "required_validator_ids",
        "missing_validator_ids",
        "critical_failures",
        "contradiction_failures",
        "reasons",
    )
    mismatches = [
        key for key in decision_fields if key in value and value.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            "self-inconsistent materials validation decision fields: " + ", ".join(mismatches)
        )
    return assessment


def canonical_record_json(assessment: ScientificAssessment) -> str:
    """Serialize a validation record deterministically for hashing/provenance."""

    return json.dumps(
        assessment.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def record_sha256(assessment: ScientificAssessment) -> str:
    return hashlib.sha256(canonical_record_json(assessment).encode("utf-8")).hexdigest()
