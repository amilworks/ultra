from __future__ import annotations

import json

import pytest

from ultra_deepagents.materials.validation import (
    EvidenceArtifact,
    ScientificStatus,
    ValidationCheck,
    ValidationOutcome,
    assess_scientific_status,
    canonical_record_json,
    parse_assessment_record,
    record_sha256,
)


_DIGEST = "a" * 64


def _evidence(name: str = "result.json") -> tuple[EvidenceArtifact, ...]:
    return (
        EvidenceArtifact(
            name=name,
            sha256=_DIGEST,
            path=f"/outputs/{name}",
            size_bytes=123,
        ),
    )


def _passing_check(validator_id: str, *, critical: bool = False) -> ValidationCheck:
    return ValidationCheck(
        validator_id=validator_id,
        outcome=ValidationOutcome.PASS,
        observed={"rgb": [1.0, 0.0, 0.0]},
        expected={"rgb": [1.0, 0.0, 0.0], "absolute_tolerance": 0.15},
        units="dimensionless",
        tolerance_rationale="orix TSL corner colors allow rendering roundoff",
        required=True,
        critical=critical,
        library_versions={"orix": "0.14.3"},
        evidence=_evidence(),
    )


def test_verified_requires_declared_passing_validators_and_hashed_evidence():
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[_passing_check("ebsd.ipf_001_red", critical=True)],
        required_validator_ids=["ebsd.ipf_001_red"],
    )

    assert result.scientific_status is ScientificStatus.VERIFIED
    assert result.verified is True
    assert result.silent_success is False
    assert result.reasons == ()


def test_succeeded_run_with_wrong_ipf_is_a_silent_scientific_failure():
    wrong_ipf = ValidationCheck(
        validator_id="ebsd.ipf_001_red",
        outcome=ValidationOutcome.FAIL,
        observed={"rgb": [0.0, 0.0, 1.0]},
        expected={"rgb": [1.0, 0.0, 0.0], "absolute_tolerance": 0.15},
        units="dimensionless",
        tolerance_rationale="declared TSL cubic IPF convention",
        required=True,
        critical=True,
        library_versions={"orix": "0.14.3"},
        evidence=_evidence("ipf_color_key.png"),
        message="[001] was blue instead of red",
    )

    result = assess_scientific_status(
        run_status="succeeded",
        checks=[wrong_ipf],
        required_validator_ids=["ebsd.ipf_001_red"],
    )

    assert result.scientific_status is ScientificStatus.FAILED
    assert result.silent_success is True
    assert result.critical_failures == ("ebsd.ipf_001_red",)


def test_required_skip_or_missing_validator_cannot_be_verified():
    skipped = ValidationCheck(
        validator_id="xrd.fcc_first_peak",
        outcome=ValidationOutcome.SKIP,
        required=True,
        critical=True,
        message="pymatgen missing",
    )
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[skipped],
        required_validator_ids=["xrd.fcc_first_peak", "structure.space_group"],
    )

    assert result.scientific_status is ScientificStatus.UNVERIFIED
    assert result.missing_validator_ids == ("structure.space_group",)
    assert any("skipped" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "missing_field",
    [
        "observed",
        "expected",
        "units",
        "tolerance_rationale",
        "library_versions",
        "evidence",
    ],
)
def test_passing_required_validator_without_complete_evidence_is_unverified(missing_field: str):
    kwargs = {
        "validator_id": "structure.space_group",
        "outcome": ValidationOutcome.PASS,
        "observed": 221,
        "expected": 221,
        "units": "dimensionless",
        "tolerance_rationale": "integer international space-group number",
        "library_versions": {"pymatgen": "2026.5.4"},
        "evidence": _evidence("structure.json"),
    }
    if missing_field in {"observed", "expected"}:
        kwargs[missing_field] = None
    elif missing_field == "tolerance_rationale":
        kwargs[missing_field] = ""
    elif missing_field == "units":
        kwargs[missing_field] = None
    elif missing_field == "library_versions":
        kwargs[missing_field] = {}
    else:
        kwargs[missing_field] = ()

    result = assess_scientific_status(
        run_status="succeeded",
        checks=[ValidationCheck(**kwargs)],
        required_validator_ids=["structure.space_group"],
    )

    assert result.scientific_status is ScientificStatus.UNVERIFIED
    assert "lack" in result.reasons[-1]


def test_task_contract_can_promote_an_optional_check_to_required():
    check = ValidationCheck(
        validator_id="structure.space_group",
        outcome=ValidationOutcome.PASS,
        observed=221,
        expected=221,
        required=False,
    )
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[check],
        required_validator_ids=["structure.space_group"],
    )

    assert result.scientific_status is ScientificStatus.UNVERIFIED
    assert "lack" in result.reasons[-1]


def test_contradiction_between_prose_and_artifact_is_failed():
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[_passing_check("ebsd.ipf_001_red")],
        required_validator_ids=["ebsd.ipf_001_red"],
        contradiction_failures=["response says [001] is blue while artifact is red"],
    )

    assert result.scientific_status is ScientificStatus.FAILED
    assert result.silent_success is True


def test_unsupported_capability_is_explicit_not_a_fake_failure_or_success():
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[],
        required_validator_ids=["dft.total_energy"],
        capability_supported=False,
    )

    assert result.scientific_status is ScientificStatus.UNSUPPORTED
    assert result.silent_success is False


def test_record_serialization_is_canonical_and_content_addressed():
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[_passing_check("ebsd.ipf_001_red")],
        required_validator_ids=["ebsd.ipf_001_red"],
    )

    encoded = canonical_record_json(result)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == "1"
    assert decoded["scientific_status"] == "verified"
    assert len(record_sha256(result)) == 64
    assert record_sha256(result) == record_sha256(result)
    reparsed = parse_assessment_record(decoded)
    assert reparsed == result


def test_serialized_record_cannot_claim_verified_when_a_check_failed():
    failed = assess_scientific_status(
        run_status="succeeded",
        checks=[
            ValidationCheck(
                validator_id="xrd.fcc_first_peak",
                outcome=ValidationOutcome.FAIL,
                critical=True,
            )
        ],
        required_validator_ids=["xrd.fcc_first_peak"],
    ).to_dict()
    failed["scientific_status"] = "verified"
    failed["verified"] = True

    with pytest.raises(ValueError, match="self-inconsistent"):
        parse_assessment_record(failed)


def test_serialized_record_requires_all_decision_fields():
    result = assess_scientific_status(
        run_status="succeeded",
        checks=[_passing_check("ebsd.ipf_001_red")],
        required_validator_ids=["ebsd.ipf_001_red"],
    ).to_dict()
    del result["capability_supported"]

    with pytest.raises(ValueError, match="missing required fields"):
        parse_assessment_record(result)


def test_invalid_evidence_digest_is_rejected():
    with pytest.raises(ValueError, match="invalid SHA-256"):
        EvidenceArtifact(name="result.json", sha256="not-a-digest", path="/outputs/result.json")


def test_duplicate_validator_ids_are_rejected():
    check = _passing_check("ebsd.ipf_001_red")
    with pytest.raises(ValueError, match="duplicate validator_id"):
        assess_scientific_status(
            run_status="succeeded",
            checks=[check, check],
            required_validator_ids=["ebsd.ipf_001_red"],
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field_name", ["observed", "expected"])
def test_validation_check_rejects_nested_nonfinite_numbers(
    field_name: str, nonfinite: float
):
    kwargs = {
        "validator_id": "xrd.fcc_first_peak",
        "outcome": ValidationOutcome.PASS,
        "observed": {"two_theta_degrees": [44.6]},
        "expected": {"two_theta_degrees": [44.5]},
        "units": "degrees",
    }
    kwargs[field_name] = {"nested": [1.0, nonfinite]}

    with pytest.raises(ValueError, match="non-finite"):
        ValidationCheck(**kwargs)


def test_serialized_record_rejects_nonfinite_number_before_recomputing_verdict():
    record = assess_scientific_status(
        run_status="succeeded",
        checks=[_passing_check("ebsd.ipf_001_red")],
        required_validator_ids=["ebsd.ipf_001_red"],
    ).to_dict()
    record["checks"][0]["observed"] = {"rgb": [1.0, float("nan"), 0.0]}

    with pytest.raises(ValueError, match="non-finite"):
        parse_assessment_record(record)
