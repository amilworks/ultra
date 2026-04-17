"""Golden-task evaluation helpers for frontier-style scientific diligence."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.evals.research_review import audit_contract_payload, score_response_quality


class GoldenTaskInput(BaseModel):
    prompt: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)


class GoldenTaskExpectation(BaseModel):
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    min_evidence_items: int = 0
    min_measurement_items: int = 0
    max_meta_narration_rate: float = 0.45
    min_answer_completeness: float = 0.65


class GoldenTaskCase(BaseModel):
    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(default="scientific_workflow", min_length=1)
    input: GoldenTaskInput
    expectation: GoldenTaskExpectation = Field(default_factory=GoldenTaskExpectation)
    baseline_notes: list[str] = Field(default_factory=list)


class GoldenTaskCheckResult(BaseModel):
    name: str
    passed: bool
    detail: str


class GoldenTaskScore(BaseModel):
    case_id: str
    passed: bool
    checks: list[GoldenTaskCheckResult] = Field(default_factory=list)
    response_quality: dict[str, Any] = Field(default_factory=dict)
    contract_audit: dict[str, Any] = Field(default_factory=dict)


class GoldenTaskSuiteResult(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_results: list[GoldenTaskScore] = Field(default_factory=list)


def _normalized_text(response_payload: dict[str, Any]) -> str:
    response_text = str(response_payload.get("response_text") or "").strip()
    contract = response_payload.get("contract")
    if response_text:
        return response_text.lower()
    if isinstance(contract, dict):
        return str(contract.get("result") or "").strip().lower()
    return ""


def evaluate_golden_task_case(
    case: GoldenTaskCase,
    response_payload: dict[str, Any],
) -> GoldenTaskScore:
    contract_audit = audit_contract_payload(response_payload)
    response_quality = score_response_quality(response_payload)
    response_text = _normalized_text(response_payload)
    evidence_count = int(contract_audit.get("evidence_count") or 0)
    measurement_count = int(contract_audit.get("measurement_count") or 0)
    meta_narration_rate = float(response_quality.get("meta_narration_rate") or 0.0)
    answer_completeness = float(response_quality.get("answer_completeness") or 0.0)

    checks = [
        GoldenTaskCheckResult(
            name="required_keywords",
            passed=all(
                keyword.lower() in response_text for keyword in case.expectation.required_keywords
            ),
            detail=(
                "All required keywords were present."
                if case.expectation.required_keywords
                else "No required keywords configured."
            ),
        ),
        GoldenTaskCheckResult(
            name="forbidden_keywords",
            passed=all(
                keyword.lower() not in response_text
                for keyword in case.expectation.forbidden_keywords
            ),
            detail=(
                "No forbidden keywords were present."
                if case.expectation.forbidden_keywords
                else "No forbidden keywords configured."
            ),
        ),
        GoldenTaskCheckResult(
            name="evidence_density",
            passed=evidence_count >= case.expectation.min_evidence_items,
            detail=(
                f"Found {evidence_count} evidence items; expected at least "
                f"{case.expectation.min_evidence_items}."
            ),
        ),
        GoldenTaskCheckResult(
            name="measurement_density",
            passed=measurement_count >= case.expectation.min_measurement_items,
            detail=(
                f"Found {measurement_count} measurement items; expected at least "
                f"{case.expectation.min_measurement_items}."
            ),
        ),
        GoldenTaskCheckResult(
            name="meta_narration_rate",
            passed=meta_narration_rate <= case.expectation.max_meta_narration_rate,
            detail=(
                f"Meta narration rate was {meta_narration_rate:.3f}; expected at most "
                f"{case.expectation.max_meta_narration_rate:.3f}."
            ),
        ),
        GoldenTaskCheckResult(
            name="answer_completeness",
            passed=answer_completeness >= case.expectation.min_answer_completeness,
            detail=(
                f"Answer completeness was {answer_completeness:.3f}; expected at least "
                f"{case.expectation.min_answer_completeness:.3f}."
            ),
        ),
    ]

    return GoldenTaskScore(
        case_id=case.case_id,
        passed=all(item.passed for item in checks),
        checks=checks,
        response_quality=response_quality,
        contract_audit=contract_audit,
    )


def evaluate_golden_task_suite(
    cases: list[GoldenTaskCase],
    responses_by_case_id: dict[str, dict[str, Any]],
) -> GoldenTaskSuiteResult:
    case_results = [
        evaluate_golden_task_case(
            case,
            responses_by_case_id.get(case.case_id, {}),
        )
        for case in cases
    ]
    passed_cases = sum(1 for item in case_results if item.passed)
    return GoldenTaskSuiteResult(
        total_cases=len(case_results),
        passed_cases=passed_cases,
        failed_cases=max(len(case_results) - passed_cases, 0),
        case_results=case_results,
    )
