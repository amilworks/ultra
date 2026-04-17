from __future__ import annotations

from src.evals.golden_tasks import (
    GoldenTaskCase,
    GoldenTaskExpectation,
    GoldenTaskInput,
    evaluate_golden_task_case,
    evaluate_golden_task_suite,
)


def test_golden_task_case_scores_contract_density_and_quality() -> None:
    case = GoldenTaskCase(
        case_id="case-1",
        title="Quantify nuclei coverage",
        input=GoldenTaskInput(prompt="Quantify nuclei coverage for the uploaded stack."),
        expectation=GoldenTaskExpectation(
            required_keywords=["coverage"],
            min_evidence_items=1,
            min_measurement_items=1,
            min_answer_completeness=0.5,
            max_meta_narration_rate=0.4,
        ),
    )
    response_payload = {
        "response_text": (
            "Coverage was 42.1 percent across the inspected field. "
            "Limitation: only one field was measured. Next step: compare a second replicate."
        ),
        "contract": {
            "result": "Coverage was 42.1 percent across the inspected field.",
            "confidence": {"level": "medium", "why": ["Single-field estimate only."]},
            "evidence": [{"source": "tool", "summary": "Segmentation mask coverage report"}],
            "measurements": [{"name": "coverage", "value": 42.1, "unit": "%"}],
            "limitations": ["Only one field was measured."],
            "next_steps": [{"action": "Compare a second replicate."}],
        },
    }

    result = evaluate_golden_task_case(case, response_payload)

    assert result.passed is True
    assert result.case_id == "case-1"
    assert any(item.name == "measurement_density" and item.passed for item in result.checks)


def test_golden_task_suite_aggregates_pass_and_fail_counts() -> None:
    cases = [
        GoldenTaskCase(
            case_id="pass-case",
            title="Passing case",
            input=GoldenTaskInput(prompt="Provide the answer."),
            expectation=GoldenTaskExpectation(
                min_answer_completeness=0.3,
                max_meta_narration_rate=1.0,
            ),
        ),
        GoldenTaskCase(
            case_id="fail-case",
            title="Failing case",
            input=GoldenTaskInput(prompt="Provide the answer."),
            expectation=GoldenTaskExpectation(required_keywords=["mitochondria"]),
        ),
    ]
    responses = {
        "pass-case": {
            "response_text": "Direct answer with limitation and next step.",
            "contract": {
                "result": "Direct answer with limitation and next step.",
                "confidence": {"level": "medium", "why": ["Draft."]},
                "limitations": ["Single run only."],
                "next_steps": [{"action": "Repeat the run."}],
            },
        },
        "fail-case": {
            "response_text": "This answer never mentions the required keyword.",
            "contract": {
                "result": "This answer never mentions the required keyword.",
                "confidence": {"level": "medium", "why": ["Draft."]},
                "limitations": ["Single run only."],
                "next_steps": [{"action": "Repeat the run."}],
            },
        },
    }

    result = evaluate_golden_task_suite(cases, responses)

    assert result.total_cases == 2
    assert result.passed_cases == 1
    assert result.failed_cases == 1
