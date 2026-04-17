#!/usr/bin/env python3
"""Evaluate code-execution reasoning modes on difficult synthetic STEM prompts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import httpx

PROBLEM_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codeexec_reasoning" / "problems.json"
)

REASONER_MODES: list[dict[str, Any]] = [
    {
        "id": "baseline_tool_workflow",
        "reasoning_mode": "deep",
        "force_execution_regime": "validated_tool",
    },
    {
        "id": "codeexec_reasoning_agent",
        "reasoning_mode": "deep",
        "force_execution_regime": None,
    },
    {
        "id": "fast_reasoning_writer",
        "reasoning_mode": "fast",
        "force_execution_regime": None,
    },
]


def load_problems(path: Path = PROBLEM_FIXTURE_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload if isinstance(item, dict)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fixture-path", default=str(PROBLEM_FIXTURE_PATH))
    parser.add_argument("--max-runtime-seconds", type=int, default=900)
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    return parser.parse_args()


def _tool_invocations(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = parsed.get("metadata") if isinstance(parsed, dict) else {}
    invocations = metadata.get("tool_invocations") if isinstance(metadata, dict) else None
    if isinstance(invocations, list):
        return [item for item in invocations if isinstance(item, dict)]
    return []


def _tool_name(invocation: dict[str, Any]) -> str:
    return (
        str(
            invocation.get("tool")
            or invocation.get("tool_name")
            or invocation.get("name")
            or invocation.get("display_name")
            or ""
        )
        .strip()
    )


def _artifact_names(execute_invocation: dict[str, Any]) -> list[str]:
    envelope = (
        dict(execute_invocation.get("output_envelope") or {})
        if isinstance(execute_invocation.get("output_envelope"), dict)
        else {}
    )
    output_files = envelope.get("output_files")
    if not isinstance(output_files, list):
        return []
    return [str(item).strip() for item in output_files if str(item).strip()]


def _measurement_count(execute_invocation: dict[str, Any]) -> int:
    summary = (
        dict(execute_invocation.get("output_summary") or {})
        if isinstance(execute_invocation.get("output_summary"), dict)
        else {}
    )
    key_measurements = summary.get("key_measurements")
    if isinstance(key_measurements, list):
        return len([item for item in key_measurements if item])
    envelope = (
        dict(execute_invocation.get("output_envelope") or {})
        if isinstance(execute_invocation.get("output_envelope"), dict)
        else {}
    )
    envelope_measurements = envelope.get("key_measurements")
    if isinstance(envelope_measurements, list):
        return len([item for item in envelope_measurements if item])
    return 0


def score_run(payload: dict[str, Any]) -> dict[str, float]:
    response_text = str(payload.get("response_text") or "")
    missing_outputs = list(payload.get("missing_expected_outputs") or [])
    success = bool(payload.get("success"))
    return {
        "artifact_completion": 1.0 if not missing_outputs else 0.0,
        "failure_honesty": (
            1.0
            if success
            or "did not complete successfully" in response_text.lower()
            or "computation failed" in response_text.lower()
            else 0.0
        ),
        "quantitative_density": float(payload.get("measurement_count") or 0),
        "scientific_prose": (
            1.0
            if "limitation" in response_text.lower() or "limitations" in response_text.lower()
            else 0.0
        ),
    }


def run_eval_prompt(
    *,
    client: httpx.Client,
    base_url: str,
    problem: dict[str, Any],
    mode: dict[str, Any],
    max_runtime_seconds: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": str(problem.get("prompt") or "")}],
        "workflow_hint": {"id": "pro_mode", "source": "slash_menu"},
        "reasoning_mode": str(mode.get("reasoning_mode") or "deep"),
        "debug": True,
        "budgets": {"max_tool_calls": 12, "max_runtime_seconds": max_runtime_seconds},
    }
    forced_execution_regime = str(mode.get("force_execution_regime") or "").strip().lower()
    if forced_execution_regime:
        payload["benchmark"] = {"force_pro_mode_execution_regime": forced_execution_regime}

    started = time.perf_counter()
    response = client.post(f"{base_url.rstrip('/')}/v1/chat", json=payload)
    response.raise_for_status()
    parsed = response.json()
    runtime_seconds = time.perf_counter() - started
    tool_invocations = _tool_invocations(parsed)
    tool_names = [_tool_name(item) for item in tool_invocations]
    execute_invocation = next(
        (
            item
            for item in reversed(tool_invocations)
            if _tool_name(item) == "execute_python_job"
        ),
        {},
    )
    artifact_names = _artifact_names(execute_invocation)
    artifact_basenames = {Path(item).name for item in artifact_names}
    expected_artifacts = [
        str(item).strip() for item in list(problem.get("expected_artifacts") or []) if str(item).strip()
    ]
    missing_expected_outputs = [
        item
        for item in expected_artifacts
        if item not in artifact_names and Path(item).name not in artifact_basenames
    ]
    envelope = (
        dict(execute_invocation.get("output_envelope") or {})
        if isinstance(execute_invocation.get("output_envelope"), dict)
        else {}
    )
    success = bool(
        execute_invocation
        and str(execute_invocation.get("status") or "").strip().lower() == "completed"
        and bool(envelope.get("success", True))
    )
    result = {
        "problem_id": str(problem.get("id") or ""),
        "mode_id": str(mode.get("id") or ""),
        "success": success,
        "runtime_seconds": round(runtime_seconds, 4),
        "response_text": str(parsed.get("response_text") or ""),
        "tool_names": tool_names,
        "expected_tools": list(problem.get("expected_tools") or []),
        "artifact_names": artifact_names,
        "missing_expected_outputs": missing_expected_outputs,
        "measurement_count": _measurement_count(execute_invocation),
        "execution_path": str(
            ((parsed.get("metadata") or {}).get("pro_mode") or {}).get("execution_path") or ""
        ),
        "response_model": str(parsed.get("model") or ""),
    }
    result["score"] = score_run(result)
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode_id") or ""), []).append(row)
    modes: dict[str, Any] = {}
    for mode_id, items in by_mode.items():
        count = len(items)
        modes[mode_id] = {
            "runs": count,
            "success_rate": round(
                sum(1 for item in items if item.get("success")) / count, 4
            )
            if count
            else 0.0,
            "mean_runtime_seconds": round(
                sum(float(item.get("runtime_seconds") or 0.0) for item in items) / count, 4
            )
            if count
            else 0.0,
            "mean_artifact_completion": round(
                sum(float((item.get("score") or {}).get("artifact_completion") or 0.0) for item in items)
                / count,
                4,
            )
            if count
            else 0.0,
            "mean_failure_honesty": round(
                sum(float((item.get("score") or {}).get("failure_honesty") or 0.0) for item in items)
                / count,
                4,
            )
            if count
            else 0.0,
            "mean_quantitative_density": round(
                sum(float((item.get("score") or {}).get("quantitative_density") or 0.0) for item in items)
                / count,
                4,
            )
            if count
            else 0.0,
        }
    return {"modes": modes, "rows": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "problem_id",
        "mode_id",
        "success",
        "runtime_seconds",
        "execution_path",
        "measurement_count",
        "missing_expected_outputs",
        "tool_names",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "problem_id": row.get("problem_id"),
                    "mode_id": row.get("mode_id"),
                    "success": row.get("success"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "execution_path": row.get("execution_path"),
                    "measurement_count": row.get("measurement_count"),
                    "missing_expected_outputs": json.dumps(
                        row.get("missing_expected_outputs") or [], ensure_ascii=False
                    ),
                    "tool_names": json.dumps(row.get("tool_names") or [], ensure_ascii=False),
                }
            )


def main() -> int:
    args = _parse_args()
    problems = load_problems(Path(args.fixture_path))
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=float(args.max_runtime_seconds) + 60.0) as client:
        for problem in problems:
            for mode in REASONER_MODES:
                rows.append(
                    run_eval_prompt(
                        client=client,
                        base_url=args.base_url,
                        problem=problem,
                        mode=mode,
                        max_runtime_seconds=int(args.max_runtime_seconds),
                    )
                )
    summary = _aggregate(rows)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.csv_output:
        _write_csv(Path(args.csv_output), rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
