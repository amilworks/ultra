#!/usr/bin/env python3
"""Capture MatTools verifier output before upstream's loose ``"ok" in`` check.

This is an additional shadow gate. It does not replace or modify the published
MatTools score produced by upstream ``result_analysis.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from mattools_safe_parser import SafeComplexDictParser

MAX_RAW_OUTPUT_CHARS = 1_000_000


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def _strict_raw_result(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "classification": "strict_failure",
            "parsed_result": None,
            "strict_exact_ok": False,
            "counter_pair": None,
        }
    if parsed == "ok":
        return {
            "classification": "strict_success",
            "parsed_result": parsed,
            "strict_exact_ok": True,
            "counter_pair": None,
        }
    if isinstance(parsed, list) and len(parsed) >= 2:
        incorrect, total = parsed[-2:]
        if type(incorrect) is int and type(total) is int and 0 <= incorrect <= total:
            return {
                "classification": "strict_partial",
                "parsed_result": parsed,
                "strict_exact_ok": False,
                "counter_pair": [incorrect, total],
            }
    return {
        "classification": "strict_failure",
        "parsed_result": parsed,
        "strict_exact_ok": False,
        "counter_pair": None,
    }


def run_shadow(snapshot_src: Path, submissions: Path) -> dict[str, Any]:
    sys.path.insert(0, str(snapshot_src))
    from docker_sandbox import DockerSandbox  # type: ignore[import-not-found]

    records = _read_jsonl(submissions)
    sandbox = DockerSandbox()
    results: list[dict[str, Any]] = []
    question_root = snapshot_src / "question_segments" / "pymatgen_analysis_defects"
    for index, record in enumerate(records, start=1):
        task_id = str(record.get("question_file_path") or "")
        function_name = str(record.get("function_name") or "")
        source = str(record.get("function") or "")
        result: dict[str, Any] = {
            "ordinal": index,
            "question_file_path": task_id,
            "function_name": function_name,
            "runnable": False,
            "classification": "strict_function_error",
            "strict_exact_ok": False,
            "counter_pair": None,
        }
        execution = sandbox.execute_code(f"{source}\nprint({function_name}())")
        if not isinstance(execution, dict):
            result["code_execution_error_sha256"] = _sha256_text(str(execution))
            results.append(result)
            continue
        stdout = str(execution.get("stdout") or "")
        result["code_stdout_sha256"] = _sha256_text(stdout)
        result["code_stdout"] = stdout[:MAX_RAW_OUTPUT_CHARS]
        result["code_stdout_truncated"] = len(stdout) > MAX_RAW_OUTPUT_CHARS
        if not stdout:
            results.append(result)
            continue
        try:
            # A fresh parser per attempt prevents placeholder/state carryover
            # between otherwise independent benchmark questions.
            generated = SafeComplexDictParser().parse(stdout)
        except Exception as exc:
            result["parse_error_type"] = type(exc).__name__
            results.append(result)
            continue
        if not isinstance(generated, dict) or not generated:
            results.append(result)
            continue
        result["runnable"] = True
        verifier = question_root / task_id / "new_unit_test.py"
        raw = sandbox.execute_file(
            params_dict=generated,
            py_filename=str(verifier),
            function_name=task_id,
        )
        raw_text = str(raw)
        result["raw_verifier_output_sha256"] = _sha256_text(raw_text)
        result["raw_verifier_output"] = raw_text[:MAX_RAW_OUTPUT_CHARS]
        result["raw_verifier_output_truncated"] = len(raw_text) > MAX_RAW_OUTPUT_CHARS
        result.update(_strict_raw_result(raw_text))
        results.append(result)
    return {
        "schema_version": "1",
        "purpose": "pre-normalization strict shadow; not the published MatTools score",
        "submission_sha256": hashlib.sha256(submissions.read_bytes()).hexdigest(),
        "result_count": len(results),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-src", required=True, type=Path)
    parser.add_argument("--submissions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_shadow(args.snapshot_src.resolve(), args.submissions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
