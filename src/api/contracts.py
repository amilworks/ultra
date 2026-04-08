from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.api.schemas import (
    AssistantContract,
    ConfidenceBlock,
    EvidenceItem,
    MeasurementItem,
    NextStepAction,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BULLET_PREFIX_RE = re.compile(r"^(?:[-*]\s+|\d+\.\s+)")
_PATH_TOKEN_RE = re.compile(r"(?P<token>(?:[A-Za-z]:\\\\|/|~\\/|\\.\\.?/|[A-Za-z0-9._-]+/)[^\s`\"'<>|]+)")
_INTERNAL_EXECUTION_TOKEN_RE = re.compile(
    r"\b(?:codejob_[a-z0-9_-]{6,64}|exec_[a-z0-9_-]{6,64}|tool_[0-9]+_[0-9]+)\b",
    flags=re.IGNORECASE,
)

_DEFAULT_LIMITATIONS = [
    "Conclusions are preliminary and should be validated with additional independent data.",
    "Current outputs may be affected by missing metadata, model assumptions, or tool availability constraints.",
]

_DEFAULT_NEXT_STEP_ACTIONS = [
    "Validate findings on an independent holdout dataset and compare key effect sizes.",
    "Run a targeted parameter sweep to test robustness of the current result.",
    "Generate or update a reproducibility report with run IDs, tool versions, and artifacts.",
]


def coerce_assistant_contract(
    response_text: str,
    run_id: str | None = None,
    *,
    minimum_next_steps: int = 1,
    minimum_limitations: int = 1,
) -> AssistantContract:
    """Parse model output into ``AssistantContract`` with safe fallbacks.

    Parameters
    ----------
    response_text : str
        Raw model response text.
    run_id : str or None, default=None
        Optional run identifier used to preserve traceability in evidence rows.
    minimum_next_steps : int, default=1
        Minimum number of next-step actions guaranteed in returned contract.
    minimum_limitations : int, default=1
        Minimum number of limitation statements guaranteed in returned contract.

    Returns
    -------
    AssistantContract
        Normalized contract payload safe for downstream rendering.
    """
    payload = _extract_json_payload(response_text)
    if isinstance(payload, dict):
        normalized = _normalize_contract_payload(dict(payload), run_id=run_id)
        if not normalized.get("result"):
            normalized["result"] = str(payload.get("answer") or response_text or "").strip()
        try:
            contract = AssistantContract.model_validate(normalized)
            return _apply_contract_defaults(
                contract,
                response_text=response_text,
                minimum_next_steps=minimum_next_steps,
                minimum_limitations=minimum_limitations,
            )
        except Exception:
            pass

    fallback_evidence: list[EvidenceItem] = []
    if run_id:
        fallback_evidence.append(
            EvidenceItem(
                source="run",
                run_id=run_id,
                summary="Model response was not in structured contract format.",
            )
        )

    fallback = AssistantContract(
        result=(response_text or "").strip() or "No response generated.",
        evidence=fallback_evidence,
        confidence=ConfidenceBlock(
            level="low",
            why=["Output was returned as unstructured text."],
        ),
        limitations=["Structured scientific response contract was not produced."],
    )
    return _apply_contract_defaults(
        fallback,
        response_text=response_text,
        minimum_next_steps=minimum_next_steps,
        minimum_limitations=minimum_limitations,
    )


def enrich_contract_with_progress_signals(
    contract: AssistantContract,
    *,
    progress_events: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> AssistantContract:
    """Augment a contract with measurements and evidence from progress events.

    Parameters
    ----------
    contract : AssistantContract
        Base contract to enrich.
    progress_events : list[dict[str, Any]] or None, default=None
        Tool progress events emitted during execution.
    run_id : str or None, default=None
        Optional run identifier used for generated evidence items.

    Returns
    -------
    AssistantContract
        Enriched and sanitized contract ready for user-facing rendering.
    """
    summary = _latest_code_execution_summary(progress_events)
    if isinstance(summary, dict) and summary.get("success") is not False:
        existing_measurements = {
            str(item.name or "").strip().lower()
            for item in contract.measurements
            if str(item.name or "").strip()
        }
        for metric in summary.get("measurements") or []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "").strip()
            if not name or name.lower() in existing_measurements:
                continue
            value = metric.get("value")
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                normalized_value: float | int | str = float(value)
            else:
                text_value = str(value).strip()
                if not text_value:
                    continue
                normalized_value = text_value
            contract.measurements.append(MeasurementItem(name=name, value=normalized_value))
            existing_measurements.add(name.lower())
            if len(contract.measurements) >= 24:
                break

        output_files = [
            str(item).strip()
            for item in (summary.get("output_files") or [])
            if str(item or "").strip()
        ]
        missing_expected_outputs = [
            str(item).strip()
            for item in (summary.get("missing_expected_outputs") or [])
            if str(item or "").strip()
        ]

        if not any(str(e.source or "").strip().lower() == "execute_python_job" for e in contract.evidence):
            details: list[str] = []
            runtime_seconds = summary.get("runtime_seconds")
            if isinstance(runtime_seconds, (int, float)):
                details.append(f"runtime={float(runtime_seconds):.3f}s")
            exit_code = summary.get("exit_code")
            if isinstance(exit_code, int):
                details.append(f"exit_code={exit_code}")
            if output_files:
                details.append(f"outputs={len(output_files)}")
            if missing_expected_outputs:
                details.append(f"missing_expected={len(missing_expected_outputs)}")
            summary_text = "Python sandbox execution completed."
            if details:
                summary_text += " (" + ", ".join(details) + ")"
            contract.evidence.append(
                EvidenceItem(
                    source="execute_python_job",
                    run_id=str(run_id or "").strip() or None,
                    artifact=output_files[0] if output_files else None,
                    summary=summary_text,
                )
            )

        repair_cycles_used = summary.get("repair_cycles_used")
        if isinstance(repair_cycles_used, int) and repair_cycles_used > 0:
            warning = (
                f"Code required {repair_cycles_used} repair cycle(s) before successful execution."
            )
            if warning not in contract.qc_warnings:
                contract.qc_warnings.append(warning)

        if missing_expected_outputs:
            warning = (
                "Some expected outputs were not produced: "
                + ", ".join(missing_expected_outputs[:6])
            )
            if warning not in contract.qc_warnings:
                contract.qc_warnings.append(warning)
            limitation = (
                "Output completeness is partial because expected artifacts are missing."
            )
            if limitation not in contract.limitations:
                contract.limitations.append(limitation)

        if not contract.measurements:
            warning = "No quantitative metrics were extracted from code execution outputs."
            if warning not in contract.qc_warnings:
                contract.qc_warnings.append(warning)
            limitation = (
                "Scientific interpretation is limited because structured numeric metrics were not captured."
            )
            if limitation not in contract.limitations:
                contract.limitations.append(limitation)

        parse_issues = [
            item
            for item in (summary.get("analysis_outputs") or [])
            if isinstance(item, dict) and str(item.get("parse_status") or "") not in {"", "ok"}
        ]
        if parse_issues:
            warning = (
                f"{len(parse_issues)} output artifact(s) could not be fully parsed for metric extraction."
            )
            if warning not in contract.qc_warnings:
                contract.qc_warnings.append(warning)

        confidence_signal = (
            "Python code executed in sandbox with reproducible artifact outputs."
        )
        if confidence_signal not in contract.confidence.why:
            contract.confidence.why.append(confidence_signal)
        if contract.confidence.level == "low" and contract.measurements:
            contract.confidence.level = "medium"

        if _result_is_too_brief(contract.result) or _result_lacks_measurement_detail(
            contract.result,
            contract.measurements,
        ):
            contract.result = _build_code_execution_result_summary(
                existing_result=contract.result,
                output_files=output_files,
                measurements=contract.measurements,
                missing_expected_outputs=missing_expected_outputs,
                runtime_seconds=summary.get("runtime_seconds"),
            )

    _enrich_contract_with_generic_tool_signals(
        contract,
        progress_events=progress_events,
        run_id=run_id,
    )

    _sanitize_contract_user_visible_fields(contract)
    return contract


def _enrich_contract_with_generic_tool_signals(
    contract: AssistantContract,
    *,
    progress_events: list[dict[str, Any]] | None,
    run_id: str | None,
) -> None:
    if not isinstance(progress_events, list):
        return
    existing_sources = {
        str(item.source or "").strip().lower()
        for item in contract.evidence
        if str(item.source or "").strip()
    }
    existing_measurements = {
        str(item.name or "").strip().lower()
        for item in contract.measurements
        if str(item.name or "").strip()
    }
    run_ref = str(run_id or "").strip() or None
    for event in progress_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("event") or "").strip().lower() != "completed":
            continue
        tool_name = str(event.get("tool") or "").strip()
        if not tool_name:
            continue
        tool_key = tool_name.lower()
        if tool_key not in existing_sources:
            summary_text = _render_generic_tool_summary(event)
            artifact_path = _first_progress_artifact_path(event)
            contract.evidence.append(
                EvidenceItem(
                    source=tool_name,
                    run_id=run_ref,
                    artifact=artifact_path,
                    summary=summary_text,
                )
            )
            existing_sources.add(tool_key)
        summary_payload = event.get("summary")
        for name, value in _extract_measurements_from_tool_summary(
            tool_name=tool_name,
            summary=summary_payload,
        ):
            metric_key = name.lower()
            if metric_key in existing_measurements:
                continue
            contract.measurements.append(MeasurementItem(name=name, value=value))
            existing_measurements.add(metric_key)
            if len(contract.measurements) >= 24:
                return


def _render_generic_tool_summary(event: dict[str, Any]) -> str:
    tool_name = str(event.get("tool") or "tool").strip()
    message = str(event.get("message") or "").strip()
    summary = event.get("summary")
    details: list[str] = []
    if isinstance(summary, dict):
        for key in (
            "processed",
            "total_files",
            "total_masks_generated",
            "coverage_percent_mean",
            "object_count",
            "coverage_fraction",
            "mean_object_size",
        ):
            value = summary.get(key)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float, str)):
                details.append(f"{key}={value}")
            if len(details) >= 4:
                break
    if details:
        if message:
            return f"{message} ({', '.join(details)})"
        return f"{tool_name} completed ({', '.join(details)})."
    if message:
        return message
    return f"{tool_name} completed."


def _first_progress_artifact_path(event: dict[str, Any]) -> str | None:
    artifacts = event.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for row in artifacts:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "").strip()
        if path:
            return path
    return None


def _extract_measurements_from_tool_summary(
    *,
    tool_name: str,
    summary: Any,
    max_metrics: int = 8,
) -> list[tuple[str, float | int | str]]:
    if not isinstance(summary, dict):
        return []
    rows: list[tuple[str, float | int | str]] = []
    for key, value in summary.items():
        if len(rows) >= max_metrics:
            break
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            rows.append((f"{tool_name}.{key}", float(value)))
            continue
        if isinstance(value, str):
            token = value.strip()
            if not token:
                continue
            if len(token) > 96:
                continue
            if re.fullmatch(r"[a-zA-Z0-9_.:+/\- ]{1,96}", token):
                rows.append((f"{tool_name}.{key}", token))
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if len(rows) >= max_metrics:
                    break
                if isinstance(sub_value, bool) or sub_value is None:
                    continue
                if isinstance(sub_value, (int, float)):
                    rows.append((f"{tool_name}.{key}.{sub_key}", float(sub_value)))
            continue
    return rows


def select_user_response_text(
    response_text: str,
    *,
    contract: AssistantContract | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> str:
    """Choose the safest and most informative user-facing response text.

    Parameters
    ----------
    response_text : str
        Raw model response text.
    contract : AssistantContract or None, default=None
        Optional structured contract extracted from the same response.
    progress_events : list[dict[str, Any]] or None, default=None
        Optional progress event trace used to decide whether enrichment is
        required.

    Returns
    -------
    str
        Sanitized response text selected for frontend display.
    """
    raw_text = sanitize_user_visible_text(response_text).strip()
    if not isinstance(contract, AssistantContract):
        return raw_text

    summary = _latest_code_execution_summary(progress_events)
    contract_result = sanitize_user_visible_text(contract.result).strip()
    parsed_payload = _extract_json_payload(response_text)
    payload_looks_contract = _looks_like_contract_payload(parsed_payload)
    raw_is_meta = _result_is_meta_summary(raw_text)
    raw_usable = bool(raw_text) and not payload_looks_contract

    if isinstance(summary, dict):
        raw_lacks_measurements = _result_lacks_measurement_detail(
            raw_text,
            contract.measurements,
        )
        raw_needs_enrichment = _code_execution_response_needs_enrichment(
            raw_text,
            contract=contract,
            progress_events=progress_events,
        )
        if summary.get("success") is False:
            if raw_usable and not raw_is_meta:
                return raw_text
            rendered = _render_contract_user_summary(contract)
            if rendered:
                return rendered
            if contract_result:
                return contract_result
            return raw_text
        if (
            raw_usable
            and not raw_is_meta
            and not _result_is_too_brief(raw_text)
            and not raw_lacks_measurements
            and not raw_needs_enrichment
        ):
            return raw_text
        rendered = _render_contract_user_summary(contract)
        if rendered:
            return rendered
        if contract_result and not _result_is_meta_summary(contract_result):
            return contract_result
        if raw_usable:
            return raw_text
        if contract_result:
            return contract_result
        return raw_text

    if raw_usable and not raw_is_meta:
        return raw_text

    if payload_looks_contract or raw_is_meta:
        if contract_result and not _result_is_meta_summary(contract_result):
            return contract_result
        rendered = _render_contract_user_summary(contract)
        if rendered:
            return rendered
        if raw_usable:
            return raw_text
        if contract_result:
            return contract_result

    if contract_result and not _result_is_meta_summary(contract_result) and not raw_text:
        return contract_result
    return raw_text


def _normalize_contract_payload(
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    normalized = dict(payload)

    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    evidence_items: list[dict[str, Any]] = []
    for item in _to_list(normalized.get("evidence")):
        if isinstance(item, dict):
            source = str(item.get("source") or item.get("tool") or "assistant").strip() or "assistant"
            summary = str(item.get("summary") or item.get("message") or item.get("result") or "").strip() or None
            artifact = str(item.get("artifact") or item.get("path") or item.get("uri") or "").strip() or None
            run_ref = str(item.get("run_id") or run_id or "").strip() or None
            entry: dict[str, Any] = {"source": source}
            if run_ref:
                entry["run_id"] = run_ref
            if artifact:
                entry["artifact"] = artifact
            if summary:
                entry["summary"] = summary
            evidence_items.append(entry)
            continue

        text = str(item or "").strip()
        if not text:
            continue
        entry = {
            "source": "assistant",
            "summary": text,
        }
        run_ref = str(run_id or "").strip()
        if run_ref:
            entry["run_id"] = run_ref
        evidence_items.append(entry)
    if "evidence" in normalized:
        normalized["evidence"] = evidence_items

    next_steps: list[dict[str, Any]] = []
    for item in _to_list(normalized.get("next_steps")):
        if isinstance(item, dict):
            action = (
                str(item.get("action") or item.get("step") or item.get("next_step") or "").strip()
            )
            if not action:
                continue
            step: dict[str, Any] = {"action": action}
            workflow = str(item.get("workflow") or "").strip()
            if workflow:
                step["workflow"] = workflow
            args_value = item.get("args")
            if isinstance(args_value, dict):
                step["args"] = args_value
            next_steps.append(step)
            continue

        action = str(item or "").strip()
        if action:
            next_steps.append({"action": action})
    if "next_steps" in normalized:
        normalized["next_steps"] = next_steps

    confidence = normalized.get("confidence")
    if isinstance(confidence, str):
        text = confidence.strip()
        normalized["confidence"] = {
            "level": "medium" if text else "low",
            "why": [text] if text else [],
        }
    elif isinstance(confidence, dict):
        coerced_confidence = dict(confidence)
        why_value = coerced_confidence.get("why")
        if isinstance(why_value, str):
            why = [why_value.strip()] if why_value.strip() else []
        elif isinstance(why_value, list):
            why = [str(item).strip() for item in why_value if str(item).strip()]
        elif why_value is None:
            why = []
        else:
            text = str(why_value).strip()
            why = [text] if text else []
        coerced_confidence["why"] = why
        level = str(coerced_confidence.get("level") or "").strip().lower()
        if level not in {"low", "medium", "high"}:
            coerced_confidence["level"] = "medium" if why else "low"
        normalized["confidence"] = coerced_confidence

    for key in ("qc_warnings", "limitations"):
        if key not in normalized:
            continue
        normalized[key] = [str(item).strip() for item in _to_list(normalized.get(key)) if str(item).strip()]

    if "measurements" in normalized:
        coerced_measurements: list[dict[str, Any]] = []
        for index, item in enumerate(_to_list(normalized.get("measurements"))):
            if isinstance(item, dict):
                raw = dict(item)
                name = (
                    str(
                        raw.get("name")
                        or raw.get("metric")
                        or raw.get("label")
                        or raw.get("key")
                        or ""
                    )
                    .strip()
                )
                value = raw.get("value")
                if value is None:
                    for alt_key in ("count", "amount", "score", "mean", "min", "max", "n"):
                        if raw.get(alt_key) is not None:
                            value = raw.get(alt_key)
                            break
                if value is None and len(raw) == 1:
                    only_key, only_value = next(iter(raw.items()))
                    if str(only_key).strip():
                        name = name or str(only_key).strip()
                        value = only_value
                if not name:
                    if value is None:
                        continue
                    name = f"measurement_{index + 1}"
                if value is None:
                    value = ""
                entry: dict[str, Any] = {"name": name, "value": value}
                unit = str(raw.get("unit") or "").strip()
                if unit:
                    entry["unit"] = unit
                ci95 = raw.get("ci95")
                if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
                    try:
                        entry["ci95"] = (float(ci95[0]), float(ci95[1]))
                    except Exception:
                        pass
                coerced_measurements.append(entry)
                continue

            if isinstance(item, (int, float)):
                coerced_measurements.append(
                    {"name": f"measurement_{index + 1}", "value": item}
                )
                continue

            text = str(item or "").strip()
            if text:
                coerced_measurements.append(
                    {"name": f"measurement_{index + 1}", "value": text}
                )
        normalized["measurements"] = coerced_measurements

    if "statistical_analysis" in normalized:
        value = normalized.get("statistical_analysis")
        if isinstance(value, dict):
            normalized["statistical_analysis"] = [value]
        elif not isinstance(value, list):
            normalized["statistical_analysis"] = []

    return normalized


def _latest_code_execution_summary(
    progress_events: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not isinstance(progress_events, list):
        return None
    for event in reversed(progress_events):
        if not isinstance(event, dict):
            continue
        if str(event.get("tool") or "").strip() != "execute_python_job":
            continue
        summary = event.get("summary")
        if isinstance(summary, dict):
            return summary
    return None


def _latest_response_quality_metrics(
    progress_events: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not isinstance(progress_events, list):
        return None
    for event in reversed(progress_events):
        if not isinstance(event, dict):
            continue
        if str(event.get("event") or "").strip().lower() != "response_quality":
            continue
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return None


def _looks_like_contract_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    core_keys = {
        "result",
        "evidence",
        "measurements",
        "statistical_analysis",
        "confidence",
        "qc_warnings",
        "limitations",
        "next_steps",
    }
    present = sum(1 for key in core_keys if key in payload)
    return present >= 2 and "result" in payload


def _result_is_too_brief(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    word_count = len(re.findall(r"\w+", normalized))
    if word_count < 24:
        return True
    numeric_count = len(re.findall(r"\d", normalized))
    return numeric_count < 2


def _result_lacks_measurement_detail(text: str, measurements: list[MeasurementItem]) -> bool:
    if not measurements:
        return False
    normalized = sanitize_user_visible_text(text).strip().lower()
    if not normalized:
        return True

    if len(re.findall(r"\d", normalized)) >= 8:
        return False

    noisy_tokens = {
        "execute",
        "execution",
        "python",
        "job",
        "durable",
        "run",
        "runtime",
        "seconds",
        "exit",
        "code",
        "attempts",
        "kind",
    }
    for measurement in measurements[:16]:
        name = str(measurement.name or "").strip().lower()
        if not name:
            continue
        tokens = [token for token in re.split(r"[^a-z0-9]+", name) if len(token) >= 4]
        signal_tokens = [token for token in tokens if token not in noisy_tokens]
        if any(token in normalized for token in signal_tokens):
            return False
    return True


def _measurement_signal_items(measurements: list[MeasurementItem]) -> list[str]:
    noise_prefixes = ("execute_python_job.", "codegen_python_plan.")
    signals: list[str] = []
    for item in measurements[:24]:
        name = str(item.name or "").strip().lower()
        if not name:
            continue
        if any(name.startswith(prefix) for prefix in noise_prefixes):
            continue
        signals.append(name)
    return signals


def _measurement_coverage_count(text: str, measurements: list[MeasurementItem]) -> int:
    normalized = sanitize_user_visible_text(text).strip().lower()
    if not normalized:
        return 0
    covered = 0
    for name in _measurement_signal_items(measurements):
        tokens = [token for token in re.split(r"[^a-z0-9]+", name) if len(token) >= 4]
        if any(token in normalized for token in tokens):
            covered += 1
    return covered


def _response_mentions_limitations(text: str) -> bool:
    normalized = sanitize_user_visible_text(text).strip().lower()
    return bool(
        re.search(
            r"\b(limitation|limited|caveat|uncertain|uncertainty|bias|constraint)s?\b",
            normalized,
        )
    )


def _response_mentions_next_steps(text: str) -> bool:
    normalized = sanitize_user_visible_text(text).strip().lower()
    return bool(
        re.search(
            r"\b(next step|next steps|recommend|recommended|should|consider|validate|follow[- ]?up|future work)\b",
            normalized,
        )
    )


def _code_execution_response_needs_enrichment(
    text: str,
    *,
    contract: AssistantContract,
    progress_events: list[dict[str, Any]] | None,
) -> bool:
    normalized = sanitize_user_visible_text(text).strip()
    if not normalized:
        return True
    word_count = len(re.findall(r"\w+", normalized))
    signal_measurements = _measurement_signal_items(contract.measurements)
    measurement_coverage = _measurement_coverage_count(normalized, contract.measurements)
    quality = _latest_response_quality_metrics(progress_events)

    if isinstance(quality, dict):
        try:
            completeness = float(quality.get("answer_completeness") or 0.0)
        except Exception:
            completeness = 0.0
        if completeness < 0.7:
            return True

    if signal_measurements and measurement_coverage < min(2, len(signal_measurements)):
        return True

    if signal_measurements and word_count < 45:
        return True

    if contract.limitations and not _response_mentions_limitations(normalized):
        return True

    if contract.next_steps and not _response_mentions_next_steps(normalized):
        return True

    return False


def _result_is_meta_summary(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    lower = normalized.lower()
    if any(
        lower.startswith(prefix)
        for prefix in (
            "provided ",
            "presented ",
            "outlined ",
            "listed ",
            "summarized ",
            "reported ",
        )
    ) and any(
        marker in lower
        for marker in (
            "comprehensive",
            "set of",
            "covering",
            "recommended",
            "next-step",
            "next step",
        )
    ):
        return True
    return False


def _format_metric_value(value: float | int | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_code_execution_result_summary(
    *,
    existing_result: str,
    output_files: list[str],
    measurements: list[MeasurementItem],
    missing_expected_outputs: list[str],
    runtime_seconds: Any,
) -> str:
    lines: list[str] = []
    if isinstance(runtime_seconds, (int, float)):
        lines.append(
            f"Python analysis executed successfully in sandbox (runtime {float(runtime_seconds):.3f}s)."
        )
    else:
        lines.append("Python analysis executed successfully in sandbox.")

    if output_files:
        lines.append(
            "Output artifacts: "
            + ", ".join(output_files[:6])
            + (", ..." if len(output_files) > 6 else "")
            + "."
        )

    metric_bits: list[str] = []
    preferred = [
        metric
        for metric in measurements
        if any(
            token in str(metric.name or "").lower()
            for token in (
                "accuracy",
                "auc",
                "f1",
                "precision",
                "recall",
                "explained_variance",
                "variance",
                "r2",
            )
        )
    ]
    ordered_metrics = preferred if preferred else measurements
    for metric in ordered_metrics[:6]:
        name = str(metric.name or "").strip()
        if not name:
            continue
        metric_bits.append(f"{name}={_format_metric_value(metric.value)}")
    if metric_bits:
        lines.append("Key quantitative results: " + "; ".join(metric_bits) + ".")

    if missing_expected_outputs:
        lines.append(
            "Missing expected outputs: " + ", ".join(missing_expected_outputs[:6]) + "."
        )

    fallback = str(existing_result or "").strip()
    if fallback and _result_is_too_brief(fallback) is False:
        lines.append(fallback)
    return " ".join(lines).strip()


def _render_contract_user_summary(contract: AssistantContract) -> str:
    sections: list[str] = []

    result_text = sanitize_user_visible_text(contract.result).strip()
    if result_text and not _result_is_meta_summary(result_text):
        sections.append(result_text)

    metric_tokens: list[str] = []
    for measurement in contract.measurements[:8]:
        name = str(measurement.name or "").strip()
        if not name:
            continue
        value = _format_metric_value(measurement.value)
        unit = str(measurement.unit or "").strip()
        if unit:
            metric_tokens.append(f"{name}={value} {unit}")
        else:
            metric_tokens.append(f"{name}={value}")
    if metric_tokens:
        sections.append("Key measurements: " + "; ".join(metric_tokens) + ".")

    warnings = [str(item).strip() for item in contract.qc_warnings if str(item).strip()]
    if warnings:
        sections.append("QC warnings: " + " ".join(warnings[:2]))

    limitations = [str(item).strip() for item in contract.limitations if str(item).strip()]
    if limitations:
        sections.append("Limitations: " + " ".join(limitations[:2]))

    next_steps = [
        str(step.action or "").strip()
        for step in contract.next_steps
        if str(step.action or "").strip()
    ]
    if next_steps:
        sections.append("Recommended next steps: " + " ".join(next_steps[:2]))

    if not sections and result_text:
        sections.append(result_text)
    return "\n\n".join(sections).strip()


def _apply_contract_defaults(
    contract: AssistantContract,
    *,
    response_text: str = "",
    minimum_next_steps: int = 1,
    minimum_limitations: int = 1,
) -> AssistantContract:
    minimum_next_steps = max(1, int(minimum_next_steps))
    minimum_limitations = max(0, int(minimum_limitations))

    existing_limitations = _clean_strings(contract.limitations)
    if len(existing_limitations) < minimum_limitations:
        inferred_limitations = _infer_limitations_from_text(response_text)
        for limitation in inferred_limitations + _DEFAULT_LIMITATIONS:
            if limitation not in existing_limitations:
                existing_limitations.append(limitation)
            if len(existing_limitations) >= minimum_limitations:
                break
    contract.limitations = existing_limitations

    existing_actions = [step for step in contract.next_steps if str(step.action or "").strip()]
    existing_action_text = {step.action.strip().lower() for step in existing_actions}
    candidate_actions = _extract_next_step_candidates(response_text) + _DEFAULT_NEXT_STEP_ACTIONS
    for candidate in candidate_actions:
        normalized = candidate.strip().lower()
        if not normalized or normalized in existing_action_text:
            continue
        if len(existing_actions) >= minimum_next_steps:
            break
        existing_actions.append(NextStepAction(action=candidate.strip()))
        existing_action_text.add(normalized)
    while len(existing_actions) < minimum_next_steps:
        existing_actions.append(
            NextStepAction(action="Run a targeted follow-up analysis to validate and extend this result.")
        )
    contract.next_steps = existing_actions

    if not contract.confidence.why:
        contract.confidence.why = ["Confidence rationale was not supplied explicitly."]
    _sanitize_contract_user_visible_fields(contract)
    return contract


def sanitize_user_visible_text(text: str | None) -> str:
    """Strip internal execution identifiers and filesystem paths from text.

    Parameters
    ----------
    text : str or None
        Raw user-visible text candidate.

    Returns
    -------
    str
        Sanitized text with internal path/token leakage reduced.
    """
    raw = str(text or "")
    if not raw:
        return ""

    def _replace(match: re.Match[str]) -> str:
        token = str(match.group("token") or "")
        core = token.rstrip(".,;:!?)]}")
        suffix = token[len(core) :]
        if not core:
            return token
        candidate = core.strip("'\"`")
        if not _looks_internal_path(candidate):
            return token
        normalized = candidate.replace("\\", "/")
        basename = Path(normalized).name
        if not basename:
            return token
        return basename + suffix

    sanitized = _PATH_TOKEN_RE.sub(_replace, raw)
    sanitized = _INTERNAL_EXECUTION_TOKEN_RE.sub("prior analysis run", sanitized)
    return sanitized


def _looks_internal_path(value: str) -> bool:
    normalized = str(value or "").strip().replace("\\", "/").lower()
    if not normalized:
        return False
    if "://" in normalized:
        return False
    if normalized.startswith(("data:", "mailto:", "tel:")):
        return False
    if normalized.startswith(("/", "~/", "./", "../")):
        return True
    return normalized.startswith(
        (
            "data/",
            "uploads/",
            "artifacts/",
            "sessions/",
            "science/",
            "tmp/",
            "var/",
            "users/",
            "home/",
            "private/",
            "opt/",
            "srv/",
        )
    )


def _sanitize_contract_user_visible_fields(contract: AssistantContract) -> None:
    contract.result = sanitize_user_visible_text(contract.result)
    contract.qc_warnings = [sanitize_user_visible_text(item) for item in contract.qc_warnings]
    contract.limitations = [sanitize_user_visible_text(item) for item in contract.limitations]
    contract.confidence.why = [sanitize_user_visible_text(item) for item in contract.confidence.why]
    for evidence in contract.evidence:
        if evidence.summary:
            evidence.summary = sanitize_user_visible_text(evidence.summary)
        if evidence.artifact:
            evidence.artifact = sanitize_user_visible_text(evidence.artifact)
    for next_step in contract.next_steps:
        next_step.action = sanitize_user_visible_text(next_step.action)


def _clean_strings(values: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _infer_limitations_from_text(text: str) -> list[str]:
    lower = (text or "").lower()
    if any(token in lower for token in ("error", "failed", "missing", "timeout", "unavailable", "not found")):
        return ["At least one requested operation failed or was unavailable, so coverage of the analysis is incomplete."]
    return []


def _extract_next_step_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _BULLET_PREFIX_RE.match(line):
            continue
        candidate = _BULLET_PREFIX_RE.sub("", line).strip(" -")
        if len(candidate) < 12:
            continue
        lower = candidate.lower()
        if any(lower.startswith(prefix) for prefix in ("result", "confidence", "limitation", "next step", "evidence")):
            continue
        candidates.append(candidate)
    return candidates


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    # First try exact JSON.
    parsed = _loads_dict(raw)
    if parsed is not None:
        return parsed

    # Then try fenced JSON snippets.
    for match in _JSON_FENCE_RE.finditer(raw):
        candidate = match.group(1).strip()
        parsed = _loads_dict(candidate)
        if parsed is not None:
            return parsed

    # Finally, attempt the broadest object span.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        parsed = _loads_dict(raw[start : end + 1])
        if parsed is not None:
            return parsed

    return None


def _loads_dict(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except Exception:
        sanitized = _escape_json_string_control_chars(raw)
        if sanitized != raw:
            try:
                data = json.loads(sanitized)
            except Exception:
                return None
        else:
            return None
    return data if isinstance(data, dict) else None


def _escape_json_string_control_chars(raw: str) -> str:
    """
    Best-effort repair for JSON-like model output that contains unescaped control
    characters inside quoted strings (commonly newlines in `result` text).
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escaped = False

    return "".join(out)
