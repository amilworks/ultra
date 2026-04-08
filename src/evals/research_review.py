from __future__ import annotations

import re
from typing import Any


_ACTION_VERBS = {
    "run",
    "compare",
    "validate",
    "quantify",
    "measure",
    "collect",
    "upload",
    "reproduce",
    "export",
    "segment",
    "detect",
    "analyze",
}
_META_PREFIXES = (
    "provided",
    "presented",
    "outlined",
    "listed",
    "summarized",
    "reported",
    "generated",
    "executed",
    "completed",
)
_META_MARKERS = (
    "comprehensive",
    "set of",
    "taxonomy",
    "covering",
    "recommended",
    "next-step",
    "next step",
    "pipeline",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any) -> str:
    return str(value or "").strip()


def _contains_numeric_text(value: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\b", str(value or "")))


def _numeric_density(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return 0.0
    numeric_hits = len(re.findall(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", text, flags=re.IGNORECASE))
    return round((float(numeric_hits) * 100.0) / float(len(words)), 3)


def _sentences(value: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", str(value or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [str(item).strip(" -•\t") for item in parts if str(item).strip(" -•\t")]


def _is_meta_sentence(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if not any(text.startswith(prefix) for prefix in _META_PREFIXES):
        return False
    return any(marker in text for marker in _META_MARKERS)


def _meta_narration_rate(value: str) -> float:
    rows = _sentences(value)
    if not rows:
        return 1.0
    meta = sum(1 for row in rows if _is_meta_sentence(row))
    return round(float(meta) / float(len(rows)), 3)


def _next_step_actions(contract: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for item in _as_list(contract.get("next_steps")):
        if isinstance(item, dict):
            action = _string(item.get("action"))
            if action:
                actions.append(action)
        elif isinstance(item, str):
            action = _string(item)
            if action:
                actions.append(action)
    return actions


def _evidence_entries(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in _as_list(contract.get("evidence")) if isinstance(entry, dict)]


def _extract_progress_quality_metrics(response_payload: dict[str, Any]) -> dict[str, Any] | None:
    progress_events = response_payload.get("progress_events")
    if not isinstance(progress_events, list):
        return None
    for event in reversed(progress_events):
        if not isinstance(event, dict):
            continue
        if str(event.get("event") or "").strip().lower() != "response_quality":
            continue
        metrics = event.get("metrics")
        if not isinstance(metrics, dict):
            continue
        return {
            "source": "progress_event",
            "meta_narration_rate": float(metrics.get("meta_narration_rate") or 0.0),
            "answer_completeness": float(metrics.get("answer_completeness") or 0.0),
            "numeric_detail_density": float(metrics.get("numeric_detail_density") or 0.0),
            "word_count": int(metrics.get("word_count") or 0),
            "repair_attempted": bool(event.get("repair_attempted", False)),
            "repair_applied": bool(event.get("repair_applied", False)),
        }
    return None


def _heuristic_response_quality(response_payload: dict[str, Any]) -> dict[str, Any]:
    contract = _as_dict(response_payload.get("contract"))
    response_text = _string(response_payload.get("response_text"))
    result_text = _string(contract.get("result"))
    primary_text = response_text or result_text
    normalized = re.sub(r"\r\n?", "\n", primary_text).strip()
    word_count = len(re.findall(r"\b[\w'-]+\b", normalized))
    sentence_count = len(_sentences(normalized))
    meta_rate = _meta_narration_rate(normalized)
    numeric_density = _numeric_density(normalized)

    limitations = [item for item in _as_list(contract.get("limitations")) if _string(item)]
    next_steps = _next_step_actions(contract)
    measurements = [item for item in _as_list(contract.get("measurements")) if isinstance(item, dict)]

    checks = {
        "direct_answer": bool(normalized) and word_count >= 32 and meta_rate < 0.5,
        "structured_detail": sentence_count >= 3 or "\n-" in normalized or "\n1." in normalized,
        "limitations_covered": bool(limitations) or any(
            token in normalized.lower() for token in ("limitation", "caveat", "uncertain")
        ),
        "next_steps_present": bool(next_steps) or "next step" in normalized.lower(),
        "numeric_detail_present": bool(measurements) or numeric_density >= 0.45,
    }
    completeness = round(
        sum(1.0 for item in checks.values() if item) / float(max(len(checks), 1)),
        3,
    )
    return {
        "source": "heuristic",
        "meta_narration_rate": meta_rate,
        "answer_completeness": completeness,
        "numeric_detail_density": numeric_density,
        "word_count": word_count,
        "repair_attempted": False,
        "repair_applied": False,
        "checks": checks,
    }


def score_response_quality(response_payload: dict[str, Any]) -> dict[str, Any]:
    heuristic_metrics = _heuristic_response_quality(response_payload)
    progress_metrics = _extract_progress_quality_metrics(response_payload)
    if not isinstance(progress_metrics, dict):
        return heuristic_metrics

    merged = dict(progress_metrics)
    overrides_applied = False

    heuristic_meta = float(heuristic_metrics.get("meta_narration_rate") or 0.0)
    progress_meta = float(progress_metrics.get("meta_narration_rate") or 0.0)
    if heuristic_meta < progress_meta:
        merged["meta_narration_rate"] = heuristic_meta
        overrides_applied = True

    heuristic_completeness = float(heuristic_metrics.get("answer_completeness") or 0.0)
    progress_completeness = float(progress_metrics.get("answer_completeness") or 0.0)
    if heuristic_completeness > progress_completeness:
        merged["answer_completeness"] = heuristic_completeness
        overrides_applied = True

    heuristic_density = float(heuristic_metrics.get("numeric_detail_density") or 0.0)
    progress_density = float(progress_metrics.get("numeric_detail_density") or 0.0)
    if heuristic_density > progress_density:
        merged["numeric_detail_density"] = heuristic_density
        overrides_applied = True

    merged["word_count"] = max(
        int(progress_metrics.get("word_count") or 0),
        int(heuristic_metrics.get("word_count") or 0),
    )
    if overrides_applied:
        merged["source"] = "progress_event+heuristic_final"
        merged["heuristic_checks"] = heuristic_metrics.get("checks")
    return merged


def audit_contract_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
    contract = _as_dict(response_payload.get("contract"))
    response_text = _string(response_payload.get("response_text"))
    result_text = _string(contract.get("result"))
    confidence = _as_dict(contract.get("confidence"))
    confidence_level = _string(confidence.get("level")).lower()
    confidence_why = [item for item in _as_list(confidence.get("why")) if _string(item)]

    evidence = _evidence_entries(contract)
    measurements = [item for item in _as_list(contract.get("measurements")) if isinstance(item, dict)]
    limitations = [item for item in _as_list(contract.get("limitations")) if _string(item)]
    next_steps = _next_step_actions(contract)

    checks: dict[str, bool] = {
        "response_text_non_empty": bool(response_text),
        "contract_result_non_empty": bool(result_text),
        "confidence_level_valid": confidence_level in {"low", "medium", "high"},
        "confidence_why_present": bool(confidence_why),
        "evidence_present": bool(evidence),
        "limitations_present": bool(limitations),
        "next_steps_present": bool(next_steps),
    }

    missing_fields: list[str] = []
    if not checks["response_text_non_empty"]:
        missing_fields.append("response_text")
    if not checks["contract_result_non_empty"]:
        missing_fields.append("contract.result")
    if not checks["confidence_level_valid"]:
        missing_fields.append("contract.confidence.level")
    if not checks["confidence_why_present"]:
        missing_fields.append("contract.confidence.why")
    if not checks["evidence_present"]:
        missing_fields.append("contract.evidence")
    if not checks["limitations_present"]:
        missing_fields.append("contract.limitations")
    if not checks["next_steps_present"]:
        missing_fields.append("contract.next_steps")

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_fields": missing_fields,
        "evidence_count": len(evidence),
        "measurement_count": len(measurements),
        "limitation_count": len(limitations),
        "next_step_count": len(next_steps),
        "confidence_level": confidence_level or None,
        "confidence_why_count": len(confidence_why),
    }


def score_research_value(response_payload: dict[str, Any]) -> dict[str, Any]:
    contract = _as_dict(response_payload.get("contract"))
    response_text = _string(response_payload.get("response_text"))
    result_text = _string(contract.get("result"))
    combined_text = f"{response_text}\n{result_text}"
    combined_text_lower = combined_text.lower()

    evidence = _evidence_entries(contract)
    measurements = [item for item in _as_list(contract.get("measurements")) if isinstance(item, dict)]
    limitations = [item for item in _as_list(contract.get("limitations")) if _string(item)]
    next_steps = _next_step_actions(contract)
    confidence = _as_dict(contract.get("confidence"))
    confidence_why = [item for item in _as_list(confidence.get("why")) if _string(item)]

    reproducibility = 0
    if evidence:
        reproducibility += 1
    if any(token in combined_text_lower for token in ("repro", "run id", "artifact", "hash", "version")):
        reproducibility += 1

    actionability = 0
    if next_steps:
        actionability += 1
    if any(_string(step).split(" ", 1)[0].lower() in _ACTION_VERBS for step in next_steps if _string(step)):
        actionability += 1

    traceability = 0
    if any(_string(item.get("run_id")) or _string(item.get("artifact")) for item in evidence):
        traceability += 1
    if measurements:
        traceability += 1

    uncertainty = 0
    if limitations:
        uncertainty += 1
    if confidence_why:
        uncertainty += 1

    scientific_specificity = 0
    if measurements:
        scientific_specificity += 1
    if _contains_numeric_text(combined_text):
        scientific_specificity += 1

    total = reproducibility + actionability + traceability + uncertainty + scientific_specificity
    max_total = 10
    recommendations: list[str] = []
    if reproducibility < 2:
        recommendations.append("Add reproducibility anchors (run IDs, artifacts, or version/hash notes).")
    if actionability < 2:
        recommendations.append("Return concrete next-step actions with clear verbs and follow-up intent.")
    if traceability < 2:
        recommendations.append("Include stronger evidence-to-measurement traceability in the final contract.")
    if uncertainty < 2:
        recommendations.append("State uncertainty with explicit confidence rationale and study limitations.")
    if scientific_specificity < 2:
        recommendations.append("Increase scientific specificity with quantitative metrics and numeric context.")

    if total >= 8:
        summary = "High research value: actionable, traceable, and quantitatively grounded."
    elif total >= 6:
        summary = "Moderate research value: useful output with room to improve reproducibility details."
    else:
        summary = "Low research value: response needs stronger evidence, actionability, and uncertainty framing."

    return {
        "score": int(total),
        "max_score": int(max_total),
        "summary": summary,
        "recommendations": recommendations,
        "dimensions": {
            "reproducibility": reproducibility,
            "actionability": actionability,
            "traceability": traceability,
            "uncertainty": uncertainty,
            "scientific_specificity": scientific_specificity,
        },
    }
