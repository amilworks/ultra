from __future__ import annotations

from collections import defaultdict
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _model_group_key(row: dict[str, Any]) -> tuple[str, str]:
    model_key = str(row.get("model_key") or "").strip().lower() or "unknown"
    model_version = str(row.get("model_version") or "").strip() or "unversioned"
    return model_key, model_version


def compute_model_health_entries(
    *,
    training_jobs: list[dict[str, Any]],
    inference_jobs: list[dict[str, Any]],
    validation_drop_watch: float = 0.10,
    drift_retrain_threshold: float = 0.35,
    min_reviewed_samples: int = 10,
) -> list[dict[str, Any]]:
    by_group_training: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_group_inference: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in training_jobs:
        if str(row.get("status") or "").strip().lower() != "succeeded":
            continue
        by_group_training[_model_group_key(row)].append(row)

    for row in inference_jobs:
        if str(row.get("status") or "").strip().lower() not in {"succeeded", "failed"}:
            continue
        by_group_inference[_model_group_key(row)].append(row)

    grouped_keys = sorted(set(by_group_training.keys()) | set(by_group_inference.keys()))
    output: list[dict[str, Any]] = []
    for group_key in grouped_keys:
        model_key, model_version = group_key
        train_rows = sorted(
            by_group_training.get(group_key, []),
            key=lambda row: str(row.get("updated_at") or row.get("finished_at") or ""),
        )
        infer_rows = sorted(
            by_group_inference.get(group_key, []),
            key=lambda row: str(row.get("updated_at") or row.get("finished_at") or ""),
        )

        validation_scores: list[float] = []
        for row in train_rows:
            result = row.get("result")
            result_dict = result if isinstance(result, dict) else {}
            metrics = result_dict.get("metrics")
            metrics_dict = metrics if isinstance(metrics, dict) else {}
            value = _to_float(
                metrics_dict.get("map50_candidate_eval")
                or metrics_dict.get("map50")
                or metrics_dict.get("best_val_dice")
                or metrics_dict.get("val_dice")
            )
            if value is not None:
                validation_scores.append(value)

        latest_validation = validation_scores[-1] if validation_scores else None
        baseline_validation = max(validation_scores) if validation_scores else None
        validation_drop_ratio = 0.0
        if (
            latest_validation is not None
            and baseline_validation is not None
            and baseline_validation > 0
        ):
            validation_drop_ratio = max(
                0.0, min(1.0, (baseline_validation - latest_validation) / baseline_validation)
            )

        drift_scores: list[float] = []
        reviewed_samples = 0
        reviewed_failures = 0
        for row in infer_rows:
            result = row.get("result")
            result_dict = result if isinstance(result, dict) else {}
            drift = _to_float(result_dict.get("drift_score_mean"))
            if drift is not None:
                drift_scores.append(drift)
            reviewed_samples += int(result_dict.get("reviewed_samples") or 0)
            reviewed_failures += int(result_dict.get("reviewed_failures") or 0)

        drift_score_mean = (
            sum(drift_scores) / float(len(drift_scores)) if drift_scores else 0.0
        )
        reviewed_failure_rate = (
            float(reviewed_failures) / float(reviewed_samples)
            if reviewed_samples > 0
            else 0.0
        )

        state = "Healthy"
        rationale: list[str] = []
        recommendation = "Continue monitoring."

        if reviewed_samples < min_reviewed_samples:
            state = "Needs Human Review"
            rationale.append(
                f"Only {reviewed_samples} reviewed inference sample(s); "
                f"minimum required is {min_reviewed_samples}."
            )
            recommendation = "Collect more reviewed inference samples."
        elif (
            validation_drop_ratio >= validation_drop_watch
            and drift_score_mean >= drift_retrain_threshold
        ):
            state = "Retrain Recommended"
            rationale.append(
                f"Validation drop is {validation_drop_ratio * 100:.1f}% "
                f"and drift mean is {drift_score_mean:.3f}."
            )
            recommendation = "Launch retraining or finetune with refreshed data."
        elif validation_drop_ratio >= validation_drop_watch or drift_score_mean >= drift_retrain_threshold:
            state = "Watch"
            rationale.append(
                f"Validation drop={validation_drop_ratio * 100:.1f}%, "
                f"drift mean={drift_score_mean:.3f}."
            )
            recommendation = "Monitor closely and prepare a finetune run."
        else:
            rationale.append(
                f"Validation drop={validation_drop_ratio * 100:.1f}%, "
                f"drift mean={drift_score_mean:.3f}, "
                f"reviewed failure rate={reviewed_failure_rate * 100:.1f}%."
            )

        output.append(
            {
                "model_key": model_key,
                "model_version": model_version,
                "status": state,
                "recommendation": recommendation,
                "rationale": rationale,
                "metrics": {
                    "latest_validation_score": latest_validation,
                    "baseline_validation_score": baseline_validation,
                    "validation_drop_ratio": validation_drop_ratio,
                    "drift_score_mean": drift_score_mean,
                    "reviewed_samples": reviewed_samples,
                    "reviewed_failures": reviewed_failures,
                    "reviewed_failure_rate": reviewed_failure_rate,
                },
                "training_runs": len(train_rows),
                "inference_runs": len(infer_rows),
            }
        )

    output.sort(key=lambda row: (str(row.get("model_key")), str(row.get("model_version"))))
    return output
