from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any

PROPOSAL_STATUS_VALUES = {
    "pending_approval",
    "approved",
    "running",
    "evaluating",
    "ready_to_promote",
    "promoted",
    "rejected",
    "failed",
}

VERSION_STATUS_VALUES = {
    "candidate",
    "canary",
    "active",
    "retired",
}

MERGE_STATUS_VALUES = {
    "open",
    "evaluating",
    "approved",
    "rejected",
    "executed",
    "failed",
}


@dataclass(frozen=True)
class ContinuousLearningPolicy:
    trigger_interval_hours: int = 6
    data_threshold_samples: int = 25
    schedule_samples: int = 10
    health_samples: int = 10
    min_days_between_updates: int = 3
    class_min_samples: int = 3
    replay_new_ratio: float = 0.60
    replay_old_ratio: float = 0.40
    l2sp_lambda: float = 1e-4
    legacy_drop_cap: float = 0.02
    worst_class_drop_cap: float = 0.05
    new_holdout_gain_min: float = 0.01
    drift_reduction_min: float = 0.10
    canonical_map50_drop_max: float = 0.02
    prairie_dog_map50_drop_max: float = 0.03
    active_map50_drop_max: float = 0.02
    canonical_fp_image_increase_max: float = 0.25


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed != parsed:
        return default
    return parsed


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_datetime(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None


def normalize_proposal_status(value: Any, *, default: str = "pending_approval") -> str:
    token = str(value or "").strip().lower()
    if token in PROPOSAL_STATUS_VALUES:
        return token
    return default


def normalize_version_status(value: Any, *, default: str = "candidate") -> str:
    token = str(value or "").strip().lower()
    if token in VERSION_STATUS_VALUES:
        return token
    return default


def normalize_merge_status(value: Any, *, default: str = "open") -> str:
    token = str(value or "").strip().lower()
    if token in MERGE_STATUS_VALUES:
        return token
    return default


def proposal_transition_allowed(current: str, target: str) -> bool:
    current_token = normalize_proposal_status(current)
    target_token = normalize_proposal_status(target)
    transitions: dict[str, set[str]] = {
        "pending_approval": {"approved", "rejected", "failed"},
        "approved": {"running", "rejected", "failed"},
        "running": {"evaluating", "failed"},
        "evaluating": {"ready_to_promote", "rejected", "failed"},
        "ready_to_promote": {"promoted", "rejected", "failed"},
        "promoted": set(),
        "rejected": set(),
        "failed": set(),
    }
    return target_token in transitions.get(current_token, set())


def version_transition_allowed(current: str, target: str) -> bool:
    current_token = normalize_version_status(current)
    target_token = normalize_version_status(target)
    transitions: dict[str, set[str]] = {
        "candidate": {"canary", "active", "retired"},
        "canary": {"active", "retired"},
        "active": {"retired"},
        "retired": set(),
    }
    return target_token in transitions.get(current_token, set())


def merge_transition_allowed(current: str, target: str) -> bool:
    current_token = normalize_merge_status(current)
    target_token = normalize_merge_status(target)
    transitions: dict[str, set[str]] = {
        "open": {"evaluating", "approved", "rejected", "failed"},
        "evaluating": {"approved", "rejected", "failed"},
        "approved": {"executed", "failed"},
        "rejected": set(),
        "executed": set(),
        "failed": set(),
    }
    return target_token in transitions.get(current_token, set())


def evaluate_trigger_policy(
    *,
    approved_new_samples: int,
    class_counts: dict[str, Any] | None,
    health_status: str | None,
    last_promoted_at: str | None,
    now: datetime | None = None,
    policy: ContinuousLearningPolicy | None = None,
) -> dict[str, Any]:
    current_policy = policy or ContinuousLearningPolicy()
    now_dt = now or datetime.utcnow()
    class_counts_dict = class_counts if isinstance(class_counts, dict) else {}
    class_min = max(1, int(current_policy.class_min_samples))
    class_coverage_ok = True
    class_coverage_issues: list[str] = []
    for raw_key, raw_count in sorted(class_counts_dict.items()):
        label = str(raw_key or "").strip() or "class"
        count = _to_int(raw_count, default=0)
        if count > 0 and count < class_min:
            class_coverage_ok = False
            class_coverage_issues.append(f"{label}:{count}")

    promoted_at = parse_datetime(last_promoted_at)
    days_since_promote = (
        (now_dt - promoted_at).total_seconds() / 86400.0 if promoted_at is not None else 10_000.0
    )
    min_spacing_ok = days_since_promote >= float(max(0, current_policy.min_days_between_updates))

    reason = "none"
    should_trigger = False
    if (
        approved_new_samples >= int(current_policy.data_threshold_samples)
        and class_coverage_ok
        and min_spacing_ok
    ):
        reason = "data_threshold"
        should_trigger = True
    elif (
        approved_new_samples >= int(current_policy.schedule_samples)
        and days_since_promote >= 7.0
        and class_coverage_ok
    ):
        reason = "schedule"
        should_trigger = True
    elif (
        str(health_status or "").strip() == "Retrain Recommended"
        and approved_new_samples >= int(current_policy.health_samples)
        and class_coverage_ok
    ):
        reason = "health"
        should_trigger = True

    return {
        "should_trigger": should_trigger,
        "reason": reason,
        "approved_new_samples": int(approved_new_samples),
        "class_coverage_ok": class_coverage_ok,
        "class_coverage_issues": class_coverage_issues,
        "days_since_promote": round(days_since_promote, 3),
        "min_spacing_days": int(current_policy.min_days_between_updates),
        "policy": {
            "trigger_interval_hours": int(current_policy.trigger_interval_hours),
            "data_threshold_samples": int(current_policy.data_threshold_samples),
            "schedule_samples": int(current_policy.schedule_samples),
            "health_samples": int(current_policy.health_samples),
            "class_min_samples": class_min,
        },
        "checked_at": now_dt.isoformat(),
    }


def deterministic_replay_sample(
    *,
    lineage_id: str,
    items: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    clamped_limit = max(0, int(limit))
    if clamped_limit <= 0 or not items:
        return []

    weighted: list[tuple[float, dict[str, Any]]] = []
    for raw in items:
        entry = raw if isinstance(raw, dict) else {}
        seed_parts = [
            str(lineage_id or "").strip(),
            str(entry.get("file_id") or "").strip(),
            str(entry.get("sample_id") or "").strip(),
        ]
        digest = sha256("|".join(seed_parts).encode("utf-8", errors="ignore")).hexdigest()
        hashed = int(digest[:12], 16) / float(16**12)
        pinned_bonus = 2.0 if bool(entry.get("pinned")) else 0.0
        weight = max(0.0, _to_float(entry.get("weight"), default=1.0))
        score = pinned_bonus + weight + hashed * 0.01
        weighted.append((score, entry))
    weighted.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in weighted[:clamped_limit]]


def build_replay_mix_plan(
    *,
    lineage_id: str,
    new_samples: int,
    replay_items: list[dict[str, Any]],
    policy: ContinuousLearningPolicy | None = None,
) -> dict[str, Any]:
    current_policy = policy or ContinuousLearningPolicy()
    new_count = max(0, int(new_samples))
    replay_ratio = max(0.0, min(1.0, float(current_policy.replay_old_ratio)))
    if new_count <= 0:
        requested_replay = min(len(replay_items), 32)
    else:
        requested_replay = int(round((new_count * replay_ratio) / max(1e-9, 1.0 - replay_ratio)))
    sampled_replay = deterministic_replay_sample(
        lineage_id=lineage_id,
        items=replay_items,
        limit=max(0, requested_replay),
    )
    return {
        "new_samples": new_count,
        "replay_samples": len(sampled_replay),
        "requested_replay_samples": max(0, requested_replay),
        "new_ratio_target": float(current_policy.replay_new_ratio),
        "replay_ratio_target": float(current_policy.replay_old_ratio),
        "l2sp_lambda": float(current_policy.l2sp_lambda),
        "sampled_replay_items": sampled_replay,
    }


def evaluate_promotion_guardrails(
    *,
    active_metrics: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None,
    policy: ContinuousLearningPolicy | None = None,
) -> dict[str, Any]:
    current_policy = policy or ContinuousLearningPolicy()
    active = active_metrics if isinstance(active_metrics, dict) else {}
    candidate = candidate_metrics if isinstance(candidate_metrics, dict) else {}

    benchmark = candidate.get("benchmark") if isinstance(candidate.get("benchmark"), dict) else {}
    if benchmark:
        baseline_before_train = (
            benchmark.get("baseline_before_train")
            if isinstance(benchmark.get("baseline_before_train"), dict)
            else {}
        )
        candidate_after_train = (
            benchmark.get("candidate_after_train")
            if isinstance(benchmark.get("candidate_after_train"), dict)
            else {}
        )
        baseline_canonical = (
            baseline_before_train.get("canonical")
            if isinstance(baseline_before_train.get("canonical"), dict)
            else {}
        )
        candidate_canonical = (
            candidate_after_train.get("canonical")
            if isinstance(candidate_after_train.get("canonical"), dict)
            else {}
        )
        baseline_active = (
            baseline_before_train.get("active")
            if isinstance(baseline_before_train.get("active"), dict)
            else {}
        )
        candidate_active = (
            candidate_after_train.get("active")
            if isinstance(candidate_after_train.get("active"), dict)
            else {}
        )

        def _metric(row: dict[str, Any], key: str) -> float | None:
            value = row.get(key)
            if isinstance(value, (float, int)):
                return float(value)
            return None

        def _class_map50(row: dict[str, Any], class_name: str) -> float | None:
            per_class = row.get("per_class")
            if not isinstance(per_class, dict):
                return None
            class_row = per_class.get(class_name)
            if not isinstance(class_row, dict):
                return None
            return _metric(class_row, "map50")

        canonical_baseline_map50 = _metric(baseline_canonical, "map50")
        canonical_candidate_map50 = _metric(candidate_canonical, "map50")
        canonical_baseline_fp = _metric(baseline_canonical, "fp_per_image")
        canonical_candidate_fp = _metric(candidate_canonical, "fp_per_image")
        prairie_baseline_map50 = _class_map50(baseline_canonical, "prairie_dog")
        prairie_candidate_map50 = _class_map50(candidate_canonical, "prairie_dog")
        active_baseline_map50 = _metric(baseline_active, "map50")
        active_candidate_map50 = _metric(candidate_active, "map50")

        complete = all(
            value is not None
            for value in (
                canonical_baseline_map50,
                canonical_candidate_map50,
                canonical_baseline_fp,
                canonical_candidate_fp,
                prairie_baseline_map50,
                prairie_candidate_map50,
                active_baseline_map50,
                active_candidate_map50,
            )
        )
        reasons: list[str] = []
        if not complete:
            reasons.append("Benchmark packet is incomplete (canonical/active metrics missing).")
            return {
                "passed": False,
                "reasons": reasons,
                "metrics": {
                    "mode": "detection_benchmark",
                    "benchmark_ready": False,
                },
            }

        canonical_drop = max(
            0.0, float(canonical_baseline_map50) - float(canonical_candidate_map50)
        )
        prairie_dog_drop = max(0.0, float(prairie_baseline_map50) - float(prairie_candidate_map50))
        active_drop = max(0.0, float(active_baseline_map50) - float(active_candidate_map50))
        baseline_fp = float(canonical_baseline_fp)
        candidate_fp = float(canonical_candidate_fp)
        fp_increase_ratio = (
            max(0.0, (candidate_fp - baseline_fp) / baseline_fp) if baseline_fp > 0.0 else 0.0
        )

        canonical_ok = canonical_drop <= float(current_policy.canonical_map50_drop_max)
        prairie_ok = prairie_dog_drop <= float(current_policy.prairie_dog_map50_drop_max)
        active_ok = active_drop <= float(current_policy.active_map50_drop_max)
        fp_ok = fp_increase_ratio <= float(current_policy.canonical_fp_image_increase_max)
        passed = bool(canonical_ok and prairie_ok and active_ok and fp_ok)
        if not canonical_ok:
            reasons.append(
                f"Canonical mAP50 drop {canonical_drop:.4f} exceeds cap "
                f"{current_policy.canonical_map50_drop_max:.4f}."
            )
        if not prairie_ok:
            reasons.append(
                f"prairie_dog canonical mAP50 drop {prairie_dog_drop:.4f} exceeds cap "
                f"{current_policy.prairie_dog_map50_drop_max:.4f}."
            )
        if not active_ok:
            reasons.append(
                f"Active-holdout mAP50 drop {active_drop:.4f} exceeds cap "
                f"{current_policy.active_map50_drop_max:.4f}."
            )
        if not fp_ok:
            reasons.append(
                f"Canonical FP/image increase {fp_increase_ratio:.4f} exceeds cap "
                f"{current_policy.canonical_fp_image_increase_max:.4f}."
            )
        return {
            "passed": passed,
            "reasons": reasons,
            "metrics": {
                "mode": "detection_benchmark",
                "benchmark_ready": True,
                "canonical_map50_drop": canonical_drop,
                "prairie_dog_map50_drop": prairie_dog_drop,
                "active_map50_drop": active_drop,
                "canonical_fp_image_increase_ratio": fp_increase_ratio,
                "thresholds": {
                    "canonical_map50_drop_max": float(current_policy.canonical_map50_drop_max),
                    "prairie_dog_map50_drop_max": float(current_policy.prairie_dog_map50_drop_max),
                    "active_map50_drop_max": float(current_policy.active_map50_drop_max),
                    "canonical_fp_image_increase_max": float(
                        current_policy.canonical_fp_image_increase_max
                    ),
                },
                "canonical_baseline_map50": canonical_baseline_map50,
                "canonical_candidate_map50": canonical_candidate_map50,
                "active_baseline_map50": active_baseline_map50,
                "active_candidate_map50": active_candidate_map50,
            },
        }

    active_new = _to_float(active.get("dice_new_holdout"), default=0.0)
    active_legacy = _to_float(active.get("dice_legacy_holdout"), default=0.0)
    active_worst = _to_float(active.get("worst_class_legacy_dice"), default=0.0)
    active_drift = _to_float(active.get("drift_score_mean"), default=0.0)

    candidate_new = _to_float(candidate.get("dice_new_holdout"), default=0.0)
    candidate_legacy = _to_float(candidate.get("dice_legacy_holdout"), default=0.0)
    candidate_worst = _to_float(candidate.get("worst_class_legacy_dice"), default=0.0)
    candidate_drift = _to_float(candidate.get("drift_score_mean"), default=0.0)
    active_map50_raw = (
        active.get("map50_candidate_eval")
        if active.get("map50_candidate_eval") is not None
        else active.get("map50")
    )
    candidate_map50_raw = (
        candidate.get("map50_candidate_eval")
        if candidate.get("map50_candidate_eval") is not None
        else candidate.get("map50")
    )
    baseline_map50_raw = (
        candidate.get("map50_baseline")
        if candidate.get("map50_baseline") is not None
        else active.get("map50_baseline")
    )
    active_map50 = _to_float(active_map50_raw, default=0.0)
    candidate_map50 = _to_float(candidate_map50_raw, default=0.0)
    baseline_map50 = _to_float(baseline_map50_raw, default=0.0)
    active_map_available = active_map50_raw is not None and active_map50 > 0.0
    candidate_map_available = candidate_map50_raw is not None and candidate_map50 > 0.0
    baseline_map_available = baseline_map50_raw is not None and baseline_map50 > 0.0

    legacy_drop = max(0.0, active_legacy - candidate_legacy)
    worst_drop = max(0.0, active_worst - candidate_worst)
    new_gain = candidate_new - active_new
    drift_reduction = max(0.0, active_drift - candidate_drift)
    drift_reduction_ratio = drift_reduction / max(1e-9, active_drift) if active_drift > 0 else 0.0
    map50_drop_from_active = max(0.0, active_map50 - candidate_map50)
    map50_drop_from_baseline = max(0.0, baseline_map50 - candidate_map50)
    map50_gain_vs_active = candidate_map50 - active_map50
    map50_gain_vs_baseline = candidate_map50 - baseline_map50

    legacy_ok = legacy_drop <= float(current_policy.legacy_drop_cap)
    worst_ok = worst_drop <= float(current_policy.worst_class_drop_cap)
    progress_ok = new_gain >= float(
        current_policy.new_holdout_gain_min
    ) or drift_reduction_ratio >= float(current_policy.drift_reduction_min)
    map_non_regression_ok = (
        True
        if not active_map_available
        else map50_drop_from_active <= float(current_policy.legacy_drop_cap)
    )
    baseline_floor_ok = True if not baseline_map_available else map50_drop_from_baseline <= 0.01

    detection_mode = bool(active_map_available or candidate_map_available or baseline_map_available)
    if detection_mode:
        progress_ok = (
            (
                active_map_available
                and map50_gain_vs_active >= float(current_policy.new_holdout_gain_min)
            )
            or (
                baseline_map_available
                and map50_gain_vs_baseline >= float(current_policy.new_holdout_gain_min)
            )
            or drift_reduction_ratio >= float(current_policy.drift_reduction_min)
        )
        passed = bool(map_non_regression_ok and baseline_floor_ok and progress_ok)
    else:
        passed = bool(legacy_ok and worst_ok and progress_ok)
    reasons: list[str] = []
    if detection_mode:
        if not map_non_regression_ok:
            reasons.append(
                f"mAP50 drop vs active is {map50_drop_from_active * 100:.2f}%, "
                f"exceeds cap {current_policy.legacy_drop_cap * 100:.2f}%."
            )
        if not baseline_floor_ok:
            reasons.append(
                f"mAP50 drop vs baseline is {map50_drop_from_baseline * 100:.2f}%, "
                "exceeds floor cap 1.00%."
            )
        if not progress_ok:
            reasons.append(
                "Candidate does not meet mAP improvement gate: "
                f"gain_vs_active={map50_gain_vs_active * 100:.2f}%, "
                f"gain_vs_baseline={map50_gain_vs_baseline * 100:.2f}%, "
                f"drift reduction={drift_reduction_ratio * 100:.2f}%."
            )
    else:
        if not legacy_ok:
            reasons.append(
                f"Legacy holdout drop {legacy_drop * 100:.2f}% exceeds cap "
                f"{current_policy.legacy_drop_cap * 100:.2f}%."
            )
        if not worst_ok:
            reasons.append(
                f"Worst-class legacy drop {worst_drop * 100:.2f}% exceeds cap "
                f"{current_policy.worst_class_drop_cap * 100:.2f}%."
            )
        if not progress_ok:
            reasons.append(
                "Candidate does not meet improvement gate: "
                f"new gain={new_gain * 100:.2f}%, drift reduction={drift_reduction_ratio * 100:.2f}%."
            )

    composite_score = (0.6 * candidate_new) + (0.4 * candidate_legacy)
    return {
        "passed": passed,
        "reasons": reasons,
        "metrics": {
            "mode": "detection" if detection_mode else "segmentation",
            "active_new_holdout": active_new,
            "candidate_new_holdout": candidate_new,
            "active_legacy_holdout": active_legacy,
            "candidate_legacy_holdout": candidate_legacy,
            "active_worst_class_legacy": active_worst,
            "candidate_worst_class_legacy": candidate_worst,
            "active_map50": active_map50,
            "candidate_map50": candidate_map50,
            "baseline_map50": baseline_map50,
            "active_map50_available": active_map_available,
            "candidate_map50_available": candidate_map_available,
            "baseline_map50_available": baseline_map_available,
            "map50_drop_from_active": map50_drop_from_active,
            "map50_drop_from_baseline": map50_drop_from_baseline,
            "map50_gain_vs_active": map50_gain_vs_active,
            "map50_gain_vs_baseline": map50_gain_vs_baseline,
            "legacy_drop": legacy_drop,
            "worst_class_drop": worst_drop,
            "new_holdout_gain": new_gain,
            "active_drift_score": active_drift,
            "candidate_drift_score": candidate_drift,
            "drift_reduction": drift_reduction,
            "drift_reduction_ratio": drift_reduction_ratio,
            "composite_score": composite_score,
        },
        "policy": {
            "legacy_drop_cap": float(current_policy.legacy_drop_cap),
            "worst_class_drop_cap": float(current_policy.worst_class_drop_cap),
            "new_holdout_gain_min": float(current_policy.new_holdout_gain_min),
            "drift_reduction_min": float(current_policy.drift_reduction_min),
        },
    }


def next_trigger_check_at(
    *, checked_at: datetime | None = None, policy: ContinuousLearningPolicy | None = None
) -> str:
    now_dt = checked_at or datetime.utcnow()
    current_policy = policy or ContinuousLearningPolicy()
    next_dt = now_dt + timedelta(hours=max(1, int(current_policy.trigger_interval_hours)))
    return next_dt.isoformat()
