"""Curated statistical analysis utilities for scientific workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable

import numpy as np


def _as_float_list(values: list[Any] | tuple[Any, ...] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    cleaned = [float(v) for v in arr if np.isfinite(v)]
    if not cleaned:
        raise ValueError("No finite numeric values were provided.")
    return cleaned


def summary_statistics(values: list[Any] | tuple[Any, ...] | np.ndarray) -> dict[str, Any]:
    """Compute descriptive statistics for a numeric sample.

    Parameters
    ----------
    values : list[Any] or tuple[Any, ...] or numpy.ndarray
        Numeric observations. Non-finite values are removed before computation.

    Returns
    -------
    dict[str, Any]
        Dictionary with sample size, central tendency, spread, and quantile
        metrics.

    Notes
    -----
    Use this as baseline context before inferential tests.
    """
    vals = np.asarray(_as_float_list(values), dtype=float)
    n = int(vals.size)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(vals, [25, 75]) if n > 1 else (vals[0], vals[0])
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(vals)),
        "std": std,
        "variance": float(std * std),
        "sem": float(std / math.sqrt(n)) if n > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _z_critical(confidence: float) -> float:
    alpha = max(1e-9, 1.0 - float(confidence))
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def confidence_interval_mean(
    values: list[Any] | tuple[Any, ...] | np.ndarray,
    confidence: float = 0.95,
    method: str = "auto",
) -> dict[str, Any]:
    """Estimate a confidence interval for the sample mean.

    Parameters
    ----------
    values : list[Any] or tuple[Any, ...] or numpy.ndarray
        Numeric observations.
    confidence : float, default=0.95
        Requested confidence level.
    method : {"auto", "t", "normal"}, default="auto"
        Interval method selector. `"auto"` prefers the t distribution when
        SciPy is available and otherwise falls back to normal.

    Returns
    -------
    dict[str, Any]
        Confidence interval payload with method metadata and CI bounds.

    Notes
    -----
    Very small samples return a degenerate interval anchored to the sample mean.
    """
    vals = np.asarray(_as_float_list(values), dtype=float)
    n = int(vals.size)
    mean = float(np.mean(vals))
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "confidence": confidence,
            "method": "degenerate",
            "ci_low": mean,
            "ci_high": mean,
        }

    std = float(np.std(vals, ddof=1))
    sem = std / math.sqrt(n)
    chosen = method
    critical = None
    if method in {"auto", "t"}:
        try:
            from scipy import stats as scipy_stats  # type: ignore

            critical = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1))
            chosen = "t"
        except Exception:
            if method == "t":
                raise ValueError("SciPy is required for method='t'.")
            chosen = "normal"

    if critical is None:
        critical = _z_critical(confidence)
        chosen = "normal"

    half = critical * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "confidence": confidence,
        "method": chosen,
        "ci_low": float(mean - half),
        "ci_high": float(mean + half),
    }


def bootstrap_confidence_interval(
    values: list[Any] | tuple[Any, ...] | np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute a bootstrap confidence interval for mean, median, or std.

    Parameters
    ----------
    values : list[Any] or tuple[Any, ...] or numpy.ndarray
        Numeric observations.
    statistic : {"mean", "median", "std"}, default="mean"
        Statistic to bootstrap.
    confidence : float, default=0.95
        Requested confidence level.
    n_resamples : int, default=2000
        Number of bootstrap samples. The implementation clamps this to a safe
        runtime range.
    random_seed : int, default=42
        Seed used for deterministic resampling.

    Returns
    -------
    dict[str, Any]
        Bootstrap estimate payload with CI bounds and run metadata.
    """
    vals = np.asarray(_as_float_list(values), dtype=float)
    n = int(vals.size)
    if n < 2:
        x = float(vals[0])
        return {
            "n": n,
            "statistic": statistic,
            "confidence": confidence,
            "n_resamples": 0,
            "estimate": x,
            "ci_low": x,
            "ci_high": x,
        }

    n_resamples = max(100, min(int(n_resamples), 20000))
    rng = np.random.default_rng(int(random_seed))
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = vals[idx]

    if statistic == "median":
        estimates = np.median(samples, axis=1)
        estimate = float(np.median(vals))
    elif statistic == "std":
        estimates = np.std(samples, axis=1, ddof=1)
        estimate = float(np.std(vals, ddof=1))
    else:
        estimates = np.mean(samples, axis=1)
        estimate = float(np.mean(vals))

    alpha = 1.0 - float(confidence)
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": n,
        "statistic": statistic,
        "confidence": confidence,
        "n_resamples": n_resamples,
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def cohen_d(group_a: list[Any], group_b: list[Any]) -> float:
    """Return Cohen's d standardized mean difference (`group_b - group_a`).

    Parameters
    ----------
    group_a : list[Any]
        Baseline group values.
    group_b : list[Any]
        Comparison group values.

    Returns
    -------
    float
        Standardized mean difference using pooled variance.
    """
    a = np.asarray(_as_float_list(group_a), dtype=float)
    b = np.asarray(_as_float_list(group_b), dtype=float)
    n1 = len(a)
    n2 = len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = float(np.var(a, ddof=1))
    s2 = float(np.var(b, ddof=1))
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / math.sqrt(pooled))


def cliffs_delta(
    group_a: list[Any],
    group_b: list[Any],
    max_samples_per_group: int = 2000,
    random_seed: int = 42,
) -> float:
    """Estimate Cliff's delta effect size between two groups.

    Parameters
    ----------
    group_a : list[Any]
        Baseline group values.
    group_b : list[Any]
        Comparison group values.
    max_samples_per_group : int, default=2000
        Maximum observations sampled from each group before pairwise
        comparison.
    random_seed : int, default=42
        Seed used when subsampling large groups.

    Returns
    -------
    float
        Cliff's delta in the range ``[-1, 1]``.
    """
    a = np.asarray(_as_float_list(group_a), dtype=float)
    b = np.asarray(_as_float_list(group_b), dtype=float)
    rng = np.random.default_rng(int(random_seed))

    if a.size > max_samples_per_group:
        a = rng.choice(a, size=max_samples_per_group, replace=False)
    if b.size > max_samples_per_group:
        b = rng.choice(b, size=max_samples_per_group, replace=False)

    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    denom = int(a.size * b.size)
    if denom <= 0:
        return 0.0
    return float((gt - lt) / denom)


def _p_value(test_name: str, a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from scipy import stats as scipy_stats  # type: ignore
    except Exception:
        return None

    try:
        if test_name == "welch_t":
            return float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
        if test_name == "mannwhitney_u":
            return float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return None
    return None


def compare_two_groups(
    group_a: list[Any],
    group_b: list[Any],
    *,
    metric_name: str = "metric",
    alpha: float = 0.05,
    test: str = "auto",
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run a two-group comparison with CIs, effect sizes, and test metadata.

    Parameters
    ----------
    group_a : list[Any]
        Baseline group values.
    group_b : list[Any]
        Comparison group values.
    metric_name : str, default="metric"
        Display name for the measured variable.
    alpha : float, default=0.05
        Significance threshold.
    test : str, default="auto"
        Statistical test selector. `"auto"` picks Welch t-test for larger
        groups and Mann-Whitney U otherwise.
    random_seed : int, default=42
        Seed for bootstrap confidence interval generation.

    Returns
    -------
    dict[str, Any]
        Combined comparison payload with descriptive summaries, effect sizes,
        CI, and hypothesis-test metadata.

    Notes
    -----
    P-values may be omitted when SciPy is unavailable in the runtime.
    """
    a = np.asarray(_as_float_list(group_a), dtype=float)
    b = np.asarray(_as_float_list(group_b), dtype=float)
    if a.size < 2 or b.size < 2:
        return {
            "metric": metric_name,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "error": "Need at least 2 observations per group for comparison.",
        }

    if test == "auto":
        selected_test = "welch_t" if (a.size >= 20 and b.size >= 20) else "mannwhitney_u"
    else:
        selected_test = str(test)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_diff = float(mean_b - mean_a)
    pct_change = float((mean_diff / mean_a) * 100.0) if abs(mean_a) > 1e-12 else None
    d = cohen_d(a.tolist(), b.tolist())
    delta = cliffs_delta(a.tolist(), b.tolist(), random_seed=random_seed)

    rng = np.random.default_rng(int(random_seed))
    n_resamples = 3000
    idx_a = rng.integers(0, a.size, size=(n_resamples, a.size))
    idx_b = rng.integers(0, b.size, size=(n_resamples, b.size))
    diff_samples = np.mean(b[idx_b], axis=1) - np.mean(a[idx_a], axis=1)
    ci_low, ci_high = np.quantile(diff_samples, [0.025, 0.975])

    p_val = _p_value(selected_test, a, b)
    if p_val is None:
        significance = "unknown"
    elif p_val < float(alpha):
        significance = "statistically_significant"
    else:
        significance = "not_statistically_significant"

    abs_d = abs(d)
    if abs_d < 0.2:
        practical = "negligible"
    elif abs_d < 0.5:
        practical = "small"
    elif abs_d < 0.8:
        practical = "medium"
    else:
        practical = "large"

    return {
        "metric": metric_name,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "summary_a": summary_statistics(a.tolist()),
        "summary_b": summary_statistics(b.tolist()),
        "mean_diff": mean_diff,
        "mean_diff_ci95": [float(ci_low), float(ci_high)],
        "median_diff": float(np.median(b) - np.median(a)),
        "percent_change_mean": pct_change,
        "effect_sizes": {
            "cohen_d": float(d),
            "cliffs_delta": float(delta),
            "practical_significance": practical,
        },
        "test": {
            "selected": selected_test,
            "alpha": float(alpha),
            "p_value": p_val,
            "significance": significance,
            "note": (
                "p-value omitted because SciPy is unavailable in runtime."
                if p_val is None
                else None
            ),
        },
        "interpretation": (
            f"{metric_name}: mean changed by {mean_diff:.4g} "
            + (f"({pct_change:.2f}% relative change); " if pct_change is not None else "")
            + f"effect size d={d:.3f} ({practical})."
        ),
    }


@dataclass(frozen=True)
class CuratedStatTool:
    name: str
    description: str
    runner: Callable[[dict[str, Any]], dict[str, Any]]


def _tool_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return summary_statistics(payload.get("values", []))


def _tool_ci_mean(payload: dict[str, Any]) -> dict[str, Any]:
    return confidence_interval_mean(
        payload.get("values", []),
        confidence=float(payload.get("confidence", 0.95)),
        method=str(payload.get("method", "auto")),
    )


def _tool_bootstrap_ci(payload: dict[str, Any]) -> dict[str, Any]:
    return bootstrap_confidence_interval(
        payload.get("values", []),
        statistic=str(payload.get("statistic", "mean")),
        confidence=float(payload.get("confidence", 0.95)),
        n_resamples=int(payload.get("n_resamples", 2000)),
        random_seed=int(payload.get("random_seed", 42)),
    )


def _tool_effect_size(payload: dict[str, Any]) -> dict[str, Any]:
    a = payload.get("group_a", [])
    b = payload.get("group_b", [])
    return {
        "cohen_d": cohen_d(a, b),
        "cliffs_delta": cliffs_delta(
            a,
            b,
            max_samples_per_group=int(payload.get("max_samples_per_group", 2000)),
            random_seed=int(payload.get("random_seed", 42)),
        ),
    }


def _tool_compare_groups(payload: dict[str, Any]) -> dict[str, Any]:
    return compare_two_groups(
        payload.get("group_a", []),
        payload.get("group_b", []),
        metric_name=str(payload.get("metric_name", "metric")),
        alpha=float(payload.get("alpha", 0.05)),
        test=str(payload.get("test", "auto")),
        random_seed=int(payload.get("random_seed", 42)),
    )


_CURATED: dict[str, CuratedStatTool] = {
    "summary_statistics": CuratedStatTool(
        name="summary_statistics",
        description="Compute descriptive stats (mean, median, std, quantiles) for one numeric vector.",
        runner=_tool_summary,
    ),
    "confidence_interval_mean": CuratedStatTool(
        name="confidence_interval_mean",
        description="Compute confidence interval for the mean using t-distribution when available.",
        runner=_tool_ci_mean,
    ),
    "bootstrap_confidence_interval": CuratedStatTool(
        name="bootstrap_confidence_interval",
        description="Compute bootstrap confidence interval for mean/median/std.",
        runner=_tool_bootstrap_ci,
    ),
    "effect_size": CuratedStatTool(
        name="effect_size",
        description="Compute standardized effect sizes between two groups (Cohen's d and Cliff's delta).",
        runner=_tool_effect_size,
    ),
    "compare_two_groups": CuratedStatTool(
        name="compare_two_groups",
        description="Full two-group comparison with test selection logic, CI, effect sizes, and interpretation.",
        runner=_tool_compare_groups,
    ),
}


def list_curated_stat_tools() -> list[dict[str, Any]]:
    """Return metadata for curated statistical tools exposed by the API.

    Returns
    -------
    list[dict[str, Any]]
        Tool descriptors with `name` and `description`.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in _CURATED.values()
    ]


def run_stat_tool(tool_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate payload and dispatch one curated statistical tool call.

    Parameters
    ----------
    tool_name : str
        Curated statistical tool identifier.
    payload : dict[str, Any] or None, default=None
        Tool-specific input payload.

    Returns
    -------
    dict[str, Any]
        Execution result envelope containing `success`, `tool_name`, and
        either `result` or `error`.
    """
    name = str(tool_name or "").strip()
    if name not in _CURATED:
        return {
            "success": False,
            "error": f"Unknown statistical tool: {name}",
            "available_tools": [item["name"] for item in list_curated_stat_tools()],
        }
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {"success": False, "error": "payload must be an object"}

    runner = _CURATED[name].runner
    try:
        result = runner(payload)
    except Exception as exc:
        return {"success": False, "error": str(exc), "tool_name": name}
    return {"success": True, "tool_name": name, "result": result}


__all__ = [
    "summary_statistics",
    "confidence_interval_mean",
    "bootstrap_confidence_interval",
    "cohen_d",
    "cliffs_delta",
    "compare_two_groups",
    "list_curated_stat_tools",
    "run_stat_tool",
]
