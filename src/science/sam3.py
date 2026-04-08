"""SAM3-based automatic segmentation for scientific image tensors.

This module provides a robust first-pass segmentation pathway that:
- handles 2D/3D/4D scientific arrays slice-wise;
- applies lightweight classical preprocessing for stability;
- proposes automatic prompt points from image statistics;
- uses sliding-window inference with 25% overlap on large slices;
- falls back to MedSAM2 when SAM3 weights/runtime are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from src.config import get_settings
from src.science.medsam2 import segment_array_with_medsam2

_SAM3_TRACKER_CACHE: dict[tuple[str, str, bool], tuple[Any, Any, Any]] = {}
_SAM3_CONCEPT_CACHE: dict[tuple[str, str, bool], tuple[Any, Any, Any]] = {}


@dataclass(frozen=True)
class _Sam3Runtime:
    backend: str
    model_ref: str
    device: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PreprocessProfile:
    name: str
    clip_lo_pct: float
    clip_hi_pct: float
    denoise_sigma: float
    tophat_radius: int
    local_contrast_window: int
    gamma: float
    unsharp_amount: float


_PREPROCESS_PROFILES: dict[str, _PreprocessProfile] = {
    "generic": _PreprocessProfile(
        name="generic",
        clip_lo_pct=0.5,
        clip_hi_pct=99.5,
        denoise_sigma=0.8,
        tophat_radius=0,
        local_contrast_window=0,
        gamma=1.0,
        unsharp_amount=0.0,
    ),
    "fluorescence": _PreprocessProfile(
        name="fluorescence",
        clip_lo_pct=0.1,
        clip_hi_pct=99.9,
        denoise_sigma=0.6,
        tophat_radius=7,
        local_contrast_window=25,
        gamma=0.9,
        unsharp_amount=0.35,
    ),
    "brightfield": _PreprocessProfile(
        name="brightfield",
        clip_lo_pct=1.0,
        clip_hi_pct=99.5,
        denoise_sigma=1.0,
        tophat_radius=0,
        local_contrast_window=11,
        gamma=1.0,
        unsharp_amount=0.1,
    ),
    "ct_like": _PreprocessProfile(
        name="ct_like",
        clip_lo_pct=2.0,
        clip_hi_pct=99.8,
        denoise_sigma=1.0,
        tophat_radius=0,
        local_contrast_window=0,
        gamma=1.0,
        unsharp_amount=0.15,
    ),
}


def _normalize_profile_name(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"fluo", "fluorescence"}:
        return "fluorescence"
    if raw in {"bf", "brightfield", "bright_field"}:
        return "brightfield"
    if raw in {"ct", "ct_like", "hu"}:
        return "ct_like"
    if raw in {"generic"}:
        return "generic"
    return "auto"


def _profile_for_name(name: str) -> _PreprocessProfile:
    return _PREPROCESS_PROFILES.get(str(name), _PREPROCESS_PROFILES["generic"])


def _torch_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch is not None)
    except Exception:
        return False


def _sam3_tracker_available() -> bool:
    try:
        from transformers import Sam3TrackerConfig, Sam3TrackerModel, Sam3TrackerProcessor  # type: ignore

        return bool(
            Sam3TrackerModel is not None
            and Sam3TrackerProcessor is not None
            and Sam3TrackerConfig is not None
        )
    except Exception:
        return False


def _sam3_concept_available() -> bool:
    try:
        from transformers import Sam3Model, Sam3Processor  # type: ignore

        return bool(Sam3Model is not None and Sam3Processor is not None)
    except Exception:
        return False


def _resolve_device(device: str | None) -> str:
    try:
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if not device:
        return "cuda" if has_cuda else "cpu"

    value = str(device).strip().lower()
    if value in {"cpu", "-1"}:
        return "cpu"
    if value in {"cuda", "gpu"}:
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return "cuda"
    if value.startswith("cuda:"):
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return value
    if value.isdigit():
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return f"cuda:{value}"
    raise ValueError(f"Invalid device value: {device!r}. Use cpu, cuda, cuda:N, or N.")


def _resolve_model_ref(model_id: str | None) -> tuple[str, list[str]]:
    settings = get_settings()
    raw = str(model_id or getattr(settings, "sam3_model_id", "facebook/sam3")).strip()
    if not raw:
        raw = "facebook/sam3"

    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), []

    default_dir = Path("data/models/sam3/facebook-sam3")
    if raw.lower() in {"default", "latest", "facebook/sam3"} and default_dir.exists():
        return str(default_dir.resolve()), []

    warning = (
        f"SAM3 model reference '{raw}' is not a local path. "
        "If remote downloads are disabled, place a local SAM3 snapshot and pass its path."
    )
    return raw, [warning]


def _get_tracker_runtime(
    model_ref: str,
    device: str,
    allow_remote_download: bool,
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    import torch  # type: ignore
    from transformers import AutoConfig, Sam3TrackerConfig, Sam3TrackerModel, Sam3TrackerProcessor  # type: ignore

    cache_key = (str(model_ref), str(device), bool(allow_remote_download))
    if cache_key in _SAM3_TRACKER_CACHE:
        model, processor, dtype = _SAM3_TRACKER_CACHE[cache_key]
        return model, processor, dtype, ()

    local_path = Path(str(model_ref)).expanduser()
    local_only = not bool(allow_remote_download) and not local_path.exists()
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    runtime_warnings: list[str] = []

    model_config = None
    try:
        base_cfg = AutoConfig.from_pretrained(
            model_ref,
            local_files_only=bool(local_only),
        )
        if str(getattr(base_cfg, "model_type", "")).strip().lower() != "sam3_tracker":
            cfg_dict = dict(base_cfg.to_dict())
            cfg_dict["model_type"] = "sam3_tracker"
            cfg_dict["architectures"] = ["Sam3TrackerModel"]
            model_config = Sam3TrackerConfig.from_dict(cfg_dict)
            runtime_warnings.append(
                "SAM3 config was coerced to tracker mode for image segmentation compatibility."
            )
        else:
            model_config = Sam3TrackerConfig.from_dict(base_cfg.to_dict())
    except Exception as exc:
        runtime_warnings.append(f"SAM3 config coercion skipped: {exc}")
        model_config = None

    model = Sam3TrackerModel.from_pretrained(
        model_ref,
        config=model_config,
        local_files_only=bool(local_only),
    )
    model.to(device=device, dtype=dtype)
    model.eval()

    processor = Sam3TrackerProcessor.from_pretrained(
        model_ref,
        local_files_only=bool(local_only),
    )
    _SAM3_TRACKER_CACHE[cache_key] = (model, processor, dtype)
    return model, processor, dtype, tuple(runtime_warnings)


def _get_concept_runtime(
    model_ref: str,
    device: str,
    allow_remote_download: bool,
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    import torch  # type: ignore
    from transformers import Sam3Model, Sam3Processor  # type: ignore

    cache_key = (str(model_ref), str(device), bool(allow_remote_download))
    if cache_key in _SAM3_CONCEPT_CACHE:
        model, processor, dtype = _SAM3_CONCEPT_CACHE[cache_key]
        return model, processor, dtype, ()

    local_path = Path(str(model_ref)).expanduser()
    local_only = not bool(allow_remote_download) and not local_path.exists()
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    runtime_warnings: list[str] = []

    model = Sam3Model.from_pretrained(
        model_ref,
        local_files_only=bool(local_only),
    )
    model.to(device=device, dtype=dtype)
    model.eval()

    processor = Sam3Processor.from_pretrained(
        model_ref,
        local_files_only=bool(local_only),
    )

    _SAM3_CONCEPT_CACHE[cache_key] = (model, processor, dtype)
    return model, processor, dtype, tuple(runtime_warnings)


def _normalize_01(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.float32)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32)


def _normalize_uint8(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    return (_normalize_01(arr, lo_pct=lo_pct, hi_pct=hi_pct) * 255.0).astype(np.uint8)


def _otsu_threshold(image_01: np.ndarray) -> float:
    x = np.clip(np.asarray(image_01, dtype=np.float32), 0.0, 1.0)
    hist, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]

    sigma_b_sq = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = int(np.nanargmax(sigma_b_sq))
    return float(centers[idx])


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 1:
        return mask.astype(bool)
    lab, n = ndi.label(mask.astype(bool))
    if n <= 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_area)
    keep[0] = False
    return keep[lab]


def _reconcile_slice_axes(slice_data: np.ndarray, order: str) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(slice_data)
    ord_chars = [ch for ch in str(order).upper() if ch.isalpha()]

    # Reconcile occasional order/shape mismatches from heterogeneous loaders.
    if arr.ndim == len(ord_chars) + 1 and "C" not in ord_chars and arr.shape[-1] in {1, 3, 4}:
        ord_chars = ord_chars + ["C"]
    while arr.ndim > len(ord_chars) and arr.shape and int(arr.shape[0]) == 1:
        arr = np.take(arr, 0, axis=0)
    if arr.ndim > len(ord_chars):
        ord_chars = ord_chars + [f"U{i}" for i in range(arr.ndim - len(ord_chars))]
    elif len(ord_chars) > arr.ndim:
        ord_chars = ord_chars[: arr.ndim]

    for axis in list(ord_chars):
        if axis not in {"C", "Y", "X"}:
            idx = ord_chars.index(axis)
            arr = np.take(arr, 0, axis=idx)
            ord_chars.pop(idx)
    return arr, ord_chars


def _slice_to_gray_float(slice_data: np.ndarray, order: str) -> np.ndarray:
    arr, ord_chars = _reconcile_slice_axes(slice_data, order)
    if arr.ndim <= 0:
        return np.zeros((1, 1), dtype=np.float32)

    if "Y" not in ord_chars or "X" not in ord_chars:
        if arr.ndim >= 2:
            flat = arr.reshape((-1,) + arr.shape[-2:])[0]
            return np.asarray(flat, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32).reshape((1, 1))

    if "C" in ord_chars:
        perm = [ord_chars.index("C"), ord_chars.index("Y"), ord_chars.index("X")]
        cyx = np.transpose(arr, perm).astype(np.float32)
        c = int(min(cyx.shape[0], 3))
        return np.mean(cyx[:c], axis=0)

    perm = [ord_chars.index("Y"), ord_chars.index("X")]
    return np.transpose(arr, perm).astype(np.float32)


def _enhance_channel(channel: np.ndarray, *, profile: _PreprocessProfile, preprocess: bool) -> np.ndarray:
    if not preprocess:
        return _normalize_uint8(channel)

    x = _normalize_01(
        np.asarray(channel, dtype=np.float32),
        lo_pct=float(profile.clip_lo_pct),
        hi_pct=float(profile.clip_hi_pct),
    )

    if int(profile.tophat_radius) > 0:
        radius = int(profile.tophat_radius)
        size = (2 * radius + 1, 2 * radius + 1)
        opened = ndi.grey_opening(x, size=size)
        x = np.clip(x - opened, 0.0, 1.0)

    if float(profile.denoise_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(profile.denoise_sigma))

    if int(profile.local_contrast_window) > 1:
        win = int(max(3, profile.local_contrast_window))
        if win % 2 == 0:
            win += 1
        local_mean = ndi.uniform_filter(x, size=win, mode="nearest")
        local_sq = ndi.uniform_filter(x * x, size=win, mode="nearest")
        local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 1e-6))
        x = np.clip((x - local_mean) / (2.5 * local_std) + 0.5, 0.0, 1.0)

    if float(profile.unsharp_amount) > 0:
        blur = ndi.gaussian_filter(x, sigma=max(0.6, float(profile.denoise_sigma) * 1.6))
        x = np.clip(x + float(profile.unsharp_amount) * (x - blur), 0.0, 1.0)

    if abs(float(profile.gamma) - 1.0) > 1e-3:
        x = np.power(np.clip(x, 0.0, 1.0), float(profile.gamma))

    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _slice_to_rgb_uint8(
    slice_data: np.ndarray,
    order: str,
    preprocess: bool,
    profile: _PreprocessProfile | None = None,
) -> np.ndarray:
    arr, ord_chars = _reconcile_slice_axes(slice_data, order)
    profile = profile or _PREPROCESS_PROFILES["generic"]
    if "Y" not in ord_chars or "X" not in ord_chars:
        gray = _slice_to_gray_float(arr, "".join(ord_chars))
        ch = _enhance_channel(gray, profile=profile, preprocess=bool(preprocess))
        return np.repeat(ch[..., None], 3, axis=-1)

    if "C" in ord_chars:
        perm = [ord_chars.index("C"), ord_chars.index("Y"), ord_chars.index("X")]
        cyx = np.transpose(arr, perm).astype(np.float32)
        if cyx.shape[0] >= 3:
            rgb = np.stack(
                [
                    _enhance_channel(cyx[i], profile=profile, preprocess=bool(preprocess))
                    for i in range(3)
                ],
                axis=-1,
            )
        else:
            gray = _enhance_channel(cyx[0], profile=profile, preprocess=bool(preprocess))
            rgb = np.repeat(gray[..., None], 3, axis=-1)
    else:
        perm = [ord_chars.index("Y"), ord_chars.index("X")]
        yx = np.transpose(arr, perm).astype(np.float32)
        gray = _enhance_channel(yx, profile=profile, preprocess=bool(preprocess))
        rgb = np.repeat(gray[..., None], 3, axis=-1)

    return np.asarray(rgb, dtype=np.uint8)


def _infer_modality_profile(
    slices: np.ndarray,
    slice_order: str,
    *,
    hint: str | None = None,
) -> tuple[str, dict[str, Any], list[str]]:
    normalized_hint = _normalize_profile_name(hint)
    warnings: list[str] = []
    if str(hint or "").strip() and normalized_hint == "auto" and str(hint).strip().lower() != "auto":
        warnings.append(
            f"Unrecognized modality_hint '{hint}'. Falling back to automatic profile inference."
        )
    if normalized_hint != "auto":
        return normalized_hint, {"strategy": "hint", "hint": normalized_hint}, warnings

    if slices.ndim < 3:
        return "generic", {"strategy": "auto_fallback"}, warnings

    z_count = int(slices.shape[0])
    sample_count = int(max(1, min(7, z_count)))
    sample_indices = np.unique(np.linspace(0, max(z_count - 1, 0), num=sample_count, dtype=int))
    sampled_values: list[np.ndarray] = []
    for z in sample_indices.tolist():
        gray = _slice_to_gray_float(slices[z], slice_order)
        finite = gray[np.isfinite(gray)]
        if finite.size <= 0:
            continue
        step = max(1, int(np.ceil(finite.size / 50000.0)))
        sampled_values.append(finite[::step])

    if not sampled_values:
        return "generic", {"strategy": "auto_no_finite"}, warnings

    vals = np.concatenate(sampled_values, axis=0).astype(np.float32)
    p1, p50, p99 = np.percentile(vals, [1.0, 50.0, 99.0]).astype(np.float64).tolist()
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = float(max(p99 - p1, 1e-6))

    zero_tol = max(1e-6, 0.001 * span)
    zero_fraction = float(np.mean(np.abs(vals) <= zero_tol))

    norm = np.clip((vals - p1) / span, 0.0, 1.0)
    dark_fraction = float(np.mean(norm <= 0.1))
    bright_fraction = float(np.mean(norm >= 0.9))
    skew_ratio = float((p99 - p50) / max(p50 - p1, 1e-6))

    decision = "generic_fallback"
    if p1 < -200.0 and p99 > 200.0 and (p99 - p1) > 500.0:
        profile_name = "ct_like"
        decision = "ct_hu_dynamic_range"
    elif dark_fraction > 0.65 and bright_fraction < 0.08 and skew_ratio > 2.5:
        profile_name = "fluorescence"
        decision = "dark_sparse_high_skew"
    elif (
        # Brightfield-ish data often has lower dark mass and moderate/high right-skew,
        # including many microscopy datasets that the previous strict skew cutoff missed.
        (
            dark_fraction < 0.55
            and skew_ratio < 2.2
            and (bright_fraction > 0.005 or zero_fraction < 0.05)
        )
        # Backup branch for bright, broad histograms.
        or (dark_fraction < 0.7 and skew_ratio < 2.6 and bright_fraction > 0.12)
    ):
        profile_name = "brightfield"
        decision = "brightfield_histogram_shape"
    else:
        profile_name = "generic"
        decision = "generic_default"

    stats = {
        "strategy": "auto",
        "profile_name": profile_name,
        "sampled_slices": [int(v) for v in sample_indices.tolist()],
        "sampled_voxels": int(vals.size),
        "min": round(vmin, 6),
        "p1": round(float(p1), 6),
        "p50": round(float(p50), 6),
        "p99": round(float(p99), 6),
        "max": round(vmax, 6),
        "zero_fraction": round(zero_fraction, 6),
        "dark_fraction": round(dark_fraction, 6),
        "bright_fraction": round(bright_fraction, 6),
        "skew_ratio": round(skew_ratio, 6),
        "decision": decision,
    }
    return profile_name, stats, warnings


def _canonical_slices(array: np.ndarray, order: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(array)
    ord_chars = [ch for ch in str(order).upper() if ch.isalpha()]

    # Drop common batch/time axes by selecting the first index.
    for axis in ("B", "N", "T"):
        if axis in ord_chars:
            idx = ord_chars.index(axis)
            arr = np.take(arr, 0, axis=idx)
            ord_chars.pop(idx)

    # Drop unsupported axes by selecting first index.
    for axis in list(ord_chars):
        if axis not in {"C", "Z", "Y", "X"}:
            idx = ord_chars.index(axis)
            arr = np.take(arr, 0, axis=idx)
            ord_chars.pop(idx)

    if "Z" in ord_chars:
        z_axis = ord_chars.index("Z")
        arr = np.moveaxis(arr, z_axis, 0)
        ord_chars.pop(z_axis)
        return arr, "".join(ord_chars)

    return arr[None, ...], "".join(ord_chars)


def _build_candidates(mask: np.ndarray, score: np.ndarray, limit: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    values = np.asarray(score, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    order = np.argsort(values)[::-1]
    if int(limit) > 0:
        order = order[: int(limit)]
    y = coords[order, 0].astype(np.float32)
    x = coords[order, 1].astype(np.float32)
    v = values[order].astype(np.float32)
    return np.stack([x, y, v], axis=1)


def _merge_candidates(candidates: list[np.ndarray], limit: int) -> np.ndarray:
    valid = [
        np.asarray(c, dtype=np.float32)
        for c in candidates
        if c is not None and np.asarray(c).size > 0
    ]
    if not valid:
        return np.zeros((0, 3), dtype=np.float32)
    merged = np.concatenate(valid, axis=0)
    if merged.ndim != 2 or merged.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    order = np.argsort(merged[:, 2])[::-1]
    if int(limit) > 0:
        order = order[: int(limit)]
    return merged[order]


def _append_spread_points(
    points: list[list[float]],
    candidates: np.ndarray,
    *,
    count: int,
    min_dist: float,
) -> int:
    if int(count) <= 0 or candidates.size <= 0:
        return 0
    added = 0
    min_dist_sq = float(min_dist * min_dist)
    for row in np.asarray(candidates, dtype=np.float32):
        x = float(row[0])
        y = float(row[1])
        keep = True
        for px, py in points:
            dx = x - float(px)
            dy = y - float(py)
            if dx * dx + dy * dy < min_dist_sq:
                keep = False
                break
        if keep:
            points.append([x, y])
            added += 1
            if added >= int(count):
                break
    return int(added)


def _append_labeled_spread_points(
    points: list[list[float]],
    labels: list[int],
    candidates: np.ndarray,
    *,
    label: int,
    count: int,
    min_dist: float,
) -> int:
    added = _append_spread_points(
        points,
        candidates,
        count=int(count),
        min_dist=float(min_dist),
    )
    if added > 0:
        labels.extend([int(label)] * int(added))
    return int(added)


def _local_maxima_mask(values: np.ndarray, size: int) -> np.ndarray:
    k = int(max(3, size))
    if k % 2 == 0:
        k += 1
    vmax = ndi.maximum_filter(np.asarray(values, dtype=np.float32), size=k, mode="nearest")
    return np.asarray(values, dtype=np.float32) >= (vmax - 1e-8)


def _robust_foreground_mask(smooth: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(smooth, dtype=np.float32)
    h, w = x.shape
    thr = float(_otsu_threshold(x))
    fg = x >= thr
    frac = float(np.mean(fg))

    if frac < 0.01:
        thr = max(thr, float(np.percentile(x, 92.0)))
        fg = x >= thr
    elif frac > 0.96:
        thr = float(np.percentile(x, 65.0))
        fg = x >= thr

    structure = np.ones((3, 3), dtype=bool)
    fg = ndi.binary_opening(fg, structure=structure)
    fg = ndi.binary_closing(fg, structure=structure)
    min_area = int(max(16, round(0.0005 * float(h * w))))
    fg = _remove_small_components(fg, min_area=min_area)

    if not np.any(fg):
        thr = float(np.percentile(x, 80.0))
        fg = _remove_small_components(x >= thr, min_area=max(8, min_area // 2))

    return fg.astype(bool), float(thr)


def _allocate_source_quotas(
    target: int,
    *,
    sources: list[tuple[str, float, int]],
) -> dict[str, int]:
    quotas = {name: 0 for name, _, _ in sources}
    active = [(name, float(weight), int(count)) for name, weight, count in sources if int(count) > 0]
    t = int(max(0, target))
    if t <= 0 or not active:
        return quotas

    total_weight = float(sum(weight for _, weight, _ in active))
    for name, weight, count in active:
        if total_weight <= 0:
            q = int(np.floor(float(t) / float(len(active))))
        else:
            q = int(round(float(t) * (float(weight) / total_weight)))
        if t >= len(active):
            q = max(1, q)
        quotas[name] = min(int(count), int(max(0, q)))

    assigned = int(sum(quotas.values()))
    while assigned < t:
        grown = False
        for name, _, count in sorted(active, key=lambda row: (row[2] - quotas[row[0]], row[1]), reverse=True):
            if quotas[name] < int(count):
                quotas[name] += 1
                assigned += 1
                grown = True
                if assigned >= t:
                    break
        if not grown:
            break

    while assigned > t:
        shrunk = False
        min_keep = 1 if t >= len(active) else 0
        for name, _, _ in sorted(active, key=lambda row: (quotas[row[0]], row[1]), reverse=True):
            if quotas[name] > min_keep:
                quotas[name] -= 1
                assigned -= 1
                shrunk = True
                if assigned <= t:
                    break
        if not shrunk:
            break

    return quotas


def _logdog_blob_candidates(
    smooth: np.ndarray,
    *,
    foreground: np.ndarray,
    target: int,
) -> np.ndarray:
    h, w = smooth.shape
    min_dim = float(min(h, w))
    min_sigma = float(max(0.9, min_dim * 0.004))
    max_sigma = float(max(min_sigma * 1.8, min_dim * 0.045))
    if max_sigma <= min_sigma:
        max_sigma = float(min_sigma * 1.8)

    sigma_levels = np.geomspace(min_sigma, max_sigma, num=5)
    gaussians = [ndi.gaussian_filter(smooth, sigma=float(s)) for s in sigma_levels]

    support = (
        ndi.binary_dilation(np.asarray(foreground, dtype=bool), iterations=1)
        if np.any(foreground)
        else np.ones_like(smooth, dtype=bool)
    )

    candidates: list[np.ndarray] = []
    for i in range(len(sigma_levels) - 1):
        sigma = float(sigma_levels[i])
        dog = np.abs(np.asarray(gaussians[i + 1]) - np.asarray(gaussians[i])) * (sigma * sigma)
        peak_size = int(max(3, round(2.5 * sigma)))
        peak_mask = _local_maxima_mask(dog, size=peak_size)

        vals = np.asarray(dog)[support]
        if vals.size <= 0:
            vals = np.asarray(dog).ravel()
        cut = float(np.percentile(vals, 86.0))
        mask = peak_mask & support & (np.asarray(dog) >= cut)
        candidates.append(
            _build_candidates(mask, np.asarray(dog), limit=max(3 * int(target), 24))
        )

    return _merge_candidates(candidates, limit=max(8 * int(target), 96))


def _distance_skeleton_candidates(
    foreground: np.ndarray,
    *,
    target: int,
) -> np.ndarray:
    fg = np.asarray(foreground, dtype=bool)
    if not np.any(fg):
        return np.zeros((0, 3), dtype=np.float32)

    dist = ndi.distance_transform_edt(fg.astype(np.uint8))
    positive = dist[dist > 0]
    if positive.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    ridge_size = int(max(3, round(0.03 * float(min(fg.shape)))))
    ridge_mask = _local_maxima_mask(dist, size=ridge_size)
    center_cut = float(np.percentile(positive, 55.0))
    mask = ridge_mask & (dist >= center_cut)
    candidates = _build_candidates(mask, dist, limit=max(4 * int(target), 32))
    if candidates.size > 0:
        return _merge_candidates([candidates], limit=max(8 * int(target), 96))

    comp, ncomp = ndi.label(fg)
    fallback: list[list[float]] = []
    for idx in range(1, int(ncomp) + 1):
        region = comp == idx
        if not np.any(region):
            continue
        score = np.where(region, dist, -1.0)
        flat = int(np.argmax(score))
        y, x = np.unravel_index(flat, dist.shape)
        fallback.append([float(x), float(y), float(dist[y, x])])
    if not fallback:
        return np.zeros((0, 3), dtype=np.float32)
    return _merge_candidates([np.asarray(fallback, dtype=np.float32)], limit=max(4 * int(target), 32))


def _watershed_basin_candidates(
    foreground: np.ndarray,
    gradient: np.ndarray,
    *,
    target: int,
) -> np.ndarray:
    fg = np.asarray(foreground, dtype=bool)
    if not np.any(fg):
        return np.zeros((0, 3), dtype=np.float32)

    frac = float(np.mean(fg))
    if frac < 0.02 or frac > 0.92:
        return np.zeros((0, 3), dtype=np.float32)

    h, w = fg.shape
    dist = ndi.distance_transform_edt(fg.astype(np.uint8))
    positive = dist[dist > 0]
    if positive.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    peak_size = int(max(3, round(0.03 * float(min(h, w)))))
    peaks = _local_maxima_mask(dist, size=peak_size) & (dist >= float(np.percentile(positive, 70.0)))
    seed_candidates = _build_candidates(peaks, dist, limit=max(5 * int(target), 64))
    if seed_candidates.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    max_seeds = int(max(4, min(seed_candidates.shape[0], max(12, 4 * int(target)))))
    seeds = np.asarray(seed_candidates[:max_seeds], dtype=np.float32)
    markers = np.zeros((h, w), dtype=np.int32)
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1

    marker_id = 2
    for row in seeds:
        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if not bool(fg[y, x]):
            continue
        if int(markers[y, x]) != 0:
            continue
        markers[y, x] = marker_id
        marker_id += 1

    if marker_id <= 3:
        return np.zeros((0, 3), dtype=np.float32)

    try:
        grad_u8 = _normalize_uint8(np.asarray(gradient, dtype=np.float32), lo_pct=2.0, hi_pct=99.5)
        ws = ndi.watershed_ift(grad_u8, markers)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)

    min_area = int(max(8, round(0.00015 * float(h * w))))
    basins: list[list[float]] = []
    for lab in range(2, marker_id):
        region = (ws == lab) & fg
        area = int(np.count_nonzero(region))
        if area < min_area:
            continue
        score = np.where(region, dist, -1.0)
        flat = int(np.argmax(score))
        y, x = np.unravel_index(flat, dist.shape)
        peak = float(dist[y, x])
        basin_score = peak * (1.0 + min(3.0, np.sqrt(float(area)) / 12.0))
        basins.append([float(x), float(y), float(basin_score)])

    if not basins:
        return np.zeros((0, 3), dtype=np.float32)
    return _merge_candidates([np.asarray(basins, dtype=np.float32)], limit=max(8 * int(target), 96))


def _hard_negative_candidates(
    smooth: np.ndarray,
    gradient: np.ndarray,
    foreground: np.ndarray,
    *,
    target: int,
) -> np.ndarray:
    smooth_x = np.asarray(smooth, dtype=np.float32)
    grad_norm = _normalize_01(np.asarray(gradient, dtype=np.float32), lo_pct=1.0, hi_pct=99.0)
    fg = np.asarray(foreground, dtype=bool)
    h, w = smooth_x.shape

    candidates: list[np.ndarray] = []
    comp, ncomp = ndi.label(fg)
    if int(ncomp) > 0:
        comp_sizes = np.bincount(comp.ravel())[1:]
        size_cut = float(np.percentile(comp_sizes, 50.0)) if comp_sizes.size > 0 else 0.0
        fg_grad_ref = float(np.median(grad_norm[fg])) if np.any(fg) else float(np.median(grad_norm))
        fg_int_ref = float(np.median(smooth_x[fg])) if np.any(fg) else float(np.median(smooth_x))
        hard_rows: list[list[float]] = []
        for idx in range(1, int(ncomp) + 1):
            region = comp == idx
            area = int(np.count_nonzero(region))
            if area <= 0:
                continue
            mean_g = float(np.mean(grad_norm[region]))
            mean_i = float(np.mean(smooth_x[region]))
            small = area <= int(max(12.0, size_cut))
            weak_edge = mean_g <= float(max(0.05, 0.85 * fg_grad_ref))
            dim_component = mean_i <= float(max(0.35, 1.05 * fg_int_ref))
            if not (small and weak_edge and dim_component):
                continue
            cy, cx = ndi.center_of_mass(region.astype(np.uint8))
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            score = (1.0 - mean_g) * 0.7 + (1.0 - min(1.0, area / float(max(h * w, 1)))) * 0.3
            hard_rows.append([float(cx), float(cy), float(score)])
        if hard_rows:
            candidates.append(
                _merge_candidates(
                    [np.asarray(hard_rows, dtype=np.float32)],
                    limit=max(3 * int(target), 24),
                )
            )

    # Background low-gradient pockets far from foreground.
    background = np.logical_not(fg)
    bg_dist = ndi.distance_transform_edt(background.astype(np.uint8))
    bg_dist_norm = _normalize_01(bg_dist, lo_pct=5.0, hi_pct=99.0)
    low_grad_cut = float(np.percentile(grad_norm, 40.0))
    bg_mask = background & (grad_norm <= low_grad_cut)
    if np.any(bg_mask):
        dist_vals = bg_dist[bg_mask]
        dist_cut = float(np.percentile(dist_vals, 70.0)) if dist_vals.size > 0 else 0.0
        pocket_mask = bg_mask & (bg_dist >= dist_cut)
        bg_score = 0.7 * (1.0 - grad_norm) + 0.3 * bg_dist_norm
        candidates.append(
            _build_candidates(
                pocket_mask,
                bg_score,
                limit=max(4 * int(target), 32),
            )
        )

    return _merge_candidates(candidates, limit=max(8 * int(target), 96))


def _propose_auto_points(
    image_rgb: np.ndarray,
    *,
    min_points: int,
    max_points: int,
    point_density: float,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    gray = np.asarray(image_rgb[..., :3], dtype=np.float32).mean(axis=-1)
    norm = _normalize_01(gray)
    smooth = ndi.gaussian_filter(norm, sigma=1.2)
    gy, gx = np.gradient(smooth)
    grad = np.hypot(gx, gy)

    fg, thr = _robust_foreground_mask(smooth)
    h, w = smooth.shape
    area = int(fg.sum()) if np.any(fg) else int(h * w)
    complexity = float(np.mean(grad[fg])) if np.any(fg) else float(np.mean(grad))

    target = int(area * float(point_density) * (1.0 + 2.5 * complexity))
    target = max(int(min_points), min(int(max_points), target))

    neg_ratio = float(min(0.35, max(0.15, 0.18 + 0.22 * complexity)))
    neg_target = int(round(float(target) * neg_ratio))
    if target <= 4:
        neg_target = 0
    elif target <= 8:
        neg_target = int(min(1, neg_target))
    else:
        neg_target = int(max(1, neg_target))
    neg_target = int(min(max(0, neg_target), max(0, target // 3)))
    pos_target = int(max(1, target - neg_target))

    pos_logdog = _logdog_blob_candidates(
        smooth,
        foreground=fg,
        target=pos_target,
    )
    pos_skeleton = _distance_skeleton_candidates(
        fg,
        target=pos_target,
    )
    pos_watershed = _watershed_basin_candidates(
        fg,
        grad,
        target=pos_target,
    )

    pos_fallback_score = 0.55 * smooth + 0.45 * grad
    if np.any(fg):
        pos_fallback_score = pos_fallback_score * (0.2 + 0.8 * fg.astype(np.float32))
    pos_fallback = _build_candidates(
        fg if np.any(fg) else np.ones_like(fg, dtype=bool),
        pos_fallback_score,
        limit=max(10 * pos_target, 96),
    )

    neg_hard = _hard_negative_candidates(
        smooth,
        grad,
        fg,
        target=max(1, neg_target),
    )
    neg_fallback_score = 0.65 * (1.0 - _normalize_01(grad, lo_pct=1.0, hi_pct=99.0)) + 0.35 * (1.0 - smooth)
    neg_support = np.logical_not(fg)
    if not np.any(neg_support):
        neg_support = np.ones_like(fg, dtype=bool)
    neg_fallback = _build_candidates(
        neg_support,
        neg_fallback_score,
        limit=max(6 * max(1, neg_target), 32),
    )

    pos_quotas = _allocate_source_quotas(
        pos_target,
        sources=[
            ("logdog", 0.35, int(pos_logdog.shape[0])),
            ("skeleton", 0.35, int(pos_skeleton.shape[0])),
            ("watershed", 0.30, int(pos_watershed.shape[0])),
        ],
    )
    neg_quotas = _allocate_source_quotas(
        neg_target,
        sources=[
            ("hardneg", 1.0, int(neg_hard.shape[0])),
        ],
    )

    points: list[list[float]] = []
    labels: list[int] = []
    min_dist = max(4, int(min(h, w) * 0.02))
    selected_counts: dict[str, int] = {}

    selected_counts["pos_logdog"] = _append_labeled_spread_points(
        points,
        labels,
        pos_logdog,
        label=1,
        count=pos_quotas.get("logdog", 0),
        min_dist=float(min_dist),
    )
    selected_counts["pos_skeleton"] = _append_labeled_spread_points(
        points,
        labels,
        pos_skeleton,
        label=1,
        count=pos_quotas.get("skeleton", 0),
        min_dist=float(min_dist),
    )
    selected_counts["pos_watershed"] = _append_labeled_spread_points(
        points,
        labels,
        pos_watershed,
        label=1,
        count=pos_quotas.get("watershed", 0),
        min_dist=float(min_dist),
    )
    remaining_pos = int(max(0, pos_target - sum(v for k, v in selected_counts.items() if k.startswith("pos_"))))
    selected_counts["pos_fallback"] = _append_labeled_spread_points(
        points,
        labels,
        pos_fallback,
        label=1,
        count=remaining_pos,
        min_dist=float(min_dist),
    )

    selected_counts["neg_hard"] = _append_labeled_spread_points(
        points,
        labels,
        neg_hard,
        label=0,
        count=neg_quotas.get("hardneg", 0),
        min_dist=float(max(3.0, 0.8 * float(min_dist))),
    )
    remaining_neg = int(max(0, neg_target - selected_counts["neg_hard"]))
    selected_counts["neg_fallback"] = _append_labeled_spread_points(
        points,
        labels,
        neg_fallback,
        label=0,
        count=remaining_neg,
        min_dist=float(max(3.0, 0.8 * float(min_dist))),
    )

    # Fill remaining budget with positive fallback first, then negatives.
    if len(points) < target:
        selected_counts["pos_fill"] = _append_labeled_spread_points(
            points,
            labels,
            pos_fallback,
            label=1,
            count=int(max(0, target - len(points))),
            min_dist=float(min_dist),
        )
    else:
        selected_counts["pos_fill"] = 0

    if len(points) < target and neg_target > 0:
        selected_counts["neg_fill"] = _append_labeled_spread_points(
            points,
            labels,
            neg_fallback,
            label=0,
            count=int(max(0, target - len(points))),
            min_dist=float(max(3.0, 0.75 * float(min_dist))),
        )
    else:
        selected_counts["neg_fill"] = 0

    if not points:
        points = [[float(w // 2), float(h // 2)]]
        labels = [1]
        selected_counts["pos_emergency"] = 1
    else:
        selected_counts["pos_emergency"] = 0

    if len(points) > target:
        points = points[:target]
        labels = labels[:target]

    if sum(int(v) for v in labels if int(v) == 1) <= 0:
        points.insert(0, [float(w // 2), float(h // 2)])
        labels.insert(0, 1)
        if len(points) > target:
            points = points[:target]
            labels = labels[:target]
        selected_counts["pos_emergency"] = max(1, selected_counts.get("pos_emergency", 0))

    pos_count = int(sum(1 for v in labels if int(v) == 1))
    neg_count = int(sum(1 for v in labels if int(v) == 0))

    meta = {
        "count": len(points),
        "target": int(target),
        "otsu_threshold": round(float(thr), 6),
        "foreground_fraction": round(float(area / float(max(h * w, 1))), 6),
        "complexity": round(float(complexity), 6),
        "min_distance_px": int(min_dist),
        "positive_count": int(pos_count),
        "negative_count": int(neg_count),
        "candidate_counts": {
            "logdog": int(pos_logdog.shape[0]),
            "skeleton": int(pos_skeleton.shape[0]),
            "watershed": int(pos_watershed.shape[0]),
            "hardneg": int(neg_hard.shape[0]),
        },
        "strategy_counts": {k: int(v) for k, v in selected_counts.items()},
    }
    return points, labels, meta


def _window_grid(height: int, width: int, window_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if height <= int(window_size) and width <= int(window_size):
        return [(0, height, 0, width)]

    ov = float(max(0.0, min(0.9, overlap)))
    step = max(1, int(round(window_size * (1.0 - ov))))
    win_h = min(int(window_size), int(height))
    win_w = min(int(window_size), int(width))

    def _coords(size: int, win: int) -> list[int]:
        if size <= win:
            return [0]
        out = list(range(0, size - win + 1, step))
        tail = size - win
        if not out or out[-1] != tail:
            out.append(tail)
        return out

    ys = _coords(int(height), win_h)
    xs = _coords(int(width), win_w)
    return [(y, y + win_h, x, x + win_w) for y in ys for x in xs]


def _select_best_tracker_mask(
    masks: np.ndarray,
    scores: np.ndarray | None,
    *,
    mask_threshold: float,
) -> tuple[np.ndarray, float]:
    def _best_candidate_index(
        candidates: np.ndarray,
        candidate_scores: np.ndarray | None,
        threshold: float,
    ) -> int:
        c = np.asarray(candidates)
        if c.ndim != 3 or c.shape[0] <= 0:
            return 0
        k = int(c.shape[0])
        pixel_count = float(max(1, int(c.shape[1] * c.shape[2])))
        area = (c > threshold).reshape(k, -1).sum(axis=1).astype(np.float64)
        frac = area / pixel_count

        nonempty = np.where(area > 0)[0]
        if nonempty.size <= 0:
            return 0
        sane = nonempty[frac[nonempty] <= 0.98]
        cand = sane if sane.size > 0 else nonempty

        if candidate_scores is not None:
            s_raw = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
            s = np.full((k,), np.nan, dtype=np.float64)
            s[: min(k, s_raw.size)] = s_raw[: min(k, s_raw.size)]
            finite = np.isfinite(s[cand])
            if np.any(finite):
                finite_vals = s[cand][finite]
                if float(np.max(finite_vals) - np.min(finite_vals)) > 1e-6:
                    scored_idx = cand[np.argmax(s[cand])]
                    if area[int(scored_idx)] > 0:
                        return int(scored_idx)

        return int(cand[np.argmax(area[cand])])

    m = np.asarray(masks)
    if m.ndim < 2:
        return np.zeros((1, 1), dtype=np.uint8), 0.0

    s = None if scores is None else np.asarray(scores, dtype=np.float32)
    th = float(mask_threshold)

    if m.ndim == 4:
        # [objects, candidates, H, W]
        obj_masks: list[np.ndarray] = []
        obj_scores: list[float] = []
        for i in range(m.shape[0]):
            if m.shape[1] <= 0:
                continue
            score_row = None if s is None or s.ndim < 2 or i >= s.shape[0] else np.asarray(s[i]).reshape(-1)
            best_j = _best_candidate_index(
                np.asarray(m[i]),
                score_row,
                threshold=th,
            )
            score = (
                float(score_row[best_j])
                if score_row is not None and score_row.size > best_j and np.isfinite(score_row[best_j])
                else 0.0
            )
            obj_masks.append(np.asarray(m[i, best_j]) > th)
            obj_scores.append(score)
        if not obj_masks:
            return np.zeros(m.shape[-2:], dtype=np.uint8), 0.0
        merged = np.any(np.stack(obj_masks, axis=0), axis=0).astype(np.uint8)
        return merged, float(np.mean(obj_scores)) if obj_scores else 0.0

    if m.ndim == 3:
        # [candidates, H, W]
        flat_scores = None if s is None else s.reshape(-1)
        best = _best_candidate_index(
            np.asarray(m),
            flat_scores,
            threshold=th,
        )
        score = (
            float(flat_scores[best])
            if flat_scores is not None and flat_scores.size > best and np.isfinite(flat_scores[best])
            else 0.0
        )
        return (np.asarray(m[best]) > th).astype(np.uint8), score

    # [H, W]
    return (np.asarray(m) > th).astype(np.uint8), 0.0


def _select_tracker_object_masks(
    masks: np.ndarray,
    scores: np.ndarray | None,
    *,
    mask_threshold: float,
    treat_3d_as_objects: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    def _best_candidate_index(
        candidates: np.ndarray,
        candidate_scores: np.ndarray | None,
        threshold: float,
    ) -> int:
        c = np.asarray(candidates)
        if c.ndim != 3 or c.shape[0] <= 0:
            return 0
        k = int(c.shape[0])
        pixel_count = float(max(1, int(c.shape[1] * c.shape[2])))
        area = (c > threshold).reshape(k, -1).sum(axis=1).astype(np.float64)
        frac = area / pixel_count

        nonempty = np.where(area > 0)[0]
        if nonempty.size <= 0:
            return 0
        sane = nonempty[frac[nonempty] <= 0.98]
        cand = sane if sane.size > 0 else nonempty

        if candidate_scores is not None:
            s_raw = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
            s = np.full((k,), np.nan, dtype=np.float64)
            s[: min(k, s_raw.size)] = s_raw[: min(k, s_raw.size)]
            finite = np.isfinite(s[cand])
            if np.any(finite):
                finite_vals = s[cand][finite]
                if float(np.max(finite_vals) - np.min(finite_vals)) > 1e-6:
                    scored_idx = cand[np.argmax(s[cand])]
                    if area[int(scored_idx)] > 0:
                        return int(scored_idx)

        return int(cand[np.argmax(area[cand])])

    m = np.asarray(masks)
    if m.ndim < 2:
        return [], []

    s = None if scores is None else np.asarray(scores, dtype=np.float32)
    th = float(mask_threshold)

    if m.ndim == 4:
        obj_masks: list[np.ndarray] = []
        obj_scores: list[float] = []
        for i in range(m.shape[0]):
            if m.shape[1] <= 0:
                continue
            score_row = (
                None
                if s is None or s.ndim < 2 or i >= s.shape[0]
                else np.asarray(s[i]).reshape(-1)
            )
            best_j = _best_candidate_index(np.asarray(m[i]), score_row, threshold=th)
            score = (
                float(score_row[best_j])
                if score_row is not None
                and score_row.size > best_j
                and np.isfinite(score_row[best_j])
                else 0.0
            )
            obj_masks.append((np.asarray(m[i, best_j]) > th).astype(np.uint8))
            obj_scores.append(score)
        return obj_masks, obj_scores

    if m.ndim == 3:
        if treat_3d_as_objects:
            obj_masks = []
            obj_scores = []
            for i in range(m.shape[0]):
                score = 0.0
                if s is not None:
                    if s.ndim >= 2 and i < s.shape[0]:
                        score_row = np.asarray(s[i]).reshape(-1)
                        finite = score_row[np.isfinite(score_row)]
                        if finite.size > 0:
                            score = float(finite[0])
                    elif s.ndim == 1 and i < s.size and np.isfinite(s[i]):
                        score = float(s[i])
                obj_masks.append((np.asarray(m[i]) > th).astype(np.uint8))
                obj_scores.append(score)
            return obj_masks, obj_scores

        flat_scores = None if s is None else s.reshape(-1)
        best = _best_candidate_index(np.asarray(m), flat_scores, threshold=th)
        score = (
            float(flat_scores[best])
            if flat_scores is not None
            and flat_scores.size > best
            and np.isfinite(flat_scores[best])
            else 0.0
        )
        return [(np.asarray(m[best]) > th).astype(np.uint8)], [score]

    return [(np.asarray(m) > th).astype(np.uint8)], [0.0]


def _compose_instance_label_map(
    object_masks: list[np.ndarray],
    object_scores: list[float] | None = None,
) -> tuple[np.ndarray, list[int], list[float]]:
    if not object_masks:
        return np.zeros((1, 1), dtype=np.uint8), [], []

    first_shape = np.asarray(object_masks[0]).shape
    if len(first_shape) < 2:
        return np.zeros((1, 1), dtype=np.uint8), [], []

    label_dtype = np.uint16 if len(object_masks) > np.iinfo(np.uint8).max else np.uint8
    label_map = np.zeros(first_shape, dtype=label_dtype)
    resolved_sizes: list[int] = []
    resolved_scores: list[float] = []
    raw_scores = list(object_scores or [])

    ranked_indices = sorted(
        range(len(object_masks)),
        key=lambda idx: (
            float(raw_scores[idx])
            if idx < len(raw_scores) and np.isfinite(raw_scores[idx])
            else float("-inf"),
            int(np.count_nonzero(np.asarray(object_masks[idx]) > 0)),
            -idx,
        ),
        reverse=True,
    )

    next_label = 1
    for idx in ranked_indices:
        candidate = np.asarray(object_masks[idx]) > 0
        if candidate.shape != label_map.shape:
            continue
        assignable = candidate & (label_map == 0)
        assign_count = int(np.count_nonzero(assignable))
        if assign_count <= 0:
            continue
        label_map[assignable] = next_label
        resolved_sizes.append(assign_count)
        resolved_scores.append(
            float(raw_scores[idx])
            if idx < len(raw_scores) and np.isfinite(raw_scores[idx])
            else 0.0
        )
        next_label += 1

    return label_map, resolved_sizes, resolved_scores


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_explicit_point_groups(points_value: Any) -> list[list[list[float]]]:
    rows = _to_python_list(points_value)
    if not rows:
        return []
    if (
        isinstance(rows[0], (list, tuple))
        and len(rows[0]) == 2
        and isinstance(rows[0][0], (int, float))
    ):
        rows = [rows]

    normalized: list[list[list[float]]] = []
    for group in rows:
        values = _to_python_list(group) if not isinstance(group, (list, tuple)) else list(group)
        group_points: list[list[float]] = []
        for point in values:
            coords = _to_python_list(point) if not isinstance(point, (list, tuple)) else list(point)
            if len(coords) < 2:
                continue
            try:
                group_points.append([float(coords[0]), float(coords[1])])
            except Exception:
                continue
        if group_points:
            normalized.append(group_points)
    return normalized


def _normalize_explicit_label_groups(
    labels_value: Any,
    *,
    point_groups: list[list[list[float]]],
) -> tuple[list[list[int]] | None, str | None]:
    if not point_groups:
        return None, None
    rows = _to_python_list(labels_value)
    if not rows:
        return None, None

    if isinstance(rows[0], (int, float, str)):
        rows = [rows]

    labels_by_group: list[list[int]] = []
    for index, group in enumerate(rows):
        if index >= len(point_groups):
            break
        raw_values = _to_python_list(group) if not isinstance(group, (list, tuple)) else list(group)
        normalized_row: list[int] = []
        for value in raw_values:
            try:
                normalized_row.append(1 if int(value) > 0 else 0)
            except Exception:
                normalized_row.append(1)
        if len(normalized_row) != len(point_groups[index]):
            return None, "input_labels must match the number of points in each prompt object."
        labels_by_group.append(normalized_row)

    if len(labels_by_group) != len(point_groups):
        return None, "input_labels must match input_points length."
    return labels_by_group, None


def _clamp_explicit_point_groups(
    point_groups: list[list[list[float]]],
    *,
    width: int,
    height: int,
) -> list[list[list[float]]]:
    if not point_groups:
        return []
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    normalized: list[list[list[float]]] = []
    for group in point_groups:
        clamped_group: list[list[float]] = []
        for point in group:
            if len(point) < 2:
                continue
            try:
                x = min(max(float(point[0]), 0.0), max_x)
                y = min(max(float(point[1]), 0.0), max_y)
            except Exception:
                continue
            clamped_group.append([x, y])
        if clamped_group:
            normalized.append(clamped_group)
    return normalized


def segment_array_with_sam3_points(
    array: np.ndarray,
    *,
    order: str,
    input_points: list[Any] | None = None,
    input_labels: list[Any] | None = None,
    mask_threshold: float = 0.5,
    model_id: str | None = None,
    device: str | None = None,
    slice_index: int | None = None,
    preprocess: bool = True,
    allow_remote_download: bool = False,
) -> dict[str, Any]:
    """Run SAM3 tracker inference with explicit point prompts on one slice."""
    if not _torch_available():
        return {
            "success": False,
            "error": "Torch runtime unavailable for SAM3 point inference.",
        }

    arr = np.asarray(array)
    if arr.size <= 0:
        return {"success": False, "error": "Input array is empty."}

    point_groups = _normalize_explicit_point_groups(input_points)
    if not point_groups:
        return {"success": False, "error": "SAM3 point mode requires input_points."}

    label_groups, label_error = _normalize_explicit_label_groups(
        input_labels,
        point_groups=point_groups,
    )
    if label_error:
        return {"success": False, "error": label_error}

    try:
        resolved_device = _resolve_device(device)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    model_ref, warnings = _resolve_model_ref(model_id)
    if not _sam3_tracker_available():
        return {
            "success": False,
            "error": "SAM3 tracker classes unavailable in installed transformers runtime.",
            "warnings": warnings,
        }

    try:
        model, processor, dtype, runtime_warnings = _get_tracker_runtime(
            model_ref,
            resolved_device,
            allow_remote_download=bool(allow_remote_download),
        )
    except Exception as exc:
        return {
            "success": False,
            "error": _sam3_unavailable_message(
                model_ref,
                allow_remote_download=bool(allow_remote_download),
            ),
            "warnings": warnings + [str(exc)],
        }

    slices, slice_order = _canonical_slices(arr, order=str(order))
    if slices.ndim < 3 or int(slices.shape[0]) <= 0:
        return {"success": False, "error": f"Invalid canonical tensor shape: {slices.shape}"}

    if slice_index is None:
        z_idx = int(slices.shape[0] // 2)
    else:
        z_idx = max(0, min(int(slice_index), int(slices.shape[0]) - 1))

    profile = _profile_for_name("generic")
    rgb = _slice_to_rgb_uint8(
        slices[z_idx],
        slice_order,
        preprocess=bool(preprocess),
        profile=profile,
    )
    height = int(rgb.shape[0])
    width = int(rgb.shape[1])

    normalized_points = _clamp_explicit_point_groups(
        point_groups,
        width=width,
        height=height,
    )
    if not normalized_points:
        return {"success": False, "error": "All explicit SAM3 points were invalid after normalization."}

    if label_groups is None:
        normalized_labels = [[1 for _ in group] for group in normalized_points]
    else:
        normalized_labels = label_groups
    if len(normalized_points) != len(normalized_labels):
        return {"success": False, "error": "input_labels must match input_points length."}

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}

    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    inputs = processor(
        images=pil,
        input_points=[normalized_points],
        input_labels=[normalized_labels],
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if key in {"original_sizes", "reshaped_input_sizes"}:
            continue
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=resolved_device, dtype=dtype)
            else:
                inputs[key] = value.to(device=resolved_device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
        apply_non_overlapping_constraints=len(normalized_points) > 1,
    )[0]
    iou_scores = getattr(outputs, "iou_scores", None)
    score_np = None if iou_scores is None else np.asarray(iou_scores.squeeze(0).cpu())
    object_masks, object_scores = _select_tracker_object_masks(
        np.asarray(masks),
        score_np,
        mask_threshold=float(mask_threshold),
        treat_3d_as_objects=True,
    )
    if not object_masks:
        return {
            "success": False,
            "error": "SAM3 point inference returned no masks.",
            "warnings": list(warnings) + list(runtime_warnings),
        }

    instance_label_map, instance_sizes, resolved_scores = _compose_instance_label_map(
        object_masks,
        object_scores,
    )
    if not instance_sizes:
        return {
            "success": False,
            "error": "SAM3 point inference returned empty object masks after non-overlap resolution.",
            "warnings": list(warnings) + list(runtime_warnings),
        }

    nonzero = int(np.count_nonzero(instance_label_map > 0))
    total = int(instance_label_map.size)
    coverage = float(nonzero / float(max(total, 1)) * 100.0)

    return {
        "success": True,
        "backend": "sam3-tracker-points",
        "model_id": Path(model_ref).name if Path(model_ref).exists() else model_ref,
        "resolved_model_ref": model_ref,
        "requested_model_id": model_id,
        "device": resolved_device,
        "slice_count": int(slices.shape[0]),
        "slices_processed": 1,
        "slice_index_used": int(z_idx),
        "input_points": normalized_points,
        "input_point_labels": normalized_labels,
        "mask_threshold": float(mask_threshold),
        "mean_score": (
          round(float(np.mean(resolved_scores)), 6)
          if resolved_scores
          else 0.0
        ),
        "estimated_instances": int(len(instance_sizes)),
        "segmented_voxels": int(nonzero),
        "coverage_percent": round(float(coverage), 4),
        "mask_shape": list(np.asarray(instance_label_map).shape),
        "mask_dtype": str(np.asarray(instance_label_map).dtype),
        "instance_area_voxels": instance_sizes,
        "instance_measurement_scope": "prompt_object_masks",
        "scores": [round(float(score), 6) for score in resolved_scores],
        "warnings": list(warnings) + list(runtime_warnings),
        "_mask": np.asarray(instance_label_map),
    }


def _compose_instance_label_map_from_masks(
    masks_value: Any,
    *,
    mask_threshold: float,
    scores: list[float] | None = None,
) -> tuple[np.ndarray, list[int], list[float]]:
    masks_list = _to_python_list(masks_value)
    if not masks_list:
        return np.zeros((1, 1), dtype=np.uint8), [], []

    object_masks: list[np.ndarray] = []
    for item in masks_list:
        arr = np.asarray(item)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            continue
        if arr.dtype == bool:
            mask_bool = arr
        else:
            mask_bool = np.asarray(arr, dtype=np.float32) >= float(mask_threshold)
        if int(np.count_nonzero(mask_bool)) <= 0:
            continue
        object_masks.append(mask_bool.astype(np.uint8))

    if not object_masks:
        return np.zeros((1, 1), dtype=np.uint8), [], []

    return _compose_instance_label_map(object_masks, scores)


def _instance_mask_sizes(masks_value: Any, *, mask_threshold: float) -> list[int]:
    masks_list = _to_python_list(masks_value)
    if not masks_list:
        return []
    sizes: list[int] = []
    for item in masks_list:
        arr = np.asarray(item)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            continue
        if arr.dtype == bool:
            mask_bool = arr
        else:
            mask_bool = np.asarray(arr, dtype=np.float32) >= float(mask_threshold)
        count = int(np.count_nonzero(mask_bool))
        if count > 0:
            sizes.append(count)
    return sizes


def _normalize_boxes_xyxy(boxes_value: Any) -> list[list[float]]:
    rows = _to_python_list(boxes_value)
    normalized: list[list[float]] = []
    for row in rows:
        values = _to_python_list(row) if not isinstance(row, (list, tuple)) else list(row)
        if len(values) < 4:
            continue
        try:
            normalized.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
        except Exception:
            continue
    return normalized


def _normalize_scores(scores_value: Any) -> list[float]:
    values = _to_python_list(scores_value)
    normalized: list[float] = []
    for value in values:
        try:
            normalized.append(float(value))
        except Exception:
            continue
    return normalized


def _sigmoid_np(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size <= 0:
        return arr.astype(np.float32)
    arr = np.clip(arr, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _extract_sam3_concept_query_scores(outputs: Any) -> np.ndarray | None:
    pred_logits = getattr(outputs, "pred_logits", None)
    if pred_logits is None:
        return None

    pred_np = (
        np.asarray(pred_logits.detach().cpu().numpy())
        if hasattr(pred_logits, "detach")
        else np.asarray(pred_logits)
    )
    pred_np = np.asarray(pred_np, dtype=np.float32)
    if pred_np.ndim <= 0:
        pred_np = pred_np.reshape(1)
    elif pred_np.ndim > 1:
        pred_np = np.asarray(pred_np[0], dtype=np.float32)
    scores = _sigmoid_np(pred_np).reshape(-1)

    presence_logits = getattr(outputs, "presence_logits", None)
    if presence_logits is not None:
        presence_np = (
            np.asarray(presence_logits.detach().cpu().numpy())
            if hasattr(presence_logits, "detach")
            else np.asarray(presence_logits)
        )
        presence_np = np.asarray(presence_np, dtype=np.float32)
        if presence_np.ndim <= 0:
            presence_np = presence_np.reshape(1)
        elif presence_np.ndim > 1:
            presence_np = np.asarray(presence_np[0], dtype=np.float32)
        presence_scores = _sigmoid_np(presence_np).reshape(-1)
        if presence_scores.size > 0:
            scores = scores * float(presence_scores[0])
    return scores.astype(np.float32)


def _fallback_sam3_concept_instances_from_raw_outputs(
    outputs: Any,
    *,
    threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray | None, list[int], list[float], str | None]:
    pred_masks = getattr(outputs, "pred_masks", None)
    if pred_masks is None:
        return None, [], [], "SAM3 concept inference returned no raw masks."

    pred_np = (
        np.asarray(pred_masks.detach().cpu().numpy())
        if hasattr(pred_masks, "detach")
        else np.asarray(pred_masks)
    )
    pred_np = np.squeeze(pred_np)
    query_scores = _extract_sam3_concept_query_scores(outputs)

    if pred_np.ndim == 2:
        if query_scores is not None and query_scores.size > 0 and float(query_scores[0]) <= float(threshold):
            return None, [], [], "no_confident_queries"
        probs = _sigmoid_np(pred_np)
        mask_bool = probs > float(mask_threshold)
        size = int(np.count_nonzero(mask_bool))
        if size <= 0:
            return None, [], [], "no_confident_queries"
        score = (
            float(query_scores[0])
            if query_scores is not None and query_scores.size > 0 and np.isfinite(query_scores[0])
            else 0.0
        )
        return mask_bool.astype(np.uint8), [size], [score], None

    if pred_np.ndim != 3:
        return None, [], [], f"Unsupported SAM3 concept mask shape: {pred_np.shape}"

    probs = _sigmoid_np(pred_np)
    candidate_count = int(probs.shape[0])
    if candidate_count <= 0:
        return None, [], [], "no_confident_queries"

    if query_scores is not None and query_scores.size > 0:
        keep_count = min(candidate_count, int(query_scores.size))
        keep_indices = [
            idx for idx in range(keep_count) if float(query_scores[idx]) > float(threshold)
        ]
        keep_scores = [
            float(query_scores[idx])
            for idx in keep_indices
            if np.isfinite(query_scores[idx])
        ]
    else:
        keep_indices = []
        keep_scores = []

    if not keep_indices:
        return None, [], [], "no_confident_queries"

    object_masks: list[np.ndarray] = []
    object_scores: list[float] = []
    for pos, idx in enumerate(keep_indices):
        mask_bool = probs[idx] > float(mask_threshold)
        if int(np.count_nonzero(mask_bool)) <= 0:
            continue
        object_masks.append(mask_bool.astype(np.uint8))
        object_scores.append(
            keep_scores[pos] if pos < len(keep_scores) and np.isfinite(keep_scores[pos]) else 0.0
        )

    if not object_masks:
        return None, [], [], "no_confident_queries"

    label_map, resolved_sizes, resolved_scores = _compose_instance_label_map(
        object_masks,
        object_scores,
    )
    if not resolved_sizes:
        return None, [], [], "no_confident_queries"

    return label_map, resolved_sizes, resolved_scores, None


def _predict_patch_with_tracker(
    image_rgb: np.ndarray,
    *,
    model: Any,
    processor: Any,
    dtype: Any,
    device: str,
    min_points: int,
    max_points: int,
    point_density: float,
    mask_threshold: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    import torch  # type: ignore

    points, labels, point_meta = _propose_auto_points(
        image_rgb,
        min_points=min_points,
        max_points=max_points,
        point_density=point_density,
    )
    if not labels or len(labels) != len(points):
        labels = [1 for _ in points]

    # [batch, objects, points, xy]
    wrapped_points = [[points]]
    wrapped_labels = [[labels]]

    pil = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    inputs = processor(
        images=pil,
        input_points=wrapped_points,
        input_labels=wrapped_labels,
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if key in ("original_sizes", "reshaped_input_sizes"):
            continue
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    iou_scores = getattr(outputs, "iou_scores", None)
    score_np = None if iou_scores is None else np.asarray(iou_scores.squeeze(0).cpu())
    mask_u8, score = _select_best_tracker_mask(
        np.asarray(masks),
        score_np,
        mask_threshold=float(mask_threshold),
    )
    return mask_u8.astype(np.uint8), float(score), point_meta


def _segment_slice(
    image_rgb: np.ndarray,
    *,
    model: Any,
    processor: Any,
    dtype: Any,
    device: str,
    window_size: int,
    window_overlap: float,
    min_points: int,
    max_points: int,
    point_density: float,
    mask_threshold: float,
    min_component_area_ratio: float,
    vote_threshold: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    windows = _window_grid(h, w, int(window_size), float(window_overlap))

    accum = np.zeros((h, w), dtype=np.float32)
    hits = np.zeros((h, w), dtype=np.float32)
    scores: list[float] = []
    point_counts: list[int] = []
    strategy_totals: dict[str, int] = {}

    for (y0, y1, x0, x1) in windows:
        patch = image_rgb[y0:y1, x0:x1]
        patch_mask, patch_score, point_meta = _predict_patch_with_tracker(
            patch,
            model=model,
            processor=processor,
            dtype=dtype,
            device=device,
            min_points=min_points,
            max_points=max_points,
            point_density=point_density,
            mask_threshold=mask_threshold,
        )
        accum[y0:y1, x0:x1] += patch_mask.astype(np.float32)
        hits[y0:y1, x0:x1] += 1.0
        scores.append(float(patch_score))
        point_counts.append(int(point_meta.get("count", 0)))
        strategy_counts = point_meta.get("strategy_counts") if isinstance(point_meta, dict) else {}
        if isinstance(strategy_counts, dict):
            for key, value in strategy_counts.items():
                try:
                    strategy_totals[str(key)] = strategy_totals.get(str(key), 0) + int(value)
                except Exception:
                    continue

    merged = np.zeros((h, w), dtype=np.uint8)
    valid = hits > 0
    ratio = np.zeros((h, w), dtype=np.float32)
    ratio[valid] = accum[valid] / hits[valid]
    thr = float(max(0.0, min(1.0, vote_threshold)))
    merged[valid] = (ratio[valid] >= thr).astype(np.uint8)
    merge_fallback = False
    if not np.any(merged) and np.any(accum > 0):
        merged[valid] = (ratio[valid] > 0.0).astype(np.uint8)
        merge_fallback = True

    min_area = int(max(4, round(float(min_component_area_ratio) * float(h * w))))
    merged = _remove_small_components(merged > 0, min_area=min_area).astype(np.uint8)

    stats = {
        "window_count": int(len(windows)),
        "avg_points_per_window": round(float(np.mean(point_counts)) if point_counts else 0.0, 4),
        "max_points_per_window": int(max(point_counts) if point_counts else 0),
        "min_points_per_window": int(min(point_counts) if point_counts else 0),
        "min_component_area_px": int(min_area),
        "vote_threshold": round(float(thr), 4),
        "vote_fallback_any_hit": bool(merge_fallback),
        "point_strategy_totals": {k: int(v) for k, v in strategy_totals.items()},
    }
    mean_score = float(np.mean(scores)) if scores else 0.0
    return merged, mean_score, stats


def _sam3_unavailable_message(model_ref: str, allow_remote_download: bool) -> str:
    local_hint = (
        "Place a local SAM3 checkpoint/snapshot directory and set SAM3_MODEL_ID to that path."
    )
    if allow_remote_download:
        return (
            f"Failed to initialize SAM3 model '{model_ref}'. "
            "Check network/auth access to Hugging Face or provide a local model path."
        )
    return (
        f"SAM3 model '{model_ref}' is not available locally and remote downloads are disabled. "
        f"{local_hint}"
    )


def _refine_mask_stack_3d(
    mask_stack: np.ndarray,
    *,
    min_component_area_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    m = np.asarray(mask_stack).astype(bool)
    if m.ndim != 3 or int(m.shape[0]) <= 1:
        return (
            np.asarray(mask_stack).astype(np.uint8),
            {
                "enabled": False,
                "reason": "not_3d",
            },
        )

    before_voxels = int(np.count_nonzero(m))

    # Fill one-slice holes when neighbors agree (temporal bridge).
    bridge = np.zeros_like(m, dtype=bool)
    bridge[1:-1] = np.logical_and(m[:-2], m[2:])
    m = np.logical_or(m, bridge)
    after_bridge_voxels = int(np.count_nonzero(m))

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_before, components_before = ndi.label(m, structure=structure)
    counts_before = np.bincount(labeled_before.ravel()) if components_before > 0 else np.zeros((1,), dtype=np.int64)
    min_component_volume = int(max(8, round(float(min_component_area_ratio) * float(m.size))))

    if components_before > 0:
        keep = counts_before >= int(min_component_volume)
        keep[0] = False

        # Remove very small components that occur in a single z-slice (typical slice flicker noise).
        boxes = ndi.find_objects(labeled_before)
        for idx, box in enumerate(boxes, start=1):
            if box is None:
                continue
            z_span = int(box[0].stop - box[0].start)
            if z_span <= 1 and int(counts_before[idx]) < int(max(min_component_volume * 4, 64)):
                keep[idx] = False

        refined = keep[labeled_before]
    else:
        refined = m

    labeled_after, components_after = ndi.label(refined, structure=structure)
    counts_after = np.bincount(labeled_after.ravel()) if components_after > 0 else np.zeros((1,), dtype=np.int64)
    after_voxels = int(np.count_nonzero(refined))

    removed_voxels = int(max(0, after_bridge_voxels - after_voxels))
    removed_components = int(
        max(
            0,
            int(np.count_nonzero(counts_before[1:] > 0)) - int(np.count_nonzero(counts_after[1:] > 0)),
        )
    )

    stats = {
        "enabled": True,
        "before_voxels": int(before_voxels),
        "after_voxels": int(after_voxels),
        "added_by_temporal_bridge": int(max(0, after_bridge_voxels - before_voxels)),
        "removed_by_component_filter": int(removed_voxels),
        "components_before": int(components_before),
        "components_after": int(components_after),
        "components_removed": int(removed_components),
        "min_component_volume": int(min_component_volume),
    }
    return refined.astype(np.uint8), stats


def segment_array_with_sam3_concept(
    array: np.ndarray,
    *,
    order: str,
    concept_prompt: str | None = None,
    input_boxes: list[list[float]] | None = None,
    input_boxes_labels: list[int] | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    model_id: str | None = None,
    device: str | None = None,
    slice_index: int | None = None,
    preprocess: bool = True,
    allow_remote_download: bool = False,
) -> dict[str, Any]:
    """Run SAM3 concept segmentation on a single representative slice.

    Supports text prompts and/or positive/negative box prompts.
    For volumetric inputs, the middle Z slice is used by default unless slice_index is provided.
    """
    if not _torch_available():
        return {"success": False, "error": "Torch runtime unavailable for SAM3 concept inference."}

    arr = np.asarray(array)
    if arr.size <= 0:
        return {"success": False, "error": "Input array is empty."}

    text_prompt = str(concept_prompt or "").strip()
    normalized_boxes: list[list[float]] = []
    for row in input_boxes or []:
        values = _to_python_list(row) if not isinstance(row, (list, tuple)) else list(row)
        if len(values) < 4:
            continue
        try:
            normalized_boxes.append(
                [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
            )
        except Exception:
            continue

    if not text_prompt and not normalized_boxes:
        return {
            "success": False,
            "error": "SAM3 concept mode requires concept_prompt and/or input_boxes.",
        }

    labels: list[int] = []
    if normalized_boxes:
        if input_boxes_labels is None:
            labels = [1 for _ in normalized_boxes]
        else:
            labels = []
            for value in input_boxes_labels:
                try:
                    labels.append(1 if int(value) > 0 else 0)
                except Exception:
                    labels.append(1)
            if len(labels) != len(normalized_boxes):
                return {
                    "success": False,
                    "error": "input_boxes_labels length must match input_boxes length.",
                }

    try:
        resolved_device = _resolve_device(device)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    model_ref, warnings = _resolve_model_ref(model_id)
    if not _sam3_concept_available():
        return {
            "success": False,
            "error": "SAM3 concept classes unavailable in installed transformers runtime.",
            "warnings": warnings,
        }

    try:
        model, processor, dtype, runtime_warnings = _get_concept_runtime(
            model_ref,
            resolved_device,
            allow_remote_download=bool(allow_remote_download),
        )
    except Exception as exc:
        return {
            "success": False,
            "error": _sam3_unavailable_message(
                model_ref, allow_remote_download=bool(allow_remote_download)
            ),
            "warnings": list(warnings) + [str(exc)],
        }

    slices, slice_order = _canonical_slices(arr, order=str(order))
    if slices.ndim < 3 or int(slices.shape[0]) <= 0:
        return {"success": False, "error": f"Invalid canonical tensor shape: {slices.shape}"}

    if slice_index is None:
        z_idx = int(slices.shape[0] // 2)
    else:
        z_idx = max(0, min(int(slice_index), int(slices.shape[0]) - 1))

    profile = _profile_for_name("generic")
    rgb = _slice_to_rgb_uint8(
        slices[z_idx],
        slice_order,
        preprocess=bool(preprocess),
        profile=profile,
    )

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}

    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    inputs_kwargs: dict[str, Any] = {
        "images": pil,
        "return_tensors": "pt",
    }
    if text_prompt:
        inputs_kwargs["text"] = text_prompt
    if normalized_boxes:
        inputs_kwargs["input_boxes"] = [normalized_boxes]
        inputs_kwargs["input_boxes_labels"] = [labels]

    inputs = processor(**inputs_kwargs)
    for key, value in list(inputs.items()):
        if key in {"original_sizes", "reshaped_input_sizes"}:
            continue
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=resolved_device, dtype=dtype)
            else:
                inputs[key] = value.to(device=resolved_device)

    with torch.no_grad():
        outputs = model(**inputs)

    all_warnings = list(warnings) + list(runtime_warnings)
    target_sizes = inputs.get("original_sizes")
    if hasattr(target_sizes, "tolist"):
        target_sizes = target_sizes.tolist()

    parsed_masks: np.ndarray | None = None
    parsed_instance_count = 0
    parsed_instance_sizes: list[int] = []
    parsed_boxes: list[list[float]] = []
    parsed_scores: list[float] = []

    try:
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            target_sizes=target_sizes,
        )
        first = post[0] if isinstance(post, list) and post else None
        if isinstance(first, dict):
            label_map, resolved_sizes, resolved_scores = _compose_instance_label_map_from_masks(
                first.get("masks"),
                mask_threshold=float(mask_threshold),
                scores=_normalize_scores(first.get("scores")),
            )
            if label_map.size > 1:
                parsed_masks = label_map
                parsed_instance_sizes = [int(value) for value in resolved_sizes if int(value) > 0]
                parsed_instance_count = int(len(parsed_instance_sizes))
                parsed_scores = [float(value) for value in resolved_scores]
            parsed_boxes = _normalize_boxes_xyxy(first.get("boxes"))
            if not parsed_scores:
                parsed_scores = _normalize_scores(first.get("scores"))
    except Exception as exc:
        all_warnings.append(f"Instance post-process fallback used: {exc}")

    if parsed_masks is None:
        fallback_masks, fallback_sizes, fallback_scores, fallback_error = (
            _fallback_sam3_concept_instances_from_raw_outputs(
                outputs,
                threshold=float(threshold),
                mask_threshold=float(mask_threshold),
            )
        )
        if fallback_masks is None:
            guidance = (
                "SAM3 concept prompt produced no confident masks at the current threshold. "
                "For geometric regions or simple synthetic shapes, prefer explicit input_boxes or input_points."
            )
            error_message = guidance if fallback_error == "no_confident_queries" else (
                fallback_error or "SAM3 concept inference returned no masks."
            )
            return {
                "success": False,
                "error": error_message,
                "warnings": all_warnings,
            }
        parsed_masks = fallback_masks
        parsed_instance_sizes = [int(value) for value in fallback_sizes if int(value) > 0]
        parsed_scores = [float(value) for value in fallback_scores if np.isfinite(value)]
        parsed_instance_count = int(len(parsed_instance_sizes))

    nonzero = int(np.count_nonzero(parsed_masks))
    total = int(parsed_masks.size)
    coverage = float(nonzero / float(max(total, 1)) * 100.0)
    component_count = int(ndi.label(parsed_masks > 0)[1]) if nonzero > 0 else 0
    mean_score = float(np.mean(parsed_scores)) if parsed_scores else 0.0

    return {
        "success": True,
        "backend": "sam3-concept",
        "model_id": Path(model_ref).name if Path(model_ref).exists() else model_ref,
        "resolved_model_ref": model_ref,
        "requested_model_id": model_id,
        "device": resolved_device,
        "slice_count": int(slices.shape[0]),
        "slices_processed": 1,
        "slice_index_used": int(z_idx),
        "concept_prompt": text_prompt or None,
        "input_boxes": normalized_boxes,
        "input_boxes_labels": labels,
        "threshold": float(threshold),
        "mask_threshold": float(mask_threshold),
        "mean_score": round(float(mean_score), 6),
        "estimated_instances": int(parsed_instance_count if parsed_instance_count > 0 else component_count),
        "segmented_voxels": int(nonzero),
        "coverage_percent": round(float(coverage), 4),
        "mask_shape": list(np.asarray(parsed_masks).shape),
        "mask_dtype": str(np.asarray(parsed_masks).dtype),
        "instance_area_voxels": [int(v) for v in parsed_instance_sizes],
        "instance_measurement_scope": (
            "model_instance_masks"
            if parsed_instance_sizes
            else "connected_components_of_final_mask"
        ),
        "boxes_xyxy": parsed_boxes,
        "scores": parsed_scores,
        "warnings": all_warnings,
        "_mask": np.asarray(parsed_masks).astype(np.uint8),
    }


def segment_array_with_sam3(
    array: np.ndarray,
    *,
    order: str,
    model_id: str | None = None,
    device: str | None = None,
    max_slices: int = 192,
    window_size: int = 1024,
    window_overlap: float = 0.25,
    min_points: int = 8,
    max_points: int = 64,
    point_density: float = 0.0015,
    mask_threshold: float = 0.5,
    min_component_area_ratio: float = 0.0001,
    vote_threshold: float = 0.5,
    modality_hint: str | None = "auto",
    preprocess: bool = True,
    refine_3d: bool = True,
    fallback_to_medsam2: bool = True,
    allow_remote_download: bool = False,
) -> dict[str, Any]:
    """Run automatic first-pass segmentation with SAM3 tracker (slice-wise).

    When SAM3 is unavailable, optionally falls back to MedSAM2.
    """
    if not _torch_available():
        return {"success": False, "error": "Torch runtime unavailable for SAM3 inference."}

    arr = np.asarray(array)
    if arr.size <= 0:
        return {"success": False, "error": "Input array is empty."}

    try:
        resolved_device = _resolve_device(device)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    model_ref, warnings = _resolve_model_ref(model_id)

    if not _sam3_tracker_available():
        if fallback_to_medsam2:
            fb = segment_array_with_medsam2(
                arr,
                order=order,
                model_id=None,
                device=resolved_device,
                max_slices=max_slices,
            )
            if fb.get("success"):
                fb["backend"] = "sam3-fallback-medsam2"
                fb["requested_model_id"] = model_id
                fb["resolved_model_ref"] = model_ref
                fb["warnings"] = [
                    "SAM3 tracker classes unavailable in installed transformers runtime; used MedSAM2 fallback."
                ] + warnings + list(fb.get("warnings") or [])
                return fb
            return {
                "success": False,
                "error": (
                    "SAM3 tracker classes unavailable and MedSAM2 fallback also failed: "
                    f"{fb.get('error') or 'unknown MedSAM2 error'}"
                ),
                "warnings": warnings + list(fb.get("warnings") or []),
            }
        return {
            "success": False,
            "error": "SAM3 tracker classes unavailable in installed transformers runtime.",
            "warnings": warnings,
        }

    try:
        model, processor, dtype, init_warnings = _get_tracker_runtime(
            model_ref,
            resolved_device,
            allow_remote_download=bool(allow_remote_download),
        )
        runtime = _Sam3Runtime(
            backend="sam3-tracker",
            model_ref=model_ref,
            device=resolved_device,
            warnings=tuple(warnings) + tuple(init_warnings),
        )
    except Exception as exc:
        if fallback_to_medsam2:
            fb = segment_array_with_medsam2(
                arr,
                order=order,
                model_id=None,
                device=resolved_device,
                max_slices=max_slices,
            )
            if fb.get("success"):
                fb["backend"] = "sam3-fallback-medsam2"
                fb["requested_model_id"] = model_id
                fb["resolved_model_ref"] = model_ref
                fb["warnings"] = [
                    _sam3_unavailable_message(model_ref, allow_remote_download=bool(allow_remote_download)),
                    f"SAM3 init error: {exc}",
                ] + warnings + list(fb.get("warnings") or [])
                return fb
            return {
                "success": False,
                "error": (
                    f"{_sam3_unavailable_message(model_ref, allow_remote_download=bool(allow_remote_download))} "
                    f"MedSAM2 fallback failed: {fb.get('error') or 'unknown MedSAM2 error'}"
                ),
                "warnings": warnings
                + [f"SAM3 init error: {exc}"]
                + list(fb.get("warnings") or []),
            }
        return {
            "success": False,
            "error": _sam3_unavailable_message(model_ref, allow_remote_download=bool(allow_remote_download)),
            "warnings": warnings + [str(exc)],
        }

    slices, slice_order = _canonical_slices(arr, order=str(order))
    if slices.ndim < 3:
        return {"success": False, "error": f"Invalid canonical tensor shape: {slices.shape}"}

    profile_name, profile_stats, profile_warnings = _infer_modality_profile(
        slices,
        slice_order,
        hint=modality_hint,
    )
    profile = _profile_for_name(profile_name)

    z_count = int(slices.shape[0])
    if z_count <= 0:
        return {"success": False, "error": "No slices available for segmentation."}

    stride = 1
    if z_count > int(max_slices):
        stride = int(np.ceil(z_count / float(max_slices)))
    indices = list(range(0, z_count, stride))

    mask_stack = np.zeros((z_count, int(slices.shape[-2]), int(slices.shape[-1])), dtype=np.uint8)
    score_by_slice: dict[int, float] = {}
    slice_stats: dict[int, dict[str, Any]] = {}
    runtime_warnings: list[str] = list(runtime.warnings) + list(profile_warnings)

    for z in indices:
        rgb = _slice_to_rgb_uint8(
            slices[z],
            slice_order,
            preprocess=bool(preprocess),
            profile=profile,
        )
        try:
            pred_mask, pred_score, stats = _segment_slice(
                rgb,
                model=model,
                processor=processor,
                dtype=dtype,
                device=runtime.device,
                window_size=int(window_size),
                window_overlap=float(window_overlap),
                min_points=int(min_points),
                max_points=int(max_points),
                point_density=float(point_density),
                mask_threshold=float(mask_threshold),
                min_component_area_ratio=float(min_component_area_ratio),
                vote_threshold=float(vote_threshold),
            )
            mask_stack[z] = pred_mask.astype(np.uint8)
            score_by_slice[z] = float(pred_score)
            slice_stats[z] = stats
        except Exception as exc:
            runtime_warnings.append(f"slice {z}: {exc}")

    if not score_by_slice:
        return {
            "success": False,
            "error": "SAM3 inference failed for all slices.",
            "warnings": runtime_warnings,
        }

    # Fill skipped slices by nearest segmented slice.
    if stride > 1 and indices:
        for z in range(z_count):
            if z in score_by_slice:
                continue
            nearest = min(indices, key=lambda k: abs(k - z))
            mask_stack[z] = mask_stack[nearest]
            score_by_slice[z] = score_by_slice.get(nearest, 0.0)
            if nearest in slice_stats:
                slice_stats[z] = dict(slice_stats[nearest])

    refine_stats: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
    }
    if bool(refine_3d) and z_count > 1:
        mask_stack, refine_stats = _refine_mask_stack_3d(
            mask_stack,
            min_component_area_ratio=float(min_component_area_ratio),
        )
    elif z_count <= 1:
        refine_stats = {"enabled": False, "reason": "not_3d"}

    if z_count == 1:
        mask_out = mask_stack[0]
    else:
        mask_out = mask_stack

    nonzero = int(np.count_nonzero(mask_out))
    total = int(mask_out.size)
    coverage = float(nonzero / float(max(total, 1)) * 100.0)
    mean_score = float(np.mean(list(score_by_slice.values()))) if score_by_slice else 0.0
    component_sizes: list[int] = []
    est_instances = 0
    if nonzero > 0:
        labels, est_instances = ndi.label(mask_out > 0)
        if int(est_instances) > 0:
            counts = np.bincount(labels.ravel())[1:]
            component_sizes = [int(item) for item in counts if int(item) > 0]

    if slice_stats:
        avg_points = float(np.mean([s.get("avg_points_per_window", 0.0) for s in slice_stats.values()]))
        avg_windows = float(np.mean([s.get("window_count", 1) for s in slice_stats.values()]))
        total_windows = int(sum(max(1, int(s.get("window_count", 1))) for s in slice_stats.values()))
        strategy_totals: dict[str, int] = {}
        for s in slice_stats.values():
            strat = s.get("point_strategy_totals") if isinstance(s, dict) else {}
            if not isinstance(strat, dict):
                continue
            for key, value in strat.items():
                try:
                    strategy_totals[str(key)] = strategy_totals.get(str(key), 0) + int(value)
                except Exception:
                    continue
        strategy_avg = {
            k: round(float(v) / float(max(total_windows, 1)), 4)
            for k, v in strategy_totals.items()
        }
    else:
        avg_points = 0.0
        avg_windows = 0.0
        strategy_avg = {}

    return {
        "success": True,
        "backend": runtime.backend,
        "model_id": Path(runtime.model_ref).name if Path(runtime.model_ref).exists() else runtime.model_ref,
        "resolved_model_ref": runtime.model_ref,
        "requested_model_id": model_id,
        "device": runtime.device,
        "slice_count": int(z_count),
        "slice_stride": int(stride),
        "slices_processed": int(len(indices)),
        "window_size": int(window_size),
        "window_overlap": float(window_overlap),
        "vote_threshold": float(vote_threshold),
        "modality_profile": profile_name,
        "modality_profile_stats": profile_stats,
        "preprocess_profile": {
            "name": profile.name,
            "clip_lo_pct": float(profile.clip_lo_pct),
            "clip_hi_pct": float(profile.clip_hi_pct),
            "denoise_sigma": float(profile.denoise_sigma),
            "tophat_radius": int(profile.tophat_radius),
            "local_contrast_window": int(profile.local_contrast_window),
            "gamma": float(profile.gamma),
            "unsharp_amount": float(profile.unsharp_amount),
        },
        "mean_score": round(float(mean_score), 6),
        "estimated_instances": int(est_instances),
        "segmented_voxels": int(nonzero),
        "coverage_percent": round(float(coverage), 4),
        "refine_3d": refine_stats,
        "auto_point_summary": {
            "avg_points_per_window": round(float(avg_points), 4),
            "avg_windows_per_slice": round(float(avg_windows), 4),
            "min_points": int(min_points),
            "max_points": int(max_points),
            "point_density": float(point_density),
            "strategy_points_per_window": strategy_avg,
        },
        "mask_shape": list(np.asarray(mask_out).shape),
        "mask_dtype": str(np.asarray(mask_out).dtype),
        "instance_area_voxels": component_sizes,
        "instance_measurement_scope": "connected_components_of_final_mask",
        "warnings": runtime_warnings,
        "_mask": np.asarray(mask_out).astype(np.uint8),
        "_slice_scores": score_by_slice,
        "_slice_stats": slice_stats,
    }


__all__ = [
    "segment_array_with_sam3",
    "segment_array_with_sam3_concept",
    "segment_array_with_sam3_points",
]
