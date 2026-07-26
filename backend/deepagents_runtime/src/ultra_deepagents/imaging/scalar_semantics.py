"""Bounded, deterministic source profiling for scalar image volumes.

The profiler is deliberately source-authoritative: callers pass the normalized
decoder source, and both viewer-info and the public volume histogram consume the
same result.  It never profiles a display pyramid or a rendered PNG.
"""

from __future__ import annotations

import math
import os
from typing import Any

PROFILE_ALGORITHM = "scalar-profile-otsu-256-v1"
MAX_PROFILE_Z_PLANES = 32
MAX_PROFILE_DECODED_BYTES = 32 * 1024 * 1024
MAX_PROFILE_DECODE_WORK_BYTES = 128 * 1024 * 1024
MAX_PROFILE_SAMPLES = 1_048_576
MAX_PROFILE_SPATIAL_REGIONS = 5
MAX_PROFILE_READS = MAX_PROFILE_Z_PLANES * MAX_PROFILE_SPATIAL_REGIONS


def _stratified_indices(size: int, count: int) -> list[int]:
    if size <= 1:
        return [0]
    count = max(1, min(size, count))
    if count == size:
        return list(range(size))
    return sorted(
        {min(size - 1, max(0, int(round(i * (size - 1) / (count - 1))))) for i in range(count)}
    )


def profile_z_indices(size: int, count: int = MAX_PROFILE_Z_PLANES) -> list[int]:
    """Return a bounded Z plan that always includes both ends and the exact centre."""
    if size <= 1:
        return [0]
    required = {0, size // 2, size - 1}
    target = max(len(required), min(size, max(1, count)))
    candidates = _stratified_indices(size, target)
    selected = set(required)
    for candidate in candidates:
        if len(selected) >= target:
            break
        selected.add(candidate)
    return sorted(selected)


def _plane_sample_regions(
    source: Any,
    *,
    byte_budget: int,
    itemsize: int,
) -> list[tuple[int, int, int, int] | None]:
    max_voxels = max(1, byte_budget // itemsize)
    total = int(source.x) * int(source.y)
    if total <= max_voxels:
        return [None]

    crop_voxels = max(1, max_voxels // MAX_PROFILE_SPATIAL_REGIONS)
    crop_w = max(1, min(int(source.x), int(math.sqrt(crop_voxels))))
    crop_h = max(1, min(int(source.y), crop_voxels // crop_w))
    # The exact centre is load-bearing: a compact object centered in a mostly
    # empty field must be represented. Four quarter-strata retain broad coverage.
    anchors = (
        (0.5, 0.5),
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    )
    regions: list[tuple[int, int, int, int] | None] = []
    for y_frac, x_frac in anchors:
        y0 = min(max(0, int(round(source.y * y_frac - crop_h / 2))), source.y - crop_h)
        x0 = min(max(0, int(round(source.x * x_frac - crop_w / 2))), source.x - crop_w)
        regions.append((y0, y0 + crop_h, x0, x0 + crop_w))
    return regions


def profile_decode_plan(
    source: Any,
) -> tuple[list[int], list[tuple[int, int, int, int] | None]]:
    """Return the bounded source-read plan shared by admission and profiling."""
    import numpy as np

    z_indices = profile_z_indices(int(source.z), MAX_PROFILE_Z_PLANES)
    per_plane_budget = max(1, MAX_PROFILE_DECODED_BYTES // len(z_indices))
    itemsize = max(1, int(np.dtype(source.dtype).itemsize))
    regions = _plane_sample_regions(
        source,
        byte_budget=per_plane_budget,
        itemsize=itemsize,
    )
    if len(z_indices) * len(regions) > MAX_PROFILE_READS:
        raise ValueError("scalar profile exceeds its bounded read envelope")
    return z_indices, regions


def _bounded_plane_sample(
    source: Any,
    *,
    t: int,
    c: int,
    z: int,
    byte_budget: int,
    regions: list[tuple[int, int, int, int] | None],
) -> tuple[Any, int, int]:
    import numpy as np

    samples = []
    decoded = 0
    read_count = 0
    for region in regions:
        kwargs: dict[str, Any] = {"t": t, "c": c, "z": z, "level": 0}
        if region is not None:
            kwargs["box"] = region
        array = np.asarray(source.read(**kwargs))
        returned_bytes = int(array.nbytes)
        if returned_bytes > byte_budget:
            raise ValueError("decoder returned an oversized scalar profile region")
        samples.append(array.reshape(-1))
        decoded += returned_bytes
        read_count += 1
    if decoded > byte_budget:
        raise ValueError("decoder returned more scalar profile bytes than requested")
    return np.concatenate(samples), decoded, read_count


def imagej_otsu_first_max(counts: Any) -> int:
    """Return ImageJ/Fiji's Otsu threshold bin with first-maximum tie semantics."""
    import numpy as np

    histogram = np.asarray(counts, dtype="float64")
    if histogram.ndim != 1 or histogram.size == 0 or float(histogram.sum()) <= 0:
        return 0
    indices = np.arange(histogram.size, dtype="float64")
    total = float(histogram.sum())
    total_mean = float((indices * histogram).sum())
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 0
    for threshold in range(histogram.size):
        background_weight += float(histogram[threshold])
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break
        background_sum += threshold * float(histogram[threshold])
        background_mean = background_sum / background_weight
        foreground_mean = (total_mean - background_sum) / foreground_weight
        between = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        # Strictly greater is intentional: ImageJ keeps the first maximum.
        if between > best_variance:
            best_variance = between
            best_threshold = threshold
    return best_threshold


def _histogram(samples: Any, dtype: Any, bins: int) -> tuple[Any, Any, float]:
    import numpy as np

    values = np.asarray(samples)
    source_dtype = np.dtype(dtype)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if bins == 256 and source_dtype.kind == "u" and source_dtype.itemsize == 1:
        counts = np.bincount(values.astype("uint8", copy=False), minlength=256)
        edges = np.arange(257, dtype="float64") - 0.5
        threshold = float(imagej_otsu_first_max(counts))
        return counts, edges, threshold
    if maximum <= minimum:
        edges = np.linspace(minimum - 0.5, maximum + 0.5, bins + 1)
        counts, edges = np.histogram(values, bins=edges)
        return counts, edges, minimum
    counts, edges = np.histogram(values.astype("float64", copy=False), bins=bins)
    threshold_index = imagej_otsu_first_max(counts)
    threshold = strict_above_threshold(edges, threshold_index, source_dtype)
    return counts, edges, threshold


def display_histogram(
    samples: list[tuple[int, Any]],
    *,
    dtype: Any,
    bins: int,
    t: int,
    algorithm: str,
) -> dict[str, Any]:
    """Build a bounded display histogram with common edges for all channels.

    This response is intentionally not threshold authority: it describes one
    representative display plane and carries no Otsu threshold or semantic
    recommendation. Common edges let the control plane combine selected
    channels without silently adding incompatible bins.
    """
    import numpy as np

    if not samples or isinstance(bins, bool) or int(bins) <= 0:
        raise ValueError("display histogram selection is empty or invalid")
    finite_samples: list[tuple[int, Any]] = []
    minimum = math.inf
    maximum = -math.inf
    for channel, raw in samples:
        values = np.asarray(raw).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("display histogram channel contains no finite samples")
        minimum = min(minimum, float(finite.min()))
        maximum = max(maximum, float(finite.max()))
        finite_samples.append((int(channel), finite))
    edges = (
        np.linspace(minimum - 0.5, maximum + 0.5, int(bins) + 1)
        if maximum <= minimum
        else np.linspace(minimum, maximum, int(bins) + 1)
    )
    channels = []
    total_samples = 0
    for channel, values in finite_samples:
        counts = np.histogram(values.astype("float64", copy=False), bins=edges)[0]
        sample_count = int(values.size)
        total_samples += sample_count
        channels.append(
            {
                "index": channel,
                "counts": [int(value) for value in counts],
                "edges": [float(value) for value in edges],
                "min": float(values.min()),
                "max": float(values.max()),
                "sample_count": sample_count,
            }
        )
    return {
        "bins": int(bins),
        "dtype": str(np.dtype(dtype)),
        "channels": channels,
        "t": int(t),
        "scope": "display",
        "sample_count": total_samples,
        "sampling": {
            "algorithm": str(algorithm),
            "scope": "display",
            "strategy": "representative-plane",
            "sample_count": total_samples,
            "read_count": len(channels),
        },
    }


def strict_above_threshold(edges: Any, threshold_index: int, dtype: Any) -> float:
    """Map an Otsu bin to a raw ``sample > threshold`` comparison exactly.

    Integer samples above the selected bin begin at ``ceil(upper_edge)``.
    Float32 samples begin at the smallest representable value greater than or
    equal to that edge, so the comparison threshold is its predecessor.
    """
    import numpy as np

    source_dtype = np.dtype(dtype)
    upper_edge = float(edges[min(int(threshold_index) + 1, len(edges) - 1)])
    if source_dtype.kind in {"u", "i"}:
        limits: Any = np.iinfo(source_dtype)
        return float(min(limits.max, max(limits.min, math.ceil(upper_edge) - 1)))
    if source_dtype.kind == "f" and source_dtype.itemsize == 4:
        first_above = np.float32(upper_edge)
        if float(first_above) < upper_edge:
            first_above = np.nextafter(first_above, np.float32(math.inf), dtype=np.float32)
        return float(np.nextafter(first_above, np.float32(-math.inf), dtype=np.float32))
    raise ValueError("source dtype cannot preserve exact mask membership")


def mask_membership_dtype(dtype: Any) -> str | None:
    """Return the exact client/server mask dtype, or None for lossy delivery."""
    import numpy as np

    source_dtype = np.dtype(dtype)
    return {
        ("u", 1): "uint8",
        ("u", 2): "uint16",
        ("i", 2): "int16",
    }.get((source_dtype.kind, source_dtype.itemsize))


def canonical_mask_threshold(value: float, dtype: Any) -> float:
    """Canonicalize once to the comparison domain shared by CPU and GLSL."""
    import numpy as np

    source_dtype = np.dtype(dtype)
    dtype_name = mask_membership_dtype(source_dtype)
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("source dtype cannot preserve exact mask membership")
    if source_dtype.kind == "f" and source_dtype.itemsize == 4:
        # Float32 masking is withheld for now because NaN/Inf membership cannot
        # be made identical across PNG slices, CPU extraction, and WebGL.
        # Preserve the caller's finite value here rather than silently rounding it.
        return threshold
    if dtype_name is None:
        raise ValueError("source dtype cannot preserve exact mask membership")
    limits: Any = np.iinfo(source_dtype)
    return float(min(limits.max, max(int(limits.min) - 1, math.floor(threshold))))


def _probability_mask_suggestion(samples: Any, threshold: float) -> bool:
    """Conservative mask heuristic; never interprets integer values as labels."""
    import numpy as np

    values = np.asarray(samples, dtype="float64")
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0:
        return False
    normalized_threshold = (threshold - minimum) / span
    low_fraction = float(np.mean(values <= minimum + span * 0.03))
    foreground_fraction = float(np.mean(values > threshold))
    high_extreme_fraction = float(np.mean(values >= minimum + span * 0.97))
    intermediate_fraction = float(
        np.mean((values > minimum + span * 0.05) & (values < minimum + span * 0.95))
    )
    occupied_codes = int(np.unique(values).size)
    # Extreme-heavy, sparse, and meaningfully bimodal. The high plateau plus
    # intermediate probability mass distinguish calibrated probability outputs
    # from sparse fluorescence and ordinary tomography with a dark background.
    return (
        occupied_codes > 2
        and low_fraction >= 0.60
        and 0.002 <= foreground_fraction <= 0.30
        and high_extreme_fraction >= 0.05
        and intermediate_fraction >= 0.005
        and 0.15 <= normalized_threshold <= 0.85
    )


def profile_scalar_volume(
    source: Any,
    path: str,
    *,
    channel: int,
    t: int,
    bins: int = 256,
) -> dict[str, Any]:
    """Profile one exact C/T selection under fixed decode and sample budgets."""
    import numpy as np

    z_indices, regions = profile_decode_plan(source)
    per_plane_budget = max(1, MAX_PROFILE_DECODED_BYTES // len(z_indices))
    decoded_chunk_bytes = getattr(source, "max_decoded_chunk_bytes", None)
    if (
        isinstance(decoded_chunk_bytes, bool)
        or not isinstance(decoded_chunk_bytes, int)
        or decoded_chunk_bytes <= 0
    ):
        raise ValueError("scalar profile decoded chunk geometry is unavailable")
    estimate_read_work = getattr(source, "estimate_read_work", None)
    if not callable(estimate_read_work):
        raise ValueError("scalar profile decoded chunk geometry is unavailable")
    planned_work: list[int] = []
    for z in z_indices:
        for region in regions:
            work = estimate_read_work(t=t, c=channel, z=z, level=0, box=region)
            if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
                raise ValueError("scalar profile decoded chunk geometry is invalid")
            planned_work.append(work)
    admitted_decode_work_bytes = sum(planned_work)
    if admitted_decode_work_bytes > MAX_PROFILE_DECODE_WORK_BYTES:
        raise ValueError("scalar profile decode work exceeds its bounded envelope")
    chunks = []
    returned_bytes = 0
    read_count = 0
    for z in z_indices:
        chunk, returned, reads = _bounded_plane_sample(
            source,
            t=t,
            c=channel,
            z=z,
            byte_budget=per_plane_budget,
            regions=regions,
        )
        chunks.append(chunk)
        returned_bytes += returned
        read_count += reads
        if returned_bytes > MAX_PROFILE_DECODED_BYTES or read_count > MAX_PROFILE_READS:
            raise ValueError("scalar profile exceeded its bounded read envelope")
    values = np.concatenate(chunks)
    if values.size > MAX_PROFILE_SAMPLES:
        indices = np.linspace(0, values.size - 1, MAX_PROFILE_SAMPLES, dtype="int64")
        values = values[indices]
    exact = (
        len(z_indices) == int(source.z)
        and int(source.x) * int(source.y) * int(source.z) <= MAX_PROFILE_SAMPLES
        and returned_bytes <= MAX_PROFILE_DECODED_BYTES
    )
    counts, edges, threshold_value = _histogram(values, source.dtype, int(bins))
    unique = np.unique(values)
    is_two_code = unique.size == 2
    membership_dtype = mask_membership_dtype(source.dtype)
    if membership_dtype is not None and exact and is_two_code:
        kind = "binary_mask"
        strength = "exact"
    elif membership_dtype is not None and is_two_code:
        kind = "binary_mask"
        strength = "suggested"
    elif membership_dtype is not None and _probability_mask_suggestion(values, threshold_value):
        kind = "probability_mask"
        strength = "suggested"
    else:
        kind = "intensity"
        strength = "unknown"
    sample_scope = "volume" if exact else "stratified_z"
    threshold = {
        "method": "otsu-256-v1",
        "value": float(threshold_value),
        "domain": "raw",
        "foreground": "above",
        "sample_scope": sample_scope,
        "sample_count": int(values.size),
        "z_samples": z_indices,
        "channel": int(channel),
        "t": int(t),
        "sampling_algorithm": PROFILE_ALGORITHM,
    }
    mask_capable = kind != "intensity"
    auto_mask = kind == "binary_mask" and strength == "exact"
    semantics = {
        "kind": kind,
        "basis": "bounded_scalar_profile",
        "strength": strength,
        "supported_modes": ["intensity", "mask"] if mask_capable else ["intensity"],
        "recommended_view": "mask" if auto_mask else "intensity",
        "threshold": threshold,
    }
    return {
        "bins": int(bins),
        "dtype": str(np.dtype(source.dtype)),
        "channels": [
            {
                "index": int(channel),
                "counts": [int(value) for value in counts],
                "edges": [float(value) for value in edges],
                "min": float(values.min()),
                "max": float(values.max()),
            }
        ],
        "channel": int(channel),
        "t": int(t),
        "scope": "volume",
        "sample_count": int(values.size),
        "sampling": {
            "algorithm": PROFILE_ALGORITHM,
            "scope": "volume",
            "strategy": "exact" if exact else "stratified-z-spatial",
            "sample_count": int(values.size),
            "returned_bytes": int(returned_bytes),
            "read_count": int(read_count),
            "max_reads": MAX_PROFILE_READS,
            "max_returned_bytes": MAX_PROFILE_DECODED_BYTES,
            "declared_max_decoded_chunk_bytes": decoded_chunk_bytes,
            "admitted_decode_work_bytes": int(admitted_decode_work_bytes),
            "max_decode_work_bytes": MAX_PROFILE_DECODE_WORK_BYTES,
            "z_samples": z_indices,
        },
        "threshold": threshold,
        "data_semantics": semantics,
    }


def profile_cache_key(path: str, channel: int, t: int, bins: int) -> tuple[Any, ...]:
    try:
        stat = os.stat(path)
        identity = (
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
            getattr(stat, "st_ino", 0),
            getattr(stat, "st_dev", 0),
        )
    except OSError:
        identity = (0, 0, 0, 0, 0)
    return (
        str(path),
        identity,
        int(channel),
        int(t),
        PROFILE_ALGORITHM,
        int(bins),
    )
