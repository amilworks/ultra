"""Bounded-memory PLY derivation for the Lens 3D scene viewer.

The source fixtures are deliberately larger than either a browser or a worker should
materialize: 2.07 million RGB points and 14.47 million Gaussian splats.  This module
keeps memory proportional to ``read batch + tier_count * output chunk`` rather than to
the number of source vertices.

Every finite source record is emitted exactly once.  Records are assigned to additive,
deterministic density tiers with a stable hash of their source row.  Tier 0 is a uniform
whole-scene preview; later tiers refine it, and the union of all tiers is the full source.
The browser therefore selects a complete tier that fits its residency budget instead of
fetching a spatially-biased prefix of a multi-gigabyte file.
"""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from ultra_deepagents.imaging.derivative_manifest import DeterministicDerivativeError
from ultra_deepagents.scene3d import chunker, ply, spark_encode

__all__ = [
    "DEFAULT_POINT_PREVIEW",
    "DEFAULT_SPLAT_PREVIEW",
    "PlyScan",
    "SplatAnalysis",
    "StreamedPlyResult",
    "analyze_splats",
    "derive_ply",
    "scan_ply",
    "tier_rates",
]

# Leave deterministic-sampling headroom below the smallest browser residency budget
# (125k splats / 300k points). Hash sampling has binomial variance, so setting a target
# equal to the browser ceiling can exceed it by a few hundred records and make a valid
# preview unloadable on that device.
DEFAULT_SPLAT_PREVIEW = 100_000
DEFAULT_POINT_PREVIEW = 280_000
ROBUST_SAMPLE_CAP = 200_000
_POSITION_NAMES = ("x", "y", "z")
_SCALE_NAMES = ("scale_0", "scale_1", "scale_2")
_ROT_NAMES = ("rot_0", "rot_1", "rot_2", "rot_3")
_DC_NAMES = ("f_dc_0", "f_dc_1", "f_dc_2")


@dataclass(frozen=True)
class PlyScan:
    source_total: int
    valid_total: int
    nonfinite: int
    bbox: list[float]
    bbox_robust: list[float]


@dataclass(frozen=True)
class StreamedPlyResult:
    scene_kind: str
    source_total: int
    total: int
    entries: list[dict[str, Any]]
    tiers: list[list[int]]
    bbox: list[float]
    bbox_robust: list[float]
    measured_sh_degree: int
    out_of_range_color_fraction: float
    nonfinite: int
    max_position_error: float
    sample: dict[str, Any]


@dataclass(frozen=True)
class SplatAnalysis:
    """Source measurements needed beside a native paged LoD artifact."""

    source_total: int
    total: int
    bbox: list[float]
    bbox_robust: list[float]
    measured_sh_degree: int
    out_of_range_color_fraction: float
    nonfinite: int
    sample: dict[str, Any]


def _positions(block: np.ndarray) -> np.ndarray:
    """One bounded float64 coordinate table for exact scan/quantization accounting."""

    count = int(block.shape[0])
    result: np.ndarray = np.empty((count, 3), dtype=np.float64)
    for axis, name in enumerate(_POSITION_NAMES):
        result[:, axis] = np.asarray(block[name], dtype=np.float64)
    return result


def scan_ply(
    path: str,
    header: ply.PlyHeader,
    *,
    iter_chunks_fn=ply.iter_chunks,
    sample_cap: int = ROBUST_SAMPLE_CAP,
) -> PlyScan:
    """First sequential pass: validate record count and compute exact/robust bounds."""

    stride = max(1, math.ceil(header.count / max(1, sample_cap)))
    seen = 0
    valid = 0
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    samples: list[np.ndarray] = []
    for block in iter_chunks_fn(path, header, names=_POSITION_NAMES):
        count = int(block.shape[0])
        xyz = _positions(block)
        finite = np.isfinite(xyz).all(axis=1)
        kept = xyz[finite]
        if kept.size:
            low = np.minimum(low, kept.min(axis=0))
            high = np.maximum(high, kept.max(axis=0))
            source_rows = np.arange(seen, seen + count, dtype=np.int64)
            sampled = finite & ((source_rows % stride) == 0)
            if np.any(sampled):
                samples.append(xyz[sampled].copy())
            valid += int(kept.shape[0])
        seen += count
    if seen != header.count:
        raise DeterministicDerivativeError("truncated_scene_source")
    if valid < 1:
        raise DeterministicDerivativeError("empty_scene_source")

    robust = np.concatenate(samples, axis=0) if samples else np.stack([low, high])
    robust_low = np.percentile(robust, 1.0, axis=0)
    robust_high = np.percentile(robust, 99.0, axis=0)
    return PlyScan(
        source_total=header.count,
        valid_total=valid,
        nonfinite=header.count - valid,
        bbox=[*(float(value) for value in low), *(float(value) for value in high)],
        bbox_robust=[
            *(float(value) for value in robust_low),
            *(float(value) for value in robust_high),
        ],
    )


def tier_rates(total: int, preview_target: int, tier_count: int) -> tuple[float, ...]:
    """Cumulative sampling rates, geometrically spaced from preview to full source."""

    if total < 1:
        raise ValueError("total must be positive")
    if preview_target < 1:
        raise ValueError("preview_target must be positive")
    if tier_count < 1:
        raise ValueError("tier_count must be positive")
    first = min(1.0, preview_target / total)
    if first >= 1.0 or tier_count == 1:
        return (1.0,)
    return tuple(
        1.0 if level == tier_count - 1 else first ** (1.0 - level / (tier_count - 1))
        for level in range(tier_count)
    )


def _splitmix64(index: np.ndarray) -> np.ndarray:
    """Stable vectorized SplitMix64; independent of Python/NumPy RNG versions."""

    with np.errstate(over="ignore"):
        value = np.asarray(index, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed: np.ndarray = value ^ (value >> np.uint64(31))
    return mixed


def _levels(source_rows: np.ndarray, rates: tuple[float, ...]) -> np.ndarray:
    if len(rates) == 1:
        only_level: np.ndarray = np.zeros(source_rows.shape[0], dtype=np.uint8)
        return only_level
    hashed = _splitmix64(source_rows)
    levels: np.ndarray = np.full(source_rows.shape[0], len(rates) - 1, dtype=np.uint8)
    unassigned = np.ones(source_rows.shape[0], dtype=bool)
    maximum = (1 << 64) - 1
    for level, rate in enumerate(rates[:-1]):
        threshold = np.uint64(min(maximum - 1, max(0, int(rate * maximum))))
        selected = unassigned & (hashed <= threshold)
        levels[selected] = level
        unassigned[selected] = False
    return levels


def _atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            os.unlink(temporary)


class _ChunkSet:
    """Fixed-capacity per-tier buffers and their emitted manifest entries."""

    def __init__(
        self,
        directory: str,
        *,
        scene_kind: str,
        tier_count: int,
        max_per_chunk: int,
        measured_sh_degree: int,
        has_alpha: bool,
    ) -> None:
        self.directory = directory
        self.scene_kind = scene_kind
        self.max_per_chunk = max_per_chunk
        self.measured_sh_degree = measured_sh_degree
        self.has_alpha = has_alpha
        self.entries: list[dict[str, Any]] = []
        self.tiers: list[list[int]] = [[] for _ in range(tier_count)]
        self.counts = np.zeros(tier_count, dtype=np.int64)
        self.max_position_error = 0.0
        self.world = [np.empty((max_per_chunk, 3), dtype=np.float64) for _ in self.tiers]
        self.ext_a: list[np.ndarray] | None
        self.ext_b: list[np.ndarray] | None
        self.rgba: list[np.ndarray] | None
        if scene_kind == "splat":
            self.ext_a = [np.empty((max_per_chunk, 4), dtype=np.uint32) for _ in self.tiers]
            self.ext_b = [np.empty((max_per_chunk, 4), dtype=np.uint32) for _ in self.tiers]
            self.rgba = None
        else:
            self.ext_a = None
            self.ext_b = None
            self.rgba = [np.empty((max_per_chunk, 4), dtype=np.uint8) for _ in self.tiers]

    def append_splats(
        self,
        level: int,
        ext_a: np.ndarray,
        ext_b: np.ndarray,
        world: np.ndarray,
    ) -> None:
        assert self.ext_a is not None and self.ext_b is not None
        offset = 0
        while offset < ext_a.shape[0]:
            used = int(self.counts[level])
            take = min(self.max_per_chunk - used, int(ext_a.shape[0]) - offset)
            self.ext_a[level][used : used + take] = ext_a[offset : offset + take]
            self.ext_b[level][used : used + take] = ext_b[offset : offset + take]
            self.world[level][used : used + take] = world[offset : offset + take]
            self.counts[level] += take
            offset += take
            if int(self.counts[level]) == self.max_per_chunk:
                self._flush(level)

    def append_points(self, level: int, world: np.ndarray, rgba: np.ndarray) -> None:
        assert self.rgba is not None
        offset = 0
        while offset < world.shape[0]:
            used = int(self.counts[level])
            take = min(self.max_per_chunk - used, int(world.shape[0]) - offset)
            self.world[level][used : used + take] = world[offset : offset + take]
            self.rgba[level][used : used + take] = rgba[offset : offset + take]
            self.counts[level] += take
            offset += take
            if int(self.counts[level]) == self.max_per_chunk:
                self._flush(level)

    def finish(self) -> None:
        for level in range(len(self.tiers)):
            self._flush(level)

    def _flush(self, level: int) -> None:
        count = int(self.counts[level])
        if count < 1:
            return
        world = self.world[level][:count]
        world_min = world.min(axis=0)
        world_max = world.max(axis=0)
        origin, xyz, _snapped = chunker.chunk_frame(np.asarray(world, dtype=np.float32))
        reconstructed = xyz.astype(np.float64) + origin.astype(np.float64)
        self.max_position_error = max(
            self.max_position_error,
            float(np.max(np.abs(reconstructed - world))),
        )
        if self.scene_kind == "splat":
            assert self.ext_a is not None and self.ext_b is not None
            ext_a = self.ext_a[level][:count]
            ext_b = self.ext_b[level][:count]
            ext_a[:, 0:3] = xyz.view(np.uint32)
            payload = spark_encode.pack_usx1_chunk(
                spark_encode.ExtSplatEncoding(
                    ext_a=ext_a,
                    ext_b=ext_b,
                    out_of_range_color_components=0,
                ),
                sh_degree=self.measured_sh_degree,
                bbox_min=xyz.min(axis=0),
                bbox_max=xyz.max(axis=0),
                origin=origin,
            )
        else:
            assert self.rgba is not None
            payload = spark_encode.pack_upc1_chunk(
                positions=xyz,
                colors_rgba=self.rgba[level][:count],
                bbox_min=xyz.min(axis=0),
                bbox_max=xyz.max(axis=0),
                origin=origin,
                has_alpha=self.has_alpha,
            )
        index = len(self.entries)
        name = f"chunk_{index:05d}.bin"
        _atomic_write(os.path.join(self.directory, name), payload)
        self.entries.append(
            {
                "index": index,
                "count": count,
                "bytes": len(payload),
                "origin": [0.0, 0.0, 0.0],
                "bbox": [
                    *(float(value) for value in world_min),
                    *(float(value) for value in world_max),
                ],
            }
        )
        self.entries[-1]["origin"] = [float(value) for value in origin]
        self.tiers[level].append(index)
        self.counts[level] = 0


class _PosterSample:
    def __init__(self, total: int, target: int) -> None:
        self.stride = max(1, math.ceil(total / max(1, target)))
        self.seen = 0
        self.positions: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.opacities: list[np.ndarray] = []
        self.radii: list[np.ndarray] = []

    def take(self, count: int) -> np.ndarray:
        rows = np.arange(self.seen, self.seen + count, dtype=np.int64)
        self.seen += count
        selected: np.ndarray = np.asarray((rows % self.stride) == 0, dtype=np.bool_)
        return selected

    def result(self, *, splat: bool) -> dict[str, Any]:
        return {
            "positions": np.concatenate(self.positions),
            "colors": np.concatenate(self.colors),
            "opacities": np.concatenate(self.opacities) if splat else None,
            "radii": np.concatenate(self.radii) if splat else None,
        }


def analyze_splats(
    *,
    path: str,
    header: ply.PlyHeader,
    poster_sample: int,
    sh_sample: int,
    iter_chunks_fn=ply.iter_chunks,
    measure_sh_fn=ply.measured_sh_degree,
) -> SplatAnalysis:
    """Measure a splat source and retain only a bounded poster sample.

    The native RAD converter owns interactive LoD generation.  This pass still owns
    Ultra's independent bounds, SH, gamut, and poster provenance; it never materializes
    the multi-million-row source.
    """

    required = (*_POSITION_NAMES, *_DC_NAMES, "opacity", *_SCALE_NAMES, *_ROT_NAMES)
    if not header.has(*required):
        raise DeterministicDerivativeError("splat_fields")
    scan = scan_ply(path, header, iter_chunks_fn=iter_chunks_fn)
    # RAD retention changes the scientific appearance and the renderer's texture-plane
    # residency. A bounded random sample is useful provenance, but it cannot prove an SH
    # band is empty: a single late coefficient can be view-dependent signal. Passing the
    # declared element count selects `measured_sh_degree`'s sequential full-scan path.
    # Keep the injectable function seam so tests and alternate readers remain bounded.
    measured = measure_sh_fn(path, header, header.count)
    sample = _PosterSample(scan.valid_total, poster_sample)
    source_seen = 0
    valid_seen = 0
    out_of_range = 0

    for block in iter_chunks_fn(path, header, names=required):
        source_count = int(block.shape[0])
        world64 = _positions(block)
        finite = np.isfinite(world64).all(axis=1)
        source_seen += source_count
        if not np.any(finite):
            continue

        world = np.asarray(world64[finite], dtype=np.float32)
        f_dc = np.stack(
            [np.asarray(block[name][finite], dtype=np.float32) for name in _DC_NAMES],
            axis=1,
        )
        opacity = np.asarray(block["opacity"][finite], dtype=np.float32)
        ln_scales = np.stack(
            [np.asarray(block[name][finite], dtype=np.float32) for name in _SCALE_NAMES],
            axis=1,
        )
        base, outside = spark_encode.dc_to_base_color(f_dc)
        out_of_range += outside
        poster_rows = sample.take(int(world.shape[0]))
        if np.any(poster_rows):
            biggest = np.sort(
                np.exp(np.clip(ln_scales[poster_rows].astype(np.float64), -700.0, 700.0)),
                axis=1,
            )[:, 1:]
            sample.positions.append(world[poster_rows].copy())
            sample.colors.append(np.asarray(np.clip(base[poster_rows], 0.0, 1.0), np.float32))
            sample.opacities.append(
                np.asarray(spark_encode.sigmoid(opacity[poster_rows]), dtype=np.float32)
            )
            sample.radii.append(np.sqrt(biggest[:, 0] * biggest[:, 1]))
        valid_seen += int(world.shape[0])

    if source_seen != header.count:
        raise DeterministicDerivativeError("truncated_scene_source")
    if valid_seen != scan.valid_total:
        raise DeterministicDerivativeError("scene_source_changed")
    return SplatAnalysis(
        source_total=scan.source_total,
        total=scan.valid_total,
        bbox=scan.bbox,
        bbox_robust=scan.bbox_robust,
        measured_sh_degree=measured,
        out_of_range_color_fraction=out_of_range / (3.0 * scan.valid_total),
        nonfinite=scan.nonfinite,
        sample=sample.result(splat=True),
    )


def _to_byte(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "f":
        byte_values: np.ndarray = np.asarray(
            np.clip(np.rint(array * 255.0), 0, 255), dtype=np.uint8
        )
        return byte_values
    byte_values = np.clip(array, 0, 255).astype(np.uint8)
    return byte_values


def derive_ply(
    *,
    path: str,
    directory: str,
    header: ply.PlyHeader,
    max_per_chunk: int,
    tier_count: int,
    preview_splats: int = DEFAULT_SPLAT_PREVIEW,
    preview_points: int = DEFAULT_POINT_PREVIEW,
    poster_sample: int,
    sh_sample: int,
    iter_chunks_fn=ply.iter_chunks,
    measure_sh_fn=ply.measured_sh_degree,
) -> StreamedPlyResult:
    """Derive every finite PLY record with bounded resident memory."""

    if max_per_chunk < 1 or max_per_chunk > 250_000:
        raise ValueError("max_per_chunk must be between 1 and 250000")
    scan = scan_ply(path, header, iter_chunks_fn=iter_chunks_fn)
    scene_kind = ply.detect_scene_kind(header)
    is_splat = scene_kind == "splat"
    has_color = header.has("red", "green", "blue")
    has_alpha = header.has("alpha")
    if is_splat:
        required = (*_POSITION_NAMES, *_DC_NAMES, "opacity", *_SCALE_NAMES, *_ROT_NAMES)
        if not header.has(*required):
            raise DeterministicDerivativeError("splat_fields")
        measured = measure_sh_fn(path, header, sh_sample)
        preview_target = preview_splats
        names = required
    else:
        if not header.has(*_POSITION_NAMES):
            raise DeterministicDerivativeError("point_fields")
        measured = 0
        color_names = (
            ("red", "green", "blue", "alpha")
            if has_alpha
            else (
                "red",
                "green",
                "blue",
            )
        )
        names = (*_POSITION_NAMES, *color_names) if has_color else _POSITION_NAMES
        preview_target = preview_points
    rates = tier_rates(scan.valid_total, preview_target, tier_count)
    os.makedirs(directory, exist_ok=True)
    chunks = _ChunkSet(
        directory,
        scene_kind=scene_kind,
        tier_count=len(rates),
        max_per_chunk=max_per_chunk,
        measured_sh_degree=measured,
        has_alpha=has_alpha,
    )
    sample = _PosterSample(scan.valid_total, poster_sample)
    source_seen = 0
    valid_seen = 0
    out_of_range = 0

    for block in iter_chunks_fn(path, header, names=names):
        source_count = int(block.shape[0])
        world64 = _positions(block)
        finite = np.isfinite(world64).all(axis=1)
        world64 = world64[finite]
        source_rows = np.arange(source_seen, source_seen + source_count, dtype=np.uint64)[finite]
        source_seen += source_count
        if world64.size == 0:
            continue
        world = np.asarray(world64, dtype=np.float32)
        levels = _levels(source_rows, rates)
        poster_rows = sample.take(int(world.shape[0]))

        if is_splat:
            ln_scales = np.stack(
                [np.asarray(block[name][finite], dtype=np.float32) for name in _SCALE_NAMES],
                axis=1,
            )
            f_dc = np.stack(
                [np.asarray(block[name][finite], dtype=np.float32) for name in _DC_NAMES],
                axis=1,
            )
            opacity = np.asarray(block["opacity"][finite], dtype=np.float32)
            encoded = spark_encode.encode_ext_splats(
                positions=world,
                ln_scales=ln_scales,
                quat_wxyz=np.stack(
                    [np.asarray(block[name][finite], dtype=np.float64) for name in _ROT_NAMES],
                    axis=1,
                ),
                raw_opacity=opacity,
                f_dc=f_dc,
            )
            out_of_range += encoded.out_of_range_color_components
            for level in range(len(rates)):
                selected = levels == level
                if np.any(selected):
                    chunks.append_splats(
                        level,
                        encoded.ext_a[selected],
                        encoded.ext_b[selected],
                        world64[selected],
                    )
            if np.any(poster_rows):
                base, _ = spark_encode.dc_to_base_color(f_dc[poster_rows])
                biggest = np.sort(np.exp(ln_scales[poster_rows].astype(np.float64)), axis=1)[:, 1:]
                sample.positions.append(world[poster_rows].copy())
                # The interactive splat wire preserves HDR tails. A PNG poster cannot,
                # so clamp only at this final display-referred raster boundary.
                sample.colors.append(np.asarray(np.clip(base, 0.0, 1.0), dtype=np.float32))
                sample.opacities.append(
                    np.asarray(spark_encode.sigmoid(opacity[poster_rows]), dtype=np.float32)
                )
                sample.radii.append(np.sqrt(biggest[:, 0] * biggest[:, 1]))
        else:
            rgba = np.full((world.shape[0], 4), 255, dtype=np.uint8)
            if has_color:
                for channel, name in enumerate(("red", "green", "blue")):
                    rgba[:, channel] = _to_byte(block[name][finite])
                if has_alpha:
                    rgba[:, 3] = _to_byte(block["alpha"][finite])
            for level in range(len(rates)):
                selected = levels == level
                if np.any(selected):
                    chunks.append_points(level, world64[selected], rgba[selected])
            if np.any(poster_rows):
                sample.positions.append(world[poster_rows].copy())
                sample.colors.append(rgba[poster_rows, :3].astype(np.float32) / np.float32(255.0))
        valid_seen += int(world.shape[0])

    if source_seen != header.count:
        raise DeterministicDerivativeError("truncated_scene_source")
    if valid_seen != scan.valid_total:
        raise DeterministicDerivativeError("scene_source_changed")
    chunks.finish()
    if sum(entry["count"] for entry in chunks.entries) != scan.valid_total:
        raise RuntimeError("streaming derive lost source records")
    return StreamedPlyResult(
        scene_kind=scene_kind,
        source_total=scan.source_total,
        total=scan.valid_total,
        entries=chunks.entries,
        tiers=chunks.tiers,
        bbox=scan.bbox,
        bbox_robust=scan.bbox_robust,
        measured_sh_degree=measured,
        out_of_range_color_fraction=(out_of_range / (3.0 * scan.valid_total)) if is_splat else 0.0,
        nonfinite=scan.nonfinite,
        max_position_error=chunks.max_position_error,
        sample=sample.result(splat=is_splat),
    )
