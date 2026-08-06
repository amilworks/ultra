"""The ``scene.derive`` batch job: convert a scene file into chunks + manifest + poster.

Mirrors ``image.derive_pyramid`` (:mod:`ultra_deepagents.imaging.job`) on the same worker
node and reuses its failure taxonomy: a
:class:`~ultra_deepagents.imaging.derivative_manifest.DeterministicDerivativeError` means
this immutable source will fail identically forever (mark it and stop), a
:class:`~ultra_deepagents.imaging.derivative_manifest.TransientDerivativeError` means try
again. The runner is pure and injectable so it is unit-tested without a real scene file;
the NATS worker wraps it.

Two passes over the source, deliberately:

1. positions (and, for splats, the importance inputs) — enough to build the octree;
2. everything else, encoded and scattered straight into its chunk output row.

The alternative — one pass holding every property in memory — costs 812 MB for the
measured 14.5M-splat file against 463 MB for the encoded output, and the second read is
sequential.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from ultra_deepagents.imaging.derivative_manifest import (
    DerivativeJobError,
    DeterministicDerivativeError,
    TransientDerivativeError,
)
from ultra_deepagents.scene3d import chunker, manifest, ply, poster, spark_encode

__all__ = [
    "CHUNK_NAME",
    "MANIFEST_NAME",
    "POSTER_NAME",
    "Scene3dDeriveJob",
    "failure_marker_path",
    "run_scene3d_derive_job",
]

ReadHeaderFn = Callable[..., ply.PlyHeader]
IterChunksFn = Callable[..., Any]
MeasureShFn = Callable[..., int]
PosterFn = Callable[..., poster.PosterResult]
PackFn = Callable[[chunker.Chunk], bytes]

FAILURE_SCHEMA = "ultra.scene3d-derive-failure.v1"
CHUNK_NAME = "chunk_{index:05d}.bin"
MANIFEST_NAME = "manifest.json"
POSTER_NAME = "poster.png"
_SCALE_NAMES = ("scale_0", "scale_1", "scale_2")
_ROT_NAMES = ("rot_0", "rot_1", "rot_2", "rot_3")
_DC_NAMES = ("f_dc_0", "f_dc_1", "f_dc_2")


@dataclass
class Scene3dDeriveJob:
    """The ``scene.derive`` payload (contract §7)."""

    resource_id: str
    src_path: str
    dst_dir: str
    max_splats_per_chunk: int = chunker.DEFAULT_MAX_PER_CHUNK
    tier_count: int = chunker.DEFAULT_TIER_COUNT
    sh_sample: int = ply.DEFAULT_SH_SAMPLE
    poster_sample: int = poster.DEFAULT_POSTER_SAMPLE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scene3dDeriveJob:
        return cls(
            resource_id=str(payload["resource_id"]),
            src_path=str(payload["src_path"]),
            dst_dir=str(payload["dst_dir"]),
            max_splats_per_chunk=int(
                payload.get("max_splats_per_chunk", chunker.DEFAULT_MAX_PER_CHUNK)
            ),
            tier_count=int(payload.get("tier_count", chunker.DEFAULT_TIER_COUNT)),
            sh_sample=int(payload.get("sh_sample", ply.DEFAULT_SH_SAMPLE)),
            poster_sample=int(payload.get("poster_sample", poster.DEFAULT_POSTER_SAMPLE)),
        )


def failure_marker_path(dst_dir: str) -> str:
    """Sidecar the control plane checks to back off a permanently-failed derive."""
    return dst_dir.rstrip("/") + ".failed"


def _atomic_write(path: str, payload: str | bytes) -> None:
    """Write via a same-directory temp file and rename, so a reader never sees a partial
    chunk or a half-written manifest."""
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        if isinstance(payload, bytes):
            stream: Any = os.fdopen(descriptor, "wb")
        else:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        with stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.remove(temporary)
        raise


def _write_failure_marker(job: Scene3dDeriveJob, exc: DeterministicDerivativeError) -> None:
    """Best-effort record that this scene's derive permanently failed, so the control
    plane stops re-enqueuing a doomed conversion on every viewer open. Never raises."""
    payload = {
        "schema": FAILURE_SCHEMA,
        "resource_id": job.resource_id,
        "src_path": job.src_path,
        "code": exc.code,
    }
    marker = failure_marker_path(job.dst_dir)
    with suppress(OSError):
        os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
        _atomic_write(marker, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _require(header: ply.PlyHeader, names: tuple[str, ...], code: str) -> None:
    if not header.has(*names):
        raise DeterministicDerivativeError(code)


def _read_columns(
    job: Scene3dDeriveJob,
    header: ply.PlyHeader,
    iter_chunks_fn: IterChunksFn,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Pass 1: pull the named columns into preallocated float32 arrays.

    Preallocated rather than list-then-concatenate: for the measured 14.5M-splat file
    that is the difference between one 406 MB working set and a 812 MB peak.
    """
    columns = {name: np.empty(header.count, dtype=np.float32) for name in names}
    offset = 0
    for block in iter_chunks_fn(job.src_path, header, names=names):
        size = int(block.shape[0])
        for name in names:
            columns[name][offset : offset + size] = block[name]
        offset += size
    if offset != header.count:
        # The header is the contract for how many records exist. A short file is a
        # deterministic defect of this immutable source, not something a retry fixes.
        raise DeterministicDerivativeError("truncated_scene_source")
    return columns


def _stack(columns: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(np.stack([columns[name] for name in names], axis=1))


def _robust_bbox(positions: np.ndarray, sample_cap: int = 2_000_000) -> list[float]:
    """The 1st..99th percentile box — what a viewer should frame on.

    Real dense reconstructions routinely carry a handful of far-field outliers: the
    measured COLMAP corridor fixture spans 3453 units end to end while the middle 99% of
    its points occupies 33, i.e. 0.9% of the diagonal. Framing a camera on the full bbox
    renders the actual scene as a speck, so the manifest carries both and the viewer
    frames on this one.

    Sampled on a stride rather than sorting every row: at 14.5M points the percentile is
    a sort, and a deterministic 2M-row stride moves the result by far less than the
    outlier tail this exists to reject.
    """
    step = max(1, int(positions.shape[0]) // max(1, sample_cap))
    sample = np.asarray(positions[::step], dtype=np.float64)
    if sample.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    low = np.percentile(sample, 1.0, axis=0)
    high = np.percentile(sample, 99.0, axis=0)
    return [*(float(v) for v in low), *(float(v) for v in high)]


def _chunk_local(positions: np.ndarray, plan: chunker.ChunkPlan) -> np.ndarray:
    """World positions rearranged into chunk-output-row order and made chunk-local."""
    local = np.empty_like(positions)
    for chunk in plan.chunks:
        rows = slice(int(plan.starts[chunk.index]), int(plan.starts[chunk.index + 1]))
        local[rows] = positions[chunk.order] - chunk.origin
    return local


def _chunk_entry(chunk: chunker.Chunk, size: int) -> dict[str, Any]:
    return {
        "index": chunk.index,
        "count": chunk.count,
        "bytes": int(size),
        "origin": [float(value) for value in chunk.origin],
        "bbox": [float(value) for value in (*chunk.bbox_min, *chunk.bbox_max)],
    }


def _derive_splats(
    job: Scene3dDeriveJob,
    header: ply.PlyHeader,
    *,
    iter_chunks_fn: IterChunksFn,
    measure_sh_fn: MeasureShFn,
) -> tuple[chunker.ChunkPlan, PackFn, dict[str, Any], dict[str, Any]]:
    """Plan and encode every splat.

    Returns the plan, a per-chunk packer (called lazily so only one payload is resident
    at a time), the measured facts the manifest needs, and the poster's stride sample.
    """
    _require(header, ("x", "y", "z", "opacity", *_SCALE_NAMES, *_ROT_NAMES), "splat_fields")
    measured = measure_sh_fn(job.src_path, header, job.sh_sample)

    columns = _read_columns(job, header, iter_chunks_fn, ("x", "y", "z", "opacity", *_SCALE_NAMES))
    positions = _stack(columns, ("x", "y", "z"))
    # Only the largest log-scale matters for importance, so reduce the three columns in
    # place rather than materializing another (n, 3) table beside `positions`.
    largest_ln_scale = np.maximum.reduce([columns[name] for name in _SCALE_NAMES])
    plan = chunker.build_chunk_plan(
        positions,
        importance=chunker.splat_importance(columns["opacity"], largest_ln_scale[:, None]),
        max_per_chunk=job.max_splats_per_chunk,
        tier_count=job.tier_count,
    )
    del columns, largest_ln_scale

    total = int(positions.shape[0])
    local = _chunk_local(positions, plan)
    stride = poster.poster_stride(total, job.poster_sample)
    # Copy the poster's sample out and drop the world table before the encoded planes are
    # allocated: a strided view would keep all 174 MB of it alive for no reason.
    poster_positions = positions[::stride].copy()
    bbox_robust = _robust_bbox(positions)
    del positions
    ext_a = np.zeros((total, 4), dtype=np.uint32)
    ext_b = np.zeros((total, 4), dtype=np.uint32)
    drawn: dict[str, list[np.ndarray]] = {"rgb": [], "alpha": [], "radius": []}
    clamped = 0
    offset = 0
    for block in iter_chunks_fn(
        job.src_path, header, names=(*_DC_NAMES, "opacity", *_SCALE_NAMES, *_ROT_NAMES)
    ):
        size = int(block.shape[0])
        rows = plan.dest[offset : offset + size]
        ln_scales = np.stack([np.asarray(block[name], np.float32) for name in _SCALE_NAMES], 1)
        f_dc = np.stack([np.asarray(block[name], np.float32) for name in _DC_NAMES], 1)
        encoded = spark_encode.encode_ext_splats(
            positions=local[rows],
            ln_scales=ln_scales,
            quat_wxyz=np.stack([np.asarray(block[name], np.float64) for name in _ROT_NAMES], 1),
            raw_opacity=np.asarray(block["opacity"], np.float32),
            f_dc=f_dc,
        )
        ext_a[rows] = encoded.ext_a
        ext_b[rows] = encoded.ext_b
        clamped += encoded.clamped_color_components
        picked = np.flatnonzero(np.arange(offset, offset + size) % stride == 0)
        if picked.size:
            base, _ = spark_encode.dc_to_base_color(f_dc[picked])
            drawn["rgb"].append(base)
            drawn["alpha"].append(spark_encode.sigmoid(block["opacity"][picked]))
            # Footprint of the projected ellipse, approximated by the equal-area circle
            # of its two largest axes.
            biggest = np.sort(np.exp(ln_scales[picked].astype(np.float64)), axis=1)[:, 1:]
            drawn["radius"].append(np.sqrt(biggest[:, 0] * biggest[:, 1]))
        offset += size
    if offset != total:
        raise DeterministicDerivativeError("truncated_scene_source")

    def pack(chunk: chunker.Chunk) -> bytes:
        rows = slice(int(plan.starts[chunk.index]), int(plan.starts[chunk.index + 1]))
        return spark_encode.pack_usx1_chunk(
            spark_encode.ExtSplatEncoding(
                ext_a=ext_a[rows], ext_b=ext_b[rows], clamped_color_components=0
            ),
            sh_degree=measured,
            bbox_min=chunk.bbox_min,
            bbox_max=chunk.bbox_max,
            origin=chunk.origin,
        )

    facts = {
        "measured_sh_degree": int(measured),
        "clamped_color_fraction": clamped / (3.0 * total),
    }
    facts["bbox_robust"] = bbox_robust
    sample = {
        "positions": poster_positions,
        "colors": np.concatenate(drawn["rgb"]),
        "opacities": np.concatenate(drawn["alpha"]),
        "radii": np.concatenate(drawn["radius"]),
    }
    return plan, pack, facts, sample


def _derive_points(
    job: Scene3dDeriveJob,
    header: ply.PlyHeader,
    *,
    iter_chunks_fn: IterChunksFn,
) -> tuple[chunker.ChunkPlan, PackFn, dict[str, Any], dict[str, Any]]:
    """Plan and encode every point (UPC1, source sRGB colours preserved)."""
    _require(header, ("x", "y", "z"), "point_fields")
    has_color = header.has("red", "green", "blue")
    has_alpha = header.has("alpha")
    columns = _read_columns(job, header, iter_chunks_fn, ("x", "y", "z"))
    positions = _stack(columns, ("x", "y", "z"))
    del columns
    # Source order is the ordering key: a scan's own order carries structure, and no
    # point is intrinsically more important than another.
    plan = chunker.build_chunk_plan(
        positions,
        importance=None,
        max_per_chunk=job.max_splats_per_chunk,
        tier_count=job.tier_count,
    )
    total = int(positions.shape[0])
    local = _chunk_local(positions, plan)
    rgba = np.full((total, 4), 255, dtype=np.uint8)
    if has_color:
        names = ("red", "green", "blue", "alpha") if has_alpha else ("red", "green", "blue")
        offset = 0
        for block in iter_chunks_fn(job.src_path, header, names=names):
            size = int(block.shape[0])
            rows = plan.dest[offset : offset + size]
            for channel, name in enumerate(names):
                rgba[rows, channel] = _to_byte(block[name])
            offset += size
        if offset != total:
            raise DeterministicDerivativeError("truncated_scene_source")

    def pack(chunk: chunker.Chunk) -> bytes:
        rows = slice(int(plan.starts[chunk.index]), int(plan.starts[chunk.index + 1]))
        return spark_encode.pack_upc1_chunk(
            positions=local[rows],
            colors_rgba=rgba[rows],
            bbox_min=chunk.bbox_min,
            bbox_max=chunk.bbox_max,
            origin=chunk.origin,
            has_alpha=has_alpha,
        )

    stride = poster.poster_stride(total, job.poster_sample)
    sample = {
        "positions": positions[::stride],
        # rgba is in output-row order; dest maps a source index to its row, so this is
        # the same stride sample of source order the positions above take.
        "colors": rgba[plan.dest[::stride], 0:3].astype(np.float32) / np.float32(255.0),
        "opacities": None,
        "radii": None,
    }
    facts = {
        "measured_sh_degree": 0,
        "clamped_color_fraction": 0.0,
        "bbox_robust": _robust_bbox(positions),
    }
    return plan, pack, facts, sample


def _to_byte(values: np.ndarray) -> np.ndarray:
    """Colour channel -> uint8. Float channels are [0,1]; integer channels are bytes."""
    array = np.asarray(values)
    if array.dtype.kind == "f":
        return np.asarray(np.clip(np.rint(array * 255.0), 0, 255), dtype=np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _resolve_job(
    job: dict[str, Any] | Scene3dDeriveJob | str,
    dst_dir: str | None,
    resource_id: str,
    options: dict[str, Any],
) -> Scene3dDeriveJob:
    if isinstance(job, str):
        if dst_dir is None:
            raise ValueError("dst_dir is required when calling with a source path")
        return Scene3dDeriveJob.from_dict(
            {"resource_id": resource_id, "src_path": job, "dst_dir": dst_dir, **options}
        )
    if options:
        raise ValueError("job options belong in the payload when passing a job payload")
    if isinstance(job, Scene3dDeriveJob):
        return job
    return Scene3dDeriveJob.from_dict(job)


def run_scene3d_derive_job(
    job: dict[str, Any] | Scene3dDeriveJob | str,
    dst_dir: str | None = None,
    *,
    resource_id: str = "",
    read_header_fn: ReadHeaderFn = ply.read_header,
    iter_chunks_fn: IterChunksFn = ply.iter_chunks,
    measure_sh_fn: MeasureShFn = ply.measured_sh_degree,
    poster_fn: PosterFn = poster.render_poster,
    **options: Any,
) -> dict[str, Any]:
    """Run a ``scene.derive`` job and return a completion-event payload.

    Accepts the queue payload (dict or :class:`Scene3dDeriveJob`) or the direct
    ``(src_path, dst_dir, **options)`` form. ``read_header_fn`` / ``iter_chunks_fn`` /
    ``measure_sh_fn`` / ``poster_fn`` are injectable so the runner is testable without a
    real scene file.

    Chunks are written first, then the poster, then ``manifest.json`` — the manifest is
    the commit record, so a reader that sees it sees a complete derive.
    """
    spec = _resolve_job(job, dst_dir, resource_id, options)
    try:
        return _run(
            spec,
            read_header_fn=read_header_fn,
            iter_chunks_fn=iter_chunks_fn,
            measure_sh_fn=measure_sh_fn,
            poster_fn=poster_fn,
        )
    except DeterministicDerivativeError as exc:
        _write_failure_marker(spec, exc)
        raise


def _run(
    job: Scene3dDeriveJob,
    *,
    read_header_fn: ReadHeaderFn,
    iter_chunks_fn: IterChunksFn,
    measure_sh_fn: MeasureShFn,
    poster_fn: PosterFn,
) -> dict[str, Any]:
    try:
        source_bytes = os.path.getsize(job.src_path)
        header = read_header_fn(job.src_path)
        scene_kind = ply.detect_scene_kind(header)
    except DerivativeJobError:
        raise
    except FileNotFoundError as exc:
        raise DeterministicDerivativeError("scene_source_missing") from exc
    except NotImplementedError as exc:
        raise DeterministicDerivativeError("unsupported_scene_encoding") from exc
    except ValueError as exc:  # PlyFormatError and the encoders' shape checks
        raise DeterministicDerivativeError("unsupported_scene_source") from exc
    except OSError as exc:
        raise TransientDerivativeError("scene_source_unavailable") from exc
    if header.count < 1:
        raise DeterministicDerivativeError("empty_scene_source")

    try:
        if scene_kind == "splat":
            plan, pack_fn, facts, sample = _derive_splats(
                job, header, iter_chunks_fn=iter_chunks_fn, measure_sh_fn=measure_sh_fn
            )
        else:
            plan, pack_fn, facts, sample = _derive_points(
                job, header, iter_chunks_fn=iter_chunks_fn
            )
    except DerivativeJobError:
        raise
    except NotImplementedError as exc:
        raise DeterministicDerivativeError("unsupported_scene_encoding") from exc
    except ValueError as exc:
        raise DeterministicDerivativeError("unsupported_scene_source") from exc
    except MemoryError as exc:
        raise TransientDerivativeError("scene_derive_resources") from exc
    except OSError as exc:
        raise TransientDerivativeError("scene_source_unavailable") from exc

    try:
        os.makedirs(job.dst_dir, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for chunk in plan.chunks:
            # Packed one at a time: materializing every payload first would hold a second
            # copy of the whole encoded scene (463 MB for the measured 14.5M-splat file).
            payload = pack_fn(chunk)
            _atomic_write(os.path.join(job.dst_dir, CHUNK_NAME.format(index=chunk.index)), payload)
            entries.append(_chunk_entry(chunk, len(payload)))
        rendered = poster_fn(
            os.path.join(job.dst_dir, POSTER_NAME),
            positions=sample["positions"],
            colors=sample["colors"],
            opacities=sample["opacities"],
            radii=sample["radii"],
            total=header.count,
        )
        document = _build_manifest(
            job, header, scene_kind, plan, entries, facts, rendered, source_bytes
        )
        _atomic_write(
            os.path.join(job.dst_dir, MANIFEST_NAME),
            json.dumps(document, indent=2, allow_nan=False) + "\n",
        )
    except DerivativeJobError:
        raise
    except OSError as exc:
        raise TransientDerivativeError("scene_output_unavailable") from exc

    # A completed derive clears the backoff marker a previous failure may have left.
    with suppress(OSError):
        os.remove(failure_marker_path(job.dst_dir))
    return {
        "resource_id": job.resource_id,
        "dst_dir": job.dst_dir,
        "status": "succeeded",
        "scene_kind": scene_kind,
        "total": plan.total,
        "chunk_count": len(plan.chunks),
        "tier_count": len(plan.tiers),
        "declared_sh_degree": ply.declared_sh_degree(header),
        "measured_sh_degree": facts["measured_sh_degree"],
        "stride_bytes": header.stride,
        "source_bytes": source_bytes,
        "manifest_path": os.path.join(job.dst_dir, MANIFEST_NAME),
        "poster_path": rendered.path,
    }


def _build_manifest(
    job: Scene3dDeriveJob,
    header: ply.PlyHeader,
    scene_kind: str,
    plan: chunker.ChunkPlan,
    entries: list[dict[str, Any]],
    facts: dict[str, Any],
    rendered: poster.PosterResult,
    source_bytes: int,
) -> dict[str, Any]:
    corners_min = [
        np.asarray(entry["bbox"][0:3]) + np.asarray(entry["origin"]) for entry in entries
    ]
    corners_max = [
        np.asarray(entry["bbox"][3:6]) + np.asarray(entry["origin"]) for entry in entries
    ]
    world_min = np.minimum.reduce(corners_min)
    world_max = np.maximum.reduce(corners_max)
    bbox = [*(float(value) for value in world_min), *(float(value) for value in world_max)]
    up_axis, up_basis = manifest.infer_up_axis(bbox)
    declared = ply.declared_sh_degree(header)
    is_splat = scene_kind == "splat"

    quantization: dict[str, Any] = (
        {
            "center": "f32-exact",
            "scale": "half-log",
            "rotation": "oct-10-10-12",
            "color": "half-display-referred",
            "clamped_color_fraction": round(float(facts["clamped_color_fraction"]), 6),
        }
        if is_splat
        else {"center": "f32-exact", "color": "u8-srgb"}
    )
    layer = manifest.build_layer(
        layer_type="splats" if is_splat else "points",
        encoding="usx-v1" if is_splat else "upc-v1",
        total=plan.total,
        chunks=entries,
        tiers=[list(tier) for tier in plan.tiers],
        activation_domain="post",
        source_frame="source",
        quantization=quantization,
    )

    limitations: list[str] = []
    if is_splat:
        limitations.extend(
            manifest.splat_limitations(
                declared_sh_degree=declared,
                measured_sh_degree=int(facts["measured_sh_degree"]),
                sh_sample=min(job.sh_sample, header.count),
                clamped_color_fraction=float(facts["clamped_color_fraction"]),
            )
        )
    else:
        limitations.append(
            "Point colours are served exactly as the source stores them, in sRGB; the renderer "
            "performs the single conversion to linear light."
        )
    limitations.append(
        f"Tier 0 loads 1 chunk in {len(plan.tiers)}, interleaved through a spatially sorted "
        f"order, so it spans the whole scene at reduced density rather than showing one corner "
        f"of it. All {plan.total:,} elements are present across the full tier set; nothing was "
        "dropped."
    )
    if plan.oversized_cells:
        limitations.append(
            f"{plan.oversized_cells} octree cell(s) reached the minimum cell size while still "
            f"holding more than {job.max_splats_per_chunk:,} elements, so their chunks exceed the "
            "target size. Every element was kept."
        )
    if plan.zero_origin_chunks:
        limitations.append(
            f"{plan.zero_origin_chunks} of {len(plan.chunks)} chunk(s) pin at least one axis to a "
            "zero origin, because a chunk-local offset could not reproduce coordinates near zero "
            "on that axis exactly in float32. Those axes keep world coordinates, so they keep the "
            "source's precision and no more."
        )
    limitations.append(
        f"The poster image is a CPU orthographic approximation along the {'xyz'[rendered.axis]} "
        f"axis, drawn from {rendered.rendered:,} of {rendered.total:,} elements one layer deep, "
        "with no blending between overlapping elements and framed on the middle 98% of the "
        "scene so outliers do not shrink it. It is not the WebGL render."
    )
    limitations.append(
        f"'{up_axis}' is a view hint for the camera controls ({up_basis}); the geometry is never "
        "rotated and stays in the source world frame."
    )
    return manifest.build_manifest(
        resource_id=job.resource_id,
        scene_kind=scene_kind,
        source_format="ply",
        writer=ply.source_writer(header),
        vertex_count=header.count,
        source_bytes=source_bytes,
        declared_sh_degree=declared,
        measured_sh_degree=int(facts["measured_sh_degree"]),
        stride_bytes=header.stride,
        bbox=bbox,
        bbox_robust=facts.get("bbox_robust"),
        up_axis=up_axis,
        up_axis_basis=up_basis,
        layers=[layer],
        limitations=limitations,
    )
