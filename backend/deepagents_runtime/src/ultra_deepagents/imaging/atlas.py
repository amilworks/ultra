"""Parallel texture-atlas assembly.

``engine.atlas()`` reads N z-planes in a serial loop inside ONE worker — and the
libbioimage binding's global lock serializes them, so a deep stack costs ~N decodes
back-to-back (~10s for 80 planes). This orchestrates the SAME per-plane work across the
runner's worker *pool*: plan -> one global-window pass -> N parallel cell reads -> compose.

On :class:`ImagePool` the N decodes run concurrently across processes (the only way past
the per-process global lock); on :class:`InlineRunner` the service keeps the monolithic
sequential path (a single process can't parallelize anyway). The composed output is
byte-identical to ``engine.atlas()`` — both use :func:`compose_atlas_png` over cells
produced by the same ``atlas_cell``.
"""

from __future__ import annotations

import asyncio
import io
import math
import operator
import os
from typing import Any

__all__ = [
    "SCALAR_VOLUME_MAX_BYTES",
    "assemble_atlas",
    "compose_atlas_png",
    "assemble_scalar_volume",
    "build_scalar_volume_dict",
    "plan_scalar_preview",
    "validate_scalar_plan",
]

SCALAR_VOLUME_MAX_BYTES = 256 * 1024 * 1024
SCALAR_PREVIEW_MAX_DIMENSION = 512
SCALAR_PREVIEW_MAX_VOXELS = 16_777_216
SCALAR_PREVIEW_POLICY = "auto-v1"
_SCALAR_DTYPE_BYTES = {"uint8": 1, "uint16": 2, "int16": 2, "float32": 4}


def plan_scalar_preview(
    width: int,
    height: int,
    depth: int,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, int | str]:
    """Plan one bounded, power-of-two, spacing-aware visualization grid.

    Dimensions are X/Y/Z and spacing is x/y/z. The axis with the finest current
    effective sampling is reduced first, which preserves anisotropic coarse axes
    (notably Z in microscopy) whenever the shared size envelope permits it.
    """
    source = [int(width), int(height), int(depth)]
    if any(value <= 0 for value in source):
        raise ValueError("scalar preview source geometry must be positive")
    physical = []
    for value in spacing:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 1.0
        physical.append(parsed if math.isfinite(parsed) and parsed > 0 else 1.0)
    factors = [1, 1, 1]

    def delivered() -> list[int]:
        return [(source[i] + factors[i] - 1) // factors[i] for i in range(3)]

    while True:
        grid = delivered()
        over_dimension = [i for i, value in enumerate(grid) if value > SCALAR_PREVIEW_MAX_DIMENSION]
        if not over_dimension and math.prod(grid) <= SCALAR_PREVIEW_MAX_VOXELS:
            break
        candidates = over_dimension or [i for i, value in enumerate(grid) if value > 1]
        if not candidates:
            raise ValueError("scalar preview geometry cannot satisfy the bounded policy")
        axis = min(candidates, key=lambda i: (physical[i] * factors[i], -grid[i], i))
        factors[axis] *= 2

    grid = delivered()
    return {
        "width": grid[0],
        "height": grid[1],
        "depth": grid[2],
        "source_width": source[0],
        "source_height": source[1],
        "source_depth": source[2],
        "downsample_x": factors[0],
        "downsample_y": factors[1],
        "downsample_z": factors[2],
        "preview_policy": SCALAR_PREVIEW_POLICY,
    }


def _scalar_plan_integer(plan: dict, field: str) -> int:
    value = plan.get(field)
    if isinstance(value, bool):
        raise ValueError(f"scalar plan {field} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"scalar plan {field} must be an integer") from exc


def validate_scalar_plan(plan: dict) -> int:
    """Validate planned output before any plane decode or volume materialization."""
    width = _scalar_plan_integer(plan, "width")
    height = _scalar_plan_integer(plan, "height")
    depth = _scalar_plan_integer(plan, "depth")
    bytes_per_voxel = _scalar_plan_integer(plan, "bytes_per_voxel")
    dtype = str(plan.get("dtype", "")).strip().lower()
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("scalar plan geometry must be positive")
    if _SCALAR_DTYPE_BYTES.get(dtype) != bytes_per_voxel:
        raise ValueError("scalar plan dtype/byte width is unsupported")
    output_bytes = width * height * depth * bytes_per_voxel
    if max(width, height, depth) > SCALAR_PREVIEW_MAX_DIMENSION:
        raise ValueError("scalar volume plan is exceeding the preview dimension limit")
    if width * height * depth > SCALAR_PREVIEW_MAX_VOXELS:
        raise ValueError("scalar volume plan is exceeding the preview voxel limit")
    if output_bytes <= 0 or output_bytes > SCALAR_VOLUME_MAX_BYTES:
        raise ValueError(
            f"scalar volume plan requires {output_bytes} bytes, exceeding the {SCALAR_VOLUME_MAX_BYTES} byte limit"
        )
    return output_bytes


# Bulk reads (a whole atlas / scalar volume) fan the per-plane decodes across the
# worker pool. Left unbounded they occupy EVERY worker for the full multi-second
# decode, so a z-scrub or the Lens viewer opened at the same time has zero free
# decoders and appears to lock up. Cap concurrent bulk chunk-reads to
# (workers - reserve) so at least `reserve` worker(s) always stay free to serve
# interactive /slice and /tile requests. The cap is global (one semaphore per
# process) so even two concurrent volume opens can't consume the reserved slot.
_bulk_semaphore: asyncio.Semaphore | None = None
_bulk_semaphore_size = 0


def _interactive_reserve() -> int:
    try:
        return max(0, int(os.environ.get("ULTRA_IMAGE_INTERACTIVE_RESERVE", "1") or "1"))
    except ValueError:
        return 1


def _bulk_semaphore_for(workers: int) -> asyncio.Semaphore:
    global _bulk_semaphore, _bulk_semaphore_size
    size = max(1, workers - _interactive_reserve())
    if _bulk_semaphore is None or _bulk_semaphore_size != size:
        _bulk_semaphore = asyncio.Semaphore(size)
        _bulk_semaphore_size = size
    return _bulk_semaphore


async def _gather_bulk(semaphore: asyncio.Semaphore, factories: list) -> list:
    """Run chunk-read coroutine factories, holding at most ``semaphore`` of them in
    flight so the reserved interactive worker(s) are never occupied by bulk work."""

    async def _guarded(make):
        async with semaphore:
            return await make()

    return await asyncio.gather(*[_guarded(make) for make in factories])


async def assemble_atlas(
    runner: Any,
    path: str,
    *,
    channels: list[int] | None = None,
    colors: list | None = None,
    level: int | None = None,
    t: int = 0,
) -> bytes:
    """Assemble the atlas PNG by fanning the per-plane reads across ``runner``'s pool.

    ``runner.call(method, *args, **kwargs)`` returns an awaitable that executes in a
    worker; gathering the per-cell calls runs them concurrently (order-preserving, so
    cell ``z`` lands at grid position ``(z // columns, z % columns)``).
    """
    plan = await runner.call(
        "atlas_plan", path, channels=channels, colors=colors, level=level, t=t
    )
    plan_t = plan.get("t", t)
    # One global-window pass first (a few sampled planes) so every cell shares the same
    # per-channel window — without it, parallel cells would each auto-scale and flicker in z.
    windows = await runner.call(
        "atlas_windows",
        path,
        depth=plan["depth"],
        level=plan["read_level"],
        channels=plan["read_channels"],
        paged=plan["paged"],
        t=plan_t,
    )
    # Fan out at WORKER granularity, not per-plane: one task per worker, each reading a
    # contiguous range of planes sequentially. The binding's lock serializes reads within a
    # worker anyway, so this gives the same cross-worker parallelism with a fraction of the
    # per-task submission/IPC overhead (80 tasks -> ~W tasks).
    workers = max(1, int(getattr(runner, "workers", 1) or 1))
    chunks = _split_chunks(plan["depth"], min(plan["depth"], workers))
    results = await _gather_bulk(
        _bulk_semaphore_for(workers),
        [
            lambda zs=zs: runner.call(
                "atlas_cells",
                path,
                zs=zs,
                level=plan["read_level"],
                channels=plan["read_channels"],
                colors=plan["cell_colors"],
                windows=windows,
                cell_w=plan["cell_w"],
                cell_h=plan["cell_h"],
                paged=plan["paged"],
                t=plan_t,
            )
            for zs in chunks
        ],
    )
    cells = [cell for chunk_cells in results for cell in chunk_cells]  # flatten in z-order
    # Compose off the event loop: numpy tiling + PNG encode release the GIL, so a worker
    # thread keeps the async reactor free for other requests.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, compose_atlas_png, cells, plan)


async def assemble_scalar_volume(runner: Any, path: str, *, channel: int = 0, t: int = 0) -> dict:
    """The scalar-volume (WebGL medical volume) sibling of :func:`assemble_atlas`: read the
    depth planes across the pool in worker-sized chunks, then stack into one float volume.
    Byte-identical to ``engine.scalar_volume()`` (same planes in z-order, same stack)."""
    plan = await runner.call("scalar_plan", path, channel=channel, t=t)
    validate_scalar_plan(plan)
    depth = plan["depth"]
    workers = max(1, int(getattr(runner, "workers", 1) or 1))
    chunks = _split_chunks(depth, min(depth, workers))
    results = await _gather_bulk(
        _bulk_semaphore_for(workers),
        [
            lambda zs=zs: runner.call(
                "scalar_planes", path, zs=zs, channel=plan["channel"], t=plan["t"], pages=plan["pages"]
            )
            for zs in chunks
        ],
    )
    planes = [plane for chunk_planes in results for plane in chunk_planes]  # flatten in z-order
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, build_scalar_volume_dict, planes, plan["channel"], plan)


def build_scalar_volume_dict(planes: list, channel: int, plan: dict) -> dict:
    """Stack per-plane float arrays into the scalar-volume payload. Pure; shared by the
    parallel orchestrator and the sequential ``engine.scalar_volume()``."""
    import numpy as np

    validate_scalar_plan(plan)
    width = _scalar_plan_integer(plan, "width")
    height = _scalar_plan_integer(plan, "height")
    depth = _scalar_plan_integer(plan, "depth")
    if len(planes) != depth:
        raise ValueError(f"scalar plane count {len(planes)} does not match planned depth {depth}")
    expected_shape = (height, width)
    expected_dtype = np.dtype({
        "uint8": "u1",
        "uint16": "<u2",
        "int16": "<i2",
        "float32": "<f4",
    }[str(plan["dtype"]).strip().lower()])
    for plane in planes:
        array = np.asarray(plane)
        if (
            array.shape != expected_shape
            or array.dtype.kind != expected_dtype.kind
            or array.dtype.itemsize != expected_dtype.itemsize
        ):
            raise ValueError("scalar plane shape/dtype does not match the validated plan")
    vol = np.ascontiguousarray(np.stack(planes, axis=0), dtype=expected_dtype)
    provenance_keys = (
        "t",
        "source_width",
        "source_height",
        "source_depth",
        "downsample_x",
        "downsample_y",
        "downsample_z",
        "preview_policy",
    )
    return {
        "data": vol.tobytes(order="C"),
        "width": int(vol.shape[2]) if vol.ndim == 3 else 0,
        "height": int(vol.shape[1]) if vol.ndim == 3 else 0,
        "depth": int(vol.shape[0]),
        "dtype": str(plan["dtype"]).strip().lower(),
        "bytes_per_voxel": expected_dtype.itemsize,
        "raw_min": float(vol.min()) if vol.size else 0.0,
        "raw_max": float(vol.max()) if vol.size else 1.0,
        "channel": int(channel),
        "scl_slope": 1.0,
        "scl_inter": 0.0,
        **{key: plan[key] for key in provenance_keys if key in plan},
    }


def _split_chunks(depth: int, n: int) -> list[list[int]]:
    """Split range(depth) into n contiguous, near-even z-ranges (preserving order)."""
    n = max(1, min(n, depth))
    base, extra = divmod(depth, n)
    chunks: list[list[int]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def compose_atlas_png(cells: list, plan: dict) -> bytes:
    """Tile the per-plane cells into the row-major grid and PNG-encode it. Pure; shared by
    the parallel orchestrator and the sequential ``engine.atlas()`` so they agree exactly."""
    import numpy as np
    from PIL import Image

    cols, rows = plan["columns"], plan["rows"]
    cell_w, cell_h = plan["cell_w"], plan["cell_h"]
    atlas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype="uint8")
    for z, cell in enumerate(cells):
        if cell is None:
            continue
        r, c = divmod(z, cols)
        atlas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w, :] = cell
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, format="PNG")
    return buf.getvalue()
