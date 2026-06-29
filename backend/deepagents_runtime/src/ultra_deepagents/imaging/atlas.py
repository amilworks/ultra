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
import os
from typing import Any

__all__ = ["assemble_atlas", "compose_atlas_png", "assemble_scalar_volume", "build_scalar_volume_dict"]


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
) -> bytes:
    """Assemble the atlas PNG by fanning the per-plane reads across ``runner``'s pool.

    ``runner.call(method, *args, **kwargs)`` returns an awaitable that executes in a
    worker; gathering the per-cell calls runs them concurrently (order-preserving, so
    cell ``z`` lands at grid position ``(z // columns, z % columns)``).
    """
    plan = await runner.call("atlas_plan", path, channels=channels, colors=colors, level=level)
    # One global-window pass first (a few sampled planes) so every cell shares the same
    # per-channel window — without it, parallel cells would each auto-scale and flicker in z.
    windows = await runner.call(
        "atlas_windows",
        path,
        depth=plan["depth"],
        level=plan["read_level"],
        channels=plan["read_channels"],
        paged=plan["paged"],
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
    depth = plan["depth"]
    if depth <= 0:
        return build_scalar_volume_dict([], plan["channel"])
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
    return await loop.run_in_executor(None, build_scalar_volume_dict, planes, plan["channel"])


def build_scalar_volume_dict(planes: list, channel: int) -> dict:
    """Stack per-plane float arrays into the scalar-volume payload. Pure; shared by the
    parallel orchestrator and the sequential ``engine.scalar_volume()``."""
    import numpy as np

    vol = np.stack(planes, axis=0) if planes else np.zeros((0, 0, 0), "float32")
    return {
        "data": vol.tobytes(),
        "width": int(vol.shape[2]) if vol.ndim == 3 else 0,
        "height": int(vol.shape[1]) if vol.ndim == 3 else 0,
        "depth": int(vol.shape[0]),
        "dtype": "float32",
        "bytes_per_voxel": 4,
        "raw_min": float(vol.min()) if vol.size else 0.0,
        "raw_max": float(vol.max()) if vol.size else 1.0,
        "channel": int(channel),
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
