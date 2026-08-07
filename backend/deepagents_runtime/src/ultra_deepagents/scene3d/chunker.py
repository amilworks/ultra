"""Octree chunking and tier construction for the ``scene3d`` derive (contract §5).

Chunking buys three things at once and the third is not optional: progressive
streaming, frustum culling, and *precision* — chunk-local coordinates keep magnitudes
small, which is what makes a 1211-unit corridor scan survive float32 centres.

The partition is exact. Every source element lands in exactly one chunk, and
:func:`build_chunk_plan` refuses to return a plan whose chunk counts do not sum to the
input count. Tier 0 is a stride sample of *every* cell, so the first pass a viewer draws
covers the whole scene at reduced density rather than a corner of it at full density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_MAX_PER_CHUNK",
    "DEFAULT_TIER_COUNT",
    "Chunk",
    "ChunkPlan",
    "build_chunk_plan",
    "splat_importance",
]

DEFAULT_MAX_PER_CHUNK = 200_000
DEFAULT_TIER_COUNT = 3
# Subdivision floor as a fraction of the root cell's longest edge: 2^-12 keeps the
# octree at most 12 levels deep, so a pathological point pile (a million duplicates at
# one coordinate) terminates instead of recursing forever. Cells that bottom out here
# keep all their elements — the chunk is oversized and counted, never truncated.
DEFAULT_MIN_CELL_FRACTION = 2.0**-12


@dataclass(frozen=True)
class Chunk:
    """One emitted chunk: which source elements, where its origin sits, its local bbox."""

    index: int
    tier: int
    cell: int  # octree cell this chunk is a density slice of; siblings share it
    order: np.ndarray  # source indices, most important first
    origin: np.ndarray  # (3,) float32, world
    bbox_min: np.ndarray  # (3,) float32, chunk-local
    bbox_max: np.ndarray  # (3,) float32, chunk-local

    @property
    def count(self) -> int:
        return int(self.order.size)


@dataclass(frozen=True)
class ChunkPlan:
    """A complete, verified partition of the source elements into chunks and tiers."""

    chunks: tuple[Chunk, ...]
    tiers: tuple[tuple[int, ...], ...]
    dest: np.ndarray  # source index -> row in the concatenated chunk output
    starts: np.ndarray  # per-chunk first output row (len == len(chunks) + 1)
    oversized_cells: int  # cells that hit the size floor and still exceed max_per_chunk
    zero_origin_chunks: int  # chunks with >=1 axis pinned to origin 0 to stay bit-exact

    @property
    def total(self) -> int:
        return int(self.starts[-1])


def splat_importance(raw_opacity: np.ndarray, ln_scales: np.ndarray) -> np.ndarray:
    """``sigmoid(opacity) * exp(max ln scale)`` — the contract's ordering key.

    A prefix of an importance-ordered chunk is a valid decimation of it, which is what
    lets a tier render partially without becoming a lie about what is on screen.
    Computed in float64: ``exp`` of the measured log-scale range (-10.6 .. 0.59) is fine
    in float32, but a degenerate export with large positive log scales would overflow.
    """
    opacity = np.asarray(raw_opacity, dtype=np.float64)
    # Reduce before widening: on the measured 14.5M-splat file, widening the whole
    # (n, 3) log-scale table to float64 first costs 347 MB for a result that is (n,).
    largest = np.asarray(ln_scales).max(axis=1).astype(np.float64)
    with np.errstate(over="ignore"):
        return np.asarray((1.0 / (1.0 + np.exp(-opacity))) * np.exp(largest))


def _pow2_grid(magnitude: float) -> float:
    """The ULP of a float32 at ``magnitude`` — the coarsest grid every coordinate on that
    axis is already aligned to. Snapping the origin to it maximizes the trailing zero
    bits of the subtraction and so maximizes the chance it is exactly representable."""
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return 1.0
    return float(np.float32(2.0 ** (math.floor(math.log2(magnitude)) - 23)))


def _chunk_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Choose an origin; return ``(origin, local, snapped)`` with ``local + origin == world``.

    The origin is the cell's low corner snapped down to a power-of-two grid, per axis.
    That is not exact for *every* float32 input: an axis spanning -345 to +0.001 cannot
    round-trip the small values through a -345 offset, because 345.001 has a coarser ULP
    than 0.001 does. The round trip is therefore verified in float32 — the arithmetic the
    shader actually performs — and any axis that fails falls back to an origin of 0,
    where ``local == world`` is exact by construction.

    The fallback is *per axis* because the failure is: a corridor scan straddling zero in
    one axis should not lose the chunk-local benefit on the other two. ``snapped`` is
    False when any axis fell back, so the manifest can say so.
    """
    low = points.min(axis=0).astype(np.float64)
    grid = np.array([_pow2_grid(float(np.abs(points[:, a]).max())) for a in range(3)])
    origin = (np.floor(low / grid) * grid).astype(np.float32)
    local = (points - origin).astype(np.float32)
    exact = np.all(local + origin == points, axis=0)
    if bool(exact.all()):
        return origin, local, True
    origin = np.where(exact, origin, np.float32(0.0)).astype(np.float32)
    return origin, (points - origin).astype(np.float32), False


def _subdivide(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    max_per_chunk: int,
    min_cell_size: float,
) -> tuple[list[np.ndarray], int]:
    """Octree-partition ``indices`` until each cell holds at most ``max_per_chunk``.

    Iterative rather than recursive: the size floor bounds the depth, but a hostile file
    should not be able to reach it through the interpreter's stack.
    """
    low = positions[indices].min(axis=0).astype(np.float64)
    high = positions[indices].max(axis=0).astype(np.float64)
    stack: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [(indices, low, high)]
    cells: list[np.ndarray] = []
    oversized = 0
    while stack:
        cell, cell_low, cell_high = stack.pop()
        edge = float((cell_high - cell_low).max())
        if cell.size <= max_per_chunk:
            cells.append(cell)
            continue
        if edge <= min_cell_size:
            # The cell cannot get smaller and still holds too much. Keep every element
            # and report the overshoot; dropping the tail here would be exactly the
            # silent decimation the contract forbids.
            cells.append(cell)
            oversized += 1
            continue
        middle = 0.5 * (cell_low + cell_high)
        points = positions[cell]
        octant = (
            (points[:, 0] >= middle[0]).astype(np.uint8) << 2
            | (points[:, 1] >= middle[1]).astype(np.uint8) << 1
            | (points[:, 2] >= middle[2]).astype(np.uint8)
        )
        for code in range(8):
            member = cell[octant == code]
            if member.size == 0:
                continue
            child_low = np.where(
                np.array([code & 4, code & 2, code & 1], dtype=bool), middle, cell_low
            )
            child_high = np.where(
                np.array([code & 4, code & 2, code & 1], dtype=bool), cell_high, middle
            )
            stack.append((member, child_low, child_high))
    return cells, oversized


def build_chunk_plan(
    positions: np.ndarray,
    *,
    importance: np.ndarray | None = None,
    max_per_chunk: int = DEFAULT_MAX_PER_CHUNK,
    tier_count: int = DEFAULT_TIER_COUNT,
    min_cell_size: float | None = None,
) -> ChunkPlan:
    """Partition ``positions`` into octree chunks and interleaved density tiers.

    ``importance`` orders elements inside a cell, most important first; ``None`` keeps
    source order (what point clouds want — their order carries scan structure and no
    element is more important than another).

    Tier ``k`` takes ranks ``k, k+T, k+2T, ...`` of every cell. Tier 0 therefore touches
    every non-empty cell (complete spatial coverage, 1/T density) and each later tier
    adds another 1/T everywhere. Chunk indices are emitted tier-major so ``tiers[k]`` is
    a contiguous run and a viewer can stream them in order.
    """
    points = np.ascontiguousarray(positions, dtype=np.float32)
    count = int(points.shape[0])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")
    if count < 1:
        raise ValueError("cannot chunk an empty element")
    if max_per_chunk < 1:
        raise ValueError("max_per_chunk must be >= 1")
    if tier_count < 1:
        raise ValueError("tier_count must be >= 1")
    if importance is not None and np.asarray(importance).shape != (count,):
        raise ValueError("importance must be (n,)")

    # Reject non-finite coordinates BEFORE subdividing. This is a LIVENESS guard, not
    # input tidiness: every comparison against NaN is False, so NaN points all fall into
    # the same octant, that cell's edge stays NaN, `edge <= min_cell_size` is never true,
    # and _subdivide recurses forever. One NaN coordinate anywhere in a source file hangs
    # a derive worker with no timeout and no error.
    #
    # Raising rather than dropping is deliberate. `order`/`dest` address the source array,
    # so filtering here would silently misalign every column the caller scatters by
    # `dest`; and quietly discarding elements is exactly what the no-silent-decimation
    # rule forbids. Callers that legitimately expect junk (the COLMAP path) filter and
    # count it themselves before calling.
    non_finite = int(count - int(np.count_nonzero(np.isfinite(points).all(axis=1))))
    if non_finite:
        raise ValueError(
            f"{non_finite} of {count} elements have a non-finite coordinate; "
            "filter them before chunking"
        )

    extent = float((points.max(axis=0) - points.min(axis=0)).max())
    floor_size = (
        extent * DEFAULT_MIN_CELL_FRACTION if min_cell_size is None else float(min_cell_size)
    )
    cells, oversized = _subdivide(
        points,
        np.arange(count, dtype=np.int64),
        max_per_chunk=max_per_chunk,
        min_cell_size=floor_size,
    )
    # Deterministic cell order: the octree stack visits octants in a fixed but
    # traversal-dependent sequence, so sort by first source index to pin the output.
    cells.sort(key=lambda cell: int(cell.min()))

    ordered: list[np.ndarray] = []
    for cell in cells:
        if importance is None:
            ordered.append(np.sort(cell))
            continue
        ranked = np.asarray(importance, dtype=np.float64)[cell]
        # Stable sort on the negated key: ties keep ascending source order, so two runs
        # over the same file produce byte-identical chunks.
        ordered.append(cell[np.argsort(-ranked, kind="stable")])

    chunks: list[Chunk] = []
    tiers: list[tuple[int, ...]] = []
    zero_origin = 0
    for tier in range(tier_count):
        members: list[int] = []
        for cell_index, cell in enumerate(ordered):
            selection = cell[tier::tier_count]
            if selection.size == 0:
                continue
            origin, local, snapped = _chunk_frame(points[selection])
            zero_origin += int(not snapped)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    tier=tier,
                    cell=cell_index,
                    order=selection,
                    origin=origin,
                    bbox_min=local.min(axis=0),
                    bbox_max=local.max(axis=0),
                )
            )
            members.append(chunks[-1].index)
        tiers.append(tuple(members))

    counts = np.array([chunk.count for chunk in chunks], dtype=np.int64)
    starts = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    if int(starts[-1]) != count:
        raise RuntimeError(
            f"chunk plan lost elements: {int(starts[-1])} of {count} placed"
        )  # pragma: no cover - guards the partition invariant
    dest = np.empty(count, dtype=np.int64)
    for chunk in chunks:
        dest[chunk.order] = np.arange(starts[chunk.index], starts[chunk.index + 1], dtype=np.int64)
    return ChunkPlan(
        chunks=tuple(chunks),
        tiers=tuple(tiers),
        dest=dest,
        starts=starts,
        oversized_cells=oversized,
        zero_origin_chunks=zero_origin,
    )
