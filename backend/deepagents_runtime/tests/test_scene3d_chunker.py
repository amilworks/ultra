"""Octree chunking: exact partition, exact chunk-local reconstruction, honest tiers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from ultra_deepagents.scene3d import chunker


def _corridor(count=20000, seed=5):
    """The measured point-cloud aspect: 449.7 x 112.6 x 1119.2, offset far from origin."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(1000.0, 1449.7, count),
            rng.uniform(-200.0, -87.4, count),
            rng.uniform(0.0, 1119.2, count),
        ],
        axis=1,
    ).astype(np.float32)


def test_every_element_lands_in_exactly_one_chunk():
    positions = _corridor()

    plan = chunker.build_chunk_plan(positions, max_per_chunk=1000, tier_count=3)

    placed = np.concatenate([chunk.order for chunk in plan.chunks])
    assert plan.total == positions.shape[0]
    assert np.array_equal(np.sort(placed), np.arange(positions.shape[0]))
    assert sum(chunk.count for chunk in plan.chunks) == positions.shape[0]


def test_chunk_local_plus_origin_reconstructs_world_coordinates_exactly():
    """float32 addition, the arithmetic the shader performs — not an approximate match."""
    positions = _corridor(count=50000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=2000, tier_count=2)

    for chunk in plan.chunks:
        world = positions[chunk.order]
        local = (world - chunk.origin).astype(np.float32)
        assert np.array_equal(local + chunk.origin, world)
        assert np.array_equal(chunk.bbox_min, local.min(axis=0))
        assert np.array_equal(chunk.bbox_max, local.max(axis=0))
    assert plan.zero_origin_chunks == 0  # a real scene never needs the fallback


def test_chunk_local_magnitudes_are_small_even_far_from_the_origin():
    """The whole point of an origin: a 1449-unit coordinate becomes a sub-cell offset."""
    positions = _corridor(count=50000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=2000, tier_count=1)

    largest_local = max(float(np.abs(chunk.bbox_max).max()) for chunk in plan.chunks)
    assert float(np.abs(positions).max()) > 1000.0
    assert largest_local < 300.0


def test_pathological_magnitude_spread_falls_back_to_a_zero_origin_and_says_so():
    """A cell spanning -1000 and 1e-6 cannot round-trip 1e-6 through a -1000 offset.

    1e-6 - (-1000) rounds to exactly 1000.0 in float32, and adding the origin back gives
    0.0 rather than 1e-6. The fallback keeps world coordinates instead, which is exact.
    """
    positions = np.array(
        [[-1000.0, -1000.0, -1000.0], [1e-6, 1e-6, 1e-6], [0.0, 0.0, 0.0]], dtype=np.float32
    )

    plan = chunker.build_chunk_plan(positions, max_per_chunk=8, tier_count=1)

    assert plan.zero_origin_chunks == 1
    chunk = plan.chunks[0]
    assert not chunk.origin.any()
    world = positions[chunk.order]
    assert np.array_equal((world - chunk.origin).astype(np.float32) + chunk.origin, world)


def test_chunks_respect_the_size_target():
    positions = _corridor(count=30000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=1000, tier_count=1)

    assert plan.oversized_cells == 0
    assert max(chunk.count for chunk in plan.chunks) <= 1000
    assert len(plan.chunks) >= 30


def test_coincident_points_stop_at_the_size_floor_and_are_counted_not_dropped():
    """A million duplicates at one coordinate must terminate, keeping every element."""
    positions = np.zeros((5000, 3), dtype=np.float32)
    positions[:, 0] = 1.0

    plan = chunker.build_chunk_plan(positions, max_per_chunk=100, tier_count=1)

    assert plan.oversized_cells >= 1
    assert plan.total == 5000  # nothing dropped to satisfy the target
    assert max(chunk.count for chunk in plan.chunks) > 100


def test_tier_zero_covers_every_cell_at_reduced_density():
    positions = _corridor(count=30000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=1000, tier_count=3)

    cells = {chunk.cell for chunk in plan.chunks}
    tier_zero = {plan.chunks[index].cell for index in plan.tiers[0]}
    assert tier_zero == cells  # complete spatial coverage, not a spatial subset
    drawn = sum(plan.chunks[index].count for index in plan.tiers[0])
    assert 0.30 < drawn / plan.total < 0.36  # roughly a third, i.e. reduced density
    assert sum(len(tier) for tier in plan.tiers) == len(plan.chunks)


def test_tiers_are_disjoint_and_together_hold_everything():
    positions = _corridor(count=12000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=500, tier_count=4)

    seen = np.concatenate([plan.chunks[index].order for tier in plan.tiers for index in tier])
    assert np.array_equal(np.sort(seen), np.arange(12000))
    assert len(plan.tiers) == 4


def test_within_a_chunk_elements_are_ordered_by_descending_importance():
    """A prefix of a chunk must be a valid decimation of it."""
    rng = np.random.default_rng(2)
    positions = rng.uniform(-1.0, 1.0, (4000, 3)).astype(np.float32)
    importance = rng.uniform(0.0, 1.0, 4000)

    plan = chunker.build_chunk_plan(
        positions, importance=importance, max_per_chunk=500, tier_count=1
    )

    for chunk in plan.chunks:
        ranked = importance[chunk.order]
        assert np.all(np.diff(ranked) <= 0.0)


def test_ordering_is_deterministic_for_tied_importance():
    positions = np.stack([np.arange(500, dtype=np.float32)] * 3, axis=1)
    importance = np.ones(500)

    first = chunker.build_chunk_plan(positions, importance=importance, max_per_chunk=100)
    second = chunker.build_chunk_plan(positions, importance=importance, max_per_chunk=100)

    for left, right in zip(first.chunks, second.chunks, strict=True):
        assert np.array_equal(left.order, right.order)
    # Ties keep ascending source order rather than an arbitrary sort permutation.
    assert np.array_equal(first.chunks[0].order, np.sort(first.chunks[0].order))


def test_points_keep_source_order_when_no_importance_is_supplied():
    positions = _corridor(count=3000)

    plan = chunker.build_chunk_plan(positions, importance=None, max_per_chunk=200, tier_count=1)

    for chunk in plan.chunks:
        assert np.array_equal(chunk.order, np.sort(chunk.order))


def test_dest_maps_each_source_element_to_its_output_row():
    positions = _corridor(count=5000)

    plan = chunker.build_chunk_plan(positions, max_per_chunk=400, tier_count=2)

    assert np.array_equal(np.sort(plan.dest), np.arange(5000))
    for chunk in plan.chunks:
        rows = plan.dest[chunk.order]
        assert np.array_equal(
            rows, np.arange(plan.starts[chunk.index], plan.starts[chunk.index + 1])
        )


def test_splat_importance_is_opacity_times_largest_axis_scale():
    raw_opacity = np.array([0.6668, -4.1553], dtype=np.float32)
    ln_scales = np.array([[-4.6392, -5.0, -6.0], [-1.0, -0.5, -2.0]], dtype=np.float32)

    ranked = chunker.splat_importance(raw_opacity, ln_scales)

    # Median opacity 0.6668 -> sigmoid 0.6608, median log-scale -4.6392 -> exp 0.00967:
    # both from the contract's Appendix A measurement of the drone scene.
    assert ranked[0] == pytest.approx(0.660783 * np.exp(-4.6392), rel=1e-4)
    # A nearly-transparent splat can still outrank it if it is far larger, which is the
    # point of the product: visual contribution, not opacity alone.
    assert ranked[1] == pytest.approx(0.015433 * np.exp(-0.5), rel=1e-3)
    assert ranked[1] > ranked[0]


def test_empty_and_malformed_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        chunker.build_chunk_plan(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        chunker.build_chunk_plan(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="tier_count"):
        chunker.build_chunk_plan(np.zeros((4, 3), dtype=np.float32), tier_count=0)


def test_non_finite_coordinates_raise_instead_of_hanging():
    """A NaN coordinate must fail fast, not spin the octree forever.

    Every comparison against NaN is False, so NaN points all land in the same octant and
    that cell's edge never shrinks below the size floor — `_subdivide` recurses without
    bound. Before this guard, a single NaN in a source file wedged a derive worker with
    no timeout and no error, which on the real queue means a redelivery loop.
    """
    positions = np.zeros((50, 3), dtype=np.float32)
    positions[1:, :] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        chunker.build_chunk_plan(positions, max_per_chunk=4, tier_count=2)


def test_infinite_coordinates_are_rejected_too():
    positions = np.zeros((16, 3), dtype=np.float32)
    positions[3, 1] = np.inf

    with pytest.raises(ValueError, match="1 of 16"):
        chunker.build_chunk_plan(positions, max_per_chunk=4, tier_count=2)
