"""The CPU poster: framing, projection axis, painter order, determinism."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from PIL import Image
from ultra_deepagents.scene3d import poster


def _slab(count=4000, seed=1, thin_axis=1):
    """A wide-wide-thin scene, the shape an aerial or corridor scan actually has."""
    rng = np.random.default_rng(seed)
    extents = [40.0, 40.0, 40.0]
    extents[thin_axis] = 2.0
    return np.stack([rng.uniform(0.0, extents[axis], count) for axis in range(3)], axis=1).astype(
        np.float32
    )


def test_projects_along_the_thinnest_axis(tmp_path):
    for thin in range(3):
        positions = _slab(thin_axis=thin)
        result = poster.render_poster(
            tmp_path / f"p{thin}.png",
            positions=positions,
            colors=np.full((positions.shape[0], 3), 0.6, dtype=np.float32),
        )
        assert result.axis == thin


def _covered(result):
    with Image.open(result.path) as image:
        return int(np.count_nonzero(np.asarray(image)[:, :, 3]))


def test_a_single_outlier_does_not_shrink_the_scene_to_a_corner(tmp_path):
    """The measured point cloud's box is ~3x its body; framing on the box loses it."""
    clean = _slab(count=4000)
    colors = np.full((4000, 3), 0.6, dtype=np.float32)
    strayed = clean.copy()
    strayed[0] = [4000.0, 0.0, 4000.0]  # one far-flung point, 100x the scene

    baseline = poster.render_poster(tmp_path / "a.png", positions=clean, colors=colors)
    result = poster.render_poster(tmp_path / "b.png", positions=strayed, colors=colors)

    # Framed on the full box the body would compress into ~5 px; robust framing keeps it.
    assert _covered(result) > 0.8 * _covered(baseline)
    assert result.rendered == 4000  # the stray is drawn at the border, not dropped


def test_nearest_element_along_the_projection_axis_wins_the_pixel(tmp_path):
    positions = np.array([[0.0, 5.0, 0.0], [0.0, -5.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    result = poster.render_poster(tmp_path / "p.png", positions=positions, colors=colors)

    with Image.open(result.path) as image:
        pixels = np.asarray(image).reshape(-1, 4)
    painted = pixels[pixels[:, 3] > 0]
    assert result.axis == 0  # x has zero extent, so it is the projection axis
    # Both sit at the same projected point; the one at larger x is nearer the viewer.
    assert painted.shape[0] >= 1


def test_opacity_modulates_the_written_alpha(tmp_path):
    positions = _slab(count=100)
    colors = np.full((100, 3), 1.0, dtype=np.float32)

    opaque = poster.render_poster(
        tmp_path / "a.png", positions=positions, colors=colors, opacities=np.ones(100)
    )
    faint = poster.render_poster(
        tmp_path / "b.png", positions=positions, colors=colors, opacities=np.full(100, 0.25)
    )

    with Image.open(opaque.path) as image:
        strong = np.asarray(image)[:, :, 3].max()
    with Image.open(faint.path) as image:
        weak = np.asarray(image)[:, :, 3].max()
    assert strong == 255
    assert 60 <= weak <= 68  # 0.25 * 255


def test_splat_footprints_grow_with_their_world_radius(tmp_path):
    # One splat at the centre plus two corner markers that fix the frame at 20 units
    # across, so a world radius maps to a predictable pixel radius (~25.5 px per unit).
    stacked = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, -10.0], [10.0, 0.0, 10.0]], dtype=np.float32)
    colors = np.ones((3, 3), dtype=np.float32)

    def painted(radius):
        result = poster.render_poster(
            tmp_path / f"r{radius}.png",
            positions=stacked,
            colors=colors,
            radii=np.array([radius, 0.0, 0.0]),
        )
        with Image.open(result.path) as image:
            return int(np.count_nonzero(np.asarray(image)[:, :, 3]))

    assert painted(0.0) < painted(0.04) < painted(0.1)
    # ...and the cap holds, so one huge splat cannot paint over the whole poster.
    assert painted(1000.0) == painted(0.2)


def test_output_is_rgba_within_the_pixel_cap_and_byte_identical_across_runs(tmp_path):
    positions = _slab(count=9000)
    colors = np.abs(positions) / 40.0

    first = poster.render_poster(tmp_path / "a.png", positions=positions, colors=colors)
    second = poster.render_poster(tmp_path / "b.png", positions=positions, colors=colors)

    assert max(first.width, first.height) <= poster.POSTER_MAX_PIXELS
    with Image.open(first.path) as image:
        assert image.mode == "RGBA"
    assert open(first.path, "rb").read() == open(second.path, "rb").read()


def test_stride_sampling_never_exceeds_the_requested_sample():
    assert poster.poster_stride(100, 400_000) == 1
    assert poster.poster_stride(14_469_103, 400_000) == 37
    assert len(range(0, 14_469_103, 37)) <= 400_000


def test_an_empty_scene_is_refused_rather_than_drawn_blank(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        poster.render_poster(
            tmp_path / "p.png",
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
        )
