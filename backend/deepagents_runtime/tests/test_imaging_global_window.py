"""Tile rendering must apply ONE global window per image, not libbioimage's per-tile
``-depth 8,D,U`` data-range — which auto-scales each tile to its own min/max and
checkerboards a tiled scalar (uint16) image. These tests pin the fix.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.engine import (
    LibBioImageEngine,
    _robust_window,
    _window_to_uint8,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_robust_window_is_p1_p99_and_never_degenerate():
    np = pytest.importorskip("numpy")
    values = np.arange(0, 1000, dtype="float32")
    lo, hi = _robust_window(values, np)
    assert lo == pytest.approx(float(np.percentile(values, 1.0)), abs=1e-2)
    assert hi == pytest.approx(float(np.percentile(values, 99.0)), abs=1e-2)
    # A constant region must not produce a zero-width window (division by zero).
    clo, chi = _robust_window(np.full(16, 7.0, dtype="float32"), np)
    assert chi > clo
    # Empty input is tolerated.
    assert _robust_window(np.zeros(0, dtype="float32"), np) == (0.0, 1.0)


def test_window_to_uint8_maps_a_value_identically_across_tiles():
    """THE anti-checkerboard property: with one shared global window, a given intensity
    maps to the same uint8 no matter which tile (local range) it appears in."""
    np = pytest.importorskip("numpy")
    window = [(0.0, 1000.0)]
    # Two tiles whose LOCAL ranges differ wildly but both contain the value 500.
    dark_tile = np.array([[0.0, 500.0], [100.0, 250.0]], dtype="float32")  # local max 500
    bright_tile = np.array([[500.0, 1000.0], [900.0, 1000.0]], dtype="float32")  # local max 1000
    out_dark = _window_to_uint8(dark_tile, window, np)
    out_bright = _window_to_uint8(bright_tile, window, np)
    # 500 -> ~127 in BOTH (global window), NOT 255 in the dark tile (which per-tile
    # data-range would have done).
    assert out_dark[0, 1] == out_bright[0, 0]
    assert 125 <= int(out_dark[0, 1]) <= 130
    assert out_dark.dtype == np.uint8


def test_window_to_uint8_handles_multichannel_and_preserves_rank():
    np = pytest.importorskip("numpy")
    chw = np.stack([np.full((2, 2), 100.0), np.full((2, 2), 800.0)]).astype("float32")
    out = _window_to_uint8(chw, [(0.0, 1000.0), (0.0, 1000.0)], np)
    assert out.shape == (2, 2, 2)
    assert int(out[0, 0, 0]) < int(out[1, 0, 0])  # channel 1 brighter than channel 0
    # 2-D in -> 2-D out.
    assert _window_to_uint8(np.zeros((4, 4), dtype="float32"), [(0.0, 1.0)], np).ndim == 2


def _scalar_engine(reads):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    scalar_meta = {
        "image_pixel_format": "unsigned integer",
        "image_pixel_depth": 16,
        "image_num_c": 1,
        "image_num_x": 5913,
        "image_num_y": 5679,
    }

    class FakeBim:
        def meta(self, path, cache):
            return scalar_meta

        def read(self, path, pipeline, cache):
            reads.append(pipeline)
            return np.linspace(0.0, 65535.0, 64, dtype="float32").reshape(8, 8)

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._np = np
    engine._Image = Image
    engine._display_window_cache = {}
    return engine


def test_scalar_tile_reads_raw_float_and_windows_globally():
    """A scalar (uint16) tile is rendered via the raw-float + global-window path, never
    the per-tile ``8,D,U`` data-range that produced the block artifacts."""
    reads: list[str] = []
    engine = _scalar_engine(reads)
    png = engine.tile("/scene.czi", level=0, col=0, row=0, tile_size=512)
    assert png[:8] == _PNG_MAGIC
    assert reads, "expected the engine to read at least the tile"
    # Every read used the raw FLOAT pipeline (32,D,F); none used per-tile 8,D,U.
    assert any("32,D,F" in p for p in reads)
    assert all("8,D,U" not in p for p in reads)


def test_scalar_global_window_is_cached():
    reads: list[str] = []
    engine = _scalar_engine(reads)
    w1 = engine._display_global_windows("/scene.czi")
    n_after_first = len(reads)
    w2 = engine._display_global_windows("/scene.czi")
    assert w1 == w2
    assert len(reads) == n_after_first  # second call hit the cache, no extra read
