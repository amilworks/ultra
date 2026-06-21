"""Unit tests for the libbioimage read-pipeline builders.

Pure-stdlib: runnable under pytest or directly (``python tests/test_imaging_pipelines.py``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging import pipelines as p


def test_tile_basic():
    assert p.tile(512, 1, 2, 3) == "-tile 512,1,2,3 -depth 8,D,U"


def test_tile_with_channels():
    assert p.tile(256, 0, 0, 0, channels=[2, 1, 3]) == "-tile 256,0,0,0 -remap 2,1,3 -depth 8,D,U"


def test_region_with_scale():
    assert p.region(10, 20, 110, 120, region_scale=0.5) == "-tile-roi 10,20,110,120,0.5 -depth 8,D,U"


def test_region_without_scale():
    assert p.region(0, 0, 256, 256) == "-tile-roi 0,0,256,256 -depth 8,D,U"


def test_slice_plane_z_and_t():
    assert p.slice_plane(z=5, t=0) == "-slice z:5,t:0 -depth 8,D,U"


def test_slice_plane_with_level_and_channels():
    assert p.slice_plane(z=2, level=1, channels=[1]) == "-slice z:2 -res-level 1 -remap 1 -depth 8,D,U"


def test_slice_plane_max_dim_caps_long_edge_for_scrub():
    # A transient z-scrub frame caps the long edge so a huge plane WITHOUT a pyramid
    # (where -res-level is a no-op) still returns a small, fast image. The resize uses
    # MX (bounding box, keep AR, no upsample) — same token /thumbnail uses.
    assert p.slice_plane(z=5, max_dim=1024) == "-slice z:5 -resize 1024,1024,BC,MX -depth 8,D,U"
    # The native (settled) read sends no max_dim -> full plane for measurements.
    assert "resize" not in p.slice_plane(z=5)


def test_page_plane_addresses_a_document_page():
    # Paged z-stacks (plain multi-page TIFF) are addressed by -page N, not -slice.
    assert p.page_plane(3) == "-page 3 -depth 8,D,U"
    assert p.page_plane(2, level=1, channels=[1]) == "-page 2 -res-level 1 -remap 1 -depth 8,D,U"


def test_page_plane_max_dim_caps_long_edge_for_scrub():
    assert p.page_plane(3, max_dim=1024) == "-page 3 -resize 1024,1024,BC,MX -depth 8,D,U"


def test_thumbnail_zscrub_frame():
    # One z-scrub frame: pick z, bound to max_size keeping aspect ratio, no upsample.
    assert p.thumbnail(256, z=3) == "-slice z:3 -resize 256,256,BC,MX -depth 8,D,U"


def test_thumbnail_flat():
    assert p.thumbnail(128) == "-resize 128,128,BC,MX -depth 8,D,U"


def test_atlas_auto():
    assert p.atlas() == "-textureatlas -depth 8,D,U"


def test_atlas_grid_and_level():
    assert p.atlas(grid=(5, 7), level=2) == "-res-level 2 -texturegrid 5,7 -depth 8,D,U"


def test_scalar_volume_plane_is_float():
    assert p.scalar_volume_plane(5, channel=0) == "-slice z:5 -remap 1 -depth 32,D,F"


def test_token_builders():
    assert p.depth(32, "D", "F") == "-depth 32,D,F"
    assert p.remap([1, 2, 3]) == "-remap 1,2,3"
    assert p.res_level(2) == "-res-level 2"
    assert p.scale(0.25) == "-scale 0.25"
    assert p.resize(640, 0, "NN") == "-resize 640,0,NN"
    assert p.nd_slice(z=2) == "-slice z:2"


def test_build_drops_empties():
    assert p.build("-a", None, "", "-b") == "-a -b"


@pytest.mark.parametrize(
    "call",
    [
        lambda: p.tile(0, 0, 0, 0),
        lambda: p.tile(512, -1, 0, 0),
        lambda: p.region(100, 0, 50, 100),  # x2 <= x1
        lambda: p.region(0, 0, 10, 10, region_scale=1.5),
        lambda: p.scale(0.0),
        lambda: p.depth(7),
        lambda: p.depth(8, "Z"),
        lambda: p.remap([]),
        lambda: p.nd_slice(),
        lambda: p.nd_slice(c=1),  # channels go through remap, not slice
        lambda: p.resize(0, 0),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def _main() -> int:
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        marks = getattr(fn, "pytestmark", [])
        param = next((m for m in marks if m.name == "parametrize"), None)
        cases = [(c,) for c in param.args[1]] if param else [()]
        for args in cases:
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL {fn.__name__}{args}")
                traceback.print_exc()
    print(f"\n{len(fns)} test fns, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
