"""NGFF reader/render/viewerinfo tests.

Requires zarr + numpy + Pillow (the ngff-service runtime deps). Skipped where those
aren't installed (e.g. the lean local dev venv); run in the ngff image / CI where they
are. Also self-runs without pytest via the __main__ block at the bottom (used to verify
inside the ngff container, which has zarr+PIL but no pytest)."""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    import zarr  # noqa: F401
    from PIL import Image
    _HAVE_DEPS = True
except Exception:  # noqa: BLE001
    _HAVE_DEPS = False

if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
    import pytest
    pytestmark = pytest.mark.skipif(not _HAVE_DEPS, reason="zarr/numpy/Pillow not installed")


def _make_zarr(path: str, sizes: list[int], *, channels: int = 1) -> None:
    """Write a minimal multiscale OME-Zarr (YX or CYX) with a gradient so tiles differ."""
    import numpy as np
    import zarr

    g = zarr.open_group(path, mode="w")
    datasets = []
    for i, s in enumerate(sizes):
        yy, xx = np.meshgrid(
            np.linspace(0, 60000, s, dtype=np.float32),
            np.linspace(0, 60000, s, dtype=np.float32),
            indexing="ij",
        )
        plane = (yy * 0.5 + xx * 0.5).astype(np.uint16)
        if channels > 1:
            data = np.stack([plane // (c + 1) for c in range(channels)], axis=0)
            shape, chunks = (channels, s, s), (1, 256, 256)
        else:
            data, shape, chunks = plane, (s, s), (256, 256)
        a = g.create_array(str(i), shape=shape, chunks=chunks, dtype="uint16")
        a[:] = data
        scale = ([1.0] if channels > 1 else []) + [2.0**i, 2.0**i]
        datasets.append({"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": scale}]})
    axes = ([{"name": "c", "type": "channel"}] if channels > 1 else []) + [
        {"name": "y", "type": "space"}, {"name": "x", "type": "space"}
    ]
    g.attrs["multiscales"] = [{"version": "0.4", "name": "t", "axes": axes, "datasets": datasets}]


def test_reader_opens_multiscale(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff

    p = str(tmp_path / "s.ome.zarr")
    _make_zarr(p, [1024, 512, 256])
    img = open_ngff(p)
    assert len(img.levels) == 3
    assert img.num_y == 1024 and img.num_x == 1024
    assert img.level_yx(1) == (512, 512)


def test_render_slice_and_tile_shapes(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.render import render_slice_png, render_tile_png

    p = str(tmp_path / "s.ome.zarr")
    _make_zarr(p, [1024, 512, 256])
    img = open_ngff(p)
    slice_png = render_slice_png(img, t=0, z=0, level=0)
    assert Image.open(io.BytesIO(slice_png)).size == (1024, 1024)
    # interior tile is full size; edge tile is cropped (1024/256 == 4 -> last col=3 full)
    interior = render_tile_png(img, level=0, col=0, row=0, tile_size=256)
    assert Image.open(io.BytesIO(interior)).size == (256, 256)
    # out-of-range tile returns a 1x1 placeholder, never an error
    oob = render_tile_png(img, level=0, col=999, row=999, tile_size=256)
    assert Image.open(io.BytesIO(oob)).size == (1, 1)
    # a tile crops the same pixels as the full slice (top-left 256x256 must match)
    full = np.asarray(Image.open(io.BytesIO(slice_png)).convert("L"))
    tile = np.asarray(Image.open(io.BytesIO(interior)).convert("L"))
    assert np.array_equal(full[:256, :256], tile)


def test_tile_scheme_threshold(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.viewerinfo import build_ngff_tile_scheme, build_ngff_viewer_info

    small = str(tmp_path / "small.ome.zarr")
    _make_zarr(small, [1024, 512])  # <= 2048 threshold -> direct
    assert build_ngff_tile_scheme(open_ngff(small)) is None
    assert build_ngff_viewer_info(open_ngff(small))["backend_mode"] == "direct"

    big = str(tmp_path / "big.ome.zarr")
    _make_zarr(big, [4096, 2048, 1024])  # > 2048 -> tiled
    ts = build_ngff_tile_scheme(open_ngff(big))
    assert ts is not None and len(ts["levels"]) == 3
    assert ts["levels"][0]["downsample"] == 1 and ts["levels"][2]["downsample"] == 4
    vi = build_ngff_viewer_info(open_ngff(big))
    assert vi["backend_mode"] == "pyramid" and vi["tile_scheme"] is not None


def _make_timelapse_zarr(path: str) -> None:
    """A 2-level T/Y/X OME-Zarr with NGFF units (time=hour, space=micrometer), a name, a
    version, and an omero block — exercises the metadata parsing the Metadata tab consumes."""
    import numpy as np
    import zarr

    g = zarr.open_group(path, mode="w")
    for i, s in enumerate((64, 32)):
        arr = np.tile(np.linspace(100, 9000, s, dtype=np.uint16), (5, s, 1))[:, :s, :s]
        a = g.create_array(str(i), shape=(5, s, s), chunks=(1, s, s), dtype="uint16")
        a[:] = arr
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "Sample-A1",
        "axes": [
            {"name": "t", "type": "time", "unit": "hour"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 0.5, 0.5]}]},
            {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 1.0, 1.0]}]},
        ],
    }]
    g.attrs["omero"] = {"name": "Sample-A1", "channels": [{"label": "Bright-field", "color": "ffffff"}]}


def test_viewer_info_metadata_enrichment(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    p = str(tmp_path / "tl.ome.zarr")
    _make_timelapse_zarr(p)
    md = build_ngff_viewer_info(open_ngff(p))["metadata"]
    assert md["array_shape"] == [5, 64, 64]  # real stored shape (T,Y,X) matching dims_order
    assert md["dims_order"] == "TYX"
    assert md["format"] == "OME-Zarr (NGFF 0.4)"
    assert md["array_min"] < md["array_max"]  # real intensity range computed
    assert md["acquisition"]["pyramid_levels"] == 2
    assert md["acquisition"]["source_name"] == "Sample-A1"
    # time axis scale 2.0 hour/frame over 5 frames -> interval "2 hours", duration "8 hours"
    assert md["microscopy"]["timelapse_interval"] == "2 hours"
    assert md["microscopy"]["total_time_duration"] == "8 hours"


def test_reader_parses_units_name_version(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff

    p = str(tmp_path / "tl.ome.zarr")
    _make_timelapse_zarr(p)
    img = open_ngff(p)
    assert img.version == "0.4" and img.name == "Sample-A1"
    assert img.units.get("x") == "micrometer" and img.units.get("t") == "hour"
    rng = img.intensity_range()
    assert rng is not None and rng[0] < rng[1]


def test_plane_cache_is_size_bounded(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff

    p = str(tmp_path / "s.ome.zarr")
    _make_zarr(p, [512, 256])
    img = open_ngff(p)
    a = img.read_plane(t=0, c=0, z=0, level=0)
    b = img.read_plane(t=0, c=0, z=0, level=0)
    # small plane is cached -> identical object returned on the second read
    assert a is b
    assert img._plane_cache_bytes > 0


if __name__ == "__main__":  # run without pytest (e.g. inside the ngff container)
    import tempfile

    if not _HAVE_DEPS:
        print("SKIP: zarr/numpy/Pillow not installed")
        sys.exit(0)
    fns = [test_reader_opens_multiscale, test_render_slice_and_tile_shapes,
           test_tile_scheme_threshold, test_plane_cache_is_size_bounded,
           test_viewer_info_metadata_enrichment, test_reader_parses_units_name_version]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            class _P:
                def __truediv__(self, name): return os.path.join(d, name)
            fn(_P())
        print("PASS", fn.__name__)
    print("ALL NGFF TESTS PASSED")
