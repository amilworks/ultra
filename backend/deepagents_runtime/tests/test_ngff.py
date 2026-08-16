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
        datasets.append(
            {"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": scale}]}
        )
    axes = ([{"name": "c", "type": "channel"}] if channels > 1 else []) + [
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    g.attrs["multiscales"] = [{"version": "0.4", "name": "t", "axes": axes, "datasets": datasets}]


def _write_single_level_ngff(
    path: str,
    *,
    dataset_transforms: list[dict[str, object]],
    global_transforms: list[dict[str, object]] | None = None,
    nested_v05: bool = False,
) -> None:
    """Write a tiny explicit v0.4/v0.5 transform fixture."""
    import zarr

    group = zarr.open_group(path, mode="w", zarr_format=3 if nested_v05 else 2)
    array_kwargs = {"dimension_names": ("y", "x")} if nested_v05 else {}
    group.create_array("0", shape=(4, 6), chunks=(4, 6), dtype="uint16", **array_kwargs)
    multiscale: dict[str, object] = {
        "name": "image-a",
        "axes": [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": [{"path": "0", "coordinateTransformations": dataset_transforms}],
    }
    if global_transforms is not None:
        multiscale["coordinateTransformations"] = global_transforms
    if nested_v05:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    else:
        multiscale["version"] = "0.4"
        group.attrs["multiscales"] = [multiscale]


def test_reader_opens_multiscale(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff

    p = str(tmp_path / "s.ome.zarr")
    _make_zarr(p, [1024, 512, 256])
    img = open_ngff(p)
    assert len(img.levels) == 3
    assert img.num_y == 1024 and img.num_x == 1024
    assert img.level_yx(1) == (512, 512)


def test_reader_rejects_unlabeled_zarr_instead_of_guessing_trailing_axes(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, is_ome_zarr, open_ngff

    path = str(tmp_path / "raw.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", shape=(2, 8, 9), chunks=(1, 4, 4), dtype="uint16")

    assert is_ome_zarr(path) is False
    with pytest.raises(NgffError, match="no OME-NGFF 'multiscales'.*not guessed"):
        open_ngff(path)


def test_reader_rejects_ambiguous_multiscales_and_accepts_explicit_selector(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    path = str(tmp_path / "multiple.ome.zarr")
    group = zarr.open_group(path, mode="w")
    group.create_array("a", shape=(4, 5), chunks=(4, 5), dtype="uint16")
    group.create_array("b", shape=(8, 9), chunks=(8, 9), dtype="uint16")
    axes = [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "first",
            "axes": axes,
            "datasets": [
                {
                    "path": "a",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
                }
            ],
        },
        {
            "version": "0.4",
            "name": "second",
            "axes": axes,
            "datasets": [
                {
                    "path": "b",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 3.0]}],
                }
            ],
        },
    ]

    with pytest.raises(NgffError, match="multiple multiscale images"):
        open_ngff(path)
    by_index = open_ngff(path, multiscale=1)
    by_name = open_ngff(path, multiscale="second")
    assert by_index.multiscale_index == 1
    assert by_index.multiscale_name == "second"
    assert by_index.levels[0].path == "b"
    assert by_name.levels[0].scale == (2.0, 3.0)


def test_reader_composes_dataset_then_global_transforms_and_exposes_units(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    path = str(tmp_path / "composed-v05.ome.zarr")
    _write_single_level_ngff(
        path,
        dataset_transforms=[
            {"type": "scale", "scale": [2.0, 3.0]},
            {"type": "translation", "translation": [5.0, 7.0]},
        ],
        global_transforms=[
            {"type": "scale", "scale": [10.0, 0.5]},
            {"type": "translation", "translation": [-1.0, 4.0]},
        ],
        nested_v05=True,
    )

    image = open_ngff(path)
    level = image.levels[0]
    # Dataset: (2p + [5,7]); global: (result * [10,.5] + [-1,4]).
    assert level.scale == pytest.approx((20.0, 1.5))
    assert level.translation == pytest.approx((49.0, 7.5))
    assert level.axis_units == ("micrometer", "nanometer")
    assert image.version == "0.5"
    assert image.physical["y"] == pytest.approx(20.0)
    assert image.translation["x"] == pytest.approx(7.5)
    assert image.units == {
        "t": "",
        "c": "",
        "z": "",
        "y": "micrometer",
        "x": "nanometer",
    }

    viewer = build_ngff_viewer_info(image)
    transform = viewer["metadata"]["ngff_coordinate_transforms"]
    assert transform["semantics"] == "effective-array-to-physical"
    assert transform["axes"] == [
        {"name": "y", "unit": "micrometer"},
        {"name": "x", "unit": "nanometer"},
    ]
    assert transform["levels"] == [
        {
            "path": "0",
            "scale": [20.0, 1.5],
            "translation": [49.0, 7.5],
        }
    ]
    assert viewer["phys"]["pixel_units"][:2] == ["nanometer", "micrometer"]


def test_reader_rejects_unsupported_order_or_dimension(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    invalid_cases = [
        (
            [{"type": "affine", "affine": [[1.0, 0.0], [0.0, 1.0]]}],
            "unsupported.*transformation type",
        ),
        (
            [
                {"type": "translation", "translation": [1.0, 2.0]},
                {"type": "scale", "scale": [3.0, 4.0]},
            ],
            "scale first",
        ),
        (
            [{"type": "scale", "scale": [1.0, 2.0, 3.0]}],
            "does not match dimensionality",
        ),
        (
            [{"type": "translation", "translation": [1.0, 2.0]}],
            "exactly one scale",
        ),
        (
            [{"type": "scale", "scale": ["1.0", 2.0]}],
            "finite JSON number",
        ),
        (
            [{"type": "scale", "scale": [0.0, 2.0]}],
            "scale must be positive",
        ),
        (
            [{"type": "scale", "scale": [1.0, 2.0], "translation": [3.0, 4.0]}],
            "conflicting 'translation' parameter",
        ),
    ]
    for index, (dataset_transforms, message) in enumerate(invalid_cases):
        path = str(tmp_path / f"invalid-{index}.ome.zarr")
        _write_single_level_ngff(path, dataset_transforms=dataset_transforms)
        with pytest.raises(NgffError, match=message):
            open_ngff(path)


def test_reader_rejects_invalid_global_transform_instead_of_ignoring_it(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    path = str(tmp_path / "invalid-global.ome.zarr")
    _write_single_level_ngff(
        path,
        dataset_transforms=[{"type": "scale", "scale": [1.0, 1.0]}],
        global_transforms=[{"type": "rotation", "angle": 0.5}],
    )
    with pytest.raises(NgffError, match="unsupported.*transformation type"):
        open_ngff(path)


def test_reader_rejects_non_finite_composed_transform(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    path = str(tmp_path / "overflow.ome.zarr")
    _write_single_level_ngff(
        path,
        dataset_transforms=[{"type": "scale", "scale": [1e308, 1.0]}],
        global_transforms=[{"type": "scale", "scale": [1e308, 1.0]}],
    )
    with pytest.raises(NgffError, match="composed.*non-finite"):
        open_ngff(path)


def test_reader_supports_bounded_path_backed_transform_vectors(tmp_path):
    import numpy as np
    import zarr
    from ultra_deepagents.ngff.reader import open_ngff

    path = str(tmp_path / "path-transform.ome.zarr")
    group = zarr.open_group(path, mode="w")
    group.create_array("0", shape=(4, 6), chunks=(4, 6), dtype="uint16")
    scale = group.create_array("scale-vector", shape=(2,), chunks=(2,), dtype="float64")
    scale[:] = np.asarray([0.25, 0.5])
    translation = group.create_array("translation-vector", shape=(2,), chunks=(2,), dtype="float64")
    translation[:] = np.asarray([10.0, -4.0])
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "path-backed",
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "path": "scale-vector"},
                        {"type": "translation", "path": "translation-vector"},
                    ],
                }
            ],
        }
    ]

    level = open_ngff(path).levels[0]
    assert level.scale == pytest.approx((0.25, 0.5))
    assert level.translation == pytest.approx((10.0, -4.0))


def test_reader_returns_canonical_yx_for_xy_stored_axis_order(tmp_path):
    import numpy as np
    import zarr
    from ultra_deepagents.ngff.reader import open_ngff

    path = str(tmp_path / "xy-order.ome.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=2)
    stored_xy = np.arange(6, dtype=np.uint16).reshape(2, 3)
    array = group.create_array("0", shape=stored_xy.shape, chunks=stored_xy.shape, dtype="uint16")
    array[:] = stored_xy
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [0.5, 2.0]}],
                }
            ],
        }
    ]

    image = open_ngff(path)
    assert image.level_yx(0) == (3, 2)
    plane = image.read_plane()
    assert plane.shape == (3, 2)
    assert np.array_equal(plane, stored_xy.T)
    region = image.read_region(level=0, y0=1, y1=3, x0=0, x1=2)
    assert np.array_equal(region, stored_xy[:, 1:3].T)


def test_reader_rejects_custom_non_singleton_axis_but_drops_lossless_singleton(tmp_path):
    import numpy as np
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    def write(path: str, custom_size: int) -> None:
        group = zarr.open_group(path, mode="w", zarr_format=2)
        array = group.create_array("0", shape=(custom_size, 2, 3), chunks=(1, 2, 3), dtype="uint16")
        array[:] = np.arange(custom_size * 6, dtype=np.uint16).reshape(custom_size, 2, 3)
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "phase", "type": "channel"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                    }
                ],
            }
        ]

    non_singleton = str(tmp_path / "custom-two.ome.zarr")
    write(non_singleton, 2)
    with pytest.raises(NgffError, match="unsupported non-singleton axis 'phase'"):
        open_ngff(non_singleton)

    singleton = str(tmp_path / "custom-one.ome.zarr")
    write(singleton, 1)
    image = open_ngff(singleton)
    assert image.read_plane().shape == (2, 3)


def test_reader_rejects_out_of_range_plane_and_level_indices(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff
    from ultra_deepagents.ngff.render import render_slice_png, render_tile_png

    path = str(tmp_path / "strict-indices.ome.zarr")
    _make_zarr(path, [8], channels=2)
    image = open_ngff(path)

    for kwargs, message in [
        ({"t": 1}, "t index 1"),
        ({"z": -1}, "z index -1"),
        ({"c": 2}, "c index 2"),
        ({"level": 1}, "multiscale level 1"),
        ({"level": -1}, "multiscale level -1"),
    ]:
        with pytest.raises(NgffError, match=message):
            image.read_plane(**kwargs)

    with pytest.raises(NgffError, match="region.*outside"):
        image.read_region(level=0, y0=0, y1=9, x0=0, x1=1)
    with pytest.raises(NgffError, match="c index 2"):
        render_slice_png(image, channels=[2])
    with pytest.raises(NgffError, match="multiscale level 3"):
        render_tile_png(image, level=3, col=0, row=0)


def test_service_channel_parser_rejects_malformed_tokens_instead_of_selecting_all():
    from ultra_deepagents.imaging.constants import MAX_COMPOSITE_CHANNELS
    from ultra_deepagents.ngff.service import _parse_channels

    assert _parse_channels("0, 2") == [0, 2]
    for malformed in ("", ",", "abc", "0,bad", "-1", "0,,1"):
        with pytest.raises(ValueError):
            _parse_channels(malformed)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_channels("0,0")
    with pytest.raises(ValueError, match=rf"at most {MAX_COMPOSITE_CHANNELS}"):
        _parse_channels(",".join(str(index) for index in range(MAX_COMPOSITE_CHANNELS + 1)))


@pytest.mark.parametrize("route", ["/slice", "/tile"])
def test_service_rejects_over_limit_channels_before_opening_storage(monkeypatch, route):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.ngff import service

    opened: list[str] = []

    def record_open(path: str):
        opened.append(path)
        raise AssertionError("storage must not open for an oversized channel selection")

    monkeypatch.setattr(service, "_get_image", record_open)
    client = TestClient(service.create_app())
    response = client.get(
        route,
        params={"path": "/untrusted/store", "channels": ",".join(str(i) for i in range(9))},
    )

    assert response.status_code == 422
    assert opened == []


@pytest.mark.parametrize("size", [0, -1, 1025])
def test_ngff_service_rejects_unbounded_tile_size_before_opening_storage(monkeypatch, size):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.ngff import service

    opened: list[str] = []

    def record_open(path: str):
        opened.append(path)
        raise AssertionError("storage must not open for an invalid tile size")

    monkeypatch.setattr(service, "_get_image", record_open)
    client = TestClient(service.create_app())

    response = client.get("/tile", params={"path": "/untrusted/store", "size": size})

    assert response.status_code == 422
    assert opened == []


def test_ngff_viewerinfo_preserves_ome_channel_defaults_in_canonical_shape(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    path = str(tmp_path / "ome-colors.ome.zarr")
    _make_zarr(path, [16], channels=3)
    group = zarr.open_group(path, mode="a")
    group.attrs["omero"] = {
        "channels": [
            {"label": "DAPI", "color": "0000FF", "active": True},
            {"label": "FITC", "color": "00FF00", "active": False},
            {"label": "TRITC", "color": "FF0000", "active": True},
        ]
    }

    viewer = build_ngff_viewer_info(open_ngff(path))

    expected_colors = [
        {"index": 0, "hex": "#0000FF", "rgb": [0, 0, 255]},
        {"index": 1, "hex": "#00FF00", "rgb": [0, 255, 0]},
        {"index": 2, "hex": "#FF0000", "rgb": [255, 0, 0]},
    ]
    assert viewer["phys"]["channel_colors"] == expected_colors
    assert viewer["channel_colors"] == expected_colors
    assert viewer["display_defaults"]["channels"] == [0, 2]
    assert viewer["display_defaults"]["channel_colors"] == ["#0000FF", "#00FF00", "#FF0000"]
    assert "channel_color" in viewer["viewer"]["display_capabilities"]
    assert "channel_lut_transport" in viewer["viewer"]["display_capabilities"]


def test_ngff_default_policy_caps_viewer_and_thumbnail_plane_reads(tmp_path, monkeypatch):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.render import render_thumbnail_png
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    path = str(tmp_path / "nine-channel.ome.zarr")
    _make_zarr(path, [8], channels=9)
    image = open_ngff(path)
    original_read_plane = image.read_plane
    read_channels: list[int] = []

    def record_read_plane(**kwargs):
        read_channels.append(int(kwargs.get("c", 0)))
        return original_read_plane(**kwargs)

    monkeypatch.setattr(image, "read_plane", record_read_plane)
    render_thumbnail_png(image)

    assert read_channels == list(range(8))
    assert build_ngff_viewer_info(image)["display_defaults"]["channels"] == list(range(8))


def test_intensity_range_is_exact_over_complete_smallest_level_or_unknown(tmp_path, monkeypatch):
    import numpy as np
    import ultra_deepagents.ngff.reader as reader
    import zarr
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    path = str(tmp_path / "range.ome.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=2)
    values = np.asarray(
        [
            [[0, 1], [1, 0]],
            [[50, 100], [150, 200]],
        ],
        dtype=np.uint16,
    )
    array = group.create_array("0", shape=values.shape, chunks=(1, 2, 2), dtype="uint16")
    array[:] = values
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                }
            ],
        }
    ]

    image = reader.open_ngff(path)
    assert image.intensity_range() == (0.0, 200.0)
    assert image.intensity_range_status == "exact_smallest_level"
    metadata = build_ngff_viewer_info(image)["metadata"]
    assert metadata["intensity_range"] == {
        "status": "exact_smallest_level",
        "scope": "complete_array",
        "level": 0,
        "minimum": 0.0,
        "maximum": 200.0,
    }
    assert (metadata["array_min"], metadata["array_max"]) == (0.0, 200.0)

    monkeypatch.setattr(reader, "_INTENSITY_RANGE_MAX_ELEMENTS", 7)
    budget_limited = reader.open_ngff(path)
    assert budget_limited.intensity_range() is None
    assert budget_limited.intensity_range_status == "unknown_budget_exceeded"


def test_reader_rejects_v05_missing_or_mismatched_dimension_names(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    for dimension_names in (None, ("x", "y")):
        suffix = "missing" if dimension_names is None else "mismatch"
        path = str(tmp_path / f"v05-{suffix}.ome.zarr")
        group = zarr.open_group(path, mode="w", zarr_format=3)
        kwargs = {} if dimension_names is None else {"dimension_names": dimension_names}
        group.create_array("0", shape=(4, 6), chunks=(4, 6), dtype="uint16", **kwargs)
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
                        }
                    ],
                }
            ],
        }

        expected = "no Zarr dimension_names" if dimension_names is None else "do not match axes"
        with pytest.raises(NgffError, match=expected):
            open_ngff(path)


def test_reader_rejects_v05_axis_name_type_count_and_order_contradictions(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    cases = [
        (
            [{"name": "y", "type": "channel"}, {"name": "x", "type": "space"}],
            (2, 3),
            "canonical axis 'y' requires type 'space'",
        ),
        (
            [
                {"name": "y", "type": "space"},
                {"name": "t", "type": "time"},
                {"name": "x", "type": "space"},
            ],
            (2, 1, 3),
            "order time first",
        ),
        (
            [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}],
            (1, 3),
            "exactly two or three space axes",
        ),
        (
            [
                {"name": "c", "type": "channel"},
                {"name": "phase", "type": "phase"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            (1, 1, 2, 3),
            "at most one channel, null, or custom axis",
        ),
    ]
    for index, (axes, shape, message) in enumerate(cases):
        path = str(tmp_path / f"axes-{index}.ome.zarr")
        group = zarr.open_group(path, mode="w", zarr_format=3)
        names = tuple(axis["name"] for axis in axes)
        group.create_array("0", shape=shape, chunks=shape, dtype="uint16", dimension_names=names)
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0] * len(shape)}
                            ],
                        }
                    ],
                }
            ],
        }

        with pytest.raises(NgffError, match=message):
            open_ngff(path)


def test_reader_rejects_reversed_resolution_shapes_and_decreasing_spatial_scale(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    def write(path: str, shapes, scales) -> None:
        group = zarr.open_group(path, mode="w", zarr_format=2)
        datasets = []
        for index, (shape, scale) in enumerate(zip(shapes, scales, strict=True)):
            group.create_array(str(index), shape=shape, chunks=shape, dtype="uint16")
            datasets.append(
                {
                    "path": str(index),
                    "coordinateTransformations": [{"type": "scale", "scale": scale}],
                }
            )
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ]

    reversed_path = str(tmp_path / "reversed.ome.zarr")
    write(reversed_path, [(4, 8), (8, 4)], [[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(NgffError, match="increases resolution dimensions.*y:4->8"):
        open_ngff(reversed_path)

    decreasing_scale_path = str(tmp_path / "decreasing-scale.ome.zarr")
    write(decreasing_scale_path, [(8, 8), (4, 4)], [[1.0, 1.0], [0.5, 2.0]])
    with pytest.raises(NgffError, match="spatial scale decreases.*y:1->0.5"):
        open_ngff(decreasing_scale_path)


def test_reader_rejects_nonspatial_dimension_changes_across_pyramid(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    for axis, axis_type in (("t", "time"), ("c", "channel"), ("phase", "phase")):
        path = str(tmp_path / f"changing-{axis}.ome.zarr")
        group = zarr.open_group(path, mode="w", zarr_format=2)
        datasets = []
        for index, shape in enumerate(((2, 8, 8), (1, 4, 4))):
            group.create_array(str(index), shape=shape, chunks=shape, dtype="uint16")
            datasets.append(
                {
                    "path": str(index),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0**index, 2.0**index]}
                    ],
                }
            )
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": axis, "type": axis_type},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ]

        with pytest.raises(NgffError, match=f"changes non-spatial dimensions.*{axis}:2->1"):
            open_ngff(path)


def test_reader_rejects_incompatible_dtype_across_pyramid(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    path = str(tmp_path / "dtype-mismatch.ome.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    group.create_array("1", shape=(4, 4), chunks=(4, 4), dtype="float32")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}],
                },
            ],
        }
    ]

    with pytest.raises(NgffError, match="dtype float32 does not match base dtype uint16"):
        open_ngff(path)


def test_reader_rejects_omero_channel_count_mismatch(tmp_path):
    import zarr
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    path = str(tmp_path / "omero-mismatch.ome.zarr")
    _make_zarr(path, [8], channels=2)
    group = zarr.open_group(path, mode="a")
    group.attrs["omero"] = {"channels": [{"label": "only-one", "color": "FFFFFF"}]}

    with pytest.raises(NgffError, match="omero.channels has 1 entries.*size 2"):
        open_ngff(path)


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


def test_first_tile_contrast_reads_only_bounded_smallest_level_patches(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.render import render_tile_png

    path = str(tmp_path / "bounded-window.ome.zarr")
    _make_zarr(path, [1024, 512])
    image = open_ngff(path)
    underlying = image.levels[-1].array
    accesses = []

    class TrackingArray:
        def __getitem__(self, index):
            accesses.append(index)
            return underlying[index]

    image.levels[-1].array = TrackingArray()
    render_tile_png(image, level=0, col=0, row=0, tile_size=256)

    assert accesses
    height, width = image.level_yx(len(image.levels) - 1)
    total_values = 0
    for y_slice, x_slice in accesses:
        y_count = int(y_slice.stop) - int(y_slice.start)
        x_count = int(x_slice.stop) - int(x_slice.start)
        total_values += y_count * x_count
        assert y_count < height or x_count < width
        assert y_count <= 128 and x_count <= 128
    assert len(accesses) <= 9
    assert total_values <= 9 * 128 * 128
    assert image._intensity_range_evaluated is False


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

    single_large = str(tmp_path / "single-large.ome.zarr")
    _make_zarr(single_large, [2049])
    single_ts = build_ngff_tile_scheme(open_ngff(single_large))
    assert single_ts is not None
    assert single_ts["levels"] == [
        {
            "level": 0,
            "width": 2049,
            "height": 2049,
            "columns": 9,
            "rows": 9,
            "downsample": 1,
        }
    ]
    assert build_ngff_viewer_info(open_ngff(single_large))["backend_mode"] == "pyramid"


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
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "Sample-A1",
            "axes": [
                {"name": "t", "type": "time", "unit": "hour"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 0.5, 0.5]}],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 1.0, 1.0]}],
                },
            ],
        }
    ]
    g.attrs["omero"] = {
        "name": "Sample-A1",
        "channels": [{"label": "Bright-field", "color": "ffffff"}],
    }


def test_viewer_info_metadata_enrichment(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

    p = str(tmp_path / "tl.ome.zarr")
    _make_timelapse_zarr(p)
    md = build_ngff_viewer_info(open_ngff(p))["metadata"]
    assert md["array_shape"] == [5, 64, 64]  # real stored shape (T,Y,X) matching dims_order
    assert md["dims_order"] == "TYX"
    assert md["format"] == "OME-Zarr (NGFF 0.4)"
    # The exact scan covers the complete smallest level, not level 0. Keep the generic
    # viewer's unqualified "Value range" absent until its label can carry that scope.
    assert "array_min" not in md and "array_max" not in md
    assert md["intensity_range"]["scope"] == "complete_smallest_multiscale_level"
    assert md["intensity_range"]["minimum"] < md["intensity_range"]["maximum"]
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
    from ultra_deepagents.ngff.reader import open_ngff, process_plane_cache_info

    p = str(tmp_path / "s.ome.zarr")
    _make_zarr(p, [512, 256])
    img = open_ngff(p)
    a = img.read_plane(t=0, c=0, z=0, level=0)
    b = img.read_plane(t=0, c=0, z=0, level=0)
    # small plane is cached -> identical object returned on the second read
    assert a is b
    assert img._plane_cache_bytes > 0
    assert process_plane_cache_info()["bytes"] >= img._plane_cache_bytes


def test_decoded_plane_cache_budget_is_process_wide_across_open_images(tmp_path):
    from ultra_deepagents.ngff import reader

    first_path = str(tmp_path / "first.ome.zarr")
    second_path = str(tmp_path / "second.ome.zarr")
    _make_zarr(first_path, [64])
    _make_zarr(second_path, [64])
    first = reader.open_ngff(first_path)
    second = reader.open_ngff(second_path)
    plane_bytes = 64 * 64 * 2
    original_max = reader._PLANE_CACHE_MAX_BYTES
    try:
        reader.clear_process_plane_cache()
        reader._PLANE_CACHE_MAX_BYTES = plane_bytes + 1
        first_plane = first.read_plane()
        assert first.read_plane() is first_plane
        second_plane = second.read_plane()
        assert second.read_plane() is second_plane

        info = reader.process_plane_cache_info()
        assert info["bytes"] == plane_bytes
        assert info["bytes"] <= info["max_bytes"]
        assert info["entries"] == 1
        assert first._plane_cache_bytes == 0
        assert not first._plane_cache
        assert second._plane_cache_bytes == plane_bytes
    finally:
        reader.clear_process_plane_cache()
        reader._PLANE_CACHE_MAX_BYTES = original_max


def test_service_cached_image_detects_chunk_only_plane_mutation(tmp_path):
    import numpy as np
    import zarr
    from ultra_deepagents.ngff import service

    path = str(tmp_path / "mutable.ome.zarr")
    _make_zarr(path, [64])
    service._open_cache.clear()
    image = service._get_image(path)
    original = image.read_plane()
    assert image.read_plane() is original
    metadata_stamp = service._stat_stamp(path)

    writer = zarr.open_group(path, mode="a")["0"]
    writer[:] = np.full(writer.shape, 4321, dtype=np.uint16)
    assert service._stat_stamp(path) == metadata_stamp
    assert service._get_image(path) is image

    refreshed = image.read_plane()
    assert refreshed is not original
    assert np.all(refreshed == 4321)
    assert image.read_plane() is refreshed


if __name__ == "__main__":  # run without pytest (e.g. inside the ngff container)
    import tempfile

    if not _HAVE_DEPS:
        print("SKIP: zarr/numpy/Pillow not installed")
        sys.exit(0)
    fns = [
        test_reader_opens_multiscale,
        test_render_slice_and_tile_shapes,
        test_tile_scheme_threshold,
        test_plane_cache_is_size_bounded,
        test_decoded_plane_cache_budget_is_process_wide_across_open_images,
        test_viewer_info_metadata_enrichment,
        test_reader_parses_units_name_version,
    ]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:

            class _P:
                def __truediv__(self, name):
                    return os.path.join(d, name)

            fn(_P())
        print("PASS", fn.__name__)
    print("ALL NGFF TESTS PASSED")
