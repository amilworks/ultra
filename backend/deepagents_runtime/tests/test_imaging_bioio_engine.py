"""Tests for the pure-Python raster engine (imaging/bioio_engine.py).

The engine is the middle tier of the ladder — it must decode REAL pixels from
real files (never placeholders), keep HDF5 on the h5py path, and map every
unreadable input to the 422 "decode" class rather than a 500.
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from ultra_deepagents.imaging.bioio_engine import (  # noqa: E402
    BioioEngine,
    _max_chunk_bytes,
    _tiff_spacing,
    _tiff_spacing_with_units,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_DECODE_MARKERS = ("empty region", "cannot encode", "cannot decode", "unsupported")


def test_decoder_chunk_shape_accounting_handles_zarr_and_dask_layouts():
    assert _max_chunk_bytes((1, 1, 924), np.dtype("uint16")) == 924 * 2
    assert (
        _max_chunk_bytes(
            ((1, 1), (1, 1, 1), (1,) * 80, (312, 312), (462, 462)),
            np.dtype("uint16"),
        )
        == 312 * 462 * 2
    )


def _size(png: bytes) -> tuple[int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(png)).size


@pytest.fixture()
def engine():
    return BioioEngine()


@pytest.fixture()
def ome_tiff(tmp_path):
    """A multichannel z-stack OME-TIFF with names + physical spacing."""
    path = str(tmp_path / "brain.ome.tiff")
    rng = np.random.default_rng(3)
    data = (rng.random((1, 3, 8, 256, 320)) * 4000).astype("uint16")
    tifffile.imwrite(
        path,
        data,
        ome=True,
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeX": 0.11,
            "PhysicalSizeY": 0.11,
            "PhysicalSizeZ": 0.29,
            "Channel": {"Name": ["DAPI", "EGFP", "CMDRP"]},
        },
    )
    return path


@pytest.fixture()
def seven_channel_ome_tiff(tmp_path):
    """Small real OME container with the authoritative T/C/Z identity."""
    path = str(tmp_path / "seven-channel.ome.tiff")
    rng = np.random.default_rng(17)
    data = rng.integers(0, 4096, size=(1, 7, 80, 12, 18), dtype="uint16")
    yy, xx = np.mgrid[:12, :18]
    structured = [
        ((xx + yy) * 700).astype("uint16"),
        (((xx // 3) + (yy // 3)) % 2 * 50_000).astype("uint16"),
        ((xx * xx + yy * yy) * 90).astype("uint16"),
    ]
    for channel, plane in zip((1, 3, 5), structured, strict=True):
        data[0, channel, :, :, :] = plane
    names = ["CMDRP_1", "CMDRP", "EGFP_1", "EGFP", "H3342_1", "H3342", "Bright_100X"]
    tifffile.imwrite(
        path,
        data,
        ome=True,
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeX": 0.1083333333333,
            "PhysicalSizeY": 0.1083333333333,
            "PhysicalSizeZ": 0.29,
            "Channel": {"Name": names},
        },
    )
    return path


@pytest.fixture()
def rgb_tiff(tmp_path):
    path = str(tmp_path / "photo.tif")
    rgb = (np.random.default_rng(4).random((120, 160, 3)) * 255).astype("uint8")
    tifffile.imwrite(path, rgb, photometric="rgb")
    return path


@pytest.fixture()
def signature_tiff(tmp_path):
    """3 channels with disjoint bright regions, so a render identifies its channel."""
    path = str(tmp_path / "signature.ome.tiff")
    data = np.zeros((1, 3, 1, 40, 60), dtype="uint16")
    data[0, 0, 0, :10, :] = 60000  # channel 0 -> top band
    data[0, 1, 0, :, :10] = 60000  # channel 1 -> left band
    data[0, 2, 0, 15:25, 25:35] = 60000  # channel 2 -> centre square
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "TCZYX"})
    return path


@pytest.fixture()
def coordinate_tiff(tmp_path):
    """Big-endian T2/C2/Z3 pixels encoding every scientific coordinate."""
    path = str(tmp_path / "coordinates.ome.tiff")
    tt, cc, zz, yy, xx = np.indices((2, 2, 3, 4, 6), dtype="uint16")
    data = (tt * 1000 + cc * 300 + zz * 50 + yy * 6 + xx).astype(">u2")
    tifffile.imwrite(
        path,
        data,
        ome=True,
        byteorder=">",
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeZ": 1.0,
            "Channel": {"Name": ["C0", "C1"]},
        },
    )
    return path, np.asarray(data, dtype="uint16")


def _which_channel(png: bytes) -> str:
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    if a[:10, :].mean() > 128:
        return "top"
    if a[:, :10].mean() > 128:
        return "left"
    if a[15:25, 25:35].mean() > 128:
        return "centre"
    return "none"


@pytest.mark.parametrize("wire_index,expected", [(0, "top"), (1, "left"), (2, "centre")])
def test_channel_indices_are_one_based_on_this_interface(
    engine, signature_tiff, wire_index, expected
):
    # service._parse_fusion_request shifts the 0-based wire request into
    # libbioimage's 1-based -remap space before calling the engine. Indexing those
    # values 0-based renders a NEIGHBOURING channel, silently, with a 200.
    png = engine.slice_plane(signature_tiff, z=0, channels=[wire_index + 1])
    assert _which_channel(png) == expected


def test_planar_rgb_is_not_transposed(engine, tmp_path):
    # planarconfig="separate" gives axes SYX; resolving the sample axis after the
    # read assumed interleaved (H,W,S) and served a transposed sliver.
    path = str(tmp_path / "planar.tif")
    rgb = (np.random.default_rng(1).random((3, 32, 48)) * 255).astype("uint8")
    tifffile.imwrite(path, rgb, photometric="rgb", planarconfig="separate")
    meta = engine.meta(path)
    assert (meta["image_num_x"], meta["image_num_y"]) == (48, 32)
    assert _size(engine.thumbnail(path, max_size=256)) == (48, 32)


def test_gray_plus_alpha_is_not_transposed(engine, tmp_path):
    path = str(tmp_path / "gray_alpha.tif")
    data = (np.random.default_rng(2).random((40, 60, 2)) * 255).astype("uint8")
    tifffile.imwrite(path, data, photometric="minisblack", extrasamples="unassalpha")
    assert _size(engine.thumbnail(path, max_size=256)) == (60, 40)


def test_plain_multipage_tiff_exposes_pages_as_depth(engine, tmp_path):
    # A paged stack reports no Z axis; collapsing to z=1 served only page 0.
    path = str(tmp_path / "pages.tif")
    tifffile.imwrite(path, (np.random.default_rng(3).random((12, 30, 40)) * 255).astype("uint8"))
    assert engine.meta(path)["image_num_z"] == 12
    assert engine.slice_plane(path, z=0) != engine.slice_plane(path, z=7)


def test_rgb_photo_keeps_colour_fidelity(engine, tmp_path):
    # A percentile stretch renders a uniform photo as solid black/white.
    path = str(tmp_path / "flat.tif")
    tifffile.imwrite(path, np.full((20, 30, 3), 128, dtype="uint8"), photometric="rgb")
    from PIL import Image

    pixels = np.asarray(Image.open(io.BytesIO(engine.thumbnail(path, max_size=64))).convert("RGB"))
    assert 120 <= pixels.mean() <= 136


def test_atlas_matches_the_advertised_scheme(engine, ome_tiff):
    # The frontend decodes the atlas with viewer-info's atlas_scheme; rendering
    # cells at native size mis-tiles the volume and inflates the PNG.
    from ultra_deepagents.imaging import viewerinfo

    meta = engine.meta(ome_tiff)
    scheme = viewerinfo.build_atlas_scheme(meta, depth=meta["image_num_z"])
    plan = engine.atlas_plan(ome_tiff)
    assert (plan["columns"], plan["rows"], plan["cell_w"], plan["cell_h"]) == (
        scheme["columns"],
        scheme["rows"],
        scheme["slice_width"],
        scheme["slice_height"],
    )
    assert _size(engine.atlas(ome_tiff)) == (scheme["atlas_width"], scheme["atlas_height"])


def test_pool_fanout_methods_exist_and_agree(engine, ome_tiff):
    # imaging/atlas.py's orchestrator calls these in every multi-process service;
    # missing them is a hard 500 on /atlas + /scalar-volume.
    for name in (
        "atlas_plan",
        "atlas_windows",
        "atlas_cell",
        "atlas_cells",
        "scalar_plan",
        "scalar_planes",
    ):
        assert callable(getattr(engine, name, None)), f"missing {name}"
    plan = engine.atlas_plan(ome_tiff)
    windows = engine.atlas_windows(
        ome_tiff,
        depth=plan["depth"],
        level=plan["read_level"],
        channels=plan["read_channels"],
        paged=plan["paged"],
    )
    cells = engine.atlas_cells(
        ome_tiff,
        zs=[0, 1],
        level=plan["read_level"],
        channels=plan["read_channels"],
        colors=plan["cell_colors"],
        windows=windows,
        cell_w=plan["cell_w"],
        cell_h=plan["cell_h"],
        paged=plan["paged"],
    )
    assert all(c.shape[:2] == (plan["cell_h"], plan["cell_w"]) for c in cells)


def test_scalar_plan_is_validated_before_materializing(engine, ome_tiff):
    from ultra_deepagents.imaging import atlas as atlas_mod

    plan = engine.scalar_plan(ome_tiff, channel=0)
    assert atlas_mod.validate_scalar_plan(plan) > 0  # raises if over budget/degenerate
    assert plan["depth"] == 8 and plan["dtype"] == "uint16"


def test_real_ome_tiff_preserves_tczyx_identity_and_signal_defaults(engine, seven_channel_ome_tiff):
    info = engine.viewer_info(seven_channel_ome_tiff)

    assert info["axis_sizes"] == {"T": 1, "C": 7, "Z": 80, "Y": 12, "X": 18}
    assert info["physical_spacing"] == pytest.approx(
        {"x": 0.1083333333333, "y": 0.1083333333333, "z": 0.29}
    )
    assert info["channel_names"] == [
        "CMDRP_1",
        "CMDRP",
        "EGFP_1",
        "EGFP",
        "H3342_1",
        "H3342",
        "Bright_100X",
    ]
    assert info["display_defaults"]["channels"] == [1, 3, 5]
    assert info["viewer"]["volume_mode"] == "slice_stack"
    assert info["viewer"]["available_surfaces"] == ["2d", "metadata", "volume"]


def test_coordinate_tiff_preserves_internal_and_source_order(engine, coordinate_tiff):
    path, _data = coordinate_tiff
    info = engine.viewer_info(path)

    assert info["axis_sizes"] == {"T": 2, "C": 2, "Z": 3, "Y": 4, "X": 6}
    assert info["dims_order"] == "TCZYX"
    assert info["metadata"]["source_dims_order"] == "XYZCT"
    assert info["metadata"]["scene_count"] == 1


def test_native_engine_routes_tiff_semantics_through_tifffile(engine, coordinate_tiff):
    """The native-present path must keep the same authoritative T/C/Z pixels."""
    from ultra_deepagents.imaging.engine import LibBioImageEngine

    path, data = coordinate_tiff
    native = object.__new__(LibBioImageEngine)
    native._semantic_tiff_engine = engine

    volume = native.scalar_volume(path, channel=1, t=1)
    assert volume["data"] == np.asarray(data[1, 1], dtype="<u2").tobytes(order="C")
    assert native.viewer_info(path)["metadata"]["reader"] == "tifffile"
    assert native.slice_plane(path, z=2, t=1, channels=[2]).startswith(_PNG_MAGIC)
    assert native.thumbnail(path, z=1, t=1, channels=[2]).startswith(_PNG_MAGIC)


def test_coordinate_tiff_threads_exact_t_c_z_across_semantic_reads(engine, coordinate_tiff):
    path, data = coordinate_tiff

    volume = engine.scalar_volume(path, channel=1, t=1)
    assert volume["data"] == np.asarray(data[1, 1], dtype="<u2").tobytes(order="C")
    assert (volume["channel"], volume["t"]) == (1, 1)

    slice_png = engine.slice_plane(path, z=2, t=1, channels=[2])
    from PIL import Image

    slice_pixels = np.asarray(Image.open(io.BytesIO(slice_png)).convert("L"))
    assert slice_pixels.shape == (4, 6)
    histogram = engine.histogram(path, bins=4, channels=[2], t=1)
    assert [entry["index"] for entry in histogram["channels"]] == [1]
    assert histogram["channels"][0]["min"] == pytest.approx(float(data[1, 1, 1].min()))
    assert histogram["channels"][0]["max"] == pytest.approx(float(data[1, 1, 1].max()))

    plan = engine.atlas_plan(path, channels=[2], t=1)
    assert plan["t"] == 1 and plan["read_channels"] == [2]
    windows = engine.atlas_windows(
        path,
        depth=plan["depth"],
        level=plan["read_level"],
        channels=plan["read_channels"],
        paged=plan["paged"],
        t=plan["t"],
    )
    cells = engine.atlas_cells(
        path,
        zs=[0, 1, 2],
        level=plan["read_level"],
        channels=plan["read_channels"],
        colors=plan["cell_colors"],
        windows=windows,
        cell_w=plan["cell_w"],
        cell_h=plan["cell_h"],
        paged=plan["paged"],
        t=plan["t"],
    )
    assert [int(cell.mean()) for cell in cells] == sorted(int(cell.mean()) for cell in cells)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        ("slice_plane", {"z": 3, "t": 1, "channels": [2]}),
        ("slice_plane", {"z": 0, "t": 2, "channels": [2]}),
        ("slice_plane", {"z": 0, "t": 1, "channels": [0]}),
        ("thumbnail", {"z": 3, "t": 1, "channels": [2]}),
        ("atlas_plan", {"t": 2, "channels": [2]}),
        ("atlas_plan", {"t": 1, "channels": [3]}),
        ("histogram", {"t": 2, "channels": [2]}),
        ("histogram", {"t": 1, "channels": [2, 2]}),
    ],
)
def test_semantic_reads_reject_invalid_or_duplicate_indices_before_read(
    engine, coordinate_tiff, monkeypatch, operation, kwargs
):
    path, _data = coordinate_tiff
    source = engine._source(path)
    reads = 0
    original_read = source.read

    def counted_read(**read_kwargs):
        nonlocal reads
        reads += 1
        return original_read(**read_kwargs)

    monkeypatch.setattr(source, "read", counted_read)
    with pytest.raises(ValueError, match="out of range|duplicate"):
        getattr(engine, operation)(path, **kwargs)
    assert reads == 0


def test_scalar_volume_is_native_little_endian_and_zero_based(engine, seven_channel_ome_tiff):
    volume = engine.scalar_volume(seven_channel_ome_tiff, channel=3, t=0)
    loaded = tifffile.imread(seven_channel_ome_tiff)
    expected = loaded[0, 3] if loaded.ndim == 5 else loaded[3]

    assert volume["dtype"] == "uint16"
    assert volume["bytes_per_voxel"] == 2
    assert volume["channel"] == 3
    assert volume["t"] == 0
    assert volume["scl_slope"] == 1.0 and volume["scl_inter"] == 0.0
    assert volume["data"] == np.asarray(expected, dtype="<u2", order="C").tobytes(order="C")


@pytest.mark.parametrize(
    ("dtype", "wire_dtype"),
    [("uint8", "uint8"), ("uint16", "uint16"), ("int16", "int16"), ("float32", "float32")],
)
def test_scalar_volume_preserves_supported_native_dtype(engine, tmp_path, dtype, wire_dtype):
    path = str(tmp_path / f"native-{wire_dtype}.ome.tiff")
    values = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

    volume = engine.scalar_volume(path)

    expected_dtype = np.dtype(dtype).newbyteorder("<")
    assert volume["dtype"] == wire_dtype
    assert volume["bytes_per_voxel"] == expected_dtype.itemsize
    assert volume["data"] == np.asarray(values, dtype=expected_dtype, order="C").tobytes(order="C")


@pytest.mark.parametrize(("channel", "time"), [(-1, 0), (7, 0), (0, -1), (0, 1)])
def test_scalar_volume_rejects_invalid_channel_and_time_before_plane_reads(
    engine, seven_channel_ome_tiff, monkeypatch, channel, time
):
    source = engine._source(seven_channel_ome_tiff)
    reads = 0
    original_read = source.read

    def counted_read(**kwargs):
        nonlocal reads
        reads += 1
        return original_read(**kwargs)

    monkeypatch.setattr(source, "read", counted_read)
    with pytest.raises(ValueError, match="out of range"):
        engine.scalar_volume(seven_channel_ome_tiff, channel=channel, t=time)
    assert reads == 0


def test_scalar_preview_plan_is_bounded_spacing_aware_and_preserves_z(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 924, 624, 80, 7, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (0.29, 0.1083333333333, 0.1083333333333)

    monkeypatch.setattr(engine, "_source", lambda _path: SyntheticSource())
    plan = engine.scalar_plan("synthetic.ome.tiff", channel=5, t=0)

    assert (plan["width"], plan["height"], plan["depth"]) == (462, 312, 80)
    assert (plan["downsample_x"], plan["downsample_y"], plan["downsample_z"]) == (2, 2, 1)
    assert (plan["source_width"], plan["source_height"], plan["source_depth"]) == (924, 624, 80)
    assert plan["preview_policy"] == "auto-v1"
    assert plan["width"] * plan["height"] * plan["depth"] <= 16_777_216
    assert plan["width"] * (0.1083333333333 * plan["downsample_x"]) == pytest.approx(
        100.1, rel=1e-3
    )
    assert plan["height"] * (0.1083333333333 * plan["downsample_y"]) == pytest.approx(
        67.6, rel=1e-3
    )
    assert plan["depth"] * (0.29 * plan["downsample_z"]) == pytest.approx(23.2)


def test_ome_physical_size_units_are_honoured(engine, tmp_path):
    # Ignoring the unit reported an nm-declared file 1000x too coarse.
    path = str(tmp_path / "nm.ome.tiff")
    tifffile.imwrite(
        path,
        np.zeros((4, 16, 16), dtype="uint16"),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 110.0,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 110.0,
            "PhysicalSizeYUnit": "nm",
        },
    )
    assert engine.meta(path)["pixel_resolution_x"] == pytest.approx(0.11)


@pytest.mark.parametrize(
    ("unit", "expected_microns"),
    [("pm", 1e-6), ("in", 25_400.0), ("ft", 304_800.0), ("Ym", 1e30)],
)
def test_ome_units_length_domain_converts_physical_units_to_microns(
    engine, tmp_path, unit, expected_microns
):
    path = str(tmp_path / f"unit-{unit}.ome.tiff")
    tifffile.imwrite(
        path,
        np.zeros((2, 4, 4), dtype="uint16"),
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeXUnit": unit,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeYUnit": unit,
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeZUnit": unit,
        },
    )
    meta = engine.meta(path)
    assert meta["pixel_resolution_x"] == pytest.approx(expected_microns)
    assert meta["pixel_resolution_unit_x"] == "um"


def test_nonphysical_and_unknown_ome_units_are_preserved_not_labeled_microns():
    metadata = (
        '<Pixels PhysicalSizeX="2" PhysicalSizeXUnit="pixel" '
        'PhysicalSizeY="3" PhysicalSizeYUnit="furlong"/>'
    )
    spacing, units = _tiff_spacing_with_units(_FakeTiff(metadata), _FakeSeries({}))
    assert spacing == (1.0, 3.0, 2.0)
    assert units == ("voxel", "furlong", "pixel")


def test_exact_one_micron_ome_spacing_beats_conflicting_resolution_tags(engine, tmp_path):
    path = str(tmp_path / "one-micron.ome.tiff")
    tifffile.imwrite(
        path,
        np.zeros((2, 8, 8), dtype="uint16"),
        ome=True,
        resolution=(100, 100),
        resolutionunit="inch",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeZ": 1.0,
        },
    )
    meta = engine.meta(path)
    assert (meta["pixel_resolution_x"], meta["pixel_resolution_y"], meta["pixel_resolution_z"]) == (
        1.0,
        1.0,
        1.0,
    )


class _FakeTag:
    def __init__(self, value):
        self.value = value


class _FakeSeries:
    def __init__(self, tags):
        self.pages = [type("Page", (), {"tags": tags})()]


class _FakeTiff:
    def __init__(self, metadata=None):
        self.ome_metadata = metadata


@pytest.mark.parametrize(
    "tags",
    [
        {"XResolution": _FakeTag((100, 1)), "YResolution": _FakeTag((100, 1))},
        {
            "XResolution": _FakeTag((100, 1)),
            "YResolution": _FakeTag((100, 1)),
            "ResolutionUnit": _FakeTag(1),
        },
        {
            "XResolution": _FakeTag((float("nan"), 1)),
            "YResolution": _FakeTag((100, 0)),
            "ResolutionUnit": _FakeTag(3),
        },
    ],
)
def test_missing_unit_unitless_and_invalid_tiff_resolution_are_not_physical_spacing(tags):
    assert _tiff_spacing(_FakeTiff(), _FakeSeries(tags)) == (1.0, 1.0, 1.0)


def test_multiple_tiff_series_report_scenes_and_withhold_volume(engine, tmp_path):
    path = str(tmp_path / "two-series.tif")
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((3, 8, 8), dtype="uint16"), metadata={"axes": "ZYX"})
        writer.write(np.zeros((4, 20, 20), dtype="uint16"), metadata={"axes": "ZYX"})

    info = engine.viewer_info(path)
    assert info["metadata"]["scene_count"] == 2
    assert info["axis_sizes"] == {"T": 1, "C": 1, "Z": 3, "Y": 8, "X": 8}
    assert info["kind"] == "unsupported" and info["decodable"] is False
    assert info["viewer"]["available_surfaces"] == ["metadata"]
    assert "slice_axes" not in info["viewer"]
    for operation, kwargs in [
        ("tile", {"level": 0, "col": 0, "row": 0}),
        ("slice_plane", {"z": 0}),
        ("thumbnail", {}),
        ("histogram", {}),
        ("scalar_plan", {}),
        ("atlas_plan", {}),
    ]:
        with pytest.raises(ValueError, match="multiple.*scene|scene.*explicit"):
            getattr(engine, operation)(path, **kwargs)


def test_scalar_preview_reads_bounded_regions_and_emits_exact_box_bytes(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 924, 624, 80, 3, 2
        dtype = np.dtype(">u2")
        spacing_zyx = (0.29, 0.1083333333333, 0.1083333333333)
        scene_count = 1
        source_order = "XYZCT"
        max_read_bytes = 0
        total_read_bytes = 0

        def read(self, *, t, c, z, level, box=None):
            assert box is not None, "scalar preview must never materialize a native plane"
            y0, y1, x0, x1 = box
            read_bytes = (y1 - y0) * (x1 - x0) * 2
            self.max_read_bytes = max(self.max_read_bytes, read_bytes)
            self.total_read_bytes += read_bytes
            yy = np.arange(y0, y1, dtype="uint32")[:, None]
            xx = np.arange(x0, x1, dtype="uint32")[None, :]
            return (t * 10_000 + c * 2_000 + z * 20 + yy * 2 + xx).astype(">u2")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    planes = engine.scalar_planes(
        "synthetic.ome.tiff",
        zs=range(80),
        channel=2,
        t=1,
        pages=0,
    )
    assert len(planes) == 80
    assert planes[0].shape == (312, 462)
    assert planes[40].shape == (312, 462)
    assert planes[79].shape == (312, 462)
    assert planes[0].dtype == np.dtype("<u2")
    expected_first = round(np.mean([14_000, 14_001, 14_002, 14_003]))
    assert int(planes[0][0, 0]) == expected_first
    assert int(planes[40][0, 0]) == expected_first + 40 * 20
    assert int(planes[79][0, 0]) == expected_first + 79 * 20
    assert source.max_read_bytes <= 4 * 1024 * 1024
    assert source.total_read_bytes == 924 * 624 * 80 * 2

    source.total_read_bytes = 0
    volume = engine.scalar_volume("synthetic.ome.tiff", channel=2, t=1)
    assert (volume["width"], volume["height"], volume["depth"]) == (462, 312, 80)
    assert (volume["source_width"], volume["source_height"], volume["source_depth"]) == (
        924,
        624,
        80,
    )
    assert (volume["downsample_x"], volume["downsample_y"], volume["downsample_z"]) == (
        2,
        2,
        1,
    )
    assert volume["preview_policy"] == "auto-v1"
    assert volume["channel"] == 2 and volume["t"] == 1
    assert volume["data"] == b"".join(plane.tobytes(order="C") for plane in planes)
    assert source.total_read_bytes == 924 * 624 * 80 * 2


def test_scalar_preview_chunks_source_planes_larger_than_four_mib(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 4096, 2048, 2, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 4096 * 2
        read_sizes: list[int] = []

        def read(self, *, t, c, z, level, box=None):
            assert box is not None
            y0, y1, x0, x1 = box
            read_bytes = (y1 - y0) * (x1 - x0) * 2
            self.read_sizes.append(read_bytes)
            return np.full((y1 - y0, x1 - x0), z + 1, dtype="uint16")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    plan = engine.scalar_plan("wide.ome.tiff")
    planes = engine.scalar_planes("wide.ome.tiff", zs=range(plan["depth"]), channel=0, t=0, pages=0)

    assert max(source.read_sizes) <= 4 * 1024 * 1024
    assert len(source.read_sizes) > source.z
    assert sum(source.read_sizes) == source.x * source.y * source.z * 2
    assert [int(plane[0, 0]) for plane in planes] == [1, 2]


def test_scalar_volume_rejects_oversize_decoder_chunk_and_total_source_work_before_read(
    engine, monkeypatch
):
    class SyntheticSource:
        x, y, z, c, t = 8192, 8192, 5, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 1

        def read(self, **_kwargs):
            raise AssertionError("source-work rejection must happen before read")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    with pytest.raises(ValueError, match="source work"):
        engine.scalar_plan("too-much-work.ome.tiff")

    source.x, source.y, source.z = 8, 8, 8
    source.max_decoded_chunk_bytes = 64 * 1024 * 1024 + 1
    with pytest.raises(ValueError, match="decoded chunk"):
        engine.scalar_plan("oversize-chunk.ome.tiff")


def test_rowsperstrip_one_tiff_with_ninety_two_mib_channel_is_plan_eligible(engine, tmp_path):
    path = str(tmp_path / "bounded-work.tif")
    shape = (80, 624, 924)
    tifffile.imwrite(
        path,
        np.zeros(shape, dtype="uint16"),
        metadata={"axes": "ZYX"},
        rowsperstrip=1,
    )

    source = engine._source(path)
    assert source.max_decoded_chunk_bytes == 924 * 2
    plan = engine.scalar_plan(path)
    assert (plan["source_width"], plan["source_height"], plan["source_depth"]) == (
        924,
        624,
        80,
    )


def test_tiff_resolution_unit_inch_is_converted(engine, tmp_path):
    path = str(tmp_path / "inch.tif")
    tifffile.imwrite(
        path, np.zeros((16, 16), dtype="uint8"), resolution=(100, 100), resolutionunit="inch"
    )
    # 1/100 inch per px = 254 micron, not 100 (the old hardcoded-cm assumption).
    assert engine.meta(path)["pixel_resolution_x"] == pytest.approx(254.0, rel=0.01)


def test_whole_slide_tiff_extensions_are_accepted(tmp_path):
    from ultra_deepagents.imaging.bioio_engine import can_read

    for name in ("slide.svs", "scan.ndpi", "multiplex.qptiff", "leica.scn"):
        assert can_read(str(tmp_path / name)), name


def test_region_scale_applies_to_the_long_edge(engine, ome_tiff):
    # Deriving the bound from the ROI width shrank tall ROIs by their aspect ratio.
    png = engine.region(ome_tiff, x1=0, y1=0, x2=40, y2=200, region_scale=0.5)
    assert _size(png) == (20, 100)


def test_meta_is_libbioimage_shaped(engine, ome_tiff):
    # The shared viewerinfo builder consumes this exact shape; getting it right is
    # what keeps the viewer contract identical to the native engine.
    meta = engine.meta(ome_tiff)
    assert meta["image_num_x"] == 320 and meta["image_num_y"] == 256
    assert meta["image_num_z"] == 8 and meta["image_num_c"] == 3 and meta["image_num_t"] == 1
    assert meta["image_pixel_depth"] == 16
    assert meta["image_pixel_format"] == "unsigned integer"
    assert meta["pixel_resolution_x"] == pytest.approx(0.11)
    assert meta["pixel_resolution_z"] == pytest.approx(0.29)
    assert meta["channels/channel:0/name"] == "DAPI"


def test_viewer_info_matches_real_geometry(engine, ome_tiff):
    vi = engine.viewer_info(ome_tiff)
    assert vi["axis_sizes"] == {"T": 1, "C": 3, "Z": 8, "Y": 256, "X": 320}
    assert vi["channel_names"] == ["DAPI", "EGFP", "CMDRP"]
    assert vi["modality"] == "microscopy"


def test_thumbnail_is_real_pixels_bounded_to_max_size(engine, ome_tiff):
    png = engine.thumbnail(ome_tiff, max_size=64)
    assert png.startswith(_PNG_MAGIC)
    width, height = _size(png)
    assert max(width, height) <= 64
    # Aspect ratio of the true plane (320x256), NOT a square placeholder.
    assert (width, height) == (64, 51)


def test_slice_frames_differ_by_z(engine, ome_tiff):
    a = engine.slice_plane(ome_tiff, z=1)
    b = engine.slice_plane(ome_tiff, z=5)
    assert _size(a) == (320, 256)
    assert a != b


def test_scrub_frame_is_bounded_but_settled_is_native(engine, ome_tiff):
    scrub = engine.slice_plane(ome_tiff, z=2, full_resolution=False)
    settled = engine.slice_plane(ome_tiff, z=2, full_resolution=True)
    assert max(_size(scrub)) <= 1024
    assert _size(settled) == (320, 256)


def test_rgb_tiff_takes_the_photo_path(engine, rgb_tiff):
    vi = engine.viewer_info(rgb_tiff)
    # red/green/blue naming is what drives viewerinfo's full-colour photo path.
    assert vi["channel_names"] == ["red", "green", "blue"]
    assert vi["axis_sizes"]["C"] == 3
    from PIL import Image

    assert Image.open(io.BytesIO(engine.thumbnail(rgb_tiff, max_size=64))).mode == "RGB"


def test_tile_region_atlas_and_volume(engine, ome_tiff):
    assert _size(engine.tile(ome_tiff, level=0, col=0, row=0, tile_size=128)) == (128, 128)
    assert _size(engine.region(ome_tiff, x1=10, y1=10, x2=140, y2=90)) == (130, 80)
    assert engine.atlas(ome_tiff).startswith(_PNG_MAGIC)
    vol = engine.scalar_volume(ome_tiff, channel=0)
    assert (vol["width"], vol["height"], vol["depth"]) == (320, 256, 8)
    assert len(vol["data"]) == vol["width"] * vol["height"] * vol["depth"] * 2


def test_histogram_per_channel(engine, ome_tiff):
    hist = engine.histogram(ome_tiff, bins=8)
    assert hist["bins"] == 8
    assert len(hist["channels"]) == 3
    assert sum(hist["channels"][0]["counts"]) > 0


def test_tile_past_the_grid_edge_is_decode_class(engine, ome_tiff):
    with pytest.raises(ValueError) as excinfo:
        engine.tile(ome_tiff, level=0, col=99, row=99, tile_size=128)
    assert any(m in str(excinfo.value).lower() for m in _DECODE_MARKERS)


@pytest.mark.parametrize("op", ["thumbnail", "slice_plane", "viewer_info", "histogram"])
def test_corrupt_file_maps_to_the_422_decode_class(engine, tmp_path, op):
    # A raw tifffile ValueError ("not a TIFF file") carries no marker, so it would
    # surface as a 500 and be retried on every grid render — must be wrapped.
    path = str(tmp_path / "broken.tif")
    with open(path, "wb") as fh:
        fh.write(b"not a tiff at all")
    with pytest.raises(ValueError) as excinfo:
        getattr(engine, op)(path)
    assert any(m in str(excinfo.value).lower() for m in _DECODE_MARKERS)


def test_unsupported_extension_is_decode_class(engine, tmp_path):
    path = str(tmp_path / "scan.lsm")
    tifffile.imwrite(path, np.zeros((8, 8), dtype="uint8"))
    with pytest.raises(ValueError) as excinfo:
        engine.viewer_info(path)
    assert "unsupported" in str(excinfo.value).lower()


def test_hdf5_still_forks_to_the_h5py_path(engine, tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "vol.h5")
    with h5py.File(path, "w") as f:
        grp = f.create_group("volume")
        grp.create_dataset(
            "ct", data=(np.random.default_rng(1).random((10, 20, 25)) * 4000).astype("int16")
        )
    info = engine.viewer_info(path)
    assert info["kind"] == "hdf5"
    assert _size(engine.thumbnail(path, max_size=512)) == (25, 20)


def test_build_engine_prefers_bioio_over_stub_when_native_is_missing(monkeypatch):
    # The load-bearing tier order: without the native wheel a user-facing service
    # must get REAL decoding, never the placeholder stub.
    from ultra_deepagents.imaging import engine as engine_mod

    def _unavailable(*_args, **_kwargs):
        raise engine_mod.EngineUnavailable("no native wheel in this environment")

    monkeypatch.setattr(engine_mod, "LibBioImageEngine", _unavailable)
    built = engine_mod.build_engine(prefer_real=True)
    assert isinstance(built, BioioEngine)


def test_build_engine_falls_to_hdf5_only_when_bioio_is_missing(monkeypatch):
    from ultra_deepagents.imaging import engine as engine_mod

    def _unavailable(*_args, **_kwargs):
        raise engine_mod.EngineUnavailable("no native wheel")

    monkeypatch.setattr(engine_mod, "LibBioImageEngine", _unavailable)
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _blocked_import(name, *args, **kwargs):
        if name.endswith("bioio_engine"):
            raise ImportError("bioio not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    built = engine_mod.build_engine(prefer_real=True)
    assert type(built).__name__ == "Hdf5OnlyEngine"


def test_stub_is_only_reachable_by_explicit_opt_out():
    from ultra_deepagents.imaging import engine as engine_mod

    assert type(engine_mod.build_engine(prefer_real=False)).__name__ == "StubEngine"
