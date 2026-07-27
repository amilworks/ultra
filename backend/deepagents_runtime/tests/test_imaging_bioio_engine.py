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

from ultra_deepagents.imaging import bioio_engine as bioio_engine_module  # noqa: E402
from ultra_deepagents.imaging.bioio_engine import (  # noqa: E402
    BioioEngine,
    _BioioPlane,
    _decoded_selection_work_bytes,
    _max_chunk_bytes,
    _tiff_spacing,
    _tiff_spacing_with_units,
    _TiffPlane,
)
from ultra_deepagents.imaging.scalar_semantics import (  # noqa: E402
    _histogram,
    canonical_mask_threshold,
    profile_z_indices,
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
    with pytest.raises(ValueError, match="decoded chunk geometry is invalid"):
        _max_chunk_bytes((True,), np.dtype("uint8"))
    with pytest.raises(ValueError, match="decoded chunk geometry is invalid"):
        _max_chunk_bytes((1,), np.dtype("V0"))


def test_intersected_chunk_accounting_includes_edges_and_every_touched_chunk():
    # A crop spanning the tail of the first Y chunk and both X chunks charges
    # both complete decoded chunks, including the short edge chunk.
    assert (
        _decoded_selection_work_bytes(
            (1, 1, 1, 7, 9),
            ((1,), (1,), (1,), (4, 3), (5, 4)),
            np.dtype("uint16"),
            (0, 0, 0, slice(3, 6), slice(4, 8)),
        )
        == 7 * 9 * 2
    )


def _naive_decoded_selection_work(shape, chunks, itemsize, selection):
    touched_axes = []
    for size, raw_chunks, selector in zip(shape, chunks, selection, strict=True):
        if isinstance(selector, int):
            start, stop = selector, selector + 1
        else:
            start, stop, step = selector.indices(size)
            assert step == 1 and start < stop
        if not isinstance(raw_chunks, (tuple, list)):
            first_chunk = start // raw_chunks
            last_chunk = (stop - 1) // raw_chunks
            touched_axes.append((last_chunk - first_chunk + 1) * raw_chunks)
            continue
        offset = 0
        touched = 0
        for chunk_size in raw_chunks:
            chunk_end = offset + chunk_size
            if offset < stop and chunk_end > start:
                touched += chunk_size
            offset = chunk_end
        touched_axes.append(touched)
    return int(np.prod(touched_axes)) * itemsize


@pytest.mark.parametrize(
    ("shape", "chunks", "selection"),
    [
        ((10,), (4,), (slice(3, 9),)),
        ((10,), ((4, 4, 2),), (slice(4, 8),)),
        ((11,), ((3, 1, 5, 2),), (slice(2, 10),)),
        ((10,), (4,), (9,)),
        ((10,), ((4, 4, 2),), (9,)),
        (
            (2, 3, 4, 7, 9),
            ((1, 1), (2, 1), (3, 1), (4, 3), (5, 4)),
            (1, 2, 3, slice(3, 6), slice(4, 8)),
        ),
    ],
)
def test_prepared_decoded_chunk_geometry_matches_naive_oracle(shape, chunks, selection):
    prepared = bioio_engine_module._prepare_decoded_chunk_geometry(
        shape,
        chunks,
        np.dtype("uint16"),
    )

    expected = _naive_decoded_selection_work(shape, chunks, 2, selection)
    assert prepared.estimate(selection) == expected
    assert _decoded_selection_work_bytes(shape, chunks, np.dtype("uint16"), selection) == expected


def test_uniform_chunks_charge_nominal_tail_and_oversized_buffers():
    nominal_tail = bioio_engine_module._prepare_decoded_chunk_geometry(
        (10,),
        (4,),
        np.dtype("uint16"),
    )
    oversized = bioio_engine_module._prepare_decoded_chunk_geometry(
        (2,),
        (4,),
        np.dtype("uint16"),
    )
    explicit_tail = bioio_engine_module._prepare_decoded_chunk_geometry(
        (10,),
        ((4, 4, 2),),
        np.dtype("uint16"),
    )

    assert nominal_tail.estimate((9,)) == 8
    assert nominal_tail.max_chunk_bytes == 8
    assert oversized.estimate((1,)) == 8
    assert oversized.max_chunk_bytes == 8
    assert explicit_tail.estimate((9,)) == 4
    assert explicit_tail.max_chunk_bytes == 8


def test_prepared_decoded_chunk_geometry_snapshots_inputs_and_keeps_uniform_axes_arithmetic():
    shape = [1_000_000_000]
    chunks = [1]
    prepared = bioio_engine_module._prepare_decoded_chunk_geometry(
        shape,
        chunks,
        np.dtype("uint8"),
    )
    shape[0] = 1
    chunks[0] = True

    assert prepared.estimate((slice(999_999_998, 1_000_000_000),)) == 2
    assert prepared.axes[0].boundaries is None


@pytest.mark.parametrize(
    ("shape", "chunks", "dtype"),
    [
        ((True,), (1,), np.dtype("uint8")),
        ((4,), (True,), np.dtype("uint8")),
        ((4,), ((2, True, 1),), np.dtype("uint8")),
        ((4,), ((2, 1),), np.dtype("uint8")),
        ((4, 4), (2,), np.dtype("uint8")),
        ((4,), (2,), np.dtype("V0")),
    ],
)
def test_prepared_decoded_chunk_geometry_rejects_invalid_descriptors(shape, chunks, dtype):
    with pytest.raises(ValueError, match="decoded chunk geometry is invalid"):
        bioio_engine_module._prepare_decoded_chunk_geometry(shape, chunks, dtype)


class _FakeChunkedArray:
    def __init__(self, chunks):
        self.shape = (1, 1, 1, 8192, 2)
        self.chunks = chunks
        self.dtype = np.dtype("uint16")
        self.reads = 0

    def __getitem__(self, _selection):
        self.reads += 1
        raise AssertionError("geometry admission must not read pixels")


def _synthetic_tiff_plane(array):
    source = object.__new__(_TiffPlane)
    source.t = source.c = source.z = 1
    source.y, source.x = 8192, 2
    source.dtype = array.dtype
    source.level_shapes = [(8192, 2)]
    source._axes = "TCZYX"
    source._samples_as_channels = False
    source._decoded_chunk_geometry_cache = {}
    source._level_array = lambda _level: array
    return source


def _synthetic_bioio_plane(array):
    source = object.__new__(_BioioPlane)
    source.t = source.c = source.z = 1
    source.y, source.x = 8192, 2
    source.level_shapes = [(8192, 2)]
    source._dask = array
    source._decoded_chunk_geometry_cache = None
    return source


@pytest.mark.parametrize("factory", [_synthetic_tiff_plane, _synthetic_bioio_plane])
@pytest.mark.parametrize(
    ("chunks", "expected_work"),
    [
        ((1, 1, 1, 1024, 2), 4096),
        (((1,), (1,), (1,), (1,) * 8192, (2,)), 4),
    ],
)
def test_plane_adapters_normalize_geometry_once(
    factory,
    chunks,
    expected_work,
    monkeypatch,
):
    array = _FakeChunkedArray(chunks)
    source = factory(array)
    prepare_calls = 0
    original_prepare = bioio_engine_module._prepare_decoded_chunk_geometry

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        bioio_engine_module,
        "_prepare_decoded_chunk_geometry",
        counted_prepare,
    )
    for y in (0, 1024, 4096, 8191):
        assert (
            source.estimate_read_work(t=0, c=0, z=0, level=0, box=(y, y + 1, 0, 1)) == expected_work
        )

    assert prepare_calls == 1
    assert array.reads == 0


@pytest.mark.parametrize("factory", [_synthetic_tiff_plane, _synthetic_bioio_plane])
def test_plane_adapters_cache_invalid_geometry_failure_without_reads(factory, monkeypatch):
    array = _FakeChunkedArray(((1,), (1,), (1,), (1,) * 8191 + (True,), (2,)))
    source = factory(array)
    prepare_calls = 0
    original_prepare = bioio_engine_module._prepare_decoded_chunk_geometry

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        bioio_engine_module,
        "_prepare_decoded_chunk_geometry",
        counted_prepare,
    )
    for _attempt in range(2):
        with pytest.raises(ValueError, match="decoded chunk geometry is invalid"):
            source.estimate_read_work(t=0, c=0, z=0, level=0, box=(0, 1, 0, 1))

    assert prepare_calls == 1
    assert array.reads == 0


def test_scalar_profile_z_plan_includes_ends_and_exact_center():
    planned = profile_z_indices(80)
    assert {0, 40, 79}.issubset(planned)
    assert len(planned) <= 32


@pytest.mark.parametrize("dtype", ["uint16", "int16", "float32"])
def test_non_uint8_otsu_threshold_strict_above_matches_bins(dtype):
    raw_values = (
        [0, 1, 7, 8, 31, 32, 63, 64, 127, 255]
        if dtype == "uint16"
        else [-64, -33, -32, -1, 0, 1, 31, 32, 63, 64]
    )
    values = np.asarray(raw_values, dtype=dtype)
    counts, edges, threshold = _histogram(values, values.dtype, 4)
    # Recompute membership from NumPy's bin partition: bins above the selected
    # Otsu bin must be exactly the raw values accepted by sample > threshold.
    from ultra_deepagents.imaging.scalar_semantics import imagej_otsu_first_max

    otsu_bin = imagej_otsu_first_max(counts)
    assigned = np.searchsorted(edges, values.astype("float64"), side="right") - 1
    assigned = np.minimum(assigned, len(counts) - 1)
    assert np.array_equal(values > threshold, assigned > otsu_bin)


@pytest.mark.parametrize(
    ("dtype", "threshold", "expected"),
    [
        ("uint8", 120.9, 120),
        ("uint8", -99, -1),
        ("uint8", 999, 255),
        ("uint16", 65_534.9, 65_534),
        ("uint16", -99, -1),
        ("uint16", 99_999, 65_535),
        ("int16", -32_768.1, -32_769),
        ("int16", 12.9, 12),
        ("int16", 99_999, 32_767),
    ],
)
def test_integer_mask_thresholds_are_membership_canonical(dtype, threshold, expected):
    canonical = canonical_mask_threshold(threshold, np.dtype(dtype))

    assert canonical == expected
    assert float(np.float32(canonical)) == canonical
    samples = np.asarray(
        [np.iinfo(dtype).min, max(np.iinfo(dtype).min, expected), np.iinfo(dtype).max],
        dtype=dtype,
    )
    assert np.array_equal(samples > threshold, samples > canonical)


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
    assert info["viewer"]["available_surfaces"] == ["2d", "metadata", "mpr", "volume"]
    assert info["scalar_mask_capability"]["dtype"] == "uint16"
    assert info["scalar_mask_capability"]["surfaces"] == ["2d", "mpr", "volume"]


def test_coordinate_tiff_preserves_internal_and_source_order(engine, coordinate_tiff):
    path, _data = coordinate_tiff
    info = engine.viewer_info(path)

    assert info["axis_sizes"] == {"T": 2, "C": 2, "Z": 3, "Y": 4, "X": 6}
    assert info["dims_order"] == "TCZYX"
    assert info["metadata"]["source_dims_order"] == "XYZCT"
    assert info["metadata"]["scene_count"] == 1
    assert info["scalar_mask_capability"]["channel_selection"] == "single"


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
    mask_threshold = 1370
    mask_png = engine.slice_plane(
        path,
        z=1,
        t=1,
        channels=[2],
        scalar_render_mode="mask",
        scalar_threshold_value=mask_threshold,
        scalar_threshold_foreground="above",
    )
    mask_pixels = np.asarray(Image.open(io.BytesIO(mask_png)).convert("L"))
    expected_mask = np.where(data[1, 1, 1] > mask_threshold, 255, 0).astype("uint8")
    np.testing.assert_array_equal(mask_pixels, expected_mask)
    histogram = engine.histogram(path, bins=4, channels=[2], t=1)
    assert [entry["index"] for entry in histogram["channels"]] == [1]
    assert histogram["channels"][0]["min"] == pytest.approx(float(data[1, 1].min()))
    assert histogram["channels"][0]["max"] == pytest.approx(float(data[1, 1].max()))
    assert histogram["t"] == 1
    assert histogram["scope"] == "volume"
    assert len(histogram["channels"][0]["edges"]) == 5
    display_histogram = engine.histogram(path, bins=4, channels=[1, 2], t=1, scope="display")
    assert display_histogram["scope"] == "display"
    assert [entry["index"] for entry in display_histogram["channels"]] == [0, 1]
    assert display_histogram["channels"][0]["edges"] == display_histogram["channels"][1]["edges"]
    assert "threshold" not in display_histogram

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
    info = engine.viewer_info(path)
    if dtype == "float32":
        assert "scalar_mask_capability" not in info
    else:
        assert info["scalar_mask_capability"]["dtype"] == wire_dtype


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


@pytest.mark.parametrize(("depth", "dtype"), [(65, "uint8"), (70, "uint16")])
def test_nearest_integer_mask_plan_is_native_and_exact(engine, monkeypatch, depth, dtype):
    class SyntheticSource:
        x, y, z, c, t = 924, 624, depth, 1, 1
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 924 * np.dtype(dtype).itemsize
        is_photo = False

        def __init__(self):
            self.dtype = np.dtype(dtype)

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            return (box[1] - box[0]) * (box[3] - box[2]) * self.dtype.itemsize

    monkeypatch.setattr(engine, "_source", lambda _path: SyntheticSource())
    monkeypatch.setattr(
        bioio_engine_module,
        "_source_generation",
        lambda _path: (1, 1, 2, 3, 4, 5),
    )

    plan = engine.scalar_plan("mask.tif", sampling="nearest")

    assert (plan["width"], plan["height"], plan["depth"]) == (924, 624, depth)
    assert (plan["downsample_x"], plan["downsample_y"], plan["downsample_z"]) == (1, 1, 1)
    assert plan["preview_policy"] == "mask-native-integer-v1"


def test_nearest_integer_mask_plan_rejects_over_traversal_before_read(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 1_000, 1_000, 100, 1, 1
        dtype = np.dtype("uint8")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 1_000
        is_photo = False
        reads = 0

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            return (box[1] - box[0]) * (box[3] - box[2])

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("over-budget exact Mask must reject before reading")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    with pytest.raises(ValueError, match="DDA crossing"):
        engine.scalar_volume("mask-too-wide.tif", sampling="nearest")
    assert source.reads == 0


def test_nearest_integer_mask_plan_sums_complete_decoder_work_before_fanout(
    engine, monkeypatch
):
    class SyntheticSource:
        x, y, z, c, t = 8, 8, 16, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 64 * 1024 * 1024
        is_photo = False
        reads = 0

        def estimate_read_work(self, **_kwargs):
            return 64 * 1024 * 1024

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError(
                "aggregate exact Mask work must reject before worker reads"
            )

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    generation = (1, 1, 2, 3, 4, 5)
    monkeypatch.setattr(
        bioio_engine_module,
        "_source_generation",
        lambda _path: generation,
    )

    with pytest.raises(ValueError, match="decode work"):
        engine.scalar_plan(
            "aggregate-mask.ome.tiff",
            channel=0,
            t=0,
            sampling="nearest",
        )
    assert source.reads == 0


def test_float_nearest_is_rejected_before_estimate_or_read(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 8, 8, 4, 1, 1
        dtype = np.dtype("float32")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 256
        estimates = 0
        reads = 0

        def estimate_read_work(self, **_kwargs):
            self.estimates += 1
            return 256

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("non-exact nearest must reject before reading")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    with pytest.raises(ValueError, match="nearest.*exact Mask"):
        engine.scalar_plan("float.tif", sampling="nearest")
    assert source.estimates == 0
    assert source.reads == 0


def test_exact_mask_worker_recomputes_complete_admission_and_rejects_forged_totals(
    engine, tmp_path, monkeypatch
):
    path = str(tmp_path / "forged-mask.tif")
    tifffile.imwrite(path, np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5))
    plan = engine.scalar_plan(path, sampling="nearest")
    source = engine._source(path)
    reads = 0

    def forbidden_read(**_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("forged admission must reject before reading")

    monkeypatch.setattr(source, "read", forbidden_read)
    forged = {
        **plan,
        "admitted_decode_work_bytes": int(plan["admitted_decode_work_bytes"]) + 1,
    }
    with pytest.raises(ValueError, match="decode work.*match|admission.*match"):
        engine.scalar_planes(
            path,
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
            plan=forged,
        )
    assert reads == 0

    forged = {
        **plan,
        "admitted_decode_read_count": int(plan["admitted_decode_read_count"]) + 1,
    }
    with pytest.raises(ValueError, match="read count.*match|admission.*match"):
        engine.scalar_planes(
            path,
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
            plan=forged,
        )
    assert reads == 0


def test_exact_mask_plan_rejects_replaced_source_and_dtype_before_read(
    engine, tmp_path, monkeypatch
):
    path = tmp_path / "replace-mask.tif"
    replacement = tmp_path / "replacement.tif"
    tifffile.imwrite(path, np.zeros((2, 4, 5), dtype="uint8"))
    plan = engine.scalar_plan(str(path), sampling="nearest")
    tifffile.imwrite(replacement, np.zeros((2, 4, 5), dtype="uint16"))
    os.replace(replacement, path)
    replacement_source = engine._source(str(path))
    reads = 0

    def forbidden_read(**_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("replaced source must reject before reading")

    monkeypatch.setattr(replacement_source, "read", forbidden_read)
    with pytest.raises(ValueError, match="generation|dtype|byte width"):
        engine.scalar_planes(
            str(path),
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
            plan=plan,
        )
    assert reads == 0


def test_exact_mask_worker_never_reads_a_source_older_than_its_generated_plan(
    engine, tmp_path, monkeypatch
):
    path = tmp_path / "racing-mask.tif"
    replacement = tmp_path / "racing-mask-replacement.tif"
    write_options = {
        "photometric": "minisblack",
        "metadata": {"axes": "ZYX"},
    }
    tifffile.imwrite(path, np.zeros((2, 4, 5), dtype="uint8"), **write_options)
    old_source = engine._source(str(path))

    tifffile.imwrite(
        replacement,
        np.full((2, 4, 5), 7, dtype="uint8"),
        **write_options,
    )
    os.replace(replacement, path)
    new_source = engine._source(str(path))
    new_plan = engine.scalar_plan(str(path), sampling="nearest")
    old_reads = 0
    original_old_read = old_source.read

    def counted_old_read(**kwargs):
        nonlocal old_reads
        old_reads += 1
        return original_old_read(**kwargs)

    monkeypatch.setattr(old_source, "read", counted_old_read)
    sources = iter((old_source, new_source))
    monkeypatch.setattr(engine, "_source", lambda _path: next(sources))

    with pytest.raises(ValueError, match="generation.*changed|selected source"):
        engine.scalar_planes(
            str(path),
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
        )
    assert old_reads == 0

    monkeypatch.setattr(engine, "_source", lambda _path: old_source)
    with pytest.raises(ValueError, match="generation.*match|selected source"):
        engine.scalar_planes(
            str(path),
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
            plan=new_plan,
        )
    assert old_reads == 0


def test_exact_mask_worker_detects_source_mutation_after_reads(
    engine, tmp_path, monkeypatch
):
    path = str(tmp_path / "mutating-mask.tif")
    tifffile.imwrite(path, np.arange(2 * 4 * 5, dtype="uint8").reshape(2, 4, 5))
    plan = engine.scalar_plan(path, sampling="nearest")
    source = engine._source(path)
    original_read = source.read
    reads = 0

    def mutating_read(**kwargs):
        nonlocal reads
        result = original_read(**kwargs)
        reads += 1
        if reads == 1:
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(source, "read", mutating_read)
    with pytest.raises(ValueError, match="generation.*changed|source.*changed"):
        engine.scalar_planes(
            path,
            zs=[0, 1],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
            plan=plan,
        )


def test_exact_mask_admission_estimates_only_selected_time_channel_across_all_z(
    engine, tmp_path, monkeypatch
):
    path = str(tmp_path / "selected-ct-mask.ome.tif")
    tifffile.imwrite(
        path,
        np.zeros((2, 1, 3, 4, 5), dtype="uint8"),
        ome=True,
        metadata={"axes": "TCZYX"},
    )
    source = engine._source(path)
    estimates: list[tuple[int, int, int]] = []
    original_estimate = source.estimate_read_work

    def recorded_estimate(*, t, c, z, **kwargs):
        estimates.append((t, c, z))
        return original_estimate(t=t, c=c, z=z, **kwargs)

    monkeypatch.setattr(source, "estimate_read_work", recorded_estimate)
    plan = engine.scalar_plan(path, channel=0, t=1, sampling="nearest")

    assert plan["admitted_decode_read_count"] == len(estimates)
    assert {entry[:2] for entry in estimates} == {(1, 0)}
    assert {entry[2] for entry in estimates} == {0, 1, 2}


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_box_scalar_plan_and_bytes_remain_unstamped_and_exact(engine, tmp_path, dtype):
    path = str(tmp_path / f"box-{dtype}.tif")
    values = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    tifffile.imwrite(
        path,
        values,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )

    plan = engine.scalar_plan(path, sampling="box")
    volume = engine.scalar_volume(path, sampling="box")

    assert not any(
        key.startswith("admitted_") or key in {"decode_admission", "source_generation"}
        for key in plan
    )
    expected_dtype = np.dtype("<f4") if dtype == "float32" else np.dtype("u1")
    assert volume["data"] == values.astype(expected_dtype).tobytes(order="C")


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
        source_generation = (1, 1, 2, 3, 4, 5)
        max_read_bytes = 0
        total_read_bytes = 0

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            y0, y1, x0, x1 = box
            return (y1 - y0) * (x1 - x0) * 2

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
    monkeypatch.setattr(
        bioio_engine_module,
        "_source_generation",
        lambda _path: (1, 1, 2, 3, 4, 5),
    )

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

    nearest = engine.scalar_planes(
        "synthetic.ome.tiff",
        zs=[0],
        channel=2,
        t=1,
        pages=0,
        sampling="nearest",
    )
    assert nearest[0].shape == (624, 924)
    assert int(nearest[0][0, 0]) == 14_000

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


def test_binary_mask_semantics_and_raw_threshold_slice_agree(engine, tmp_path):
    path = str(tmp_path / "binary-mask.tif")
    data = np.zeros((5, 24, 32), dtype="uint8")
    data[:, 6:18, 9:23] = 255
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"}, photometric="minisblack")

    info = engine.viewer_info(path)
    assert info["scalar_mask_capability"] == {
        "version": 1,
        "source_authority": "original",
        "source_format": "tiff",
        "dtype": "uint8",
        "threshold_domain": "raw",
        "threshold_foreground": "above",
        "slice_delivery": "thresholded_png",
        "volume_delivery": "raw_scalar",
        "volume_sampling": "nearest",
        "channel_selection": "single",
        "time_selection": "single",
        "surfaces": ["2d", "mpr", "volume"],
    }
    assert info["data_semantics"]["kind"] == "binary_mask"
    assert info["data_semantics"]["strength"] == "exact"
    assert info["data_semantics"]["recommended_view"] == "mask"
    assert info["display_defaults"]["scalar_render_mode"] == "auto"
    assert info["display_defaults"]["scalar_threshold_value"] == 0

    histogram = engine.histogram(path, bins=256, channels=[1], t=0)
    assert histogram["threshold"]["method"] == "otsu-256-v1"
    assert histogram["threshold"]["value"] == 0
    assert histogram["sampling"]["strategy"] == "exact"

    from PIL import Image

    png = engine.slice_plane(
        path,
        z=2,
        channels=[1],
        scalar_render_mode="mask",
        scalar_threshold_value=0,
    )
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    assert np.array_equal(pixels > 0, data[2] > 0)


def test_quantized_intensity_ramp_is_not_advertised_as_mask(engine, tmp_path):
    path = str(tmp_path / "ramp.tif")
    plane = np.tile(np.arange(256, dtype="uint8"), (32, 1))
    data = np.stack([plane] * 4, axis=0)
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"}, photometric="minisblack")

    info = engine.viewer_info(path)
    assert info["data_semantics"]["kind"] == "intensity"
    assert info["data_semantics"]["supported_modes"] == ["intensity"]


def test_tomm_like_probability_volume_is_suggested_as_mask_near_120(engine, tmp_path):
    path = str(tmp_path / "tomm-probability-mask.tif")
    values = np.concatenate(
        [
            np.zeros(7500, dtype="uint8"),
            np.full(500, 120, dtype="uint8"),
            np.full(2000, 255, dtype="uint8"),
        ]
    )
    np.random.default_rng(120).shuffle(values)
    data = values.reshape(4, 50, 50)
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"}, photometric="minisblack")

    info = engine.viewer_info(path)
    semantics = info["data_semantics"]
    assert semantics["kind"] == "probability_mask"
    assert semantics["basis"] == "bounded_scalar_profile"
    assert semantics["strength"] == "suggested"
    assert semantics["supported_modes"] == ["intensity", "mask"]
    assert semantics["recommended_view"] == "intensity"
    assert semantics["threshold"]["value"] == 120

    histogram = engine.histogram(path, bins=256, channels=[1], t=0)
    assert histogram["threshold"]["value"] == 120
    assert histogram["data_semantics"] == semantics


def test_large_two_code_volume_is_mask_capable_but_not_auto_selected(engine, tmp_path):
    path = str(tmp_path / "large-binary-mask.tif")
    data = np.zeros((33, 256, 256), dtype="uint8")
    data[:, 48:208, 64:192] = 255
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"}, photometric="minisblack")

    semantics = engine.viewer_info(path)["data_semantics"]
    assert semantics["kind"] == "binary_mask"
    assert semantics["strength"] == "suggested"
    assert semantics["supported_modes"] == ["intensity", "mask"]
    assert semantics["recommended_view"] == "intensity"


@pytest.mark.parametrize("distribution", ["sparse-fluorescence", "tomography"])
def test_intensity_distributions_are_not_advertised_as_masks(engine, tmp_path, distribution):
    rng = np.random.default_rng(44)
    if distribution == "sparse-fluorescence":
        data = rng.poisson(2, size=(8, 128, 128)).astype("uint16")
        bright = rng.random(data.shape) < 0.01
        data[bright] += rng.integers(50, 900, size=int(bright.sum()), dtype="uint16")
    else:
        data = np.clip(rng.normal(1_500, 320, size=(8, 128, 128)), 0, 4095).astype("uint16")
    path = str(tmp_path / f"{distribution}.tif")
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"}, photometric="minisblack")

    semantics = engine.viewer_info(path)["data_semantics"]
    assert semantics["kind"] == "intensity"
    assert semantics["supported_modes"] == ["intensity"]
    assert engine.viewer_info(path)["scalar_mask_capability"]["dtype"] == "uint16"


def test_scalar_preview_chunks_source_planes_larger_than_four_mib(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 4096, 2048, 2, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 4096 * 2
        read_sizes: list[int] = []

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            y0, y1, x0, x1 = box
            return (y1 - y0) * (x1 - x0) * 2

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


def test_nearest_preview_chunks_both_source_axes(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 3_000_000, 2, 1, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 2
        read_sizes: list[int] = []

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            y0, y1, x0, x1 = box
            return (y1 - y0) * (x1 - x0) * 2

        def read(self, *, t, c, z, level, box=None):
            assert box is not None
            y0, y1, x0, x1 = box
            read_bytes = (y1 - y0) * (x1 - x0) * 2
            self.read_sizes.append(read_bytes)
            yy = np.arange(y0, y1, dtype="uint16")[:, None]
            xx = np.arange(x0, x1, dtype="uint32")[None, :]
            return (yy + xx).astype("uint16")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    with pytest.raises(ValueError, match="DDA crossing|dimension|bytes"):
        engine.scalar_planes(
            "wide.ome.tiff",
            zs=[0],
            channel=0,
            t=0,
            pages=0,
            sampling="nearest",
        )
    assert source.read_sizes == []


def test_exact_mask_plane_chunks_source_reads(engine):
    class SyntheticSource:
        x, y, z, c, t = 3_000_000, 2, 1, 1, 1
        dtype = np.dtype("uint16")
        read_sizes: list[int] = []

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            y0, y1, x0, x1 = box
            return (y1 - y0) * (x1 - x0) * 2

        def read(self, *, t, c, z, level, box=None):
            assert box is not None
            y0, y1, x0, x1 = box
            read_bytes = (y1 - y0) * (x1 - x0) * 2
            self.read_sizes.append(read_bytes)
            values = np.arange(x0, x1, dtype="uint32")[None, :]
            return np.broadcast_to(values, (y1 - y0, x1 - x0)).astype("uint16")

    source = SyntheticSource()
    rendered = engine._bounded_mask_plane(source, t=0, channel=0, z=0, threshold=120)

    assert rendered.shape == (2, 3_000_000)
    assert rendered[0, 120] == 0
    assert rendered[0, 121] == 255
    assert max(source.read_sizes) <= 4 * 1024 * 1024


def test_exact_mask_plane_sums_all_decode_work_before_first_read(engine):
    class SyntheticSource:
        x, y, z, c, t = 3_000_000, 2, 1, 1, 1
        dtype = np.dtype("uint16")
        reads = 0

        def estimate_read_work(self, **_kwargs):
            # Every small intersection re-decodes the same oversized source chunk.
            return 256 * 1024 * 1024

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("over-budget mask work must reject before reading")

    source = SyntheticSource()
    with pytest.raises(ValueError, match="decode work"):
        engine._bounded_mask_plane(
            source,
            t=0,
            channel=0,
            z=0,
            threshold=120,
        )
    assert source.reads == 0


def test_float32_mask_membership_is_rejected_before_source_read(engine):
    class SyntheticSource:
        x, y, z, c, t = 1, 1, 1, 1, 1
        dtype = np.dtype("float32")
        reads = 0

        def read(self, **_kwargs):
            self.reads += 1
            return np.asarray([[1.0]], dtype="float32")

    source = SyntheticSource()
    with pytest.raises(ValueError, match="exact mask membership"):
        engine._bounded_mask_plane(
            source,
            t=0,
            channel=0,
            z=0,
            threshold=0.9999999701976776,
        )
    assert source.reads == 0


def test_float64_mask_membership_is_rejected_before_source_read(engine):
    class SyntheticSource:
        x, y, z, c, t = 1, 1, 1, 1, 1
        dtype = np.dtype("float64")
        reads = 0

        def read(self, **_kwargs):
            self.reads += 1
            return np.asarray([[1.0]], dtype="float64")

    source = SyntheticSource()
    with pytest.raises(ValueError, match="exact mask membership"):
        engine._bounded_mask_plane(
            source,
            t=0,
            channel=0,
            z=0,
            threshold=0.5,
        )
    assert source.reads == 0


@pytest.mark.parametrize("chunk_bytes", [None, 5 * 1024 * 1024])
def test_scalar_profile_rejects_unbounded_decoder_work_before_read(
    engine, monkeypatch, chunk_bytes
):
    class SyntheticSource:
        x, y, z, c, t = 1, 1, 32, 1, 1
        dtype = np.dtype("uint16")
        scene_count = 1
        max_decoded_chunk_bytes = chunk_bytes
        reads = 0

        def read(self, **_kwargs):
            self.reads += 1
            return np.zeros((1, 1), dtype="uint16")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    with pytest.raises(ValueError, match="chunk geometry|decode work"):
        engine.histogram("unbounded-profile.ome.tiff", channels=[1])
    assert source.reads == 0


def test_scalar_profile_samples_exact_xy_center(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 4096, 4096, 3, 1, 1
        dtype = np.dtype("uint8")
        scene_count = 1
        max_decoded_chunk_bytes = 64 * 1024

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            return (box[1] - box[0]) * (box[3] - box[2])

        def read(self, *, box=None, **_kwargs):
            assert box is not None
            output = np.zeros((box[1] - box[0], box[3] - box[2]), dtype="uint8")
            if box[0] <= self.y // 2 < box[1] and box[2] <= self.x // 2 < box[3]:
                output[:, output.shape[1] // 2 :] = 255
            return output

    monkeypatch.setattr(engine, "_source", lambda _path: SyntheticSource())
    histogram = engine.histogram("center-signal.ome.tiff", channels=[1])
    assert histogram["data_semantics"]["kind"] == "binary_mask"


def test_scalar_profile_sums_intersected_chunk_work_before_read(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 4096, 4096, 3, 1, 1
        dtype = np.dtype("uint16")
        scene_count = 1
        max_decoded_chunk_bytes = 32 * 1024 * 1024
        reads = 0

        def estimate_read_work(self, **_kwargs):
            return self.max_decoded_chunk_bytes

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("over-budget plan must fail before pixel reads")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    with pytest.raises(ValueError, match="decode work"):
        engine.histogram("over-budget.ome.tiff", channels=[1])
    assert source.reads == 0


@pytest.mark.parametrize("estimated_work", [1024, 256 * 1024 * 1024])
def test_scalar_mask_capability_admission_is_preplanned_without_reads(
    engine, monkeypatch, estimated_work
):
    class SyntheticSource:
        x, y, z, c, t = 32, 24, 5, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 1024
        is_photo = False
        reads = 0

        def estimate_read_work(self, **_kwargs):
            return estimated_work

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("capability admission must not read pixels")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    monkeypatch.setattr(
        bioio_engine_module,
        "_source_generation",
        lambda _path: (1, 1, 2, 3, 4, 5),
    )
    if estimated_work > 1024:
        with pytest.raises(ValueError, match="decode work"):
            engine._admit_scalar_mask_surfaces("mask.ome.tiff", source)
    else:
        admission = engine._admit_scalar_mask_surfaces("mask.ome.tiff", source)
        assert admission["surfaces"]["exact_plane"]["admitted_decode_work_bytes"] > 0
        assert admission["surfaces"]["nearest_volume"]["read_count"] > 0
    assert source.reads == 0


def test_scalar_mask_capability_admits_each_surface_independently(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 1, 1, 200, 1, 1
        dtype = np.dtype("uint8")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 2 * 1024 * 1024
        is_photo = False
        reads = 0

        def estimate_read_work(self, **_kwargs):
            return self.max_decoded_chunk_bytes

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("capability admission must not read pixels")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)
    monkeypatch.setattr(
        bioio_engine_module,
        "_source_generation",
        lambda _path: (1, 1, 2, 3, 4, 5),
    )

    admission = engine._admit_scalar_mask_surfaces("mask.ome.tiff", source)

    assert set(admission["surfaces"]) == {"exact_plane", "histogram", "nearest_volume"}
    assert admission["surfaces"]["exact_plane"]["read_count"] == 1
    assert admission["surfaces"]["histogram"]["read_count"] <= 160
    assert admission["surfaces"]["nearest_volume"]["read_count"] == 200
    assert source.reads == 0


def test_scalar_mask_capability_admission_is_bounded_for_skinny_huge_z(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 1, 1, 100_000, 1, 1
        dtype = np.dtype("uint8")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 1
        is_photo = False
        estimate_calls = 0

        def estimate_read_work(self, **_kwargs):
            self.estimate_calls += 1
            return 1

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    with pytest.raises(ValueError, match="dimension|DDA crossing"):
        engine._admit_scalar_mask_surfaces("skinny-mask.ome.tiff", source)
    assert source.estimate_calls == 0


def test_box_scalar_volume_preplans_cumulative_chunk_work_before_first_read(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 8, 8, 16, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 64 * 1024 * 1024
        reads = 0

        def estimate_read_work(self, **_kwargs):
            return self.max_decoded_chunk_bytes

        def read(self, **_kwargs):
            self.reads += 1
            raise AssertionError("over-budget box work must reject before reading")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    with pytest.raises(ValueError, match="decode work"):
        engine.scalar_planes(
            "repeated-chunk.ome.tiff",
            zs=range(source.z),
            channel=0,
            t=0,
            pages=0,
            sampling="box",
        )
    assert source.reads == 0


def test_box_scalar_volume_safe_admission_reads_each_output_once(engine, monkeypatch):
    class SyntheticSource:
        x, y, z, c, t = 4, 3, 5, 1, 1
        dtype = np.dtype("uint16")
        spacing_zyx = (1.0, 1.0, 1.0)
        scene_count = 1
        max_decoded_chunk_bytes = 24
        reads = 0

        def estimate_read_work(self, *, box=None, **_kwargs):
            assert box is not None
            return (box[1] - box[0]) * (box[3] - box[2]) * 2

        def read(self, *, z, box=None, **_kwargs):
            assert box is not None
            self.reads += 1
            return np.full((box[1] - box[0], box[3] - box[2]), z, dtype="uint16")

    source = SyntheticSource()
    monkeypatch.setattr(engine, "_source", lambda _path: source)

    planes = engine.scalar_planes(
        "safe-box.ome.tiff",
        zs=range(source.z),
        channel=0,
        t=0,
        pages=0,
        sampling="box",
    )

    assert [int(plane[0, 0]) for plane in planes] == list(range(source.z))
    assert source.reads == source.z


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
    histogram = engine.histogram(path, channels=[1])
    sampling = histogram["sampling"]
    assert sampling["declared_max_decoded_chunk_bytes"] == 924 * 2
    assert sampling["admitted_decode_work_bytes"] <= sampling["max_decode_work_bytes"]


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
