"""Unit tests for the h5py-backed HDF5 data viewer (imaging/hdf5.py).

Fixtures generate tiny synthetic ``.h5`` / ``.dream3d`` files in-test (no committed
binary fixtures), then exercise every reader entry point + edge cases + the
viewer-info ``kind:"hdf5"`` detection. Gated on h5py so a bare environment skips
rather than errors.
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("PIL")

from ultra_deepagents.imaging import hdf5  # noqa: E402

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def general_h5(tmp_path):
    """A general HDF5 file: 2D image, 3D scalar/label/vector/rgb volumes, a compound
    table, a 1-D series, a string dataset, an empty group, and string/scalar attrs."""
    path = str(tmp_path / "sample.h5")
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.attrs["description"] = "synthetic sample"
        f.attrs["scale"] = np.float64(1.5)
        f.attrs["dims"] = np.array([20, 40, 50], dtype="i4")
        img = f.create_group("image2d")
        img.create_dataset("gray", data=(rng.random((64, 80)) * 60000).astype("uint16"))
        vol = f.create_group("volume")
        ct = vol.create_dataset("ct", data=(rng.random((20, 40, 50)) * 4000).astype("int16"))
        ct.attrs["units"] = "HU"
        vol.create_dataset("labels", data=rng.integers(0, 5, (20, 40, 50)).astype("uint8"))
        vol.create_dataset("euler", data=rng.random((20, 40, 50, 3)).astype("float32"))
        vol.create_dataset("ipf", data=rng.integers(0, 255, (20, 40, 50, 3)).astype("uint8"))
        f.create_group("empty_group")
        comp_dt = np.dtype([("id", "i4"), ("size", "f4"), ("name", "S8")])
        rows = np.zeros(50, dtype=comp_dt)
        rows["id"] = np.arange(50)
        rows["size"] = rng.random(50).astype("f4")
        rows["name"] = b"grain"
        f.create_dataset("table", data=rows)
        f.create_dataset("series", data=rng.random(200).astype("float64"))
        f.create_dataset("strings", data=np.array([b"alpha", b"beta", b"gamma"], dtype="S8"))
    return path


@pytest.fixture()
def dream3d_h5(tmp_path):
    """A minimal DREAM.3D-shaped file: /DataContainers/.../CellData + CellFeatureData
    + CellEnsembleData/PhaseName + _SIMPL_GEOMETRY."""
    path = str(tmp_path / "synthetic.dream3d")
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        f.attrs["DREAM3D Version"] = "6.5.0"
        dc = f.create_group("DataContainers")
        syn = dc.create_group("SyntheticVolume")
        geom = syn.create_group("_SIMPL_GEOMETRY")
        geom.create_dataset("DIMENSIONS", data=np.array([50, 40, 20], dtype="i8"))
        geom.create_dataset("SPACING", data=np.array([0.5, 0.5, 0.5], dtype="f4"))
        geom.create_dataset("ORIGIN", data=np.array([0.0, 0.0, 0.0], dtype="f4"))
        cell = syn.create_group("CellData")
        cell.create_dataset("FeatureIds", data=rng.integers(0, 30, (20, 40, 50, 1)).astype("i4"))
        cell.create_dataset("Confidence Index", data=rng.random((20, 40, 50, 1)).astype("f4"))
        cell.create_dataset("IPFColors", data=rng.integers(0, 255, (20, 40, 50, 3)).astype("u1"))
        cell.create_dataset("EulerAngles", data=rng.random((20, 40, 50, 3)).astype("f4"))
        feat = syn.create_group("CellFeatureData")
        feat.create_dataset("EquivalentDiameters", data=np.abs(rng.random(30)).astype("f4"))
        feat.create_dataset("NumNeighbors", data=rng.integers(0, 20, 30).astype("i4"))
        feat.create_dataset("AvgEulerAngles", data=rng.random((30, 3)).astype("f4"))
        ens = syn.create_group("CellEnsembleData")
        ens.create_dataset("PhaseName", data=np.array([b"Invalid Phase", b"Nickel"], dtype="S16"))
    return path


@pytest.fixture()
def feature_filter_h5(tmp_path):
    """Co-registered CellData with one Feature ID in disconnected locations."""
    path = str(tmp_path / "feature-filter.dream3d")
    ids = np.full((2, 3, 4, 1), 7, dtype="u4")
    ids[0, 0, 0, 0] = 25
    ids[1, 2, 3, 0] = 25
    ids[0, 1, 2, 0] = 0
    ipf = np.arange(2 * 3 * 4 * 3, dtype="u1").reshape(2, 3, 4, 3)
    eulers = np.arange(2 * 3 * 4 * 3, dtype="f4").reshape(2, 3, 4, 3)
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([4, 3, 2], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=ids)
        cell.create_dataset("EulerAngles", data=eulers)
        cell.create_dataset("IPFColor", data=ipf)
    return path


@pytest.fixture()
def dream3d_real_layout_h5(tmp_path):
    """Small regression fixture matching the naming/layout of the real 6.9 GiB file.

    StatsGenerator owns the first (empty) geometry and repeats the ensemble phase
    metadata.  The usable image geometry and per-grain arrays live in the later
    SyntheticVolumeDataContainer, whose feature group is named ``Grain Data``.
    """
    path = str(tmp_path / "real-layout.dream3d")
    grain_count = 5_000
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        dc = f.create_group("DataContainers")

        stats = dc.create_group("StatsGeneratorDataContainer")
        stats.create_group("_SIMPL_GEOMETRY")
        stats_ensemble = stats.create_group("CellEnsembleData")
        stats_ensemble.create_dataset(
            "PhaseName",
            data=np.array([b"Unknown Phase", b"Primary"], dtype="S16"),
        )

        syn = dc.create_group("SyntheticVolumeDataContainer")
        geom = syn.create_group("_SIMPL_GEOMETRY")
        geom.create_dataset("DIMENSIONS", data=np.array([20, 25, 10], dtype="i8"))
        geom.create_dataset("SPACING", data=np.array([0.5, 0.5, 0.5], dtype="f4"))
        geom.create_dataset("ORIGIN", data=np.array([1.0, 2.0, 3.0], dtype="f4"))

        cell = syn.create_group("CellData")
        cell.create_dataset(
            "FeatureIds",
            data=np.arange(1, grain_count + 1, dtype="i4").reshape(10, 25, 20, 1),
        )
        cell.create_dataset("IPFColor", data=np.zeros((10, 25, 20, 3), dtype="u1"))

        # Row zero is DREAM.3D's reserved feature sentinel.  Row one deliberately
        # contains legitimate zero measurements/orientation and must be preserved.
        grain = syn.create_group("Grain Data")
        volumes = np.ones((grain_count + 1, 1), dtype="f4")
        volumes[0, 0] = 0.0
        volumes[1, 0] = 0.0
        grain.create_dataset("Volumes", data=volumes)
        neighbors = np.full((grain_count + 1, 1), 4, dtype="i4")
        neighbors[0, 0] = 0
        neighbors[1, 0] = 0
        grain.create_dataset("NumNeighbors", data=neighbors)
        # Neighbor lists are not tuple arrays and can be far longer than grain count.
        grain.create_dataset("NeighborList", data=np.zeros(grain_count * 2, dtype="i8"))
        eulers = np.full((grain_count + 1, 3), 0.25, dtype="f4")
        eulers[0, :] = 0.0
        eulers[1, :] = 0.0
        grain.create_dataset("EulerAngles", data=eulers)

        ensemble = syn.create_group("CellEnsembleData")
        ensemble.create_dataset(
            "PhaseName",
            data=np.array([b"Unknown Phase", b"Primary"], dtype="S16"),
        )
    return path


def _assert_json(obj):
    """Every response is cast directly by the FE (no normalizer), so it must be strict
    JSON — no NaN/Inf/bytes/numpy scalars leaking through."""
    json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_is_hdf5_data_file_extension_gate():
    # Generic HDF5 containers.
    assert hdf5.is_hdf5_data_file("vol.h5")
    assert hdf5.is_hdf5_data_file("VOL.HDF5")
    assert hdf5.is_hdf5_data_file("micro.hdf")
    assert hdf5.is_hdf5_data_file("grid.he5")        # HDF-EOS5
    # Materials.
    assert hdf5.is_hdf5_data_file("micro.dream3d")
    assert hdf5.is_hdf5_data_file("orientation.h5ebsd")  # EBSD scans
    assert hdf5.is_hdf5_data_file("SCAN.H5EBSD")          # case-insensitive
    # Single-cell bio + NeXus.
    assert hdf5.is_hdf5_data_file("cells.h5ad")      # AnnData
    assert hdf5.is_hdf5_data_file("matrix.loom")     # Loom
    assert hdf5.is_hdf5_data_file("beamline.nxs")    # NeXus
    # Page/series suffix tolerated across the family.
    assert hdf5.is_hdf5_data_file("series.h5_3")
    assert hdf5.is_hdf5_data_file("series.h5_12")
    assert hdf5.is_hdf5_data_file("scan.h5ebsd_2")
    # HDF5-based image/non-image formats must NOT be captured by the data explorer.
    assert not hdf5.is_hdf5_data_file("cells.ims")   # Imaris -> libbioimage/bioio
    assert not hdf5.is_hdf5_data_file("data.mat")    # only v7.3 is HDF5
    assert not hdf5.is_hdf5_data_file("scan.nc")     # NetCDF (v3 is not HDF5)
    assert not hdf5.is_hdf5_data_file("scan.nc4")
    assert not hdf5.is_hdf5_data_file("photo.tif")
    # .he5 must not be shadowed by the .h5 entry, and .h5ad must not by .h5.
    assert not hdf5.is_hdf5_data_file("plain.he5.txt")
    # The series tolerance is anchored to the END: an ".h5_" substring mid-name
    # must never hijack a raster into the h5py path.
    assert not hdf5.is_hdf5_data_file("scan.h5_v2.tif")
    assert not hdf5.is_hdf5_data_file("scan.h5_2.tif")
    assert not hdf5.is_hdf5_data_file("map.h5ebsd.tif")


# ---------------------------------------------------------------------------
# viewerinfo
# ---------------------------------------------------------------------------
def test_viewerinfo_general_structure(general_h5):
    vi = hdf5.build_hdf5_viewer_info(general_h5, file_id="fid1")
    _assert_json(vi)
    assert vi["kind"] == "hdf5"
    assert vi["reader"] == "h5py"
    block = vi["hdf5"]
    assert block["enabled"] and block["supported"] and block["status"] == "ready"
    assert block["error"] is None
    assert set(block["root_keys"]) == {"image2d", "volume", "empty_group", "table", "series", "strings"}
    # root attributes are JSON-safe and present.
    assert block["root_attributes"]["description"] == "synthetic sample"
    assert block["summary"]["dataset_count"] == 8
    assert block["summary"]["group_count"] == 3  # image2d, volume, empty_group
    assert block["summary"]["truncated"] is False
    assert block["summary"]["dataset_kinds"]  # non-empty
    assert block["materials"] is None
    # default is a renderable dataset path.
    assert block["default_dataset_path"] and block["default_dataset_path"].startswith("/volume/")


def test_viewerinfo_tree_nodes(general_h5):
    block = hdf5.build_hdf5_viewer_info(general_h5)["hdf5"]

    def _find(nodes, path):
        for n in nodes:
            if n["path"] == path:
                return n
            hit = _find(n["children"], path)
            if hit is not None:
                return hit
        return None

    empty = _find(block["tree"], "/empty_group")
    assert empty is not None and empty["node_type"] == "group"
    assert empty["child_count"] == 0 and empty["children"] == []
    ct = _find(block["tree"], "/volume/ct")
    assert ct is not None and ct["node_type"] == "dataset"
    assert ct["shape"] == [20, 40, 50]
    assert ct["preview_kind"] == "scalar_volume"
    assert ct["dtype"] and ct["child_count"] == 0
    labels = _find(block["tree"], "/volume/labels")
    assert labels["preview_kind"] == "label_volume"  # name-hinted categorical


def test_viewerinfo_corrupt_file_is_metadata_only(tmp_path):
    path = str(tmp_path / "corrupt.h5")
    with open(path, "wb") as fh:
        fh.write(b"this is definitely not an HDF5 container")
    vi = hdf5.build_hdf5_viewer_info(path)
    _assert_json(vi)
    assert vi["kind"] == "hdf5"
    block = vi["hdf5"]
    assert block["supported"] is False
    assert block["status"] == "unsupported"
    assert block["error"]
    assert block["tree"] == []


def test_viewerinfo_empty_file(tmp_path):
    path = str(tmp_path / "empty.h5")
    with h5py.File(path, "w"):
        pass
    block = hdf5.build_hdf5_viewer_info(path)["hdf5"]
    assert block["supported"] is True
    assert block["tree"] == []
    assert block["summary"]["dataset_count"] == 0
    assert block["default_dataset_path"] is None


def test_attribute_metadata_is_count_array_and_byte_bounded(tmp_path):
    path = str(tmp_path / "bounded-attributes.h5")
    with h5py.File(path, "w") as f:
        f.attrs["description"] = "small"
        f.attrs["oversized_vlen"] = "X" * (hdf5.MAX_METADATA_ITEM_BYTES + 1)
        f.attrs["oversized_array"] = np.arange(hdf5.MAX_ATTR_ARRAY + 1, dtype="i8")
        for index in range(hdf5.MAX_ATTRS + 20):
            f.attrs[f"small_{index:03d}"] = index
        dset = f.create_dataset("values", data=np.arange(8, dtype="i4"))
        dset.attrs["units"] = "count"
        dset.attrs["oversized_array"] = np.arange(hdf5.MAX_ATTR_ARRAY + 1, dtype="i8")

    viewer_attrs = hdf5.build_hdf5_viewer_info(path)["hdf5"]["root_attributes"]
    summary_attrs = hdf5.dataset_summary(path, "/values")["attributes"]
    assert len(viewer_attrs) <= hdf5.MAX_ATTRS
    assert "oversized_vlen" not in viewer_attrs
    assert "oversized_array" not in viewer_attrs
    assert summary_attrs == {"units": "count"}


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
def test_summary_scalar_volume(general_h5):
    s = hdf5.dataset_summary(general_h5, "/volume/ct", file_id="fid1")
    _assert_json(s)
    assert s["preview_kind"] == "scalar_volume"
    assert s["render_policy"] == "scalar"
    assert s["delivery_mode"] == "scalar"
    assert s["texture_policy"] == "linear"
    assert s["shape"] == [20, 40, 50]
    assert s["rank"] == 3
    assert s["element_count"] == 20 * 40 * 50
    assert s["estimated_bytes"] == 20 * 40 * 50 * 2  # int16
    assert s["dimension_summary"] == {"z": 20, "y": 40, "x": 50}
    assert s["axis_sizes"] == {"T": 1, "C": 1, "Z": 20, "Y": 40, "X": 50}
    assert s["slice_axes"] == ["z", "y", "x"]
    assert s["volume_eligible"] is True
    assert "volume" in s["capabilities"] and "histogram" in s["capabilities"]
    # required collections must be arrays/objects, never null (FE casts, no normalize).
    assert isinstance(s["structured_fields"], list)
    assert isinstance(s["attributes"], dict)
    assert isinstance(s["materials_domain_tags"], list)
    assert isinstance(s["display_capabilities"], list)
    assert isinstance(s["viewer_capabilities"], list)
    # every slice axis has a preview plane whose pixel_size matches the served plane.
    for axis in s["slice_axes"]:
        plane = s["preview_planes"][axis]
        assert plane["axis"] == axis
        assert plane["pixel_size"]["width"] > 0 and plane["pixel_size"]["height"] > 0
    # z-plane pixel size equals (X, Y).
    assert s["preview_planes"]["z"]["pixel_size"] == {"width": 50, "height": 40}
    assert s["preview_planes"]["y"]["pixel_size"] == {"width": 50, "height": 20}
    assert s["preview_planes"]["x"]["pixel_size"] == {"width": 40, "height": 20}
    # scalar volume uses the scalar-volume delivery, not atlas.
    assert s["atlas_scheme"] is None
    assert s["sample_statistics"]["sample_count"] > 0
    assert s["sample_statistics"]["min"] is not None


def test_summary_label_volume(general_h5):
    s = hdf5.dataset_summary(general_h5, "/volume/labels")
    assert s["preview_kind"] == "label_volume"
    assert s["render_policy"] == "categorical"
    assert s["texture_policy"] == "nearest"
    assert s["delivery_mode"] == "atlas"
    assert s["atlas_scheme"] is not None
    scheme = s["atlas_scheme"]
    assert scheme["slice_count"] == 20
    assert scheme["atlas_width"] == scheme["slice_width"] * scheme["columns"]
    assert scheme["atlas_height"] == scheme["slice_height"] * scheme["rows"]
    assert scheme["format"] == "png"


def test_summary_publishes_exact_feature_registration_for_compatible_cell_maps(feature_filter_h5):
    source = "/DataContainers/Image/CellData/FeatureIds"
    for dataset_path in (
        source,
        "/DataContainers/Image/CellData/EulerAngles",
        "/DataContainers/Image/CellData/IPFColor",
    ):
        summary = hdf5.dataset_summary(feature_filter_h5, dataset_path)
        assert "feature_filter" in summary["capabilities"]
        assert summary["feature_filter"] == {
            "supported": True,
            "source_dataset_path": source,
            "max_ids": 64,
            "background_id": 0,
            "provenance": "co_registered_raw_integer_feature_ids",
            "registration_key": f"{source}|2x3x4|1x1x1",
            "target_role": summary["semantic_role"],
            "native_shape": [2, 3, 4],
            "preview_shape": [2, 3, 4],
            "preview_stride": {"z": 1, "y": 1, "x": 1},
        }

    unrelated = hdf5.dataset_summary(feature_filter_h5, "/DataContainers/Image/CellData/IPFColor")
    assert unrelated["feature_filter"]["source_dataset_path"] == source


def test_summary_vector_volume_components(general_h5):
    s = hdf5.dataset_summary(general_h5, "/volume/euler")
    assert s["preview_kind"] == "vector_volume"
    assert s["component_count"] == 3
    assert s["component_labels"] == ["phi1", "Phi", "phi2"]
    assert s["render_policy"] == "analysis"
    assert s["atlas_scheme"] is not None
    assert "histogram" in s["capabilities"]


def test_summary_rgb_volume(general_h5):
    s = hdf5.dataset_summary(general_h5, "/volume/ipf")
    assert s["preview_kind"] == "rgb_volume"
    assert s["render_policy"] == "display"
    assert s["component_count"] == 3
    assert s["atlas_scheme"] is not None
    # rgb is displayed directly, not histogrammed.
    assert "histogram" not in s["capabilities"]


def test_summary_2d_image_not_volume(general_h5):
    s = hdf5.dataset_summary(general_h5, "/image2d/gray")
    assert s["preview_kind"] == "scalar_volume"
    assert s["dimension_summary"] == {"z": 1, "y": 64, "x": 80}
    assert s["slice_axes"] == ["z"]
    assert s["volume_eligible"] is False
    assert s["volume_reason"]  # human sentence
    assert s["atlas_scheme"] is None


def test_summary_compound_table(general_h5):
    s = hdf5.dataset_summary(general_h5, "/table")
    assert s["preview_kind"] == "table"
    names = [f["name"] for f in s["structured_fields"]]
    assert names == ["id", "size", "name"]
    assert s["volume_eligible"] is False
    assert "histogram" not in s["capabilities"]
    assert "volume" not in s["capabilities"]


def test_summary_series_and_string(general_h5):
    ser = hdf5.dataset_summary(general_h5, "/series")
    assert ser["preview_kind"] == "series"
    st = hdf5.dataset_summary(general_h5, "/strings")
    assert st["preview_kind"] == "table"
    _assert_json(ser)
    _assert_json(st)


def test_summary_unknown_dataset_raises_not_found(general_h5):
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.dataset_summary(general_h5, "/nope/missing")
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.dataset_summary(general_h5, "")  # empty dataset_path
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.dataset_summary(general_h5, "/volume")  # a group, not a dataset


# ---------------------------------------------------------------------------
# Slice / atlas / scalar-volume
# ---------------------------------------------------------------------------
def test_slice_png_all_axes(general_h5):
    for axis in ("z", "y", "x"):
        png = hdf5.slice_png(general_h5, "/volume/ct", axis=axis, index=2)
        assert png.startswith(_PNG_MAGIC)


def test_slice_png_dims_match_preview_planes(general_h5):
    import io as _io

    from PIL import Image

    s = hdf5.dataset_summary(general_h5, "/volume/ct")
    for axis in s["slice_axes"]:
        png = hdf5.slice_png(general_h5, "/volume/ct", axis=axis, index=1)
        img = Image.open(_io.BytesIO(png))
        expected = s["preview_planes"][axis]["pixel_size"]
        assert (img.width, img.height) == (expected["width"], expected["height"])


def test_slice_index_out_of_range_clamps(general_h5):
    # A summary/slice race can send an index past the range; must clamp, not raise.
    png = hdf5.slice_png(general_h5, "/volume/ct", axis="z", index=99999)
    assert png.startswith(_PNG_MAGIC)
    png2 = hdf5.slice_png(general_h5, "/volume/ct", axis="z", index=-5)
    assert png2.startswith(_PNG_MAGIC)


def test_slice_label_and_rgb_and_vector(general_h5):
    assert hdf5.slice_png(general_h5, "/volume/labels", axis="z", index=3).startswith(_PNG_MAGIC)
    assert hdf5.slice_png(general_h5, "/volume/ipf", axis="z", index=3).startswith(_PNG_MAGIC)
    assert hdf5.slice_png(general_h5, "/volume/euler", axis="z", index=3, component=2).startswith(_PNG_MAGIC)


def test_default_dataset_path_prefers_materials_recommendation(general_h5, dream3d_h5):
    # General file: the largest renderable dataset from the tree walk.
    assert hdf5.default_dataset_path(general_h5) is not None
    # DREAM.3D: the recommended materials map wins (IPFColors per the probe).
    recommended = hdf5.default_dataset_path(dream3d_h5)
    assert recommended is not None
    assert "CellData" in recommended


def test_default_dataset_path_is_memoized(general_h5, monkeypatch):
    # The grid z-scrub resolves the default dataset once per FRAME; the walk +
    # DREAM.3D materials probe (a full FeatureIds scan) must not rerun per call.
    hdf5._DEFAULT_DATASET_CACHE.clear()
    opens = {"n": 0}
    real_open = hdf5._open

    def counting_open(path):
        opens["n"] += 1
        return real_open(path)

    monkeypatch.setattr(hdf5, "_open", counting_open)
    first = hdf5.default_dataset_path(general_h5)
    second = hdf5.default_dataset_path(general_h5)
    assert first == second
    assert opens["n"] == 1


def test_default_dataset_path_corrupt_file_maps_to_decode_class(tmp_path):
    path = str(tmp_path / "not-really.h5")
    with open(path, "wb") as f:
        f.write(b"junk that is not HDF5")
    with pytest.raises(hdf5.Hdf5Error) as excinfo:
        hdf5.default_dataset_path(path)
    # "unsupported" is a service _DECODE_ERROR_MARKERS token -> 422, not 500.
    assert "unsupported" in str(excinfo.value)


def test_viewer_info_top_level_axis_sizes(general_h5, dream3d_h5):
    # The grid hover z-scrub reads top-level axis_sizes from the viewer info;
    # they must reflect the DEFAULT dataset's preview grid.
    info = hdf5.build_hdf5_viewer_info(general_h5)
    assert info["axis_sizes"]["Z"] == 20
    assert info["axis_sizes"]["Y"] == 40
    assert info["axis_sizes"]["X"] == 50
    d3 = hdf5.build_hdf5_viewer_info(dream3d_h5)
    assert d3["axis_sizes"]["Z"] == 20


def test_thumbnail_png_renders_and_bounds(general_h5):
    import io as _io

    from PIL import Image

    png = hdf5.thumbnail_png(general_h5, max_size=32)
    assert png.startswith(_PNG_MAGIC)
    img = Image.open(_io.BytesIO(png))
    assert max(img.width, img.height) <= 32
    # A max_size larger than the plane returns the native-size slice untouched.
    native = hdf5.thumbnail_png(general_h5, max_size=4096)
    assert native.startswith(_PNG_MAGIC)


def test_thumbnail_png_dream3d(dream3d_h5):
    assert hdf5.thumbnail_png(dream3d_h5, max_size=64).startswith(_PNG_MAGIC)


def test_thumbnail_png_no_renderable_dataset_raises(tmp_path):
    # Tables/series only — nothing sliceable; the engine surfaces the standard
    # thumbnail error and the grid falls back to its icon.
    path = str(tmp_path / "tables-only.h5")
    with h5py.File(path, "w") as f:
        comp_dt = np.dtype([("id", "i4"), ("size", "f4")])
        f.create_dataset("table", data=np.zeros(10, dtype=comp_dt))
    with pytest.raises(hdf5.Hdf5Error) as excinfo:
        hdf5.thumbnail_png(path)
    assert "unsupported" in str(excinfo.value)  # -> 422 through the service mapping


def test_atlas_png_matches_scheme(general_h5):
    import io as _io

    from PIL import Image

    s = hdf5.dataset_summary(general_h5, "/volume/labels")
    scheme = s["atlas_scheme"]
    png = hdf5.atlas_png(general_h5, "/volume/labels")
    assert png.startswith(_PNG_MAGIC)
    img = Image.open(_io.BytesIO(png))
    assert (img.width, img.height) == (scheme["atlas_width"], scheme["atlas_height"])


def test_large_label_atlas_resizes_with_palette_preserving_nearest_neighbor(tmp_path):
    import io as _io

    from PIL import Image

    path = str(tmp_path / "large-labels.h5")
    yy, xx = np.indices((520, 516))
    plane = ((yy + xx) % 7 + 1).astype("i4")
    plane[(yy % 11 == 0) & (xx % 13 == 0)] = 0
    with h5py.File(path, "w") as f:
        f.create_dataset("FeatureIds", data=np.stack([plane, plane[::-1]], axis=0))

    first = hdf5.atlas_png(path, "/FeatureIds")
    second = hdf5.atlas_png(path, "/FeatureIds")
    assert first == second

    decoded = np.asarray(Image.open(_io.BytesIO(first)).convert("RGB"), dtype="uint8")
    actual_colors = {tuple(int(channel) for channel in pixel) for pixel in decoded.reshape(-1, 3)}
    palette = hdf5._label_to_rgb(np.arange(8, dtype="i4"))
    expected_colors = {tuple(int(channel) for channel in pixel) for pixel in palette.reshape(-1, 3)}
    assert actual_colors == expected_colors
    assert actual_colors - {(0, 0, 0)}


@pytest.mark.parametrize(
    "raw",
    ["", "0", "-1", "+1", "1,,2", "1 2", "4294967296", "1,"],
)
def test_feature_filter_rejects_malformed_or_nonpositive_ids(feature_filter_h5, raw):
    with pytest.raises(hdf5.Hdf5Error):
        hdf5.atlas_png(
            feature_filter_h5,
            "/DataContainers/Image/CellData/IPFColor",
            feature_ids=raw,
        )


def test_feature_filter_rejects_more_than_64_unique_ids(feature_filter_h5):
    with pytest.raises(hdf5.Hdf5Error, match="at most 64"):
        hdf5.atlas_png(
            feature_filter_h5,
            "/DataContainers/Image/CellData/IPFColor",
            feature_ids=",".join(str(value) for value in range(1, 66)),
        )


def test_filtered_atlas_keeps_every_disconnected_occurrence_of_selected_id(feature_filter_h5):
    from PIL import Image

    target_path = "/DataContainers/Image/CellData/IPFColor"
    summary = hdf5.dataset_summary(feature_filter_h5, target_path)
    scheme = summary["atlas_scheme"]
    filtered = Image.open(
        io.BytesIO(hdf5.atlas_png(feature_filter_h5, target_path, feature_ids="25"))
    ).convert("RGBA")
    atlas = np.asarray(filtered, dtype="u1")
    selected = []
    for z in range(scheme["slice_count"]):
        row, column = divmod(z, scheme["columns"])
        cell = atlas[
            row * scheme["slice_height"]:(row + 1) * scheme["slice_height"],
            column * scheme["slice_width"]:(column + 1) * scheme["slice_width"],
        ]
        selected.extend(np.argwhere(cell[..., 3] == 255).tolist())
        assert np.all(cell[cell[..., 3] == 0] == 0)

    # ID 25 occurs in two disconnected corners on different Z planes. Filtering
    # is global raw-ID equality, not a connected-component fill around one click.
    assert len(selected) == 2


def test_filtered_slice_uses_union_of_raw_ids_and_preserves_transparency(feature_filter_h5):
    from PIL import Image

    target_path = "/DataContainers/Image/CellData/FeatureIds"
    image = Image.open(
        io.BytesIO(
            hdf5.slice_png(
                feature_filter_h5,
                target_path,
                axis="z",
                index=0,
                feature_ids="25,7,25",
            )
        )
    ).convert("RGBA")
    rgba = np.asarray(image, dtype="u1")
    assert int(np.count_nonzero(rgba[..., 3] == 255)) == 11
    assert np.all(rgba[rgba[..., 3] == 0] == 0)


@pytest.mark.parametrize("dataset_name", ["FeatureIds", "EulerAngles", "IPFColor"])
def test_filtered_atlas_applies_multi_id_union_globally_to_compatible_maps(
    feature_filter_h5, dataset_name
):
    from PIL import Image

    dataset_path = f"/DataContainers/Image/CellData/{dataset_name}"
    rgba = np.asarray(
        Image.open(
            io.BytesIO(hdf5.atlas_png(feature_filter_h5, dataset_path, feature_ids="25,7"))
        ).convert("RGBA"),
        dtype="u1",
    )
    assert int(np.count_nonzero(rgba[..., 3] == 255)) == 23
    assert np.all(rgba[rgba[..., 3] == 0] == 0)


def test_feature_filter_fails_closed_for_soft_linked_identity(tmp_path):
    path = str(tmp_path / "soft-feature-ids.dream3d")
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        hidden = image.create_dataset("HiddenIds", data=np.ones((2, 2, 2, 1), dtype="u4"))
        cell = image.create_group("CellData")
        cell["FeatureIds"] = h5py.SoftLink(hidden.name)
        cell.create_dataset("IPFColor", data=np.ones((2, 2, 2, 3), dtype="u1"))

    summary = hdf5.dataset_summary(path, "/DataContainers/Image/CellData/IPFColor")
    assert "feature_filter" not in summary["capabilities"]
    assert summary["feature_filter"] is None
    with pytest.raises(hdf5.Hdf5Error, match="hard-linked"):
        hdf5.atlas_png(path, "/DataContainers/Image/CellData/IPFColor", feature_ids="1")


def test_feature_filter_rejects_distinct_feature_and_grain_identities(feature_filter_h5):
    with h5py.File(feature_filter_h5, "r+") as file:
        cell = file["/DataContainers/Image/CellData"]
        cell.create_dataset("GrainIds", data=np.asarray(cell["FeatureIds"]))
    summary = hdf5.dataset_summary(
        feature_filter_h5, "/DataContainers/Image/CellData/IPFColor"
    )
    assert summary["feature_filter"] is None
    with pytest.raises(hdf5.Hdf5Error, match="ambiguous"):
        hdf5.atlas_png(
            feature_filter_h5,
            "/DataContainers/Image/CellData/IPFColor",
            feature_ids="7",
        )


def test_feature_filter_accepts_feature_and_grain_aliases(feature_filter_h5):
    with h5py.File(feature_filter_h5, "r+") as file:
        cell = file["/DataContainers/Image/CellData"]
        cell["GrainIds"] = cell["FeatureIds"]
    summary = hdf5.dataset_summary(
        feature_filter_h5, "/DataContainers/Image/CellData/IPFColor"
    )
    assert summary["feature_filter"]["source_dataset_path"].endswith("/FeatureIds")


def test_geometry_qualified_rank3_feature_ids_with_x_three_are_zyx(tmp_path):
    path = str(tmp_path / "rank3-x3.dream3d")
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([3, 7, 8], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((8, 7, 3), dtype="u4"))
        cell.create_dataset("IPFColor", data=np.ones((8, 7, 3, 3), dtype="u1"))

    summary = hdf5.dataset_summary(path, "/DataContainers/Image/CellData/FeatureIds")
    assert summary["preview_kind"] == "label_volume"
    assert summary["axis_sizes"] == {"T": 1, "C": 1, "Z": 8, "Y": 7, "X": 3}
    assert summary["feature_filter"]["native_shape"] == [8, 7, 3]


def test_dataset_summary_calibrates_preview_spacing_to_native_geometry_extent(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "preview-spacing.dream3d")
    native_zyx = (11, 9, 7)
    raw_spacing_xyz = (0.37, 0.61, 1.13)
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([7, 9, 11], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.array(raw_spacing_xyz, dtype="f8"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

        image_b = file["DataContainers"].create_group("ImageB")
        geometry_b = image_b.create_group("_SIMPL_GEOMETRY")
        geometry_b.create_dataset("DIMENSIONS", data=np.array([7, 9, 11], dtype="u8"))
        geometry_b.create_dataset("SPACING", data=np.array([2.0, 3.0, 4.0], dtype="f8"))
        geometry_b.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell_b = image_b.create_group("CellData")
        cell_b.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

        file.create_group("Other").create_dataset(
            "Scalar", data=np.ones(native_zyx, dtype="f4")
        )

    monkeypatch.setattr(hdf5, "ATLAS_CELL_CAP", 4)
    monkeypatch.setattr(hdf5, "PREVIEW_MAX_PLANE", 5)

    summary = hdf5.dataset_summary(
        path, "/DataContainers/Image/CellData/FeatureIds"
    )
    assert summary["geometry"]["path"].endswith("/Image/_SIMPL_GEOMETRY")
    assert summary["geometry"]["dimensions"] == [7, 9, 11]
    assert summary["geometry"]["spacing"] == pytest.approx(raw_spacing_xyz)
    assert summary["axis_sizes"] == {"T": 1, "C": 1, "Z": 4, "Y": 5, "X": 4}

    expected_spacing = {
        "x": raw_spacing_xyz[0] * 7 / 4,
        "y": raw_spacing_xyz[1] * 9 / 5,
        "z": raw_spacing_xyz[2] * 11 / 4,
    }
    assert summary["physical_spacing"] == pytest.approx(expected_spacing)
    for axis, native_count in zip(("z", "y", "x"), native_zyx, strict=True):
        preview_count = summary["axis_sizes"][axis.upper()]
        raw_spacing = raw_spacing_xyz[{"x": 0, "y": 1, "z": 2}[axis]]
        assert preview_count * summary["physical_spacing"][axis] == pytest.approx(
            native_count * raw_spacing
        )

    assert summary["preview_planes"]["z"]["spacing"] == pytest.approx(
        {"row": expected_spacing["y"], "col": expected_spacing["x"]}
    )
    assert summary["preview_planes"]["y"]["spacing"] == pytest.approx(
        {"row": expected_spacing["z"], "col": expected_spacing["x"]}
    )
    assert summary["preview_planes"]["x"]["spacing"] == pytest.approx(
        {"row": expected_spacing["z"], "col": expected_spacing["y"]}
    )
    assert summary["preview_planes"]["z"]["world_size"] == pytest.approx(
        {"width": 7 * raw_spacing_xyz[0], "height": 9 * raw_spacing_xyz[1]}
    )
    assert summary["preview_planes"]["y"]["world_size"] == pytest.approx(
        {"width": 7 * raw_spacing_xyz[0], "height": 11 * raw_spacing_xyz[2]}
    )
    assert summary["preview_planes"]["x"]["world_size"] == pytest.approx(
        {"width": 9 * raw_spacing_xyz[1], "height": 11 * raw_spacing_xyz[2]}
    )

    summary_b = hdf5.dataset_summary(
        path, "/DataContainers/ImageB/CellData/FeatureIds"
    )
    assert summary_b["geometry"]["path"].endswith("/ImageB/_SIMPL_GEOMETRY")
    assert summary_b["geometry"]["spacing"] == [2.0, 3.0, 4.0]
    assert summary_b["physical_spacing"] == pytest.approx(
        {"x": 2.0 * 7 / 4, "y": 3.0 * 9 / 5, "z": 4.0 * 11 / 4}
    )

    mismatched = hdf5.dataset_summary(path, "/Other/Scalar")
    assert mismatched["geometry"] is None
    assert mismatched["physical_spacing"] == {"x": None, "y": None, "z": None}
    assert mismatched["measurement_policy"] == "pixel-only"


def test_dataset_summary_rejects_overflowed_effective_preview_spacing(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "overflow-spacing.dream3d")
    native_zyx = (11, 9, 7)
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([7, 9, 11], dtype="u8"))
        geometry.create_dataset(
            "SPACING", data=np.array([np.finfo("f8").max, 0.61, 1.13], dtype="f8")
        )
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

    monkeypatch.setattr(hdf5, "ATLAS_CELL_CAP", 4)
    monkeypatch.setattr(hdf5, "PREVIEW_MAX_PLANE", 5)

    summary = hdf5.dataset_summary(
        path, "/DataContainers/Image/CellData/FeatureIds"
    )
    assert summary["geometry"]["spacing"][0] == np.finfo("f8").max
    assert summary["axis_sizes"] == {"T": 1, "C": 1, "Z": 4, "Y": 5, "X": 4}
    assert summary["physical_spacing"] == {"x": None, "y": None, "z": None}
    assert summary["measurement_policy"] == "pixel-only"


def test_dataset_summary_rejects_nonfinite_native_extent_before_preview_planes(tmp_path):
    path = str(tmp_path / "native-extent-overflow.dream3d")
    native_zyx = (2, 2, 2)
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2], dtype="u8"))
        geometry.create_dataset(
            "SPACING", data=np.array([np.finfo("f8").max, 0.5, 0.5], dtype="f8")
        )
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

    summary = hdf5.dataset_summary(
        path, "/DataContainers/Image/CellData/FeatureIds"
    )
    assert summary["geometry"]["spacing"][0] == np.finfo("f8").max
    assert summary["axis_sizes"] == {"T": 1, "C": 1, "Z": 2, "Y": 2, "X": 2}
    assert summary["physical_spacing"] == {"x": None, "y": None, "z": None}
    assert summary["measurement_policy"] == "pixel-only"
    for plane in summary["preview_planes"].values():
        assert plane["world_size"] == {"width": 2.0, "height": 2.0}
        assert all(np.isfinite(value) for value in plane["world_size"].values())
    _assert_json(summary)


def test_dataset_summary_rejects_cross_container_hard_link_geometry_alias(tmp_path):
    path = str(tmp_path / "ambiguous-geometry-alias.dream3d")
    native_zyx = (2, 3, 4)
    with h5py.File(path, "w") as file:
        containers = file.create_group("DataContainers")
        image_a = containers.create_group("ImageA")
        geometry_a = image_a.create_group("_SIMPL_GEOMETRY")
        geometry_a.create_dataset("DIMENSIONS", data=np.array([4, 3, 2], dtype="u8"))
        geometry_a.create_dataset("SPACING", data=np.ones(3, dtype="f8"))
        geometry_a.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell_a = image_a.create_group("CellData")
        feature_ids = cell_a.create_dataset(
            "FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4")
        )

        image_b = containers.create_group("ImageB")
        geometry_b = image_b.create_group("_SIMPL_GEOMETRY")
        geometry_b.create_dataset("DIMENSIONS", data=np.array([4, 3, 2], dtype="u8"))
        geometry_b.create_dataset("SPACING", data=np.full(3, 2.0, dtype="f8"))
        geometry_b.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell_b = image_b.create_group("CellData")
        cell_b["FeatureIds"] = feature_ids

    for container_name in ("ImageA", "ImageB"):
        summary = hdf5.dataset_summary(
            path, f"/DataContainers/{container_name}/CellData/FeatureIds"
        )
        assert summary["geometry"] is None
        assert summary["physical_spacing"] == {"x": None, "y": None, "z": None}
        assert summary["measurement_policy"] == "pixel-only"


def test_dataset_summary_rejects_geometry_group_shared_across_containers(tmp_path):
    path = str(tmp_path / "shared-geometry-group.dream3d")
    native_zyx = (2, 3, 4)
    with h5py.File(path, "w") as file:
        containers = file.create_group("DataContainers")
        image_a = containers.create_group("ImageA")
        geometry = image_a.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([4, 3, 2], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.array([0.5, 0.75, 1.25], dtype="f8"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f8"))
        cell_a = image_a.create_group("CellData")
        cell_a.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

        image_b = containers.create_group("ImageB")
        image_b["_SIMPL_GEOMETRY"] = geometry
        cell_b = image_b.create_group("CellData")
        cell_b.create_dataset("FeatureIds", data=np.ones((*native_zyx, 1), dtype="u4"))

    for container_name in ("ImageA", "ImageB"):
        summary = hdf5.dataset_summary(
            path, f"/DataContainers/{container_name}/CellData/FeatureIds"
        )
        assert summary["geometry"] is None
        assert summary["physical_spacing"] == {"x": None, "y": None, "z": None}
        assert summary["measurement_policy"] == "pixel-only"


def test_preview_stride_selection_is_algebraic_for_huge_sparse_shapes():
    shape = (10**12, 10**12, 10**12)
    volume = hdf5._interpret_volume(shape, np.dtype("u1"), exact_zyx=shape)
    assert volume is not None
    assert volume["pz"] * volume["py"] * volume["px"] <= hdf5.PREVIEW_MAX_VOXELS
    assert min(volume["sz"], volume["sy"], volume["sx"]) > 1


def test_delivery_grid_is_bounded_to_256_per_axis_for_deep_wide_metadata():
    shape = (1000, 300, 513)
    volume = hdf5._interpret_volume(shape, np.dtype("u4"), exact_zyx=shape)
    assert volume is not None
    scheme = hdf5._atlas_scheme(volume["px"], volume["py"], volume["pz"])
    assert volume["sz"] >= 4
    assert scheme["slice_count"] <= hdf5.ATLAS_CELL_CAP
    assert scheme["slice_height"] <= hdf5.ATLAS_CELL_CAP
    assert scheme["slice_width"] <= hdf5.ATLAS_CELL_CAP
    assert (
        scheme["slice_count"] + scheme["slice_height"] + scheme["slice_width"]
        <= 3 * hdf5.ATLAS_CELL_CAP
    )
    assert (
        scheme["slice_count"]
        * scheme["slice_height"]
        * scheme["slice_width"]
        * np.dtype("u4").itemsize
        <= 64 * 1024 * 1024
    )


def test_deep_feature_filter_downsamples_z_and_stays_aligned(tmp_path):
    from PIL import Image

    path = str(tmp_path / "deep-feature-ids.dream3d")
    native_depth = 1000
    identities = np.arange(1, native_depth + 1, dtype="u4").reshape(native_depth, 1, 1, 1)
    colors = np.repeat((identities % 251).astype("u1"), 3, axis=-1)
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([1, 1, native_depth], dtype="u4"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=identities, chunks=(25, 1, 1, 1))
        cell.create_dataset("IPFColor", data=colors, chunks=(25, 1, 1, 3))

    summary = hdf5.dataset_summary(path, "/DataContainers/Image/CellData/IPFColor")
    registration = summary["feature_filter"]
    assert registration["preview_shape"][0] <= hdf5.ATLAS_CELL_CAP
    assert registration["preview_stride"]["z"] == 4
    assert summary["atlas_scheme"]["slice_count"] == registration["preview_shape"][0]

    rgba = np.asarray(
        Image.open(
            io.BytesIO(
                hdf5.atlas_png(
                    path,
                    "/DataContainers/Image/CellData/IPFColor",
                    feature_ids="1,997",
                )
            )
        ).convert("RGBA"),
        dtype="u1",
    )
    assert int(np.count_nonzero(rgba[..., 3] == 255)) == 2
    assert np.all(rgba[rgba[..., 3] == 0] == 0)


def test_nonintegral_nearest_grid_aligns_label_atlas_and_filter(tmp_path, monkeypatch):
    from PIL import Image

    path = str(tmp_path / "nearest-14-to-5.dream3d")
    ids = np.arange(1, 15, dtype="u4").reshape(1, 1, 14, 1)
    colors = np.repeat(ids.astype("u1"), 3, axis=-1)
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([14, 1, 1], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=ids)
        cell.create_dataset("IPFColor", data=colors)

    real_atlas_scheme = hdf5._atlas_scheme
    monkeypatch.setattr(
        hdf5,
        "_atlas_scheme",
        lambda px, py, pz, **_kwargs: real_atlas_scheme(px, py, pz, cell_cap=5),
    )
    delivered_ids = hdf5._resize_nearest(ids[0, :, :, 0], 1, 5)

    label_atlas = np.asarray(
        Image.open(io.BytesIO(hdf5.atlas_png(path, "/DataContainers/Image/CellData/FeatureIds"))).convert("RGB")
    )
    assert np.array_equal(label_atlas[0, :5], hdf5._label_to_rgb(delivered_ids[0]))

    filtered = np.asarray(
        Image.open(
            io.BytesIO(
                hdf5.atlas_png(
                    path,
                    "/DataContainers/Image/CellData/IPFColor",
                    feature_ids=",".join(str(value) for value in range(1, 15)),
                )
            )
        ).convert("RGBA")
    )
    assert np.array_equal(filtered[0, :5, 0], delivered_ids[0].astype("u1"))
    assert np.all(filtered[0, :5, 3] == 255)


def test_atlas_accepts_noop_extras(general_h5):
    # The URL builder supports enhancement/fusion_method/negative/channels; the reader
    # must accept + ignore them.
    png = hdf5.atlas_png(general_h5, "/volume/ct", enhancement="d", negative="false", channels="0")
    assert png.startswith(_PNG_MAGIC)


def test_scalar_volume_headers_and_size(general_h5):
    s = hdf5.dataset_summary(general_h5, "/volume/ct")
    vol = hdf5.scalar_volume(general_h5, "/volume/ct")
    # dims agree with the summary axis_sizes / preview grid.
    assert vol["width"] == s["axis_sizes"]["X"]
    assert vol["height"] == s["axis_sizes"]["Y"]
    assert vol["depth"] == s["axis_sizes"]["Z"]
    assert vol["dtype"] == "float32" and vol["bytes_per_voxel"] == 4
    assert len(vol["data"]) == vol["width"] * vol["height"] * vol["depth"] * 4
    assert vol["raw_max"] > vol["raw_min"]
    assert (vol["source_width"], vol["source_height"], vol["source_depth"]) == (
        50,
        40,
        20,
    )
    assert (vol["downsample_x"], vol["downsample_y"], vol["downsample_z"]) == (1, 1, 1)
    assert vol["t"] == 0 and vol["preview_policy"] == "stride-v1"


@pytest.mark.parametrize("channel", [-1, 1, 1.5, True])
def test_scalar_volume_rejects_nonexact_or_out_of_range_channel(general_h5, channel):
    with pytest.raises((ValueError, hdf5.Hdf5Error), match="channel|component|integer|range"):
        hdf5.scalar_volume(general_h5, "/volume/ct", channel=channel)


def test_slice_and_scalar_volume_not_found(general_h5):
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.slice_png(general_h5, "/missing")
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.scalar_volume(general_h5, "/missing")


def test_non_numeric_slice_is_unsupported(general_h5):
    # A string dataset cannot be rendered; the message carries the 422-mapped marker.
    with pytest.raises(hdf5.Hdf5Error) as exc:
        hdf5.slice_png(general_h5, "/strings")
    assert "unsupported" in str(exc.value)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def test_histogram_continuous(general_h5):
    h = hdf5.dataset_histogram(general_h5, "/volume/ct", bins=24, file_id="fid1")
    _assert_json(h)
    assert h["discrete"] is False
    assert len(h["bins"]) == 24
    assert h["sample_count"] > 0
    assert h["min"] is not None and h["max"] is not None
    for b in h["bins"]:
        assert set(b) >= {"label", "start", "end", "count"}
        assert isinstance(b["count"], int)


def test_histogram_discrete_labels(general_h5):
    h = hdf5.dataset_histogram(general_h5, "/volume/labels", bins=24)
    assert h["discrete"] is True
    # one bin per distinct label (0..4).
    assert 1 <= len(h["bins"]) <= 5
    assert sum(b["count"] for b in h["bins"]) == h["sample_count"]


def test_histogram_bins_bounds(general_h5):
    # bins is clamped to [8, 256].
    assert len(hdf5.dataset_histogram(general_h5, "/volume/ct", bins=2)["bins"]) == 8
    assert len(hdf5.dataset_histogram(general_h5, "/volume/ct", bins=99999)["bins"]) == 256


def test_histogram_vector_component(general_h5):
    h = hdf5.dataset_histogram(general_h5, "/volume/euler", component=1, bins=16)
    assert h["component_index"] == 1
    assert h["component_label"] == "Phi"


# ---------------------------------------------------------------------------
# Table preview
# ---------------------------------------------------------------------------
def test_table_compound_pagination_and_charts(general_h5):
    t = hdf5.table_preview(general_h5, "/table", offset=0, limit=5, file_id="fid1")
    _assert_json(t)
    assert t["total_rows"] == 50
    assert t["offset"] == 0 and t["limit"] == 5
    assert len(t["rows"]) == 5
    assert [c["key"] for c in t["columns"]] == ["id", "size", "name"]
    # each row is keyed by column key + carries row_index.
    assert t["rows"][0]["id"] == 0
    assert t["rows"][0]["row_index"] == 0
    # numeric flag is honest.
    numeric = {c["key"]: c["numeric"] for c in t["columns"]}
    assert numeric["id"] and numeric["size"] and not numeric["name"]
    # charts follow the recharts contract (numeric y_key; scatter uses y_key "value").
    for chart in t["charts"]:
        assert chart["kind"] in ("scatter", "histogram")
        for row in chart["data"]:
            assert chart["x_key"] in row and chart["y_key"] in row
        if chart["kind"] == "scatter":
            assert chart["y_key"] == "value"


def test_table_pagination_offset_echo(general_h5):
    t = hdf5.table_preview(general_h5, "/table", offset=48, limit=12)
    assert t["offset"] == 48
    assert len(t["rows"]) == 2  # only 50 rows total
    assert t["rows"][0]["row_index"] == 48


def test_table_series_single_column(general_h5):
    t = hdf5.table_preview(general_h5, "/series", offset=10, limit=5)
    assert t["preview_kind"] == "series"
    assert [c["key"] for c in t["columns"]] == ["value"]
    assert t["total_rows"] == 200
    assert len(t["rows"]) == 5
    assert t["rows"][0]["row_index"] == 10


def test_table_string_dataset(general_h5):
    t = hdf5.table_preview(general_h5, "/strings", offset=0, limit=5)
    assert [c["key"] for c in t["columns"]] == ["value"]
    assert [r["value"] for r in t["rows"]] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Materials (DREAM.3D)
# ---------------------------------------------------------------------------
def test_dream3d_viewerinfo_detects_materials(dream3d_h5):
    vi = hdf5.build_hdf5_viewer_info(dream3d_h5, file_id="fid2")
    _assert_json(vi)
    assert vi["modality"] == "materials"
    materials = vi["hdf5"]["materials"]
    assert materials is not None
    assert materials["detected"] is True
    assert materials["schema"] == "dream3d"
    assert materials["recommended_view"] == "materials"
    assert "grain_metrics" in materials["capabilities"]
    assert "orientation" in materials["capabilities"]
    assert materials["phase_names"] == ["Nickel"]  # "Invalid Phase" filtered out
    # default points at the recommended map (IPF colors).
    assert vi["hdf5"]["default_dataset_path"].endswith("/IPFColors")


def test_dream3d_dashboard(dream3d_h5):
    d = hdf5.materials_dashboard(dream3d_h5, file_id="fid2")
    _assert_json(d)
    assert d["schema"] == "dream3d"
    assert d["file_id"] == "fid2"
    # maps are the spatial CellData volumes only (no per-feature arrays leaked in).
    map_names = {m["title"] for m in d["maps"]}
    assert "IPFColors" in map_names and "FeatureIds" in map_names
    assert "AvgEulerAngles" not in map_names
    for m in d["maps"]:
        assert m["dataset_path"].startswith("/DataContainers/")
        assert m["semantic_role"]
    assert d["overview"]["recommended_map_dataset_path"].endswith("/IPFColors")
    assert d["overview"]["feature_count"] == 29  # stored rows minus reserved feature zero
    assert d["overview"]["geometry"]["spacing"] == [0.5, 0.5, 0.5]
    # grain + orientation charts obey the chart-data contract.
    for chart in d["grain_charts"]:
        assert chart["kind"] in ("bar", "histogram", "scatter")
        assert chart["source_paths"]
        for row in chart["data"]:
            assert chart["x_key"] in row and chart["y_key"] in row
    for chart in d["orientation_charts"]:
        assert chart["kind"] == "scatter"
        assert chart["y_key"] == "value"
        for row in chart["data"]:
            assert chart["x_key"] in row and "value" in row


def test_dream3d_real_layout_selects_complete_image_geometry(dream3d_real_layout_h5):
    d = hdf5.materials_dashboard(dream3d_real_layout_h5)

    geometry = d["overview"]["geometry"]
    assert geometry["path"].endswith("/SyntheticVolumeDataContainer/_SIMPL_GEOMETRY")
    assert geometry["dimensions"] == [20, 25, 10]
    assert geometry["spacing"] == [0.5, 0.5, 0.5]
    assert geometry["origin"] == [1.0, 2.0, 3.0]


def test_dream3d_geometry_ranking_requires_cell_shape_consistency(tmp_path):
    path = str(tmp_path / "geometry-shape-consistency.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        containers = f.create_group("DataContainers")

        misleading = containers.create_group("MisleadingLargeContainer")
        misleading_geometry = misleading.create_group("_SIMPL_GEOMETRY")
        misleading_geometry.create_dataset("DIMENSIONS", data=np.array([100, 100, 100]))
        misleading_geometry.create_dataset("SPACING", data=np.ones(3))
        misleading_geometry.create_dataset("ORIGIN", data=np.zeros(3))
        misleading_cell = misleading.create_group("CellData")
        misleading_cell.create_dataset("FeatureIds", data=np.zeros((2, 2, 2, 1), dtype="i4"))

        image = containers.create_group("ActualImageContainer")
        image_geometry = image.create_group("_SIMPL_GEOMETRY")
        image_geometry.create_dataset("DIMENSIONS", data=np.array([4, 3, 2]))
        image_geometry.create_dataset("SPACING", data=np.array([0.5, 0.5, 1.0]))
        image_geometry.create_dataset("ORIGIN", data=np.zeros(3))
        image_cell = image.create_group("CellData")
        image_cell.create_dataset("FeatureIds", data=np.zeros((2, 3, 4, 1), dtype="i4"))

    geometry = hdf5.materials_dashboard(path)["overview"]["geometry"]
    assert geometry["path"].endswith("/ActualImageContainer/_SIMPL_GEOMETRY")
    assert geometry["cell_data_consistent"] is True
    assert geometry["complete"] is True


def test_dream3d_phase_names_are_scoped_to_selected_geometry_container(tmp_path):
    path = str(tmp_path / "phase-scope.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        containers = f.create_group("DataContainers")

        stale = containers.create_group("StatsGeneratorDataContainer")
        stale.create_group("_SIMPL_GEOMETRY")
        stale_ensemble = stale.create_group("CellEnsembleData")
        stale_ensemble.create_dataset(
            "PhaseName", data=np.array([b"Unknown Phase", b"Stale phase"], dtype="S16")
        )

        image = containers.create_group("SyntheticVolumeDataContainer")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([4, 3, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.zeros((2, 3, 4, 1), dtype="i4"))
        ensemble = image.create_group("CellEnsembleData")
        ensemble.create_dataset(
            "PhaseName", data=np.array([b"Unknown Phase", b"Selected phase"], dtype="S20")
        )

    overview = hdf5.materials_dashboard(path)["overview"]
    assert overview["phase_names"] == ["Selected phase"]


def test_untrusted_links_are_never_dereferenced(tmp_path):
    outside = str(tmp_path / "outside-secret.h5")
    source = str(tmp_path / "linked.dream3d")
    with h5py.File(outside, "w") as f:
        f.create_dataset("secret", data=np.array([b"OUTSIDE-SECRET"], dtype="S32"))
        f.create_dataset("PhaseName", data=np.array([b"OUTSIDE-PHASE"], dtype="S32"))
        f.create_dataset("FeatureIds", data=np.ones((2, 2, 2, 1), dtype="i4"))

    with h5py.File(source, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        intensity = cell.create_dataset("Intensity", data=np.ones((2, 2, 2, 1), dtype="f4"))
        cell["SoftFeatureIds"] = h5py.SoftLink(intensity.name)
        cell["FeatureIds"] = h5py.ExternalLink(outside, "/FeatureIds")
        cell["ExternalSecret"] = h5py.ExternalLink(outside, "/secret")
        feature = image.create_group("CellFeatureData")
        feature.create_dataset("Volumes", data=np.array([[0.0], [1.0]], dtype="f4"))
        ensemble = image.create_group("CellEnsembleData")
        ensemble["PhaseName"] = h5py.ExternalLink(outside, "/PhaseName")
        f["SoftSecret"] = h5py.SoftLink("/DataContainers/Image/CellData/ExternalSecret")
        f["ExternalSecret"] = h5py.ExternalLink(outside, "/secret")

    viewer = hdf5.build_hdf5_viewer_info(source)
    dashboard = hdf5.materials_dashboard(source)
    encoded = json.dumps(viewer) + json.dumps(dashboard)
    assert "OUTSIDE-SECRET" not in encoded
    assert "OUTSIDE-PHASE" not in encoded
    assert dashboard["overview"]["phase_names"] == []
    assert dashboard["overview"]["feature_id_scan_complete"] is False
    assert dashboard["overview"]["referenced_positive_feature_count"] is None
    assert dashboard["overview"]["grain_count"] is None
    assert dashboard["overview"]["feature_zero_reserved"] is False

    def _tree_paths(nodes):
        for node in nodes:
            yield node["path"]
            yield from _tree_paths(node["children"])

    paths = set(_tree_paths(viewer["hdf5"]["tree"]))
    # External links, and soft links that CHAIN to an external link, are never
    # dereferenced (the outside file is never opened).
    assert "/ExternalSecret" not in paths
    assert "/SoftSecret" not in paths  # soft -> external chain, rejected at the external hop
    assert "/DataContainers/Image/CellData/FeatureIds" not in paths  # external link

    for linked_path in (
        "/ExternalSecret",
        "/SoftSecret",
        "/DataContainers/Image/CellData/ExternalSecret",
        "/DataContainers/Image/CellData/FeatureIds",
    ):
        with pytest.raises(hdf5.Hdf5DatasetNotFound):
            hdf5.dataset_summary(source, linked_path)

    # An INTERNAL soft link (NeXus-style, staying inside this file) resolves to its
    # in-file target — it is not treated as untrusted. No outside file is touched.
    resolved = hdf5.dataset_summary(source, "/DataContainers/Image/CellData/SoftFeatureIds")
    assert resolved is not None
    assert resolved["geometry"] is None
    assert resolved["physical_spacing"] == {"x": None, "y": None, "z": None}
    assert resolved["measurement_policy"] == "pixel-only"
    assert "OUTSIDE-SECRET" not in json.dumps(resolved)


def test_phase_metadata_requires_string_ensemble_data_and_is_byte_bounded(tmp_path):
    path = str(tmp_path / "bounded-phases.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((2, 2, 2, 1), dtype="i4"))

        # A matching basename outside a typed ensemble group is not phase metadata.
        image.create_dataset("PhaseName", data=np.array([b"UNSCOPED"], dtype="S16"))
        ensemble = image.create_group("CellEnsembleData")
        ensemble.create_dataset("MaterialName", data=np.array([1, 2, 3], dtype="i8"))
        ensemble.create_dataset(
            "PhaseName",
            data=np.array(
                [
                    b"Invalid Phase",
                    b"Nickel",
                    b"X" * (hdf5.MATERIAL_PHASE_MAX_ITEM_BYTES + 1),
                ],
                dtype=f"S{hdf5.MATERIAL_PHASE_MAX_ITEM_BYTES + 1}",
            ),
        )
        vlen = h5py.string_dtype(encoding="utf-8")
        ensemble.create_dataset(
            "PhaseNames",
            data=np.array(["Copper", "Y" * (hdf5.MATERIAL_PHASE_MAX_ITEM_BYTES + 1)], dtype=object),
            dtype=vlen,
        )
        scalar_ensemble = image.create_group("EnsembleData")
        scalar_ensemble.create_dataset("MaterialName", data=np.bytes_("Nickel"))

    overview = hdf5.materials_dashboard(path)["overview"]
    assert overview["phase_names"] == ["Copper", "Nickel"]
    assert sum(len(name.encode("utf-8")) for name in overview["phase_names"]) <= (
        hdf5.MATERIAL_PHASE_MAX_TOTAL_BYTES
    )


def test_geometry_vectors_reject_nonnumeric_or_oversized_items(tmp_path):
    path = str(tmp_path / "geometry-metadata-bounds.dream3d")
    huge_float = np.dtype("V4096")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        containers = f.create_group("DataContainers")
        malformed = containers.create_group("Malformed")
        malformed_geometry = malformed.create_group("_SIMPL_GEOMETRY")
        malformed_geometry.create_dataset(
            "DIMENSIONS", data=np.array([b"2", b"2", b"2"], dtype="S2")
        )
        malformed_geometry.create_dataset("SPACING", shape=(3,), dtype=huge_float)
        malformed_geometry.create_dataset("ORIGIN", data=np.zeros(3))

        image = containers.create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        image.create_group("CellData").create_dataset(
            "FeatureIds", data=np.ones((2, 2, 2, 1), dtype="i4")
        )

    selected = hdf5.materials_dashboard(path)["overview"]["geometry"]
    assert selected["path"].endswith("/Image/_SIMPL_GEOMETRY")


def test_orientation_capability_requires_an_orientation_array(tmp_path):
    path = str(tmp_path / "no-orientation.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.ones((2, 2, 2, 1), dtype="i4"))
        feature = image.create_group("CellFeatureData")
        feature.create_dataset("Volumes", data=np.array([[0.0], [8.0]], dtype="f4"))

    dashboard = hdf5.materials_dashboard(path)
    assert "grain_metrics" in dashboard["overview"]["capabilities"]
    assert "orientation" not in dashboard["overview"]["capabilities"]
    assert dashboard["orientation_charts"] == []


def test_grain_count_requires_complete_consistent_feature_id_relationship(tmp_path):
    path = str(tmp_path / "relational-counts.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        image.create_group("CellData").create_dataset(
            "FeatureIds", data=np.zeros((2, 2, 2, 1), dtype="i4")
        )
        feature = image.create_group("CellFeatureData")
        feature.create_dataset("Volumes", data=np.array([[0.0], [1.0], [2.0], [3.0]]))

    overview = hdf5.materials_dashboard(path)["overview"]
    assert overview["declared_feature_tuple_count"] == 3
    assert overview["referenced_positive_feature_count"] == 0
    assert overview["feature_id_scan_complete"] is True
    assert overview["feature_id_consistency"] is False
    assert overview["feature_count"] == 3
    assert overview["grain_count"] is None


def test_multiple_feature_groups_are_selected_by_feature_id_relationship(tmp_path):
    path = str(tmp_path / "multiple-feature-groups.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        ids = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype="i4").reshape(2, 2, 2, 1)
        image.create_group("CellData").create_dataset("FeatureIds", data=ids)

        wrong = image.create_group("CellFeatureData")
        wrong.create_dataset("Volumes", data=np.arange(6, dtype="f4")[:, None])
        wrong.create_dataset("NumNeighbors", data=np.arange(6, dtype="i4")[:, None])
        right = image.create_group("Grain Data")
        right.create_dataset("Volumes", data=np.arange(4, dtype="f4")[:, None])

    dashboard = hdf5.materials_dashboard(path)
    overview = dashboard["overview"]
    assert overview["declared_feature_tuple_count"] == 3
    assert overview["referenced_positive_feature_count"] == 3
    assert overview["feature_id_scan_complete"] is True
    assert overview["feature_id_consistency"] is True
    assert overview["grain_count"] == 3
    assert dashboard["grain_charts"]
    assert all("/Grain Data/" in chart["source_paths"][0] for chart in dashboard["grain_charts"])


def test_feature_id_scan_is_complete_across_bounded_hyperslabs(tmp_path, monkeypatch):
    path = str(tmp_path / "chunked-feature-ids.dream3d")
    monkeypatch.setattr(hdf5, "FEATURE_ID_SCAN_CHUNK_VALUES", 2)
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        ids = np.array([0, 1, 2, 3, 1, 2, 3, 0], dtype="i4").reshape(2, 2, 2, 1)
        image.create_group("CellData").create_dataset("FeatureIds", data=ids)
        image.create_group("CellFeatureData").create_dataset(
            "Volumes", data=np.arange(4, dtype="f4")[:, None]
        )

    overview = hdf5.materials_dashboard(path)["overview"]
    assert overview["feature_id_scan_complete"] is True
    assert overview["referenced_positive_feature_count"] == 3
    assert overview["feature_id_consistency"] is True
    assert overview["grain_count"] == 3


def test_feature_id_identity_cap_fails_closed(tmp_path, monkeypatch):
    path = str(tmp_path / "identity-cap.dream3d")
    monkeypatch.setattr(hdf5, "FEATURE_ID_MAX_TRACKED_IDENTITIES", 2)
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        ids = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype="i4").reshape(2, 2, 2, 1)
        image.create_group("CellData").create_dataset("FeatureIds", data=ids)
        image.create_group("CellFeatureData").create_dataset(
            "Volumes", data=np.arange(4, dtype="f4")[:, None]
        )

    overview = hdf5.materials_dashboard(path)["overview"]
    assert overview["declared_feature_tuple_count"] == 3
    assert overview["feature_id_scan_complete"] is False
    assert overview["referenced_positive_feature_count"] is None
    assert overview["feature_id_consistency"] is None
    assert overview["grain_count"] is None


def test_dream3d_units_are_metadata_backed_not_inferred_from_dataset_name(tmp_path):
    path = str(tmp_path / "units-provenance.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        cell.create_dataset("FeatureIds", data=np.zeros((2, 2, 2, 1), dtype="i4"))
        grain = image.create_group("Grain Data")
        grain.create_dataset("EquivalentDiameters", data=np.array([[0.0], [1.0], [2.0]]))
        eulers = grain.create_dataset(
            "EulerAngles", data=np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        )
        eulers.attrs["Units"] = "degrees"

    diameter = hdf5.dataset_summary(
        path, "/DataContainers/Image/Grain Data/EquivalentDiameters"
    )
    eulers = hdf5.dataset_summary(path, "/DataContainers/Image/Grain Data/EulerAngles")
    charts = hdf5.materials_dashboard(path)["orientation_charts"]
    assert diameter["units_hint"] is None
    assert eulers["units_hint"] == "degrees"
    assert charts[0]["units_hint"] == "degrees"


def test_dream3d_feature_row_zero_is_not_removed_without_schema_evidence(tmp_path):
    path = str(tmp_path / "no-reserved-feature-zero.dream3d")
    with h5py.File(path, "w") as f:
        f.attrs["FileVersion"] = "7.0"
        image = f.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 2]))
        geometry.create_dataset("SPACING", data=np.ones(3))
        geometry.create_dataset("ORIGIN", data=np.zeros(3))
        cell = image.create_group("CellData")
        # No FeatureIds/GrainIds array and no explicit sentinel attribute.
        cell.create_dataset("Intensity", data=np.ones((2, 2, 2, 1), dtype="f4"))
        grain = image.create_group("Grain Data")
        grain.create_dataset("Volumes", data=np.array([[1.0], [2.0], [3.0]], dtype="f4"))

    dashboard = hdf5.materials_dashboard(path)
    assert dashboard["overview"]["declared_feature_tuple_count"] == 3
    assert dashboard["overview"]["referenced_positive_feature_count"] is None
    assert dashboard["overview"]["feature_id_scan_complete"] is False
    assert dashboard["overview"]["feature_id_consistency"] is None
    assert dashboard["overview"]["grain_count"] is None
    chart = dashboard["grain_charts"][0]
    assert sum(row["count"] for row in chart["data"]) == 3
    assert "no reserved feature row" in chart["provenance"].lower()


def test_dream3d_real_layout_recognizes_grain_data_and_reserved_row(dream3d_real_layout_h5):
    d = hdf5.materials_dashboard(dream3d_real_layout_h5)

    assert d["overview"]["feature_count"] == 5_000
    assert d["overview"]["grain_count"] == 5_000
    assert "grain_metrics" in d["overview"]["capabilities"]

    volume_chart = next(
        chart
        for chart in d["grain_charts"]
        if chart["source_paths"][0].endswith("/Grain Data/Volumes")
    )
    sampled = int(volume_chart["description"].split()[0])
    assert 1 < sampled <= hdf5.TABLE_CHART_MAX_ROWS
    assert sum(row["count"] for row in volume_chart["data"]) == sampled
    # The reserved first row is excluded by position, but the legitimate zero in
    # row one remains in the distribution as a singleton histogram bin.
    assert any(row["count"] == 1 for row in volume_chart["data"])
    assert "reserved" in volume_chart["provenance"].lower()


def test_dream3d_real_layout_emits_bounded_orientation_chart(dream3d_real_layout_h5):
    d = hdf5.materials_dashboard(dream3d_real_layout_h5)

    chart = next(
        chart
        for chart in d["orientation_charts"]
        if chart["source_paths"][0].endswith("/Grain Data/EulerAngles")
    )
    assert len(chart["data"]) <= 2_000
    # Real feature row one is a valid zero orientation; only sentinel row zero drops.
    assert chart["data"][0] == {"phi1": 0.0, "value": 0.0}
    assert "reserved" in chart["provenance"].lower()


def test_dream3d_phase_names_are_deduplicated_stored_metadata(dream3d_real_layout_h5):
    vi = hdf5.build_hdf5_viewer_info(dream3d_real_layout_h5)
    materials = vi["hdf5"]["materials"]
    assert materials["phase_names"] == ["Primary"]
    assert materials["phase_names_source"] == "stored_metadata"
    assert "no phase-identification algorithm" in materials["phase_names_provenance"].lower()

    overview = hdf5.materials_dashboard(dream3d_real_layout_h5)["overview"]
    assert overview["phase_names"] == ["Primary"]
    assert overview["phase_names_source"] == "stored_metadata"


def test_dashboard_on_non_dream3d_raises(general_h5):
    with pytest.raises(hdf5.Hdf5DatasetNotFound):
        hdf5.materials_dashboard(general_h5)


def test_dream3d_map_summary_and_slice(dream3d_h5):
    # The recommended IPF map summarizes + slices as an rgb volume.
    dp = "/DataContainers/SyntheticVolume/CellData/IPFColors"
    s = hdf5.dataset_summary(dream3d_h5, dp)
    assert s["preview_kind"] == "rgb_volume"
    assert s["geometry"] is not None
    assert s["physical_spacing"]["x"] == 0.5
    assert s["measurement_policy"] == "spacing-aware"
    assert hdf5.slice_png(dream3d_h5, dp, axis="z", index=5).startswith(_PNG_MAGIC)
