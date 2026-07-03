"""Unit tests for the h5py-backed HDF5 data viewer (imaging/hdf5.py).

Fixtures generate tiny synthetic ``.h5`` / ``.dream3d`` files in-test (no committed
binary fixtures), then exercise every reader entry point + edge cases + the
viewer-info ``kind:"hdf5"`` detection. Gated on h5py so a bare environment skips
rather than errors.
"""

from __future__ import annotations

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


def _assert_json(obj):
    """Every response is cast directly by the FE (no normalizer), so it must be strict
    JSON — no NaN/Inf/bytes/numpy scalars leaking through."""
    json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_is_hdf5_data_file_extension_gate():
    assert hdf5.is_hdf5_data_file("vol.h5")
    assert hdf5.is_hdf5_data_file("VOL.HDF5")
    assert hdf5.is_hdf5_data_file("micro.hdf")
    assert hdf5.is_hdf5_data_file("micro.dream3d")
    assert hdf5.is_hdf5_data_file("series.h5_3")  # page/series suffix tolerated
    # HDF5-based image/non-image formats must NOT be captured by the data explorer.
    assert not hdf5.is_hdf5_data_file("cells.ims")   # Imaris -> libbioimage
    assert not hdf5.is_hdf5_data_file("data.mat")    # MATLAB v7.3
    assert not hdf5.is_hdf5_data_file("scan.nc")     # NetCDF4
    assert not hdf5.is_hdf5_data_file("photo.tif")


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
    from PIL import Image
    import io as _io

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


def test_atlas_png_matches_scheme(general_h5):
    from PIL import Image
    import io as _io

    s = hdf5.dataset_summary(general_h5, "/volume/labels")
    scheme = s["atlas_scheme"]
    png = hdf5.atlas_png(general_h5, "/volume/labels")
    assert png.startswith(_PNG_MAGIC)
    img = Image.open(_io.BytesIO(png))
    assert (img.width, img.height) == (scheme["atlas_width"], scheme["atlas_height"])


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
    assert d["overview"]["feature_count"] == 30
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
