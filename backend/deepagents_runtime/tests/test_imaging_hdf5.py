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
