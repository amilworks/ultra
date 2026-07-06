"""Service tests using the in-process stub engine (no native lib, no subprocesses)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402
from ultra_deepagents.imaging.pool import InlineRunner  # noqa: E402
from ultra_deepagents.imaging.service import create_app  # noqa: E402


@pytest.fixture()
def client():
    runner = InlineRunner(prefer_real=False)  # deterministic stub
    app = create_app(runner=runner)
    with TestClient(app) as c:
        yield c
    runner.shutdown()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tile_returns_png(client):
    r = client.get("/tile", params={"path": "a.czi", "level": 0, "col": 1, "row": 2, "size": 64})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_slice_full_resolution_param_is_threaded_through(client):
    # full_resolution selects native vs a bounded scrub level; the stub seeds its
    # output by it, so the two responses differ — proving the param reaches the engine.
    r = client.get("/slice", params={"path": "a.czi", "z": 2, "full_resolution": "true"})
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG\r\n\x1a\n")
    settled = client.get("/slice", params={"path": "a.czi", "z": 2, "full_resolution": "true"}).content
    scrub = client.get("/slice", params={"path": "a.czi", "z": 2, "full_resolution": "false"}).content
    assert settled != scrub


def test_thumbnail_zscrub_frames_differ_by_z(client):
    a = client.get("/thumbnail", params={"path": "a.czi", "max_size": 48, "z": 1}).content
    b = client.get("/thumbnail", params={"path": "a.czi", "max_size": 48, "z": 2}).content
    assert a != b  # scrubbing z yields distinct frames


def test_meta_and_formats(client):
    assert client.get("/meta", params={"path": "a.czi"}).json()["image_num_z"] >= 1
    assert "czi" in client.get("/formats").json()["formats"]


def test_histogram(client):
    r = client.get("/histogram", params={"path": "a.czi", "bins": 16})
    assert r.json()["bins"] == 16


def test_viewerinfo_has_tile_scheme(client):
    r = client.get("/viewerinfo", params={"path": "a.czi"})
    assert r.status_code == 200
    vi = r.json()
    assert vi["axis_sizes"]["X"] == 2048
    assert vi["reader"] == "libbioimage"
    assert vi["tile_scheme"] is not None and len(vi["tile_scheme"]["levels"]) == 4


def test_scalar_volume_octet_stream_and_headers(client):
    r = client.get("/scalar-volume", params={"path": "a.nii", "channel": 0})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-volume-dtype"] == "float32"
    assert int(r.headers["x-volume-bytes-per-voxel"]) == 4
    w = int(r.headers["x-volume-width"]); h = int(r.headers["x-volume-height"]); d = int(r.headers["x-volume-depth"])
    assert d >= 1
    assert len(r.content) == w * h * d * 4


class _RaisingRunner:
    """A runner whose every op raises a chosen exception, for error-mapping tests."""

    workers = 1

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def call(self, method, *args, **kwargs):
        raise self._exc

    def shutdown(self) -> None:
        pass


def test_undecodable_image_maps_to_422_not_500():
    # A malformed/unsupported file that the engine reads as a 0-sized region (after its
    # own transient-retry) must surface as 422 "preview unavailable", not a 500 that looks
    # like a server fault. Reproduced live on an undecodable .czi (libbioimage -> 0x0).
    app = create_app(runner=_RaisingRunner(ValueError("engine returned an empty region (shape (0, 0, 0))")))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/thumbnail", params={"path": "/bad.czi", "max_size": 512})
        assert r.status_code == 422
        assert "decod" in r.json()["error"].lower()


def test_unexpected_value_error_still_500():
    # A ValueError that is NOT a decode failure is a real bug and must keep the 500.
    app = create_app(runner=_RaisingRunner(ValueError("some unexpected internal failure")))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/thumbnail", params={"path": "/x.tif", "max_size": 512})
        assert r.status_code == 500


# --- HDF5 data viewer routes -------------------------------------------------
# Driven end-to-end through the StubEngine, whose hdf5_* ops read a REAL synthetic
# file with h5py (the reader is engine-independent). Gated on h5py.

h5py = pytest.importorskip("h5py")


@pytest.fixture()
def hdf5_path(tmp_path):
    import numpy as np

    path = str(tmp_path / "sample.h5")
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        vol = f.create_group("volume")
        vol.create_dataset("ct", data=(rng.random((12, 24, 30)) * 4000).astype("int16"))
        vol.create_dataset("labels", data=rng.integers(0, 4, (12, 24, 30)).astype("uint8"))
        f.create_dataset("series", data=rng.random(120).astype("float64"))
    return path


@pytest.fixture()
def dream3d_path(tmp_path):
    import numpy as np

    path = str(tmp_path / "synthetic.dream3d")
    rng = np.random.default_rng(8)
    with h5py.File(path, "w") as f:
        f.attrs["DREAM3D Version"] = "6.5.0"
        cell = f.create_group("DataContainers").create_group("Vol").create_group("CellData")
        cell.create_dataset("FeatureIds", data=rng.integers(0, 20, (10, 20, 25, 1)).astype("i4"))
        cell.create_dataset("IPFColors", data=rng.integers(0, 255, (10, 20, 25, 3)).astype("u1"))
    return path


def test_plain_viewerinfo_detects_hdf5_by_path_extension(client, hdf5_path):
    # The generic /viewerinfo endpoint forks to the hdf5 payload when the path (or the
    # optional name hint) is an HDF5-data file, so no image pipeline touches it.
    r = client.get("/viewerinfo", params={"path": hdf5_path})
    assert r.status_code == 200
    assert r.json()["kind"] == "hdf5"


def test_hdf5_dataset_route(client, hdf5_path):
    r = client.get("/hdf5/dataset", params={"path": hdf5_path, "dataset_path": "/volume/ct", "file_id": "fid"})
    assert r.status_code == 200
    s = r.json()
    assert s["preview_kind"] == "scalar_volume"
    assert s["dimension_summary"] == {"z": 12, "y": 24, "x": 30}


def test_hdf5_dataset_unknown_is_404(client, hdf5_path):
    r = client.get("/hdf5/dataset", params={"path": hdf5_path, "dataset_path": "/nope"})
    assert r.status_code == 404
    assert "error" in r.json()


def test_hdf5_slice_png(client, hdf5_path):
    r = client.get("/hdf5/preview/slice", params={"path": hdf5_path, "dataset_path": "/volume/ct", "axis": "z", "index": 3})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_hdf5_slice_bad_index_type_is_422(client, hdf5_path):
    # FastAPI query validation: a non-int index → 422 automatically.
    r = client.get("/hdf5/preview/slice", params={"path": hdf5_path, "dataset_path": "/volume/ct", "index": "abc"})
    assert r.status_code == 422


def test_hdf5_atlas_png(client, hdf5_path):
    r = client.get("/hdf5/preview/atlas", params={"path": hdf5_path, "dataset_path": "/volume/labels"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_hdf5_scalar_volume_headers(client, hdf5_path):
    r = client.get("/hdf5/preview/scalar-volume", params={"path": hdf5_path, "dataset_path": "/volume/ct"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-volume-dtype"] == "float32"
    w = int(r.headers["x-volume-width"]); h = int(r.headers["x-volume-height"]); d = int(r.headers["x-volume-depth"])
    assert (w, h, d) == (30, 24, 12)
    assert len(r.content) == w * h * d * 4
    assert float(r.headers["x-volume-raw-max"]) > float(r.headers["x-volume-raw-min"])


def test_hdf5_histogram_route(client, hdf5_path):
    r = client.get("/hdf5/preview/histogram", params={"path": hdf5_path, "dataset_path": "/volume/ct", "bins": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["discrete"] is False
    assert len(body["bins"]) == 24


def test_hdf5_table_route(client, hdf5_path):
    r = client.get("/hdf5/preview/table", params={"path": hdf5_path, "dataset_path": "/series", "offset": 0, "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["preview_kind"] == "series"
    assert body["offset"] == 0 and body["limit"] == 5
    assert len(body["rows"]) == 5


def test_hdf5_materials_dashboard_route(client, dream3d_path):
    r = client.get("/hdf5/materials/dashboard", params={"path": dream3d_path, "file_id": "fid"})
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == "dream3d"
    assert any(m["dataset_path"].endswith("/IPFColors") for m in body["maps"])


def test_hdf5_materials_dashboard_non_dream3d_is_404(client, hdf5_path):
    r = client.get("/hdf5/materials/dashboard", params={"path": hdf5_path})
    assert r.status_code == 404
