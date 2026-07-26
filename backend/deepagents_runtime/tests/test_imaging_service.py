"""Service tests using the in-process stub engine (no native lib, no subprocesses)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
np = pytest.importorskip("numpy")
from fastapi.testclient import TestClient  # noqa: E402
from ultra_deepagents.imaging import service as service_module  # noqa: E402
from ultra_deepagents.imaging.pool import InlineRunner  # noqa: E402
from ultra_deepagents.imaging.service import _scalar_volume_envelope, create_app  # noqa: E402


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


def test_warm_localized_pyramid_recency_does_not_mutate_source_generation(
    tmp_path,
    monkeypatch,
):
    derived = tmp_path / "derived"
    derived.mkdir()
    source = derived / "fixture__pyramid.tif"
    source.write_bytes(b"stable pyramid bytes")
    cache = tmp_path / "cache"
    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_ENABLED", True)
    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_DIR", str(cache))

    localized = service_module.localize_pyramid(str(source))
    assert os.path.basename(os.path.dirname(localized)) == "derived"
    assert os.path.basename(localized).endswith("__pyramid.tif")
    initial = os.stat(localized)
    generation_before = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    time.sleep(0.002)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        localized_paths = list(
            executor.map(
                service_module.localize_pyramid,
                [str(source)] * 8,
            )
        )
    final = os.stat(localized)
    generation_after = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    )

    assert localized_paths == [localized] * 8
    assert generation_after == generation_before


def test_pyramid_cache_eviction_covers_owned_subdir_and_legacy_root(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "cache"
    derived = cache / "derived"
    derived.mkdir(parents=True)
    legacy = cache / "legacy.tif"
    owned = derived / "owned__pyramid.tif"
    legacy.write_bytes(b"12")
    owned.write_bytes(b"34")
    service_module._touch_pyramid_access_marker(str(legacy))
    service_module._touch_pyramid_access_marker(str(owned))
    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_DIR", str(cache))
    monkeypatch.setenv("ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE_BYTES", "1")

    service_module._evict_pyramid_cache(incoming=1)

    assert not legacy.exists()
    assert not owned.exists()
    assert not os.path.exists(service_module._pyramid_access_marker(str(legacy)))
    assert not os.path.exists(service_module._pyramid_access_marker(str(owned)))


@pytest.mark.parametrize(
    "path",
    [
        "/cache/not-derived/sample__pyramid.tif",
        "/cache/derived/nested/sample__pyramid.tif",
        "/cache/derived/sample.tif",
        "/cache/derived/sample__pyramid.tiff",
    ],
)
def test_localize_pyramid_does_not_promote_ordinary_tiffs_to_owned_identity(path):
    assert service_module._is_derived_pyramid(path) is False
    assert service_module.localize_pyramid(path) == path


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
    settled = client.get(
        "/slice", params={"path": "a.czi", "z": 2, "full_resolution": "true"}
    ).content
    scrub = client.get(
        "/slice", params={"path": "a.czi", "z": 2, "full_resolution": "false"}
    ).content
    assert settled != scrub


def test_thumbnail_zscrub_frames_differ_by_z(client):
    a = client.get("/thumbnail", params={"path": "a.czi", "max_size": 48, "z": 1}).content
    b = client.get("/thumbnail", params={"path": "a.czi", "max_size": 48, "z": 2}).content
    assert a != b  # scrubbing z yields distinct frames


def test_meta_and_formats(client):
    assert client.get("/meta", params={"path": "a.czi"}).json()["image_num_z"] >= 1
    assert "czi" in client.get("/formats").json()["formats"]


def test_histogram(client):
    r = client.get(
        "/histogram",
        params={"path": "a.czi", "bins": 16, "channel": 0, "t": 0, "scope": "volume"},
    )
    assert r.json()["bins"] == 16


def test_display_histogram_preserves_composite_channels_with_common_edges(client):
    r = client.get(
        "/histogram",
        params={"path": "a.czi", "bins": 16, "channels": "0,2", "t": 1},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["scope"] == "display"
    assert payload["t"] == 1
    assert [entry["index"] for entry in payload["channels"]] == [0, 2]
    assert payload["channels"][0]["edges"] == payload["channels"][1]["edges"]
    assert all(sum(entry["counts"]) == entry["sample_count"] for entry in payload["channels"])


def test_viewerinfo_has_tile_scheme(client):
    r = client.get("/viewerinfo", params={"path": "a.czi"})
    assert r.status_code == 200
    vi = r.json()
    assert vi["axis_sizes"]["X"] == 2048
    assert vi["reader"] == "libbioimage"
    assert vi["tile_scheme"] is not None and len(vi["tile_scheme"]["levels"]) == 4


def test_scalar_volume_octet_stream_and_headers(client):
    r = client.get(
        "/scalar-volume",
        params={"path": "a.nii", "channel": 0, "sampling": "box"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-volume-dtype"] == "float32"
    assert int(r.headers["x-volume-bytes-per-voxel"]) == 4
    assert float(r.headers["x-volume-scl-slope"]) == 1.0
    assert float(r.headers["x-volume-scl-inter"]) == 0.0
    assert int(r.headers["x-volume-time"]) == 0
    assert r.headers["x-volume-sampling"] == "box"
    w = int(r.headers["x-volume-width"])
    h = int(r.headers["x-volume-height"])
    d = int(r.headers["x-volume-depth"])
    assert d >= 1
    assert len(r.content) == w * h * d * 4


def test_scalar_volume_local_weighted_budget_rejects_before_second_decode_and_releases(
    monkeypatch,
):
    plan = {
        "width": 2,
        "height": 1,
        "depth": 1,
        "dtype": "uint16",
        "bytes_per_voxel": 2,
        "channel": 0,
        "t": 0,
        "pages": 0,
        "source_width": 2,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "native-exact-v1",
        "sampling": "box",
    }
    volume = {
        **plan,
        "raw_min": 1,
        "raw_max": 2,
        "scl_slope": 1,
        "scl_inter": 0,
        "data": b"\x01\x00\x02\x00",
    }

    class BlockingRunner:
        workers = 1

        def __init__(self):
            self.scalar_reads = 0
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        async def call(self, method, _path, **_kwargs):
            if method == "scalar_plan":
                return dict(plan)
            if method != "scalar_volume":
                raise AssertionError(method)
            self.scalar_reads += 1
            if self.scalar_reads == 1:
                self.first_started.set()
                assert await asyncio.to_thread(self.release_first.wait, 5)
            return dict(volume)

    # One response requires 4 body bytes + one 4-byte plane array + one 4-byte
    # contiguous stack. The second request must fail before any scalar-plane work.
    monkeypatch.setenv("ULTRA_IMGSVC_SCALAR_VOLUME_INFLIGHT_BYTES", "12")
    runner = BlockingRunner()
    app = create_app(runner=runner)
    first_result: list[object] = []
    with TestClient(app) as local_client:
        first = threading.Thread(
            target=lambda: first_result.append(
                local_client.get(
                    "/scalar-volume",
                    params={"path": "mask.tif", "sampling": "box"},
                )
            )
        )
        first.start()
        assert runner.first_started.wait(timeout=5)

        rejected = local_client.get(
            "/scalar-volume",
            params={"path": "mask.tif", "sampling": "box"},
        )
        assert rejected.status_code == 503
        assert runner.scalar_reads == 1

        runner.release_first.set()
        first.join(timeout=5)
        assert not first.is_alive()
        assert first_result[0].status_code == 200

        admitted_after_release = local_client.get(
            "/scalar-volume",
            params={"path": "mask.tif", "sampling": "box"},
        )
        assert admitted_after_release.status_code == 200
        assert runner.scalar_reads == 2


def test_scalar_volume_local_weighted_budget_releases_after_decode_error(monkeypatch):
    plan = {
        "width": 1,
        "height": 1,
        "depth": 1,
        "dtype": "uint8",
        "bytes_per_voxel": 1,
        "channel": 0,
        "t": 0,
        "pages": 0,
        "source_width": 1,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "native-exact-v1",
        "sampling": "box",
    }

    class FailingOnceRunner:
        workers = 1

        def __init__(self):
            self.scalar_reads = 0

        async def call(self, method, _path, **_kwargs):
            if method == "scalar_plan":
                return dict(plan)
            self.scalar_reads += 1
            if self.scalar_reads == 1:
                raise RuntimeError("synthetic decode failure")
            return {
                **plan,
                "data": b"\x07",
                "raw_min": 7,
                "raw_max": 7,
                "scl_slope": 1,
                "scl_inter": 0,
            }

    monkeypatch.setenv("ULTRA_IMGSVC_SCALAR_VOLUME_INFLIGHT_BYTES", "3")
    runner = FailingOnceRunner()
    app = create_app(runner=runner)
    with TestClient(app) as local_client:
        with pytest.raises(RuntimeError, match="synthetic decode failure"):
            local_client.get(
                "/scalar-volume",
                params={"path": "mask.tif", "sampling": "box"},
            )
        recovered = local_client.get(
            "/scalar-volume",
            params={"path": "mask.tif", "sampling": "box"},
        )
    assert recovered.status_code == 200
    assert recovered.content == b"\x07"


@pytest.mark.parametrize(
    "failure_message",
    [
        "exact Mask source generation changed",
        "exact Mask decode work does not match its admission plan",
        "exact Mask read count does not match its admission plan",
    ],
)
def test_exact_mask_admission_refusal_is_422_and_releases_budget(
    monkeypatch,
    failure_message,
):
    plan = {
        "width": 1,
        "height": 1,
        "depth": 1,
        "dtype": "uint8",
        "bytes_per_voxel": 1,
        "channel": 0,
        "t": 0,
        "pages": 0,
        "source_width": 1,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "mask-native-integer-v1",
        "sampling": "nearest",
        "decode_admission": "complete-selected-scalar-v1",
        "admitted_decode_work_bytes": 1,
        "admitted_decode_read_count": 1,
        "admitted_source_dtype": "uint8",
        "admitted_source_bytes_per_voxel": 1,
        "source_generation": (1, 1, 2, 3, 4, 5),
    }

    class RefusingOnceRunner:
        workers = 2

        def __init__(self):
            self.plane_calls = 0

        async def call(self, method, _path, **_kwargs):
            if method == "scalar_plan":
                return dict(plan)
            if method != "scalar_planes":
                raise AssertionError(method)
            self.plane_calls += 1
            if self.plane_calls == 1:
                raise ValueError(failure_message)
            return [np.array([[7]], dtype="uint8")]

    monkeypatch.setenv("ULTRA_IMGSVC_SCALAR_VOLUME_INFLIGHT_BYTES", "3")
    runner = RefusingOnceRunner()
    app = create_app(runner=runner)
    with TestClient(app) as local_client:
        refused = local_client.get(
            "/scalar-volume",
            params={"path": "mask.tif", "sampling": "nearest"},
        )
        recovered = local_client.get(
            "/scalar-volume",
            params={"path": "mask.tif", "sampling": "nearest"},
        )

    assert refused.status_code == 422
    assert recovered.status_code == 200
    assert recovered.content == b"\x07"
    assert runner.plane_calls == 2


def test_scalar_volume_service_rejects_nonexact_nearest_before_decode_or_fanout():
    plan = {
        "width": 1,
        "height": 1,
        "depth": 1,
        "dtype": "float32",
        "bytes_per_voxel": 4,
        "channel": 0,
        "t": 0,
        "pages": 0,
        "source_width": 1,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "nearest-source-grid-v1",
        "sampling": "nearest",
    }

    class RejectingRunner:
        workers = 4

        def __init__(self):
            self.decode_calls = 0

        async def call(self, method, _path, **_kwargs):
            if method == "scalar_plan":
                return dict(plan)
            self.decode_calls += 1
            raise AssertionError("non-exact nearest must reject before decode")

    runner = RejectingRunner()
    app = create_app(runner=runner)
    with TestClient(app) as local_client:
        response = local_client.get(
            "/scalar-volume",
            params={"path": "float.tif", "sampling": "nearest"},
        )

    assert response.status_code == 422
    assert runner.decode_calls == 0


def test_scalar_volume_envelope_exposes_preview_provenance_and_actual_time():
    data, headers = _scalar_volume_envelope(
        {
            "width": 2,
            "height": 1,
            "depth": 1,
            "dtype": "uint16",
            "bytes_per_voxel": 2,
            "raw_min": 1,
            "raw_max": 2,
            "scl_slope": 1,
            "scl_inter": 0,
            "channel": 3,
            "t": 2,
            "source_width": 4,
            "source_height": 1,
            "source_depth": 1,
            "downsample_x": 2,
            "downsample_y": 1,
            "downsample_z": 1,
            "preview_policy": "auto-v1",
            "data": b"\x01\x00\x02\x00",
        }
    )

    assert data == b"\x01\x00\x02\x00"
    assert headers["x-volume-channel"] == "3"
    assert headers["x-volume-time"] == "2"
    assert headers["x-volume-source-width"] == "4"
    assert headers["x-volume-downsample-x"] == "2"
    assert headers["x-volume-preview-policy"] == "auto-v1"
    assert headers["x-volume-sampling"] == "box"


def test_scalar_volume_envelope_requires_explicit_consistent_identity_provenance():
    base = {
        "width": 2,
        "height": 1,
        "depth": 1,
        "dtype": "uint16",
        "bytes_per_voxel": 2,
        "raw_min": 1,
        "raw_max": 2,
        "scl_slope": 1,
        "scl_inter": 0,
        "channel": 0,
        "t": 0,
        "source_width": 4,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 2,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "auto-v1",
        "data": b"\x01\x00\x02\x00",
    }
    for field in (
        "t",
        "source_width",
        "source_height",
        "source_depth",
        "downsample_x",
        "downsample_y",
        "downsample_z",
        "preview_policy",
    ):
        malformed = dict(base)
        del malformed[field]
        with pytest.raises((KeyError, ValueError)):
            _scalar_volume_envelope(malformed)
    with pytest.raises(ValueError, match="source geometry|delivery grid|provenance"):
        _scalar_volume_envelope({**base, "source_width": 5})


def test_scalar_volume_envelope_rejects_fractional_geometry():
    with pytest.raises(ValueError, match="width must be an integer"):
        _scalar_volume_envelope(
            {
                "width": 1.5,
                "height": 1,
                "depth": 1,
                "dtype": "uint8",
                "bytes_per_voxel": 1,
                "raw_min": 0,
                "raw_max": 0,
                "scl_slope": 1,
                "scl_inter": 0,
                "channel": 0,
                "data": b"\x00",
            }
        )


def test_scalar_volume_envelope_rejects_body_length_mismatch():
    with pytest.raises(ValueError, match="body length"):
        _scalar_volume_envelope(
            {
                "width": 2,
                "height": 1,
                "depth": 1,
                "dtype": "uint8",
                "bytes_per_voxel": 1,
                "raw_min": 0,
                "raw_max": 0,
                "scl_slope": 1,
                "scl_inter": 0,
                "channel": 0,
                "data": b"\x00",
            }
        )


def test_scalar_volume_envelope_requires_explicit_finite_rescale():
    base = {
        "width": 1,
        "height": 1,
        "depth": 1,
        "dtype": "uint8",
        "bytes_per_voxel": 1,
        "raw_min": 0,
        "raw_max": 0,
        "scl_inter": 0,
        "channel": 0,
        "data": b"\x00",
    }
    with pytest.raises(KeyError, match="scl_slope"):
        _scalar_volume_envelope(base)
    with pytest.raises(ValueError, match="intensity metadata"):
        _scalar_volume_envelope({**base, "scl_slope": 0})


class _RaisingRunner:
    """A runner whose every op raises a chosen exception, for error-mapping tests."""

    workers = 1

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def call(self, method, *args, **kwargs):
        raise self._exc

    def shutdown(self) -> None:
        pass


@pytest.mark.parametrize(
    "message",
    [
        "engine returned an empty region (shape (0, 0, 0))",
        "scalar volume channel index 7 is out of range for C=3",
        "multiple scenes require an explicit scene identity for volume preview",
        "source plane input exceeds the bounded native semantic limit",
    ],
)
def test_undecodable_image_maps_to_422_not_500(message):
    # A malformed/unsupported file that the engine reads as a 0-sized region (after its
    # own transient-retry) must surface as 422 "preview unavailable", not a 500 that looks
    # like a server fault. Reproduced live on an undecodable .czi (libbioimage -> 0x0).
    app = create_app(runner=_RaisingRunner(ValueError(message)))
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
        zz, yy, xx = np.indices((12, 24, 30))
        vol.create_dataset(
            "euler",
            data=np.stack([xx + zz, yy * 2 + zz, ((xx + yy + zz) % 5) * 10], axis=-1).astype(
                "float32"
            ),
        )
        f.create_dataset("series", data=rng.random(120).astype("float64"))
    return path


@pytest.fixture()
def hdf5_feature_path(tmp_path):
    import numpy as np

    path = str(tmp_path / "feature-filter.dream3d")
    with h5py.File(path, "w") as file:
        image = file.create_group("DataContainers").create_group("Image")
        geometry = image.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([2, 2, 1], dtype="u8"))
        geometry.create_dataset("SPACING", data=np.ones(3, dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.zeros(3, dtype="f4"))
        cell = image.create_group("CellData")
        cell.create_dataset(
            "FeatureIds",
            data=np.array([[[[25], [7]], [[7], [7]]]], dtype="u4"),
        )
        cell.create_dataset(
            "IPFColor",
            data=np.array(
                [[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]],
                dtype="u1",
            ),
        )
    return path


def test_plain_viewerinfo_detects_hdf5_by_path_extension(client, hdf5_path):
    # The generic /viewerinfo endpoint forks to the hdf5 payload when the path (or the
    # optional name hint) is an HDF5-data file, so no image pipeline touches it.
    r = client.get("/viewerinfo", params={"path": hdf5_path})
    assert r.status_code == 200
    assert r.json()["kind"] == "hdf5"


def test_hdf5_dataset_route(client, hdf5_path):
    r = client.get(
        "/hdf5/dataset", params={"path": hdf5_path, "dataset_path": "/volume/ct", "file_id": "fid"}
    )
    assert r.status_code == 200
    s = r.json()
    assert s["preview_kind"] == "scalar_volume"
    assert s["dimension_summary"] == {"z": 12, "y": 24, "x": 30}


def test_hdf5_dataset_unknown_is_404(client, hdf5_path):
    r = client.get("/hdf5/dataset", params={"path": hdf5_path, "dataset_path": "/nope"})
    assert r.status_code == 404
    assert "error" in r.json()


def test_hdf5_slice_png(client, hdf5_path):
    r = client.get(
        "/hdf5/preview/slice",
        params={"path": hdf5_path, "dataset_path": "/volume/ct", "axis": "z", "index": 3},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_hdf5_slice_bad_index_type_is_422(client, hdf5_path):
    # FastAPI query validation: a non-int index → 422 automatically.
    r = client.get(
        "/hdf5/preview/slice",
        params={"path": hdf5_path, "dataset_path": "/volume/ct", "index": "abc"},
    )
    assert r.status_code == 422


def test_hdf5_atlas_png(client, hdf5_path):
    r = client.get(
        "/hdf5/preview/atlas", params={"path": hdf5_path, "dataset_path": "/volume/labels"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_hdf5_atlas_threads_vector_component_to_reader(client, hdf5_path):
    responses = [
        client.get(
            "/hdf5/preview/atlas",
            params={
                "path": hdf5_path,
                "dataset_path": "/volume/euler",
                "component": component,
            },
        )
        for component in range(3)
    ]
    assert all(response.status_code == 200 for response in responses)
    assert len({response.content for response in responses}) == 3


def test_hdf5_atlas_rejects_out_of_range_vector_component(client, hdf5_path):
    response = client.get(
        "/hdf5/preview/atlas",
        params={"path": hdf5_path, "dataset_path": "/volume/euler", "component": 3},
    )
    assert response.status_code == 422
    assert "component" in response.json()["detail"].lower()


def test_hdf5_service_threads_feature_ids_to_slice_and_atlas(client, hdf5_feature_path):
    import io

    import numpy as np
    from PIL import Image

    params = {
        "path": hdf5_feature_path,
        "dataset_path": "/DataContainers/Image/CellData/IPFColor",
        "feature_ids": "25",
    }
    atlas = client.get("/hdf5/preview/atlas", params=params)
    slice_response = client.get("/hdf5/preview/slice", params={**params, "axis": "z", "index": 0})
    assert atlas.status_code == 200
    assert slice_response.status_code == 200
    for response in (atlas, slice_response):
        rgba = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGBA"))
        assert int(np.count_nonzero(rgba[..., 3] == 255)) == 1
        assert np.all(rgba[rgba[..., 3] == 0] == 0)


def test_hdf5_scalar_volume_headers(client, hdf5_path):
    r = client.get(
        "/hdf5/preview/scalar-volume", params={"path": hdf5_path, "dataset_path": "/volume/ct"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-volume-dtype"] == "float32"
    assert float(r.headers["x-volume-scl-slope"]) == 1.0
    assert float(r.headers["x-volume-scl-inter"]) == 0.0
    w = int(r.headers["x-volume-width"])
    h = int(r.headers["x-volume-height"])
    d = int(r.headers["x-volume-depth"])
    assert (w, h, d) == (30, 24, 12)
    assert len(r.content) == w * h * d * 4
    assert float(r.headers["x-volume-raw-max"]) > float(r.headers["x-volume-raw-min"])
    for header, expected in {
        "x-volume-time": "0",
        "x-volume-source-width": "30",
        "x-volume-source-height": "24",
        "x-volume-source-depth": "12",
        "x-volume-downsample-x": "1",
        "x-volume-downsample-y": "1",
        "x-volume-downsample-z": "1",
        "x-volume-preview-policy": "stride-v1",
    }.items():
        assert r.headers[header] == expected


def test_hdf5_histogram_route(client, hdf5_path):
    r = client.get(
        "/hdf5/preview/histogram",
        params={"path": hdf5_path, "dataset_path": "/volume/ct", "bins": 24},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["discrete"] is False
    assert len(body["bins"]) == 24


def test_hdf5_table_route(client, hdf5_path):
    r = client.get(
        "/hdf5/preview/table",
        params={"path": hdf5_path, "dataset_path": "/series", "offset": 0, "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["preview_kind"] == "series"
    assert body["offset"] == 0 and body["limit"] == 5
    assert len(body["rows"]) == 5
