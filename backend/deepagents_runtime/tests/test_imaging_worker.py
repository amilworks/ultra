"""Transport-logic tests for the image.derive_pyramid convert worker.

The convert job logic lives in test_imaging_job.py; this covers _handle_message's
ack/nak/term decisions — in particular that a POISON job (one that always fails)
is terminated at the delivery cap instead of looping forever (criterion C4).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_deepagents.imaging import bioio_engine, worker
from ultra_deepagents.imaging import engine as imaging_engine
from ultra_deepagents.imaging.convert import ConvertResult
from ultra_deepagents.imaging.derivative_manifest import (
    DeterministicDerivativeError,
    StaleDerivativeJobError,
    TransientDerivativeError,
)
from ultra_deepagents.imaging.job import run_derive_pyramid_job
from ultra_deepagents.imaging.transcode import TranscodeResult


class FakeMsg:
    """Minimal JetStream message stand-in recording the ack op the worker chose."""

    def __init__(self, payload: dict, num_delivered: int = 1):
        self.data = json.dumps(payload).encode("utf-8")
        self.metadata = SimpleNamespace(num_delivered=num_delivered)
        self.acked = False
        self.naked = False
        self.termed = False
        self.nak_delays: list[float | None] = []
        self.in_progress_calls = 0

    async def ack(self) -> None:
        self.acked = True

    async def nak(self, delay: float | None = None) -> None:
        self.naked = True
        self.nak_delays.append(delay)

    async def term(self) -> None:
        self.termed = True

    async def in_progress(self) -> None:
        self.in_progress_calls += 1


_JOB = {
    "src_path": "/in.tif",
    "dst_path": "/out.tif",
    "resource_id": "res_1",
    "source_sha256": "a" * 64,
    "source_size_bytes": 123,
}


def _strict_viewer_info():
    return {
        "dims_order": "YX",
        "axis_sizes": {"T": 1, "C": 1, "Z": 1, "Y": 4, "X": 4},
        "dtype": "uint16",
        "channel_names": ["Channel 1"],
        "physical_spacing": {"x": 1.0, "y": 1.0, "z": 1.0},
        "metadata": {
            "selected_scene_id": "scene-0",
            "spacing_units": {"x": "voxel", "y": "voxel", "z": "voxel"},
        },
        "viewer": {"tile_scheme": None, "atlas_scheme": None},
        "tile_scheme": None,
    }


def test_production_source_resolver_uses_strict_scene_zero_publication(monkeypatch):
    public_metadata_only = {
        "kind": "unsupported",
        "decodable": False,
        "axis_sizes": {"T": 1, "C": 2, "Z": 1, "Y": 8, "X": 12},
        "scene_count": 2,
        "selected_scene_index": 0,
        "selected_scene_id": "Scene:0",
        "viewer": {"available_surfaces": ["metadata"]},
    }
    strict_scene_zero = {
        **public_metadata_only,
        "kind": "image",
        "decodable": True,
        "viewer": {"available_surfaces": ["2d", "metadata"]},
    }

    class FakeBioioEngine:
        def viewer_info(self, _path):
            return public_metadata_only

        def strict_publication_viewer_info(self, _path):
            return strict_scene_zero

    monkeypatch.setattr(bioio_engine, "BioioEngine", FakeBioioEngine)
    resolver = worker._resolve_source_viewer_info_fn()

    assert resolver is not None
    assert resolver("multi-scene.czi") == strict_scene_zero


def test_generic_metadata_only_bioio_scene_zero_publishes_through_worker_resolver(
    monkeypatch, tmp_path
):
    class FakeDask:
        shape = (1, 2, 1, 8, 12)
        chunks = ((1,), (1, 1), (1,), (8,), (12,))
        dtype = "uint16"

    class FakeBioImage:
        scenes = ("Scene:0", "Scene:1")
        physical_pixel_sizes = None

        def __init__(self, _path):
            self.current_scene = "Scene:1"

        def set_scene(self, scene):
            self.current_scene = self.scenes[scene] if isinstance(scene, int) else scene

        @property
        def dims(self):
            shape = (1, 2, 1, 8, 12) if self.current_scene == "Scene:0" else (1, 1, 1, 2, 3)
            return SimpleNamespace(order="TCZYX", shape=shape)

        @property
        def channel_names(self):
            return ("A", "B") if self.current_scene == "Scene:0" else ("wrong-scene",)

        def get_image_dask_data(self, order):
            assert order == "TCZYX"
            assert self.current_scene == "Scene:0"
            return FakeDask()

    monkeypatch.setitem(sys.modules, "bioio", SimpleNamespace(BioImage=FakeBioImage))
    source = tmp_path / "multi-scene.czi"
    source.write_bytes(b"multi-scene-source")
    destination = tmp_path / "derived" / "res_1__pyramid.tif"
    engine = bioio_engine.BioioEngine()
    public = engine.viewer_info(str(source))
    strict = engine.strict_publication_viewer_info(str(source))
    assert public["kind"] == "unsupported" and public["viewer"]["available_surfaces"] == [
        "metadata"
    ]
    assert strict["kind"] == "image" and strict["selected_scene_index"] == 0
    derived = json.loads(json.dumps(strict))
    derived["display_defaults"] = {"channels": [1]}

    def transcode(_src, dst, *, prefer):
        assert prefer == "first"
        with open(dst, "wb") as stream:
            stream.write(b"intermediate")
        return TranscodeResult(
            dst,
            2,
            0,
            "Scene:0",
            num_c=2,
            num_z=1,
            num_t=1,
            dtype="uint16",
            series_names=["Scene:0", "Scene:1"],
        )

    def convert(src, dst, *, spec):
        assert src.endswith(".transcode.ome.tif")
        with open(dst, "wb") as stream:
            stream.write(b"derived")
        return ConvertResult(src, dst, 0, "", "")

    source_resolver = worker._resolve_source_viewer_info_fn()
    assert source_resolver is not None
    result = run_derive_pyramid_job(
        {
            "resource_id": "res_1",
            "src_path": str(source),
            "dst_path": str(destination),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size_bytes": source.stat().st_size,
            "fmt": "auto",
        },
        convert_fn=convert,
        meta_fn=lambda _path: {"image_num_x": 12, "image_num_y": 8, "image_num_c": 2},
        transcode_fn=transcode,
        viewer_info_fn=lambda _path: derived,
        source_viewer_info_fn=source_resolver,
        require_manifest=True,
    )

    manifest = json.loads(destination.with_suffix(".manifest.json").read_text())
    assert result["status"] == "succeeded"
    assert manifest["producer"] == {
        "reader": "bioio",
        "series_count": 2,
        "series_index": 0,
        "series_name": "Scene:0",
    }
    assert manifest["semantics"]["display"]["default_channels"] == [0, 1]


def _run(
    msg: FakeMsg,
    *,
    max_deliver: int = worker.DEFAULT_MAX_DELIVER,
    viewer_info_fn=None,
    ack_progress_interval_seconds: float = worker.DEFAULT_ACK_PROGRESS_INTERVAL_SECONDS,
) -> None:
    asyncio.run(
        worker._handle_message(
            msg,
            meta_fn=None,
            viewer_info_fn=viewer_info_fn,
            max_deliver=max_deliver,
            ack_progress_interval_seconds=ack_progress_interval_seconds,
        )
    )


def test_successful_job_is_acked(monkeypatch):
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda job, **kwargs: {"resource_id": "res_1", "levels": 4},
    )
    msg = FakeMsg(_JOB)
    _run(msg)
    assert msg.acked and not msg.naked and not msg.termed


def test_scene_derive_job_is_dispatched_instead_of_acknowledged_as_unrelated(monkeypatch):
    calls: list[dict] = []

    def derive(job):
        calls.append(dict(job))
        return {
            "resource_id": job["resource_id"],
            "status": "succeeded",
            "chunk_count": 4,
        }

    monkeypatch.setattr(worker, "run_scene3d_derive_job", derive, raising=False)
    msg = FakeMsg(
        {
            "job_id": "imgjob_scene_1",
            "job_type": "scene.derive",
            "metadata": {
                "resource_id": "file_scene_1",
                "src_path": "/in.ply",
                "dst_dir": "/derived/file_scene_1__scene3d.v3.sha256-" + "b" * 64,
                "source_sha256": "b" * 64,
                "source_size_bytes": 456,
                "splat_delivery": "spark-rad-v1",
            },
        }
    )

    _run(msg)

    assert msg.acked and not msg.naked and not msg.termed
    assert calls == [
        {
            "resource_id": "file_scene_1",
            "src_path": "/in.ply",
            "dst_dir": "/derived/file_scene_1__scene3d.v3.sha256-" + "b" * 64,
            "source_sha256": "b" * 64,
            "source_size_bytes": 456,
            "splat_delivery": "spark-rad-v1",
            "force_id": "imgjob_scene_1",
        }
    ]


def test_long_conversion_extends_jetstream_ack_deadline(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocking_job(job, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return {"resource_id": job["resource_id"], "levels": 4}

    monkeypatch.setattr(worker, "run_derive_pyramid_job", blocking_job)
    msg = FakeMsg(_JOB)
    finished = threading.Event()

    def handle() -> None:
        try:
            _run(msg, ack_progress_interval_seconds=0.005)
        finally:
            finished.set()

    thread = threading.Thread(target=handle)
    thread.start()
    assert started.wait(timeout=5)
    deadline = time.monotonic() + 2
    while msg.in_progress_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert msg.in_progress_calls > 0
    release.set()
    thread.join(timeout=5)
    assert finished.is_set() and msg.acked
    progress_after_ack = msg.in_progress_calls
    time.sleep(0.02)
    assert msg.in_progress_calls == progress_after_ack


def test_foreign_job_type_is_acked_not_redelivered(monkeypatch):
    # A message for another job type must be ack'd so JetStream stops routing it here,
    # never nak'd (which would redeliver it to us forever).
    def _should_not_run(job, meta_fn=None):  # pragma: no cover - must not be called
        raise AssertionError("foreign job must not be executed")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _should_not_run)
    msg = FakeMsg({"job_type": "something.else", "metadata": {"x": 1}})
    _run(msg)
    assert msg.acked and not msg.naked and not msg.termed


def test_legacy_job_missing_source_identity_is_retired_without_marker(monkeypatch, tmp_path):
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    msg = FakeMsg({"resource_id": "res_1", "src_path": "/in.tif", "dst_path": str(dst)})

    _run(msg)

    assert msg.acked and not msg.naked and not msg.termed
    assert not (tmp_path / "derived" / "res_1__pyramid.failed").exists()


def test_transient_failure_below_cap_is_naked_for_retry(monkeypatch):
    def _boom(job, **kwargs):
        raise TransientDerivativeError("viewer_info_unavailable")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB, num_delivered=1)
    _run(msg, max_deliver=5)
    assert msg.naked and not msg.termed and not msg.acked
    assert msg.nak_delays == [worker.DEFAULT_RETRY_DELAY_SECONDS]


def test_poison_job_at_cap_is_terminated_not_redelivered(monkeypatch):
    # The crux: a job that always fails must be term()'d once it reaches the delivery
    # cap so it leaves the consumer for good — it cannot wedge the worker forever.
    def _boom(job, **kwargs):
        raise TransientDerivativeError("engine_unavailable")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB, num_delivered=5)
    _run(msg, max_deliver=5)
    assert msg.termed and not msg.naked and not msg.acked


def test_poison_job_writes_failure_marker(monkeypatch, tmp_path):
    # On permanent failure, the worker drops a <fileID>__pyramid.failed sidecar so the
    # control plane can back off re-enqueuing a doomed convert (the 707k-redelivery fix).
    src = tmp_path / "source.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {
        "src_path": str(src),
        "dst_path": str(dst),
        "resource_id": "res_1",
        "source_sha256": "a" * 64,
        "source_size_bytes": 123,
        "tile_size": 512,
        "compression": "lzw",
        "layout": "topdirs",
        "fmt": "auto",
    }

    def _boom(job, **kwargs):
        raise DeterministicDerivativeError("unsupported_source")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(job, num_delivered=5)
    _run(msg, max_deliver=5)

    marker = tmp_path / "derived" / "res_1__pyramid.failed"
    assert msg.termed and marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["resource_id"] == "res_1"
    assert payload["code"] == "unsupported_source"
    assert "error" not in payload
    assert "src_path" not in payload
    assert payload["source_sha256"] == "a" * 64
    assert payload["conversion_spec"]["tile_size"] == 512
    assert stat.S_IMODE(marker.stat().st_mode) == 0o644


def test_failure_marker_cannot_publish_after_source_deletion(monkeypatch, tmp_path):
    import fcntl

    src = tmp_path / "source.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {**_JOB, "src_path": str(src), "dst_path": str(dst)}
    lock_path = tmp_path / ".locks" / ".res_1__pyramid.lock"
    lock_path.parent.mkdir()
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DeterministicDerivativeError("unsupported_source")
        ),
    )
    msg = FakeMsg(job, num_delivered=5)
    finished = threading.Event()

    def handle_message() -> None:
        try:
            _run(msg, max_deliver=5, ack_progress_interval_seconds=0.005)
        finally:
            finished.set()

    thread = threading.Thread(target=handle_message)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while msg.in_progress_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not finished.is_set(), "failure marker ignored the lifecycle lock"
        assert msg.in_progress_calls > 0, "failure-marker publication stopped AckWait heartbeats"
        src.unlink()
        lock_path.unlink()
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
        thread.join(timeout=5)

    assert finished.is_set()
    assert msg.acked and not msg.termed and not dst.with_suffix(".failed").exists()
    progress_after_ack = msg.in_progress_calls
    time.sleep(0.02)
    assert msg.in_progress_calls == progress_after_ack


def test_fetch_failures_use_capped_backoff_and_timeout_resets_streak(monkeypatch):
    class FailingSubscription:
        def __init__(self):
            self.calls = 0

        async def fetch(self, _batch, *, timeout):
            assert timeout == 5
            self.calls += 1
            raise RuntimeError("transport unavailable")

    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(worker.random, "uniform", lambda _low, _high: 1.0)
    monkeypatch.setattr(worker.asyncio, "sleep", record_sleep)
    subscription = FailingSubscription()

    async def exercise_failures():
        failures = 0
        for _ in range(7):
            messages, failures = await worker._fetch_one(
                subscription, consecutive_failures=failures
            )
            assert messages == []
        return failures

    failures = asyncio.run(exercise_failures())
    assert failures == 7
    assert subscription.calls == 7
    assert delays == [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0]

    class TimeoutSubscription:
        async def fetch(self, _batch, *, timeout):
            raise TimeoutError

    messages, reset = asyncio.run(
        worker._fetch_one(TimeoutSubscription(), consecutive_failures=failures)
    )
    assert messages == [] and reset == 0
    assert len(delays) == 7


def test_failure_marker_io_failure_is_naked_not_terminated(monkeypatch, tmp_path):
    src = tmp_path / "source.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {**_JOB, "src_path": str(src), "dst_path": str(dst)}
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DeterministicDerivativeError("unsupported_source")
        ),
    )
    monkeypatch.setattr(
        worker.tempfile,
        "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk temporarily unavailable")),
    )

    msg = FakeMsg(job, num_delivered=5)
    _run(msg, max_deliver=5)

    assert msg.naked and not msg.termed and not msg.acked
    assert msg.nak_delays == [worker.DEFAULT_RETRY_DELAY_SECONDS]


def test_transient_failure_does_not_write_failure_marker(monkeypatch, tmp_path):
    # A below-cap (retryable) failure must NOT mark the resource as permanently failed.
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {**_JOB, "dst_path": str(dst)}
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda job, **kwargs: (_ for _ in ()).throw(
            TransientDerivativeError("viewer_info_unavailable")
        ),
    )
    msg = FakeMsg(job, num_delivered=1)
    _run(msg, max_deliver=5)
    assert msg.naked and not (tmp_path / "derived" / "res_1__pyramid.failed").exists()


def test_failure_marker_path_swaps_tif_suffix():
    assert worker._failure_marker_path("/a/b/res__pyramid.tif") == "/a/b/res__pyramid.failed"
    assert worker._failure_marker_path("/a/b/res__pyramid") == "/a/b/res__pyramid.failed"


def test_missing_delivery_metadata_defaults_to_retry(monkeypatch):
    # If the transport doesn't expose num_delivered, treat it as attempt 1 (retry),
    # never as an immediate terminate — we'd rather retry a recoverable job once more.
    def _boom(job, **kwargs):
        raise TransientDerivativeError("engine_unavailable")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB)
    msg.metadata = SimpleNamespace()  # no num_delivered attribute
    _run(msg, max_deliver=5)
    assert msg.naked and not msg.termed


def test_stale_job_is_acked_without_marker(monkeypatch, tmp_path):
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {**_JOB, "dst_path": str(dst)}
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StaleDerivativeJobError("source_generation_changed")
        ),
    )

    msg = FakeMsg(job)
    _run(msg)

    assert msg.acked and not msg.naked and not msg.termed
    assert not dst.with_suffix(".failed").exists()


def test_nonregular_source_generation_is_acked_without_marker_through_worker(tmp_path):
    for source_kind in ("directory", "symlink"):
        case = tmp_path / source_kind
        case.mkdir()
        if source_kind == "directory":
            source = case / "source.tif"
            source.mkdir()
        else:
            referent = case / "referent.tif"
            referent.write_bytes(b"source")
            source = case / "source.tif"
            source.symlink_to(referent)
        dst = case / "derived" / "res_1__pyramid.tif"
        msg = FakeMsg(
            {
                **_JOB,
                "src_path": str(source),
                "dst_path": str(dst),
                "source_size_bytes": 6,
            }
        )

        _run(msg, viewer_info_fn=lambda _path: _strict_viewer_info())

        assert msg.acked and not msg.naked and not msg.termed
        assert not dst.with_suffix(".failed").exists()


def test_invalid_format_terms_with_source_bound_marker_before_converter(tmp_path):
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    msg = FakeMsg(
        {
            **_JOB,
            "src_path": str(source),
            "dst_path": str(dst),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size_bytes": source.stat().st_size,
            "fmt": "not-a-real-format",
        }
    )

    _run(msg, viewer_info_fn=lambda _path: _strict_viewer_info())

    marker = dst.with_suffix(".failed")
    assert msg.termed and not msg.naked and not msg.acked
    marker_payload = json.loads(marker.read_text())
    assert marker_payload["code"] == "invalid_conversion_spec"
    assert marker_payload["conversion_spec"]["fmt"] == "not-a-real-format"


def test_plain_value_error_is_not_misclassified_as_deterministic(monkeypatch, tmp_path):
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {**_JOB, "dst_path": str(dst)}
    monkeypatch.setattr(
        worker,
        "run_derive_pyramid_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("missing viewer info")),
    )

    msg = FakeMsg(job)
    _run(msg)

    assert msg.naked and not msg.termed and not msg.acked
    assert not dst.with_suffix(".failed").exists()


def test_ack_happens_only_after_manifest_commit(monkeypatch, tmp_path):
    manifest = tmp_path / "res_1__pyramid.manifest.json"
    artifact = tmp_path / "res_1__pyramid.sha256-abc.tif"

    class ManifestCheckingMsg(FakeMsg):
        async def ack(self) -> None:
            assert manifest.is_file()
            await super().ack()

    def _publish(job, **kwargs):
        assert kwargs["require_manifest"] is True
        manifest.write_text("{}")
        return {
            "resource_id": "res_1",
            "derived_path": str(artifact),
            "manifest_path": str(manifest),
            "status": "succeeded",
        }

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _publish)
    msg = ManifestCheckingMsg(_JOB)
    _run(msg)
    assert msg.acked and not msg.naked and not msg.termed


def test_viewer_info_resolver_accepts_non_libbio_engine(monkeypatch):
    class SemanticEngine:
        def viewer_info(self, path):
            return {"path": path}

    semantic_engine = SemanticEngine()
    monkeypatch.setattr(imaging_engine, "build_engine", lambda prefer_real: semantic_engine)

    resolved = worker._resolve_viewer_info_fn()

    assert resolved is not None
    assert resolved("/source.czi") == {"path": "/source.czi"}
