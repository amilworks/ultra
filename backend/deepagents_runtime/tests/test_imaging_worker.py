"""Transport-logic tests for the image.derive_pyramid convert worker.

The convert job logic lives in test_imaging_job.py; this covers _handle_message's
ack/nak/term decisions — in particular that a POISON job (one that always fails)
is terminated at the delivery cap instead of looping forever (criterion C4).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_deepagents.imaging import worker


class FakeMsg:
    """Minimal JetStream message stand-in recording the ack op the worker chose."""

    def __init__(self, payload: dict, num_delivered: int = 1):
        self.data = json.dumps(payload).encode("utf-8")
        self.metadata = SimpleNamespace(num_delivered=num_delivered)
        self.acked = False
        self.naked = False
        self.termed = False

    async def ack(self) -> None:
        self.acked = True

    async def nak(self) -> None:
        self.naked = True

    async def term(self) -> None:
        self.termed = True


_JOB = {"src_path": "/in.tif", "dst_path": "/out.tif", "resource_id": "res_1"}


def _run(msg: FakeMsg, *, max_deliver: int = worker.DEFAULT_MAX_DELIVER) -> None:
    asyncio.run(worker._handle_message(msg, meta_fn=None, max_deliver=max_deliver))


def test_successful_job_is_acked(monkeypatch):
    monkeypatch.setattr(worker, "run_derive_pyramid_job", lambda job, meta_fn=None: {"resource_id": "res_1", "levels": 4})
    msg = FakeMsg(_JOB)
    _run(msg)
    assert msg.acked and not msg.naked and not msg.termed


def test_foreign_job_type_is_acked_not_redelivered(monkeypatch):
    # A message for another job type must be ack'd so JetStream stops routing it here,
    # never nak'd (which would redeliver it to us forever).
    def _should_not_run(job, meta_fn=None):  # pragma: no cover - must not be called
        raise AssertionError("foreign job must not be executed")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _should_not_run)
    msg = FakeMsg({"job_type": "something.else", "metadata": {"x": 1}})
    _run(msg)
    assert msg.acked and not msg.naked and not msg.termed


def test_transient_failure_below_cap_is_naked_for_retry(monkeypatch):
    def _boom(job, meta_fn=None):
        raise RuntimeError("transient imgcnv hiccup")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB, num_delivered=1)
    _run(msg, max_deliver=5)
    assert msg.naked and not msg.termed and not msg.acked


def test_poison_job_at_cap_is_terminated_not_redelivered(monkeypatch):
    # The crux: a job that always fails must be term()'d once it reaches the delivery
    # cap so it leaves the consumer for good — it cannot wedge the worker forever.
    def _boom(job, meta_fn=None):
        raise RuntimeError("unsupported source; will always fail")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB, num_delivered=5)
    _run(msg, max_deliver=5)
    assert msg.termed and not msg.naked and not msg.acked


def test_poison_job_writes_failure_marker(monkeypatch, tmp_path):
    # On permanent failure, the worker drops a <fileID>__pyramid.failed sidecar so the
    # control plane can back off re-enqueuing a doomed convert (the 707k-redelivery fix).
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {"src_path": "/in.tif", "dst_path": str(dst), "resource_id": "res_1"}

    def _boom(job, meta_fn=None):
        raise RuntimeError("imgcnv conversion failed (exit 3): Input format is not supported")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(job, num_delivered=5)
    _run(msg, max_deliver=5)

    marker = tmp_path / "derived" / "res_1__pyramid.failed"
    assert msg.termed and marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["resource_id"] == "res_1" and "not supported" in payload["error"]


def test_transient_failure_does_not_write_failure_marker(monkeypatch, tmp_path):
    # A below-cap (retryable) failure must NOT mark the resource as permanently failed.
    dst = tmp_path / "derived" / "res_1__pyramid.tif"
    job = {"src_path": "/in.tif", "dst_path": str(dst), "resource_id": "res_1"}
    monkeypatch.setattr(worker, "run_derive_pyramid_job", lambda job, meta_fn=None: (_ for _ in ()).throw(RuntimeError("transient")))
    msg = FakeMsg(job, num_delivered=1)
    _run(msg, max_deliver=5)
    assert msg.naked and not (tmp_path / "derived" / "res_1__pyramid.failed").exists()


def test_failure_marker_path_swaps_tif_suffix():
    assert worker._failure_marker_path("/a/b/res__pyramid.tif") == "/a/b/res__pyramid.failed"
    assert worker._failure_marker_path("/a/b/res__pyramid") == "/a/b/res__pyramid.failed"


def test_missing_delivery_metadata_defaults_to_retry(monkeypatch):
    # If the transport doesn't expose num_delivered, treat it as attempt 1 (retry),
    # never as an immediate terminate — we'd rather retry a recoverable job once more.
    def _boom(job, meta_fn=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "run_derive_pyramid_job", _boom)
    msg = FakeMsg(_JOB)
    msg.metadata = SimpleNamespace()  # no num_delivered attribute
    _run(msg, max_deliver=5)
    assert msg.naked and not msg.termed
