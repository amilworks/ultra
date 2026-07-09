"""Tests for the worker-side compute-service client (offloads GPU work)."""

from __future__ import annotations

import pytest

import ultra_deepagents.training.compute_client as cc


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def _config() -> cc.ComputeServiceConfig:
    return cc.ComputeServiceConfig(base_url="http://svc:8030", token="tok", poll_interval=1.0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cc.time, "sleep", lambda *_a, **_k: None)


def test_from_env_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ULTRA_COMPUTE_SERVICE_URL", raising=False)
    assert cc.ComputeServiceConfig.from_env() is None


def test_submit_and_wait_success_forwards_progress(monkeypatch):
    posts: list = []
    monkeypatch.setattr(cc.httpx, "post", lambda url, **kw: posts.append((url, kw)) or _Resp(202, {"job_id": "J1"}))
    gets = iter(
        [
            _Resp(200, {"status": "running", "progress": {"completed": 3, "total": 40, "message": "epoch 3/40"}}),
            _Resp(200, {"status": "running", "progress": {"completed": 20, "total": 40, "message": "epoch 20/40"}}),
            _Resp(200, {"status": "succeeded", "result": {"weights_uri": "/out/best.pt"}, "progress": {"completed": 40, "total": 40}}),
        ]
    )
    monkeypatch.setattr(cc.httpx, "get", lambda url, **kw: next(gets))

    seen: list = []
    client = cc.ComputeServiceClient(_config())
    result = client.submit_and_wait(
        executor="rarespot.train",
        spec={"a": 1},
        idempotency_key="J:finetune",
        on_progress=lambda p: seen.append((p.get("completed"), p.get("total"))),
    )
    assert result == {"weights_uri": "/out/best.pt"}
    assert (3, 40) in seen and (20, 40) in seen  # progress forwarded, deduped on change
    assert posts[0][1]["json"]["idempotency_key"] == "J:finetune"


def test_submit_and_wait_raises_on_failure(monkeypatch):
    monkeypatch.setattr(cc.httpx, "post", lambda url, **kw: _Resp(202, {"job_id": "J2"}))
    monkeypatch.setattr(cc.httpx, "get", lambda url, **kw: _Resp(200, {"status": "failed", "error": "boom"}))
    client = cc.ComputeServiceClient(_config())
    with pytest.raises(cc.ComputeServiceError, match="boom"):
        client.submit_and_wait(executor="rarespot.train", spec={})


def test_submit_and_wait_cancels_when_asked(monkeypatch):
    cancels: list = []

    def _post(url, **kw):
        if url.endswith("/cancel"):
            cancels.append(url)
            return _Resp(200, {"status": "cancelled"})
        return _Resp(202, {"job_id": "J3"})

    monkeypatch.setattr(cc.httpx, "post", _post)
    monkeypatch.setattr(cc.httpx, "get", lambda url, **kw: _Resp(200, {"status": "running", "progress": {}}))
    client = cc.ComputeServiceClient(_config())
    with pytest.raises(cc.ComputeJobCancelled):
        client.submit_and_wait(executor="rarespot.train", spec={}, should_cancel=lambda: True)
    assert any(u.endswith("/v1/jobs/J3/cancel") for u in cancels)


def test_submit_rejects_non_202(monkeypatch):
    monkeypatch.setattr(cc.httpx, "post", lambda url, **kw: _Resp(400, {}, text="bad executor"))
    client = cc.ComputeServiceClient(_config())
    with pytest.raises(cc.ComputeServiceError, match="rejected"):
        client.submit(executor="nope.gone", spec={})


def test_poll_tolerates_transient_get_failures(monkeypatch):
    monkeypatch.setattr(cc.httpx, "post", lambda url, **kw: _Resp(202, {"job_id": "J4"}))
    calls = {"n": 0}

    def _get(url, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise cc.httpx.ConnectError("blip")
        return _Resp(200, {"status": "succeeded", "result": {"ok": True}})

    monkeypatch.setattr(cc.httpx, "get", _get)
    client = cc.ComputeServiceClient(_config())
    assert client.submit_and_wait(executor="x.y", spec={}) == {"ok": True}
