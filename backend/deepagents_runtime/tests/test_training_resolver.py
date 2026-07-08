"""ServingWeightsResolver (§8.1) — fail-open wire client + the rarespot seam.

The urlopen monkeypatch pins the actual wire (URL, method, worker-token
header, timeout) rather than a private helper; the config-seam tests pin the
env gate + the baked-weights fallback ladder.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ultra_deepagents.rarespot.config import (
    RARESPOT_MODEL_KEY,
    RareSpotConfig,
    resolve_serving_weights,
)
from ultra_deepagents.training import resolver as resolver_module
from ultra_deepagents.training.resolver import ServingWeightsResolver

RESOLUTION = {
    "model_key": "yolov5_rarespot",
    "weights_uri": "/mnt/barrel-data/ultra/training/weights/yolov5_rarespot/v1/weights.pt",
    "version_id": "yolov5_rarespot-v1",
    "is_canary": False,
}


def _settings(**overrides):
    values = {
        "control_base_url": "http://control.test:8088",
        "control_worker_token": "worker-token",
        "control_status_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _pin_urlopen(monkeypatch, payload=RESOLUTION):
    calls: list[tuple[object, float | None]] = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return _FakeResponse(payload)

    monkeypatch.setattr(resolver_module.urllib_request, "urlopen", fake_urlopen)
    return calls


def test_resolve_happy_path_pins_wire(monkeypatch):
    calls = _pin_urlopen(monkeypatch)
    resolver = ServingWeightsResolver(_settings())
    resolution = resolver.resolve("yolov5_rarespot", "run-123")
    assert resolution == RESOLUTION
    request, timeout = calls[0]
    assert (
        request.full_url
        == "http://control.test:8088/v2/training/models/yolov5_rarespot/resolve?run_id=run-123"
    )
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.headers["X-ultra-worker-token"] == "worker-token"
    assert timeout == 2.0


def test_resolve_passes_canary_flag_through(monkeypatch):
    _pin_urlopen(monkeypatch, {**RESOLUTION, "is_canary": True})
    resolution = ServingWeightsResolver(_settings()).resolve("yolov5_rarespot", "run-0")
    assert resolution is not None
    assert resolution["is_canary"] is True


def test_resolve_ttl_cache_skips_second_http_call(monkeypatch):
    calls = _pin_urlopen(monkeypatch)
    resolver = ServingWeightsResolver(_settings(), ttl_seconds=30.0)
    first = resolver.resolve("yolov5_rarespot", "run-123")
    second = resolver.resolve("yolov5_rarespot", "run-123")
    assert first == second == RESOLUTION
    assert len(calls) == 1
    # The split is per-run: a different run_id must NOT reuse the cached answer.
    resolver.resolve("yolov5_rarespot", "run-456")
    assert len(calls) == 2


def test_resolve_never_raises_and_fails_open(monkeypatch):
    def broken_urlopen(request, timeout=None):
        raise OSError("control plane unreachable")

    monkeypatch.setattr(resolver_module.urllib_request, "urlopen", broken_urlopen)
    assert ServingWeightsResolver(_settings()).resolve("yolov5_rarespot", "run-1") is None
    # Unconfigured resolver (no base URL) also fails open.
    assert ServingWeightsResolver(_settings(control_base_url="")).resolve("x", "run-1") is None


def test_resolve_rejects_payload_without_weights_uri(monkeypatch):
    _pin_urlopen(monkeypatch, {**RESOLUTION, "weights_uri": ""})
    assert ServingWeightsResolver(_settings()).resolve("yolov5_rarespot", "run-1") is None


def test_record_canary_observation_posts_and_swallows_errors(monkeypatch):
    calls = _pin_urlopen(monkeypatch, {"observation_id": "obs-1"})
    resolver = ServingWeightsResolver(_settings())
    payload = {"run_id": "run-1", "canary_version_id": "v1", "canary_metrics": {"detection_count": 3}}
    result = resolver.record_canary_observation("yolov5_rarespot", payload)
    assert result == {"observation_id": "obs-1"}
    request, _ = calls[0]
    assert (
        request.full_url
        == "http://control.test:8088/v2/training/models/yolov5_rarespot/canary-observations"
    )
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == payload

    def broken_urlopen(request, timeout=None):
        raise OSError("control plane unreachable")

    monkeypatch.setattr(resolver_module.urllib_request, "urlopen", broken_urlopen)
    assert resolver.record_canary_observation("yolov5_rarespot", payload) is None


# --------------------------------------------------------------- config seam


def _config(tmp_path):
    weights = tmp_path / "baked.pt"
    weights.write_bytes(b"baked")
    return RareSpotConfig.from_paths(
        weights_path=weights, yolov5_path=tmp_path, artifact_root=tmp_path
    )


def test_config_seam_env_off_serves_baked_weights(monkeypatch, tmp_path):
    monkeypatch.delenv("ULTRA_RARESPOT_SERVING_RESOLVER", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        ServingWeightsResolver, "resolve", lambda self, mk, rid: calls.append(mk)
    )
    config = _config(tmp_path)
    path, info = resolve_serving_weights(config, "run-1", settings=_settings())
    assert path == config.weights_path
    assert info is None
    assert calls == []


def test_config_seam_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRA_RARESPOT_SERVING_RESOLVER", "1")
    monkeypatch.setattr(
        ServingWeightsResolver,
        "resolve",
        lambda self, mk, rid: {
            "model_key": mk,
            "weights_uri": str(tmp_path / "missing.pt"),
            "version_id": "yolov5_rarespot-v9",
            "is_canary": True,
        },
    )
    config = _config(tmp_path)
    path, info = resolve_serving_weights(config, "run-1", settings=_settings())
    assert path == config.weights_path
    assert info is None


def test_config_seam_serves_resolved_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRA_RARESPOT_SERVING_RESOLVER", "1")
    served = tmp_path / "canary.pt"
    served.write_bytes(b"canary")
    seen: list[tuple[str, str]] = []

    def fake_resolve(self, model_key, run_id):
        seen.append((model_key, run_id))
        return {
            "model_key": model_key,
            "weights_uri": str(served),
            "version_id": "yolov5_rarespot-v2",
            "is_canary": True,
        }

    monkeypatch.setattr(ServingWeightsResolver, "resolve", fake_resolve)
    config = _config(tmp_path)
    path, info = resolve_serving_weights(config, "run-42", settings=_settings())
    assert path == served.resolve()
    assert info is not None
    assert info["version_id"] == "yolov5_rarespot-v2"
    assert info["is_canary"] is True
    assert seen == [(RARESPOT_MODEL_KEY, "run-42")]
