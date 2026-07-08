"""GoldGate training worker unit tests — no network, no torch, no NATS.

Covers: envelope parse/validation, the OS-thread lease+ack keepalive (renewal,
in_progress scheduling, the 409 kill flag), the adapter registry, the five
phase branches through a fake control client, gold content-hash determinism,
and the generic fail-closed freeze defenses.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.training.adapters import (
    ADAPTERS,
    MaterializeContext,
    ModelAdapter,
    NoAdapterRegistered,
    TrainContext,
    get_adapter,
)
from ultra_deepagents.training.control_client import (
    TrainingControlClient,
    TrainingJobLease,
    TrainingLeaseConflict,
    TrainingLeaseUnavailable,
)
from ultra_deepagents.training.envelope import TrainingJobEnvelope
from ultra_deepagents.training.keepalive import (
    TrainingLeaseKeepalive,
    training_keepalive_interval,
)
from ultra_deepagents.training.manifests import RARESPOT_MANIFEST
from ultra_deepagents.training.worker import (
    NATSTrainingWorker,
    TrainingWorkerSettings,
    compute_gold_content_hash,
    run_generic_leakage_defenses,
)


def _runtime_settings(**overrides) -> RuntimeSettings:
    values = {
        "openai_base_url": "http://example.test/v1",
        "openai_model": "deepseek_v4",
        "control_base_url": "https://control.example.test",
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def _worker_settings(**overrides) -> TrainingWorkerSettings:
    values = {
        "runtime": _runtime_settings(),
        "lease_ttl_seconds": 30.0,
        "redelivery_delay_seconds": 1.0,
        "worker_id": "training-worker-test",
    }
    values.update(overrides)
    return TrainingWorkerSettings(**values)


def _job_payload(**overrides) -> dict:
    payload = {
        "job_id": "training_job_1",
        "dispatch_id": "dispatch-1",
        "model_key": "yolov5_rarespot",
        "job_type": "training.sync",
        "gpu_pool": "titan",
        "owner_user_id": "agent-user",
        "owner_org_id": "agent-org",
        "params": {"frame_source_dir": "/data/frames"},
        "metadata": {"origin": "test"},
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------- envelope


def test_envelope_parses_full_payload() -> None:
    job = TrainingJobEnvelope.from_dict(_job_payload())
    assert job.job_id == "training_job_1"
    assert job.dispatch_id == "dispatch-1"
    assert job.model_key == "yolov5_rarespot"
    assert job.job_type == "training.sync"
    assert job.gpu_pool == "titan"
    assert job.owner_user_id == "agent-user"
    assert job.owner_org_id == "agent-org"
    assert job.params == {"frame_source_dir": "/data/frames"}
    assert job.metadata == {"origin": "test"}
    assert job.principal_headers() == {
        "X-Ultra-User-Id": "agent-user",
        "X-Ultra-Org-Id": "agent-org",
        "X-Ultra-Role": "researcher",
    }


def test_envelope_requires_identity_and_phase() -> None:
    with pytest.raises(ValueError, match="job_id"):
        TrainingJobEnvelope.from_dict(_job_payload(job_id=""))
    with pytest.raises(ValueError, match="model_key"):
        TrainingJobEnvelope.from_dict(_job_payload(model_key=""))
    with pytest.raises(ValueError, match="training.bogus"):
        TrainingJobEnvelope.from_dict(_job_payload(job_type="training.bogus"))
    with pytest.raises(ValueError, match="object"):
        TrainingJobEnvelope.from_dict("not-a-dict")  # type: ignore[arg-type]


def test_envelope_is_tolerant_of_wrong_typed_optional_fields() -> None:
    job = TrainingJobEnvelope.from_dict(
        _job_payload(params="not-a-dict", metadata=42, owner_user_id="", owner_org_id=None)
    )
    assert job.params == {}
    assert job.metadata == {}
    assert job.owner_user_id == "local-user"
    assert job.owner_org_id == "local-org"


def test_worker_token_rides_principal_headers() -> None:
    job = TrainingJobEnvelope.from_dict(_job_payload())
    settings = _runtime_settings(control_worker_token="secret-token")
    assert job.principal_headers(settings)["X-Ultra-Worker-Token"] == "secret-token"


# ------------------------------------------------------------------ keepalive


class FakeLeaseClient:
    def __init__(self, conflict_on_renewal: int | None = None) -> None:
        self.renewals = 0
        self.released: list[str] = []
        self._conflict_on_renewal = conflict_on_renewal

    def renew_lease(self, lease: TrainingJobLease, *, ttl_seconds: float) -> TrainingJobLease:
        self.renewals += 1
        if self._conflict_on_renewal is not None and self.renewals >= self._conflict_on_renewal:
            raise TrainingLeaseConflict("lease handed to another worker")
        return TrainingJobLease(
            job_id=lease.job_id,
            worker_id=lease.worker_id,
            lease_token=f"token-{self.renewals}",
        )

    def release_lease(self, lease: TrainingJobLease) -> None:
        self.released.append(lease.lease_token)


class FakeMessage:
    def __init__(self) -> None:
        self.in_progress_count = 0

    async def in_progress(self) -> None:
        self.in_progress_count += 1


def test_keepalive_interval_is_a_third_of_ttl() -> None:
    assert training_keepalive_interval(600.0) == pytest.approx(200.0)
    assert training_keepalive_interval(0.0) == pytest.approx(0.1)


def test_keepalive_renews_lease_and_schedules_in_progress_from_os_thread() -> None:
    async def scenario():
        client = FakeLeaseClient()
        message = FakeMessage()
        lease = TrainingJobLease(job_id="job-1", worker_id="w-1", lease_token="token-0")
        keepalive = TrainingLeaseKeepalive(
            lease,
            client,
            loop=asyncio.get_running_loop(),
            ttl_seconds=1.0,
            message=message,
            interval_seconds=0.05,
        )
        keepalive.start()
        await asyncio.sleep(0.4)
        keepalive.stop()
        return client, message, keepalive

    client, message, keepalive = asyncio.run(scenario())
    assert client.renewals >= 2
    assert message.in_progress_count >= 1
    # The rotating token was picked up by the sole renewer.
    assert keepalive.current_lease().lease_token == f"token-{client.renewals}"
    assert keepalive.lost is False


def test_keepalive_409_sets_killed_flag_and_fires_on_lost() -> None:
    async def scenario():
        lost = asyncio.Event()
        client = FakeLeaseClient(conflict_on_renewal=1)
        lease = TrainingJobLease(job_id="job-2", worker_id="w-1", lease_token="token-0")
        keepalive = TrainingLeaseKeepalive(
            lease,
            client,
            loop=asyncio.get_running_loop(),
            ttl_seconds=1.0,
            interval_seconds=0.02,
            on_lost=lost.set,
        )
        keepalive.start()
        await asyncio.wait_for(lost.wait(), timeout=2.0)
        keepalive.stop()
        return client, keepalive

    client, keepalive = asyncio.run(scenario())
    assert keepalive.lost is True
    assert client.renewals == 1


def test_keepalive_survives_transient_renewal_errors() -> None:
    class FlakyClient(FakeLeaseClient):
        def renew_lease(self, lease, *, ttl_seconds):
            self.renewals += 1
            if self.renewals == 1:
                raise OSError("control plane unreachable")
            return super().renew_lease(lease, ttl_seconds=ttl_seconds)

    async def scenario():
        client = FlakyClient()
        lease = TrainingJobLease(job_id="job-3", worker_id="w-1", lease_token="token-0")
        keepalive = TrainingLeaseKeepalive(
            lease,
            client,
            loop=asyncio.get_running_loop(),
            ttl_seconds=1.0,
            interval_seconds=0.03,
        )
        keepalive.start()
        await asyncio.sleep(0.25)
        keepalive.stop()
        return client, keepalive

    client, keepalive = asyncio.run(scenario())
    assert client.renewals >= 2
    assert keepalive.lost is False


# ------------------------------------------------------------------- registry


def test_get_adapter_unknown_model_key_raises() -> None:
    with pytest.raises(NoAdapterRegistered, match="model_key=unknown_model"):
        get_adapter("unknown_model")
    with pytest.raises(NoAdapterRegistered):
        get_adapter("")


def test_get_adapter_resolves_rarespot() -> None:
    adapter = get_adapter("yolov5_rarespot")
    assert adapter.model_key == "yolov5_rarespot"
    # Every manifest-declared extra defense must resolve to an implementation.
    for name in RARESPOT_MANIFEST.leakage_defenses_extra:
        assert name in adapter.implemented_leakage_checks


def test_materialize_context_validates_purpose() -> None:
    # The plan-3.5 purposes are all legal - including the ASSEMBLE mix.
    MaterializeContext(purpose="finetune_mix")
    with pytest.raises(ValueError, match="purpose"):
        MaterializeContext(purpose="bogus_purpose")


# ---------------------------------------------------------------- worker flow


class FakeAck:
    def __init__(self, payload: dict) -> None:
        self.data = json.dumps(payload).encode("utf-8")
        self.acked = 0
        self.naked = 0
        self.termed = 0
        self.nak_delays: list[float | None] = []
        self.in_progress_count = 0

    async def ack(self) -> None:
        self.acked += 1

    async def nak(self, delay=None) -> None:
        self.naked += 1
        self.nak_delays.append(delay)

    async def term(self) -> None:
        self.termed += 1

    async def in_progress(self) -> None:
        self.in_progress_count += 1


class FakeControlClient:
    def __init__(self, *, lease_error: Exception | None = None) -> None:
        self.lease_error = lease_error
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.statuses: list[tuple[str, str]] = []
        self.events: list[str] = []
        self.model_status_payloads: list[dict] = []
        self.gold_payloads: list[dict] = []
        self.benchmark_payloads: list[dict] = []
        self.registered_versions: list[tuple[str, dict]] = []

    def acquire_lease(self, job_id: str, *, ttl_seconds: float) -> TrainingJobLease:
        if self.lease_error is not None:
            raise self.lease_error
        self.acquired.append(job_id)
        return TrainingJobLease(job_id=job_id, worker_id="w-test", lease_token="token-0")

    def renew_lease(self, lease: TrainingJobLease, *, ttl_seconds: float) -> TrainingJobLease:
        return lease

    def release_lease(self, lease: TrainingJobLease) -> None:
        self.released.append(lease.lease_token)

    def append_event(self, job_id: str, **kwargs) -> None:
        self.events.append(str(kwargs.get("event_type")))

    def update_job_status(
        self, job_id: str, *, status: str, error: str = "", output_summary=None
    ) -> None:
        self.statuses.append((status, error))

    def upsert_model_status(self, job_id: str, payload: dict) -> None:
        self.model_status_payloads.append(payload)

    def finalize_gold_set(self, job_id: str, payload: dict) -> None:
        self.gold_payloads.append(payload)

    def insert_benchmark_run(self, job_id: str, payload: dict) -> None:
        self.benchmark_payloads.append(payload)

    def register_model_version(self, model_key: str, payload: dict) -> None:
        self.registered_versions.append((model_key, payload))


async def _no_heartbeat(*_args, **_kwargs) -> None:
    return None


def _worker(client: FakeControlClient, **settings_overrides) -> NATSTrainingWorker:
    return NATSTrainingWorker(
        settings=_worker_settings(**settings_overrides),
        client_factory=lambda job: client,
        worker_heartbeat_func=_no_heartbeat,
    )


class FakeGoldAdapter(ModelAdapter):
    """Registered under the rarespot key for worker-flow tests (monkeypatched)."""

    model_key = "yolov5_rarespot"
    implemented_leakage_checks = ("aerial_geospatial_overlap",)

    def __init__(self, *, items: list[dict], extra_violations: list[dict] | None = None) -> None:
        self._items = items
        self._extra_violations = list(extra_violations or [])
        self.extra_called_with: dict | None = None

    def materialize_dataset(self, ctx: MaterializeContext) -> dict:
        return {"model_key": self.model_key, "purpose": ctx.purpose, "items": list(self._items)}

    def train(self, ctx: TrainContext) -> dict:
        raise NotImplementedError("GPU execution runs on the GPU-node worker deployment")

    def evaluate(self, weights_uri, gold_manifest, slice=None):  # noqa: A002
        return {
            "schema": "detection.v1",
            "aggregate": {"map50": 0.5},
            "per_class": {},
            "per_slice": {},
            "eval": {"passes": 3, "kernel": "yolov5_two_pass/v1", "wall_clock_s": 0.1},
        }

    def smoke_test(self, weights_uri, gold_sample) -> None:
        return None

    def extra_leakage_checks(self, train_pool, gold_items, *, params=None):
        self.extra_called_with = dict(params or {})
        return list(self._extra_violations)


def _gold_items() -> list[dict]:
    return [
        {
            "item_id": "frame-1",
            "source_ref": {"bisque_image_id": "00-abc", "frame_stem": "frame-1"},
            "content_sha256": "a" * 64,
            "gt_label_sha256": "b" * 64,
            "phash": "0f0f0f0f0f0f0f0f",
            "label_stats": {
                "tile_count": 165,
                "box_count": 3,
                "per_class_box_count": {"burrow": 3},
            },
        },
        {
            "item_id": "frame-2",
            "source_ref": {"bisque_image_id": "00-def", "frame_stem": "frame-2"},
            "content_sha256": "c" * 64,
            "gt_label_sha256": "d" * 64,
            "phash": "f0f0f0f0f0f0f0f0",
            "label_stats": {
                "tile_count": 165,
                "box_count": 2,
                "per_class_box_count": {"prairie_dog": 2},
            },
        },
    ]


def test_unknown_model_key_fails_terminally_before_lease() -> None:
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(model_key="no_such_model"))
    asyncio.run(worker._process_message(message))
    assert client.acquired == []  # hard-fail BEFORE leasing
    assert client.statuses[0][0] == "failed"
    assert "no adapter registered" in client.statuses[0][1]
    assert client.events == ["training.job.adapter_missing"]
    assert message.termed == 1
    assert message.acked == 0


def test_malformed_payload_is_terminated_not_redelivered() -> None:
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(job_type="training.not_a_phase"))
    asyncio.run(worker._process_message(message))
    assert message.termed == 1
    assert client.statuses == []


def test_lease_conflict_naks_for_redelivery() -> None:
    client = FakeControlClient(lease_error=TrainingLeaseConflict("held elsewhere"))
    worker = _worker(client, ack_wait_seconds=120.0, redelivery_delay_seconds=1.0)
    message = FakeAck(_job_payload())
    asyncio.run(worker._process_message(message))
    assert message.naked == 1
    assert message.acked == 0
    assert client.statuses == []
    # Conflict retries are spaced a whole AckWait apart, NOT the short
    # redelivery delay: a crashed worker's stale lease persists up to the
    # lease TTL, and MaxDeliver-spaced retries must outlast it.
    assert message.nak_delays == [120.0]


def test_lease_unavailable_naks_with_short_delay() -> None:
    client = FakeControlClient(lease_error=TrainingLeaseUnavailable("control plane down"))
    worker = _worker(client, ack_wait_seconds=120.0, redelivery_delay_seconds=1.0)
    message = FakeAck(_job_payload())
    asyncio.run(worker._process_message(message))
    assert message.naked == 1
    assert message.acked == 0
    assert client.statuses == []
    assert message.nak_delays == [1.0]


def test_finetune_not_implemented_fails_cleanly_with_reason(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")  # force default registration before patching
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", FakeGoldAdapter(items=[]))
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(job_type="training.finetune"))
    asyncio.run(worker._process_message(message))
    assert client.statuses[0] == ("running", "")
    status, error = client.statuses[-1]
    assert status == "failed"
    assert "GPU-node worker deployment" in error
    assert client.acquired == ["training_job_1"]
    assert client.released == ["token-0"]
    assert message.acked == 1


def test_gold_freeze_happy_path_finalizes_frozen(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")
    adapter = FakeGoldAdapter(items=_gold_items())
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", adapter)
    client = FakeControlClient()
    worker = _worker(client)
    params = {
        "exclusion_inventory_uri": "/srv/training-store/inventory.json",
        "inventory_sha256": "e" * 64,
        "default_slice": "prior_train",
        "split_manifest_uri": "store:/gold/yolov5_rarespot/gs-1/manifest.json",
    }
    message = FakeAck(_job_payload(job_type="training.gold_freeze", params=params))
    asyncio.run(worker._process_message(message))

    assert len(client.gold_payloads) == 1
    payload = client.gold_payloads[0]
    assert payload["status"] == "frozen"
    assert payload["item_count"] == 2
    assert payload["failure_reasons"] == []
    assert payload["content_hash"] == compute_gold_content_hash(_gold_items())
    assert payload["provenance"]["inventory_sha256"] == "e" * 64
    assert all(item["slice"] == "prior_train" for item in payload["items"])
    assert payload["label_stats"]["per_class_box_count"] == {"burrow": 3, "prairie_dog": 2}
    # The adapter extras received the freeze params (inventory URI + sha).
    assert adapter.extra_called_with is not None
    assert adapter.extra_called_with["inventory_sha256"] == "e" * 64
    assert client.statuses[-1][0] == "succeeded"
    assert message.acked == 1


def test_gold_freeze_violations_fail_closed(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")
    violation = {
        "check": "phash_near_dup",
        "item_id": "frame-2",
        "reason": "near-dup of trained tile",
    }
    adapter = FakeGoldAdapter(items=_gold_items(), extra_violations=[violation])
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", adapter)
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(job_type="training.gold_freeze", params={}))
    asyncio.run(worker._process_message(message))

    payload = client.gold_payloads[0]
    assert payload["status"] == "failed"
    assert violation in payload["failure_reasons"]
    status, error = client.statuses[-1]
    assert status == "failed"
    assert "leakage violation" in error


def test_gold_freeze_requires_phash_is_structural(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")
    items = _gold_items()
    items[0]["phash"] = None  # a buggy materializer cannot skip defense 3 silently
    adapter = FakeGoldAdapter(items=items)
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", adapter)
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(job_type="training.gold_freeze", params={}))
    asyncio.run(worker._process_message(message))

    payload = client.gold_payloads[0]
    assert payload["status"] == "failed"
    assert any(reason["check"] == "requires_phash" for reason in payload["failure_reasons"])


def test_sync_phase_upserts_model_status_counts(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")
    adapter = FakeGoldAdapter(items=_gold_items())
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", adapter)
    client = FakeControlClient()
    worker = _worker(client)
    message = FakeAck(_job_payload(job_type="training.sync"))
    asyncio.run(worker._process_message(message))

    assert len(client.model_status_payloads) == 1
    payload = client.model_status_payloads[0]
    assert payload["model_key"] == "yolov5_rarespot"
    assert payload["reviewed_images"] == 2
    assert payload["retrain_gate_counts"] == {
        "reviewed_images": 2,
        "total_objects": 5,
        "per_class": {"burrow": 3, "prairie_dog": 2},
    }
    assert client.statuses[-1][0] == "succeeded"


def test_benchmark_phase_inserts_benchmark_run(monkeypatch) -> None:
    get_adapter("yolov5_rarespot")
    adapter = FakeGoldAdapter(items=[])
    monkeypatch.setitem(ADAPTERS, "yolov5_rarespot", adapter)
    client = FakeControlClient()
    worker = _worker(client)
    params = {
        "weights_uri": "/srv/model-store/weights/v0.pt",
        "gold_manifest": {"slices": {}},
        "model_version_id": "yolov5_rarespot-v0",
        "gold_set_id": "gs-1",
        "gold_set_content_hash": "f" * 64,
    }
    message = FakeAck(_job_payload(job_type="training.benchmark", params=params))
    asyncio.run(worker._process_message(message))

    assert len(client.benchmark_payloads) == 1
    payload = client.benchmark_payloads[0]
    assert payload["model_version_id"] == "yolov5_rarespot-v0"
    assert payload["gold_set_content_hash"] == "f" * 64
    assert payload["metric_schema"] == "detection.v1"
    assert payload["kernel_version"] == "yolov5_two_pass/v1"
    assert payload["metrics"]["aggregate"]["map50"] == 0.5
    assert client.statuses[-1][0] == "succeeded"


# --------------------------------------------------------------- gold hashing


def test_gold_content_hash_is_order_independent_and_label_sensitive() -> None:
    items = _gold_items()
    reversed_hash = compute_gold_content_hash(list(reversed(items)))
    assert compute_gold_content_hash(items) == reversed_hash
    mutated = [dict(item) for item in items]
    mutated[0]["gt_label_sha256"] = "9" * 64  # an ignore-box edit moves the label sha
    assert compute_gold_content_hash(mutated) != reversed_hash
    assert len(reversed_hash) == 64


def test_generic_defenses_catch_train_gold_overlap() -> None:
    items = _gold_items()
    train_pool = [
        {
            "item_id": "frame-1",
            "source_ref": {"bisque_image_id": "00-abc"},
            "content_sha256": "z" * 64,
        },
        {
            "item_id": "other",
            "source_ref": {"bisque_image_id": "00-zzz"},
            "content_sha256": "c" * 64,
        },
    ]
    violations = run_generic_leakage_defenses(RARESPOT_MANIFEST, train_pool, items)
    checks = {violation["check"] for violation in violations}
    assert "source_id_disjointness" in checks  # frame-1 id in both pools
    assert "content_hash_disjointness" in checks  # frame-2 sha in the train pool
    assert run_generic_leakage_defenses(RARESPOT_MANIFEST, [], items) == []


# ------------------------------------------------------------------- settings


def test_worker_settings_env_defaults(monkeypatch) -> None:
    for name in (
        "ULTRA_CONTROL_NATS_TRAINING_JOBS_SUBJECT",
        "ULTRA_CONTROL_NATS_TRAINING_WORKER_DURABLE",
        "ULTRA_TRAINING_NATS_ACK_WAIT_SECONDS",
        "ULTRA_TRAINING_CONTROL_LEASE_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = TrainingWorkerSettings.from_env(_runtime_settings())
    assert settings.jobs_subject == "ultra.training.jobs"
    assert settings.durable == "ultra-training-worker"
    # Short AckWait bounds crash recovery (in_progress() keepalive touches
    # carry long jobs); the lease TTL is the crash-takeover window.
    assert settings.ack_wait_seconds == 300.0
    assert settings.lease_ttl_seconds == 3600.0
    assert settings.max_deliver == 20

    monkeypatch.setenv("ULTRA_CONTROL_NATS_TRAINING_JOBS_SUBJECT", "ultra.training.jobs.test")
    monkeypatch.setenv("ULTRA_CONTROL_NATS_TRAINING_WORKER_DURABLE", "ultra-training-worker-test")
    settings = TrainingWorkerSettings.from_env(_runtime_settings())
    assert settings.jobs_subject == "ultra.training.jobs.test"
    assert settings.durable == "ultra-training-worker-test"


def test_control_client_lease_urls_and_conflict(monkeypatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    def fake_request(self, url, method, payload, *, timeout=None):
        calls.append((method, url, payload))
        return {"job_id": "j1", "worker_id": "w", "lease_token": "t-1"}

    monkeypatch.setattr(TrainingControlClient, "_request", fake_request)
    client = TrainingControlClient(_runtime_settings(), worker_id="w")
    lease = client.acquire_lease("j 1", ttl_seconds=60)
    assert lease is not None and lease.lease_token == "t-1"
    method, url, payload = calls[0]
    assert method == "POST"
    assert url == "https://control.example.test/v2/training/jobs/j%201/lease"
    assert payload == {"worker_id": "w", "ttl_seconds": 60}

    client.update_job_status("j1", status="running")
    assert calls[-1][0] == "PATCH"
    assert calls[-1][1].endswith("/v2/training/jobs/j1/status")
    client.finalize_gold_set("j1", {"status": "frozen"})
    assert calls[-1][1].endswith("/v2/training/jobs/j1/gold-result")
    client.insert_benchmark_run("j1", {})
    assert calls[-1][1].endswith("/v2/training/jobs/j1/benchmark-result")
    client.upsert_model_status("j1", {})
    assert calls[-1][1].endswith("/v2/training/jobs/j1/status-result")
    client.register_model_version("yolov5_rarespot", {})
    assert calls[-1][1].endswith("/v2/training/models/yolov5_rarespot/versions")
