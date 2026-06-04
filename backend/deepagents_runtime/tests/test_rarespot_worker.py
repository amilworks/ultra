import asyncio
import contextlib
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import ControlPlaneRunLease, RunLeaseConflict
from ultra_deepagents.rarespot.store import EventPublishError, NATSRunEventStore
from ultra_deepagents.rarespot.worker import (
    RareSpotJobRunner,
    _handle_rarespot_cancel_payload,
    _subscribe_rarespot_cancels,
    host_readable_local_paths,
    process_rarespot_job,
)
from ultra_deepagents.rarespot.worker import (
    build_rarespot_store,
    build_rarespot_consumer_config,
    apply_rarespot_metadata_overrides,
    rarespot_ack_extension_interval,
    run_worker_loop,
    subscribe_rarespot_jobs,
)


class FakeStore:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str, str]] = []
        self.events: list[tuple[str, str, str]] = []
        self.event_records: list[dict] = []
        self.artifacts: list[dict] = []
        self.heartbeat_seen: asyncio.Event | None = None

    async def update_run_status(self, run_id: str, status: str, *, response_text: str = "", error: str = ""):
        self.statuses.append((run_id, status, error))

    async def append_run_event(
        self,
        run_id: str,
        thread_id: str,
        event_kind: str,
        message: str,
        payload: dict | None = None,
        *,
        event_id: str | None = None,
    ):
        self.events.append((run_id, event_kind, message))
        self.event_records.append(
            {
                "event_id": event_id,
                "run_id": run_id,
                "thread_id": thread_id,
                "event_kind": event_kind,
                "message": message,
                "payload": payload or {},
            }
        )
        if event_kind == "run.heartbeat" and self.heartbeat_seen is not None:
            self.heartbeat_seen.set()

    async def create_artifact(self, artifact: dict):
        artifact = dict(artifact)
        artifact["artifact_id"] = f"artifact-{len(self.artifacts) + 1}"
        self.artifacts.append(artifact)
        return artifact

    async def get_run(self, run_id: str) -> dict:
        return {"run_id": run_id, "status": "queued"}


class FakeAck:
    def __init__(self) -> None:
        self.acked = False
        self.nacked = False
        self.in_progress_count = 0
        self.progress_seen: asyncio.Event | None = None

    async def ack(self) -> None:
        self.acked = True

    async def nak(self, delay=None) -> None:
        _ = delay
        self.nacked = True

    async def in_progress(self) -> None:
        self.in_progress_count += 1
        if self.progress_seen is not None:
            self.progress_seen.set()


async def no_control_plane_run_lease(_run_id: str, _settings: RuntimeSettings):
    return None


async def no_control_plane_run_status(_run_id: str, _settings: RuntimeSettings):
    return None


class SuccessfulRunner(RareSpotJobRunner):
    async def run(self, job: dict) -> dict:
        output_dir = Path(job["artifact_root"]) / job["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / "predictions.json"
        prediction_path.write_text("{}", encoding="utf-8")
        return {
            "counts_by_class": {"prairie_dog": 2, "burrow": 1},
            "top_spectral_review_candidates": [{"file_name": "tile.png", "score": 2.0}],
            "artifacts": [
                {
                    "kind": "json",
                    "path": "predictions.json",
                    "title": "Predictions",
                    "mime_type": "application/json",
                    "category": "prediction",
                }
            ],
        }


class FailingRunner(RareSpotJobRunner):
    async def run(self, job: dict) -> dict:
        raise RuntimeError("model load failed")


class ProgressGatedRunner(RareSpotJobRunner):
    def __init__(self, progress_seen: asyncio.Event) -> None:
        self.progress_seen = progress_seen

    async def run(self, job: dict) -> dict:
        await asyncio.wait_for(self.progress_seen.wait(), timeout=1.0)
        return {
            "counts_by_class": {},
            "top_spectral_review_candidates": [],
            "artifacts": [],
        }


class HeartbeatGatedRunner(RareSpotJobRunner):
    def __init__(self, heartbeat_seen: asyncio.Event) -> None:
        self.heartbeat_seen = heartbeat_seen

    async def run(self, job: dict) -> dict:
        await asyncio.wait_for(self.heartbeat_seen.wait(), timeout=1.0)
        return {
            "counts_by_class": {},
            "top_spectral_review_candidates": [],
            "artifacts": [],
        }


class CancellableRunner(RareSpotJobRunner):
    def __init__(self, cancelled: asyncio.Event) -> None:
        self.cancelled = cancelled

    async def run(self, job: dict) -> dict:
        _ = job
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class StartedCancellableRunner(RareSpotJobRunner):
    def __init__(self, started: asyncio.Event, cancelled: asyncio.Event) -> None:
        self.started = started
        self.cancelled = cancelled

    async def run(self, job: dict) -> dict:
        _ = job
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FakeNATSConnection:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, object]] = []

    async def subscribe(self, subject: str, cb):
        self.subscriptions.append((subject, cb))
        return {"subject": subject, "callback": cb}


class FakeCancelMessage:
    def __init__(self, payload: bytes) -> None:
        self.data = payload


class FakeConsumerInfo:
    def __init__(self, config: object) -> None:
        self.config = config


class FakeJetStream:
    def __init__(self, *, existing_config: object | None = None, add_error: Exception | None = None) -> None:
        self.existing_config = existing_config
        self.add_error = add_error
        self.added_configs: list[object] = []
        self.deleted_consumers: list[tuple[str, str]] = []
        self.pull_calls: list[dict] = []

    async def consumer_info(self, stream: str, consumer: str) -> FakeConsumerInfo:
        if self.existing_config is None:
            from nats.js.errors import NotFoundError

            raise NotFoundError
        return FakeConsumerInfo(self.existing_config)

    async def add_consumer(self, stream: str, *, config: object):
        self.added_configs.append(config)
        if self.add_error is not None:
            error = self.add_error
            self.add_error = None
            raise error
        return FakeConsumerInfo(config)

    async def delete_consumer(self, stream: str, consumer: str) -> bool:
        self.deleted_consumers.append((stream, consumer))
        return True

    async def pull_subscribe(self, subject: str, *, durable: str, stream: str, config: object):
        self.pull_calls.append(
            {
                "subject": subject,
                "durable": durable,
                "stream": stream,
                "config": config,
            }
        )
        return {"subscription": "ok"}


class CapturingEventJetStream:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict, dict]] = []

    async def publish(self, subject: str, data: bytes, **kwargs):
        import json

        self.published.append((subject, json.loads(data.decode("utf-8")), kwargs.get("headers") or {}))


class FailingEventJetStream:
    async def publish(self, subject: str, data: bytes, **_kwargs):
        _ = subject, data
        raise RuntimeError("nats publish unavailable")


def test_nats_run_event_store_publishes_normalized_rarespot_events(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_events_subject="ultra.runs.events",
    )
    js = CapturingEventJetStream()
    store = NATSRunEventStore(js, settings)
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    result = asyncio.run(
        process_rarespot_job(
            payload,
            store=store,
            runner=SuccessfulRunner(),
            message=ack,
            run_status_func=no_control_plane_run_status,
            run_lease_func=no_control_plane_run_lease,
        )
    )

    assert ack.acked is True
    assert result["artifact_ids"] == ["artifact_run-1_rarespot_000001"]
    subjects = [subject for subject, _event, _headers in js.published]
    assert subjects == ["ultra.runs.events", "ultra.runs.events", "ultra.runs.events"]
    events = [event for _subject, event, _headers in js.published]
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "artifact.created",
        "run.completed",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert [event["event_id"] for event in events] == [
        "evt_run-1_rarespot_000001",
        "evt_run-1_rarespot_000002",
        "evt_run-1_rarespot_000003",
    ]
    artifact_payload = events[1]["payload"]
    assert artifact_payload["artifact_id"] == "artifact_run-1_rarespot_000001"
    assert artifact_payload["tool_name"] == "rarespot_ecology_inference"
    assert artifact_payload["kind"] == "json"
    assert artifact_payload["path"] == "predictions.json"
    completed_payload = events[2]["payload"]
    assert completed_payload["counts_by_class"] == {"prairie_dog": 2, "burrow": 1}
    assert "response_text" in completed_payload


def test_nats_run_event_store_publishes_events_with_deterministic_jetstream_message_id(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_events_subject="ultra.runs.events",
    )
    js = CapturingEventJetStream()
    store = NATSRunEventStore(js, settings)

    asyncio.run(
        store.append_run_event(
            "run-1",
            "thread-1",
            "run.started",
            "RareSpot ecology inference started.",
        )
    )

    _subject, event, headers = js.published[0]
    assert event["event_id"] == "evt_run-1_rarespot_000001"
    assert headers["Nats-Msg-Id"] == "event:evt_run-1_rarespot_000001"


def test_process_rarespot_job_naks_when_nats_event_publish_fails(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_events_subject="ultra.runs.events",
    )
    store = NATSRunEventStore(FailingEventJetStream(), settings)
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    try:
        asyncio.run(
            process_rarespot_job(
                payload,
                store=store,
                runner=SuccessfulRunner(),
                message=ack,
                run_status_func=no_control_plane_run_status,
                run_lease_func=no_control_plane_run_lease,
            )
        )
    except EventPublishError:
        pass
    else:
        raise AssertionError("process_rarespot_job should raise event publish errors")

    assert ack.acked is False
    assert ack.nacked is True


def test_build_rarespot_store_uses_nats_events_when_database_url_is_missing():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_database_url="",
    )

    store = build_rarespot_store(CapturingEventJetStream(), settings)

    assert isinstance(store, NATSRunEventStore)


def test_apply_rarespot_metadata_overrides_updates_inference_config(tmp_path: Path):
    from ultra_deepagents.rarespot.config import RareSpotConfig

    config = RareSpotConfig(
        weights_path=tmp_path / "weights.pt",
        yolov5_path=tmp_path / "yolov5",
        artifact_root=tmp_path / "artifacts",
        allowed_input_roots=(tmp_path,),
        conf=0.25,
        iou=0.45,
        tile_overlap=0.25,
        tile_size=1280,
        spectral=True,
    )

    updated = apply_rarespot_metadata_overrides(
        config,
        {
            "rarespot_config": {
                "confidence_threshold": 0.6,
                "iou_threshold": 0.35,
                "tile_overlap": 0.1,
                "image_size": 960,
                "spectral": False,
            }
        },
    )

    assert updated.conf == 0.6
    assert updated.iou == 0.35
    assert updated.tile_overlap == 0.1
    assert updated.tile_size == 960
    assert updated.spectral is False
    assert updated.weights_path == config.weights_path


def test_host_readable_local_paths_drop_docker_workspace_paths():
    host_upload = "/tmp/ultra/data/uploads/file-1__prairie.jpg"
    assert host_readable_local_paths(
        [
            "/workspace/staged_uploads/file-1/prairie.jpg",
            host_upload,
        ]
    ) == [host_upload]


def test_process_rarespot_job_success_registers_artifacts_and_acks(tmp_path: Path):
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {"local_paths": [str(tmp_path)]},
    }

    result = asyncio.run(
        process_rarespot_job(
            payload,
            store=store,
            runner=SuccessfulRunner(),
            message=ack,
            run_status_func=no_control_plane_run_status,
            run_lease_func=no_control_plane_run_lease,
        )
    )

    assert ack.acked is True
    assert store.statuses[0] == ("run-1", "running", "")
    assert store.statuses[-1][1] == "succeeded"
    assert store.artifacts[0]["tool_name"] == "rarespot_ecology_inference"
    assert result["artifact_ids"] == ["artifact-1"]
    assert result["counts_by_class"] == {"prairie_dog": 2, "burrow": 1}


def test_process_rarespot_job_posts_worker_heartbeats(tmp_path: Path):
    heartbeat_calls: list[dict[str, object]] = []

    async def post_worker_heartbeat(settings, *, status, current_run_id=None, metadata=None):
        heartbeat_calls.append(
            {
                "worker_id": settings.worker_id,
                "worker_kind": settings.worker_kind,
                "status": status,
                "current_run_id": current_run_id,
                "metadata": metadata or {},
            }
        )

    async def scenario():
        store = FakeStore()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            rarespot_worker_id="rarespot-worker-1",
            rarespot_worker_kind="rarespot",
            rarespot_artifact_root=str(tmp_path / "artifacts"),
        )
        ack = FakeAck()
        payload = {
            "run_id": "run-rarespot",
            "thread_id": "thread-1",
            "artifact_root": str(tmp_path / "artifacts"),
            "metadata": {"local_paths": [str(tmp_path)]},
        }
        result = await process_rarespot_job(
            payload,
            store=store,
            runner=SuccessfulRunner(),
            message=ack,
            settings=settings,
            run_status_func=no_control_plane_run_status,
            run_lease_func=no_control_plane_run_lease,
            worker_heartbeat_func=post_worker_heartbeat,
            ack_progress_interval_seconds=0,
        )
        return result, ack

    result, ack = asyncio.run(scenario())

    assert result["run_id"] == "run-rarespot"
    assert ack.acked is True
    assert [call["status"] for call in heartbeat_calls] == ["busy", "idle"]
    assert heartbeat_calls[0]["worker_id"] == "rarespot-worker-1"
    assert heartbeat_calls[0]["worker_kind"] == "rarespot"
    assert heartbeat_calls[0]["current_run_id"] == "run-rarespot"
    assert heartbeat_calls[0]["metadata"] == {"active_tasks": 1}
    assert heartbeat_calls[1]["current_run_id"] is None
    assert heartbeat_calls[1]["metadata"] == {"active_tasks": 0}


def test_process_rarespot_job_claims_and_releases_control_plane_run_lease(tmp_path: Path):
    calls: list[tuple[str, str]] = []

    async def acquire_lease(run_id: str, settings: RuntimeSettings):
        calls.append(("acquire", run_id))
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id=settings.worker_id,
            lease_token="lease-token-1",
        )

    async def renew_lease(lease: ControlPlaneRunLease, _settings: RuntimeSettings):
        calls.append(("renew", lease.lease_token))
        return lease

    async def release_lease(lease: ControlPlaneRunLease, _settings: RuntimeSettings):
        calls.append(("release", lease.lease_token))

    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        worker_id="rarespot-worker-a",
    )
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    result = asyncio.run(
        process_rarespot_job(
            payload,
            store=store,
            runner=SuccessfulRunner(),
            message=ack,
            settings=settings,
            run_status_func=no_control_plane_run_status,
            run_lease_func=acquire_lease,
            renew_run_lease_func=renew_lease,
            release_run_lease_func=release_lease,
            ack_progress_interval_seconds=0,
        )
    )

    assert ack.acked is True
    assert ack.nacked is False
    assert result["run_id"] == "run-1"
    assert calls == [
        ("acquire", "run-1"),
        ("release", "lease-token-1"),
    ]


def test_process_rarespot_job_naks_without_compute_when_control_plane_run_lease_is_held(tmp_path: Path):
    class RunnerShouldNotStart(RareSpotJobRunner):
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, job: dict) -> dict:
            _ = job
            self.calls += 1
            return {}

    async def acquire_lease(_run_id: str, _settings: RuntimeSettings):
        raise RunLeaseConflict("run already leased")

    runner = RunnerShouldNotStart()
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        worker_id="rarespot-worker-a",
    )
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    result = asyncio.run(
        process_rarespot_job(
            payload,
            store=store,
            runner=runner,
            message=ack,
            settings=settings,
            run_status_func=no_control_plane_run_status,
            run_lease_func=acquire_lease,
            ack_progress_interval_seconds=0.01,
        )
    )

    assert result == {"run_id": "run-1", "lease_conflict": True}
    assert runner.calls == 0
    assert ack.acked is False
    assert ack.nacked is True
    assert store.statuses == []
    assert store.events == []


def test_process_rarespot_job_naks_and_stops_compute_when_control_plane_run_lease_is_lost(tmp_path: Path):
    calls: list[tuple[str, str]] = []
    cancelled = asyncio.Event()

    async def acquire_lease(run_id: str, settings: RuntimeSettings):
        calls.append(("acquire", run_id))
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id=settings.worker_id,
            lease_token="lease-token-1",
        )

    async def renew_lost_lease(lease: ControlPlaneRunLease, _settings: RuntimeSettings):
        calls.append(("renew", lease.lease_token))
        raise RunLeaseConflict("lease moved to another worker")

    async def release_lease(lease: ControlPlaneRunLease, _settings: RuntimeSettings):
        calls.append(("release", lease.lease_token))

    async def scenario() -> tuple[FakeAck, FakeStore]:
        settings = RuntimeSettings(
            openai_base_url="http://127.0.0.1:8003/v1",
            openai_model="deepseek_v4",
            worker_id="rarespot-worker-a",
        )
        store = FakeStore()
        ack = FakeAck()
        payload = {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "artifact_root": str(tmp_path),
            "metadata": {},
        }
        await asyncio.wait_for(
            process_rarespot_job(
                payload,
                store=store,
                runner=CancellableRunner(cancelled),
                message=ack,
                settings=settings,
                run_status_func=no_control_plane_run_status,
                run_lease_func=acquire_lease,
                renew_run_lease_func=renew_lost_lease,
                release_run_lease_func=release_lease,
                ack_progress_interval_seconds=0.01,
            ),
            timeout=0.5,
        )
        return ack, store

    ack, store = asyncio.run(scenario())

    assert cancelled.is_set()
    assert ack.acked is False
    assert ack.nacked is True
    assert calls == [
        ("acquire", "run-1"),
        ("renew", "lease-token-1"),
        ("release", "lease-token-1"),
    ]
    assert ("run-1", "run.failed", "RareSpot ecology inference failed.") not in store.events


def test_process_rarespot_job_acks_terminal_control_plane_run_without_compute(tmp_path: Path):
    class RunnerShouldNotStart(RareSpotJobRunner):
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, job: dict) -> dict:
            _ = job
            self.calls += 1
            return {}

    async def canceled_status(_run_id: str, _settings: RuntimeSettings):
        return "canceled"

    async def acquire_lease_should_not_run(_run_id: str, _settings: RuntimeSettings):
        raise AssertionError("lease should not be acquired for a terminal run")

    runner = RunnerShouldNotStart()
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    result = asyncio.run(
        process_rarespot_job(
            payload,
            store=store,
            runner=runner,
            message=ack,
            run_status_func=canceled_status,
            run_lease_func=acquire_lease_should_not_run,
        )
    )

    assert result == {"run_id": "run-1", "skipped_status": "canceled"}
    assert runner.calls == 0
    assert ack.acked is True
    assert ack.nacked is False
    assert store.statuses == []
    assert store.events == []


def test_run_worker_loop_validates_rarespot_config_before_connecting(monkeypatch, tmp_path: Path):
    def fail_validate(_settings: RuntimeSettings) -> None:
        raise RuntimeError("RareSpot weights not found")

    async def fail_connect(*_args, **_kwargs):
        raise AssertionError("worker should validate config before connecting to NATS")

    monkeypatch.setattr("ultra_deepagents.rarespot.worker.validate_rarespot_runtime_config", fail_validate)
    monkeypatch.setattr("nats.connect", fail_connect)

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        rarespot_artifact_root=str(tmp_path / "artifacts"),
    )

    try:
        asyncio.run(run_worker_loop(settings))
    except RuntimeError as exc:
        assert "RareSpot weights not found" in str(exc)
    else:
        raise AssertionError("bad RareSpot config should stop the worker before NATS connect")


def test_process_rarespot_job_acks_and_records_canceled_event_when_cancel_requested(tmp_path: Path):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cancel_reasons = {"run-1": "user requested stop"}
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    async def scenario() -> dict:
        task = asyncio.create_task(
            process_rarespot_job(
                payload,
                store=store,
                runner=StartedCancellableRunner(started, cancelled),
                message=ack,
                run_status_func=no_control_plane_run_status,
                run_lease_func=no_control_plane_run_lease,
                cancel_reason_func=lambda run_id: cancel_reasons.get(run_id),
                ack_progress_interval_seconds=0,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        return await asyncio.wait_for(task, timeout=1.0)

    result = asyncio.run(scenario())

    assert result == {"run_id": "run-1", "canceled": True}
    assert cancelled.is_set()
    assert ack.acked is True
    assert ack.nacked is False
    assert store.statuses[-1] == ("run-1", "canceled", "user requested stop")
    assert store.event_records[-1] == {
        "event_id": "evt_run-1_canceled",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "event_kind": "run.canceled",
        "message": "RareSpot ecology inference canceled.",
        "payload": {"reason": "user requested stop"},
    }


def test_rarespot_cancel_payload_records_reason_and_cancels_active_task():
    async def scenario() -> tuple[dict[str, str], bool]:
        active_task = asyncio.create_task(asyncio.Event().wait())
        active_tasks = {"run-1": active_task}
        cancel_reasons: dict[str, str] = {}

        _handle_rarespot_cancel_payload(
            {
                "run_id": "run-1",
                "reason": "operator canceled",
            },
            active_tasks=active_tasks,
            cancel_reasons=cancel_reasons,
        )
        with contextlib.suppress(asyncio.CancelledError):
            await active_task
        return cancel_reasons, active_task.cancelled()

    cancel_reasons, was_cancelled = asyncio.run(scenario())

    assert cancel_reasons == {"run-1": "operator canceled"}
    assert was_cancelled is True


def test_subscribe_rarespot_cancels_uses_control_cancel_subject():
    async def scenario():
        nc = FakeNATSConnection()
        settings = RuntimeSettings(
            openai_base_url="http://127.0.0.1:8003/v1",
            openai_model="deepseek_v4",
            nats_cancel_subject="ultra.runs.cancel",
        )
        active_tasks: dict[str, asyncio.Task] = {}
        cancel_reasons: dict[str, str] = {}

        subscription = await _subscribe_rarespot_cancels(
            nc,
            settings,
            active_tasks=active_tasks,
            cancel_reasons=cancel_reasons,
        )
        return nc, subscription

    nc, subscription = asyncio.run(scenario())

    assert subscription == {"subject": "ultra.runs.cancel", "callback": nc.subscriptions[0][1]}
    assert nc.subscriptions[0][0] == "ultra.runs.cancel"


def test_build_rarespot_consumer_config_uses_long_running_ack_wait():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=300,
    )

    config = build_rarespot_consumer_config(settings)

    assert config.durable_name == "rarespot-ecology-worker"
    assert config.ack_wait == 300
    assert config.filter_subject == "ultra.runs.rarespot.jobs"


def test_rarespot_ack_extension_interval_stays_below_ack_wait_for_long_running_leases():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=120,
        rarespot_nats_ack_progress_interval_seconds=180,
    )

    assert rarespot_ack_extension_interval(settings) == 60


def test_subscribe_rarespot_jobs_reconciles_existing_consumer_ack_wait_before_binding():
    old_settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=30,
    )
    new_settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=300,
    )
    js = FakeJetStream(existing_config=build_rarespot_consumer_config(old_settings))

    subscription = asyncio.run(subscribe_rarespot_jobs(js, new_settings))

    assert subscription == {"subscription": "ok"}
    assert len(js.added_configs) == 1
    assert js.added_configs[0].ack_wait == 300
    assert js.deleted_consumers == []
    assert js.pull_calls[0]["config"].ack_wait == 300


def test_subscribe_rarespot_jobs_recreates_consumer_when_update_is_rejected():
    old_settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=30,
    )
    new_settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        rarespot_nats_ack_wait_seconds=300,
    )
    js = FakeJetStream(
        existing_config=build_rarespot_consumer_config(old_settings),
        add_error=RuntimeError("consumer configuration requests do not match"),
    )

    asyncio.run(subscribe_rarespot_jobs(js, new_settings))

    assert js.deleted_consumers == [("ULTRA_RUNS", "rarespot-ecology-worker")]
    assert len(js.added_configs) == 2
    assert js.added_configs[-1].ack_wait == 300


def test_process_rarespot_job_extends_jetstream_ack_while_runner_is_active(tmp_path: Path):
    async def scenario() -> FakeAck:
        store = FakeStore()
        ack = FakeAck()
        ack.progress_seen = asyncio.Event()
        payload = {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "artifact_root": str(tmp_path),
            "metadata": {},
        }

        await process_rarespot_job(
            payload,
            store=store,
            runner=ProgressGatedRunner(ack.progress_seen),
            message=ack,
            run_status_func=no_control_plane_run_status,
            run_lease_func=no_control_plane_run_lease,
            ack_progress_interval_seconds=0.01,
        )
        return ack

    ack = asyncio.run(scenario())

    assert ack.in_progress_count >= 1
    assert ack.acked is True


def test_process_rarespot_job_persists_run_heartbeat_while_runner_is_active(tmp_path: Path):
    async def scenario() -> FakeStore:
        store = FakeStore()
        store.heartbeat_seen = asyncio.Event()
        ack = FakeAck()
        payload = {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "artifact_root": str(tmp_path),
            "metadata": {},
        }

        await process_rarespot_job(
            payload,
            store=store,
            runner=HeartbeatGatedRunner(store.heartbeat_seen),
            message=ack,
            run_status_func=no_control_plane_run_status,
            run_lease_func=no_control_plane_run_lease,
            ack_progress_interval_seconds=0.01,
        )
        return store

    store = asyncio.run(scenario())

    event_kinds = [event_kind for _run_id, event_kind, _message in store.events]
    assert event_kinds == ["run.started", "run.heartbeat", "run.completed"]
    assert store.statuses[0] == ("run-1", "running", "")
    assert store.statuses[1] == ("run-1", "running", "")
    assert store.statuses[-1][1] == "succeeded"


def test_process_rarespot_job_failure_marks_failed_and_acks(tmp_path: Path):
    store = FakeStore()
    ack = FakeAck()
    payload = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "artifact_root": str(tmp_path),
        "metadata": {},
    }

    try:
        asyncio.run(
            process_rarespot_job(
                payload,
                store=store,
                runner=FailingRunner(),
                message=ack,
                run_status_func=no_control_plane_run_status,
                run_lease_func=no_control_plane_run_lease,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "model load failed"
    else:
        raise AssertionError("process_rarespot_job should raise on runner failure")

    assert ack.acked is True
    assert store.statuses[-1] == ("run-1", "failed", "model load failed")
    assert ("run-1", "run.failed", "RareSpot ecology inference failed.") in store.events
