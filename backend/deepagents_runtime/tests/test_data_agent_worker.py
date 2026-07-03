from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error
from urllib import request as urllib_request

import nats
import pytest
import ultra_deepagents.data_agent.worker as data_agent_worker_module
from nats.js.api import AckPolicy
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.data_agent.worker import (
    TERMINAL_DATA_AGENT_JOB_STATUSES,
    ControlPlaneDataAgentJobLease,
    DataAgentJobEnvelope,
    DataAgentLeaseConflict,
    DefaultDataAgentProcessor,
    NATSDataAgentWorker,
    acquire_control_plane_data_agent_job_lease,
    append_control_plane_data_agent_job_event,
    build_data_agent_consumer_config,
    data_agent_ack_extension_interval,
    fetch_control_plane_data_agent_job_status,
    release_control_plane_data_agent_job_lease,
    renew_control_plane_data_agent_job_lease,
    update_control_plane_data_agent_job_status,
)


class FakeAck:
    def __init__(self, payload: bytes) -> None:
        self.data = payload
        self.acked = 0
        self.naked = 0
        self.nak_delays: list[float | None] = []
        self.in_progress_count = 0

    async def ack(self) -> None:
        self.acked += 1

    async def nak(self, delay=None) -> None:
        self.naked += 1
        self.nak_delays.append(delay)

    async def in_progress(self) -> None:
        self.in_progress_count += 1


async def no_worker_heartbeat(*_args, **_kwargs) -> None:
    return None


async def no_data_agent_job_status(_job: DataAgentJobEnvelope, _settings: RuntimeSettings) -> None:
    return None


def _settings(**overrides) -> RuntimeSettings:
    values = {
        "openai_base_url": "http://example.test/v1",
        "openai_model": "deepseek_v4",
        "control_base_url": "https://control.example.test",
        "worker_ack_wait_seconds": 1.0,
        "worker_ack_progress_interval_seconds": 0,
        "data_agent_worker_id": "data-agent-worker-a",
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def _job_payload(**overrides) -> dict:
    payload = {
        "job_id": "data_agent_job_1",
        "dispatch_id": "dispatch-1",
        "owner_user_id": "agent-user",
        "owner_org_id": "agent-org",
        "project_id": "nph-study",
        "job_type": "extract_metadata",
        "resource_ids": ["resource-1", "resource-2"],
        "resource_count": 2,
        "input_selector": {"resource_ids": ["resource-1", "resource-2"]},
        "metadata": {"priority": "field"},
    }
    payload.update(overrides)
    return payload


def test_data_agent_job_envelope_preserves_queue_context():
    job = DataAgentJobEnvelope.from_dict(_job_payload())

    assert job.job_id == "data_agent_job_1"
    assert job.dispatch_id == "dispatch-1"
    assert job.owner_user_id == "agent-user"
    assert job.owner_org_id == "agent-org"
    assert job.project_id == "nph-study"
    assert job.job_type == "extract_metadata"
    assert job.resource_ids == ("resource-1", "resource-2")
    assert job.resource_count == 2
    assert job.input_selector == {"resource_ids": ["resource-1", "resource-2"]}
    assert job.metadata == {"priority": "field"}
    assert job.principal_headers() == {
        "X-Ultra-User-Id": "agent-user",
        "X-Ultra-Org-Id": "agent-org",
        "X-Ultra-Role": "researcher",
    }


def test_default_data_agent_processor_returns_template_specific_summary():
    progress_calls: list[dict[str, object]] = []

    async def progress(**kwargs):
        progress_calls.append(kwargs)

    async def scenario(job_type: str, **overrides):
        processor = DefaultDataAgentProcessor()
        job = DataAgentJobEnvelope.from_dict(_job_payload(job_type=job_type, **overrides))
        return await processor(job, progress)

    metadata = asyncio.run(scenario("extract_metadata"))
    captions = asyncio.run(scenario("caption_resources"))
    batch_tags = asyncio.run(
        scenario(
            "batch_tag_resources",
            input_selector={
                "resource_ids": ["resource-1", "resource-2"],
                "tags": ["NPH", "Under 70", "NPH"],
            },
        )
    )
    dedupe = asyncio.run(scenario("deduplicate_resources"))
    quality = asyncio.run(scenario("quality_check_resources"))

    assert metadata["summary_kind"] == "metadata_extraction"
    assert metadata["resource_ids"] == ["resource-1", "resource-2"]
    assert captions["summary_kind"] == "caption_generation"
    assert len(captions["captions"]) == 2
    assert batch_tags["summary_kind"] == "batch_tagging"
    assert batch_tags["tags_added"] == ["NPH", "Under 70"]
    assert dedupe["summary_kind"] == "duplicate_detection"
    assert dedupe["resource_count"] == 2
    assert quality["summary_kind"] == "quality_check"
    assert quality["warnings"] == []
    assert [call["message"] for call in progress_calls] == [
        "Data Agent metadata extraction completed.",
        "Data Agent caption generation completed.",
        "Data Agent batch tagging completed.",
        "Data Agent duplicate detection completed.",
        "Data Agent quality check completed.",
    ]


def test_data_agent_consumer_config_uses_control_plane_subject_and_durable():
    settings = _settings(
        data_agent_nats_jobs_subject="ultra.data_agent.jobs",
        data_agent_nats_worker_durable="ultra-data-agent-worker",
        data_agent_nats_ack_wait_seconds=240.0,
        worker_max_deliver=7,
        worker_max_concurrency=3,
    )

    config = build_data_agent_consumer_config(settings)

    assert config.filter_subject == "ultra.data_agent.jobs"
    assert config.durable_name == "ultra-data-agent-worker"
    assert config.ack_wait == 240.0
    assert config.max_deliver == 7
    assert config.max_ack_pending == 3
    assert data_agent_ack_extension_interval(settings) == 60.0


def test_data_agent_consumer_config_match_rejects_non_explicit_ack_policy():
    settings = _settings(
        data_agent_nats_jobs_subject="ultra.data_agent.jobs",
        data_agent_nats_worker_durable="ultra-data-agent-worker",
    )
    desired = build_data_agent_consumer_config(settings)
    existing = SimpleNamespace(
        filter_subject=desired.filter_subject,
        ack_wait=desired.ack_wait,
        max_deliver=desired.max_deliver,
        max_ack_pending=desired.max_ack_pending,
        ack_policy=AckPolicy.NONE,
    )

    assert not data_agent_worker_module._data_agent_consumer_config_matches(existing, desired)


def test_data_agent_control_plane_helpers_send_owner_headers(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, body: bytes = b"{}") -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request, timeout):
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "headers": dict(request.header_items()),
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        if request.get_method() == "DELETE":
            return FakeResponse()
        return FakeResponse(
            b'{"job_id":"data_agent_job_1","worker_id":"data-agent-worker-a",'
            b'"lease_token":"lease-token-1","lease_expires_at":"2026-06-08T01:02:03Z",'
            b'"created_at":"2026-06-08T01:00:00Z","updated_at":"2026-06-08T01:00:00Z"}'
        )

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)
    settings = _settings(
        control_status_timeout_seconds=3.0,
        control_worker_token="data-agent-worker-secret",
        data_agent_control_lease_ttl_seconds=120.0,
    )
    job = DataAgentJobEnvelope.from_dict(_job_payload())

    lease = asyncio.run(acquire_control_plane_data_agent_job_lease(job, settings))
    renewed = asyncio.run(renew_control_plane_data_agent_job_lease(lease, settings, job=job))
    asyncio.run(
        update_control_plane_data_agent_job_status(
            job,
            settings,
            status="running",
            progress_completed=1,
            progress_total=2,
            message="Extracted metadata for one resource.",
            event_metadata={"stage": "metadata"},
        )
    )
    asyncio.run(
        append_control_plane_data_agent_job_event(
            job,
            settings,
            event_id="data_agent_job_event_data_agent_job_1_dispatch_1_skipped_initial_status",
            event_type="data_agent.job.skipped",
            message="Data Agent job delivery skipped before worker lease.",
            metadata={"stage": "initial_status_check", "control_status": "succeeded"},
        )
    )
    asyncio.run(release_control_plane_data_agent_job_lease(renewed, settings, job=job))

    assert lease.lease_token == "lease-token-1"
    assert renewed.lease_token == "lease-token-1"
    assert [call["method"] for call in captured] == ["POST", "PATCH", "PATCH", "POST", "DELETE"]
    assert captured[0]["url"] == "https://control.example.test/v2/data-agent/jobs/data_agent_job_1/lease"
    assert captured[2]["url"] == "https://control.example.test/v2/data-agent/jobs/data_agent_job_1/status"
    assert captured[3]["url"] == "https://control.example.test/v2/data-agent/jobs/data_agent_job_1/events"
    for call in captured:
        assert call["headers"]["X-ultra-user-id"] == "agent-user"
        assert call["headers"]["X-ultra-org-id"] == "agent-org"
        assert call["headers"]["X-ultra-worker-token"] == "data-agent-worker-secret"
    assert captured[0]["payload"] == {"worker_id": "data-agent-worker-a", "ttl_seconds": 120.0}
    assert captured[1]["payload"] == {"lease_token": "lease-token-1", "ttl_seconds": 120.0}
    assert captured[2]["payload"]["status"] == "running"
    assert captured[2]["payload"]["progress_completed"] == 1
    assert captured[2]["payload"]["event_metadata"] == {"stage": "metadata"}
    assert captured[3]["payload"] == {
        "event_id": "data_agent_job_event_data_agent_job_1_dispatch_1_skipped_initial_status",
        "event_type": "data_agent.job.skipped",
        "message": "Data Agent job delivery skipped before worker lease.",
        "metadata": {"stage": "initial_status_check", "control_status": "succeeded"},
    }
    assert captured[4]["payload"] == {"lease_token": "lease-token-1"}


def test_data_agent_control_plane_status_lookup_sends_owner_headers_and_maps_404(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request, timeout):
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "headers": dict(request.header_items()),
            }
        )
        if len(captured) == 2:
            raise urllib_error.HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)
        return FakeResponse(b'{"job":{"job_id":"data_agent_job_1","status":"succeeded"}}')

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)
    settings = _settings(control_status_timeout_seconds=3.0)
    job = DataAgentJobEnvelope.from_dict(_job_payload())

    status = asyncio.run(fetch_control_plane_data_agent_job_status(job, settings))
    missing = asyncio.run(fetch_control_plane_data_agent_job_status(job, settings))

    assert status == "succeeded"
    assert missing == "not_found"
    assert [call["method"] for call in captured] == ["GET", "GET"]
    assert captured[0]["url"] == "https://control.example.test/v2/data-agent/jobs/data_agent_job_1"
    for call in captured:
        assert call["headers"]["X-ultra-user-id"] == "agent-user"
        assert call["headers"]["X-ultra-org-id"] == "agent-org"


def test_worker_processes_data_agent_job_updates_terminal_status_and_acks():
    calls: list[tuple[str, object]] = []

    async def processor(job: DataAgentJobEnvelope, progress):
        calls.append(("processor", job.job_id))
        await progress(
            progress_completed=1,
            progress_total=2,
            message="Summarized one resource.",
            event_metadata={"resource_id": "resource-1"},
        )
        return {"summary": "2 resources inspected", "resource_count": job.resource_count}

    async def acquire(job: DataAgentJobEnvelope, settings: RuntimeSettings):
        calls.append(("acquire", job.owner_user_id))
        return ControlPlaneDataAgentJobLease(
            job_id=job.job_id,
            worker_id=settings.data_agent_worker_id,
            lease_token="lease-token-1",
        )

    async def renew(lease, settings, *, job):
        calls.append(("renew", lease.lease_token))
        return lease

    async def release(lease, settings, *, job):
        calls.append(("release", lease.lease_token))

    async def update_status(job, settings, **kwargs):
        calls.append(("status", kwargs))

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(),
            processor=processor,
            job_status_func=no_data_agent_job_status,
            lease_func=acquire,
            renew_lease_func=renew,
            release_lease_func=release,
            status_update_func=update_status,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert calls[0] == ("acquire", "agent-user")
    assert calls[1][0] == "status"
    assert calls[1][1]["status"] == "running"
    assert calls[2] == ("processor", "data_agent_job_1")
    assert calls[3][0] == "status"
    assert calls[3][1]["status"] == "running"
    assert calls[3][1]["progress_completed"] == 1
    assert calls[4][0] == "status"
    assert calls[4][1]["status"] == "succeeded"
    assert calls[4][1]["output_summary"] == {
        "summary": "2 resources inspected",
        "resource_count": 2,
    }
    assert calls[-1] == ("release", "lease-token-1")


def test_worker_posts_skip_event_before_ack_for_terminal_control_plane_data_agent_job():
    calls: list[tuple[str, object]] = []

    async def terminal_status(job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        calls.append(("status", job.job_id))
        return "succeeded"

    async def append_skip_event(job: DataAgentJobEnvelope, _settings: RuntimeSettings, **kwargs):
        calls.append(("event", kwargs))
        assert job.job_id == "data_agent_job_1"
        return {"event": {"event_type": kwargs["event_type"], "metadata": kwargs["metadata"]}}

    async def acquire_should_not_run(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        raise AssertionError("lease should not be acquired for terminal Data Agent jobs")

    async def processor_should_not_run(_job: DataAgentJobEnvelope, _progress):
        raise AssertionError("processor should not run for terminal Data Agent jobs")

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(),
            processor=processor_should_not_run,
            job_status_func=terminal_status,
            lease_func=acquire_should_not_run,
            event_append_func=append_skip_event,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert calls[0] == ("status", "data_agent_job_1")
    assert calls[1][0] == "event"
    skip = calls[1][1]
    assert skip["event_type"] == "data_agent.job.skipped"
    assert skip["event_id"] == "data_agent_job_event_data_agent_job_1_dispatch_1_skipped_initial_status"
    assert skip["message"] == "Data Agent job delivery skipped before worker lease."
    assert skip["metadata"] == {
        "ack_action": "ack",
        "control_status": "succeeded",
        "dispatch_id": "dispatch-1",
        "job_type": "extract_metadata",
        "stage": "initial_status_check",
        "worker_id": "data-agent-worker-a",
    }
    assert message.acked == 1
    assert message.naked == 0


def test_worker_naks_terminal_control_plane_data_agent_job_when_skip_event_fails():
    async def terminal_status(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        return "canceled"

    async def append_skip_event(_job: DataAgentJobEnvelope, _settings: RuntimeSettings, **_kwargs):
        raise RuntimeError("control plane unavailable")

    async def acquire_should_not_run(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        raise AssertionError("lease should not be acquired for terminal Data Agent jobs")

    async def processor_should_not_run(_job: DataAgentJobEnvelope, _progress):
        raise AssertionError("processor should not run for terminal Data Agent jobs")

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(data_agent_nats_ack_progress_interval_seconds=30.0),
            processor=processor_should_not_run,
            job_status_func=terminal_status,
            lease_func=acquire_should_not_run,
            event_append_func=append_skip_event,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [30.0]


def test_worker_naks_missing_control_plane_data_agent_job_when_skip_event_cannot_persist():
    skip_events: list[dict[str, object]] = []

    async def missing_status(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        return "not_found"

    async def append_skip_event(_job: DataAgentJobEnvelope, _settings: RuntimeSettings, **kwargs):
        skip_events.append(kwargs)
        raise RuntimeError("job event append returned 404")

    async def acquire_should_not_run(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        raise AssertionError("lease should not be acquired for missing Data Agent jobs")

    async def processor_should_not_run(_job: DataAgentJobEnvelope, _progress):
        raise AssertionError("processor should not run for missing Data Agent jobs")

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(data_agent_nats_ack_progress_interval_seconds=45.0),
            processor=processor_should_not_run,
            job_status_func=missing_status,
            lease_func=acquire_should_not_run,
            event_append_func=append_skip_event,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [45.0]
    assert skip_events[0]["event_type"] == "data_agent.job.skipped"
    assert skip_events[0]["event_id"] == "data_agent_job_event_data_agent_job_1_dispatch_1_skipped_initial_status"
    assert skip_events[0]["metadata"]["control_status"] == "not_found"


def test_worker_marks_data_agent_job_failed_and_acks_when_processor_raises():
    statuses: list[dict[str, object]] = []
    released: list[str] = []

    async def processor(_job: DataAgentJobEnvelope, _progress):
        raise RuntimeError("metadata extractor crashed")

    async def acquire(job: DataAgentJobEnvelope, settings: RuntimeSettings):
        return ControlPlaneDataAgentJobLease(
            job_id=job.job_id,
            worker_id=settings.data_agent_worker_id,
            lease_token="lease-token-1",
        )

    async def update_status(_job, _settings, **kwargs):
        statuses.append(kwargs)

    async def release(lease, _settings, *, job):
        released.append(lease.lease_token)

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(),
            processor=processor,
            job_status_func=no_data_agent_job_status,
            lease_func=acquire,
            release_lease_func=release,
            status_update_func=update_status,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert statuses[-1]["status"] == "failed"
    assert statuses[-1]["error"] == "metadata extractor crashed"
    assert released == ["lease-token-1"]


def test_worker_naks_without_processing_when_data_agent_job_lease_is_held():
    processor_calls = 0

    async def processor(_job: DataAgentJobEnvelope, _progress):
        nonlocal processor_calls
        processor_calls += 1
        return {}

    async def acquire(_job: DataAgentJobEnvelope, _settings: RuntimeSettings):
        raise DataAgentLeaseConflict("job already leased")

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(data_agent_nats_ack_progress_interval_seconds=30.0),
            processor=processor,
            job_status_func=no_data_agent_job_status,
            lease_func=acquire,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert processor_calls == 0
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [30.0]


def test_worker_naks_and_stops_compute_when_data_agent_lease_is_lost():
    cancelled = asyncio.Event()
    released: list[str] = []

    async def processor(_job: DataAgentJobEnvelope, _progress):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def acquire(job: DataAgentJobEnvelope, settings: RuntimeSettings):
        return ControlPlaneDataAgentJobLease(
            job_id=job.job_id,
            worker_id=settings.data_agent_worker_id,
            lease_token="lease-token-1",
        )

    async def renew(_lease, _settings, *, job):
        raise DataAgentLeaseConflict("lease moved to another worker")

    async def release(lease, _settings, *, job):
        released.append(lease.lease_token)

    async def update_status(_job, _settings, **_kwargs):
        return None

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(
                data_agent_nats_ack_wait_seconds=1.0,
                data_agent_nats_ack_progress_interval_seconds=0.01,
            ),
            processor=processor,
            job_status_func=no_data_agent_job_status,
            lease_func=acquire,
            renew_lease_func=renew,
            release_lease_func=release,
            status_update_func=update_status,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(json.dumps(_job_payload()).encode("utf-8"))
        await asyncio.wait_for(worker._process_message(message), timeout=0.5)
        return message

    message = asyncio.run(scenario())

    assert cancelled.is_set()
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [0.01]
    assert released == ["lease-token-1"]


def test_worker_acks_malformed_data_agent_payload():
    async def processor(_job: DataAgentJobEnvelope, _progress):
        raise AssertionError("processor should not run for malformed payload")

    async def scenario():
        worker = NATSDataAgentWorker(
            _settings(),
            processor=processor,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        message = FakeAck(b"{not-json")
        await worker._process_message(message)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0


def test_data_agent_worker_live_nats_smoke():
    nats_url = os.getenv("ULTRA_DEEPAGENTS_TEST_NATS_URL", "").strip()
    if not nats_url:
        pytest.skip("ULTRA_DEEPAGENTS_TEST_NATS_URL is not set")

    async def scenario():
        suffix = uuid.uuid4().hex
        stream = f"ULTRA_DATA_AGENT_TEST_{suffix}"
        subject = f"ultra.test.{suffix}.data_agent.jobs"
        durable = f"data-agent-test-{suffix}"
        statuses: list[dict[str, object]] = []

        async def processor(job: DataAgentJobEnvelope, progress):
            await progress(
                progress_completed=job.resource_count,
                progress_total=job.resource_count,
                message="live nats smoke progressed",
            )
            return {"smoke": True, "job_id": job.job_id}

        async def update_status(_job, _settings, **kwargs):
            statuses.append(kwargs)

        settings = _settings(
            control_base_url="",
            data_agent_nats_url=nats_url,
            data_agent_nats_stream=stream,
            data_agent_nats_jobs_subject=subject,
            data_agent_nats_worker_durable=durable,
            data_agent_nats_ack_wait_seconds=30.0,
            data_agent_nats_ack_progress_interval_seconds=0,
        )
        worker = NATSDataAgentWorker(
            settings,
            processor=processor,
            job_status_func=no_data_agent_job_status,
            status_update_func=update_status,
            worker_heartbeat_func=no_worker_heartbeat,
        )
        nc = await nats.connect(nats_url)
        js = nc.jetstream()
        try:
            await worker._ensure_stream(js)
            subscription = await worker._subscribe(js)
            await js.publish(subject, json.dumps(_job_payload()).encode("utf-8"))
            messages = await subscription.fetch(batch=1, timeout=2.0)
            assert len(messages) == 1
            await worker._process_message(messages[0])
            await _wait_for_consumer_ack_pending_zero(js, stream, durable)
        finally:
            with contextlib.suppress(Exception):
                await js.delete_stream(stream)
            await nc.drain()
        return statuses

    statuses = asyncio.run(scenario())

    assert [status["status"] for status in statuses] == ["running", "running", "succeeded"]
    assert statuses[-1]["output_summary"] == {"smoke": True, "job_id": "data_agent_job_1"}


def test_data_agent_worker_go_control_plane_live_smoke(tmp_path):
    if os.getenv("ULTRA_DEEPAGENTS_TEST_GO_CONTROL_PLANE", "").strip() != "1":
        pytest.skip("set ULTRA_DEEPAGENTS_TEST_GO_CONTROL_PLANE=1 to run the Go control-plane smoke")
    nats_url = os.getenv("ULTRA_DEEPAGENTS_TEST_NATS_URL", "").strip()
    if not nats_url:
        pytest.skip("ULTRA_DEEPAGENTS_TEST_NATS_URL is not set")
    if shutil.which("go") is None:
        pytest.skip("go is not available")

    suffix = uuid.uuid4().hex
    port = _free_local_port()
    control_base_url = f"http://127.0.0.1:{port}"
    stream = f"ULTRA_DA_E2E_{suffix}"
    jobs_subject = f"ultra.e2e.{suffix}.data_agent.jobs"
    durable = f"ultra-da-e2e-{suffix}"
    principal_headers = {
        "X-Ultra-User-Id": "agent-user",
        "X-Ultra-Org-Id": "agent-org",
        "X-Ultra-Role": "researcher",
    }
    repo_backend = Path(__file__).resolve().parents[2]
    controlplane_dir = repo_backend / "controlplane"
    env = os.environ.copy()
    env.update(
        {
            "ULTRA_CONTROL_HTTP_ADDR": f"127.0.0.1:{port}",
            "ULTRA_CONTROL_NATS_URL": nats_url,
            "ULTRA_CONTROL_NATS_STREAM": stream,
            "ULTRA_CONTROL_NATS_JOBS_SUBJECT": f"ultra.e2e.{suffix}.runs.jobs",
            "ULTRA_CONTROL_NATS_DATA_AGENT_JOBS_SUBJECT": jobs_subject,
            "ULTRA_CONTROL_NATS_EVENTS_SUBJECT": f"ultra.e2e.{suffix}.runs.events",
            "ULTRA_CONTROL_NATS_CANCEL_SUBJECT": f"ultra.e2e.{suffix}.runs.cancel",
            "ULTRA_CONTROL_NATS_EVENT_CONSUMER": f"ultra-control-e2e-{suffix}",
            "ULTRA_CONTROL_NATS_WORKER_DURABLE": f"ultra-worker-e2e-{suffix}",
            "ULTRA_CONTROL_NATS_DATA_AGENT_WORKER_DURABLE": durable,
            "ULTRA_CONTROL_UPLOAD_ROOT": str(tmp_path / "uploads"),
            "ULTRA_CONTROL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "ULTRA_CONTROL_RUN_RECOVERY_ENABLED": "false",
            "ULTRA_CONTROL_AUTH_PROVIDER": "dev",
        }
    )
    env.pop("ULTRA_CONTROL_DATABASE_URL", None)
    env.pop("RUN_STORE_PATH", None)

    proc = subprocess.Popen(
        ["go", "run", "./cmd/ultra-control", "serve"],
        cwd=controlplane_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_control_plane(control_base_url, proc)
        upload = _upload_smoke_resource(
            control_base_url,
            filename="nph-e2e.nii.gz",
            content=b"nifti smoke bytes",
            headers=principal_headers,
        )
        resource_id = upload["uploaded"][0]["file_id"]
        created = _request_json(
            "POST",
            f"{control_base_url}/v2/data-agent/jobs",
            {
                "job_type": "extract_metadata",
                "resource_ids": [resource_id],
                "project_id": "nph-smoke",
                "metadata": {"source": "python_go_control_plane_e2e_smoke"},
            },
            headers=principal_headers,
        )
        job_id = created["job"]["job_id"]

        async def scenario():
            settings = _settings(
                control_base_url=control_base_url,
                data_agent_nats_url=nats_url,
                data_agent_nats_stream=stream,
                data_agent_nats_jobs_subject=jobs_subject,
                data_agent_nats_worker_durable=durable,
                data_agent_nats_ack_wait_seconds=30.0,
                data_agent_nats_ack_progress_interval_seconds=0,
                data_agent_worker_id=durable,
                data_agent_control_lease_required=True,
            )
            worker = NATSDataAgentWorker(settings, worker_heartbeat_func=no_worker_heartbeat)
            nc = await nats.connect(nats_url)
            js = nc.jetstream()
            try:
                await worker._ensure_stream(js)
                subscription = await worker._subscribe(js)
                messages = await subscription.fetch(batch=1, timeout=10.0)
                assert len(messages) == 1
                await worker._process_message(messages[0])
                await _wait_for_consumer_ack_pending_zero(js, stream, durable)
            finally:
                await nc.drain()

        asyncio.run(scenario())
        loaded = _wait_for_data_agent_job_status(
            control_base_url,
            job_id,
            want_status="succeeded",
            headers=principal_headers,
        )
    finally:
        _terminate_process(proc)
        asyncio.run(_delete_nats_stream(nats_url, stream))

    assert loaded["job"]["progress_completed"] == 1
    assert loaded["job"]["progress_total"] == 1
    assert loaded["job"]["output_summary"] == {
        "job_type": "extract_metadata",
        "metadata": {"source": "python_go_control_plane_e2e_smoke"},
        "resource_count": 1,
        "resource_ids": [resource_id],
        "selector": {"resource_ids": [resource_id]},
        "summary": "Prepared metadata extraction for 1 resources from the Data Agent queue envelope.",
        "summary_kind": "metadata_extraction",
    }
    assert [event["event_type"] for event in loaded["events"]] == [
        "data_agent.job.created",
        "data_agent.job.dispatched",
        "data_agent.job.leased",
        "data_agent.job.progressed",
        "data_agent.job.progressed",
        "data_agent.job.completed",
    ]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_control_plane(base_url: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"control plane exited early:\n{_process_output(proc)}")
        try:
            payload = _request_json("GET", f"{base_url}/v1/health")
            if payload.get("status") == "ok":
                return
        except Exception:
            time.sleep(0.2)
    raise AssertionError(f"control plane did not become healthy:\n{_process_output(proc)}")


def _upload_smoke_resource(
    base_url: str,
    *,
    filename: str,
    content: bytes,
    headers: dict[str, str],
) -> dict[str, object]:
    boundary = f"ultra-smoke-{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib_request.Request(
        f"{base_url}/v2/uploads",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            **headers,
        },
    )
    with urllib_request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib_request.Request(
        url,
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib_request.urlopen(request, timeout=10.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc


def _wait_for_data_agent_job_status(
    base_url: str,
    job_id: str,
    *,
    want_status: str,
    headers: dict[str, str],
) -> dict[str, object]:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        payload = _request_json("GET", f"{base_url}/v2/data-agent/jobs/{job_id}", headers=headers)
        status = str(payload.get("job", {}).get("status", ""))
        if status == want_status:
            return payload
        if status in TERMINAL_DATA_AGENT_JOB_STATUSES:
            raise AssertionError(f"data-agent job reached {status}: {payload}")
        time.sleep(0.2)
    raise AssertionError(f"data-agent job {job_id} did not reach {want_status}")


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=5.0)


def _process_output(proc: subprocess.Popen[str]) -> str:
    if proc.stdout is None:
        return ""
    try:
        output, _ = proc.communicate(timeout=1.0)
        return output
    except subprocess.TimeoutExpired:
        return "<process still running>"


async def _delete_nats_stream(nats_url: str, stream: str) -> None:
    if not nats_url or not stream:
        return
    with contextlib.suppress(Exception):
        nc = await nats.connect(nats_url)
        try:
            await nc.jetstream().delete_stream(stream)
        finally:
            await nc.drain()


async def _wait_for_consumer_ack_pending_zero(js, stream: str, durable: str) -> None:
    deadline = time.monotonic() + 3.0
    last_pending = None
    while time.monotonic() < deadline:
        consumer = await js.consumer_info(stream, durable)
        last_pending = getattr(consumer, "num_ack_pending", 0)
        if last_pending == 0:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"consumer {durable} ack pending = {last_pending}, want 0")
