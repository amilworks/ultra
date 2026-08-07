from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import nats
from nats.js.api import AckPolicy, ConsumerConfig

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
    WorkerHeartbeatFunc,
    fetch_job_messages,
    post_control_plane_worker_heartbeat,
)

logger = logging.getLogger(__name__)

TERMINAL_DATA_AGENT_JOB_STATUSES = {"succeeded", "failed", "canceled"}
CONTROL_PLANE_DATA_AGENT_JOB_NOT_FOUND_STATUS = "not_found"
SKIP_DATA_AGENT_JOB_STATUSES = {
    *TERMINAL_DATA_AGENT_JOB_STATUSES,
    CONTROL_PLANE_DATA_AGENT_JOB_NOT_FOUND_STATUS,
}


class DataAgentLeaseConflict(RuntimeError):
    pass


class DataAgentLeaseUnavailable(RuntimeError):
    pass


class RetryableDataAgentProcessingError(RuntimeError):
    """A processor-side operational failure that must leave the delivery retryable."""


@dataclass(frozen=True)
class ControlPlaneDataAgentJobLease:
    job_id: str
    worker_id: str
    lease_token: str
    lease_expires_at: str = ""


@dataclass(frozen=True)
class DataAgentJobEnvelope:
    job_id: str
    dispatch_id: str = ""
    owner_user_id: str = ""
    owner_org_id: str = ""
    project_id: str = ""
    job_type: str = ""
    resource_ids: tuple[str, ...] = ()
    resource_count: int = 0
    input_selector: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataAgentJobEnvelope:
        if not isinstance(payload, dict):
            raise ValueError("data-agent job payload must be an object")
        job_id = _string(payload.get("job_id"))
        if not job_id:
            raise ValueError("data-agent job payload missing job_id")
        resource_ids = tuple(_string_list(payload.get("resource_ids")))
        resource_count = _int(payload.get("resource_count"), default=len(resource_ids))
        return cls(
            job_id=job_id,
            dispatch_id=_string(payload.get("dispatch_id")),
            owner_user_id=_string(payload.get("owner_user_id")) or "local-user",
            owner_org_id=_string(payload.get("owner_org_id")) or "local-org",
            project_id=_string(payload.get("project_id")),
            job_type=_string(payload.get("job_type")),
            resource_ids=resource_ids,
            resource_count=resource_count,
            input_selector=_dict(payload.get("input_selector")),
            metadata=_dict(payload.get("metadata")),
        )

    def principal_headers(self, settings: Any | None = None) -> dict[str, str]:
        headers = {
            "X-Ultra-User-Id": self.owner_user_id or "local-user",
            "X-Ultra-Org-Id": self.owner_org_id or "local-org",
            "X-Ultra-Role": "researcher",
        }
        token = str(getattr(settings, "control_worker_token", "") or "").strip() if settings else ""
        if token:
            headers["X-Ultra-Worker-Token"] = token
        return headers


DataAgentProgressFunc = Callable[..., Awaitable[None]]
DataAgentProcessorFunc = Callable[
    [DataAgentJobEnvelope, DataAgentProgressFunc],
    Awaitable[dict[str, Any] | None],
]
DataAgentJobStatusFunc = Callable[
    [DataAgentJobEnvelope, RuntimeSettings],
    Awaitable[str | None],
]
DataAgentLeaseFunc = Callable[
    [DataAgentJobEnvelope, RuntimeSettings],
    Awaitable[ControlPlaneDataAgentJobLease | None],
]


class RenewDataAgentLeaseFunc(Protocol):
    async def __call__(
        self,
        lease: ControlPlaneDataAgentJobLease,
        settings: RuntimeSettings,
        *,
        job: DataAgentJobEnvelope,
    ) -> ControlPlaneDataAgentJobLease | None: ...


class ReleaseDataAgentLeaseFunc(Protocol):
    async def __call__(
        self,
        lease: ControlPlaneDataAgentJobLease,
        settings: RuntimeSettings,
        *,
        job: DataAgentJobEnvelope,
    ) -> None: ...


DataAgentStatusUpdateFunc = Callable[..., Awaitable[dict[str, Any] | None]]
DataAgentEventAppendFunc = Callable[..., Awaitable[dict[str, Any] | None]]


def build_data_agent_consumer_config(settings: RuntimeSettings) -> ConsumerConfig:
    return ConsumerConfig(
        durable_name=settings.data_agent_nats_worker_durable,
        filter_subject=settings.data_agent_nats_jobs_subject,
        ack_policy=AckPolicy.EXPLICIT,
        ack_wait=settings.data_agent_nats_ack_wait_seconds,
        max_deliver=settings.worker_max_deliver,
        max_ack_pending=settings.worker_max_concurrency,
    )


def data_agent_ack_extension_interval(settings: RuntimeSettings) -> float:
    ack_wait_seconds = max(0.1, float(settings.data_agent_nats_ack_wait_seconds))
    safe_ceiling = max(0.1, ack_wait_seconds / 2.0)
    configured_interval = float(settings.data_agent_nats_ack_progress_interval_seconds)
    if configured_interval <= 0:
        return safe_ceiling
    return min(configured_interval, safe_ceiling)


async def fetch_control_plane_data_agent_job_status(
    job: DataAgentJobEnvelope,
    settings: RuntimeSettings,
) -> str | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    url = _data_agent_job_url(settings, job, suffix="")

    def fetch() -> str | None:
        request = urllib_request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                **job.principal_headers(settings),
            },
        )
        timeout = max(0.1, float(settings.control_status_timeout_seconds))
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return CONTROL_PLANE_DATA_AGENT_JOB_NOT_FOUND_STATUS
            logger.warning(
                "Control-plane data-agent job status lookup returned HTTP error; proceeding with job.",
                extra={"job_id": job.job_id, "status_code": exc.code},
            )
            return None
        except Exception:
            logger.warning(
                "Control-plane data-agent job status lookup failed; proceeding with job.",
                extra={"job_id": job.job_id},
                exc_info=True,
            )
            return None
        if not isinstance(payload, dict):
            return None
        record = payload.get("job")
        if isinstance(record, dict):
            status = _string(record.get("status")).lower()
            return status or None
        status = _string(payload.get("status")).lower()
        return status or None

    return await asyncio.to_thread(fetch)


async def acquire_control_plane_data_agent_job_lease(
    job: DataAgentJobEnvelope,
    settings: RuntimeSettings,
) -> ControlPlaneDataAgentJobLease | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    url = _data_agent_job_url(settings, job, suffix="/lease")
    payload = {
        "worker_id": settings.data_agent_worker_id,
        "ttl_seconds": settings.data_agent_control_lease_ttl_seconds,
    }
    return await asyncio.to_thread(
        _request_control_plane_data_agent_job_lease,
        url,
        "POST",
        payload,
        settings,
        job,
    )


async def renew_control_plane_data_agent_job_lease(
    lease: ControlPlaneDataAgentJobLease,
    settings: RuntimeSettings,
    *,
    job: DataAgentJobEnvelope,
) -> ControlPlaneDataAgentJobLease | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return lease
    url = _data_agent_job_url(settings, job, suffix="/lease")
    payload = {
        "lease_token": lease.lease_token,
        "ttl_seconds": settings.data_agent_control_lease_ttl_seconds,
    }
    renewed = await asyncio.to_thread(
        _request_control_plane_data_agent_job_lease,
        url,
        "PATCH",
        payload,
        settings,
        job,
    )
    return renewed or lease


async def release_control_plane_data_agent_job_lease(
    lease: ControlPlaneDataAgentJobLease,
    settings: RuntimeSettings,
    *,
    job: DataAgentJobEnvelope,
) -> None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return
    url = _data_agent_job_url(settings, job, suffix="/lease")
    payload = {"lease_token": lease.lease_token}
    await asyncio.to_thread(
        _request_control_plane_data_agent_job_lease,
        url,
        "DELETE",
        payload,
        settings,
        job,
    )


async def update_control_plane_data_agent_job_status(
    job: DataAgentJobEnvelope,
    settings: RuntimeSettings,
    *,
    status: str,
    progress_completed: int = 0,
    progress_total: int = 0,
    error: str = "",
    message: str = "",
    output_summary: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    url = _data_agent_job_url(settings, job, suffix="/status")
    payload = {
        "status": status,
        "progress_completed": progress_completed,
        "progress_total": progress_total,
        "error": error,
        "message": message,
        "output_summary": output_summary or {},
        "metadata": metadata or {},
        "event_metadata": event_metadata or {},
    }
    return await asyncio.to_thread(
        _request_control_plane_json,
        url,
        "PATCH",
        payload,
        settings,
        job,
    )


async def append_control_plane_data_agent_job_event(
    job: DataAgentJobEnvelope,
    settings: RuntimeSettings,
    *,
    event_id: str = "",
    event_type: str,
    message: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    url = _data_agent_job_url(settings, job, suffix="/events")
    payload = {
        "event_id": event_id,
        "event_type": event_type,
        "message": message,
        "metadata": metadata or {},
    }
    return await asyncio.to_thread(
        _request_control_plane_json,
        url,
        "POST",
        payload,
        settings,
        job,
    )


def _request_control_plane_data_agent_job_lease(
    url: str,
    method: str,
    payload: dict[str, Any],
    settings: RuntimeSettings,
    job: DataAgentJobEnvelope,
) -> ControlPlaneDataAgentJobLease | None:
    try:
        data = _request_control_plane_json(url, method, payload, settings, job)
    except urllib_error.HTTPError as exc:
        if exc.code == 409:
            raise DataAgentLeaseConflict(
                "control plane data-agent job lease is already held"
            ) from exc
        if settings.data_agent_control_lease_required:
            raise DataAgentLeaseUnavailable(
                f"control plane data-agent lease request failed with HTTP {exc.code}"
            ) from exc
        logger.warning(
            "Control-plane data-agent lease request returned HTTP error; proceeding without durable lease.",
            extra={"job_id": job.job_id, "status_code": exc.code},
        )
        return None
    except Exception as exc:
        if settings.data_agent_control_lease_required:
            raise DataAgentLeaseUnavailable(
                "control plane data-agent lease request failed"
            ) from exc
        logger.warning(
            "Control-plane data-agent lease request failed; proceeding without durable lease.",
            extra={"job_id": job.job_id},
            exc_info=True,
        )
        return None
    if method == "DELETE" or not isinstance(data, dict):
        return None
    return ControlPlaneDataAgentJobLease(
        job_id=str(data.get("job_id") or ""),
        worker_id=str(data.get("worker_id") or ""),
        lease_token=str(data.get("lease_token") or ""),
        lease_expires_at=str(data.get("lease_expires_at") or ""),
    )


def _request_control_plane_json(
    url: str,
    method: str,
    payload: dict[str, Any],
    settings: RuntimeSettings,
    job: DataAgentJobEnvelope,
) -> dict[str, Any] | None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **job.principal_headers(settings),
        },
    )
    timeout = max(0.1, float(settings.control_status_timeout_seconds))
    with urllib_request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else None


def _data_agent_job_url(
    settings: RuntimeSettings,
    job: DataAgentJobEnvelope,
    *,
    suffix: str,
) -> str:
    quoted_job_id = urllib_parse.quote(job.job_id, safe="")
    return f"{settings.control_base_url.rstrip('/')}/v2/data-agent/jobs/{quoted_job_id}{suffix}"


def _data_agent_skip_event_id(job: DataAgentJobEnvelope) -> str:
    dispatch_part = _event_id_part(job.dispatch_id) or "no_dispatch"
    return (
        f"data_agent_job_event_{_event_id_part(job.job_id)}_{dispatch_part}_skipped_initial_status"
    )


def _event_id_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in _string(value))
    return cleaned.strip("_")


class DefaultDataAgentProcessor:
    async def __call__(
        self,
        job: DataAgentJobEnvelope,
        progress: DataAgentProgressFunc,
    ) -> dict[str, Any]:
        output_summary = _default_data_agent_output_summary(job)
        await progress(
            progress_completed=job.resource_count,
            progress_total=job.resource_count,
            message=f"Data Agent {_data_agent_template_display_name(job.job_type)} completed.",
            event_metadata={"stage": output_summary.get("summary_kind", "metadata_summary")},
        )
        return output_summary


def _default_data_agent_output_summary(job: DataAgentJobEnvelope) -> dict[str, Any]:
    job_type = _string(job.job_type) or "extract_metadata"
    base: dict[str, Any] = {
        "job_type": job_type,
        "resource_count": job.resource_count,
        "resource_ids": list(job.resource_ids),
        "summary": _default_data_agent_summary_text(job_type, job),
    }
    if job_type == "caption_resources":
        return {
            **base,
            "summary_kind": "caption_generation",
            "captions": [
                {
                    "resource_id": resource_id,
                    "caption": f"{resource_id} is queued for deterministic metadata captioning.",
                    "caption_source": "deterministic_queue_context",
                }
                for resource_id in job.resource_ids
            ],
        }
    if job_type == "batch_tag_resources":
        tags_added = _data_agent_tags(job)
        return {
            **base,
            "summary_kind": "batch_tagging",
            "tagged_resource_count": job.resource_count,
            "tags_added": tags_added,
        }
    if job_type == "organize_resources":
        return {
            **base,
            "summary_kind": "organization_plan",
            "collection_suggestions": [
                {
                    "name": job.project_id or "Unassigned project",
                    "rule": "project",
                    "reason": "Groups resources by queue project context.",
                    "resource_ids": list(job.resource_ids),
                    "resource_count": job.resource_count,
                }
            ],
        }
    if job_type == "deduplicate_resources":
        return {
            **base,
            "summary_kind": "duplicate_detection",
            "duplicate_groups": [],
            "duplicate_group_count": 0,
            "duplicate_resource_count": 0,
            "unique_resource_count": job.resource_count,
        }
    if job_type == "quality_check_resources":
        return {
            **base,
            "summary_kind": "quality_check",
            "warning_count": 0,
            "warnings": [],
        }
    return {
        **base,
        "summary_kind": "metadata_extraction",
        "selector": job.input_selector or {},
        "metadata": job.metadata or {},
    }


def _data_agent_template_display_name(job_type: str) -> str:
    normalized = _string(job_type)
    if normalized == "caption_resources":
        return "caption generation"
    if normalized == "batch_tag_resources":
        return "batch tagging"
    if normalized == "organize_resources":
        return "organization plan"
    if normalized == "deduplicate_resources":
        return "duplicate detection"
    if normalized == "quality_check_resources":
        return "quality check"
    if normalized == "extract_metadata":
        return "metadata extraction"
    return "metadata summary"


def _default_data_agent_summary_text(job_type: str, job: DataAgentJobEnvelope) -> str:
    label = _data_agent_template_display_name(job_type)
    return (
        f"Prepared {label} for {job.resource_count} resources from the Data Agent queue envelope."
    )


def _data_agent_tags(job: DataAgentJobEnvelope) -> list[str]:
    tags = _string_list((job.input_selector or {}).get("tags"))
    if not tags:
        tags = _string_list((job.metadata or {}).get("tags"))
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


class NATSDataAgentWorker:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        processor: DataAgentProcessorFunc | None = None,
        job_status_func: DataAgentJobStatusFunc = fetch_control_plane_data_agent_job_status,
        lease_func: DataAgentLeaseFunc = acquire_control_plane_data_agent_job_lease,
        renew_lease_func: RenewDataAgentLeaseFunc = renew_control_plane_data_agent_job_lease,
        release_lease_func: ReleaseDataAgentLeaseFunc = release_control_plane_data_agent_job_lease,
        status_update_func: DataAgentStatusUpdateFunc = update_control_plane_data_agent_job_status,
        event_append_func: DataAgentEventAppendFunc = append_control_plane_data_agent_job_event,
        worker_heartbeat_func: WorkerHeartbeatFunc = post_control_plane_worker_heartbeat,
    ) -> None:
        self.settings = _data_agent_worker_identity(settings or RuntimeSettings.from_env())
        self._processor = processor or DefaultDataAgentProcessor()
        self._job_status = job_status_func
        self._lease = lease_func
        self._renew_lease = renew_lease_func
        self._release_lease = release_lease_func
        self._update_status = status_update_func
        self._append_event = event_append_func
        self._worker_heartbeat = worker_heartbeat_func
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False

    async def run_forever(self) -> None:
        nc = await nats.connect(self.settings.data_agent_nats_url)
        js = nc.jetstream()
        try:
            await self._ensure_stream(js)
            subscription = await self._subscribe(js)
            active_message_tasks: set[asyncio.Task] = set()
            max_concurrency = max(1, self.settings.worker_max_concurrency)
            last_idle_heartbeat_at = 0.0
            while True:
                active_message_tasks = _harvest_finished_message_tasks(active_message_tasks)
                available_slots = max_concurrency - len(active_message_tasks)
                if available_slots <= 0:
                    active_message_tasks = await _wait_for_message_task_capacity(
                        active_message_tasks,
                        timeout=0.2,
                    )
                    continue
                messages = await fetch_job_messages(
                    subscription,
                    batch=available_slots,
                    timeout=2.0,
                )
                if not messages:
                    if active_message_tasks:
                        active_message_tasks = await _wait_for_message_task_capacity(
                            active_message_tasks,
                            timeout=0.2,
                        )
                    else:
                        last_idle_heartbeat_at = await self._maybe_post_idle_worker_heartbeat(
                            last_idle_heartbeat_at,
                        )
                        await asyncio.sleep(0.2)
                    continue
                for message in messages:
                    active_message_tasks.add(asyncio.create_task(self._process_message(message)))
        finally:
            self._shutting_down = True
            if "active_message_tasks" in locals():
                await _cancel_message_tasks(active_message_tasks)
            await nc.drain()

    async def _ensure_stream(self, js: Any) -> None:
        try:
            await js.add_stream(
                name=self.settings.data_agent_nats_stream,
                subjects=[self.settings.data_agent_nats_jobs_subject],
            )
        except Exception as exc:
            if "stream name already in use" not in str(exc).lower():
                raise

    async def _subscribe(self, js: Any) -> Any:
        config = build_data_agent_consumer_config(self.settings)
        await _reconcile_data_agent_consumer(js, self.settings, config=config)
        return await js.pull_subscribe(
            self.settings.data_agent_nats_jobs_subject,
            durable=self.settings.data_agent_nats_worker_durable,
            stream=self.settings.data_agent_nats_stream,
            config=config,
        )

    async def _process_message(self, message: Any) -> None:
        ack_progress_task = _start_ack_progress_task(
            message,
            interval_seconds=data_agent_ack_extension_interval(self.settings),
        )
        control_lease: ControlPlaneDataAgentJobLease | None = None
        data_agent_heartbeat_task: asyncio.Task | None = None
        should_ack = True
        control_lease_lost = False

        def mark_control_lease_lost() -> None:
            nonlocal control_lease_lost
            control_lease_lost = True

        try:
            try:
                payload = json.loads(message.data.decode("utf-8"))
                job = DataAgentJobEnvelope.from_dict(payload)
            except Exception:
                logger.exception("Discarding malformed Data Agent job payload.")
                await _ack_message(message)
                return

            control_status = await self._safe_job_status(job)
            if control_status in SKIP_DATA_AGENT_JOB_STATUSES:
                logger.info(
                    "Skipping Data Agent job because control plane job is not runnable.",
                    extra={"job_id": job.job_id, "status": control_status},
                )
                try:
                    await self._append_event(
                        job,
                        self.settings,
                        event_id=_data_agent_skip_event_id(job),
                        event_type="data_agent.job.skipped",
                        message="Data Agent job delivery skipped before worker lease.",
                        metadata={
                            "ack_action": "ack",
                            "control_status": control_status,
                            "dispatch_id": job.dispatch_id,
                            "job_type": job.job_type,
                            "stage": "initial_status_check",
                            "worker_id": self.settings.data_agent_worker_id,
                        },
                    )
                except Exception:
                    logger.warning(
                        "Data Agent skipped-job audit failed; NAKing delivery for retry.",
                        extra={"job_id": job.job_id, "status": control_status},
                        exc_info=True,
                    )
                    await _nak_message(
                        message,
                        delay_seconds=data_agent_redelivery_delay(self.settings),
                    )
                    return
                await _ack_message(message)
                return

            current_task = asyncio.current_task()
            if current_task is not None:
                self._active_tasks[job.job_id] = current_task
            try:
                try:
                    control_lease = await self._lease(job, self.settings)
                except DataAgentLeaseConflict:
                    should_ack = False
                    await _nak_message(
                        message,
                        delay_seconds=data_agent_redelivery_delay(self.settings),
                    )
                    return
                except DataAgentLeaseUnavailable:
                    should_ack = False
                    await _nak_message(
                        message,
                        delay_seconds=data_agent_redelivery_delay(self.settings),
                    )
                    return

                await self._post_worker_heartbeat(
                    "busy",
                    current_run_id=job.job_id,
                    metadata={"active_tasks": len(self._active_tasks), "job_type": job.job_type},
                )
                await self._update_status(
                    job,
                    self.settings,
                    status="running",
                    progress_completed=0,
                    progress_total=job.resource_count,
                    message="Data Agent job started.",
                    event_metadata={"stage": "worker_started", "dispatch_id": job.dispatch_id},
                )

                async def progress(**kwargs: Any) -> None:
                    await self._update_status(
                        job,
                        self.settings,
                        status="running",
                        progress_completed=_int(kwargs.get("progress_completed"), default=0),
                        progress_total=_int(
                            kwargs.get("progress_total"), default=job.resource_count
                        ),
                        message=_string(kwargs.get("message")),
                        output_summary=_dict(kwargs.get("output_summary")),
                        metadata=_dict(kwargs.get("metadata")),
                        event_metadata=_dict(kwargs.get("event_metadata")),
                    )

                data_agent_heartbeat_task = _start_data_agent_heartbeat_task(
                    control_lease,
                    self.settings,
                    job=job,
                    renew_lease_func=self._renew_lease,
                    worker_task=current_task,
                    on_lease_lost=mark_control_lease_lost,
                    interval_seconds=data_agent_ack_extension_interval(self.settings),
                )
                try:
                    output_summary = await self._processor(job, progress)
                finally:
                    await _stop_data_agent_heartbeat_task(data_agent_heartbeat_task)
                    data_agent_heartbeat_task = None
                output_summary_dict = _dict(output_summary)
                terminal_status = _data_agent_output_terminal_status(output_summary_dict)
                terminal_progress_completed = job.resource_count
                if terminal_status != "succeeded":
                    terminal_progress_completed = max(
                        0,
                        min(
                            _int(output_summary_dict.get("processed"), default=0),
                            max(0, job.resource_count),
                        ),
                    )
                terminal_message = _data_agent_terminal_message(terminal_status)
                await self._update_status(
                    job,
                    self.settings,
                    status=terminal_status,
                    progress_completed=terminal_progress_completed,
                    progress_total=job.resource_count,
                    error=_data_agent_terminal_error(terminal_status, output_summary_dict),
                    message=terminal_message,
                    output_summary=output_summary_dict,
                    event_metadata={
                        "stage": _data_agent_terminal_stage(terminal_status),
                        "dispatch_id": job.dispatch_id,
                    },
                )
            except asyncio.CancelledError:
                if control_lease_lost:
                    should_ack = False
                    await _nak_message(
                        message,
                        delay_seconds=data_agent_redelivery_delay(self.settings),
                    )
                    return
                if self._shutting_down:
                    should_ack = False
                    await _nak_message(message)
                    return
                raise
            except RetryableDataAgentProcessingError:
                should_ack = False
                logger.warning(
                    "Data Agent processor is temporarily unavailable; NAKing delivery for retry.",
                    extra={"job_id": job.job_id, "job_type": job.job_type},
                    exc_info=True,
                )
                await _nak_message(
                    message,
                    delay_seconds=data_agent_redelivery_delay(self.settings),
                )
                return
            except Exception as exc:
                try:
                    await self._update_status(
                        job,
                        self.settings,
                        status="failed",
                        progress_completed=0,
                        progress_total=job.resource_count,
                        error=str(exc) or exc.__class__.__name__,
                        message="Data Agent job failed.",
                        event_metadata={
                            "stage": "worker_failed",
                            "error_type": exc.__class__.__name__,
                            "dispatch_id": job.dispatch_id,
                        },
                    )
                except Exception:
                    should_ack = False
                    await _nak_message(message)
                    raise
            finally:
                await _stop_data_agent_heartbeat_task(data_agent_heartbeat_task)
                if self._active_tasks.get(job.job_id) is current_task:
                    self._active_tasks.pop(job.job_id, None)
                await self._post_worker_heartbeat(
                    "idle",
                    metadata={"active_tasks": len(self._active_tasks)},
                )
                if control_lease is not None:
                    with contextlib.suppress(Exception):
                        await self._release_lease(control_lease, self.settings, job=job)
            if should_ack:
                await _ack_message(message)
        finally:
            await _stop_ack_progress_task(ack_progress_task)

    async def _maybe_post_idle_worker_heartbeat(self, last_posted_at: float) -> float:
        interval_seconds = self.settings.worker_heartbeat_interval_seconds
        if interval_seconds <= 0:
            return last_posted_at
        now = time.monotonic()
        if now - last_posted_at < interval_seconds:
            return last_posted_at
        await self._post_worker_heartbeat(
            "idle",
            metadata={"active_tasks": len(self._active_tasks)},
        )
        return now

    async def _post_worker_heartbeat(
        self,
        status: str,
        *,
        current_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        heartbeat_metadata = {
            "active_tasks": len(self._active_tasks),
            "max_concurrency": max(1, self.settings.worker_max_concurrency),
        }
        heartbeat_metadata.update(metadata or {})
        try:
            await self._worker_heartbeat(
                self.settings,
                status=status,
                current_run_id=current_run_id,
                metadata=heartbeat_metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Data Agent worker heartbeat failed; keeping worker alive.",
                extra={"worker_id": self.settings.worker_id, "status": status},
                exc_info=True,
            )

    async def _safe_job_status(self, job: DataAgentJobEnvelope) -> str | None:
        try:
            return await self._job_status(job, self.settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Control-plane data-agent job status hook failed; proceeding with job.",
                extra={"job_id": job.job_id},
                exc_info=True,
            )
            return None


def _data_agent_worker_identity(settings: RuntimeSettings) -> RuntimeSettings:
    return replace(
        settings,
        worker_id=settings.data_agent_worker_id,
        worker_kind=settings.data_agent_worker_kind,
    )


def data_agent_redelivery_delay(settings: RuntimeSettings) -> float | None:
    interval = data_agent_ack_extension_interval(settings)
    if interval <= 0:
        return None
    return interval


def _data_agent_output_terminal_status(output_summary: dict[str, Any]) -> str:
    status = _string(output_summary.get("terminal_status")).lower()
    if status in TERMINAL_DATA_AGENT_JOB_STATUSES:
        return status
    if output_summary.get("canceled") is True:
        return "canceled"
    return "succeeded"


def _data_agent_terminal_stage(status: str) -> str:
    if status == "canceled":
        return "worker_canceled"
    if status == "failed":
        return "worker_failed"
    return "worker_completed"


def _data_agent_terminal_message(status: str) -> str:
    if status == "canceled":
        return "Data Agent job canceled."
    if status == "failed":
        return "Data Agent job failed."
    return "Data Agent job completed."


def _data_agent_terminal_error(status: str, output_summary: dict[str, Any]) -> str:
    if status != "failed":
        return ""
    return _string(output_summary.get("error")) or _string(output_summary.get("summary"))


def _start_data_agent_heartbeat_task(
    control_lease: ControlPlaneDataAgentJobLease | None,
    settings: RuntimeSettings,
    *,
    job: DataAgentJobEnvelope,
    renew_lease_func: RenewDataAgentLeaseFunc,
    worker_task: asyncio.Task | None,
    on_lease_lost: Callable[[], None] | None,
    interval_seconds: float,
) -> asyncio.Task | None:
    if interval_seconds <= 0:
        return None
    return asyncio.create_task(
        _data_agent_heartbeat_loop(
            control_lease,
            settings,
            job=job,
            renew_lease_func=renew_lease_func,
            worker_task=worker_task,
            on_lease_lost=on_lease_lost,
            interval_seconds=interval_seconds,
        )
    )


async def _data_agent_heartbeat_loop(
    control_lease: ControlPlaneDataAgentJobLease | None,
    settings: RuntimeSettings,
    *,
    job: DataAgentJobEnvelope,
    renew_lease_func: RenewDataAgentLeaseFunc,
    worker_task: asyncio.Task | None,
    on_lease_lost: Callable[[], None] | None,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if control_lease is not None:
                renewed = await renew_lease_func(control_lease, settings, job=job)
                if renewed is not None:
                    control_lease = renewed
        except asyncio.CancelledError:
            raise
        except (DataAgentLeaseConflict, DataAgentLeaseUnavailable):
            if on_lease_lost is not None:
                on_lease_lost()
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
            logger.warning(
                "Data Agent job lease renewal failed; stopping active job.",
                extra={"job_id": job.job_id},
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Data Agent job lease heartbeat failed; keeping active job running.",
                extra={"job_id": job.job_id},
                exc_info=True,
            )


async def _stop_data_agent_heartbeat_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _start_ack_progress_task(
    message: Any | None, *, interval_seconds: float
) -> asyncio.Task | None:
    if message is None or interval_seconds <= 0 or not hasattr(message, "in_progress"):
        return None
    return asyncio.create_task(_ack_progress_loop(message, interval_seconds=interval_seconds))


async def _ack_progress_loop(message: Any, *, interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        with contextlib.suppress(Exception):
            await message.in_progress()


async def _stop_ack_progress_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _ack_message(message: Any) -> None:
    if hasattr(message, "ack"):
        with contextlib.suppress(Exception):
            await message.ack()


async def _nak_message(message: Any, *, delay_seconds: float | None = None) -> None:
    if not hasattr(message, "nak"):
        return
    with contextlib.suppress(Exception):
        if delay_seconds is None:
            await message.nak()
        else:
            try:
                await message.nak(delay=delay_seconds)
            except TypeError:
                await message.nak()


async def _reconcile_data_agent_consumer(
    js: Any,
    settings: RuntimeSettings,
    *,
    config: ConsumerConfig,
) -> None:
    from nats.js.errors import NotFoundError

    try:
        existing = await js.consumer_info(
            settings.data_agent_nats_stream,
            settings.data_agent_nats_worker_durable,
        )
    except NotFoundError:
        await js.add_consumer(settings.data_agent_nats_stream, config=config)
        return
    if _data_agent_consumer_config_matches(getattr(existing, "config", None), config):
        return
    try:
        await js.add_consumer(settings.data_agent_nats_stream, config=config)
    except Exception as exc:
        if not _consumer_update_rejected(exc):
            raise
        await js.delete_consumer(
            settings.data_agent_nats_stream,
            settings.data_agent_nats_worker_durable,
        )
        await js.add_consumer(settings.data_agent_nats_stream, config=config)


def _data_agent_consumer_config_matches(
    existing_config: Any, desired_config: ConsumerConfig
) -> bool:
    if existing_config is None:
        return False
    return (
        getattr(existing_config, "filter_subject", None) == desired_config.filter_subject
        and _ack_policy_equal(
            getattr(existing_config, "ack_policy", None), desired_config.ack_policy
        )
        and _float_equal(getattr(existing_config, "ack_wait", None), desired_config.ack_wait)
        and getattr(existing_config, "max_deliver", None) == desired_config.max_deliver
        and getattr(existing_config, "max_ack_pending", None) == desired_config.max_ack_pending
    )


def _ack_policy_equal(left: Any, right: Any) -> bool:
    return bool(getattr(left, "value", left) == getattr(right, "value", right))


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.001
    except (TypeError, ValueError):
        return bool(left == right)


def _consumer_update_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "configuration" in text
        or "already exists" in text
        or "consumer name already in use" in text
    )


def _harvest_finished_message_tasks(tasks: set[asyncio.Task]) -> set[asyncio.Task]:
    pending: set[asyncio.Task] = set()
    for task in tasks:
        if task.done():
            _log_message_task_result(task)
        else:
            pending.add(task)
    return pending


async def _wait_for_message_task_capacity(
    tasks: set[asyncio.Task],
    *,
    timeout: float,
) -> set[asyncio.Task]:
    if not tasks:
        return tasks
    done, pending = await asyncio.wait(
        tasks,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in done:
        _log_message_task_result(task)
    return set(pending)


async def _cancel_message_tasks(tasks: set[asyncio.Task]) -> None:
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            logger.error(
                "Data Agent worker message task failed during shutdown.",
                exc_info=(type(result), result, result.__traceback__),
            )


def _log_message_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.error(
            "Data Agent worker message task failed.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _string(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


async def amain() -> None:
    await NATSDataAgentWorker().run_forever()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
