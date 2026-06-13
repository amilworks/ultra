from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
import fcntl
import importlib.metadata
import json
import logging
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import nats
import nats.errors
from nats.js.api import AckPolicy, ConsumerConfig

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope

logger = logging.getLogger(__name__)

TERMINAL_CONTROL_PLANE_RUN_STATUSES = {"succeeded", "failed", "canceled"}
CONTROL_PLANE_RUN_NOT_FOUND_STATUS = "not_found"
SKIP_CONTROL_PLANE_RUN_STATUSES = {
    *TERMINAL_CONTROL_PLANE_RUN_STATUSES,
    CONTROL_PLANE_RUN_NOT_FOUND_STATUS,
}
CONTROL_PLANE_STATUSES_WITH_AUTHORITATIVE_TERMINAL_EVENT = {
    "succeeded",
    "failed",
    CONTROL_PLANE_RUN_NOT_FOUND_STATUS,
}
WORKER_TERMINAL_EVENT_KINDS = {"run.completed", "run.failed", "run.canceled"}


# The user that owns the job currently being processed. Control-plane HTTP
# calls attach it as X-Ultra-User-Id so dev-mode deployments without a worker
# token still resolve user-scoped runs. asyncio tasks created while handling a
# job (status monitor, heartbeats) inherit the value automatically.
_control_plane_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ultra_control_plane_user_id", default=None
)


def _control_plane_headers(
    settings: RuntimeSettings,
    *,
    json_body: bool = False,
) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    token = str(getattr(settings, "control_worker_token", "") or "").strip()
    if token:
        headers["X-Ultra-Worker-Token"] = token
    user_id = _control_plane_user_id.get()
    if user_id:
        headers["X-Ultra-User-Id"] = user_id
    return headers


def _control_plane_request_has_identity(settings: RuntimeSettings) -> bool:
    if str(getattr(settings, "control_worker_token", "") or "").strip():
        return True
    return bool(_control_plane_user_id.get())


def control_lease_validity_window(settings: RuntimeSettings) -> float:
    """How long a freshly acquired/renewed control-plane run lease can be
    trusted while renewals fail. 80% of the TTL leaves a safety margin so this
    worker stops before lease-expiry recovery hands the run to another worker.
    """
    ttl = max(1.0, float(settings.control_run_lease_ttl_seconds))
    return ttl * 0.8


class EventPublishError(RuntimeError):
    pass


class RunLeaseConflict(RuntimeError):
    pass


class RunLeaseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ControlPlaneRunLease:
    run_id: str
    worker_id: str
    lease_token: str
    lease_expires_at: str = ""


WorkerHeartbeatFunc = Callable[..., Awaitable[None]]


class RunLock:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def release(self) -> None:
        with contextlib.suppress(Exception):
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            self._handle.close()

    def __enter__(self) -> RunLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = exc_type, exc, traceback
        self.release()


def build_job_consumer_config(settings: RuntimeSettings) -> ConsumerConfig:
    return ConsumerConfig(
        durable_name=settings.nats_worker_durable,
        filter_subject=settings.nats_jobs_subject,
        ack_policy=AckPolicy.EXPLICIT,
        ack_wait=settings.worker_ack_wait_seconds,
        max_deliver=settings.worker_max_deliver,
        max_ack_pending=settings.worker_max_concurrency,
    )


def job_ack_extension_interval(settings: RuntimeSettings) -> float:
    ack_wait_seconds = max(0.1, float(settings.worker_ack_wait_seconds))
    safe_ceiling = max(0.1, ack_wait_seconds / 2.0)
    configured_interval = float(settings.worker_ack_progress_interval_seconds)
    if configured_interval <= 0:
        return safe_ceiling
    return min(configured_interval, safe_ceiling)


def active_duplicate_redelivery_delay(settings: RuntimeSettings) -> float:
    interval = job_ack_extension_interval(settings)
    if interval > 0:
        return interval
    return 1.0


def try_acquire_run_lock(settings: RuntimeSettings, run_id: str) -> RunLock | None:
    workspace_dir = Path(settings.workspace_root).expanduser() / run_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    handle = (workspace_dir / "run.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return None
        raise
    return RunLock(handle)


async def fetch_control_plane_run_status(
    run_id: str,
    settings: RuntimeSettings,
) -> str | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    quoted_run_id = urllib_parse.quote(run_id, safe="")
    url = f"{base_url}/v2/runs/{quoted_run_id}"
    timeout = max(0.1, float(settings.control_status_timeout_seconds))

    request_has_identity = _control_plane_request_has_identity(settings)

    def fetch() -> str | None:
        request = urllib_request.Request(
            url,
            method="GET",
            headers=_control_plane_headers(settings),
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                if request_has_identity:
                    return CONTROL_PLANE_RUN_NOT_FOUND_STATUS
                # An anonymous 404 is indistinguishable from an auth failure
                # (the control plane hides runs it cannot scope). Treating it
                # as authoritative would silently drop the job.
                logger.warning(
                    "Control-plane run status lookup returned 404 without worker identity; "
                    "proceeding with job.",
                    extra={"run_id": run_id},
                )
                return None
            logger.warning(
                "Control-plane run status lookup returned HTTP error; proceeding with job.",
                extra={"run_id": run_id, "status_code": exc.code},
            )
            return None
        except Exception:
            logger.warning(
                "Control-plane run status lookup failed; proceeding with job.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return None
        if not isinstance(payload, dict):
            return None
        status = str(payload.get("status") or "").strip().lower()
        return status or None

    return await asyncio.to_thread(fetch)


async def fetch_control_plane_user_profile(
    run_id: str,
    settings: RuntimeSettings,
) -> dict[str, Any] | None:
    """Fetch the run owner's self-described profile from the control plane.

    Uses the worker-scoped ``GET /v2/runs/{run_id}/user-profile`` endpoint; the
    browser's ``/v2/me`` is bound to a WorkOS session and cannot be read with a
    worker token. Returns the profile mapping or ``None``. Best-effort: never
    raises, so a profile lookup can't fail a run.
    """
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    if not _control_plane_request_has_identity(settings):
        return None
    quoted_run_id = urllib_parse.quote(run_id, safe="")
    url = f"{base_url}/v2/runs/{quoted_run_id}/user-profile"
    timeout = max(0.1, float(settings.control_status_timeout_seconds))

    def fetch() -> dict[str, Any] | None:
        request = urllib_request.Request(
            url, method="GET", headers=_control_plane_headers(settings)
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            logger.warning(
                "Control-plane profile lookup failed; proceeding without profile.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return None
        if not isinstance(payload, dict):
            return None
        profile = payload.get("profile")
        if isinstance(profile, dict) and any(
            str(value or "").strip() for value in profile.values()
        ):
            return profile
        return None

    return await asyncio.to_thread(fetch)


async def fetch_control_plane_run_max_sequence(
    run_id: str,
    settings: RuntimeSettings,
) -> int:
    """Return the run's highest persisted event sequence in the control plane,
    or 0 when none/unreachable. Pages forward from the last seen cursor."""
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return 0
    quoted_run_id = urllib_parse.quote(run_id, safe="")
    timeout = max(0.1, float(settings.control_status_timeout_seconds))
    page_limit = 500

    def fetch_page(after_sequence: int) -> list[dict[str, Any]]:
        query = urllib_parse.urlencode(
            {"limit": str(page_limit), "after_sequence": str(after_sequence)}
        )
        url = f"{base_url}/v2/runs/{quoted_run_id}/events?{query}"
        request = urllib_request.Request(
            url, method="GET", headers=_control_plane_headers(settings)
        )
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return []
        events = payload.get("events")
        return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []

    def fetch_max() -> int:
        cursor = 0
        try:
            while True:
                page = fetch_page(cursor)
                if not page:
                    break
                page_max = cursor
                for event in page:
                    sequence = int(event.get("sequence") or 0)
                    if sequence > page_max:
                        page_max = sequence
                if page_max <= cursor:
                    break
                cursor = page_max
                if len(page) < page_limit:
                    break
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return 0
            raise
        return cursor

    return await asyncio.to_thread(fetch_max)


async def fetch_control_plane_run_usage_summary(
    run_id: str,
    settings: RuntimeSettings,
) -> dict[str, Any] | None:
    """Return deduped token usage already persisted as run.token_usage events."""
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    if not _control_plane_request_has_identity(settings):
        return None
    quoted_run_id = urllib_parse.quote(run_id, safe="")
    timeout = max(0.1, float(settings.control_status_timeout_seconds))
    page_limit = 500

    def fetch_page(after_sequence: int) -> list[dict[str, Any]]:
        query = urllib_parse.urlencode(
            {"limit": str(page_limit), "after_sequence": str(after_sequence)}
        )
        url = f"{base_url}/v2/runs/{quoted_run_id}/events?{query}"
        request = urllib_request.Request(
            url, method="GET", headers=_control_plane_headers(settings)
        )
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return []
        events = payload.get("events")
        return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []

    def fetch_all() -> dict[str, Any] | None:
        after_sequence = 0
        seen_usage_ids: set[str] = set()
        summary = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        model = ""
        try:
            while True:
                events = fetch_page(after_sequence)
                if not events:
                    break
                page_max = after_sequence
                for event in events:
                    sequence = _event_sequence(event) or 0
                    if sequence > page_max:
                        page_max = sequence
                    if str(event.get("event_kind") or event.get("event_type") or "") != "run.token_usage":
                        continue
                    payload = event.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    usage_event_id = str(
                        payload.get("usage_event_id") or event.get("event_id") or ""
                    ).strip()
                    if not usage_event_id or usage_event_id in seen_usage_ids:
                        continue
                    seen_usage_ids.add(usage_event_id)
                    input_tokens = _positive_int(payload.get("input_tokens"))
                    output_tokens = _positive_int(payload.get("output_tokens"))
                    total_tokens = _positive_int(payload.get("total_tokens"))
                    if total_tokens <= 0:
                        total_tokens = input_tokens + output_tokens
                    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
                        continue
                    summary["input_tokens"] += input_tokens
                    summary["output_tokens"] += output_tokens
                    summary["total_tokens"] += total_tokens
                    if not model:
                        model = str(payload.get("model") or "").strip()
                if page_max <= after_sequence:
                    break
                after_sequence = page_max
                if len(events) < page_limit:
                    break
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        if summary["input_tokens"] <= 0 and summary["output_tokens"] <= 0 and summary["total_tokens"] <= 0:
            return None
        if model:
            summary["model"] = model
        return summary

    try:
        return await asyncio.to_thread(fetch_all)
    except Exception:
        logger.warning(
            "Could not read run token usage summary; proceeding without prior usage.",
            extra={"run_id": run_id},
            exc_info=True,
        )
        return None


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


async def acquire_control_plane_run_lease(
    run_id: str,
    settings: RuntimeSettings,
) -> ControlPlaneRunLease | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return None
    quoted_run_id = urllib_parse.quote(run_id, safe="")
    url = f"{base_url}/v2/runs/{quoted_run_id}/lease"
    payload = {
        "worker_id": settings.worker_id,
        "ttl_seconds": settings.control_run_lease_ttl_seconds,
    }
    return await asyncio.to_thread(
        _request_control_plane_run_lease,
        url,
        "POST",
        payload,
        settings,
    )


async def renew_control_plane_run_lease(
    lease: ControlPlaneRunLease,
    settings: RuntimeSettings,
) -> ControlPlaneRunLease | None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return lease
    quoted_run_id = urllib_parse.quote(lease.run_id, safe="")
    url = f"{base_url}/v2/runs/{quoted_run_id}/lease"
    payload = {
        "lease_token": lease.lease_token,
        "ttl_seconds": settings.control_run_lease_ttl_seconds,
    }
    renewed = await asyncio.to_thread(
        _request_control_plane_run_lease,
        url,
        "PATCH",
        payload,
        settings,
    )
    return renewed or lease


async def release_control_plane_run_lease(
    lease: ControlPlaneRunLease,
    settings: RuntimeSettings,
) -> None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return
    quoted_run_id = urllib_parse.quote(lease.run_id, safe="")
    url = f"{base_url}/v2/runs/{quoted_run_id}/lease"
    payload = {"lease_token": lease.lease_token}
    await asyncio.to_thread(
        _request_control_plane_run_lease,
        url,
        "DELETE",
        payload,
        settings,
    )


async def post_control_plane_worker_heartbeat(
    settings: RuntimeSettings,
    *,
    status: str,
    current_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return
    url = f"{base_url}/v2/workers/heartbeat"
    payload = {
        "worker_id": settings.worker_id,
        "worker_kind": settings.worker_kind,
        "status": status,
        "current_run_id": current_run_id,
        "hostname": socket.gethostname(),
        "version": _package_version(),
        "metadata": metadata or {},
    }
    await asyncio.to_thread(
        _request_control_plane_worker_heartbeat,
        url,
        payload,
        settings,
    )


def _request_control_plane_run_lease(
    url: str,
    method: str,
    payload: dict[str, Any],
    settings: RuntimeSettings,
) -> ControlPlaneRunLease | None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method=method,
        headers=_control_plane_headers(settings, json_body=True),
    )
    timeout = max(0.1, float(settings.control_status_timeout_seconds))
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            if method == "DELETE":
                _ = response.read()
                return None
            data = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        if exc.code == 409:
            raise RunLeaseConflict("control plane run lease is already held") from exc
        if settings.control_run_lease_required:
            raise RunLeaseUnavailable(
                f"control plane run lease request failed with HTTP {exc.code}"
            ) from exc
        logger.warning(
            "Control-plane run lease request returned HTTP error; proceeding without durable lease.",
            extra={"status_code": exc.code},
        )
        return None
    except Exception as exc:
        if settings.control_run_lease_required:
            raise RunLeaseUnavailable("control plane run lease request failed") from exc
        logger.warning(
            "Control-plane run lease request failed; proceeding without durable lease.",
            exc_info=True,
        )
        return None
    if not isinstance(data, dict):
        return None
    return ControlPlaneRunLease(
        run_id=str(data.get("run_id") or ""),
        worker_id=str(data.get("worker_id") or ""),
        lease_token=str(data.get("lease_token") or ""),
        lease_expires_at=str(data.get("lease_expires_at") or ""),
    )


def _request_control_plane_worker_heartbeat(
    url: str,
    payload: dict[str, Any],
    settings: RuntimeSettings,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers=_control_plane_headers(settings, json_body=True),
    )
    timeout = max(0.1, float(settings.control_status_timeout_seconds))
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            _ = response.read()
    except Exception:
        logger.warning(
            "Control-plane worker heartbeat failed; keeping worker alive.",
            extra={"worker_id": settings.worker_id, "status": payload.get("status")},
            exc_info=True,
        )


def _package_version() -> str:
    try:
        return importlib.metadata.version("ultra-deepagents")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


async def fetch_job_messages(
    subscription: Any,
    *,
    batch: int = 1,
    timeout: float = 1.0,
) -> list[Any]:
    try:
        return await subscription.fetch(batch=batch, timeout=timeout)
    except nats.errors.TimeoutError:
        return []


class NATSDeepAgentsWorker:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        run_job_func: Callable[..., Awaitable[str]] = run_job,
        run_status_func: Callable[[str, RuntimeSettings], Awaitable[str | None]] = (
            fetch_control_plane_run_status
        ),
        run_lease_func: Callable[
            [str, RuntimeSettings], Awaitable[ControlPlaneRunLease | None]
        ] = acquire_control_plane_run_lease,
        renew_run_lease_func: Callable[
            [ControlPlaneRunLease, RuntimeSettings],
            Awaitable[ControlPlaneRunLease | None],
        ] = renew_control_plane_run_lease,
        release_run_lease_func: Callable[
            [ControlPlaneRunLease, RuntimeSettings], Awaitable[None]
        ] = release_control_plane_run_lease,
        worker_heartbeat_func: WorkerHeartbeatFunc = post_control_plane_worker_heartbeat,
        user_profile_func: Callable[
            [str, RuntimeSettings], Awaitable[dict[str, Any] | None]
        ] = fetch_control_plane_user_profile,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self._run_job = run_job_func
        self._run_status = run_status_func
        self._run_lease = run_lease_func
        self._renew_run_lease = renew_run_lease_func
        self._release_run_lease = release_run_lease_func
        self._worker_heartbeat = worker_heartbeat_func
        self._user_profile = user_profile_func
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._canceled_run_reasons: dict[str, str] = {}
        self._shutting_down = False
        self._checkpoint_store: Any | None = None
        self._checkpointer: Any | None = None

    async def run_forever(self) -> None:
        nc = await nats.connect(self.settings.nats_url)
        js = nc.jetstream()
        cancel_subscription = None
        try:
            await self._ensure_stream(js)
            async def handle_cancel_message(message: Any) -> None:
                await self._handle_cancel_message(message, js)

            cancel_subscription = await nc.subscribe(
                self.settings.nats_cancel_subject,
                cb=handle_cancel_message,
            )
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
                    active_message_tasks.add(asyncio.create_task(self._process_message(message, js)))
        finally:
            self._shutting_down = True
            if "active_message_tasks" in locals():
                await _cancel_message_tasks(active_message_tasks)
            if cancel_subscription is not None:
                with contextlib.suppress(Exception):
                    await cancel_subscription.unsubscribe()
            await nc.drain()

    async def _ensure_stream(self, js: Any) -> None:
        subjects = [
            self.settings.nats_jobs_subject,
            self.settings.nats_events_subject,
            self.settings.nats_cancel_subject,
        ]
        try:
            await js.add_stream(name=self.settings.nats_stream, subjects=subjects)
        except Exception as exc:
            if "stream name already in use" not in str(exc).lower():
                raise

    async def _subscribe(self, js: Any) -> Any:
        config = build_job_consumer_config(self.settings)
        await _reconcile_consumer(js, self.settings, config=config)
        return await js.pull_subscribe(
            self.settings.nats_jobs_subject,
            durable=self.settings.nats_worker_durable,
            stream=self.settings.nats_stream,
            config=config,
        )

    async def _process_message(self, message: Any, js: Any) -> None:
        progress_task = _start_ack_progress_task(
            message,
            interval_seconds=job_ack_extension_interval(self.settings),
        )
        user_context_token: contextvars.Token | None = None
        try:
            try:
                payload = json.loads(message.data.decode("utf-8"))
                job = RunJobEnvelope.from_dict(payload)
            except Exception:
                logger.exception("Discarding malformed Deep Agents job payload.")
                await _ack_message(message)
                return
            user_context_token = _control_plane_user_id.set(
                str(job.user_id or "").strip() or None
            )

            last_published_sequence = 0
            terminal_event_published = False
            publish_lock = asyncio.Lock()

            async def publish_event(event: dict[str, Any]) -> None:
                nonlocal last_published_sequence, terminal_event_published
                async with publish_lock:
                    await self._publish_event(js, event)
                    event_sequence = _event_sequence(event)
                    if event_sequence is not None:
                        last_published_sequence = max(last_published_sequence, event_sequence)
                    event_kind = str(event.get("event_kind") or event.get("event_type") or "").strip()
                    if event_kind in WORKER_TERMINAL_EVENT_KINDS:
                        terminal_event_published = True

            heartbeat_index = 0
            control_lease: ControlPlaneRunLease | None = None
            lease_validity_deadline = float("inf")

            async def publish_worker_heartbeat() -> None:
                nonlocal heartbeat_index, last_published_sequence, control_lease
                nonlocal lease_validity_deadline
                async with publish_lock:
                    if terminal_event_published:
                        return
                    if control_lease is not None:
                        try:
                            renewed = await self._renew_run_lease(control_lease, self.settings)
                        except RunLeaseConflict:
                            # Another worker owns the run now: stop immediately.
                            raise
                        except RunLeaseUnavailable:
                            # The control plane is unreachable but the stored
                            # lease is still valid. Keep computing and retry on
                            # the next heartbeat; only give up once the lease
                            # itself can no longer be alive.
                            if time.monotonic() >= lease_validity_deadline:
                                raise
                            logger.warning(
                                "Control-plane lease renewal unavailable; retrying while the lease is still valid.",
                                extra={"run_id": job.run_id},
                            )
                            renewed = None
                        if renewed is not None:
                            control_lease = renewed
                            lease_validity_deadline = (
                                time.monotonic() + control_lease_validity_window(self.settings)
                            )
                    heartbeat_index += 1
                    sequence = last_published_sequence + 1
                    await self._publish_event(
                        js,
                        {
                            "event_id": (
                                f"evt_{job.run_id}_worker_heartbeat_{heartbeat_index:06d}"
                            ),
                            "sequence": sequence,
                            "run_id": job.run_id,
                            "thread_id": job.thread_id,
                            "event_kind": "run.heartbeat",
                            "event_type": "run",
                            "node_name": "worker",
                            "agent_role": "worker",
                            "level": "debug",
                            "message": "Worker heartbeat.",
                            "payload": {
                                "status": "alive",
                                "heartbeat_index": heartbeat_index,
                            },
                        },
                    )
                    last_published_sequence = sequence
                await self._post_worker_heartbeat(
                    "busy",
                    current_run_id=job.run_id,
                    metadata={"active_tasks": len(self._active_tasks)},
                )

            current_task = asyncio.current_task()
            run_lock: RunLock | None = None
            control_status = await self._safe_run_status(job.run_id)
            if control_status in SKIP_CONTROL_PLANE_RUN_STATUSES:
                logger.info(
                    "Skipping Deep Agents job because control plane run is not runnable.",
                    extra={"run_id": job.run_id, "status": control_status},
                )
                await _ack_message(message)
                return
            existing_task = self._active_tasks.get(job.run_id)
            if existing_task is not None:
                if existing_task.done():
                    self._active_tasks.pop(job.run_id, None)
                else:
                    logger.warning(
                        "Received duplicate delivery for active Deep Agents run; redelivering later.",
                        extra={"run_id": job.run_id},
                    )
                    await _nak_message(
                        message,
                        delay_seconds=active_duplicate_redelivery_delay(self.settings),
                    )
                    return
            run_lock = try_acquire_run_lock(self.settings, job.run_id)
            if run_lock is None:
                logger.warning(
                    "Received cross-worker duplicate delivery for locked Deep Agents run; redelivering later.",
                    extra={"run_id": job.run_id},
                )
                await _nak_message(
                    message,
                    delay_seconds=active_duplicate_redelivery_delay(self.settings),
                )
                return
            try:
                control_lease = await self._run_lease(job.run_id, self.settings)
                if control_lease is not None:
                    lease_validity_deadline = (
                        time.monotonic() + control_lease_validity_window(self.settings)
                    )
            except RunLeaseConflict:
                logger.warning(
                    "Received duplicate delivery for control-plane leased Deep Agents run; redelivering later.",
                    extra={"run_id": job.run_id},
                )
                await _nak_message(
                    message,
                    delay_seconds=active_duplicate_redelivery_delay(self.settings),
                )
                run_lock.release()
                run_lock = None
                return
            except RunLeaseUnavailable:
                logger.warning(
                    "Control-plane run lease is required but unavailable; redelivering job.",
                    extra={"run_id": job.run_id},
                    exc_info=True,
                )
                await _nak_message(
                    message,
                    delay_seconds=active_duplicate_redelivery_delay(self.settings),
                )
                run_lock.release()
                run_lock = None
                return
            if current_task is not None:
                self._active_tasks[job.run_id] = current_task
            await self._post_worker_heartbeat(
                "busy",
                current_run_id=job.run_id,
                metadata={"active_tasks": len(self._active_tasks)},
            )
            should_ack = True
            status_monitor_task: asyncio.Task | None = None
            heartbeat_task: asyncio.Task | None = None
            control_terminal_status: str | None = None
            control_lease_lost = False

            def mark_control_terminal_status(status: str) -> None:
                nonlocal control_terminal_status
                control_terminal_status = status

            def mark_control_lease_lost() -> None:
                nonlocal control_lease_lost
                control_lease_lost = True

            try:
                try:
                    control_status = await self._safe_run_status(job.run_id)
                    if control_status in SKIP_CONTROL_PLANE_RUN_STATUSES:
                        logger.info(
                            "Skipping Deep Agents job because control plane run became non-runnable before compute.",
                            extra={"run_id": job.run_id, "status": control_status},
                        )
                        return
                    if job.run_id in self._canceled_run_reasons:
                        await self._publish_canceled_event(
                            js,
                            job,
                            reason=self._canceled_run_reasons[job.run_id],
                            sequence=last_published_sequence + 1,
                        )
                        return
                    status_monitor_task = _start_control_status_monitor_task(
                        job.run_id,
                        self.settings,
                        self._run_status,
                        current_task,
                        on_terminal_status=mark_control_terminal_status,
                    )
                    heartbeat_task = _start_run_heartbeat_task(
                        publish_worker_heartbeat,
                        interval_seconds=job_ack_extension_interval(self.settings),
                        worker_task=current_task,
                        on_lease_lost=mark_control_lease_lost,
                    )
                    resume_kwargs: dict[str, Any] = {}
                    checkpointer = await self._ensure_checkpointer()
                    if checkpointer is not None:
                        # Seed the floor above the run's already-persisted events
                        # so a resumed run's event ids never collide with the
                        # original partial run's (which would be deduped/dropped).
                        resume_kwargs["checkpointer"] = checkpointer
                        resume_kwargs["sequence_floor"] = (
                            await self._run_event_sequence_floor(job.run_id)
                        )
                    prior_usage = await self._run_token_usage_summary(job.run_id)
                    if prior_usage:
                        resume_kwargs["prior_usage"] = prior_usage
                    user_profile = await self._load_user_profile(job.run_id)
                    if user_profile:
                        resume_kwargs["user_profile"] = user_profile
                    try:
                        response_text = await self._run_job(
                            job,
                            self.settings,
                            publish_event=publish_event,
                            **resume_kwargs,
                        )
                    finally:
                        await _stop_run_heartbeat_task(heartbeat_task)
                        heartbeat_task = None
                    if not terminal_event_published:
                        await self._publish_completed_event(
                            js,
                            job,
                            response_text=response_text,
                            sequence=last_published_sequence + 1,
                        )
                except asyncio.CancelledError:
                    if control_lease_lost:
                        should_ack = False
                        await _nak_message(
                            message,
                            delay_seconds=active_duplicate_redelivery_delay(self.settings),
                        )
                        return
                    if self._shutting_down and job.run_id not in self._canceled_run_reasons:
                        should_ack = False
                        await _nak_message(message)
                        return
                    if control_terminal_status in CONTROL_PLANE_STATUSES_WITH_AUTHORITATIVE_TERMINAL_EVENT:
                        return
                    await self._publish_canceled_event(
                        js,
                        job,
                        reason=self._canceled_run_reasons.get(job.run_id, "canceled"),
                        sequence=last_published_sequence + 1,
                    )
                except EventPublishError:
                    raise
                except Exception as exc:
                    logger.exception("Deep Agents job failed; terminal event should already be published.")
                    if not terminal_event_published:
                        await self._publish_failed_event(
                            js,
                            job,
                            exc,
                            sequence=last_published_sequence + 1,
                        )
            except EventPublishError:
                should_ack = False
                logger.exception("Deep Agents event publish failed; redelivering job.")
                await _nak_message(message)
            finally:
                await _stop_run_heartbeat_task(heartbeat_task)
                await _stop_control_status_monitor_task(status_monitor_task)
                if self._active_tasks.get(job.run_id) is current_task:
                    self._active_tasks.pop(job.run_id, None)
                await self._post_worker_heartbeat(
                    "idle",
                    metadata={"active_tasks": len(self._active_tasks)},
                )
                if control_lease is not None:
                    with contextlib.suppress(Exception):
                        await self._release_run_lease(control_lease, self.settings)
                if run_lock is not None:
                    run_lock.release()
                if should_ack:
                    self._clear_checkpointer_thread(job.run_id)
                    await _ack_message(message)
        finally:
            if user_context_token is not None:
                _control_plane_user_id.reset(user_context_token)
            await _stop_ack_progress_task(progress_task)

    async def _ensure_checkpointer(self) -> Any | None:
        """Lazily build the durable run checkpointer. Returns None when
        checkpointing is disabled or no control-plane database is configured,
        which preserves the prior restart-from-scratch behavior."""
        if not self.settings.checkpointer_enabled:
            return None
        if self._checkpointer is not None:
            return self._checkpointer
        from ultra_deepagents.checkpointing import (
            DurableCheckpointer,
            build_checkpoint_state_store,
        )

        try:
            store = build_checkpoint_state_store(self.settings.control_database_url)
        except Exception:
            logger.warning(
                "Could not build durable checkpoint store; runs will not resume.",
                exc_info=True,
            )
            return None
        if store is None:
            return None
        self._checkpoint_store = store
        self._checkpointer = DurableCheckpointer(store)
        logger.info("Durable run checkpointing enabled.")
        return self._checkpointer

    def _clear_checkpointer_thread(self, run_id: str) -> None:
        checkpointer = self._checkpointer
        if checkpointer is None:
            return
        clear_thread = getattr(checkpointer, "clear_thread", None)
        if clear_thread is None:
            return
        try:
            clear_thread(run_id)
        except Exception:
            logger.warning(
                "Checkpoint runtime cleanup failed; continuing after terminal run.",
                extra={"run_id": run_id},
                exc_info=True,
            )

    async def _run_event_sequence_floor(self, run_id: str) -> int:
        """The run's current max event sequence in the control plane, used to
        seed the worker sequencer so resumed events append without colliding
        with the original partial run's already-persisted events."""
        try:
            return await fetch_control_plane_run_max_sequence(run_id, self.settings)
        except Exception:
            logger.warning(
                "Could not read run event sequence floor; using 0.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return 0

    async def _run_token_usage_summary(self, run_id: str) -> dict[str, Any] | None:
        """Best-effort prior token usage for final payload continuity."""
        return await fetch_control_plane_run_usage_summary(run_id, self.settings)

    async def _load_user_profile(self, run_id: str) -> dict[str, Any] | None:
        """Best-effort fetch of the run owner's profile for memory seeding."""
        try:
            return await self._user_profile(run_id, self.settings)
        except Exception:
            logger.warning(
                "Could not load user profile for memory seeding; continuing.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return None

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
                "Control-plane worker heartbeat hook failed; keeping worker running.",
                extra={"worker_id": self.settings.worker_id, "status": status},
                exc_info=True,
            )

    async def _safe_run_status(self, run_id: str) -> str | None:
        try:
            return await self._run_status(run_id, self.settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Control-plane run status lookup failed; proceeding with Deep Agents job.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return None

    async def _handle_cancel_message(self, message: Any, js: Any) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
        except Exception:
            logger.exception("Discarding malformed Deep Agents cancel payload.")
            return
        await self._handle_cancel_payload(payload, js)

    async def _handle_cancel_payload(self, payload: dict[str, Any], js: Any) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        reason = str(payload.get("reason") or "canceled").strip() or "canceled"
        self._canceled_run_reasons[run_id] = reason
        task = self._active_tasks.get(run_id)
        if task is not None:
            task.cancel()

    async def _publish_event(self, js: Any, event: dict[str, Any]) -> None:
        headers = {}
        message_id = _event_message_id(event)
        if message_id:
            headers["Nats-Msg-Id"] = message_id
        try:
            await js.publish(
                self.settings.nats_events_subject,
                json.dumps(event, default=str).encode("utf-8"),
                headers=headers or None,
            )
        except Exception as exc:
            raise EventPublishError(str(exc)) from exc

    async def _publish_canceled_event(
        self,
        js: Any,
        job: RunJobEnvelope,
        *,
        reason: str,
        sequence: int,
    ) -> None:
        await self._publish_event(
            js,
            {
                "event_id": f"evt_{job.run_id}_canceled",
                "sequence": sequence,
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "event_kind": "run.canceled",
                "event_type": "run",
                "node_name": "coordinator",
                "agent_role": "coordinator",
                "level": "info",
                "message": "Run canceled.",
                "payload": {"reason": reason},
            },
        )

    async def _publish_failed_event(
        self,
        js: Any,
        job: RunJobEnvelope,
        exc: Exception,
        *,
        sequence: int,
    ) -> None:
        await self._publish_event(
            js,
            {
                "event_id": f"evt_{job.run_id}_worker_failed",
                "sequence": sequence,
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "event_kind": "run.failed",
                "event_type": "run",
                "node_name": "coordinator",
                "agent_role": "coordinator",
                "level": "error",
                "message": "Run failed.",
                "payload": {
                    "error": str(exc) or exc.__class__.__name__,
                    "error_type": exc.__class__.__name__,
                    "stage": "worker",
                },
            },
        )

    async def _publish_completed_event(
        self,
        js: Any,
        job: RunJobEnvelope,
        *,
        response_text: Any,
        sequence: int,
    ) -> None:
        await self._publish_event(
            js,
            {
                "event_id": f"evt_{job.run_id}_worker_completed",
                "sequence": sequence,
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "event_kind": "run.completed",
                "event_type": "run",
                "node_name": "coordinator",
                "agent_role": "coordinator",
                "level": "info",
                "message": "Run completed.",
                "payload": {"response_text": str(response_text or "")},
            },
        )


def _event_sequence(event: dict[str, Any]) -> int | None:
    try:
        sequence = int(event.get("sequence"))
    except (TypeError, ValueError):
        return None
    if sequence < 0:
        return None
    return sequence


def _event_message_id(event: dict[str, Any]) -> str:
    event_id = str(event.get("event_id") or "").strip()
    if event_id:
        return f"event:{event_id}"
    run_id = str(event.get("run_id") or "").strip()
    sequence = _event_sequence(event)
    if run_id and sequence is not None:
        return f"event:{run_id}:{sequence}"
    return ""


async def _reconcile_consumer(js: Any, settings: RuntimeSettings, *, config: ConsumerConfig) -> None:
    from nats.js.errors import NotFoundError

    try:
        existing = await js.consumer_info(settings.nats_stream, settings.nats_worker_durable)
    except NotFoundError:
        await js.add_consumer(settings.nats_stream, config=config)
        return
    if _consumer_config_matches(getattr(existing, "config", None), config):
        return
    try:
        await js.add_consumer(settings.nats_stream, config=config)
    except Exception as exc:
        if not _consumer_update_rejected(exc):
            raise
        await js.delete_consumer(settings.nats_stream, settings.nats_worker_durable)
        await js.add_consumer(settings.nats_stream, config=config)


def _consumer_config_matches(existing_config: Any, desired_config: ConsumerConfig) -> bool:
    if existing_config is None:
        return False
    return (
        getattr(existing_config, "filter_subject", None) == desired_config.filter_subject
        and _float_equal(getattr(existing_config, "ack_wait", None), desired_config.ack_wait)
        and getattr(existing_config, "max_deliver", None) == desired_config.max_deliver
        and getattr(existing_config, "max_ack_pending", None) == desired_config.max_ack_pending
    )


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.001
    except (TypeError, ValueError):
        return left == right


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
                "Deep Agents worker message task failed during shutdown.",
                exc_info=(type(result), result, result.__traceback__),
            )


def _log_message_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.error(
            "Deep Agents worker message task failed.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _start_ack_progress_task(message: Any | None, *, interval_seconds: float) -> asyncio.Task | None:
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


def _start_run_heartbeat_task(
    publish_heartbeat: Callable[[], Awaitable[None]],
    *,
    interval_seconds: float,
    worker_task: asyncio.Task | None = None,
    on_lease_lost: Callable[[], None] | None = None,
) -> asyncio.Task | None:
    if interval_seconds <= 0:
        return None
    return asyncio.create_task(
        _run_heartbeat_loop(
            publish_heartbeat,
            interval_seconds=interval_seconds,
            worker_task=worker_task,
            on_lease_lost=on_lease_lost,
        )
    )


async def _run_heartbeat_loop(
    publish_heartbeat: Callable[[], Awaitable[None]],
    *,
    interval_seconds: float,
    worker_task: asyncio.Task | None = None,
    on_lease_lost: Callable[[], None] | None = None,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await publish_heartbeat()
        except asyncio.CancelledError:
            raise
        except (RunLeaseConflict, RunLeaseUnavailable):
            if on_lease_lost is not None:
                on_lease_lost()
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
            logger.warning(
                "Control-plane run lease renewal failed; stopping active Deep Agents job.",
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Deep Agents run heartbeat publish failed; keeping active job running.",
                exc_info=True,
            )


async def _stop_run_heartbeat_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _start_control_status_monitor_task(
    run_id: str,
    settings: RuntimeSettings,
    run_status_func: Callable[[str, RuntimeSettings], Awaitable[str | None]],
    worker_task: asyncio.Task | None,
    *,
    on_terminal_status: Callable[[str], None] | None = None,
) -> asyncio.Task | None:
    interval_seconds = settings.control_status_poll_interval_seconds
    if worker_task is None or interval_seconds <= 0:
        return None
    return asyncio.create_task(
        _control_status_monitor_loop(
            run_id,
            settings,
            run_status_func,
            worker_task,
            interval_seconds=interval_seconds,
            on_terminal_status=on_terminal_status,
        )
    )


async def _control_status_monitor_loop(
    run_id: str,
    settings: RuntimeSettings,
    run_status_func: Callable[[str, RuntimeSettings], Awaitable[str | None]],
    worker_task: asyncio.Task,
    *,
    interval_seconds: float,
    on_terminal_status: Callable[[str], None] | None = None,
) -> None:
    while not worker_task.done():
        await asyncio.sleep(interval_seconds)
        if worker_task.done():
            return
        try:
            status = await run_status_func(run_id, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Control-plane run status monitor failed; keeping active job running.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            continue
        if status in SKIP_CONTROL_PLANE_RUN_STATUSES:
            logger.info(
                "Canceling active Deep Agents job because control plane run is no longer runnable.",
                extra={"run_id": run_id, "status": status},
            )
            if on_terminal_status is not None:
                on_terminal_status(status)
            worker_task.cancel()
            return


async def _stop_control_status_monitor_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _ack_message(message: Any) -> None:
    if not hasattr(message, "ack"):
        return
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


async def amain() -> None:
    await NATSDeepAgentsWorker().run_forever()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
