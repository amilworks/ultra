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
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import nats
import nats.errors
from nats.js.api import AckPolicy, ConsumerConfig

from ultra_deepagents.code_execution.cleanup import reap_orphaned_sandbox_containers
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import RunEventSequencer, run_job
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


WorkerHeartbeatFunc = Callable[..., Awaitable[None]]


class RunLock:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def release(self) -> None:
        with contextlib.suppress(Exception):
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            self._handle.close()


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


async def fetch_control_plane_run_events_snapshot(
    run_id: str,
    settings: RuntimeSettings,
) -> tuple[int, dict[str, Any] | None]:
    """One pagination pass over the run's persisted control-plane events that
    simultaneously computes the highest event sequence (resume floor) and the
    deduped run.token_usage summary, so job start pages the (potentially huge)
    event history once instead of twice. Returns ``(0, None)`` when the run has
    no events or does not exist. The usage half requires a request identity;
    the max sequence is computed regardless."""
    base_url = settings.control_base_url.rstrip("/")
    if not base_url:
        return 0, None
    include_usage = _control_plane_request_has_identity(settings)
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

    def fetch_snapshot() -> tuple[int, dict[str, Any] | None]:
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
                    if not include_usage:
                        continue
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
                return 0, None
            raise
        if not include_usage or (
            summary["input_tokens"] <= 0
            and summary["output_tokens"] <= 0
            and summary["total_tokens"] <= 0
        ):
            return after_sequence, None
        if model:
            summary["model"] = model
        return after_sequence, summary

    return await asyncio.to_thread(fetch_snapshot)


async def fetch_control_plane_run_max_sequence(
    run_id: str,
    settings: RuntimeSettings,
) -> int:
    """Return the run's highest persisted event sequence in the control plane,
    or 0 when none/unreachable. Pages forward from the last seen cursor."""
    max_sequence, _ = await fetch_control_plane_run_events_snapshot(run_id, settings)
    return max_sequence


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
    try:
        _, summary = await fetch_control_plane_run_events_snapshot(run_id, settings)
        return summary
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
    # A failed renewal must be visible (None), never masked by returning the
    # stale lease: masking silently extends trust in a lease the control
    # plane may already have handed to another worker.
    return renewed


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


# Connection-level NATS failures (dropped/drained link, no servers, stale connection)
# that must reconnect the worker's message pump rather than crash the process and orphan
# the in-flight run. nats.errors.Error is the base of all of them (DrainTimeoutError,
# ConnectionClosedError, ...); the normal idle-fetch TimeoutError is already swallowed in
# fetch_job_messages, so it never reaches the supervisor.
_RECOVERABLE_NATS_ERRORS = (nats.errors.Error, OSError)


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
        self._checkpointer: Any | None = None
        # Serialize sandbox-container sweeps so the startup sweep and the first periodic
        # tick (or overlapping ticks) never issue redundant docker calls on the same IDs.
        self._sandbox_reaper_lock = asyncio.Lock()

    async def run_forever(self) -> None:
        """Supervise the NATS message pump: serve one connection until a recoverable
        connection-level failure (dropped/drained link, common when heavy load starves
        the event loop and NATS heartbeats are missed), then reconnect with backoff
        instead of crashing the process — a crash orphans the in-flight run. A clean
        shutdown returns; non-NATS errors still propagate."""
        backoff = 1.0
        while True:
            try:
                await self._serve_one_connection()
                return
            except _RECOVERABLE_NATS_ERRORS as exc:
                logger.warning(
                    "NATS connection lost (%s); reconnecting in %.1fs",
                    type(exc).__name__,
                    backoff,
                    exc_info=exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _serve_one_connection(self) -> None:
        self._shutting_down = False
        nc = await nats.connect(self.settings.nats_url)
        js = nc.jetstream()
        cancel_subscription = None
        reaper_task: asyncio.Task | None = None
        try:
            await self._ensure_stream(js)
            # Clear any orphaned/leftover sandbox containers from a previous (possibly
            # crashed) worker before taking new jobs, then keep a lightweight periodic
            # sweep running. Both are best-effort and never block the message pump.
            await self._reap_sandbox_containers_once()
            if self.settings.sandbox_reaper_interval_seconds > 0:
                reaper_task = asyncio.create_task(self._sandbox_reaper_loop())
            async def handle_cancel_message(message: Any) -> None:
                await self._handle_cancel_message(message)

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
            if reaper_task is not None:
                reaper_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reaper_task
            if "active_message_tasks" in locals():
                await _cancel_message_tasks(active_message_tasks)
            if cancel_subscription is not None:
                with contextlib.suppress(Exception):
                    await cancel_subscription.unsubscribe()
            with contextlib.suppress(Exception):
                await nc.drain()

    async def _sandbox_reaper_loop(self) -> None:
        """Periodically reap orphaned/leftover sandbox containers until cancelled."""
        interval = self.settings.sandbox_reaper_interval_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._reap_sandbox_containers_once()

    async def _reap_sandbox_containers_once(self) -> None:
        """Run one best-effort sandbox-container sweep off the event loop.

        Never raises: a docker/daemon problem only logs. Serialized by a lock so the
        startup sweep and periodic ticks never duplicate work on the same container IDs.
        """
        try:
            async with self._sandbox_reaper_lock:
                reaped = await asyncio.to_thread(
                    reap_orphaned_sandbox_containers,
                    default_cap_seconds=self.settings.sandbox_timeout_seconds,
                    grace_seconds=self.settings.sandbox_reaper_grace_seconds,
                    max_per_sweep=self.settings.sandbox_reaper_max_per_sweep,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Sandbox container reap failed; continuing.", exc_info=True)
            return
        if reaped:
            logger.info(
                "Reaped orphaned sandbox containers.",
                extra={"reaped": len(reaped)},
            )

    async def _ensure_stream(self, js: Any) -> None:
        subjects = [
            self.settings.nats_jobs_subject,
            self.settings.nats_events_subject,
            _partitioned_events_wildcard(self.settings.nats_events_subject),
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

            # One sequencer for the whole run. Streamed events (via run_job) and
            # worker lifecycle events (heartbeats, terminal events, skips) all
            # allocate their source_sequence from this single counter, so two
            # different events can never claim the same (run_id, source_sequence)
            # and the per-run sequence stays gapless. Created below before any
            # event is emitted.
            run_sequencer: RunEventSequencer | None = None
            terminal_event_published = False
            publish_lock = asyncio.Lock()

            async def publish_event(event: dict[str, Any]) -> None:
                nonlocal terminal_event_published
                async with publish_lock:
                    await self._publish_event(js, event)
                    # Keep the shared sequencer at or above every published
                    # sequence so a subsequent worker-side lifecycle event never
                    # reuses a number an already-published event took. Normal
                    # streamed events come from sequencer.stamp(), so this is a
                    # no-op for them; it only matters if some path publishes an
                    # explicit sequence. Both reads/writes are await-free, so
                    # this is atomic against concurrent next_sequence() calls.
                    if run_sequencer is not None:
                        event_sequence = _event_sequence(event)
                        if event_sequence is not None and event_sequence > run_sequencer.sequence:
                            run_sequencer.sequence = event_sequence
                    event_kind = str(event.get("event_kind") or event.get("event_type") or "").strip()
                    if event_kind in WORKER_TERMINAL_EVENT_KINDS:
                        terminal_event_published = True

            heartbeat_index = 0
            control_lease: ControlPlaneRunLease | None = None

            async def publish_worker_heartbeat() -> None:
                nonlocal heartbeat_index, control_lease
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
                            # The control plane is unreachable. GPU work in
                            # flight is expensive; keep computing and retry the
                            # renewal on every heartbeat. The control plane's
                            # token-matched renewal revives even an expired
                            # lease after it heals, and if it requeued the run
                            # meanwhile the next renewal gets an authoritative
                            # 409 (RunLeaseConflict) which stops this worker
                            # within one heartbeat interval.
                            logger.warning(
                                "Control-plane lease renewal unavailable; continuing compute and retrying.",
                                extra={"run_id": job.run_id},
                            )
                            renewed = None
                        if renewed is not None:
                            control_lease = renewed
                    heartbeat_index += 1
                    sequence = run_sequencer.next_sequence()
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
                await self._post_worker_heartbeat(
                    "busy",
                    current_run_id=job.run_id,
                    metadata={"active_tasks": len(self._active_tasks)},
                )

            current_task = asyncio.current_task()
            run_lock: RunLock | None = None
            # Seed with a fresh floor (0); re-seeded to the resume floor below
            # before compute starts. Skip/cancel paths only ever emit a single
            # terminal event, and a terminal run drops late events, so their
            # low sequence never matters.
            run_sequencer = RunEventSequencer(job.run_id)
            control_status = await self._safe_run_status(job.run_id)
            if control_status in SKIP_CONTROL_PLANE_RUN_STATUSES:
                logger.info(
                    "Skipping Deep Agents job because control plane run is not runnable.",
                    extra={"run_id": job.run_id, "status": control_status},
                )
                try:
                    await self._publish_skipped_event(
                        js,
                        job,
                        control_status=control_status,
                        stage="initial_status_check",
                        sequence=run_sequencer.next_sequence(),
                    )
                except EventPublishError:
                    logger.exception("Deep Agents skip event publish failed; redelivering job.")
                    await _nak_message(message)
                    return
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
                        await self._publish_skipped_event(
                            js,
                            job,
                            control_status=control_status,
                            stage="pre_compute_recheck",
                            sequence=run_sequencer.next_sequence(),
                        )
                        return
                    if job.run_id in self._canceled_run_reasons:
                        await self._publish_canceled_event(
                            js,
                            job,
                            reason=self._canceled_run_reasons[job.run_id],
                            sequence=run_sequencer.next_sequence(),
                        )
                        return
                    status_monitor_task = _start_control_status_monitor_task(
                        job.run_id,
                        self.settings,
                        self._run_status,
                        current_task,
                        on_terminal_status=mark_control_terminal_status,
                    )
                    resume_kwargs: dict[str, Any] = {}
                    checkpointer = await self._ensure_checkpointer()
                    if checkpointer is not None:
                        resume_kwargs["checkpointer"] = checkpointer
                    # One pass over the run's persisted events yields both the
                    # resume sequence floor and any prior token usage, instead
                    # of paging the full event history twice. Seed the shared
                    # sequencer above the run's already-persisted events BEFORE
                    # starting the heartbeat, so a resumed run's events —
                    # streamed and heartbeat alike — never reuse a
                    # source_sequence the original partial run already used
                    # (which the control plane would dedup or, under strict
                    # partition ingest, stall on).
                    resume_floor, prior_usage = await self._run_events_snapshot(job.run_id)
                    if resume_floor > run_sequencer.sequence:
                        run_sequencer.sequence = resume_floor
                    heartbeat_task = _start_run_heartbeat_task(
                        publish_worker_heartbeat,
                        interval_seconds=job_ack_extension_interval(self.settings),
                        worker_task=current_task,
                        on_lease_lost=mark_control_lease_lost,
                    )
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
                            sequencer=run_sequencer,
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
                            sequence=run_sequencer.next_sequence(),
                        )
                except asyncio.CancelledError:
                    # Cancellation is otherwise handled silently below, which makes an
                    # orphaned "running" run (terminal event never published) impossible to
                    # diagnose. Record which branch fires and the deciding state so a stuck
                    # run's cause is visible in the worker log (stderr).
                    logger.warning(
                        "Deep Agents run task cancelled; classifying for terminal handling.",
                        extra={
                            "run_id": job.run_id,
                            "control_lease_lost": control_lease_lost,
                            "shutting_down": self._shutting_down,
                            "control_terminal_status": control_terminal_status,
                            "explicitly_canceled": job.run_id in self._canceled_run_reasons,
                            "terminal_event_published": terminal_event_published,
                        },
                    )
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
                        await self._publish_skipped_event(
                            js,
                            job,
                            control_status=control_terminal_status,
                            stage="status_monitor",
                            sequence=run_sequencer.next_sequence(),
                        )
                        return
                    await self._publish_canceled_event(
                        js,
                        job,
                        reason=self._canceled_run_reasons.get(job.run_id, "canceled"),
                        sequence=run_sequencer.next_sequence(),
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
                            sequence=run_sequencer.next_sequence(),
                        )
            except EventPublishError:
                should_ack = False
                logger.exception("Deep Agents event publish failed; redelivering job.")
                await _nak_message(
                    message,
                    delay_seconds=active_duplicate_redelivery_delay(self.settings),
                )
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
                    await self._delete_checkpointer_thread(job.run_id)
                    await _ack_message(message)
                # This worker is done with the run either way (terminal ack or
                # NAK back to the queue), so drop its cancel bookkeeping. A
                # redelivery that lost the in-memory fast path still skips via
                # the authoritative control-plane status check above.
                self._canceled_run_reasons.pop(job.run_id, None)
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
        self._checkpointer = DurableCheckpointer(store)
        logger.info("Durable run checkpointing enabled.")
        # Reap abandoned checkpoint rows (runs that crashed before terminal ack)
        # on worker startup. Completed runs already delete their own row on ack;
        # this backstops stragglers without a long-lived background loop.
        retention = self.settings.checkpoint_retention_seconds
        if retention > 0:
            try:
                reaped = await self._checkpointer.gc(retention)
                if reaped:
                    logger.info(
                        "Reaped abandoned durable checkpoint rows.",
                        extra={"reaped": reaped},
                    )
            except Exception:
                logger.warning("Durable checkpoint GC at startup failed.", exc_info=True)
        return self._checkpointer

    async def _delete_checkpointer_thread(self, run_id: str) -> None:
        """Drop a terminal run's in-memory slice AND its durable row.

        A run that reached terminal ack (succeeded/failed/canceled) will never be
        redelivered or resumed, so its durable checkpoint row must be deleted to
        keep the shared control-plane Postgres from accumulating completed-run
        state forever. Best-effort: a cleanup failure never blocks the ack."""
        checkpointer = self._checkpointer
        if checkpointer is None:
            return
        try:
            delete_thread = getattr(checkpointer, "delete_thread", None)
            if delete_thread is not None:
                await delete_thread(run_id)
                return
            clear_thread = getattr(checkpointer, "clear_thread", None)
            if clear_thread is not None:
                clear_thread(run_id)
        except Exception:
            logger.warning(
                "Checkpoint cleanup failed; continuing after terminal run.",
                extra={"run_id": run_id},
                exc_info=True,
            )

    async def _run_events_snapshot(self, run_id: str) -> tuple[int, dict[str, Any] | None]:
        """Single-pass fetch of the run's max persisted event sequence (used to
        seed the worker sequencer so resumed events append without colliding
        with the original partial run's already-persisted events) and its
        best-effort prior token usage for final payload continuity. Degrades to
        (0, None) — fresh floor, no prior usage — when unreadable."""
        try:
            return await fetch_control_plane_run_events_snapshot(run_id, self.settings)
        except Exception:
            logger.warning(
                "Could not read run events snapshot; using sequence floor 0 and no prior usage.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return 0, None

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

    async def _handle_cancel_message(self, message: Any) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
        except Exception:
            logger.exception("Discarding malformed Deep Agents cancel payload.")
            return
        await self._handle_cancel_payload(payload)

    async def _handle_cancel_payload(self, payload: dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        reason = str(payload.get("reason") or "canceled").strip() or "canceled"
        self._canceled_run_reasons[run_id] = reason
        # Cancels fan out to every worker, so bound the map: entries for runs
        # this worker processes are popped when the job finishes, but runs
        # handled elsewhere would otherwise accumulate forever. Evicting an old
        # entry only shifts a redelivered canceled run from the in-memory fast
        # path to the authoritative control-plane status check. (Single event
        # loop; dicts preserve insertion order.)
        while len(self._canceled_run_reasons) > 1024:
            self._canceled_run_reasons.pop(next(iter(self._canceled_run_reasons)))
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
                _event_publish_subject(self.settings, event),
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

    async def _publish_skipped_event(
        self,
        js: Any,
        job: RunJobEnvelope,
        *,
        control_status: str,
        stage: str,
        sequence: int,
    ) -> None:
        await self._publish_event(
            js,
            {
                "event_id": f"evt_{job.run_id}_worker_skipped",
                "sequence": sequence,
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "event_kind": "run.worker_skipped",
                "event_type": "run",
                "node_name": "worker",
                "agent_role": "worker",
                "level": "warning",
                "message": "Worker skipped job because the control plane run is not runnable.",
                "payload": {
                    "ack_action": "ack",
                    "control_status": control_status,
                    "reason": "control_plane_not_runnable",
                    "stage": stage,
                    "worker_id": self.settings.worker_id,
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


def _event_partition_count(settings: RuntimeSettings) -> int:
    return min(256, max(1, int(settings.nats_event_partitions or 1)))


def _fnv1a_32(value: str) -> int:
    hash_value = 2166136261
    for byte in value.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def _partitioned_events_wildcard(events_subject: str) -> str:
    return f"{str(events_subject or 'ultra.runs.events').strip() or 'ultra.runs.events'}.p.*"


def _event_publish_subject(settings: RuntimeSettings, event: dict[str, Any]) -> str:
    events_subject = str(settings.nats_events_subject or "ultra.runs.events").strip()
    if not events_subject:
        events_subject = "ultra.runs.events"
    run_id = str(event.get("run_id") or "")
    partition = _fnv1a_32(run_id) % _event_partition_count(settings)
    return f"{events_subject}.p.{partition}"


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
        and _ack_policy_equal(getattr(existing_config, "ack_policy", None), desired_config.ack_policy)
        and _float_equal(getattr(existing_config, "ack_wait", None), desired_config.ack_wait)
        and getattr(existing_config, "max_deliver", None) == desired_config.max_deliver
        and getattr(existing_config, "max_ack_pending", None) == desired_config.max_ack_pending
    )


def _ack_policy_equal(left: Any, right: Any) -> bool:
    return getattr(left, "value", left) == getattr(right, "value", right)


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
