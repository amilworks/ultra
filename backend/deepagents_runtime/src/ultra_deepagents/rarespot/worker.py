from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
    ControlPlaneRunLease,
    RunLeaseConflict,
    RunLeaseUnavailable,
    SKIP_CONTROL_PLANE_RUN_STATUSES,
    WorkerHeartbeatFunc,
    acquire_control_plane_run_lease,
    fetch_control_plane_run_status,
    post_control_plane_worker_heartbeat,
    release_control_plane_run_lease,
    renew_control_plane_run_lease,
)
from ultra_deepagents.rarespot.config import RareSpotConfig
from ultra_deepagents.rarespot.inference import run_rarespot_inference
from ultra_deepagents.rarespot.staging import stage_inputs
from ultra_deepagents.rarespot.store import EventPublishError, NATSRunEventStore, PostgresRunStore
from ultra_deepagents.rarespot.uploads import resolve_uploaded_file_ids, unique_file_ids

logger = logging.getLogger(__name__)


class RareSpotJobRunner(Protocol):
    async def run(self, job: dict[str, Any]) -> dict[str, Any]:
        ...


RunLeaseFunc = Callable[[str, RuntimeSettings], Awaitable[ControlPlaneRunLease | None]]
RunStatusFunc = Callable[[str, RuntimeSettings], Awaitable[str | None]]
RenewRunLeaseFunc = Callable[
    [ControlPlaneRunLease, RuntimeSettings],
    Awaitable[ControlPlaneRunLease | None],
]
ReleaseRunLeaseFunc = Callable[[ControlPlaneRunLease, RuntimeSettings], Awaitable[None]]
CancelReasonFunc = Callable[[str], str | None]


class DefaultRareSpotJobRunner:
    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or RuntimeSettings.from_env()

    async def run(self, job: dict[str, Any]) -> dict[str, Any]:
        config = RareSpotConfig.from_settings(self.settings)
        run_id = str(job["run_id"])
        thread_id = str(job.get("thread_id") or "")
        artifact_root = Path(str(job.get("artifact_root") or config.artifact_root)).expanduser().resolve()
        output_dir = artifact_root / run_id
        metadata = dict(job.get("metadata") or {})
        selection_context = metadata.get("selection_context")
        if not isinstance(selection_context, dict):
            selection_context = {}
        config = apply_rarespot_metadata_overrides(config, metadata)
        local_paths = host_readable_local_paths(
            job.get("local_paths") or metadata.get("local_paths") or metadata.get("input_paths") or [],
            workspace_root=self.settings.workspace_root,
        )
        file_ids = unique_file_ids(
            [
                *(job.get("file_ids") or []),
                *(metadata.get("file_ids") or []),
                *(selection_context.get("focused_file_ids") or []),
            ]
        )
        upload_resolution = await resolve_uploaded_file_ids(
            file_ids,
            upload_roots=config.upload_roots,
            allowed_roots=config.allowed_input_roots,
            database_url=config.upload_database_url,
        )
        local_paths.extend(str(path) for path in upload_resolution.image_paths)
        resource_uris = list(job.get("resource_uris") or metadata.get("resource_uris") or [])
        dataset_uris = list(job.get("dataset_uris") or metadata.get("dataset_uris") or [])
        if file_ids and not local_paths and not resource_uris and not dataset_uris:
            missing = ", ".join(upload_resolution.missing_file_ids)
            raise FileNotFoundError(f"No uploaded image files could be resolved for RareSpot inference: {missing}")
        staged = await stage_inputs(
            local_paths=[str(path) for path in local_paths],
            resource_uris=[str(uri) for uri in resource_uris],
            dataset_uris=[str(uri) for uri in dataset_uris],
            stage_dir=output_dir / "staged",
            config=config,
        )
        image_paths = list(staged["image_paths"])
        return await asyncio.to_thread(
            run_rarespot_inference,
            image_paths=image_paths,
            run_id=run_id,
            thread_id=thread_id,
            output_dir=output_dir,
            config=config,
        )


RARESPOT_CONSUMER_DURABLE = "rarespot-ecology-worker"


def host_readable_local_paths(paths: list[str] | tuple[str, ...], *, workspace_root: str = "") -> list[str]:
    readable: list[str] = []
    workspace = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root else None
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        if path.startswith("/workspace/"):
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        if workspace is not None and (resolved == workspace or workspace in resolved.parents):
            continue
        readable.append(path)
    return readable


def apply_rarespot_metadata_overrides(
    config: RareSpotConfig,
    metadata: dict[str, Any] | None,
) -> RareSpotConfig:
    raw = (metadata or {}).get("rarespot_config")
    if not isinstance(raw, dict):
        return config
    updates: dict[str, Any] = {}
    confidence = _optional_float(raw.get("confidence_threshold"))
    if confidence is not None:
        updates["conf"] = _bounded(confidence, minimum=0.001, maximum=1.0)
    iou = _optional_float(raw.get("iou_threshold"))
    if iou is not None:
        updates["iou"] = _bounded(iou, minimum=0.001, maximum=1.0)
    tile_overlap = _optional_float(raw.get("tile_overlap"))
    if tile_overlap is not None:
        updates["tile_overlap"] = _bounded(tile_overlap, minimum=0.0, maximum=0.9)
    image_size = _optional_int(raw.get("image_size"))
    if image_size is not None:
        updates["tile_size"] = max(128, min(4096, image_size))
    if "spectral" in raw:
        updates["spectral"] = _optional_bool(raw.get("spectral"), default=config.spectral)
    return replace(config, **updates) if updates else config


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _bounded(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_rarespot_store(js: Any, settings: RuntimeSettings) -> Any:
    if settings.rarespot_database_url:
        return PostgresRunStore(settings.rarespot_database_url)
    return NATSRunEventStore(js, settings)


def build_rarespot_consumer_config(settings: RuntimeSettings) -> Any:
    from nats.js import api

    return api.ConsumerConfig(
        durable_name=RARESPOT_CONSUMER_DURABLE,
        ack_wait=settings.rarespot_nats_ack_wait_seconds,
        filter_subject=settings.rarespot_nats_jobs_subject,
    )


def rarespot_ack_extension_interval(settings: RuntimeSettings) -> float:
    ack_wait_seconds = max(0.1, float(settings.rarespot_nats_ack_wait_seconds))
    safe_ceiling = max(0.1, ack_wait_seconds / 2.0)
    configured_interval = float(settings.rarespot_nats_ack_progress_interval_seconds)
    if configured_interval <= 0:
        return safe_ceiling
    return min(configured_interval, safe_ceiling)


async def subscribe_rarespot_jobs(js: Any, settings: RuntimeSettings) -> Any:
    config = build_rarespot_consumer_config(settings)
    await _reconcile_rarespot_consumer(js, settings, config=config)
    return await js.pull_subscribe(
        settings.rarespot_nats_jobs_subject,
        durable=RARESPOT_CONSUMER_DURABLE,
        stream=settings.rarespot_nats_stream,
        config=config,
    )


async def _reconcile_rarespot_consumer(
    js: Any,
    settings: RuntimeSettings,
    *,
    config: Any,
) -> None:
    from nats.js.errors import NotFoundError

    try:
        existing = await js.consumer_info(settings.rarespot_nats_stream, RARESPOT_CONSUMER_DURABLE)
    except NotFoundError:
        await js.add_consumer(settings.rarespot_nats_stream, config=config)
        return

    if _rarespot_consumer_config_matches(getattr(existing, "config", None), config):
        return

    try:
        await js.add_consumer(settings.rarespot_nats_stream, config=config)
    except Exception as exc:
        if not _is_consumer_update_rejected(exc):
            raise
        await js.delete_consumer(settings.rarespot_nats_stream, RARESPOT_CONSUMER_DURABLE)
        await js.add_consumer(settings.rarespot_nats_stream, config=config)


def _rarespot_consumer_config_matches(existing_config: Any, desired_config: Any) -> bool:
    if existing_config is None:
        return False
    return (
        getattr(existing_config, "filter_subject", None) == getattr(desired_config, "filter_subject", None)
        and _float_equal(
            getattr(existing_config, "ack_wait", None),
            getattr(desired_config, "ack_wait", None),
        )
    )


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.001
    except (TypeError, ValueError):
        return left == right


def _is_consumer_update_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "configuration" in text
        or "already exists" in text
        or "consumer name already in use" in text
    )


async def process_rarespot_job(
    payload: dict[str, Any],
    *,
    store: Any,
    runner: RareSpotJobRunner,
    message: Any | None = None,
    settings: RuntimeSettings | None = None,
    run_status_func: RunStatusFunc = fetch_control_plane_run_status,
    run_lease_func: RunLeaseFunc = acquire_control_plane_run_lease,
    renew_run_lease_func: RenewRunLeaseFunc = renew_control_plane_run_lease,
    release_run_lease_func: ReleaseRunLeaseFunc = release_control_plane_run_lease,
    cancel_reason_func: CancelReasonFunc | None = None,
    worker_heartbeat_func: WorkerHeartbeatFunc = post_control_plane_worker_heartbeat,
    ack_progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    run_id = str(payload["run_id"])
    thread_id = str(payload.get("thread_id") or "")
    settings = settings or RuntimeSettings.from_env()
    settings = _rarespot_worker_identity(settings)
    artifact_ids: list[str] = []
    ack_progress_task = _start_ack_progress_task(
        message,
        interval_seconds=ack_progress_interval_seconds,
    )
    run_heartbeat_task: asyncio.Task | None = None
    control_lease: ControlPlaneRunLease | None = None
    control_lease_lost = False

    def mark_control_lease_lost() -> None:
        nonlocal control_lease_lost
        control_lease_lost = True

    try:
        control_status = await _safe_run_status(run_id, settings, run_status_func)
        if control_status in SKIP_CONTROL_PLANE_RUN_STATUSES:
            logger.info(
                "Skipping RareSpot job because control plane run is not runnable.",
                extra={"run_id": run_id, "status": control_status},
            )
            if message is not None:
                await message.ack()
            return {"run_id": run_id, "skipped_status": control_status}
        try:
            control_lease = await run_lease_func(run_id, settings)
        except RunLeaseConflict:
            logger.warning(
                "RareSpot job is already held by another control-plane run lease; redelivering later.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            await _nak_message(
                message,
                delay_seconds=_rarespot_redelivery_delay(ack_progress_interval_seconds),
            )
            return {"run_id": run_id, "lease_conflict": True}
        except RunLeaseUnavailable:
            logger.warning(
                "RareSpot control-plane run lease is required but unavailable; redelivering job.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            await _nak_message(
                message,
                delay_seconds=_rarespot_redelivery_delay(ack_progress_interval_seconds),
            )
            return {"run_id": run_id, "lease_unavailable": True}
        await _post_worker_heartbeat(
            worker_heartbeat_func,
            settings,
            status="busy",
            current_run_id=run_id,
            metadata={"active_tasks": 1},
        )
        await store.update_run_status(run_id, "running")
        await store.append_run_event(
            run_id,
            thread_id,
            "run.started",
            "RareSpot ecology inference started.",
            {"workflow_kind": "rarespot_ecology"},
        )
        run_heartbeat_task = _start_run_heartbeat_task(
            store,
            run_id,
            thread_id,
            interval_seconds=ack_progress_interval_seconds,
            control_lease=control_lease,
            settings=settings,
            renew_run_lease_func=renew_run_lease_func,
            worker_task=asyncio.current_task(),
            on_lease_lost=mark_control_lease_lost,
        )
        try:
            result = await runner.run(payload)
        finally:
            await _stop_run_heartbeat_task(run_heartbeat_task)
            run_heartbeat_task = None
        for artifact in result.get("artifacts") or []:
            artifact.setdefault("run_id", run_id)
            artifact.setdefault("thread_id", thread_id)
            artifact.setdefault("tool_name", "rarespot_ecology_inference")
            created = await store.create_artifact(artifact)
            artifact_ids.append(str(created.get("artifact_id") or ""))
        summary = {
            "run_id": run_id,
            "configuration": result.get("configuration") or {},
            "counts_by_class": result.get("counts_by_class") or {},
            "confidence_summary": result.get("confidence_summary") or {},
            "top_spectral_review_candidates": result.get("top_spectral_review_candidates") or [],
            "artifact_ids": [artifact_id for artifact_id in artifact_ids if artifact_id],
        }
        await store.update_run_status(run_id, "succeeded", response_text=json.dumps(summary))
        await store.append_run_event(
            run_id,
            thread_id,
            "run.completed",
            "RareSpot ecology inference completed.",
            summary,
        )
        if message is not None:
            await message.ack()
        return summary
    except asyncio.CancelledError:
        if control_lease_lost:
            await _nak_message(
                message,
                delay_seconds=_rarespot_redelivery_delay(ack_progress_interval_seconds),
            )
            return {"run_id": run_id, "lease_lost": True}
        cancel_reason = _cancel_reason(run_id, cancel_reason_func)
        if cancel_reason:
            try:
                await store.update_run_status(run_id, "canceled", error=cancel_reason)
                await store.append_run_event(
                    run_id,
                    thread_id,
                    "run.canceled",
                    "RareSpot ecology inference canceled.",
                    {"reason": cancel_reason},
                    event_id=f"evt_{run_id}_canceled",
                )
            except EventPublishError:
                await _nak_message(message)
                raise
            if message is not None:
                await message.ack()
            return {"run_id": run_id, "canceled": True}
        await _nak_message(message)
        raise
    except EventPublishError:
        await _nak_message(message)
        raise
    except Exception as exc:
        try:
            await store.update_run_status(run_id, "failed", error=str(exc))
            await store.append_run_event(
                run_id,
                thread_id,
                "run.failed",
                "RareSpot ecology inference failed.",
                {"error": str(exc)},
            )
        except EventPublishError:
            await _nak_message(message)
            raise
        if message is not None:
            await message.ack()
        raise
    finally:
        await _stop_run_heartbeat_task(run_heartbeat_task)
        await _stop_ack_progress_task(ack_progress_task)
        await _post_worker_heartbeat(
            worker_heartbeat_func,
            settings,
            status="idle",
            metadata={"active_tasks": 0},
        )
        if control_lease is not None:
            with contextlib.suppress(Exception):
                await release_run_lease_func(control_lease, settings)


async def _safe_run_status(
    run_id: str,
    settings: RuntimeSettings,
    run_status_func: RunStatusFunc,
) -> str | None:
    try:
        return await run_status_func(run_id, settings)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "RareSpot control-plane run status lookup failed; proceeding with job.",
            extra={"run_id": run_id},
            exc_info=True,
        )
        return None


def _cancel_reason(run_id: str, cancel_reason_func: CancelReasonFunc | None) -> str:
    if cancel_reason_func is None:
        return ""
    try:
        reason = str(cancel_reason_func(run_id) or "").strip()
    except Exception:
        logger.warning(
            "RareSpot cancel reason lookup failed; treating cancellation as retryable.",
            extra={"run_id": run_id},
            exc_info=True,
        )
        return ""
    return reason or "canceled"


def _rarespot_worker_identity(settings: RuntimeSettings) -> RuntimeSettings:
    return replace(
        settings,
        worker_id=settings.rarespot_worker_id,
        worker_kind=settings.rarespot_worker_kind,
    )


async def _post_worker_heartbeat(
    heartbeat_func: WorkerHeartbeatFunc,
    settings: RuntimeSettings,
    *,
    status: str,
    current_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        await heartbeat_func(
            settings,
            status=status,
            current_run_id=current_run_id,
            metadata=metadata or {},
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "RareSpot worker heartbeat failed; keeping worker running.",
            extra={"worker_id": settings.worker_id, "status": status},
            exc_info=True,
        )


async def _maybe_post_idle_worker_heartbeat(
    last_posted_at: float,
    settings: RuntimeSettings,
    *,
    active_tasks: dict[str, asyncio.Task],
) -> float:
    interval_seconds = settings.worker_heartbeat_interval_seconds
    if interval_seconds <= 0:
        return last_posted_at
    now = time.monotonic()
    if now - last_posted_at < interval_seconds:
        return last_posted_at
    await _post_worker_heartbeat(
        post_control_plane_worker_heartbeat,
        settings,
        status="idle",
        metadata={"active_tasks": len(active_tasks)},
    )
    return now


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


def _rarespot_redelivery_delay(interval_seconds: float) -> float | None:
    if interval_seconds <= 0:
        return None
    return interval_seconds


async def _nak_message(message: Any | None, *, delay_seconds: float | None = None) -> None:
    if message is None or not hasattr(message, "nak"):
        return
    with contextlib.suppress(Exception):
        if delay_seconds is None:
            await message.nak()
        else:
            try:
                await message.nak(delay=delay_seconds)
            except TypeError:
                await message.nak()


def _start_run_heartbeat_task(
    store: Any,
    run_id: str,
    thread_id: str,
    *,
    interval_seconds: float,
    control_lease: ControlPlaneRunLease | None = None,
    settings: RuntimeSettings | None = None,
    renew_run_lease_func: RenewRunLeaseFunc = renew_control_plane_run_lease,
    worker_task: asyncio.Task | None = None,
    on_lease_lost: Callable[[], None] | None = None,
) -> asyncio.Task | None:
    if interval_seconds <= 0:
        return None
    return asyncio.create_task(
        _run_heartbeat_loop(
            store,
            run_id,
            thread_id,
            interval_seconds=interval_seconds,
            control_lease=control_lease,
            settings=settings,
            renew_run_lease_func=renew_run_lease_func,
            worker_task=worker_task,
            on_lease_lost=on_lease_lost,
        )
    )


async def _run_heartbeat_loop(
    store: Any,
    run_id: str,
    thread_id: str,
    *,
    interval_seconds: float,
    control_lease: ControlPlaneRunLease | None = None,
    settings: RuntimeSettings | None = None,
    renew_run_lease_func: RenewRunLeaseFunc = renew_control_plane_run_lease,
    worker_task: asyncio.Task | None = None,
    on_lease_lost: Callable[[], None] | None = None,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if control_lease is not None:
                if settings is None:
                    raise RunLeaseUnavailable("missing runtime settings for RareSpot lease renewal")
                renewed = await renew_run_lease_func(control_lease, settings)
                if renewed is not None:
                    control_lease = renewed
            await store.update_run_status(run_id, "running")
            await store.append_run_event(
                run_id,
                thread_id,
                "run.heartbeat",
                "RareSpot ecology inference heartbeat.",
                {
                    "workflow_kind": "rarespot_ecology",
                    "status": "alive",
                },
            )
        except asyncio.CancelledError:
            raise
        except (RunLeaseConflict, RunLeaseUnavailable):
            if on_lease_lost is not None:
                on_lease_lost()
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
            logger.warning(
                "RareSpot control-plane run lease renewal failed; stopping active job.",
                extra={"run_id": run_id},
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "RareSpot run heartbeat persist failed; keeping active job running.",
                extra={"run_id": run_id},
                exc_info=True,
            )


async def _stop_run_heartbeat_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def run_worker_loop(settings: RuntimeSettings | None = None) -> None:
    settings = settings or RuntimeSettings.from_env()
    settings = _rarespot_worker_identity(settings)
    import nats
    from nats.errors import TimeoutError as NATSTimeoutError

    nc = await nats.connect(settings.rarespot_nats_url)
    js = nc.jetstream()
    store = build_rarespot_store(js, settings)
    await store.connect()
    active_tasks: dict[str, asyncio.Task] = {}
    cancel_reasons: dict[str, str] = {}
    cancel_subscription = None
    try:
        await js.add_stream(
            name=settings.rarespot_nats_stream,
            subjects=[
                settings.rarespot_nats_jobs_subject,
                settings.rarespot_nats_events_subject,
                settings.nats_cancel_subject,
            ],
        )
    except Exception as exc:
        if "stream name already in use" not in str(exc).lower():
            raise
    cancel_subscription = await _subscribe_rarespot_cancels(
        nc,
        settings,
        active_tasks=active_tasks,
        cancel_reasons=cancel_reasons,
    )
    subscription = await subscribe_rarespot_jobs(js, settings)
    runner = DefaultRareSpotJobRunner(settings)
    last_idle_heartbeat_at = 0.0
    try:
        while True:
            try:
                messages = await subscription.fetch(batch=1, timeout=2)
            except NATSTimeoutError:
                last_idle_heartbeat_at = await _maybe_post_idle_worker_heartbeat(
                    last_idle_heartbeat_at,
                    settings,
                    active_tasks=active_tasks,
                )
                await asyncio.sleep(0.2)
                continue
            for message in messages:
                try:
                    payload = json.loads(message.data.decode("utf-8"))
                except Exception:
                    logger.exception("Discarding malformed RareSpot job payload.")
                    await message.ack()
                    continue
                payload.setdefault("artifact_root", settings.rarespot_artifact_root)
                run_id = str(payload.get("run_id") or "")
                if run_id and run_id in cancel_reasons:
                    await message.ack()
                    continue
                task: asyncio.Task | None = None
                try:
                    task = asyncio.create_task(
                        process_rarespot_job(
                            payload,
                            store=store,
                            runner=runner,
                            message=message,
                            settings=settings,
                            cancel_reason_func=lambda candidate_run_id: cancel_reasons.get(candidate_run_id),
                            ack_progress_interval_seconds=rarespot_ack_extension_interval(settings),
                        )
                    )
                    if run_id:
                        active_tasks[run_id] = task
                    await task
                except EventPublishError:
                    logger.exception("RareSpot event publish failed; job was nacked for redelivery.")
                except Exception:
                    logger.exception("RareSpot job failed after terminal event handling.")
                finally:
                    if run_id and task is not None and active_tasks.get(run_id) is task:
                        active_tasks.pop(run_id, None)
    finally:
        if cancel_subscription is not None and hasattr(cancel_subscription, "unsubscribe"):
            with contextlib.suppress(Exception):
                await cancel_subscription.unsubscribe()
        for task in list(active_tasks.values()):
            task.cancel()
        for task in list(active_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await store.close()
        await nc.drain()


async def _subscribe_rarespot_cancels(
    nc: Any,
    settings: RuntimeSettings,
    *,
    active_tasks: dict[str, asyncio.Task],
    cancel_reasons: dict[str, str],
) -> Any:
    async def handle_cancel_message(message: Any) -> None:
        await _handle_rarespot_cancel_message(
            message,
            active_tasks=active_tasks,
            cancel_reasons=cancel_reasons,
        )

    return await nc.subscribe(settings.nats_cancel_subject, cb=handle_cancel_message)


async def _handle_rarespot_cancel_message(
    message: Any,
    *,
    active_tasks: dict[str, asyncio.Task],
    cancel_reasons: dict[str, str],
) -> None:
    try:
        payload = json.loads(message.data.decode("utf-8"))
    except Exception:
        logger.exception("Discarding malformed RareSpot cancel payload.")
        return
    _handle_rarespot_cancel_payload(
        payload,
        active_tasks=active_tasks,
        cancel_reasons=cancel_reasons,
    )


def _handle_rarespot_cancel_payload(
    payload: dict[str, Any],
    *,
    active_tasks: dict[str, asyncio.Task],
    cancel_reasons: dict[str, str],
) -> None:
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        return
    reason = str(payload.get("reason") or "canceled").strip() or "canceled"
    cancel_reasons[run_id] = reason
    task = active_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
