"""Job lifecycle for the compute service — model-agnostic.

Owns everything that is NOT model-specific: submit → queue → run off-loop under a
per-GPU semaphore → progress → cancel → terminal state → restart recovery, plus
bounded history so the service runs indefinitely. It dispatches to an
:class:`Executor` by key and knows nothing about what the executor does; that is
what lets a new model plug in without touching this file.

Reliability contract (the user's "launch back to back to back with no errors"):
- every job gets a FRESH job-scoped workdir, wiped before use;
- the per-GPU semaphore is released in ``finally`` no matter how a job ends;
- a job failure is recorded on the job, never propagated to crash the worker loop;
- a cancel event exists from submit (no race window) and cancel kills the whole
  process group so a half-run training leaves no orphaned GPU process;
- records + workdirs are evicted oldest-first past a bound.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shutil
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .executors import JobCancelled, JobContext, get_executor

logger = logging.getLogger("ultra.compute.jobs")

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED})
IDEMPOTENCY_CONFLICT_DETAIL = "Idempotency key is already bound to a different request."
INVALID_JOB_SPEC_DETAIL = "Job spec must contain only finite JSON values."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IdempotencyConflictError(RuntimeError):
    """Raised when a retained idempotency key is reused for another request."""

    def __init__(self) -> None:
        super().__init__(IDEMPOTENCY_CONFLICT_DETAIL)


class InvalidJobSpecError(ValueError):
    """Raised when a job spec cannot be represented faithfully as JSON."""

    def __init__(self) -> None:
        super().__init__(INVALID_JOB_SPEC_DETAIL)


def _typed_json_value(value: Any) -> list[Any]:
    """Return a deterministic JSON value that preserves each source JSON type."""
    value_type = type(value)
    if value is None:
        return ["null"]
    if value_type is bool:
        return ["bool", value]
    if value_type is int:
        return ["int", str(value)]
    if value_type is float:
        if not math.isfinite(value):
            raise InvalidJobSpecError
        return ["float", value.hex()]
    if value_type is str:
        return ["string", value]
    if value_type is list:
        return ["array", [_typed_json_value(item) for item in value]]
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise InvalidJobSpecError
        return [
            "object",
            [[key, _typed_json_value(value[key])] for key in sorted(value)],
        ]
    raise InvalidJobSpecError


def validate_job_spec(spec: dict[str, Any]) -> None:
    """Reject values that cannot be represented faithfully in persisted JSON."""
    _typed_json_value(spec)


def _request_identity(executor_key: str, spec: dict[str, Any]) -> bytes:
    canonical = _typed_json_value({"executor": executor_key, "spec": spec})
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


class JobProgress(BaseModel):
    completed: int = 0
    total: int = 0
    message: str = ""
    phase: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    updated_at: str = ""


class JobRecord(BaseModel):
    job_id: str
    executor: str
    status: str
    created_at: str
    updated_at: str
    spec: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    progress: JobProgress = Field(default_factory=JobProgress)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobManager:
    """Async-facing manager; the blocking executor call runs on a worker thread.

    Thread-safety: ``_records`` / ``_contexts`` / ``_cancel_events`` are touched
    from BOTH the event loop (submit, cancel, get) and the executor worker thread
    (progress, finish), so every access is under ``_lock`` (a plain threading lock,
    since the executor side is not on the loop). Disk persistence is best-effort
    durability for restart recovery; the in-memory record is authoritative while a
    job is live so progress reads are always fresh.
    """

    def __init__(
        self,
        *,
        job_root: Path,
        workspace_root: Path,
        max_concurrent: int,
        history_limit: int = 200,
        progress_persist_interval: float = 5.0,
    ) -> None:
        self.job_root = Path(job_root)
        self.workspace_root = Path(workspace_root)
        self.max_concurrent = max(1, int(max_concurrent))
        self.history_limit = max(16, int(history_limit))
        self.progress_persist_interval = float(progress_persist_interval)

        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._contexts: dict[str, JobContext] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._idempotency: dict[str, str] = {}
        self._last_persist: dict[str, float] = {}

        # Created on the running loop in :meth:`start`.
        self.queue: asyncio.Queue[str] | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self.worker_tasks: list[asyncio.Task[Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ startup
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        requeued = self._recover_on_startup()
        # One worker task per concurrency slot; the semaphore is the real bound but
        # multiple pumps let independent GPUs (max_concurrent>1) run in parallel.
        self.worker_tasks = [
            loop.create_task(self._worker_loop(), name=f"compute-worker-{i}")
            for i in range(self.max_concurrent)
        ]
        for job_id in requeued:
            assert self.queue is not None
            self.queue.put_nowait(job_id)

    async def shutdown(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        for task in self.worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.worker_tasks = []

    def _recover_on_startup(self) -> list[str]:
        """Load persisted records and choose authoritative queued attempts."""
        for path in sorted(self.job_root.glob("*.json")):
            try:
                record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("dropping unreadable job record %s: %s", path.name, exc)
                continue
            if record.status == STATUS_RUNNING:
                record.status = STATUS_FAILED
                record.error = "compute service restarted while the job was running"
                record.finished_at = _utc_now()
                record.updated_at = record.finished_at
                self._write_disk(record)
            self._records[record.job_id] = record
            # A requeued job MUST get a cancel event or a later cancel() would find
            # none and silently no-op (and _run_job_sync would synthesize a private
            # one that cancel() can't see).
            self._cancel_events.setdefault(record.job_id, threading.Event())
        self._evict_locked()
        self._rebuild_idempotency_index_locked()
        return self._recovery_queue_locked()

    @staticmethod
    def _attempt_order(record: JobRecord) -> tuple[str, str]:
        """Order server-created attempts deterministically, newest last."""
        return (record.created_at, record.job_id)

    def _idempotency_attempts_locked(self, idempotency_key: str) -> list[JobRecord]:
        return [
            record
            for record in self._records.values()
            if record.idempotency_key == idempotency_key
        ]

    @staticmethod
    def _consistent_request_identity(attempts: list[JobRecord]) -> bytes | None:
        identities: set[bytes] = set()
        try:
            for attempt in attempts:
                identities.add(_request_identity(attempt.executor, attempt.spec))
        except InvalidJobSpecError:
            return None
        if len(identities) != 1:
            return None
        return next(iter(identities))

    def _recovery_queue_locked(self) -> list[str]:
        histories: dict[str, list[JobRecord]] = {}
        for record in self._records.values():
            if record.idempotency_key is not None:
                histories.setdefault(record.idempotency_key, []).append(record)

        authoritative_queued: set[str] = set()
        for attempts in histories.values():
            if self._consistent_request_identity(attempts) is None:
                continue
            newest = max(attempts, key=self._attempt_order)
            if newest.status == STATUS_QUEUED:
                authoritative_queued.add(newest.job_id)

        return [
            record.job_id
            for record in self._records.values()
            if record.status == STATUS_QUEUED
            and (
                record.idempotency_key is None
                or record.job_id in authoritative_queued
            )
        ]

    def _rebuild_idempotency_index_locked(self) -> None:
        self._idempotency.clear()
        for record in self._records.values():
            key = record.idempotency_key
            if key is None:
                continue
            current_id = self._idempotency.get(key)
            current = self._records.get(current_id) if current_id is not None else None
            if current is None or self._attempt_order(record) > self._attempt_order(current):
                self._idempotency[key] = record.job_id

    # -------------------------------------------------------------- persistence
    def _record_path(self, job_id: str) -> Path:
        return self.job_root / f"{job_id}.json"

    def _write_disk(self, record: JobRecord) -> None:
        try:
            tmp = self._record_path(record.job_id).with_suffix(".json.tmp")
            tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self._record_path(record.job_id))
        except Exception:  # noqa: BLE001 - durability is best-effort; never crash a job on it
            logger.exception("failed to persist job record %s", record.job_id)

    # -------------------------------------------------------------------- reads
    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            return record.model_copy(deep=True) if record is not None else None

    def list_jobs(self, limit: int = 100) -> list[JobRecord]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
            return [r.model_copy(deep=True) for r in records[: max(1, int(limit))]]

    # ------------------------------------------------------------------- submit
    def submit(
        self,
        *,
        executor_key: str,
        spec: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> JobRecord:
        admitted_spec = deepcopy(spec or {})
        validate_job_spec(admitted_spec)
        request_identity = (
            _request_identity(executor_key, admitted_spec)
            if idempotency_key is not None
            else None
        )
        with self._lock:
            if idempotency_key is not None:
                attempts = self._idempotency_attempts_locked(idempotency_key)
                if attempts:
                    # Inspect every retained attempt before liveness/terminal state
                    # so legacy mixed histories fail closed regardless of pointer.
                    retained_identity = self._consistent_request_identity(attempts)
                    if retained_identity is None or retained_identity != request_identity:
                        raise IdempotencyConflictError
                    existing = max(attempts, key=self._attempt_order)
                    self._idempotency[idempotency_key] = existing.job_id
                    # A same-request redelivery resumes queued/running work and
                    # reuses success, while failure/cancellation may retry under
                    # the existing policy by minting a fresh job below.
                    if existing.status not in (STATUS_FAILED, STATUS_CANCELLED):
                        return existing.model_copy(deep=True)
            job_id = uuid4().hex
            now = _utc_now()
            record = JobRecord(
                job_id=job_id,
                executor=executor_key,
                status=STATUS_QUEUED,
                created_at=now,
                updated_at=now,
                spec=admitted_spec,
                idempotency_key=idempotency_key,
            )
            self._records[job_id] = record
            self._cancel_events[job_id] = threading.Event()
            if idempotency_key is not None:
                self._idempotency[idempotency_key] = job_id
            self._write_disk(record)
            self._evict_locked()
            snapshot = record.model_copy(deep=True)
        # submit() runs off the event loop (app.py calls it via to_thread), and
        # asyncio.Queue is NOT thread-safe: put_nowait mutates loop-owned state and
        # its getter-wakeup must run on the loop. Schedule it thread-safely.
        assert self.queue is not None and self._loop is not None, "JobManager.start() not called"
        self._loop.call_soon_threadsafe(self.queue.put_nowait, job_id)
        logger.info("queued job %s executor=%s", job_id, executor_key)
        return snapshot

    # -------------------------------------------------------------------- cancel
    def cancel(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            if record.status in TERMINAL_STATUSES:
                return record.model_copy(deep=True)
            event = self._cancel_events.get(job_id)
            context = self._contexts.get(job_id)
            if event is not None:
                event.set()
            if context is None:
                # Still queued (not yet running): mark cancelled now; the worker
                # loop will skip it because status is no longer queued.
                record.status = STATUS_CANCELLED
                record.error = "cancelled before start"
                record.finished_at = _utc_now()
                record.updated_at = record.finished_at
                self._write_disk(record)
                snapshot = record.model_copy(deep=True)
                context = None
            else:
                snapshot = record.model_copy(deep=True)
        if context is not None:
            context._kill_processes()  # noqa: SLF001 - manager owns the context internals
        return snapshot

    # ---------------------------------------------------------------- execution
    async def _worker_loop(self) -> None:
        assert self.queue is not None and self.semaphore is not None
        while True:
            job_id = await self.queue.get()
            try:
                current = self.get(job_id)
                if current is None or current.status != STATUS_QUEUED:
                    continue  # cancelled/removed while queued
                async with self.semaphore:
                    await asyncio.to_thread(self._run_job_sync, job_id)
            except Exception:  # noqa: BLE001 - a job must never break the pump
                logger.exception("worker loop error on job %s", job_id)
            finally:
                self.queue.task_done()

    def _run_job_sync(self, job_id: str) -> None:
        """Runs on a worker thread. Owns the transition queued→running→terminal."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status != STATUS_QUEUED:
                return
            # Use the REGISTERED cancel event (the one cancel() sets); register a
            # fresh one only if somehow missing, so cancel() and this run share it.
            event = self._cancel_events.get(job_id)
            if event is None:
                event = self._cancel_events[job_id] = threading.Event()
            if event.is_set():
                self._finish_locked(job_id, STATUS_CANCELLED, error="cancelled before start")
                return
            record.status = STATUS_RUNNING
            record.started_at = _utc_now()
            record.updated_at = record.started_at
            self._write_disk(record)
            executor_key = record.executor
            spec = deepcopy(record.spec)

        executor = get_executor(executor_key)
        if executor is None:
            self._finish(job_id, STATUS_FAILED, error=f"unknown executor: {executor_key}")
            return

        # Everything from workdir setup onward is inside the try so that ANY failure
        # (e.g. a read-only/full workspace on mkdir) transitions the job to a
        # terminal FAILED instead of leaving it stuck RUNNING forever.
        try:
            workdir = self.workspace_root / job_id
            shutil.rmtree(workdir, ignore_errors=True)
            workdir.mkdir(parents=True, exist_ok=True)
            context = JobContext(
                job_id=job_id,
                workdir=workdir,
                _emit=lambda update, _jid=job_id: self._on_progress(_jid, update),
                _cancel_event=event,
            )
            with self._lock:
                self._contexts[job_id] = context
            context.check_cancelled()
            result = executor.run(spec, context)
            self._finish(job_id, STATUS_SUCCEEDED, result=_jsonable(result))
            logger.info("job %s succeeded", job_id)
        except JobCancelled as exc:
            self._finish(job_id, STATUS_CANCELLED, error=str(exc))
            logger.info("job %s cancelled", job_id)
        except Exception as exc:  # noqa: BLE001
            self._finish(job_id, STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
            logger.exception("job %s failed", job_id)
        finally:
            with self._lock:
                self._contexts.pop(job_id, None)

    def _on_progress(self, job_id: str, update: dict[str, Any]) -> None:
        """Called from the executor thread; throttled disk persistence."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            progress = record.progress
            if "completed" in update:
                progress.completed = int(update["completed"])
            if "total" in update:
                progress.total = int(update["total"])
            if "message" in update:
                progress.message = str(update["message"])
            if "phase" in update:
                progress.phase = str(update["phase"])
            if "metrics" in update and isinstance(update["metrics"], dict):
                progress.metrics.update(update["metrics"])
            progress.updated_at = _utc_now()
            record.updated_at = progress.updated_at
            now = time.monotonic()
            if now - self._last_persist.get(job_id, 0.0) >= self.progress_persist_interval:
                self._write_disk(record)
                self._last_persist[job_id] = now

    def _finish(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            self._finish_locked(job_id, status, result=result, error=error)

    def _finish_locked(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        record = self._records.get(job_id)
        if record is None:
            return
        record.status = status
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        record.finished_at = _utc_now()
        record.updated_at = record.finished_at
        self._write_disk(record)
        self._last_persist.pop(job_id, None)
        self._evict_locked()

    # --------------------------------------------------------------- eviction
    def _evict_locked(self) -> None:
        """Bound in-memory records + on-disk workdirs to history_limit (oldest first)."""
        if len(self._records) <= self.history_limit:
            return
        terminal = sorted(
            (r for r in self._records.values() if r.status in TERMINAL_STATUSES),
            key=lambda r: r.finished_at or r.created_at,
        )
        overflow = len(self._records) - self.history_limit
        evicted = False
        for record in terminal[:overflow]:
            self._records.pop(record.job_id, None)
            self._contexts.pop(record.job_id, None)
            self._cancel_events.pop(record.job_id, None)
            self._last_persist.pop(record.job_id, None)
            self._record_path(record.job_id).unlink(missing_ok=True)
            shutil.rmtree(self.workspace_root / record.job_id, ignore_errors=True)
            evicted = True
        if evicted:
            self._rebuild_idempotency_index_locked()


def _jsonable(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"result": value}
