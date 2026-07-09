"""The executor seam — the ONE thing you write to add a model to the compute service.

An executor is a self-contained unit of GPU work for a single
``model_key.capability`` (e.g. ``rarespot.train``, ``rarespot.evaluate``,
``megaseg.infer``). The service framework owns everything else — auth, the job
queue, the per-GPU semaphore, progress storage, SSE, cancellation, health — so a
new model plugs in as one small class plus a registry line, never a change to the
service internals. That is the whole modularity contract.

Executors run OFF the event loop (the job manager calls ``run`` via
``asyncio.to_thread``) and are handed a :class:`JobContext` to report progress and
observe cancellation. They must be synchronous, self-contained, and leave nothing
behind between jobs (fresh workdir per job, subprocesses that exit and free the
GPU) so launches are reliable back to back.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


class JobCancelled(RuntimeError):
    """Raised by an executor when it observes ``ctx.cancelled`` and stops."""


@dataclass
class JobContext:
    """The handle an executor uses to talk back to the framework while it runs.

    ``progress`` is throttled/stored by the job manager and surfaced on
    ``GET /v1/jobs/{id}`` and the SSE stream. ``cancelled`` flips when a client
    calls ``POST /v1/jobs/{id}/cancel`` — long executors must poll it (and, for a
    subprocess-based executor, register the child via ``register_process`` so the
    framework can kill it on cancel).
    """

    job_id: str
    workdir: Path
    _emit: Callable[[dict[str, Any]], None]
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _processes: list[Any] = field(default_factory=list)
    _proc_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise JobCancelled(f"job {self.job_id} cancelled")

    def progress(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        message: str = "",
        metrics: dict[str, Any] | None = None,
        phase: str = "",
    ) -> None:
        """Report progress. All fields optional; the manager merges + stores it."""
        update: dict[str, Any] = {}
        if completed is not None:
            update["completed"] = int(completed)
        if total is not None:
            update["total"] = int(total)
        if message:
            update["message"] = message
        if phase:
            update["phase"] = phase
        if metrics:
            update["metrics"] = dict(metrics)
        if update:
            self._emit(update)

    def register_process(self, proc: Any) -> None:
        """Register a live subprocess.Popen so cancel can terminate it."""
        with self._proc_lock:
            self._processes.append(proc)

    def _kill_processes(self, grace_seconds: float = 10.0) -> None:
        """Terminate every registered child by process GROUP, escalating
        SIGTERM -> (grace) -> SIGKILL.

        A GPU training run forks DataLoader workers; killing only the direct child
        leaves them (and the CUDA context) orphaned, which is exactly what makes the
        next launch fail. And a child parked in an uninterruptible CUDA/driver call
        ignores SIGTERM entirely — without the SIGKILL escalation the reaping
        ``process.wait()`` would block forever and wedge the job slot. Executors that
        register a Popen MUST start it with ``start_new_session=True`` so the group
        id is the child pid. Idempotent; safe to call from cancel, a watchdog, and
        the executor's finally.
        """
        with self._proc_lock:
            procs = list(self._processes)
        for proc in procs:
            self._signal_group(proc, signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, grace_seconds)
        for proc in procs:
            while self._still_running(proc) and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._still_running(proc):
                self._signal_group(proc, signal.SIGKILL)

    @staticmethod
    def _still_running(proc: Any) -> bool:
        try:
            return proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _signal_group(proc: Any, sig: int) -> None:
        pid = getattr(proc, "pid", None)
        if pid is None or not JobContext._still_running(proc):
            return
        try:
            os.killpg(os.getpgid(pid), sig)
        except Exception:  # noqa: BLE001 - not a group leader / already gone
            try:
                proc.send_signal(sig)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


@dataclass(frozen=True)
class ExecutorManifest:
    """Declarative description of an executor — introspectable via GET /v1/executors."""

    model_key: str
    capability: str  # "train" | "evaluate" | "infer" | ...
    description: str = ""
    gpu: bool = True
    # Free-form resource hint mirroring the training ModelManifest.executor block
    # (min_vram_gb, wall_clock_budget_s, ...). The framework does not interpret it;
    # it is surfaced for operators + future schedulers.
    resource: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model_key}.{self.capability}"


@runtime_checkable
class Executor(Protocol):
    """Implement this + register it. That is the entire extension surface."""

    manifest: ExecutorManifest

    def run(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        """Do the GPU work described by ``spec``; return a JSON-able result dict.

        Raise on failure (the framework records it as a failed job); raise
        :class:`JobCancelled` when ``ctx.cancelled`` is observed. Report progress
        through ``ctx.progress(...)``. Must be synchronous and self-contained.
        """
        ...
