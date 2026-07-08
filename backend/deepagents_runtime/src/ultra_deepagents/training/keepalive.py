"""Loop-independent lease + ack keepalive for training jobs (plan section 9.2).

Ported from the run worker's ``_LeaseKeepalive`` OS-thread design (nats_worker,
commit 86e47b6b): a dedicated ``threading.Thread`` on a wall-clock timer renews
the control-plane job lease with SYNCHRONOUS HTTP and schedules the NATS
``message.in_progress()`` ack extension onto the event loop via
``asyncio.run_coroutine_threadsafe``. A synchronous stall of the event loop (a
large barrel copy/hash done outside ``to_thread``, a long GC pause, an epoch's
blocking step) therefore cannot lapse either liveness signal — as long as the
worker PROCESS is alive it holds the lease and keeps the delivery in flight,
so multi-hour finetunes are never reaped and restarted from scratch. If the
process truly dies, the thread dies with it, the lease expires, and recovery
correctly takes over.

An authoritative 409 on renewal means the control plane handed the job to
another worker: the keepalive sets its ``lost`` kill flag (which the job loop
checks / gets cancelled on) and stops — never fight a hand-off.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from ultra_deepagents.training.control_client import (
    TrainingControlClient,
    TrainingJobLease,
    TrainingLeaseConflict,
)

logger = logging.getLogger(__name__)


def training_keepalive_interval(ttl_seconds: float) -> float:
    """Renew at TTL/3 so two consecutive missed HTTP calls (a control-plane
    hiccup) still leave the lease valid. The 0.1s floor only guards a
    misconfigured zero and lets tests drive sub-second intervals."""
    return max(0.1, float(ttl_seconds) / 3.0)


class TrainingLeaseKeepalive:
    """Wall-clock OS-thread keepalive: lease renewal + ``in_progress()``.

    The thread is the SOLE lease renewer while active, so a rotating lease
    token never races between two renewers. ``stop()`` signals and joins.
    """

    def __init__(
        self,
        lease: TrainingJobLease,
        client: TrainingControlClient | Any,
        *,
        loop: asyncio.AbstractEventLoop,
        ttl_seconds: float,
        message: Any | None = None,
        interval_seconds: float | None = None,
        on_lost: Callable[[], None] | None = None,
    ) -> None:
        self._client = client
        self._loop = loop
        self._ttl = float(ttl_seconds)
        self._message = message
        self._interval = (
            max(0.1, float(interval_seconds))
            if interval_seconds is not None
            else training_keepalive_interval(ttl_seconds)
        )
        self._on_lost = on_lost
        self._lock = threading.Lock()
        self._lease = lease
        self._stop = threading.Event()
        self._lost = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run,
            name=f"training-lease-keepalive-{self._lease.job_id[:16]}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def current_lease(self) -> TrainingJobLease:
        with self._lock:
            return self._lease

    @property
    def lost(self) -> bool:
        """The kill flag: True once a renewal came back 409 (lease handed
        elsewhere). The job loop must abort processing when it is set."""
        return self._lost

    def _run(self) -> None:
        # stop.wait returns True the instant stop() is called (prompt teardown);
        # False on timeout means a tick is due.
        while not self._stop.wait(self._interval):
            with self._lock:
                lease = self._lease
            # (1) Lease renewal — synchronous HTTP, safe in this thread; keeps
            # the job out of the control plane's expired-lease recovery path.
            try:
                renewed = self._client.renew_lease(lease, ttl_seconds=self._ttl)
            except TrainingLeaseConflict:
                self._lost = True
                if self._on_lost is not None:
                    with contextlib.suppress(RuntimeError):
                        self._loop.call_soon_threadsafe(self._on_lost)
                return
            except Exception:
                # Transient (control plane unreachable / timeout): the TTL still
                # has headroom, retry next tick. A real hand-off surfaces as the
                # 409 above.
                logger.warning(
                    "Training lease keepalive renewal failed; retrying next tick.",
                    extra={"job_id": lease.job_id},
                    exc_info=True,
                )
            else:
                if renewed is not None:
                    with self._lock:
                        self._lease = renewed
            # (2) Ack extension — schedule message.in_progress() onto the event
            # loop from this thread so the JetStream redelivery clock resets even
            # while the loop itself is busy elsewhere. Fire-and-forget.
            message = self._message
            if message is not None and hasattr(message, "in_progress"):
                try:
                    future = asyncio.run_coroutine_threadsafe(message.in_progress(), self._loop)
                    future.add_done_callback(_swallow_result)
                except RuntimeError:
                    logger.debug(
                        "Training keepalive could not schedule in_progress; loop closed.",
                        extra={"job_id": lease.job_id},
                    )


def _swallow_result(future: Any) -> None:
    with contextlib.suppress(Exception):
        future.result()
