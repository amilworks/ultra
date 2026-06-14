"""Execution runners for the image service.

The ``libbioimage`` binding serializes every call on a global lock, so decode
concurrency must come from *processes*, not threads. :class:`ImagePool` runs a
pool of worker processes, each holding one engine instance (one in-flight decode
per process). :class:`InlineRunner` runs the engine in-process — used by tests
and by deployments that front the service with an external process manager.

Both expose the same minimal async interface the FastAPI app depends on:
``await runner.call(method, *args, **kwargs)``, a ``workers`` count, and
``shutdown()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any

__all__ = ["ImagePool", "InlineRunner", "default_workers"]

logger = logging.getLogger("ultra_deepagents.imaging.pool")

# Module-global engine, initialized once per worker process.
_ENGINE: Any = None


def default_workers() -> int:
    return max(1, os.cpu_count() or 2)


def _init_worker(prefer_real: bool, cache_size: int) -> None:
    global _ENGINE
    from ultra_deepagents.imaging.engine import build_engine

    _ENGINE = build_engine(prefer_real=prefer_real, cache_size=cache_size)


def _invoke(method: str, args: tuple, kwargs: dict) -> Any:
    if _ENGINE is None:  # pragma: no cover - defensive
        raise RuntimeError("image engine not initialized in worker process")
    return getattr(_ENGINE, method)(*args, **kwargs)


class ImagePool:
    """A process pool of engine workers, resilient to worker crashes.

    A malformed file that segfaults / OOM-kills a libbioimage worker would otherwise
    leave the underlying ``ProcessPoolExecutor`` permanently *broken* — every
    subsequent submit raises ``BrokenProcessPool`` and image serving dies for ALL
    users until the service restarts. :meth:`call` detects a broken pool, rebuilds it
    once (replacing the dead worker set), and retries — so one poison file only fails
    its own request, not the service.
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        prefer_real: bool = True,
        cache_size: int = 4,
        call_timeout: float | None = None,
    ) -> None:
        self.workers = workers or default_workers()
        self._prefer_real = prefer_real
        self._cache_size = cache_size
        # A per-call ceiling so a worker hung on a pathological/malformed file (an
        # infinite decode, not a crash) is reclaimed instead of consuming a worker
        # forever — N hangs would otherwise starve the pool. Generous (>> any
        # legitimate tile/slice/atlas/volume op) so it only fires on a true hang.
        if call_timeout is None:
            raw = os.getenv("ULTRA_IMAGE_CALL_TIMEOUT_S", "").strip()
            try:
                call_timeout = float(raw) if raw else 180.0
            except ValueError:
                call_timeout = 180.0
        self.call_timeout = call_timeout if call_timeout and call_timeout > 0 else None
        self._rebuild_lock = asyncio.Lock()
        self._ex = self._new_executor()

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(self._prefer_real, self._cache_size),
        )

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # Two attempts: if a worker died (BrokenExecutor on submit OR while awaiting
        # the result), rebuild the pool once and retry the request on fresh workers.
        for attempt in range(2):
            ex = self._ex
            try:
                fut = asyncio.wrap_future(ex.submit(_invoke, method, args, kwargs))
                if self.call_timeout:
                    return await asyncio.wait_for(fut, self.call_timeout)
                return await fut
            except BrokenExecutor:
                if attempt == 1:
                    raise
                logger.warning("image worker pool broke (likely a worker crash); rebuilding")
                await self._rebuild(ex)
            except (asyncio.TimeoutError, TimeoutError):
                # The worker is hung on this task; reclaim it (rebuild kills the dead
                # worker set) and fail THIS request — do not retry a hang.
                logger.warning("image op %r exceeded %ss; recycling the worker pool", method, self.call_timeout)
                await self._rebuild(ex)
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _rebuild(self, broken: ProcessPoolExecutor) -> None:
        # Serialize so concurrent callers that all hit the same breakage rebuild once.
        async with self._rebuild_lock:
            if self._ex is broken:
                self._ex = self._new_executor()
                broken.shutdown(wait=False, cancel_futures=True)

    def shutdown(self) -> None:
        self._ex.shutdown(wait=True)


class InlineRunner:
    """Runs the engine in the current process (tests / externally-managed workers)."""

    def __init__(self, *, engine: Any = None, prefer_real: bool = True, cache_size: int = 4) -> None:
        if engine is None:
            from ultra_deepagents.imaging.engine import build_engine

            engine = build_engine(prefer_real=prefer_real, cache_size=cache_size)
        self._engine = engine
        self.workers = 1

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._engine, method)(*args, **kwargs)

    def shutdown(self) -> None:  # symmetry with ImagePool
        pass
