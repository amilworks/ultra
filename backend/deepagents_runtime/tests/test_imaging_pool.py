"""ImagePool resilience: a crashed worker must not take down the whole service."""

from __future__ import annotations

import asyncio
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from ultra_deepagents.imaging.pool import ImagePool  # noqa: E402

pytest.importorskip("PIL")  # the StubEngine worker needs Pillow


def test_image_pool_recovers_from_worker_crash():
    # A malformed file that segfaults / OOM-kills a libbioimage worker would otherwise
    # leave the ProcessPoolExecutor permanently broken (every subsequent submit raises
    # BrokenProcessPool) — image serving dead for ALL users. ImagePool must rebuild + recover.
    async def run():
        pool = ImagePool(workers=2, prefer_real=False)  # StubEngine workers
        try:
            assert await pool.call("formats")  # warm + spin up workers
            pids = list(getattr(pool._ex, "_processes", {}).keys())
            assert pids, "expected at least one worker process"
            os.kill(pids[0], signal.SIGKILL)  # crash a worker
            await asyncio.sleep(0.6)  # let the executor notice the dead worker
            # Must NOT raise BrokenProcessPool — the pool rebuilds and the request retries.
            assert await pool.call("formats")
            for _ in range(5):  # and stays healthy afterward
                assert await pool.call("formats")
        finally:
            pool.shutdown()

    asyncio.run(run())


def test_image_pool_reclaims_hung_worker_on_timeout():
    # A worker hung on a pathological op (here: the StubEngine's pure-Python big-tile
    # render) must NOT block forever — call_timeout reclaims it and the request fails
    # cleanly, and the pool keeps serving afterward.
    async def run():
        pool = ImagePool(workers=2, prefer_real=False, call_timeout=0.2)
        try:
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await pool.call("tile", "x.tif", level=0, col=0, row=0, tile_size=6000)
            # The 0.2s timeout above is intentionally tiny to force the hung op
            # path. Do not require a freshly spawned worker process to cold-start
            # and serve the health probe inside that same artificial budget.
            pool.call_timeout = 5.0
            assert await pool.call("formats")  # pool recovered, still serving
        finally:
            pool.shutdown()

    asyncio.run(run())
