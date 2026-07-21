"""Parallel atlas orchestration (imaging/atlas.py) — no native lib required."""

from __future__ import annotations

import asyncio
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import ultra_deepagents.imaging.atlas as atlas_mod  # noqa: E402
from PIL import Image  # noqa: E402
from ultra_deepagents.imaging.atlas import (  # noqa: E402
    assemble_atlas,
    assemble_scalar_volume,
    build_scalar_volume_dict,
    compose_atlas_png,
    plan_scalar_preview,
    validate_scalar_plan,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_scalar_preflight_accepts_reference_440_by_440_by_30_volume():
    assert validate_scalar_plan(
        {
            "width": 440,
            "height": 440,
            "depth": 30,
            "dtype": "float32",
            "bytes_per_voxel": 4,
        }
    ) == 440 * 440 * 30 * 4


def test_preview_policy_uses_power_of_two_spacing_aware_factors_without_allocating():
    plan = plan_scalar_preview(924, 624, 80, spacing=(0.1083333333333, 0.1083333333333, 0.29))
    assert plan == {
        "width": 462,
        "height": 312,
        "depth": 80,
        "source_width": 924,
        "source_height": 624,
        "source_depth": 80,
        "downsample_x": 2,
        "downsample_y": 2,
        "downsample_z": 1,
        "preview_policy": "auto-v1",
    }


@pytest.mark.parametrize(
    ("dtype", "bytes_per_voxel"),
    [("uint8", 1), ("uint16", 2), ("int16", 2), ("float32", 4)],
)
def test_scalar_assembly_supports_canonical_native_dtypes(dtype, bytes_per_voxel):
    planes = [np.arange(12, dtype=dtype).reshape(3, 4), np.arange(12, 24, dtype=dtype).reshape(3, 4)]
    plan = {
        "width": 4,
        "height": 3,
        "depth": 2,
        "dtype": dtype,
        "bytes_per_voxel": bytes_per_voxel,
        "channel": 2,
        "t": 3,
        "source_width": 4,
        "source_height": 3,
        "source_depth": 2,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "preview_policy": "auto-v1",
    }

    volume = build_scalar_volume_dict(planes, 2, plan)

    assert volume["dtype"] == dtype
    assert volume["t"] == 3
    assert volume["data"] == np.stack(planes).astype(np.dtype(dtype).newbyteorder("<")).tobytes(order="C")


class FakeParallelRunner:
    """Mimics ImagePool: each call is awaitable; atlas_cell(z) -> a constant-valued cell."""

    workers = 4

    def __init__(self, plan, windows):
        self.plan = plan
        self.windows = windows
        self.cell_calls: list[int] = []

    async def call(self, method, path, **kw):
        await asyncio.sleep(0)  # force interleaving, like real concurrent workers
        if method == "atlas_plan":
            return self.plan
        if method == "atlas_windows":
            return self.windows
        if method == "atlas_cells":
            cells = []
            for z in kw["zs"]:
                self.cell_calls.append(z)
                cells.append(np.full((kw["cell_h"], kw["cell_w"], 3), (z * 17) % 256, dtype="uint8"))
            return cells
        raise AssertionError(f"unexpected method {method}")


def test_parallel_atlas_places_cells_in_grid_order():
    # gather() preserves submission order, so cell z must land at (row=z//cols, col=z%cols)
    # even though the reads complete concurrently/out of order.
    plan = {
        "depth": 5, "columns": 3, "rows": 2, "cell_w": 4, "cell_h": 4,
        "read_level": 0, "read_channels": [1], "cell_colors": [(1.0, 1.0, 1.0)], "paged": False,
    }
    runner = FakeParallelRunner(plan, [(0.0, 255.0)])
    png = asyncio.run(assemble_atlas(runner, "vol.ome.tif", channels=[1], colors=[(1.0, 1.0, 1.0)]))
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    assert img.shape == (4 * 2, 4 * 3, 3)  # rows*cell_h x cols*cell_w
    for z in range(5):
        r, c = divmod(z, 3)
        block = img[r * 4:(r + 1) * 4, c * 4:(c + 1) * 4, 0]
        assert (block == (z * 17) % 256).all(), f"cell {z} misplaced"
    assert (img[4:8, 8:12, 0] == 0).all()  # the 6th (empty) grid slot stays zero
    assert sorted(runner.cell_calls) == [0, 1, 2, 3, 4]  # every plane requested exactly once


def test_bulk_read_reserves_a_worker_for_interactive(monkeypatch):
    """A bulk scalar-volume/atlas read must never occupy EVERY decode worker — at
    least `reserve` worker(s) stay free so a concurrent z-scrub /slice or viewer
    /tile is served immediately (the fix for the 'open fMRI locks up the viewer')."""
    monkeypatch.setenv("ULTRA_IMAGE_INTERACTIVE_RESERVE", "1")
    atlas_mod._bulk_semaphore = None  # force a fresh semaphore for this worker count
    atlas_mod._bulk_semaphore_size = 0

    class ConcurrencyRunner:
        workers = 6

        def __init__(self) -> None:
            self.active = 0
            self.peak = 0

        async def call(self, method, path, **kw):
            if method == "scalar_plan":
                return {
                    "width": 2,
                    "height": 2,
                    "depth": 60,
                    "dtype": "float32",
                    "bytes_per_voxel": 4,
                    "channel": 0,
                    "t": 0,
                    "pages": 0,
                }
            if method == "scalar_planes":
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0.01)  # hold the worker so concurrency is observable
                self.active -= 1
                return [np.zeros((2, 2), dtype="float32") for _ in kw["zs"]]
            raise AssertionError(f"unexpected method {method}")

    runner = ConcurrencyRunner()
    asyncio.run(assemble_scalar_volume(runner, "vol.nii"))
    # 6 workers, reserve 1 → at most 5 concurrent bulk reads; one worker stays free.
    assert runner.peak == 5, f"bulk peak concurrency {runner.peak} should be workers - reserve (5)"
    assert atlas_mod._interactive_reserve() == 1


def test_parallel_scalar_preflight_rejects_oversize_before_plane_reads():
    class OversizeRunner:
        workers = 4

        def __init__(self) -> None:
            self.plane_reads = 0

        async def call(self, method, path, **kw):
            if method == "scalar_plan":
                return {
                    "width": 16_384,
                    "height": 16_384,
                    "depth": 1,
                    "dtype": "float32",
                    "bytes_per_voxel": 4,
                    "channel": 0,
                    "t": 0,
                    "pages": 0,
                }
            if method == "scalar_planes":
                self.plane_reads += 1
                raise AssertionError("oversize plan must not dispatch plane reads")
            raise AssertionError(f"unexpected method {method}")

    runner = OversizeRunner()
    with pytest.raises(ValueError, match="exceeding"):
        asyncio.run(assemble_scalar_volume(runner, "oversize.nii"))
    assert runner.plane_reads == 0


def test_compose_atlas_png_row_major_layout():
    plan = {"columns": 2, "rows": 2, "cell_w": 3, "cell_h": 3, "depth": 4}
    cells = [np.full((3, 3, 3), i + 1, dtype="uint8") for i in range(4)]
    img = np.asarray(Image.open(io.BytesIO(compose_atlas_png(cells, plan))).convert("RGB"))
    assert img.shape == (6, 6, 3)
    assert (img[0:3, 0:3, 0] == 1).all() and (img[0:3, 3:6, 0] == 2).all()
    assert (img[3:6, 0:3, 0] == 3).all() and (img[3:6, 3:6, 0] == 4).all()
