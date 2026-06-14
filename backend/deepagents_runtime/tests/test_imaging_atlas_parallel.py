"""Parallel atlas orchestration (imaging/atlas.py) — no native lib required."""

from __future__ import annotations

import asyncio
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from ultra_deepagents.imaging.atlas import assemble_atlas, compose_atlas_png  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


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


def test_compose_atlas_png_row_major_layout():
    plan = {"columns": 2, "rows": 2, "cell_w": 3, "cell_h": 3, "depth": 4}
    cells = [np.full((3, 3, 3), i + 1, dtype="uint8") for i in range(4)]
    img = np.asarray(Image.open(io.BytesIO(compose_atlas_png(cells, plan))).convert("RGB"))
    assert img.shape == (6, 6, 3)
    assert (img[0:3, 0:3, 0] == 1).all() and (img[0:3, 3:6, 0] == 2).all()
    assert (img[3:6, 0:3, 0] == 3).all() and (img[3:6, 3:6, 0] == 4).all()
