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
    SCALAR_MASK_MAX_GRID_CROSSINGS,
    SCALAR_MASK_NATIVE_POLICY,
    assemble_atlas,
    assemble_scalar_volume,
    build_scalar_volume_dict,
    compose_atlas_png,
    plan_scalar_mask_native,
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


@pytest.mark.parametrize(("depth", "dtype", "bytes_per_voxel"), [(65, "uint8", 1), (70, "uint16", 2)])
def test_exact_integer_mask_policy_keeps_reference_microscopy_grid_native(
    depth, dtype, bytes_per_voxel
):
    plan = plan_scalar_mask_native(
        924,
        624,
        depth,
        dtype=dtype,
        bytes_per_voxel=bytes_per_voxel,
    )

    assert (plan["width"], plan["height"], plan["depth"]) == (924, 624, depth)
    assert (plan["downsample_x"], plan["downsample_y"], plan["downsample_z"]) == (1, 1, 1)
    assert plan["preview_policy"] == SCALAR_MASK_NATIVE_POLICY
    assert validate_scalar_plan(plan) == 924 * 624 * depth * bytes_per_voxel
    assert plan["width"] + plan["height"] + plan["depth"] <= SCALAR_MASK_MAX_GRID_CROSSINGS


def test_exact_integer_mask_policy_fails_closed_before_materialization_when_traversal_is_too_large():
    with pytest.raises(ValueError, match="DDA crossing"):
        plan_scalar_mask_native(
            1_000,
            1_000,
            100,
            dtype="uint8",
            bytes_per_voxel=1,
        )


def test_exact_integer_mask_policy_uses_the_prepared_gpu_byte_limit():
    # 900 + 900 + 100 stays within the dimension and DDA envelopes, while the
    # native uint16 texture is ~154 MiB and must not be advertised/uploaded.
    with pytest.raises(ValueError, match="134217728|128 MiB|Mask"):
        plan_scalar_mask_native(
            900,
            900,
            100,
            dtype="uint16",
            bytes_per_voxel=2,
        )


@pytest.mark.parametrize(
    ("dtype", "bytes_per_voxel"),
    [("float32", 4), ("uint16", 1), ("uint32", 4)],
)
def test_exact_integer_mask_policy_rejects_unsupported_dtype_or_byte_width(
    dtype, bytes_per_voxel
):
    with pytest.raises(ValueError, match="dtype|byte width"):
        plan_scalar_mask_native(
            16,
            16,
            16,
            dtype=dtype,
            bytes_per_voxel=bytes_per_voxel,
        )


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


def test_parallel_nonexact_nearest_rejects_before_fanout():
    class FloatNearestRunner:
        workers = 4
        plane_reads = 0

        async def call(self, method, path, **kw):
            if method == "scalar_plan":
                return {
                    "width": 2,
                    "height": 2,
                    "depth": 2,
                    "dtype": "float32",
                    "bytes_per_voxel": 4,
                    "channel": 0,
                    "t": 0,
                    "pages": 0,
                    "sampling": "nearest",
                    "preview_policy": "nearest-source-grid-v1",
                }
            if method == "scalar_planes":
                self.plane_reads += 1
            raise AssertionError(method)

    runner = FloatNearestRunner()
    with pytest.raises(ValueError, match="nearest.*exact Mask"):
        asyncio.run(assemble_scalar_volume(runner, "float.tif", sampling="nearest"))
    assert runner.plane_reads == 0


def test_parallel_exact_mask_requires_and_forwards_the_complete_decoder_admission():
    plan = {
        "width": 2,
        "height": 2,
        "depth": 4,
        "source_width": 2,
        "source_height": 2,
        "source_depth": 4,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "dtype": "uint8",
        "bytes_per_voxel": 1,
        "channel": 1,
        "t": 2,
        "pages": 0,
        "sampling": "nearest",
        "preview_policy": SCALAR_MASK_NATIVE_POLICY,
        "decode_admission": "complete-selected-scalar-v1",
        "admitted_decode_work_bytes": 16,
        "admitted_decode_read_count": 4,
        "admitted_source_dtype": "uint8",
        "admitted_source_bytes_per_voxel": 1,
        "source_generation": (1, 1, 2, 16, 3, 4),
    }

    class ExactMaskRunner:
        workers = 2

        def __init__(self, admitted_plan):
            self.plan = admitted_plan
            self.plane_reads = 0

        async def call(self, method, path, **kw):
            if method == "scalar_planes":
                self.plane_reads += 1
                assert kw["plan"] is self.plan
                return [
                    np.full((2, 2), output_z, dtype="uint8")
                    for output_z in kw["zs"]
                ]
            raise AssertionError(f"unexpected method {method}")

    runner = ExactMaskRunner(plan)
    volume = asyncio.run(
        assemble_scalar_volume(
            runner,
            "mask.tif",
            channel=1,
            t=2,
            sampling="nearest",
            plan=plan,
        )
    )
    assert runner.plane_reads == 2
    assert volume["data"] == np.stack(
        [
            np.full((2, 2), output_z, dtype="uint8")
            for output_z in range(4)
        ]
    ).tobytes()

    missing_admission = dict(plan)
    missing_admission.pop("decode_admission")
    rejected = ExactMaskRunner(missing_admission)
    with pytest.raises(ValueError, match="decoder admission"):
        asyncio.run(
            assemble_scalar_volume(
                rejected,
                "mask.tif",
                channel=1,
                t=2,
                sampling="nearest",
                plan=missing_admission,
            )
        )
    assert rejected.plane_reads == 0


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    ("plan_field", "plan_value", "requested_channel", "requested_t", "requested_sampling"),
    [
        ("channel", 1, 0, 0, "nearest"),
        ("t", 1, 0, 0, "nearest"),
        ("sampling", "nearest", 0, 0, "box"),
    ],
)
def test_scalar_assembly_rejects_a_supplied_plan_for_another_selection_before_fanout(
    workers,
    plan_field,
    plan_value,
    requested_channel,
    requested_t,
    requested_sampling,
):
    plan = {
        "width": 1,
        "height": 1,
        "depth": 1,
        "source_width": 1,
        "source_height": 1,
        "source_depth": 1,
        "downsample_x": 1,
        "downsample_y": 1,
        "downsample_z": 1,
        "dtype": "uint8",
        "bytes_per_voxel": 1,
        "channel": 0,
        "t": 0,
        "pages": 0,
        "sampling": "nearest",
        "preview_policy": SCALAR_MASK_NATIVE_POLICY,
        "decode_admission": "complete-selected-scalar-v1",
        "admitted_decode_work_bytes": 1,
        "admitted_decode_read_count": 1,
        "admitted_source_dtype": "uint8",
        "admitted_source_bytes_per_voxel": 1,
        "source_generation": (1, 1, 2, 1, 3, 4),
    }
    plan[plan_field] = plan_value

    class SelectionRunner:
        def __init__(self):
            self.workers = workers
            self.plane_reads = 0

        async def call(self, method, _path, **_kwargs):
            if method == "scalar_planes":
                self.plane_reads += 1
            raise AssertionError(method)

    runner = SelectionRunner()
    with pytest.raises(ValueError, match="plan.*request|selection"):
        asyncio.run(
            assemble_scalar_volume(
                runner,
                "mask.tif",
                channel=requested_channel,
                t=requested_t,
                sampling=requested_sampling,
                plan=plan,
            )
        )
    assert runner.plane_reads == 0


def test_compose_atlas_png_row_major_layout():
    plan = {"columns": 2, "rows": 2, "cell_w": 3, "cell_h": 3, "depth": 4}
    cells = [np.full((3, 3, 3), i + 1, dtype="uint8") for i in range(4)]
    img = np.asarray(Image.open(io.BytesIO(compose_atlas_png(cells, plan))).convert("RGB"))
    assert img.shape == (6, 6, 3)
    assert (img[0:3, 0:3, 0] == 1).all() and (img[0:3, 3:6, 0] == 2).all()
    assert (img[3:6, 0:3, 0] == 3).all() and (img[3:6, 3:6, 0] == 4).all()
