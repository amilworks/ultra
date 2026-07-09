"""``rarespot.smoke`` executor — structural sanity check on candidate weights.

A thin wrapper over ``RareSpotAdapter.smoke_test_local`` (plan 6.2: assert names/nc
on the checkpoint + one real detect pass). It needs torch + the yolov5 fork, which
the CPU training worker doesn't have, so the worker offloads it here. It runs
BEFORE model-version registration; a failure raises, the job fails, and the training
phase fails cleanly with the candidate never registered.

spec:
    weights_uri  (required) candidate weights, path readable in-container
    gold_sample  (optional) the plan-6.2 sample (tile dir / manifest) to detect on
returns:
    {ok: true}
"""

from __future__ import annotations

from typing import Any

from ultra_deepagents.training.rarespot_adapter import RareSpotAdapter

from .base import ExecutorManifest, JobContext


class RareSpotSmokeExecutor:
    manifest = ExecutorManifest(
        model_key="rarespot",
        capability="smoke",
        description="structural smoke test of candidate weights (names/nc + one detect pass)",
        resource={"min_vram_gb": 4, "wall_clock_budget_s": 600},
    )

    def run(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        weights_uri = str(spec.get("weights_uri") or "").strip()
        if not weights_uri:
            raise ValueError("rarespot.smoke: weights_uri is required")
        gold_sample = spec.get("gold_sample")
        ctx.progress(total=1, message="running smoke test", phase="smoke")
        ctx.check_cancelled()
        # Raises if the candidate weights are structurally unusable.
        RareSpotAdapter().smoke_test_local(weights_uri, gold_sample)
        ctx.progress(completed=1, total=1, message="smoke test passed", phase="smoke")
        return {"ok": True}


from . import register  # noqa: E402 - register after the class is defined

register(RareSpotSmokeExecutor())
