"""``rarespot.evaluate`` executor — the two-pass detection kernel on the GPU node.

A deliberately THIN wrapper over the adapter's ``RareSpotAdapter.evaluate`` so the
service and the worker share ONE implementation of the ``detection.v1`` contract
(val.py at conf 0.001 for curve metrics + the production detect path at conf 0.25
for operating-point metrics). No logic is duplicated here; drift is impossible.

The worker assembles the gold eval slices (CPU) and passes the manifest by
reference — every path in it must be readable inside this container.

spec:
    weights_uri   (required) candidate weights, path readable in-container
    gold_manifest (required) the eval manifest (slices with data_yaml/tiles_dir/...)
    slice         (optional) restrict to one slice; default all
returns:
    {detection: <detection.v1 blob>}
"""

from __future__ import annotations

from typing import Any

from ultra_deepagents.training.rarespot_adapter import RareSpotAdapter

from .base import ExecutorManifest, JobContext


class RareSpotEvaluateExecutor:
    manifest = ExecutorManifest(
        model_key="rarespot",
        capability="evaluate",
        description="two-pass detection.v1 eval (mAP curve + operating point) for a candidate",
        resource={"min_vram_gb": 8, "wall_clock_budget_s": 3600},
    )

    def run(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        weights_uri = str(spec.get("weights_uri") or "").strip()
        gold_manifest = spec.get("gold_manifest") or {}
        if not weights_uri:
            raise ValueError("rarespot.evaluate: weights_uri is required")
        if not isinstance(gold_manifest, dict) or not gold_manifest:
            raise ValueError("rarespot.evaluate: gold_manifest is required")
        slice_ = spec.get("slice")

        # Isolate eval scratch inside the job workdir so back-to-back runs never
        # collide on a shared temp dir.
        manifest = dict(gold_manifest)
        params = dict(manifest.get("params") or {})
        params.setdefault("workdir", str(ctx.workdir / "eval"))
        manifest["params"] = params

        ctx.progress(total=1, message="running two-pass evaluation", phase="benchmark")
        ctx.check_cancelled()
        # GPU pinning is at the container level (CUDA_VISIBLE_DEVICES set before torch
        # inits). evaluate_local = the kernel directly; never the routing evaluate(),
        # so a service node that (mis)configures ULTRA_COMPUTE_SERVICE_URL can't recurse.
        detection = RareSpotAdapter().evaluate_local(weights_uri, manifest, slice=slice_)
        aggregate = (detection or {}).get("aggregate") or {}
        ctx.progress(
            completed=1,
            total=1,
            message="evaluation complete",
            phase="benchmark",
            metrics={
                "map50": aggregate.get("map50"),
                "map50_95": aggregate.get("map50_95"),
                "precision_at_op": aggregate.get("precision_at_op"),
                "recall_at_op": aggregate.get("recall_at_op"),
            },
        )
        return {"detection": detection}


from . import register  # noqa: E402 - register after the class is defined

register(RareSpotEvaluateExecutor())
