"""``megaseg.infer`` executor — the proof that the framework is model-agnostic.

Adding a whole new model + capability is THIS FILE: a class with a ``manifest`` and
a ``run(spec, ctx)``, plus the one ``register(...)`` line at the bottom. Nothing in
the job manager, the app, or the other executors changes. Enabling it on a node is
purely operational: install the megaseg code (``src.science.megaseg_core``), set
``ULTRA_MEGASEG_CHECKPOINT_PATH``, and add ``megaseg_infer`` to
``ULTRA_COMPUTE_EXECUTORS`` (or leave the default = all).

The megaseg model + torch are imported LAZILY inside ``run`` so this module always
imports and registers — a node without the megaseg code still advertises the
capability and returns a clear error if actually asked to run it, rather than
silently dropping the executor.

spec:
    sources          (required) list of input file paths readable in-container
    structure_channel/nucleus_channel/channel_index_base/mask_threshold/... (optional)
    checkpoint_path  (optional) override ULTRA_MEGASEG_CHECKPOINT_PATH
    device           (optional) default env
returns:
    {result: <megaseg batch result>, output_dir, artifacts: [...]}
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .base import ExecutorManifest, JobContext

DEFAULT_CHECKPOINT = str(os.getenv("ULTRA_MEGASEG_CHECKPOINT_PATH") or "").strip()
DEFAULT_DEVICE = str(os.getenv("ULTRA_MEGASEG_DEVICE") or os.getenv("ULTRA_COMPUTE_GPU_DEVICE") or "cuda").strip()


class MegasegInferExecutor:
    manifest = ExecutorManifest(
        model_key="megaseg",
        capability="infer",
        description="megaseg segmentation inference over one or more input volumes",
        resource={"min_vram_gb": 16},
    )

    def __init__(self) -> None:
        # Cache the built model across jobs, keyed by checkpoint, guarded by a lock
        # (jobs are serialized by the semaphore, but the lock keeps this correct if
        # max_concurrent > 1 for a multi-GPU deployment).
        self._model: Any | None = None
        self._model_checkpoint: str | None = None
        self._lock = threading.Lock()

    def _get_model(self, checkpoint: str, device: str) -> Any:
        from src.science.megaseg_core import build_megaseg_model  # lazy import

        with self._lock:
            if self._model is not None and self._model_checkpoint == checkpoint:
                return self._model
            self._model = build_megaseg_model(checkpoint_path=Path(checkpoint), device=device)
            self._model_checkpoint = checkpoint
            return self._model

    def run(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        from src.science.megaseg_core import run_megaseg_batch  # lazy import

        sources = [str(s) for s in (spec.get("sources") or []) if str(s).strip()]
        if not sources:
            raise ValueError("megaseg.infer: at least one source path is required")
        checkpoint = str(spec.get("checkpoint_path") or DEFAULT_CHECKPOINT).strip()
        if not checkpoint:
            raise ValueError("megaseg.infer: checkpoint_path (or ULTRA_MEGASEG_CHECKPOINT_PATH) is required")
        device = str(spec.get("device") or DEFAULT_DEVICE).strip() or "cuda"
        output_dir = ctx.workdir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        ctx.progress(total=len(sources), message=f"segmenting {len(sources)} input(s)", phase="infer")
        ctx.check_cancelled()
        model = self._get_model(checkpoint, device)
        result = run_megaseg_batch(
            file_paths=sources,
            output_dir=output_dir,
            checkpoint_path=Path(checkpoint),
            structure_channel=int(spec.get("structure_channel") or 4),
            nucleus_channel=(int(spec["nucleus_channel"]) if spec.get("nucleus_channel") is not None else 6),
            channel_index_base=int(spec.get("channel_index_base") or 1),
            mask_threshold=float(spec.get("mask_threshold") or 0.5),
            save_visualizations=bool(spec.get("save_visualizations", True)),
            generate_report=bool(spec.get("generate_report", True)),
            device=device,
            structure_name=str(spec.get("structure_name") or "structure"),
            model=model,
            amp_enabled=bool(spec.get("amp_enabled", False)),
        )
        artifacts = [
            p.relative_to(output_dir).as_posix()
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        ]
        ctx.progress(
            completed=len(sources),
            total=len(sources),
            message="segmentation complete",
            phase="infer",
        )
        return {"result": result, "output_dir": str(output_dir), "artifacts": artifacts}


from . import register  # noqa: E402 - register after the class is defined

register(MegasegInferExecutor())
