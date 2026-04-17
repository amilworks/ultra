from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import BaseTrainingAdapter, DynUNETAdapter, MedSAMAdapter, YOLOv5Adapter


class TrainingCancelledError(RuntimeError):
    pass


def build_model_version(model_key: str, job_id: str) -> str:
    token = str(model_key or "").strip().lower() or "model"
    jid = str(job_id or "").strip().lower()
    return f"{token}:{jid[:10] or 'latest'}"


@dataclass
class TrainingRunner:
    adapters: dict[str, BaseTrainingAdapter]

    def __init__(self) -> None:
        self.adapters = {
            "dynunet": DynUNETAdapter(),
            "medsam": MedSAMAdapter(),
            "yolov5_rarespot": YOLOv5Adapter(),
        }

    def get_adapter(self, model_key: str) -> BaseTrainingAdapter:
        token = str(model_key or "").strip().lower()
        adapter = self.adapters.get(token)
        if adapter is None:
            raise ValueError(f"Unsupported model key: {model_key}")
        return adapter

    def run_training(
        self,
        *,
        model_key: str,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: Callable[[dict[str, Any]], None],
        control_callback: Callable[[], None],
        initial_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        adapter = self.get_adapter(model_key)
        if not adapter.supports_training and not adapter.supports_finetune:
            raise ValueError(f"Model '{model_key}' does not support training or finetuning.")
        return adapter.train(
            manifest=manifest,
            config=config,
            output_dir=output_dir,
            progress_callback=progress_callback,
            control_callback=control_callback,
            initial_checkpoint_path=initial_checkpoint_path,
        )

    def run_inference(
        self,
        *,
        model_key: str,
        model_artifact_path: str | None,
        input_paths: list[str],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: Callable[[dict[str, Any]], None],
        control_callback: Callable[[], None],
    ) -> dict[str, Any]:
        adapter = self.get_adapter(model_key)
        if not adapter.supports_inference:
            raise ValueError(f"Model '{model_key}' does not support inference.")
        return adapter.infer(
            model_artifact_path=model_artifact_path,
            input_paths=input_paths,
            config=config,
            output_dir=output_dir,
            progress_callback=progress_callback,
            control_callback=control_callback,
        )

    def run_benchmark(
        self,
        *,
        model_key: str,
        model_artifact_path: str | None,
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: Callable[[dict[str, Any]], None],
        control_callback: Callable[[], None],
    ) -> dict[str, Any]:
        adapter = self.get_adapter(model_key)
        return adapter.benchmark(
            model_artifact_path=model_artifact_path,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            control_callback=control_callback,
        )
