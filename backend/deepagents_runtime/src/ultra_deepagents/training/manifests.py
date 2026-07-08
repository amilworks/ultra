"""Declarative model manifests for the GoldGate training subsystem.

A manifest is the source of truth for what the control-plane registry row must
contain (the DB row is what Go validates against without importing Python).
The parity test pins every manifest here to the shared fixture at
backend/contracts/training/<model_key>.manifest.json, which the Go seed test
pins from the other side — so the two languages cannot drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CAPABILITY_SYNC = "SYNC"
CAPABILITY_ASSEMBLE = "ASSEMBLE"
CAPABILITY_FINETUNE = "FINETUNE"
CAPABILITY_BENCHMARK = "BENCHMARK"

# The five generic job phases. Frozen at M1's "one enum PR"; adapters map
# capabilities onto the phases the control plane may dispatch (plan section 3.5).
TRAINING_JOB_PHASES = (
    "training.sync",
    "training.assemble",
    "training.finetune",
    "training.benchmark",
    "training.gold_freeze",
)

CAPABILITY_ALLOWED_PHASES: dict[str, tuple[str, ...]] = {
    CAPABILITY_SYNC: ("training.sync",),
    CAPABILITY_ASSEMBLE: ("training.assemble",),
    CAPABILITY_FINETUNE: ("training.finetune",),
    # A benchmark-only model materializes and freezes gold through the
    # gold_freeze job - no hand INSERT (plan section 3.5, round-2 fix).
    CAPABILITY_BENCHMARK: ("training.benchmark", "training.gold_freeze"),
}


@dataclass(frozen=True)
class ModelManifest:
    model_key: str
    task_type: str
    display_name: str
    dataset_format: str
    metric_schema: str
    requires_phash: bool
    capabilities: tuple[str, ...]
    classes: dict[str, str] | None
    leakage_defenses_extra: tuple[str, ...] = ()
    gt_layer_priority: tuple[str, ...] = ()
    executor: dict[str, Any] = field(default_factory=dict)

    def allowed_phases(self) -> tuple[str, ...]:
        phases: list[str] = []
        for capability in self.capabilities:
            for phase in CAPABILITY_ALLOWED_PHASES.get(capability, ()):
                if phase not in phases:
                    phases.append(phase)
        return tuple(phases)

    def to_contract_projection(self) -> dict[str, Any]:
        """The exact shape of the shared fixture's 'manifest' object."""
        return {
            "model_key": self.model_key,
            "task_type": self.task_type,
            "display_name": self.display_name,
            "dataset_format": self.dataset_format,
            "metric_schema": self.metric_schema,
            "requires_phash": self.requires_phash,
            "capabilities": list(self.capabilities),
            "classes": dict(self.classes or {}),
            "leakage_defenses_extra": list(self.leakage_defenses_extra),
            "gt_layer_priority": list(self.gt_layer_priority),
            "executor": dict(self.executor),
        }


RARESPOT_MANIFEST = ModelManifest(
    model_key="yolov5_rarespot",
    task_type="detection",
    display_name="RareSpot prairie-dog/burrow",
    dataset_format="yolo_txt_tiles_512",
    metric_schema="detection.v1",
    requires_phash=True,
    capabilities=(
        CAPABILITY_SYNC,
        CAPABILITY_ASSEMBLE,
        CAPABILITY_FINETUNE,
        CAPABILITY_BENCHMARK,
    ),
    classes={"0": "prairie_dog", "1": "burrow"},
    leakage_defenses_extra=("aerial_geospatial_overlap",),
    gt_layer_priority=("gt2", "New Ground Truth"),
    executor={
        "gpu_pool": "titan",
        "min_vram_gb": 8,
        "shm_gb": 8,
        "wall_clock_budget_s": 21600,
        "cpu_only": False,
    },
)

MANIFESTS: dict[str, ModelManifest] = {
    RARESPOT_MANIFEST.model_key: RARESPOT_MANIFEST,
}
