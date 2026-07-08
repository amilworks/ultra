"""GoldGate training adapters (M0 skeleton).

The declarative half of the ModelAdapter contract lives here from M0 so the
model-key spelling and manifest values are pinned cross-language before any
executor exists. The imperative adapter hooks (materialize_dataset, train,
evaluate, smoke_test, extra_leakage_checks) arrive with the M1 training worker.
Design: planning/2026-07-07-goldgate-continual-finetuning-plan.md (section 3.5).
"""

from ultra_deepagents.training.manifests import (
    MANIFESTS,
    ModelManifest,
    RARESPOT_MANIFEST,
)

__all__ = ["MANIFESTS", "ModelManifest", "RARESPOT_MANIFEST"]
