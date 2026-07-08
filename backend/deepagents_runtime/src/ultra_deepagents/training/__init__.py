"""GoldGate training subsystem.

The declarative half of the ModelAdapter contract (manifests) is pinned
cross-language from M0. The M1 training worker adds the imperative half: the
five adapter hooks (adapters.py, rarespot_adapter.py), the job envelope, the
sync control-plane client, the OS-thread lease keepalive, and the NATS worker
with the five generic phase branches (worker.py).
Design: planning/2026-07-07-goldgate-continual-finetuning-plan.md.
"""

from ultra_deepagents.training.adapters import (
    ADAPTERS,
    MaterializeContext,
    ModelAdapter,
    NoAdapterRegistered,
    TrainContext,
    get_adapter,
    register_adapter,
)
from ultra_deepagents.training.envelope import TrainingJobEnvelope
from ultra_deepagents.training.manifests import (
    MANIFESTS,
    RARESPOT_MANIFEST,
    TRAINING_JOB_PHASES,
    ModelManifest,
)

__all__ = [
    "ADAPTERS",
    "MANIFESTS",
    "RARESPOT_MANIFEST",
    "TRAINING_JOB_PHASES",
    "MaterializeContext",
    "ModelAdapter",
    "ModelManifest",
    "NoAdapterRegistered",
    "TrainContext",
    "TrainingJobEnvelope",
    "get_adapter",
    "register_adapter",
]
