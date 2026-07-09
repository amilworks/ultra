"""The imperative half of the ModelAdapter contract (plan section 3.5).

Exactly five hooks, registry-resolved by ``model_key``. The training worker
dispatches by JOB PHASE and calls these hooks — never a model-named branch.
An unknown ``model_key`` raises ``NoAdapterRegistered`` BEFORE the worker
leases work it can't do.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Plan 3.5 capability table: SYNC materializes "sync"; ASSEMBLE materializes
# params.purpose in {"finetune_mix", "gold_cut"}.
MATERIALIZE_PURPOSES = ("sync", "gold_cut", "finetune_mix")


class NoAdapterRegistered(LookupError):  # noqa: N818 - contract name from the plan
    """No adapter is registered for the requested model_key."""


@dataclass(frozen=True)
class MaterializeContext:
    """Context for ``materialize_dataset``; ``purpose`` is registry-validated."""

    purpose: str
    params: dict[str, Any] = field(default_factory=dict)
    workdir: Path = Path(".")
    settings: Any | None = None

    def __post_init__(self) -> None:
        if self.purpose not in MATERIALIZE_PURPOSES:
            raise ValueError(
                f"materialize purpose {self.purpose!r} is not one of {list(MATERIALIZE_PURPOSES)}"
            )


@dataclass(frozen=True)
class TrainContext:
    """Context for ``train`` (the assemble/finetune phases).

    ``job_id`` / ``progress_cb`` / ``should_cancel`` are optional runtime handles the
    worker injects so an adapter that offloads GPU work to the compute service can
    (a) key the job idempotently on the training job id — a redelivery RESUMES the
    running service job rather than launching a duplicate; (b) forward live progress
    back to the control plane; (c) observe run cancellation and stop the GPU job.
    They are None for pure-local adapters and callers that don't need them.
    """

    params: dict[str, Any] = field(default_factory=dict)
    workdir: Path = Path(".")
    settings: Any | None = None
    job_id: str = ""
    progress_cb: Callable[[dict[str, Any]], None] | None = None
    should_cancel: Callable[[], bool] | None = None


class ModelAdapter(abc.ABC):
    """Five hooks per plan 3.5. Everything else in the pipeline is generic."""

    model_key: str = ""
    # Named extra leakage checks this adapter actually implements. The freeze
    # asserts every manifest.leakage_defenses_extra name resolves here AT
    # EXECUTION (plan 4.4) — a manifest string cannot silently outrun the code.
    implemented_leakage_checks: tuple[str, ...] = ()

    @abc.abstractmethod
    def materialize_dataset(self, ctx: MaterializeContext) -> dict[str, Any]:
        """Source pull + label materialization into standard items
        ``{item_id, source_ref, content_sha256, phash|None, label_kind,
        gt_label_sha256, label_uri, width, height, label_stats, strata_tags}``."""

    @abc.abstractmethod
    def train(self, ctx: TrainContext) -> dict[str, Any]:
        """Build+run the finetune; returns
        ``{weights_uri, train_log_uri, hyp_sha256, data_manifest_sha256}``."""

    @abc.abstractmethod
    def evaluate(
        self,
        weights_uri: str,
        gold_manifest: dict[str, Any],
        slice: str | None = None,  # noqa: A002 - the plan-3.5 contract name
    ) -> dict[str, Any]:
        """Metrics dict conforming to ``manifest.metric_schema``. ``slice`` may
        be None ("all required slices" — the normal benchmark call); the blob
        always emits every required slice key (empty => label_count 0 + null
        metrics, plan 7.3's emission contract)."""

    @abc.abstractmethod
    def smoke_test(self, weights_uri: str, gold_sample: Any) -> None:
        """Raise if the candidate weights are structurally unusable; run by the
        worker BEFORE version registration."""

    @abc.abstractmethod
    def extra_leakage_checks(
        self,
        train_pool: list[dict[str, Any]],
        gold_items: list[dict[str, Any]],
        *,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Adapter-declared defenses, run fail-closed on the worker during the
        gold-freeze job. Returns violations (empty list = pass)."""


ADAPTERS: dict[str, ModelAdapter] = {}

_defaults_loaded = False


def register_adapter(adapter: ModelAdapter) -> None:
    key = str(adapter.model_key or "").strip()
    if not key:
        raise ValueError("adapter has no model_key")
    ADAPTERS[key] = adapter


def get_adapter(model_key: str) -> ModelAdapter:
    _ensure_default_adapters()
    normalized = str(model_key or "").strip()
    adapter = ADAPTERS.get(normalized)
    if adapter is None:
        raise NoAdapterRegistered(f"no adapter registered for model_key={normalized or '<empty>'}")
    return adapter


def _ensure_default_adapters() -> None:
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True
    # Registers RareSpotAdapter on import; import here to avoid a module cycle.
    import ultra_deepagents.training.rarespot_adapter  # noqa: F401
