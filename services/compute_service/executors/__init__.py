"""Executor registry — the model-agnostic dispatch table.

``register(executor)`` at import time; the service looks executors up by
``"{model_key}.{capability}"``. Import failures for one model's executor MUST NOT
take down the others (a missing torch dep for megaseg should not stop rarespot
training), so each built-in executor is imported defensively in
:func:`load_builtin_executors` and a failure is logged, not raised.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .base import Executor, ExecutorManifest, JobCancelled, JobContext

logger = logging.getLogger("ultra.compute.executors")

_REGISTRY: dict[str, Executor] = {}
_LOAD_ERRORS: dict[str, str] = {}

__all__ = [
    "Executor",
    "ExecutorManifest",
    "JobCancelled",
    "JobContext",
    "register",
    "get_executor",
    "list_manifests",
    "load_errors",
    "load_builtin_executors",
]


def register(executor: Executor) -> Executor:
    """Register an executor under ``model_key.capability``. Last write wins with a warning."""
    key = executor.manifest.key
    if key in _REGISTRY:
        logger.warning("executor %s already registered; overwriting", key)
    _REGISTRY[key] = executor
    logger.info("registered executor %s", key)
    return executor


def get_executor(key: str) -> Executor | None:
    return _REGISTRY.get(key)


def list_manifests() -> list[ExecutorManifest]:
    return [ex.manifest for ex in _REGISTRY.values()]


def load_errors() -> dict[str, str]:
    """Executors that failed to import — surfaced on /health for diagnosability."""
    return dict(_LOAD_ERRORS)


def load_builtin_executors(only: Iterable[str] | None = None) -> None:
    """Import + register the in-tree executors, each isolated from the others.

    ``only`` (a set of module stems, e.g. {"rarespot_train"}) restricts which are
    loaded — used by the deploy env to run a training-only or inference-only node.
    """
    builtins = ("rarespot_train", "rarespot_evaluate", "rarespot_smoke", "megaseg_infer")
    import importlib

    for stem in builtins:
        if only is not None and stem not in only:
            continue
        try:
            # Relative to THIS package so the service works under any top-level
            # mount name (services.compute_service.* or otherwise).
            importlib.import_module(f".{stem}", package=__name__)
        except Exception as exc:  # noqa: BLE001 - one bad executor must not sink the rest
            _LOAD_ERRORS[stem] = f"{type(exc).__name__}: {exc}"
            logger.warning("executor module %s failed to load: %s", stem, exc)
