"""Framework-level settings for the compute service.

Only the model-AGNOSTIC knobs live here (auth, roots, concurrency, which executors
to load). Model-specific configuration (yolov5 path, training image, gpu device
selection, ...) is read by each executor from its own ``ULTRA_<MODEL>_*`` env so a
new model never has to touch this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name) or "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # noqa: TRY003
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(slots=True)
class ServiceSettings:
    api_key: str
    job_root: Path
    workspace_root: Path
    max_concurrent: int = 1
    history_limit: int = 200
    # None => load every built-in executor; a set restricts to those module stems
    # (e.g. {"rarespot_train", "rarespot_evaluate"} for a training-only node).
    executors: frozenset[str] | None = None

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        api_key = _env("ULTRA_COMPUTE_API_KEY")
        if not api_key:
            raise RuntimeError("ULTRA_COMPUTE_API_KEY is required for the compute service.")
        job_root = Path(
            _env("ULTRA_COMPUTE_JOB_ROOT", "/srv/ultra/compute-service/jobs")
        ).expanduser()
        workspace_root = Path(
            _env("ULTRA_COMPUTE_WORKSPACE_ROOT", "/srv/ultra/compute-service/work")
        ).expanduser()
        executors_raw = _env("ULTRA_COMPUTE_EXECUTORS")
        executors = (
            frozenset(part.strip() for part in executors_raw.split(",") if part.strip())
            if executors_raw
            else None
        )
        return cls(
            api_key=api_key,
            job_root=job_root,
            workspace_root=workspace_root,
            max_concurrent=max(1, _env_int("ULTRA_COMPUTE_MAX_CONCURRENT", 1)),
            history_limit=max(16, _env_int("ULTRA_COMPUTE_HISTORY_LIMIT", 200)),
            executors=executors,
        )
