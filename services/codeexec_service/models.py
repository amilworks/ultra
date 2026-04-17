"""Settings and helpers for the code execution service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ServiceSettings:
    """Runtime settings for the private code execution service."""

    job_root: Path
    artifact_root: Path
    api_key: str
    worker_image: str
    docker_network: str = "none"
    max_concurrent_jobs: int = 1

