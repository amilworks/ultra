"""Settings and helpers for the code execution service."""

from __future__ import annotations

import os
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

    @classmethod
    def from_env(cls) -> ServiceSettings:
        job_root = Path(
            os.environ.get("CODEEXEC_JOB_ROOT", "/srv/ultra/codeexec-service/jobs")
        ).resolve()
        artifact_root = Path(
            os.environ.get("CODEEXEC_ARTIFACT_ROOT", "/srv/ultra/codeexec-service/artifacts")
        ).resolve()
        return cls(
            job_root=job_root,
            artifact_root=artifact_root,
            api_key=str(os.environ.get("CODEEXEC_API_KEY", "") or "").strip(),
            worker_image=str(
                os.environ.get("CODEEXEC_WORKER_IMAGE", "ultra-codeexec-job:current") or ""
            ).strip()
            or "ultra-codeexec-job:current",
            docker_network=str(os.environ.get("CODEEXEC_DOCKER_NETWORK", "none") or "").strip()
            or "none",
            max_concurrent_jobs=max(
                1,
                int(os.environ.get("CODEEXEC_MAX_CONCURRENT_JOBS", "1") or "1"),
            ),
        )
