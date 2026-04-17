"""Execution runner seam for the code execution service."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_codeexec_attempt(
    *,
    job_id: str,
    work_dir: Path,
    request: dict[str, Any],
    worker_image: str,
    docker_network: str,
) -> dict[str, Any]:
    """Run one execution attempt inside the curated worker image."""

    raise NotImplementedError("Implement the Docker worker launch in Task 7.")

