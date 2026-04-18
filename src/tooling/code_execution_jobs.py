"""Helpers for packaging prepared code jobs for remote execution."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def load_job_spec_inputs(job_dir: Path) -> list[dict[str, Any]]:
    """Load the normalized inputs array from a persisted job spec."""

    spec = json.loads((job_dir / "job_spec.json").read_text(encoding="utf-8"))
    return list(spec.get("inputs") or [])


def build_service_submission_bundle(
    job_dir: Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pack a prepared job spec and source tree into one tarball."""

    spec = json.loads((job_dir / "job_spec.json").read_text(encoding="utf-8"))
    remote_inputs: list[dict[str, Any]] = []
    local_inputs: list[dict[str, Any]] = []
    for item in list(spec.get("inputs") or []):
        path_value = str(item.get("path") or "").strip()
        if path_value.startswith(("s3://", "http://", "https://")):
            remote_inputs.append(item)
        else:
            local_inputs.append(item)

    temp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    bundle_path = Path(temp_file.name)
    temp_file.close()
    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(job_dir / "job_spec.json", arcname="job_spec.json")
        archive.add(job_dir / "source", arcname="source")
    return bundle_path, remote_inputs, local_inputs


__all__ = ["build_service_submission_bundle", "load_job_spec_inputs"]
