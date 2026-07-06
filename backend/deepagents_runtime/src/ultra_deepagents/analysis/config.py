"""Configuration for the batch model-inference worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _split_roots(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(":") if part.strip())


def _analysis_upload_roots() -> tuple[str, ...]:
    for var in ("ULTRA_ANALYSIS_UPLOAD_ROOTS", "ULTRA_UPLOAD_ROOTS", "ULTRA_CONTROL_UPLOAD_ROOT"):
        raw = _env_str(var)
        if raw:
            roots = _split_roots(raw)
            if roots:
                return roots
    return ("data/uploads",)


@dataclass(slots=True)
class AnalysisSettings:
    """Where to read inputs, where to write outputs, and how to reach the GPU + control plane."""

    control_base_url: str
    control_status_timeout_seconds: float
    # Outputs are written here; this MUST be the same volume the control plane serves
    # resources from (ULTRA_CONTROL_UPLOAD_ROOT) so registration is metadata-only.
    upload_root: Path
    # Roots scanned to resolve an input resource_id to its source file on disk.
    upload_roots: tuple[Path, ...]
    upload_database_url: str
    megaseg_service_url: str
    megaseg_service_api_key: str
    megaseg_timeout_seconds: float
    control_worker_token: str = ""
    output_prefix: str = "analysis"

    @classmethod
    def from_env(cls) -> AnalysisSettings:
        roots = tuple(Path(p).expanduser() for p in _analysis_upload_roots())
        upload_root_raw = (
            _env_str("ULTRA_ANALYSIS_OUTPUT_ROOT")
            or _env_str("ULTRA_CONTROL_UPLOAD_ROOT")
            or (str(roots[0]) if roots else "data/uploads")
        )
        return cls(
            control_base_url=_env_str("ULTRA_CONTROL_BASE_URL", "http://127.0.0.1:8088").rstrip("/"),
            control_status_timeout_seconds=_env_float("ULTRA_ANALYSIS_CONTROL_TIMEOUT_SECONDS", 30.0),
            upload_root=Path(upload_root_raw).expanduser(),
            upload_roots=roots,
            upload_database_url=(
                _env_str("ULTRA_ANALYSIS_UPLOAD_DATABASE_URL")
                or _env_str("ULTRA_RARESPOT_UPLOAD_DATABASE_URL")
            ),
            megaseg_service_url=_env_str("MEGASEG_SERVICE_URL").rstrip("/"),
            megaseg_service_api_key=_env_str("MEGASEG_SERVICE_API_KEY"),
            # Distinct from MEGASEG_SERVICE_TIMEOUT_SECONDS (the old submit-poll client's
            # short request timeout): /v1/infer runs a full synchronous segmentation, so this
            # must cover the whole inference, not just the request handshake.
            megaseg_timeout_seconds=_env_float("ULTRA_ANALYSIS_MEGASEG_TIMEOUT_SECONDS", 1800.0),
            control_worker_token=_env_str("ULTRA_CONTROL_WORKER_TOKEN"),
        )
