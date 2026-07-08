"""Wire envelope for GoldGate training jobs (plan section 9.2).

Mirrors the Go side's ``eventbus.TrainingJob`` field-for-field: the subject is
model-neutral, ``model_key`` travels as a validated attribute, and ``params``
reference rows/barrel paths — NATS stays dumb delivery, never payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ultra_deepagents.training.manifests import TRAINING_JOB_PHASES


@dataclass(frozen=True)
class TrainingJobEnvelope:
    job_id: str
    model_key: str
    job_type: str
    dispatch_id: str = ""
    gpu_pool: str = ""
    owner_user_id: str = "local-user"
    owner_org_id: str = "local-org"
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingJobEnvelope:
        """Tolerant parse: unknown keys ignored, wrong-typed dicts coerced to
        empty — but identity (job_id, model_key) and the phase enum are hard
        requirements. An invalid job_type is a poison message, never work."""
        if not isinstance(payload, dict):
            raise ValueError("training job payload must be an object")
        job_id = _string(payload.get("job_id"))
        if not job_id:
            raise ValueError("training job payload missing job_id")
        model_key = _string(payload.get("model_key"))
        if not model_key:
            raise ValueError("training job payload missing model_key")
        job_type = _string(payload.get("job_type"))
        if job_type not in TRAINING_JOB_PHASES:
            raise ValueError(
                f"training job_type {job_type!r} is not one of the five generic phases "
                f"{list(TRAINING_JOB_PHASES)}"
            )
        return cls(
            job_id=job_id,
            model_key=model_key,
            job_type=job_type,
            dispatch_id=_string(payload.get("dispatch_id")),
            gpu_pool=_string(payload.get("gpu_pool")),
            owner_user_id=_string(payload.get("owner_user_id")) or "local-user",
            owner_org_id=_string(payload.get("owner_org_id")) or "local-org",
            params=_dict(payload.get("params")),
            metadata=_dict(payload.get("metadata")),
        )

    def principal_headers(self, settings: Any | None = None) -> dict[str, str]:
        """Same principal shape the data-agent worker sends to the control plane."""
        headers = {
            "X-Ultra-User-Id": self.owner_user_id or "local-user",
            "X-Ultra-Org-Id": self.owner_org_id or "local-org",
            "X-Ultra-Role": "researcher",
        }
        token = str(getattr(settings, "control_worker_token", "") or "").strip() if settings else ""
        if token:
            headers["X-Ultra-Worker-Token"] = token
        return headers


def _string(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
