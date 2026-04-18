"""Shared contract models for code execution service requests and results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CodeExecutionInput(BaseModel):
    """Normalized description of one execution input."""

    name: str
    kind: Literal["file", "directory"] = "file"
    sandbox_path: str
    description: str | None = None
    local_path: str | None = None
    uri: str | None = None

    @field_validator("sandbox_path")
    @classmethod
    def _validate_sandbox_path(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text.startswith("/inputs/"):
            raise ValueError("sandbox_path must live under /inputs/")
        return text


class CodeExecutionArtifact(BaseModel):
    """Artifact produced by one execution attempt."""

    name: str
    relative_path: str
    size_bytes: int
    kind: Literal["file"] = "file"


class CodeExecutionAttemptResult(BaseModel):
    """Terminal result for one execution attempt."""

    success: bool
    job_id: str
    execution_backend: Literal["docker", "service"]
    exit_code: int | None = None
    runtime_seconds: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_class: str | None = None
    error_message: str | None = None
    repair_hint: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    missing_expected_outputs: list[str] = Field(default_factory=list)
    analysis_outputs: list[dict[str, Any]] = Field(default_factory=list)
    measurement_candidates: list[dict[str, Any]] = Field(default_factory=list)
    key_measurements: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class CodeExecutionJobRequest(BaseModel):
    """Service submission payload for one prepared job."""

    job_id: str
    timeout_seconds: int = 900
    cpu_limit: float = 2.0
    memory_mb: int = 4096
    expected_outputs: list[str] = Field(default_factory=list)
    inputs: list[CodeExecutionInput] = Field(default_factory=list)


class CodeExecutionJobRecord(BaseModel):
    """Persisted service-side lifecycle record."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    request: CodeExecutionJobRequest
    artifact_manifest: list[CodeExecutionArtifact] = Field(default_factory=list)
    result: CodeExecutionAttemptResult | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


__all__ = [
    "CodeExecutionArtifact",
    "CodeExecutionAttemptResult",
    "CodeExecutionInput",
    "CodeExecutionJobRecord",
    "CodeExecutionJobRequest",
]
