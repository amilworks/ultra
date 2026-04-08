"""Typed graph events and stream envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.agents.contracts import GraphEventRecord as LegacyGraphEventRecord

GraphEventType = Literal[
    "run.started",
    "run.completed",
    "run.failed",
    "node.started",
    "node.completed",
    "state.updated",
    "task.queued",
    "task.progress",
    "task.completed",
    "artifact.created",
    "interrupt.raised",
    "checkpoint.created",
]

GraphStreamEventType = Literal["token", "done", "error", "run_event"]


class GraphEvent(BaseModel):
    """Typed event emitted by the new LangGraph execution layer."""

    event_type: GraphEventType
    workflow_kind: str = "interactive_chat"
    phase: str
    status: str
    agent_role: str | None = None
    node: str | None = None
    domain_id: str | None = None
    task_id: str | None = None
    checkpoint_id: str | None = None
    scope_id: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize_fields(self) -> "GraphEvent":
        self.phase = str(self.phase or "").strip() or "unknown"
        self.status = str(self.status or "").strip() or "progress"
        return self


class GraphStreamEnvelope(BaseModel):
    """SSE-ready frame for graph tokens, events, and completion payloads."""

    event: GraphStreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    retry: int | None = None


def coerce_graph_event(value: GraphEvent | LegacyGraphEventRecord | dict[str, Any]) -> GraphEvent:
    """Coerce legacy or dict payloads into the new graph event model."""

    if isinstance(value, GraphEvent):
        return value
    if isinstance(value, LegacyGraphEventRecord):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
    return GraphEvent(
        event_type=str(payload.get("event_type") or "state.updated"),
        workflow_kind=str(payload.get("workflow_kind") or "interactive_chat"),
        phase=str(payload.get("phase") or payload.get("node") or "unknown"),
        status=str(payload.get("status") or "progress"),
        agent_role=payload.get("agent_role"),
        node=payload.get("node"),
        domain_id=payload.get("domain_id"),
        task_id=payload.get("task_id"),
        checkpoint_id=payload.get("checkpoint_id"),
        scope_id=payload.get("scope_id"),
        message=payload.get("message"),
        payload=dict(payload.get("payload") or {}),
    )
