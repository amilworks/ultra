"""Typed graph-facing models for the LangGraph backend foundation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.contracts import KnowledgeContext, ReasoningMode

GraphWorkflowKind = Literal[
    "interactive_chat",
    "scientific_plan",
    "upload_sync",
    "code_job",
    "training_job",
    "evaluation_job",
    "repro_report",
]

GraphExecutionStatus = Literal[
    "queued",
    "running",
    "waiting_for_input",
    "waiting_for_task",
    "succeeded",
    "failed",
    "canceled",
]

GraphNodeName = Literal[
    "preflight",
    "deliberation",
    "route",
    "fast_direct",
    "solve",
    "verify",
    "repair",
    "synthesize",
    "finalize",
    "task",
    "checkpoint",
]

GraphCheckpointKind = Literal[
    "initial",
    "node",
    "task",
    "interrupt",
    "final",
]


class GraphThreadRef(BaseModel):
    """User-facing thread identity projected from graph state."""

    thread_id: str
    user_id: str | None = None
    conversation_id: str | None = None
    workflow_kind: GraphWorkflowKind = "interactive_chat"


class GraphCheckpointRef(BaseModel):
    """Stable checkpoint handle for durable resume and replay."""

    checkpoint_id: str
    kind: GraphCheckpointKind = "node"
    node: GraphNodeName | None = None
    task_id: str | None = None
    previous_checkpoint_id: str | None = None


class GraphTaskRef(BaseModel):
    """Task projection for long-running or side-effecting work."""

    task_id: str
    queue_name: str
    kind: str
    status: GraphExecutionStatus = "queued"
    payload: dict[str, Any] = Field(default_factory=dict)
    attempt_count: int = 0


class GraphArtifactRef(BaseModel):
    """Artifact reference emitted by a graph node or task."""

    artifact_id: str
    run_id: str | None = None
    thread_id: str | None = None
    title: str
    path: str | None = None
    source_path: str | None = None
    preview_path: str | None = None
    mime_type: str | None = None
    category: str | None = None
    tool_name: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphExecutionMetadata(BaseModel):
    """Execution metadata preserved across graph and API projections."""

    thread_id: str | None = None
    run_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    workflow_kind: GraphWorkflowKind = "interactive_chat"
    reasoning_mode: ReasoningMode = "auto"
    selected_tool_names: list[str] = Field(default_factory=list)
    workflow_hint: dict[str, Any] = Field(default_factory=dict)
    knowledge_context: KnowledgeContext = Field(default_factory=KnowledgeContext)
    model: str | None = None
    trace_group_id: str | None = None
    budget_state: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
