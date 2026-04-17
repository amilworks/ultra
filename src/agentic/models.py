from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant", "tool"]
RunStatus = Literal["queued", "running", "waiting_for_input", "succeeded", "failed", "canceled"]


class AgenticAttachment(BaseModel):
    kind: Literal["file_id", "resource_uri", "dataset_uri"]
    value: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticMessageInput(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)
    attachments: list[AgenticAttachment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticRunBudget(BaseModel):
    max_tool_calls: int = Field(default=12, ge=1, le=64)
    max_runtime_seconds: int = Field(default=900, ge=1, le=86400)


class AgenticRunRequest(BaseModel):
    goal: str | None = None
    messages: list[AgenticMessageInput] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)
    knowledge_context: dict[str, Any] = Field(default_factory=dict)
    selection_context: dict[str, Any] = Field(default_factory=dict)
    workflow_hint: dict[str, Any] = Field(default_factory=dict)
    reasoning_mode: Literal["auto", "fast", "deep"] = "deep"
    budgets: AgenticRunBudget = Field(default_factory=AgenticRunBudget)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticExecutionResult(BaseModel):
    response_text: str
    current_step: str = "completed"
    checkpoint_state: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class ScientistPlan(BaseModel):
    summary: str = ""
    use_tools: bool = False
    recommended_tools: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class ScientistVerification(BaseModel):
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class ScientistDraft(BaseModel):
    answer: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    needs_escalation: bool = False
    notes: list[str] = Field(default_factory=list)


class ScientistTurnPlan(BaseModel):
    primary_domain: str = "core"
    selected_domains: list[str] = Field(default_factory=list)
    solve_mode: Literal["direct_response", "single_specialist", "team"] = "single_specialist"
    summary: str = ""
    use_tools: bool = False
    verification_focus: list[str] = Field(default_factory=list)
    requires_verification: bool = False
    notes: list[str] = Field(default_factory=list)


class ScientistSolvePacket(BaseModel):
    answer: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    resolution_mode: str = "single_specialist"
    notes: list[str] = Field(default_factory=list)
    requires_verification: bool = False


class ChemistryHydrogenCountResult(BaseModel):
    final_product_summary: str = ""
    hydrogen_signal_count: int | None = None
    chemically_distinct_sites: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
