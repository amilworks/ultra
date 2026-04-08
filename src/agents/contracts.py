"""Typed internal contracts for chat-centric agent workflows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


IssueSeverity = Literal["low", "medium", "high"]
ReasoningMode = Literal["auto", "fast", "deep"]
BenchmarkAnswerFormat = Literal["mcq_letter"]
BenchmarkVisibleAnswerStyle = Literal["natural", "mcq"]
BiologyProblemClass = Literal[
    "molecular_mechanism",
    "sequence_design",
    "assay_quantification",
    "microscopy_quantification",
    "comparative_statistics",
    "image_interpretation",
    "conceptual_only",
]
SolveMode = Literal[
    "fast_direct",
    "specialist_only",
    "specialist_plus_verifier",
    "specialist_plus_verifier_plus_code",
]
AgentRole = Literal[
    "triage",
    "planner",
    "domain_specialist",
    "biology",
    "verifier",
    "synthesizer",
    "coder",
    "vision",
    "medical",
    "safety_governor",
]
GraphStatus = Literal["started", "completed", "failed", "progress"]
OperationIntent = Literal[
    "analyze",
    "answer",
    "detect",
    "segment",
    "count",
    "search",
    "load",
    "upload",
    "diagnose",
]
ArtifactModality = Literal[
    "unknown",
    "microscopy_image",
    "clinical_image",
    "clinical_volume",
    "materials_image",
    "table",
    "dataset",
    "resource",
]
RouteSource = Literal[
    "heuristic",
    "route_judge",
    "intent_guardrail",
    "selected_tool_constraint",
    "evidence_reconciler",
    "fallback",
]


class ArtifactBinding(BaseModel):
    """Explicit artifact reference shared between bounded solve steps."""

    key: str
    value: str | list[str] | dict[str, Any] | None = None
    source_task_id: str | None = None


class EvidenceArtifact(BaseModel):
    """Renderable artifact emitted by a tool or verification step."""

    kind: str
    title: str
    path: str | None = None
    mime_type: str | None = None
    source_tool: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSnippet(BaseModel):
    """Small file-backed reference snippet injected into specialist handoffs."""

    pack_id: str
    title: str
    source_path: str
    excerpt: str
    match_reason: str = ""


class KnowledgeContext(BaseModel):
    """Optional collaborator/project selectors for local context-pack lookup."""

    collaborator_id: str | None = None
    project_id: str | None = None
    pack_ids: list[str] = Field(default_factory=list)


class ToolResultEnvelope(BaseModel):
    """Canonical envelope for tool outputs used by orchestration and UI."""

    success: bool = True
    summary: str = ""
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    ui_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    download_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    structured_outputs: dict[str, Any] = Field(default_factory=dict)
    next_actions: list[str] = Field(default_factory=list)
    artifact_bindings: list[ArtifactBinding] = Field(default_factory=list)


class GraphEventRecord(BaseModel):
    """Typed workflow event persisted for chat progress UIs."""

    kind: Literal["graph", "tool"] = "graph"
    workflow_kind: str = "interactive_chat"
    phase: str
    status: GraphStatus
    agent_role: AgentRole | None = None
    node: str | None = None
    domain_id: str | None = None
    task_id: str | None = None
    scope_id: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ResourceFocus(BaseModel):
    """Structured focus carried from uploads, Use in Chat, or deictic selection."""

    context_id: str | None = None
    source: str | None = None
    focused_file_ids: list[str] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    originating_message_id: str | None = None
    originating_user_text: str | None = None
    suggested_domain: str | None = None
    suggested_tool_names: list[str] = Field(default_factory=list)


class TurnIntent(BaseModel):
    """Immutable per-turn intent ledger shared across routing and handoffs."""

    original_user_text: str = ""
    normalized_user_text: str = ""
    resolved_context_text: str | None = None
    resolved_context_source: str | None = None
    selected_tool_names: list[str] = Field(default_factory=list)
    workflow_hint: dict[str, Any] = Field(default_factory=dict)
    resource_focus: ResourceFocus = Field(default_factory=ResourceFocus)
    knowledge_context: KnowledgeContext = Field(default_factory=KnowledgeContext)
    operation_intent: OperationIntent = "analyze"
    artifact_modality: ArtifactModality = "unknown"
    scientific_domain: str | None = None
    evidence_signals: list[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    """Domain routing result for one user turn."""

    selected_domains: list[str] = Field(default_factory=list)
    primary_domain: str | None = None
    secondary_domains: list[str] = Field(default_factory=list)
    operation_intent: OperationIntent = "analyze"
    artifact_modality: ArtifactModality = "unknown"
    score_by_domain: dict[str, float] = Field(default_factory=dict)
    evidence_signals: list[str] = Field(default_factory=list)
    route_source: RouteSource = "heuristic"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    used_model_classifier: bool = False

    @model_validator(mode="after")
    def _sync_domains(self) -> "RouteDecision":
        if self.primary_domain:
            ordered = [str(self.primary_domain).strip()]
            ordered.extend(
                token
                for token in self.secondary_domains
                if str(token).strip() and str(token).strip() not in ordered
            )
            self.selected_domains = ordered
        elif self.selected_domains:
            ordered = [str(token).strip() for token in self.selected_domains if str(token).strip()]
            self.selected_domains = ordered
            self.primary_domain = ordered[0] if ordered else None
            self.secondary_domains = ordered[1:]
        else:
            self.primary_domain = None
            self.secondary_domains = []
        if not self.reason:
            self.reason = self.route_source
        return self


class BenchmarkConfig(BaseModel):
    """Benchmark-only configuration knobs for challenger experiments."""

    enabled: bool = False
    experiment_label: str | None = None
    hidden_answer_format: BenchmarkAnswerFormat = "mcq_letter"
    visible_answer_style: BenchmarkVisibleAnswerStyle = "natural"
    duplicate_solve_enabled: bool = False
    duplicate_solve_passes: int = Field(default=2, ge=1, le=4)
    strict_option_elimination: bool = False
    chemistry_reasoning_boost: bool = False
    biology_reasoning_boost: bool = False
    biology_quant_planner_enabled: bool = False
    biology_parallel_critic_enabled: bool = False
    force_verifier: bool = False
    force_code_verification: bool = False
    allow_retry_reconciliation: bool = True


class BiologyPlan(BaseModel):
    """Structured biology-planning output used to guide tooling and critique."""

    problem_class: BiologyProblemClass = "conceptual_only"
    requires_tools: bool = False
    recommended_tools: list[str] = Field(default_factory=list)
    require_code_verification: bool = False
    require_parallel_critic: bool = False
    evidence_requirements: list[str] = Field(default_factory=list)
    reasoning_focus: str = ""


class BenchmarkHiddenAnswer(BaseModel):
    """Hidden benchmark-only grading payload."""

    letter: str | None = None
    source: str | None = None


class BenchmarkResult(BaseModel):
    """Benchmark-only result payload surfaced to offline evaluators."""

    enabled: bool = False
    experiment_label: str | None = None
    hidden_answer: BenchmarkHiddenAnswer | None = None
    visible_answer_style: BenchmarkVisibleAnswerStyle = "natural"
    trace: dict[str, Any] = Field(default_factory=dict)


class DeliberationPolicy(BaseModel):
    """Evidence-gated solve policy for one chat turn."""

    reasoning_mode: ReasoningMode = "auto"
    solve_mode: SolveMode = "specialist_only"
    route: RouteDecision = Field(default_factory=RouteDecision)
    verifier_enabled: bool = True
    code_verification_enabled: bool = False
    signals: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class SolveTask(BaseModel):
    """Per-domain solve task specification."""

    domain_id: str
    prompt: str


class Claim(BaseModel):
    """One normalized claim produced by a domain solver."""

    text: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceRef(BaseModel):
    """Reference to evidence backing a claim."""

    source: str = ""
    detail: str | None = None


class ToolInsight(BaseModel):
    """Structured interpretation derived from one tool invocation."""

    tool: str
    headline: str
    details: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Structured output from one domain solve node."""

    domain_id: str
    success: bool = True
    summary: str = ""
    raw_output: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    tool_insights: list[ToolInsight] = Field(default_factory=list)
    error: str | None = None
    tool_calls: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationIssue(BaseModel):
    """One verifier finding."""

    code: str
    severity: IssueSeverity = "medium"
    message: str
    domain_id: str | None = None
    correctable: bool = True


class VerificationReport(BaseModel):
    """Verification outcome across all solve node outputs."""

    passed: bool = True
    issues: list[VerificationIssue] = Field(default_factory=list)
    retry_domains: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CodeVerificationResult(BaseModel):
    """Outcome from the bounded code-verifier stage."""

    attempted: bool = False
    verified: bool | None = None
    recommendation: Literal["accept", "repair", "escalate", "not_needed"] = "not_needed"
    summary: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    remaining_uncertainty: list[str] = Field(default_factory=list)
    raw_output: str = ""
    tool_calls: int = 0


class SynthesisPacket(BaseModel):
    """Canonical payload delivered to synthesis."""

    user_text: str = ""
    deliberation: DeliberationPolicy = Field(default_factory=DeliberationPolicy)
    route: RouteDecision = Field(default_factory=RouteDecision)
    agent_results: list[AgentResult] = Field(default_factory=list)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    code_verification: CodeVerificationResult | None = None


class HandoffPacket(BaseModel):
    """Bounded specialist handoff packet with filtered context only."""

    domain_id: str
    turn_intent: TurnIntent = Field(default_factory=TurnIntent)
    route: RouteDecision = Field(default_factory=RouteDecision)
    conversation_excerpt: list[str] = Field(default_factory=list)
    reference_context: list[ContextSnippet] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)
    retry_feedback: str | None = None
    notes: list[str] = Field(default_factory=list)


class WorkGraphExecution(BaseModel):
    """Bounded workgraph execution result."""

    route: RouteDecision = Field(default_factory=RouteDecision)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    retry_count: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
