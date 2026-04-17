from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

from src.agno_backend.knowledge import ScientificKnowledgeScope
from src.agno_backend.memory import ScientificMemoryPolicy

ChatRole = Literal["system", "user", "assistant", "tool"]
ConfidenceLevel = Literal["low", "medium", "high"]
BenchmarkAnswerFormat = Literal["mcq_letter"]
BenchmarkVisibleAnswerStyle = Literal["natural", "mcq"]
# Keep the full forced-regime surface for evaluation and benchmark override paths.
# Production routing only auto-selects a smaller default set.
ForcedProModeExecutionRegime = Literal[
    "fast_dialogue",
    "validated_tool",
    "iterative_research",
    "autonomous_cycle",
    "focused_team",
    "reasoning_solver",
    "proof_workflow",
    "expert_council",
]
ChatWorkflowHintId = Literal[
    "find_bisque_assets",
    "search_bisque_resources",
    "bisque_advanced_search",
    "load_bisque_resource",
    "bisque_download_resource",
    "bisque_download_dataset",
    "upload_to_bisque",
    "bisque_create_dataset",
    "bisque_add_to_dataset",
    "bisque_add_gobjects",
    "add_tags_to_resource",
    "bisque_fetch_xml",
    "delete_bisque_resource",
    "segment_sam3",
    "detect_prairie_dog",
    "detect_yolo",
    "estimate_depth_pro",
    "pro_mode",
    "scientific_calculator",
    "chemistry_workbench",
    "run_bisque_module",
]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ToolBudget(BaseModel):
    max_tool_calls: int = Field(default=12, ge=1, le=64)
    max_runtime_seconds: int = Field(default=900, ge=1, le=86400)


class ChatBenchmarkConfig(BaseModel):
    enabled: bool = True
    experiment_label: str | None = Field(default=None, max_length=128)
    hidden_answer_format: BenchmarkAnswerFormat = "mcq_letter"
    visible_answer_style: BenchmarkVisibleAnswerStyle = "natural"
    force_pro_mode_execution_regime: ForcedProModeExecutionRegime | None = None
    use_autonomy_agno_controller: bool = False
    disable_autonomy_memory_knowledge: bool = False
    disable_autonomy_focused_team_delegate: bool = False
    disable_autonomy_resume: bool = False
    autonomy_max_cycles: int | None = Field(default=None, ge=1, le=6)
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


class ChatWorkflowHint(BaseModel):
    id: ChatWorkflowHintId
    source: Literal["slash_menu"] = "slash_menu"


class KnowledgeContext(BaseModel):
    collaborator_id: str | None = None
    project_id: str | None = None
    pack_ids: list[str] = Field(default_factory=list)


class SelectionContext(BaseModel):
    context_id: str | None = None
    source: str | None = None
    focused_file_ids: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    artifact_handles: dict[str, list[str]] = Field(default_factory=dict)
    originating_message_id: str | None = None
    originating_user_text: str | None = None
    suggested_domain: str | None = None
    suggested_tool_names: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    uploaded_files: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(
        default_factory=list,
        description="Opaque file IDs returned by /v1/uploads.",
    )
    resource_uris: list[str] = Field(
        default_factory=list,
        description="Optional BisQue resource URIs or view URLs to import into this chat turn.",
    )
    dataset_uris: list[str] = Field(
        default_factory=list,
        description="Optional BisQue dataset URIs or view URLs to fan out into this chat turn.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation id used to persist and recover run state.",
    )
    goal: str | None = None
    selected_tool_names: list[str] = Field(
        default_factory=list,
        description="Optional explicit tool allowlist for this chat turn.",
    )
    knowledge_context: KnowledgeContext | None = Field(
        default=None,
        description="Optional collaborator/project knowledge-pack selectors for this chat turn.",
    )
    selection_context: SelectionContext | None = Field(
        default=None,
        description="Optional structured context carried from a prior selection or Use in Chat action.",
    )
    workflow_hint: ChatWorkflowHint | None = Field(
        default=None,
        description="Optional structured workflow selection from the slash composer.",
    )
    reasoning_mode: Literal["auto", "fast", "deep"] = Field(
        default="deep",
        description="Reasoning preference for chat orchestration. Defaults to the highest-quality setting.",
    )
    debug: bool = Field(
        default=False,
        description="When true, include compact execution diagnostics in response metadata and persisted run metadata.",
    )
    budgets: ToolBudget = Field(default_factory=ToolBudget)
    benchmark: ChatBenchmarkConfig | None = None


class ChatTitleRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Conversation messages used to derive a concise title.",
    )
    max_words: int = Field(default=4, ge=2, le=8)


class ChatTitleResponse(BaseModel):
    title: str
    model: str
    strategy: Literal["llm", "fallback"] = "llm"


class EvidenceItem(BaseModel):
    source: str
    run_id: str | None = None
    artifact: str | None = None
    summary: str | None = None


class MeasurementItem(BaseModel):
    name: str
    value: float | int | str
    unit: str | None = None
    ci95: tuple[float, float] | None = None


class ConfidenceBlock(BaseModel):
    level: ConfidenceLevel = "low"
    why: list[str] = Field(default_factory=list)


class NextStepAction(BaseModel):
    action: str
    workflow: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class AssistantContract(BaseModel):
    result: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    measurements: list[MeasurementItem] = Field(default_factory=list)
    statistical_analysis: list[dict[str, Any]] = Field(default_factory=list)
    confidence: ConfidenceBlock = Field(default_factory=ConfidenceBlock)
    qc_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[NextStepAction] = Field(default_factory=list)


class ChatResponse(BaseModel):
    run_id: str
    model: str
    response_text: str
    duration_seconds: float
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    benchmark: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class RunResultResponse(BaseModel):
    run_id: str
    status: Literal["pending", "running", "succeeded", "failed", "canceled"]
    result: ChatResponse | None = None


class AnalysisRunSummary(BaseModel):
    run_id: str
    conversation_id: str | None = None
    goal: str
    status: Literal["pending", "running", "succeeded", "failed", "canceled"]
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    tools: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None


class AnalysisHistoryResponse(BaseModel):
    count: int
    analyses: list[AnalysisRunSummary]


class ContractAuditRequest(BaseModel):
    run_ids: list[str] = Field(
        default_factory=list,
        description="Optional run IDs to audit. When empty, audits the latest runs for the active user.",
    )
    limit: int = Field(default=25, ge=1, le=200)


class ContractAuditRecord(BaseModel):
    run_id: str
    status: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    measurement_count: int = 0
    limitation_count: int = 0
    next_step_count: int = 0
    confidence_level: str | None = None
    confidence_why_count: int = 0
    research_score: int = 0
    research_max_score: int = 0
    research_summary: str = ""
    recommendations: list[str] = Field(default_factory=list)


class ContractAuditResponse(BaseModel):
    count: int
    passed: int
    failed: int
    average_research_score: float
    records: list[ContractAuditRecord] = Field(default_factory=list)


class CreateRunRequest(BaseModel):
    goal: str = Field(..., description="Human goal")
    plan: dict[str, Any] = Field(..., description="Structured workflow plan")


class CreateRunResponse(BaseModel):
    run_id: str
    status: str


class RunResponse(BaseModel):
    run_id: str
    goal: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    workflow_kind: str = "workflow_plan"
    mode: str = "durable"
    parent_run_id: str | None = None
    planner_version: str | None = None
    agent_role: str | None = None
    checkpoint_state: dict[str, Any] | None = None
    budget_state: dict[str, Any] | None = None
    trace_group_id: str | None = None


class RunEventRecord(BaseModel):
    ts: datetime | str | None = None
    level: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RunEventsResponse(BaseModel):
    run_id: str
    events: list[RunEventRecord] = Field(default_factory=list)


V2ThreadStatus = Literal["active", "archived", "deleted"]
V2RunStatus = Literal[
    "queued",
    "running",
    "waiting_for_input",
    "waiting_for_task",
    "succeeded",
    "failed",
    "canceled",
]
V2EventKind = Literal[
    "run.started",
    "node.started",
    "node.completed",
    "state.updated",
    "task.queued",
    "task.progress",
    "task.completed",
    "artifact.created",
    "interrupt.raised",
    "checkpoint.created",
    "run.completed",
    "run.failed",
    "message.delta",
    "error",
]
V2ArtifactKind = Literal["artifact", "preview", "report", "dataset", "file"]


class V2ThreadMessage(BaseModel):
    message_id: str | None = None
    thread_id: str | None = None
    role: str
    content: str
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class V2ThreadRecord(BaseModel):
    thread_id: str
    user_id: str | None = None
    title: str | None = None
    status: V2ThreadStatus = "active"
    created_at: datetime
    updated_at: datetime
    latest_run_id: str | None = None
    checkpoint_id: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2ThreadListResponse(BaseModel):
    count: int
    threads: list[V2ThreadRecord] = Field(default_factory=list)


class V2ThreadMessageListResponse(BaseModel):
    thread_id: str
    count: int
    messages: list[V2ThreadMessage] = Field(default_factory=list)


class V2ThreadCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_messages: list[V2ThreadMessage] = Field(default_factory=list)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class V2ThreadUpsertRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[V2ThreadMessage] = Field(default_factory=list)


class V2RunCreateRequest(BaseModel):
    goal: str | None = None
    messages: list[V2ThreadMessage] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)
    knowledge_context: KnowledgeContext | None = None
    selection_context: SelectionContext | None = None
    workflow_hint: ChatWorkflowHint | None = None
    reasoning_mode: Literal["auto", "fast", "deep"] = "deep"
    budgets: ToolBudget = Field(default_factory=ToolBudget)
    benchmark: ChatBenchmarkConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2RunResumeRequest(BaseModel):
    decision: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2RunCancelRequest(BaseModel):
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2GraphEventRecord(BaseModel):
    event_id: int | str | None = None
    run_id: str
    thread_id: str | None = None
    event_kind: V2EventKind
    event_type: str | None = None
    node_name: str | None = None
    task_id: str | None = None
    checkpoint_id: str | None = None
    scope_id: str | None = None
    agent_role: str | None = None
    level: str | None = None
    ts: datetime | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class V2RunRecord(BaseModel):
    run_id: str
    thread_id: str | None = None
    user_id: str | None = None
    goal: str
    status: V2RunStatus
    workflow_kind: str = "interactive_chat"
    mode: str | None = None
    current_node: str | None = None
    parent_run_id: str | None = None
    planner_version: str | None = None
    agent_role: str | None = None
    trace_group_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_state: dict[str, Any] | None = None
    budget_state: dict[str, Any] | None = None
    response_text: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2RunListResponse(BaseModel):
    count: int
    runs: list[V2RunRecord] = Field(default_factory=list)


class V2RunEventsResponse(BaseModel):
    run_id: str
    count: int
    events: list[V2GraphEventRecord] = Field(default_factory=list)


class V2ArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    thread_id: str | None = None
    kind: V2ArtifactKind = "artifact"
    path: str | None = None
    source_path: str | None = None
    preview_path: str | None = None
    title: str | None = None
    result_group_id: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    storage_uri: str | None = None
    tool_name: str | None = None
    category: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2ArtifactListResponse(BaseModel):
    run_id: str
    count: int
    artifacts: list[V2ArtifactRecord] = Field(default_factory=list)


class V2ArtifactResponse(BaseModel):
    artifact: V2ArtifactRecord


TrainingDatasetSplit = Literal["train", "val", "test"]
TrainingDatasetRole = Literal["image", "mask", "annotation"]
TrainingJobType = Literal["training", "inference"]
TrainingJobStatus = Literal["queued", "running", "paused", "succeeded", "failed", "canceled"]
ModelHealthStatus = Literal["Healthy", "Watch", "Retrain Recommended", "Needs Human Review"]
TrainingDomainOwnerScope = Literal["shared", "private"]
TrainingLineageScope = Literal["shared", "fork"]
TrainingVersionStatus = Literal["candidate", "canary", "active", "retired"]
TrainingProposalStatus = Literal[
    "pending_approval",
    "approved",
    "running",
    "evaluating",
    "ready_to_promote",
    "promoted",
    "rejected",
    "failed",
]
TrainingMergeStatus = Literal["open", "evaluating", "approved", "rejected", "executed", "failed"]
PrairieBenchmarkMode = Literal["canonical_only", "promotion_packet"]
TrainingTriggerReason = Literal["data_threshold", "schedule", "health", "manual", "merge", "none"]


class TrainingModelRecord(BaseModel):
    key: str
    name: str
    framework: str
    task_type: str = "segmentation"
    description: str
    supports_training: bool
    supports_finetune: bool
    supports_inference: bool
    dimensions: list[str] = Field(default_factory=list)
    default_config: dict[str, Any] = Field(default_factory=dict)


class TrainingModelsResponse(BaseModel):
    count: int
    models: list[TrainingModelRecord] = Field(default_factory=list)


class TrainingDatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingDatasetItemAssignment(BaseModel):
    file_id: str = Field(min_length=1, max_length=128)
    split: TrainingDatasetSplit
    role: TrainingDatasetRole
    sample_id: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingDatasetItemsRequest(BaseModel):
    items: list[TrainingDatasetItemAssignment] = Field(default_factory=list)
    replace: bool = False


class TrainingDatasetRecord(BaseModel):
    dataset_id: str
    user_id: str
    name: str
    description: str | None = None
    item_count: int = 0
    split_counts: dict[str, int] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TrainingDatasetResponse(BaseModel):
    dataset: TrainingDatasetRecord
    manifest: dict[str, Any] = Field(default_factory=dict)


class TrainingDatasetListResponse(BaseModel):
    count: int
    datasets: list[TrainingDatasetRecord] = Field(default_factory=list)


class TrainingJobCreateRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=128)
    model_key: str = Field(min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)
    confirm_launch: bool = True
    initial_checkpoint_path: str | None = None


class TrainingPreflightRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=128)
    model_key: str = Field(min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingPreflightResponse(BaseModel):
    dataset_id: str
    model_key: str
    config: dict[str, Any] = Field(default_factory=dict)
    recommended_launch: bool = False
    report: dict[str, Any] = Field(default_factory=dict)


class TrainingJobControlRequest(BaseModel):
    action: Literal["pause", "resume", "cancel", "restart"]


class InferenceJobCreateRequest(BaseModel):
    model_key: str = Field(min_length=1, max_length=64)
    model_version: str | None = Field(default=None, max_length=256)
    file_ids: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    reviewed_samples: int = Field(default=0, ge=0, le=1_000_000)
    reviewed_failures: int = Field(default=0, ge=0, le=1_000_000)
    confirm_launch: bool = True


class TrainingJobRecord(BaseModel):
    job_id: str
    user_id: str
    job_type: TrainingJobType
    dataset_id: str | None = None
    model_key: str
    model_version: str | None = None
    status: TrainingJobStatus
    artifact_run_id: str | None = None
    error: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    control: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_heartbeat_at: datetime | None = None


class TrainingJobResponse(BaseModel):
    job: TrainingJobRecord


class ModelHealthRecord(BaseModel):
    model_key: str
    model_version: str
    status: ModelHealthStatus
    recommendation: str
    rationale: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    training_runs: int = 0
    inference_runs: int = 0
    user_id: str | None = None


class ModelHealthResponse(BaseModel):
    count: int
    models: list[ModelHealthRecord] = Field(default_factory=list)


class TrainingDomainCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    owner_scope: TrainingDomainOwnerScope = "shared"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingDomainRecord(BaseModel):
    domain_id: str
    name: str
    description: str | None = None
    owner_scope: TrainingDomainOwnerScope
    owner_user_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TrainingDomainListResponse(BaseModel):
    count: int
    domains: list[TrainingDomainRecord] = Field(default_factory=list)


class TrainingLineageRecord(BaseModel):
    lineage_id: str
    domain_id: str
    scope: TrainingLineageScope
    owner_user_id: str
    model_key: str
    parent_lineage_id: str | None = None
    active_version_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TrainingLineageListResponse(BaseModel):
    count: int
    lineages: list[TrainingLineageRecord] = Field(default_factory=list)


class TrainingForkLineageRequest(BaseModel):
    model_key: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingModelVersionRecord(BaseModel):
    version_id: str
    lineage_id: str
    source_job_id: str | None = None
    artifact_run_id: str | None = None
    status: TrainingVersionStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TrainingModelVersionListResponse(BaseModel):
    count: int
    versions: list[TrainingModelVersionRecord] = Field(default_factory=list)


class TrainingUpdateProposalRecord(BaseModel):
    proposal_id: str
    lineage_id: str
    trigger_reason: TrainingTriggerReason | str
    trigger_snapshot: dict[str, Any] = Field(default_factory=dict)
    dataset_snapshot: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    status: TrainingProposalStatus | str
    idempotency_key: str | None = None
    approved_by: str | None = None
    rejected_by: str | None = None
    linked_job_id: str | None = None
    candidate_version_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TrainingUpdateProposalListResponse(BaseModel):
    count: int
    proposals: list[TrainingUpdateProposalRecord] = Field(default_factory=list)


class TrainingUpdateProposalPreviewRequest(BaseModel):
    lineage_id: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(min_length=1, max_length=128)
    approved_new_samples: int | None = Field(default=None, ge=0, le=1_000_000)
    class_counts: dict[str, int] = Field(default_factory=dict)
    health_status: ModelHealthStatus | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    trigger_reason_override: TrainingTriggerReason | None = None
    idempotency_key: str | None = Field(default=None, max_length=256)
    persist: bool = False


class TrainingUpdateProposalPreviewResponse(BaseModel):
    trigger: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)
    proposal: TrainingUpdateProposalRecord | None = None


class TrainingUpdateProposalDecisionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=4000)
    confirm_launch: bool = True


class TrainingUpdateProposalResponse(BaseModel):
    proposal: TrainingUpdateProposalRecord


class TrainingVersionPromoteRequest(BaseModel):
    note: str | None = Field(default=None, max_length=4000)


class TrainingVersionRollbackRequest(BaseModel):
    target_version_id: str | None = Field(default=None, max_length=128)
    note: str | None = Field(default=None, max_length=4000)


class TrainingModelVersionResponse(BaseModel):
    version: TrainingModelVersionRecord
    lineage: TrainingLineageRecord


class TrainingMergeRequestCreateRequest(BaseModel):
    source_lineage_id: str = Field(min_length=1, max_length=128)
    target_lineage_id: str = Field(min_length=1, max_length=128)
    candidate_version_id: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=4000)


class TrainingMergeRequestDecisionRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=4000)


class TrainingMergeRequestRecord(BaseModel):
    merge_id: str
    source_lineage_id: str
    target_lineage_id: str
    candidate_version_id: str
    requested_by: str
    status: TrainingMergeStatus | str
    decision_by: str | None = None
    notes: str | None = None
    evaluation: dict[str, Any] = Field(default_factory=dict)
    linked_proposal_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    decided_at: datetime | None = None
    executed_at: datetime | None = None


class TrainingMergeRequestResponse(BaseModel):
    merge_request: TrainingMergeRequestRecord


class TrainingMergeRequestListResponse(BaseModel):
    count: int
    merge_requests: list[TrainingMergeRequestRecord] = Field(default_factory=list)


class PrairieSyncResponse(BaseModel):
    success: bool
    dataset_name: str
    dataset_id: str | None = None
    bisque_dataset_uri: str | None = None
    synced_images: int = 0
    reviewed_images: int = 0
    unreviewed_images: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    unsupported_class_counts: dict[str, int] = Field(default_factory=dict)
    last_sync_at: str | None = None
    errors: list[str] = Field(default_factory=list)


class PrairieStatusResponse(BaseModel):
    dataset_name: str
    dataset_id: str | None = None
    last_sync_at: str | None = None
    next_sync_at: str | None = None
    active_model_version: str | None = None
    model_health: ModelHealthStatus | str = "Needs Human Review"
    reviewed_images: int = 0
    unreviewed_images: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    unsupported_class_counts: dict[str, int] = Field(default_factory=dict)
    detection_counts: dict[str, int] = Field(default_factory=dict)
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    benchmark_baseline: dict[str, Any] = Field(default_factory=dict)
    benchmark_latest_candidate: dict[str, Any] = Field(default_factory=dict)
    last_benchmark_at: str | None = None
    benchmark_ready: bool = False
    canonical_benchmark_ready: bool = False
    promotion_benchmark_ready: bool = False
    retrain_gate: bool = False
    retrain_gate_reasons: list[str] = Field(default_factory=list)
    retrain_gate_counts: dict[str, int] = Field(default_factory=dict)


class PrairieRetrainRequest(BaseModel):
    confirm_launch: bool = True
    note: str | None = Field(default=None, max_length=4000)


class PrairieRetrainRecord(BaseModel):
    request_id: str
    training_job_id: str
    status: TrainingJobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    model_version: str | None = None
    note: str | None = None
    error: str | None = None
    gating_summary: dict[str, Any] = Field(default_factory=dict)
    benchmark_report_artifact_id: str | None = None


class PrairieRetrainListResponse(BaseModel):
    count: int
    requests: list[PrairieRetrainRecord] = Field(default_factory=list)


class PrairieBenchmarkRunResponse(BaseModel):
    run_id: str
    model_version: str | None = None
    mode: PrairieBenchmarkMode = "canonical_only"
    benchmark_ready: bool = False
    canonical_benchmark_ready: bool = False
    promotion_benchmark_ready: bool = False
    report: dict[str, Any] = Field(default_factory=dict)


class PrairieBenchmarkRunRequest(BaseModel):
    mode: PrairieBenchmarkMode = "canonical_only"


class ArtifactRecord(BaseModel):
    path: str
    size_bytes: int
    mime_type: str | None = None
    modified_at: datetime
    source_path: str | None = None
    title: str | None = None
    result_group_id: str | None = None


class ArtifactListResponse(BaseModel):
    run_id: str
    root: str
    artifact_count: int
    artifacts: list[ArtifactRecord]


class StatsToolRecord(BaseModel):
    name: str
    description: str


class StatsToolsResponse(BaseModel):
    tool_count: int
    tools: list[StatsToolRecord]


class StatsRunRequest(BaseModel):
    tool_name: str = Field(..., description="Curated statistical tool name")
    payload: dict[str, Any] = Field(default_factory=dict, description="Tool-specific payload")


class StatsRunResponse(BaseModel):
    success: bool
    tool_name: str
    result: dict[str, Any] | None = None
    error: str | None = None


class ReproReportRequest(BaseModel):
    run_id: str | None = None
    title: str | None = None
    result_summary: str | None = None
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    statistical_analysis: list[dict[str, Any]] | dict[str, Any] | None = None
    qc_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    next_steps: list[dict[str, Any] | str] = Field(default_factory=list)
    output_dir: str | None = None


class ReproReportResponse(BaseModel):
    success: bool
    run_id: str | None = None
    report_markdown_path: str
    report_json_path: str
    report_sha256: str
    report_bundle_sha256: str


class ImageLoadRequest(BaseModel):
    file_path: str = Field(..., description="Path to local image/volume file")
    scene: int | str | None = None
    use_aicspylibczi: bool = False
    array_mode: Literal["plane", "volume", "tczyx"] = "plane"
    t_index: int | None = None
    c_index: int | None = None
    z_index: int | None = None
    save_array: bool = True
    include_array: bool = False
    max_inline_elements: int = Field(default=16384, ge=64, le=1_000_000)


class ImageLoadResponse(BaseModel):
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class UploadedFileRecord(BaseModel):
    file_id: str
    original_name: str
    content_type: str | None = None
    size_bytes: int
    sha256: str
    created_at: datetime
    sync_status: str | None = None
    canonical_resource_uniq: str | None = None
    canonical_resource_uri: str | None = None
    client_view_url: str | None = None
    image_service_url: str | None = None
    sync_error: str | None = None
    sync_run_id: str | None = None


class UploadFilesResponse(BaseModel):
    file_count: int
    uploaded: list[UploadedFileRecord]


class ViewerPlaneDescriptor(BaseModel):
    axis: Literal["z", "y", "x"]
    label: str
    axes: list[str] = Field(default_factory=list)
    pixel_size: dict[str, int] = Field(default_factory=dict)
    spacing: dict[str, float] = Field(default_factory=dict)
    world_size: dict[str, float] = Field(default_factory=dict)
    aspect_ratio: float = 1.0


class ViewerServiceUrls(BaseModel):
    preview: str | None = None
    display: str | None = None
    slice: str | None = None
    tile: str | None = None
    atlas: str | None = None
    scalar_volume: str | None = None
    histogram: str | None = None
    dataset: str | None = None
    table: str | None = None


class ViewerDisplayDefaults(BaseModel):
    enhancement: str = "d"
    negative: bool = False
    rotate: int = 0
    fusion_method: str = "m"
    channel_mode: str = "composite"
    channels: list[int] = Field(default_factory=list)
    channel_colors: list[str] = Field(default_factory=list)
    time_index: int = 0
    z_index: int = 0
    volume_channel: int | None = None
    volume_clip_min: dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    volume_clip_max: dict[str, float] = Field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )


class ViewerTileLevel(BaseModel):
    level: int
    width: int
    height: int
    columns: int
    rows: int
    downsample: float


class ViewerTileScheme(BaseModel):
    tile_size: int
    format: str = "png"
    levels: list[ViewerTileLevel] = Field(default_factory=list)


class ViewerAtlasScheme(BaseModel):
    slice_count: int
    columns: int
    rows: int
    slice_width: int
    slice_height: int
    atlas_width: int
    atlas_height: int
    downsample: float
    format: str = "png"


class Hdf5TreeNode(BaseModel):
    path: str
    name: str
    node_type: Literal["group", "dataset"]
    child_count: int = 0
    attributes_count: int = 0
    shape: list[int] | None = None
    dtype: str | None = None
    preview_kind: str | None = None
    children: list[Hdf5TreeNode] = Field(default_factory=list)


class Hdf5GeometrySummary(BaseModel):
    path: str | None = None
    dimensions: list[int] | None = None
    spacing: list[float] | None = None
    origin: list[float] | None = None


class Hdf5Summary(BaseModel):
    group_count: int = 0
    dataset_count: int = 0
    dataset_kinds: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False
    geometry: Hdf5GeometrySummary | None = None


class Hdf5MaterialsPayload(BaseModel):
    detected: bool = False
    schema_name: Literal["dream3d"] | None = Field(
        default=None,
        validation_alias=AliasChoices("schema", "schema_name"),
        serialization_alias="schema",
    )
    capabilities: list[str] = Field(default_factory=list)
    roles: dict[str, str] = Field(default_factory=dict)
    phase_names: list[str] = Field(default_factory=list)
    feature_count: int | None = None
    grain_count: int | None = None
    recommended_view: Literal["materials", "explorer"] = "explorer"


class Hdf5ViewerPayload(BaseModel):
    enabled: bool = True
    supported: bool = True
    status: str = "ready"
    error: str | None = None
    root_keys: list[str] = Field(default_factory=list)
    root_attributes: dict[str, Any] = Field(default_factory=dict)
    summary: Hdf5Summary = Field(default_factory=Hdf5Summary)
    tree: list[Hdf5TreeNode] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    selected_dataset_path: str | None = None
    default_dataset_path: str | None = None
    materials: Hdf5MaterialsPayload | None = None


class Hdf5DatasetField(BaseModel):
    name: str
    dtype: str


class Hdf5DatasetStatistics(BaseModel):
    sample_count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    unique_values: int | None = None


class Hdf5DatasetSummaryResponse(BaseModel):
    file_id: str
    dataset_path: str
    dataset_name: str
    preview_kind: str | None = None
    semantic_role: str | None = None
    units_hint: str | None = None
    materials_domain_tags: list[str] = Field(default_factory=list)
    dtype: str
    shape: list[int] = Field(default_factory=list)
    rank: int = 0
    element_count: int = 0
    estimated_bytes: int | None = None
    dimension_summary: dict[str, int] | None = None
    capabilities: list[str] = Field(default_factory=list)
    render_policy: Literal["scalar", "categorical", "display", "analysis"] = "analysis"
    delivery_mode: Literal["direct", "scalar", "atlas", "deferred_multiscale"] = "direct"
    diagnostic_surface: Literal["mpr", "none"] = "none"
    first_paint_mode: Literal["image", "webgl"] = "image"
    measurement_policy: Literal["pixel-only", "spacing-aware", "orientation-aware"] = "pixel-only"
    texture_policy: Literal["linear", "nearest"] = "nearest"
    display_capabilities: list[str] = Field(default_factory=list)
    viewer_capabilities: list[str] = Field(default_factory=list)
    volume_eligible: bool = False
    volume_reason: str | None = None
    axis_sizes: dict[str, int] | None = None
    physical_spacing: dict[str, float] | None = None
    atlas_scheme: ViewerAtlasScheme | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    geometry: Hdf5GeometrySummary | None = None
    structured_fields: list[Hdf5DatasetField] = Field(default_factory=list)
    component_count: int = 1
    component_labels: list[str] = Field(default_factory=list)
    slice_axes: list[Literal["z", "y", "x"]] = Field(default_factory=list)
    preview_planes: dict[str, Any] = Field(default_factory=dict)
    sample_shape: list[int] = Field(default_factory=list)
    sample_values: Any | None = None
    sample_statistics: Hdf5DatasetStatistics | None = None


class Hdf5HistogramBin(BaseModel):
    label: str
    start: float | None = None
    end: float | None = None
    count: int = 0


class Hdf5DatasetHistogramResponse(BaseModel):
    file_id: str
    dataset_path: str
    preview_kind: str | None = None
    component_index: int | None = None
    component_label: str | None = None
    sample_count: int = 0
    discrete: bool = False
    min: float | None = None
    max: float | None = None
    bins: list[Hdf5HistogramBin] = Field(default_factory=list)


class Hdf5TablePreviewColumn(BaseModel):
    key: str
    label: str
    dtype: str
    numeric: bool = False


class Hdf5TablePreviewChart(BaseModel):
    kind: Literal["scatter", "histogram"]
    title: str
    description: str | None = None
    x_key: str
    y_key: str
    data: list[dict[str, Any]] = Field(default_factory=list)


class Hdf5DatasetTablePreviewResponse(BaseModel):
    file_id: str
    dataset_path: str
    preview_kind: str | None = None
    offset: int = 0
    limit: int = 0
    total_rows: int = 0
    total_columns: int = 0
    columns: list[Hdf5TablePreviewColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[Hdf5TablePreviewChart] = Field(default_factory=list)


class Hdf5MaterialsMapResponse(BaseModel):
    title: str
    description: str | None = None
    dataset_path: str
    semantic_role: str
    preview_kind: str | None = None


class Hdf5MaterialsChartResponse(BaseModel):
    kind: Literal["scatter", "histogram", "bar"]
    title: str
    description: str | None = None
    x_key: str
    y_key: str
    data: list[dict[str, Any]] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    units_hint: str | None = None
    provenance: str | None = None


class Hdf5MaterialsDatasetLinkResponse(BaseModel):
    label: str
    dataset_path: str
    semantic_role: str
    group: str


class Hdf5MaterialsOverviewResponse(BaseModel):
    geometry: Hdf5GeometrySummary | None = None
    spacing_note: str | None = None
    phase_names: list[str] = Field(default_factory=list)
    feature_count: int | None = None
    grain_count: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    recommended_map_dataset_path: str | None = None


class Hdf5MaterialsDashboardResponse(BaseModel):
    file_id: str
    schema_name: Literal["dream3d"] = Field(
        validation_alias=AliasChoices("schema", "schema_name"),
        serialization_alias="schema",
    )
    overview: Hdf5MaterialsOverviewResponse
    maps: list[Hdf5MaterialsMapResponse] = Field(default_factory=list)
    grain_charts: list[Hdf5MaterialsChartResponse] = Field(default_factory=list)
    orientation_charts: list[Hdf5MaterialsChartResponse] = Field(default_factory=list)
    synthetic_stats: list[Hdf5MaterialsChartResponse] = Field(default_factory=list)
    dataset_links: list[Hdf5MaterialsDatasetLinkResponse] = Field(default_factory=list)


class UploadViewerResponse(BaseModel):
    kind: Literal["image", "hdf5"] = "image"
    file_id: str
    original_name: str
    modality: str | None = None
    dims_order: str = "TCZYX"
    backend_mode: str | None = None
    axis_sizes: dict[str, int] = Field(default_factory=dict)
    selected_indices: dict[str, int] = Field(default_factory=dict)
    is_volume: bool = False
    is_timeseries: bool = False
    is_multichannel: bool = False
    phys: dict[str, Any] | None = None
    display_defaults: ViewerDisplayDefaults | None = None
    service_urls: ViewerServiceUrls | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    viewer: dict[str, Any] = Field(default_factory=dict)
    hdf5: Hdf5ViewerPayload | None = None


class ResourceRecord(BaseModel):
    file_id: str
    original_name: str
    content_type: str | None = None
    size_bytes: int
    sha256: str
    created_at: datetime
    source_type: str = "upload"
    resource_kind: str = "file"
    source_uri: str | None = None
    client_view_url: str | None = None
    image_service_url: str | None = None
    has_thumbnail: bool = False
    thumbnail_url: str | None = None
    preview_url: str | None = None
    sync_status: str | None = None
    sync_error: str | None = None
    canonical_resource_uniq: str | None = None
    canonical_resource_uri: str | None = None
    cache_ready: bool = False
    staged_locally: bool = False
    sync_run_id: str | None = None


class ResourceListResponse(BaseModel):
    count: int
    resources: list[ResourceRecord]


class ResourceComputationLookupRequest(BaseModel):
    file_ids: list[str] = Field(
        default_factory=list,
        description="Upload file_ids to check for prior computation runs.",
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description="Optional tool names to constrain lookup (segment/yolo/depth aliases accepted).",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional user prompt used to infer tool intent when tool_names is empty.",
    )
    limit_per_file_tool: int = Field(default=1, ge=1, le=5)


class ResourceComputationSuggestion(BaseModel):
    requested_file_id: str
    requested_file_name: str
    requested_file_sha256: str
    tool_name: str
    run_id: str
    run_status: str
    run_goal: str | None = None
    run_updated_at: str
    conversation_id: str | None = None
    conversation_title: str | None = None
    conversation_updated_at: str | None = None
    match_type: Literal["sha256", "filename"]


class ResourceComputationLookupResponse(BaseModel):
    count: int
    suggestions: list[ResourceComputationSuggestion] = Field(default_factory=list)


class ResumableUploadInitRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=2048)
    size_bytes: int = Field(ge=1)
    content_type: str | None = None
    fingerprint: str = Field(min_length=1, max_length=512)
    chunk_size_bytes: int = Field(default=5 * 1024 * 1024, ge=256 * 1024, le=64 * 1024 * 1024)


class ResumableUploadSessionResponse(BaseModel):
    upload_id: str
    file_name: str
    size_bytes: int
    content_type: str | None = None
    chunk_size_bytes: int
    bytes_received: int
    status: Literal["active", "completed", "failed"]
    uploaded: UploadedFileRecord | None = None
    error: str | None = None


class ResumableUploadChunkResponse(BaseModel):
    upload_id: str
    bytes_received: int
    size_bytes: int
    complete: bool
    status: Literal["active", "completed", "failed"]


class ResumableUploadCompleteResponse(BaseModel):
    upload_id: str
    uploaded: UploadedFileRecord


class Sam3PointAnnotation(BaseModel):
    x: float
    y: float
    label: int | str = 1


class Sam3BoxAnnotation(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: int | str = 1


class Sam3ImageAnnotation(BaseModel):
    file_id: str = Field(min_length=1, max_length=128)
    points: list[Sam3PointAnnotation] = Field(default_factory=list)
    boxes: list[Sam3BoxAnnotation] = Field(default_factory=list)


class Sam3InteractiveRequest(BaseModel):
    file_ids: list[str] = Field(
        default_factory=list,
        description="Upload file_ids returned by /v1/uploads.",
    )
    annotations: list[Sam3ImageAnnotation] = Field(default_factory=list)
    model: str | None = Field(
        default="medsam",
        description=(
            "Interactive segmentation backend. Supported values: medsam, medsam2, sam3. "
            "MedSAM2 is the default scientific path; SAM3 is retained only for explicit legacy requests."
        ),
    )
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    concept_prompt: str | None = None
    save_visualizations: bool = True
    preset: str | None = "balanced"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    point_box_size: float = Field(
        default=20.0,
        ge=2.0,
        le=512.0,
        description="Point radius converted into SAM3 box prompts in pixel units.",
    )
    min_points: int | None = Field(default=None, ge=1, le=1024)
    max_points: int | None = Field(default=None, ge=1, le=2048)
    tracker_prompt_mode: Literal[
        "single_object_refine",
        "per_positive_point_instance",
    ] = Field(
        default="single_object_refine",
        description=(
            "How SAM3 tracker point prompts are grouped. "
            "'single_object_refine' treats all positive/negative clicks as one object prompt "
            "(SAM2-style refinement). "
            "'per_positive_point_instance' treats each positive click as its own object prompt."
        ),
    )
    force_rerun: bool = False


class Sam3InteractiveResponse(BaseModel):
    success: bool
    run_id: str
    response_text: str
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ConversationUpsertRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=256)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    state: dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(BaseModel):
    conversation_id: str
    title: str
    created_at_ms: int
    updated_at_ms: int
    preview: str = ""
    message_count: int = 0
    preferred_panel: Literal["chat"] = "chat"
    running: bool = False
    state: dict[str, Any] = Field(default_factory=dict)


class ConversationListResponse(BaseModel):
    count: int
    total_count: int
    limit: int
    offset: int
    has_more: bool
    conversations: list[ConversationRecord]


class ConversationSearchResponse(BaseModel):
    query: str
    count: int
    matches: list[dict[str, Any]] = Field(default_factory=list)


class SantaBarbaraWeatherResponse(BaseModel):
    success: bool
    location: str = "Santa Barbara, CA"
    micro_location: str | None = None
    observed_at: str | None = None
    temperature_f: float | None = None
    apparent_temperature_f: float | None = None
    weather_code: int | None = None
    weather_label: str | None = None
    wind_speed_mph: float | None = None
    daily_high_f: float | None = None
    daily_low_f: float | None = None
    precipitation_probability_percent: float | None = None
    wave_height_ft: float | None = None
    swell_wave_height_ft: float | None = None
    wave_period_seconds: float | None = None
    blip: str | None = None
    summary: str
    source: str = "open-meteo"


class BisqueImportRequest(BaseModel):
    resources: list[str] = Field(
        default_factory=list,
        description=(
            "BisQue resource references to import (client_service/view URLs, "
            "data_service URLs, image_service URLs, or resource IDs)."
        ),
    )


class BisqueImportItem(BaseModel):
    input_url: str
    resource_uri: str | None = None
    resource_uniq: str | None = None
    client_view_url: str | None = None
    image_service_url: str | None = None
    status: Literal["imported", "reused", "error"]
    download_source: str | None = None
    error: str | None = None
    uploaded: UploadedFileRecord | None = None


class BisqueImportResponse(BaseModel):
    file_count: int
    uploaded: list[UploadedFileRecord]
    imports: list[BisqueImportItem]


class BisqueAuthLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)


class BisqueGuestProfile(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    email: str = Field(min_length=3, max_length=320)
    affiliation: str = Field(min_length=1, max_length=512)


class BisqueAuthGuestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    email: str = Field(min_length=3, max_length=320)
    affiliation: str = Field(min_length=1, max_length=512)


class BisqueAuthSessionResponse(BaseModel):
    authenticated: bool
    username: str | None = None
    bisque_root: str | None = None
    expires_at: datetime | None = None
    mode: Literal["bisque", "guest"] | None = None
    guest_profile: BisqueGuestProfile | None = None
    is_admin: bool = False


class AdminPlatformKpis(BaseModel):
    total_users: int = 0
    active_users_24h: int = 0
    total_conversations: int = 0
    conversations_started_24h: int = 0
    total_messages: int = 0
    messages_last_24h: int = 0
    user_messages_last_24h: int = 0
    assistant_messages_last_24h: int = 0
    total_runs: int = 0
    runs_last_24h: int = 0
    success_rate_last_24h: float = 0.0
    running_runs: int = 0
    failed_runs_24h: int = 0
    total_uploads: int = 0
    soft_deleted_uploads: int = 0
    total_storage_bytes: int = 0
    avg_messages_per_conversation: float = 0.0


class AdminUsageBucket(BaseModel):
    bucket_start: datetime
    runs_total: int = 0
    runs_succeeded: int = 0
    runs_failed: int = 0
    uploads: int = 0
    new_users: int = 0


class AdminToolUsageRecord(BaseModel):
    tool_name: str
    count: int = 0
    succeeded: int = 0
    failed: int = 0


class AdminUserSummary(BaseModel):
    user_id: str
    conversations: int = 0
    messages: int = 0
    runs_total: int = 0
    runs_running: int = 0
    runs_failed: int = 0
    runs_succeeded: int = 0
    uploads: int = 0
    storage_bytes: int = 0
    last_activity_at: datetime | None = None


class AdminRunRecord(BaseModel):
    run_id: str
    user_id: str | None = None
    conversation_id: str | None = None
    goal: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    duration_seconds: float | None = None
    tool_names: list[str] = Field(default_factory=list)


class AdminIssueRecord(BaseModel):
    issue_type: Literal["failed_run", "failed_upload_session", "stalled_run"]
    severity: Literal["high", "medium", "low"] = "medium"
    user_id: str | None = None
    run_id: str | None = None
    upload_id: str | None = None
    conversation_id: str | None = None
    message: str
    occurred_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminOverviewResponse(BaseModel):
    generated_at: datetime
    kpis: AdminPlatformKpis
    usage_last_24h: list[AdminUsageBucket] = Field(default_factory=list)
    tool_usage_7d: list[AdminToolUsageRecord] = Field(default_factory=list)
    top_users: list[AdminUserSummary] = Field(default_factory=list)
    recent_issues: list[AdminIssueRecord] = Field(default_factory=list)


class AdminUserListResponse(BaseModel):
    count: int
    users: list[AdminUserSummary] = Field(default_factory=list)


class AdminRunListResponse(BaseModel):
    count: int
    runs: list[AdminRunRecord] = Field(default_factory=list)


class AdminIssueListResponse(BaseModel):
    count: int
    issues: list[AdminIssueRecord] = Field(default_factory=list)


class AdminRunActionResponse(BaseModel):
    run_id: str
    previous_status: str
    status: str
    updated: bool


class AdminConversationActionResponse(BaseModel):
    conversation_id: str
    user_id: str
    deleted: bool


class V3AttachmentRecord(BaseModel):
    kind: Literal["file_id", "resource_uri", "dataset_uri"]
    value: str = Field(min_length=1)
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V3SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    status: str = "active"
    summary: str | None = None
    memory_policy: ScientificMemoryPolicy = Field(default_factory=ScientificMemoryPolicy)
    knowledge_scope: ScientificKnowledgeScope = Field(default_factory=ScientificKnowledgeScope)
    metadata: dict[str, Any] = Field(default_factory=dict)


class V3SessionRecord(BaseModel):
    session_id: str
    user_id: str | None = None
    title: str
    status: str
    summary: str | None = None
    memory_policy: ScientificMemoryPolicy = Field(default_factory=ScientificMemoryPolicy)
    knowledge_scope: ScientificKnowledgeScope = Field(default_factory=ScientificKnowledgeScope)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class V3SessionListResponse(BaseModel):
    count: int
    sessions: list[V3SessionRecord] = Field(default_factory=list)


class V3MessageRecord(BaseModel):
    message_id: str
    session_id: str
    role: ChatRole
    content: str
    attachments: list[V3AttachmentRecord] = Field(default_factory=list)
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class V3MessageInput(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)
    attachments: list[V3AttachmentRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class V3SessionMessageListResponse(BaseModel):
    session_id: str
    count: int
    messages: list[V3MessageRecord] = Field(default_factory=list)


class V3RunCreateRequest(BaseModel):
    goal: str | None = None
    messages: list[V3MessageInput] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    dataset_uris: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)
    knowledge_context: KnowledgeContext | None = None
    selection_context: SelectionContext | None = None
    workflow_hint: ChatWorkflowHint | None = None
    reasoning_mode: Literal["auto", "fast", "deep"] = "deep"
    debug: bool = False
    budgets: ToolBudget = Field(default_factory=ToolBudget)
    metadata: dict[str, Any] = Field(default_factory=dict)


class V3RunRecord(BaseModel):
    run_id: str
    session_id: str
    user_id: str | None = None
    workflow_name: str
    status: str
    current_step: str | None = None
    checkpoint_state: dict[str, Any] = Field(default_factory=dict)
    budget_state: dict[str, Any] = Field(default_factory=dict)
    response_text: str | None = None
    trace_group_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class V3RunEventRecord(BaseModel):
    event_id: str
    run_id: str
    session_id: str | None = None
    user_id: str | None = None
    event_kind: str
    event_type: str
    agent_name: str | None = None
    tool_name: str | None = None
    level: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class V3RunEventListResponse(BaseModel):
    run_id: str
    count: int
    events: list[V3RunEventRecord] = Field(default_factory=list)


class V3ArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    session_id: str | None = None
    user_id: str | None = None
    kind: str
    title: str | None = None
    path: str | None = None
    source_path: str | None = None
    preview_path: str | None = None
    result_group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class V3ArtifactListResponse(BaseModel):
    run_id: str
    count: int
    artifacts: list[V3ArtifactRecord] = Field(default_factory=list)


class V3ApprovalRecord(BaseModel):
    approval_id: str
    run_id: str
    session_id: str | None = None
    user_id: str | None = None
    action_type: str
    tool_name: str | None = None
    status: str
    request_payload: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class V3ApprovalResolveRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class V3ApprovalResolveResponse(BaseModel):
    approval: V3ApprovalRecord
    run: V3RunRecord
