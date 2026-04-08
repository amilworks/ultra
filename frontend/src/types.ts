export type ChatRole = "system" | "user" | "assistant" | "tool";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ToolBudget = {
  max_tool_calls: number;
  max_runtime_seconds: number;
};

export type ChatBenchmarkConfig = {
  enabled?: boolean;
  experiment_label?: string | null;
  hidden_answer_format?: "mcq_letter";
  visible_answer_style?: "natural" | "mcq";
  force_pro_mode_execution_regime?:
    | "fast_dialogue"
    | "validated_tool"
    | "iterative_research"
    | "autonomous_cycle"
    | "focused_team"
    | "reasoning_solver"
    | "proof_workflow"
    | "expert_council"
    | null;
  use_autonomy_agno_controller?: boolean;
  disable_autonomy_memory_knowledge?: boolean;
  disable_autonomy_focused_team_delegate?: boolean;
  disable_autonomy_resume?: boolean;
  autonomy_max_cycles?: number | null;
  duplicate_solve_enabled?: boolean;
  duplicate_solve_passes?: number;
  strict_option_elimination?: boolean;
  chemistry_reasoning_boost?: boolean;
  biology_reasoning_boost?: boolean;
  biology_quant_planner_enabled?: boolean;
  biology_parallel_critic_enabled?: boolean;
  force_verifier?: boolean;
  force_code_verification?: boolean;
  allow_retry_reconciliation?: boolean;
};

export type ChatWorkflowHintId =
  | "find_bisque_assets"
  | "search_bisque_resources"
  | "bisque_advanced_search"
  | "load_bisque_resource"
  | "bisque_download_resource"
  | "bisque_download_dataset"
  | "upload_to_bisque"
  | "bisque_create_dataset"
  | "bisque_add_to_dataset"
  | "bisque_add_gobjects"
  | "add_tags_to_resource"
  | "bisque_fetch_xml"
  | "delete_bisque_resource"
  | "segment_sam3"
  | "detect_prairie_dog"
  | "detect_yolo"
  | "estimate_depth_pro"
  | "pro_mode"
  | "scientific_calculator"
  | "chemistry_workbench"
  | "run_bisque_module";

export type ChatWorkflowHint = {
  id: ChatWorkflowHintId;
  source: "slash_menu";
};

export type KnowledgeContext = {
  collaborator_id?: string | null;
  project_id?: string | null;
  pack_ids?: string[];
};

export type V3MemoryPolicy = {
  mode?: "off" | "conservative";
  session_summary?: boolean;
  user_memory?: boolean;
  project_notebook?: boolean;
};

export type V3KnowledgeScope = {
  mode?: "default" | "project_notebook";
  project_id?: string | null;
  namespaces?: string[];
  include_curated_packs?: boolean;
  include_uploads?: boolean;
  include_project_notes?: boolean;
};

export type SelectionContext = {
  context_id?: string | null;
  source?: string | null;
  focused_file_ids?: string[];
  resource_uris?: string[];
  dataset_uris?: string[];
  artifact_handles?: Record<string, string[]>;
  originating_message_id?: string | null;
  originating_user_text?: string | null;
  suggested_domain?: string | null;
  suggested_tool_names?: string[];
};

export type ChatRequest = {
  messages: ChatMessage[];
  uploaded_files: string[];
  file_ids?: string[];
  resource_uris?: string[];
  dataset_uris?: string[];
  conversation_id?: string | null;
  goal?: string | null;
  selected_tool_names?: string[];
  knowledge_context?: KnowledgeContext | null;
  selection_context?: SelectionContext | null;
  workflow_hint?: ChatWorkflowHint | null;
  reasoning_mode?: "auto" | "fast" | "deep";
  debug?: boolean;
  budgets: ToolBudget;
  benchmark?: ChatBenchmarkConfig | null;
};

export type ChatTitleRequest = {
  messages: ChatMessage[];
  max_words?: number;
};

export type ChatTitleResponse = {
  title: string;
  model: string;
  strategy: "llm" | "fallback";
};

export type ConfidenceBlock = {
  level: "low" | "medium" | "high";
  why: string[];
};

export type EvidenceItem = {
  source: string;
  run_id?: string | null;
  artifact?: string | null;
  summary?: string | null;
};

export type MeasurementItem = {
  name: string;
  value: number | string;
  unit?: string | null;
  ci95?: [number, number] | null;
};

export type NextStepAction = {
  action: string;
  workflow?: string | null;
  args?: Record<string, unknown>;
};

export type AssistantContract = {
  result: string;
  evidence: EvidenceItem[];
  measurements: MeasurementItem[];
  statistical_analysis: Array<Record<string, unknown>>;
  confidence: ConfidenceBlock;
  qc_warnings: string[];
  limitations: string[];
  next_steps: NextStepAction[];
};

export type ProgressEvent = {
  event: string;
  level?: string;
  message?: string;
  ts?: string;
  tool?: string;
  elapsed_s?: number;
  [key: string]: unknown;
};

export type ChatResponse = {
  run_id: string;
  model: string;
  response_text: string;
  duration_seconds: number;
  progress_events?: ProgressEvent[];
  benchmark?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
};

export type RunResultResponse = {
  run_id: string;
  status: "pending" | "running" | "succeeded" | "failed" | "canceled";
  result?: ChatResponse | null;
};

export type AnalysisRunSummary = {
  run_id: string;
  conversation_id?: string | null;
  goal: string;
  status: "pending" | "running" | "succeeded" | "failed" | "canceled";
  created_at: string;
  updated_at: string;
  error?: string | null;
  tools: string[];
  file_names: string[];
  duration_seconds?: number | null;
};

export type AnalysisHistoryResponse = {
  count: number;
  analyses: AnalysisRunSummary[];
};

export type ContractAuditRequest = {
  run_ids?: string[];
  limit?: number;
};

export type ContractAuditRecord = {
  run_id: string;
  status: string;
  passed: boolean;
  checks: Record<string, boolean>;
  missing_fields: string[];
  evidence_count: number;
  measurement_count: number;
  limitation_count: number;
  next_step_count: number;
  confidence_level?: string | null;
  confidence_why_count: number;
  research_score: number;
  research_max_score: number;
  research_summary: string;
  recommendations: string[];
};

export type ContractAuditResponse = {
  count: number;
  passed: number;
  failed: number;
  average_research_score: number;
  records: ContractAuditRecord[];
};

export type ArtifactRecord = {
  path: string;
  size_bytes: number;
  mime_type?: string | null;
  modified_at: string;
  source_path?: string | null;
  title?: string | null;
};

export type ArtifactListResponse = {
  run_id: string;
  root: string;
  artifact_count: number;
  artifacts: ArtifactRecord[];
};

export type RunEvent = {
  event_type: string;
  level?: string;
  payload?: Record<string, unknown>;
  ts?: string;
};

export type RunEventsResponse = {
  run_id: string;
  events: RunEvent[];
};

export type RunResponse = {
  run_id: string;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
  error?: string | null;
  workflow_kind: string;
  mode: string;
  parent_run_id?: string | null;
  planner_version?: string | null;
  agent_role?: string | null;
  checkpoint_state?: Record<string, unknown> | null;
  budget_state?: Record<string, unknown> | null;
  trace_group_id?: string | null;
};

export type TrainingDatasetSplit = "train" | "val" | "test";
export type TrainingDatasetRole = "image" | "mask" | "annotation";
export type TrainingJobType = "training" | "inference";
export type TrainingJobStatus =
  | "queued"
  | "running"
  | "paused"
  | "succeeded"
  | "failed"
  | "canceled";
export type ModelHealthStatus =
  | "Healthy"
  | "Watch"
  | "Retrain Recommended"
  | "Needs Human Review";

export type TrainingModelRecord = {
  key: string;
  name: string;
  framework: string;
  task_type: string;
  description: string;
  supports_training: boolean;
  supports_finetune: boolean;
  supports_inference: boolean;
  dimensions: string[];
  default_config: Record<string, unknown>;
};

export type TrainingModelsResponse = {
  count: number;
  models: TrainingModelRecord[];
};

export type TrainingDatasetCreateRequest = {
  name: string;
  description?: string | null;
  metadata?: Record<string, unknown>;
};

export type TrainingDatasetItemAssignment = {
  file_id: string;
  split: TrainingDatasetSplit;
  role: TrainingDatasetRole;
  sample_id?: string | null;
  metadata?: Record<string, unknown>;
};

export type TrainingDatasetItemsRequest = {
  items: TrainingDatasetItemAssignment[];
  replace?: boolean;
};

export type TrainingDatasetRecord = {
  dataset_id: string;
  user_id: string;
  name: string;
  description?: string | null;
  item_count: number;
  split_counts: Record<string, number>;
  created_at: string;
  updated_at: string;
};

export type TrainingDatasetResponse = {
  dataset: TrainingDatasetRecord;
  manifest: Record<string, unknown>;
};

export type TrainingDatasetListResponse = {
  count: number;
  datasets: TrainingDatasetRecord[];
};

export type TrainingJobCreateRequest = {
  dataset_id: string;
  model_key: string;
  config?: Record<string, unknown>;
  confirm_launch?: boolean;
  initial_checkpoint_path?: string | null;
};

export type TrainingPreflightRequest = {
  dataset_id: string;
  model_key: string;
  config?: Record<string, unknown>;
};

export type TrainingPreflightResponse = {
  dataset_id: string;
  model_key: string;
  config: Record<string, unknown>;
  recommended_launch: boolean;
  report: Record<string, unknown>;
};

export type TrainingJobControlRequest = {
  action: "pause" | "resume" | "cancel" | "restart";
};

export type InferenceJobCreateRequest = {
  model_key: string;
  model_version?: string | null;
  file_ids: string[];
  config?: Record<string, unknown>;
  reviewed_samples?: number;
  reviewed_failures?: number;
  confirm_launch?: boolean;
};

export type TrainingJobRecord = {
  job_id: string;
  user_id: string;
  job_type: TrainingJobType;
  dataset_id?: string | null;
  model_key: string;
  model_version?: string | null;
  status: TrainingJobStatus;
  artifact_run_id?: string | null;
  error?: string | null;
  request: Record<string, unknown>;
  result: Record<string, unknown>;
  control: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  last_heartbeat_at?: string | null;
};

export type TrainingJobResponse = {
  job: TrainingJobRecord;
};

export type ModelHealthRecord = {
  model_key: string;
  model_version: string;
  status: ModelHealthStatus;
  recommendation: string;
  rationale: string[];
  metrics: Record<string, unknown>;
  training_runs: number;
  inference_runs: number;
  user_id?: string | null;
};

export type ModelHealthResponse = {
  count: number;
  models: ModelHealthRecord[];
};

export type TrainingDomainOwnerScope = "shared" | "private";
export type TrainingLineageScope = "shared" | "fork";
export type TrainingVersionStatus = "candidate" | "canary" | "active" | "retired";
export type TrainingProposalStatus =
  | "pending_approval"
  | "approved"
  | "running"
  | "evaluating"
  | "ready_to_promote"
  | "promoted"
  | "rejected"
  | "failed";
export type TrainingMergeStatus =
  | "open"
  | "evaluating"
  | "approved"
  | "rejected"
  | "executed"
  | "failed";
export type TrainingTriggerReason =
  | "data_threshold"
  | "schedule"
  | "health"
  | "manual"
  | "merge"
  | "none";

export type TrainingDomainCreateRequest = {
  name: string;
  description?: string | null;
  owner_scope?: TrainingDomainOwnerScope;
  metadata?: Record<string, unknown>;
};

export type TrainingDomainRecord = {
  domain_id: string;
  name: string;
  description?: string | null;
  owner_scope: TrainingDomainOwnerScope;
  owner_user_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type TrainingDomainListResponse = {
  count: number;
  domains: TrainingDomainRecord[];
};

export type TrainingLineageRecord = {
  lineage_id: string;
  domain_id: string;
  scope: TrainingLineageScope;
  owner_user_id: string;
  model_key: string;
  parent_lineage_id?: string | null;
  active_version_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type TrainingLineageListResponse = {
  count: number;
  lineages: TrainingLineageRecord[];
};

export type TrainingForkLineageRequest = {
  model_key?: string | null;
  metadata?: Record<string, unknown>;
};

export type TrainingModelVersionRecord = {
  version_id: string;
  lineage_id: string;
  source_job_id?: string | null;
  artifact_run_id?: string | null;
  status: TrainingVersionStatus;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type TrainingModelVersionListResponse = {
  count: number;
  versions: TrainingModelVersionRecord[];
};

export type TrainingUpdateProposalRecord = {
  proposal_id: string;
  lineage_id: string;
  trigger_reason: TrainingTriggerReason | string;
  trigger_snapshot: Record<string, unknown>;
  dataset_snapshot: Record<string, unknown>;
  config: Record<string, unknown>;
  status: TrainingProposalStatus | string;
  idempotency_key?: string | null;
  approved_by?: string | null;
  rejected_by?: string | null;
  linked_job_id?: string | null;
  candidate_version_id?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  approved_at?: string | null;
  rejected_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

export type TrainingUpdateProposalListResponse = {
  count: number;
  proposals: TrainingUpdateProposalRecord[];
};

export type TrainingUpdateProposalPreviewRequest = {
  lineage_id: string;
  dataset_id: string;
  approved_new_samples?: number | null;
  class_counts?: Record<string, number>;
  health_status?: ModelHealthStatus | null;
  config?: Record<string, unknown>;
  trigger_reason_override?: TrainingTriggerReason | null;
  idempotency_key?: string | null;
  persist?: boolean;
};

export type TrainingUpdateProposalPreviewResponse = {
  trigger: Record<string, unknown>;
  preview: Record<string, unknown>;
  proposal?: TrainingUpdateProposalRecord | null;
};

export type TrainingUpdateProposalDecisionRequest = {
  note?: string | null;
  confirm_launch?: boolean;
};

export type TrainingUpdateProposalResponse = {
  proposal: TrainingUpdateProposalRecord;
};

export type TrainingVersionPromoteRequest = {
  note?: string | null;
};

export type TrainingVersionRollbackRequest = {
  target_version_id?: string | null;
  note?: string | null;
};

export type TrainingModelVersionResponse = {
  version: TrainingModelVersionRecord;
  lineage: TrainingLineageRecord;
};

export type TrainingMergeRequestCreateRequest = {
  source_lineage_id: string;
  target_lineage_id: string;
  candidate_version_id: string;
  notes?: string | null;
};

export type TrainingMergeRequestDecisionRequest = {
  notes?: string | null;
};

export type TrainingMergeRequestRecord = {
  merge_id: string;
  source_lineage_id: string;
  target_lineage_id: string;
  candidate_version_id: string;
  requested_by: string;
  status: TrainingMergeStatus | string;
  decision_by?: string | null;
  notes?: string | null;
  evaluation: Record<string, unknown>;
  linked_proposal_id?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  decided_at?: string | null;
  executed_at?: string | null;
};

export type TrainingMergeRequestResponse = {
  merge_request: TrainingMergeRequestRecord;
};

export type TrainingMergeRequestListResponse = {
  count: number;
  merge_requests: TrainingMergeRequestRecord[];
};

export type PrairieSyncResponse = {
  success: boolean;
  dataset_name: string;
  dataset_id?: string | null;
  bisque_dataset_uri?: string | null;
  synced_images: number;
  reviewed_images: number;
  unreviewed_images: number;
  class_counts: Record<string, number>;
  unsupported_class_counts: Record<string, number>;
  last_sync_at?: string | null;
  errors: string[];
};

export type PrairieStatusResponse = {
  dataset_name: string;
  dataset_id?: string | null;
  last_sync_at?: string | null;
  next_sync_at?: string | null;
  active_model_version?: string | null;
  model_health: ModelHealthStatus | string;
  reviewed_images: number;
  unreviewed_images: number;
  class_counts: Record<string, number>;
  unsupported_class_counts: Record<string, number>;
  detection_counts: Record<string, number>;
  latest_metrics: Record<string, unknown>;
  benchmark_baseline: Record<string, unknown>;
  benchmark_latest_candidate: Record<string, unknown>;
  last_benchmark_at?: string | null;
  benchmark_ready: boolean;
  canonical_benchmark_ready: boolean;
  promotion_benchmark_ready: boolean;
  retrain_gate: boolean;
  retrain_gate_reasons: string[];
  retrain_gate_counts: Record<string, number>;
};

export type PrairieRetrainRequest = {
  confirm_launch?: boolean;
  note?: string | null;
};

export type PrairieRetrainRecord = {
  request_id: string;
  training_job_id: string;
  status: TrainingJobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  model_version?: string | null;
  note?: string | null;
  error?: string | null;
  gating_summary: Record<string, unknown>;
  benchmark_report_artifact_id?: string | null;
};

export type PrairieRetrainListResponse = {
  count: number;
  requests: PrairieRetrainRecord[];
};

export type PrairieBenchmarkRunResponse = {
  run_id: string;
  model_version?: string | null;
  mode: "canonical_only" | "promotion_packet";
  benchmark_ready: boolean;
  canonical_benchmark_ready: boolean;
  promotion_benchmark_ready: boolean;
  report: Record<string, unknown>;
};

export type PrairieBenchmarkRunRequest = {
  mode?: "canonical_only" | "promotion_packet";
};

export type ReproReportRequest = {
  run_id?: string | null;
  title?: string | null;
  result_summary?: string | null;
  measurements?: Array<Record<string, unknown>>;
  statistical_analysis?: Array<Record<string, unknown>> | Record<string, unknown> | null;
  qc_warnings?: string[];
  limitations?: string[];
  provenance?: Record<string, unknown>;
  next_steps?: Array<Record<string, unknown> | string>;
  output_dir?: string | null;
};

export type ReproReportResponse = {
  success: boolean;
  run_id?: string | null;
  report_markdown_path: string;
  report_json_path: string;
  report_sha256: string;
  report_bundle_sha256: string;
};

export type UploadedFileRecord = {
  file_id: string;
  original_name: string;
  content_type?: string | null;
  size_bytes: number;
  sha256: string;
  created_at: string;
  sync_status?: string | null;
  canonical_resource_uniq?: string | null;
  canonical_resource_uri?: string | null;
  client_view_url?: string | null;
  image_service_url?: string | null;
  sync_error?: string | null;
  sync_run_id?: string | null;
};

export type UploadFilesResponse = {
  file_count: number;
  uploaded: UploadedFileRecord[];
};

export type ResourceRecord = {
  file_id: string;
  original_name: string;
  content_type?: string | null;
  size_bytes: number;
  sha256: string;
  created_at: string;
  source_type: "upload" | "bisque_import" | string;
  resource_kind: "image" | "video" | "table" | "file" | string;
  source_uri?: string | null;
  client_view_url?: string | null;
  image_service_url?: string | null;
  has_thumbnail: boolean;
  thumbnail_url?: string | null;
  preview_url?: string | null;
  sync_status?: string | null;
  sync_error?: string | null;
  canonical_resource_uniq?: string | null;
  canonical_resource_uri?: string | null;
  cache_ready?: boolean;
  staged_locally?: boolean;
  sync_run_id?: string | null;
};

export type ResourceListResponse = {
  count: number;
  resources: ResourceRecord[];
};

export type ResourceComputationLookupRequest = {
  file_ids: string[];
  tool_names?: string[];
  prompt?: string | null;
  limit_per_file_tool?: number;
};

export type ResourceComputationSuggestion = {
  requested_file_id: string;
  requested_file_name: string;
  requested_file_sha256: string;
  tool_name: "segment_image_sam3" | "yolo_detect" | "estimate_depth_pro" | string;
  run_id: string;
  run_status: string;
  run_goal?: string | null;
  run_updated_at: string;
  conversation_id?: string | null;
  conversation_title?: string | null;
  conversation_updated_at?: string | null;
  match_type: "sha256" | "filename";
};

export type ResourceComputationLookupResponse = {
  count: number;
  suggestions: ResourceComputationSuggestion[];
};

export type ResumableUploadInitRequest = {
  file_name: string;
  size_bytes: number;
  content_type?: string | null;
  fingerprint: string;
  chunk_size_bytes?: number;
};

export type ResumableUploadSessionResponse = {
  upload_id: string;
  file_name: string;
  size_bytes: number;
  content_type?: string | null;
  chunk_size_bytes: number;
  bytes_received: number;
  status: "active" | "completed" | "failed";
  uploaded?: UploadedFileRecord | null;
  error?: string | null;
};

export type ResumableUploadChunkResponse = {
  upload_id: string;
  bytes_received: number;
  size_bytes: number;
  complete: boolean;
  status: "active" | "completed" | "failed";
};

export type ResumableUploadCompleteResponse = {
  upload_id: string;
  uploaded: UploadedFileRecord;
};

export type ConversationRecord = {
  conversation_id: string;
  title: string;
  created_at_ms: number;
  updated_at_ms: number;
  preview: string;
  message_count: number;
  preferred_panel: "chat";
  running: boolean;
  state: Record<string, unknown>;
};

export type ConversationListResponse = {
  count: number;
  total_count: number;
  limit: number;
  offset: number;
  has_more: boolean;
  conversations: ConversationRecord[];
};

export type ConversationSearchResponse = {
  query: string;
  count: number;
  matches: Array<Record<string, unknown>>;
};

export type AdminPlatformKpis = {
  total_users: number;
  active_users_24h: number;
  total_conversations: number;
  conversations_started_24h: number;
  total_messages: number;
  messages_last_24h: number;
  user_messages_last_24h: number;
  assistant_messages_last_24h: number;
  total_runs: number;
  runs_last_24h: number;
  success_rate_last_24h: number;
  running_runs: number;
  failed_runs_24h: number;
  total_uploads: number;
  soft_deleted_uploads: number;
  total_storage_bytes: number;
  avg_messages_per_conversation: number;
};

export type AdminUsageBucket = {
  bucket_start: string;
  runs_total: number;
  runs_succeeded: number;
  runs_failed: number;
  uploads: number;
  new_users: number;
};

export type AdminToolUsageRecord = {
  tool_name: string;
  count: number;
  succeeded: number;
  failed: number;
};

export type AdminUserSummary = {
  user_id: string;
  conversations: number;
  messages: number;
  runs_total: number;
  runs_running: number;
  runs_failed: number;
  runs_succeeded: number;
  uploads: number;
  storage_bytes: number;
  last_activity_at?: string | null;
};

export type AdminRunRecord = {
  run_id: string;
  user_id?: string | null;
  conversation_id?: string | null;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
  error?: string | null;
  duration_seconds?: number | null;
  tool_names: string[];
};

export type AdminIssueRecord = {
  issue_type: "failed_run" | "failed_upload_session" | "stalled_run";
  severity: "high" | "medium" | "low";
  user_id?: string | null;
  run_id?: string | null;
  upload_id?: string | null;
  conversation_id?: string | null;
  message: string;
  occurred_at: string;
  metadata: Record<string, unknown>;
};

export type AdminOverviewResponse = {
  generated_at: string;
  kpis: AdminPlatformKpis;
  usage_last_24h: AdminUsageBucket[];
  tool_usage_7d: AdminToolUsageRecord[];
  top_users: AdminUserSummary[];
  recent_issues: AdminIssueRecord[];
};

export type AdminUserListResponse = {
  count: number;
  users: AdminUserSummary[];
};

export type AdminRunListResponse = {
  count: number;
  runs: AdminRunRecord[];
};

export type AdminIssueListResponse = {
  count: number;
  issues: AdminIssueRecord[];
};

export type AdminRunActionResponse = {
  run_id: string;
  previous_status: string;
  status: string;
  updated: boolean;
};

export type AdminConversationActionResponse = {
  conversation_id: string;
  user_id: string;
  deleted: boolean;
};

export type SantaBarbaraWeatherResponse = {
  success: boolean;
  location: string;
  micro_location?: string | null;
  observed_at?: string | null;
  temperature_f?: number | null;
  apparent_temperature_f?: number | null;
  weather_code?: number | null;
  weather_label?: string | null;
  wind_speed_mph?: number | null;
  daily_high_f?: number | null;
  daily_low_f?: number | null;
  precipitation_probability_percent?: number | null;
  wave_height_ft?: number | null;
  swell_wave_height_ft?: number | null;
  wave_period_seconds?: number | null;
  blip?: string | null;
  summary: string;
  source: string;
};

export type BisqueImportItem = {
  input_url: string;
  resource_uri?: string | null;
  resource_uniq?: string | null;
  client_view_url?: string | null;
  image_service_url?: string | null;
  status: "imported" | "reused" | "error";
  download_source?: "image_service" | "resource_blob" | "resource_uri" | "bqapi_blob" | string | null;
  error?: string | null;
  uploaded?: UploadedFileRecord | null;
};

export type BisqueImportResponse = {
  file_count: number;
  uploaded: UploadedFileRecord[];
  imports: BisqueImportItem[];
};

export type Hdf5ViewerTreeNode = {
  path: string;
  name: string;
  node_type: "group" | "dataset" | string;
  child_count: number;
  attributes_count: number;
  shape?: number[] | null;
  dtype?: string | null;
  preview_kind?: string | null;
  children: Hdf5ViewerTreeNode[];
};

export type Hdf5DatasetField = {
  name: string;
  dtype: string;
};

export type Hdf5MaterialsPayload = {
  detected: boolean;
  schema?: "dream3d" | null;
  capabilities: string[];
  roles: Record<string, string>;
  phase_names: string[];
  feature_count?: number | null;
  grain_count?: number | null;
  recommended_view: "materials" | "explorer";
};

export type Hdf5DatasetSummary = {
  file_id: string;
  dataset_path: string;
  dataset_name: string;
  preview_kind?: string | null;
  semantic_role?: string | null;
  units_hint?: string | null;
  materials_domain_tags: string[];
  dtype: string;
  shape: number[];
  rank: number;
  element_count: number;
  estimated_bytes?: number | null;
  dimension_summary?: Record<string, number> | null;
  capabilities: string[];
  render_policy: "scalar" | "categorical" | "display" | "analysis" | string;
  delivery_mode: "direct" | "scalar" | "atlas" | "deferred_multiscale" | string;
  diagnostic_surface: "mpr" | "none" | string;
  first_paint_mode: "image" | "webgl" | string;
  measurement_policy: "pixel-only" | "spacing-aware" | "orientation-aware" | string;
  texture_policy: "linear" | "nearest" | string;
  display_capabilities: string[];
  viewer_capabilities: string[];
  volume_eligible: boolean;
  volume_reason?: string | null;
  axis_sizes?: {
    T: number;
    C: number;
    Z: number;
    Y: number;
    X: number;
  } | null;
  physical_spacing?: {
    x?: number | null;
    y?: number | null;
    z?: number | null;
  } | null;
  atlas_scheme?: {
    slice_count: number;
    columns: number;
    rows: number;
    slice_width: number;
    slice_height: number;
    atlas_width: number;
    atlas_height: number;
    downsample: number;
    format: "png" | string;
  } | null;
  attributes: Record<string, unknown>;
  geometry?: {
    path?: string | null;
    dimensions?: number[] | null;
    spacing?: number[] | null;
    origin?: number[] | null;
  } | null;
  structured_fields: Hdf5DatasetField[];
  component_count: number;
  component_labels: string[];
  slice_axes: Array<"z" | "y" | "x">;
  preview_planes: Record<
    string,
    {
      axis: "z" | "y" | "x";
      label: string;
      axes: string[];
      pixel_size: {
        width: number;
        height: number;
      };
      spacing: {
        row: number;
        col: number;
      };
      world_size: {
        width: number;
        height: number;
      };
      aspect_ratio: number;
    }
  >;
  sample_shape: number[];
  sample_values?: unknown;
  sample_statistics?: {
    sample_count: number;
    min?: number | null;
    max?: number | null;
    mean?: number | null;
    unique_values?: number | null;
  } | null;
};

export type Hdf5DatasetHistogramResponse = {
  file_id: string;
  dataset_path: string;
  preview_kind?: string | null;
  component_index?: number | null;
  component_label?: string | null;
  sample_count: number;
  discrete: boolean;
  min?: number | null;
  max?: number | null;
  bins: Array<{
    label: string;
    start?: number | null;
    end?: number | null;
    count: number;
  }>;
};

export type Hdf5DatasetTablePreviewResponse = {
  file_id: string;
  dataset_path: string;
  preview_kind?: string | null;
  offset: number;
  limit: number;
  total_rows: number;
  total_columns: number;
  columns: Array<{
    key: string;
    label: string;
    dtype: string;
    numeric: boolean;
  }>;
  rows: Array<Record<string, unknown>>;
  charts: Array<{
    kind: "scatter" | "histogram";
    title: string;
    description?: string | null;
    x_key: string;
    y_key: string;
    data: Array<Record<string, unknown>>;
  }>;
};

export type Hdf5MaterialsChartResponse = {
  kind: "scatter" | "histogram" | "bar";
  title: string;
  description?: string | null;
  x_key: string;
  y_key: string;
  data: Array<Record<string, unknown>>;
  source_paths: string[];
  units_hint?: string | null;
  provenance?: string | null;
};

export type Hdf5MaterialsMapResponse = {
  title: string;
  description?: string | null;
  dataset_path: string;
  semantic_role: string;
  preview_kind?: string | null;
};

export type Hdf5MaterialsDatasetLinkResponse = {
  label: string;
  dataset_path: string;
  semantic_role: string;
  group: string;
};

export type Hdf5MaterialsDashboardResponse = {
  file_id: string;
  schema: "dream3d";
  overview: {
    geometry?: {
      path?: string | null;
      dimensions?: number[] | null;
      spacing?: number[] | null;
      origin?: number[] | null;
    } | null;
    spacing_note?: string | null;
    phase_names: string[];
    feature_count?: number | null;
    grain_count?: number | null;
    capabilities: string[];
    recommended_map_dataset_path?: string | null;
  };
  maps: Hdf5MaterialsMapResponse[];
  grain_charts: Hdf5MaterialsChartResponse[];
  orientation_charts: Hdf5MaterialsChartResponse[];
  synthetic_stats: Hdf5MaterialsChartResponse[];
  dataset_links: Hdf5MaterialsDatasetLinkResponse[];
};

export type UploadViewerInfo = {
  kind?: "image" | "hdf5" | string;
  file_id: string;
  original_name: string;
  modality?: "microscopy" | "medical" | "geospatial" | "materials" | "image" | "unknown" | string;
  dims_order: string;
  backend_mode?: "direct" | "pyramid" | "atlas" | "scalar" | "hdf5" | string;
  axis_sizes: {
    T: number;
    C: number;
    Z: number;
    Y: number;
    X: number;
  };
  selected_indices: {
    T: number;
    C: number;
    Z: number;
  };
  is_volume: boolean;
  is_timeseries: boolean;
  is_multichannel: boolean;
  phys?: {
    resource_uniq?: string;
    name?: string;
    x?: number;
    y?: number;
    z?: number;
    t?: number;
    ch?: number;
    pixel_depth?: number;
    pixel_format?: "u" | "s" | "f" | string;
    pixel_size?: number[];
    pixel_units?: string[];
    channel_names?: string[];
    display_channels?: number[];
    channel_colors?: Array<{
      index: number;
      hex: string;
      rgb: number[];
    }>;
    units?: string;
    dicom?: {
      modality?: string | null;
      wnd_center?: number | null;
      wnd_width?: number | null;
    } | null;
    geo?: Record<string, unknown> | null;
    coordinates?: Record<string, unknown> | null;
  };
  display_defaults?: {
    enhancement: string;
    negative: boolean;
    rotate: number;
    fusion_method: "m" | "a" | string;
    channel_mode: "composite" | "single" | string;
    channels: number[];
    channel_colors: string[];
    time_index: number;
    z_index: number;
    volume_channel?: number | null;
    volume_clip_min?: { x: number; y: number; z: number };
    volume_clip_max?: { x: number; y: number; z: number };
  };
  service_urls?: {
    preview?: string;
    display?: string;
    slice?: string;
    tile?: string;
    atlas?: string;
    scalar_volume?: string;
    histogram?: string;
    dataset?: string;
    table?: string;
  };
  metadata: {
    reader: string;
    dims_order: string;
    array_shape: number[];
    array_dtype: string;
    array_min?: number;
    array_max?: number;
    intensity_stats?: {
      min: number;
      max: number;
    };
    physical_spacing?: {
      z?: number | null;
      y?: number | null;
      x?: number | null;
    } | null;
    scene?: string | null;
    scene_count: number;
    header?: Record<string, string>;
    filename_hints?: Record<string, unknown>;
    exif?: Record<string, string>;
    geo?: Record<string, unknown> | null;
    dicom?: {
      modality?: string | null;
      wnd_center?: number | null;
      wnd_width?: number | null;
    } | null;
    microscopy?: {
      channel_names?: string[];
      dimensions_present?: string;
      objective?: string;
      imaging_datetime?: string;
      binning?: string;
      position_index?: number | string | null;
      row?: number | string | null;
      column?: number | string | null;
      timelapse_interval?: number | string | null;
      total_time_duration?: number | string | null;
      current_scene?: string;
      scene_names?: string[];
    } | null;
    warnings: string[];
  };
  viewer: {
    status: "ready" | "preview-ready" | "warming" | "degraded-fallback" | string;
    warmup_mode: "lazy" | "hybrid" | "precomputed" | string;
    backend_mode?: "direct" | "pyramid" | "atlas" | "scalar" | string;
    default_surface: "2d" | "mpr" | "volume" | "metadata" | string;
    available_surfaces: string[];
    default_axis: "z" | "y" | "x";
    slice_axes: Array<"z" | "y" | "x">;
    channel_mode: "composite" | "single" | string;
    tile_scheme: {
      tile_size: number;
      format: "png" | string;
      levels: Array<{
        level: number;
        width: number;
        height: number;
        columns: number;
        rows: number;
        downsample: number;
      }>;
    };
    atlas_scheme?: {
      slice_count: number;
      columns: number;
      rows: number;
      slice_width: number;
      slice_height: number;
      atlas_width: number;
      atlas_height: number;
      downsample: number;
      format: "png" | string;
    };
    default_plane: {
      axis: "z" | "y" | "x";
      label: string;
      axes: string[];
      pixel_size: {
        width: number;
        height: number;
      };
      spacing: {
        row: number;
        col: number;
      };
      world_size: {
        width: number;
        height: number;
      };
      aspect_ratio: number;
    };
    planes: Record<
      string,
      {
        axis: "z" | "y" | "x";
        label: string;
        axes: string[];
        pixel_size: {
          width: number;
          height: number;
        };
        spacing: {
          row: number;
          col: number;
        };
        world_size: {
          width: number;
          height: number;
        };
        aspect_ratio: number;
      }
    >;
    volume_mode: "none" | "slice_stack" | "atlas" | "scalar" | string;
    render_policy?: "scalar" | "categorical" | "display" | "analysis" | string;
    delivery_mode?: "direct" | "scalar" | "atlas" | "deferred_multiscale" | string;
    diagnostic_surface?: "mpr" | "none" | string;
    first_paint_mode?: "image" | "webgl" | string;
    measurement_policy?: "pixel-only" | "spacing-aware" | "orientation-aware" | string;
    texture_policy?: "linear" | "nearest" | string;
    display_capabilities?: string[];
    viewer_capabilities?: string[];
    orientation?: {
      frame: "pixel" | "voxel" | "patient" | "geospatial" | string;
      row_axis: string;
      col_axis: string;
      slice_axis?: string | null;
      axis_labels?: {
        x?: { positive?: string | null; negative?: string | null };
        y?: { positive?: string | null; negative?: string | null };
        z?: { positive?: string | null; negative?: string | null };
      };
      labels?: {
        top?: string | null;
        bottom?: string | null;
        left?: string | null;
        right?: string | null;
        front?: string | null;
        back?: string | null;
      };
    };
    asset_preparation?: {
      status: "ready" | "preview-ready" | "warming" | "degraded-fallback" | string;
      native_supported: boolean;
      tile_pyramid: "lazy" | "hybrid" | "precomputed" | "none" | string;
      volume_representation: "none" | "slice_stack" | "chunks" | "atlas" | "scalar" | string;
    };
    chunk_scheme?: {
      mode: "none" | "slice_stack" | "bricks" | "atlas" | "scalar" | string;
      axis?: "z" | "y" | "x";
      sample_count?: number;
    };
    display_defaults?: {
      enhancement: string;
      negative: boolean;
      rotate: number;
      fusion_method: "m" | "a" | string;
      channel_mode: "composite" | "single" | string;
      channels: number[];
      channel_colors: string[];
      time_index: number;
      z_index: number;
      volume_channel?: number | null;
    };
    service_urls?: {
      preview?: string;
      display?: string;
      slice?: string;
      tile?: string;
      atlas?: string;
      scalar_volume?: string;
      histogram?: string;
      dataset?: string;
      table?: string;
    };
    fallback_urls?: {
      preview?: string;
      slice?: string;
    };
  };
  hdf5?: {
    enabled: boolean;
    supported: boolean;
    status: "ready" | "disabled" | "unsupported" | string;
    error?: string | null;
    root_keys: string[];
    root_attributes: Record<string, unknown>;
    summary: {
      group_count: number;
      dataset_count: number;
      dataset_kinds: Record<string, number>;
      truncated: boolean;
      geometry?: {
        path?: string | null;
        dimensions?: number[] | null;
        spacing?: number[] | null;
        origin?: number[] | null;
      } | null;
    };
    tree: Hdf5ViewerTreeNode[];
    limitations: string[];
    selected_dataset_path?: string | null;
    default_dataset_path?: string | null;
    materials?: Hdf5MaterialsPayload | null;
  } | null;
};

export type UploadViewerHistogramResponse = {
  file_id: string;
  bins: number;
  histogram: {
    bins: number[];
    edges: number[];
    min: number;
    max: number;
    channel_indices: number[];
    time_index: number;
  };
};

export type UploadCaptionResponse = {
  file_id: string;
  caption: string;
  source: "llm" | "fallback" | "cache";
};

export type Sam3PointAnnotation = {
  x: number;
  y: number;
  label: 0 | 1 | "positive" | "negative" | "include" | "exclude";
};

export type Sam3BoxAnnotation = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: 0 | 1 | "positive" | "negative" | "include" | "exclude";
};

export type Sam3ImageAnnotation = {
  file_id: string;
  points: Sam3PointAnnotation[];
  boxes: Sam3BoxAnnotation[];
};

export type Sam3InteractiveRequest = {
  file_ids: string[];
  annotations: Sam3ImageAnnotation[];
  model?: "sam3" | "medsam" | "medsam2" | string | null;
  conversation_id?: string | null;
  concept_prompt?: string | null;
  save_visualizations?: boolean;
  preset?: "fast" | "balanced" | "high_quality" | string | null;
  threshold?: number | null;
  point_box_size?: number;
  min_points?: number | null;
  max_points?: number | null;
  tracker_prompt_mode?:
    | "single_object_refine"
    | "per_positive_point_instance";
  force_rerun?: boolean;
};

export type Sam3InteractiveResponse = {
  success: boolean;
  run_id: string;
  response_text: string;
  progress_events: ProgressEvent[];
  result: {
    processed: number;
    total_files: number;
    total_masks_generated: number;
    files_processed: Array<Record<string, unknown>>;
    preferred_upload_paths: string[];
    visualization_paths: Array<Record<string, unknown>>;
    output_directories: string[];
    coverage_percent_mean?: number | null;
    coverage_percent_min?: number | null;
    coverage_percent_max?: number | null;
    model?: string | null;
    annotations: Array<Record<string, unknown>>;
    run_id: string;
  };
  warnings: string[];
};

export type PublicConfigResponse = {
  bisque_root?: string | null;
  bisque_browser_url?: string | null;
  bisque_auth_enabled?: boolean;
  bisque_auth_mode?: "local" | "oidc" | "dual";
  bisque_oidc_enabled?: boolean;
  bisque_guest_enabled?: boolean;
  admin_enabled?: boolean;
  bisque_urls?: {
    home?: string | null;
    images?: string | null;
    datasets?: string | null;
    tables?: string | null;
  } | null;
};

export type BisqueAuthSessionResponse = {
  authenticated: boolean;
  username?: string | null;
  bisque_root?: string | null;
  expires_at?: string | null;
  mode?: "bisque" | "guest" | null;
  guest_profile?: {
    name: string;
    email: string;
    affiliation: string;
  } | null;
  is_admin?: boolean;
};

export type BisqueAuthLoginRequest = {
  username: string;
  password: string;
};

export type BisqueGuestAuthRequest = {
  name: string;
  email: string;
  affiliation: string;
};

export type V3AttachmentRecord = {
  kind: "file_id" | "resource_uri" | "dataset_uri";
  value: string;
  name?: string | null;
  metadata?: Record<string, unknown>;
};

export type V3SessionCreateRequest = {
  title?: string | null;
  status?: string;
  summary?: string | null;
  memory_policy?: V3MemoryPolicy;
  knowledge_scope?: V3KnowledgeScope;
  metadata?: Record<string, unknown>;
};

export type V3SessionRecord = {
  session_id: string;
  user_id?: string | null;
  title: string;
  status: string;
  summary?: string | null;
  memory_policy: V3MemoryPolicy;
  knowledge_scope: V3KnowledgeScope;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type V3SessionListResponse = {
  count: number;
  sessions: V3SessionRecord[];
};

export type V3MessageInput = {
  role: ChatRole;
  content: string;
  attachments?: V3AttachmentRecord[];
  metadata?: Record<string, unknown>;
};

export type V3MessageRecord = {
  message_id: string;
  session_id: string;
  role: ChatRole;
  content: string;
  attachments: V3AttachmentRecord[];
  run_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type V3SessionMessageListResponse = {
  session_id: string;
  count: number;
  messages: V3MessageRecord[];
};

export type V3RunCreateRequest = {
  goal?: string | null;
  messages: V3MessageInput[];
  file_ids?: string[];
  resource_uris?: string[];
  dataset_uris?: string[];
  selected_tool_names?: string[];
  knowledge_context?: KnowledgeContext | null;
  selection_context?: SelectionContext | null;
  workflow_hint?: ChatWorkflowHint | null;
  reasoning_mode?: "auto" | "fast" | "deep";
  debug?: boolean;
  budgets?: ToolBudget;
  metadata?: Record<string, unknown>;
};

export type V3RunRecord = {
  run_id: string;
  session_id: string;
  user_id?: string | null;
  workflow_name: string;
  status: string;
  current_step?: string | null;
  checkpoint_state: Record<string, unknown>;
  budget_state: Record<string, unknown>;
  response_text?: string | null;
  trace_group_id?: string | null;
  metrics: Record<string, unknown>;
  error?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type V3RunEventRecord = {
  event_id: string;
  run_id: string;
  session_id?: string | null;
  user_id?: string | null;
  event_kind: string;
  event_type: string;
  agent_name?: string | null;
  tool_name?: string | null;
  level?: string | null;
  payload: Record<string, unknown>;
  created_at: string;
};

export type V3RunEventListResponse = {
  run_id: string;
  count: number;
  events: V3RunEventRecord[];
};

export type V3ArtifactRecord = {
  artifact_id: string;
  run_id: string;
  session_id?: string | null;
  user_id?: string | null;
  kind: string;
  title?: string | null;
  path?: string | null;
  source_path?: string | null;
  preview_path?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type V3ArtifactListResponse = {
  run_id: string;
  count: number;
  artifacts: V3ArtifactRecord[];
};

export type V3ApprovalRecord = {
  approval_id: string;
  run_id: string;
  session_id?: string | null;
  user_id?: string | null;
  action_type: string;
  tool_name?: string | null;
  status: string;
  request_payload: Record<string, unknown>;
  resolution: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type V3ApprovalResolveRequest = {
  decision: "approve" | "reject";
  note?: string | null;
  metadata?: Record<string, unknown>;
};

export type V3ApprovalResolveResponse = {
  approval: V3ApprovalRecord;
  run: V3RunRecord;
};

export * from "./types-v2";
