export type ChatRole = "system" | "user" | "assistant" | "tool";

export type ChatMessage = {
  role: ChatRole;
  content: string;
  // Durable wire identity (present on server-persisted thread messages).
  message_id?: string;
  thread_id?: string | null;
  run_id?: string | null;
  created_at?: string;
  metadata?: Record<string, unknown> | null;
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
  // Keep the full forced-regime surface for evaluation and benchmark overrides.
  // Production routing auto-selects only the simplified default set.
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
  | "bisque_download_resource"
  | "upload_to_bisque"
  | "bisque_create_dataset"
  | "run_bisque_module"
  | "rarespot_ecology"
  | "detect_prairie_dog"
  | "pro_mode"
  | "goal_driven_build"
  | "quantitative_analysis"
  | "image_analysis"
  | "megaseg"
  | "knowledge";

export type ChatWorkflowHint = {
  id: ChatWorkflowHintId;
  source: "slash_menu";
};

export type KnowledgeContext = {
  collaborator_id?: string | null;
  project_id?: string | null;
  pack_ids?: string[];
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

export type RemoteMutationIntent = "bisque.upload" | "bisque.create_dataset";

export type ChatRequest = {
  messages: ChatMessage[];
  uploaded_files: string[];
  file_ids?: string[];
  resource_uris?: string[];
  dataset_uris?: string[];
  conversation_id?: string | null;
  goal?: string | null;
  selected_tool_names?: string[];
  remote_mutation_intents?: RemoteMutationIntent[];
  knowledge_context?: KnowledgeContext | null;
  selection_context?: SelectionContext | null;
  workflow_hint?: ChatWorkflowHint | null;
  reasoning_mode?: "auto" | "fast" | "deep";
  debug?: boolean;
  budgets?: ToolBudget | null;
  benchmark?: ChatBenchmarkConfig | null;
  idempotency_key?: string | null;
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

export type ArtifactRecord = {
  path: string;
  size_bytes: number;
  mime_type?: string | null;
  modified_at: string;
  source_path?: string | null;
  title?: string | null;
  result_group_id?: string | null;
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

export type RunTokenUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  model?: string;
};

export type CurrentUserProfile = {
  display_name?: string;
  title?: string;
  institution?: string;
  research_interests?: string;
  bio?: string;
};

export type CurrentUserResponse = {
  user: {
    user_id: string;
    email?: string;
    display_name?: string;
    role?: string;
    org_id?: string;
  };
  profile: CurrentUserProfile;
};

export type TokenUsageSummary = {
  lifetime_input_tokens: number;
  lifetime_output_tokens: number;
  lifetime_total_tokens: number;
  peak_daily_total: number;
  longest_task_seconds: number;
  current_streak_days: number;
  longest_streak_days: number;
  active_days: number;
  last_active_day?: string;
};

export type TokenUsageDailyPoint = {
  day: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  run_count: number;
};

export type TokenUsageResponse = {
  days: number;
  summary: TokenUsageSummary;
  daily: TokenUsageDailyPoint[];
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

export type TrainingDomainOwnerScope = "shared" | "private";
export type TrainingLineageScope = "shared" | "fork";
export type TrainingVersionStatus = "candidate" | "canary" | "active" | "retired" | "rejected";

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

export type TrainingModelVersionRecord = {
  version_id: string;
  lineage_id: string;
  source_job_id?: string | null;
  artifact_run_id?: string | null;
  status: TrainingVersionStatus;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  activated_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type TrainingModelVersionListResponse = {
  count: number;
  versions: TrainingModelVersionRecord[];
};

export type TrainingVersionPromoteRequest = {
  note?: string | null;
  // Required by the server when promoting canary -> active while the gold
  // set's held-out slice is pending (plan section 8.2); audited.
  override_reason?: string | null;
};

export type TrainingVersionRollbackRequest = {
  target_version_id?: string | null;
  note?: string | null;
};

export type TrainingModelVersionResponse = {
  version: TrainingModelVersionRecord;
  lineage: TrainingLineageRecord;
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
  // Target version: omit for the baseline (defaults to active); pass the
  // candidate's id when benchmarking a candidate, or the server benchmarks
  // the active version instead and the candidate never gets a verdict.
  version_id?: string;
};

// --- GoldGate training UI (M1.5) -------------------------------------------
// Status-read echoes (plan section 3.6/14.5). Each is individually optional:
// an older backend omits it and the UI degrades per the section-14.5 table.

export type TrainingGoldFreezeState = "blocked" | "ready" | "freezing" | "frozen" | "failed";

export type TrainingGoldEcho = {
  gold_set_id?: string | null;
  gold_set_version?: number | null;
  content_hash?: string | null;
  freeze_state?: TrainingGoldFreezeState;
  qualifying_prior_frames?: number;
  required_prior_frames?: number;
  freeze_failure_reasons?: string[];
  held_out_state?: "pending_new_survey" | "populated";
  per_slice_counts?: Record<string, number>;
};

export type TrainingCanaryEcho = {
  canary_version_id: string;
  soak_started_at: string;
  runs_observed: number;
  min_soak_runs: number;
  min_soak_hours: number;
  traffic_fraction: number;
  drift_note?: string | null;
};

export type TrainingRecentEvent = {
  ts: string;
  kind: string;
  version_id?: string;
  gold_hash_short?: string;
  report_uri?: string;
  summary: string;
};

export type TrainingRunningBenchmark = {
  version_id: string; // a version id, or the literal 'baseline'
  started_at: string;
};

// Live finetune progress echoed on the status read from the in-flight job's
// latest training.progress event (completed/total epochs + best mAP so far).
export type TrainingRunningFinetune = {
  job_id: string;
  job_type: string;
  status: string;
  started_at: string;
  message?: string;
  completed?: number;
  total?: number;
  map50?: number;
  map50_95?: number;
};

// The PINNED metadata.guardrails.clauses[] element shape (plan section 7.6,
// stamped by the M2 gate engine at benchmark time).
export type GateClauseWire = {
  clause_key: string;
  metric_path: string;
  slice?: string | null;
  comparator: "max_drop_vs_active" | "abs_floor" | "max_rise_vs_active" | "abs_ceiling";
  value: number;
  candidate_value?: number | null;
  baseline_value?: number | null;
  outcome: "passed" | "failed" | "excluded";
  reason?: string | null;
};

export type GateGuardrailsWire = {
  passed?: boolean;
  reasons?: string[];
  clauses?: GateClauseWire[];
  benchmarked_at?: string | null;
  gold_set_version?: number | null;
  gold_set_content_hash?: string | null;
  report_uri?: string | null;
};

// Normalized, model-agnostic status the rebuilt dashboard consumes; superset
// of the prairie wire shape with per-class counts as plain Records.
export type TrainingModelStatus = PrairieStatusResponse & {
  model_key?: string;
  per_class_new_objects?: Record<string, number>;
  retrain_gate_thresholds?: Record<string, unknown>;
  active_version_activated_at?: string | null;
  last_retrain_at?: string | null;
  gold?: TrainingGoldEcho | null;
  canary?: TrainingCanaryEcho | null;
  running_benchmark?: TrainingRunningBenchmark | null;
  running_finetune?: TrainingRunningFinetune | null;
  recent_events?: TrainingRecentEvent[];
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
  project_id?: string | null;
  client_view_url?: string | null;
  image_service_url?: string | null;
  sync_error?: string | null;
  sync_run_id?: string | null;
};

export type UploadFilesResponse = {
  file_count: number;
  uploaded: UploadedFileRecord[];
};

export type UploadSessionRecord = {
  session_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  project_id?: string | null;
  source_type: string;
  status: "active" | "paused" | "completed" | "failed" | "canceled" | string;
  total_bytes: number;
  bytes_received: number;
  bytes_verified: number;
  bytes_committed: number;
  idempotency_key?: string | null;
  browser_fingerprint?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type UploadSessionFileInit = {
  file_token: string;
  original_name: string;
  relative_path?: string | null;
  content_type?: string | null;
  size_bytes: number;
  declared_sha256?: string | null;
};

export type UploadSessionCreateRequest = {
  idempotency_key?: string | null;
  browser_fingerprint?: string | null;
  project_id?: string | null;
  total_bytes: number;
  files: UploadSessionFileInit[];
};

export type UploadSessionFileRecord = {
  session_id: string;
  file_token: string;
  resource_id?: string | null;
  original_name: string;
  relative_path?: string | null;
  content_type?: string | null;
  size_bytes: number;
  declared_sha256?: string | null;
  computed_sha256?: string | null;
  status: "pending" | "uploading" | "completed" | "failed" | string;
  error?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type UploadChunkRecord = {
  session_id: string;
  file_token: string;
  chunk_index: number;
  offset: number;
  size_bytes: number;
  sha256: string;
  status: "received" | "verified" | "failed" | string;
  storage_uri?: string | null;
  received_at?: string | null;
  verified_at?: string | null;
  error?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type UploadSessionLimits = {
  max_parallel_files: number;
  max_parallel_chunks: number;
  max_files_per_session: number;
};

export type UploadSessionEventRecord = {
  event_id: string;
  session_id: string;
  actor_user_id?: string | null;
  actor_org_id?: string | null;
  event_type: string;
  ts: string;
  metadata?: Record<string, unknown> | null;
};

export type UploadSessionResponse = {
  session: UploadSessionRecord;
  files: UploadSessionFileRecord[];
  chunks?: UploadChunkRecord[];
  events?: UploadSessionEventRecord[];
  limits?: UploadSessionLimits | null;
};

export type UploadChunkResponse = {
  session: UploadSessionRecord;
  file: UploadSessionFileRecord;
  chunk: UploadChunkRecord;
};

export type UploadSessionFileCompleteResponse = {
  session: UploadSessionRecord;
  file: UploadSessionFileRecord;
  resource: UploadedFileRecord;
};

export type ResourceRecord = {
  file_id: string;
  original_name: string;
  content_type?: string | null;
  size_bytes: number;
  sha256: string;
  created_at: string;
  status?: string | null;
  source_type: "upload" | "bisque_import" | string;
  resource_kind: "image" | "video" | "table" | "document" | "file" | string;
  source_uri?: string | null;
  project_id?: string | null;
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
  tags?: string[];
  metadata?: Record<string, unknown> | null;
  share_summary?: ResourceShareSummary | null;
};

export type ResourceTextFormat = "csv" | "json" | "yaml" | "xml" | "markdown" | "text";

export type ResourceTextHead = {
  file_id: string;
  original_name: string;
  content_type: string;
  format: ResourceTextFormat | string;
  total_size_bytes: number;
  offset: number;
  returned_bytes: number;
  next_offset: number;
  truncated: boolean;
  encoding: string;
  eol: "lf" | "crlf" | "none" | string;
  line_count: number;
  approx_total_lines: number;
  text: string;
};

export type ResourceCsvRows = {
  file_id: string;
  original_name: string;
  delimiter: string;
  columns?: string[] | null;
  rows: string[][];
  offset_bytes: number;
  next_offset_bytes: number;
  returned_rows: number;
  has_more: boolean;
  approx_total_rows: number;
  total_size_bytes: number;
};

export type ResourceShareSummary = {
  share_status: "private" | "shared_by_me" | "shared_with_me" | "public" | string;
  active_grant_count: number;
  shared_by_me?: boolean | null;
  shared_with_me?: boolean | null;
  public?: boolean | null;
};

export type ResourceMetadataFilter = {
  path: string;
  operator: "eq" | "contains" | "exists" | "lt" | "lte" | "gt" | "gte" | string;
  value?: string | number | boolean | null;
};

export type ResourceListResponse = {
  count: number;
  resources: ResourceRecord[];
};

export type ResourceResponse = {
  resource: ResourceRecord;
};

export type ResourceEventRecord = {
  event_id: string;
  resource_id: string;
  actor_user_id?: string | null;
  actor_org_id?: string | null;
  event_type: string;
  ts: string;
  metadata?: Record<string, unknown> | null;
};

export type ResourceBulkTagRequest = {
  resource_ids: string[];
  tags: string[];
  metadata?: Record<string, unknown> | null;
};

export type ResourceBulkTagResponse = {
  count: number;
  resources: ResourceRecord[];
  events: ResourceEventRecord[];
};

export type ResourceBulkLifecycleRequest = {
  resource_ids: string[];
};

export type ResourceBulkLifecycleResponse = {
  count: number;
  resources: ResourceRecord[];
  events: ResourceEventRecord[];
};

export type ResourceShareGrantRecord = {
  grant_id: string;
  resource_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  grantee_user_id?: string | null;
  grantee_org_id?: string | null;
  role: "read" | string;
  status: "active" | "revoked" | string;
  created_by_user_id?: string | null;
  created_at: string;
  updated_at: string;
  revoked_at?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type ResourceShareGrantCreateRequest = {
  grantee_user_id?: string | null;
  grantee_org_id?: string | null;
  public?: boolean | null;
  role?: "read" | string;
  metadata?: Record<string, unknown> | null;
};

export type ResourceShareGrantsCreateRequest = ResourceShareGrantCreateRequest & {
  resource_ids: string[];
};

export type ResourceShareGrantResponse = {
  grant: ResourceShareGrantRecord;
};

export type ResourceShareGrantsCreateResponse = {
  count: number;
  grants: ResourceShareGrantRecord[];
};

export type ResourceCollectionShareGrantsCreateResponse = {
  count: number;
  collection: ResourceCollectionRecord;
  grants: ResourceShareGrantRecord[];
};

/** One pickable share grantee: a same-org person or the org itself. */
export type ShareTargetRecord = {
  kind: "user" | "org" | string;
  grantee_user_id?: string;
  grantee_org_id?: string;
  label: string;
  detail?: string;
};

export type ShareTargetListResponse = {
  targets: ShareTargetRecord[];
};

/** Collection-level grant rows (same wire shape, keyed by collection_id). */
export type ResourceCollectionShareGrantRecord = Omit<
  ResourceShareGrantRecord,
  "resource_id"
> & {
  collection_id: string;
};

export type ResourceCollectionShareGrantListResponse = {
  grants: ResourceCollectionShareGrantRecord[];
};

export type ResourceCollectionShareGrantRevokeResponse = {
  grant: ResourceCollectionShareGrantRecord;
};

export type ResourceShareGrantListResponse = {
  resource_id: string;
  count: number;
  grants: ResourceShareGrantRecord[];
};

export type ResourceCollectionRecord = {
  collection_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  project_id?: string | null;
  parent_collection_id?: string | null;
  name: string;
  description?: string | null;
  collection_type: "collection" | "folder" | "dataset" | string;
  status: "active" | "deleted" | string;
  resource_count: number;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown> | null;
};

export type ResourceCollectionCreateRequest = {
  name: string;
  description?: string | null;
  collection_type?: "collection" | "folder" | "dataset" | string | null;
  project_id?: string | null;
  parent_collection_id?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type ResourceCollectionPatchRequest = {
  name: string;
};

export type ResourceCollectionResponse = {
  collection: ResourceCollectionRecord;
};

export type ResourceCollectionListResponse = {
  count: number;
  collections: ResourceCollectionRecord[];
};

export type ResourceCollectionMembershipRecord = {
  collection_id: string;
  resource_id: string;
  position: number;
  added_by_user_id?: string | null;
  added_at: string;
  metadata?: Record<string, unknown> | null;
};

export type ResourceCollectionAddResourcesResponse = {
  collection: ResourceCollectionRecord;
  added_count: number;
  memberships: ResourceCollectionMembershipRecord[];
};

export type ResourceCollectionRemoveResourcesResponse = {
  collection: ResourceCollectionRecord;
  removed_count: number;
  memberships: ResourceCollectionMembershipRecord[];
};

export type DatasetSnapshotRecord = {
  snapshot_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  project_id?: string | null;
  source_collection_id?: string | null;
  name: string;
  description?: string | null;
  status: "active" | "deleted" | string;
  resource_count: number;
  total_bytes: number;
  created_by_user_id?: string | null;
  created_at: string;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotResourceRecord = {
  snapshot_id: string;
  resource_id: string;
  position: number;
  original_name: string;
  content_type?: string | null;
  size_bytes: number;
  sha256?: string | null;
  source_type: "upload" | "bisque_import" | string;
  resource_kind: "image" | "video" | "table" | "document" | "file" | string;
  storage_uri?: string | null;
  source_uri?: string | null;
  project_id?: string | null;
  resource_created_at?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotCreateRequest = {
  name: string;
  description?: string | null;
  source_collection_id?: string | null;
  resource_ids?: string[];
  resource_query?: DatasetSnapshotResourceQuery | null;
  project_id?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotResourceQuery = {
  q?: string;
  kind?: string;
  source?: string;
  project_id?: string | null;
  sharing?: string;
  tags?: string[];
  descriptors?: string[];
  metadata_filters?: ResourceMetadataFilter[];
  created_after?: string;
  created_before?: string;
  processing_status?: string;
};

export type DatasetSnapshotResponse = {
  snapshot: DatasetSnapshotRecord;
  resources: DatasetSnapshotResourceRecord[];
};

export type DatasetSnapshotListResponse = {
  count: number;
  snapshots: DatasetSnapshotRecord[];
};

export type DatasetSnapshotEventRecord = {
  event_id: string;
  snapshot_id: string;
  actor_user_id?: string | null;
  actor_org_id?: string | null;
  event_type: string;
  ts: string;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotEventListResponse = {
  snapshot_id: string;
  count: number;
  total_count: number;
  limit: number;
  offset: number;
  events: DatasetSnapshotEventRecord[];
};

export type DatasetSnapshotShareGrantRecord = {
  grant_id: string;
  snapshot_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  grantee_user_id?: string | null;
  grantee_org_id?: string | null;
  role: "read" | string;
  status: "active" | "revoked" | string;
  created_by_user_id?: string | null;
  created_at: string;
  updated_at: string;
  revoked_at?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotShareGrantCreateRequest = {
  grantee_user_id?: string | null;
  grantee_org_id?: string | null;
  role?: "read" | string;
  metadata?: Record<string, unknown> | null;
};

export type DatasetSnapshotShareGrantResponse = {
  grant: DatasetSnapshotShareGrantRecord;
};

export type DatasetSnapshotShareGrantListResponse = {
  count: number;
  grants: DatasetSnapshotShareGrantRecord[];
};

export type DataAgentJobRecord = {
  job_id: string;
  owner_user_id: string;
  owner_org_id?: string | null;
  owner_role?: string | null;
  project_id?: string | null;
  job_type:
    | "caption_resources"
    | "extract_metadata"
    | "organize_resources"
    | "deduplicate_resources"
    | "quality_check_resources"
    | "batch_tag_resources"
    | "create_dataset_snapshot"
    | string;
  status: "queued" | "running" | "succeeded" | "failed" | "canceled" | string;
  resource_count: number;
  progress_completed: number;
  progress_total: number;
  error?: string | null;
  created_by_user_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  input_selector?: Record<string, unknown> | null;
  output_summary?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
};

export type DataAgentJobEventRecord = {
  event_id: string;
  job_id: string;
  sequence: number;
  event_type: string;
  actor_user_id?: string | null;
  actor_org_id?: string | null;
  ts: string;
  message?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type DataAgentJobCreateRequest = {
  job_type:
    | "caption_resources"
    | "extract_metadata"
    | "organize_resources"
    | "deduplicate_resources"
    | "quality_check_resources"
    | "batch_tag_resources"
    | "create_dataset_snapshot"
    | string;
  resource_ids?: string[];
  source_collection_id?: string | null;
  project_id?: string | null;
  resource_query?: DatasetSnapshotResourceQuery | null;
  input_selector?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
};

export type DataAgentJobControlRequest = {
  action: "cancel" | "retry" | string;
  reason?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type DataAgentJobResponse = {
  job: DataAgentJobRecord;
  events: DataAgentJobEventRecord[];
};

export type DataAgentJobListResponse = {
  count: number;
  jobs: DataAgentJobRecord[];
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
  stale_running_runs: number;
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

export type AdminActivityPeriod = {
  label: string;
  window: string;
  messages: number;
  user_messages: number;
  assistant_messages: number;
  tool_calls: number;
  active_users: number;
  runs: number;
  failed_runs: number;
  artifacts: number;
};

export type AdminRuntimeSummary = {
  app_version?: string;
  store_backend: string;
  dispatch_mode: string;
  job_transport: string;
  event_transport: string;
  stub_worker_enabled: boolean;
  nats_configured: boolean;
  nats_stream?: string;
  nats_jobs_subject?: string;
  nats_rarespot_jobs_subject?: string;
  nats_events_subject?: string;
  nats_cancel_subject?: string;
  nats_event_consumer?: string;
  artifact_root?: string;
  upload_root?: string;
  run_recovery_enabled?: boolean;
  run_recovery_interval_seconds?: number;
  run_recovery_batch_limit?: number;
};

export type AdminQueueConsumerDiagnostic = {
  name: string;
  role?: string;
  subject?: string;
  active: boolean;
  ack_wait_seconds?: number;
  max_deliver?: number;
  pending_messages: number;
  in_flight_messages: number;
  redelivered_messages: number;
  waiting_pull_requests: number;
  delivered_stream_sequence?: number;
  ack_floor_stream_sequence?: number;
  error?: string;
};

export type AdminQueueDiagnostics = {
  available: boolean;
  mode: string;
  stream?: string;
  stream_subjects?: string[];
  stream_messages: number;
  stream_bytes: number;
  first_sequence: number;
  last_sequence: number;
  consumer_count: number;
  consumers: AdminQueueConsumerDiagnostic[];
  error?: string;
};

export type AdminDatabasePoolStats = {
  max_conns: number;
  total_conns: number;
  acquired_conns: number;
  idle_conns: number;
  constructing_conns: number;
  acquire_count: number;
  empty_acquire_count: number;
  canceled_acquire_count: number;
  new_conns_count: number;
  max_lifetime_destroy_count: number;
  max_idle_destroy_count: number;
  acquire_duration_seconds: number;
  empty_acquire_wait_seconds: number;
  saturation: number;
  wait_ratio: number;
};

export type AdminDatabaseQueryStats = {
  query_id: string;
  calls: number;
  mean_exec_ms: number;
  total_exec_ms: number;
  rows: number;
  shared_blocks_hit: number;
  shared_blocks_read: number;
  temp_blocks_written: number;
  query: string;
};

export type AdminDatabaseDiagnostics = {
  available: boolean;
  pool: AdminDatabasePoolStats;
  top_queries: AdminDatabaseQueryStats[];
  error?: string;
};

export type AdminWorkerRecord = {
  worker_id: string;
  worker_kind: string;
  status: string;
  current_run_id?: string | null;
  hostname?: string | null;
  version?: string | null;
  started_at: string;
  last_heartbeat_at: string;
  updated_at: string;
  heartbeat_age_seconds?: number | null;
  active: boolean;
  stale: boolean;
  metadata: Record<string, unknown>;
};

export type AdminUserSummary = {
  user_id: string;
  email?: string | null;
  display_name?: string | null;
  role?: string | null;
  status?: string | null;
  org_id?: string | null;
  created_at?: string | null;
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
  last_event_kind?: string | null;
  last_event_at?: string | null;
  last_event_sequence?: number | null;
  last_activity_age_seconds?: number | null;
  event_count: number;
  message_delta_count: number;
  tool_call_count: number;
  artifact_count: number;
  heartbeat_count: number;
  last_tool_name?: string | null;
  last_tool_at?: string | null;
  first_delta_latency_seconds?: number | null;
  first_tool_latency_seconds?: number | null;
  first_artifact_latency_seconds?: number | null;
  lease_worker_id?: string | null;
  lease_expires_at?: string | null;
  lease_active?: boolean;
  lease_expired?: boolean;
  lease_seconds_remaining?: number | null;
  lease_last_renewed_at?: string | null;
  lease_last_renewed_age_seconds?: number | null;
  stale: boolean;
  stale_reason?: string | null;
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
  runtime: AdminRuntimeSummary;
  queue: AdminQueueDiagnostics;
  database: AdminDatabaseDiagnostics;
  kpis: AdminPlatformKpis;
  activity: AdminActivityPeriod[];
  usage_last_24h: AdminUsageBucket[];
  tool_usage_7d: AdminToolUsageRecord[];
  workers: AdminWorkerRecord[];
  top_users: AdminUserSummary[];
  resource_projects: AdminResourceOwnerSummary[];
  recent_issues: AdminIssueRecord[];
};

export type AdminResourceOwnerSummary = {
  id: string;
  uploads: number;
  storage_bytes: number;
};

export type AdminUserListResponse = {
  count: number;
  users: AdminUserSummary[];
};

export type AdminOrganization = {
  org_id: string;
  name: string;
  status?: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  uploads?: number;
  storage_bytes?: number;
};

export type AdminOrganizationListResponse = {
  count: number;
  organizations: AdminOrganization[];
};

export type AdminCreateOrganizationRequest = {
  org_id?: string;
  name?: string;
  status?: string;
  metadata?: Record<string, unknown>;
};

export type AdminCreateUserRequest = {
  user_id?: string;
  email?: string;
  display_name?: string;
  role?: string;
  status?: string;
  org_id?: string;
  metadata?: Record<string, unknown>;
};

export type AdminUserStatus = "active" | "pending" | "disabled" | "rejected";

export type AdminUpdateUserStatusRequest = {
  status: AdminUserStatus;
};

export type AdminUserAccount = {
  user_id: string;
  email?: string | null;
  display_name?: string | null;
  role?: string | null;
  status?: string | null;
  org_id?: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
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

export type AdminMetricWeekPoint = {
  week_start: string;
  value: number;
};

export type AdminMetricNorthStar = {
  label: string;
  definition: string;
  current_week: number;
  previous_week: number;
  delta_pct: number | null;
  weekly: AdminMetricWeekPoint[];
};

export type AdminMetricKpis = {
  wau: number;
  mau: number;
  stickiness_pct: number | null;
  new_users: number;
  activation_rate_pct: number | null;
  activation_window_days: number;
  week4_retention_pct: number | null;
  useful_run_rate_pct: number | null;
  useful_runs: number;
  total_runs: number;
};

export type AdminMetricCohort = {
  cohort_start: string;
  size: number;
  values_pct: Array<number | null>;
  retained: number[];
};

export type AdminMetricRetention = {
  unit: string;
  max_periods: number;
  cohorts: AdminMetricCohort[];
};

export type AdminMetricPowerBucket = {
  days_active: number;
  users: number;
};

export type AdminMetricPowerCurve = {
  window_days: number;
  total_users: number;
  power_user_threshold: number;
  power_users: number;
  power_user_share_pct: number | null;
  buckets: AdminMetricPowerBucket[];
};

export type AdminMetricFunnelStage = {
  stage: string;
  users: number;
  of_previous_pct: number | null;
  of_top_pct: number | null;
};

export type AdminMetricModelCost = {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  runs: number;
  cost: number | null;
  priced: boolean;
};

export type AdminMetricCostDay = {
  day: string;
  total_tokens: number;
  cost: number | null;
};

export type AdminMetricCost = {
  currency: string;
  priced: boolean;
  total_tokens: number;
  total_cost: number | null;
  cost_per_useful_run: number | null;
  tokens_per_useful_run: number | null;
  useful_runs: number;
  unpriced_models: string[];
  by_model: AdminMetricModelCost[];
  daily: AdminMetricCostDay[];
};

export type AdminMetricsResponse = {
  generated_at: string;
  available: boolean;
  range_days: number;
  north_star: AdminMetricNorthStar;
  kpis: AdminMetricKpis;
  retention_cohorts: AdminMetricRetention;
  power_user_curve: AdminMetricPowerCurve;
  activation_funnel: AdminMetricFunnelStage[];
  cost: AdminMetricCost;
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

export type BisqueResourceRecord = {
  resource_uri: string;
  name?: string | null;
  resource_uniq?: string | null;
  resource_type?: string | null;
  tags?: Record<string, string>;
  client_view_url?: string | null;
  image_service_url?: string | null;
};

export type BisqueSearchRequest = {
  resourceType?: string;
  tagQuery?: string;
  tagOrder?: string;
  query?: string;
  nameContains?: string;
  extensions?: string[];
  scope?: string;
  sort?: string;
  limit?: number;
  offset?: number;
  countAll?: boolean;
};

export type BisqueSearchResponse = {
  count: number;
  results: BisqueResourceRecord[];
};

export type BisqueUploadRecord = {
  file_id?: string | null;
  artifact_id?: string | null;
  resource_uri: string;
  name?: string | null;
  resource_uniq?: string | null;
  client_view_url?: string | null;
};

export type BisqueDatasetRecord = {
  collection_id?: string | null;
  name: string;
  resource_uri: string;
  resource_uniq?: string | null;
  member_count: number;
  client_view_url?: string | null;
};

export type BisquePushRequest = {
  fileIds?: string[];
  collectionIds?: string[];
  datasetName?: string;
};

export type BisquePushResponse = {
  count: number;
  uploads: BisqueUploadRecord[];
  datasets: BisqueDatasetRecord[];
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

export type Hdf5DatasetSummary = {
  file_id: string;
  dataset_path: string;
  dataset_name: string;
  materials_domain_tags: string[];
  preview_kind?: string | null;
  semantic_role?: string | null;
  feature_filter?: Hdf5FeatureFilter | null;
  units_hint?: string | null;
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
    cell_data_path?: string | null;
    cell_data_consistent?: boolean | null;
    complete?: boolean | null;
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

export type Hdf5FeatureFilter = {
  supported: true;
  source_dataset_path: string;
  max_ids: number;
  background_id: 0;
  provenance: "co_registered_raw_integer_feature_ids" | string;
  registration_key: string;
  target_role: "feature_ids" | "euler_angles" | "ipf_colors" | string;
  native_shape: [number, number, number];
  preview_shape: [number, number, number];
  preview_stride: { z: number; y: number; x: number };
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

export type CiftiStructure = {
  name: string;
  count: number;
};

export type CiftiViewerData = {
  /** Human label, e.g. "dense timeseries", "parcellated connectivity". */
  cifti_type: string;
  /** Which views apply to this file: "carpet" and/or "connectivity". */
  views: ("carpet" | "connectivity" | string)[];
  /** Brain-location rows and the second-axis columns of the data matrix. */
  rows: number;
  cols: number;
  structures: CiftiStructure[];
  column_axis: { role?: string; size?: number; step?: number; unit?: string };
  service_urls: { carpet?: string; connectivity?: string; download?: string };
};

export type UploadViewerInfo = {
  kind?: "image" | "hdf5" | "cifti" | "unsupported" | string;
  file_id: string;
  original_name: string;
  /**
   * False when the image engine recognized the file but cannot decode it (e.g. a
   * Leica .lif — registered but non-functional in this libbioimage build). The
   * viewer renders a calm "preview unavailable, download instead" card rather than a
   * broken 1×1 canvas stuck on "Loading…". Absent/true means a normal decodable image.
   */
  decodable?: boolean;
  /** Human-readable explanation shown when `decodable` is false. */
  message?: string;
  modality?: "microscopy" | "medical" | "geospatial" | "materials" | "image" | "unknown" | string;
  /** Canonical in-memory order used by every semantic viewer operation. */
  dims_order: string;
  /** Axis order reported by the source container before canonicalization. */
  source_dims_order?: string;
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
  data_semantics?: {
    kind: "intensity" | "binary_mask" | "probability_mask";
    basis: "authoritative" | "exact" | "suggested" | "unknown" | string;
    strength: "authoritative" | "exact" | "suggested" | "unknown" | string;
    supported_modes: Array<"intensity" | "mask">;
    recommended_view: "intensity" | "mask";
    threshold?: {
      method: "otsu-256-v1";
      value: number;
      domain: "raw";
      foreground: "above";
      sample_scope: string;
      sample_count: number;
      z_samples: number[];
      channel: number;
      t: number;
      sampling_algorithm: string;
      sampling_strategy?: "exact" | "stratified-z-spatial";
      source_sha256?: string;
      bins?: number;
    };
  };
  scalar_mask_capability?: {
    version: 1;
    source_authority: "original";
    source_format: "tiff" | "ome-tiff";
    source_sha256: string;
    dtype: "uint8" | "uint16" | "int16";
    threshold_domain: "raw";
    threshold_foreground: "above";
    slice_delivery: "thresholded_png";
    volume_delivery: "raw_scalar";
    volume_sampling: "nearest";
    channel_selection: "single";
    time_selection: "single";
    surfaces: Array<"2d" | "mpr" | "volume">;
  };
  viewer_calibrations?: {
    version: 1;
    source_sha256: string;
    selections: Record<
      string,
      {
        revision: number;
        channel: number;
        t: number;
        render_mode: "auto" | "intensity" | "mask";
        threshold_method: "otsu-256-v1" | "manual";
        threshold_value: number;
        threshold_foreground: "above";
        threshold_provenance: {
          method: "otsu-256-v1";
          value: number;
          domain: "raw";
          foreground: "above";
          channel: number;
          t: number;
          sample_scope: "volume" | "stratified_z";
          sample_count: number;
          sampling_algorithm: string;
          sampling_strategy: "exact" | "stratified-z-spatial";
          z_samples: number[];
          source_sha256: string;
          bins: number;
        };
      }
    >;
  };
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
    scalar_colormap?: string | null;
    volume_signal_floor?: number | null;
    volume_density?: number | null;
    volume_lighting?: boolean | null;
    volume_lighting_strength?: number | null;
    volume_channel?: number | null;
    volume_view_preset?: string | null;
    volume_camera_mode?: string | null;
    // 3D ray-projection, decoupled from the 2D `fusion_method` (which combines
    // channels in a flat image). MIP maxes each ray; composite integrates front-to-
    // back. Undefined → the renderer's per-source default (composite for multichannel
    // fluorescence, which is dense; MIP would flatten it into a cloud).
    volume_projection?: "mip" | "composite" | null;
    volume_clip_min?: { x: number; y: number; z: number };
    volume_clip_max?: { x: number; y: number; z: number };
    // Z-cursor cutaway: when true the Volume tab cuts the volume at the live Z
    // slice and exposes a high-resolution interior cross-section (overview camera),
    // independent of the manual volume_clip box.
    volume_cutaway?: boolean | null;
    // Per-channel intensity window (normalized [0,1]) for the multichannel volume
    // LUT, indexed by channel. Absent -> full range per channel.
    volume_channel_windows?: Array<{ low: number; high: number } | null> | null;
    // Gamma tone-curve exponent for the multichannel volume (vole-core GAMMA_SCALE):
    // default 1 (linear); >1 darkens midtones, <1 lifts faint structure.
    volume_gamma?: number | null;
    scalar_render_mode?: "auto" | "intensity" | "mask";
    scalar_threshold_method?: "otsu-256-v1" | "manual";
    scalar_threshold_value?: number | null;
    scalar_threshold_foreground?: "above";
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
    format?: string;
    acquisition?: Record<string, string | number>;
    dims_order: string;
    array_shape: number[];
    array_dtype: string;
    sha256?: string;
    size_bytes?: number;
    content_type?: string;
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
    /** Unit for physical_spacing and derived physical extents (for example, mm or um). */
    physical_spacing_unit?: string | null;
    spacing_units?: {
      z?: string | null;
      y?: string | null;
      x?: string | null;
    } | null;
    source_dims_order?: string;
    scene?: string | null;
    scene_count: number;
    selected_scene_index?: number | null;
    selected_scene_id?: string | null;
    /** Tiled-mosaic acquisition (multi-field stage scan). Null/absent for a normal
     *  single-field image. An unstitched mosaic shows per-field illumination seams. */
    mosaic?: {
      tiles: number;
      stitched?: boolean;
      overlap?: number;
    } | null;
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
      scalar_colormap?: string | null;
      volume_signal_floor?: number | null;
      volume_density?: number | null;
      volume_lighting?: boolean | null;
      volume_lighting_strength?: number | null;
      volume_channel?: number | null;
      volume_view_preset?: string | null;
      volume_camera_mode?: string | null;
      volume_projection?: "mip" | "composite" | null;
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
        cell_data_path?: string | null;
        cell_data_consistent?: boolean | null;
        complete?: boolean | null;
      } | null;
    };
    tree: Hdf5ViewerTreeNode[];
    limitations: string[];
    selected_dataset_path?: string | null;
    default_dataset_path?: string | null;
    materials?: Hdf5MaterialsPayload | null;
  } | null;
  /** Present for kind:"cifti" — drives the grayordinate carpet + connectivity views. */
  cifti?: CiftiViewerData | null;
};

export type CiftiCarpetResponse = {
  rows: number;
  cols: number;
  clip_z: number;
  source_rows: number;
  structures: { name: string; start: number; end: number }[];
  column_axis: { role?: string; size?: number; sampled?: number; step?: number; unit?: string };
  cifti_type?: string;
  /** base64 uint8, rows*cols, row-major; z-score maps 0..255 -> -clip_z..+clip_z. */
  data: string;
};

export type CiftiConnectivityResponse = {
  n: number;
  min: number;
  max: number;
  dtype: string;
  /** true when correlation was computed from a timeseries; false = stored matrix. */
  computed: boolean;
  labels?: string[];
  cifti_type?: string;
  /** base64 little-endian float32, n*n, row-major. */
  data: string;
};

export type UploadViewerHistogramResponse = {
  file_id: string;
  bins: number;
  dtype?: string;
  channels?: number[];
  channel?: number;
  t?: number;
  source?: string;
  sample_count?: number;
  scope?: "volume" | string;
  sampling?: Record<string, unknown>;
  threshold?: {
    method: "otsu-256-v1" | string;
    value: number;
    domain?: "raw" | string;
    foreground?: "above" | string;
    sample_scope?: string;
    sample_count?: number;
    z_samples?: number[];
    channel?: number;
    t?: number;
    sampling_algorithm?: string;
    sampling_strategy?: "exact" | "stratified-z-spatial" | string;
    source_sha256?: string;
    bins?: number;
  };
  data_semantics?: UploadViewerInfo["data_semantics"];
  histogram: {
    bins: number[];
    edges: number[];
    min: number;
    max: number;
    channel_indices: number[];
    time_index: number;
    sampling?: Record<string, unknown>;
    threshold?: UploadViewerHistogramResponse["threshold"];
  };
};

export type PublicConfigResponse = {
  app_name?: string | null;
  app_version?: string | null;
  features?: Record<string, boolean>;
  bisque_root?: string | null;
  bisque_browser_url?: string | null;
  bisque_auth_enabled?: boolean;
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
  mode?: "bisque" | "guest" | "workos" | null;
  provider?: "local" | "workos" | string | null;
  account_status?: AdminUserStatus | "not_configured" | string | null;
  account_email?: string | null;
  account_user_id?: string | null;
  message?: string | null;
  authorization_url?: string | null;
  logout_url?: string | null;
  user?: {
    id?: string | null;
    username?: string | null;
    email?: string | null;
    org_id?: string | null;
    role?: string | null;
  } | null;
  guest_profile?: {
    name: string;
    email: string;
    affiliation: string;
  } | null;
  is_admin?: boolean;
  bisque_linked?: boolean | null;
};

export type BisqueAuthLoginRequest = {
  username: string;
  password: string;
};

export type AccountRequestPayload = {
  name: string;
  email: string;
  affiliation: string;
};

export type AccountRequestResponse = BisqueAuthSessionResponse;

// ---------------------------------------------------------------------------
// Ultra execution tier — capability visibility + admission wire contract (v1).
// Mirrors the Go control-plane views in internal/httpapi/ultra_*.go.
// ---------------------------------------------------------------------------

export type UltraCapabilityRetryMode =
  | "not_retryable"
  | "after_state_change"
  | "retry_at"
  | "requote";

export type UltraCapabilityDisabledReason =
  | "phase2_capabilities_unavailable"
  | "policy_disabled"
  | "kill_switch_active"
  | "entitlement_missing"
  | "capability_missing"
  | "capability_version_mismatch"
  | "capability_surface_mismatch"
  | "capability_unhealthy"
  | "user_quota_unavailable"
  | "org_quota_unavailable"
  | "global_capacity_unavailable"
  | "endpoint_capacity_unavailable"
  | "resource_authorization_failed"
  | "profile_incompatible"
  | "account_binding_invalid"
  | "scheduler_unavailable"
  | "worker_class_unavailable";

export type UltraCapabilityRetryDirective = {
  mode: UltraCapabilityRetryMode;
  retry_at?: string | null;
};

export type UltraCapabilityAvailability = {
  available: boolean;
  reason?: UltraCapabilityDisabledReason | null;
  retry?: UltraCapabilityRetryDirective | null;
};

export type UltraCapabilityCatalogResponse = {
  schema_version: "ultra.capability_catalog.v1";
  scope: "visibility_only_not_authorization";
  observed_at: string;
  ultra: {
    visible: boolean;
    quote_available: boolean;
    admission_available: boolean;
    dispatch_available: boolean;
    requirements_published: boolean;
    capability_binding_sha256: string;
    availability: UltraCapabilityAvailability;
  };
};

export type UltraRetryDirectiveV1 = {
  mode: UltraCapabilityRetryMode;
  retry_at?: string;
};

export type UltraTokenCallBudgetV1 = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  model_calls: number;
  tool_calls: number;
};

export type UltraRoleBudgetV1 = {
  role_id: string;
  limits: UltraTokenCallBudgetV1;
};

export type UltraEffectiveBudgetV1 = {
  schema_version: "ultra.effective_budget.v1";
  root_deadline: string;
  root_limits: UltraTokenCallBudgetV1;
  role_budgets: UltraRoleBudgetV1[];
  dispatch: {
    autonomous_cycles: number;
    dynamic_dispatches: number;
    concurrent_dynamic_dispatches: number;
    durable_tasks: number;
    concurrent_leases: number;
    retries_per_task: number;
    retries_per_phase: number;
  };
  compute: {
    sandbox_executions: number;
    sandbox_compute_milliseconds: number;
  };
  evidence: {
    events: number;
    event_bytes: number;
    checkpoints: number;
    checkpoint_bytes: number;
    output_bytes: number;
  };
  quotas: {
    user_concurrent_runs: number;
    org_concurrent_runs: number;
    fleet_concurrent_runs: number;
    endpoint_concurrent_calls: number;
    sandbox_concurrent_actions: number;
  };
};

export type UltraEstimateRangeV1 = {
  lower: number;
  upper: number;
};

export type UltraEstimateBandV1 = {
  schema_version: "ultra.estimate_band.v1";
  estimator_id: string;
  estimator_version: string;
  estimator_sha256: string;
  confidence_basis_points: number;
  wall_clock_milliseconds: UltraEstimateRangeV1;
  input_tokens: UltraEstimateRangeV1;
  output_tokens: UltraEstimateRangeV1;
  total_tokens: UltraEstimateRangeV1;
  model_calls: UltraEstimateRangeV1;
  tool_calls: UltraEstimateRangeV1;
  dynamic_dispatches: UltraEstimateRangeV1;
  durable_tasks: UltraEstimateRangeV1;
  sandbox_executions: UltraEstimateRangeV1;
  checkpoints: UltraEstimateRangeV1;
  output_bytes: UltraEstimateRangeV1;
};

export type UltraAcceptanceCriterionViewV1 = {
  ordinal: number;
  criterion_id: string;
  statement: string;
  required_evidence: string[];
};

export type UltraAcceptanceIntentViewV1 = {
  schema_version: "ultra.acceptance_intent.v1";
  intent_sha256: string;
  criteria: UltraAcceptanceCriterionViewV1[];
  deliverables: string[];
  exclusions: string[];
};

export type UltraAdmissionQuoteViewV1 = {
  schema_version: "ultra.admission_quote.v1";
  canonicalization_version: "ultra.canonical_json.v1";
  quote_id: string;
  quote_sha256: string;
  request_sha256: string;
  issued_at: string;
  expires_at: string;
  hold_expires_at: string | null;
  effective_tier: "ultra";
  policy_id: string;
  policy_version: string;
  effective_budget: UltraEffectiveBudgetV1;
  estimate_band: UltraEstimateBandV1;
  confirmation_requirement: "explicit_required" | "not_required";
  acceptance_intent: UltraAcceptanceIntentViewV1;
};

export type UltraAdmissionQuoteDecisionV1 = {
  schema_version: "ultra.quote_issuance_decision.v1";
  outcome: "quoted" | "rejected";
  decided_at: string;
  reason?: string;
  retry?: UltraRetryDirectiveV1;
  quote?: UltraAdmissionQuoteViewV1;
};

export type UltraConfirmedRunV1 = {
  run_id: string;
  goal: string;
  status: string;
  workflow_kind: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  thread_id?: string;
  user_id?: string;
  mode?: string;
  requested_execution_tier?: "ultra";
  effective_execution_tier?: "ultra";
  resolved_reasoning_profile_id?: string;
  current_node?: string;
  parent_run_id?: string;
  planner_version?: string;
  agent_role?: string;
  trace_group_id?: string;
  checkpoint_id?: string;
  checkpoint_state?: Record<string, unknown> | null;
  budget_state?: Record<string, unknown> | null;
  response_text?: string;
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
};

export type UltraQuoteConfirmationResponseV1 = {
  schema_version: "ultra.quote_confirmation_response.v1";
  outcome: "created" | "replayed" | "rejected";
  reason?: string;
  retry?: UltraRetryDirectiveV1;
  run?: UltraConfirmedRunV1;
};

// --- DREAM3D microstructure viewer types (restored: viewer functionality removed by #40) ---
export type Hdf5MaterialsPayload = {
  detected: boolean;
  schema?: "dream3d" | null;
  capabilities: string[];
  roles: Record<string, string>;
  phase_names: string[];
  phase_names_source?: string | null;
  phase_names_provenance?: string | null;
  feature_count?: number | null;
  grain_count?: number | null;
  declared_feature_tuple_count?: number | null;
  referenced_positive_feature_count?: number | null;
  feature_id_scan_complete: boolean;
  feature_id_consistency?: boolean | null;
  feature_zero_reserved?: boolean | null;
  recommended_view: "materials" | "explorer";
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
      cell_data_path?: string | null;
      cell_data_consistent?: boolean | null;
      complete?: boolean | null;
    } | null;
    spacing_note?: string | null;
    phase_names: string[];
    phase_names_source?: string | null;
    phase_names_provenance?: string | null;
    feature_count?: number | null;
    grain_count?: number | null;
    declared_feature_tuple_count?: number | null;
    referenced_positive_feature_count?: number | null;
    feature_id_scan_complete: boolean;
    feature_id_consistency?: boolean | null;
    feature_zero_reserved?: boolean | null;
    capabilities: string[];
    recommended_map_dataset_path?: string | null;
  };
  maps: Hdf5MaterialsMapResponse[];
  grain_charts: Hdf5MaterialsChartResponse[];
  orientation_charts: Hdf5MaterialsChartResponse[];
  synthetic_stats: Hdf5MaterialsChartResponse[];
  dataset_links: Hdf5MaterialsDatasetLinkResponse[];
};
