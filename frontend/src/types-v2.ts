export type V2JsonPrimitive = string | number | boolean | null;
export type V2JsonValue = V2JsonPrimitive | V2JsonObject | V2JsonValue[];
export type V2JsonObject = { [key: string]: V2JsonValue | undefined };

export type V2MessageRole = "system" | "user" | "assistant" | "tool" | "developer" | (string & {});

export type V2ThreadStatus = "active" | "archived" | "deleted" | (string & {});
export type V2RunStatus =
  | "queued"
  | "running"
  | "waiting_for_input"
  | "waiting_for_task"
  | "succeeded"
  | "failed"
  | "canceled"
  | (string & {});

export type V2EventKind =
  | "run.started"
  | "node.started"
  | "node.completed"
  | "state.updated"
  | "task.queued"
  | "task.progress"
  | "task.completed"
  | "artifact.created"
  | "interrupt.raised"
  | "checkpoint.created"
  | "run.completed"
  | "run.failed"
  | "message.delta"
  | "error"
  | (string & {});

export type V2ArtifactKind = "artifact" | "preview" | "report" | "dataset" | "file" | (string & {});

export type V2ThreadMessage = {
  message_id?: string | null;
  thread_id?: string | null;
  role: V2MessageRole;
  content: string;
  created_at?: string | null;
  metadata?: V2JsonObject | null;
  run_id?: string | null;
};

export type V2ThreadRecord = {
  thread_id: string;
  user_id?: string | null;
  title?: string | null;
  status: V2ThreadStatus;
  created_at: string;
  updated_at: string;
  latest_run_id?: string | null;
  checkpoint_id?: string | null;
  summary?: string | null;
  metadata: V2JsonObject;
};

export type V2ThreadListResponse = {
  count: number;
  threads: V2ThreadRecord[];
};

export type V2ThreadMessageListResponse = {
  thread_id: string;
  count: number;
  messages: V2ThreadMessage[];
};

export type V2ThreadCreateRequest = {
  title?: string | null;
  metadata?: V2JsonObject | null;
  initial_messages?: V2ThreadMessage[];
  conversation_id?: string | null;
};

export type V2ThreadUpsertRequest = {
  title?: string | null;
  metadata?: V2JsonObject | null;
  messages?: V2ThreadMessage[];
};

export type V2RunBudget = {
  max_tool_calls?: number | null;
  max_runtime_seconds?: number | null;
  [key: string]: V2JsonValue | undefined;
};

export type V2RunCreateRequest = {
  goal?: string | null;
  messages: V2ThreadMessage[];
  file_ids?: string[];
  resource_uris?: string[];
  dataset_uris?: string[];
  selected_tool_names?: string[];
  knowledge_context?: V2JsonObject | null;
  selection_context?: V2JsonObject | null;
  workflow_hint?: V2JsonObject | null;
  reasoning_mode?: "auto" | "fast" | "deep";
  budgets?: V2RunBudget | null;
  benchmark?: V2JsonObject | null;
  metadata?: V2JsonObject | null;
};

export type V2RunResumeRequest = {
  decision?: "approve" | "reject" | string;
  note?: string | null;
  metadata?: V2JsonObject | null;
};

export type V2RunCancelRequest = {
  reason?: string | null;
  metadata?: V2JsonObject | null;
};

export type V2GraphEventRecord = {
  event_id?: number | string | null;
  run_id: string;
  thread_id?: string | null;
  event_kind: V2EventKind;
  event_type?: string | null;
  node_name?: string | null;
  task_id?: string | null;
  checkpoint_id?: string | null;
  scope_id?: string | null;
  agent_role?: string | null;
  level?: string | null;
  ts?: string | null;
  message?: string | null;
  payload: V2JsonObject;
};

export type V2RunRecord = {
  run_id: string;
  thread_id?: string | null;
  user_id?: string | null;
  goal: string;
  status: V2RunStatus;
  workflow_kind: string;
  mode?: string | null;
  current_node?: string | null;
  parent_run_id?: string | null;
  planner_version?: string | null;
  agent_role?: string | null;
  trace_group_id?: string | null;
  checkpoint_id?: string | null;
  checkpoint_state?: V2JsonObject | null;
  budget_state?: V2JsonObject | null;
  response_text?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  metadata: V2JsonObject;
};

export type V2RunListResponse = {
  count: number;
  runs: V2RunRecord[];
};

export type V2RunEventsResponse = {
  run_id: string;
  count: number;
  events: V2GraphEventRecord[];
};

export type V2ArtifactRecord = {
  artifact_id: string;
  run_id: string;
  thread_id?: string | null;
  kind: V2ArtifactKind;
  path?: string | null;
  source_path?: string | null;
  preview_path?: string | null;
  title?: string | null;
  result_group_id?: string | null;
  mime_type?: string | null;
  size_bytes?: number | null;
  sha256?: string | null;
  storage_uri?: string | null;
  tool_name?: string | null;
  category?: string | null;
  created_at: string;
  updated_at?: string | null;
  metadata: V2JsonObject;
};

export type V2ArtifactListResponse = {
  run_id: string;
  count: number;
  artifacts: V2ArtifactRecord[];
};

export type V2ArtifactResponse = {
  artifact: V2ArtifactRecord;
};

export type V2StreamTokenEvent = {
  event: "token";
  data: {
    delta: string;
    index?: number | null;
    [key: string]: V2JsonValue | undefined;
  };
};

export type V2StreamDoneEvent = {
  event: "done";
  data: V2RunRecord;
};

export type V2StreamErrorEvent = {
  event: "error";
  data: {
    message: string;
    detail?: V2JsonValue;
    [key: string]: V2JsonValue | undefined;
  };
};

export type V2StreamRunEvent = {
  event: "run_event";
  data: V2GraphEventRecord;
};

export type V2StreamGenericEvent = {
  event: string;
  data: unknown;
};

export type V2StreamEvent =
  | V2StreamTokenEvent
  | V2StreamDoneEvent
  | V2StreamErrorEvent
  | V2StreamRunEvent
  | V2StreamGenericEvent;

export type V2StreamHandlers = {
  onToken?: (delta: string, event: V2StreamTokenEvent) => void;
  onRunEvent?: (event: V2GraphEventRecord, envelope: V2StreamRunEvent) => void;
  onDone?: (run: V2RunRecord, envelope: V2StreamDoneEvent) => void;
  onError?: (error: V2StreamErrorEvent["data"], envelope: V2StreamErrorEvent) => void;
  onEvent?: (event: V2StreamEvent) => void;
};
