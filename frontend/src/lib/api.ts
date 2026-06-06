import type {
  AccountRequestPayload,
  AccountRequestResponse,
  AdminConversationActionResponse,
  AdminCreateOrganizationRequest,
  AdminCreateUserRequest,
  AdminIssueListResponse,
  AdminOrganization,
  AdminOrganizationListResponse,
  AdminOverviewResponse,
  AdminRunActionResponse,
  AdminRunListResponse,
  AdminUpdateUserStatusRequest,
  AdminUserAccount,
  AdminUserStatus,
  AdminUserListResponse,
  AnalysisHistoryResponse,
  ArtifactRecord,
  ArtifactListResponse,
  BisqueAuthLoginRequest,
  BisqueAuthSessionResponse,
  BisqueGuestAuthRequest,
  BisqueImportResponse,
  BisqueSearchRequest,
  BisqueSearchResponse,
  BisqueUploadResponse,
  ChatRequest,
  ChatResponse,
  ChatTitleRequest,
  ChatTitleResponse,
  ContractAuditRequest,
  ContractAuditResponse,
  ConversationListResponse,
  ConversationRecord,
  ConversationSearchResponse,
  Hdf5DatasetHistogramResponse,
  Hdf5MaterialsDashboardResponse,
  Hdf5DatasetSummary,
  Hdf5DatasetTablePreviewResponse,
  PublicConfigResponse,
  PrairieBenchmarkRunRequest,
  PrairieBenchmarkRunResponse,
  PrairieRetrainListResponse,
  PrairieRetrainRequest,
  PrairieStatusResponse,
  PrairieSyncResponse,
  ProgressEvent,
  ResourceListResponse,
  ResourceComputationLookupRequest,
  ResourceComputationLookupResponse,
  ResumableUploadChunkResponse,
  ResumableUploadCompleteResponse,
  ResumableUploadInitRequest,
  ResumableUploadSessionResponse,
  ReproReportRequest,
  ReproReportResponse,
  InferenceJobCreateRequest,
  ModelHealthResponse,
  TrainingDomainCreateRequest,
  TrainingDomainListResponse,
  TrainingDomainRecord,
  TrainingForkLineageRequest,
  TrainingLineageListResponse,
  TrainingLineageRecord,
  TrainingMergeRequestCreateRequest,
  TrainingMergeRequestDecisionRequest,
  TrainingMergeRequestListResponse,
  TrainingMergeRequestResponse,
  TrainingModelVersionListResponse,
  TrainingModelVersionResponse,
  TrainingUpdateProposalDecisionRequest,
  TrainingUpdateProposalListResponse,
  TrainingUpdateProposalPreviewRequest,
  TrainingUpdateProposalPreviewResponse,
  TrainingUpdateProposalResponse,
  TrainingVersionPromoteRequest,
  TrainingVersionRollbackRequest,
  RunResultResponse,
  RunEventsResponse,
  RunEvent,
  RunResponse,
  Sam3InteractiveRequest,
  Sam3InteractiveResponse,
  SantaBarbaraWeatherResponse,
  TrainingDatasetCreateRequest,
  TrainingDatasetItemsRequest,
  TrainingDatasetListResponse,
  TrainingDatasetResponse,
  TrainingJobControlRequest,
  TrainingJobCreateRequest,
  TrainingPreflightRequest,
  TrainingPreflightResponse,
  TrainingJobResponse,
  TrainingModelsResponse,
  UploadCaptionResponse,
  UploadViewerHistogramResponse,
  UploadViewerInfo,
  UploadFilesResponse,
} from "../types";
import { normalizeUploadViewerInfo } from "./viewerManifest";

export type ApiClientOptions = {
  baseUrl: string;
  apiKey?: string;
};

export type ChatStreamHandlers = {
  onToken?: (delta: string) => void;
  onDone?: (payload: ChatResponse) => void;
  onRunStarted?: (payload: { runId: string; model?: string | null }) => void;
  onRunEvent?: (payload: RunEvent) => void;
};

export type ChatStreamOptions = ChatStreamHandlers & {
  signal?: AbortSignal;
};

export type RunStreamOptions = ChatStreamOptions & {
  afterSequence?: number;
};

export type UpsertConversationOptions = {
  titleSource?: "manual";
};

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export type ScalarVolumePayload = {
  data: ArrayBuffer;
  width: number;
  height: number;
  depth: number;
  dtype: string;
  bytesPerVoxel: number;
  rawMin: number;
  rawMax: number;
  channel: number | null;
};

const buildUrl = (baseUrl: string, path: string, params?: Record<string, string>): string => {
  const url = new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
  }
  return url.toString();
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const V2_CHAT_THREAD_MAP_STORAGE_KEY = "bisque-ultra:v2-chat-thread-map";
const V2_CONVERSATION_STATE_METADATA_KEY = "frontend_state";
const V2_TITLE_STATE_METADATA_KEY = "title_state";

const asPlainString = (value: unknown): string => String(value ?? "");

const asTrimmedString = (value: unknown): string => String(value ?? "").trim();

const asOptionalString = (value: unknown): string | null => {
  const text = asTrimmedString(value);
  return text.length > 0 ? text : null;
};

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .map((item) => asTrimmedString(item))
        .filter((item) => item.length > 0)
    : [];

const mergeStringArrays = (...values: unknown[]): string[] => {
  const seen = new Set<string>();
  const merged: string[] = [];
  values.forEach((value) => {
    asStringArray(value).forEach((item) => {
      if (seen.has(item)) {
        return;
      }
      seen.add(item);
      merged.push(item);
    });
  });
  return merged;
};

const asFiniteNumber = (value: unknown, fallback = 0): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const asMillis = (value: unknown, fallback: number): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Date.parse(asPlainString(value));
  return Number.isFinite(parsed) ? parsed : fallback;
};

const v2ThreadMetadata = (thread: Record<string, unknown>): Record<string, unknown> =>
  isRecord(thread.metadata) ? thread.metadata : {};

const v2ThreadConversationId = (thread: Record<string, unknown>): string => {
  const metadata = v2ThreadMetadata(thread);
  return asOptionalString(metadata.conversation_id) ?? asTrimmedString(thread.thread_id);
};

const isDefaultConversationTitle = (value: unknown): boolean => {
  const normalized = asTrimmedString(value).replace(/\s+/g, " ");
  return normalized.length === 0 || normalized === "New conversation";
};

const TITLE_STOP_WORDS = new Set([
  "a",
  "about",
  "also",
  "analyze",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "build",
  "by",
  "calculate",
  "can",
  "compare",
  "compute",
  "could",
  "create",
  "durable",
  "explain",
  "find",
  "for",
  "from",
  "generate",
  "how",
  "i",
  "in",
  "into",
  "is",
  "it",
  "latest",
  "list",
  "make",
  "me",
  "of",
  "on",
  "or",
  "our",
  "please",
  "produce",
  "real",
  "run",
  "show",
  "that",
  "the",
  "this",
  "to",
  "train",
  "using",
  "visualize",
  "we",
  "what",
  "with",
  "would",
  "write",
  "you",
]);

const normalizeTitleWord = (word: string): string => {
  const trimmed = word.replace(/^[^\w]+|[^\w]+$/g, "");
  if (!trimmed) {
    return "";
  }
  if (/[A-Z]/.test(trimmed.slice(1)) || /\d|[.+/#-]/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1).toLowerCase()}`;
};

const fallbackConversationTitleFromSeed = (value: string, maxWords = 6): string => {
  const singleLine = value.replace(/\s+/g, " ").trim().replace(/^["'`]+|["'`]+$/g, "");
  if (!singleLine) {
    return "New conversation";
  }
  const candidates =
    singleLine
      .match(/[A-Za-z0-9][A-Za-z0-9.+/#-]*/g)
      ?.map(normalizeTitleWord)
      .filter(Boolean) ?? [];
  const keywords = candidates.filter(
    (word) =>
      !TITLE_STOP_WORDS.has(word.toLowerCase()) &&
      (word.length > 1 || /^[A-Z]$/.test(word) || /\d/.test(word))
  );
  const words = (keywords.length > 0 ? keywords : candidates).slice(0, Math.max(1, maxWords));
  const title = words.join(" ").trim();
  return title || "New conversation";
};

const conversationTitleFromStoredOrSeed = (
  storedTitle: unknown,
  fallbackSeed: string
): string => {
  const normalized = asTrimmedString(storedTitle).replace(/\s+/g, " ").replace(/^["'`]+|["'`]+$/g, "");
  const title = isDefaultConversationTitle(normalized)
    ? fallbackConversationTitleFromSeed(fallbackSeed)
    : normalized;
  return title.length <= 120 ? title : `${title.slice(0, 119)}...`;
};

const v2ThreadToConversationRecord = (
  thread: Record<string, unknown>,
  includeState: boolean
): ConversationRecord => {
  const now = Date.now();
  const metadata = v2ThreadMetadata(thread);
  const conversationId = v2ThreadConversationId(thread);
  const state = includeState && isRecord(metadata[V2_CONVERSATION_STATE_METADATA_KEY])
    ? metadata[V2_CONVERSATION_STATE_METADATA_KEY]
    : {};
  const createdAt = asMillis(metadata.created_at_ms, asMillis(thread.created_at, now));
  const updatedAt = asMillis(metadata.updated_at_ms, asMillis(thread.updated_at, createdAt));
  const preview = asOptionalString(metadata.preview) ?? asOptionalString(thread.summary) ?? "";
  return {
    conversation_id: conversationId,
    title: conversationTitleFromStoredOrSeed(thread.title, preview),
    created_at_ms: createdAt,
    updated_at_ms: updatedAt,
    preview,
    message_count: Math.max(0, Math.floor(asFiniteNumber(metadata.message_count, 0))),
    preferred_panel: "chat",
    running: Boolean(metadata.running),
    state,
  };
};

const v2ListRecordNeedsDurableTitleHydration = (
  thread: Record<string, unknown>,
  record: ConversationRecord
): boolean =>
  Boolean(asTrimmedString(thread.thread_id)) &&
  isDefaultConversationTitle(record.title) &&
  (!asTrimmedString(record.preview) || record.message_count === 0);

const isEmptyDefaultV2HistoryRecord = (record: ConversationRecord): boolean =>
  isDefaultConversationTitle(record.title) &&
  !asTrimmedString(record.preview) &&
  record.message_count === 0 &&
  !record.running;

const v2ConversationMetadataFromRecord = (record: ConversationRecord): Record<string, unknown> => ({
  conversation_id: record.conversation_id,
  created_at_ms: record.created_at_ms,
  updated_at_ms: record.updated_at_ms,
  preview: record.preview,
  message_count: record.message_count,
  preferred_panel: record.preferred_panel,
  running: record.running,
  [V2_CONVERSATION_STATE_METADATA_KEY]: record.state ?? {},
});

const withManualTitleState = (metadata: Record<string, unknown>): Record<string, unknown> => ({
  ...metadata,
  [V2_TITLE_STATE_METADATA_KEY]: {
    source: "manual",
    updated_at_ms: Date.now(),
  },
});

const lastUserMessageContent = (request: ChatRequest): string => {
  for (let idx = request.messages.length - 1; idx >= 0; idx -= 1) {
    const message = request.messages[idx];
    if (message.role === "user") {
      return asTrimmedString(message.content);
    }
  }
  return "";
};

const chatTitleFromRequest = (request: ChatRequest): string => {
  const raw = asTrimmedString(request.goal) || lastUserMessageContent(request) || "New conversation";
  const title = fallbackConversationTitleFromSeed(raw);
  return title.length > 80 ? `${title.slice(0, 77).trimEnd()}...` : title;
};

const chatTitleFromMessages = (request: ChatTitleRequest): string => {
  const maxWords =
    Number.isFinite(Number(request.max_words)) && Number(request.max_words) > 0
      ? Math.floor(Number(request.max_words))
      : 4;
  let raw = "";
  for (let idx = request.messages.length - 1; idx >= 0; idx -= 1) {
    const message = request.messages[idx];
    if (message.role === "user") {
      raw = asTrimmedString(message.content);
      break;
    }
  }
  return fallbackConversationTitleFromSeed(raw, maxWords);
};

const stateMessagesFromState = (state: Record<string, unknown>): Array<Record<string, unknown>> =>
  Array.isArray(state.messages) ? state.messages.filter(isRecord) : [];

const stateMessageRole = (message: Record<string, unknown>): string =>
  asTrimmedString(message.role).toLowerCase();

const stateMessageRunId = (message: Record<string, unknown>): string | null =>
  asOptionalString(message.runId ?? message.run_id);

const isActiveV2RunStatus = (status: unknown): boolean => {
  const normalized = asTrimmedString(status).toLowerCase();
  return (
    normalized === "queued" ||
    normalized === "pending" ||
    normalized === "running" ||
    normalized === "waiting_for_input" ||
    normalized === "waiting_for_task"
  );
};

const frontendStateNeedsV2RunReconciliation = (
  thread: Record<string, unknown>,
  state: Record<string, unknown>
): boolean => {
  const latestRunId = asOptionalString(thread.latest_run_id);
  if (!latestRunId) {
    return Boolean(state.sending) || Boolean(asOptionalString(state.streamingMessageId));
  }

  const messages = stateMessagesFromState(state);
  const latestAssistant = messages.find(
    (message) =>
      stateMessageRole(message) === "assistant" && stateMessageRunId(message) === latestRunId
  );
  if (!latestAssistant) {
    return true;
  }
  if (Boolean(state.sending) || Boolean(asOptionalString(state.streamingMessageId))) {
    return true;
  }
  return !asOptionalString(latestAssistant.content);
};

const threadMessageToStateMessage = (
  message: Record<string, unknown>,
  index: number,
  threadId: string,
  conversationId: string,
  fallbackTime: number
): Record<string, unknown> => ({
  id: asOptionalString(message.message_id) ?? `${threadId || conversationId}-message-${index}`,
  role: asOptionalString(message.role) ?? "user",
  content: asPlainString(message.content),
  createdAt: asMillis(message.created_at, fallbackTime),
  runId: asOptionalString(message.run_id) ?? undefined,
});

const findAssistantPatchIndex = (
  messages: Array<Record<string, unknown>>,
  latestRunId: string
): number => {
  const ownedIndex = messages.findIndex(
    (message) =>
      stateMessageRole(message) === "assistant" && stateMessageRunId(message) === latestRunId
  );
  if (ownedIndex >= 0) {
    return ownedIndex;
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (stateMessageRole(message) === "assistant" && !stateMessageRunId(message)) {
      return index;
    }
  }
  return -1;
};

const normalizeV2RunEvent = (event: Record<string, unknown>): RunEvent => {
  const payload = isRecord(event.payload) ? event.payload : {};
  const eventType = asTrimmedString(event.event_kind) || asTrimmedString(event.event_type) || "run_event";
  const sequence = asFiniteNumber(event.sequence, 0);
  return {
    event_type: eventType,
    level: asOptionalString(event.level) ?? undefined,
    payload: {
      ...payload,
      event_id: event.event_id,
      sequence,
      event_kind: eventType,
      run_id: event.run_id,
      thread_id: event.thread_id,
      node_name: event.node_name,
      task_id: event.task_id,
      checkpoint_id: event.checkpoint_id,
      scope_id: event.scope_id,
      agent_role: event.agent_role,
      message: event.message,
    },
    ts: asOptionalString(event.ts) ?? undefined,
  };
};

const progressEventFromV2RunEvent = (event: Record<string, unknown>): ProgressEvent => {
  const payload = isRecord(event.payload) ? event.payload : {};
  const eventType = asTrimmedString(event.event_kind) || asTrimmedString(event.event_type) || "run_event";
  return {
    ...payload,
    event: eventType,
    level: asOptionalString(event.level) ?? undefined,
    message: asOptionalString(event.message) ?? asOptionalString(payload.message) ?? undefined,
    ts: asOptionalString(event.ts) ?? undefined,
  };
};

const isV2TerminalEventKind = (eventKind: string): boolean =>
  eventKind === "run.completed" || eventKind === "run.failed" || eventKind === "run.canceled";

const isV2ThreadID = (value: unknown): boolean => asTrimmedString(value).startsWith("thread_");

const normalizeRunResultStatus = (
  status: unknown
): RunResultResponse["status"] => {
  const normalized = asTrimmedString(status).toLowerCase();
  if (normalized === "queued") {
    return "pending";
  }
  if (
    normalized === "pending" ||
    normalized === "running" ||
    normalized === "succeeded" ||
    normalized === "failed" ||
    normalized === "canceled"
  ) {
    return normalized;
  }
  return "pending";
};

const v2DeltaFromRunEvent = (event: Record<string, unknown>): string => {
  const payload = isRecord(event.payload) ? event.payload : {};
  return asPlainString(payload.delta ?? payload.text ?? event.message);
};

const responseTextFromV2TerminalEvent = (event: Record<string, unknown>): string => {
  const payload = isRecord(event.payload) ? event.payload : {};
  return asTrimmedString(payload.response_text ?? payload.text ?? payload.message ?? event.message);
};

const normalizeV2RunResponse = (
  run: Record<string, unknown>,
  fallback: {
    runId: string;
    responseText: string;
    progressEvents: ProgressEvent[];
    durationSeconds?: number | null;
    metadata?: Record<string, unknown> | null;
  }
): ChatResponse => {
  const runMetadata = isRecord(run.metadata) ? run.metadata : null;
  const fallbackMetadata = fallback.metadata ?? null;
  const metadata =
    runMetadata || fallbackMetadata
      ? {
          ...(fallbackMetadata ?? {}),
          ...(runMetadata ?? {}),
        }
      : null;
  return {
    run_id: asTrimmedString(run.run_id) || fallback.runId,
    model: asTrimmedString(run.model) || "deep_agents",
    response_text: asTrimmedString(run.response_text) || fallback.responseText,
    duration_seconds: asFiniteNumber(run.duration_seconds, fallback.durationSeconds ?? 0),
    progress_events: fallback.progressEvents,
    benchmark: isRecord(run.benchmark) ? run.benchmark : null,
    metadata,
  };
};

async function parseError(response: Response): Promise<never> {
  const text = await response.text();
  let detail: unknown = text;
  try {
    detail = JSON.parse(text);
  } catch {
    // keep raw text
  }
  throw new ApiError(`Request failed with status ${response.status}`, response.status, detail);
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private v2ThreadIdsByConversation: Map<string, string> | null = null;
  private readonly v2ArtifactIdsByRunPath = new Map<string, string>();

  constructor(options: ApiClientOptions) {
    this.baseUrl = options.baseUrl;
    this.apiKey = options.apiKey?.trim() || undefined;
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = { ...(extra ?? {}) };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    return headers;
  }

  private browserStorage(): Storage | null {
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        return window.localStorage;
      }
      if (typeof localStorage !== "undefined") {
        return localStorage;
      }
    } catch {
      // Storage can be unavailable in private browsing or during tests.
    }
    return null;
  }

  private v2ThreadMap(): Map<string, string> {
    if (this.v2ThreadIdsByConversation) {
      return this.v2ThreadIdsByConversation;
    }
    const map = new Map<string, string>();
    const storage = this.browserStorage();
    if (storage) {
      try {
        const parsed = JSON.parse(storage.getItem(V2_CHAT_THREAD_MAP_STORAGE_KEY) ?? "{}");
        if (isRecord(parsed)) {
          Object.entries(parsed).forEach(([key, value]) => {
            const threadId = asTrimmedString(value);
            if (isV2ThreadID(threadId)) {
              map.set(key, threadId);
            }
          });
        }
      } catch {
        // A corrupt cache should not block chat.
      }
    }
    this.v2ThreadIdsByConversation = map;
    return map;
  }

  private rememberV2Thread(conversationId: string, threadId: string): void {
    const conversationKey = asTrimmedString(conversationId);
    const resolvedThreadId = asTrimmedString(threadId);
    if (!conversationKey || !isV2ThreadID(resolvedThreadId)) {
      return;
    }
    const map = this.v2ThreadMap();
    map.set(conversationKey, resolvedThreadId);
    const storage = this.browserStorage();
    if (!storage) {
      return;
    }
    try {
      storage.setItem(
        V2_CHAT_THREAD_MAP_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(map.entries()))
      );
    } catch {
      // Non-fatal; the current in-memory map still works for this session.
    }
  }

  private forgetV2Thread(conversationId: string): void {
    const conversationKey = asTrimmedString(conversationId);
    if (!conversationKey) {
      return;
    }
    const map = this.v2ThreadMap();
    map.delete(conversationKey);
    const storage = this.browserStorage();
    if (!storage) {
      return;
    }
    try {
      storage.setItem(
        V2_CHAT_THREAD_MAP_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(map.entries()))
      );
    } catch {
      // Non-fatal; a stale persisted cache can be overwritten on the next success.
    }
  }

  private async getExistingV2ThreadID(threadId: string): Promise<string | null> {
    if (!isV2ThreadID(threadId)) {
      return null;
    }
    try {
      const thread = await this.fetchJson<Record<string, unknown>>(
        `/v2/threads/${encodeURIComponent(threadId)}`,
        { method: "GET" }
      );
      return asTrimmedString(thread.thread_id) || threadId;
    } catch (error) {
      if (error instanceof ApiError && (error.status === 400 || error.status === 404)) {
        return null;
      }
      throw error;
    }
  }

  private conversationIdFromRequest(request: ChatRequest): string {
    return asTrimmedString(request.conversation_id) || `local-${Date.now().toString(36)}`;
  }

  private async listV2Conversations(
    limit: number,
    offset: number,
    includeState: boolean
  ): Promise<ConversationListResponse> {
    const requestedLimit = Math.max(1, Math.floor(Number(limit) || 25));
    const requestedOffset = Math.max(0, Math.floor(Number(offset) || 0));
    const payload = await this.fetchJson<Record<string, unknown>>(
      "/v2/threads",
      { method: "GET" },
      {
        limit: String(requestedLimit),
        offset: String(requestedOffset),
      }
    );
    const threads = Array.isArray(payload.threads)
      ? payload.threads.filter(isRecord)
      : [];
    const pageCount = Math.max(
      threads.length,
      Math.floor(asFiniteNumber(payload.count, threads.length))
    );
    const conversations = (
      await Promise.all(
        threads.map(async (thread): Promise<ConversationRecord | null> => {
          let record = v2ThreadToConversationRecord(thread, includeState);
          if (v2ListRecordNeedsDurableTitleHydration(thread, record)) {
            const hydrated = await this.hydrateV2ConversationRecord(thread, record);
            if (!includeState && isEmptyDefaultV2HistoryRecord(hydrated)) {
              return null;
            }
            record = includeState ? hydrated : { ...hydrated, state: {} };
          } else if (!includeState && isEmptyDefaultV2HistoryRecord(record)) {
            return null;
          }
          const threadId = asTrimmedString(thread.thread_id);
          if (record.conversation_id && threadId) {
            this.rememberV2Thread(record.conversation_id, threadId);
          }
          return record;
        })
      )
    ).filter((record): record is ConversationRecord => record !== null);
    const totalCount = Math.max(
      pageCount,
      Math.floor(asFiniteNumber(payload.total_count ?? payload.count, pageCount))
    );
    return {
      count: pageCount,
      total_count: totalCount,
      limit: requestedLimit,
      offset: requestedOffset,
      has_more: requestedOffset + pageCount < totalCount,
      conversations,
    };
  }

  private async getV2Conversation(conversationId: string): Promise<ConversationRecord> {
    const normalizedConversationId = asTrimmedString(conversationId);
    const knownThreadId = this.v2ThreadMap().get(normalizedConversationId);
    const candidateThreadId = knownThreadId || (
      normalizedConversationId.startsWith("thread_") ? normalizedConversationId : ""
    );
    if (candidateThreadId) {
      const thread = await this.fetchJson<Record<string, unknown>>(
        `/v2/threads/${encodeURIComponent(candidateThreadId)}`,
        { method: "GET" }
      );
      const record = v2ThreadToConversationRecord(thread, true);
      this.rememberV2Thread(record.conversation_id, asTrimmedString(thread.thread_id));
      return this.hydrateV2ConversationRecord(thread, record);
    }

    const payload = await this.fetchJson<Record<string, unknown>>(
      "/v2/threads",
      { method: "GET" },
      {
        limit: "1000",
        offset: "0",
      }
    );
    const foundThread = Array.isArray(payload.threads)
      ? payload.threads.filter(isRecord).find(
          (thread) => v2ThreadConversationId(thread) === normalizedConversationId
        )
      : null;
    if (!foundThread) {
      throw new ApiError("Conversation was not found", 404, null);
    }
    const record = v2ThreadToConversationRecord(foundThread, true);
    this.rememberV2Thread(record.conversation_id, asTrimmedString(foundThread.thread_id));
    return this.hydrateV2ConversationRecord(foundThread, record);
  }

  private async hydrateV2ConversationRecord(
    thread: Record<string, unknown>,
    record: ConversationRecord
  ): Promise<ConversationRecord> {
    const existingState = isRecord(record.state) ? record.state : {};
    const hasExistingState = Object.keys(existingState).length > 0;
    const latestRunId = asOptionalString(thread.latest_run_id);

    if (
      hasExistingState &&
      !frontendStateNeedsV2RunReconciliation(thread, existingState)
    ) {
      return record;
    }

    if (hasExistingState && !latestRunId) {
      return {
        ...record,
        running: false,
        state: {
          ...existingState,
          sending: false,
          streamingMessageId: null,
        },
      };
    }

    const threadId = asTrimmedString(thread.thread_id);
    const messages: Array<Record<string, unknown>> = [];
    if (threadId) {
      try {
        const payload = await this.fetchJson<Record<string, unknown>>(
          `/v2/threads/${encodeURIComponent(threadId)}/messages`,
          { method: "GET" }
        );
        if (Array.isArray(payload.messages)) {
          messages.push(...payload.messages.filter(isRecord));
        }
      } catch {
        // A V2 thread without message hydration should still open instead of spinning forever.
      }
    }

    const stateMessages: Array<Record<string, unknown>> = hasExistingState
      ? stateMessagesFromState(existingState).map((message) => ({ ...message }))
      : messages.map((message, index) =>
          threadMessageToStateMessage(
            message,
            index,
            threadId,
            record.conversation_id,
            record.updated_at_ms
          )
        );

    if (hasExistingState && stateMessages.length === 0 && messages.length > 0) {
      stateMessages.push(
        ...messages.map((message, index) =>
          threadMessageToStateMessage(
            message,
            index,
            threadId,
            record.conversation_id,
            record.updated_at_ms
          )
        )
      );
    }

    let latestRun: Record<string, unknown> | null = null;
    if (latestRunId) {
      try {
        latestRun = await this.fetchJson<Record<string, unknown>>(
          `/v2/runs/${encodeURIComponent(latestRunId)}`,
          { method: "GET" }
        );
      } catch {
        // Keep the durable message reconstruction if the latest run record is unavailable.
      }
    }

    const latestRunActive = latestRun ? isActiveV2RunStatus(latestRun.status) : false;
    const latestRunMetadata = latestRun && isRecord(latestRun.metadata) ? latestRun.metadata : null;
    const latestRunResponseText = latestRun ? asOptionalString(latestRun.response_text) : null;
    const durableLatestAssistant = latestRunId
      ? messages.find(
          (message) =>
            asTrimmedString(message.role).toLowerCase() === "assistant" &&
            asOptionalString(message.run_id) === latestRunId
        )
      : null;
    const latestAssistantText =
      latestRunResponseText ?? asOptionalString(durableLatestAssistant?.content) ?? "";

    if (latestRunId) {
      if (!stateMessages.some((message) => stateMessageRole(message) === "assistant" && stateMessageRunId(message) === latestRunId)) {
        const patchIndex = findAssistantPatchIndex(stateMessages, latestRunId);
        const createdAt = asMillis(
          durableLatestAssistant?.created_at,
          asMillis(latestRun?.completed_at, asMillis(latestRun?.updated_at, record.updated_at_ms))
        );
        if (patchIndex >= 0) {
          const existing = stateMessages[patchIndex];
          stateMessages[patchIndex] = {
            ...existing,
            role: "assistant",
            content: latestAssistantText || asPlainString(existing.content),
            createdAt: asMillis(existing.createdAt, createdAt),
            runId: latestRunId,
            responseMetadata: latestRunMetadata ?? existing.responseMetadata ?? null,
          };
        } else if (latestAssistantText || latestRunActive) {
          stateMessages.push({
            id: `${latestRunId}-assistant`,
            role: "assistant",
            content: latestAssistantText,
            createdAt,
            runId: latestRunId,
            responseMetadata: latestRunMetadata,
          });
        }
      } else if (latestAssistantText || latestRunMetadata || latestRunActive) {
        const patchIndex = findAssistantPatchIndex(stateMessages, latestRunId);
        if (patchIndex >= 0) {
          const existing = stateMessages[patchIndex];
          stateMessages[patchIndex] = {
            ...existing,
            content: latestAssistantText || asPlainString(existing.content),
            runId: latestRunId,
            responseMetadata: latestRunMetadata ?? existing.responseMetadata ?? null,
          };
        }
      }
    }

    const preview = record.preview || asOptionalString(
      [...stateMessages].reverse().find((message) => message.role === "user")?.content
    ) || "";
    return {
      ...record,
      title: conversationTitleFromStoredOrSeed(record.title, preview),
      preview,
      message_count: stateMessages.length || record.message_count,
      running: latestRunActive,
      state: {
        ...(hasExistingState ? existingState : {}),
        preferredPanel: asOptionalString(existingState.preferredPanel) ?? "chat",
        prompt: "",
        messages: stateMessages,
        uploadedFiles: hasExistingState ? existingState.uploadedFiles ?? [] : [],
        stagedUploadFileIds: hasExistingState ? existingState.stagedUploadFileIds ?? [] : [],
        activeSelectionContext: hasExistingState ? existingState.activeSelectionContext ?? null : null,
        failedUploadPreviewIds: hasExistingState ? existingState.failedUploadPreviewIds ?? {} : {},
        bisqueLinksByFileId: hasExistingState ? existingState.bisqueLinksByFileId ?? {} : {},
        composerWorkflowPreset: hasExistingState ? existingState.composerWorkflowPreset ?? null : null,
        selectionImportPending: false,
        sending: latestRunActive,
        chatError: latestRunActive ? null : existingState.chatError ?? null,
        streamingMessageId: latestRunActive
          ? asOptionalString(
              stateMessages.find(
                (message) =>
                  stateMessageRole(message) === "assistant" &&
                  stateMessageRunId(message) === latestRunId
              )?.id
            )
          : null,
      },
    };
  }

  private async upsertV2Conversation(
    record: ConversationRecord,
    options?: UpsertConversationOptions
  ): Promise<ConversationRecord> {
    const conversationId = asTrimmedString(record.conversation_id);
    const threadId = this.v2ThreadMap().get(conversationId);
    const baseMetadata = v2ConversationMetadataFromRecord(record);
    const metadata =
      options?.titleSource === "manual" ? withManualTitleState(baseMetadata) : baseMetadata;

    if (threadId) {
      const response = await fetch(
        buildUrl(this.baseUrl, `/v2/threads/${encodeURIComponent(threadId)}`),
        {
          method: "PUT",
          headers: this.headers({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            title: record.title,
            metadata,
          }),
          credentials: "include",
        }
      );
      if (response.ok) {
        const thread = (await response.json()) as Record<string, unknown>;
        const updated = v2ThreadToConversationRecord(thread, true);
        this.rememberV2Thread(updated.conversation_id, asTrimmedString(thread.thread_id));
        return updated;
      }
      if (response.status !== 404 && response.status !== 405) {
        return parseError(response);
      }
      return record;
    }

    const created = await this.fetchJson<Record<string, unknown>>("/v2/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: record.title,
        metadata,
      }),
    });
    const createdThreadId = asTrimmedString(created.thread_id);
    if (createdThreadId) {
      this.rememberV2Thread(conversationId, createdThreadId);
    }
    return v2ThreadToConversationRecord(created, true);
  }

  private async deleteV2Conversation(
    conversationId: string
  ): Promise<{ deleted: boolean; conversation_id: string }> {
    const normalizedConversationId = asTrimmedString(conversationId);
    const knownThreadId = this.v2ThreadMap().get(normalizedConversationId);
    let threadId =
      knownThreadId ||
      (normalizedConversationId.startsWith("thread_") ? normalizedConversationId : "");

    if (!threadId) {
      const payload = await this.fetchJson<Record<string, unknown>>(
        "/v2/threads",
        { method: "GET" },
        {
          limit: "1000",
          offset: "0",
        }
      );
      const foundThread = Array.isArray(payload.threads)
        ? payload.threads.filter(isRecord).find(
            (thread) => v2ThreadConversationId(thread) === normalizedConversationId
          )
        : null;
      threadId = foundThread ? asTrimmedString(foundThread.thread_id) : "";
    }

    if (!threadId) {
      throw new ApiError("Conversation was not found", 404, null);
    }

    const response = await fetch(buildUrl(this.baseUrl, `/v2/threads/${encodeURIComponent(threadId)}`), {
      method: "DELETE",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok && response.status !== 404) {
      return parseError(response);
    }
    this.forgetV2Thread(normalizedConversationId);
    return {
      deleted: true,
      conversation_id: normalizedConversationId,
    };
  }

  private v2ArtifactKey(runId: string, path: string): string {
    return `${runId}\n${path}`;
  }

  private rememberV2Artifact(runId: string, artifactId: string, paths: Array<unknown>): void {
    const resolvedRunId = asTrimmedString(runId);
    const resolvedArtifactId = asTrimmedString(artifactId);
    if (!resolvedRunId || !resolvedArtifactId) {
      return;
    }
    paths
      .map((path) => asTrimmedString(path))
      .filter((path) => path.length > 0)
      .forEach((path) => {
        this.v2ArtifactIdsByRunPath.set(this.v2ArtifactKey(resolvedRunId, path), resolvedArtifactId);
      });
  }

  private rememberV2ArtifactEvent(event: Record<string, unknown>): void {
    const payload = isRecord(event.payload) ? event.payload : {};
    this.rememberV2Artifact(asTrimmedString(event.run_id), asTrimmedString(payload.artifact_id), [
      payload.path,
      payload.relative_path,
      payload.source_path,
      payload.preview_path,
    ]);
  }

  private async fetchJson<T>(
    path: string,
    init: RequestInit = {},
    params?: Record<string, string>
  ): Promise<T> {
    const initHeaders = Object.fromEntries(new Headers(init.headers).entries());
    const response = await fetch(buildUrl(this.baseUrl, path, params), {
      ...init,
      headers: this.headers(initHeaders),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as T;
  }

  private async ensureV2Thread(request: ChatRequest, conversationId: string): Promise<string> {
    const existing = this.v2ThreadMap().get(conversationId);
    if (existing) {
      const existingThreadId = await this.getExistingV2ThreadID(existing);
      if (existingThreadId) {
        this.rememberV2Thread(conversationId, existingThreadId);
        return existingThreadId;
      }
      this.forgetV2Thread(conversationId);
    }

    if (conversationId.startsWith("thread_")) {
      const threadId = await this.getExistingV2ThreadID(conversationId);
      if (threadId) {
        this.rememberV2Thread(conversationId, threadId);
        return threadId;
      }
    }

    const created = await this.fetchJson<Record<string, unknown>>("/v2/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: chatTitleFromRequest(request),
        metadata: {
          conversation_id: conversationId,
          frontend_bridge: "v2-chat",
          [V2_TITLE_STATE_METADATA_KEY]: {
            source: "auto",
            strategy: "initial_request",
          },
        },
      }),
    });
    const threadId = asTrimmedString(created.thread_id);
    if (!threadId) {
      throw new ApiError("V2 thread creation did not return a thread id", 502, created);
    }
    this.rememberV2Thread(conversationId, threadId);
    return threadId;
  }

  private async createV2Run(
    threadId: string,
    request: ChatRequest,
    idempotencyKey: string | null,
    signal?: AbortSignal
  ): Promise<Record<string, unknown>> {
    return this.fetchJson<Record<string, unknown>>(
      `/v2/threads/${encodeURIComponent(threadId)}/runs`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
        },
        body: JSON.stringify(this.buildV2RunRequest(request)),
        signal,
      }
    );
  }

  private buildV2RunRequest(request: ChatRequest): Record<string, unknown> {
    const selectionContext = isRecord(request.selection_context) ? request.selection_context : {};
    return {
      goal: asOptionalString(request.goal) ?? lastUserMessageContent(request),
      messages: request.messages.map((message) => ({
        role: message.role,
        content: asPlainString(message.content),
      })),
      file_ids: request.file_ids ?? [],
      resource_uris: mergeStringArrays(request.resource_uris, selectionContext.resource_uris),
      dataset_uris: mergeStringArrays(request.dataset_uris, selectionContext.dataset_uris),
      selected_tool_names: request.selected_tool_names ?? [],
      knowledge_context: request.knowledge_context ?? null,
      selection_context: request.selection_context ?? null,
      workflow_hint: request.workflow_hint ?? null,
      reasoning_mode: request.reasoning_mode ?? "auto",
      budgets: request.budgets ?? null,
      benchmark: request.benchmark ?? null,
      idempotency_key: asOptionalString(request.idempotency_key),
      metadata: {
        conversation_id: request.conversation_id ?? null,
        uploaded_files: request.uploaded_files ?? [],
        frontend_bridge: "v2-chat",
      },
    };
  }

  private async getV2Run(runId: string): Promise<Record<string, unknown> | null> {
    try {
      return await this.fetchJson<Record<string, unknown>>(
        `/v2/runs/${encodeURIComponent(runId)}`,
        { method: "GET" }
      );
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async health(): Promise<{ status: string; ts: string }> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/health"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as { status: string; ts: string };
  }

  async getPublicConfig(): Promise<PublicConfigResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/config/public"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PublicConfigResponse;
  }

  async getSantaBarbaraWeather(): Promise<SantaBarbaraWeatherResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/fun/weather/santa-barbara"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as SantaBarbaraWeatherResponse;
  }

  async getAdminOverview(options?: {
    topUsers?: number;
    issueLimit?: number;
  }): Promise<AdminOverviewResponse> {
    const params: Record<string, string> = {
      top_users: String(Math.max(1, Number(options?.topUsers) || 8)),
      issue_limit: String(Math.max(1, Number(options?.issueLimit) || 12)),
    };
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/overview", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminOverviewResponse;
  }

  async listAdminUsers(options?: {
    limit?: number;
    query?: string;
  }): Promise<AdminUserListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Number(options?.limit) || 200)),
    };
    const query = String(options?.query || "").trim();
    if (query) {
      params.q = query;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/users", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminUserListResponse;
  }

  async listAdminOrganizations(options?: {
    limit?: number;
    query?: string;
  }): Promise<AdminOrganizationListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Number(options?.limit) || 200)),
    };
    const query = String(options?.query || "").trim();
    if (query) {
      params.q = query;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/orgs", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminOrganizationListResponse;
  }

  async createAdminOrganization(
    payload: AdminCreateOrganizationRequest
  ): Promise<AdminOrganization> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/orgs"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      credentials: "include",
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminOrganization;
  }

  async createAdminUser(payload: AdminCreateUserRequest): Promise<AdminUserAccount> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/users"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      credentials: "include",
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminUserAccount;
  }

  async updateAdminUserStatus(
    userId: string,
    status: AdminUserStatus
  ): Promise<AdminUserAccount> {
    const payload: AdminUpdateUserStatusRequest = { status };
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/admin/users/${encodeURIComponent(userId)}/status`),
      {
        method: "PATCH",
        headers: this.headers({ "Content-Type": "application/json" }),
        credentials: "include",
        body: JSON.stringify(payload),
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminUserAccount;
  }

  async deleteAdminUser(userId: string): Promise<AdminUserAccount> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/admin/users/${encodeURIComponent(userId)}`),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminUserAccount;
  }

  async listAdminRuns(options?: {
    limit?: number;
    offset?: number;
    status?: string;
    userId?: string;
    query?: string;
  }): Promise<AdminRunListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Number(options?.limit) || 200)),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const status = String(options?.status || "").trim();
    if (status) {
      params.status = status;
    }
    const userId = String(options?.userId || "").trim();
    if (userId) {
      params.user_id = userId;
    }
    const query = String(options?.query || "").trim();
    if (query) {
      params.q = query;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/runs", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminRunListResponse;
  }

  async listAdminIssues(limit = 25): Promise<AdminIssueListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, "/v2/admin/issues", {
        limit: String(Math.max(1, Number(limit) || 25)),
      }),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminIssueListResponse;
  }

  async cancelAdminRun(runId: string): Promise<AdminRunActionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/admin/runs/${encodeURIComponent(runId)}/cancel`),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminRunActionResponse;
  }

  async requeueAdminRun(runId: string, reason = "admin requeue"): Promise<AdminRunActionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/admin/runs/${encodeURIComponent(runId)}/requeue`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        credentials: "include",
        body: JSON.stringify({ reason }),
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminRunActionResponse;
  }

  async deleteAdminConversation(
    conversationId: string,
    userId: string
  ): Promise<AdminConversationActionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/admin/conversations/${encodeURIComponent(conversationId)}`, {
        user_id: userId,
      }),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminConversationActionResponse;
  }

  async listTrainingModels(): Promise<TrainingModelsResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/models"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingModelsResponse;
  }

  async syncPrairieActiveLearningDataset(): Promise<PrairieSyncResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/prairie/sync"), {
      method: "POST",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieSyncResponse;
  }

  async getPrairieActiveLearningStatus(): Promise<PrairieStatusResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/prairie/status"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieStatusResponse;
  }

  async runPrairieBenchmark(
    request?: PrairieBenchmarkRunRequest
  ): Promise<PrairieBenchmarkRunResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/prairie/benchmark/run"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request ?? { mode: "canonical_only" }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieBenchmarkRunResponse;
  }

  async requestPrairieRetrain(request: PrairieRetrainRequest): Promise<TrainingJobResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/prairie/retrain-request"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async listPrairieRetrainRequests(): Promise<PrairieRetrainListResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/prairie/retrain-requests"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieRetrainListResponse;
  }

  async createTrainingDataset(
    request: TrainingDatasetCreateRequest
  ): Promise<TrainingDatasetResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/datasets"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDatasetResponse;
  }

  async listTrainingDatasets(options?: { limit?: number }): Promise<TrainingDatasetListResponse> {
    const params: Record<string, string> = {};
    if (typeof options?.limit === "number" && Number.isFinite(options.limit)) {
      params.limit = String(Math.max(1, Math.floor(options.limit)));
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        "/v2/training/datasets",
        Object.keys(params).length > 0 ? params : undefined
      ),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDatasetListResponse;
  }

  async getTrainingDataset(datasetId: string): Promise<TrainingDatasetResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/datasets/${encodeURIComponent(datasetId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDatasetResponse;
  }

  async assignTrainingDatasetItems(
    datasetId: string,
    request: TrainingDatasetItemsRequest
  ): Promise<TrainingDatasetResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/datasets/${encodeURIComponent(datasetId)}/items`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDatasetResponse;
  }

  async createTrainingJob(request: TrainingJobCreateRequest): Promise<TrainingJobResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/jobs"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async previewTrainingJob(
    request: TrainingPreflightRequest
  ): Promise<TrainingPreflightResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/preflight"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingPreflightResponse;
  }

  async getTrainingJob(jobId: string): Promise<TrainingJobResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/jobs/${encodeURIComponent(jobId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async controlTrainingJob(
    jobId: string,
    request: TrainingJobControlRequest
  ): Promise<TrainingJobResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/jobs/${encodeURIComponent(jobId)}/control`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async createInferenceJob(request: InferenceJobCreateRequest): Promise<TrainingJobResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/inference/jobs"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async getInferenceJobResult(jobId: string): Promise<TrainingJobResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/inference/jobs/${encodeURIComponent(jobId)}/result`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingJobResponse;
  }

  async getModelHealth(): Promise<ModelHealthResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/model-health"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ModelHealthResponse;
  }

  async getAdminModelHealth(): Promise<ModelHealthResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/model-health"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ModelHealthResponse;
  }

  async createTrainingDomain(
    request: TrainingDomainCreateRequest
  ): Promise<TrainingDomainRecord> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/domains"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDomainRecord;
  }

  async listTrainingDomains(limit = 200): Promise<TrainingDomainListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, "/v2/training/domains", { limit: String(limit) }),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingDomainListResponse;
  }

  async listDomainLineages(
    domainId: string,
    options?: { limit?: number }
  ): Promise<TrainingLineageListResponse> {
    const params: Record<string, string> = {};
    if (options?.limit != null) {
      params.limit = String(options.limit);
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/training/domains/${encodeURIComponent(domainId)}/lineages`,
        Object.keys(params).length > 0 ? params : undefined
      ),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingLineageListResponse;
  }

  async forkTrainingLineage(
    lineageId: string,
    request: TrainingForkLineageRequest
  ): Promise<TrainingLineageRecord> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/lineages/${encodeURIComponent(lineageId)}/fork`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingLineageRecord;
  }

  async listLineageVersions(
    lineageId: string,
    options?: { limit?: number }
  ): Promise<TrainingModelVersionListResponse> {
    const params: Record<string, string> = {};
    if (options?.limit != null) {
      params.limit = String(options.limit);
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/training/lineages/${encodeURIComponent(lineageId)}/versions`,
        Object.keys(params).length > 0 ? params : undefined
      ),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingModelVersionListResponse;
  }

  async previewTrainingUpdateProposal(
    request: TrainingUpdateProposalPreviewRequest
  ): Promise<TrainingUpdateProposalPreviewResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/update-proposals/preview"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingUpdateProposalPreviewResponse;
  }

  async listTrainingUpdateProposals(options?: {
    lineageId?: string;
    status?: string;
    limit?: number;
  }): Promise<TrainingUpdateProposalListResponse> {
    const params: Record<string, string> = {};
    if (options?.lineageId) {
      params.lineage_id = options.lineageId;
    }
    if (options?.status) {
      params.status = options.status;
    }
    if (options?.limit != null) {
      params.limit = String(options.limit);
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        "/v2/training/update-proposals",
        Object.keys(params).length > 0 ? params : undefined
      ),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingUpdateProposalListResponse;
  }

  async approveTrainingUpdateProposal(
    proposalId: string,
    request: TrainingUpdateProposalDecisionRequest
  ): Promise<TrainingUpdateProposalResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/update-proposals/${encodeURIComponent(proposalId)}/approve`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingUpdateProposalResponse;
  }

  async rejectTrainingUpdateProposal(
    proposalId: string,
    request: TrainingUpdateProposalDecisionRequest
  ): Promise<TrainingUpdateProposalResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/update-proposals/${encodeURIComponent(proposalId)}/reject`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingUpdateProposalResponse;
  }

  async promoteTrainingModelVersion(
    versionId: string,
    request: TrainingVersionPromoteRequest
  ): Promise<TrainingModelVersionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/model-versions/${encodeURIComponent(versionId)}/promote`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingModelVersionResponse;
  }

  async rollbackTrainingModelVersion(
    versionId: string,
    request: TrainingVersionRollbackRequest
  ): Promise<TrainingModelVersionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/model-versions/${encodeURIComponent(versionId)}/rollback`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingModelVersionResponse;
  }

  async createTrainingMergeRequest(
    request: TrainingMergeRequestCreateRequest
  ): Promise<TrainingMergeRequestResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/training/merge-requests"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingMergeRequestResponse;
  }

  async approveTrainingMergeRequest(
    mergeId: string,
    request: TrainingMergeRequestDecisionRequest
  ): Promise<TrainingMergeRequestResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/merge-requests/${encodeURIComponent(mergeId)}/approve`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingMergeRequestResponse;
  }

  async rejectTrainingMergeRequest(
    mergeId: string,
    request: TrainingMergeRequestDecisionRequest
  ): Promise<TrainingMergeRequestResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/merge-requests/${encodeURIComponent(mergeId)}/reject`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(request),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingMergeRequestResponse;
  }

  async listTrainingMergeRequests(options?: {
    status?: string;
    limit?: number;
  }): Promise<TrainingMergeRequestListResponse> {
    const params: Record<string, string> = {};
    if (options?.status) {
      params.status = options.status;
    }
    if (options?.limit != null) {
      params.limit = String(options.limit);
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        "/v2/training/merge-requests",
        Object.keys(params).length > 0 ? params : undefined
      ),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TrainingMergeRequestListResponse;
  }

  async getBisqueSession(): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/session"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async loginBisque(payload: BisqueAuthLoginRequest): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/login"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async startHostedAuth(): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/login"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({}),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async logoutBisque(): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/logout"), {
      method: "POST",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async unlinkBisqueAccount(): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/bisque/unlink"), {
      method: "POST",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async requestAccount(payload: AccountRequestPayload): Promise<AccountRequestResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/request-account"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AccountRequestResponse;
  }

  async continueAsGuest(payload: BisqueGuestAuthRequest): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/auth/guest"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.chatStream(request);
  }

  async chatTitle(request: ChatTitleRequest): Promise<ChatTitleResponse> {
    return {
      title: chatTitleFromMessages(request),
      model: "frontend-local",
      strategy: "fallback",
    };
  }

  async chatStream(request: ChatRequest, options?: ChatStreamOptions): Promise<ChatResponse> {
    const conversationId = this.conversationIdFromRequest(request);
    let threadId = await this.ensureV2Thread(request, conversationId);
    const idempotencyKey = asOptionalString(request.idempotency_key);
    let createdRun: Record<string, unknown>;
    try {
      createdRun = await this.createV2Run(threadId, request, idempotencyKey, options?.signal);
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) {
        throw error;
      }
      this.forgetV2Thread(conversationId);
      const recoveredThreadId = await this.ensureV2Thread(request, conversationId);
      if (recoveredThreadId === threadId) {
        throw error;
      }
      threadId = recoveredThreadId;
      createdRun = await this.createV2Run(threadId, request, idempotencyKey, options?.signal);
    }
    const runId = asTrimmedString(createdRun.run_id);
    if (!runId) {
      throw new ApiError("V2 run creation did not return a run id", 502, createdRun);
    }
    options?.onRunStarted?.({
      runId,
      model: asOptionalString(createdRun.model) ?? "deep_agents",
    });

    return this.consumeV2RunEventStream(runId, createdRun, options);
  }

  async resumeRunStream(runId: string, options?: RunStreamOptions): Promise<ChatResponse> {
    const normalizedRunId = asTrimmedString(runId);
    if (!normalizedRunId) {
      throw new ApiError("Run stream resume requires a run id", 400, null);
    }
    return this.consumeV2RunEventStream(normalizedRunId, { run_id: normalizedRunId }, options);
  }

  private async consumeV2RunEventStream(
    runId: string,
    fallbackRun: Record<string, unknown>,
    options?: RunStreamOptions
  ): Promise<ChatResponse> {
    const streamParams: Record<string, string> = { stream: "true" };
    const afterSequence = Math.max(0, Math.floor(Number(options?.afterSequence ?? 0)));
    streamParams.after_sequence = String(afterSequence);

    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/runs/${encodeURIComponent(runId)}/events`, streamParams),
      {
        method: "GET",
        headers: this.headers({ Accept: "text/event-stream" }),
        credentials: "include",
        signal: options?.signal,
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    if (!response.body) {
      throw new ApiError("Run event stream did not include a readable body", 502, null);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const progressEvents: ProgressEvent[] = [];
    let buffer = "";
    let streamedText = "";
    let terminalEventSeen = false;
    let terminalStatus: "succeeded" | "failed" | "canceled" | null = null;
    let terminalDetail: unknown = null;
    let terminalResponseText = "";

    const handleStreamEvent = (eventName: string, payload: unknown): void => {
      if (eventName === "heartbeat") {
        return;
      }
      if (eventName === "token" && isRecord(payload)) {
        const delta = asPlainString(payload.delta);
        if (delta) {
          streamedText += delta;
          options?.onToken?.(delta);
        }
        return;
      }
      if (eventName === "error") {
        terminalEventSeen = true;
        terminalStatus = "failed";
        terminalDetail = payload;
        return;
      }
      if (eventName !== "run_event" || !isRecord(payload)) {
        return;
      }

      const eventKind =
        asTrimmedString(payload.event_kind) || asTrimmedString(payload.event_type) || "run_event";
      if (eventKind === "artifact.created") {
        this.rememberV2ArtifactEvent(payload);
      }

      const normalized = normalizeV2RunEvent(payload);
      progressEvents.push(progressEventFromV2RunEvent(payload));
      options?.onRunEvent?.(normalized);

      if (eventKind === "message.delta") {
        const delta = v2DeltaFromRunEvent(payload);
        if (delta) {
          streamedText += delta;
          options?.onToken?.(delta);
        }
      }

      if (!isV2TerminalEventKind(eventKind)) {
        return;
      }
      terminalEventSeen = true;
      terminalResponseText = responseTextFromV2TerminalEvent(payload);
      terminalStatus =
        eventKind === "run.completed" ? "succeeded" : eventKind === "run.canceled" ? "canceled" : "failed";
      terminalDetail = isRecord(payload.payload) ? payload.payload : payload;
    };

    const parseEventBlock = (rawBlock: string): void => {
      const block = rawBlock.trim();
      if (!block) {
        return;
      }
      let eventName = "message";
      const dataLines: string[] = [];
      block.split("\n").forEach((line) => {
        if (line.startsWith(":")) {
          return;
        }
        if (line.startsWith("event:")) {
          eventName = line.slice("event:".length).trim() || "message";
          return;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice("data:".length).trimStart());
        }
      });
      if (dataLines.length === 0) {
        return;
      }
      const rawData = dataLines.join("\n");
      let payload: unknown = rawData;
      try {
        payload = JSON.parse(rawData);
      } catch {
        // keep raw text payload
      }
      handleStreamEvent(eventName, payload);
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        while (true) {
          const boundary = buffer.indexOf("\n\n");
          if (boundary === -1) {
            break;
          }
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          parseEventBlock(block);
          if (terminalEventSeen) {
            break;
          }
        }
        if (terminalEventSeen) {
          break;
        }
      }
      if (!terminalEventSeen) {
        buffer += decoder.decode().replace(/\r\n/g, "\n");
      }
      if (buffer.trim().length > 0) {
        parseEventBlock(buffer);
      }
    } finally {
      if (terminalEventSeen) {
        try {
          await reader.cancel();
        } catch {
          // The stream is already complete or no longer cancelable.
        }
      }
      reader.releaseLock();
    }

    if (terminalStatus === "failed") {
      throw new ApiError("Run failed", 500, terminalDetail);
    }
    if (terminalStatus === "canceled") {
      throw new ApiError("Run canceled", 499, terminalDetail);
    }

    const finalRun = await this.getV2Run(runId);
    if (!terminalEventSeen) {
      const finalStatus = normalizeRunResultStatus(finalRun?.status ?? fallbackRun.status);
      if (finalStatus === "pending" || finalStatus === "running") {
        throw new ApiError("Run stream ended before a terminal event", 503, {
          run_id: runId,
          status: finalStatus,
          response_text: streamedText,
        });
      }
      if (finalStatus === "failed") {
        throw new ApiError("Run failed", 500, finalRun ?? fallbackRun);
      }
      if (finalStatus === "canceled") {
        throw new ApiError("Run canceled", 499, finalRun ?? fallbackRun);
      }
    }
    const responseText =
      terminalResponseText || asTrimmedString(finalRun?.response_text) || streamedText;
    const completedPayload = normalizeV2RunResponse(finalRun ?? fallbackRun, {
      runId,
      responseText,
      progressEvents,
      metadata: isRecord(terminalDetail) ? terminalDetail : null,
    });
    options?.onDone?.(completedPayload);
    return completedPayload;
  }

  async uploadFiles(files: File[]): Promise<UploadFilesResponse> {
    if (files.length === 0) {
      return { file_count: 0, uploaded: [] };
    }
    const payload = new FormData();
    files.forEach((file) => payload.append("files", file, file.name));

    const response = await fetch(buildUrl(this.baseUrl, "/v2/uploads"), {
      method: "POST",
      headers: this.headers(),
      body: payload,
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadFilesResponse;
  }

  private async fileFingerprint(file: File): Promise<string> {
    const base = `${file.name}:${file.size}:${file.lastModified}:${file.type || "application/octet-stream"}`;
    if (typeof window === "undefined" || !window.crypto?.subtle) {
      return base;
    }
    try {
      const seed = new TextEncoder().encode(base);
      const digest = await window.crypto.subtle.digest("SHA-256", seed);
      const hash = Array.from(new Uint8Array(digest))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
      return `${base}:${hash}`;
    } catch {
      return base;
    }
  }

  private async initResumableUpload(
    payload: ResumableUploadInitRequest
  ): Promise<ResumableUploadSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/uploads/resumable/init"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResumableUploadSessionResponse;
  }

  private async uploadResumableChunk(
    uploadId: string,
    offset: number,
    chunk: Blob
  ): Promise<ResumableUploadChunkResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/uploads/resumable/${encodeURIComponent(uploadId)}/chunk`, {
        offset: String(Math.max(0, Math.floor(offset))),
      }),
      {
        method: "PUT",
        headers: this.headers({ "Content-Type": "application/octet-stream" }),
        body: chunk,
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResumableUploadChunkResponse;
  }

  private async getResumableUploadStatus(uploadId: string): Promise<ResumableUploadSessionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/uploads/resumable/${encodeURIComponent(uploadId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResumableUploadSessionResponse;
  }

  private async completeResumableUpload(
    uploadId: string
  ): Promise<ResumableUploadCompleteResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/uploads/resumable/${encodeURIComponent(uploadId)}/complete`),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResumableUploadCompleteResponse;
  }

  private isRecoverableResumableUploadError(error: unknown): boolean {
    if (!(error instanceof ApiError)) {
      return false;
    }
    const detail =
      typeof error.detail === "string"
        ? error.detail
        : error.detail
          ? JSON.stringify(error.detail)
          : "";
    return (
      detail.includes("missing uploaded file") ||
      detail.includes("Upload session was stale and has been reset")
    );
  }

  private async uploadFileResumable(file: File): Promise<UploadFilesResponse["uploaded"][number]> {
    const fingerprint = await this.fileFingerprint(file);
    const runAttempt = async (allowRetry: boolean): Promise<UploadFilesResponse["uploaded"][number]> => {
      const init = await this.initResumableUpload({
        file_name: file.name,
        size_bytes: file.size,
        content_type: file.type || "application/octet-stream",
        fingerprint,
      });

      if (init.status === "completed" && init.uploaded) {
        return init.uploaded;
      }
      if (init.status === "failed") {
        throw new ApiError(
          `Upload session failed for ${file.name}`,
          409,
          init.error || "Upload session failed",
        );
      }

      const uploadId = init.upload_id;
      const chunkSize = Math.max(256 * 1024, Number(init.chunk_size_bytes) || 5 * 1024 * 1024);
      let offset = Math.max(0, Number(init.bytes_received) || 0);
      let offsetMismatchRetries = 0;

      while (offset < file.size) {
        const end = Math.min(offset + chunkSize, file.size);
        const chunk = file.slice(offset, end);
        try {
          const chunkResponse = await this.uploadResumableChunk(uploadId, offset, chunk);
          offset = Math.max(offset, Number(chunkResponse.bytes_received) || end);
          offsetMismatchRetries = 0;
        } catch (error) {
          if (error instanceof ApiError && error.status === 409) {
            const detail = error.detail;
            const expectedOffset =
              detail && typeof detail === "object"
                ? (detail as Record<string, unknown>).expected_offset
                : null;
            if (typeof expectedOffset === "number" && Number.isFinite(expectedOffset)) {
              offset = Math.max(0, Math.floor(expectedOffset));
              offsetMismatchRetries += 1;
              if (offsetMismatchRetries <= 5) {
                continue;
              }
            }
            const statusPayload = await this.getResumableUploadStatus(uploadId);
            offset = Math.max(0, Number(statusPayload.bytes_received) || offset);
            if (statusPayload.status === "active" && offset < file.size) {
              offsetMismatchRetries += 1;
              if (offsetMismatchRetries <= 5) {
                continue;
              }
            }
          }
          throw error;
        }
      }

      try {
        const complete = await this.completeResumableUpload(uploadId);
        return complete.uploaded;
      } catch (error) {
        if (allowRetry && this.isRecoverableResumableUploadError(error)) {
          return runAttempt(false);
        }
        throw error;
      }
    };

    return runAttempt(true);
  }

  async importBisqueResources(resources: string[]): Promise<BisqueImportResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/uploads/from-bisque"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ resources }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueImportResponse;
  }

  async searchBisqueResources(options: BisqueSearchRequest = {}): Promise<BisqueSearchResponse> {
    const extensions = asStringArray(options.extensions);
    const response = await fetch(buildUrl(this.baseUrl, "/v2/bisque/search"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        resource_type: asTrimmedString(options.resourceType) || "image",
        tag_query: asTrimmedString(options.tagQuery),
        ...(asTrimmedString(options.tagOrder) ? { tag_order: asTrimmedString(options.tagOrder) } : {}),
        query: asTrimmedString(options.query),
        ...(asTrimmedString(options.nameContains)
          ? { name_contains: asTrimmedString(options.nameContains) }
          : {}),
        ...(extensions.length > 0 ? { extensions } : {}),
        ...(asTrimmedString(options.scope) ? { scope: asTrimmedString(options.scope) } : {}),
        ...(asTrimmedString(options.sort) ? { sort: asTrimmedString(options.sort) } : {}),
        limit: Math.max(1, Math.min(100, Number(options.limit) || 25)),
        ...(options.countAll ? { count_all: true } : {}),
        ...(Number.isFinite(Number(options.offset)) && Number(options.offset) > 0
          ? { offset: Math.floor(Number(options.offset)) }
          : {}),
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueSearchResponse;
  }

  async downloadBisqueResources(resources: string[]): Promise<BisqueImportResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/bisque/download"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ resources }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueImportResponse;
  }

  async uploadBisqueOutputs(options: {
    fileIds?: string[];
    artifactIds?: string[];
  }): Promise<BisqueUploadResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/bisque/upload"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        file_ids: options.fileIds ?? [],
        artifact_ids: options.artifactIds ?? [],
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueUploadResponse;
  }

  async listResources(options?: {
    limit?: number;
    offset?: number;
    query?: string;
    kind?: "image" | "video" | "table" | "file";
    source?: "upload" | "bisque_import";
  }): Promise<ResourceListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const query = String(options?.query ?? "").trim();
    if (query) {
      params.q = query;
    }
    const kind = String(options?.kind ?? "").trim();
    if (kind) {
      params.kind = kind;
    }
    const source = String(options?.source ?? "").trim();
    if (source) {
      params.source = source;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resources", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceListResponse;
  }

  async lookupResourceReuse(
    request: ResourceComputationLookupRequest
  ): Promise<ResourceComputationLookupResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/resources/reuse-suggestions"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceComputationLookupResponse;
  }

  async deleteResource(fileId: string): Promise<{ deleted: boolean; file_id: string }> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId)}`),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as { deleted: boolean; file_id: string };
  }

  resourceThumbnailUrl(fileId: string): string {
    const safeFileId = encodeURIComponent(fileId);
    return buildUrl(this.baseUrl, `/v2/resources/${safeFileId}/thumbnail`);
  }

  uploadPreviewUrl(fileId: string): string {
    const safeFileId = encodeURIComponent(fileId);
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/preview`);
  }

  uploadDisplayUrl(
    fileId: string,
    explicitPath?: string | null,
    config?: {
      enhancement?: string;
      negative?: boolean;
      gamma?: number | null;
      channels?: number[];
      channelColors?: string[];
      cacheKey?: string;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const path =
      explicitPath && String(explicitPath).trim()
        ? String(explicitPath)
        : `/v2/uploads/${safeFileId}/display`;
    const params: Record<string, string> = {};
    if (config?.enhancement) {
      params.enhancement = config.enhancement;
    }
    if (typeof config?.negative === "boolean") {
      params.negative = config.negative ? "true" : "false";
    }
    if (typeof config?.gamma === "number" && Number.isFinite(config.gamma) && config.gamma > 0) {
      params.gamma = String(config.gamma);
    }
    if (Array.isArray(config?.channels) && config.channels.length > 0) {
      params.channels = config.channels
        .filter((value) => Number.isFinite(value))
        .map((value) => String(Math.max(0, Math.floor(value))))
        .join(",");
    }
    if (Array.isArray(config?.channelColors) && config.channelColors.length > 0) {
      params.channel_colors = config.channelColors
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(",");
    }
    const cacheKey = String(config?.cacheKey ?? "").trim();
    if (cacheKey) {
      params.cache_key = cacheKey;
    }
    return buildUrl(this.baseUrl, path, params);
  }

  uploadSliceUrl(
    fileId: string,
    indices?: {
      axis?: "z" | "y" | "x";
      x?: number | null;
      y?: number | null;
      z?: number | null;
      c?: number | null;
      t?: number | null;
      enhancement?: string;
      fusionMethod?: string;
      negative?: boolean;
      channels?: number[];
      channelColors?: string[];
      fullResolution?: boolean;
      cacheKey?: string;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (indices?.axis) {
      params.axis = indices.axis;
    }
    if (typeof indices?.x === "number" && Number.isFinite(indices.x)) {
      params.x = String(Math.max(0, Math.floor(indices.x)));
    }
    if (typeof indices?.y === "number" && Number.isFinite(indices.y)) {
      params.y = String(Math.max(0, Math.floor(indices.y)));
    }
    if (typeof indices?.z === "number" && Number.isFinite(indices.z)) {
      params.z = String(Math.max(0, Math.floor(indices.z)));
    }
    if (typeof indices?.c === "number" && Number.isFinite(indices.c)) {
      params.c = String(Math.max(0, Math.floor(indices.c)));
    }
    if (typeof indices?.t === "number" && Number.isFinite(indices.t)) {
      params.t = String(Math.max(0, Math.floor(indices.t)));
    }
    if (indices?.enhancement) {
      params.enhancement = indices.enhancement;
    }
    if (indices?.fusionMethod) {
      params.fusion_method = indices.fusionMethod;
    }
    if (typeof indices?.negative === "boolean") {
      params.negative = indices.negative ? "true" : "false";
    }
    if (Array.isArray(indices?.channels) && indices.channels.length > 0) {
      params.channels = indices.channels
        .filter((value) => Number.isFinite(value))
        .map((value) => String(Math.max(0, Math.floor(value))))
        .join(",");
    }
    if (Array.isArray(indices?.channelColors) && indices.channelColors.length > 0) {
      params.channel_colors = indices.channelColors
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(",");
    }
    if (typeof indices?.fullResolution === "boolean") {
      params.full_resolution = indices.fullResolution ? "true" : "false";
    }
    const cacheKey = String(indices?.cacheKey ?? "").trim();
    if (cacheKey) {
      params.cache_key = cacheKey;
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/slice`, params);
  }

  async getUploadScalarVolume(
    fileId: string,
    config?: { t?: number | null; channel?: number | null }
  ): Promise<ScalarVolumePayload> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (typeof config?.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    if (typeof config?.channel === "number" && Number.isFinite(config.channel)) {
      params.channel = String(Math.max(0, Math.floor(config.channel)));
    }
    const response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/scalar-volume`, params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return {
      data: await response.arrayBuffer(),
      width: Number(response.headers.get("x-volume-width") ?? 0),
      height: Number(response.headers.get("x-volume-height") ?? 0),
      depth: Number(response.headers.get("x-volume-depth") ?? 0),
      dtype: String(response.headers.get("x-volume-dtype") ?? "uint16"),
      bytesPerVoxel: Number(response.headers.get("x-volume-bytes-per-voxel") ?? 2),
      rawMin: Number(response.headers.get("x-volume-raw-min") ?? 0),
      rawMax: Number(response.headers.get("x-volume-raw-max") ?? 1),
      channel:
        response.headers.get("x-volume-channel") == null
          ? null
          : Number(response.headers.get("x-volume-channel")),
    };
  }

  uploadTileUrl(
    fileId: string,
    config: {
      axis: "z" | "y" | "x";
      level: number;
      tileX: number;
      tileY: number;
      z?: number | null;
      c?: number | null;
      t?: number | null;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const safeAxis = encodeURIComponent(config.axis);
    const safeLevel = Math.max(0, Math.floor(config.level));
    const safeTileX = Math.max(0, Math.floor(config.tileX));
    const safeTileY = Math.max(0, Math.floor(config.tileY));
    const params: Record<string, string> = {};
    if (typeof config.z === "number" && Number.isFinite(config.z)) {
      params.z = String(Math.max(0, Math.floor(config.z)));
    }
    if (typeof config.c === "number" && Number.isFinite(config.c)) {
      params.c = String(Math.max(0, Math.floor(config.c)));
    }
    if (typeof config.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    return buildUrl(
      this.baseUrl,
      `/v2/uploads/${safeFileId}/tiles/${safeAxis}/${safeLevel}/${safeTileX}/${safeTileY}`,
      params
    );
  }

  uploadAtlasUrl(
    fileId: string,
    config?: {
      enhancement?: string;
      fusionMethod?: string;
      negative?: boolean;
      channels?: number[];
      channelColors?: string[];
      t?: number | null;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (config?.enhancement) {
      params.enhancement = config.enhancement;
    }
    if (config?.fusionMethod) {
      params.fusion_method = config.fusionMethod;
    }
    if (typeof config?.negative === "boolean") {
      params.negative = config.negative ? "true" : "false";
    }
    if (Array.isArray(config?.channels) && config.channels.length > 0) {
      params.channels = config.channels
        .filter((value) => Number.isFinite(value))
        .map((value) => String(Math.max(0, Math.floor(value))))
        .join(",");
    }
    if (Array.isArray(config?.channelColors) && config.channelColors.length > 0) {
      params.channel_colors = config.channelColors
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(",");
    }
    if (typeof config?.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/atlas`, params);
  }

  async getUploadHistogram(
    fileId: string,
    config?: { channels?: number[]; t?: number | null; bins?: number | null }
  ): Promise<UploadViewerHistogramResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (Array.isArray(config?.channels) && config.channels.length > 0) {
      params.channels = config.channels
        .filter((value) => Number.isFinite(value))
        .map((value) => String(Math.max(0, Math.floor(value))))
        .join(",");
    }
    if (typeof config?.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    if (typeof config?.bins === "number" && Number.isFinite(config.bins)) {
      params.bins = String(Math.max(8, Math.floor(config.bins)));
    }
    const response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/histogram`, params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadViewerHistogramResponse;
  }

  async getUploadViewer(fileId: string): Promise<UploadViewerInfo> {
    const safeFileId = encodeURIComponent(fileId);
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 15000);
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/viewer`), {
        method: "GET",
        headers: this.headers(),
        signal: controller.signal,
        credentials: "include",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("Viewer metadata request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return normalizeUploadViewerInfo(await response.json());
  }

  async getHdf5DatasetSummary(fileId: string, datasetPath: string): Promise<Hdf5DatasetSummary> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: datasetPath,
    };
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 15000);
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/dataset`, params), {
        method: "GET",
        headers: this.headers(),
        signal: controller.signal,
        credentials: "include",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("HDF5 dataset request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Hdf5DatasetSummary;
  }

  async getHdf5MaterialsDashboard(fileId: string): Promise<Hdf5MaterialsDashboardResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 20000);
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/materials/dashboard`), {
        method: "GET",
        headers: this.headers(),
        signal: controller.signal,
        credentials: "include",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("HDF5 materials dashboard request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Hdf5MaterialsDashboardResponse;
  }

  hdf5SlicePreviewUrl(
    fileId: string,
    config: {
      datasetPath: string;
      axis?: "z" | "y" | "x";
      index?: number | null;
      component?: number | null;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: config.datasetPath,
    };
    if (config.axis) {
      params.axis = config.axis;
    }
    if (typeof config.index === "number" && Number.isFinite(config.index)) {
      params.index = String(Math.max(0, Math.floor(config.index)));
    }
    if (typeof config.component === "number" && Number.isFinite(config.component)) {
      params.component = String(Math.max(0, Math.floor(config.component)));
    }
    return buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/preview/slice`, params);
  }

  hdf5AtlasPreviewUrl(
    fileId: string,
    config: {
      datasetPath: string;
      enhancement?: string;
      fusionMethod?: string;
      negative?: boolean;
      channels?: number[];
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: config.datasetPath,
    };
    if (config.enhancement) {
      params.enhancement = config.enhancement;
    }
    if (config.fusionMethod) {
      params.fusion_method = config.fusionMethod;
    }
    if (typeof config.negative === "boolean") {
      params.negative = config.negative ? "true" : "false";
    }
    if (Array.isArray(config.channels) && config.channels.length > 0) {
      params.channels = config.channels
        .filter((value) => Number.isFinite(value))
        .map((value) => String(Math.max(0, Math.floor(value))))
        .join(",");
    }
    return buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/preview/atlas`, params);
  }

  async getHdf5ScalarVolume(
    fileId: string,
    config: { datasetPath: string; channel?: number | null }
  ): Promise<ScalarVolumePayload> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: config.datasetPath,
    };
    if (typeof config.channel === "number" && Number.isFinite(config.channel)) {
      params.channel = String(Math.max(0, Math.floor(config.channel)));
    }
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/preview/scalar-volume`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return {
      data: await response.arrayBuffer(),
      width: Number(response.headers.get("x-volume-width") ?? 0),
      height: Number(response.headers.get("x-volume-height") ?? 0),
      depth: Number(response.headers.get("x-volume-depth") ?? 0),
      dtype: String(response.headers.get("x-volume-dtype") ?? "uint16"),
      bytesPerVoxel: Number(response.headers.get("x-volume-bytes-per-voxel") ?? 2),
      rawMin: Number(response.headers.get("x-volume-raw-min") ?? 0),
      rawMax: Number(response.headers.get("x-volume-raw-max") ?? 1),
      channel:
        response.headers.get("x-volume-channel") == null
          ? null
          : Number(response.headers.get("x-volume-channel")),
    };
  }

  async getHdf5DatasetHistogram(
    fileId: string,
    datasetPath: string,
    config?: { component?: number | null; bins?: number | null }
  ): Promise<Hdf5DatasetHistogramResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: datasetPath,
    };
    if (typeof config?.component === "number" && Number.isFinite(config.component)) {
      params.component = String(Math.max(0, Math.floor(config.component)));
    }
    if (typeof config?.bins === "number" && Number.isFinite(config.bins)) {
      params.bins = String(Math.max(8, Math.floor(config.bins)));
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 15000);
    let response: Response;
    try {
      response = await fetch(
        buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/preview/histogram`, params),
        {
          method: "GET",
          headers: this.headers(),
          signal: controller.signal,
          credentials: "include",
        }
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("HDF5 histogram request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Hdf5DatasetHistogramResponse;
  }

  async getHdf5DatasetTablePreview(
    fileId: string,
    datasetPath: string,
    config?: { offset?: number | null; limit?: number | null }
  ): Promise<Hdf5DatasetTablePreviewResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: datasetPath,
    };
    if (typeof config?.offset === "number" && Number.isFinite(config.offset)) {
      params.offset = String(Math.max(0, Math.floor(config.offset)));
    }
    if (typeof config?.limit === "number" && Number.isFinite(config.limit)) {
      params.limit = String(Math.max(1, Math.floor(config.limit)));
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 15000);
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/hdf5/preview/table`, params), {
        method: "GET",
        headers: this.headers(),
        signal: controller.signal,
        credentials: "include",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("HDF5 table preview request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Hdf5DatasetTablePreviewResponse;
  }

  async getUploadCaption(fileId: string): Promise<UploadCaptionResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 18000);
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/caption`), {
        method: "GET",
        headers: this.headers(),
        signal: controller.signal,
        credentials: "include",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("Caption request timed out", 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadCaptionResponse;
  }

  async sam3InteractiveSegment(
    request: Sam3InteractiveRequest
  ): Promise<Sam3InteractiveResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/segment/sam3/interactive"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Sam3InteractiveResponse;
  }

  async getRun(runId: string): Promise<RunResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/runs/${runId}`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as RunResponse;
  }

  async getRunResult(runId: string): Promise<RunResultResponse> {
    const v2Run = await this.getV2Run(runId);
    if (!v2Run) {
      throw new ApiError("Run was not found", 404, null);
    }

    const status = normalizeRunResultStatus(v2Run.status);
    const result =
      status === "succeeded"
        ? normalizeV2RunResponse(v2Run, {
            runId,
            responseText: "",
            progressEvents: [],
          })
        : null;

    return {
      run_id: asTrimmedString(v2Run.run_id) || runId,
      status,
      result,
    };
  }

  async getAnalysisHistory(limit = 5, query?: string): Promise<AnalysisHistoryResponse> {
    const params: Record<string, string> = { limit: String(limit) };
    if (query && query.trim()) {
      params.q = query.trim();
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v1/history/analyses", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AnalysisHistoryResponse;
  }

  async evaluateContractHealth(request: ContractAuditRequest): Promise<ContractAuditResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/evals/contracts"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        run_ids: request.run_ids ?? [],
        limit: request.limit ?? 25,
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ContractAuditResponse;
  }

  async listConversations(
    limit = 25,
    offset = 0,
    includeState = false
  ): Promise<ConversationListResponse> {
    return this.listV2Conversations(limit, offset, includeState);
  }

  async getConversation(conversationId: string): Promise<ConversationRecord> {
    return this.getV2Conversation(conversationId);
  }

  async upsertConversation(
    record: ConversationRecord,
    options?: UpsertConversationOptions
  ): Promise<ConversationRecord> {
    return this.upsertV2Conversation(record, options);
  }

  async deleteConversation(conversationId: string): Promise<{ deleted: boolean; conversation_id: string }> {
    return this.deleteV2Conversation(conversationId);
  }

  async searchConversations(query: string, limit = 50): Promise<ConversationSearchResponse> {
    const normalizedQuery = asTrimmedString(query);
    const needle = normalizedQuery.toLowerCase();
    const conversations = await this.listV2Conversations(limit, 0, true);
    const matches = conversations.conversations.filter((conversation) => {
      if (!needle) {
        return true;
      }
      const searchable = [
        conversation.title,
        conversation.preview,
        conversation.conversation_id,
        JSON.stringify(conversation.state ?? {}),
      ]
        .join("\n")
        .toLowerCase();
      return searchable.includes(needle);
    });
    return {
      query: normalizedQuery,
      count: matches.length,
      matches,
    };
  }

  async getRunEvents(runId: string, limit = 200): Promise<RunEventsResponse> {
    const requestedLimit = Math.max(1, Math.floor(asFiniteNumber(limit, 200)));
    const events: RunEvent[] = [];
    let resolvedRunId = runId;
    let afterSequence = 0;
    while (true) {
      const payload = await this.fetchJson<Record<string, unknown>>(
        `/v2/runs/${encodeURIComponent(runId)}/events`,
        { method: "GET" },
        {
          limit: String(requestedLimit),
          after_sequence: String(afterSequence),
        }
      );
      resolvedRunId = asTrimmedString(payload.run_id) || resolvedRunId;
      const rawEvents = Array.isArray(payload.events) ? payload.events.filter(isRecord) : [];
      events.push(...rawEvents.map((event) => normalizeV2RunEvent(event)));
      const nextAfterSequence = rawEvents.reduce((current, event) => {
        const sequence = Math.floor(asFiniteNumber(event.sequence, 0));
        return sequence > current ? sequence : current;
      }, afterSequence);
      if (rawEvents.length < requestedLimit || nextAfterSequence <= afterSequence) {
        break;
      }
      afterSequence = nextAfterSequence;
    }
    return {
      run_id: resolvedRunId,
      events,
    };
  }

  async listArtifacts(runId: string, limit = 500): Promise<ArtifactListResponse> {
    const payload = await this.fetchJson<Record<string, unknown>>(
      `/v2/runs/${encodeURIComponent(runId)}/artifacts`,
      { method: "GET" },
      { limit: String(limit) }
    );
    const artifacts = Array.isArray(payload.artifacts)
      ? payload.artifacts
          .map((artifact): ArtifactRecord | null => {
            if (!isRecord(artifact)) {
              return null;
            }
            const path =
              asOptionalString(artifact.path) ??
              asOptionalString(artifact.relative_path) ??
              asOptionalString(artifact.source_path) ??
              asOptionalString(artifact.preview_path) ??
              asTrimmedString(artifact.artifact_id);
            const artifactId = asTrimmedString(artifact.artifact_id);
            this.rememberV2Artifact(runId, artifactId, [
              artifact.path,
              artifact.relative_path,
              artifact.source_path,
              artifact.preview_path,
              path,
            ]);
            return {
              path,
              size_bytes: asFiniteNumber(artifact.size_bytes, 0),
              mime_type: asOptionalString(artifact.mime_type),
              modified_at:
                asOptionalString(artifact.updated_at) ??
                asOptionalString(artifact.created_at) ??
                new Date(0).toISOString(),
              source_path: asOptionalString(artifact.source_path),
              title: asOptionalString(artifact.title),
              result_group_id: asOptionalString(artifact.result_group_id),
            } satisfies ArtifactRecord;
          })
          .filter((artifact): artifact is ArtifactRecord => artifact !== null)
      : [];
    return {
      run_id: asTrimmedString(payload.run_id) || runId,
      root: "",
      artifact_count: asFiniteNumber(payload.count, artifacts.length),
      artifacts,
    };
  }

  artifactDownloadUrl(runId: string, path: string): string {
    const artifactId = this.v2ArtifactIdsByRunPath.get(this.v2ArtifactKey(runId, path));
    if (artifactId) {
      return buildUrl(this.baseUrl, `/v2/artifacts/${encodeURIComponent(artifactId)}/download`);
    }
    return buildUrl(this.baseUrl, `/v2/runs/${encodeURIComponent(runId)}/artifacts/download`, {
      path,
    });
  }

  async createReproReport(req: ReproReportRequest): Promise<ReproReportResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/workflows/repro-report"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(req),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ReproReportResponse;
  }

}

export * from "./api-v2";
