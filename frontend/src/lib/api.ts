import type {
  AdminConversationActionResponse,
  AdminIssueListResponse,
  AdminOverviewResponse,
  AdminRunActionResponse,
  AdminRunListResponse,
  AdminUserListResponse,
  AnalysisHistoryResponse,
  ArtifactListResponse,
  BisqueAuthLoginRequest,
  BisqueAuthSessionResponse,
  BisqueGuestAuthRequest,
  BisqueImportResponse,
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
  V3ApprovalResolveRequest,
  V3ApprovalResolveResponse,
  V3ArtifactListResponse,
  V3RunCreateRequest,
  V3RunEventListResponse,
  V3RunRecord,
  V3SessionCreateRequest,
  V3SessionListResponse,
  V3SessionMessageListResponse,
  V3SessionRecord,
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

const normalizeChatResponsePayload = (
  value: unknown,
  fallback?: {
    runId?: string | null;
    model?: string | null;
    responseText?: string | null;
    durationSeconds?: number | null;
  }
): ChatResponse | null => {
  const root = isRecord(value)
    ? isRecord(value.response)
      ? value.response
      : isRecord(value.result) && isRecord(value.result.response)
        ? value.result.response
        : value
    : null;
  if (!root) {
    return null;
  }

  const rawResponseText =
    root.response_text ?? root.content ?? root.output_text ?? fallback?.responseText ?? "";
  const runId = String(root.run_id ?? fallback?.runId ?? "").trim();
  const model = String(root.model ?? fallback?.model ?? "").trim();
  const durationRaw = root.duration_seconds ?? fallback?.durationSeconds ?? 0;
  const durationSeconds = Number.isFinite(Number(durationRaw)) ? Number(durationRaw) : 0;
  const progressEvents = Array.isArray(root.progress_events) ? root.progress_events : [];
  const benchmark = isRecord(root.benchmark) ? root.benchmark : null;
  const metadata = isRecord(root.metadata) ? root.metadata : null;

  if (!runId && !model && String(rawResponseText || "").trim().length === 0) {
    return null;
  }

  return {
    run_id: runId,
    model,
    response_text: String(rawResponseText || ""),
    duration_seconds: durationSeconds,
    progress_events: progressEvents,
    benchmark,
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

  async health(): Promise<{ status: string; ts: string }> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/health"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/config/public"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/admin/overview", params), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/admin/users", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminUserListResponse;
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/admin/runs", params), {
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
      buildUrl(this.baseUrl, "/v1/admin/issues", {
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
      buildUrl(this.baseUrl, `/v1/admin/runs/${encodeURIComponent(runId)}/cancel`),
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

  async deleteAdminConversation(
    conversationId: string,
    userId: string
  ): Promise<AdminConversationActionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/admin/conversations/${encodeURIComponent(conversationId)}`, {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/models"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/prairie/sync"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/prairie/status"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/prairie/benchmark/run"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/prairie/retrain-request"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/prairie/retrain-requests"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/datasets"), {
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
        "/v1/training/datasets",
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
      buildUrl(this.baseUrl, `/v1/training/datasets/${encodeURIComponent(datasetId)}`),
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
      buildUrl(this.baseUrl, `/v1/training/datasets/${encodeURIComponent(datasetId)}/items`),
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/jobs"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/preflight"), {
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
      buildUrl(this.baseUrl, `/v1/training/jobs/${encodeURIComponent(jobId)}`),
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
      buildUrl(this.baseUrl, `/v1/training/jobs/${encodeURIComponent(jobId)}/control`),
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/inference/jobs"), {
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
      buildUrl(this.baseUrl, `/v1/inference/jobs/${encodeURIComponent(jobId)}/result`),
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/model-health"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/admin/model-health"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/domains"), {
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
      buildUrl(this.baseUrl, "/v1/training/domains", { limit: String(limit) }),
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
        `/v1/training/domains/${encodeURIComponent(domainId)}/lineages`,
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
      buildUrl(this.baseUrl, `/v1/training/lineages/${encodeURIComponent(lineageId)}/fork`),
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
        `/v1/training/lineages/${encodeURIComponent(lineageId)}/versions`,
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/update-proposals/preview"), {
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
        "/v1/training/update-proposals",
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
      buildUrl(this.baseUrl, `/v1/training/update-proposals/${encodeURIComponent(proposalId)}/approve`),
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
      buildUrl(this.baseUrl, `/v1/training/update-proposals/${encodeURIComponent(proposalId)}/reject`),
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
      buildUrl(this.baseUrl, `/v1/training/model-versions/${encodeURIComponent(versionId)}/promote`),
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
      buildUrl(this.baseUrl, `/v1/training/model-versions/${encodeURIComponent(versionId)}/rollback`),
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/training/merge-requests"), {
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
      buildUrl(this.baseUrl, `/v1/training/merge-requests/${encodeURIComponent(mergeId)}/approve`),
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
      buildUrl(this.baseUrl, `/v1/training/merge-requests/${encodeURIComponent(mergeId)}/reject`),
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
        "/v1/training/merge-requests",
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/auth/session"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  getBisqueOidcStartUrl(redirectUrl?: string): string {
    const params: Record<string, string> = {};
    const normalizedRedirect = String(redirectUrl ?? "").trim();
    if (normalizedRedirect) {
      params.next = normalizedRedirect;
    }
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(
      this.baseUrl,
      "/v1/auth/oidc/start",
      Object.keys(params).length > 0 ? params : undefined
    );
  }

  getBisqueBrowserLogoutUrl(redirectUrl?: string): string {
    const params: Record<string, string> = {};
    const normalizedRedirect = String(redirectUrl ?? "").trim();
    if (normalizedRedirect) {
      params.next = normalizedRedirect;
    }
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(
      this.baseUrl,
      "/v1/auth/logout/browser",
      Object.keys(params).length > 0 ? params : undefined
    );
  }

  async loginBisque(payload: BisqueAuthLoginRequest): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/auth/login"), {
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

  async logoutBisque(): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/auth/logout"), {
      method: "POST",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisqueAuthSessionResponse;
  }

  async continueAsGuest(payload: BisqueGuestAuthRequest): Promise<BisqueAuthSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/auth/guest"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/chat"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    const payload = await response.json();
    const normalized = normalizeChatResponsePayload(payload);
    if (!normalized) {
      throw new ApiError("Chat response did not include a valid completion payload", 502, payload);
    }
    return normalized;
  }

  async chatTitle(request: ChatTitleRequest): Promise<ChatTitleResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/chat/title"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ChatTitleResponse;
  }

  async chatStream(request: ChatRequest, options?: ChatStreamOptions): Promise<ChatResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v1/chat/stream"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
      signal: options?.signal,
    });
    if (!response.ok) {
      if (response.status === 404 || response.status === 405 || response.status === 501) {
        const fallback = await this.chat(request);
        options?.onDone?.(fallback);
        return fallback;
      }
      return parseError(response);
    }
    if (!response.body) {
      throw new ApiError("Stream response did not include a readable body", 502, null);
    }
    options?.onRunStarted?.({
      runId: String(response.headers.get("X-Run-Id") ?? "").trim(),
      model: String(response.headers.get("X-Model") ?? "").trim() || null,
    });
    const headerRunId = String(response.headers.get("X-Run-Id") ?? "").trim();
    const headerModel = String(response.headers.get("X-Model") ?? "").trim();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let completedPayload: ChatResponse | null = null;
    let terminalEventSeen = false;
    let streamedText = "";

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
          eventName = line.slice("event:".length).trim();
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

      if (eventName === "token" && payload && typeof payload === "object") {
        const delta = String((payload as Record<string, unknown>).delta ?? "");
        if (delta) {
          streamedText += delta;
          options?.onToken?.(delta);
        }
        return;
      }

      if (eventName === "done") {
        const normalized = normalizeChatResponsePayload(payload, {
          runId: headerRunId || null,
          model: headerModel || null,
          responseText: streamedText || null,
          durationSeconds: 0,
        });
        if (!normalized) {
          return;
        }
        completedPayload = normalized;
        terminalEventSeen = true;
        options?.onDone?.(normalized);
        return;
      }

      if (eventName === "run_event" && payload && typeof payload === "object") {
        options?.onRunEvent?.(payload as RunEvent);
        return;
      }

      if (eventName === "error") {
        const detail =
          payload && typeof payload === "object"
            ? (payload as Record<string, unknown>).detail ?? payload
            : payload;
        const statusRaw =
          payload && typeof payload === "object"
            ? (payload as Record<string, unknown>).status_code
            : null;
        const statusCode =
          typeof statusRaw === "number"
            ? statusRaw
            : typeof statusRaw === "string" && /^\d+$/.test(statusRaw)
              ? Number(statusRaw)
              : 500;
        throw new ApiError(`Stream failed with status ${statusCode}`, statusCode, detail);
      }
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

    if (!completedPayload && streamedText.trim().length > 0) {
      completedPayload = {
        run_id: headerRunId,
        model: headerModel,
        response_text: streamedText,
        duration_seconds: 0,
        progress_events: [],
        benchmark: null,
        metadata: null,
      };
      options?.onDone?.(completedPayload);
    }

    if (!completedPayload) {
      throw new ApiError("Stream ended before a completion payload was received", 502, null);
    }
    return completedPayload;
  }

  async uploadFiles(files: File[]): Promise<UploadFilesResponse> {
    if (files.length === 0) {
      return { file_count: 0, uploaded: [] };
    }
    try {
      const uploaded = [];
      for (const file of files) {
        const row = await this.uploadFileResumable(file);
        uploaded.push(row);
      }
      return {
        file_count: uploaded.length,
        uploaded,
      };
    } catch (error) {
      if (!(error instanceof ApiError) || ![404, 405, 501].includes(error.status)) {
        throw error;
      }
    }

    // Fallback path for servers without resumable endpoints.
    const payload = new FormData();
    files.forEach((file) => payload.append("files", file, file.name));

    const response = await fetch(buildUrl(this.baseUrl, "/v1/uploads"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/uploads/from-bisque"), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/resources", params), {
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
      buildUrl(this.baseUrl, `/v1/resources/${encodeURIComponent(fileId)}`),
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
    const params = this.apiKey ? { api_key: this.apiKey } : undefined;
    return buildUrl(this.baseUrl, `/v1/resources/${safeFileId}/thumbnail`, params);
  }

  uploadPreviewUrl(fileId: string): string {
    const safeFileId = encodeURIComponent(fileId);
    const params = this.apiKey ? { api_key: this.apiKey } : undefined;
    return buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/preview`, params);
  }

  uploadDisplayUrl(fileId: string, explicitPath?: string | null): string {
    const safeFileId = encodeURIComponent(fileId);
    const path =
      explicitPath && String(explicitPath).trim()
        ? String(explicitPath)
        : `/v1/uploads/${safeFileId}/display`;
    const params = this.apiKey ? { api_key: this.apiKey } : undefined;
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/slice`, params);
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    const response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/scalar-volume`, params), {
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(
      this.baseUrl,
      `/v1/uploads/${safeFileId}/tiles/${safeAxis}/${safeLevel}/${safeTileX}/${safeTileY}`,
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/atlas`, params);
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    let response: Response;
    try {
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/histogram`, params), {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      });
    } catch (error) {
      throw error;
    }
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
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/viewer`), {
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
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
    if (this.apiKey) {
      params.api_key = this.apiKey;
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
      response = await fetch(buildUrl(this.baseUrl, `/v1/uploads/${safeFileId}/caption`), {
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
    const response = await fetch(buildUrl(this.baseUrl, "/v1/segment/sam3/interactive"), {
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
    const response = await fetch(buildUrl(this.baseUrl, `/v1/runs/${runId}`), {
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
    const response = await fetch(buildUrl(this.baseUrl, `/v1/runs/${runId}/result`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as RunResultResponse;
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
    const response = await fetch(
      buildUrl(this.baseUrl, "/v1/conversations", {
        limit: String(limit),
        offset: String(offset),
        include_state: includeState ? "true" : "false",
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
    return (await response.json()) as ConversationListResponse;
  }

  async getConversation(conversationId: string): Promise<ConversationRecord> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/conversations/${encodeURIComponent(conversationId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ConversationRecord;
  }

  async upsertConversation(record: ConversationRecord): Promise<ConversationRecord> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/conversations/${encodeURIComponent(record.conversation_id)}`),
      {
        method: "PUT",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(record),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ConversationRecord;
  }

  async deleteConversation(conversationId: string): Promise<{ deleted: boolean; conversation_id: string }> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/conversations/${encodeURIComponent(conversationId)}`),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as { deleted: boolean; conversation_id: string };
  }

  async searchConversations(query: string, limit = 50): Promise<ConversationSearchResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, "/v1/conversations/search", {
        q: query,
        limit: String(limit),
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
    return (await response.json()) as ConversationSearchResponse;
  }

  async getRunEvents(runId: string, limit = 200): Promise<RunEventsResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/runs/${runId}/events`, { limit: String(limit) }),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as RunEventsResponse;
  }

  async listArtifacts(runId: string, limit = 500): Promise<ArtifactListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v1/artifacts/${runId}`, { limit: String(limit) }),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ArtifactListResponse;
  }

  artifactDownloadUrl(runId: string, path: string): string {
    const params: Record<string, string> = { path };
    if (this.apiKey) {
      params.api_key = this.apiKey;
    }
    return buildUrl(this.baseUrl, `/v1/artifacts/${runId}/download`, params);
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

  async createV3Session(req: V3SessionCreateRequest): Promise<V3SessionRecord> {
    const response = await fetch(buildUrl(this.baseUrl, "/v3/sessions"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(req),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3SessionRecord;
  }

  async listV3Sessions(limit = 100): Promise<V3SessionListResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v3/sessions", { limit: String(limit) }), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3SessionListResponse;
  }

  async getV3Session(sessionId: string): Promise<V3SessionRecord> {
    const response = await fetch(buildUrl(this.baseUrl, `/v3/sessions/${encodeURIComponent(sessionId)}`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3SessionRecord;
  }

  async getV3SessionMessages(sessionId: string, limit = 500): Promise<V3SessionMessageListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v3/sessions/${encodeURIComponent(sessionId)}/messages`, {
        limit: String(limit),
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
    return (await response.json()) as V3SessionMessageListResponse;
  }

  async createV3Run(sessionId: string, req: V3RunCreateRequest): Promise<V3RunRecord> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v3/sessions/${encodeURIComponent(sessionId)}/runs`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(req),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3RunRecord;
  }

  async getV3Run(runId: string): Promise<V3RunRecord> {
    const response = await fetch(buildUrl(this.baseUrl, `/v3/runs/${encodeURIComponent(runId)}`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3RunRecord;
  }

  async getV3RunEvents(runId: string, limit = 500): Promise<V3RunEventListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v3/runs/${encodeURIComponent(runId)}/events`, {
        limit: String(limit),
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
    return (await response.json()) as V3RunEventListResponse;
  }

  async getV3RunArtifacts(runId: string, limit = 500): Promise<V3ArtifactListResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v3/runs/${encodeURIComponent(runId)}/artifacts`, {
        limit: String(limit),
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
    return (await response.json()) as V3ArtifactListResponse;
  }

  async resolveV3Approval(
    runId: string,
    approvalId: string,
    req: V3ApprovalResolveRequest
  ): Promise<V3ApprovalResolveResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v3/runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(approvalId)}`
      ),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(req),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as V3ApprovalResolveResponse;
  }
}

export * from "./api-v2";
