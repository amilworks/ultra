import { bundleRootForRelativePath } from "./pendingBundles";
import type {
  AccountRequestPayload,
  AccountRequestResponse,
  AdminConversationActionResponse,
  CurrentUserProfile,
  CurrentUserResponse,
  TokenUsageResponse,
  AdminCreateOrganizationRequest,
  AdminCreateUserRequest,
  AdminIssueListResponse,
  AdminOrganization,
  AdminOrganizationListResponse,
  AdminMetricsResponse,
  AdminOverviewResponse,
  AdminRunActionResponse,
  AdminRunListResponse,
  AdminUpdateUserStatusRequest,
  AdminUserAccount,
  AdminUserStatus,
  AdminUserListResponse,
  ArtifactRecord,
  ArtifactListResponse,
  BisqueAuthLoginRequest,
  BisqueAuthSessionResponse,
  BisqueImportResponse,
  BisquePushRequest,
  BisquePushResponse,
  BisqueSearchRequest,
  BisqueSearchResponse,
  ChatRequest,
  ChatResponse,
  ConversationListResponse,
  ConversationRecord,
  Hdf5DatasetHistogramResponse,
  Hdf5DatasetSummary,
  Hdf5DatasetTablePreviewResponse,
  Hdf5MaterialsDashboardResponse,
  PublicConfigResponse,
  PrairieBenchmarkRunRequest,
  PrairieBenchmarkRunResponse,
  PrairieRetrainListResponse,
  PrairieRetrainRequest,
  PrairieStatusResponse,
  PrairieSyncResponse,
  ProgressEvent,
  DataAgentJobControlRequest,
  DataAgentJobCreateRequest,
  DataAgentJobListResponse,
  DataAgentJobResponse,
  DatasetSnapshotCreateRequest,
  DatasetSnapshotEventListResponse,
  DatasetSnapshotListResponse,
  DatasetSnapshotResponse,
  DatasetSnapshotShareGrantCreateRequest,
  DatasetSnapshotShareGrantListResponse,
  DatasetSnapshotShareGrantResponse,
  ResourceCollectionAddResourcesResponse,
  ResourceCollectionCreateRequest,
  ResourceCollectionListResponse,
  ResourceCollectionPatchRequest,
  ResourceCollectionRemoveResourcesResponse,
  ResourceCollectionResponse,
  ResourceCollectionShareGrantsCreateResponse,
  ResourceBulkLifecycleRequest,
  ResourceBulkLifecycleResponse,
  ResourceBulkTagRequest,
  ResourceBulkTagResponse,
  ResourceListResponse,
  ResourceMetadataFilter,
  ResourceRecord,
  ResourceResponse,
  ResourceTextHead,
  ResourceCsvRows,
  ResourceShareGrantCreateRequest,
  ResourceCollectionShareGrantListResponse,
  ResourceCollectionShareGrantRevokeResponse,
  ShareTargetListResponse,
  ResourceShareGrantListResponse,
  ResourceShareGrantResponse,
  ResourceShareGrantsCreateRequest,
  ResourceShareGrantsCreateResponse,
  InferenceJobCreateRequest,
  TrainingDomainListResponse,
  TrainingLineageListResponse,
  TrainingModelVersionListResponse,
  TrainingModelVersionResponse,
  TrainingVersionPromoteRequest,
  TrainingVersionRollbackRequest,
  RunResultResponse,
  RunEventsResponse,
  RunEvent,
  TrainingJobResponse,
  TrainingModelsResponse,
  UploadViewerHistogramResponse,
  UploadViewerInfo,
  CiftiCarpetResponse,
  CiftiConnectivityResponse,
  Scene3dManifestResponse,
  UploadChunkResponse,
  UploadedFileRecord,
  UploadFilesResponse,
  UploadSessionCreateRequest,
  UploadSessionFileCompleteResponse,
  UploadSessionResponse,
} from "../types";
import type * as ViewerManifest from "./viewerManifest";
import { reportClientError } from "./client-diagnostics";
import { isEphemeralDeltaEventKind } from "@/features/chat/run-events";

export type ApiClientOptions = {
  baseUrl: string;
  apiKey?: string;
};

export type StreamTokenEvent = {
  sequence?: number;
  eventId?: string;
  runId?: string;
  eventKind?: string;
};

export type ChatStreamHandlers = {
  onToken?: (delta: string, event?: StreamTokenEvent) => void;
  onDone?: (payload: ChatResponse) => void;
  onRunStarted?: (payload: { runId: string; model?: string | null }) => void;
  onRunEvent?: (payload: RunEvent) => void;
};

export type ChatStreamOptions = ChatStreamHandlers & {
  signal?: AbortSignal;
};

export type RunStreamOptions = ChatStreamOptions & {
  afterSequence?: number;
  /** Reconnect if the stream delivers no bytes for this long (server heartbeats
   *  every 15s; the default 60s = four missed beats — the dead-but-open socket
   *  signature after OS sleep). Tests inject a small value. */
  inactivityTimeoutMs?: number;
  /** Base backoff between reconnect attempts (default 1000ms; capped at 15s).
   *  Tests inject a small value. */
  retryBaseDelayMs?: number;
};

export type UploadProgressEvent = {
  id: string;
  fileName: string;
  fileIndex: number;
  fileToken?: string;
  sessionId?: string;
  fingerprint?: string;
  relativePath?: string;
  contentType?: string;
  chunkSizeBytes?: number;
  status: "creating" | "uploading" | "verifying" | "paused" | "completed" | "failed";
  totalBytes: number;
  bytesVerified: number;
  bytesCommitted: number;
  error?: string | null;
};

export type UploadPauseSignal = {
  isPaused: (sessionId: string, fileToken?: string | null) => boolean;
};

export type UploadResumeSessionOptions = {
  sessionId: string;
  fileToken?: string | null;
  progressId?: string | null;
};

export type UploadFilesOptions = {
  onProgress?: (event: UploadProgressEvent) => void;
  resumeSession?: UploadResumeSessionOptions;
  pauseSignal?: UploadPauseSignal;
};

type V2UploadFilePlan = {
  file: File;
  fileIndex: number;
  fileToken: string;
  fingerprint: string;
  relativePath: string | null;
  contentType: string;
  chunkSize: number;
  progressID: string;
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

export type RunSteerMessageRecord = {
  steer_id: string;
  run_id: string;
  thread_id: string;
  message_id: string;
  content: string;
  file_ids?: string[];
  status: "pending" | "applied" | "missed";
  created_at: string;
  applied_at?: string;
  updated_at: string;
};

export type NoteRecord = {
  note_id: string;
  title: string;
  body_markdown: string;
  pinned: boolean;
  created_at: string;
  updated_at: string;
};

export type NoteListItem = {
  note_id: string;
  title: string;
  snippet: string;
  pinned: boolean;
  updated_at: string;
};

export type NoteListResponse = {
  notes: NoteListItem[];
  total_count: number;
};

export type NoteWritePayload = {
  title?: string;
  body_markdown?: string;
  pinned?: boolean;
};

/** The steer 409 that means "run terminal or finalizing" — fall back to Phase 0 queueing. */
export const isSteeringClosedError = (error: unknown): boolean =>
  error instanceof ApiError &&
  error.status === 409 &&
  typeof error.detail === "object" &&
  error.detail !== null &&
  (error.detail as { code?: string }).code === "steering_closed";

export class UploadPausedError extends Error {
  readonly sessionId: string;
  readonly fileToken?: string | null;

  constructor(sessionId: string, fileToken?: string | null) {
    super("Upload paused");
    this.name = "UploadPausedError";
    this.sessionId = sessionId;
    this.fileToken = fileToken ?? null;
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
  channel: number;
  time: number;
  sourceWidth: number;
  sourceHeight: number;
  sourceDepth: number;
  downsampleX: number;
  downsampleY: number;
  downsampleZ: number;
  previewPolicy: string;
  sampling: "box" | "nearest";
  /** NIfTI physical intensity = sclSlope * stored code + sclInter. */
  sclSlope: number;
  sclInter: number;
};

const MAX_SCALAR_VOLUME_BYTES = 256 * 1024 * 1024;
const MAX_SCALAR_VOLUME_WORKING_SET_BYTES = 256 * 1024 * 1024;
const SCALAR_HALF_FLOAT_BYTES_PER_VOXEL = 2;
const SCALAR_RAW_FLOAT_BYTES_PER_VOXEL = 4;
const SCALAR_DTYPE_BYTES: Readonly<Record<string, number>> = {
  uint8: 1,
  uint16: 2,
  int16: 2,
  float32: 4,
};

const invalidScalarVolumeResponse = (message: string): ApiError =>
  new ApiError(`Invalid scalar volume response: ${message}`, 502, null);

const requiredScalarHeaderNumber = (response: Response, header: string): number => {
  const raw = response.headers.get(header);
  if (raw == null || raw.trim() === "") {
    throw invalidScalarVolumeResponse(`missing ${header}`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw invalidScalarVolumeResponse(`non-finite ${header}`);
  }
  return value;
};

const requiredScalarHeaderInteger = (response: Response, header: string): number => {
  const value = requiredScalarHeaderNumber(response, header);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw invalidScalarVolumeResponse(`invalid ${header}`);
  }
  return value;
};

const requiredScalarHeaderIndex = (response: Response, header: string): number => {
  const value = requiredScalarHeaderNumber(response, header);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw invalidScalarVolumeResponse(`invalid ${header}`);
  }
  return value;
};

export const requireNonNegativeSafeInteger = (value: unknown, field: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${field} must be a non-negative safe integer.`);
  }
  return value;
};

const throwIfAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) {
    throw new DOMException("The operation was aborted.", "AbortError");
  }
};

// --- Run event stream reconnection -------------------------------------------------
// A run outlives any single SSE connection: laptops sleep, proxies restart, networks
// blip. The stream consumer therefore treats one connection as one ATTEMPT and holds
// every cross-attempt invariant (event cursor, token dedupe keys, accumulated text)
// in this context object, so a reconnect resumes exactly where the last attempt died
// and a replay overlap can never double-apply an event or a token.

const RUN_STREAM_INACTIVITY_TIMEOUT_MS = 60_000;
const RUN_STREAM_RETRY_MAX_DELAY_MS = 15_000;

type V2RunStreamContext = {
  lastRunEventSequence: number;
  seenTokenDeliveryKeys: Set<string>;
  progressEvents: ProgressEvent[];
  streamedText: string;
  terminalEventSeen: boolean;
  terminalStatus: "succeeded" | "failed" | "canceled" | null;
  terminalDetail: unknown;
  terminalResponseText: string;
};

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === "AbortError";

// Retryable = the transport or an intermediary failed; the RUN's own outcome is
// unknown. Auth failures, not-found, and other 4xx are real answers — rethrow.
const isRetryableRunStreamError = (error: unknown): boolean => {
  if (error instanceof ApiError) {
    return (
      error.status === 0 ||
      error.status === 408 ||
      error.status === 429 ||
      error.status >= 500
    );
  }
  if (error instanceof DOMException) {
    // Our own inactivity cancel surfaces as a TimeoutError in engines that
    // reject the pending read on cancel.
    return error.name === "TimeoutError";
  }
  // fetch() network failures (connection cut, mid-body reset) are TypeErrors.
  return error instanceof TypeError;
};

// Exponential backoff between reconnect attempts, cut short the moment the tab
// becomes visible again or the browser regains network — waking from sleep should
// reconnect NOW, not after the remainder of a 15s backoff.
const waitBeforeStreamRetry = async (
  attempt: number,
  signal?: AbortSignal,
  baseDelayMs = 1000
): Promise<void> => {
  throwIfAborted(signal);
  if (attempt <= 1) {
    return; // first reconnect is immediate: the common case is a clean sever
  }
  const exponent = Math.min(attempt - 2, 4);
  const delayMs =
    Math.min(RUN_STREAM_RETRY_MAX_DELAY_MS, baseDelayMs * 2 ** exponent) +
    Math.floor(Math.random() * 250);
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisible);
      }
      if (typeof window !== "undefined") {
        window.removeEventListener("online", finish);
      }
      signal?.removeEventListener("abort", onAbort);
    };
    const finish = () => {
      if (!settled) {
        settled = true;
        cleanup();
        resolve();
      }
    };
    const onVisible = () => {
      if (typeof document === "undefined" || document.visibilityState === "visible") {
        finish();
      }
    };
    const onAbort = () => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(new DOMException("The operation was aborted.", "AbortError"));
      }
    };
    const timer = setTimeout(finish, delayMs);
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisible);
    }
    if (typeof window !== "undefined") {
      window.addEventListener("online", finish);
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
};

const readExactScalarVolumeBody = async (
  response: Response,
  expectedLength: number,
  signal?: AbortSignal
): Promise<ArrayBuffer> => {
  throwIfAborted(signal);
  const reader = response.body?.getReader();
  if (!reader) {
    throw invalidScalarVolumeResponse("streaming response body is unavailable");
  }
  const output = new Uint8Array(expectedLength);
  let offset = 0;
  const readWithAbort = (): Promise<ReadableStreamReadResult<Uint8Array>> => {
    throwIfAborted(signal);
    if (!signal) {
      return reader.read();
    }
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        void reader.cancel(signal.reason).catch(() => undefined);
        reject(new DOMException("The operation was aborted.", "AbortError"));
      };
      signal.addEventListener("abort", onAbort, { once: true });
      reader.read().then(resolve, reject).finally(() => {
        signal.removeEventListener("abort", onAbort);
      });
    });
  };
  try {
    while (true) {
      const { done, value } = await readWithAbort();
      throwIfAborted(signal);
      if (done) {
        break;
      }
      if (!value) {
        continue;
      }
      if (offset + value.byteLength > expectedLength) {
        throw invalidScalarVolumeResponse("body exceeds geometry length");
      }
      output.set(value, offset);
      offset += value.byteLength;
    }
  } catch (error) {
    void reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (offset !== expectedLength) {
    throw invalidScalarVolumeResponse(`body length ${offset} does not match geometry ${expectedLength}`);
  }
  return output.buffer;
};

const parseScalarVolumeResponse = async (
  response: Response,
  signal?: AbortSignal
): Promise<ScalarVolumePayload> => {
  throwIfAborted(signal);
  const width = requiredScalarHeaderInteger(response, "x-volume-width");
  const height = requiredScalarHeaderInteger(response, "x-volume-height");
  const depth = requiredScalarHeaderInteger(response, "x-volume-depth");
  const dtype = String(response.headers.get("x-volume-dtype") ?? "").trim().toLowerCase();
  const bytesPerVoxel = requiredScalarHeaderInteger(response, "x-volume-bytes-per-voxel");
  const expectedBytesPerVoxel = SCALAR_DTYPE_BYTES[dtype];
  if (expectedBytesPerVoxel == null || bytesPerVoxel !== expectedBytesPerVoxel) {
    throw invalidScalarVolumeResponse(`unsupported dtype/byte width ${dtype || "missing"}/${bytesPerVoxel}`);
  }
  const rawMin = requiredScalarHeaderNumber(response, "x-volume-raw-min");
  const rawMax = requiredScalarHeaderNumber(response, "x-volume-raw-max");
  if (rawMax < rawMin) {
    throw invalidScalarVolumeResponse("raw extrema are not ordered");
  }
  const sclSlope = requiredScalarHeaderNumber(response, "x-volume-scl-slope");
  const sclInter = requiredScalarHeaderNumber(response, "x-volume-scl-inter");
  if (sclSlope === 0) {
    throw invalidScalarVolumeResponse("zero x-volume-scl-slope");
  }
  const physicalMin = rawMin * sclSlope + sclInter;
  const physicalMax = rawMax * sclSlope + sclInter;
  if (!Number.isFinite(physicalMin) || !Number.isFinite(physicalMax)) {
    throw invalidScalarVolumeResponse("non-finite physical intensity transform");
  }
  const channel = requiredScalarHeaderIndex(response, "x-volume-channel");
  const time = requiredScalarHeaderIndex(response, "x-volume-time");
  const sourceWidth = requiredScalarHeaderInteger(response, "x-volume-source-width");
  const sourceHeight = requiredScalarHeaderInteger(response, "x-volume-source-height");
  const sourceDepth = requiredScalarHeaderInteger(response, "x-volume-source-depth");
  const downsampleX = requiredScalarHeaderInteger(response, "x-volume-downsample-x");
  const downsampleY = requiredScalarHeaderInteger(response, "x-volume-downsample-y");
  const downsampleZ = requiredScalarHeaderInteger(response, "x-volume-downsample-z");
  const previewPolicy = String(response.headers.get("x-volume-preview-policy") ?? "").trim();
  if (!previewPolicy) {
    throw invalidScalarVolumeResponse("invalid x-volume-preview-policy");
  }
  // Rolling compatibility: older intensity servers omitted this header and used
  // BOX delivery. Nearest remains fail-closed because mask membership depends on it.
  const sampling = String(response.headers.get("x-volume-sampling") ?? "").trim() || "box";
  if (sampling !== "box" && sampling !== "nearest") {
    throw invalidScalarVolumeResponse("invalid x-volume-sampling");
  }
  const previewAxes = [
    ["x", sourceWidth, downsampleX, width],
    ["y", sourceHeight, downsampleY, height],
    ["z", sourceDepth, downsampleZ, depth],
  ] as const;
  for (const [axis, sourceSize, factor, deliveredSize] of previewAxes) {
    if (Math.ceil(sourceSize / factor) !== deliveredSize) {
      throw invalidScalarVolumeResponse(`inconsistent ${axis}-axis preview provenance`);
    }
  }
  const voxelCount = width * height * depth;
  const expectedLength = voxelCount * bytesPerVoxel;
  const nativeIntegerMaskZeroCopy =
    previewPolicy === "mask-native-integer-v1" &&
    sampling === "nearest" &&
    (dtype === "uint8" || dtype === "uint16" || dtype === "int16") &&
    width === sourceWidth &&
    height === sourceHeight &&
    depth === sourceDepth &&
    downsampleX === 1 &&
    downsampleY === 1 &&
    downsampleZ === 1;
  const stagingLength =
    nativeIntegerMaskZeroCopy
      ? 0
      : voxelCount *
        (sampling === "nearest"
          ? SCALAR_RAW_FLOAT_BYTES_PER_VOXEL
          : SCALAR_HALF_FLOAT_BYTES_PER_VOXEL);
  const workingSetLength = expectedLength + stagingLength;
  if (
    !Number.isSafeInteger(voxelCount) ||
    !Number.isSafeInteger(expectedLength) ||
    !Number.isSafeInteger(stagingLength) ||
    !Number.isSafeInteger(workingSetLength) ||
    expectedLength <= 0
  ) {
    throw invalidScalarVolumeResponse("geometry overflows safe integer bounds");
  }
  if (expectedLength > MAX_SCALAR_VOLUME_BYTES) {
    throw invalidScalarVolumeResponse(
      `payload ${expectedLength} bytes exceeds the ${MAX_SCALAR_VOLUME_BYTES} byte preview limit`
    );
  }
  if (workingSetLength > MAX_SCALAR_VOLUME_WORKING_SET_BYTES) {
    throw invalidScalarVolumeResponse(
      `wire plus scalar texture staging requires ${workingSetLength} bytes, exceeding the ${MAX_SCALAR_VOLUME_WORKING_SET_BYTES} byte working-set limit`
    );
  }
  const declaredLengthRaw = response.headers.get("content-length");
  if (declaredLengthRaw != null && declaredLengthRaw.trim() !== "") {
    const declaredLength = Number(declaredLengthRaw);
    if (!Number.isSafeInteger(declaredLength) || declaredLength !== expectedLength) {
      throw invalidScalarVolumeResponse("content-length does not match geometry");
    }
  }
  const data = await readExactScalarVolumeBody(response, expectedLength, signal);
  throwIfAborted(signal);
  return {
    data,
    width,
    height,
    depth,
    dtype,
    bytesPerVoxel,
    rawMin,
    rawMax,
    sclSlope,
    sclInter,
    channel,
    time,
    sourceWidth,
    sourceHeight,
    sourceDepth,
    downsampleX,
    downsampleY,
    downsampleZ,
    previewPolicy,
    sampling,
  };
};

let viewerManifestModulePromise: Promise<typeof ViewerManifest> | null = null;

const loadViewerManifestModule = () => {
  viewerManifestModulePromise ??= import("./viewerManifest").catch((error: unknown) => {
    viewerManifestModulePromise = null;
    throw error;
  });
  return viewerManifestModulePromise;
};

const buildUrl = (
  baseUrl: string,
  path: string,
  params?: Record<string, string | string[]>
): string => {
  const url = new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((item) => url.searchParams.append(key, item));
        return;
      }
      url.searchParams.set(key, value);
    });
  }
  return url.toString();
};

/**
 * Add a composite channel selection to an image-service request.
 *
 * Viewer display state stores colors as a source-channel-indexed palette, while
 * the image-service contract requires one color per selected channel in exactly
 * the same order as `channels`. Projecting at this shared request boundary keeps
 * display, slice, and atlas URLs consistent and prevents cardinality mismatches.
 */
const MAX_IMAGE_CHANNEL_SELECTION = 8;

const applyImageChannelSelection = (
  params: Record<string, string>,
  channels?: number[],
  channelPalette?: string[]
): void => {
  const selectedChannels = Array.isArray(channels) ? channels : [];
  if (selectedChannels.length === 0) {
    return;
  }
  if (selectedChannels.length > MAX_IMAGE_CHANNEL_SELECTION) {
    throw new RangeError(`At most ${MAX_IMAGE_CHANNEL_SELECTION} image channels may be selected.`);
  }
  if (selectedChannels.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Image channel indices must be non-negative safe integers.");
  }
  if (new Set(selectedChannels).size !== selectedChannels.length) {
    throw new RangeError("Duplicate image channel indices are not allowed.");
  }
  params.channels = selectedChannels.map(String).join(",");

  if (!Array.isArray(channelPalette) || channelPalette.length === 0) {
    return;
  }
  const normalizedPalette = channelPalette.map((value) => String(value || "").trim());
  const selectedColors = selectedChannels.map((channel) => normalizedPalette[channel] ?? "");
  if (selectedColors.every(Boolean)) {
    params.channel_colors = selectedColors.join(",");
  }
};

const hexDigest = (digest: ArrayBuffer): string =>
  Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const V2_CHAT_THREAD_MAP_STORAGE_KEY = "bisque-ultra:v2-chat-thread-map";
const V2_CONVERSATION_STATE_METADATA_KEY = "frontend_state";
const V2_TITLE_STATE_METADATA_KEY = "title_state";
const V2_UPLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024;
const V2_UPLOAD_MAX_PARALLEL_FILES = 4;
const V2_UPLOAD_MAX_PARALLEL_CHUNKS = 8; // more in-flight chunks better saturate the link
const V2_UPLOAD_HARD_MAX_PARALLEL = 16;
const V2_UPLOAD_CHUNK_RETRY_DELAYS_MS = [120, 420];

const boundedUploadConcurrency = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 1) {
    return fallback;
  }
  return Math.max(1, Math.min(Math.floor(numeric), V2_UPLOAD_HARD_MAX_PARALLEL));
};

const uploadSessionMaxParallelFiles = (state: UploadSessionResponse): number =>
  boundedUploadConcurrency(state.limits?.max_parallel_files, V2_UPLOAD_MAX_PARALLEL_FILES);

const uploadSessionMaxParallelChunks = (state: UploadSessionResponse): number =>
  boundedUploadConcurrency(state.limits?.max_parallel_chunks, V2_UPLOAD_MAX_PARALLEL_CHUNKS);

const asPlainString = (value: unknown): string => String(value ?? "");

const asTrimmedString = (value: unknown): string => String(value ?? "").trim();

const uniqueTrimmedStrings = (values: readonly unknown[] | undefined): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  values?.forEach((value) => {
    const trimmed = asTrimmedString(value);
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    result.push(trimmed);
  });
  return result;
};

const resourceMetadataFilterSpecs = (
  filters: readonly ResourceMetadataFilter[] | undefined
): string[] => {
  const specs: string[] = [];
  filters?.forEach((filter) => {
    const path = asTrimmedString(filter.path);
    const operator = asTrimmedString(filter.operator).toLowerCase();
    const value = asTrimmedString(filter.value);
    if (!path || !operator) {
      return;
    }
    specs.push(`${path}:${operator}:${value}`);
  });
  return specs;
};

const asOptionalString = (value: unknown): string | null => {
  const text = asTrimmedString(value);
  return text.length > 0 ? text : null;
};

const browserUserTimeZone = (): string | null => {
  try {
    return asOptionalString(Intl.DateTimeFormat().resolvedOptions().timeZone);
  } catch {
    return null;
  }
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

/**
 * Runs the user deliberately removed, so hydration stops resurrecting them.
 *
 * Deleting or editing away the most recent turn leaves the saved state with no
 * assistant message carrying `thread.latest_run_id` — which is exactly the
 * condition reconciliation treats as "the snapshot is stale, refetch the run".
 * It then pushes the deleted answer back from `control_runs.response_text`, so
 * the message reappears on the next load and the delete looks broken.
 *
 * Capped, because this rides along in thread metadata on every snapshot write.
 * Oldest ids fall off first; losing an ancient tombstone is harmless, since the
 * run it names is long past being `latest_run_id`.
 */
const DELETED_RUN_ID_LIMIT = 200;

export const readDeletedRunIds = (state: Record<string, unknown>): string[] => {
  const raw = state.deletedRunIds;
  if (!Array.isArray(raw)) {
    return [];
  }
  const seen = new Set<string>();
  for (const entry of raw) {
    const id = asOptionalString(entry);
    if (id) {
      seen.add(id);
    }
  }
  return [...seen].slice(-DELETED_RUN_ID_LIMIT);
};

const frontendStateNeedsV2RunReconciliation = (
  thread: Record<string, unknown>,
  state: Record<string, unknown>
): boolean => {
  const latestRunId = asOptionalString(thread.latest_run_id);
  if (!latestRunId) {
    return Boolean(state.sending) || Boolean(asOptionalString(state.streamingMessageId));
  }

  // The user removed this turn on purpose. Its absence is the intended state,
  // not a stale snapshot to be repaired.
  if (readDeletedRunIds(state).includes(latestRunId)) {
    return false;
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
): Record<string, unknown> => {
  const state: Record<string, unknown> = {
    id: asOptionalString(message.message_id) ?? `${threadId || conversationId}-message-${index}`,
    role: asOptionalString(message.role) ?? "user",
    content: asPlainString(message.content),
    createdAt: asMillis(message.created_at, fallbackTime),
    runId: asOptionalString(message.run_id) ?? undefined,
  };
  // Steering rows keep their identity through hydration: the transcript shows
  // "Steered mid-run" instead of a bare user bubble.
  const metadata = isRecord(message.metadata) ? message.metadata : {};
  if (asOptionalString(metadata.kind) === "steering") {
    state.steering = "historic";
    state.steerId = asOptionalString(metadata.steer_id) ?? undefined;
  }
  return state;
};

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

const normalizeUploadProgressError = (error: unknown): string => {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return asPlainString(error) || "Upload failed";
};

const sleep = (delayMs: number): Promise<void> =>
  new Promise((resolve) => globalThis.setTimeout(resolve, Math.max(0, delayMs)));

const isRetryableUploadChunkError = (error: unknown): boolean => {
  if (error instanceof ApiError) {
    return error.status === 408 || error.status === 425 || error.status === 429 || error.status >= 500;
  }
  return error instanceof TypeError;
};

const isUploadSessionPausedConflict = (error: unknown): boolean =>
  error instanceof ApiError &&
  error.status === 409 &&
  /upload session is paused/i.test(error.message);

const isUploadSessionPaused = (
  options: UploadFilesOptions,
  sessionId: string,
  fileToken?: string | null
): boolean => Boolean(options.pauseSignal?.isPaused(sessionId, fileToken));

// Canonicalize a Feature-ID selection for DREAM3D microstructure filtering: dedup,
// reject non-digit / non-uint32 / zero (background), sort, cap. Throws RangeError on
// invalid input so the viewer can surface a clear message instead of a bad request.
export const canonicalizeHdf5FeatureIds = (values: readonly string[], maxIds = 64): string[] => {
  const unique = new Set<number>();
  values.forEach((raw) => {
    const token = String(raw);
    if (!/^[0-9]+$/.test(token)) {
      throw new RangeError("Feature IDs must contain digits only.");
    }
    const value = Number(token);
    if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffff_ffff) {
      throw new RangeError("Feature IDs must be positive uint32 values; 0 is background.");
    }
    unique.add(value);
  });
  const safeMaxIds = Math.max(1, Math.floor(maxIds));
  if (unique.size > safeMaxIds) {
    throw new RangeError(`Select at most ${safeMaxIds} unique Feature IDs.`);
  }
  return [...unique].sort((a, b) => a - b).map(String);
};

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
    let messagesFetchFailed = false;
    let messagesFetchError: unknown = null;
    if (threadId) {
      // The durable user + assistant turns live in control_thread_messages and
      // are read back here. When there is no client snapshot to fall back on,
      // this fetch is the ONLY source of the user's message — a transient
      // failure (swallowed before) would rebuild an assistant-only transcript
      // and silently drop the user's turn. Retry briefly before giving up.
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const payload = await this.fetchJson<Record<string, unknown>>(
            `/v2/threads/${encodeURIComponent(threadId)}/messages`,
            { method: "GET" }
          );
          messagesFetchFailed = false;
          messagesFetchError = null;
          if (Array.isArray(payload.messages)) {
            messages.push(...payload.messages.filter(isRecord));
          }
          break;
        } catch (error) {
          messagesFetchFailed = true;
          messagesFetchError = error;
          if (attempt < 2) {
            await new Promise<void>((resolve) => {
              setTimeout(resolve, 200 * (attempt + 1));
            });
          }
        }
      }
    }

    if (messagesFetchFailed) {
      // With a snapshot we can still open from local state (recovered); without
      // one, returning now would drop the user's message, so surface a
      // recoverable error instead of overwriting local state with a lossy rebuild.
      reportClientError(messagesFetchError, {
        source: "hydration-messages-fetch",
        recovered: hasExistingState,
        extra: { conversationId: record.conversation_id, hasExistingState },
      });
      if (!hasExistingState) {
        throw new ApiError(
          "This conversation could not be fully loaded. Please try again.",
          503,
          null
        );
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

    // A tombstoned run must not be rebuilt from the durable transcript. This
    // guard matters more than the one in the gate above: findAssistantPatchIndex
    // falls back to "last assistant with no runId", so without it the deleted
    // answer can be written over an unrelated surviving message rather than
    // merely reappearing.
    const tombstonedRunIds = readDeletedRunIds(existingState);
    if (latestRunId && !tombstonedRunIds.includes(latestRunId)) {
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

    // Instrumentation: a reconstructed transcript that has an assistant turn but
    // no user turn is the exact "only the response is left" symptom. Capture it
    // (handled, non-fatal) so the real-world trigger is visible in diagnostics.
    const hasUserTurn = stateMessages.some((message) => stateMessageRole(message) === "user");
    const hasAssistantTurn = stateMessages.some(
      (message) => stateMessageRole(message) === "assistant"
    );
    if (!hasExistingState && hasAssistantTurn && !hasUserTurn) {
      reportClientError(
        new Error("Hydrated transcript has an assistant turn but no user message"),
        {
          source: "hydration-anomaly",
          recovered: true,
          extra: {
            conversationId: record.conversation_id,
            latestRunId: latestRunId ?? null,
            messageCount: stateMessages.length,
            threadMessageCount: messages.length,
          },
        }
      );
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
      remote_mutation_intents: request.remote_mutation_intents ?? [],
      knowledge_context: request.knowledge_context ?? null,
      selection_context: request.selection_context ?? null,
      workflow_hint: request.workflow_hint ?? null,
      reasoning_mode: request.reasoning_mode ?? "auto",
      budgets: request.budgets ?? null,
      benchmark: request.benchmark ?? null,
      idempotency_key: asOptionalString(request.idempotency_key),
      metadata: {
        conversation_id: request.conversation_id ?? null,
        user_timezone: browserUserTimeZone(),
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

  async getCurrentUser(): Promise<CurrentUserResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/me"), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as CurrentUserResponse;
  }

  async updateCurrentUser(profile: CurrentUserProfile): Promise<CurrentUserResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/me"), {
      method: "PATCH",
      headers: this.headers({ "Content-Type": "application/json" }),
      credentials: "include",
      body: JSON.stringify(profile),
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as CurrentUserResponse;
  }

  async getTokenUsage(days?: number): Promise<TokenUsageResponse> {
    const params: Record<string, string> = {};
    const requestedDays = Math.max(1, Math.min(730, Number(days) || 365));
    params.days = String(requestedDays);
    const response = await fetch(buildUrl(this.baseUrl, "/v2/me/token-usage", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as TokenUsageResponse;
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

  async getAdminMetrics(options?: { rangeDays?: number }): Promise<AdminMetricsResponse> {
    const params: Record<string, string> = {};
    const rangeDays = Number(options?.rangeDays);
    if (Number.isFinite(rangeDays) && rangeDays > 0) {
      params.range_days = String(Math.round(rangeDays));
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/admin/metrics", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as AdminMetricsResponse;
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

  /**
   * Cancel a user's own run on the control plane. Stopping a chat must stop the
   * backend run (which keeps consuming worker capacity and model tokens), not
   * just disconnect the local stream. Best-effort: a run that is already
   * terminal returns 404/409, which callers treat as already-stopped.
   */
  async cancelRun(runId: string, reason?: string): Promise<void> {
    const normalizedRunId = asTrimmedString(runId);
    if (!normalizedRunId) {
      return;
    }
    const trimmedReason = asTrimmedString(reason);
    await this.fetchJson<Record<string, unknown>>(
      `/v2/runs/${encodeURIComponent(normalizedRunId)}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reason: trimmedReason || "Canceled from chat composer",
        }),
      }
    );
  }

  /**
   * Steer an in-flight run (Phase 1 of double texting): the text is applied
   * to the RUNNING agent at its next model-call boundary instead of waiting
   * for the turn to finish. A 409 with code "steering_closed" means the run
   * is terminal or finalizing — callers fall back to Phase 0 queueing.
   */
  async steerRun(
    runId: string,
    input: { steerId: string; text: string; fileIds?: string[] }
  ): Promise<RunSteerMessageRecord> {
    return await this.fetchJson<RunSteerMessageRecord>(
      `/v2/runs/${encodeURIComponent(runId)}/steer`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steer_id: input.steerId,
          text: input.text,
          // Attachments must ride the steer itself: the worker only grants
          // staging authority to control-plane-stamped ids, never to ids
          // that merely appear in the text.
          file_ids: input.fileIds ?? [],
        }),
      }
    );
  }

  /** The run's steering messages — used to verify whether a steer POST whose
   * response was lost actually landed before returning text to the draft. */
  async listRunSteerMessages(runId: string): Promise<RunSteerMessageRecord[]> {
    const response = await this.fetchJson<{ steer_messages?: RunSteerMessageRecord[] }>(
      `/v2/runs/${encodeURIComponent(runId)}/steer`
    );
    return response.steer_messages ?? [];
  }

  /* Notes — the personal layer. Markdown is the source of truth; every call
     is owner-scoped by the session. */
  async listNotes(options?: { query?: string; limit?: number; offset?: number }): Promise<NoteListResponse> {
    const params = new URLSearchParams();
    if (options?.query?.trim()) params.set("query", options.query.trim());
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.offset) params.set("offset", String(options.offset));
    const suffix = params.size > 0 ? `?${params.toString()}` : "";
    return await this.fetchJson<NoteListResponse>(`/v2/notes${suffix}`);
  }

  async createNote(payload: NoteWritePayload = {}): Promise<NoteRecord> {
    return await this.fetchJson<NoteRecord>(`/v2/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async getNote(noteId: string): Promise<NoteRecord> {
    return await this.fetchJson<NoteRecord>(`/v2/notes/${encodeURIComponent(noteId)}`);
  }

  async updateNote(noteId: string, payload: NoteWritePayload): Promise<NoteRecord> {
    return await this.fetchJson<NoteRecord>(`/v2/notes/${encodeURIComponent(noteId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async deleteNote(noteId: string): Promise<void> {
    await this.fetchJson<{ status: string }>(`/v2/notes/${encodeURIComponent(noteId)}`, {
      method: "DELETE",
    });
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

  async syncTrainingModel(modelKey: string): Promise<PrairieSyncResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/sync`), {
      method: "POST",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieSyncResponse;
  }

  async getTrainingModelStatus(modelKey: string): Promise<PrairieStatusResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/status`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieStatusResponse;
  }

  async runTrainingBenchmark(
    modelKey: string,
    request?: PrairieBenchmarkRunRequest
  ): Promise<PrairieBenchmarkRunResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/benchmark/run`), {
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

  async requestTrainingRetrain(modelKey: string, request: PrairieRetrainRequest): Promise<TrainingJobResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/retrain-request`), {
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

  async listTrainingRetrainRequests(modelKey: string): Promise<PrairieRetrainListResponse> {
    const response = await fetch(buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/retrain-requests`), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as PrairieRetrainListResponse;
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

  // GoldGate generic write routes (minted generic in the plan's section 3.6;
  // they 404 until the M1 backend lands - the capability-gated UI never calls
  // them before their echoes exist, and a stray 404 renders inline).
  async createGoldSetDraft(modelKey: string): Promise<Record<string, unknown>> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/models/${encodeURIComponent(modelKey)}/gold-sets`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({}),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Record<string, unknown>;
  }

  async freezeGoldSet(modelKey: string, goldSetId: string): Promise<Record<string, unknown>> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/training/models/${encodeURIComponent(modelKey)}/gold-sets/${encodeURIComponent(goldSetId)}/freeze`
      ),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ confirm: true }),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Record<string, unknown>;
  }

  async rejectModelVersion(versionId: string, reason?: string): Promise<Record<string, unknown>> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/training/model-versions/${encodeURIComponent(versionId)}/reject`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ reason: reason ?? "" }),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as Record<string, unknown>;
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

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.chatStream(request);
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
    // Reconnect-with-cursor orchestration around single-connection attempts.
    //
    // One severed or dead-but-open connection used to surface here as a thrown
    // "ended before a terminal event" / network error, which the App rendered
    // as a failed run while the run kept executing server-side — the guaranteed
    // outcome of closing a laptop on an overnight run. Instead: retry from the
    // shared cursor until the run itself reaches a terminal state. Caller
    // aborts and non-retryable API answers (auth, not-found) still throw
    // immediately; a run that finished while we were disconnected settles from
    // the run record exactly like one whose terminal event arrived in-stream.
    const ctx: V2RunStreamContext = {
      lastRunEventSequence: Math.max(0, Math.floor(Number(options?.afterSequence ?? 0))),
      seenTokenDeliveryKeys: new Set<string>(),
      progressEvents: [],
      streamedText: "",
      terminalEventSeen: false,
      terminalStatus: null,
      terminalDetail: null,
      terminalResponseText: "",
    };
    let settledRun: Record<string, unknown> | null = null;
    let attempt = 0;
    while (!ctx.terminalEventSeen) {
      throwIfAborted(options?.signal);
      try {
        await this.consumeV2RunEventStreamAttempt(runId, ctx, options);
        if (ctx.terminalEventSeen) {
          break;
        }
        // Clean stream end without a terminal event: either the run finished
        // while we were disconnected, or the connection was cut. Settle when
        // the run record is terminal; otherwise reconnect from the cursor.
        const snapshot = await this.getV2Run(runId).catch(() => null);
        const snapshotStatus = normalizeRunResultStatus(snapshot?.status ?? "");
        if (snapshotStatus && snapshotStatus !== "pending" && snapshotStatus !== "running") {
          settledRun = snapshot;
          break;
        }
      } catch (error) {
        if (isAbortError(error) || options?.signal?.aborted) {
          throw error;
        }
        if (!isRetryableRunStreamError(error)) {
          throw error;
        }
      }
      attempt += 1;
      await waitBeforeStreamRetry(attempt, options?.signal, options?.retryBaseDelayMs);
    }

    if (ctx.terminalStatus === "failed") {
      throw new ApiError("Run failed", 500, ctx.terminalDetail);
    }
    if (ctx.terminalStatus === "canceled") {
      throw new ApiError("Run canceled", 499, ctx.terminalDetail);
    }
    const finalRun = settledRun ?? (await this.getV2Run(runId).catch(() => null));
    if (!ctx.terminalEventSeen) {
      const finalStatus = normalizeRunResultStatus(finalRun?.status ?? fallbackRun.status);
      if (finalStatus === "failed") {
        throw new ApiError("Run failed", 500, finalRun ?? fallbackRun);
      }
      if (finalStatus === "canceled") {
        throw new ApiError("Run canceled", 499, finalRun ?? fallbackRun);
      }
      // pending/running is unreachable here: the loop above only exits on a
      // terminal in-stream event or a terminal run snapshot.
    }
    const responseText =
      ctx.terminalResponseText || asTrimmedString(finalRun?.response_text) || ctx.streamedText;
    const completedPayload = normalizeV2RunResponse(finalRun ?? fallbackRun, {
      runId,
      responseText,
      progressEvents: ctx.progressEvents,
      metadata: isRecord(ctx.terminalDetail) ? ctx.terminalDetail : null,
    });
    options?.onDone?.(completedPayload);
    return completedPayload;
  }

  private async consumeV2RunEventStreamAttempt(
    runId: string,
    ctx: V2RunStreamContext,
    options?: RunStreamOptions
  ): Promise<void> {
    const streamParams: Record<string, string> = { stream: "true" };
    streamParams.after_sequence = String(ctx.lastRunEventSequence);

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
    let buffer = "";
    // Dead-but-open sockets (the post-sleep signature) deliver no bytes and no
    // error. The server heartbeats every 15s, so a silent minute means the
    // connection is gone: cancel the read and let the orchestrator reconnect.
    const inactivityTimeoutMs = Math.max(
      1000,
      Math.floor(Number(options?.inactivityTimeoutMs ?? RUN_STREAM_INACTIVITY_TIMEOUT_MS))
    );
    let idleTimer: ReturnType<typeof setTimeout> | undefined;
    const armIdleTimer = () => {
      if (idleTimer !== undefined) {
        clearTimeout(idleTimer);
      }
      idleTimer = setTimeout(() => {
        void reader
          .cancel(new DOMException("Run stream idle past heartbeat window", "TimeoutError"))
          .catch(() => undefined);
      }, inactivityTimeoutMs);
    };

    const tokenDeliveryKey = (event: StreamTokenEvent): string | null => {
      const eventID = asTrimmedString(event.eventId);
      if (eventID) {
        return `event_id:${eventID}`;
      }
      const sequence = Math.floor(Number(event.sequence ?? 0));
      if (!Number.isFinite(sequence) || sequence <= 0) {
        return null;
      }
      const tokenRunID = asTrimmedString(event.runId) || runId;
      return `sequence:${tokenRunID}:${sequence}`;
    };

    const emitTokenOnce = (delta: string, event: StreamTokenEvent): void => {
      const key = tokenDeliveryKey(event);
      if (key) {
        // Shared across reconnect attempts: a replayed delta after resume is
        // dropped here even when the server's replay overlaps the cursor.
        if (ctx.seenTokenDeliveryKeys.has(key)) {
          return;
        }
        ctx.seenTokenDeliveryKeys.add(key);
      }
      ctx.streamedText += delta;
      options?.onToken?.(delta, event);
    };

    const handleStreamEvent = (eventName: string, payload: unknown): void => {
      if (eventName === "heartbeat") {
        return;
      }
      if (eventName === "token" && isRecord(payload)) {
        const delta = asPlainString(payload.delta);
        if (delta) {
          const sequence = Math.floor(asFiniteNumber(payload.sequence, 0));
          emitTokenOnce(delta, {
            sequence: sequence > 0 ? sequence : undefined,
            eventId: asTrimmedString(payload.event_id ?? payload.eventId) || undefined,
            runId: asTrimmedString(payload.run_id) || runId,
            eventKind: "token",
          });
        }
        return;
      }
      if (eventName === "error") {
        ctx.terminalEventSeen = true;
        ctx.terminalStatus = "failed";
        ctx.terminalDetail = payload;
        return;
      }
      if (eventName !== "run_event" || !isRecord(payload)) {
        return;
      }
      // The server delivers run events in strictly increasing sequence order;
      // anything at or below the last seen sequence is a duplicate (e.g. a
      // replay overlap after reconnect) and must not be appended again.
      const eventSequence = Math.floor(asFiniteNumber(payload.sequence, 0));
      if (eventSequence > 0) {
        if (eventSequence <= ctx.lastRunEventSequence) {
          return;
        }
        ctx.lastRunEventSequence = eventSequence;
      }

      const eventKind =
        asTrimmedString(payload.event_kind) || asTrimmedString(payload.event_type) || "run_event";

      // Token deltas are ephemeral and belong on EXACTLY one path: the rAF-batched text
      // stream (onToken -> streamedText). They must never enter the runEvents/progressEvents
      // arrays. Routing each of the ~37k deltas/turn through onRunEvent rebuilt the entire
      // messages array and ran an O(n) dedup scan PER TOKEN, saturating the main thread and
      // freezing the tab on long conversations. The sequence-dedup above still runs first, so
      // reconnect/replay overlap stays correct — a re-delivered delta is dropped before here.
      if (eventKind === "message.delta") {
        const delta = v2DeltaFromRunEvent(payload);
        if (delta) {
          const eventID = asTrimmedString(payload.event_id);
          const tokenEvent: StreamTokenEvent = {
            sequence: eventSequence > 0 ? eventSequence : undefined,
            eventId: eventID || undefined,
            runId: asTrimmedString(payload.run_id) || undefined,
            eventKind,
          };
          emitTokenOnce(delta, tokenEvent);
        }
        return;
      }
      // Other ephemeral per-token deltas (subagent/trace text) carry no durable trace meaning and
      // are not rendered token-by-token; drop them so they never bloat runEvents either. (The
      // sequence-dedup above already ran, so the reconnect cursor still advances past them.)
      if (isEphemeralDeltaEventKind(eventKind)) {
        return;
      }

      if (eventKind === "artifact.created") {
        this.rememberV2ArtifactEvent(payload);
      }

      const normalized = normalizeV2RunEvent(payload);
      // Reasoning deltas are cumulative live-thinking snapshots: onRunEvent coalesces
      // them into a single runEvents slot, and no progress-event consumer renders
      // them, so keeping them out of progressEvents stops the array (and every
      // persisted snapshot of it) growing by one entry per ~0.4s for the whole run.
      if (eventKind !== "trace.reasoning.delta") {
        ctx.progressEvents.push(progressEventFromV2RunEvent(payload));
      }
      options?.onRunEvent?.(normalized);

      if (!isV2TerminalEventKind(eventKind)) {
        return;
      }
      ctx.terminalEventSeen = true;
      ctx.terminalResponseText = responseTextFromV2TerminalEvent(payload);
      ctx.terminalStatus =
        eventKind === "run.completed" ? "succeeded" : eventKind === "run.canceled" ? "canceled" : "failed";
      ctx.terminalDetail = isRecord(payload.payload) ? payload.payload : payload;
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
      armIdleTimer();
      while (true) {
        const { value, done } = await reader.read();
        armIdleTimer();
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
          if (ctx.terminalEventSeen) {
            break;
          }
        }
        if (ctx.terminalEventSeen) {
          break;
        }
      }
      if (!ctx.terminalEventSeen) {
        buffer += decoder.decode().replace(/\r\n/g, "\n");
      }
      if (buffer.trim().length > 0) {
        parseEventBlock(buffer);
      }
    } finally {
      if (idleTimer !== undefined) {
        clearTimeout(idleTimer);
      }
      if (ctx.terminalEventSeen) {
        try {
          await reader.cancel();
        } catch {
          // The stream is already complete or no longer cancelable.
        }
      }
      reader.releaseLock();
    }
  }

  async uploadFiles(files: File[], options: UploadFilesOptions = {}): Promise<UploadFilesResponse> {
    if (files.length === 0) {
      return { file_count: 0, uploaded: [] };
    }
    // Bundle finalization lives only in the multi-file session path, so a
    // single-member zarr store (one .zattrs, a lone chunk) must route there
    // too — the single-file path would land its bytes without ever cataloging
    // the bundle resource.
    const singleFileIsBundleMember =
      files.length === 1 &&
      Boolean(bundleRootForRelativePath(files[0].webkitRelativePath ?? ""));
    if (files.length > 1 || singleFileIsBundleMember) {
      return this.uploadMultipleFilesWithV2Session(files, options);
    }
    const uploaded: UploadFilesResponse["uploaded"] = [];
    for (const [index, file] of files.entries()) {
      uploaded.push(await this.uploadFileWithV2Session(file, index, options));
    }
    return { file_count: uploaded.length, uploaded };
  }

  async getUploadSessionStatus(sessionId: string): Promise<UploadSessionResponse> {
    return this.getUploadSession(sessionId);
  }

  async pauseUploadSession(sessionId: string): Promise<UploadSessionResponse> {
    return this.updateUploadSessionControl(
      `/v2/upload-sessions/${encodeURIComponent(sessionId)}/pause`
    );
  }

  async resumeUploadSession(sessionId: string): Promise<UploadSessionResponse> {
    return this.updateUploadSessionControl(
      `/v2/upload-sessions/${encodeURIComponent(sessionId)}/resume`
    );
  }

  async cancelUploadSession(sessionId: string): Promise<UploadSessionResponse> {
    return this.updateUploadSessionControl(
      `/v2/upload-sessions/${encodeURIComponent(sessionId)}/cancel`
    );
  }

  private async createUploadSession(
    payload: UploadSessionCreateRequest
  ): Promise<UploadSessionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/upload-sessions"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadSessionResponse;
  }

  private async getUploadSession(sessionId: string): Promise<UploadSessionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/upload-sessions/${encodeURIComponent(sessionId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadSessionResponse;
  }

  private async updateUploadSessionControl(path: string): Promise<UploadSessionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, path),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadSessionResponse;
  }

  private async uploadSessionChunk(
    sessionId: string,
    fileToken: string,
    chunkIndex: number,
    offset: number,
    chunk: Blob
  ): Promise<UploadChunkResponse> {
    const chunkSha = await this.sha256Blob(chunk);
    const chunkUrl = buildUrl(
      this.baseUrl,
      `/v2/upload-sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(
        fileToken
      )}/chunks/${encodeURIComponent(String(chunkIndex))}`
    );
    for (let attemptIndex = 0; ; attemptIndex += 1) {
      try {
        const response = await fetch(chunkUrl, {
          method: "PUT",
          headers: this.headers({
            "Content-Type": "application/octet-stream",
            "X-Upload-Offset": String(Math.max(0, Math.floor(offset))),
            "X-Upload-Chunk-Sha256": chunkSha,
          }),
          body: chunk,
          credentials: "include",
        });
        if (!response.ok) {
          await parseError(response);
        }
        return (await response.json()) as UploadChunkResponse;
      } catch (error) {
        const retryDelayMs = V2_UPLOAD_CHUNK_RETRY_DELAYS_MS[attemptIndex];
        if (retryDelayMs === undefined || !isRetryableUploadChunkError(error)) {
          throw error;
        }
        await sleep(retryDelayMs);
      }
    }
  }

  private async completeUploadSessionFile(
    sessionId: string,
    fileToken: string
  ): Promise<UploadSessionFileCompleteResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/upload-sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(
          fileToken
        )}/complete`
      ),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadSessionFileCompleteResponse;
  }

  // Commit directory-format bundles (OME-Zarr folder uploads) after all member files have
  // completed: the server reconstructed the tree under bundles/{id} and finalize-bundle
  // creates ONE catalog resource per bundle. Returns the bundle resources.
  private async finalizeUploadBundle(sessionId: string): Promise<UploadedFileRecord[]> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/upload-sessions/${encodeURIComponent(sessionId)}/finalize-bundle`),
      { method: "POST", headers: this.headers(), credentials: "include" }
    );
    if (!response.ok) {
      return parseError(response);
    }
    const data = (await response.json()) as { bundles?: UploadedFileRecord[] };
    return data.bundles ?? [];
  }

  private async buildV2UploadFilePlan(file: File, fileIndex: number): Promise<V2UploadFilePlan> {
    const fileToken = await this.uploadFileToken(file, fileIndex);
    const fingerprint = await this.fileFingerprint(file);
    const relativePath = this.fileRelativePath(file);
    const contentType = file.type || "application/octet-stream";
    const chunkSize = Math.max(256 * 1024, V2_UPLOAD_CHUNK_SIZE_BYTES);
    return {
      file,
      fileIndex,
      fileToken,
      fingerprint,
      relativePath,
      contentType,
      chunkSize,
      progressID: `${fileToken}:${file.size}:${file.lastModified}`,
    };
  }

  private emitV2UploadProgress(
    plan: V2UploadFilePlan,
    options: UploadFilesOptions,
    event: Omit<UploadProgressEvent, "id" | "fileName" | "fileIndex" | "totalBytes" | "fingerprint" | "relativePath" | "contentType" | "chunkSizeBytes">
  ): void {
    options.onProgress?.({
      id: plan.progressID,
      fileName: plan.file.name || "upload.bin",
      fileIndex: plan.fileIndex,
      totalBytes: plan.file.size,
      fingerprint: plan.fingerprint,
      relativePath: plan.relativePath ?? undefined,
      contentType: plan.contentType,
      chunkSizeBytes: plan.chunkSize,
      ...event,
    });
  }

  private async uploadBatchFingerprint(plans: V2UploadFilePlan[]): Promise<string> {
    const manifest = plans
      .map((plan) => `${plan.fingerprint}\t${plan.fileToken}\t${plan.relativePath ?? ""}`)
      .join("\n");
    const digest = await this.sha256Text(manifest);
    return `batch:${plans.length}:${digest}`;
  }

  private async uploadMultipleFilesWithV2Session(
    files: File[],
    options: UploadFilesOptions
  ): Promise<UploadFilesResponse> {
    const plans = await Promise.all(files.map((file, index) => this.buildV2UploadFilePlan(file, index)));
    const latestProgressById = new Map<string, UploadProgressEvent>();
    const trackedOptions: UploadFilesOptions = {
      ...options,
      onProgress: (event) => {
        latestProgressById.set(event.id, event);
        options.onProgress?.(event);
      },
    };
    plans.forEach((plan) => {
      this.emitV2UploadProgress(plan, trackedOptions, {
        fileToken: plan.fileToken,
        status: "creating",
        bytesVerified: 0,
        bytesCommitted: 0,
      });
    });
    try {
      const idempotencyKey = await this.uploadBatchFingerprint(plans);
      let state = await this.createUploadSession({
        idempotency_key: idempotencyKey,
        browser_fingerprint: idempotencyKey,
        total_bytes: plans.reduce((sum, plan) => sum + plan.file.size, 0),
        files: plans.map((plan) => ({
          file_token: plan.fileToken,
          original_name: plan.file.name || "upload.bin",
          relative_path: plan.relativePath ?? undefined,
          content_type: plan.contentType,
          size_bytes: plan.file.size,
        })),
      });
      const sessionId = state.session.session_id;
      if (!sessionId) {
        throw new ApiError("Upload session response was missing session state", 500, state);
      }
      if (state.session.status === "paused") {
        state = await this.resumeUploadSession(sessionId);
      }
      const uploaded: UploadFilesResponse["uploaded"] = Array.from({ length: plans.length });
      let nextPlanTaskIndex = 0;
      const uploadNextFile = async () => {
        for (;;) {
          const plan = plans[nextPlanTaskIndex];
          nextPlanTaskIndex += 1;
          if (!plan) {
            return;
          }
        const sessionFile = state.files.find((item) => item.file_token === plan.fileToken);
        if (!sessionFile?.file_token) {
          throw new ApiError("Upload session response was missing session file state", 500, {
            session_id: sessionId,
            file_token: plan.fileToken,
          });
        }
          uploaded[plan.fileIndex] = await this.uploadFilePlanInV2Session(
            plan,
            state,
            sessionId,
            sessionFile,
            trackedOptions
          );
        }
      };
      const workerCount = Math.min(uploadSessionMaxParallelFiles(state), plans.length);
      await Promise.all(Array.from({ length: workerCount }, () => uploadNextFile()));
      // Directory-format bundles (OME-Zarr folder upload): the per-member completes wrote
      // into a bundle tree but created no per-file resources — finalize commits ONE resource
      // per bundle and we return those instead of the N member stubs.
      const bundleMap = state.session.metadata?.bundles;
      if (bundleMap && typeof bundleMap === "object" && Object.keys(bundleMap as Record<string, unknown>).length > 0) {
        const bundles = await this.finalizeUploadBundle(sessionId);
        if (bundles.length > 0) {
          return { file_count: bundles.length, uploaded: bundles };
        }
      }
      return { file_count: uploaded.length, uploaded };
    } catch (error) {
      const pausedError =
        error instanceof UploadPausedError || isUploadSessionPausedConflict(error)
          ? error instanceof UploadPausedError
            ? error
            : new UploadPausedError("")
          : null;
      plans.forEach((plan) => {
        const latestProgress = latestProgressById.get(plan.progressID);
        if (latestProgress?.status === "completed") {
          return;
        }
        if (pausedError) {
          this.emitV2UploadProgress(plan, trackedOptions, {
            sessionId: latestProgress?.sessionId,
            fileToken: latestProgress?.fileToken ?? plan.fileToken,
            status: "paused",
            bytesVerified: latestProgress?.bytesVerified ?? 0,
            bytesCommitted: latestProgress?.bytesCommitted ?? 0,
          });
          return;
        }
        this.emitV2UploadProgress(plan, trackedOptions, {
          fileToken: plan.fileToken,
          status: "failed",
          bytesVerified: latestProgress?.bytesVerified ?? 0,
          bytesCommitted: latestProgress?.bytesCommitted ?? 0,
          error: normalizeUploadProgressError(error),
        });
      });
      if (pausedError) {
        throw pausedError;
      }
      throw error;
    }
  }

  private async uploadFilePlanInV2Session(
    plan: V2UploadFilePlan,
    state: UploadSessionResponse,
    sessionId: string,
    sessionFile: UploadSessionResponse["files"][number],
    options: UploadFilesOptions
  ): Promise<UploadFilesResponse["uploaded"][number]> {
    if (sessionFile.status === "completed" && sessionFile.resource_id) {
      const complete = await this.completeUploadSessionFile(sessionId, sessionFile.file_token);
      this.emitV2UploadProgress(plan, options, {
        sessionId,
        fileToken: sessionFile.file_token,
        status: "completed",
        bytesVerified: plan.file.size,
        bytesCommitted: plan.file.size,
      });
      return complete.resource;
    }

    const verifiedChunks = new Map<number, { offset: number; size: number }>();
    (state.chunks ?? [])
      .filter((chunk) => chunk.file_token === sessionFile.file_token && chunk.status === "verified")
      .forEach((chunk) => {
        verifiedChunks.set(chunk.chunk_index, { offset: chunk.offset, size: chunk.size_bytes });
      });

    const missingChunks: Array<{ chunkIndex: number; offset: number; end: number }> = [];
    const verifiedChunkSizes = new Map<number, number>();
    for (let offset = 0, chunkIndex = 0; offset < plan.file.size; chunkIndex += 1) {
      const end = Math.min(offset + plan.chunkSize, plan.file.size);
      const existing = verifiedChunks.get(chunkIndex);
      if (existing && existing.offset === offset && existing.size === end - offset) {
        verifiedChunkSizes.set(chunkIndex, existing.size);
      } else {
        missingChunks.push({ chunkIndex, offset, end });
      }
      offset = end;
    }

    let nextChunkTaskIndex = 0;
    const verifiedBytes = () =>
      Array.from(verifiedChunkSizes.values()).reduce((sum, size) => sum + size, 0);
    const emitPaused = () => {
      this.emitV2UploadProgress(plan, options, {
        sessionId,
        fileToken: sessionFile.file_token,
        status: "paused",
        bytesVerified: verifiedBytes(),
        bytesCommitted: 0,
      });
    };
    const throwIfPaused = () => {
      if (isUploadSessionPaused(options, sessionId, sessionFile.file_token)) {
        emitPaused();
        throw new UploadPausedError(sessionId, sessionFile.file_token);
      }
    };
    const uploadNextChunk = async () => {
      for (;;) {
        throwIfPaused();
        const task = missingChunks[nextChunkTaskIndex];
        nextChunkTaskIndex += 1;
        if (!task) {
          return;
        }
        let chunkResponse: UploadChunkResponse;
        try {
          chunkResponse = await this.uploadSessionChunk(
            sessionId,
            sessionFile.file_token,
            task.chunkIndex,
            task.offset,
            plan.file.slice(task.offset, task.end)
          );
        } catch (error) {
          if (isUploadSessionPausedConflict(error)) {
            emitPaused();
            throw new UploadPausedError(sessionId, sessionFile.file_token);
          }
          throw error;
        }
        verifiedChunkSizes.set(task.chunkIndex, chunkResponse.chunk.size_bytes);
        this.emitV2UploadProgress(plan, options, {
          sessionId,
          fileToken: sessionFile.file_token,
          status: "uploading",
          bytesVerified: verifiedBytes(),
          bytesCommitted: 0,
        });
        throwIfPaused();
      }
    };
    const workerCount = Math.min(uploadSessionMaxParallelChunks(state), missingChunks.length);
    await Promise.all(Array.from({ length: workerCount }, () => uploadNextChunk()));

    throwIfPaused();
    const complete = await this.completeUploadSessionFile(sessionId, sessionFile.file_token);
    this.emitV2UploadProgress(plan, options, {
      sessionId,
      fileToken: sessionFile.file_token,
      status: "completed",
      bytesVerified: plan.file.size,
      bytesCommitted: plan.file.size,
    });
    return complete.resource;
  }

  private async uploadFileWithV2Session(
    file: File,
    fileIndex: number,
    options: UploadFilesOptions
  ): Promise<UploadFilesResponse["uploaded"][number]> {
    const resumeSessionId = asTrimmedString(options.resumeSession?.sessionId);
    const resumeFileToken = asTrimmedString(options.resumeSession?.fileToken);
    const resumeProgressId = asTrimmedString(options.resumeSession?.progressId);
    const fileToken = resumeFileToken || (await this.uploadFileToken(file, fileIndex));
    const fingerprint = await this.fileFingerprint(file);
    const relativePath = this.fileRelativePath(file);
    const contentType = file.type || "application/octet-stream";
    const chunkSize = Math.max(256 * 1024, V2_UPLOAD_CHUNK_SIZE_BYTES);
    const progressID = resumeProgressId || `${fileToken}:${file.size}:${file.lastModified}`;
    const emitProgress = (event: Omit<UploadProgressEvent, "id" | "fileName" | "fileIndex" | "totalBytes">) => {
      options.onProgress?.({
        id: progressID,
        fileName: file.name || "upload.bin",
        fileIndex,
        totalBytes: file.size,
        fingerprint,
        relativePath: relativePath ?? undefined,
        contentType,
        chunkSizeBytes: chunkSize,
        ...event,
      });
    };
    emitProgress({
      sessionId: resumeSessionId || undefined,
      fileToken,
      status: "creating",
      bytesVerified: 0,
      bytesCommitted: 0,
    });
    let latestSessionId: string | undefined;
    let latestFileToken = fileToken;
    let latestBytesVerified = 0;
    let latestBytesCommitted = 0;
    try {
      let state = resumeSessionId
        ? await this.getUploadSession(resumeSessionId)
        : await this.createUploadSession({
            idempotency_key: fingerprint,
            browser_fingerprint: fingerprint,
            total_bytes: file.size,
            files: [
              {
                file_token: fileToken,
                original_name: file.name || "upload.bin",
                relative_path: relativePath ?? undefined,
                content_type: contentType,
                size_bytes: file.size,
              },
            ],
          });
      const sessionId = state.session.session_id;
      if (!sessionId) {
        throw new ApiError("Upload session response was missing session file state", 500, state);
      }
      latestSessionId = sessionId;
      latestBytesVerified = Math.max(latestBytesVerified, state.session.bytes_verified ?? 0);
      latestBytesCommitted = Math.max(latestBytesCommitted, state.session.bytes_committed ?? 0);
      if (state.session.status === "paused") {
        state = await this.resumeUploadSession(sessionId);
        latestBytesVerified = Math.max(latestBytesVerified, state.session.bytes_verified ?? 0);
        latestBytesCommitted = Math.max(latestBytesCommitted, state.session.bytes_committed ?? 0);
      }
      const sessionFile = resumeFileToken
        ? state.files.find((item) => item.file_token === resumeFileToken)
        : state.files.find((item) => item.file_token === fileToken) ?? state.files[0];
      if (!sessionFile?.file_token) {
        throw new ApiError("Upload session response was missing session file state", 500, state);
      }
      latestFileToken = sessionFile.file_token;
      if (state.session.status === "completed" && sessionFile.resource_id) {
        const complete = await this.completeUploadSessionFile(sessionId, sessionFile.file_token);
        latestBytesVerified = Math.max(latestBytesVerified, complete.session.bytes_verified);
        latestBytesCommitted = Math.max(latestBytesCommitted, complete.session.bytes_committed);
        emitProgress({
          sessionId,
          fileToken: sessionFile.file_token,
          status: "completed",
          bytesVerified: latestBytesVerified,
          bytesCommitted: latestBytesCommitted,
        });
        return complete.resource;
      }

      const verifiedChunks = new Map<number, { offset: number; size: number }>();
      (state.chunks ?? [])
        .filter((chunk) => chunk.file_token === sessionFile.file_token && chunk.status === "verified")
        .forEach((chunk) => {
          verifiedChunks.set(chunk.chunk_index, { offset: chunk.offset, size: chunk.size_bytes });
        });

      const missingChunks: Array<{ chunkIndex: number; offset: number; end: number }> = [];
      for (let offset = 0, chunkIndex = 0; offset < file.size; chunkIndex += 1) {
        const end = Math.min(offset + chunkSize, file.size);
        const existing = verifiedChunks.get(chunkIndex);
        if (!existing || existing.offset !== offset || existing.size !== end - offset) {
          missingChunks.push({ chunkIndex, offset, end });
        }
        offset = end;
      }
      let nextChunkTaskIndex = 0;
      const emitPaused = () => {
        emitProgress({
          sessionId,
          fileToken: sessionFile.file_token,
          status: "paused",
          bytesVerified: latestBytesVerified,
          bytesCommitted: latestBytesCommitted,
        });
      };
      const throwIfPaused = () => {
        if (isUploadSessionPaused(options, sessionId, sessionFile.file_token)) {
          emitPaused();
          throw new UploadPausedError(sessionId, sessionFile.file_token);
        }
      };
      const uploadNextChunk = async () => {
        for (;;) {
          throwIfPaused();
          const task = missingChunks[nextChunkTaskIndex];
          nextChunkTaskIndex += 1;
          if (!task) {
            return;
          }
          let chunkResponse: UploadChunkResponse;
          try {
            chunkResponse = await this.uploadSessionChunk(
              sessionId,
              sessionFile.file_token,
              task.chunkIndex,
              task.offset,
              file.slice(task.offset, task.end)
            );
          } catch (error) {
            if (isUploadSessionPausedConflict(error)) {
              emitPaused();
              throw new UploadPausedError(sessionId, sessionFile.file_token);
            }
            throw error;
          }
          latestBytesVerified = Math.max(latestBytesVerified, chunkResponse.session.bytes_verified);
          latestBytesCommitted = Math.max(latestBytesCommitted, chunkResponse.session.bytes_committed);
          emitProgress({
            sessionId,
            fileToken: sessionFile.file_token,
            status: "uploading",
            bytesVerified: latestBytesVerified,
            bytesCommitted: latestBytesCommitted,
          });
          throwIfPaused();
        }
      };
      const workerCount = Math.min(uploadSessionMaxParallelChunks(state), missingChunks.length);
      await Promise.all(Array.from({ length: workerCount }, () => uploadNextChunk()));

      throwIfPaused();
      try {
        const complete = await this.completeUploadSessionFile(sessionId, sessionFile.file_token);
        latestBytesVerified = Math.max(latestBytesVerified, complete.session.bytes_verified);
        latestBytesCommitted = Math.max(latestBytesCommitted, complete.session.bytes_committed);
        emitProgress({
          sessionId,
          fileToken: sessionFile.file_token,
          status: "completed",
          bytesVerified: latestBytesVerified,
          bytesCommitted: latestBytesCommitted,
        });
        return complete.resource;
      } catch (error) {
        if (this.isRecoverableResumableUploadError(error)) {
          state = await this.getUploadSession(sessionId);
          latestBytesVerified = Math.max(latestBytesVerified, state.session.bytes_verified ?? 0);
          latestBytesCommitted = Math.max(latestBytesCommitted, state.session.bytes_committed ?? 0);
          if (state.session.status === "completed") {
            const complete = await this.completeUploadSessionFile(sessionId, sessionFile.file_token);
            latestBytesVerified = Math.max(latestBytesVerified, complete.session.bytes_verified);
            latestBytesCommitted = Math.max(latestBytesCommitted, complete.session.bytes_committed);
            emitProgress({
              sessionId,
              fileToken: sessionFile.file_token,
              status: "completed",
              bytesVerified: latestBytesVerified,
              bytesCommitted: latestBytesCommitted,
            });
            return complete.resource;
          }
        }
        throw error;
      }
    } catch (error) {
      if (error instanceof UploadPausedError) {
        throw error;
      }
      if (isUploadSessionPausedConflict(error) && latestSessionId) {
        emitProgress({
          sessionId: latestSessionId,
          fileToken: latestFileToken,
          status: "paused",
          bytesVerified: latestBytesVerified,
          bytesCommitted: latestBytesCommitted,
        });
        throw new UploadPausedError(latestSessionId, latestFileToken);
      }
      emitProgress({
        sessionId: latestSessionId,
        fileToken: latestFileToken,
        status: "failed",
        bytesVerified: latestBytesVerified,
        bytesCommitted: latestBytesCommitted,
        error: normalizeUploadProgressError(error),
      });
      throw error;
    }
  }

  private async fileFingerprint(file: File): Promise<string> {
    const identityName = this.fileRelativePath(file) ?? (file.name || "upload.bin");
    const base = `${identityName}:${file.size}:${file.lastModified}:${file.type || "application/octet-stream"}`;
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

  private fileRelativePath(file: File): string | null {
    const rawPath = String((file as File & { webkitRelativePath?: string }).webkitRelativePath ?? "").trim();
    if (!rawPath) {
      return null;
    }
    const normalized = rawPath
      .replace(/\\/g, "/")
      .split("/")
      .map((segment) => segment.trim())
      .filter((segment) => segment.length > 0 && segment !== "." && segment !== "..")
      .join("/");
    return normalized.length > 0 ? normalized : null;
  }

  private async uploadFileToken(file: File, fileIndex: number): Promise<string> {
    const fingerprint = await this.fileFingerprint(file);
    const digest = await this.sha256Text(fingerprint);
    return `file-${fileIndex}-${digest.slice(0, 16)}`;
  }

  private async sha256Text(value: string): Promise<string> {
    const crypto = globalThis.crypto;
    if (crypto?.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
      return hexDigest(digest);
    }
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return Math.abs(hash >>> 0).toString(16).padStart(8, "0").repeat(8);
  }

  private async sha256Blob(blob: Blob): Promise<string> {
    const crypto = globalThis.crypto;
    if (!crypto?.subtle) {
      throw new ApiError("Browser crypto is required for verified resumable uploads", 0, null);
    }
    const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    return hexDigest(digest);
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

  async pushResourcesToBisque(options: BisquePushRequest): Promise<BisquePushResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/bisque/push"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        file_ids: options.fileIds ?? [],
        collection_ids: options.collectionIds ?? [],
        dataset_name: options.datasetName ?? "",
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as BisquePushResponse;
  }

  async listResources(options?: {
    limit?: number;
    offset?: number;
    query?: string;
    kind?: "image" | "video" | "table" | "file";
    source?: "upload" | "bisque_import";
    sharing?: string;
    processingStatus?: string;
    status?: string;
    tags?: string[];
    descriptors?: string[];
    metadataFilters?: ResourceMetadataFilter[];
    createdAfter?: string;
    createdBefore?: string;
  }): Promise<ResourceListResponse> {
    const params: Record<string, string | string[]> = {
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
    const sharing = String(options?.sharing ?? "").trim();
    if (sharing && sharing !== "all") {
      params.sharing = sharing;
    }
    const processingStatus = String(options?.processingStatus ?? "").trim();
    if (processingStatus && processingStatus !== "all") {
      params.processing_status = processingStatus;
    }
    const status = String(options?.status ?? "").trim();
    if (status && status !== "active") {
      params.status = status;
    }
    const tags = uniqueTrimmedStrings(options?.tags);
    if (tags.length > 0) {
      params.tags = tags.join(",");
    }
    const descriptors = uniqueTrimmedStrings(options?.descriptors);
    if (descriptors.length > 0) {
      params.descriptors = descriptors.join(",");
    }
    const metadataFilterSpecs = resourceMetadataFilterSpecs(options?.metadataFilters);
    if (metadataFilterSpecs.length > 0) {
      params.metadata_filter = metadataFilterSpecs;
    }
    const createdAfter = String(options?.createdAfter ?? "").trim();
    if (createdAfter) {
      params.created_after = createdAfter;
    }
    const createdBefore = String(options?.createdBefore ?? "").trim();
    if (createdBefore) {
      params.created_before = createdBefore;
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

  async bulkTagResources(request: ResourceBulkTagRequest): Promise<ResourceBulkTagResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resources/tags/bulk"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        resource_ids: uniqueTrimmedStrings(request.resource_ids),
        tags: uniqueTrimmedStrings(request.tags),
        ...(request.metadata ? { metadata: request.metadata } : {}),
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceBulkTagResponse;
  }

  async deleteResources(
    request: ResourceBulkLifecycleRequest
  ): Promise<ResourceBulkLifecycleResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resources/delete/bulk"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        resource_ids: uniqueTrimmedStrings(request.resource_ids),
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceBulkLifecycleResponse;
  }

  async restoreResources(
    request: ResourceBulkLifecycleRequest
  ): Promise<ResourceBulkLifecycleResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resources/restore/bulk"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        resource_ids: uniqueTrimmedStrings(request.resource_ids),
      }),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceBulkLifecycleResponse;
  }

  async restoreResource(fileId: string): Promise<ResourceResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId.trim())}/restore`),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceResponse;
  }

  async patchResourceMetadata(
    fileId: string,
    metadata: Record<string, unknown>
  ): Promise<ResourceResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId.trim())}`),
      {
        method: "PATCH",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ metadata }),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceResponse;
  }

  async renameResource(fileId: string, originalName: string): Promise<ResourceResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId.trim())}`),
      {
        method: "PATCH",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ original_name: originalName.trim() }),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceResponse;
  }

  async listResourceShareGrants(
    fileId: string,
    options?: {
      limit?: number;
      status?: string;
    }
  ): Promise<ResourceShareGrantListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
    };
    const status = String(options?.status ?? "").trim();
    if (status) {
      params.status = status;
    }
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId)}/shares`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceShareGrantListResponse;
  }

  async createResourceShareGrant(
    fileId: string,
    request: ResourceShareGrantCreateRequest
  ): Promise<ResourceShareGrantResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId)}/shares`),
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
    return (await response.json()) as ResourceShareGrantResponse;
  }

  async createResourceShareGrants(
    request: ResourceShareGrantsCreateRequest
  ): Promise<ResourceShareGrantsCreateResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resources/shares/bulk"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceShareGrantsCreateResponse;
  }

  // Pickable grantees: same-org people + the org itself. The reliability core
  // of the sharing redesign — grantees are chosen, never typed as raw ids.
  async listShareTargets(query: string): Promise<ShareTargetListResponse> {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return this.fetchJson<ShareTargetListResponse>("/v2/share-targets", { method: "GET" });
    }
    return this.fetchJson<ShareTargetListResponse>(
      `/v2/share-targets?q=${encodeURIComponent(trimmedQuery)}`,
      { method: "GET" }
    );
  }

  async listResourceCollectionShareGrants(
    collectionId: string
  ): Promise<ResourceCollectionShareGrantListResponse> {
    return this.fetchJson<ResourceCollectionShareGrantListResponse>(
      `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}/shares`,
      { method: "GET" }
    );
  }

  async revokeResourceCollectionShareGrant(
    collectionId: string,
    grantId: string
  ): Promise<ResourceCollectionShareGrantRevokeResponse> {
    return this.fetchJson<ResourceCollectionShareGrantRevokeResponse>(
      `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}/shares/${encodeURIComponent(grantId.trim())}`,
      { method: "DELETE" }
    );
  }

  async createResourceCollectionShareGrants(
    collectionId: string,
    request: ResourceShareGrantCreateRequest
  ): Promise<ResourceCollectionShareGrantsCreateResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}/shares`
      ),
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
    return (await response.json()) as ResourceCollectionShareGrantsCreateResponse;
  }

  async revokeResourceShareGrant(
    fileId: string,
    grantId: string
  ): Promise<ResourceShareGrantResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/resources/${encodeURIComponent(fileId)}/shares/${encodeURIComponent(grantId)}`
      ),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceShareGrantResponse;
  }

  async createResourceCollection(
    request: ResourceCollectionCreateRequest
  ): Promise<ResourceCollectionResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resource-collections"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionResponse;
  }

  async listResourceCollections(options?: {
    limit?: number;
    offset?: number;
    query?: string;
    collectionType?: "collection" | "folder" | "dataset" | string;
    projectId?: string;
    status?: "active" | "deleted" | string;
  }): Promise<ResourceCollectionListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const query = String(options?.query ?? "").trim();
    if (query) {
      params.q = query;
    }
    const collectionType = String(options?.collectionType ?? "").trim();
    if (collectionType) {
      params.collection_type = collectionType;
    }
    const projectId = String(options?.projectId ?? "").trim();
    if (projectId) {
      params.project_id = projectId;
    }
    const status = String(options?.status ?? "").trim();
    if (status && status !== "active") {
      params.status = status;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/resource-collections", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionListResponse;
  }

  async deleteResourceCollection(collectionId: string): Promise<ResourceCollectionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}`),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionResponse;
  }

  async patchResourceCollection(
    collectionId: string,
    request: ResourceCollectionPatchRequest
  ): Promise<ResourceCollectionResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}`),
      {
        method: "PATCH",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ name: request.name.trim() }),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionResponse;
  }

  async restoreResourceCollection(collectionId: string): Promise<ResourceCollectionResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}/restore`
      ),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionResponse;
  }

  async addResourcesToCollection(
    collectionId: string,
    resourceIds: string[],
    metadata?: Record<string, unknown>
  ): Promise<ResourceCollectionAddResourcesResponse> {
    const body: { resource_ids: string[]; metadata?: Record<string, unknown> } = {
      resource_ids: resourceIds,
    };
    if (metadata) {
      body.metadata = metadata;
    }
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resource-collections/${encodeURIComponent(collectionId)}/resources`),
      {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionAddResourcesResponse;
  }

  async removeResourceFromCollection(
    collectionId: string,
    resourceId: string
  ): Promise<ResourceCollectionRemoveResourcesResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/resource-collections/${encodeURIComponent(collectionId.trim())}/resources/${encodeURIComponent(resourceId.trim())}`
      ),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceCollectionRemoveResourcesResponse;
  }

  async listResourceCollectionResources(
    collectionId: string,
    options?: {
      limit?: number;
      offset?: number;
      query?: string;
      kind?: "image" | "video" | "table" | "file";
      source?: "upload" | "bisque_import";
    projectId?: string;
    sharing?: string;
      processingStatus?: string;
      status?: string;
      tags?: string[];
      descriptors?: string[];
      metadataFilters?: ResourceMetadataFilter[];
      createdAfter?: string;
      createdBefore?: string;
    }
  ): Promise<ResourceListResponse> {
    const params: Record<string, string | string[]> = {
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
    const projectId = String(options?.projectId ?? "").trim();
    if (projectId) {
      params.project_id = projectId;
    }
    const sharing = String(options?.sharing ?? "").trim();
    if (sharing && sharing !== "all") {
      params.sharing = sharing;
    }
    const processingStatus = String(options?.processingStatus ?? "").trim();
    if (processingStatus && processingStatus !== "all") {
      params.processing_status = processingStatus;
    }
    const status = String(options?.status ?? "").trim();
    if (status && status !== "active") {
      params.status = status;
    }
    const tags = uniqueTrimmedStrings(options?.tags);
    if (tags.length > 0) {
      params.tags = tags.join(",");
    }
    const descriptors = uniqueTrimmedStrings(options?.descriptors);
    if (descriptors.length > 0) {
      params.descriptors = descriptors.join(",");
    }
    const metadataFilterSpecs = resourceMetadataFilterSpecs(options?.metadataFilters);
    if (metadataFilterSpecs.length > 0) {
      params.metadata_filter = metadataFilterSpecs;
    }
    const createdAfter = String(options?.createdAfter ?? "").trim();
    if (createdAfter) {
      params.created_after = createdAfter;
    }
    const createdBefore = String(options?.createdBefore ?? "").trim();
    if (createdBefore) {
      params.created_before = createdBefore;
    }
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resource-collections/${encodeURIComponent(collectionId)}/resources`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as ResourceListResponse;
  }

  async createDatasetSnapshot(request: DatasetSnapshotCreateRequest): Promise<DatasetSnapshotResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/dataset-snapshots"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotResponse;
  }

  async listDatasetSnapshots(options?: {
    limit?: number;
    offset?: number;
    query?: string;
    projectId?: string;
    sourceCollectionId?: string;
    status?: string;
  }): Promise<DatasetSnapshotListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const query = String(options?.query ?? "").trim();
    if (query) {
      params.q = query;
    }
    const projectId = String(options?.projectId ?? "").trim();
    if (projectId) {
      params.project_id = projectId;
    }
    const sourceCollectionId = String(options?.sourceCollectionId ?? "").trim();
    if (sourceCollectionId) {
      params.source_collection_id = sourceCollectionId;
    }
    const status = String(options?.status ?? "").trim();
    if (status && status !== "active") {
      params.status = status;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/dataset-snapshots", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotListResponse;
  }

  async deleteDatasetSnapshot(snapshotId: string): Promise<DatasetSnapshotResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}`),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotResponse;
  }

  async restoreDatasetSnapshot(snapshotId: string): Promise<DatasetSnapshotResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}/restore`),
      {
        method: "POST",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotResponse;
  }

  async getDatasetSnapshot(snapshotId: string): Promise<DatasetSnapshotResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotResponse;
  }

  async listDatasetSnapshotEvents(
    snapshotId: string,
    options?: {
      limit?: number;
      offset?: number;
      eventType?: string;
      actorUserId?: string;
    }
  ): Promise<DatasetSnapshotEventListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const eventType = String(options?.eventType ?? "").trim();
    if (eventType) {
      params.event_type = eventType;
    }
    const actorUserId = String(options?.actorUserId ?? "").trim();
    if (actorUserId) {
      params.actor_user_id = actorUserId;
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}/events`,
        params
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
    return (await response.json()) as DatasetSnapshotEventListResponse;
  }

  async listDatasetSnapshotShareGrants(
    snapshotId: string,
    options?: { limit?: number; status?: string }
  ): Promise<DatasetSnapshotShareGrantListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
    };
    const status = String(options?.status ?? "").trim();
    if (status) {
      params.status = status;
    }
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}/shares`,
        params
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
    return (await response.json()) as DatasetSnapshotShareGrantListResponse;
  }

  async createDatasetSnapshotShareGrant(
    snapshotId: string,
    request: DatasetSnapshotShareGrantCreateRequest
  ): Promise<DatasetSnapshotShareGrantResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}/shares`),
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
    return (await response.json()) as DatasetSnapshotShareGrantResponse;
  }

  async revokeDatasetSnapshotShareGrant(
    snapshotId: string,
    grantId: string
  ): Promise<DatasetSnapshotShareGrantResponse> {
    const response = await fetch(
      buildUrl(
        this.baseUrl,
        `/v2/dataset-snapshots/${encodeURIComponent(snapshotId)}/shares/${encodeURIComponent(grantId)}`
      ),
      {
        method: "DELETE",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DatasetSnapshotShareGrantResponse;
  }

  async createDataAgentJob(request: DataAgentJobCreateRequest): Promise<DataAgentJobResponse> {
    const response = await fetch(buildUrl(this.baseUrl, "/v2/data-agent/jobs"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DataAgentJobResponse;
  }

  async listDataAgentJobs(options?: {
    limit?: number;
    offset?: number;
    status?: string;
    jobType?: string;
    projectId?: string;
  }): Promise<DataAgentJobListResponse> {
    const params: Record<string, string> = {
      limit: String(Math.max(1, Math.min(1000, Number(options?.limit) || 200))),
      offset: String(Math.max(0, Number(options?.offset) || 0)),
    };
    const status = String(options?.status ?? "").trim();
    if (status) {
      params.status = status;
    }
    const jobType = String(options?.jobType ?? "").trim();
    if (jobType) {
      params.job_type = jobType;
    }
    const projectId = String(options?.projectId ?? "").trim();
    if (projectId) {
      params.project_id = projectId;
    }
    const response = await fetch(buildUrl(this.baseUrl, "/v2/data-agent/jobs", params), {
      method: "GET",
      headers: this.headers(),
      credentials: "include",
    });
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DataAgentJobListResponse;
  }

  async getDataAgentJob(jobId: string): Promise<DataAgentJobResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/data-agent/jobs/${encodeURIComponent(jobId)}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as DataAgentJobResponse;
  }

  async controlDataAgentJob(
    jobId: string,
    request: DataAgentJobControlRequest
  ): Promise<DataAgentJobResponse> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/data-agent/jobs/${encodeURIComponent(jobId)}/control`),
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
    return (await response.json()) as DataAgentJobResponse;
  }

  // Fetch a single resource by id (GET /v2/resources/{file_id}). Used to restore a
  // Lens deep-link / shared URL when the resource isn't already in the loaded list.
  async getResource(fileId: string): Promise<ResourceRecord> {
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/resources/${encodeURIComponent(fileId.trim())}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    const payload = (await response.json()) as ResourceResponse;
    return payload.resource;
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

  resourceThumbnailUrl(
    resourceOrFileId: string | Pick<ResourceRecord, "file_id" | "has_thumbnail" | "thumbnail_url">
  ): string {
    const fileId = typeof resourceOrFileId === "string" ? resourceOrFileId : resourceOrFileId.file_id;
    const safeFileId = encodeURIComponent(fileId);
    const canonical = buildUrl(this.baseUrl, `/v2/resources/${safeFileId}/thumbnail`);
    if (typeof resourceOrFileId === "string" || resourceOrFileId.has_thumbnail !== true) {
      return canonical;
    }
    const advertised = String(resourceOrFileId.thumbnail_url ?? "").trim();
    if (!advertised) {
      return canonical;
    }
    try {
      const base = new URL(this.baseUrl.endsWith("/") ? this.baseUrl : `${this.baseUrl}/`);
      const candidate = new URL(advertised, base);
      if (
        (candidate.protocol !== "http:" && candidate.protocol !== "https:") ||
        candidate.origin !== base.origin ||
        candidate.username !== "" ||
        candidate.password !== ""
      ) {
        return canonical;
      }
      return candidate.toString();
    } catch {
      return canonical;
    }
  }

  resourceDownloadUrl(fileId: string): string {
    const safeFileId = encodeURIComponent(fileId);
    return buildUrl(this.baseUrl, `/v2/resources/${safeFileId}/download`);
  }

  resourceCollectionDownloadUrl(collectionId: string): string {
    const safeCollectionId = encodeURIComponent(collectionId);
    return buildUrl(this.baseUrl, `/v2/resource-collections/${safeCollectionId}/download`);
  }

  // resourceTextHead fetches a bounded, UTF-8-safe window of a text/data resource
  // plus metadata (total size, truncation, encoding, line estimate). O(window).
  async resourceTextHead(
    fileId: string,
    options: { maxBytes?: number; offset?: number } = {}
  ): Promise<ResourceTextHead> {
    const params: Record<string, string> = {};
    if (typeof options.maxBytes === "number") {
      params.max_bytes = String(Math.max(1, Math.floor(options.maxBytes)));
    }
    if (typeof options.offset === "number" && options.offset > 0) {
      params.offset = String(Math.floor(options.offset));
    }
    return this.fetchJson<ResourceTextHead>(
      `/v2/resources/${encodeURIComponent(fileId)}/text-head`,
      {},
      params
    );
  }

  // resourceCsvRows fetches one quote-aware page of CSV/TSV rows using byte-offset
  // (cursor) pagination. offset 0 also returns the header as columns. O(page_size).
  async resourceCsvRows(
    fileId: string,
    options: { offsetBytes?: number; limit?: number; delimiter?: string } = {}
  ): Promise<ResourceCsvRows> {
    const params: Record<string, string> = {};
    if (typeof options.offsetBytes === "number" && options.offsetBytes > 0) {
      params.offset_bytes = String(Math.floor(options.offsetBytes));
    }
    if (typeof options.limit === "number") {
      params.limit = String(Math.max(1, Math.floor(options.limit)));
    }
    if (options.delimiter) {
      params.delimiter = options.delimiter;
    }
    return this.fetchJson<ResourceCsvRows>(
      `/v2/resources/${encodeURIComponent(fileId)}/csv/rows`,
      {},
      params
    );
  }

  // fetchResourceRangeBytes reads a bounded byte window of a resource via an HTTP
  // Range request against /download (ServeContent supports 206). Used for gzip
  // (.gz) heads that the server does not transcode — the caller decompresses.
  async fetchResourceRangeBytes(
    fileId: string,
    options: { offset?: number; maxBytes?: number } = {}
  ): Promise<{ bytes: Uint8Array; totalBytes: number | null; truncated: boolean }> {
    const offset = Math.max(0, Math.floor(options.offset ?? 0));
    const maxBytes = Math.max(1, Math.floor(options.maxBytes ?? 1 << 20));
    const end = offset + maxBytes - 1;
    const response = await fetch(this.resourceDownloadUrl(fileId), {
      method: "GET",
      headers: this.headers({ Range: `bytes=${offset}-${end}` }),
      credentials: "include",
    });
    if (!response.ok && response.status !== 206) {
      return parseError(response);
    }
    const buffer = new Uint8Array(await response.arrayBuffer());
    let totalBytes: number | null = null;
    let truncated = false;
    const contentRange = response.headers.get("Content-Range");
    if (contentRange) {
      const match = /\/(\d+)\s*$/.exec(contentRange);
      if (match) {
        totalBytes = Number(match[1]);
      }
      truncated = totalBytes !== null && offset + buffer.byteLength < totalBytes;
    } else {
      const length = response.headers.get("Content-Length");
      if (length) {
        totalBytes = Number(length);
      }
    }
    return { bytes: buffer, totalBytes, truncated };
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
      /** Source-channel-indexed color palette. */
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
    applyImageChannelSelection(params, config?.channels, config?.channelColors);
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
      /** Source-channel-indexed color palette. */
      channelColors?: string[];
      fullResolution?: boolean;
      cacheKey?: string;
      scalarRenderMode?: "intensity" | "mask";
      scalarThresholdValue?: number | null;
      scalarThresholdForeground?: "above";
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
    applyImageChannelSelection(params, indices?.channels, indices?.channelColors);
    if (typeof indices?.fullResolution === "boolean") {
      params.full_resolution = indices.fullResolution ? "true" : "false";
    }
    if (indices?.scalarRenderMode) {
      params.scalar_render_mode = indices.scalarRenderMode;
    }
    if (
      typeof indices?.scalarThresholdValue === "number" &&
      Number.isFinite(indices.scalarThresholdValue)
    ) {
      params.scalar_threshold_value = String(indices.scalarThresholdValue);
    }
    if (indices?.scalarThresholdForeground) {
      params.scalar_threshold_foreground = indices.scalarThresholdForeground;
    }
    const cacheKey = String(indices?.cacheKey ?? "").trim();
    if (cacheKey) {
      params.cache_key = cacheKey;
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/slice`, params);
  }

  async getUploadScalarVolume(
    fileId: string,
    config?: {
      t?: number | null;
      channel?: number | null;
      sampling?: "box" | "nearest";
      signal?: AbortSignal;
    }
  ): Promise<ScalarVolumePayload> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (config?.t != null) {
      params.t = String(requireNonNegativeSafeInteger(config.t, "time index"));
    }
    if (config?.channel != null) {
      params.channel = String(requireNonNegativeSafeInteger(config.channel, "channel index"));
    }
    if (config?.sampling) {
      params.sampling = config.sampling;
    }
    const payload = await this.fetchScalarVolumeWithTimeout(
      buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/scalar-volume`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
        // Volume payloads are tens of MB; skip the HTTP disk cache so we neither
        // bloat it nor fail the fetch on a cache-write error for large responses.
        cache: "no-store",
        signal: config?.signal,
      },
      120000, // generous: a large volume read is legitimately slow; only a true hang trips it
      "Volume request",
    );
    const expectedSampling = config?.sampling ?? "box";
    if (payload.sampling !== expectedSampling) {
      throw invalidScalarVolumeResponse(
        `sampling mismatch: requested ${expectedSampling}, received ${payload.sampling}`
      );
    }
    return payload;
  }

  uploadTileUrl(
    fileId: string,
    config: {
      axis: "z" | "y" | "x";
      level: number;
      tileX: number;
      tileY: number;
      size?: number | null;
      z?: number | null;
      c?: number | null;
      t?: number | null;
      channels?: number[];
      /** Source-channel-indexed color palette. */
      channelColors?: string[];
      cacheKey?: string;
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const safeAxis = encodeURIComponent(config.axis);
    const safeLevel = Math.max(0, Math.floor(config.level));
    const safeTileX = Math.max(0, Math.floor(config.tileX));
    const safeTileY = Math.max(0, Math.floor(config.tileY));
    const params: Record<string, string> = {};
    // The engine retiles the served file (which may be a derived pyramid with a
    // different native tile size) at this requested size, so it MUST match the grid
    // the canvas computes from tile_scheme.tile_size — otherwise tile (col,row) maps
    // to the wrong region/scale and out-of-grid tiles 500.
    if (typeof config.size === "number" && Number.isFinite(config.size) && config.size > 0) {
      params.size = String(Math.floor(config.size));
    }
    if (typeof config.z === "number" && Number.isFinite(config.z)) {
      params.z = String(Math.max(0, Math.floor(config.z)));
    }
    if (typeof config.c === "number" && Number.isFinite(config.c)) {
      params.c = String(Math.max(0, Math.floor(config.c)));
    }
    if (typeof config.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    applyImageChannelSelection(params, config.channels, config.channelColors);
    const cacheKey = String(config.cacheKey ?? "").trim();
    if (cacheKey) {
      params.cache_key = cacheKey;
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
      /** Source-channel-indexed color palette. */
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
    applyImageChannelSelection(params, config?.channels, config?.channelColors);
    if (typeof config?.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/atlas`, params);
  }

  async getUploadHistogram(
    fileId: string,
    config?: {
      channel?: number | null;
      channels?: number[];
      t?: number | null;
      bins?: number | null;
      scope?: "volume";
      signal?: AbortSignal;
    }
  ): Promise<UploadViewerHistogramResponse> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {};
    if (config?.scope === "volume") {
      if (
        config.channel != null &&
        Array.isArray(config.channels) &&
        config.channels.length > 0
      ) {
        throw new RangeError(
          "volume histogram requires one unambiguous channel selector"
        );
      }
      const volumeChannel =
        config.channel ??
        (Array.isArray(config.channels) && config.channels.length === 1
          ? config.channels[0]
          : null);
      if (volumeChannel == null) {
        throw new RangeError("volume histogram requires exactly one channel");
      }
      params.channel = String(
        requireNonNegativeSafeInteger(volumeChannel, "histogram channel")
      );
    } else if (config?.channel != null) {
      params.channel = String(requireNonNegativeSafeInteger(config.channel, "histogram channel"));
    } else if (Array.isArray(config?.channels) && config.channels.length > 0) {
      const channels = config.channels.map((channel) =>
        requireNonNegativeSafeInteger(channel, "histogram channel")
      );
      if (new Set(channels).size !== channels.length) {
        throw new RangeError("histogram channels must not contain duplicates");
      }
      params.channels = channels.join(",");
    }
    if (typeof config?.t === "number" && Number.isFinite(config.t)) {
      params.t = String(Math.max(0, Math.floor(config.t)));
    }
    if (typeof config?.bins === "number" && Number.isFinite(config.bins)) {
      params.bins = String(Math.max(8, Math.floor(config.bins)));
    }
    if (config?.scope) {
      params.scope = config.scope;
    }
    const response = await this.fetchWithTimeout(
      buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/histogram`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
        signal: config?.signal,
      },
      30000,
      "Histogram request",
    );
    if (!response.ok) {
      return parseError(response);
    }
    return (await response.json()) as UploadViewerHistogramResponse;
  }

  // Wraps fetch with an abort-based timeout so a hung image-service request surfaces a
  // 504 the viewer can render (the viewer already has error states for these calls)
  // instead of spinning forever. Mirrors the inline pattern getUploadViewer uses below.
  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    timeoutMs: number,
    label: string,
  ): Promise<Response> {
    const controller = new AbortController();
    const callerSignal = init.signal;
    let timedOut = false;
    const abortFromCaller = () => controller.abort(callerSignal?.reason);
    if (callerSignal?.aborted) {
      abortFromCaller();
    } else {
      callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
    }
    const timeoutId = window.setTimeout(() => {
      if (!controller.signal.aborted) {
        timedOut = true;
        controller.abort();
      }
    }, timeoutMs);
    try {
      return await fetch(url, { ...init, signal: controller.signal });
    } catch (error) {
      if (timedOut && error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError(`${label} timed out`, 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
      callerSignal?.removeEventListener("abort", abortFromCaller);
    }
  }

  private async fetchScalarVolumeWithTimeout(
    url: string,
    init: RequestInit,
    timeoutMs: number,
    label: string
  ): Promise<ScalarVolumePayload> {
    const controller = new AbortController();
    const callerSignal = init.signal;
    let timedOut = false;
    const abortFromCaller = () => controller.abort(callerSignal?.reason);
    if (callerSignal?.aborted) {
      abortFromCaller();
    } else {
      callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
    }
    const timeoutId = window.setTimeout(() => {
      if (!controller.signal.aborted) {
        timedOut = true;
        controller.abort();
      }
    }, timeoutMs);
    try {
      const response = await fetch(url, { ...init, signal: controller.signal });
      if (!response.ok) {
        return parseError(response);
      }
      return await parseScalarVolumeResponse(response, controller.signal);
    } catch (error) {
      if (timedOut && error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError(`${label} timed out`, 504, null);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
      callerSignal?.removeEventListener("abort", abortFromCaller);
    }
  }

  async getUploadViewer(fileId: string): Promise<UploadViewerInfo> {
    const safeFileId = encodeURIComponent(fileId);
    const response = await this.fetchWithTimeout(
      buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/viewer`),
      { method: "GET", headers: this.headers(), credentials: "include" },
      // Cold viewer metadata for a large microscopy source (e.g. a 600MB OME-TIFF)
      // is a one-time libbioimage decode over remote NFS that can take ~10-30s; the
      // control-plane image-service client allows 60s, so match it rather than abort
      // at 15s. (Repeat views are served warm in well under a second.)
      60000,
      "Viewer metadata request",
    );
    if (!response.ok) {
      return parseError(response);
    }
    const { normalizeUploadViewerInfo } = await loadViewerManifestModule();
    return normalizeUploadViewerInfo(await response.json());
  }

  // CIFTI grayordinate views: a bounded, downsampled carpet (grayordinate × time)
  // and an N×N connectivity matrix. Both return small payloads regardless of the
  // (often ~400MB) source, so the default JSON path with a generous timeout is fine.
  async getCiftiCarpet(fileId: string, options: { maxRows?: number; maxCols?: number } = {}): Promise<CiftiCarpetResponse> {
    const params: Record<string, string> = {};
    if (options.maxRows) params.max_rows = String(Math.floor(options.maxRows));
    if (options.maxCols) params.max_cols = String(Math.floor(options.maxCols));
    return this.fetchJson<CiftiCarpetResponse>(
      `/v2/uploads/${encodeURIComponent(fileId)}/cifti/carpet`,
      {},
      params
    );
  }

  async getCiftiConnectivity(fileId: string, options: { nodes?: number } = {}): Promise<CiftiConnectivityResponse> {
    const params: Record<string, string> = {};
    if (options.nodes) params.nodes = String(Math.floor(options.nodes));
    return this.fetchJson<CiftiConnectivityResponse>(
      `/v2/uploads/${encodeURIComponent(fileId)}/cifti/connectivity`,
      {},
      params
    );
  }

  // Lens scene3d: the derive job's manifest. Small JSON regardless of the (often
  // multi-GB) source, because the control plane never parses a scene file in the
  // request path — it serves what the worker already derived. While the job is
  // still running the same endpoint answers with `{status:"deriving"}`, so callers
  // poll this rather than treating a non-ready answer as an error.
  async getScene3dManifest(fileId: string): Promise<Scene3dManifestResponse> {
    return this.fetchJson<Scene3dManifestResponse>(
      `/v2/uploads/${encodeURIComponent(fileId)}/scene3d/manifest`
    );
  }

  // A single derived chunk (USX1 splats or UPC1 points), returned as raw bytes so
  // the caller can build zero-copy typed-array views over the planar payload. The
  // response is immutable + ETagged, and the caller passes a signal so in-flight
  // tier fetches abort when the viewer unmounts mid-stream.
  async fetchScene3dChunk(
    fileId: string,
    index: number,
    options: { signal?: AbortSignal } = {}
  ): Promise<ArrayBuffer> {
    const safeFileId = encodeURIComponent(fileId);
    const safeIndex = Math.max(0, Math.floor(Number(index) || 0));
    const response = await fetch(
      buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/scene3d/chunk/${safeIndex}`),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
        signal: options.signal,
      }
    );
    if (!response.ok) {
      return parseError(response);
    }
    return response.arrayBuffer();
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
      response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/dataset`, params), {
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
      response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/materials/dashboard`), {
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
      featureIds?: readonly string[];
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
    if (config.featureIds?.length) {
      params.feature_ids = canonicalizeHdf5FeatureIds(config.featureIds).join(",");
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/preview/slice`, params);
  }

  hdf5AtlasPreviewUrl(
    fileId: string,
    config: {
      datasetPath: string;
      component?: number | null;
      enhancement?: string;
      fusionMethod?: string;
      negative?: boolean;
      channels?: number[];
      featureIds?: readonly string[];
    }
  ): string {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: config.datasetPath,
    };
    if (typeof config.component === "number" && Number.isFinite(config.component)) {
      params.component = String(Math.max(0, Math.floor(config.component)));
    }
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
    if (config.featureIds?.length) {
      params.feature_ids = canonicalizeHdf5FeatureIds(config.featureIds).join(",");
    }
    return buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/preview/atlas`, params);
  }

  async getHdf5ScalarVolume(
    fileId: string,
    config: { datasetPath: string; channel?: number | null; signal?: AbortSignal }
  ): Promise<ScalarVolumePayload> {
    const safeFileId = encodeURIComponent(fileId);
    const params: Record<string, string> = {
      dataset_path: config.datasetPath,
    };
    if (config.channel != null) {
      params.channel = String(requireNonNegativeSafeInteger(config.channel, "channel index"));
    }
    return this.fetchScalarVolumeWithTimeout(
      buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/preview/scalar-volume`, params),
      {
        method: "GET",
        headers: this.headers(),
        credentials: "include",
        cache: "no-store",
        signal: config.signal,
      },
      120000,
      "HDF5 volume request",
    );
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
        buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/preview/histogram`, params),
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
      response = await fetch(buildUrl(this.baseUrl, `/v2/uploads/${safeFileId}/hdf5/preview/table`, params), {
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

  async getRunEvents(
    runId: string,
    limit = 200,
    options?: { afterSequence?: number }
  ): Promise<RunEventsResponse> {
    const requestedLimit = Math.max(1, Math.floor(asFiniteNumber(limit, 200)));
    const events: RunEvent[] = [];
    let resolvedRunId = runId;
    // Callers polling a live run pass the last sequence they already hold so
    // each poll only transfers new events instead of re-paging from zero.
    let afterSequence = Math.max(0, Math.floor(asFiniteNumber(options?.afterSequence, 0)));
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

  // getRunArtifactCaption fetches a lazily-generated, server-cached academic caption
  // for a run-output figure. Always resolves (caption is "" when captioning is
  // disabled/unavailable), so callers never need to special-case failure.
  async getRunArtifactCaption(runId: string, path: string): Promise<{ caption: string; enabled: boolean }> {
    return this.fetchJson<{ caption: string; enabled: boolean }>(
      `/v2/runs/${encodeURIComponent(runId)}/artifacts/caption`,
      {},
      { path }
    );
  }

}
