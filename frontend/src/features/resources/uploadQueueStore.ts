import type { UploadProgressEvent } from "@/lib/api";
import type { UploadSessionResponse } from "@/types";

export const RESOURCE_UPLOAD_QUEUE_STORAGE_KEY = "bisque.frontend.resourceUploadQueue.v1";

const RESOURCE_UPLOAD_QUEUE_DB_NAME = "bisque-ultra-resource-upload-queue";
const RESOURCE_UPLOAD_QUEUE_DB_VERSION = 1;
const RESOURCE_UPLOAD_QUEUE_OBJECT_STORE = "uploads";
const DEFAULT_UPLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024;
const UPLOAD_RESELECT_MESSAGE =
  "Upload interrupted. Select the same file or folder to resume from verified chunks.";
const UPLOAD_SESSION_EXPIRED_MESSAGE = "Upload session expired. Start this upload again.";

export type ResourceUploadQueueStatus =
  | "queued"
  | "creating"
  | "uploading"
  | "verifying"
  | "paused"
  | "needs_file"
  | "completed"
  | "failed"
  | "canceled";

export type ResourceUploadQueueRecord = {
  id: string;
  fingerprint: string;
  fileName: string;
  relativePath: string | null;
  fileToken: string;
  sessionId: string | null;
  contentType: string;
  totalBytes: number;
  bytesVerified: number;
  bytesCommitted: number;
  chunkSizeBytes: number;
  status: ResourceUploadQueueStatus;
  error: string | null;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
};

export type ResourceUploadProgressSummary = {
  id: string;
  fingerprint?: string | null;
  sessionId?: string | null;
  fileToken?: string | null;
  fileName: string;
  relativePath?: string | null;
  status: ResourceUploadQueueStatus;
  totalBytes: number;
  bytesVerified: number;
  error?: string | null;
};

export type ResourceUploadSessionLoader = (sessionId: string) => Promise<UploadSessionResponse>;

export type ResourceUploadQueueUpsertInput = {
  id: string;
  fingerprint: string;
  fileName: string;
  relativePath?: string | null;
  fileToken: string;
  sessionId?: string | null;
  contentType?: string | null;
  totalBytes: number;
  bytesVerified?: number;
  bytesCommitted?: number;
  chunkSizeBytes?: number;
  status?: ResourceUploadQueueStatus;
  error?: string | null;
  createdAt?: string;
  updatedAt?: string;
  completedAt?: string | null;
};

export type ResourceUploadQueueProgressPatch = {
  sessionId?: string | null;
  bytesVerified?: number;
  bytesCommitted?: number;
  status?: ResourceUploadQueueStatus;
  error?: string | null;
  updatedAt?: string;
  completedAt?: string | null;
};

export type ResourceUploadQueueStore = {
  list: () => Promise<ResourceUploadQueueRecord[]>;
  get: (id: string) => Promise<ResourceUploadQueueRecord | null>;
  upsert: (input: ResourceUploadQueueUpsertInput) => Promise<ResourceUploadQueueRecord>;
  updateProgress: (
    id: string,
    patch: ResourceUploadQueueProgressPatch
  ) => Promise<ResourceUploadQueueRecord | null>;
  remove: (id: string) => Promise<void>;
  clear: () => Promise<void>;
};

type ResourceUploadQueueDriver = {
  list: () => Promise<unknown[]>;
  get: (id: string) => Promise<unknown | null>;
  put: (record: ResourceUploadQueueRecord) => Promise<void>;
  delete: (id: string) => Promise<void>;
  clear: () => Promise<void>;
};

export type CreateResourceUploadQueueStoreOptions = {
  now?: () => string;
  localStorage?: Storage | null;
  indexedDB?: IDBFactory | null;
  preferIndexedDB?: boolean;
  driver?: ResourceUploadQueueDriver;
};

export type HydrateResourceUploadProgressOptions = {
  limit?: number;
  loadUploadSession?: ResourceUploadSessionLoader;
};

const uploadQueueStatuses = new Set<ResourceUploadQueueStatus>([
  "queued",
  "creating",
  "uploading",
  "verifying",
  "paused",
  "needs_file",
  "completed",
  "failed",
  "canceled",
]);

const asObject = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const asTrimmedString = (value: unknown): string => String(value ?? "").trim();

const asOptionalString = (value: unknown): string | null => {
  const text = asTrimmedString(value);
  return text.length > 0 ? text : null;
};

const asBytes = (value: unknown, fallback = 0, max = Number.POSITIVE_INFINITY): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.min(max, Math.max(0, Math.floor(numeric)));
};

const asStatus = (value: unknown): ResourceUploadQueueStatus => {
  const normalized = asTrimmedString(value) as ResourceUploadQueueStatus;
  return uploadQueueStatuses.has(normalized) ? normalized : "queued";
};

const asDateString = (value: unknown): string | null => {
  const text = asTrimmedString(value);
  if (!text) {
    return null;
  }
  const millis = Date.parse(text);
  return Number.isFinite(millis) ? new Date(millis).toISOString() : null;
};

const sortByRecentUpdate = (left: ResourceUploadQueueRecord, right: ResourceUploadQueueRecord): number =>
  Date.parse(right.updatedAt) - Date.parse(left.updatedAt) || left.fileName.localeCompare(right.fileName);

const normalizeResourceUploadQueueRecord = (value: unknown): ResourceUploadQueueRecord | null => {
  const record = asObject(value);
  if (!record) {
    return null;
  }
  const id = asOptionalString(record.id);
  const fingerprint = asOptionalString(record.fingerprint);
  const fileName = asOptionalString(record.fileName);
  const fileToken = asOptionalString(record.fileToken);
  if (!id || !fingerprint || !fileName || !fileToken) {
    return null;
  }
  const totalBytes = asBytes(record.totalBytes);
  const bytesVerified = asBytes(record.bytesVerified, 0, totalBytes);
  const bytesCommitted = asBytes(record.bytesCommitted, 0, totalBytes);
  const chunkSizeBytes =
    asBytes(record.chunkSizeBytes) > 0 ? asBytes(record.chunkSizeBytes) : DEFAULT_UPLOAD_CHUNK_SIZE_BYTES;
  const createdAt = asDateString(record.createdAt);
  const updatedAt = asDateString(record.updatedAt);
  if (!createdAt || !updatedAt) {
    return null;
  }
  return {
    id,
    fingerprint,
    fileName,
    relativePath: asOptionalString(record.relativePath),
    fileToken,
    sessionId: asOptionalString(record.sessionId),
    contentType: asOptionalString(record.contentType) ?? "application/octet-stream",
    totalBytes,
    bytesVerified,
    bytesCommitted,
    chunkSizeBytes,
    status: asStatus(record.status),
    error: asOptionalString(record.error),
    createdAt,
    updatedAt,
    completedAt: asDateString(record.completedAt),
  };
};

const requireString = (value: string, field: string): string => {
  const text = asOptionalString(value);
  if (!text) {
    throw new Error(`Resource upload queue record is missing ${field}`);
  }
  return text;
};

const buildResourceUploadQueueRecord = (
  input: ResourceUploadQueueUpsertInput,
  existing: ResourceUploadQueueRecord | null,
  timestamp: string
): ResourceUploadQueueRecord => {
  const totalBytes = asBytes(input.totalBytes, existing?.totalBytes ?? 0);
  const status = input.status ?? existing?.status ?? "queued";
  const bytesVerified = asBytes(
    Math.max(existing?.bytesVerified ?? 0, input.bytesVerified ?? 0),
    0,
    totalBytes
  );
  const bytesCommitted = asBytes(
    Math.max(existing?.bytesCommitted ?? 0, input.bytesCommitted ?? 0),
    0,
    totalBytes
  );
  return {
    id: requireString(input.id, "id"),
    fingerprint: requireString(input.fingerprint, "fingerprint"),
    fileName: requireString(input.fileName, "fileName"),
    relativePath: asOptionalString(input.relativePath) ?? existing?.relativePath ?? null,
    fileToken: requireString(input.fileToken, "fileToken"),
    sessionId: asOptionalString(input.sessionId) ?? existing?.sessionId ?? null,
    contentType: asOptionalString(input.contentType) ?? existing?.contentType ?? "application/octet-stream",
    totalBytes,
    bytesVerified,
    bytesCommitted,
    chunkSizeBytes:
      asBytes(input.chunkSizeBytes) > 0
        ? asBytes(input.chunkSizeBytes)
        : existing?.chunkSizeBytes ?? DEFAULT_UPLOAD_CHUNK_SIZE_BYTES,
    status,
    error: input.error === undefined ? existing?.error ?? null : asOptionalString(input.error),
    createdAt: asDateString(input.createdAt) ?? existing?.createdAt ?? timestamp,
    updatedAt: asDateString(input.updatedAt) ?? timestamp,
    completedAt:
      status === "completed"
        ? asDateString(input.completedAt) ?? existing?.completedAt ?? timestamp
        : asDateString(input.completedAt) ?? existing?.completedAt ?? null,
  };
};

const createLocalStorageDriver = (storage: Storage | null): ResourceUploadQueueDriver => {
  if (!storage) {
    return createMemoryDriver();
  }
  const readRawRecords = (): unknown[] => {
    try {
      const raw = storage.getItem(RESOURCE_UPLOAD_QUEUE_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };
  const writeRawRecords = (records: unknown[]): void => {
    try {
      storage.setItem(RESOURCE_UPLOAD_QUEUE_STORAGE_KEY, JSON.stringify(records));
    } catch {
      // Upload sessions remain authoritative; local queue persistence is best effort.
    }
  };
  return {
    list: async () => readRawRecords(),
    get: async (id: string) => readRawRecords().find((record) => asObject(record)?.id === id) ?? null,
    put: async (record: ResourceUploadQueueRecord) => {
      const next = readRawRecords().filter((item) => asObject(item)?.id !== record.id);
      next.unshift(record);
      writeRawRecords(next);
    },
    delete: async (id: string) => {
      writeRawRecords(readRawRecords().filter((item) => asObject(item)?.id !== id));
    },
    clear: async () => {
      writeRawRecords([]);
    },
  };
};

const createMemoryDriver = (): ResourceUploadQueueDriver => {
  let records: ResourceUploadQueueRecord[] = [];
  return {
    list: async () => records,
    get: async (id: string) => records.find((record) => record.id === id) ?? null,
    put: async (record: ResourceUploadQueueRecord) => {
      records = [record, ...records.filter((item) => item.id !== record.id)];
    },
    delete: async (id: string) => {
      records = records.filter((record) => record.id !== id);
    },
    clear: async () => {
      records = [];
    },
  };
};

const idbRequest = <T>(request: IDBRequest<T>): Promise<T> =>
  new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB upload queue request failed"));
  });

const createIndexedDbDriver = (factory: IDBFactory): ResourceUploadQueueDriver => {
  let dbPromise: Promise<IDBDatabase> | null = null;
  const openDatabase = (): Promise<IDBDatabase> => {
    dbPromise ??= new Promise((resolve, reject) => {
      const request = factory.open(RESOURCE_UPLOAD_QUEUE_DB_NAME, RESOURCE_UPLOAD_QUEUE_DB_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(RESOURCE_UPLOAD_QUEUE_OBJECT_STORE)) {
          const store = database.createObjectStore(RESOURCE_UPLOAD_QUEUE_OBJECT_STORE, { keyPath: "id" });
          store.createIndex("updatedAt", "updatedAt");
          store.createIndex("status", "status");
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB upload queue open failed"));
    });
    return dbPromise;
  };
  const objectStore = async (mode: IDBTransactionMode): Promise<IDBObjectStore> => {
    const database = await openDatabase();
    return database.transaction(RESOURCE_UPLOAD_QUEUE_OBJECT_STORE, mode).objectStore(RESOURCE_UPLOAD_QUEUE_OBJECT_STORE);
  };
  return {
    list: async () => idbRequest((await objectStore("readonly")).getAll()),
    get: async (id: string) => idbRequest((await objectStore("readonly")).get(id)),
    put: async (record: ResourceUploadQueueRecord) => {
      await idbRequest((await objectStore("readwrite")).put(record));
    },
    delete: async (id: string) => {
      await idbRequest((await objectStore("readwrite")).delete(id));
    },
    clear: async () => {
      await idbRequest((await objectStore("readwrite")).clear());
    },
  };
};

const resolveBrowserLocalStorage = (
  override: Storage | null | undefined
): Storage | null => {
  if (override !== undefined) {
    return override;
  }
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      return window.localStorage;
    }
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

const resolveBrowserIndexedDB = (
  override: IDBFactory | null | undefined
): IDBFactory | null => {
  if (override !== undefined) {
    return override;
  }
  try {
    if (typeof window !== "undefined" && window.indexedDB) {
      return window.indexedDB;
    }
    return globalThis.indexedDB ?? null;
  } catch {
    return null;
  }
};

const createDefaultDriver = (options: CreateResourceUploadQueueStoreOptions): ResourceUploadQueueDriver => {
  const indexedDBFactory = resolveBrowserIndexedDB(options.indexedDB);
  if (options.preferIndexedDB !== false && indexedDBFactory) {
    return createIndexedDbDriver(indexedDBFactory);
  }
  return createLocalStorageDriver(resolveBrowserLocalStorage(options.localStorage));
};

export const createResourceUploadQueueStore = (
  options: CreateResourceUploadQueueStoreOptions = {}
): ResourceUploadQueueStore => {
  const driver = options.driver ?? createDefaultDriver(options);
  const now = options.now ?? (() => new Date().toISOString());
  const listNormalizedRecords = async (): Promise<ResourceUploadQueueRecord[]> =>
    (await driver.list())
      .map(normalizeResourceUploadQueueRecord)
      .filter((record): record is ResourceUploadQueueRecord => Boolean(record))
      .sort(sortByRecentUpdate);

  const getNormalizedRecord = async (id: string): Promise<ResourceUploadQueueRecord | null> => {
    const direct = normalizeResourceUploadQueueRecord(await driver.get(id));
    if (direct) {
      return direct;
    }
    return (await listNormalizedRecords()).find((record) => record.id === id) ?? null;
  };

  return {
    list: listNormalizedRecords,
    get: getNormalizedRecord,
    upsert: async (input: ResourceUploadQueueUpsertInput) => {
      const existing = await getNormalizedRecord(input.id);
      const record = buildResourceUploadQueueRecord(input, existing, now());
      await driver.put(record);
      return record;
    },
    updateProgress: async (id: string, patch: ResourceUploadQueueProgressPatch) => {
      const existing = await getNormalizedRecord(id);
      if (!existing) {
        return null;
      }
      const timestamp = patch.updatedAt ?? now();
      const requestedStatus = patch.status ?? existing.status;
      const preserveCompleted = existing.status === "completed" && requestedStatus !== "completed";
      const status = preserveCompleted ? "completed" : requestedStatus;
      const record: ResourceUploadQueueRecord = {
        ...existing,
        sessionId: patch.sessionId === undefined ? existing.sessionId : asOptionalString(patch.sessionId),
        bytesVerified: asBytes(Math.max(existing.bytesVerified, patch.bytesVerified ?? 0), 0, existing.totalBytes),
        bytesCommitted: asBytes(Math.max(existing.bytesCommitted, patch.bytesCommitted ?? 0), 0, existing.totalBytes),
        status,
        error: preserveCompleted ? existing.error : patch.error === undefined ? existing.error : asOptionalString(patch.error),
        updatedAt: asDateString(timestamp) ?? now(),
        completedAt:
          status === "completed"
            ? asDateString(patch.completedAt) ?? existing.completedAt ?? timestamp
            : asDateString(patch.completedAt) ?? existing.completedAt,
      };
      await driver.put(record);
      return record;
    },
    remove: async (id: string) => {
      await driver.delete(id);
    },
    clear: async () => {
      await driver.clear();
    },
  };
};

export const persistResourceUploadProgressEvent = async (
  store: Pick<ResourceUploadQueueStore, "get" | "upsert" | "updateProgress">,
  event: UploadProgressEvent
): Promise<ResourceUploadQueueRecord | null> => {
  const id = asOptionalString(event.id);
  const fingerprint = asOptionalString(event.fingerprint);
  const fileName = asOptionalString(event.fileName);
  const fileToken = asOptionalString(event.fileToken);
  if (!id || !fingerprint || !fileName || !fileToken) {
    return null;
  }
  const existing = await store.get(id);
  if (existing) {
    return store.updateProgress(id, {
      sessionId: event.sessionId ?? existing.sessionId,
      bytesVerified: event.bytesVerified,
      bytesCommitted: event.bytesCommitted,
      status: event.status,
      error: event.error ?? null,
    });
  }
  return store.upsert({
    id,
    fingerprint,
    fileName,
    relativePath: event.relativePath ?? null,
    fileToken,
    sessionId: event.sessionId ?? null,
    contentType: event.contentType ?? "application/octet-stream",
    totalBytes: event.totalBytes,
    bytesVerified: event.bytesVerified,
    bytesCommitted: event.bytesCommitted,
    chunkSizeBytes: event.chunkSizeBytes ?? DEFAULT_UPLOAD_CHUNK_SIZE_BYTES,
    status: event.status,
    error: event.error ?? null,
  });
};

const recoverableHydrationStatuses = new Set<ResourceUploadQueueStatus>([
  "queued",
  "creating",
  "uploading",
  "verifying",
  "paused",
  "needs_file",
  "failed",
]);

const terminalHydrationStatuses = new Set<ResourceUploadQueueStatus>([
  "completed",
  "canceled",
]);

const verifiedBytesForUploadSessionFile = (
  session: UploadSessionResponse,
  fileToken: string,
  fallbackBytes: number
): number => {
  const chunks = session.chunks ?? [];
  const verifiedBytes = chunks
    .filter((chunk) => chunk.file_token === fileToken && chunk.status === "verified")
    .reduce((total, chunk) => total + asBytes(chunk.size_bytes), 0);
  return Math.max(fallbackBytes, verifiedBytes);
};

const failedUploadChunkForFile = (
  session: UploadSessionResponse,
  fileToken: string
) =>
  (session.chunks ?? []).find((chunk) => chunk.file_token === fileToken && chunk.status === "failed") ?? null;

const uploadSessionErrorStatus = (error: unknown): number | null => {
  const record = asObject(error);
  if (!record) {
    return null;
  }
  const status = Number(record.status);
  return Number.isFinite(status) ? Math.floor(status) : null;
};

const isMissingUploadSessionError = (error: unknown): boolean => {
  const status = uploadSessionErrorStatus(error);
  return status === 404 || status === 410;
};

const reconcileUploadQueueRecordWithSession = async (
  store: Pick<ResourceUploadQueueStore, "updateProgress">,
  record: ResourceUploadQueueRecord,
  loadUploadSession: ResourceUploadSessionLoader
): Promise<ResourceUploadQueueRecord> => {
  if (!record.sessionId) {
    return record;
  }
  try {
    const session = await loadUploadSession(record.sessionId);
    const sessionFile = session.files.find((file) => file.file_token === record.fileToken) ?? null;
    const failedChunk = failedUploadChunkForFile(session, record.fileToken);
    const completed =
      session.session.status === "completed" || sessionFile?.status === "completed";
    const canceled = session.session.status === "canceled";
    const paused = session.session.status === "paused";
    const failed =
      session.session.status === "failed" || sessionFile?.status === "failed" || Boolean(failedChunk);
    const serverVerifiedBytes = completed && sessionFile
      ? sessionFile.size_bytes
      : verifiedBytesForUploadSessionFile(session, record.fileToken, record.bytesVerified);
    const serverCommittedBytes = completed && sessionFile
      ? sessionFile.size_bytes
      : Math.max(record.bytesCommitted, sessionFile?.resource_id ? sessionFile.size_bytes : 0);
    const status: ResourceUploadQueueStatus = completed
      ? "completed"
      : canceled
        ? "canceled"
        : failed
          ? "failed"
          : paused
            ? "paused"
            : "needs_file";
    const error = completed
      ? null
      : failedChunk?.error || sessionFile?.error || session.session.error || (status === "needs_file" ? UPLOAD_RESELECT_MESSAGE : record.error);
    return (
      (await store.updateProgress(record.id, {
        sessionId: session.session.session_id,
        bytesVerified: serverVerifiedBytes,
        bytesCommitted: serverCommittedBytes,
        status,
        error,
        completedAt: completed ? sessionFile?.completed_at ?? session.session.completed_at ?? undefined : undefined,
      })) ?? record
    );
  } catch (error) {
    if (isMissingUploadSessionError(error)) {
      return (
        (await store.updateProgress(record.id, {
          bytesVerified: record.bytesVerified,
          bytesCommitted: record.bytesCommitted,
          status: "failed",
          error: UPLOAD_SESSION_EXPIRED_MESSAGE,
        })) ?? {
          ...record,
          status: "failed",
          error: UPLOAD_SESSION_EXPIRED_MESSAGE,
        }
      );
    }
    return record;
  }
};

export const hydrateResourceUploadProgressFromQueueStore = async (
  store: Pick<ResourceUploadQueueStore, "list" | "updateProgress">,
  options: HydrateResourceUploadProgressOptions = {}
): Promise<ResourceUploadProgressSummary[]> => {
  const limit = Math.max(1, Math.floor(options.limit ?? 12));
  const records = await store.list();
  const progress: ResourceUploadProgressSummary[] = [];
  const sessionStatusById = new Map<string, Promise<UploadSessionResponse>>();
  const uploadSessionLoader = options.loadUploadSession;
  const loadUploadSession = uploadSessionLoader
    ? (sessionId: string): Promise<UploadSessionResponse> => {
        const normalizedSessionId = asTrimmedString(sessionId);
        if (!normalizedSessionId) {
          return Promise.reject(new Error("Upload session id is required"));
        }
        const existing = sessionStatusById.get(normalizedSessionId);
        if (existing) {
          return existing;
        }
        const next = uploadSessionLoader(normalizedSessionId);
        sessionStatusById.set(normalizedSessionId, next);
        return next;
      }
    : null;
  for (const localRecord of records) {
    if (progress.length >= limit) {
      break;
    }
    if (!recoverableHydrationStatuses.has(localRecord.status)) {
      continue;
    }
    const record = loadUploadSession
      ? await reconcileUploadQueueRecordWithSession(store, localRecord, loadUploadSession)
      : localRecord;
    const terminalReconciled = Boolean(options.loadUploadSession) && terminalHydrationStatuses.has(record.status);
    if (!recoverableHydrationStatuses.has(record.status) && !terminalReconciled) {
      continue;
    }
    const status =
      record.status === "failed" ||
      record.status === "needs_file" ||
      record.status === "paused" ||
      terminalHydrationStatuses.has(record.status)
        ? record.status
        : "needs_file";
    const error =
      status === "needs_file"
        ? record.error || UPLOAD_RESELECT_MESSAGE
        : record.error;
    if (status !== record.status || error !== record.error) {
      await store.updateProgress(record.id, {
        status,
        bytesVerified: record.bytesVerified,
        bytesCommitted: record.bytesCommitted,
        error,
      });
    }
    const summary: ResourceUploadProgressSummary = {
      id: record.id,
      fingerprint: record.fingerprint,
      sessionId: record.sessionId,
      fileToken: record.fileToken,
      fileName: record.fileName,
      status,
      totalBytes: record.totalBytes,
      bytesVerified: record.bytesVerified,
      error,
    };
    if (record.relativePath) {
      summary.relativePath = record.relativePath;
    }
    progress.push(summary);
  }
  return progress;
};
