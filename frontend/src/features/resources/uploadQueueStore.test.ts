import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  RESOURCE_UPLOAD_QUEUE_STORAGE_KEY,
  createResourceUploadQueueStore,
  hydrateResourceUploadProgressFromQueueStore,
  persistResourceUploadProgressEvent,
} from "./uploadQueueStore";
import type { UploadSessionResponse } from "@/types";

const createMemoryStorage = (): Storage => {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => Array.from(values.keys())[index] ?? null,
    removeItem: (key: string) => {
      values.delete(key);
    },
    setItem: (key: string, value: string) => {
      values.set(key, String(value));
    },
  };
};

describe("resource upload queue store", () => {
  let storage: Storage;
  let nowIndex = 0;
  const nextNow = () => {
    nowIndex += 1;
    return `2026-06-08T00:00:0${nowIndex}.000Z`;
  };

  beforeEach(() => {
    storage = createMemoryStorage();
    nowIndex = 0;
    Object.defineProperty(window, "localStorage", {
      value: storage,
      configurable: true,
    });
    Object.defineProperty(window, "indexedDB", {
      value: undefined,
      configurable: true,
    });
    vi.stubGlobal("localStorage", storage);
    vi.stubGlobal("indexedDB", undefined);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("persists recoverable upload session state across store instances", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });

    const saved = await store.upsert({
      id: "file-0-brain",
      fingerprint: "brain.nii:1048576:1780915200000:application/x-nifti",
      fileName: "brain.nii",
      fileToken: "file-0-brain",
      sessionId: "upload_session_123",
      contentType: "application/x-nifti",
      totalBytes: 1_048_576,
      bytesVerified: 524_288,
      bytesCommitted: 0,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "uploading",
    });

    expect(saved).toMatchObject({
      id: "file-0-brain",
      fileName: "brain.nii",
      sessionId: "upload_session_123",
      status: "uploading",
      bytesVerified: 524_288,
      bytesCommitted: 0,
      error: null,
      createdAt: "2026-06-08T00:00:01.000Z",
      updatedAt: "2026-06-08T00:00:01.000Z",
    });
    expect(storage.getItem(RESOURCE_UPLOAD_QUEUE_STORAGE_KEY)).toContain("upload_session_123");

    const rehydratedStore = createResourceUploadQueueStore({ now: nextNow });
    await expect(rehydratedStore.list()).resolves.toEqual([saved]);
  });

  it("keeps byte counters monotonic when parallel chunk updates arrive out of order", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-0-stack",
      fingerprint: "stack.ome.tiff:25165827:1780915200000:image/tiff",
      fileName: "stack.ome.tiff",
      fileToken: "file-0-stack",
      sessionId: "upload_session_parallel",
      contentType: "image/tiff",
      totalBytes: 25_165_827,
      bytesVerified: 16_777_216,
      bytesCommitted: 0,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "uploading",
    });

    const staleUpdate = await store.updateProgress("file-0-stack", {
      bytesVerified: 8_388_608,
      bytesCommitted: 0,
      status: "uploading",
    });
    const freshUpdate = await store.updateProgress("file-0-stack", {
      bytesVerified: 25_165_827,
      bytesCommitted: 25_165_827,
      status: "completed",
    });

    expect(staleUpdate?.bytesVerified).toBe(16_777_216);
    expect(freshUpdate).toMatchObject({
      bytesVerified: 25_165_827,
      bytesCommitted: 25_165_827,
      status: "completed",
      completedAt: "2026-06-08T00:00:03.000Z",
    });
  });

  it("does not let stale failure events demote a completed upload", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-0-alpha",
      fingerprint: "alpha.ome.tiff:5:1780915200000:image/tiff",
      fileName: "alpha.ome.tiff",
      fileToken: "file-0-alpha",
      sessionId: "upload_session_folder",
      contentType: "image/tiff",
      totalBytes: 5,
      bytesVerified: 5,
      bytesCommitted: 5,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "completed",
      error: null,
    });

    const staleFailure = await store.updateProgress("file-0-alpha", {
      status: "failed",
      bytesVerified: 0,
      bytesCommitted: 0,
      error: "Request failed with status 400",
    });

    expect(staleFailure).toMatchObject({
      status: "completed",
      bytesVerified: 5,
      bytesCommitted: 5,
      error: null,
      completedAt: "2026-06-08T00:00:01.000Z",
    });
  });

  it("drops malformed persisted queue entries instead of blocking hydration", async () => {
    storage.setItem(
      RESOURCE_UPLOAD_QUEUE_STORAGE_KEY,
      JSON.stringify([
        { id: "", fileName: "missing-id.nii" },
        {
          id: "file-valid",
          fingerprint: "valid:42:1780915200000:image/png",
          fileName: "valid.png",
          fileToken: "file-valid",
          sessionId: "upload_session_valid",
          contentType: "image/png",
          totalBytes: 42,
          bytesVerified: -100,
          bytesCommitted: 99,
          chunkSizeBytes: 0,
          status: "mystery",
          createdAt: "2026-06-08T00:00:00.000Z",
          updatedAt: "2026-06-08T00:00:00.000Z",
        },
      ])
    );

    const store = createResourceUploadQueueStore({ now: nextNow });

    await expect(store.list()).resolves.toEqual([
      {
        id: "file-valid",
        fingerprint: "valid:42:1780915200000:image/png",
        fileName: "valid.png",
        relativePath: null,
        fileToken: "file-valid",
        sessionId: "upload_session_valid",
        contentType: "image/png",
        totalBytes: 42,
        bytesVerified: 0,
        bytesCommitted: 42,
        chunkSizeBytes: 8 * 1024 * 1024,
        status: "queued",
        error: null,
        createdAt: "2026-06-08T00:00:00.000Z",
        updatedAt: "2026-06-08T00:00:00.000Z",
        completedAt: null,
      },
    ]);
  });

  it("persists API upload progress events as recoverable queue records", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });

    const initial = await persistResourceUploadProgressEvent(store, {
      id: "file-0-brain:1048576:1780915200000",
      fileName: "brain.nii",
      fileIndex: 0,
      fileToken: "file-0-brain",
      sessionId: "upload_session_456",
      fingerprint: "brain.nii:1048576:1780915200000:application/x-nifti:hash",
      relativePath: "study-1/brain.nii",
      contentType: "application/x-nifti",
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "creating",
      totalBytes: 1_048_576,
      bytesVerified: 0,
      bytesCommitted: 0,
    });
    const updated = await persistResourceUploadProgressEvent(store, {
      id: "file-0-brain:1048576:1780915200000",
      fileName: "brain.nii",
      fileIndex: 0,
      fileToken: "file-0-brain",
      sessionId: "upload_session_456",
      fingerprint: "brain.nii:1048576:1780915200000:application/x-nifti:hash",
      relativePath: "study-1/brain.nii",
      contentType: "application/x-nifti",
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "uploading",
      totalBytes: 1_048_576,
      bytesVerified: 524_288,
      bytesCommitted: 0,
    });

    expect(initial).toMatchObject({
      id: "file-0-brain:1048576:1780915200000",
      status: "creating",
      createdAt: "2026-06-08T00:00:01.000Z",
    });
    expect(updated).toMatchObject({
      id: "file-0-brain:1048576:1780915200000",
      status: "uploading",
      sessionId: "upload_session_456",
      relativePath: "study-1/brain.nii",
      bytesVerified: 524_288,
      createdAt: "2026-06-08T00:00:01.000Z",
      updatedAt: "2026-06-08T00:00:02.000Z",
    });
    await expect(store.list()).resolves.toHaveLength(1);
  });

  it("hydrates reload progress from recoverable queue records and marks interrupted uploads", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-active",
      fingerprint: "active.nii:1000:1780915200000:application/x-nifti",
      fileName: "active.nii",
      fileToken: "file-active",
      sessionId: "upload_session_active",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 400,
      bytesCommitted: 0,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "uploading",
    });
    await store.upsert({
      id: "file-done",
      fingerprint: "done.png:10:1780915200000:image/png",
      fileName: "done.png",
      fileToken: "file-done",
      sessionId: "upload_session_done",
      contentType: "image/png",
      totalBytes: 10,
      bytesVerified: 10,
      bytesCommitted: 10,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "completed",
    });
    await store.upsert({
      id: "file-folder",
      fingerprint: "folder-brain.nii:2000:1780915200000:application/x-nifti",
      fileName: "brain.nii",
      relativePath: "field-study/session-1/brain.nii",
      fileToken: "file-folder",
      sessionId: "upload_session_folder",
      contentType: "application/x-nifti",
      totalBytes: 2000,
      bytesVerified: 768,
      bytesCommitted: 0,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "failed",
      error: "Connection dropped while uploading.",
    });
    await store.upsert({
      id: "file-canceled",
      fingerprint: "canceled.png:10:1780915200000:image/png",
      fileName: "canceled.png",
      fileToken: "file-canceled",
      sessionId: "upload_session_canceled",
      contentType: "image/png",
      totalBytes: 10,
      bytesVerified: 0,
      bytesCommitted: 0,
      chunkSizeBytes: 8 * 1024 * 1024,
      status: "canceled",
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store);

    expect(progress).toEqual([
      {
        id: "file-folder",
        fingerprint: "folder-brain.nii:2000:1780915200000:application/x-nifti",
        fileToken: "file-folder",
        fileName: "brain.nii",
        relativePath: "field-study/session-1/brain.nii",
        sessionId: "upload_session_folder",
        status: "failed",
        totalBytes: 2000,
        bytesVerified: 768,
        error: "Connection dropped while uploading.",
      },
      {
        id: "file-active",
        fingerprint: "active.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-active",
        fileName: "active.nii",
        sessionId: "upload_session_active",
        status: "needs_file",
        totalBytes: 1000,
        bytesVerified: 400,
        error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
      },
    ]);
    await expect(store.get("file-active")).resolves.toMatchObject({
      status: "needs_file",
      bytesVerified: 400,
      error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
    });
  });

  it("reconciles reload progress with authoritative upload-session status", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-active",
      fingerprint: "active.nii:1000:1780915200000:application/x-nifti",
      fileName: "active.nii",
      fileToken: "file-active",
      sessionId: "upload_session_active",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 0,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store, {
      loadUploadSession: async (sessionId): Promise<UploadSessionResponse> => {
        expect(sessionId).toBe("upload_session_active");
        return {
          session: {
            session_id: "upload_session_active",
            owner_user_id: "field-user",
            source_type: "upload",
            status: "active",
            total_bytes: 1000,
            bytes_received: 512,
            bytes_verified: 512,
            bytes_committed: 0,
            created_at: "2026-06-08T00:00:00.000Z",
            updated_at: "2026-06-08T00:00:02.000Z",
            metadata: {},
          },
          files: [
            {
              session_id: "upload_session_active",
              file_token: "file-active",
              original_name: "active.nii",
              content_type: "application/x-nifti",
              size_bytes: 1000,
              status: "uploading",
              created_at: "2026-06-08T00:00:00.000Z",
              updated_at: "2026-06-08T00:00:02.000Z",
              metadata: {},
            },
          ],
          chunks: [
            {
              session_id: "upload_session_active",
              file_token: "file-active",
              chunk_index: 0,
              offset: 0,
              size_bytes: 512,
              sha256: "verified-sha",
              status: "verified",
              received_at: "2026-06-08T00:00:01.000Z",
              verified_at: "2026-06-08T00:00:01.000Z",
              metadata: {},
            },
          ],
        };
      },
    });

    expect(progress).toEqual([
      {
        id: "file-active",
        fingerprint: "active.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-active",
        fileName: "active.nii",
        sessionId: "upload_session_active",
        status: "needs_file",
        totalBytes: 1000,
        bytesVerified: 512,
        error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
      },
    ]);
    await expect(store.get("file-active")).resolves.toMatchObject({
      status: "needs_file",
      bytesVerified: 512,
      error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
    });
  });

  it("marks missing durable upload sessions as expired instead of surfacing raw HTTP errors", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-expired",
      fingerprint: "expired.nii:1000:1780915200000:application/x-nifti",
      fileName: "expired.nii",
      fileToken: "file-expired",
      sessionId: "upload_session_missing",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 256,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
      error: "Request failed with status 404",
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store, {
      loadUploadSession: async () => {
        const error = new Error("Request failed with status 404") as Error & { status: number };
        error.status = 404;
        throw error;
      },
    });

    expect(progress).toEqual([
      {
        id: "file-expired",
        fingerprint: "expired.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-expired",
        fileName: "expired.nii",
        sessionId: "upload_session_missing",
        status: "failed",
        totalBytes: 1000,
        bytesVerified: 256,
        error: "Upload session expired. Start this upload again.",
      },
    ]);
    await expect(store.get("file-expired")).resolves.toMatchObject({
      status: "failed",
      bytesVerified: 256,
      error: "Upload session expired. Start this upload again.",
    });
  });

  it("keeps paused upload sessions visible as paused after refresh reconciliation", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-paused",
      fingerprint: "paused.nii:1000:1780915200000:application/x-nifti",
      fileName: "paused.nii",
      fileToken: "file-paused",
      sessionId: "upload_session_paused",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 256,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store, {
      loadUploadSession: async (): Promise<UploadSessionResponse> => ({
        session: {
          session_id: "upload_session_paused",
          owner_user_id: "field-user",
          source_type: "upload",
          status: "paused",
          total_bytes: 1000,
          bytes_received: 512,
          bytes_verified: 512,
          bytes_committed: 0,
          created_at: "2026-06-08T00:00:00.000Z",
          updated_at: "2026-06-08T00:00:02.000Z",
          metadata: {},
        },
        files: [
          {
            session_id: "upload_session_paused",
            file_token: "file-paused",
            original_name: "paused.nii",
            content_type: "application/x-nifti",
            size_bytes: 1000,
            status: "uploading",
            created_at: "2026-06-08T00:00:00.000Z",
            updated_at: "2026-06-08T00:00:02.000Z",
            metadata: {},
          },
        ],
        chunks: [
          {
            session_id: "upload_session_paused",
            file_token: "file-paused",
            chunk_index: 0,
            offset: 0,
            size_bytes: 512,
            sha256: "verified-sha",
            status: "verified",
            received_at: "2026-06-08T00:00:01.000Z",
            verified_at: "2026-06-08T00:00:01.000Z",
            metadata: {},
          },
        ],
      }),
    });

    expect(progress).toEqual([
      {
        id: "file-paused",
        fingerprint: "paused.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-paused",
        fileName: "paused.nii",
        sessionId: "upload_session_paused",
        status: "paused",
        totalBytes: 1000,
        bytesVerified: 512,
        error: null,
      },
    ]);
    await expect(store.get("file-paused")).resolves.toMatchObject({
      status: "paused",
      bytesVerified: 512,
      error: null,
    });
  });

  it("reconciles multi-file folder refresh with one status request per upload session", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-folder-a",
      fingerprint: "folder/session-a.nii:1000:1780915200000:application/x-nifti",
      fileName: "session-a.nii",
      relativePath: "study-1/session-a.nii",
      fileToken: "file-folder-a",
      sessionId: "upload_session_folder",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 0,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });
    await store.upsert({
      id: "file-folder-b",
      fingerprint: "folder/session-b.nii:1000:1780915200000:application/x-nifti",
      fileName: "session-b.nii",
      relativePath: "study-1/session-b.nii",
      fileToken: "file-folder-b",
      sessionId: "upload_session_folder",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 0,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });
    const loadUploadSession = vi.fn(async (sessionId: string): Promise<UploadSessionResponse> => {
      expect(sessionId).toBe("upload_session_folder");
      return {
        session: {
          session_id: "upload_session_folder",
          owner_user_id: "field-user",
          source_type: "upload",
          status: "active",
          total_bytes: 2000,
          bytes_received: 768,
          bytes_verified: 512,
          bytes_committed: 0,
          created_at: "2026-06-08T00:00:00.000Z",
          updated_at: "2026-06-08T00:00:02.000Z",
          metadata: {},
        },
        files: [
          {
            session_id: "upload_session_folder",
            file_token: "file-folder-a",
            original_name: "session-a.nii",
            relative_path: "study-1/session-a.nii",
            content_type: "application/x-nifti",
            size_bytes: 1000,
            status: "uploading",
            created_at: "2026-06-08T00:00:00.000Z",
            updated_at: "2026-06-08T00:00:02.000Z",
            metadata: {},
          },
          {
            session_id: "upload_session_folder",
            file_token: "file-folder-b",
            original_name: "session-b.nii",
            relative_path: "study-1/session-b.nii",
            content_type: "application/x-nifti",
            size_bytes: 1000,
            status: "uploading",
            created_at: "2026-06-08T00:00:00.000Z",
            updated_at: "2026-06-08T00:00:02.000Z",
            metadata: {},
          },
        ],
        chunks: [
          {
            session_id: "upload_session_folder",
            file_token: "file-folder-a",
            chunk_index: 0,
            offset: 0,
            size_bytes: 512,
            sha256: "verified-a",
            status: "verified",
            received_at: "2026-06-08T00:00:01.000Z",
            verified_at: "2026-06-08T00:00:01.000Z",
            metadata: {},
          },
          {
            session_id: "upload_session_folder",
            file_token: "file-folder-b",
            chunk_index: 0,
            offset: 0,
            size_bytes: 256,
            sha256: "failed-b",
            status: "failed",
            received_at: "2026-06-08T00:00:01.000Z",
            error: "Chunk checksum mismatch.",
            metadata: {},
          },
        ],
      };
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store, {
      loadUploadSession,
    });

    expect(loadUploadSession).toHaveBeenCalledTimes(1);
    expect(progress).toEqual([
      {
        id: "file-folder-b",
        fingerprint: "folder/session-b.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-folder-b",
        fileName: "session-b.nii",
        relativePath: "study-1/session-b.nii",
        sessionId: "upload_session_folder",
        status: "failed",
        totalBytes: 1000,
        bytesVerified: 0,
        error: "Chunk checksum mismatch.",
      },
      {
        id: "file-folder-a",
        fingerprint: "folder/session-a.nii:1000:1780915200000:application/x-nifti",
        fileToken: "file-folder-a",
        fileName: "session-a.nii",
        relativePath: "study-1/session-a.nii",
        sessionId: "upload_session_folder",
        status: "needs_file",
        totalBytes: 1000,
        bytesVerified: 512,
        error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
      },
    ]);
    await expect(store.get("file-folder-a")).resolves.toMatchObject({
      status: "needs_file",
      bytesVerified: 512,
    });
    await expect(store.get("file-folder-b")).resolves.toMatchObject({
      status: "failed",
      bytesVerified: 0,
      error: "Chunk checksum mismatch.",
    });
  });

  it("returns authoritative terminal refresh states so stale visible progress can be cleared", async () => {
    const store = createResourceUploadQueueStore({ now: nextNow });
    await store.upsert({
      id: "file-completed",
      fingerprint: "completed.nii:1000:1780915200000:application/x-nifti",
      fileName: "completed.nii",
      fileToken: "file-completed",
      sessionId: "upload_session_completed",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 256,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });
    await store.upsert({
      id: "file-canceled",
      fingerprint: "canceled.nii:1000:1780915200000:application/x-nifti",
      fileName: "canceled.nii",
      fileToken: "file-canceled",
      sessionId: "upload_session_canceled",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 256,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });
    await store.upsert({
      id: "file-failed",
      fingerprint: "failed.nii:1000:1780915200000:application/x-nifti",
      fileName: "failed.nii",
      fileToken: "file-failed",
      sessionId: "upload_session_failed",
      contentType: "application/x-nifti",
      totalBytes: 1000,
      bytesVerified: 256,
      bytesCommitted: 0,
      chunkSizeBytes: 256,
      status: "uploading",
    });

    const progress = await hydrateResourceUploadProgressFromQueueStore(store, {
      loadUploadSession: async (sessionId): Promise<UploadSessionResponse> => {
        const status = sessionId.replace("upload_session_", "");
        const fileToken = `file-${status}`;
        return {
          session: {
            session_id: sessionId,
            owner_user_id: "field-user",
            source_type: "upload",
            status,
            total_bytes: 1000,
            bytes_received: status === "completed" ? 1000 : 256,
            bytes_verified: status === "completed" ? 1000 : 256,
            bytes_committed: status === "completed" ? 1000 : 0,
            error: status === "failed" ? "Virus scan rejected upload." : null,
            created_at: "2026-06-08T00:00:00.000Z",
            updated_at: "2026-06-08T00:00:02.000Z",
            completed_at: status === "completed" ? "2026-06-08T00:00:03.000Z" : null,
            metadata: {},
          },
          files: [
            {
              session_id: sessionId,
              file_token: fileToken,
              original_name: `${status}.nii`,
              content_type: "application/x-nifti",
              size_bytes: 1000,
              status,
              resource_id: status === "completed" ? "file_resource_completed" : null,
              error: status === "failed" ? "Virus scan rejected upload." : null,
              created_at: "2026-06-08T00:00:00.000Z",
              updated_at: "2026-06-08T00:00:02.000Z",
              completed_at: status === "completed" ? "2026-06-08T00:00:03.000Z" : null,
              metadata: {},
            },
          ],
          chunks: [],
        };
      },
    });

    expect(progress.map((item) => [item.id, item.status, item.bytesVerified, item.error])).toEqual([
      ["file-failed", "failed", 256, "Virus scan rejected upload."],
      ["file-canceled", "canceled", 256, null],
      ["file-completed", "completed", 1000, null],
    ]);
    await expect(store.get("file-completed")).resolves.toMatchObject({
      status: "completed",
      bytesVerified: 1000,
      bytesCommitted: 1000,
      completedAt: "2026-06-08T00:00:03.000Z",
    });
    await expect(store.get("file-canceled")).resolves.toMatchObject({
      status: "canceled",
      bytesVerified: 256,
      error: null,
    });
    await expect(store.get("file-failed")).resolves.toMatchObject({
      status: "failed",
      bytesVerified: 256,
      error: "Virus scan rejected upload.",
    });
  });
});
