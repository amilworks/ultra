import { describe, expect, it } from "vitest";

import type { UploadProgressEvent } from "@/lib/api";

import { createResourceUploadProgressFrameBatcher } from "./uploadProgressBatcher";

const progressEvent = (
  id: string,
  bytesVerified: number,
  status: UploadProgressEvent["status"] = "uploading"
): UploadProgressEvent => ({
  id,
  fileName: `${id}.nii`,
  fileIndex: 0,
  fileToken: id,
  sessionId: "upload_session_large_folder",
  fingerprint: `${id}:1024:1780915200000:application/x-nifti`,
  relativePath: `study/${id}.nii`,
  contentType: "application/x-nifti",
  chunkSizeBytes: 8 * 1024 * 1024,
  status,
  totalBytes: 1024,
  bytesVerified,
  bytesCommitted: status === "completed" ? bytesVerified : 0,
});

describe("resource upload progress frame batcher", () => {
  it("coalesces many upload progress events into one scheduled UI flush", () => {
    const scheduled: Array<() => void> = [];
    const flushes: UploadProgressEvent[][] = [];
    const batcher = createResourceUploadProgressFrameBatcher({
      schedule: (flush) => {
        scheduled.push(flush);
      },
      onFlush: (events) => {
        flushes.push(events);
      },
    });

    batcher.enqueue(progressEvent("file_a", 256));
    batcher.enqueue(progressEvent("file_a", 512));
    batcher.enqueue(progressEvent("file_b", 128));

    expect(scheduled).toHaveLength(1);
    expect(flushes).toHaveLength(0);

    scheduled[0]?.();

    expect(flushes).toHaveLength(1);
    expect(flushes[0]).toHaveLength(2);
    expect(flushes[0]?.find((event) => event.id === "file_a")?.bytesVerified).toBe(512);
    expect(flushes[0]?.find((event) => event.id === "file_b")?.bytesVerified).toBe(128);
  });

  it("can flush synchronously before upload cleanup removes completed rows", () => {
    const scheduled: Array<() => void> = [];
    const flushes: UploadProgressEvent[][] = [];
    const batcher = createResourceUploadProgressFrameBatcher({
      schedule: (flush) => {
        scheduled.push(flush);
      },
      onFlush: (events) => {
        flushes.push(events);
      },
    });

    batcher.enqueue(progressEvent("file_done", 1024, "completed"));
    batcher.flush();
    scheduled[0]?.();

    expect(flushes).toHaveLength(1);
    expect(flushes[0]).toEqual([progressEvent("file_done", 1024, "completed")]);
  });
});
