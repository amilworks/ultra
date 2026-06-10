import type { ResourceUploadProgress } from "@/components/ResourceBrowser";
import type { UploadProgressEvent } from "@/lib/api";

export type ResourceUploadProgressFrameBatcher = {
  enqueue: (event: UploadProgressEvent) => void;
  flush: () => void;
  clear: () => void;
};

type ResourceUploadProgressFrameBatcherOptions = {
  onFlush: (events: UploadProgressEvent[]) => void;
  schedule?: (flush: () => void) => void;
};

const scheduleNextFrame = (flush: () => void): void => {
  if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
    window.requestAnimationFrame(flush);
    return;
  }
  setTimeout(flush, 16);
};

export const mergeResourceUploadProgress = (
  items: ResourceUploadProgress[],
  event: UploadProgressEvent
): ResourceUploadProgress[] => {
  const nextItem: ResourceUploadProgress = {
    id: event.id,
    fingerprint: event.fingerprint ?? null,
    sessionId: event.sessionId ?? null,
    fileToken: event.fileToken ?? null,
    fileName: event.fileName,
    relativePath: event.relativePath ?? null,
    status: event.status,
    totalBytes: event.totalBytes,
    bytesVerified: event.bytesVerified,
    error: event.error ?? null,
  };
  const withoutCurrent = items.filter((item) => item.id !== event.id);
  return [nextItem, ...withoutCurrent].slice(0, 12);
};

export const createResourceUploadProgressFrameBatcher = ({
  onFlush,
  schedule = scheduleNextFrame,
}: ResourceUploadProgressFrameBatcherOptions): ResourceUploadProgressFrameBatcher => {
  const pending = new Map<string, UploadProgressEvent>();
  let scheduled = false;

  const flush = (): void => {
    scheduled = false;
    if (pending.size === 0) {
      return;
    }
    const events = Array.from(pending.values());
    pending.clear();
    onFlush(events);
  };

  return {
    enqueue: (event) => {
      pending.set(event.id, event);
      if (scheduled) {
        return;
      }
      scheduled = true;
      schedule(flush);
    },
    flush,
    clear: () => {
      pending.clear();
      scheduled = false;
    },
  };
};
