import type { ResourceTextHead } from "../types";

// Shared cache + concurrency gate for Resources-list content-preview thumbnails.
// A 50-card grid must never fire 50 fetches at once: previews are requested only
// when a card scrolls into view (the component's IntersectionObserver), then
// funneled through here so at most MAX_CONCURRENT run concurrently and each file
// is fetched once (results + in-flight promises are cached by file id).

const MAX_CONCURRENT = 5;
const PREVIEW_BYTES = 4096;

const cache = new Map<string, ResourceTextHead>();
const inflight = new Map<string, Promise<ResourceTextHead>>();

let active = 0;
const waiters: Array<() => void> = [];

function acquire(): Promise<void> {
  if (active < MAX_CONCURRENT) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    waiters.push(() => {
      active += 1;
      resolve();
    });
  });
}

function release(): void {
  active -= 1;
  const next = waiters.shift();
  if (next) {
    next();
  }
}

export function getCachedResourcePreview(fileId: string): ResourceTextHead | undefined {
  return cache.get(fileId);
}

export function getResourcePreview(
  fileId: string,
  fetchHead: (fileId: string, maxBytes: number) => Promise<ResourceTextHead>,
  maxBytes: number = PREVIEW_BYTES
): Promise<ResourceTextHead> {
  const cached = cache.get(fileId);
  if (cached) {
    return Promise.resolve(cached);
  }
  const existing = inflight.get(fileId);
  if (existing) {
    return existing;
  }
  const promise = (async () => {
    await acquire();
    try {
      const result = await fetchHead(fileId, maxBytes);
      cache.set(fileId, result);
      return result;
    } finally {
      release();
      inflight.delete(fileId);
    }
  })();
  inflight.set(fileId, promise);
  return promise;
}
