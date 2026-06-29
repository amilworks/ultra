// Shared cache + concurrency gate for lazy figure captions, mirroring the
// resource-preview thumbnail cache. The backend already caches captions on disk and
// bounds VLM concurrency; this just dedupes within a session and avoids firing a
// burst of caption requests when many figures scroll into view at once.

const MAX_CONCURRENT = 4;

const cache = new Map<string, string>(); // key -> caption ("" = no caption / disabled)
const inflight = new Map<string, Promise<string>>();

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

export function figureCaptionKey(runId: string, path: string): string {
  return `${runId}|${path}`;
}

export function getCachedFigureCaption(key: string): string | undefined {
  return cache.get(key);
}

export function fetchFigureCaption(
  key: string,
  fetcher: () => Promise<{ caption: string; enabled: boolean }>
): Promise<string> {
  const cached = cache.get(key);
  if (cached !== undefined) {
    return Promise.resolve(cached);
  }
  const existing = inflight.get(key);
  if (existing) {
    return existing;
  }
  const promise = (async () => {
    await acquire();
    try {
      const result = await fetcher();
      const caption = result.enabled ? (result.caption ?? "").trim() : "";
      cache.set(key, caption);
      return caption;
    } catch {
      // Never cache a transient failure as "no caption" — allow a later retry.
      return "";
    } finally {
      release();
      inflight.delete(key);
    }
  })();
  inflight.set(key, promise);
  return promise;
}
