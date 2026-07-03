import { lazy, type ComponentType } from "react";

import {
  isChunkLoadError,
  reloadOnceForChunkError,
  reportClientError,
} from "./client-diagnostics";

type RetryOptions = {
  /** Number of extra attempts after the first failure (chunk/transport errors only). */
  retries?: number;
  /** Base backoff in ms; grows linearly per attempt. */
  delayMs?: number;
  /** Human label used in diagnostics. */
  label?: string;
};

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });

/**
 * Run a dynamic import with bounded retries. Transient network blips are retried;
 * a persistent chunk-load failure (stale deploy) triggers a single guarded reload
 * instead of bubbling a `ChunkLoadError` up to an error boundary. The returned
 * promise intentionally never settles in the reload case so any `<Suspense>`
 * fallback stays visible until the page reloads.
 */
export async function retryDynamicImport<T>(
  factory: () => Promise<T>,
  { retries = 1, delayMs = 350, label }: RetryOptions = {}
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await factory();
    } catch (error) {
      lastError = error;
      // Only retry chunk/transport failures; real module errors should surface fast.
      if (!isChunkLoadError(error) || attempt === retries) {
        break;
      }
      await wait(delayMs * (attempt + 1));
    }
  }

  if (isChunkLoadError(lastError) && reloadOnceForChunkError(label)) {
    // Reload scheduled — keep the Suspense fallback up by never resolving.
    return new Promise<T>(() => {});
  }

  reportClientError(lastError, { source: "lazy-import", extra: { label } });
  throw lastError;
}

/** `React.lazy` for a named export, hardened with retry + stale-deploy reload. */
export function lazyNamedWithRetry<TModule extends Record<string, unknown>>(
  loader: () => Promise<TModule>,
  exportName: keyof TModule
) {
  return lazy(() =>
    retryDynamicImport(loader, { label: String(exportName) }).then((module) => ({
      // Permissive prop typing mirrors the prior inline helper so existing call
      // sites that pass props to these lazy components keep type-checking.
      default: module[exportName] as ComponentType<any>,
    }))
  );
}
