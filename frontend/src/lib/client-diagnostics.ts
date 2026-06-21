/**
 * Lightweight, self-contained client error diagnostics.
 *
 * The app historically had a single top-level error boundary and no telemetry,
 * so a render-time throw (most often an unhandled `ChunkLoadError` after a
 * redeploy rotated Vite chunk hashes) replaced the whole interface with a
 * generic "Something went wrong" screen with no way to learn what failed.
 *
 * This module captures errors from React boundaries and global handlers,
 * classifies chunk-load failures, keeps a small inspectable ring buffer, and
 * recovers from stale-deploy chunk errors with a single guarded reload. It has
 * no external dependencies and never throws.
 */

export type ClientErrorSource =
  | "react-render"
  | "error-boundary"
  | "message-row"
  | "unhandledrejection"
  | "window-error"
  | "lazy-import"
  | "vite-preload"
  | "markdown-lexer"
  | string;

export type ClientErrorContext = {
  source: ClientErrorSource;
  componentStack?: string;
  /** Set by recovery paths so a reload/retry is not double-counted as a crash. */
  recovered?: boolean;
  extra?: Record<string, unknown>;
};

type StoredClientError = {
  at: number;
  source: ClientErrorSource;
  name: string;
  message: string;
  chunkLoad: boolean;
  recovered: boolean;
  componentStack?: string;
  extra?: Record<string, unknown>;
};

const STORAGE_KEY = "ultra:client-errors";
const CHUNK_RELOAD_KEY = "ultra:chunk-reload-at";
const MAX_STORED_ERRORS = 25;
/** Cooldown that prevents a stale-chunk reload from looping if the new bundle is also broken. */
const CHUNK_RELOAD_COOLDOWN_MS = 30_000;

const CHUNK_LOAD_PATTERNS = [
  /loading chunk [\w-]+ failed/i,
  /failed to fetch dynamically imported module/i,
  /error loading dynamically imported module/i,
  /importing a module script failed/i,
  /'?text\/html'? is not a valid javascript mime type/i,
  /unable to preload css/i,
];

const toError = (value: unknown): Error =>
  value instanceof Error ? value : new Error(typeof value === "string" ? value : String(value));

/** True when an error is a stale/missing JS chunk or a failed dynamic import preload. */
export const isChunkLoadError = (value: unknown): boolean => {
  if (!value) {
    return false;
  }
  const error = value instanceof Error ? value : null;
  const name = error?.name ?? "";
  if (name === "ChunkLoadError") {
    return true;
  }
  const message = error?.message ?? (typeof value === "string" ? value : "");
  return CHUNK_LOAD_PATTERNS.some((pattern) => pattern.test(message));
};

const readStored = (): StoredClientError[] => {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as StoredClientError[]) : [];
  } catch {
    return [];
  }
};

const persist = (entry: StoredClientError): void => {
  try {
    const next = [...readStored(), entry].slice(-MAX_STORED_ERRORS);
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // sessionStorage may be full or blocked; reporting is best effort.
  }
};

/**
 * Record a client error: structured console output (kept in production so crash
 * diagnostics survive), a sessionStorage ring buffer, and a `window` handle that
 * support can read (`window.__ultraClientErrors`).
 */
export const reportClientError = (value: unknown, context: ClientErrorContext): void => {
  const error = toError(value);
  const chunkLoad = isChunkLoadError(error);
  const entry: StoredClientError = {
    at: Date.now(),
    source: context.source,
    name: error.name,
    message: error.message,
    chunkLoad,
    recovered: Boolean(context.recovered),
    componentStack: context.componentStack,
    extra: context.extra,
  };

  // Keep a structured, greppable line in the console even in production builds.
  // Handled/recovered events are warnings; genuine crashes are errors.
  const tag = entry.recovered ? "recovered" : chunkLoad ? "chunk-load" : "error";
  const log = entry.recovered ? console.warn : console.error;
  log(`[ultra:${context.source}] ${tag}: ${error.name}: ${error.message}`, {
    error,
    componentStack: context.componentStack,
    recovered: entry.recovered,
    extra: context.extra,
  });

  persist(entry);

  try {
    const win = window as typeof window & { __ultraClientErrors?: StoredClientError[] };
    win.__ultraClientErrors = readStored();
  } catch {
    // Ignore: exposing the buffer on window is a convenience, not a requirement.
  }
};

/**
 * Reload once to recover from a stale-deploy chunk error. Guarded by a short
 * cooldown so a genuinely broken bundle does not reload-loop. Returns true when
 * a reload was scheduled (callers should then keep their Suspense fallback up).
 */
export const reloadOnceForChunkError = (label?: string): boolean => {
  const now = Date.now();
  try {
    const last = Number(window.sessionStorage.getItem(CHUNK_RELOAD_KEY) || 0);
    if (last && now - last < CHUNK_RELOAD_COOLDOWN_MS) {
      return false;
    }
    window.sessionStorage.setItem(CHUNK_RELOAD_KEY, String(now));
  } catch {
    // If storage is unavailable we still attempt a single reload below.
  }
  reportClientError(new Error(`Recovering from chunk-load failure${label ? `: ${label}` : ""}`), {
    source: "lazy-import",
    recovered: true,
    extra: { label },
  });
  window.location.reload();
  return true;
};

let globalHandlersInstalled = false;

/** Install idempotent global listeners for unhandled rejections, errors, and Vite preload failures. */
export const installGlobalErrorReporting = (): void => {
  if (globalHandlersInstalled || typeof window === "undefined") {
    return;
  }
  globalHandlersInstalled = true;

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    if (isChunkLoadError(reason)) {
      if (reloadOnceForChunkError("unhandledrejection")) {
        event.preventDefault();
        return;
      }
    }
    reportClientError(reason, { source: "unhandledrejection" });
  });

  window.addEventListener("error", (event) => {
    // Resource load errors (e.g. <img>) surface here without an Error object; ignore those.
    if (!event.error) {
      return;
    }
    if (isChunkLoadError(event.error)) {
      reloadOnceForChunkError("window-error");
      return;
    }
    reportClientError(event.error, { source: "window-error" });
  });

  // Vite fires this when a module preload fails (typically a rotated chunk hash).
  window.addEventListener("vite:preloadError", (event) => {
    if (reloadOnceForChunkError("vite-preload")) {
      event.preventDefault();
      return;
    }
    reportClientError((event as unknown as { payload?: unknown }).payload ?? event, {
      source: "vite-preload",
    });
  });
};

/** Read the captured error buffer (for support tooling / debugging). */
export const getClientErrorLog = (): StoredClientError[] => readStored();
