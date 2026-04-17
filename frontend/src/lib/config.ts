const readEnvString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

export const DEFAULT_API_BASE_URL =
  readEnvString(import.meta.env.VITE_API_BASE_URL) ??
  (typeof window === "undefined"
    ? "http://localhost:8000"
    : window.location.origin);

export const DEFAULT_API_KEY =
  import.meta.env.DEV
    ? readEnvString(import.meta.env.VITE_ORCHESTRATOR_API_KEY) ?? ""
    : "";

export const DEFAULT_BISQUE_BROWSER_URL =
  readEnvString(import.meta.env.VITE_BISQUE_BROWSER_URL) ?? "";

export const DEFAULT_MAX_TOOL_CALLS = 12;
export const DEFAULT_MAX_RUNTIME_SECONDS = 900;
export const DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS = 7200;
