const readEnvString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const readEnvBoolean = (value: unknown): boolean | undefined => {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  const token = value.trim().toLowerCase();
  if (token === "1" || token === "true" || token === "yes" || token === "on") {
    return true;
  }
  if (token === "0" || token === "false" || token === "no" || token === "off") {
    return false;
  }
  return undefined;
};

export const DEFAULT_API_BASE_URL =
  readEnvString(import.meta.env.VITE_API_BASE_URL) ??
  (typeof window === "undefined"
    ? "http://localhost:8000"
    : window.location.origin);

export const DEFAULT_API_KEY =
  readEnvString(import.meta.env.VITE_ORCHESTRATOR_API_KEY) ?? "";

export const DEFAULT_BISQUE_BROWSER_URL =
  readEnvString(import.meta.env.VITE_BISQUE_BROWSER_URL) ?? "";

export const DEFAULT_MAX_TOOL_CALLS = 12;
export const DEFAULT_MAX_RUNTIME_SECONDS = 900;
export const DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS = 7200;
