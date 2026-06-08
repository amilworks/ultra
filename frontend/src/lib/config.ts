const readEnvString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const PRODUCTION_BISQUE_ROOT_URL = "https://bisque2.ece.ucsb.edu";

const toBisqueClientServiceUrl = (root: string): string => {
  const normalizedRoot = root.trim().replace(/\/+$/, "");
  if (/\/client_service$/i.test(normalizedRoot)) {
    return `${normalizedRoot}/`;
  }
  return `${normalizedRoot}/client_service/`;
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
  readEnvString(import.meta.env.VITE_BISQUE_BROWSER_URL) ??
  toBisqueClientServiceUrl(
    readEnvString(import.meta.env.VITE_BISQUE_ROOT_URL) ?? PRODUCTION_BISQUE_ROOT_URL
  );
