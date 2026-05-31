export type BisqueNavLinks = {
  home: string;
  datasets: string;
  images: string;
  tables: string;
};

const normalizeBisqueClientServiceBase = (root: string): string => {
  const candidate = String(root || "").trim();
  if (!candidate) {
    return "";
  }

  try {
    const parsed = new URL(candidate);
    const pathname = parsed.pathname.replace(/\/+$/, "");
    const clientServiceIndex = pathname.toLowerCase().indexOf("/client_service");
    const normalizedPath =
      clientServiceIndex >= 0
        ? pathname.slice(0, clientServiceIndex + "/client_service".length)
        : `${pathname === "" || pathname === "/" ? "" : pathname}/client_service`;
    return `${parsed.protocol}//${parsed.host}${normalizedPath}`;
  } catch {
    const withoutQuery = candidate.split(/[?#]/, 1)[0].replace(/\/+$/, "");
    const clientServiceIndex = withoutQuery.toLowerCase().indexOf("/client_service");
    return clientServiceIndex >= 0
      ? withoutQuery.slice(0, clientServiceIndex + "/client_service".length)
      : `${withoutQuery}/client_service`;
  }
};

export const buildBisqueNavLinks = (root: string): BisqueNavLinks => {
  const clientServiceBase = normalizeBisqueClientServiceBase(root);
  return {
    home: `${clientServiceBase}/`,
    datasets: `${clientServiceBase}/browser?resource=/data_service/dataset`,
    images: `${clientServiceBase}/browser?resource=/data_service/image`,
    tables: `${clientServiceBase}/browser?resource=/data_service/table`,
  };
};

export const inferBisqueRootFromUrl = (urlValue: string): string | null => {
  const candidate = String(urlValue || "").trim();
  if (!candidate) {
    return null;
  }
  try {
    const parsed =
      typeof window !== "undefined" && window.location?.origin
        ? new URL(candidate, window.location.origin)
        : new URL(candidate);
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return null;
  }
};
