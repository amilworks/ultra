export type BisqueNavLinks = {
  home: string;
  datasets: string;
  images: string;
  tables: string;
};

export const buildBisqueNavLinks = (root: string): BisqueNavLinks => {
  const trimmedRoot = String(root || "").trim().replace(/\/+$/, "");
  return {
    home: `${trimmedRoot}/client_service/`,
    datasets: `${trimmedRoot}/client_service/browser?resource=/data_service/dataset`,
    images: `${trimmedRoot}/client_service/browser?resource=/data_service/image`,
    tables: `${trimmedRoot}/client_service/browser?resource=/data_service/table`,
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
