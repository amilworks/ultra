import type { Hdf5DatasetSummary } from "@/types";

export const formatPathValue = (value: unknown): string => {
  if (value == null) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(" x ");
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
};

export const formatCount = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Not available";
  }
  return Math.max(0, Math.round(value)).toLocaleString();
};

export const formatByteEstimate = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return "Not available";
  }
  if (value < 1024) {
    return `${Math.round(value)} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = value / 1024;
  let unitIndex = 0;
  while (scaled >= 1024 && unitIndex < units.length - 1) {
    scaled /= 1024;
    unitIndex += 1;
  }
  return `${scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2)} ${units[unitIndex]}`;
};

export const formatSampleValue = (value: unknown): string => {
  if (value == null) {
    return "No sample extracted.";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

export const formatRatio = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Not available";
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)}%`;
};

export const formatSummaryToken = (value: string | null | undefined): string => {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "Unknown";
  }
  return normalized
    .split("_")
    .map((part) => {
      if (!part) {
        return part;
      }
      if (part.length <= 3 && /^[a-z0-9]+$/i.test(part)) {
        return part.toUpperCase();
      }
      return part[0].toUpperCase() + part.slice(1);
    })
    .join(" ");
};

export const buildSampleCoverage = (summary: Hdf5DatasetSummary): string | null => {
  const sampleCount = summary.sample_statistics?.sample_count;
  if (typeof sampleCount !== "number" || !Number.isFinite(sampleCount) || summary.element_count <= 0) {
    return null;
  }
  const ratio = (sampleCount / Math.max(1, summary.element_count)) * 100;
  return `${formatCount(sampleCount)} sampled values of ${formatCount(summary.element_count)} (${formatRatio(ratio)})`;
};

export const describeGeometryProvenance = (
  summary: Hdf5DatasetSummary
): { label: string; detail: string } | null => {
  if (!summary.geometry) {
    return null;
  }
  if (summary.geometry.path) {
    return {
      label: "Geometry metadata",
      detail: `Linked through ${summary.geometry.path}. Treat spacing and origin as associated geometry metadata, not a guarantee for every dataset.`,
    };
  }
  return {
    label: "Geometry hint",
    detail: "Derived from available structural metadata. Verify calibration before using it for quantitative interpretation.",
  };
};
