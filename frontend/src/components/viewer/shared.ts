import type { UploadViewerInfo } from "@/types";

export type ViewerSurface = "2d" | "mpr" | "volume" | "metadata";

export type ViewerIndices = {
  x: number;
  y: number;
  z: number;
  t: number;
};

const VIEWER_SURFACE_LABELS: Record<ViewerSurface, string> = {
  "2d": "2D",
  mpr: "Slice Views",
  volume: "Volume",
  metadata: "Metadata",
};

export const clampViewerIndex = (value: number, size: number, fallback = 0): number => {
  if (size <= 0) {
    return 0;
  }
  if (!Number.isFinite(value)) {
    return Math.min(size - 1, Math.max(0, fallback));
  }
  return Math.min(size - 1, Math.max(0, Math.round(value)));
};

export const getPlaneDescriptor = (
  viewerInfo: UploadViewerInfo,
  axis: "z" | "y" | "x"
): UploadViewerInfo["viewer"]["default_plane"] => {
  return viewerInfo.viewer.planes[axis] ?? viewerInfo.viewer.default_plane;
};

export const formatViewerSurfaceLabel = (surface: string): string => {
  const normalized = String(surface || "").toLowerCase() as ViewerSurface;
  return VIEWER_SURFACE_LABELS[normalized] ?? (String(surface || "").trim() || "View");
};

const fallbackAxisLabels = {
  x: { positive: "X", negative: "-X" },
  y: { positive: "Y", negative: "-Y" },
  z: { positive: "Z", negative: "-Z" },
};

const normalizeAxisKey = (value: string): "x" | "y" | "z" => {
  const safe = String(value || "").trim().toLowerCase();
  if (safe.endsWith("x")) {
    return "x";
  }
  if (safe.endsWith("y")) {
    return "y";
  }
  return "z";
};

export const getPlaneOrientationLabels = (
  viewerInfo: UploadViewerInfo,
  axis: "z" | "y" | "x"
): { top: string; bottom: string; left: string; right: string } => {
  const plane = getPlaneDescriptor(viewerInfo, axis);
  const orientation = viewerInfo.viewer.orientation;
  const axisLabels = orientation?.axis_labels ?? fallbackAxisLabels;
  const rowKey = normalizeAxisKey(plane.axes[0] ?? "Y");
  const colKey = normalizeAxisKey(plane.axes[1] ?? "X");
  const row = axisLabels[rowKey] ?? fallbackAxisLabels[rowKey];
  const col = axisLabels[colKey] ?? fallbackAxisLabels[colKey];
  return {
    top: String(row.negative ?? fallbackAxisLabels[rowKey].negative),
    bottom: String(row.positive ?? fallbackAxisLabels[rowKey].positive),
    left: String(col.negative ?? fallbackAxisLabels[colKey].negative),
    right: String(col.positive ?? fallbackAxisLabels[colKey].positive),
  };
};

export const buildInitialViewerIndices = (viewerInfo: UploadViewerInfo): ViewerIndices => ({
  x: Math.max(0, Math.floor((Number(viewerInfo.axis_sizes.X ?? 1) - 1) / 2)),
  y: Math.max(0, Math.floor((Number(viewerInfo.axis_sizes.Y ?? 1) - 1) / 2)),
  z: Number(viewerInfo.selected_indices?.Z ?? 0),
  t: Number(viewerInfo.selected_indices?.T ?? 0),
});

export const getPlaneCursor = (
  axis: "z" | "y" | "x",
  indices: ViewerIndices
): { row: number; col: number } => {
  if (axis === "y") {
    return { row: indices.z, col: indices.x };
  }
  if (axis === "x") {
    return { row: indices.z, col: indices.y };
  }
  return { row: indices.y, col: indices.x };
};

export const mapPlanePointToViewerIndices = (
  axis: "z" | "y" | "x",
  point: { row: number; col: number },
  current: ViewerIndices
): ViewerIndices => {
  if (axis === "y") {
    return { ...current, z: point.row, x: point.col };
  }
  if (axis === "x") {
    return { ...current, z: point.row, y: point.col };
  }
  return { ...current, y: point.row, x: point.col };
};
