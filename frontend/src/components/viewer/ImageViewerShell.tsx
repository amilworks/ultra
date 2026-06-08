import { useEffect, useState } from "react";
import { ChevronDown, Focus, RotateCcw, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerHistogramResponse, UploadViewerInfo } from "@/types";

import { DirectPlaneImage } from "./DirectPlaneImage";
import { scalarVolumePayloadValueAt } from "./scalarVolume";
import { SlicePlaneCanvas } from "./SlicePlaneCanvas";
import { SliceStackVolumeCanvas } from "./SliceStackVolumeCanvas";
import {
  formatViewerSurfaceLabel,
  getOrientationSummary,
  getPlaneCursor,
  getPlaneOrientationLabels,
  getSpacingSummary,
  mapPlanePointToViewerIndices,
  type ViewerIndices,
  type ViewerSurface,
} from "./shared";
import { SCALAR_VOLUME_COLOR_MAPS, resolveScalarVolumeColorMap } from "./volumeColorMap";
import { VOLUME_CAMERA_MODES } from "./volumeCameraMode";
import { computePhysicalVolumeGeometry } from "./volumeGeometry";
import { resolveScalarVolumeLighting } from "./volumeLighting";
import { resolveScalarVolumeTransferFunction, type ScalarVolumeTransferFunction } from "./volumeTransferFunction";
import { VOLUME_VIEW_PRESETS } from "./volumeViewPreset";

type ViewerDisplayState = NonNullable<UploadViewerInfo["display_defaults"]>;

type UploadHistogramState = {
  key: string;
  histogram: UploadViewerHistogramResponse | null;
  error: string | null;
};

type ScalarProbeState = {
  key: string;
  volume: ScalarVolumePayload | null;
  error: string | null;
};

type ImageViewerShellProps = {
  viewerInfo: UploadViewerInfo;
  apiClient: ApiClient;
  selectedSurface: ViewerSurface;
  onSurfaceChange: (surface: string) => void;
  selectedDisplayState: ViewerDisplayState | null;
  updateSelectedDisplay: (patch: Partial<ViewerDisplayState>) => void;
  clampedIndices: ViewerIndices;
  debouncedX: number;
  debouncedY: number;
  debouncedZ: number;
  debouncedT: number;
  xAxisSize: number;
  yAxisSize: number;
  zAxisSize: number;
  tAxisSize: number;
  setSelectedIndex: (axis: keyof ViewerIndices, value: number) => void;
  selectedCaption: string;
  captionLoading: boolean;
};

type MetadataCard = {
  label: string;
  value: string;
};

type MetadataSection = {
  title: string;
  rows: Array<{ label: string; value: string }>;
};

type PlaneAxis = "z" | "y" | "x";

type PlanePoint = {
  row: number;
  col: number;
};

type PlaneMeasurement = {
  start: PlanePoint;
  end: PlanePoint;
};

type MeasurementDraft =
  | {
      axis: PlaneAxis;
      start: PlanePoint;
    }
  | null;

const MIN_CLIP_SPAN = 0.02;
const INTERIOR_FOCUS_CLIP_SPAN = 0.56;
const SCALAR_VOLUME_TRANSFER_PRESETS = [
  { id: "custom", label: "Custom", signalFloor: null, densityScale: null },
  { id: "full", label: "Full range", signalFloor: 0, densityScale: 1 },
  { id: "soft", label: "Soft contrast", signalFloor: 0.15, densityScale: 1.2 },
  { id: "crisp", label: "Crisp structures", signalFloor: 0.35, densityScale: 1.4 },
] as const;

type ScalarVolumeTransferPresetId = (typeof SCALAR_VOLUME_TRANSFER_PRESETS)[number]["id"];

const formatNumber = (value: number): string => value.toLocaleString();

const formatPixelType = (dtype: string, bitDepth?: number | null): string => {
  const normalized = String(dtype || "").trim();
  const sourceBits = /^(?:u?int|float)(\d+)$/i.exec(normalized)?.[1];
  const parsedSourceBits = sourceBits ? Number(sourceBits) : null;
  if (
    typeof bitDepth === "number" &&
    Number.isFinite(bitDepth) &&
    bitDepth > 0 &&
    (parsedSourceBits == null || parsedSourceBits === bitDepth)
  ) {
    return `${normalized} • ${bitDepth}-bit`;
  }
  return normalized || "unknown";
};

const formatIntensityValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 100 || Number.isInteger(value)) {
    return value.toFixed(0);
  }
  return value.toFixed(2);
};

const roundClipFraction = (value: number): number => Number(value.toFixed(4));

const centeredClipAxisBounds = ({
  count,
  index,
  span = INTERIOR_FOCUS_CLIP_SPAN,
}: {
  count: number;
  index: number;
  span?: number;
}): { min: number; max: number } => {
  const safeCount = Math.max(1, Math.floor(Number(count) || 1));
  const safeSpan = Math.max(MIN_CLIP_SPAN, Math.min(1, Number(span) || INTERIOR_FOCUS_CLIP_SPAN));
  const clampedIndex = Math.max(0, Math.min(safeCount - 1, Math.floor(Number(index) || 0)));
  const center = (clampedIndex + 0.5) / safeCount;
  let min = center - safeSpan / 2;
  let max = center + safeSpan / 2;
  if (min < 0) {
    max = Math.min(1, max - min);
    min = 0;
  }
  if (max > 1) {
    min = Math.max(0, min - (max - 1));
    max = 1;
  }
  return {
    min: roundClipFraction(min),
    max: roundClipFraction(max),
  };
};

export function buildInteriorVolumeClipBounds({
  axisSizes,
  indices,
  span = INTERIOR_FOCUS_CLIP_SPAN,
}: {
  axisSizes: UploadViewerInfo["axis_sizes"];
  indices: ViewerIndices;
  span?: number;
}): {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
} {
  const x = centeredClipAxisBounds({ count: axisSizes.X, index: indices.x, span });
  const y = centeredClipAxisBounds({ count: axisSizes.Y, index: indices.y, span });
  const z = centeredClipAxisBounds({ count: axisSizes.Z, index: indices.z, span });
  return {
    min: { x: x.min, y: y.min, z: z.min },
    max: { x: x.max, y: y.max, z: z.max },
  };
}

const isVolumeClipActive = ({
  min,
  max,
}: {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}): boolean =>
  Math.abs(min.x) > 0.0001 ||
  Math.abs(min.y) > 0.0001 ||
  Math.abs(min.z) > 0.0001 ||
  Math.abs(1 - max.x) > 0.0001 ||
  Math.abs(1 - max.y) > 0.0001 ||
  Math.abs(1 - max.z) > 0.0001;

const getScalarVolumeTransferPresetId = (
  transfer: ScalarVolumeTransferFunction
): ScalarVolumeTransferPresetId => {
  const close = (left: number, right: number) => Math.abs(left - right) < 0.005;
  const preset = SCALAR_VOLUME_TRANSFER_PRESETS.find(
    (candidate) =>
      candidate.signalFloor != null &&
      candidate.densityScale != null &&
      close(transfer.signalFloor, candidate.signalFloor) &&
      close(transfer.densityScale, candidate.densityScale)
  );
  return preset?.id ?? "custom";
};

const histogramSampleLabel = (histogram: UploadViewerHistogramResponse | null): string => {
  if (!histogram) {
    return "";
  }
  const dtype = histogram.dtype ? `${histogram.dtype} • ` : "";
  const samples =
    typeof histogram.sample_count === "number" && Number.isFinite(histogram.sample_count)
      ? `${formatNumber(histogram.sample_count)} samples`
      : "sampled";
  return `${dtype}${samples}`;
};

const formatJsonishValue = (value: unknown): string => {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => formatJsonishValue(item))
      .filter(Boolean)
      .join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
};

const recordToRows = (record: Record<string, unknown> | null | undefined) =>
  Object.entries(record ?? {})
    .map(([label, value]) => ({
      label,
      value: formatJsonishValue(value),
    }))
    .filter((row) => row.value);

const splitMetadataSummaryRows = (
  cards: MetadataCard[]
): { primary: MetadataCard[]; secondary: MetadataCard[] } => {
  const primaryLabels = new Set(["Shape", "Axes", "Type", "Spacing"]);
  const primary = cards.filter((card) => primaryLabels.has(card.label));
  const secondary = cards.filter((card) => !primaryLabels.has(card.label));
  return {
    primary: primary.length > 0 ? primary : cards.slice(0, 4),
    secondary: primary.length > 0 ? secondary : cards.slice(4),
  };
};

const clampPoint = (
  point: PlanePoint,
  descriptor: UploadViewerInfo["viewer"]["default_plane"]
): PlanePoint => ({
  row: Math.max(0, Math.min(Math.round(point.row), Math.max(0, descriptor.pixel_size.height - 1))),
  col: Math.max(0, Math.min(Math.round(point.col), Math.max(0, descriptor.pixel_size.width - 1))),
});

const parseWindowLevel = (
  enhancement: string | undefined,
  fallbackCenter: number,
  fallbackWidth: number
): { center: number; width: number } => {
  const safe = String(enhancement || "");
  if (safe.startsWith("hounsfield:")) {
    const parts = safe.split(":");
    const center = Number(parts[1]);
    const width = Number(parts[2]);
    return {
      center: Number.isFinite(center) ? center : fallbackCenter,
      width: Number.isFinite(width) && width > 0 ? width : fallbackWidth,
    };
  }
  return { center: fallbackCenter, width: fallbackWidth };
};

const buildWindowEnhancement = (center: number, width: number): string =>
  `hounsfield:${center.toFixed(3)}:${Math.max(1, width).toFixed(3)}`;

const hexColorOrDefault = (value: string | undefined, fallback: string): string => {
  const safe = String(value || "").trim();
  return /^#?[0-9a-fA-F]{6}$/.test(safe) ? (safe.startsWith("#") ? safe : `#${safe}`) : fallback;
};

const clampUnitInterval = (value: number, fallback: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(0, Math.min(1, numeric));
};

const normalizeClipBounds = (
  displayState: ViewerDisplayState | null | undefined
): { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } => {
  const rawMin = displayState?.volume_clip_min ?? { x: 0, y: 0, z: 0 };
  const rawMax = displayState?.volume_clip_max ?? { x: 1, y: 1, z: 1 };
  const outputMin = {
    x: clampUnitInterval(rawMin.x, 0),
    y: clampUnitInterval(rawMin.y, 0),
    z: clampUnitInterval(rawMin.z, 0),
  };
  const outputMax = {
    x: clampUnitInterval(rawMax.x, 1),
    y: clampUnitInterval(rawMax.y, 1),
    z: clampUnitInterval(rawMax.z, 1),
  };
  (["x", "y", "z"] as const).forEach((axis) => {
    if (outputMax[axis] - outputMin[axis] < MIN_CLIP_SPAN) {
      if (outputMin[axis] <= 1 - MIN_CLIP_SPAN) {
        outputMax[axis] = Math.min(1, outputMin[axis] + MIN_CLIP_SPAN);
      } else {
        outputMin[axis] = Math.max(0, outputMax[axis] - MIN_CLIP_SPAN);
      }
    }
  });
  return { min: outputMin, max: outputMax };
};

const getSpatialUnit = (viewerInfo: UploadViewerInfo): string | null => {
  const coordinates = viewerInfo.phys?.coordinates;
  if (!coordinates || typeof coordinates !== "object") {
    return null;
  }
  const units = (coordinates as Record<string, unknown>).space_units;
  if (!units || typeof units !== "object") {
    return null;
  }
  const spatial = (units as Record<string, unknown>).spatial;
  return typeof spatial === "string" && spatial.trim() ? spatial.trim() : null;
};

const measurementUnitLabel = (viewerInfo: UploadViewerInfo): string => {
  if (viewerInfo.viewer.measurement_policy === "orientation-aware") {
    return getSpatialUnit(viewerInfo) ?? "vox";
  }
  if (viewerInfo.viewer.measurement_policy === "spacing-aware") {
    return "vox";
  }
  return viewerInfo.is_volume ? "vox" : "px";
};

const formatDistance = (value: number, unit: string): string => {
  if (!Number.isFinite(value)) {
    return `0 ${unit}`;
  }
  const rounded = value >= 100 ? value.toFixed(1) : value >= 10 ? value.toFixed(2) : value.toFixed(3);
  return `${rounded} ${unit}`;
};

const computeMeasurementDistance = (
  measurement: PlaneMeasurement,
  descriptor: UploadViewerInfo["viewer"]["default_plane"],
  viewerInfo: UploadViewerInfo
): string => {
  const rowDelta = Math.abs(measurement.end.row - measurement.start.row);
  const colDelta = Math.abs(measurement.end.col - measurement.start.col);
  const usePhysicalScale = viewerInfo.viewer.measurement_policy !== "pixel-only";
  const rowScale = usePhysicalScale ? Number(descriptor.spacing.row || 1) : 1;
  const colScale = usePhysicalScale ? Number(descriptor.spacing.col || 1) : 1;
  const distance = Math.sqrt((rowDelta * rowScale) ** 2 + (colDelta * colScale) ** 2);
  return formatDistance(distance, measurementUnitLabel(viewerInfo));
};

const computeCursorWorldPosition = (
  viewerInfo: UploadViewerInfo,
  indices: ViewerIndices
): Array<{ label: string; value: string }> => {
  const output: Array<{ label: string; value: string }> = [
    {
      label: viewerInfo.is_volume ? "Voxel" : "Pixel",
      value: `x=${indices.x}, y=${indices.y}${viewerInfo.is_volume ? `, z=${indices.z}` : ""}, t=${indices.t}`,
    },
  ];
  const coordinates = viewerInfo.phys?.coordinates;
  if (!coordinates || typeof coordinates !== "object") {
    return output;
  }
  const affine = (coordinates as Record<string, unknown>).affine;
  const axisCodes = Array.isArray((coordinates as Record<string, unknown>).axis_codes)
    ? ((coordinates as Record<string, unknown>).axis_codes as unknown[]).map((value) => String(value))
    : [];
  if (!Array.isArray(affine) || affine.length < 3) {
    return output;
  }
  const matrix = affine
    .map((row) => (Array.isArray(row) ? row.map((value) => Number(value)) : []))
    .filter((row) => row.length >= 4);
  if (matrix.length < 3) {
    return output;
  }
  const voxel = [indices.x, indices.y, indices.z, 1];
  const world = matrix.slice(0, 3).map((row) =>
    row.reduce((sum, value, index) => sum + value * (voxel[index] ?? 0), 0)
  );
  if (!world.every((value) => Number.isFinite(value))) {
    return output;
  }
  const unit = getSpatialUnit(viewerInfo) ?? "units";
  output.push({
    label: "Position",
    value: world
      .map((value, index) => `${axisCodes[index] ?? ["X", "Y", "Z"][index]}=${value.toFixed(2)} ${unit}`)
      .join(" • "),
  });
  return output;
};

export function ImageViewerShell({
  viewerInfo,
  apiClient,
  selectedSurface,
  onSurfaceChange,
  selectedDisplayState,
  updateSelectedDisplay,
  clampedIndices,
  debouncedX,
  debouncedY,
  debouncedZ,
  debouncedT,
  xAxisSize,
  yAxisSize,
  zAxisSize,
  tAxisSize,
  setSelectedIndex,
}: ImageViewerShellProps) {
  const [measurementMode, setMeasurementMode] = useState(false);
  const [measurementDraft, setMeasurementDraft] = useState<MeasurementDraft>(null);
  const [measurementsByAxis, setMeasurementsByAxis] = useState<Partial<Record<PlaneAxis, PlaneMeasurement>>>({});
  const [activeMeasurementAxis, setActiveMeasurementAxis] = useState<PlaneAxis>("z");
  const [advancedControlsOpen, setAdvancedControlsOpen] = useState(false);
  const [metadataDetailsOpen, setMetadataDetailsOpen] = useState(false);
  const [uploadHistogramState, setUploadHistogramState] = useState<UploadHistogramState>({
    key: "",
    histogram: null,
    error: null,
  });
  const [scalarProbeState, setScalarProbeState] = useState<ScalarProbeState>({
    key: "",
    volume: null,
    error: null,
  });

  const metadataSummaryCards: MetadataCard[] = (() => {
    const cards: MetadataCard[] = [];
    const spacing = getSpacingSummary(viewerInfo);
    const orientation = getOrientationSummary(viewerInfo);
    const bitDepth = viewerInfo.phys?.pixel_depth;
    cards.push({ label: "Shape", value: viewerInfo.metadata.array_shape.join(" × ") });
    cards.push({ label: "Axes", value: viewerInfo.metadata.dims_order || viewerInfo.dims_order });
    cards.push({
      label: "Type",
      value: formatPixelType(viewerInfo.metadata.array_dtype, bitDepth),
    });
    if (spacing) {
      cards.push({ label: "Spacing", value: spacing });
    }
    if (orientation) {
      cards.push({ label: "Orientation", value: orientation });
    }
    if (viewerInfo.axis_sizes.C > 1) {
      cards.push({ label: "Channels", value: `${formatNumber(viewerInfo.axis_sizes.C)}` });
    }
    if (viewerInfo.axis_sizes.T > 1) {
      cards.push({ label: "Timepoints", value: `${formatNumber(viewerInfo.axis_sizes.T)}` });
    }
    if (viewerInfo.metadata.scene || viewerInfo.metadata.scene_count > 1) {
      cards.push({
        label: "Scene",
        value: viewerInfo.metadata.scene
          ? `${viewerInfo.metadata.scene}${
              viewerInfo.metadata.scene_count > 1 ? ` • ${viewerInfo.metadata.scene_count} scenes` : ""
            }`
          : `${viewerInfo.metadata.scene_count} scenes`,
      });
    }
    return cards;
  })();

  const metadataSections: MetadataSection[] = (() => {
    const sections: MetadataSection[] = [];
    const headerRows = recordToRows(viewerInfo.metadata.header);
    if (headerRows.length > 0) {
      sections.push({ title: "Image Header", rows: headerRows });
    }
    const exifRows = recordToRows(viewerInfo.metadata.exif);
    if (exifRows.length > 0) {
      sections.push({ title: "EXIF Tags", rows: exifRows });
    }
    const dicomRows = [
      viewerInfo.metadata.dicom?.modality
        ? { label: "Modality", value: viewerInfo.metadata.dicom.modality }
        : null,
      typeof viewerInfo.metadata.dicom?.wnd_center === "number"
        ? { label: "Window center", value: String(viewerInfo.metadata.dicom.wnd_center) }
        : null,
      typeof viewerInfo.metadata.dicom?.wnd_width === "number"
        ? { label: "Window width", value: String(viewerInfo.metadata.dicom.wnd_width) }
        : null,
    ].filter(Boolean) as MetadataSection["rows"];
    if (dicomRows.length > 0) {
      sections.push({ title: "DICOM Header", rows: dicomRows });
    }
    const geoRows = recordToRows(viewerInfo.metadata.geo);
    if (geoRows.length > 0) {
      sections.push({ title: "Geospatial Metadata", rows: geoRows });
    }
    const microscopyRows = [
      viewerInfo.metadata.microscopy?.channel_names?.length
        ? { label: "Channel names", value: viewerInfo.metadata.microscopy.channel_names.join(", ") }
        : null,
      viewerInfo.metadata.microscopy?.dimensions_present
        ? { label: "Dimensions", value: viewerInfo.metadata.microscopy.dimensions_present }
        : null,
      viewerInfo.metadata.microscopy?.objective
        ? { label: "Objective", value: viewerInfo.metadata.microscopy.objective }
        : null,
      viewerInfo.metadata.microscopy?.imaging_datetime
        ? { label: "Acquired", value: viewerInfo.metadata.microscopy.imaging_datetime }
        : null,
      viewerInfo.metadata.microscopy?.binning
        ? { label: "Binning", value: viewerInfo.metadata.microscopy.binning }
        : null,
      viewerInfo.metadata.microscopy?.position_index != null
        ? { label: "Position index", value: String(viewerInfo.metadata.microscopy.position_index) }
        : null,
      viewerInfo.metadata.microscopy?.row != null
        ? { label: "Row", value: String(viewerInfo.metadata.microscopy.row) }
        : null,
      viewerInfo.metadata.microscopy?.column != null
        ? { label: "Column", value: String(viewerInfo.metadata.microscopy.column) }
        : null,
    ].filter(Boolean) as MetadataSection["rows"];
    if (microscopyRows.length > 0) {
      sections.push({ title: "Microscopy Metadata", rows: microscopyRows });
    }
    const orientationRows = [
      viewerInfo.viewer.orientation?.frame
        ? { label: "Frame", value: String(viewerInfo.viewer.orientation.frame) }
        : null,
      viewerInfo.viewer.orientation?.axis_labels?.x
        ? {
            label: "X axis",
            value: `${viewerInfo.viewer.orientation.axis_labels.x.negative ?? "-X"} ↔ ${viewerInfo.viewer.orientation.axis_labels.x.positive ?? "X"}`,
          }
        : null,
      viewerInfo.viewer.orientation?.axis_labels?.y
        ? {
            label: "Y axis",
            value: `${viewerInfo.viewer.orientation.axis_labels.y.negative ?? "-Y"} ↔ ${viewerInfo.viewer.orientation.axis_labels.y.positive ?? "Y"}`,
          }
        : null,
      viewerInfo.is_volume && viewerInfo.viewer.orientation?.axis_labels?.z
        ? {
            label: "Z axis",
            value: `${viewerInfo.viewer.orientation.axis_labels.z.negative ?? "-Z"} ↔ ${viewerInfo.viewer.orientation.axis_labels.z.positive ?? "Z"}`,
          }
        : null,
    ].filter(Boolean) as MetadataSection["rows"];
    if (orientationRows.length > 0) {
      sections.push({ title: "Orientation", rows: orientationRows });
    }
    const coordinates = viewerInfo.phys?.coordinates ?? null;
    const coordinateRows = [
      typeof coordinates?.space === "string" ? { label: "Space", value: coordinates.space } : null,
      Array.isArray(coordinates?.axis_codes)
        ? { label: "Axis codes", value: coordinates.axis_codes.join(" / ") }
        : null,
      coordinates?.space_units && typeof coordinates.space_units === "object"
        ? { label: "Units", value: formatJsonishValue(coordinates.space_units) }
        : null,
    ].filter(Boolean) as MetadataSection["rows"];
    if (coordinateRows.length > 0) {
      sections.push({ title: "Coordinate Transform", rows: coordinateRows });
    }
    return sections;
  })();
  const metadataSummaryRows = splitMetadataSummaryRows(metadataSummaryCards);

  const displayCapabilities = new Set((viewerInfo.viewer.display_capabilities ?? []).map((value) => String(value)));
  const selectedChannelIndices = Array.isArray(selectedDisplayState?.channels)
    ? selectedDisplayState.channels.filter((value) => Number.isFinite(value)).map((value) => Math.max(0, Math.floor(value)))
    : [];
  const selectedChannelKey = selectedChannelIndices.join(",");
  const selectedChannelColorKey = (selectedDisplayState?.channel_colors ?? []).map((value) => String(value || "").trim()).join(",");
  const hasMultipleChannels = Boolean(viewerInfo.is_multichannel) || viewerInfo.axis_sizes.C > 1;
  const canControlChannels =
    hasMultipleChannels ||
    displayCapabilities.has("channel_visibility") ||
    displayCapabilities.has("channel_mix") ||
    displayCapabilities.has("channel_color");
  const canControlChannelColor = displayCapabilities.has("channel_color");
  const canLoadScalarVolumeHistogram =
    viewerInfo.is_volume && viewerInfo.viewer.volume_mode === "scalar";
  const canLoadUploadHistogram =
    Boolean(viewerInfo.service_urls?.histogram) &&
    (displayCapabilities.has("histogram") || displayCapabilities.has("intensity_window")) &&
    (!viewerInfo.is_volume || canLoadScalarVolumeHistogram);
  const uploadHistogramRequestKey = canLoadUploadHistogram
    ? [viewerInfo.file_id, canControlChannels ? selectedChannelKey : "all"].join("\u0000")
    : "";
  const currentUploadHistogramState =
    uploadHistogramState.key === uploadHistogramRequestKey
      ? uploadHistogramState
      : { key: uploadHistogramRequestKey, histogram: null, error: null };
  const uploadHistogram = currentUploadHistogramState.histogram;
  const uploadHistogramError = currentUploadHistogramState.error;
  useEffect(() => {
    if (!uploadHistogramRequestKey || uploadHistogramState.key === uploadHistogramRequestKey) {
      return;
    }
    let active = true;
    const histogramChannels = selectedChannelKey
      ? selectedChannelKey.split(",").map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : [];
    const histogramConfig =
      canControlChannels && histogramChannels.length > 0
        ? { bins: 256, channels: histogramChannels }
        : { bins: 256 };
    apiClient
      .getUploadHistogram(viewerInfo.file_id, histogramConfig)
      .then((response) => {
        if (!active) {
          return;
        }
        setUploadHistogramState({ key: uploadHistogramRequestKey, histogram: response, error: null });
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        setUploadHistogramState({
          key: uploadHistogramRequestKey,
          histogram: null,
          error: error instanceof Error ? error.message : "Histogram unavailable.",
        });
      });
    return () => {
      active = false;
    };
  }, [
    apiClient,
    canControlChannels,
    selectedChannelKey,
    uploadHistogramRequestKey,
    uploadHistogramState.key,
    viewerInfo.file_id,
  ]);
  const isScalarMpr =
    viewerInfo.viewer.render_policy === "scalar" && viewerInfo.viewer.diagnostic_surface === "mpr";
  const canControlScalarVolumeColorMap =
    viewerInfo.viewer.volume_mode === "scalar" && selectedSurface === "volume";
  const scalarVolumeColorMap = resolveScalarVolumeColorMap(selectedDisplayState?.scalar_colormap);
  const scalarVolumeTransfer = resolveScalarVolumeTransferFunction(selectedDisplayState);
  const scalarVolumeTransferPresetId = getScalarVolumeTransferPresetId(scalarVolumeTransfer);
  const scalarVolumeLighting = resolveScalarVolumeLighting(selectedDisplayState);
  const canLoadScalarProbe =
    isScalarMpr &&
    selectedSurface === "mpr" &&
    viewerInfo.viewer.volume_mode === "scalar" &&
    Boolean(viewerInfo.service_urls?.scalar_volume) &&
    displayCapabilities.has("scalar_probe");
  const histogramMin = Number(uploadHistogram?.histogram.min);
  const histogramMax = Number(uploadHistogram?.histogram.max);
  const metadataArrayMin = Number(viewerInfo.metadata.array_min ?? viewerInfo.metadata.intensity_stats?.min ?? 0);
  const metadataArrayMax = Number(viewerInfo.metadata.array_max ?? viewerInfo.metadata.intensity_stats?.max ?? 1);
  const arrayMin = Number.isFinite(histogramMin) ? histogramMin : metadataArrayMin;
  const arrayMax = Number.isFinite(histogramMax) ? histogramMax : metadataArrayMax;
  const defaultCenter = Number(viewerInfo.metadata.dicom?.wnd_center ?? (arrayMin + arrayMax) / 2);
  const defaultWidth = Number(
    viewerInfo.metadata.dicom?.wnd_width ?? Math.max(1, Math.abs(arrayMax - arrayMin))
  );
  const parsedWindow = parseWindowLevel(selectedDisplayState?.enhancement, defaultCenter, defaultWidth);
  const intensityRangeSpan = Math.max(1, Math.abs(arrayMax - arrayMin));
  const intensityStep = intensityRangeSpan <= 16 ? 0.1 : 1;
  const sourceIntensityReady = Boolean(
    selectedDisplayState &&
      uploadHistogram &&
      uploadHistogram.histogram.bins.length > 0 &&
      Number.isFinite(arrayMin) &&
      Number.isFinite(arrayMax)
  );
  const directIntensityReady = Boolean(!viewerInfo.is_volume && sourceIntensityReady);
  const volumeIntensityReady = Boolean(viewerInfo.is_volume && sourceIntensityReady);
  const histogramMaxCount = Math.max(1, ...(uploadHistogram?.histogram.bins ?? [1]));
  const clipBounds = normalizeClipBounds(selectedDisplayState);
  const channelNames =
    viewerInfo.metadata.microscopy?.channel_names?.length || viewerInfo.phys?.channel_names?.length
      ? (viewerInfo.metadata.microscopy?.channel_names ?? viewerInfo.phys?.channel_names ?? []).map((value) =>
          String(value)
        )
      : Array.from({ length: viewerInfo.axis_sizes.C }, (_value, index) => `Channel ${index + 1}`);
  const channelColors = channelNames.map((_, index) =>
    hexColorOrDefault(selectedDisplayState?.channel_colors?.[index], viewerInfo.phys?.channel_colors?.[index]?.hex ?? "#ffffff")
  );
  const isSliceStackVolume = viewerInfo.viewer.volume_mode === "slice_stack";
  const showSliceStack2DControls = Boolean(
    selectedSurface === "2d" && viewerInfo.is_volume && isSliceStackVolume && selectedDisplayState
  );
  const volumeChannelIndex = Math.max(
    0,
    Math.min(
      Number(selectedDisplayState?.volume_channel ?? viewerInfo.selected_indices.C ?? 0),
      Math.max(0, channelNames.length - 1)
    )
  );
  const physicalVolumeGeometry = viewerInfo.is_volume
    ? computePhysicalVolumeGeometry({
        planePixelSize: viewerInfo.viewer.default_plane.pixel_size,
        volumeDepth: viewerInfo.axis_sizes.Z,
        physicalSpacing: viewerInfo.metadata.physical_spacing,
      })
    : null;
  const canShowVolumeGeometry =
    viewerInfo.is_volume && displayCapabilities.has("physical_scale") && physicalVolumeGeometry != null;
  const volumeSummaryRows: MetadataCard[] = (() => {
    if (!viewerInfo.is_volume || !selectedDisplayState) {
      return [];
    }
    if (isSliceStackVolume) {
      const rows: MetadataCard[] = [
        {
          label: "Channel",
          value: channelNames[selectedChannelIndices[0] ?? volumeChannelIndex] ?? `Channel ${volumeChannelIndex + 1}`,
        },
        {
          label: "Z slice",
          value: `${clampedIndices.z + 1}/${zAxisSize}`,
        },
        {
          label: "Stack",
          value: `${viewerInfo.axis_sizes.X} x ${viewerInfo.axis_sizes.Y} x ${viewerInfo.axis_sizes.Z}`,
        },
      ];
      if (canShowVolumeGeometry && physicalVolumeGeometry) {
        rows.push({ label: "Spacing", value: physicalVolumeGeometry.spacingLabel });
        rows.push({
          label: "Sampling",
          value: physicalVolumeGeometry.isAnisotropic ? "Anisotropic voxels" : "Isotropic voxels",
        });
      }
      return rows;
    }
    const rows: MetadataCard[] = [
      {
        label: "Projection",
        value: selectedDisplayState.fusion_method === "m" ? "Maximum intensity" : "Composite",
      },
      isScalarMpr
        ? {
            label: "Window",
            value: `${parsedWindow.center.toFixed(1)} / ${parsedWindow.width.toFixed(1)}`,
          }
        : {
            label: "Enhancement",
            value:
              selectedDisplayState.enhancement === "f"
                ? "Full range"
                : selectedDisplayState.enhancement?.startsWith("hounsfield")
                  ? "Windowed"
                  : "Dynamic",
          },
    ];
    if (canShowVolumeGeometry && physicalVolumeGeometry) {
      rows.push({ label: "Volume", value: physicalVolumeGeometry.aspectLabel });
      rows.push({ label: "Spacing", value: physicalVolumeGeometry.spacingLabel });
      rows.push({
        label: "Sampling",
        value: physicalVolumeGeometry.isAnisotropic ? "Anisotropic voxels" : "Isotropic voxels",
      });
    }
    if (viewerInfo.viewer.volume_mode === "scalar" && channelNames.length > 1) {
      rows.push({ label: "Channel", value: channelNames[volumeChannelIndex] ?? `Channel ${volumeChannelIndex + 1}` });
    }
    return rows;
  })();
  const scalarProbeChannelIndex =
    selectedSurface === "mpr" && selectedChannelIndices.length === 1
      ? Math.max(0, Math.min(selectedChannelIndices[0] ?? 0, Math.max(0, channelNames.length - 1)))
      : volumeChannelIndex;
  const scalarProbeRequestKey = canLoadScalarProbe
    ? [viewerInfo.file_id, debouncedT, scalarProbeChannelIndex].join("\u0000")
    : "";
  const currentScalarProbeState =
    scalarProbeState.key === scalarProbeRequestKey
      ? scalarProbeState
      : { key: scalarProbeRequestKey, volume: null, error: null };
  const scalarProbeVolume = currentScalarProbeState.volume;
  const scalarProbeError = currentScalarProbeState.error;
  useEffect(() => {
    if (!scalarProbeRequestKey || scalarProbeState.key === scalarProbeRequestKey) {
      return;
    }
    let active = true;
    apiClient
      .getUploadScalarVolume(viewerInfo.file_id, {
        t: debouncedT,
        channel: scalarProbeChannelIndex,
      })
      .then((payload) => {
        if (!active) {
          return;
        }
        setScalarProbeState({ key: scalarProbeRequestKey, volume: payload, error: null });
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        setScalarProbeState({
          key: scalarProbeRequestKey,
          volume: null,
          error: error instanceof Error ? error.message : "Scalar probe unavailable.",
        });
      });
    return () => {
      active = false;
    };
  }, [
    apiClient,
    debouncedT,
    scalarProbeChannelIndex,
    scalarProbeRequestKey,
    scalarProbeState.key,
    viewerInfo.file_id,
  ]);
  const displayTransformKey = [
    selectedDisplayState?.enhancement ?? "d",
    selectedDisplayState?.negative ? "negative" : "positive",
    canControlChannels ? selectedChannelKey || "channels-default" : "channels-all",
    canControlChannelColor ? selectedChannelColorKey || "colors-default" : "colors-source",
  ].join(":");
  const previewCacheKey = `windowed-v2:${
    String(viewerInfo.metadata.sha256 ?? viewerInfo.file_id).trim() || viewerInfo.file_id
  }:${displayTransformKey}`;

  const mprSliceUrls = {
    z: apiClient.uploadSliceUrl(viewerInfo.file_id, {
      axis: "z",
      x: debouncedX,
      y: debouncedY,
      z: debouncedZ,
      t: debouncedT,
      enhancement: selectedDisplayState?.enhancement,
      fusionMethod: selectedDisplayState?.fusion_method,
      negative: selectedDisplayState?.negative,
      channels: selectedDisplayState?.channels,
      channelColors: selectedDisplayState?.channel_colors,
      cacheKey: previewCacheKey,
    }),
    y: apiClient.uploadSliceUrl(viewerInfo.file_id, {
      axis: "y",
      x: debouncedX,
      y: debouncedY,
      z: debouncedZ,
      t: debouncedT,
      enhancement: selectedDisplayState?.enhancement,
      fusionMethod: selectedDisplayState?.fusion_method,
      negative: selectedDisplayState?.negative,
      channels: selectedDisplayState?.channels,
      channelColors: selectedDisplayState?.channel_colors,
      cacheKey: previewCacheKey,
    }),
    x: apiClient.uploadSliceUrl(viewerInfo.file_id, {
      axis: "x",
      x: debouncedX,
      y: debouncedY,
      z: debouncedZ,
      t: debouncedT,
      enhancement: selectedDisplayState?.enhancement,
      fusionMethod: selectedDisplayState?.fusion_method,
      negative: selectedDisplayState?.negative,
      channels: selectedDisplayState?.channels,
      channelColors: selectedDisplayState?.channel_colors,
      cacheKey: previewCacheKey,
    }),
  };
  const direct2dSliceUrl = apiClient.uploadSliceUrl(viewerInfo.file_id, {
    axis: "z",
    z: debouncedZ,
    t: debouncedT,
    enhancement: selectedDisplayState?.enhancement,
    fusionMethod: selectedDisplayState?.fusion_method,
    negative: selectedDisplayState?.negative,
    channels: selectedDisplayState?.channels,
    channelColors: selectedDisplayState?.channel_colors,
    fullResolution: true,
    cacheKey: previewCacheKey,
  });
  const direct2dDisplayUrl =
    !viewerInfo.is_volume && viewerInfo.service_urls?.display
      ? apiClient.uploadDisplayUrl(viewerInfo.file_id, viewerInfo.service_urls.display, {
          enhancement: selectedDisplayState?.enhancement,
          negative: selectedDisplayState?.negative,
          channels: canControlChannels ? selectedChannelIndices : undefined,
          channelColors: canControlChannelColor ? selectedDisplayState?.channel_colors : undefined,
          cacheKey: previewCacheKey,
        })
      : null;
  const direct2dImageUrl = direct2dDisplayUrl ?? direct2dSliceUrl;
  const direct2dPreviewUrl = apiClient.uploadPreviewUrl(viewerInfo.file_id);

  const scalarProbeValue =
    canLoadScalarProbe && scalarProbeVolume
      ? scalarVolumePayloadValueAt(scalarProbeVolume, clampedIndices)
      : null;
  const cursorReadoutRows = [
    ...computeCursorWorldPosition(viewerInfo, clampedIndices),
    ...(scalarProbeValue != null
      ? [{ label: "Voxel value", value: formatIntensityValue(scalarProbeValue) }]
      : scalarProbeError
        ? [{ label: "Voxel value", value: "Unavailable" }]
        : []),
  ];
  const activeMeasurementPlaneLabel =
    viewerInfo.viewer.planes[activeMeasurementAxis]?.label ?? activeMeasurementAxis.toUpperCase();
  const metadataOverviewRows = [
    ...metadataSummaryRows.primary.map((row) => ({ ...row, tone: "primary" as const })),
    ...metadataSummaryRows.secondary.map((row) => ({ ...row, tone: "secondary" as const })),
  ];

  const updateVolumeClipEdge = (
    edge: "min" | "max",
    axis: "x" | "y" | "z",
    nextValue: number
  ) => {
    const nextMin = { ...clipBounds.min };
    const nextMax = { ...clipBounds.max };
    if (edge === "min") {
      nextMin[axis] = clampUnitInterval(nextValue, clipBounds.min[axis]);
      nextMax[axis] = Math.max(nextMax[axis], Math.min(1, nextMin[axis] + MIN_CLIP_SPAN));
    } else {
      nextMax[axis] = clampUnitInterval(nextValue, clipBounds.max[axis]);
      nextMin[axis] = Math.min(nextMin[axis], Math.max(0, nextMax[axis] - MIN_CLIP_SPAN));
    }
    updateSelectedDisplay({
      volume_clip_min: nextMin,
      volume_clip_max: nextMax,
    });
  };

  const resetVolumeClip = () => {
    updateSelectedDisplay({
      volume_clip_min: { x: 0, y: 0, z: 0 },
      volume_clip_max: { x: 1, y: 1, z: 1 },
    });
  };

  const focusInteriorVolume = () => {
    const nextClip = buildInteriorVolumeClipBounds({
      axisSizes: viewerInfo.axis_sizes,
      indices: clampedIndices,
    });
    updateSelectedDisplay({
      volume_clip_min: nextClip.min,
      volume_clip_max: nextClip.max,
      volume_camera_mode: "perspective",
    });
  };

  const volumeClipActive = isVolumeClipActive(clipBounds);

  const activeMeasurement = measurementsByAxis[activeMeasurementAxis] ?? null;
  const activeMeasurementDescriptor =
    activeMeasurement != null ? viewerInfo.viewer.planes[activeMeasurementAxis] : null;
  const activeMeasurementDistance =
    activeMeasurement && activeMeasurementDescriptor
      ? computeMeasurementDistance(activeMeasurement, activeMeasurementDescriptor, viewerInfo)
      : null;

  const handlePlaneSelect = (axis: PlaneAxis, point: PlanePoint) => {
    const descriptor = viewerInfo.viewer.planes[axis];
    const clampedPoint = clampPoint(point, descriptor);
    const next = mapPlanePointToViewerIndices(axis, clampedPoint, clampedIndices);
    setSelectedIndex("x", next.x);
    setSelectedIndex("y", next.y);
    setSelectedIndex("z", next.z);
  };

  const handlePlaneMeasure = (axis: PlaneAxis, point: PlanePoint) => {
    const descriptor = viewerInfo.viewer.planes[axis];
    const clampedPoint = clampPoint(point, descriptor);
    setActiveMeasurementAxis(axis);
    setMeasurementsByAxis((previous) => {
      if (!measurementMode) {
        return previous;
      }
      if (!measurementDraft || measurementDraft.axis !== axis) {
        return {
          ...previous,
          [axis]: { start: clampedPoint, end: clampedPoint },
        };
      }
      return {
        ...previous,
        [axis]: {
          start: measurementDraft.start,
          end: clampedPoint,
        },
      };
    });
    if (!measurementMode) {
      return;
    }
    setMeasurementDraft((previous) => {
      if (!previous || previous.axis !== axis) {
        return { axis, start: clampedPoint };
      }
      return null;
    });
  };

  const clearMeasurements = () => {
    setMeasurementDraft(null);
    setMeasurementsByAxis({});
  };

  const handleSurfaceChange = (surface: string) => {
    if (surface === "mpr" || surface === "volume") {
      setAdvancedControlsOpen(false);
    }
    onSurfaceChange(surface);
  };

  const renderChannelControls = () => {
    if (!selectedDisplayState || !canControlChannels || channelNames.length <= 1) {
      return null;
    }
    const resolvedChannelMode = String(
      selectedDisplayState.channel_mode ?? viewerInfo.viewer.channel_mode ?? "",
    ).toLowerCase();
    const singleChannelMode = viewerInfo.viewer.volume_mode === "scalar" || resolvedChannelMode === "single";
    return (
      <div className="viewer-channel-controls" data-viewer-channel-controls="true">
        {channelNames.map((label, index) => {
          const active = selectedChannelIndices.includes(index);
          return (
            <div key={`${label}-${index}`} className="viewer-channel-chip">
              <button
                type="button"
                className={active ? "viewer-channel-toggle is-active" : "viewer-channel-toggle"}
                aria-pressed={active}
                onClick={() => {
                  if (singleChannelMode) {
                    updateSelectedDisplay({
                      channels: [index],
                      volume_channel: index,
                    });
                    return;
                  }
                  const current = new Set(selectedChannelIndices);
                  if (active && current.size > 1) {
                    current.delete(index);
                  } else if (!active) {
                    current.add(index);
                  }
                  const nextChannels = Array.from(current).sort((a, b) => a - b);
                  const patch: Partial<ViewerDisplayState> = { channels: nextChannels };
                  if (viewerInfo.viewer.volume_mode === "scalar" && nextChannels.length === 1) {
                    patch.volume_channel = nextChannels[0];
                  }
                  updateSelectedDisplay(patch);
                }}
              >
                <span
                  className="viewer-channel-swatch"
                  style={{ backgroundColor: channelColors[index] }}
                  aria-hidden="true"
                />
                {label}
              </button>
              {canControlChannelColor ? (
                <input
                  type="color"
                  aria-label={`${label} color`}
                  value={channelColors[index]}
                  onChange={(event) => {
                    const nextColors = [...(selectedDisplayState.channel_colors ?? [])];
                    nextColors[index] = event.target.value;
                    updateSelectedDisplay({ channel_colors: nextColors });
                  }}
                />
              ) : null}
            </div>
          );
        })}
      </div>
    );
  };

  const renderIntensityHistogramPanel = () => (
    <div className="viewer-intensity-histogram-panel">
      <div className="viewer-intensity-histogram" aria-label="Source histogram">
        {uploadHistogram?.histogram.bins.map((count, index) => (
          <span
            key={`histogram-bin-${index}`}
            style={{ height: `${Math.max(4, (count / histogramMaxCount) * 100)}%` }}
          />
        ))}
      </div>
      <strong>
        {formatIntensityValue(arrayMin)}-{formatIntensityValue(arrayMax)}
      </strong>
      <span>{histogramSampleLabel(uploadHistogram)}</span>
    </div>
  );

  const renderVolumeGeometryPanel = () => {
    if (!canShowVolumeGeometry || !physicalVolumeGeometry) {
      return null;
    }
    return (
      <div className="viewer-volume-geometry-panel" data-viewer-volume-geometry="true">
        <span>Physical volume</span>
        <strong>{physicalVolumeGeometry.aspectLabel}</strong>
        <span>Spacing {physicalVolumeGeometry.spacingLabel}</span>
        {physicalVolumeGeometry.isAnisotropic ? <em>Anisotropic voxels</em> : <em>Isotropic voxels</em>}
      </div>
    );
  };

  const renderCompactSurfaceReadout = (rows: MetadataCard[], label: string) => {
    if (rows.length === 0) {
      return null;
    }
    return (
      <section
        className="viewer-volume-readout"
        data-viewer-volume-summary="true"
        data-viewer-volume-readout="compact"
        aria-label={label}
      >
        <dl className="viewer-surface-readout-list viewer-surface-readout-list-compact">
          {rows.map((row) => (
            <div key={row.label} className="viewer-surface-readout-item">
              <dt>{row.label}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>
      </section>
    );
  };

  const hasMprIndexControls =
    selectedSurface === "mpr" && (xAxisSize > 1 || yAxisSize > 1 || zAxisSize > 1 || tAxisSize > 1);
  const has2DIndexControls =
    selectedSurface === "2d" && !showSliceStack2DControls && (zAxisSize > 1 || tAxisSize > 1);

  return (
    <>
      <Tabs value={selectedSurface} onValueChange={handleSurfaceChange} className="viewer-surface-tabs">
        <div className="viewer-surface-toolbar">
          <TabsList className="viewer-surface-list">
            {viewerInfo.viewer.available_surfaces.map((surface) => (
              <TabsTrigger key={surface} value={surface}>
                {formatViewerSurfaceLabel(surface)}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        <TabsContent value="2d" className="viewer-surface-panel">
          <div className={viewerInfo.is_volume ? "viewer-volume-layout viewer-volume-layout-2d" : undefined}>
            {isSliceStackVolume ? renderCompactSurfaceReadout(volumeSummaryRows, "Stack summary") : null}
            <div className="viewer-canvas-shell viewer-canvas-shell-2d">
              <div
                data-viewer-surface="2d"
                data-viewer-backend="direct"
                data-viewer-aspect={viewerInfo.viewer.default_plane.aspect_ratio.toFixed(4)}
              >
                <DirectPlaneImage
                  imageUrl={direct2dImageUrl}
                  placeholderUrl={direct2dPreviewUrl}
                  descriptor={viewerInfo.viewer.default_plane}
                  title="2d-plane"
                  className="viewer-canvas-root"
                  interactive={true}
                  orientationLabels={getPlaneOrientationLabels(viewerInfo, "z")}
                />
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="mpr" className="viewer-surface-panel">
          <div className="viewer-mpr-toolbar" data-viewer-mpr-tools="true">
            <dl className="viewer-mpr-readouts" data-viewer-cursor-readout="true">
              {cursorReadoutRows.map((row) => (
                <div key={row.label} className="viewer-mpr-readout">
                  <dt>{row.label}</dt>
                  <dd>{row.value}</dd>
                </div>
              ))}
            </dl>
            <div className="viewer-mpr-actions" data-viewer-measurement-readout="true">
              <span className="viewer-mpr-plane-readout">{activeMeasurementPlaneLabel}</span>
              {activeMeasurementDistance ? (
                <span className="viewer-mpr-distance-readout">{activeMeasurementDistance}</span>
              ) : null}
              <label className="viewer-inline-control viewer-inline-control-switch viewer-mpr-measure-toggle">
                <span>Measure</span>
                <Switch checked={measurementMode} onCheckedChange={setMeasurementMode} size="sm" />
              </label>
              {(activeMeasurement || measurementDraft) && (
                <Button type="button" size="sm" variant="outline" onClick={clearMeasurements}>
                  Clear
                </Button>
              )}
            </div>
          </div>
          <div className="viewer-mpr-grid">
            {(["z", "y", "x"] as const).map((axis) => (
              <article key={axis} className="viewer-mpr-card">
                <div className="viewer-mpr-header">
                  <span>{viewerInfo.viewer.planes[axis]?.label ?? axis.toUpperCase()}</span>
                  <span>{viewerInfo.viewer.planes[axis]?.axes.join("/")}</span>
                </div>
                <SlicePlaneCanvas
                  imageUrl={mprSliceUrls[axis]}
                  descriptor={viewerInfo.viewer.planes[axis]}
                  title={`${axis}-plane`}
                  className="viewer-canvas-root viewer-mpr-canvas"
                  orientationLabels={getPlaneOrientationLabels(viewerInfo, axis)}
                  crosshair={getPlaneCursor(viewerInfo, axis, clampedIndices)}
                  measurement={measurementsByAxis[axis] ?? null}
                  onSelectPoint={(point) => handlePlaneSelect(axis, point)}
                  onMeasurePoint={(point) => handlePlaneMeasure(axis, point)}
                  measureMode={measurementMode}
                />
              </article>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="volume" className="viewer-surface-panel">
          <div className="viewer-volume-layout">
            {renderCompactSurfaceReadout(volumeSummaryRows, "Volume summary")}
            {selectedDisplayState ? (
              <div
                className="viewer-volume-inspection-toolbar"
                data-viewer-volume-inspection-toolbar="true"
                data-viewer-interior-active={volumeClipActive ? "true" : "false"}
              >
                <div className="viewer-volume-inspection-status">
                  {volumeClipActive ? "Interior cutaway active" : "Full volume"}
                </div>
                <div className="viewer-volume-inspection-actions">
                  <Button
                    type="button"
                    variant={volumeClipActive ? "secondary" : "outline"}
                    size="sm"
                    onClick={focusInteriorVolume}
                    aria-pressed={volumeClipActive}
                  >
                    <Focus data-icon="inline-start" />
                    Interior focus
                  </Button>
                  {volumeClipActive ? (
                    <Button type="button" variant="ghost" size="sm" onClick={resetVolumeClip}>
                      <RotateCcw data-icon="inline-start" />
                      Full volume
                    </Button>
                  ) : null}
                </div>
              </div>
            ) : null}
            <div className="viewer-canvas-shell viewer-canvas-shell-volume">
              <SliceStackVolumeCanvas
                apiClient={apiClient}
                fileId={viewerInfo.file_id}
                viewerInfo={viewerInfo}
                xIndex={debouncedX}
                yIndex={debouncedY}
                zIndex={debouncedZ}
                tIndex={debouncedT}
                displayState={selectedDisplayState}
              />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="metadata" className="viewer-surface-panel">
          <section
            className="viewer-metadata-overview"
            data-viewer-metadata-summary="true"
            data-viewer-metadata-layout="facts"
            aria-label="Metadata at a glance"
          >
            <dl className="viewer-metadata-overview-list">
              {metadataOverviewRows.map((row) => (
                <div
                  key={row.label}
                  className={
                    row.tone === "secondary"
                      ? "viewer-metadata-overview-row viewer-metadata-overview-row-secondary"
                      : "viewer-metadata-overview-row"
                  }
                >
                  <dt>{row.label}</dt>
                  <dd>{row.value}</dd>
                </div>
              ))}
            </dl>
          </section>
          {viewerInfo.metadata.warnings.length > 0 ? (
            <div className="viewer-metadata-note">
              <strong>Viewer notes</strong>
              <span>{viewerInfo.metadata.warnings.join(" ")}</span>
            </div>
          ) : null}
          {metadataSections.length > 0 ? (
            <Collapsible
              open={metadataDetailsOpen}
              onOpenChange={setMetadataDetailsOpen}
              className="viewer-metadata-details"
              data-viewer-metadata-details="true"
            >
              <div className="viewer-advanced-header viewer-metadata-details-header">
                <CollapsibleTrigger asChild>
                  <Button type="button" variant="outline" size="sm" className="viewer-advanced-trigger">
                    <SlidersHorizontal data-icon="inline-start" />
                    Technical details
                    <ChevronDown data-icon="inline-end" />
                  </Button>
                </CollapsibleTrigger>
              </div>
              <CollapsibleContent data-viewer-metadata-raw="true">
                <div className="viewer-metadata-grid viewer-metadata-grid-raw">
                  {metadataSections.map((section) => (
                    <section key={section.title} className="viewer-metadata-card viewer-metadata-card-wide">
                      <strong>{section.title}</strong>
                      <dl className="viewer-metadata-list">
                        {section.rows.map((row) => (
                          <div key={`${section.title}-${row.label}`} className="viewer-metadata-row">
                            <dt>{row.label}</dt>
                            <dd>{row.value}</dd>
                          </div>
                        ))}
                      </dl>
                    </section>
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          ) : null}
        </TabsContent>
      </Tabs>

      {!viewerInfo.is_volume && selectedDisplayState && canControlChannels && channelNames.length > 1 ? (
        <div
          className="viewer-display-controls viewer-display-controls-direct viewer-display-controls-channels"
          data-viewer-direct-channel-controls="true"
        >
          {renderChannelControls()}
        </div>
      ) : null}

      {showSliceStack2DControls ? (
        <div
          className="viewer-display-controls viewer-display-controls-direct viewer-display-controls-stack"
          data-viewer-stack-controls="true"
        >
          {zAxisSize > 1 ? (
            <label className="viewer-inline-control viewer-inline-control-wide">
              <span>Z slice</span>
              <input
                type="range"
                aria-label="Z slice"
                min={0}
                max={Math.max(0, zAxisSize - 1)}
                value={clampedIndices.z}
                onChange={(event) => setSelectedIndex("z", Number(event.target.value))}
              />
              <strong>
                {clampedIndices.z + 1}/{zAxisSize}
              </strong>
            </label>
          ) : null}
          {tAxisSize > 1 ? (
            <label className="viewer-inline-control viewer-inline-control-wide">
              <span>Time</span>
              <input
                type="range"
                aria-label="Time"
                min={0}
                max={Math.max(0, tAxisSize - 1)}
                value={clampedIndices.t}
                onChange={(event) => setSelectedIndex("t", Number(event.target.value))}
              />
              <strong>
                {clampedIndices.t + 1}/{tAxisSize}
              </strong>
            </label>
          ) : null}
          {renderChannelControls()}
        </div>
      ) : null}

      {viewerInfo.is_volume && selectedDisplayState && (selectedSurface === "mpr" || selectedSurface === "volume") ? (
        <Collapsible
          open={advancedControlsOpen}
          onOpenChange={setAdvancedControlsOpen}
          className="viewer-advanced-controls"
          data-viewer-advanced-controls="true"
        >
          <div className="viewer-advanced-header">
            <CollapsibleTrigger asChild>
              <Button type="button" variant="outline" size="sm" className="viewer-advanced-trigger">
                <SlidersHorizontal data-icon="inline-start" />
                Advanced rendering
                <ChevronDown data-icon="inline-end" />
              </Button>
            </CollapsibleTrigger>
          </div>
          <CollapsibleContent data-viewer-advanced-content="true">
            <div className="viewer-display-controls viewer-display-controls-volume">
              {selectedSurface === "mpr" ? renderVolumeGeometryPanel() : null}
              {volumeIntensityReady ? (
                <div className="viewer-volume-intensity-panel" data-viewer-volume-intensity-summary="true">
                  {renderIntensityHistogramPanel()}
                </div>
              ) : null}
              {isScalarMpr ? (
                <>
                  <label className="viewer-inline-control">
                    <span>Window level</span>
                    <input
                      type="range"
                      min={Math.floor(arrayMin)}
                      max={Math.ceil(arrayMax)}
                      step="1"
                      value={parsedWindow.center}
                      onChange={(event) =>
                        updateSelectedDisplay({
                          enhancement: buildWindowEnhancement(Number(event.target.value), parsedWindow.width),
                        })
                      }
                    />
                    <strong>{parsedWindow.center.toFixed(1)}</strong>
                  </label>
                  <label className="viewer-inline-control">
                    <span>Window width</span>
                    <input
                      type="range"
                      min={1}
                      max={Math.max(1, Math.ceil(Math.abs(arrayMax - arrayMin)))}
                      step="1"
                      value={parsedWindow.width}
                      onChange={(event) =>
                        updateSelectedDisplay({
                          enhancement: buildWindowEnhancement(parsedWindow.center, Number(event.target.value)),
                        })
                      }
                    />
                    <strong>{parsedWindow.width.toFixed(1)}</strong>
                  </label>
                </>
              ) : (
                <div className="viewer-inline-control">
                  <span>Enhancement</span>
                  <Select
                    value={selectedDisplayState.enhancement}
                    onValueChange={(value) => updateSelectedDisplay({ enhancement: value })}
                  >
                    <SelectTrigger aria-label="Enhancement" className="viewer-select-trigger">
                      <SelectValue placeholder="Enhancement" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="d">Dynamic</SelectItem>
                        <SelectItem value="f">Full range</SelectItem>
                        {viewerInfo.display_defaults?.enhancement?.startsWith("hounsfield") ? (
                          <SelectItem value={viewerInfo.display_defaults.enhancement}>
                            DICOM window
                          </SelectItem>
                        ) : null}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div className="viewer-inline-control">
                <span>{selectedSurface === "volume" ? "Projection" : "Fusion"}</span>
                <Select
                  value={selectedDisplayState.fusion_method}
                  onValueChange={(value) =>
                    updateSelectedDisplay({
                      fusion_method: value as ViewerDisplayState["fusion_method"],
                    })
                  }
                >
                  <SelectTrigger
                    aria-label={selectedSurface === "volume" ? "Projection" : "Fusion"}
                    className="viewer-select-trigger"
                  >
                    <SelectValue placeholder={selectedSurface === "volume" ? "Projection" : "Fusion"} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectItem value="m">
                        {selectedSurface === "volume" ? "MIP" : "Maximum"}
                      </SelectItem>
                      <SelectItem value="a">
                        {selectedSurface === "volume" ? "Composite" : "Average"}
                      </SelectItem>
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              {selectedSurface === "volume" ? (
                <div className="viewer-inline-control" data-viewer-volume-view-control="true">
                  <span>Volume view</span>
                  <Select
                    value={selectedDisplayState.volume_view_preset ?? "iso"}
                    onValueChange={(value) =>
                      updateSelectedDisplay({
                        volume_view_preset: value as ViewerDisplayState["volume_view_preset"],
                      })
                    }
                  >
                    <SelectTrigger aria-label="Volume view" className="viewer-select-trigger">
                      <SelectValue placeholder="Volume view" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {VOLUME_VIEW_PRESETS.map((preset) => (
                          <SelectItem key={preset.id} value={preset.id}>
                            {preset.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
              {selectedSurface === "volume" ? (
                <div className="viewer-inline-control" data-viewer-camera-mode-control="true">
                  <span>Camera</span>
                  <Select
                    value={selectedDisplayState.volume_camera_mode ?? "perspective"}
                    onValueChange={(value) =>
                      updateSelectedDisplay({
                        volume_camera_mode: value as ViewerDisplayState["volume_camera_mode"],
                      })
                    }
                  >
                    <SelectTrigger aria-label="Camera" className="viewer-select-trigger">
                      <SelectValue placeholder="Camera" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {VOLUME_CAMERA_MODES.map((mode) => (
                          <SelectItem key={mode.id} value={mode.id}>
                            {mode.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
              {canControlScalarVolumeColorMap ? (
                <div className="viewer-inline-control" data-viewer-scalar-colormap-control="true">
                  <span>Scalar colormap</span>
                  <Select
                    value={scalarVolumeColorMap.id}
                    onValueChange={(value) =>
                      updateSelectedDisplay({
                        scalar_colormap: value as ViewerDisplayState["scalar_colormap"],
                      })
                    }
                  >
                    <SelectTrigger aria-label="Scalar colormap" className="viewer-select-trigger">
                      <SelectValue placeholder="Scalar colormap" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {SCALAR_VOLUME_COLOR_MAPS.map((colorMap) => (
                          <SelectItem key={colorMap.id} value={colorMap.id}>
                            {colorMap.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
              {canControlScalarVolumeColorMap ? (
                <>
                  <div className="viewer-inline-control" data-viewer-transfer-preset-control="true">
                    <span>Transfer preset</span>
                    <Select
                      value={scalarVolumeTransferPresetId}
                      onValueChange={(value) => {
                        const preset = SCALAR_VOLUME_TRANSFER_PRESETS.find(
                          (candidate) => candidate.id === value
                        );
                        if (!preset || preset.signalFloor == null || preset.densityScale == null) {
                          return;
                        }
                        updateSelectedDisplay({
                          volume_signal_floor: preset.signalFloor,
                          volume_density: preset.densityScale,
                        });
                      }}
                    >
                      <SelectTrigger aria-label="Transfer preset" className="viewer-select-trigger">
                        <SelectValue placeholder="Transfer preset" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {SCALAR_VOLUME_TRANSFER_PRESETS.map((preset) => (
                            <SelectItem key={preset.id} value={preset.id}>
                              {preset.label}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                  <label className="viewer-inline-control viewer-inline-control-wide" data-viewer-signal-floor-control="true">
                    <span>Signal floor</span>
                    <input
                      type="range"
                      aria-label="Signal floor"
                      min={0}
                      max={95}
                      step={1}
                      value={Math.round(scalarVolumeTransfer.signalFloor * 100)}
                      onChange={(event) =>
                        updateSelectedDisplay({
                          volume_signal_floor: Number(event.target.value) / 100,
                        })
                      }
                    />
                    <strong>{scalarVolumeTransfer.signalFloorLabel}</strong>
                  </label>
                  <label className="viewer-inline-control viewer-inline-control-wide" data-viewer-density-control="true">
                    <span>Density</span>
                    <input
                      type="range"
                      aria-label="Density"
                      min={0.1}
                      max={3}
                      step={0.05}
                      value={scalarVolumeTransfer.densityScale}
                      onChange={(event) =>
                        updateSelectedDisplay({
                          volume_density: Number(event.target.value),
                        })
                      }
                    />
                    <strong>{scalarVolumeTransfer.densityLabel}</strong>
                  </label>
                  <label
                    className="viewer-inline-control viewer-inline-control-switch"
                    data-viewer-depth-lighting-control="true"
                  >
                    <span>Depth lighting</span>
                    <Switch
                      aria-label="Depth lighting"
                      checked={scalarVolumeLighting.enabled}
                      onCheckedChange={(checked) =>
                        updateSelectedDisplay({
                          volume_lighting: checked,
                          volume_lighting_strength: scalarVolumeLighting.strength,
                        })
                      }
                    />
                  </label>
                  {scalarVolumeLighting.enabled ? (
                    <label
                      className="viewer-inline-control viewer-inline-control-wide"
                      data-viewer-lighting-strength-control="true"
                    >
                      <span>Lighting strength</span>
                      <input
                        type="range"
                        aria-label="Lighting strength"
                        min={0}
                        max={100}
                        step={1}
                        value={Math.round(scalarVolumeLighting.strength * 100)}
                        onChange={(event) =>
                          updateSelectedDisplay({
                            volume_lighting_strength: Number(event.target.value) / 100,
                          })
                        }
                      />
                      <strong>{scalarVolumeLighting.strengthLabel}</strong>
                    </label>
                  ) : null}
                </>
              ) : null}
              <label className="viewer-inline-control viewer-inline-control-switch">
                <span>Negative</span>
                <Switch
                  checked={selectedDisplayState.negative}
                  onCheckedChange={(checked) => updateSelectedDisplay({ negative: checked })}
                />
              </label>
              {viewerInfo.viewer.volume_mode === "scalar" && canControlChannels && channelNames.length > 1 ? (
                <div className="viewer-inline-control" data-viewer-volume-channel-control="true">
                  <span>Volume channel</span>
                  <Select
                    value={String(volumeChannelIndex)}
                    onValueChange={(value) => {
                      const nextChannel = Math.max(0, Math.floor(Number(value)));
                      updateSelectedDisplay({
                        channels: [nextChannel],
                        volume_channel: nextChannel,
                      });
                    }}
                  >
                    <SelectTrigger aria-label="Volume channel" className="viewer-select-trigger">
                      <SelectValue placeholder="Volume channel" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {channelNames.map((label, index) => (
                          <SelectItem key={`${label}-${index}`} value={String(index)}>
                            {label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
              {selectedSurface === "volume" ? (
                <div className="viewer-volume-clip-panel" data-viewer-volume-clip-controls="true">
                  <div className="viewer-volume-clip-header">
                    <span>Cutaway</span>
                    <Button type="button" variant="ghost" size="sm" onClick={resetVolumeClip}>
                      Reset
                    </Button>
                  </div>
                  <div className="viewer-volume-clip-grid">
                    {(["x", "y", "z"] as const).map((axis) => (
                      <div key={axis} className="viewer-volume-clip-row">
                        <span className="viewer-volume-clip-label">{axis.toUpperCase()}</span>
                        <div className="viewer-volume-clip-sliders">
                          <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            aria-label={`Clip ${axis.toUpperCase()} start`}
                            value={Math.round(clipBounds.min[axis] * 100)}
                            onChange={(event) =>
                              updateVolumeClipEdge("min", axis, Number(event.target.value) / 100)
                            }
                          />
                          <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            aria-label={`Clip ${axis.toUpperCase()} end`}
                            value={Math.round(clipBounds.max[axis] * 100)}
                            onChange={(event) =>
                              updateVolumeClipEdge("max", axis, Number(event.target.value) / 100)
                            }
                          />
                        </div>
                        <strong className="viewer-volume-clip-readout">
                          {Math.round(clipBounds.min[axis] * 100)}-{Math.round(clipBounds.max[axis] * 100)}%
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {renderChannelControls()}
            </div>
          </CollapsibleContent>
        </Collapsible>
      ) : null}

      {directIntensityReady && selectedDisplayState ? (
        <div className="viewer-display-controls viewer-display-controls-direct" data-viewer-intensity-controls="true">
          {renderIntensityHistogramPanel()}
          <label className="viewer-inline-control viewer-inline-control-wide">
            <span>Window center</span>
            <input
              type="range"
              aria-label="Window center"
              min={arrayMin}
              max={arrayMax}
              step={intensityStep}
              value={Math.max(arrayMin, Math.min(arrayMax, parsedWindow.center))}
              onChange={(event) =>
                updateSelectedDisplay({
                  enhancement: buildWindowEnhancement(Number(event.target.value), parsedWindow.width),
                })
              }
            />
            <strong>{formatIntensityValue(parsedWindow.center)}</strong>
          </label>
          <label className="viewer-inline-control viewer-inline-control-wide">
            <span>Window width</span>
            <input
              type="range"
              aria-label="Window width"
              min={intensityStep}
              max={intensityRangeSpan}
              step={intensityStep}
              value={Math.max(intensityStep, Math.min(intensityRangeSpan, parsedWindow.width))}
              onChange={(event) =>
                updateSelectedDisplay({
                  enhancement: buildWindowEnhancement(parsedWindow.center, Number(event.target.value)),
                })
              }
            />
            <strong>{formatIntensityValue(parsedWindow.width)}</strong>
          </label>
          <div className="viewer-intensity-actions">
            <Button type="button" size="sm" variant="outline" onClick={() => updateSelectedDisplay({ enhancement: "d" })}>
              Auto
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={() => updateSelectedDisplay({ enhancement: "f" })}>
              Full
            </Button>
          </div>
          <label className="viewer-inline-control viewer-inline-control-switch">
            <span>Negative</span>
            <Switch
              checked={selectedDisplayState.negative}
              onCheckedChange={(checked) => updateSelectedDisplay({ negative: checked })}
            />
          </label>
        </div>
      ) : uploadHistogramError ? (
        <div className="viewer-metadata-note" data-viewer-intensity-error="true">
          <strong>Histogram unavailable</strong>
          <span>{uploadHistogramError}</span>
        </div>
      ) : null}

      {hasMprIndexControls || has2DIndexControls ? (
        <div className="viewer-controls">
          {selectedSurface === "mpr" ? (
            <>
              {xAxisSize > 1 ? (
                <label className="viewer-slider">
                  <span>X position</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, xAxisSize - 1)}
                    value={clampedIndices.x}
                    onChange={(event) => setSelectedIndex("x", Number(event.target.value))}
                  />
                  <strong>
                    {clampedIndices.x + 1}/{xAxisSize}
                  </strong>
                </label>
              ) : null}
              {yAxisSize > 1 ? (
                <label className="viewer-slider">
                  <span>Y position</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, yAxisSize - 1)}
                    value={clampedIndices.y}
                    onChange={(event) => setSelectedIndex("y", Number(event.target.value))}
                  />
                  <strong>
                    {clampedIndices.y + 1}/{yAxisSize}
                  </strong>
                </label>
              ) : null}
            </>
          ) : null}
          {zAxisSize > 1 ? (
            <label className="viewer-slider">
              <span>Z slice</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, zAxisSize - 1)}
                value={clampedIndices.z}
                onChange={(event) => setSelectedIndex("z", Number(event.target.value))}
              />
              <strong>
                {clampedIndices.z + 1}/{zAxisSize}
              </strong>
            </label>
          ) : null}
          {tAxisSize > 1 ? (
            <label className="viewer-slider">
              <span>Time</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, tAxisSize - 1)}
                value={clampedIndices.t}
                onChange={(event) => setSelectedIndex("t", Number(event.target.value))}
              />
              <strong>
                {clampedIndices.t + 1}/{tAxisSize}
              </strong>
            </label>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
