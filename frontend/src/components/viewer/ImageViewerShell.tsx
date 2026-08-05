import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  Copy,
  Download,
  Focus,
  ImageDown,
  Info,
  Maximize2,
  Minimize2,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";

import { formatBytes } from "@/lib/format";

import { Button } from "@/components/ui/button";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Slider } from "@/components/ui/slider";
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
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import { normalizeViewerCalibrations } from "@/lib/viewerManifest";
import {
  sliceAxisCoordinates,
  sliceChannelSelection,
  supportsSliceChannelColor,
} from "@/lib/viewerSliceContract";
import type { UploadViewerHistogramResponse, UploadViewerInfo } from "@/types";

import { DeepZoomCanvas } from "./DeepZoomCanvas";
import { DirectPlaneImage } from "./DirectPlaneImage";
import { ChannelControls, MAX_COMPOSITE_CHANNELS } from "./ChannelControls";
import {
  canCopyImageToClipboard,
  copyBlobToClipboard,
  downloadBlob,
  downloadFromUrl,
  exportFileStem,
  type ViewerCanvasHandle,
} from "./captureView";
import {
  SCALAR_MASK_NATIVE_POLICY,
  scalarVolumePayloadValueAt,
  validateScalarVolumeIdentity,
} from "./scalarVolume";
import { SlicePlaneCanvas } from "./SlicePlaneCanvas";
import {
  buildScalarSliceSource,
  cancelPendingSlicePrefetches,
  prefetchSliceBitmaps,
  type ScalarSliceSource,
} from "./sliceImageCache";
import {
  mapNearestDeliveryPlanePointToSource,
  mapSourcePlanePointToNearestDelivery,
  mapSourceSliceToNearestDelivery,
  mapSourceVoxelToNearestDelivery,
  scalarPayloadUsesNativeGrid,
  scalarSliceDimensions,
  type ScalarSliceAxis,
} from "./scalarSlice";
import { SliceStackVolumeCanvas } from "./SliceStackVolumeCanvas";
import {
  formatViewerSurfaceLabel,
  getPlaneCursor,
  getPlaneOrientationLabels,
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
import { isTypingTarget, keyToFullscreenAction } from "./fullscreenState";
import { createChromeFadeController, prefersReducedMotionSafe } from "./chromeVisibility";
import {
  canonicalMaskThreshold,
  isExactMaskDtype,
  isMaskCapableScalarVolume,
  resolveScalarRendering,
} from "./scalarMask";

type ViewerDisplayState = NonNullable<UploadViewerInfo["display_defaults"]>;

type UploadHistogramState = {
  key: string;
  histogram: UploadViewerHistogramResponse | null;
  error: string | null;
};

type ScalarProbeState = {
  key: string;
  client: ApiClient | null;
  volume: ScalarVolumePayload | null;
  error: string | null;
};

type SavedMaskCalibration = {
  selectionKey: string;
  revision: number;
  channel: number;
  time: number;
  renderMode: "auto" | "intensity" | "mask";
  thresholdMethod: "otsu-256-v1" | "manual";
  thresholdValue: number;
  thresholdProvenance:
    | NonNullable<
        NonNullable<
          UploadViewerInfo["viewer_calibrations"]
        >["selections"][string]
      >["threshold_provenance"]
    | null;
};

const sameMaskThresholdProvenance = (
  left: SavedMaskCalibration["thresholdProvenance"],
  right: SavedMaskCalibration["thresholdProvenance"]
): boolean =>
  Boolean(
    left &&
      right &&
      left.method === right.method &&
      left.value === right.value &&
      left.domain === right.domain &&
      left.foreground === right.foreground &&
      left.channel === right.channel &&
      left.t === right.t &&
      left.sample_scope === right.sample_scope &&
      left.sample_count === right.sample_count &&
      left.sampling_algorithm === right.sampling_algorithm &&
      left.sampling_strategy === right.sampling_strategy &&
      left.source_sha256 === right.source_sha256 &&
      left.bins === right.bins &&
      left.z_samples.length === right.z_samples.length &&
      left.z_samples.every((value, index) => value === right.z_samples[index])
  );

const sameMaskCalibrationContent = (
  left: SavedMaskCalibration | undefined,
  right: SavedMaskCalibration
): boolean =>
  Boolean(
    left &&
      left.selectionKey === right.selectionKey &&
      left.revision === right.revision &&
      left.channel === right.channel &&
      left.time === right.time &&
      left.renderMode === right.renderMode &&
      left.thresholdMethod === right.thresholdMethod &&
      left.thresholdValue === right.thresholdValue &&
      sameMaskThresholdProvenance(
        left.thresholdProvenance,
        right.thresholdProvenance
      )
  );

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
  onViewerCalibrationsChange?: (
    calibrations: UploadViewerInfo["viewer_calibrations"] | undefined
  ) => void;
};

type MetadataCard = {
  label: string;
  value: string;
};

type MetadataSection = {
  title: string;
  rows: Array<{ label: string; value: string }>;
};

type MetadataDetail = {
  label: string;
  value: string;
  /** Render the value in a monospace font (hashes, ids). */
  mono?: boolean;
  /** Show a copy-to-clipboard affordance. */
  copyable?: boolean;
  /** A small trailing unit/format hint (e.g. "mm", the raw content type). */
  hint?: string;
  /** A leading color swatch (channel LUT color, as a #hex) — data color, not a UI accent. */
  swatch?: string;
};

type MetadataGroup = {
  title: string;
  details: MetadataDetail[];
};

const READER_FORMAT_NAMES: Record<string, string> = {
  "nifti-1": "NIfTI-1",
  "nifti-2": "NIfTI-2",
  dicom: "DICOM",
};

const formatReaderName = (reader: string, contentType?: string): string => {
  const key = String(reader || "").trim().toLowerCase();
  if (READER_FORMAT_NAMES[key]) {
    return READER_FORMAT_NAMES[key];
  }
  if (key.includes("ome")) {
    return "OME-TIFF";
  }
  if (key.includes("tiff")) {
    return "TIFF";
  }
  if (key.includes("nifti")) {
    return "NIfTI";
  }
  if (key) {
    return reader;
  }
  return String(contentType || "").trim() || "Unknown";
};

const titleCaseLabel = (value: string): string => {
  const safe = String(value || "").trim();
  return safe ? `${safe.charAt(0).toUpperCase()}${safe.slice(1)}` : safe;
};

const finitePositive = (value: number | null | undefined): number =>
  typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;

const formatExtent = (value: number): string => (value >= 100 ? value.toFixed(0) : value.toFixed(1));

// "px"/"pixel" are not real physical units — treat them as absent so a medical
// volume's millimetre spacing is not mislabelled.
const isMeaningfulSpatialUnit = (value: unknown): value is string => {
  if (typeof value !== "string") {
    return false;
  }
  const normalized = value.trim().toLowerCase();
  return (
    normalized !== "" &&
    !["px", "pixel", "pixels", "voxel", "voxels", "unknown"].includes(normalized)
  );
};

const resolveMetadataSpatialUnit = (viewerInfo: UploadViewerInfo): string => {
  if (isMeaningfulSpatialUnit(viewerInfo.metadata.physical_spacing_unit)) {
    return viewerInfo.metadata.physical_spacing_unit.trim();
  }
  const coordinates = viewerInfo.phys?.coordinates;
  const spaceUnits =
    coordinates && typeof coordinates === "object"
      ? (coordinates as Record<string, unknown>).space_units
      : null;
  const spatial =
    spaceUnits && typeof spaceUnits === "object" ? (spaceUnits as Record<string, unknown>).spatial : null;
  if (isMeaningfulSpatialUnit(spatial)) {
    return spatial.trim();
  }
  const pixelUnits = viewerInfo.phys?.pixel_units;
  const uniformPixelUnits = Array.isArray(pixelUnits)
    ? pixelUnits.slice(0, 3).map((unit) =>
        isMeaningfulSpatialUnit(unit) ? unit.trim() : null
      )
    : [];
  if (
    uniformPixelUnits.length === 3 &&
    uniformPixelUnits.every((unit): unit is string => unit != null) &&
    new Set(uniformPixelUnits.map((unit) => unit.toLowerCase())).size === 1
  ) {
    return uniformPixelUnits[0] ?? "";
  }
  if (uniformPixelUnits.length === 3 && uniformPixelUnits.some((unit) => unit != null)) {
    return "";
  }
  const spacingUnits = viewerInfo.metadata.spacing_units;
  const uniformSpacingUnits = [spacingUnits?.x, spacingUnits?.y, spacingUnits?.z]
    .map((unit) => (isMeaningfulSpatialUnit(unit) ? unit.trim() : null));
  if (
    uniformSpacingUnits.every((unit): unit is string => unit != null) &&
    new Set(uniformSpacingUnits.map((unit) => unit.toLowerCase())).size === 1
  ) {
    return uniformSpacingUnits[0] ?? "";
  }
  if (uniformSpacingUnits.some((unit) => unit != null)) {
    return "";
  }
  if (viewerInfo.metadata.physical_spacing && String(viewerInfo.modality) === "medical") {
    return "mm";
  }
  return "";
};

function MetadataDetailValue({ detail }: { detail: MetadataDetail }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      return;
    }
    navigator.clipboard
      .writeText(detail.value)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => {});
  };
  return (
    <dd className={`viewer-metadata-kv-value${detail.mono ? " viewer-metadata-kv-value-mono" : ""}`}>
      {detail.swatch ? (
        <span
          className="viewer-metadata-kv-swatch"
          style={{ backgroundColor: detail.swatch }}
          aria-hidden="true"
        />
      ) : null}
      <span className="viewer-metadata-kv-text">{detail.value}</span>
      {detail.hint ? <span className="viewer-metadata-kv-hint">{detail.hint}</span> : null}
      {detail.copyable ? (
        <button
          type="button"
          className="viewer-metadata-copy"
          onClick={handleCopy}
          aria-label={copied ? `${detail.label} copied` : `Copy ${detail.label}`}
          title={copied ? "Copied" : "Copy"}
        >
          {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
        </button>
      ) : null}
    </dd>
  );
}

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
// How many neighbouring slices to warm in the background so scrubbing stays ahead
// of the network. MPR prefetches three axes at once, so it uses a smaller radius.
const SLICE_PREFETCH_RADIUS_2D = 4;
const SLICE_PREFETCH_RADIUS_MPR = 2;
const SCALAR_VOLUME_TRANSFER_PRESETS = [
  { id: "custom", label: "Custom", signalFloor: null, densityScale: null },
  { id: "full", label: "Full range", signalFloor: 0, densityScale: 1 },
  { id: "soft", label: "Soft contrast", signalFloor: 0.15, densityScale: 1.2 },
  { id: "crisp", label: "Crisp structures", signalFloor: 0.35, densityScale: 1.4 },
] as const;

type ScalarVolumeTransferPresetId = (typeof SCALAR_VOLUME_TRANSFER_PRESETS)[number]["id"];

// Standard radiology CT windows (Hounsfield center / width). A tight brain window
// makes the gradient-modulated volume renderer reveal ventricle walls far more
// strongly than a wide default window, since boundary emphasis is computed on the
// windowed signal.
const CT_WINDOW_PRESETS = [
  { id: "brain", label: "Brain", center: 40, width: 80 },
  { id: "subdural", label: "Subdural", center: 80, width: 200 },
  // Narrow acute-stroke window (W8) maximizes gray-white differentiation for
  // early infarct; this is the textbook stroke window, not a wide brain detail.
  { id: "stroke", label: "Stroke", center: 35, width: 8 },
  { id: "soft", label: "Soft tissue", center: 40, width: 400 },
  { id: "fossa", label: "Posterior fossa", center: 40, width: 120 },
  { id: "bone", label: "Bone", center: 600, width: 2800 },
  { id: "lung", label: "Lung", center: -600, width: 1500 },
] as const;

type CtWindowPresetId = (typeof CT_WINDOW_PRESETS)[number]["id"];

// computeRobustHistogramWindow returns the [loPct, hiPct] percentile intensity
// range from a histogram. MR/microscopy have no absolute units, so the default
// window must be robust: a single hot voxel inflates the raw max and washes the
// image out under a min..max stretch. Percentile edges keep tissue legible.
export function computeRobustHistogramWindow(
  histogram: { histogram?: { min?: unknown; max?: unknown; bins?: unknown } } | null | undefined,
  loPct: number,
  hiPct: number
): { min: number; max: number } | null {
  const h = histogram?.histogram;
  const bins = Array.isArray(h?.bins) ? (h?.bins as unknown[]) : null;
  const min = Number(h?.min);
  const max = Number(h?.max);
  if (!bins || bins.length === 0 || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return null;
  }
  const counts = bins.map((c) => Math.max(0, Number(c) || 0));
  const total = counts.reduce((sum, c) => sum + c, 0);
  if (total <= 0) {
    return null;
  }
  const valueAtPercentile = (pct: number): number => {
    const target = pct * total;
    let cumulative = 0;
    for (let i = 0; i < counts.length; i += 1) {
      cumulative += counts[i];
      if (cumulative >= target) {
        return min + ((i + 0.5) / counts.length) * (max - min);
      }
    }
    return max;
  };
  const lo = valueAtPercentile(loPct);
  const hi = valueAtPercentile(hiPct);
  return hi > lo ? { min: lo, max: hi } : null;
}

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

const getActiveCtWindowPresetId = (center: number, width: number): CtWindowPresetId | null => {
  const preset = CT_WINDOW_PRESETS.find(
    (candidate) => Math.abs(candidate.center - center) < 0.5 && Math.abs(candidate.width - width) < 0.5
  );
  return preset?.id ?? null;
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
  const axisUnits = viewerInfo.metadata.spacing_units;
  const unitsByAxis: Record<string, unknown> = {
    X: axisUnits?.x,
    Y: axisUnits?.y,
    Z: axisUnits?.z,
  };
  const planeUnits = descriptor.axes.map((axis) => {
    const unit = unitsByAxis[axis];
    return isMeaningfulSpatialUnit(unit) ? unit.trim() : null;
  });
  const physicalPlaneUnit =
    planeUnits.length === 2 &&
    planeUnits.every((unit): unit is string => unit != null) &&
    new Set(planeUnits.map((unit) => unit.toLowerCase())).size === 1
      ? planeUnits[0]
      : null;
  const usePhysicalScale =
    viewerInfo.viewer.measurement_policy !== "pixel-only" && physicalPlaneUnit != null;
  const rowScale = usePhysicalScale ? Number(descriptor.spacing.row || 1) : 1;
  const colScale = usePhysicalScale ? Number(descriptor.spacing.col || 1) : 1;
  const distance = Math.sqrt((rowDelta * rowScale) ** 2 + (colDelta * colScale) ** 2);
  return formatDistance(distance, physicalPlaneUnit ?? (viewerInfo.is_volume ? "vox" : "px"));
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

// Cross-browser current fullscreen element (Safari uses the webkit-prefixed form).
const currentFullscreenElement = (): Element | null =>
  document.fullscreenElement ??
  (document as unknown as { webkitFullscreenElement?: Element | null }).webkitFullscreenElement ??
  null;

const normalizeSelectedChannelIndices = (
  values: readonly number[] | null | undefined,
  channelCount: number,
): number[] => {
  if (!Array.isArray(values)) {
    return [];
  }
  const normalized: number[] = [];
  const seen = new Set<number>();
  for (const value of values) {
    if (
      !Number.isFinite(value) ||
      !Number.isInteger(value) ||
      value < 0 ||
      value >= channelCount ||
      seen.has(value)
    ) {
      continue;
    }
    normalized.push(value);
    seen.add(value);
    if (normalized.length === MAX_COMPOSITE_CHANNELS) {
      break;
    }
  }
  return normalized;
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
  onViewerCalibrationsChange,
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
    client: null,
    volume: null,
    error: null,
  });
  const uploadHistogramRequestRef = useRef<{
    key: string;
    controller: AbortController;
    abortTimer: number | null;
  } | null>(null);
  const scalarProbeRequestRef = useRef<{
    key: string;
    client: ApiClient;
    controller: AbortController;
    abortTimer: number | null;
  } | null>(null);
  const [maskDrafts, setMaskDrafts] = useState(
    () => new Map<string, SavedMaskCalibration>()
  );
  const maskDraftsRef = useRef(maskDrafts);
  useEffect(() => {
    maskDraftsRef.current = maskDrafts;
  }, [maskDrafts]);
  const [savedMaskBaselines, setSavedMaskBaselines] = useState(
    () => new Map<string, SavedMaskCalibration>()
  );
  const [maskCalibrationSaveRecord, setMaskCalibrationSaveRecord] = useState<{
    selectionKey: string;
    status: "idle" | "saving" | "saved" | "error";
  }>({ selectionKey: "", status: "idle" });
  const [maskThresholdEditing, setMaskThresholdEditing] = useState(false);
  const [serverMaskThresholdState, setServerMaskThresholdState] = useState<{
    selectionKey: string;
    value: number;
  }>({ selectionKey: "", value: 0 });
  // --- Immersive fullscreen (native Fullscreen API on the shell wrapper) ---------
  const shellRef = useRef<HTMLDivElement | null>(null);
  // Imperative handle from whichever 2D canvas is mounted (direct-plane or deep-zoom),
  // so the canvas context menu can fit/reset and export/copy the current view.
  const canvasHandleRef = useRef<ViewerCanvasHandle | null>(null);
  // The context menu portals into the shell node (captured after mount) so it stays
  // visible when the shell is in native fullscreen.
  const [shellPortalContainer, setShellPortalContainer] = useState<HTMLElement | null>(null);
  useEffect(() => {
    setShellPortalContainer(shellRef.current);
  }, []);
  const viewerHoveredRef = useRef(false);
  const fullscreenTriggerRef = useRef<HTMLElement | null>(null);
  const isFullscreenRef = useRef(false);
  const cssFullscreenRef = useRef(false);
  // Native Fullscreen API state (desktop / iPad / Android).
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  // CSS pseudo-fullscreen state — used on iPhone Safari, which exposes no element
  // Fullscreen API at all (requestFullscreen/webkitRequestFullscreen are absent).
  const [cssFullscreen, setCssFullscreen] = useState(false);
  const isFullscreen = nativeFullscreen || cssFullscreen;
  useEffect(() => {
    isFullscreenRef.current = isFullscreen;
    cssFullscreenRef.current = cssFullscreen;
  }, [isFullscreen, cssFullscreen]);

  const exitShellFullscreen = useCallback(() => {
    if (cssFullscreenRef.current) {
      setCssFullscreen(false);
      return;
    }
    const exit =
      document.exitFullscreen ??
      (document as unknown as { webkitExitFullscreen?: () => void }).webkitExitFullscreen;
    if (currentFullscreenElement()) {
      exit?.call(document);
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
    const shell = shellRef.current as
      | (HTMLDivElement & { webkitRequestFullscreen?: () => void })
      | null;
    if (!shell) {
      return;
    }
    if (cssFullscreenRef.current || currentFullscreenElement()) {
      exitShellFullscreen();
      return;
    }
    // Remember where focus was so we can restore it on exit (no focus trap).
    fullscreenTriggerRef.current = (document.activeElement as HTMLElement | null) ?? null;
    const request = shell.requestFullscreen ?? shell.webkitRequestFullscreen;
    if (typeof request !== "function") {
      // iPhone Safari: no element Fullscreen API → CSS pseudo-fullscreen.
      setCssFullscreen(true);
      return;
    }
    try {
      const result = request.call(shell) as Promise<void> | undefined;
      if (result && typeof result.catch === "function") {
        result.catch(() => setCssFullscreen(true));
      }
    } catch {
      // Fullscreen denied — fall back to CSS pseudo-fullscreen rather than staying docked.
      setCssFullscreen(true);
    }
  }, [exitShellFullscreen]);

  // Sync state from the native event (covers UA-driven Esc exit) + manage focus.
  useEffect(() => {
    const onChange = () => {
      const active = currentFullscreenElement() === shellRef.current;
      setNativeFullscreen(active);
      if (active) {
        const focusTarget =
          shellRef.current?.querySelector<HTMLElement>('[role="group"]') ?? shellRef.current;
        focusTarget?.focus?.();
      } else {
        fullscreenTriggerRef.current?.focus?.();
        fullscreenTriggerRef.current = null;
      }
    };
    document.addEventListener("fullscreenchange", onChange);
    document.addEventListener("webkitfullscreenchange", onChange as EventListener);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      document.removeEventListener("webkitfullscreenchange", onChange as EventListener);
    };
  }, []);

  // F toggles, Escape exits. Capture-phase so we can stop the fullscreen-exit Esc
  // from also closing a host sheet; ignored while typing or holding a modifier.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (
        event.key === "Escape" &&
        target?.closest('[role="dialog"], [data-radix-dismissable-layer]')
      ) {
        return;
      }
      const action = keyToFullscreenAction(
        event.key,
        isTypingTarget(event.target),
        event.ctrlKey || event.metaKey || event.altKey
      );
      if (!action) {
        return;
      }
      if (action === "toggle") {
        event.preventDefault();
        toggleFullscreen();
      } else if (isFullscreenRef.current || currentFullscreenElement()) {
        event.preventDefault();
        event.stopPropagation();
        exitShellFullscreen();
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [toggleFullscreen, exitShellFullscreen]);

  // iPhone Safari ignores touch-action for its proprietary pinch gesture, which
  // would zoom the whole page and hijack the viewer's own pinch/pan. Suppress the
  // gesture events on the viewer surface only, so page accessibility zoom stays
  // intact everywhere else.
  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) {
      return;
    }
    const prevent = (event: Event) => event.preventDefault();
    shell.addEventListener("gesturestart", prevent, { passive: false });
    shell.addEventListener("gesturechange", prevent, { passive: false });
    shell.addEventListener("gestureend", prevent, { passive: false });
    return () => {
      shell.removeEventListener("gesturestart", prevent);
      shell.removeEventListener("gesturechange", prevent);
      shell.removeEventListener("gestureend", prevent);
    };
  }, []);

  // CSS pseudo-fullscreen (iPhone): lock page scroll behind the overlay and move
  // focus into the viewer, restoring it on exit — mirroring the native path.
  useEffect(() => {
    if (!cssFullscreen) {
      return;
    }
    const root = document.documentElement;
    const previousOverflow = root.style.overflow;
    root.style.overflow = "hidden";
    const focusTarget =
      shellRef.current?.querySelector<HTMLElement>('[role="group"]') ?? shellRef.current;
    focusTarget?.focus?.();
    return () => {
      root.style.overflow = previousOverflow;
      fullscreenTriggerRef.current?.focus?.();
      fullscreenTriggerRef.current = null;
    };
  }, [cssFullscreen]);

  // Arrow-key scrubbing on the 2D surface: Up/Down step Z, Left/Right step T (and
  // fall back to Z when there is no time axis). Scoped to when the viewer is hovered
  // or focused so it never hijacks page scrolling; skipped while a slider/input is
  // focused (isTypingTarget) so the native range-input arrows still work there.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (selectedSurface !== "2d") return;
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      if (isTypingTarget(event.target)) return;
      const scoped =
        viewerHoveredRef.current ||
        Boolean(shellRef.current && shellRef.current.contains(document.activeElement));
      if (!scoped) return;
      const hasZ = zAxisSize > 1;
      const hasT = tAxisSize > 1;
      const stepZ = (delta: number) =>
        setSelectedIndex("z", Math.max(0, Math.min(zAxisSize - 1, clampedIndices.z + delta)));
      const stepT = (delta: number) =>
        setSelectedIndex("t", Math.max(0, Math.min(tAxisSize - 1, clampedIndices.t + delta)));
      let handled = true;
      switch (event.key) {
        case "ArrowUp":
          if (hasZ) stepZ(1); else handled = false; break;
        case "ArrowDown":
          if (hasZ) stepZ(-1); else handled = false; break;
        case "ArrowRight":
          if (hasT) stepT(1); else if (hasZ) stepZ(1); else handled = false; break;
        case "ArrowLeft":
          if (hasT) stepT(-1); else if (hasZ) stepZ(-1); else handled = false; break;
        default:
          handled = false;
      }
      if (handled) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedSurface, zAxisSize, tAxisSize, clampedIndices.z, clampedIndices.t, setSelectedIndex]);

  // Immersive idle-fade for the whole viewer chrome. The in-canvas toolbar already
  // recedes on idle, but the orientation labels, caption, and controls (which are
  // siblings outside the canvas node) never did — so the "idle" state still framed
  // the image with chrome on every side. A shell-scoped controller writes
  // data-chrome-faded on the viewer-shell; CSS dims every `.viewer-chrome-fade`
  // descendant to a faint level and snaps it back on pointer activity / keyboard
  // focus. Opacity-only (stays tab-navigable) and pinned visible under
  // prefers-reduced-motion. Pointer events bubble from the canvas, so canvas
  // interaction keeps the whole frame revealed in sync with the per-canvas driver.
  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) {
      return;
    }
    const controller = createChromeFadeController(shell, {
      reducedMotion: prefersReducedMotionSafe(),
    });
    controller.reveal();
    const reveal = () => controller.reveal();
    // Immediate-fade on pointer-exit is a MOUSE behavior. On touch, releasing a tap
    // fires pointerleave, which would instantly re-fade the chrome the same tap just
    // revealed; let touch fall through to the controller's idle timer instead.
    const fade = (event: PointerEvent) => {
      if (event.pointerType !== "touch") {
        controller.fadeNow();
      }
    };
    shell.addEventListener("pointermove", reveal, { passive: true });
    shell.addEventListener("pointerdown", reveal, { passive: true });
    shell.addEventListener("focusin", reveal);
    shell.addEventListener("pointerleave", fade, { passive: true });
    return () => {
      controller.dispose();
      shell.removeEventListener("pointermove", reveal);
      shell.removeEventListener("pointerdown", reveal);
      shell.removeEventListener("focusin", reveal);
      shell.removeEventListener("pointerleave", fade);
      delete shell.dataset.chromeFaded;
    };
  }, []);

  const metadataGroups: MetadataGroup[] = (() => {
    const groups: MetadataGroup[] = [];
    const md = viewerInfo.metadata;
    const axisSizes = viewerInfo.axis_sizes;
    const spatialUnit = resolveMetadataSpatialUnit(viewerInfo);

    const fileDetails: MetadataDetail[] = [
      { label: "Name", value: viewerInfo.original_name },
      // Prefer the real container format ("OME-TIFF"/"BigTIFF"); fall back to the
      // reader name only when the backend didn't report a format.
      {
        label: "Format",
        value: md.format && md.format.trim() ? md.format : formatReaderName(md.reader, md.content_type),
        hint: md.content_type,
      },
    ];
    if (viewerInfo.modality) {
      fileDetails.push({ label: "Modality", value: titleCaseLabel(String(viewerInfo.modality)) });
    }
    if (typeof md.size_bytes === "number" && Number.isFinite(md.size_bytes)) {
      fileDetails.push({ label: "File size", value: formatBytes(md.size_bytes) });
    }
    if (md.sha256) {
      fileDetails.push({ label: "SHA-256", value: md.sha256, mono: true, copyable: true });
    }
    fileDetails.push({ label: "File ID", value: viewerInfo.file_id, mono: true, copyable: true });
    groups.push({ title: "File", details: fileDetails });

    // Tiled-mosaic acquisition (a multi-field stage scan). When it was saved
    // unstitched, the assembled image shows per-field illumination seams that look
    // like an artifact but are the raw data — surface it so it isn't mistaken for a bug.
    const mosaic = md.mosaic;
    if (mosaic && mosaic.tiles > 1) {
      const mosaicDetails: MetadataDetail[] = [{ label: "Fields", value: `${mosaic.tiles} tiles` }];
      if (typeof mosaic.overlap === "number" && Number.isFinite(mosaic.overlap)) {
        mosaicDetails.push({ label: "Overlap", value: `${Math.round(mosaic.overlap * 100)}%` });
      }
      if (typeof mosaic.stitched === "boolean") {
        mosaicDetails.push({
          label: "Stitched",
          value: mosaic.stitched ? "Yes" : "No — fields may show illumination seams; stitch upstream for a seamless view",
        });
      }
      groups.push({ title: "Mosaic", details: mosaicDetails });
    }

    // Acquisition / provenance — software, capture date, and (for OME-TIFF) the
    // instrument context libbioimage parses from the embedded OME-XML. Hidden when
    // the file carries none. Fields render in a fixed, scientist-readable order.
    const acquisition = md.acquisition;
    if (acquisition) {
      const acquisitionFields: Array<[string, string]> = [
        ["software", "Software"],
        ["acquired", "Acquired"],
        ["acquisition_mode", "Acquisition mode"],
        ["objective", "Objective"],
        ["objective_medium", "Objective medium"],
        ["refractive_index", "Refractive index"],
        ["detector_binning", "Detector binning"],
        ["experimenter", "Experimenter"],
        ["source_name", "Source"],
      ];
      const acquisitionDetails: MetadataDetail[] = [];
      for (const [key, label] of acquisitionFields) {
        const value = acquisition[key];
        if (value != null && String(value).trim()) {
          acquisitionDetails.push({ label, value: String(value) });
        }
      }
      const levels = acquisition.pyramid_levels;
      if (typeof levels === "number" && levels > 1) {
        acquisitionDetails.push({ label: "Pyramid levels", value: String(levels) });
      }
      if (acquisitionDetails.length > 0) {
        groups.push({ title: "Acquisition", details: acquisitionDetails });
      }
    }

    const totalVoxels = md.array_shape.reduce((acc, value) => acc * Math.max(1, Math.floor(value)), 1);
    const dimensionDetails: MetadataDetail[] = [
      { label: "Array shape", value: md.array_shape.join(" × "), hint: md.dims_order || viewerInfo.dims_order },
      { label: "Width (X)", value: formatNumber(axisSizes.X) },
      { label: "Height (Y)", value: formatNumber(axisSizes.Y) },
    ];
    if (viewerInfo.is_volume || axisSizes.Z > 1) {
      dimensionDetails.push({ label: "Depth (Z)", value: formatNumber(axisSizes.Z) });
    }
    if (axisSizes.C > 1) {
      dimensionDetails.push({ label: "Channels (C)", value: formatNumber(axisSizes.C) });
    }
    if (axisSizes.T > 1) {
      dimensionDetails.push({ label: "Timepoints (T)", value: formatNumber(axisSizes.T) });
    }
    // Time-lapse cadence (e.g. OME-Zarr "1 hour" every frame over "60 hours"). Backend
    // formats the value + unit; shown next to the timepoint count.
    const timeInterval = md.microscopy?.timelapse_interval;
    if (axisSizes.T > 1 && timeInterval != null && String(timeInterval).trim()) {
      dimensionDetails.push({ label: "Time interval", value: String(timeInterval) });
    }
    const timeDuration = md.microscopy?.total_time_duration;
    if (axisSizes.T > 1 && timeDuration != null && String(timeDuration).trim()) {
      dimensionDetails.push({ label: "Duration", value: String(timeDuration) });
    }
    if (md.scene || md.scene_count > 1) {
      dimensionDetails.push({
        label: "Scenes",
        value: md.scene
          ? `${md.scene}${md.scene_count > 1 ? ` (${md.scene_count})` : ""}`
          : formatNumber(md.scene_count),
      });
    }
    dimensionDetails.push({ label: "Total voxels", value: formatNumber(totalVoxels) });
    groups.push({ title: "Dimensions", details: dimensionDetails });

    // Channels — names + their LUT color, for any multichannel image (works for
    // fluorescence channel names like DAPI/EGFP as well as RGBA bands). Replaces the
    // old comma-joined channel string and shows each channel's own color as a swatch.
    const channelNames = viewerInfo.phys?.channel_names ?? [];
    const channelColors = viewerInfo.phys?.channel_colors ?? [];
    if (channelNames.length > 1) {
      const channelDetails: MetadataDetail[] = channelNames.map((name, index) => ({
        label: `Channel ${index + 1}`,
        value: String(name),
        swatch: channelColors[index]?.hex,
      }));
      groups.push({ title: "Channels", details: channelDetails });
    }

    const dataDetails: MetadataDetail[] = [
      { label: "Pixel type", value: formatPixelType(md.array_dtype, viewerInfo.phys?.pixel_depth) },
    ];
    const colorSpace = md.acquisition?.color_space;
    if (typeof colorSpace === "string" && colorSpace.trim()) {
      dataDetails.push({ label: "Color space", value: colorSpace });
    }
    const intensityMin = md.array_min ?? md.intensity_stats?.min;
    const intensityMax = md.array_max ?? md.intensity_stats?.max;
    // Only show the intensity range when the backend actually computed it (a real,
    // non-degenerate range). Absent stats arrive as NaN, so a meaningless "0 → 0"
    // never renders. (typeof narrows; Number.isFinite excludes NaN.)
    if (
      typeof intensityMin === "number" &&
      typeof intensityMax === "number" &&
      Number.isFinite(intensityMin) &&
      Number.isFinite(intensityMax) &&
      intensityMax > intensityMin
    ) {
      dataDetails.push({
        label: "Value range",
        value: `${formatIntensityValue(intensityMin)} → ${formatIntensityValue(intensityMax)}`,
      });
      dataDetails.push({ label: "Span", value: formatIntensityValue(Math.abs(intensityMax - intensityMin)) });
    }
    groups.push({ title: "Data & intensity", details: dataDetails });

    const spacing = md.physical_spacing;
    if (spacing) {
      const sx = finitePositive(spacing.x);
      const sy = finitePositive(spacing.y);
      const sz = finitePositive(spacing.z);
      const geometryDetails: MetadataDetail[] = [];
      const explicitAxisUnits = md.spacing_units;
      const physicalAxisUnits = viewerInfo.phys?.pixel_units;
      const resolveAxisUnit = (metadataUnit: unknown, axisIndex: number): string => {
        for (const candidate of [metadataUnit, physicalAxisUnits?.[axisIndex], spatialUnit]) {
          if (typeof candidate === "string" && candidate.trim()) {
            return candidate.trim();
          }
        }
        return "";
      };
      const spacingAxes = [
        { label: "X", value: sx, unit: resolveAxisUnit(explicitAxisUnits?.x, 0) },
        { label: "Y", value: sy, unit: resolveAxisUnit(explicitAxisUnits?.y, 1) },
        { label: "Z", value: sz, unit: resolveAxisUnit(explicitAxisUnits?.z, 2) },
      ].filter((axis) => axis.value > 0);
      const hasAxisUnit = spacingAxes.some((axis) => axis.unit);
      const mixedAxisUnits =
        hasAxisUnit &&
        (spacingAxes.some((axis) => !axis.unit) ||
          new Set(spacingAxes.map((axis) => axis.unit.toLowerCase())).size > 1);
      const spacingParts = spacingAxes.map((axis) =>
        mixedAxisUnits
          ? `${axis.label} ${axis.value.toFixed(3)} ${axis.unit || "unit unknown"}`
          : `${axis.label} ${axis.value.toFixed(3)}`
      );
      if (spacingParts.length > 0) {
        geometryDetails.push({
          label: "Voxel spacing",
          value: spacingParts.join(" · "),
          hint: mixedAxisUnits ? undefined : spatialUnit || undefined,
        });
      }
      const extentParts: string[] = [];
      if (sx) extentParts.push(formatExtent(axisSizes.X * sx));
      if (sy) extentParts.push(formatExtent(axisSizes.Y * sy));
      if (sz && (viewerInfo.is_volume || axisSizes.Z > 1)) extentParts.push(formatExtent(axisSizes.Z * sz));
      if (extentParts.length > 0 && !mixedAxisUnits) {
        geometryDetails.push({
          label: "Field of view",
          value: extentParts.join(" × "),
          hint: spatialUnit || undefined,
        });
      }
      const positiveSpacings = [sx, sy, sz].filter((value) => value > 0);
      if (positiveSpacings.length >= 2 && !mixedAxisUnits) {
        const ratio = Math.max(...positiveSpacings) / Math.min(...positiveSpacings);
        geometryDetails.push({
          label: "Sampling",
          value: ratio > 1.05 ? `Anisotropic (${ratio.toFixed(1)}×)` : "Isotropic",
        });
      }
      if (geometryDetails.length > 0) {
        groups.push({ title: "Geometry & spacing", details: geometryDetails });
      }
    }

    const orientation = viewerInfo.viewer.orientation;
    if (orientation) {
      const axisLabels = orientation.axis_labels;
      const orientationDetails: MetadataDetail[] = [
        { label: "Frame", value: titleCaseLabel(String(orientation.frame || "pixel")) },
      ];
      if (axisLabels?.x) {
        orientationDetails.push({ label: "X axis", value: `${axisLabels.x.negative ?? "-X"} → ${axisLabels.x.positive ?? "X"}` });
      }
      if (axisLabels?.y) {
        orientationDetails.push({ label: "Y axis", value: `${axisLabels.y.negative ?? "-Y"} → ${axisLabels.y.positive ?? "Y"}` });
      }
      if (viewerInfo.is_volume && axisLabels?.z) {
        orientationDetails.push({ label: "Z axis", value: `${axisLabels.z.negative ?? "-Z"} → ${axisLabels.z.positive ?? "Z"}` });
      }
      if (orientationDetails.length > 1) {
        groups.push({ title: "Orientation", details: orientationDetails });
      }
    }

    if (md.dicom) {
      const dicomDetails: MetadataDetail[] = [];
      if (md.dicom.modality) dicomDetails.push({ label: "Modality", value: String(md.dicom.modality) });
      if (typeof md.dicom.wnd_center === "number") {
        dicomDetails.push({ label: "Window center", value: String(md.dicom.wnd_center) });
      }
      if (typeof md.dicom.wnd_width === "number") {
        dicomDetails.push({ label: "Window width", value: String(md.dicom.wnd_width) });
      }
      if (dicomDetails.length > 0) {
        groups.push({ title: "DICOM", details: dicomDetails });
      }
    }

    // Channels, objective, acquired, and binning now live in the Channels +
    // Acquisition groups; only surface the stage position here when present (rare
    // multi-position / well-plate acquisitions), so nothing is duplicated.
    const microscopy = md.microscopy;
    if (microscopy?.position_index != null) {
      groups.push({
        title: "Stage position",
        details: [{ label: "Index", value: String(microscopy.position_index) }],
      });
    }

    return groups;
  })();

  // Raw, verbose key/value dumps shown under "Technical details". Curated
  // orientation / DICOM / microscopy facts live in the metadata groups above.
  const metadataSections: MetadataSection[] = (() => {
    const sections: MetadataSection[] = [];
    const headerRows = recordToRows(viewerInfo.metadata.header);
    if (headerRows.length > 0) {
      sections.push({ title: "Image header", rows: headerRows });
    }
    const exifRows = recordToRows(viewerInfo.metadata.exif);
    if (exifRows.length > 0) {
      sections.push({ title: "EXIF tags", rows: exifRows });
    }
    const filenameHintRows = recordToRows(viewerInfo.metadata.filename_hints);
    if (filenameHintRows.length > 0) {
      sections.push({ title: "Filename hints", rows: filenameHintRows });
    }
    const geoRows = recordToRows(viewerInfo.metadata.geo);
    if (geoRows.length > 0) {
      sections.push({ title: "Geospatial metadata", rows: geoRows });
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
      sections.push({ title: "Coordinate transform", rows: coordinateRows });
    }
    return sections;
  })();

  const displayCapabilities = new Set((viewerInfo.viewer.display_capabilities ?? []).map((value) => String(value)));
  const channelCount = Math.max(0, Math.floor(Number(viewerInfo.axis_sizes.C) || 0));
  const microscopyChannelNames = viewerInfo.metadata.microscopy?.channel_names;
  const physicalChannelNames = viewerInfo.phys?.channel_names;
  const channelNames = useMemo(
    () =>
      Array.from({ length: channelCount }, (_value, index) => {
        const microscopyName = String(microscopyChannelNames?.[index] ?? "").trim();
        const physicalName = String(physicalChannelNames?.[index] ?? "").trim();
        return microscopyName || physicalName || `Channel ${index + 1}`;
      }),
    [channelCount, microscopyChannelNames, physicalChannelNames],
  );
  const selectedChannelIndices = useMemo(
    () => normalizeSelectedChannelIndices(selectedDisplayState?.channels, channelCount),
    [channelCount, selectedDisplayState?.channels],
  );
  const channelColors = useMemo(
    () =>
      channelNames.map((_, index) =>
        hexColorOrDefault(
          selectedDisplayState?.channel_colors?.[index],
          viewerInfo.phys?.channel_colors?.[index]?.hex ?? "#ffffff",
        ),
      ),
    [channelNames, selectedDisplayState?.channel_colors, viewerInfo.phys?.channel_colors],
  );
  const selectedChannelKey = selectedChannelIndices.join(",");
  const selectedChannelColorKey = selectedChannelIndices
    .map((index) => channelColors[index])
    .join(",");
  const hasMultipleChannels = Boolean(viewerInfo.is_multichannel) || viewerInfo.axis_sizes.C > 1;
  // An RGB(A) photo (render_policy "display") renders its native colors directly — its
  // bands are not science channels, so the per-channel pills + LUT colors don't apply
  // (e.g. an RGBA orthomosaic must not expose Red/Green/Blue/Alpha composite controls).
  const isDisplayPhoto =
    viewerInfo.viewer.render_policy === "display" &&
    viewerInfo.axis_sizes.T === 1 &&
    viewerInfo.axis_sizes.Z === 1;
  const canControlChannels =
    !isDisplayPhoto &&
    (hasMultipleChannels ||
      displayCapabilities.has("channel_visibility") ||
      displayCapabilities.has("channel_mix") ||
      displayCapabilities.has("channel_color"));
  const canControlChannelColor = displayCapabilities.has("channel_color");
  const sourceSha256 = String(viewerInfo.metadata.sha256 ?? "").trim();
  const scalarIdentity = resolveScalarRendering({
    viewerInfo,
    displayState: selectedDisplayState,
    settledTime: debouncedT,
    requestedMode: "intensity",
    threshold: null,
  });
  const maskHistogramChannel = scalarIdentity?.channel ?? -1;
  const resolvedScalarTime = scalarIdentity?.time ?? -1;
  const maskSelectionId = `c${maskHistogramChannel}:t${resolvedScalarTime}`;
  const maskSelectionKey = [
    viewerInfo.file_id,
    sourceSha256,
    maskSelectionId,
  ].join("\u0000");
  const persistedMaskSelection =
    isMaskCapableScalarVolume(viewerInfo) &&
    viewerInfo.viewer_calibrations?.source_sha256 === sourceSha256
      ? viewerInfo.viewer_calibrations.selections[maskSelectionId]
      : undefined;
  const viewerMaskCapable = isMaskCapableScalarVolume(viewerInfo);
  const canLoadScalarVolumeHistogram =
    viewerInfo.is_volume &&
    (viewerInfo.viewer.volume_mode === "scalar" ||
      viewerInfo.viewer.volume_mode === "slice_stack");
  const volumeHistogramDemand =
    viewerMaskCapable &&
    (selectedDisplayState?.scalar_render_mode === "mask" ||
      persistedMaskSelection?.render_mode === "mask");
  const canLoadUploadHistogram =
    Boolean(viewerInfo.service_urls?.histogram) &&
    (!viewerInfo.is_volume || scalarIdentity != null) &&
    (viewerMaskCapable ||
      displayCapabilities.has("histogram") ||
      displayCapabilities.has("intensity_window")) &&
    (!viewerInfo.is_volume ||
      (canLoadScalarVolumeHistogram && volumeHistogramDemand));
  const uploadHistogramRequestKey = canLoadUploadHistogram
    ? [
        viewerInfo.file_id,
        viewerInfo.is_volume ? maskHistogramChannel : canControlChannels ? selectedChannelKey : "all",
        viewerInfo.is_volume ? resolvedScalarTime : "still",
        viewerInfo.is_volume ? "volume" : "plane",
      ].join("\u0000")
    : "";
  const currentUploadHistogramState =
    uploadHistogramState.key === uploadHistogramRequestKey
      ? uploadHistogramState
      : { key: uploadHistogramRequestKey, histogram: null, error: null };
  const uploadHistogram = currentUploadHistogramState.histogram;
  const uploadHistogramError = currentUploadHistogramState.error;
  useEffect(() => {
    const existingRequest = uploadHistogramRequestRef.current;
    if (
      existingRequest?.key === uploadHistogramRequestKey &&
      !existingRequest.controller.signal.aborted
    ) {
      if (existingRequest.abortTimer !== null) {
        window.clearTimeout(existingRequest.abortTimer);
        existingRequest.abortTimer = null;
      }
      return;
    }
    if (!uploadHistogramRequestKey || uploadHistogramState.key === uploadHistogramRequestKey) {
      return;
    }
    if (existingRequest) {
      if (existingRequest.abortTimer !== null) {
        window.clearTimeout(existingRequest.abortTimer);
      }
      existingRequest.controller.abort();
    }
    const requestController = new AbortController();
    const request = {
      key: uploadHistogramRequestKey,
      controller: requestController,
      abortTimer: null as number | null,
    };
    uploadHistogramRequestRef.current = request;
    const histogramConfig = viewerInfo.is_volume
      ? {
          bins: 256,
          channel: maskHistogramChannel,
          t: resolvedScalarTime,
          scope: "volume" as const,
          signal: requestController.signal,
        }
      : canControlChannels && selectedChannelIndices.length > 0
        ? {
            bins: 256,
            channels: selectedChannelIndices,
            signal: requestController.signal,
          }
        : { bins: 256, signal: requestController.signal };
    apiClient
      .getUploadHistogram(viewerInfo.file_id, histogramConfig)
      .then((response) => {
        if (
          uploadHistogramRequestRef.current !== request ||
          requestController.signal.aborted
        ) {
          return;
        }
        setUploadHistogramState({ key: uploadHistogramRequestKey, histogram: response, error: null });
      })
      .catch((error: unknown) => {
        if (
          uploadHistogramRequestRef.current !== request ||
          requestController.signal.aborted
        ) {
          return;
        }
        setUploadHistogramState({
          key: uploadHistogramRequestKey,
          histogram: null,
          error: error instanceof Error ? error.message : "Histogram unavailable.",
        });
      });
    return () => {
      request.abortTimer = window.setTimeout(() => {
        if (
          uploadHistogramRequestRef.current === request &&
          request.abortTimer !== null
        ) {
          requestController.abort();
          uploadHistogramRequestRef.current = null;
        }
      }, 0);
    };
  }, [
    apiClient,
    canControlChannels,
    maskHistogramChannel,
    resolvedScalarTime,
    selectedChannelIndices,
    selectedChannelKey,
    uploadHistogramRequestKey,
    viewerInfo.file_id,
    viewerInfo.is_volume,
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
  // The default window anchors on a robust p1..p99 range (not the raw min..max)
  // so non-CT scalars aren't washed out by outliers. The slider still spans the
  // full arrayMin..arrayMax. CT supplies an explicit Hounsfield window, and a
  // real DICOM acquisition window takes precedence over both.
  const robustWindow = useMemo(
    () => computeRobustHistogramWindow(uploadHistogram, 0.01, 0.99),
    [uploadHistogram]
  );
  const windowAnchorMin = robustWindow ? robustWindow.min : arrayMin;
  const windowAnchorMax = robustWindow ? robustWindow.max : arrayMax;
  const defaultCenter = Number(
    viewerInfo.metadata.dicom?.wnd_center ?? (windowAnchorMin + windowAnchorMax) / 2
  );
  const defaultWidth = Number(
    viewerInfo.metadata.dicom?.wnd_width ?? Math.max(1, Math.abs(windowAnchorMax - windowAnchorMin))
  );
  const parsedWindow = parseWindowLevel(selectedDisplayState?.enhancement, defaultCenter, defaultWidth);
  const showCtWindowPresets = isScalarMpr && viewerInfo.modality === "medical";
  const activeCtWindowPresetId = getActiveCtWindowPresetId(parsedWindow.center, parsedWindow.width);
  const intensityRangeSpan = Math.max(1, Math.abs(arrayMax - arrayMin));
  const intensityStep = intensityRangeSpan <= 16 ? 0.1 : 1;
  const sourceIntensityReady = Boolean(
    selectedDisplayState &&
      uploadHistogram &&
      uploadHistogram.histogram.bins.length > 0 &&
      Number.isFinite(arrayMin) &&
      Number.isFinite(arrayMax)
  );
  // These controls are only honest when the selected 2D renderer consumes the
  // display transform. Scientific slice/tile URLs are intentionally exact and
  // ignore window/enhancement/negative, so exposing the controls there creates
  // a convincing no-op. Native display photos keep the effective controls.
  const directIntensityReady = Boolean(
    isDisplayPhoto &&
      !viewerInfo.is_volume &&
      viewerInfo.service_urls?.display &&
      sourceIntensityReady
  );
  const volumeIntensityReady = Boolean(viewerInfo.is_volume && sourceIntensityReady);
  const histogramMaxCount = Math.max(1, ...(uploadHistogram?.histogram.bins ?? [1]));
  const scalarDtype = String(
    uploadHistogram?.dtype ?? viewerInfo.metadata.array_dtype ?? ""
  ).toLowerCase();
  const canonicalHistogramThreshold = canonicalMaskThreshold(
    uploadHistogram?.threshold?.value,
    scalarDtype
  );
  const exactMaskHistogram = Boolean(
    uploadHistogram &&
      uploadHistogram.channel === maskHistogramChannel &&
      uploadHistogram.t === resolvedScalarTime &&
      uploadHistogram.scope === "volume" &&
      uploadHistogram.threshold?.method === "otsu-256-v1" &&
      uploadHistogram.threshold.domain === "raw" &&
      uploadHistogram.threshold.foreground === "above" &&
      uploadHistogram.threshold.channel === maskHistogramChannel &&
      uploadHistogram.threshold.t === resolvedScalarTime &&
      canonicalHistogramThreshold !== null &&
      String(uploadHistogram.threshold.sampling_algorithm ?? "").trim() &&
      (uploadHistogram.threshold.sampling_strategy === "exact" ||
        uploadHistogram.threshold.sampling_strategy ===
          "stratified-z-spatial") &&
      uploadHistogram.threshold.sample_scope ===
        (uploadHistogram.threshold.sampling_strategy === "exact"
          ? "volume"
          : "stratified_z") &&
      uploadHistogram.threshold.source_sha256 === sourceSha256 &&
      Number.isSafeInteger(uploadHistogram.threshold.bins) &&
      (uploadHistogram.threshold.bins ?? 0) >= 8 &&
      Array.isArray(uploadHistogram.threshold.z_samples) &&
      uploadHistogram.threshold.z_samples.length > 0
  );
  const exactHistogramThreshold =
    exactMaskHistogram && uploadHistogram?.threshold
      ? uploadHistogram.threshold
      : null;
  const viewerThreshold = viewerInfo.data_semantics?.threshold;
  const canonicalViewerThreshold = canonicalMaskThreshold(
    viewerThreshold?.value,
    scalarDtype
  );
  const matchingViewerThreshold =
    viewerThreshold?.channel === maskHistogramChannel &&
    viewerThreshold.t === resolvedScalarTime &&
    canonicalViewerThreshold !== null
      ? viewerThreshold
      : null;
  const histogramSemanticsThreshold = canonicalMaskThreshold(
    uploadHistogram?.data_semantics?.threshold?.value,
    scalarDtype
  );
  const histogramSelectionSemantics =
    exactMaskHistogram &&
    uploadHistogram?.data_semantics?.threshold?.channel === maskHistogramChannel &&
    uploadHistogram.data_semantics.threshold.t === resolvedScalarTime &&
    histogramSemanticsThreshold !== null &&
    histogramSemanticsThreshold === canonicalHistogramThreshold
      ? uploadHistogram.data_semantics
      : undefined;
  const viewerSelectionSemantics =
    matchingViewerThreshold && viewerInfo.data_semantics
      ? viewerInfo.data_semantics
      : undefined;
  const selectionSemantics =
    histogramSelectionSemantics ?? viewerSelectionSemantics;
  const persistedThresholdValue = canonicalMaskThreshold(
    persistedMaskSelection?.threshold_value,
    scalarDtype
  );
  const persistedProvenanceValue = canonicalMaskThreshold(
    persistedMaskSelection?.threshold_provenance.value,
    scalarDtype
  );
  const validPersistedMaskSelection =
    persistedMaskSelection &&
    persistedMaskSelection.channel === maskHistogramChannel &&
    persistedMaskSelection.t === resolvedScalarTime &&
    persistedThresholdValue !== null &&
    persistedProvenanceValue !== null &&
    (persistedMaskSelection.threshold_method !== "otsu-256-v1" ||
      persistedThresholdValue === persistedProvenanceValue)
      ? persistedMaskSelection
      : undefined;
  const inferredOtsuThreshold =
    canonicalHistogramThreshold ??
    persistedProvenanceValue ??
    canonicalViewerThreshold;
  const histogramSamplingAlgorithm = String(
    exactHistogramThreshold?.sampling_algorithm ??
      (uploadHistogram?.sampling?.algorithm as string | undefined) ??
      ""
  ).trim();
  const exactThresholdProvenance =
    exactHistogramThreshold && canonicalHistogramThreshold !== null
    ? {
        method: "otsu-256-v1" as const,
        value: canonicalHistogramThreshold,
        domain: "raw" as const,
        foreground: "above" as const,
        channel: maskHistogramChannel,
        t: resolvedScalarTime,
        sample_scope: exactHistogramThreshold.sample_scope as
          | "volume"
          | "stratified_z",
        sample_count: Number(
          exactHistogramThreshold.sample_count ?? uploadHistogram?.sample_count ?? 0
        ),
        sampling_algorithm: histogramSamplingAlgorithm,
        sampling_strategy: exactHistogramThreshold.sampling_strategy as
          | "exact"
          | "stratified-z-spatial",
        z_samples: [...(exactHistogramThreshold.z_samples ?? [])],
        source_sha256: sourceSha256,
        bins: exactHistogramThreshold.bins as number,
      }
    : null;
  const storedBaseline: SavedMaskCalibration | null = validPersistedMaskSelection
    ? {
        selectionKey: maskSelectionKey,
        revision: validPersistedMaskSelection.revision,
        channel: validPersistedMaskSelection.channel,
        time: validPersistedMaskSelection.t,
        renderMode: validPersistedMaskSelection.render_mode,
        thresholdMethod: validPersistedMaskSelection.threshold_method,
        thresholdValue: persistedThresholdValue as number,
        thresholdProvenance: {
          ...validPersistedMaskSelection.threshold_provenance,
          value: persistedProvenanceValue as number,
        },
      }
    : null;
  const inferredBaseline: SavedMaskCalibration | null =
    inferredOtsuThreshold !== null &&
        (exactThresholdProvenance || matchingViewerThreshold)
      ? {
          selectionKey: maskSelectionKey,
          revision: 0,
          channel: maskHistogramChannel,
          time: resolvedScalarTime,
          renderMode: "auto",
          thresholdMethod: "otsu-256-v1",
          thresholdValue: inferredOtsuThreshold,
          thresholdProvenance: exactThresholdProvenance,
        }
      : null;
  const locallySavedBaseline = savedMaskBaselines.get(maskSelectionKey);
  const savedBaseline =
    locallySavedBaseline ?? storedBaseline ?? inferredBaseline;
  const maskDraft = maskDrafts.get(maskSelectionKey);
  const requestedScalarRenderMode =
    maskDraft?.renderMode ?? savedBaseline?.renderMode ?? "auto";
  const maskThresholdMethod =
    maskDraft?.thresholdMethod ??
    savedBaseline?.thresholdMethod ??
    "otsu-256-v1";
  const canonicalDraftThreshold = canonicalMaskThreshold(
    maskDraft?.thresholdProvenance || maskDraft?.thresholdMethod === "manual"
      ? maskDraft?.thresholdValue
      : null,
    scalarDtype
  );
  const maskThresholdValue =
    canonicalDraftThreshold ??
    savedBaseline?.thresholdValue ??
    inferredOtsuThreshold ??
    canonicalMaskThreshold(arrayMin, scalarDtype) ??
    arrayMin;
  const maskCapable =
    viewerMaskCapable &&
    scalarIdentity != null &&
    isExactMaskDtype(viewerInfo.scalar_mask_capability?.dtype);
  const scalarModeViewer = {
    ...viewerInfo,
    data_semantics: selectionSemantics,
  };
  const hasMaskRenderThresholdEvidence = Boolean(
    exactThresholdProvenance ||
      maskDraft?.thresholdProvenance ||
      savedBaseline?.thresholdProvenance ||
      matchingViewerThreshold
  );
  const resolvedScalarRendering = resolveScalarRendering({
    viewerInfo: scalarModeViewer,
    displayState: selectedDisplayState,
    settledTime: resolvedScalarTime,
    requestedMode: requestedScalarRenderMode,
    threshold: hasMaskRenderThresholdEvidence ? maskThresholdValue : null,
  });
  const effectiveScalarRenderMode =
    resolvedScalarRendering?.effectiveMode ?? "intensity";
  const effectiveMaskMode =
    maskCapable && resolvedScalarRendering?.effectiveMode === "mask";
  // Raw nearest scalar data is also the semantic authority for microscopy Mask
  // MPR, even when the ordinary intensity volume mode is a slice stack.
  const canLoadScalarVolume =
    (viewerInfo.viewer.volume_mode === "scalar" ||
      (viewerMaskCapable && effectiveMaskMode)) &&
    (selectedSurface === "mpr" ||
      (selectedSurface === "2d" && !effectiveMaskMode)) &&
    Boolean(viewerInfo.service_urls?.scalar_volume) &&
    typeof apiClient.getUploadScalarVolume === "function";
  const maskThresholdSpan = Math.max(0.000001, arrayMax - arrayMin);
  const maskThresholdStep =
    scalarDtype.includes("float") || scalarDtype.includes("double")
      ? Math.max(maskThresholdSpan / 1000, 0.0001)
      : 1;
  const maskThresholdMarker = Math.max(
    0,
    Math.min(100, ((maskThresholdValue - arrayMin) / maskThresholdSpan) * 100)
  );
  const baselineForDirty = savedBaseline ?? storedBaseline;
  const maskCalibrationDirty =
    maskCapable &&
    Boolean(
      maskDraft &&
        (!baselineForDirty ||
          requestedScalarRenderMode !== baselineForDirty.renderMode ||
          maskThresholdMethod !== baselineForDirty.thresholdMethod ||
          Math.abs(maskThresholdValue - baselineForDirty.thresholdValue) > 1e-9)
    );
  const maskCalibrationSaveState =
    maskCalibrationSaveRecord.selectionKey === maskSelectionKey
      ? maskCalibrationSaveRecord.status
      : "idle";
  const maskCalibrationProvenance =
    exactThresholdProvenance ??
    maskDraft?.thresholdProvenance ??
    savedBaseline?.thresholdProvenance ??
    null;
  const hasSavableMaskCalibrationEvidence = Boolean(
    maskCalibrationProvenance &&
      maskCalibrationProvenance.channel === maskHistogramChannel &&
      maskCalibrationProvenance.t === resolvedScalarTime &&
      maskCalibrationProvenance.sample_count > 0 &&
      maskCalibrationProvenance.sampling_algorithm &&
      maskCalibrationProvenance.source_sha256 === sourceSha256 &&
      maskCalibrationProvenance.bins >= 8 &&
      maskCalibrationProvenance.z_samples.length > 0
  );
  const serverMaskThresholdValue =
    serverMaskThresholdState.selectionKey === maskSelectionKey
      ? canonicalMaskThreshold(serverMaskThresholdState.value, scalarDtype) ??
        maskThresholdValue
      : maskThresholdValue;
  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setServerMaskThresholdState({
        selectionKey: maskSelectionKey,
        value: maskThresholdValue,
      });
    }, 160);
    return () => window.clearTimeout(timeoutId);
  }, [maskSelectionKey, maskThresholdValue]);
  const clipBounds = normalizeClipBounds(selectedDisplayState);
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
        physicalSpacing: resolveMetadataSpatialUnit(viewerInfo)
          ? viewerInfo.metadata.physical_spacing
          : null,
      })
    : null;
  const physicalVolumeUnit = resolveMetadataSpatialUnit(viewerInfo);
  const volumeGeometryUnit = physicalVolumeUnit || "vox";
  const withPhysicalUnit = (value: string): string => `${value} ${volumeGeometryUnit}`;
  const volumeSpacingLabel = (() => {
    if (physicalVolumeUnit && physicalVolumeGeometry) {
      return `${physicalVolumeGeometry.spacingLabel} ${physicalVolumeUnit}`;
    }
    const spacing = viewerInfo.metadata.physical_spacing;
    const units = viewerInfo.metadata.spacing_units;
    return (["x", "y", "z"] as const)
      .map((axis) => {
        const value = finitePositive(spacing?.[axis]) || 1;
        const unit = String(units?.[axis] || "voxel");
        return `${axis.toUpperCase()} ${value.toFixed(2)} ${unit}`;
      })
      .join(" · ");
  })();
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
        rows.push({ label: "Spacing", value: volumeSpacingLabel });
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
      rows.push({ label: "Extent", value: withPhysicalUnit(physicalVolumeGeometry.aspectLabel) });
      rows.push({ label: "Spacing", value: volumeSpacingLabel });
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
    resolvedScalarRendering?.channel ?? scalarIdentity?.channel ?? volumeChannelIndex;
  const scalarProbeTime =
    resolvedScalarRendering?.time ?? scalarIdentity?.time ?? resolvedScalarTime;
  const scalarProbeSampling = effectiveMaskMode ? "nearest" : "box";
  const scalarProbeExpectedPolicy = effectiveMaskMode
    ? SCALAR_MASK_NATIVE_POLICY
    : null;
  const scalarProbeExpectedMaskDtype = effectiveMaskMode
    ? viewerInfo.scalar_mask_capability?.dtype
    : null;
  const scalarProbeSourceWidth = viewerInfo.axis_sizes.X;
  const scalarProbeSourceHeight = viewerInfo.axis_sizes.Y;
  const scalarProbeSourceDepth = viewerInfo.axis_sizes.Z;
  const scalarProbeRequestKey = canLoadScalarVolume
    ? [
        viewerInfo.file_id,
        sourceSha256,
        scalarProbeTime,
        scalarProbeChannelIndex,
        scalarProbeSampling,
        scalarProbeSourceWidth,
        scalarProbeSourceHeight,
        scalarProbeSourceDepth,
        scalarProbeExpectedPolicy ?? "box-policy:any",
        scalarProbeExpectedMaskDtype ?? "mask-dtype:none",
      ].join("\u0000")
    : "";
  const currentScalarProbeState =
    scalarProbeState.key === scalarProbeRequestKey &&
    scalarProbeState.client === apiClient
      ? scalarProbeState
      : {
          key: scalarProbeRequestKey,
          client: apiClient,
          volume: null,
          error: null,
        };
  const scalarProbeVolume = currentScalarProbeState.volume;
  const scalarProbeError = currentScalarProbeState.error;
  useEffect(() => {
    const existingRequest = scalarProbeRequestRef.current;
    if (!scalarProbeRequestKey) {
      if (existingRequest) {
        if (existingRequest.abortTimer !== null) {
          window.clearTimeout(existingRequest.abortTimer);
        }
        existingRequest.controller.abort();
        scalarProbeRequestRef.current = null;
      }
      if (
        scalarProbeState.key ||
        scalarProbeState.volume ||
        scalarProbeState.error
      ) {
        const clearTimer = window.setTimeout(() => {
          setScalarProbeState({
            key: "",
            client: null,
            volume: null,
            error: null,
          });
        }, 0);
        return () => window.clearTimeout(clearTimer);
      }
      return;
    }
    if (
      existingRequest?.key === scalarProbeRequestKey &&
      existingRequest.client === apiClient &&
      !existingRequest.controller.signal.aborted
    ) {
      if (existingRequest.abortTimer !== null) {
        window.clearTimeout(existingRequest.abortTimer);
        existingRequest.abortTimer = null;
      }
      return;
    }
    if (
      scalarProbeState.key === scalarProbeRequestKey &&
      scalarProbeState.client === apiClient
    ) {
      return;
    }
    if (existingRequest) {
      if (existingRequest.abortTimer !== null) {
        window.clearTimeout(existingRequest.abortTimer);
      }
      existingRequest.controller.abort();
    }
    const requestController = new AbortController();
    const request = {
      key: scalarProbeRequestKey,
      client: apiClient,
      controller: requestController,
      abortTimer: null as number | null,
    };
    scalarProbeRequestRef.current = request;
    apiClient
      .getUploadScalarVolume(viewerInfo.file_id, {
        t: scalarProbeTime,
        channel: scalarProbeChannelIndex,
        sampling: scalarProbeSampling,
        signal: requestController.signal,
      })
      .then((payload) => {
        if (
          scalarProbeRequestRef.current !== request ||
          requestController.signal.aborted
        ) {
          return;
        }
        validateScalarVolumeIdentity(payload, {
          channel: scalarProbeChannelIndex,
          time: scalarProbeTime,
          sourceGrid: {
            width: scalarProbeSourceWidth,
            height: scalarProbeSourceHeight,
            depth: scalarProbeSourceDepth,
          },
          policy: scalarProbeExpectedPolicy ?? payload.previewPolicy,
        });
        if (payload.sampling !== scalarProbeSampling) {
          throw new RangeError(
            `Scalar volume sampling mismatch: expected ${scalarProbeSampling}, received ${String(payload.sampling)}.`
          );
        }
        if (
          scalarProbeExpectedMaskDtype &&
          payload.dtype !== scalarProbeExpectedMaskDtype
        ) {
          throw new RangeError(
            `Scalar Mask dtype mismatch: expected ${scalarProbeExpectedMaskDtype}, received ${String(payload.dtype)}.`
          );
        }
        setScalarProbeState({
          key: scalarProbeRequestKey,
          client: apiClient,
          volume: payload,
          error: null,
        });
      })
      .catch((error: unknown) => {
        if (
          scalarProbeRequestRef.current !== request ||
          requestController.signal.aborted
        ) {
          return;
        }
        setScalarProbeState({
          key: scalarProbeRequestKey,
          client: apiClient,
          volume: null,
          error: error instanceof Error ? error.message : "Scalar probe unavailable.",
        });
      });
    return () => {
      request.abortTimer = window.setTimeout(() => {
        if (
          scalarProbeRequestRef.current === request &&
          request.abortTimer !== null
        ) {
          requestController.abort();
          scalarProbeRequestRef.current = null;
        }
      }, 0);
    };
  }, [
    apiClient,
    scalarProbeChannelIndex,
    scalarProbeExpectedMaskDtype,
    scalarProbeExpectedPolicy,
    scalarProbeState.client,
    scalarProbeState.error,
    scalarProbeState.key,
    scalarProbeState.volume,
    scalarProbeSampling,
    scalarProbeSourceDepth,
    scalarProbeSourceHeight,
    scalarProbeSourceWidth,
    scalarProbeTime,
    scalarProbeRequestKey,
    viewerInfo.file_id,
  ]);
  const displayTransformKey = [
    isDisplayPhoto
      ? `${selectedDisplayState?.enhancement ?? "d"}:${selectedDisplayState?.negative ? "negative" : "positive"}`
      : "scientific-server-exact",
    canControlChannels ? selectedChannelKey || "channels-default" : "channels-all",
    canControlChannelColor ? selectedChannelColorKey || "colors-default" : "colors-source",
    effectiveMaskMode ? "mask" : "intensity",
    effectiveMaskMode ? serverMaskThresholdValue : "no-threshold",
  ].join(":");
  const previewSourceIdentity = effectiveMaskMode
    ? sourceSha256
    : String(viewerInfo.metadata.sha256 ?? viewerInfo.file_id).trim() ||
      viewerInfo.file_id;
  const previewCacheKey = `windowed-v2:${previewSourceIdentity}:${displayTransformKey}`;
  const resolvedSliceChannel = resolvedScalarRendering?.channel;
  const resolvedSliceTime = resolvedScalarRendering?.time;

  const buildMprSliceUrl = useCallback(
    (axis: "x" | "y" | "z", indices: { x: number; y: number; z: number }) =>
      apiClient.uploadSliceUrl(viewerInfo.file_id, {
        axis,
        ...sliceAxisCoordinates(viewerInfo, axis, indices),
        t: resolvedSliceTime,
        ...sliceChannelSelection(
          viewerInfo,
          effectiveMaskMode && resolvedSliceChannel !== undefined
            ? [resolvedSliceChannel]
            : selectedChannelIndices,
          channelColors,
          resolvedSliceChannel
        ),
        scalarRenderMode: effectiveMaskMode ? "mask" : "intensity",
        scalarThresholdValue: effectiveMaskMode ? serverMaskThresholdValue : undefined,
        scalarThresholdForeground: effectiveMaskMode ? "above" : undefined,
        cacheKey: previewCacheKey,
      }),
    [
      apiClient,
      previewCacheKey,
      effectiveMaskMode,
      resolvedSliceChannel,
      resolvedSliceTime,
      serverMaskThresholdValue,
      channelColors,
      selectedChannelIndices,
      viewerInfo,
    ]
  );
  const buildDirect2dSliceUrl = useCallback(
    (z: number, t: number = resolvedSliceTime ?? resolvedScalarTime) =>
      apiClient.uploadSliceUrl(viewerInfo.file_id, {
        axis: "z",
        z,
        t,
        ...sliceChannelSelection(
          viewerInfo,
          effectiveMaskMode && resolvedSliceChannel !== undefined
            ? [resolvedSliceChannel]
            : selectedChannelIndices,
          channelColors,
          resolvedSliceChannel
        ),
        scalarRenderMode: effectiveMaskMode ? "mask" : "intensity",
        scalarThresholdValue: effectiveMaskMode ? serverMaskThresholdValue : undefined,
        scalarThresholdForeground: effectiveMaskMode ? "above" : undefined,
        fullResolution: true,
        cacheKey: previewCacheKey,
      }),
    [
      apiClient,
      previewCacheKey,
      effectiveMaskMode,
      resolvedSliceChannel,
      resolvedSliceTime,
      resolvedScalarTime,
      serverMaskThresholdValue,
      channelColors,
      selectedChannelIndices,
      viewerInfo,
    ]
  );
  const mprSliceUrls = {
    z: effectiveMaskMode
      ? ""
      : buildMprSliceUrl("z", { x: debouncedX, y: debouncedY, z: debouncedZ }),
    y: effectiveMaskMode
      ? ""
      : buildMprSliceUrl("y", { x: debouncedX, y: debouncedY, z: debouncedZ }),
    x: effectiveMaskMode
      ? ""
      : buildMprSliceUrl("x", { x: debouncedX, y: debouncedY, z: debouncedZ }),
  };
  const direct2dSliceUrl = buildDirect2dSliceUrl(debouncedZ);

  // --- Client-side slice rendering (instant scrub + window/level, zero network) ---
  // The full volume is already loaded; slices extracted from it match the backend
  // PNG pixel-for-pixel. Enabled for single-channel scalar volumes with an explicit
  // Hounsfield window (the medical case); anything else uses the cached PNG path.
  // Uses the immediate (non-debounced) indices, since there is no network to gate.
  const enhancementIsHounsfield = String(selectedDisplayState?.enhancement ?? "").startsWith("hounsfield:");
  const clientMaskSliceEnabled =
    effectiveMaskMode &&
    scalarProbeVolume?.sampling === "nearest" &&
    scalarProbeVolume.channel === scalarProbeChannelIndex &&
    scalarProbeVolume.time === scalarProbeTime;
  const clientIntensitySliceEnabled =
    !effectiveMaskMode &&
    Boolean(scalarProbeVolume) &&
    enhancementIsHounsfield &&
    viewerInfo.axis_sizes.C <= 1;
  const clientMprSliceEnabled =
    clientMaskSliceEnabled || clientIntensitySliceEnabled;
  const clientDirect2dSliceEnabled =
    !effectiveMaskMode && clientIntensitySliceEnabled;
  const scalarWindowLow = parsedWindow.center - parsedWindow.width / 2;
  const scalarWindowHigh = parsedWindow.center + parsedWindow.width / 2;
  const scalarInvert = Boolean(selectedDisplayState?.negative);
  const makeScalarSlice = useCallback(
    (
      axis: ScalarSliceAxis,
      sourceSliceIndex: number,
      enabled: boolean
    ): ScalarSliceSource | null => {
      if (!enabled || !scalarProbeVolume) {
        return null;
      }
      const sliceIndex = mapSourceSliceToNearestDelivery(
        scalarProbeVolume,
        axis,
        sourceSliceIndex
      ).deliveryIndex;
      return buildScalarSliceSource({
        fileId: viewerInfo.file_id,
        sourceIdentity: sourceSha256,
        payload: scalarProbeVolume,
        axis,
        sliceIndex,
        windowLow: clientMaskSliceEnabled
          ? serverMaskThresholdValue
          : scalarWindowLow,
        windowHigh: clientMaskSliceEnabled
          ? serverMaskThresholdValue + 1
          : scalarWindowHigh,
        invert: clientMaskSliceEnabled ? false : scalarInvert,
      });
    },
    [
      clientMaskSliceEnabled,
      scalarInvert,
      scalarProbeVolume,
      scalarWindowHigh,
      scalarWindowLow,
      serverMaskThresholdValue,
      sourceSha256,
      viewerInfo.file_id,
    ]
  );
  const direct2dScalarSlice = useMemo(
    () => makeScalarSlice("z", clampedIndices.z, clientDirect2dSliceEnabled),
    [makeScalarSlice, clampedIndices.z, clientDirect2dSliceEnabled]
  );
  const mprScalarSlices = useMemo(
    () => ({
      z: makeScalarSlice("z", clampedIndices.z, clientMprSliceEnabled),
      y: makeScalarSlice("y", clampedIndices.y, clientMprSliceEnabled),
      x: makeScalarSlice("x", clampedIndices.x, clientMprSliceEnabled),
    }),
    [
      makeScalarSlice,
      clampedIndices.x,
      clampedIndices.y,
      clampedIndices.z,
      clientMprSliceEnabled,
    ]
  );

  // Warm neighbouring slices so moving through the stack hits the cache instead of
  // a fresh ~150ms backend round-trip per slice. Skipped when slices render
  // client-side (extraction is instant, so there is nothing to warm).
  useEffect(() => {
    const activeClientSliceEnabled =
      selectedSurface === "mpr" ? clientMprSliceEnabled : clientDirect2dSliceEnabled;
    if (
      !viewerInfo.is_volume ||
      activeClientSliceEnabled ||
      (effectiveMaskMode && selectedSurface === "mpr")
    ) {
      return;
    }
    if (
      effectiveMaskMode &&
      (maskThresholdEditing ||
        Math.abs(serverMaskThresholdValue - maskThresholdValue) > 1e-9)
    ) {
      cancelPendingSlicePrefetches();
      return;
    }
    const urls: string[] = [];
    if (selectedSurface === "2d") {
      for (let step = 1; step <= SLICE_PREFETCH_RADIUS_2D; step += 1) {
        // Warm Z neighbors at the current timepoint...
        if (debouncedZ + step < zAxisSize) urls.push(buildDirect2dSliceUrl(debouncedZ + step));
        if (debouncedZ - step >= 0) urls.push(buildDirect2dSliceUrl(debouncedZ - step));
        // ...and T neighbors at the current plane, so time scrubbing is as smooth as
        // Z scrubbing (the cache key includes t, so these resolve instantly on arrival).
        if (tAxisSize > 1 && !effectiveMaskMode) {
          if (debouncedT + step < tAxisSize) urls.push(buildDirect2dSliceUrl(debouncedZ, debouncedT + step));
          if (debouncedT - step >= 0) urls.push(buildDirect2dSliceUrl(debouncedZ, debouncedT - step));
        }
      }
    } else if (selectedSurface === "mpr") {
      for (let step = 1; step <= SLICE_PREFETCH_RADIUS_MPR; step += 1) {
        if (debouncedZ + step < zAxisSize) urls.push(buildMprSliceUrl("z", { x: debouncedX, y: debouncedY, z: debouncedZ + step }));
        if (debouncedZ - step >= 0) urls.push(buildMprSliceUrl("z", { x: debouncedX, y: debouncedY, z: debouncedZ - step }));
        if (debouncedY + step < yAxisSize) urls.push(buildMprSliceUrl("y", { x: debouncedX, y: debouncedY + step, z: debouncedZ }));
        if (debouncedY - step >= 0) urls.push(buildMprSliceUrl("y", { x: debouncedX, y: debouncedY - step, z: debouncedZ }));
        if (debouncedX + step < xAxisSize) urls.push(buildMprSliceUrl("x", { x: debouncedX + step, y: debouncedY, z: debouncedZ }));
        if (debouncedX - step >= 0) urls.push(buildMprSliceUrl("x", { x: debouncedX - step, y: debouncedY, z: debouncedZ }));
      }
    }
    if (urls.length > 0) {
      prefetchSliceBitmaps(urls);
    }
  }, [
    buildDirect2dSliceUrl,
    buildMprSliceUrl,
    clientDirect2dSliceEnabled,
    clientMprSliceEnabled,
    debouncedX,
    debouncedY,
    debouncedZ,
    debouncedT,
    effectiveMaskMode,
    maskThresholdEditing,
    maskThresholdValue,
    selectedSurface,
    serverMaskThresholdValue,
    viewerInfo.is_volume,
    xAxisSize,
    yAxisSize,
    zAxisSize,
    tAxisSize,
  ]);
  const direct2dDisplayUrl =
    isDisplayPhoto && !viewerInfo.is_volume && viewerInfo.service_urls?.display
      ? apiClient.uploadDisplayUrl(viewerInfo.file_id, viewerInfo.service_urls.display, {
          enhancement: selectedDisplayState?.enhancement,
          negative: selectedDisplayState?.negative,
          channels: canControlChannels ? selectedChannelIndices : undefined,
          channelColors: canControlChannelColor ? channelColors : undefined,
          cacheKey: previewCacheKey,
        })
      : null;
  // Scientific images render through selector-aware /slice regardless of reader. The
  // legacy /display path is reserved for native-color RGB(A) photos: it is not allowed
  // to flatten or silently replace a failed T/Z/C/LUT scientific request.
  const direct2dImageUrl =
    isDisplayPhoto ? (direct2dDisplayUrl ?? direct2dSliceUrl) : direct2dSliceUrl;
  const direct2dPreviewUrl = apiClient.uploadPreviewUrl(viewerInfo.file_id);
  const canUseDeepZoom2D =
    !viewerInfo.is_volume &&
    viewerInfo.axis_sizes.T === 1 &&
    viewerInfo.axis_sizes.Z === 1 &&
    (viewerInfo.backend_mode === "pyramid" ||
      viewerInfo.viewer.backend_mode === "pyramid" ||
      viewerInfo.viewer.delivery_mode === "deferred_multiscale") &&
    viewerInfo.viewer.tile_scheme.levels.length > 0;

  const validatedNearestMaskMapping =
    effectiveMaskMode && clientMaskSliceEnabled;
  const nativeExactIntensityMapping =
    !effectiveMaskMode &&
    canLoadScalarProbe &&
    scalarProbeVolume?.sampling === "box" &&
    scalarPayloadUsesNativeGrid(scalarProbeVolume) &&
    (scalarProbeVolume.previewPolicy === "native-exact-v1" ||
      scalarProbeVolume.previewPolicy === "exact-v1");
  const scalarProbeMapping =
    scalarProbeVolume && (validatedNearestMaskMapping || nativeExactIntensityMapping)
      ? mapSourceVoxelToNearestDelivery(scalarProbeVolume, clampedIndices)
      : null;
  const scalarProbeValue =
    scalarProbeVolume && scalarProbeMapping?.exact
      ? scalarVolumePayloadValueAt(scalarProbeVolume, scalarProbeMapping.delivery)
      : null;
  const mprCrosshairIndices =
    scalarProbeMapping
      ? { ...clampedIndices, ...scalarProbeMapping.sampledSource }
      : clampedIndices;
  const mprUsesDeliveryCoordinates = Boolean(
    clientMprSliceEnabled && scalarProbeVolume
  );
  const mprCoordinateGrid = (axis: PlaneAxis) =>
    mprUsesDeliveryCoordinates && scalarProbeVolume
      ? scalarSliceDimensions(scalarProbeVolume, axis)
      : null;
  const mapMprSourcePointToDisplay = (
    axis: PlaneAxis,
    point: PlanePoint
  ): PlanePoint =>
    mprUsesDeliveryCoordinates && scalarProbeVolume
      ? mapSourcePlanePointToNearestDelivery(
          scalarProbeVolume,
          axis,
          point
        ).delivery
      : point;
  const mapMprDisplayPointToSource = (
    axis: PlaneAxis,
    point: PlanePoint
  ): PlanePoint =>
    mprUsesDeliveryCoordinates && scalarProbeVolume
      ? mapNearestDeliveryPlanePointToSource(
          scalarProbeVolume,
          axis,
          point
        )
      : point;
  const cursorReadoutRows = [
    ...computeCursorWorldPosition(viewerInfo, clampedIndices),
    ...(scalarProbeValue != null
      ? [{ label: "Voxel value", value: formatIntensityValue(scalarProbeValue) }]
      : scalarProbeMapping && !scalarProbeMapping.exact
        ? [
            { label: "Voxel value", value: "Unavailable (not sampled in preview)" },
            {
              label: "Preview sample",
              value: `x=${scalarProbeMapping.sampledSource.x}, y=${scalarProbeMapping.sampledSource.y}, z=${scalarProbeMapping.sampledSource.z}`,
            },
          ]
      : scalarProbeError
        ? [{ label: "Voxel value", value: "Unavailable" }]
        : []),
  ];
  const activeMeasurementPlaneLabel =
    viewerInfo.viewer.planes[activeMeasurementAxis]?.label ?? activeMeasurementAxis.toUpperCase();

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
      volume_cutaway: false,
    });
  };

  // Cutaway follows the live Z slice. Camera, projection, and window settings
  // remain untouched so enabling it changes only which portion is visualized.
  const enableVolumeCutaway = () => {
    updateSelectedDisplay({
      volume_cutaway: true,
    });
  };

  const cutawayActive = Boolean(selectedDisplayState?.volume_cutaway);
  const volumeClipActive = cutawayActive || isVolumeClipActive(clipBounds);

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

  // --- Canvas context-menu actions (right-click / long-press on the 2D surface) ---
  const viewExportName = `${exportFileStem(viewerInfo.original_name)}-view.png`;
  const metadataAvailable = viewerInfo.viewer.available_surfaces.includes("metadata");

  const handleResetView = () => {
    canvasHandleRef.current?.fitView();
  };

  const handleExportView = async () => {
    const blob = await canvasHandleRef.current?.captureViewToBlob();
    if (blob) {
      downloadBlob(blob, viewExportName);
    }
  };

  const handleCopyView = async () => {
    const blob = await canvasHandleRef.current?.captureViewToBlob();
    if (!blob) {
      return;
    }
    // Fall back to a file download if the clipboard write fails (unsupported / denied).
    const copied = await copyBlobToClipboard(blob);
    if (!copied) {
      downloadBlob(blob, viewExportName);
    }
  };

  const handleDownloadOriginal = () => {
    downloadFromUrl(apiClient.resourceDownloadUrl(viewerInfo.file_id), viewerInfo.original_name ?? "image");
  };

  const renderChannelControls = () => {
    if (!selectedDisplayState || !canControlChannels || channelNames.length <= 1) {
      return null;
    }
    const resolvedChannelMode = String(
      selectedDisplayState.channel_mode ?? viewerInfo.viewer.channel_mode ?? "",
    ).toLowerCase();
    const singleChannelMode = viewerInfo.viewer.volume_mode === "scalar" || resolvedChannelMode === "single";
    const toggleChannel = (index: number, active: boolean) => {
      if (singleChannelMode) {
        updateSelectedDisplay({ channels: [index], volume_channel: index });
        return;
      }
      const current = new Set(selectedChannelIndices);
      if (active && current.size > 1) {
        current.delete(index);
      } else if (!active && current.size < MAX_COMPOSITE_CHANNELS) {
        current.add(index);
      }
      const nextChannels = Array.from(current);
      const patch: Partial<ViewerDisplayState> = { channels: nextChannels };
      if (viewerInfo.viewer.volume_mode === "scalar" && nextChannels.length === 1) {
        patch.volume_channel = nextChannels[0];
      }
      updateSelectedDisplay(patch);
    };
    const setChannelColor = (index: number, hex: string) => {
      const nextColors = [...(selectedDisplayState.channel_colors ?? [])];
      nextColors[index] = hex;
      updateSelectedDisplay({ channel_colors: nextColors });
    };
    return (
      <ChannelControls
        channelNames={channelNames}
        channelColors={channelColors}
        selectedIndices={selectedChannelIndices}
        canEditColor={canControlChannelColor}
        singleChannelMode={singleChannelMode}
        portalContainer={shellPortalContainer}
        onToggleChannel={toggleChannel}
        onSetChannelColor={setChannelColor}
      />
    );
  };

  const updateMaskDisplay = (patch: Partial<ViewerDisplayState>) => {
    const nextThresholdValue =
      canonicalMaskThreshold(
        patch.scalar_threshold_value ?? maskThresholdValue,
        scalarDtype
      ) ?? maskThresholdValue;
    const nextCalibration: SavedMaskCalibration = {
      selectionKey: maskSelectionKey,
      revision: savedBaseline?.revision ?? 0,
      channel: maskHistogramChannel,
      time: resolvedScalarTime,
      renderMode: patch.scalar_render_mode ?? requestedScalarRenderMode,
      thresholdMethod: patch.scalar_threshold_method ?? maskThresholdMethod,
      thresholdValue: nextThresholdValue,
      thresholdProvenance: maskCalibrationProvenance,
    };
    setMaskDrafts((current) => {
      const next = new Map(current);
      next.set(maskSelectionKey, nextCalibration);
      maskDraftsRef.current = next;
      return next;
    });
    setMaskCalibrationSaveRecord({ selectionKey: maskSelectionKey, status: "idle" });
    updateSelectedDisplay(patch);
  };

  const saveMaskCalibration = async () => {
    if (
      !maskCapable ||
      !maskCalibrationDirty ||
      !sourceSha256 ||
      !Number.isFinite(maskThresholdValue) ||
      !maskCalibrationProvenance ||
      !hasSavableMaskCalibrationEvidence
    ) {
      return;
    }
    const canonicalThresholdValue = canonicalMaskThreshold(
      maskThresholdValue,
      scalarDtype
    );
    if (canonicalThresholdValue === null) {
      return;
    }
    const snapshot: SavedMaskCalibration = {
      selectionKey: maskSelectionKey,
      revision: savedBaseline?.revision ?? 0,
      channel: maskHistogramChannel,
      time: resolvedScalarTime,
      renderMode: requestedScalarRenderMode,
      thresholdMethod: maskThresholdMethod,
      thresholdValue: canonicalThresholdValue,
      thresholdProvenance: maskCalibrationProvenance,
    };
    setMaskCalibrationSaveRecord({ selectionKey: maskSelectionKey, status: "saving" });
    try {
      const response = await apiClient.patchResourceMetadata(viewerInfo.file_id, {
        ultra_viewer_calibration_v1: {
          version: 1,
          source_sha256: sourceSha256,
          selections: {
            [maskSelectionId]: {
              channel: snapshot.channel,
              t: snapshot.time,
              render_mode: snapshot.renderMode,
              threshold_method: snapshot.thresholdMethod,
              threshold_value: snapshot.thresholdValue,
              threshold_foreground: "above",
              expected_revision: snapshot.revision,
              threshold_provenance: snapshot.thresholdProvenance,
            },
          },
        },
      });
      const returnedCalibrations = normalizeViewerCalibrations(
        response.resource?.metadata?.ultra_viewer_calibration_v1,
        sourceSha256
      );
      if (!returnedCalibrations) {
        throw new Error("calibration PATCH response omitted the persisted selection");
      }
      const echoedSelection = returnedCalibrations.selections[maskSelectionId];
      if (
        !echoedSelection ||
        echoedSelection.channel !== snapshot.channel ||
        echoedSelection.t !== snapshot.time ||
        echoedSelection.render_mode !== snapshot.renderMode ||
        echoedSelection.threshold_method !== snapshot.thresholdMethod ||
        echoedSelection.threshold_value !== snapshot.thresholdValue ||
        echoedSelection.threshold_foreground !== "above" ||
        echoedSelection.revision !== snapshot.revision + 1 ||
        !sameMaskThresholdProvenance(
          echoedSelection.threshold_provenance,
          snapshot.thresholdProvenance
        )
      ) {
        throw new Error("calibration PATCH response did not echo the submitted selection");
      }
      const acknowledged: SavedMaskCalibration = {
        ...snapshot,
        revision: echoedSelection.revision,
        thresholdProvenance: echoedSelection.threshold_provenance,
      };
      const snapshotStillCurrent = sameMaskCalibrationContent(
        maskDraftsRef.current.get(maskSelectionKey),
        snapshot
      );
      onViewerCalibrationsChange?.(returnedCalibrations);
      setSavedMaskBaselines((current) => {
        const existing = current.get(maskSelectionKey);
        if (existing && existing.revision > acknowledged.revision) {
          return current;
        }
        const next = new Map(current);
        next.set(maskSelectionKey, acknowledged);
        return next;
      });
      setMaskDrafts((current) => {
        const existing = current.get(maskSelectionKey);
        if (!sameMaskCalibrationContent(existing, snapshot)) {
          return current;
        }
        const next = new Map(current);
        next.set(maskSelectionKey, acknowledged);
        maskDraftsRef.current = next;
        return next;
      });
      if (snapshotStillCurrent) {
        setMaskCalibrationSaveRecord({
          selectionKey: maskSelectionKey,
          status: "saved",
        });
      }
    } catch {
      // The edited values intentionally stay in display state so a failed save
      // remains dirty and can be retried without reconstructing the calibration.
      if (
        sameMaskCalibrationContent(
          maskDraftsRef.current.get(maskSelectionKey),
          snapshot
        )
      ) {
        setMaskCalibrationSaveRecord({
          selectionKey: maskSelectionKey,
          status: "error",
        });
      }
    }
  };

  const renderScalarRenderingControl = () => (
    <TooltipProvider delayDuration={120}>
      <div className="viewer-inline-control-group">
        <div className="viewer-inline-control" data-viewer-scalar-render-mode="true">
          <Select
            value={requestedScalarRenderMode}
            onValueChange={(value) =>
              updateMaskDisplay({
                scalar_render_mode: value as ViewerDisplayState["scalar_render_mode"],
              })
            }
          >
            <SelectTrigger aria-label="Scalar rendering" className="viewer-select-trigger">
              <SelectValue placeholder="Scalar rendering" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="auto">
                  Auto · {effectiveScalarRenderMode === "mask" ? "Mask" : "Intensity"}
                </SelectItem>
                <SelectItem value="intensity">Intensity</SelectItem>
                <SelectItem value="mask">Mask</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label="About scalar rendering"
              >
                <Info />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">
              Auto follows source semantics. Mask uses raw values above the threshold.
            </TooltipContent>
          </Tooltip>
        </div>
        {channelNames.length > 1 ? (
          <div className="viewer-inline-control" data-viewer-mask-channel="true">
            <Select
              value={String(maskHistogramChannel)}
              onValueChange={(value) =>
                updateSelectedDisplay({ volume_channel: Number(value) })
              }
            >
              <SelectTrigger aria-label="Mask channel" className="viewer-select-trigger">
                <SelectValue placeholder="Mask channel" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {channelNames.map((label, index) => (
                    <SelectItem key={`mask-${label}-${index}`} value={String(index)}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        ) : null}
      </div>
    </TooltipProvider>
  );

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

  const renderMaskCalibrationPanel = () => (
    <section
      className="viewer-volume-intensity-panel"
      data-viewer-mask-calibration="true"
      aria-label="Mask threshold calibration"
    >
      {volumeIntensityReady ? (
        <div className="viewer-intensity-histogram-panel">
          <div
            className="viewer-intensity-histogram"
            aria-label="Source volume histogram with mask threshold"
            style={{ position: "relative" }}
          >
            {uploadHistogram?.histogram.bins.map((count, index) => (
              <span
                key={`mask-histogram-bin-${index}`}
                style={{ height: `${Math.max(4, (count / histogramMaxCount) * 100)}%` }}
              />
            ))}
            <i
              aria-hidden="true"
              style={{
                position: "absolute",
                insetBlock: 0,
                left: `${maskThresholdMarker}%`,
                width: 2,
                background: "currentColor",
                opacity: 0.9,
              }}
            />
          </div>
          <strong>
            {formatIntensityValue(arrayMin)}-{formatIntensityValue(arrayMax)}
          </strong>
          <span>{histogramSampleLabel(uploadHistogram)}</span>
        </div>
      ) : uploadHistogramError ? (
        <div className="viewer-metadata-note" data-viewer-mask-histogram-error="true">
          <strong>Histogram unavailable</strong>
          <span>{uploadHistogramError}</span>
        </div>
      ) : null}
      <label className="viewer-inline-control viewer-inline-control-wide">
        <span>Raw threshold</span>
        <input
          type="range"
          aria-label="Mask raw threshold"
          min={arrayMin}
          max={arrayMax}
          step={maskThresholdStep}
          value={maskThresholdValue}
          onPointerDown={() => {
            cancelPendingSlicePrefetches();
            setMaskThresholdEditing(true);
          }}
          onPointerUp={() => setMaskThresholdEditing(false)}
          onPointerCancel={() => setMaskThresholdEditing(false)}
          onKeyDown={() => {
            cancelPendingSlicePrefetches();
            setMaskThresholdEditing(true);
          }}
          onKeyUp={() => setMaskThresholdEditing(false)}
          onBlur={() => setMaskThresholdEditing(false)}
          onChange={(event) =>
            updateMaskDisplay({
              scalar_threshold_method: "manual",
              scalar_threshold_value: Number(event.target.value),
              scalar_threshold_foreground: "above",
            })
          }
        />
        <strong>{formatIntensityValue(maskThresholdValue)}</strong>
      </label>
      <TooltipProvider delayDuration={120}>
        <div
          className="viewer-volume-inspection-actions"
          style={{ flexWrap: "wrap" }}
        >
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={!Number.isFinite(inferredOtsuThreshold)}
            onClick={() =>
              updateMaskDisplay({
                scalar_threshold_method: "otsu-256-v1",
                scalar_threshold_value: inferredOtsuThreshold,
                scalar_threshold_foreground: "above",
              })
            }
          >
            Use Otsu
          </Button>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label="About Otsu threshold"
              >
                <Info />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">
              Otsu computes one deterministic raw-value threshold from this exact
              channel and timepoint. It is unavailable until matching histogram
              evidence loads.
            </TooltipContent>
          </Tooltip>
          <Button
            type="button"
            size="sm"
            disabled={
              !maskCalibrationDirty ||
              !sourceSha256 ||
              !hasSavableMaskCalibrationEvidence ||
              maskCalibrationSaveState === "saving"
            }
            onClick={() => void saveMaskCalibration()}
          >
            {maskCalibrationSaveState === "saving" ? "Saving…" : "Save resource default"}
          </Button>
          <span aria-live="polite">
            {maskCalibrationSaveState === "error"
              ? "Save failed; changes remain unsaved."
              : maskCalibrationSaveState === "saved" && !maskCalibrationDirty
                ? "Saved."
                : maskCalibrationDirty
                  ? "Unsaved changes"
                  : ""}
          </span>
        </div>
      </TooltipProvider>
      {!sourceSha256 ? (
        <div className="viewer-metadata-note">
          Source identity is unavailable, so this calibration cannot be saved safely.
        </div>
      ) : null}
    </section>
  );

  const renderVolumeGeometryPanel = () => {
    if (!canShowVolumeGeometry || !physicalVolumeGeometry) {
      return null;
    }
    return (
      <div className="viewer-volume-geometry-panel" data-viewer-volume-geometry="true">
        <span>Physical volume</span>
        <strong>{withPhysicalUnit(physicalVolumeGeometry.aspectLabel)}</strong>
        <span>Spacing {withPhysicalUnit(physicalVolumeGeometry.spacingLabel)}</span>
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

  // Data-first caption shown beneath the canvas (the specimen "label"): name +
  // a concise facts line. For volumes the per-surface readout (projection/window/
  // spacing) carries the detail; for 2D images this single line stands in.
  const captionMeta = [
    viewerInfo.modality ? titleCaseLabel(String(viewerInfo.modality)) : "",
    viewerInfo.is_volume ? "Volume" : viewerInfo.axis_sizes.Z > 1 ? "Stack" : "Image",
    `${formatNumber(viewerInfo.axis_sizes.X)} × ${formatNumber(viewerInfo.axis_sizes.Y)}${
      viewerInfo.is_volume || viewerInfo.axis_sizes.Z > 1
        ? ` × ${formatNumber(viewerInfo.axis_sizes.Z)}`
        : ""
    } vox`,
    viewerInfo.dims_order ? `Axes ${viewerInfo.dims_order}` : "",
  ]
    .filter(Boolean)
    .join("  ·  ");

  // An unstitched tiled-mosaic acquisition shows per-field illumination seams that look
  // like a render artifact but are the raw data — flag it right on the image.
  const mosaicInfo = viewerInfo.metadata.mosaic;
  const unstitchedMosaic = mosaicInfo && mosaicInfo.tiles > 1 && mosaicInfo.stitched === false ? mosaicInfo : null;
  const mosaicBadgeTitle = unstitchedMosaic
    ? `Unstitched ${unstitchedMosaic.tiles}-field mosaic${
        typeof unstitchedMosaic.overlap === "number" ? ` (${Math.round(unstitchedMosaic.overlap * 100)}% overlap)` : ""
      }. The seams between fields are in the source data, not a rendering error — stitch upstream (e.g. ZEN) for a seamless image.`
    : "";

  const hasMprIndexControls =
    selectedSurface === "mpr" && (xAxisSize > 1 || yAxisSize > 1 || zAxisSize > 1 || tAxisSize > 1);
  const has2DIndexControls =
    selectedSurface === "2d" && !showSliceStack2DControls && (zAxisSize > 1 || tAxisSize > 1);
  const effectiveVolumeDisplayState: ViewerDisplayState | null = selectedDisplayState
    ? {
        ...selectedDisplayState,
        channels: selectedChannelIndices,
        scalar_render_mode: effectiveMaskMode ? "mask" : "intensity",
        scalar_threshold_method: maskThresholdMethod,
        scalar_threshold_value: maskThresholdValue,
        scalar_threshold_foreground: "above" as const,
      }
    : null;

  return (
    <div
      ref={shellRef}
      className={cssFullscreen ? "viewer-shell viewer-shell-css-fullscreen" : "viewer-shell"}
      tabIndex={-1}
      onPointerEnter={() => {
        viewerHoveredRef.current = true;
      }}
      onPointerLeave={() => {
        viewerHoveredRef.current = false;
      }}
      data-viewer-fullscreen={isFullscreen ? "true" : "false"}
    >
      <Tabs value={selectedSurface} onValueChange={handleSurfaceChange} className="viewer-surface-tabs">
        <div className="viewer-surface-toolbar">
          <TabsList className="viewer-surface-list">
            {viewerInfo.viewer.available_surfaces.map((surface) => (
              <TabsTrigger key={surface} value={surface}>
                {formatViewerSurfaceLabel(surface)}
              </TabsTrigger>
            ))}
          </TabsList>
          {maskCapable && selectedDisplayState
            ? renderScalarRenderingControl()
            : null}
          <Button
            type="button"
            variant="outline"
            size="icon-xs"
            className="viewer-fullscreen-toggle"
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            aria-pressed={isFullscreen}
            aria-keyshortcuts="f"
            onClick={toggleFullscreen}
          >
            {isFullscreen ? <Minimize2 data-icon="inline-start" /> : <Maximize2 data-icon="inline-start" />}
          </Button>
        </div>

        <TabsContent value="2d" className="viewer-surface-panel">
          <div
            className={
              viewerInfo.is_volume
                ? "viewer-volume-layout viewer-volume-layout-2d viewer-canvas-layout-2d"
                : "viewer-canvas-layout-2d"
            }
          >
            <div className="viewer-canvas-shell viewer-canvas-shell-2d">
              <ContextMenu>
                <ContextMenuTrigger asChild>
                  <div
                    data-viewer-surface="2d"
                    data-viewer-backend={canUseDeepZoom2D ? "pyramid" : "direct"}
                    data-viewer-aspect={viewerInfo.viewer.default_plane.aspect_ratio.toFixed(4)}
                    onContextMenu={(event) => {
                      // Don't hijack right-click over the zoom toolbar's own buttons.
                      if ((event.target as HTMLElement).closest("[data-viewer-image-control]")) {
                        event.preventDefault();
                      }
                    }}
                  >
                    {canUseDeepZoom2D ? (
                      <DeepZoomCanvas
                        ref={canvasHandleRef}
                        apiClient={apiClient}
                        fileId={viewerInfo.file_id}
                        viewerInfo={viewerInfo}
                        axis="z"
                        zIndex={clampedIndices.z}
                        tIndex={clampedIndices.t}
                        channels={isDisplayPhoto ? undefined : selectedChannelIndices}
                        channelColors={
                          isDisplayPhoto || !supportsSliceChannelColor(viewerInfo)
                            ? undefined
                            : channelColors
                        }
                        cacheKey={isDisplayPhoto ? undefined : previewCacheKey}
                        className="viewer-canvas-root"
                      />
                    ) : (
                      <DirectPlaneImage
                        ref={canvasHandleRef}
                        imageUrl={direct2dImageUrl}
                        placeholderUrl={direct2dPreviewUrl}
                        descriptor={viewerInfo.viewer.default_plane}
                        title="2d-plane"
                        className="viewer-canvas-root"
                        interactive={true}
                        orientationLabels={getPlaneOrientationLabels(viewerInfo, "z")}
                        scalarSlice={direct2dScalarSlice}
                      />
                    )}
                  </div>
                </ContextMenuTrigger>
                <ContextMenuContent container={shellPortalContainer} className="viewer-context-menu">
                  <ContextMenuItem onSelect={handleResetView}>
                    <RotateCcw data-icon="inline-start" />
                    Reset view
                  </ContextMenuItem>
                  <ContextMenuItem onSelect={() => void handleCopyView()} disabled={!canCopyImageToClipboard()}>
                    <Copy data-icon="inline-start" />
                    Copy current view
                  </ContextMenuItem>
                  <ContextMenuItem onSelect={() => void handleExportView()}>
                    <ImageDown data-icon="inline-start" />
                    Export current view (PNG)
                  </ContextMenuItem>
                  <ContextMenuSeparator />
                  <ContextMenuItem onSelect={handleDownloadOriginal}>
                    <Download data-icon="inline-start" />
                    Download original image
                  </ContextMenuItem>
                  {metadataAvailable ? (
                    <ContextMenuItem onSelect={() => handleSurfaceChange("metadata")}>
                      <Info data-icon="inline-start" />
                      View metadata
                    </ContextMenuItem>
                  ) : null}
                  <ContextMenuItem onSelect={toggleFullscreen}>
                    {isFullscreen ? <Minimize2 data-icon="inline-start" /> : <Maximize2 data-icon="inline-start" />}
                    {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                  </ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
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
                {effectiveMaskMode && !clientMaskSliceEnabled ? (
                  <div
                    className="viewer-canvas-root viewer-mpr-canvas viewer-empty-state"
                    data-viewer-mpr-mask-unavailable={axis}
                  >
                    Mask Slice Views unavailable until the nearest scalar preview is ready.
                  </div>
                ) : (
                  <SlicePlaneCanvas
                    imageUrl={mprSliceUrls[axis]}
                    descriptor={viewerInfo.viewer.planes[axis]}
                    title={`${axis}-plane`}
                    className="viewer-canvas-root viewer-mpr-canvas"
                    orientationLabels={getPlaneOrientationLabels(viewerInfo, axis)}
                    crosshair={mapMprSourcePointToDisplay(
                      axis,
                      getPlaneCursor(axis, mprCrosshairIndices)
                    )}
                    measurement={
                      measurementsByAxis[axis]
                        ? {
                            start: mapMprSourcePointToDisplay(
                              axis,
                              measurementsByAxis[axis].start
                            ),
                            end: mapMprSourcePointToDisplay(
                              axis,
                              measurementsByAxis[axis].end
                            ),
                          }
                        : null
                    }
                    onSelectPoint={(point) =>
                      handlePlaneSelect(
                        axis,
                        mapMprDisplayPointToSource(axis, point)
                      )
                    }
                    onMeasurePoint={(point) =>
                      handlePlaneMeasure(
                        axis,
                        mapMprDisplayPointToSource(axis, point)
                      )
                    }
                    measureMode={measurementMode}
                    coordinateGrid={mprCoordinateGrid(axis)}
                    scalarSlice={mprScalarSlices[axis]}
                  />
                )}
              </article>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="volume" className="viewer-surface-panel">
          <div className="viewer-volume-layout">
            <div className="viewer-canvas-shell viewer-canvas-shell-volume">
              <SliceStackVolumeCanvas
                apiClient={apiClient}
                fileId={viewerInfo.file_id}
                viewerInfo={viewerInfo}
                xIndex={clampedIndices.x}
                yIndex={clampedIndices.y}
                zIndex={clampedIndices.z}
                tIndex={resolvedScalarRendering?.time ?? resolvedScalarTime}
                displayState={effectiveVolumeDisplayState}
                resolvedScalarRendering={resolvedScalarRendering}
              />
            </div>
            {selectedDisplayState ? (
              <div
                className="viewer-volume-inspection-toolbar"
                data-viewer-volume-inspection-toolbar="true"
                data-viewer-interior-active={volumeClipActive ? "true" : "false"}
              >
                <div className="viewer-volume-inspection-status">
                  {volumeClipActive ? "Cutaway active" : "Full volume"}
                </div>
                <div className="viewer-volume-inspection-actions">
                  <Button
                    type="button"
                    variant={volumeClipActive ? "secondary" : "outline"}
                    size="sm"
                    onClick={enableVolumeCutaway}
                    aria-pressed={volumeClipActive}
                  >
                    <Focus data-icon="inline-start" />
                    Cutaway
                  </Button>
                  {volumeClipActive ? (
                    <Button type="button" variant="ghost" size="sm" onClick={resetVolumeClip}>
                      <RotateCcw data-icon="inline-start" />
                      Full volume
                    </Button>
                  ) : null}
                </div>
                {cutawayActive && zAxisSize > 1 ? (
                  <label className="viewer-volume-cutaway-depth" data-viewer-cutaway-depth="true">
                    <span>Depth (Z)</span>
                    <input
                      type="range"
                      aria-label="Cutaway depth"
                      min={0}
                      max={Math.max(0, zAxisSize - 1)}
                      step={1}
                      value={clampedIndices.z}
                      onChange={(event) => setSelectedIndex("z", Number(event.target.value))}
                    />
                    <strong>{`${clampedIndices.z + 1}/${zAxisSize}`}</strong>
                  </label>
                ) : null}
              </div>
            ) : null}
          </div>
        </TabsContent>

        <TabsContent value="metadata" className="viewer-surface-panel">
          <div
            className="viewer-metadata-groups"
            data-viewer-metadata-summary="true"
            data-viewer-metadata-layout="groups"
            aria-label="Image metadata"
          >
            {metadataGroups.map((group) => (
              <section
                key={group.title}
                className="viewer-metadata-group"
                data-viewer-metadata-group={group.title}
              >
                <h3 className="viewer-metadata-group-title">{group.title}</h3>
                <dl className="viewer-metadata-kv">
                  {group.details.map((detail) => (
                    <div key={`${group.title}-${detail.label}`} className="viewer-metadata-kv-row">
                      <dt>{detail.label}</dt>
                      <MetadataDetailValue detail={detail} />
                    </div>
                  ))}
                </dl>
              </section>
            ))}
          </div>
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

      {selectedSurface !== "metadata" ? (
        <div className="viewer-caption viewer-chrome-fade" data-viewer-caption="true">
          <div className="viewer-caption-name">
            <span className="viewer-caption-filename" title={viewerInfo.original_name}>
              {viewerInfo.original_name}
            </span>
            {unstitchedMosaic ? (
              <span className="viewer-caption-badge" title={mosaicBadgeTitle}>
                Unstitched mosaic
              </span>
            ) : null}
          </div>
          {viewerInfo.is_volume ? (
            renderCompactSurfaceReadout(volumeSummaryRows, isSliceStackVolume ? "Stack summary" : "Volume summary")
          ) : captionMeta ? (
            <div className="viewer-caption-meta">{captionMeta}</div>
          ) : null}
        </div>
      ) : null}

      {!viewerInfo.is_volume && selectedDisplayState && canControlChannels && channelNames.length > 1 ? (
        <div
          className="viewer-display-controls viewer-display-controls-direct viewer-display-controls-channels"
          data-viewer-direct-channel-controls="true"
        >
          {renderChannelControls()}
        </div>
      ) : null}

      {/* Multichannel 3D volume: channel composition is the primary interaction, so
          surface the per-channel toggle/color strip directly (not buried in the
          Advanced panel). Toggling a channel loads/composites its full-res volume. */}
      {viewerInfo.is_volume &&
      selectedSurface === "volume" &&
      Boolean(viewerInfo.is_multichannel) &&
      viewerInfo.viewer.volume_mode !== "scalar" &&
      selectedDisplayState &&
      canControlChannels &&
      channelNames.length > 1 ? (
        <div
          className="viewer-display-controls viewer-display-controls-direct viewer-display-controls-channels"
          data-viewer-volume-channel-controls="true"
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
            <div className="viewer-inline-control viewer-inline-control-wide viewer-slider-field">
              <div className="viewer-slider-field-head">
                <span>Z slice</span>
                <strong>
                  {clampedIndices.z + 1}/{zAxisSize}
                </strong>
              </div>
              <Slider
                aria-label="Z slice"
                min={0}
                max={Math.max(0, zAxisSize - 1)}
                value={[clampedIndices.z]}
                onValueChange={(values) => setSelectedIndex("z", values[0] ?? clampedIndices.z)}
              />
            </div>
          ) : null}
          {tAxisSize > 1 ? (
            <div className="viewer-inline-control viewer-inline-control-wide viewer-slider-field">
              <div className="viewer-slider-field-head">
                <span>Time</span>
                <strong>
                  {clampedIndices.t + 1}/{tAxisSize}
                </strong>
              </div>
              <Slider
                aria-label="Time"
                min={0}
                max={Math.max(0, tAxisSize - 1)}
                value={[clampedIndices.t]}
                onValueChange={(values) => setSelectedIndex("t", values[0] ?? clampedIndices.t)}
              />
            </div>
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
              {effectiveMaskMode ? renderMaskCalibrationPanel() : null}
              {!effectiveMaskMode && volumeIntensityReady ? (
                <div className="viewer-volume-intensity-panel" data-viewer-volume-intensity-summary="true">
                  {renderIntensityHistogramPanel()}
                </div>
              ) : !effectiveMaskMode && uploadHistogramError ? (
                <div className="viewer-metadata-note" data-viewer-intensity-error="true">
                  <strong>Histogram unavailable</strong>
                  <span>{uploadHistogramError}</span>
                </div>
              ) : null}
              {!effectiveMaskMode && isScalarMpr ? (
                <>
                  {showCtWindowPresets ? (
                    <div
                      className="viewer-window-presets"
                      role="group"
                      aria-label="CT window presets"
                      data-viewer-window-presets="true"
                      data-viewer-active-window-preset={activeCtWindowPresetId ?? "custom"}
                    >
                      <span className="viewer-window-presets-label">Window presets</span>
                      <div className="viewer-window-presets-row">
                        {CT_WINDOW_PRESETS.map((preset) => {
                          const active = activeCtWindowPresetId === preset.id;
                          return (
                            <Button
                              key={preset.id}
                              type="button"
                              size="sm"
                              variant={active ? "secondary" : "outline"}
                              aria-pressed={active}
                              data-viewer-window-preset={preset.id}
                              data-active={active ? "true" : "false"}
                              title={`${preset.label} — center ${preset.center} / width ${preset.width} HU`}
                              onClick={() =>
                                updateSelectedDisplay({
                                  enhancement: buildWindowEnhancement(preset.center, preset.width),
                                })
                              }
                            >
                              {preset.label}
                            </Button>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}
                  <label className="viewer-inline-control">
                    <span>Window level</span>
                    <input
                      type="range"
                      aria-label="Window level"
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
                      aria-label="Window width"
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
              ) : !effectiveMaskMode ? (
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
              ) : null}
              {!effectiveMaskMode ? (
                <div className="viewer-inline-control">
                <span>{selectedSurface === "volume" ? "Projection" : "Fusion"}</span>
                {selectedSurface === "volume" ? (
                  // 3D ray-projection — its OWN setting, decoupled from the 2D
                  // `fusion_method`. Defaults to Composite (MIP flattens dense
                  // fluorescence into a cloud); MIP stays available for sparse data.
                  <Select
                    value={selectedDisplayState.volume_projection ?? "composite"}
                    onValueChange={(value) =>
                      updateSelectedDisplay({
                        volume_projection: value as ViewerDisplayState["volume_projection"],
                      })
                    }
                  >
                    <SelectTrigger aria-label="Projection" className="viewer-select-trigger">
                      <SelectValue placeholder="Projection" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="composite">Composite</SelectItem>
                        <SelectItem value="mip">MIP</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                ) : (
                  <Select
                    value={selectedDisplayState.fusion_method}
                    onValueChange={(value) =>
                      updateSelectedDisplay({
                        fusion_method: value as ViewerDisplayState["fusion_method"],
                      })
                    }
                  >
                    <SelectTrigger aria-label="Fusion" className="viewer-select-trigger">
                      <SelectValue placeholder="Fusion" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="m">Maximum</SelectItem>
                        <SelectItem value="a">Average</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                )}
                </div>
              ) : null}
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
              {canControlScalarVolumeColorMap && !effectiveMaskMode ? (
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
              {canControlScalarVolumeColorMap && !effectiveMaskMode ? (
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
                  {selectedSurface === "volume" &&
                  viewerInfo.is_volume &&
                  viewerInfo.viewer.volume_mode !== "scalar" ? (
                    <label className="viewer-inline-control viewer-inline-control-wide" data-viewer-volume-gamma-control="true">
                      <span>Gamma</span>
                      <input
                        type="range"
                        aria-label="Gamma"
                        min={0.3}
                        max={3}
                        step={0.05}
                        value={Number(selectedDisplayState?.volume_gamma) > 0 ? Number(selectedDisplayState.volume_gamma) : 1}
                        onChange={(event) => updateSelectedDisplay({ volume_gamma: Number(event.target.value) })}
                      />
                      <strong>
                        {(Number(selectedDisplayState?.volume_gamma) > 0
                          ? Number(selectedDisplayState.volume_gamma)
                          : 1
                        ).toFixed(2)}
                      </strong>
                    </label>
                  ) : null}
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
              {!effectiveMaskMode ? (
                <label className="viewer-inline-control viewer-inline-control-switch">
                  <span>Negative</span>
                  <Switch
                    checked={selectedDisplayState.negative}
                    onCheckedChange={(checked) => updateSelectedDisplay({ negative: checked })}
                  />
                </label>
              ) : null}
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
    </div>
  );
}
