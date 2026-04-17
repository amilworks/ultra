import { useState } from "react";

import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ApiClient } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

import { DirectPlaneImage } from "./DirectPlaneImage";
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

type ViewerDisplayState = NonNullable<UploadViewerInfo["display_defaults"]>;

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

const formatNumber = (value: number): string => value.toLocaleString();

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
  selectedCaption,
  captionLoading,
}: ImageViewerShellProps) {
  const [measurementMode, setMeasurementMode] = useState(false);
  const [measurementDraft, setMeasurementDraft] = useState<MeasurementDraft>(null);
  const [measurementsByAxis, setMeasurementsByAxis] = useState<Partial<Record<PlaneAxis, PlaneMeasurement>>>({});
  const [activeMeasurementAxis, setActiveMeasurementAxis] = useState<PlaneAxis>("z");

  const metadataLine = (() => {
    const shape = Array.isArray(viewerInfo.metadata.array_shape)
      ? viewerInfo.metadata.array_shape.join(" x ")
      : "unknown";
    const spacing = getSpacingSummary(viewerInfo);
    const orientation = getOrientationSummary(viewerInfo);
    return [
      `Axes ${viewerInfo.dims_order}`,
      `Shape ${shape}`,
      `${viewerInfo.metadata.array_dtype}`,
      `Range ${viewerInfo.metadata.array_min ?? 0} to ${viewerInfo.metadata.array_max ?? 0}`,
      spacing ? `Spacing ${spacing}` : null,
      orientation ? `Orientation ${orientation}` : null,
    ]
      .filter(Boolean)
      .join(" • ");
  })();

  const fallbackCaption = (() => {
    const plane = viewerInfo.viewer?.default_plane;
    if (!plane) {
      return "Caption: Native viewer metadata loaded.";
    }
    const surfaceLabel =
      viewerInfo.viewer.diagnostic_surface === "mpr"
        ? "measurement-ready scientific volume"
        : viewerInfo.is_volume
          ? "multiplanar scientific volume"
          : "direct scientific image";
    return `Caption: ${surfaceLabel} (${plane.pixel_size.width}×${plane.pixel_size.height}) prepared for the native viewer.`;
  })();

  const viewerContextText =
    selectedCaption ||
    (captionLoading ? "Caption: Generating metadata-based image caption…" : fallbackCaption || "Caption: Native viewer ready.");

  const metadataCards: MetadataCard[] = (() => {
    const cards: MetadataCard[] = [];
    const spacing = getSpacingSummary(viewerInfo);
    const orientation = getOrientationSummary(viewerInfo);
    const bitDepth = viewerInfo.phys?.pixel_depth;
    cards.push({
      label: "Pixel dimensions",
      value: `${formatNumber(viewerInfo.axis_sizes.X)} × ${formatNumber(viewerInfo.axis_sizes.Y)} px`,
    });
    cards.push({ label: "Axes order", value: viewerInfo.dims_order });
    cards.push({ label: "Stored shape", value: viewerInfo.metadata.array_shape.join(" × ") });
    cards.push({
      label: "Pixel type",
      value:
        typeof bitDepth === "number" && Number.isFinite(bitDepth) && bitDepth > 0
          ? `${viewerInfo.metadata.array_dtype} • ${bitDepth}-bit`
          : viewerInfo.metadata.array_dtype,
    });
    cards.push({ label: "Channels", value: `${formatNumber(viewerInfo.axis_sizes.C)}` });
    if (viewerInfo.axis_sizes.Z > 1) {
      cards.push({ label: "Z slices", value: `${formatNumber(viewerInfo.axis_sizes.Z)}` });
    }
    if (viewerInfo.axis_sizes.T > 1) {
      cards.push({ label: "Timepoints", value: `${formatNumber(viewerInfo.axis_sizes.T)}` });
    }
    if (spacing) {
      cards.push({ label: "Spacing", value: spacing });
    }
    if (orientation) {
      cards.push({ label: "Orientation", value: orientation });
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

  const displayCapabilities = new Set((viewerInfo.viewer.display_capabilities ?? []).map((value) => String(value)));
  const isScalarMpr =
    viewerInfo.viewer.render_policy === "scalar" && viewerInfo.viewer.diagnostic_surface === "mpr";
  const isMicroscopyChannelView =
    displayCapabilities.has("channel_visibility") || displayCapabilities.has("channel_color");
  const arrayMin = Number(viewerInfo.metadata.array_min ?? 0);
  const arrayMax = Number(viewerInfo.metadata.array_max ?? 1);
  const defaultCenter = Number(viewerInfo.metadata.dicom?.wnd_center ?? (arrayMin + arrayMax) / 2);
  const defaultWidth = Number(
    viewerInfo.metadata.dicom?.wnd_width ?? Math.max(1, Math.abs(arrayMax - arrayMin))
  );
  const parsedWindow = parseWindowLevel(selectedDisplayState?.enhancement, defaultCenter, defaultWidth);
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
  const volumeChannelIndex = Math.max(
    0,
    Math.min(
      Number(selectedDisplayState?.volume_channel ?? viewerInfo.selected_indices.C ?? 0),
      Math.max(0, channelNames.length - 1)
    )
  );

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
  });
  const direct2dDisplayUrl =
    viewerInfo.viewer.render_policy === "display" && viewerInfo.service_urls?.display
      ? apiClient.uploadDisplayUrl(viewerInfo.file_id, viewerInfo.service_urls.display)
      : null;
  const direct2dImageUrl = direct2dDisplayUrl ?? direct2dSliceUrl;
  const direct2dPreviewUrl = apiClient.uploadPreviewUrl(viewerInfo.file_id);

  const cursorReadoutRows = computeCursorWorldPosition(viewerInfo, clampedIndices);

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

  return (
    <>
      <Tabs value={selectedSurface} onValueChange={onSurfaceChange} className="viewer-surface-tabs">
        <div className="viewer-surface-toolbar">
          <TabsList className="viewer-surface-list">
            {viewerInfo.viewer.available_surfaces.map((surface) => (
              <TabsTrigger key={surface} value={surface}>
                {formatViewerSurfaceLabel(surface)}
              </TabsTrigger>
            ))}
          </TabsList>
          <HoverCard openDelay={120} closeDelay={80}>
            <HoverCardTrigger asChild>
              <Button type="button" variant="outline" size="sm" className="viewer-context-trigger">
                Context
              </Button>
            </HoverCardTrigger>
            <HoverCardContent align="end" className="viewer-context-card">
              <div className="viewer-context-card-grid">
                <div className="viewer-context-card-section">
                  <strong>Viewer</strong>
                  <p>{viewerContextText}</p>
                </div>
                <div className="viewer-context-card-section">
                  <strong>Source metadata</strong>
                  <p>{metadataLine}</p>
                </div>
              </div>
            </HoverCardContent>
          </HoverCard>
        </div>

        <TabsContent value="2d" className="viewer-surface-panel">
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
        </TabsContent>

        <TabsContent value="mpr" className="viewer-surface-panel">
          <div className="viewer-inspection-bar" data-viewer-mpr-tools="true">
            <section className="viewer-inspection-card" data-viewer-cursor-readout="true">
              <strong>Cursor</strong>
              <dl className="viewer-inspection-list">
                {cursorReadoutRows.map((row) => (
                  <div key={row.label} className="viewer-inspection-row">
                    <dt>{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
            <section className="viewer-inspection-card" data-viewer-measurement-readout="true">
              <div className="viewer-inspection-card-header">
                <strong>Measurement</strong>
                <div className="viewer-inspection-actions">
                  <label className="viewer-inline-control viewer-inline-control-switch">
                    <span>Measure</span>
                    <Switch checked={measurementMode} onCheckedChange={setMeasurementMode} />
                  </label>
                  {(activeMeasurement || measurementDraft) && (
                    <Button type="button" size="sm" variant="outline" onClick={clearMeasurements}>
                      Clear
                    </Button>
                  )}
                </div>
              </div>
              <dl className="viewer-inspection-list">
                <div className="viewer-inspection-row">
                  <dt>Plane</dt>
                  <dd>{viewerInfo.viewer.planes[activeMeasurementAxis]?.label ?? activeMeasurementAxis.toUpperCase()}</dd>
                </div>
                <div className="viewer-inspection-row">
                  <dt>Status</dt>
                  <dd>
                    {measurementDraft
                      ? "Select the second point."
                      : activeMeasurement
                        ? "Measurement ready."
                        : "Enable Measure, then click two points."}
                  </dd>
                </div>
                {activeMeasurementDistance ? (
                  <div className="viewer-inspection-row">
                    <dt>Distance</dt>
                    <dd>{activeMeasurementDistance}</dd>
                  </div>
                ) : null}
              </dl>
            </section>
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
          <div className="viewer-canvas-shell viewer-canvas-shell-volume">
            <SliceStackVolumeCanvas
              apiClient={apiClient}
              fileId={viewerInfo.file_id}
              viewerInfo={viewerInfo}
              zIndex={debouncedZ}
              tIndex={debouncedT}
              displayState={selectedDisplayState}
            />
          </div>
        </TabsContent>

        <TabsContent value="metadata" className="viewer-surface-panel">
          <div className="viewer-metadata-grid">
            {metadataCards.map((card) => (
              <div key={card.label} className="viewer-metadata-card">
                <strong>{card.label}</strong>
                <span>{card.value}</span>
              </div>
            ))}
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
          {viewerInfo.metadata.warnings.length > 0 ? (
            <div className="viewer-metadata-note">
              <strong>Viewer notes</strong>
              <span>{viewerInfo.metadata.warnings.join(" ")}</span>
            </div>
          ) : null}
        </TabsContent>
      </Tabs>

      {viewerInfo.is_volume && selectedDisplayState ? (
        <div className="viewer-display-controls">
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
            <label className="viewer-inline-control">
              <span>Enhancement</span>
              <select
                value={selectedDisplayState.enhancement}
                onChange={(event) => updateSelectedDisplay({ enhancement: event.target.value })}
              >
                <option value="d">Dynamic</option>
                <option value="f">Full range</option>
                {viewerInfo.display_defaults?.enhancement?.startsWith("hounsfield") ? (
                  <option value={viewerInfo.display_defaults.enhancement}>DICOM window</option>
                ) : null}
              </select>
            </label>
          )}
          <label className="viewer-inline-control">
            <span>Fusion</span>
            <select
              value={selectedDisplayState.fusion_method}
              onChange={(event) =>
                updateSelectedDisplay({
                  fusion_method: event.target.value as ViewerDisplayState["fusion_method"],
                })
              }
            >
              <option value="m">Maximum</option>
              <option value="a">Average</option>
            </select>
          </label>
          <label className="viewer-inline-control viewer-inline-control-switch">
            <span>Negative</span>
            <Switch
              checked={selectedDisplayState.negative}
              onCheckedChange={(checked) => updateSelectedDisplay({ negative: checked })}
            />
          </label>
          {selectedSurface === "volume" &&
          viewerInfo.viewer.volume_mode === "scalar" &&
          isMicroscopyChannelView &&
          channelNames.length > 1 ? (
            <label className="viewer-inline-control" data-viewer-volume-channel-control="true">
              <span>Volume channel</span>
              <select
                value={String(volumeChannelIndex)}
                onChange={(event) =>
                  updateSelectedDisplay({
                    volume_channel: Number(event.target.value),
                  })
                }
              >
                {channelNames.map((label, index) => (
                  <option key={`${label}-${index}`} value={String(index)}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
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
          {isMicroscopyChannelView ? (
            <div className="viewer-channel-controls" data-viewer-channel-controls="true">
              {channelNames.map((label, index) => {
                const active = (selectedDisplayState.channels ?? []).includes(index);
                return (
                  <div key={label} className="viewer-channel-chip">
                    <button
                      type="button"
                      className={active ? "viewer-channel-toggle is-active" : "viewer-channel-toggle"}
                      onClick={() => {
                        const current = new Set(selectedDisplayState.channels ?? []);
                        if (active && current.size > 1) {
                          current.delete(index);
                        } else if (!active) {
                          current.add(index);
                        }
                        updateSelectedDisplay({ channels: Array.from(current).sort((a, b) => a - b) });
                      }}
                    >
                      <span
                        className="viewer-channel-swatch"
                        style={{ backgroundColor: channelColors[index] }}
                        aria-hidden="true"
                      />
                      {label}
                    </button>
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
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      ) : null}

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
    </>
  );
}
