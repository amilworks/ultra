import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerHistogramResponse, UploadViewerInfo } from "@/types";

import { ImageViewerShell } from "./ImageViewerShell";

vi.mock("./DirectPlaneImage", () => ({
  DirectPlaneImage: ({
    imageUrl,
    scalarSlice,
  }: {
    imageUrl: string;
    scalarSlice?: { sliceIndex: number } | null;
  }) => (
    <div
      data-testid="direct-plane-image"
      data-image-url={imageUrl}
      data-scalar-slice-index={scalarSlice?.sliceIndex ?? ""}
    />
  ),
}));

vi.mock("./DeepZoomCanvas", () => ({
  DeepZoomCanvas: ({
    apiClient,
    fileId,
    viewerInfo,
    axis = "z",
    zIndex,
    tIndex,
    channels,
    channelColors,
    cacheKey,
  }: {
    apiClient: ApiClient;
    fileId: string;
    viewerInfo: UploadViewerInfo;
    axis?: "z" | "y" | "x";
    zIndex: number;
    tIndex: number;
    channels?: number[];
    channelColors?: string[];
    cacheKey?: string;
  }) => (
    <div
      data-testid="deep-zoom-canvas"
      data-file-id={fileId}
      data-level-count={String(viewerInfo.viewer.tile_scheme.levels.length)}
      data-z-index={String(zIndex)}
      data-t-index={String(tIndex)}
      data-channels={channels?.join(",") ?? ""}
      data-cache-key={cacheKey ?? ""}
      data-tile-url={apiClient.uploadTileUrl(fileId, {
        axis,
        level: viewerInfo.viewer.tile_scheme.levels[0]?.level ?? 0,
        tileX: 0,
        tileY: 0,
        z: zIndex,
        t: tIndex,
        channels,
        channelColors,
        cacheKey,
      })}
    />
  ),
}));

vi.mock("./SlicePlaneCanvas", () => ({
  SlicePlaneCanvas: ({
    imageUrl,
    title,
    scalarSlice,
    crosshair,
    coordinateGrid,
    measureMode,
    onMeasurePoint,
  }: {
    imageUrl: string;
    title: string;
    scalarSlice?: { sliceIndex: number } | null;
    crosshair?: { row: number; col: number };
    coordinateGrid?: { width: number; height: number } | null;
    measureMode?: boolean;
    onMeasurePoint?: (point: { row: number; col: number }) => void;
  }) => (
    <div
      data-testid="slice-plane-canvas"
      data-image-url={imageUrl}
      data-title={title}
      data-scalar-slice-index={scalarSlice?.sliceIndex ?? ""}
      data-crosshair-row={crosshair?.row ?? ""}
      data-crosshair-col={crosshair?.col ?? ""}
      data-coordinate-grid-width={coordinateGrid?.width ?? ""}
      data-coordinate-grid-height={coordinateGrid?.height ?? ""}
      data-measure-mode={measureMode ? "true" : "false"}
    >
      <button type="button" aria-label={`measure ${title} start`} onClick={() => onMeasurePoint?.({ row: 0, col: 0 })} />
      <button type="button" aria-label={`measure ${title} end`} onClick={() => onMeasurePoint?.({ row: 1, col: 1 })} />
    </div>
  ),
}));

vi.mock("./SliceStackVolumeCanvas", () => ({
  SliceStackVolumeCanvas: ({
    displayState,
    xIndex,
    yIndex,
    zIndex,
    tIndex,
  }: {
    displayState?: {
      scalar_colormap?: string;
      volume_signal_floor?: number;
      volume_density?: number;
      volume_lighting?: boolean;
      volume_lighting_strength?: number;
      volume_view_preset?: string;
      volume_camera_mode?: string;
      volume_clip_min?: { x: number; y: number; z: number };
      volume_clip_max?: { x: number; y: number; z: number };
      volume_cutaway?: boolean | null;
      scalar_render_mode?: "auto" | "intensity" | "mask";
      scalar_threshold_method?: "otsu-256-v1" | "manual";
      scalar_threshold_value?: number | null;
    } | null;
    xIndex?: number;
    yIndex?: number;
    zIndex?: number;
    tIndex?: number;
  }) => (
    <div
      data-testid="slice-stack-volume-canvas"
      data-cutaway={displayState?.volume_cutaway ? "true" : "false"}
      data-scalar-render-mode={displayState?.scalar_render_mode ?? ""}
      data-scalar-threshold-method={displayState?.scalar_threshold_method ?? ""}
      data-scalar-threshold-value={
        displayState?.scalar_threshold_value == null ? "" : String(displayState.scalar_threshold_value)
      }
      data-scalar-colormap={displayState?.scalar_colormap ?? ""}
      data-signal-floor={displayState?.volume_signal_floor == null ? "" : String(displayState.volume_signal_floor)}
      data-density={displayState?.volume_density == null ? "" : String(displayState.volume_density)}
      data-lighting={displayState?.volume_lighting == null ? "" : String(displayState.volume_lighting)}
      data-lighting-strength={
        displayState?.volume_lighting_strength == null ? "" : String(displayState.volume_lighting_strength)
      }
      data-view-preset={displayState?.volume_view_preset ?? ""}
      data-camera-mode={displayState?.volume_camera_mode ?? ""}
      data-clip-x-min={displayState?.volume_clip_min == null ? "" : String(displayState.volume_clip_min.x)}
      data-clip-x-max={displayState?.volume_clip_max == null ? "" : String(displayState.volume_clip_max.x)}
      data-clip-y-min={displayState?.volume_clip_min == null ? "" : String(displayState.volume_clip_min.y)}
      data-clip-y-max={displayState?.volume_clip_max == null ? "" : String(displayState.volume_clip_max.y)}
      data-clip-z-min={displayState?.volume_clip_min == null ? "" : String(displayState.volume_clip_min.z)}
      data-clip-z-max={displayState?.volume_clip_max == null ? "" : String(displayState.volume_clip_max.z)}
      data-x-index={xIndex == null ? "" : String(xIndex)}
      data-y-index={yIndex == null ? "" : String(yIndex)}
      data-z-index={zIndex == null ? "" : String(zIndex)}
      data-t-index={tIndex == null ? "" : String(tIndex)}
    />
  ),
}));

if (!HTMLElement.prototype.hasPointerCapture) {
  Object.defineProperty(HTMLElement.prototype, "hasPointerCapture", {
    configurable: true,
    value: () => false,
  });
}

if (!HTMLElement.prototype.setPointerCapture) {
  Object.defineProperty(HTMLElement.prototype, "setPointerCapture", {
    configurable: true,
    value: () => undefined,
  });
}

if (!HTMLElement.prototype.releasePointerCapture) {
  Object.defineProperty(HTMLElement.prototype, "releasePointerCapture", {
    configurable: true,
    value: () => undefined,
  });
}

if (!HTMLElement.prototype.scrollIntoView) {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    value: () => undefined,
  });
}

const defaultPlane = {
  axis: "z" as const,
  label: "XY",
  axes: ["Y", "X"],
  pixel_size: { width: 4, height: 1 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 4, height: 1 },
  aspect_ratio: 4,
};

const viewerInfo: UploadViewerInfo = {
  kind: "image",
  file_id: "file-123",
  original_name: "histology.ome.tiff",
  modality: "microscopy",
  dims_order: "YX",
  backend_mode: "direct",
  axis_sizes: { T: 1, C: 1, Z: 1, Y: 1, X: 4 },
  selected_indices: { T: 0, C: 0, Z: 0 },
  is_volume: false,
  is_timeseries: false,
  is_multichannel: false,
  display_defaults: {
    enhancement: "d",
    negative: false,
    rotate: 0,
    fusion_method: "m",
    channel_mode: "single",
    channels: [0],
    channel_colors: ["#ffffff"],
    time_index: 0,
    z_index: 0,
  },
  service_urls: {
    preview: "/v2/uploads/file-123/preview",
    display: "/v2/uploads/file-123/display",
    slice: "/v2/uploads/file-123/slice",
    histogram: "/v2/uploads/file-123/histogram",
  },
  metadata: {
    reader: "go-image+tiff",
    dims_order: "YX",
    array_shape: [1, 4],
    array_dtype: "uint16",
    sha256: "abc123",
    scene_count: 1,
    warnings: [],
  },
  viewer: {
    status: "preview-ready",
    warmup_mode: "deferred",
    backend_mode: "direct",
    default_surface: "2d",
    available_surfaces: ["2d", "metadata"],
    default_axis: "z",
    slice_axes: ["z"],
    channel_mode: "single",
    tile_scheme: { tile_size: 256, format: "png", levels: [] },
    default_plane: defaultPlane,
    planes: { z: defaultPlane },
    volume_mode: "none",
    render_policy: "scalar",
    delivery_mode: "direct",
    diagnostic_surface: "none",
    first_paint_mode: "image",
    measurement_policy: "pixel-only",
    texture_policy: "linear",
    display_capabilities: ["intensity_window", "histogram"],
    viewer_capabilities: ["2d", "metadata"],
    orientation: {
      frame: "pixel",
      row_axis: "Y",
      col_axis: "X",
      slice_axis: null,
    },
  },
};

const histogram: UploadViewerHistogramResponse = {
  file_id: "file-123",
  bins: 4,
  dtype: "uint16",
  source: "decoded-image",
  sample_count: 4,
  channels: [0],
  histogram: {
    bins: [1, 1, 1, 1],
    edges: [1000, 1001, 1002, 1003, 1004],
    min: 1000,
    max: 1003,
    channel_indices: [0],
    time_index: 0,
  },
};

const physicalScalarViewerInfo = (
  spacingUnits: { x: string; y: string; z: string }
): UploadViewerInfo => {
  const planeZ = {
    axis: "z" as const,
    label: "XY plane",
    axes: ["Y", "X"],
    pixel_size: { width: 64, height: 32 },
    spacing: { row: 0.5, col: 0.5 },
    world_size: { width: 32, height: 16 },
    aspect_ratio: 2,
  };
  return {
    ...viewerInfo,
    original_name: "physical-volume.ome.tiff",
    modality: "microscopy",
    dims_order: "ZYX",
    backend_mode: "scalar",
    axis_sizes: { T: 1, C: 1, Z: 12, Y: 32, X: 64 },
    selected_indices: { T: 0, C: 0, Z: 6 },
    is_volume: true,
    display_defaults: {
      ...(viewerInfo.display_defaults as NonNullable<UploadViewerInfo["display_defaults"]>),
      channels: [0],
      channel_colors: ["#ffffff"],
      volume_channel: 0,
    },
    metadata: {
      ...viewerInfo.metadata,
      dims_order: "ZYX",
      array_shape: [12, 32, 64],
      physical_spacing: { x: 0.5, y: 0.5, z: 2 },
      physical_spacing_unit:
        spacingUnits.x === spacingUnits.y && spacingUnits.y === spacingUnits.z
          ? spacingUnits.x
          : null,
      spacing_units: spacingUnits,
    },
    viewer: {
      ...viewerInfo.viewer,
      available_surfaces: ["2d", "mpr", "volume", "metadata"],
      default_surface: "volume",
      default_axis: "z",
      slice_axes: ["z", "y", "x"],
      default_plane: planeZ,
      planes: {
        z: planeZ,
        y: {
          ...planeZ,
          axis: "y" as const,
          label: "XZ plane",
          axes: ["Z", "X"],
          pixel_size: { width: 64, height: 12 },
          spacing: { row: 2, col: 0.5 },
          world_size: { width: 32, height: 24 },
          aspect_ratio: 4 / 3,
        },
        x: {
          ...planeZ,
          axis: "x" as const,
          label: "YZ plane",
          axes: ["Z", "Y"],
          pixel_size: { width: 32, height: 12 },
          spacing: { row: 2, col: 0.5 },
          world_size: { width: 16, height: 24 },
          aspect_ratio: 2 / 3,
        },
      },
      volume_mode: "scalar",
      render_policy: "scalar",
      diagnostic_surface: "mpr",
      measurement_policy: "spacing-aware",
      display_capabilities: ["physical_scale", "diagnostic_mpr"],
      viewer_capabilities: ["volume", "metadata", "physical_scale"],
    },
  };
};

const renderPhysicalViewer = (
  info: UploadViewerInfo,
  selectedSurface: "mpr" | "volume"
) =>
  render(
    <ImageViewerShell
      viewerInfo={info}
      apiClient={
        {
          getUploadHistogram: vi.fn(async () => histogram),
          uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
          uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
        } as unknown as ApiClient
      }
      selectedSurface={selectedSurface}
      onSurfaceChange={() => {}}
      selectedDisplayState={info.display_defaults ?? null}
      updateSelectedDisplay={() => {}}
      clampedIndices={{ x: 0, y: 0, z: 6, t: 0 }}
      debouncedX={0}
      debouncedY={0}
      debouncedZ={6}
      debouncedT={0}
      xAxisSize={64}
      yAxisSize={32}
      zAxisSize={12}
      tAxisSize={1}
      setSelectedIndex={() => {}}
      selectedCaption=""
      captionLoading={false}
    />
  );

type DisplayUrlConfig = {
  enhancement?: string;
  negative?: boolean;
  gamma?: number | null;
  cacheKey?: string;
  channels?: number[];
  channelColors?: string[];
};

const buildDisplayUrl = (
  fileId: string,
  explicitPath?: string | null,
  config?: DisplayUrlConfig
) => {
  const params = new URLSearchParams();
  if (config?.enhancement) {
    params.set("enhancement", config.enhancement);
  }
  if (typeof config?.negative === "boolean") {
    params.set("negative", config.negative ? "true" : "false");
  }
  if (typeof config?.gamma === "number") {
    params.set("gamma", String(config.gamma));
  }
  if (Array.isArray(config?.channels) && config.channels.length > 0) {
    params.set("channels", config.channels.join(","));
    const projected = config.channels.map((index) => config.channelColors?.[index] ?? "");
    if (projected.every(Boolean)) {
      params.set("channel_colors", projected.join(","));
    }
  }
  if (Array.isArray(config?.channelColors) && config.channelColors.length > 0) {
    params.set("channel_colors", config.channelColors.join(","));
  }
  if (config?.cacheKey) {
    params.set("cache_key", config.cacheKey);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return `https://ultra.example.org${explicitPath ?? `/v2/uploads/${fileId}/display`}${suffix}`;
};

type SliceUrlConfig = DisplayUrlConfig & {
  axis?: string;
  x?: number;
  y?: number;
  z?: number;
  t?: number;
  fullResolution?: boolean;
  scalarRenderMode?: "intensity" | "mask";
  scalarThresholdValue?: number;
};

const buildSliceUrl = (fileId: string, config?: SliceUrlConfig) => {
  const params = new URLSearchParams();
  if (config?.axis) {
    params.set("axis", config.axis);
  }
  for (const axis of ["x", "y", "z", "t"] as const) {
    if (typeof config?.[axis] === "number") {
      params.set(axis, String(config[axis]));
    }
  }
  if (config?.enhancement) {
    params.set("enhancement", config.enhancement);
  }
  if (typeof config?.negative === "boolean") {
    params.set("negative", config.negative ? "true" : "false");
  }
  if (Array.isArray(config?.channels) && config.channels.length > 0) {
    params.set("channels", config.channels.join(","));
    const projected = config.channels.map((index) => config.channelColors?.[index] ?? "");
    if (projected.every(Boolean)) {
      params.set("channel_colors", projected.join(","));
    }
  }
  if (config?.cacheKey) {
    params.set("cache_key", config.cacheKey);
  }
  if (config?.scalarRenderMode) {
    params.set("scalar_render_mode", config.scalarRenderMode);
  }
  if (typeof config?.scalarThresholdValue === "number") {
    params.set("scalar_threshold_value", String(config.scalarThresholdValue));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return `https://ultra.example.org/v2/uploads/${fileId}/slice${suffix}`;
};

type TileUrlConfig = DisplayUrlConfig & {
  axis: "z" | "y" | "x";
  level: number;
  tileX: number;
  tileY: number;
  z?: number;
  t?: number;
};

const buildTileUrl = (fileId: string, config: TileUrlConfig) => {
  const params = new URLSearchParams();
  if (typeof config.z === "number") params.set("z", String(config.z));
  if (typeof config.t === "number") params.set("t", String(config.t));
  if (config.channels?.length) {
    params.set("channels", config.channels.join(","));
    const projected = config.channels.map((index) => config.channelColors?.[index] ?? "");
    if (projected.every(Boolean)) params.set("channel_colors", projected.join(","));
  }
  if (config.cacheKey) params.set("cache_key", config.cacheKey);
  return `https://ultra.example.org/v2/uploads/${fileId}/tiles/${config.axis}/${config.level}/${config.tileX}/${config.tileY}?${params.toString()}`;
};

const openAdvancedControls = () => {
  fireEvent.click(screen.getByRole("button", { name: "Advanced rendering" }));
};

const chooseSelectOption = async (name: string, optionName: string): Promise<void> => {
  fireEvent.pointerDown(screen.getByRole("combobox", { name }), {
    button: 0,
    ctrlKey: false,
    pointerId: 1,
    pointerType: "mouse",
  });
  fireEvent.click(await screen.findByRole("option", { name: optionName }));
};

describe("ImageViewerShell", () => {
  const maskMprPlanes = {
    z: {
      axis: "z" as const,
      label: "Axial",
      axes: ["Y", "X"],
      pixel_size: { width: 10, height: 9 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 10, height: 9 },
      aspect_ratio: 10 / 9,
    },
    y: {
      axis: "y" as const,
      label: "Coronal",
      axes: ["Z", "X"],
      pixel_size: { width: 10, height: 7 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 10, height: 7 },
      aspect_ratio: 10 / 7,
    },
    x: {
      axis: "x" as const,
      label: "Sagittal",
      axes: ["Z", "Y"],
      pixel_size: { width: 9, height: 7 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 9, height: 7 },
      aspect_ratio: 9 / 7,
    },
  };
  const maskMprThreshold = {
    method: "otsu-256-v1" as const,
    value: 120,
    domain: "raw" as const,
    foreground: "above" as const,
    sample_scope: "volume",
    sample_count: 36,
    z_samples: [0, 3, 6],
    channel: 1,
    t: 1,
    sampling_algorithm: "scalar-profile-otsu-256-v1",
  };
  const makeMaskMprViewerInfo = (): UploadViewerInfo => ({
    ...viewerInfo,
    file_id: "file-mask-mpr",
    original_name: "mask.ome.tiff",
    modality: "microscopy",
    dims_order: "TCZYX",
    backend_mode: "direct",
    axis_sizes: { T: 2, C: 2, Z: 7, Y: 9, X: 10 },
    selected_indices: { T: 1, C: 1, Z: 6 },
    is_volume: true,
    is_timeseries: true,
    is_multichannel: true,
    display_defaults: {
      ...(viewerInfo.display_defaults as NonNullable<UploadViewerInfo["display_defaults"]>),
      channels: [1],
      time_index: 1,
      z_index: 6,
      scalar_render_mode: "mask",
      scalar_threshold_method: "otsu-256-v1",
      scalar_threshold_value: 120,
      scalar_threshold_foreground: "above",
    },
    metadata: {
      ...viewerInfo.metadata,
      reader: "tifffile",
      dims_order: "TCZYX",
      array_shape: [2, 2, 7, 9, 10],
      array_dtype: "uint16",
      sha256: "mask-mpr-sha",
    },
    data_semantics: {
      kind: "binary_mask",
      basis: "bounded_scalar_profile",
      strength: "exact",
      supported_modes: ["intensity", "mask"],
      recommended_view: "mask",
      threshold: maskMprThreshold,
    },
    scalar_mask_capability: {
      version: 1,
      source_authority: "original",
      source_format: "ome-tiff",
      source_sha256: "mask-mpr-sha",
      dtype: "uint16",
      threshold_domain: "raw",
      threshold_foreground: "above",
      slice_delivery: "thresholded_png",
      volume_delivery: "raw_scalar",
      volume_sampling: "nearest",
      channel_selection: "single",
      time_selection: "single",
      surfaces: ["2d", "mpr", "volume"],
    },
    service_urls: {
      ...viewerInfo.service_urls,
      histogram: undefined,
      scalar_volume: "/v2/uploads/file-mask-mpr/scalar-volume",
    },
    viewer: {
      ...viewerInfo.viewer,
      default_surface: "mpr",
      available_surfaces: ["2d", "mpr", "volume", "metadata"],
      default_axis: "z",
      slice_axes: ["z", "y", "x"],
      default_plane: maskMprPlanes.z,
      planes: maskMprPlanes,
      volume_mode: "slice_stack",
      render_policy: "scalar",
      diagnostic_surface: "none",
      display_capabilities: ["channel_visibility"],
      viewer_capabilities: ["2d", "mpr", "volume", "metadata"],
      service_urls: {
        slice: "/v2/uploads/file-mask-mpr/slice",
        scalar_volume: "/v2/uploads/file-mask-mpr/scalar-volume",
      },
    },
  });
  const makeMaskMprPayload = (
    overrides: Partial<ScalarVolumePayload> = {}
  ): ScalarVolumePayload => {
    const sourceWidth = overrides.sourceWidth ?? 10;
    const sourceHeight = overrides.sourceHeight ?? 9;
    const sourceDepth = overrides.sourceDepth ?? 7;
    const width = overrides.width ?? sourceWidth;
    const height = overrides.height ?? sourceHeight;
    const depth = overrides.depth ?? sourceDepth;
    return {
      data:
        overrides.data ??
        new Uint16Array(width * height * depth)
          .map((_, index) => index + 100)
          .buffer,
      width,
      height,
      depth,
      dtype: "uint16",
      bytesPerVoxel: 2,
      rawMin: 100,
      rawMax: 100 + width * height * depth - 1,
      channel: 1,
      time: 1,
      sourceWidth,
      sourceHeight,
      sourceDepth,
      downsampleX: 1,
      downsampleY: 1,
      downsampleZ: 1,
      previewPolicy: "mask-native-integer-v1",
      sampling: "nearest",
      sclSlope: 1,
      sclInter: 0,
      ...overrides,
    };
  };
  const makeMaskMprShellProps = (
    maskViewerInfo: UploadViewerInfo,
    apiClient: ApiClient
  ) => ({
    viewerInfo: maskViewerInfo,
    apiClient,
    onSurfaceChange: () => {},
    selectedDisplayState: maskViewerInfo.display_defaults ?? null,
    updateSelectedDisplay: () => {},
    clampedIndices: { x: 4, y: 5, z: 6, t: 1 },
    debouncedX: 4,
    debouncedY: 5,
    debouncedZ: 6,
    debouncedT: 1,
    xAxisSize: 10,
    yAxisSize: 9,
    zAxisSize: 7,
    tAxisSize: 2,
    setSelectedIndex: () => {},
    selectedCaption: "",
    captionLoading: false,
  });
  const makeIntensityMprViewerInfo = (): UploadViewerInfo => ({
    ...viewerInfo,
    file_id: "file-box-mpr",
    original_name: "intensity.nii",
    modality: "medical",
    dims_order: "ZYX",
    backend_mode: "scalar",
    axis_sizes: { T: 1, C: 1, Z: 7, Y: 9, X: 10 },
    selected_indices: { T: 0, C: 0, Z: 6 },
    is_volume: true,
    is_timeseries: false,
    is_multichannel: false,
    display_defaults: {
      ...(viewerInfo.display_defaults as NonNullable<UploadViewerInfo["display_defaults"]>),
      enhancement: "hounsfield:120:240",
      channels: [0],
      time_index: 0,
      z_index: 6,
      volume_channel: 0,
    },
    service_urls: {
      ...viewerInfo.service_urls,
      histogram: undefined,
      scalar_volume: "/v2/uploads/file-box-mpr/scalar-volume",
    },
    metadata: {
      ...viewerInfo.metadata,
      reader: "nifti",
      dims_order: "ZYX",
      array_shape: [7, 9, 10],
      array_dtype: "uint16",
      sha256: "box-mpr-sha",
    },
    viewer: {
      ...viewerInfo.viewer,
      default_surface: "mpr",
      available_surfaces: ["mpr", "volume", "metadata"],
      default_axis: "z",
      slice_axes: ["z", "y", "x"],
      default_plane: maskMprPlanes.z,
      planes: maskMprPlanes,
      volume_mode: "scalar",
      render_policy: "scalar",
      diagnostic_surface: "mpr",
      display_capabilities: ["scalar_probe"],
      viewer_capabilities: ["mpr", "volume", "metadata"],
    },
  });

  it("calibrates mask rendering with exact volume provenance and saves a SHA-bound default", async () => {
    const maskHistogram: UploadViewerHistogramResponse = {
      file_id: "file-mask",
      bins: 4,
      dtype: "uint8",
      channels: [1],
      source: "image-service-source",
      sample_count: 8,
      scope: "volume",
      channel: 1,
      t: 0,
      sampling: {
        algorithm: "scalar-profile-otsu-256-v1",
        scope: "volume",
        strategy: "exact",
        sample_count: 8,
        z_samples: [0, 1],
      },
      threshold: {
        method: "otsu-256-v1",
        value: 120,
        domain: "raw",
        foreground: "above",
        sample_scope: "volume",
        sample_count: 8,
        z_samples: [0, 1],
        channel: 1,
        t: 0,
        sampling_algorithm: "scalar-profile-otsu-256-v1",
        sampling_strategy: "exact",
        source_sha256: "mask-sha-256",
        bins: 256,
      },
      histogram: {
        bins: [4, 1, 1, 2],
        edges: [0, 64, 128, 192, 256],
        min: 0,
        max: 255,
        channel_indices: [1],
        time_index: 0,
      },
    };
    const maskViewer: UploadViewerInfo = {
      ...viewerInfo,
      file_id: "file-mask",
      original_name: "mask.tif",
      dims_order: "ZYX",
      axis_sizes: { T: 3, C: 2, Z: 2, Y: 2, X: 2 },
      is_volume: true,
      is_timeseries: true,
      is_multichannel: true,
      selected_indices: { T: 0, C: 1, Z: 0 },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint8",
        sha256: "mask-sha-256",
      },
      data_semantics: {
        kind: "binary_mask",
        basis: "exact_two_code_volume",
        strength: "exact",
        supported_modes: ["intensity", "mask"],
        recommended_view: "mask",
        threshold: {
          method: "otsu-256-v1",
          value: 120,
          domain: "raw",
          foreground: "above",
          sample_scope: "volume",
          sample_count: 8,
          z_samples: [0, 1],
          channel: 1,
          t: 0,
          sampling_algorithm: "scalar-profile-otsu-256-v1",
        },
      },
      scalar_mask_capability: {
        version: 1,
        source_authority: "original",
        source_format: "tiff",
        source_sha256: "mask-sha-256",
        dtype: "uint8",
        threshold_domain: "raw",
        threshold_foreground: "above",
        slice_delivery: "thresholded_png",
        volume_delivery: "raw_scalar",
        volume_sampling: "nearest",
        channel_selection: "single",
        time_selection: "single",
        surfaces: ["2d", "mpr", "volume"],
      },
      display_defaults: {
        ...(viewerInfo.display_defaults as NonNullable<UploadViewerInfo["display_defaults"]>),
        scalar_render_mode: "auto",
        scalar_threshold_method: "otsu-256-v1",
        scalar_threshold_value: 120,
        scalar_threshold_foreground: "above",
        channels: [1],
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-mask/scalar-volume",
      },
      viewer: {
        ...viewerInfo.viewer,
        default_surface: "volume",
        available_surfaces: ["2d", "mpr", "volume", "metadata"],
        volume_mode: "slice_stack",
        service_urls: {
          slice: "/v2/uploads/file-mask/slice",
          scalar_volume: "/v2/uploads/file-mask/scalar-volume",
        },
      },
    };
    const uploadSliceUrl = vi.fn(buildSliceUrl);
    let persistedSelections: Record<string, unknown> = {};
    let resolveTime2Histogram:
      | ((value: UploadViewerHistogramResponse) => void)
      | undefined;
    let time2HistogramResolved = false;
    let deferCalibrationSaves = false;
    const pendingCalibrationSaves: Array<{
      selectionId: string;
      resolve: () => void;
    }> = [];
    const histogramForTime = (time: number): UploadViewerHistogramResponse => {
      const threshold = time === 2 ? 220 : 120;
      return {
        ...maskHistogram,
        t: time,
        threshold: {
          ...maskHistogram.threshold!,
          value: threshold,
          t: time,
        },
        histogram: {
          ...maskHistogram.histogram,
          time_index: time,
        },
        data_semantics: {
          ...maskViewer.data_semantics!,
          threshold: {
            ...maskViewer.data_semantics!.threshold!,
            value: threshold,
            t: time,
          },
        },
      };
    };
    const apiClient = {
      getUploadHistogram: vi.fn(
        async (_fileId: string, config?: { t?: number }) => {
          const time = config?.t ?? 0;
          if (time === 1) {
            throw new Error("selection histogram rejected");
          }
          if (time === 2 && !time2HistogramResolved) {
            return await new Promise<UploadViewerHistogramResponse>((resolve) => {
              resolveTime2Histogram = resolve;
            });
          }
          return histogramForTime(time);
        }
      ),
      uploadSliceUrl,
      uploadPreviewUrl: vi.fn(() => "/preview"),
      uploadAtlasUrl: vi.fn(() => "/atlas"),
      patchResourceMetadata: vi.fn(
        async (_fileId: string, metadata: Record<string, unknown>) => ({
          resource: {
            metadata: {
              ...metadata,
              ultra_viewer_calibration_v1: {
                ...(metadata.ultra_viewer_calibration_v1 as Record<string, unknown>),
                selections: {
                  ...persistedSelections,
                  ...(
                    (metadata.ultra_viewer_calibration_v1 as Record<string, unknown>)
                      .selections as Record<string, unknown>
                  ),
                },
              },
            },
          },
        })
      ),
    } as unknown as ApiClient;
    vi.mocked(apiClient.patchResourceMetadata).mockImplementation(
      async (_fileId: string, metadata: Record<string, unknown>) => {
        const calibration = metadata.ultra_viewer_calibration_v1 as Record<string, unknown>;
        const requestSelections = calibration.selections as Record<string, unknown>;
        const acknowledgedSelections = Object.fromEntries(
          Object.entries(requestSelections).map(([key, rawSelection]) => {
            const {
              expected_revision: expectedRevision,
              ...selection
            } = rawSelection as Record<string, unknown>;
            return [
              key,
              {
                ...selection,
                revision: Number(expectedRevision) + 1,
              },
            ];
          })
        );
        const selectionId = Object.keys(requestSelections)[0] ?? "";
        if (deferCalibrationSaves) {
          return await new Promise((resolve) => {
            pendingCalibrationSaves.push({
              selectionId,
              resolve: () =>
                resolve({
                  resource: {
                    metadata: {
                      ultra_viewer_calibration_v1: {
                        ...calibration,
                        selections: acknowledgedSelections,
                      },
                    },
                  },
                } as never),
            });
          });
        }
        persistedSelections = {
          ...persistedSelections,
          ...acknowledgedSelections,
        };
        return {
          resource: {
            metadata: {
              ultra_viewer_calibration_v1: {
                ...calibration,
                selections: persistedSelections,
              },
            },
          },
        } as never;
      }
    );

    function Harness() {
      const [currentViewer, setCurrentViewer] = useState(maskViewer);
      const [displayState, setDisplayState] = useState(maskViewer.display_defaults ?? null);
      const [time, setTime] = useState(0);
      const [surface, setSurface] = useState<"2d" | "volume">("volume");
      const [mounted, setMounted] = useState(true);
      return (
        <>
          <button type="button" onClick={() => setTime(0)}>Time 0</button>
          <button type="button" onClick={() => setTime(1)}>Time 1</button>
          <button type="button" onClick={() => setTime(2)}>Time 2</button>
          <button type="button" onClick={() => setSurface("2d")}>2D surface</button>
          <button type="button" onClick={() => setSurface("volume")}>Volume surface</button>
          <button
            type="button"
            onClick={() => {
              setMounted(false);
              setDisplayState(maskViewer.display_defaults ?? null);
            }}
          >
            Switch away
          </button>
          <button type="button" onClick={() => setMounted(true)}>Switch back</button>
          {mounted ? <ImageViewerShell
            viewerInfo={currentViewer}
            apiClient={apiClient}
            selectedSurface={surface}
            onSurfaceChange={(nextSurface) =>
              setSurface(nextSurface === "2d" ? "2d" : "volume")
            }
            selectedDisplayState={displayState}
            updateSelectedDisplay={(patch) =>
              setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
            }
            clampedIndices={{ x: 0, y: 0, z: 0, t: time }}
            debouncedX={0}
            debouncedY={0}
            debouncedZ={0}
            debouncedT={time}
            xAxisSize={2}
            yAxisSize={2}
            zAxisSize={2}
            tAxisSize={3}
            setSelectedIndex={() => {}}
            selectedCaption=""
            captionLoading={false}
            onViewerCalibrationsChange={(calibrations) => {
              if (!calibrations) {
                return;
              }
              setCurrentViewer((previous) => ({
                ...previous,
                viewer_calibrations:
                  previous.viewer_calibrations?.source_sha256 ===
                  calibrations.source_sha256
                    ? {
                        version: 1,
                        source_sha256: calibrations.source_sha256,
                        selections: {
                          ...previous.viewer_calibrations.selections,
                          ...calibrations.selections,
                        },
                      }
                    : calibrations,
              }));
            }}
          /> : null}
        </>
      );
    }

    render(<Harness />);

    expect(apiClient.getUploadHistogram).not.toHaveBeenCalled();
    expect(screen.getByRole("combobox", { name: "Scalar rendering" })).toHaveTextContent(
      "Auto · Mask"
    );
    expect(
      screen.queryByRole("button", { name: "About scalar rendering" })
    ).not.toBeInTheDocument();
    await chooseSelectOption("Scalar rendering", "Mask");
    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenCalledWith(
        "file-mask",
        expect.objectContaining({
          bins: 256,
          channel: 1,
          t: 0,
          scope: "volume",
          signal: expect.any(AbortSignal),
        })
      )
    );
    expect(apiClient.getUploadHistogram).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.mocked(apiClient.getUploadHistogram).mock.results[0]?.value;
    });
    openAdvancedControls();
    await waitFor(() =>
      expect(screen.getByLabelText("Mask raw threshold")).toHaveValue("120")
    );
    expect(screen.queryByRole("combobox", { name: "Projection" })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Mask raw threshold"), { target: { value: "130" } });
    expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute(
      "data-scalar-threshold-value",
      "130"
    );
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save resource default" }));

    await waitFor(() =>
      expect(apiClient.patchResourceMetadata).toHaveBeenCalledWith("file-mask", {
        ultra_viewer_calibration_v1: {
          version: 1,
          source_sha256: "mask-sha-256",
          selections: {
            "c1:t0": {
              channel: 1,
              t: 0,
              render_mode: "mask",
              threshold_method: "manual",
              threshold_value: 130,
              threshold_foreground: "above",
              expected_revision: 0,
              threshold_provenance: {
                method: "otsu-256-v1",
                value: 120,
                domain: "raw",
                foreground: "above",
                channel: 1,
                t: 0,
                sample_scope: "volume",
                sample_count: 8,
                sampling_algorithm: "scalar-profile-otsu-256-v1",
                sampling_strategy: "exact",
                z_samples: [0, 1],
                source_sha256: "mask-sha-256",
                bins: 256,
              },
            },
          },
        },
      })
    );
    expect(await screen.findByText("Saved.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Switch away" }));
    fireEvent.click(screen.getByRole("button", { name: "Switch back" }));
    openAdvancedControls();
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("130");

    fireEvent.click(screen.getByRole("button", { name: "Time 2" }));
    await chooseSelectOption("Scalar rendering", "Mask");
    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenCalledWith(
        "file-mask",
        expect.objectContaining({
          bins: 256,
          channel: 1,
          t: 2,
          scope: "volume",
          signal: expect.any(AbortSignal),
        })
      )
    );
    expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute(
      "data-scalar-render-mode",
      "intensity"
    );
    time2HistogramResolved = true;
    await act(async () => {
      resolveTime2Histogram?.(histogramForTime(2));
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Mask raw threshold")).toHaveValue("220")
    );
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "230" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save resource default" }));
    await waitFor(() =>
      expect(apiClient.patchResourceMetadata).toHaveBeenLastCalledWith(
        "file-mask",
        expect.objectContaining({
          ultra_viewer_calibration_v1: expect.objectContaining({
            selections: {
              "c1:t2": expect.objectContaining({
                channel: 1,
                t: 2,
                threshold_value: 230,
                threshold_provenance: expect.objectContaining({
                  value: 220,
                  t: 2,
                }),
              }),
            },
          }),
        })
      )
    );

    fireEvent.click(screen.getByRole("button", { name: "Time 0" }));
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("130");
    await new Promise((resolvePromise) => window.setTimeout(resolvePromise, 200));

    uploadSliceUrl.mockClear();
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "140" },
    });
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "150" },
    });
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "160" },
    });
    fireEvent.click(screen.getByRole("button", { name: "2D surface" }));
    expect(screen.getByTestId("direct-plane-image")).toHaveAttribute(
      "data-image-url",
      expect.stringContaining("scalar_threshold_value=130")
    );
    await waitFor(
      () =>
        expect(screen.getByTestId("direct-plane-image")).toHaveAttribute(
          "data-image-url",
          expect.stringContaining("scalar_threshold_value=160")
        ),
      { timeout: 1000 }
    );
    const requestedThresholds = new Set(
      uploadSliceUrl.mock.calls
        .map((call) => call[1]?.scalarThresholdValue)
        .filter((value): value is number => typeof value === "number")
    );
    expect(requestedThresholds).toEqual(new Set([130, 160]));

    fireEvent.click(screen.getByRole("button", { name: "Time 1" }));
    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenCalledWith(
        "file-mask",
        expect.objectContaining({
          bins: 256,
          channel: 1,
          t: 1,
          scope: "volume",
          signal: expect.any(AbortSignal),
        })
      )
    );
    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").getAttribute("data-image-url") ?? "";
      expect(imageUrl).not.toContain("scalar_render_mode=mask");
      expect(imageUrl).not.toContain("scalar_threshold_value=");
    });

    fireEvent.click(screen.getByRole("button", { name: "Time 0" }));
    fireEvent.click(screen.getByRole("button", { name: "Volume surface" }));
    await waitFor(() =>
      expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute(
        "data-t-index",
        "0"
      )
    );
    if (
      screen
        .getByRole("button", { name: "Advanced rendering" })
        .getAttribute("aria-expanded") !== "true"
    ) {
      openAdvancedControls();
    }
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("160");
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "170" },
    });
    deferCalibrationSaves = true;
    fireEvent.click(screen.getByRole("button", { name: "Save resource default" }));
    await waitFor(() =>
      expect(pendingCalibrationSaves.map((pending) => pending.selectionId)).toContain(
        "c1:t0"
      )
    );
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "180" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Time 2" }));
    await waitFor(() =>
      expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute(
        "data-t-index",
        "2"
      )
    );
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "Scalar rendering" })
      ).toHaveTextContent("Mask")
    );
    if (
      screen
        .getByRole("button", { name: "Advanced rendering" })
        .getAttribute("aria-expanded") !== "true"
    ) {
      openAdvancedControls();
    }
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("230");
    fireEvent.change(screen.getByLabelText("Mask raw threshold"), {
      target: { value: "240" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save resource default" }));
    await waitFor(() =>
      expect(pendingCalibrationSaves.map((pending) => pending.selectionId)).toEqual(
        expect.arrayContaining(["c1:t0", "c1:t2"])
      )
    );

    await act(async () => {
      pendingCalibrationSaves.find(
        (pending) => pending.selectionId === "c1:t2"
      )?.resolve();
    });
    expect(await screen.findByText("Saved.")).toBeInTheDocument();
    await act(async () => {
      pendingCalibrationSaves.find(
        (pending) => pending.selectionId === "c1:t0"
      )?.resolve();
    });

    fireEvent.click(screen.getByRole("button", { name: "Time 0" }));
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("180");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Time 2" }));
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("240");

    fireEvent.click(screen.getByRole("button", { name: "Switch away" }));
    fireEvent.click(screen.getByRole("button", { name: "Switch back" }));
    openAdvancedControls();
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("240");
    fireEvent.click(screen.getByRole("button", { name: "Time 0" }));
    expect(await screen.findByLabelText("Mask raw threshold")).toHaveValue("170");
  });

  it("uses shadcn selects instead of native browser dropdowns", () => {
    const source = readFileSync(resolve(process.cwd(), "src/components/viewer/ImageViewerShell.tsx"), "utf8");

    expect(source).not.toMatch(/<select\b/);
    expect(source).toMatch(/<SelectTrigger/);
  });

  it("keeps scientific slice URLs exact when upload histograms load", async () => {
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(viewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={viewerInfo}
          apiClient={apiClient}
          selectedSurface="2d"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={4}
          yAxisSize={1}
          zAxisSize={1}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenCalledWith(
        "file-123",
        expect.objectContaining({
          bins: 256,
          signal: expect.any(AbortSignal),
        })
      )
    );
    await waitFor(() => {
      expect(screen.queryByLabelText("Window center")).toBeNull();
      expect(screen.queryByLabelText("Window width")).toBeNull();
      expect(screen.queryByRole("button", { name: "Auto" })).toBeNull();
      expect(screen.queryByRole("button", { name: "Full" })).toBeNull();
      expect(screen.queryByText("Negative")).toBeNull();
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).not.toContain("enhancement=");
      expect(imageUrl).not.toContain("window_min=");
      expect(imageUrl).not.toContain("window_max=");
    });
  });

  it("offers a curated right-click context menu on the 2D surface", async () => {
    const resourceDownloadUrl = vi.fn(() => "https://ultra.example.org/v2/resources/file-123/download");
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
      resourceDownloadUrl,
    } as unknown as ApiClient;

    const onSurfaceChange = vi.fn();
    render(
      <ImageViewerShell
        viewerInfo={viewerInfo}
        apiClient={apiClient}
        selectedSurface="2d"
        onSurfaceChange={onSurfaceChange}
        selectedDisplayState={viewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={0}
        debouncedT={0}
        xAxisSize={4}
        yAxisSize={1}
        zAxisSize={1}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    const surface = document.querySelector('[data-viewer-surface="2d"]');
    expect(surface).not.toBeNull();
    fireEvent.contextMenu(surface as HTMLElement);

    // Curated, high-value actions are present; zoom in/out are intentionally NOT in the
    // menu (they duplicate the always-visible toolbar buttons).
    expect(await screen.findByText("Reset view")).toBeInTheDocument();
    expect(screen.getByText("Copy current view")).toBeInTheDocument();
    expect(screen.getByText("Export current view (PNG)")).toBeInTheDocument();
    expect(screen.getByText("Download original image")).toBeInTheDocument();
    expect(screen.getByText("View metadata")).toBeInTheDocument();
    expect(screen.queryByText("Zoom in")).not.toBeInTheDocument();
    expect(screen.queryByText("Zoom out")).not.toBeInTheDocument();

    // "View metadata" jumps to the metadata surface via the shell's surface change.
    fireEvent.click(screen.getByText("View metadata"));
    expect(onSurfaceChange).toHaveBeenCalledWith("metadata");
  });

  it("tags the 2D canvas wrapper so it fills the available height (volume + non-volume)", () => {
    // The fill chain (.viewer-workspace-surface-2d .viewer-canvas-layout-2d { height:100% })
    // only reaches the canvas if BOTH the volume and non-volume wrappers carry the
    // shared class — a non-volume image previously had an unclassed wrapper, so the
    // canvas left empty panel space below it. Guard the class on both branches.
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;
    const props = {
      apiClient,
      selectedSurface: "2d" as const,
      onSurfaceChange: () => {},
      updateSelectedDisplay: () => {},
      clampedIndices: { x: 0, y: 0, z: 0, t: 0 },
      debouncedX: 0,
      debouncedY: 0,
      debouncedZ: 0,
      debouncedT: 0,
      xAxisSize: 4,
      yAxisSize: 1,
      zAxisSize: 1,
      tAxisSize: 1,
      setSelectedIndex: () => {},
      selectedCaption: "",
      captionLoading: false,
    };
    // Non-volume photo: wrapper carries the shared fill class, NOT the volume class.
    const { container, unmount } = render(
      <ImageViewerShell {...props} viewerInfo={viewerInfo} selectedDisplayState={viewerInfo.display_defaults ?? null} />
    );
    const flat = container.querySelector(".viewer-canvas-layout-2d");
    expect(flat).not.toBeNull();
    expect(flat?.classList.contains("viewer-volume-layout-2d")).toBe(false);
    unmount();

    // Volume: wrapper carries BOTH the volume layout class and the shared fill class.
    const volumeInfo: UploadViewerInfo = {
      ...viewerInfo,
      is_volume: true,
      axis_sizes: { ...viewerInfo.axis_sizes, Z: 40 },
    };
    const { container: volContainer } = render(
      <ImageViewerShell {...props} viewerInfo={volumeInfo} selectedDisplayState={volumeInfo.display_defaults ?? null} />
    );
    const volWrapper = volContainer.querySelector(".viewer-volume-layout-2d");
    expect(volWrapper).not.toBeNull();
    expect(volWrapper?.classList.contains("viewer-canvas-layout-2d")).toBe(true);
  });

  it("lets direct multichannel images choose the visualized channel", async () => {
    const multichannelViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      axis_sizes: { ...viewerInfo.axis_sizes, C: 3 },
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "m",
          channel_mode: "composite",
          channels: [0, 1, 2],
          channel_colors: [],
          time_index: 0,
          z_index: 0,
        }),
        channel_mode: "composite",
        channels: [0, 1, 2],
        channel_colors: ["#ff0000", "#00ff00", "#0000ff"],
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "YXC",
        array_shape: [1, 4, 3],
        array_dtype: "uint8",
      },
      viewer: {
        ...viewerInfo.viewer,
        // A real direct multichannel science image is the scalar (window/level) path;
        // render_policy "display" is reserved for RGB(A) photos, which have no science
        // channels to choose (so they intentionally hide the per-channel controls).
        render_policy: "scalar",
        channel_mode: "composite",
        display_capabilities: ["histogram", "channel_visibility"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => ({
        ...histogram,
        channels: [0, 1, 2],
        histogram: { ...histogram.histogram, channel_indices: [0, 1, 2] },
      })),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(multichannelViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={multichannelViewerInfo}
          apiClient={apiClient}
          selectedSurface="2d"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={4}
          yAxisSize={1}
          zAxisSize={1}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    expect(await screen.findByTestId("direct-plane-image")).toBeInTheDocument();
    expect(screen.getByText("Channel 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Channel 1, source channel 0" }));
    fireEvent.click(screen.getByRole("button", { name: "Channel 3, source channel 2" }));

    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("channels=1");
    });
  });

  it("keeps hyperspectral channel controls bounded and searchable", async () => {
    const channelNames = Array.from({ length: 260 }, (_value, index) => `Band ${index + 1}`);
    const channelColors = Array.from(
      { length: 260 },
      (_value, index) => ["#3b82f6", "#22c55e", "#ef4444"][index % 3],
    );
    const hyperspectralViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      axis_sizes: { ...viewerInfo.axis_sizes, T: 4, C: 260, Z: 7 },
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "m",
          channel_mode: "composite",
          channels: [5, 1, 3],
          channel_colors: [],
        time_index: 2,
        z_index: 4,
        }),
        channel_mode: "composite",
        channels: [5, 1, 3, 5, -1, 1.5, 260, Number.NaN],
        channel_colors: channelColors,
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "TCZYX",
        array_shape: [4, 260, 7, 4, 4],
        array_dtype: "uint16",
        microscopy: {
          ...viewerInfo.metadata.microscopy,
          channel_names: channelNames,
        },
      },
      viewer: {
        ...viewerInfo.viewer,
        render_policy: "scalar",
        channel_mode: "composite",
        display_capabilities: [
          "histogram",
          "channel_visibility",
          "channel_color",
          "channel_lut_transport",
        ],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => ({
        ...histogram,
        channels: [5, 1, 3],
        histogram: { ...histogram.histogram, channel_indices: [5, 1, 3] },
      })),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(hyperspectralViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={hyperspectralViewerInfo}
          apiClient={apiClient}
          selectedSurface="2d"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 4, t: 2 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={4}
          debouncedT={2}
          xAxisSize={4}
          yAxisSize={4}
          zAxisSize={7}
          tAxisSize={4}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    const { container } = render(<Harness />);

    expect(await screen.findByTestId("direct-plane-image")).toBeInTheDocument();
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("/slice?");
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("z=4");
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("t=2");
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("channels=5%2C1%2C3");
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain(
      "channel_colors=%23ef4444%2C%2322c55e%2C%233b82f6",
    );
    expect(apiClient.uploadDisplayUrl).not.toHaveBeenCalled();
    expect(
      Array.from(container.querySelectorAll(".viewer-channel-toggle")).map((element) =>
        element.getAttribute("aria-label"),
      ),
    ).toEqual([
      "Band 6, source channel 5",
      "Band 2, source channel 1",
      "Band 4, source channel 3",
    ]);
    expect(screen.queryByRole("button", { name: "Band 260, source channel 259" })).toBeNull();
    expect(container.querySelectorAll('[data-viewer-channel-chip="true"]')).toHaveLength(3);
    expect(container.querySelectorAll('[data-viewer-channel-row="true"]')).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Enter fullscreen" }));
    await waitFor(() => {
      expect(container.querySelector(".viewer-shell")).toHaveAttribute(
        "data-viewer-fullscreen",
        "true",
      );
    });

    const chooserTrigger = screen.getByRole("button", {
      name: "Choose channels, 3 selected of 260",
    });
    fireEvent.click(chooserTrigger);
    const channelDialog = await screen.findByRole("dialog", { name: "Channels" });
    expect(channelDialog).toHaveClass("viewer-channel-browser-dialog");
    expect(channelDialog.closest(".viewer-shell")).not.toBeNull();
    let search = await screen.findByPlaceholderText("Search 260 channels");
    expect(search).toHaveFocus();
    expect(document.querySelectorAll('[data-viewer-channel-row="true"]').length).toBeLessThanOrEqual(12);

    fireEvent.change(search, { target: { value: "C259" } });
    const focusedVirtualRow = await screen.findByRole("button", {
      name: "Band 260, source channel 259",
    });
    focusedVirtualRow.focus();
    expect(focusedVirtualRow).toHaveFocus();
    fireEvent.keyDown(focusedVirtualRow, { key: "Escape", code: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Channels" })).toBeNull());
    expect(container.querySelector(".viewer-shell")).toHaveAttribute(
      "data-viewer-fullscreen",
      "true",
    );
    expect(chooserTrigger).toHaveFocus();

    fireEvent.click(chooserTrigger);
    search = await screen.findByPlaceholderText("Search 260 channels");
    expect(search).toHaveFocus();
    expect(search).toHaveValue("");
    fireEvent.change(search, { target: { value: "C259" } });
    fireEvent.click(
      await screen.findByRole("button", { name: "Band 260, source channel 259" }),
    );

    await waitFor(() => {
      expect(screen.getByText(/4 selected/)).toBeInTheDocument();
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("/slice?");
      expect(imageUrl).toContain("channels=5%2C1%2C3%2C259");
      expect(imageUrl).toContain("z=4");
      expect(imageUrl).toContain("t=2");
      expect(apiClient.uploadSliceUrl).toHaveBeenLastCalledWith(
        hyperspectralViewerInfo.file_id,
        expect.objectContaining({
          z: 4,
          t: 2,
          channels: [5, 1, 3, 259],
          channelColors,
          fullResolution: true,
        }),
      );
    });

    for (const sourceIndex of [200, 201, 202, 203]) {
      fireEvent.change(search, { target: { value: `C${sourceIndex}` } });
      fireEvent.click(
        await screen.findByRole("button", {
          name: `Band ${sourceIndex + 1}, source channel ${sourceIndex}`,
        }),
      );
    }
    await waitFor(() => {
      expect(screen.getByText(/8 selected/)).toBeInTheDocument();
    });

    fireEvent.change(search, { target: { value: "C204" } });
    expect(
      await screen.findByRole("button", { name: "Band 205, source channel 204" }),
    ).toBeDisabled();
    expect(screen.getByText("Remove a channel to choose another.")).toBeInTheDocument();
  });

  it("hides per-channel controls for an RGB(A) display photo", async () => {
    // An RGBA orthomosaic (render_policy "display", channel_mode "single") is color
    // data, not science channels — the Red/Green/Blue/Alpha pills must NOT appear even
    // though C>1.
    const photoViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      modality: "image",
      axis_sizes: { ...viewerInfo.axis_sizes, C: 4 },
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d", negative: false, rotate: 0, fusion_method: "m",
          channel_mode: "single", channels: [0], channel_colors: [], time_index: 0, z_index: 0,
        }),
        channel_mode: "single",
        channels: [0],
      },
      viewer: {
        ...viewerInfo.viewer,
        render_policy: "display",
        channel_mode: "single",
        display_capabilities: ["histogram", "channel_visibility"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={photoViewerInfo}
        apiClient={apiClient}
        selectedSurface="2d"
        onSurfaceChange={() => {}}
        selectedDisplayState={photoViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0} debouncedY={0} debouncedZ={0} debouncedT={0}
        xAxisSize={4} yAxisSize={1} zAxisSize={1} tAxisSize={1}
        setSelectedIndex={() => {}} selectedCaption="" captionLoading={false}
      />
    );

    expect(await screen.findByTestId("direct-plane-image")).toBeInTheDocument();
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("/display");
    expect(apiClient.uploadDisplayUrl).toHaveBeenCalledWith(
      photoViewerInfo.file_id,
      photoViewerInfo.service_urls?.display,
      expect.objectContaining({ channels: undefined, channelColors: undefined }),
    );
    // No per-channel pills (the composite UI is suppressed for a photo).
    expect(screen.queryByRole("button", { name: "Channel 1, source channel 0" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Channel 2, source channel 1" })).toBeNull();
    expect(await screen.findByLabelText("Window center")).toBeInTheDocument();
    expect(screen.getByLabelText("Window width")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Auto" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Full" })).toBeInTheDocument();
    expect(screen.getByText("Negative")).toBeInTheDocument();
  });

  it("renders a flat RGB time series through an exact T-aware scientific slice", () => {
    const timeSeriesViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "rgb-timeseries.ome.tif",
      backend_mode: "pyramid",
      dims_order: "TCYX",
      axis_sizes: { T: 3, C: 3, Z: 1, Y: 1024, X: 1024 },
      is_timeseries: true,
      display_defaults: {
        ...viewerInfo.display_defaults!,
        enhancement: "hounsfield:50:100",
        negative: true,
        fusion_method: "m",
        channels: [0, 1, 2],
      },
      viewer: {
        ...viewerInfo.viewer,
        backend_mode: "pyramid",
        delivery_mode: "deferred_multiscale",
        render_policy: "display",
        tile_scheme: {
          tile_size: 512,
          format: "png",
          levels: [{ level: 0, width: 1024, height: 1024, columns: 2, rows: 2, downsample: 1 }],
        },
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
      uploadTileUrl: vi.fn(buildTileUrl),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={timeSeriesViewerInfo}
        apiClient={apiClient}
        selectedSurface="2d"
        onSurfaceChange={() => {}}
        selectedDisplayState={timeSeriesViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 2 }}
        debouncedX={0} debouncedY={0} debouncedZ={0} debouncedT={2}
        xAxisSize={1024} yAxisSize={1024} zAxisSize={1} tAxisSize={3}
        setSelectedIndex={() => {}} selectedCaption="" captionLoading={false}
      />
    );

    expect(screen.queryByTestId("deep-zoom-canvas")).not.toBeInTheDocument();
    const imageUrl = new URL(screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "");
    expect(imageUrl.pathname).toContain("/slice");
    expect(imageUrl.searchParams.get("t")).toBe("2");
    expect(imageUrl.searchParams.has("enhancement")).toBe(false);
    expect(imageUrl.searchParams.has("negative")).toBe(false);
    expect(imageUrl.searchParams.get("cache_key")).not.toContain("hounsfield");
    expect(imageUrl.searchParams.get("cache_key")).not.toContain("negative");
    expect(apiClient.uploadDisplayUrl).not.toHaveBeenCalled();
  });

  it("uses deep zoom tiles for pyramid-backed 2D images", () => {
    const pyramidColors = Array.from(
      { length: 260 },
      (_value, index) => ["#3b82f6", "#22c55e", "#ef4444"][index % 3],
    );
    const pyramidPlane = {
      ...defaultPlane,
      pixel_size: { width: 95174, height: 91416 },
      world_size: { width: 95174, height: 91416 },
      aspect_ratio: 95174 / 91416,
    };
    const pyramidViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "large-pyramid.tif",
      backend_mode: "pyramid",
      axis_sizes: { T: 1, C: 260, Z: 1, Y: 91416, X: 95174 },
      is_multichannel: true,
      display_defaults: {
        ...viewerInfo.display_defaults!,
        channel_mode: "composite",
        channels: [5, 1, 259],
        channel_colors: pyramidColors,
      },
      phys: {
        ...viewerInfo.phys!,
        channel_names: ["DAPI", "FITC"],
        channel_colors: pyramidColors.map((hex, index) => ({ index, hex, rgb: [0, 0, 0] })),
      },
      service_urls: {
        ...viewerInfo.service_urls,
        tile: "/v2/uploads/file-123/tiles/{axis}/{level}/{tile_x}/{tile_y}",
      },
      metadata: {
        ...viewerInfo.metadata,
        reader: "libbioimage",
        array_shape: [260, 91416, 95174],
        array_dtype: "uint8",
        microscopy: {
          ...viewerInfo.metadata.microscopy,
          channel_names: ["DAPI", ""],
        },
      },
      viewer: {
        ...viewerInfo.viewer,
        backend_mode: "pyramid",
        delivery_mode: "deferred_multiscale",
        first_paint_mode: "webgl",
        render_policy: "scalar",
        channel_mode: "composite",
        display_capabilities: [
          "histogram",
          "channel_visibility",
          "channel_color",
          "channel_lut_transport",
        ],
        tile_scheme: {
          tile_size: 512,
          format: "png",
          levels: [
            { level: 8, width: 371, height: 357, columns: 1, rows: 1, downsample: 256 },
            { level: 7, width: 743, height: 714, columns: 2, rows: 2, downsample: 128 },
            { level: 0, width: 95174, height: 91416, columns: 186, rows: 179, downsample: 1 },
          ],
        },
        default_plane: pyramidPlane,
        planes: { z: pyramidPlane },
        viewer_capabilities: ["2d", "metadata", "deep_zoom"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
      uploadTileUrl: vi.fn(buildTileUrl),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={pyramidViewerInfo}
        apiClient={apiClient}
        selectedSurface="2d"
        onSurfaceChange={() => {}}
        selectedDisplayState={pyramidViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={0}
        debouncedT={0}
        xAxisSize={95174}
        yAxisSize={91416}
        zAxisSize={1}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    expect(screen.getByTestId("deep-zoom-canvas")).toHaveAttribute("data-level-count", "3");
    expect(screen.getByTestId("deep-zoom-canvas")).toHaveAttribute("data-z-index", "0");
    expect(screen.getByTestId("deep-zoom-canvas")).toHaveAttribute("data-channels", "5,1,259");
    expect(screen.getByRole("button", { name: "Channel 260, source channel 259" })).toBeInTheDocument();
    const tileUrl = new URL(screen.getByTestId("deep-zoom-canvas").dataset.tileUrl ?? "");
    expect(tileUrl.searchParams.get("channels")).toBe("5,1,259");
    expect(tileUrl.searchParams.get("channel_colors")).toBe(
      [pyramidColors[5], pyramidColors[1], pyramidColors[259]].join(","),
    );
    expect(tileUrl.searchParams.get("cache_key")).toContain(":5,1,259:");
    expect(screen.queryByTestId("direct-plane-image")).not.toBeInTheDocument();
  });

  it("keeps pyramid-backed RGB(A) tiles on their native display path", () => {
    const photoViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "rgba-orthomosaic.tif",
      backend_mode: "pyramid",
      axis_sizes: { T: 1, C: 4, Z: 1, Y: 4096, X: 4096 },
      is_multichannel: true,
      display_defaults: {
        ...viewerInfo.display_defaults!,
        channel_mode: "single",
        channels: [0, 1, 2, 3],
        channel_colors: ["#ff0000", "#00ff00", "#0000ff", "#ffffff"],
      },
      viewer: {
        ...viewerInfo.viewer,
        backend_mode: "pyramid",
        delivery_mode: "deferred_multiscale",
        render_policy: "display",
        channel_mode: "single",
        tile_scheme: {
          tile_size: 512,
          format: "png",
          levels: [{ level: 0, width: 4096, height: 4096, columns: 8, rows: 8, downsample: 1 }],
        },
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
      uploadTileUrl: vi.fn(buildTileUrl),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={photoViewerInfo}
        apiClient={apiClient}
        selectedSurface="2d"
        onSurfaceChange={() => {}}
        selectedDisplayState={photoViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={0}
        debouncedT={0}
        xAxisSize={4096}
        yAxisSize={4096}
        zAxisSize={1}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    const canvas = screen.getByTestId("deep-zoom-canvas");
    expect(canvas).toHaveAttribute("data-channels", "");
    expect(canvas).toHaveAttribute("data-cache-key", "");
    const tileUrl = new URL(canvas.dataset.tileUrl ?? "");
    expect(tileUrl.searchParams.has("channels")).toBe(false);
    expect(tileUrl.searchParams.has("channel_colors")).toBe(false);
    expect(tileUrl.searchParams.has("cache_key")).toBe(false);
  });

  it("presents OME TIFF stacks with single-channel slice controls", async () => {
    const stackViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "cell-stack.ome.tiff",
      dims_order: "ZCYX",
      axis_sizes: { T: 1, C: 3, Z: 2, Y: 1, X: 2 },
      selected_indices: { T: 0, C: 2, Z: 1 },
      is_volume: true,
      is_multichannel: true,
      phys: {
        channel_names: ["DAPI", "EGFP", "Brightfield"],
        channel_colors: [
          { index: 0, hex: "#0000ff", rgb: [0, 0, 255] },
          { index: 1, hex: "#00ff00", rgb: [0, 255, 0] },
          { index: 2, hex: "#ffffff", rgb: [255, 255, 255] },
        ],
      },
      display_defaults: {
        enhancement: "d",
        negative: false,
        rotate: 0,
        fusion_method: "m",
        channel_mode: "single",
        channels: [2],
        channel_colors: ["#0000ff", "#00ff00", "#ffffff"],
        time_index: 0,
        z_index: 1,
        volume_channel: 2,
      },
      metadata: {
        ...viewerInfo.metadata,
        reader: "ome-tiff+xml+go-image",
        dims_order: "ZCYX",
        array_shape: [2, 3, 1, 2],
        physical_spacing: { x: 0.5, y: 0.5, z: 1.25 },
        microscopy: {
          channel_names: ["DAPI", "EGFP", "Brightfield"],
          dimensions_present: "ZCYX",
        },
      },
      viewer: {
        ...viewerInfo.viewer,
        volume_mode: "slice_stack",
        channel_mode: "single",
        display_capabilities: [
          "slice_navigation",
          "channel_visibility",
          "channel_color",
          "channel_lut_transport",
          "physical_scale",
        ],
        viewer_capabilities: ["webgl_first_paint", "direct_delivery", "channel_selection"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(stackViewerInfo.display_defaults ?? null);
      const [indices, setIndices] = useState({ x: 0, y: 0, z: 1, t: 0 });
      return (
        <ImageViewerShell
          viewerInfo={stackViewerInfo}
          apiClient={apiClient}
          selectedSurface="2d"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={indices}
          debouncedX={indices.x}
          debouncedY={indices.y}
          debouncedZ={indices.z}
          debouncedT={indices.t}
          xAxisSize={2}
          yAxisSize={1}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={(axis, value) => setIndices((previous) => ({ ...previous, [axis]: value }))}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    expect(screen.getByLabelText("Stack summary")).toBeInTheDocument();
    expect(screen.getAllByText("Brightfield").length).toBeGreaterThan(0);
    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("z=1");
      expect(imageUrl).toContain("channels=2");
      expect(imageUrl).toContain("channel_colors=%23ffffff");
    });

    fireEvent.click(screen.getByRole("button", { name: "DAPI, source channel 0" }));
    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("channels=0");
      expect(imageUrl).toContain("channel_colors=%230000ff");
      expect(imageUrl).not.toContain("channels=0%2C2");
      expect(imageUrl).not.toContain("channels=0,2");
    });

    // The Z slice control is now the calm shadcn Slider (a Radix component, not a
    // native range), so it is driven via keyboard: hovering the 2D viewer and pressing
    // arrow keys steps Z. z starts at 1, so ArrowDown -> z=0, then ArrowUp -> z=1.
    const shell = document.querySelector(".viewer-shell") as HTMLElement;
    fireEvent.pointerEnter(shell);
    fireEvent.keyDown(shell, { key: "ArrowDown" });
    await waitFor(() => expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("z=0"));
    fireEvent.keyDown(shell, { key: "ArrowUp" });
    await waitFor(() => expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("z=1"));
  });

  it("maps microscopy Mask MPR through the validated native integer payload", async () => {
    const maskPlanes = {
      z: {
        axis: "z" as const,
        label: "Axial",
        axes: ["Y", "X"],
        pixel_size: { width: 10, height: 9 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 10, height: 9 },
        aspect_ratio: 10 / 9,
      },
      y: {
        axis: "y" as const,
        label: "Coronal",
        axes: ["Z", "X"],
        pixel_size: { width: 10, height: 7 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 10, height: 7 },
        aspect_ratio: 10 / 7,
      },
      x: {
        axis: "x" as const,
        label: "Sagittal",
        axes: ["Z", "Y"],
        pixel_size: { width: 9, height: 7 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 9, height: 7 },
        aspect_ratio: 9 / 7,
      },
    };
    const threshold = {
      method: "otsu-256-v1" as const,
      value: 120,
      domain: "raw" as const,
      foreground: "above" as const,
      sample_scope: "volume",
      sample_count: 36,
      z_samples: [0, 3, 6],
      channel: 1,
      t: 1,
      sampling_algorithm: "scalar-profile-otsu-256-v1",
    };
    const maskViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      file_id: "file-mask-mpr",
      original_name: "mask.ome.tiff",
      modality: "microscopy",
      dims_order: "TCZYX",
      backend_mode: "direct",
      axis_sizes: { T: 2, C: 2, Z: 7, Y: 9, X: 10 },
      selected_indices: { T: 1, C: 1, Z: 6 },
      is_volume: true,
      is_timeseries: true,
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults as NonNullable<UploadViewerInfo["display_defaults"]>),
        channels: [1],
        time_index: 1,
        z_index: 6,
        scalar_render_mode: "mask",
        scalar_threshold_method: "otsu-256-v1",
        scalar_threshold_value: 120,
        scalar_threshold_foreground: "above",
      },
      metadata: {
        ...viewerInfo.metadata,
        reader: "tifffile",
        dims_order: "TCZYX",
        array_shape: [2, 2, 7, 9, 10],
        array_dtype: "uint16",
        sha256: "mask-mpr-sha",
      },
      data_semantics: {
        kind: "binary_mask",
        basis: "bounded_scalar_profile",
        strength: "exact",
        supported_modes: ["intensity", "mask"],
        recommended_view: "mask",
        threshold,
      },
      scalar_mask_capability: {
        version: 1,
        source_authority: "original",
        source_format: "ome-tiff",
        source_sha256: "mask-mpr-sha",
        dtype: "uint16",
        threshold_domain: "raw",
        threshold_foreground: "above",
        slice_delivery: "thresholded_png",
        volume_delivery: "raw_scalar",
        volume_sampling: "nearest",
        channel_selection: "single",
        time_selection: "single",
        surfaces: ["2d", "mpr", "volume"],
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-mask-mpr/scalar-volume",
      },
      viewer: {
        ...viewerInfo.viewer,
        default_surface: "mpr",
        available_surfaces: ["2d", "mpr", "volume", "metadata"],
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        default_plane: maskPlanes.z,
        planes: maskPlanes,
        volume_mode: "slice_stack",
        render_policy: "scalar",
        diagnostic_surface: "none",
        display_capabilities: ["channel_visibility"],
        viewer_capabilities: ["2d", "mpr", "volume", "metadata"],
        service_urls: {
          slice: "/v2/uploads/file-mask-mpr/slice",
          scalar_volume: "/v2/uploads/file-mask-mpr/scalar-volume",
        },
      },
    };
    const scalarVolumeResponse = {
      data: new Uint16Array(10 * 9 * 7)
        .map((_, index) => index + 100)
        .buffer,
      width: 10,
      height: 9,
      depth: 7,
      dtype: "uint16" as const,
      bytesPerVoxel: 2,
      rawMin: 100,
      rawMax: 729,
      channel: 1,
      time: 1,
      sourceWidth: 10,
      sourceHeight: 9,
      sourceDepth: 7,
      downsampleX: 1,
      downsampleY: 1,
      downsampleZ: 1,
      previewPolicy: "mask-native-integer-v1",
      sampling: "nearest" as const,
      sclSlope: 1,
      sclInter: 0,
    };
    let resolveScalarVolume!: (value: typeof scalarVolumeResponse) => void;
    const getUploadScalarVolume = vi.fn(
      () =>
        new Promise<typeof scalarVolumeResponse>((resolve) => {
          resolveScalarVolume = resolve;
        })
    );
    const uploadSliceUrl = vi.fn(buildSliceUrl);
    const apiClient = {
      getUploadScalarVolume,
      getUploadHistogram: vi.fn(async () => ({
        file_id: "file-mask-mpr",
        bins: 256,
        dtype: "uint16",
        channels: [1],
        source: "image-service-source",
        sample_count: 36,
        scope: "volume",
        channel: 1,
        t: 1,
        sampling: {
          algorithm: "scalar-profile-otsu-256-v1",
          scope: "volume",
          strategy: "exact",
          sample_count: 36,
          z_samples: [0, 3, 6],
        },
        threshold: { ...threshold, source_sha256: "mask-mpr-sha", bins: 256 },
        histogram: {
          bins: [18, 18],
          edges: [0, 120, 65535],
          min: 0,
          max: 65535,
          channel_indices: [1],
          time_index: 1,
        },
      })),
      uploadSliceUrl,
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;
    const shellProps = {
      viewerInfo: maskViewerInfo,
      apiClient,
      onSurfaceChange: () => {},
      selectedDisplayState: maskViewerInfo.display_defaults ?? null,
      updateSelectedDisplay: () => {},
      clampedIndices: { x: 4, y: 5, z: 6, t: 1 },
      debouncedX: 4,
      debouncedY: 5,
      debouncedZ: 6,
      debouncedT: 1,
      xAxisSize: 10,
      yAxisSize: 9,
      zAxisSize: 7,
      tAxisSize: 2,
      setSelectedIndex: () => {},
      selectedCaption: "",
      captionLoading: false,
    };
    const { rerender } = render(
      <ImageViewerShell {...shellProps} selectedSurface="mpr" />
    );

    await waitFor(() =>
      expect(getUploadScalarVolume).toHaveBeenCalledWith(
        "file-mask-mpr",
        expect.objectContaining({
          channel: 1,
          t: 1,
          sampling: "nearest",
          signal: expect.any(AbortSignal),
        })
      )
    );
    expect(document.querySelectorAll("[data-viewer-mpr-mask-unavailable]")).toHaveLength(3);
    expect(screen.queryAllByTestId("slice-plane-canvas")).toHaveLength(0);
    expect(uploadSliceUrl).not.toHaveBeenCalledWith(
      "file-mask-mpr",
      expect.objectContaining({ axis: "x" })
    );
    expect(uploadSliceUrl).not.toHaveBeenCalledWith(
      "file-mask-mpr",
      expect.objectContaining({ axis: "y" })
    );
    await act(async () => resolveScalarVolume(scalarVolumeResponse));
    await waitFor(() => expect(screen.getAllByTestId("slice-plane-canvas")).toHaveLength(3));
    expect(document.querySelectorAll("[data-viewer-mpr-mask-unavailable]")).toHaveLength(0);
    const canvases = Object.fromEntries(
      screen.getAllByTestId("slice-plane-canvas").map((canvas) => [
        canvas.dataset.title?.slice(0, 1),
        canvas,
      ])
    );
    expect(canvases.z).toHaveAttribute("data-scalar-slice-index", "6");
    expect(canvases.y).toHaveAttribute("data-scalar-slice-index", "5");
    expect(canvases.x).toHaveAttribute("data-scalar-slice-index", "4");
    expect(canvases.z).toHaveAttribute("data-crosshair-row", "5");
    expect(canvases.z).toHaveAttribute("data-crosshair-col", "4");
    expect(canvases.y).toHaveAttribute("data-crosshair-row", "6");
    expect(canvases.y).toHaveAttribute("data-crosshair-col", "4");
    expect(canvases.x).toHaveAttribute("data-crosshair-row", "6");
    expect(canvases.x).toHaveAttribute("data-crosshair-col", "5");
    expect(canvases.z).toHaveAttribute("data-coordinate-grid-width", "10");
    expect(canvases.z).toHaveAttribute("data-coordinate-grid-height", "9");
    expect(canvases.y).toHaveAttribute("data-coordinate-grid-width", "10");
    expect(canvases.y).toHaveAttribute("data-coordinate-grid-height", "7");
    expect(canvases.x).toHaveAttribute("data-coordinate-grid-width", "9");
    expect(canvases.x).toHaveAttribute("data-coordinate-grid-height", "7");
    expect(screen.getByText("Voxel value")).toBeInTheDocument();
    expect(screen.getByText("694")).toBeInTheDocument();

    rerender(<ImageViewerShell {...shellProps} selectedSurface="2d" />);
    const directPlane = await screen.findByTestId("direct-plane-image");
    expect(directPlane).toHaveAttribute("data-scalar-slice-index", "");
    expect(directPlane).toHaveAttribute(
      "data-image-url",
      expect.stringContaining("scalar_render_mode=mask")
    );
    expect(uploadSliceUrl).toHaveBeenCalledWith(
      "file-mask-mpr",
      expect.objectContaining({
        axis: "z",
        z: 6,
        t: 1,
        channels: [1],
        fullResolution: true,
        scalarRenderMode: "mask",
      })
    );
  });

  it.each([
    ["wrong source grid", { sourceWidth: 11 }],
    ["wrong nearest policy", { previewPolicy: "auto-v1" }],
    [
      "wrong Mask dtype",
      {
        data: new Uint8Array(4 * 3 * 3).buffer,
        dtype: "uint8",
        bytesPerVoxel: 1,
      },
    ],
  ] as const)("fails closed when Mask MPR receives %s provenance", async (_caseName, overrides) => {
    const maskViewerInfo = makeMaskMprViewerInfo();
    const payload = makeMaskMprPayload(overrides);
    const apiClient = {
      getUploadScalarVolume: vi.fn(async () => payload),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        {...makeMaskMprShellProps(maskViewerInfo, apiClient)}
        selectedSurface="mpr"
      />
    );

    await waitFor(() => expect(screen.getByText("Unavailable")).toBeInTheDocument());
    expect(document.querySelectorAll("[data-viewer-mpr-mask-unavailable]")).toHaveLength(3);
    expect(screen.queryAllByTestId("slice-plane-canvas")).toHaveLength(0);
    expect(screen.queryByText("Preview sample")).not.toBeInTheDocument();
    expect(screen.queryByText("Unavailable (not sampled in preview)")).not.toBeInTheDocument();
  });

  it("keeps the exact server Mask slice authoritative without loading a full volume in ordinary 2D", async () => {
    const maskViewerInfo = makeMaskMprViewerInfo();
    const getUploadScalarVolume = vi.fn();
    const uploadSliceUrl = vi.fn(buildSliceUrl);
    const apiClient = {
      getUploadScalarVolume,
      uploadSliceUrl,
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        {...makeMaskMprShellProps(maskViewerInfo, apiClient)}
        selectedSurface="2d"
      />
    );
    expect(getUploadScalarVolume).not.toHaveBeenCalled();

    const directPlane = await screen.findByTestId("direct-plane-image");
    await waitFor(() =>
      expect(directPlane).toHaveAttribute("data-scalar-slice-index", "")
    );
    expect(directPlane).toHaveAttribute(
      "data-image-url",
      expect.stringContaining("scalar_render_mode=mask")
    );
    expect(uploadSliceUrl).toHaveBeenCalledWith(
      "file-mask-mpr",
      expect.objectContaining({
        axis: "z",
        z: 6,
        t: 1,
        channels: [1],
        fullResolution: true,
        scalarRenderMode: "mask",
      })
    );
  });

  it("aborts Mask MPR volume work when leaving MPR and starts cleanly on return", async () => {
    const maskViewerInfo = makeMaskMprViewerInfo();
    const getUploadScalarVolume = vi.fn(
      (fileId: string, options: { signal?: AbortSignal }) => {
        void fileId;
        void options;
        return new Promise<ScalarVolumePayload>(() => {});
      }
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;
    const props = makeMaskMprShellProps(maskViewerInfo, apiClient);
    const { rerender } = render(
      <ImageViewerShell {...props} selectedSurface="mpr" />
    );

    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));
    const firstSignal = getUploadScalarVolume.mock.calls[0]?.[1]?.signal;
    expect(firstSignal?.aborted).toBe(false);

    rerender(<ImageViewerShell {...props} selectedSurface="2d" />);
    await waitFor(() => expect(firstSignal?.aborted).toBe(true));

    rerender(<ImageViewerShell {...props} selectedSurface="mpr" />);
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(2));
    expect(getUploadScalarVolume.mock.calls[1]?.[1]?.signal?.aborted).toBe(false);
  });

  it("revalidates Mask payloads when source expectations or the ApiClient identity changes", async () => {
    const initialViewerInfo = makeMaskMprViewerInfo();
    const resizedViewerInfo: UploadViewerInfo = {
      ...initialViewerInfo,
      axis_sizes: { ...initialViewerInfo.axis_sizes, X: 11 },
      metadata: {
        ...initialViewerInfo.metadata,
        array_shape: [2, 2, 7, 9, 11],
      },
    };
    const clientOneResolvers: Array<(value: ScalarVolumePayload) => void> = [];
    const clientOneGetScalarVolume = vi.fn(
      () =>
        new Promise<ScalarVolumePayload>((resolve) => {
          clientOneResolvers.push(resolve);
        })
    );
    const clientOne = {
      getUploadScalarVolume: clientOneGetScalarVolume,
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;
    let resolveClientTwo!: (value: ScalarVolumePayload) => void;
    const clientTwoGetScalarVolume = vi.fn(
      () =>
        new Promise<ScalarVolumePayload>((resolve) => {
          resolveClientTwo = resolve;
        })
    );
    const clientTwo = {
      getUploadScalarVolume: clientTwoGetScalarVolume,
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;
    const { rerender } = render(
      <ImageViewerShell
        {...makeMaskMprShellProps(initialViewerInfo, clientOne)}
        selectedSurface="mpr"
      />
    );

    await waitFor(() => expect(clientOneGetScalarVolume).toHaveBeenCalledTimes(1));
    await act(async () => clientOneResolvers[0]?.(makeMaskMprPayload()));
    await waitFor(() => expect(screen.getAllByTestId("slice-plane-canvas")).toHaveLength(3));

    rerender(
      <ImageViewerShell
        {...makeMaskMprShellProps(resizedViewerInfo, clientOne)}
        selectedSurface="mpr"
        xAxisSize={11}
      />
    );
    expect(document.querySelectorAll("[data-viewer-mpr-mask-unavailable]")).toHaveLength(3);
    expect(screen.queryAllByTestId("slice-plane-canvas")).toHaveLength(0);
    await waitFor(() => expect(clientOneGetScalarVolume).toHaveBeenCalledTimes(2));
    await act(async () =>
      clientOneResolvers[1]?.(makeMaskMprPayload({ sourceWidth: 11 }))
    );
    await waitFor(() => expect(screen.getAllByTestId("slice-plane-canvas")).toHaveLength(3));

    rerender(
      <ImageViewerShell
        {...makeMaskMprShellProps(resizedViewerInfo, clientTwo)}
        selectedSurface="mpr"
        xAxisSize={11}
      />
    );
    expect(document.querySelectorAll("[data-viewer-mpr-mask-unavailable]")).toHaveLength(3);
    expect(screen.queryAllByTestId("slice-plane-canvas")).toHaveLength(0);
    await waitFor(() => expect(clientTwoGetScalarVolume).toHaveBeenCalledTimes(1));
    await act(async () =>
      resolveClientTwo(makeMaskMprPayload({ sourceWidth: 11 }))
    );
    await waitFor(() => expect(screen.getAllByTestId("slice-plane-canvas")).toHaveLength(3));
  });

  it("does not snap or disclose exact values for a downsampled BOX intensity preview", async () => {
    const intensityViewerInfo = makeIntensityMprViewerInfo();
    const payload = makeMaskMprPayload({
      data: new Uint16Array(4 * 3 * 3)
        .map((_, index) => index + 100)
        .buffer,
      width: 4,
      height: 3,
      depth: 3,
      rawMax: 135,
      downsampleX: 3,
      downsampleY: 4,
      downsampleZ: 3,
      channel: 0,
      time: 0,
      previewPolicy: "auto-v1",
      sampling: "box",
    });
    let resolveScalarVolume!: (value: ScalarVolumePayload) => void;
    const getUploadScalarVolume = vi.fn(
      () =>
        new Promise<ScalarVolumePayload>((resolve) => {
          resolveScalarVolume = resolve;
        })
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={intensityViewerInfo}
        apiClient={apiClient}
        selectedSurface="mpr"
        onSurfaceChange={() => {}}
        selectedDisplayState={intensityViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 4, y: 5, z: 6, t: 0 }}
        debouncedX={4}
        debouncedY={5}
        debouncedZ={6}
        debouncedT={0}
        xAxisSize={10}
        yAxisSize={9}
        zAxisSize={7}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));
    await act(async () => resolveScalarVolume(payload));
    await waitFor(() => expect(screen.getAllByTestId("slice-plane-canvas")).toHaveLength(3));
    const canvases = Object.fromEntries(
      screen.getAllByTestId("slice-plane-canvas").map((canvas) => [
        canvas.dataset.title?.slice(0, 1),
        canvas,
      ])
    );
    expect(canvases.z).toHaveAttribute("data-scalar-slice-index", "2");
    expect(canvases.y).toHaveAttribute("data-scalar-slice-index", "1");
    expect(canvases.x).toHaveAttribute("data-scalar-slice-index", "1");
    expect(canvases.z).toHaveAttribute("data-crosshair-row", "1");
    expect(canvases.z).toHaveAttribute("data-crosshair-col", "1");
    expect(canvases.y).toHaveAttribute("data-crosshair-row", "2");
    expect(canvases.y).toHaveAttribute("data-crosshair-col", "1");
    expect(canvases.x).toHaveAttribute("data-crosshair-row", "2");
    expect(canvases.x).toHaveAttribute("data-crosshair-col", "1");
    expect(canvases.z).toHaveAttribute("data-coordinate-grid-width", "4");
    expect(canvases.z).toHaveAttribute("data-coordinate-grid-height", "3");
    expect(canvases.y).toHaveAttribute("data-coordinate-grid-width", "4");
    expect(canvases.x).toHaveAttribute("data-coordinate-grid-width", "3");
    expect(screen.queryByText("Preview sample")).not.toBeInTheDocument();
    expect(screen.queryByText("Voxel value")).not.toBeInTheDocument();
  });

  it("shows source voxel values for the single selected scalar MPR channel", async () => {
    const scalarPlanes = {
      z: {
        axis: "z" as const,
        label: "Axial",
        axes: ["Y", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      y: {
        axis: "y" as const,
        label: "Coronal",
        axes: ["Z", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      x: {
        axis: "x" as const,
        label: "Sagittal",
        axes: ["Z", "Y"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "brain.nii",
      modality: "medical",
      dims_order: "ZYXC",
      backend_mode: "direct",
      axis_sizes: { T: 1, C: 2, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 1, Z: 0 },
      is_volume: true,
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [1],
          channel_colors: [],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channels: [1],
        volume_channel: 1,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYXC",
        array_shape: [2, 2, 2, 2],
        array_dtype: "uint16",
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["mpr", "volume", "metadata"],
        default_surface: "mpr",
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        default_plane: scalarPlanes.z,
        planes: scalarPlanes,
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["scalar_probe", "channel_visibility"],
        viewer_capabilities: ["mpr", "volume", "metadata"],
      },
    };
    const source = new Uint16Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const apiClient = {
      getUploadScalarVolume: vi.fn(async () => ({
        data: source.buffer,
        width: 2,
        height: 2,
        depth: 2,
        dtype: "uint16",
        bytesPerVoxel: 2,
        rawMin: 10,
        rawMax: 80,
        channel: 1,
        time: 0,
        sourceWidth: 2,
        sourceHeight: 2,
        sourceDepth: 2,
        downsampleX: 1,
        downsampleY: 1,
        downsampleZ: 1,
        previewPolicy: "native-exact-v1",
        sampling: "box",
        sclSlope: 1,
        sclInter: 0,
      })),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={scalarViewerInfo}
        apiClient={apiClient}
        selectedSurface="mpr"
        onSurfaceChange={() => {}}
        selectedDisplayState={scalarViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 1, y: 0, z: 1, t: 0 }}
        debouncedX={1}
        debouncedY={0}
        debouncedZ={1}
        debouncedT={0}
        xAxisSize={2}
        yAxisSize={2}
        zAxisSize={2}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    await waitFor(() =>
      expect(apiClient.getUploadScalarVolume).toHaveBeenCalledWith(
        "file-123",
        expect.objectContaining({
          t: 0,
          channel: 1,
          sampling: "box",
          signal: expect.any(AbortSignal),
        })
      )
    );
    expect(await screen.findByText("Voxel value")).toBeInTheDocument();
    expect(screen.getByText("60")).toBeInTheDocument();
  });

  it("lets scalar Slice Views choose one active channel for slices and probes", async () => {
    const scalarPlanes = {
      z: {
        axis: "z" as const,
        label: "XY plane",
        axes: ["Y", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      y: {
        axis: "y" as const,
        label: "XZ plane",
        axes: ["Z", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      x: {
        axis: "x" as const,
        label: "YZ plane",
        axes: ["Z", "Y"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "two-channel-volume.nii",
      modality: "medical",
      dims_order: "CZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 2, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff", "#00ff00"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff", "#00ff00"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "CZYX",
        array_shape: [2, 2, 2, 2],
        array_dtype: "uint16",
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["mpr", "volume", "metadata"],
        default_surface: "mpr",
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        default_plane: scalarPlanes.z,
        planes: scalarPlanes,
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["scalar_probe", "channel_visibility"],
        viewer_capabilities: ["mpr", "volume", "metadata"],
      },
    };
    const apiClient = {
      getUploadScalarVolume: vi.fn(async (_fileId: string, config?: { channel?: number | null }) => {
        const values = config?.channel === 1 ? [100, 200, 300, 400, 500, 600, 700, 800] : [10, 20, 30, 40, 50, 60, 70, 80];
        const source = new Uint16Array(values);
        return {
          data: source.buffer,
          width: 2,
          height: 2,
          depth: 2,
          dtype: "uint16",
          bytesPerVoxel: 2,
          rawMin: values[0],
          rawMax: values[values.length - 1],
          channel: config?.channel ?? 0,
          time: 0,
          sourceWidth: 2,
          sourceHeight: 2,
          sourceDepth: 2,
          downsampleX: 1,
          downsampleY: 1,
          downsampleZ: 1,
          previewPolicy: "native-exact-v1",
          sampling: "box",
          sclSlope: 1,
          sclInter: 0,
        };
      }),
      uploadSliceUrl: vi.fn((_fileId: string, indices?: { channels?: number[] }) => {
        const params = new URLSearchParams();
        if (Array.isArray(indices?.channels) && indices.channels.length > 0) {
          params.set("channels", indices.channels.join(","));
        }
        const suffix = params.toString() ? `?${params.toString()}` : "";
        return `https://ultra.example.org/v2/uploads/file-123/slice${suffix}`;
      }),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="mpr"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 1, y: 0, z: 1, t: 0 }}
          debouncedX={1}
          debouncedY={0}
          debouncedZ={1}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    await waitFor(() =>
      expect(apiClient.getUploadScalarVolume).toHaveBeenCalledWith(
        "file-123",
        expect.objectContaining({
          t: 0,
          channel: 0,
          sampling: "box",
          signal: expect.any(AbortSignal),
        })
      )
    );
    openAdvancedControls();
    await chooseSelectOption("Volume channel", "Channel 2");

    await waitFor(() =>
      expect(apiClient.getUploadScalarVolume).toHaveBeenLastCalledWith(
        "file-123",
        expect.objectContaining({
          t: 0,
          channel: 1,
          sampling: "box",
          signal: expect.any(AbortSignal),
        })
      )
    );
    expect(screen.getAllByTestId("slice-plane-canvas")[0].dataset.imageUrl).toContain("channels=1");
    expect(await screen.findByText("600")).toBeInTheDocument();
  });

  it("does not eagerly fetch a volume histogram on a cold intensity load", async () => {
    const scalarPlanes = {
      z: {
        axis: "z" as const,
        label: "XY plane",
        axes: ["Y", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      y: {
        axis: "y" as const,
        label: "XZ plane",
        axes: ["Z", "X"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
      x: {
        axis: "x" as const,
        label: "YZ plane",
        axes: ["Z", "Y"],
        pixel_size: { width: 2, height: 2 },
        spacing: { row: 1, col: 1 },
        world_size: { width: 2, height: 2 },
        aspect_ratio: 1,
      },
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "two-channel-volume.nii",
      modality: "medical",
      dims_order: "CZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 2, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 1, Z: 0 },
      is_volume: true,
      is_multichannel: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [1],
          channel_colors: ["#ffffff", "#00ff00"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [1],
        channel_colors: ["#ffffff", "#00ff00"],
        volume_channel: 1,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
        histogram: "/v2/uploads/file-123/histogram",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "CZYX",
        array_shape: [2, 2, 2, 2],
        array_dtype: "int16",
        array_min: 0,
        array_max: 1,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["mpr", "volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        default_plane: scalarPlanes.z,
        planes: scalarPlanes,
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["histogram", "intensity_window", "channel_visibility"],
        viewer_capabilities: ["mpr", "volume", "metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(
        async (_fileId: string, config?: { channel?: number; channels?: number[] }) => {
        const channel = config?.channel ?? config?.channels?.[0] ?? 0;
        return {
          ...histogram,
          dtype: "int16",
          source: "scalar-volume",
          sample_count: 8,
          channels: [channel],
          histogram: {
            ...histogram.histogram,
            min: channel === 1 ? -5 : 10,
            max: channel === 1 ? 500 : 80,
            bins: [1, 2, 2, 3],
            edges: channel === 1 ? [-5, 120, 250, 375, 500] : [10, 30, 50, 70, 80],
            channel_indices: [channel],
          },
        };
        }
      ),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    await act(async () => {
      await Promise.resolve();
    });
    expect(apiClient.getUploadHistogram).not.toHaveBeenCalled();
    openAdvancedControls();
    fireEvent.click(screen.getByRole("button", { name: "Channel 1, source channel 0" }));
    expect(apiClient.getUploadHistogram).not.toHaveBeenCalled();
  });

  it("lets scalar volume rendering switch to a perceptual colormap", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
        histogram: "/v2/uploads/file-123/histogram",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["histogram", "intensity_window", "palette"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    expect(screen.queryByLabelText("Scalar colormap")).not.toBeInTheDocument();
    openAdvancedControls();
    await chooseSelectOption("Scalar colormap", "Viridis");

    await waitFor(() =>
      expect(screen.getByTestId("slice-stack-volume-canvas").dataset.scalarColormap).toBe("viridis")
    );
  });

  it("passes cutaway range changes into the volume renderer state", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    openAdvancedControls();
    fireEvent.change(screen.getByLabelText("Clip X start"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("Clip Z end"), { target: { value: "75" } });

    await waitFor(() => {
      const canvas = screen.getByTestId("slice-stack-volume-canvas");
      expect(canvas.dataset.clipXMin).toBe("0.2");
      expect(canvas.dataset.clipXMax).toBe("1");
      expect(canvas.dataset.clipZMin).toBe("0");
      expect(canvas.dataset.clipZMax).toBe("0.75");
    });
    expect(screen.getByText("20-100%")).toBeInTheDocument();
    expect(screen.getByText("0-75%")).toBeInTheDocument();
  });

  const buildMedicalScalarVolume = (modality: string): UploadViewerInfo => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    return {
      ...viewerInfo,
      original_name: "head-ct.nii",
      modality,
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
        enhancement: "hounsfield:350.000:1800.000",
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "int16",
        array_min: -1024,
        array_max: 3071,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["mpr", "volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
  };

  const renderMedicalScalarVolume = (info: UploadViewerInfo) => {
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;
    function Harness() {
      const [displayState, setDisplayState] = useState(info.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={info}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }
    render(<Harness />);
  };

  it("offers one-click CT window presets that retune the window for medical volumes", async () => {
    renderMedicalScalarVolume(buildMedicalScalarVolume("medical"));
    openAdvancedControls();

    // Wide default window (350/1800) matches no preset.
    const brain = await screen.findByRole("button", { name: "Brain" });
    expect(brain.getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(brain);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Brain" }).getAttribute("aria-pressed")).toBe("true");
      expect((screen.getByLabelText("Window level") as HTMLInputElement).value).toBe("40");
      expect((screen.getByLabelText("Window width") as HTMLInputElement).value).toBe("80");
    });

    fireEvent.click(screen.getByRole("button", { name: "Bone" }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Bone" }).getAttribute("aria-pressed")).toBe("true");
      expect(screen.getByRole("button", { name: "Brain" }).getAttribute("aria-pressed")).toBe("false");
      expect((screen.getByLabelText("Window level") as HTMLInputElement).value).toBe("600");
      expect((screen.getByLabelText("Window width") as HTMLInputElement).value).toBe("2800");
    });
  });

  it("hides CT window presets for non-medical scalar volumes but keeps the window sliders", () => {
    renderMedicalScalarVolume(buildMedicalScalarVolume("microscopy"));
    openAdvancedControls();

    expect(screen.queryByRole("button", { name: "Brain" })).toBeNull();
    expect(screen.queryByRole("group", { name: "CT window presets" })).toBeNull();
    expect(screen.getByLabelText("Window level")).toBeInTheDocument();
  });

  it("enables the Z-cursor cutaway without changing camera mode and exposes a depth scrubber", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 10, height: 10 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 10, height: 10 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 10, Y: 10, X: 10 },
      selected_indices: { T: 0, C: 0, Z: 6 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
        volume_camera_mode: "orthographic",
        volume_clip_min: { x: 0, y: 0, z: 0 },
        volume_clip_max: { x: 1, y: 1, z: 1 },
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [10, 10, 10],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 4, y: 5, z: 6, t: 0 }}
          debouncedX={4}
          debouncedY={5}
          debouncedZ={6}
          debouncedT={0}
          xAxisSize={10}
          yAxisSize={10}
          zAxisSize={10}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    expect(screen.queryByRole("button", { name: "Interior focus" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cutaway" }));

    await waitFor(() => {
      const canvas = screen.getByTestId("slice-stack-volume-canvas");
      // Cutaway changes clipping only; it preserves the user's camera mode.
      expect(canvas.dataset.cutaway).toBe("true");
      expect(canvas.dataset.cameraMode).toBe("orthographic");
    });
    expect(screen.getByText("Cutaway active")).toBeInTheDocument();
    // The user sweeps the cut through Z from the Volume tab.
    expect(screen.getByLabelText("Cutaway depth")).toBeInTheDocument();
  });

  it("passes volume view preset changes into the volume renderer state", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
        volume_view_preset: "iso",
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    openAdvancedControls();
    await chooseSelectOption("Volume view", "XY");

    await waitFor(() =>
      expect(screen.getByTestId("slice-stack-volume-canvas").dataset.viewPreset).toBe("xy")
    );
  });

  it("passes volume camera mode changes into the volume renderer state", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
        volume_camera_mode: "perspective",
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    openAdvancedControls();
    await chooseSelectOption("Camera", "Orthographic");

    await waitFor(() =>
      expect(screen.getByTestId("slice-stack-volume-canvas").dataset.cameraMode).toBe("orthographic")
    );
  });

  it("passes linked X/Y/Z cursor indices into the 3D volume renderer", () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 4, height: 3 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 4, height: 3 },
      aspect_ratio: 4 / 3,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "linked-volume.nii",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 5, Y: 4, X: 6 },
      selected_indices: { T: 0, C: 0, Z: 1 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 1,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [5, 4, 6],
        array_dtype: "uint16",
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["window_level"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={scalarViewerInfo}
        apiClient={apiClient}
        selectedSurface="volume"
        onSurfaceChange={() => {}}
        selectedDisplayState={scalarViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 4, y: 2, z: 3, t: 0 }}
        debouncedX={4}
        debouncedY={2}
        debouncedZ={3}
        debouncedT={0}
        xAxisSize={6}
        yAxisSize={4}
        zAxisSize={5}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute("data-x-index", "4");
    expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute("data-y-index", "2");
    expect(screen.getByTestId("slice-stack-volume-canvas")).toHaveAttribute("data-z-index", "3");
  });

  it("lets scalar volume rendering tune the transfer function", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
        histogram: "/v2/uploads/file-123/histogram",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["histogram", "intensity_window", "palette"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    openAdvancedControls();
    await chooseSelectOption("Transfer preset", "Crisp structures");

    await waitFor(() => {
      const canvas = screen.getByTestId("slice-stack-volume-canvas");
      expect(canvas.dataset.signalFloor).toBe("0.35");
      expect(canvas.dataset.density).toBe("1.4");
    });

    fireEvent.change(screen.getByLabelText("Signal floor"), { target: { value: "20" } });

    await waitFor(() => {
      expect(screen.getByTestId("slice-stack-volume-canvas").dataset.signalFloor).toBe("0.2");
    });
  });

  it("lets scalar volume rendering enable depth lighting", async () => {
    const scalarPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 2, height: 2 },
      spacing: { row: 1, col: 1 },
      world_size: { width: 2, height: 2 },
      aspect_ratio: 1,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "scalar-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 0,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
        histogram: "/v2/uploads/file-123/histogram",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [2, 2, 2],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 100,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z"],
        default_plane: scalarPlane,
        planes: { z: scalarPlane },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        display_capabilities: ["histogram", "intensity_window", "palette"],
        viewer_capabilities: ["volume", "metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    function Harness() {
      const [displayState, setDisplayState] = useState(scalarViewerInfo.display_defaults ?? null);
      return (
        <ImageViewerShell
          viewerInfo={scalarViewerInfo}
          apiClient={apiClient}
          selectedSurface="volume"
          onSurfaceChange={() => {}}
          selectedDisplayState={displayState}
          updateSelectedDisplay={(patch) =>
            setDisplayState((previous) => (previous ? { ...previous, ...patch } : previous))
          }
          clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
          debouncedX={0}
          debouncedY={0}
          debouncedZ={0}
          debouncedT={0}
          xAxisSize={2}
          yAxisSize={2}
          zAxisSize={2}
          tAxisSize={1}
          setSelectedIndex={() => {}}
          selectedCaption=""
          captionLoading={false}
        />
      );
    }

    render(<Harness />);

    openAdvancedControls();
    fireEvent.click(await screen.findByLabelText("Depth lighting"));

    await waitFor(() => {
      const canvas = screen.getByTestId("slice-stack-volume-canvas");
      expect(canvas.dataset.lighting).toBe("true");
      expect(canvas.dataset.lightingStrength).toBe("0.65");
    });

    fireEvent.change(screen.getByLabelText("Lighting strength"), { target: { value: "85" } });

    await waitFor(() => {
      expect(screen.getByTestId("slice-stack-volume-canvas").dataset.lightingStrength).toBe("0.85");
    });
  });

  it.each([
    {
      name: "uniform physical axes",
      units: { x: "um", y: "um", z: "um" },
      expected: "2.062 um",
    },
    {
      name: "mixed physical and voxel axes",
      units: { x: "um", y: "um", z: "voxel" },
      expected: "1.414 vox",
    },
  ])("keeps MPR measurement units honest for $name", async ({ units, expected }) => {
    renderPhysicalViewer(physicalScalarViewerInfo(units), "mpr");

    fireEvent.click(screen.getByLabelText("Measure"));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "measure y-plane start" }).parentElement
      ).toHaveAttribute("data-measure-mode", "true")
    );
    fireEvent.click(screen.getByRole("button", { name: "measure y-plane start" }));
    await waitFor(() => expect(screen.getByText(/0\.000 (?:um|vox)/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "measure y-plane end" }));

    await waitFor(() =>
      expect(document.querySelector(".viewer-mpr-distance-readout")).not.toHaveTextContent("0.000")
    );
    expect(document.querySelector(".viewer-mpr-distance-readout")).toHaveTextContent(expected);
  });

  it("uses one global unit only for uniform axes and keeps mixed-axis volume geometry in voxel space", async () => {
    const uniform = renderPhysicalViewer(
      physicalScalarViewerInfo({ x: "um", y: "um", z: "um" }),
      "volume"
    );
    let summary = await screen.findByLabelText("Volume summary");
    expect(within(summary).getByText("32 x 16 x 24 um")).toBeInTheDocument();
    expect(within(summary).getByText("0.50 x 0.50 x 2.00 um")).toBeInTheDocument();

    uniform.unmount();
    renderPhysicalViewer(
      physicalScalarViewerInfo({ x: "um", y: "um", z: "voxel" }),
      "volume"
    );
    summary = await screen.findByLabelText("Volume summary");
    expect(within(summary).getByText("64 x 32 x 12 vox")).toBeInTheDocument();
    expect(
      within(summary).getByText("X 0.50 um · Y 0.50 um · Z 2.00 voxel")
    ).toBeInTheDocument();
    expect(within(summary).queryByText(/32 x 16 x 24 um/)).not.toBeInTheDocument();
  });

  it("shows physical volume geometry for spacing-aware 3D data", async () => {
    const anisotropicPlane = {
      axis: "z" as const,
      label: "XY plane",
      axes: ["Y", "X"],
      pixel_size: { width: 64, height: 32 },
      spacing: { row: 0.5, col: 0.5 },
      world_size: { width: 32, height: 16 },
      aspect_ratio: 2,
    };
    const scalarViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "anisotropic-volume.nii",
      modality: "medical",
      dims_order: "ZYX",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 12, Y: 32, X: 64 },
      selected_indices: { T: 0, C: 0, Z: 6 },
      is_volume: true,
      display_defaults: {
        ...(viewerInfo.display_defaults ?? {
          enhancement: "d",
          negative: false,
          rotate: 0,
          fusion_method: "a",
          channel_mode: "single",
          channels: [0],
          channel_colors: ["#ffffff"],
          time_index: 0,
          z_index: 6,
        }),
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        volume_channel: 0,
      },
      service_urls: {
        ...viewerInfo.service_urls,
        scalar_volume: "/v2/uploads/file-123/scalar-volume",
        histogram: "/v2/uploads/file-123/histogram",
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [12, 32, 64],
        array_dtype: "uint16",
        array_min: 0,
        array_max: 1024,
        physical_spacing: { x: 0.5, y: 0.5, z: 2 },
        physical_spacing_unit: "mm",
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["volume", "metadata"],
        default_surface: "volume",
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        default_plane: anisotropicPlane,
        planes: {
          z: anisotropicPlane,
          y: {
            ...anisotropicPlane,
            axis: "y" as const,
            label: "XZ plane",
            axes: ["Z", "X"],
            pixel_size: { width: 64, height: 12 },
            spacing: { row: 2, col: 0.5 },
            world_size: { width: 32, height: 24 },
            aspect_ratio: 1.3333,
          },
          x: {
            ...anisotropicPlane,
            axis: "x" as const,
            label: "YZ plane",
            axes: ["Z", "Y"],
            pixel_size: { width: 32, height: 12 },
            spacing: { row: 2, col: 0.5 },
            world_size: { width: 16, height: 24 },
            aspect_ratio: 0.6667,
          },
        },
        volume_mode: "scalar",
        render_policy: "scalar",
        diagnostic_surface: "mpr",
        measurement_policy: "spacing-aware",
        display_capabilities: ["histogram", "physical_scale", "volume_context", "window_level"],
        viewer_capabilities: ["volume", "metadata", "physical_scale"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => ({
        ...histogram,
        source: "scalar-volume",
        sample_count: 24576,
        histogram: {
          ...histogram.histogram,
          min: 0,
          max: 1024,
        },
      })),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={scalarViewerInfo}
        apiClient={apiClient}
        selectedSurface="volume"
        onSurfaceChange={() => {}}
        selectedDisplayState={scalarViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 6, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={6}
        debouncedT={0}
        xAxisSize={64}
        yAxisSize={32}
        zAxisSize={12}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    expect(screen.getByTestId("slice-stack-volume-canvas")).toBeInTheDocument();
    expect(screen.queryByLabelText("Window level")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Advanced rendering" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Advanced" })).not.toBeInTheDocument();
    const volumeSummary = await screen.findByLabelText("Volume summary");
    expect(volumeSummary).toHaveAttribute("data-viewer-volume-readout", "compact");
    expect(within(volumeSummary).getByText("Extent")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("32 x 16 x 24 mm")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("Spacing")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("0.50 x 0.50 x 2.00 mm")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("Anisotropic voxels")).toBeInTheDocument();
  });

  it("keeps metadata summary readable and raw metadata collapsed", async () => {
    const metadataViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "brain-ct.nii.gz",
      modality: "medical",
      dims_order: "ZYX",
      axis_sizes: { T: 1, C: 1, Z: 36, Y: 246, X: 246 },
      is_volume: true,
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [36, 246, 246],
        array_dtype: "int16",
        physical_spacing: { x: 0.48, y: 0.48, z: 5 },
        header: { scanner: "CT-1" },
        dicom: { modality: "CT", wnd_center: 1023.5, wnd_width: 4094.9 },
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["metadata"],
        default_surface: "metadata",
        volume_mode: "scalar",
        render_policy: "scalar",
        display_capabilities: [],
        viewer_capabilities: ["metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={metadataViewerInfo}
        apiClient={apiClient}
        selectedSurface="metadata"
        onSurfaceChange={() => {}}
        selectedDisplayState={metadataViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={0}
        debouncedT={0}
        xAxisSize={246}
        yAxisSize={246}
        zAxisSize={36}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    const metadataSummary = screen.getByLabelText("Image metadata");
    expect(metadataSummary).toHaveAttribute("data-viewer-metadata-layout", "groups");
    expect(within(metadataSummary).getByText("Array shape")).toBeInTheDocument();
    expect(screen.getByText("36 × 246 × 246")).toBeInTheDocument();
    expect(within(metadataSummary).getByText("Voxel spacing")).toBeInTheDocument();
    expect(screen.getByText("X 0.480 · Y 0.480 · Z 5.000")).toBeInTheDocument();
    expect(within(metadataSummary).getByText("Field of view")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Technical details" })).toBeInTheDocument();
    expect(screen.queryByText("Image header")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Technical details" }));

    expect(await screen.findByText("Image header")).toBeInTheDocument();
    expect(screen.getByText("scanner")).toBeInTheDocument();
    expect(screen.getByText("CT-1")).toBeInTheDocument();
  });

  it("renders backend-shaped phys pixel units per axis without a mixed composite quantity", () => {
    const mixedUnitViewerInfo: UploadViewerInfo = {
      ...viewerInfo,
      original_name: "mixed-calibration.ome.tiff",
      dims_order: "ZYX",
      axis_sizes: { T: 1, C: 1, Z: 8, Y: 20, X: 40 },
      is_volume: true,
      phys: {
        ...viewerInfo.phys,
        pixel_units: ["um", "um", "pixel", "frame"],
      },
      metadata: {
        ...viewerInfo.metadata,
        dims_order: "ZYX",
        array_shape: [8, 20, 40],
        physical_spacing: { x: 0.5, y: 0.5, z: 1 },
        physical_spacing_unit: undefined,
        spacing_units: undefined,
      },
      viewer: {
        ...viewerInfo.viewer,
        available_surfaces: ["metadata"],
        default_surface: "metadata",
        display_capabilities: [],
        viewer_capabilities: ["metadata"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/preview"),
    } as unknown as ApiClient;

    render(
      <ImageViewerShell
        viewerInfo={mixedUnitViewerInfo}
        apiClient={apiClient}
        selectedSurface="metadata"
        onSurfaceChange={() => {}}
        selectedDisplayState={mixedUnitViewerInfo.display_defaults ?? null}
        updateSelectedDisplay={() => {}}
        clampedIndices={{ x: 0, y: 0, z: 0, t: 0 }}
        debouncedX={0}
        debouncedY={0}
        debouncedZ={0}
        debouncedT={0}
        xAxisSize={40}
        yAxisSize={20}
        zAxisSize={8}
        tAxisSize={1}
        setSelectedIndex={() => {}}
        selectedCaption=""
        captionLoading={false}
      />
    );

    const metadataSummary = screen.getByLabelText("Image metadata");
    expect(
      within(metadataSummary).getByText("X 0.500 um · Y 0.500 um · Z 1.000 pixel")
    ).toBeInTheDocument();
    expect(within(metadataSummary).queryByText("Field of view")).not.toBeInTheDocument();
    expect(within(metadataSummary).queryByText("Sampling")).not.toBeInTheDocument();
    expect(
      within(metadataSummary).queryByText("X 0.500 · Y 0.500 · Z 1.000")
    ).not.toBeInTheDocument();
  });

  it("hydrates same-file NGFF LUT transport into the actual rendered slice URL", async () => {
    const baseNgffInfo: UploadViewerInfo = {
      ...viewerInfo,
      file_id: "same-ngff-file",
      original_name: "channels.ome.zarr",
      dims_order: "CYX",
      axis_sizes: { T: 1, C: 2, Z: 1, Y: 8, X: 12 },
      is_multichannel: true,
      phys: {
        ...viewerInfo.phys,
        channel_names: ["DAPI", "EGFP"],
        channel_colors: [
          { index: 0, hex: "#0000ff", rgb: [0, 0, 255] },
          { index: 1, hex: "#00ff00", rgb: [0, 255, 0] },
        ],
      },
      display_defaults: {
        ...viewerInfo.display_defaults!,
        channel_mode: "composite",
        channels: [0, 1],
        channel_colors: ["#0000ff", "#00ff00"],
      },
      metadata: {
        ...viewerInfo.metadata,
        reader: "ngff",
        dims_order: "CYX",
        array_shape: [2, 8, 12],
      },
      viewer: {
        ...viewerInfo.viewer,
        render_policy: "scalar",
        channel_mode: "composite",
        display_capabilities: ["channel_visibility", "channel_color"],
      },
    };
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadSliceUrl: vi.fn(buildSliceUrl),
      uploadPreviewUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/same-ngff-file/preview"),
    } as unknown as ApiClient;
    const props = {
      apiClient,
      selectedSurface: "2d" as const,
      onSurfaceChange: () => {},
      selectedDisplayState: baseNgffInfo.display_defaults ?? null,
      updateSelectedDisplay: () => {},
      clampedIndices: { x: 0, y: 0, z: 0, t: 0 },
      debouncedX: 0,
      debouncedY: 0,
      debouncedZ: 0,
      debouncedT: 0,
      xAxisSize: 12,
      yAxisSize: 8,
      zAxisSize: 1,
      tAxisSize: 1,
      setSelectedIndex: () => {},
      selectedCaption: "",
      captionLoading: false,
    };

    const { rerender } = render(<ImageViewerShell {...props} viewerInfo={baseNgffInfo} />);
    expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).not.toContain(
      "channel_colors="
    );

    const hydratedNgffInfo: UploadViewerInfo = {
      ...baseNgffInfo,
      viewer: {
        ...baseNgffInfo.viewer,
        display_capabilities: [
          "channel_visibility",
          "channel_color",
          "channel_lut_transport",
        ],
      },
    };
    rerender(<ImageViewerShell {...props} viewerInfo={hydratedNgffInfo} />);

    await waitFor(() => {
      const renderedUrl = new URL(
        screen.getByTestId("direct-plane-image").dataset.imageUrl ?? ""
      );
      expect(renderedUrl.searchParams.get("channels")).toBe("0,1");
      expect(renderedUrl.searchParams.get("channel_colors")).toBe("#0000ff,#00ff00");
    });
  });
});
