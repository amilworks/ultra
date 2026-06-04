import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { UploadViewerHistogramResponse, UploadViewerInfo } from "@/types";

import { ImageViewerShell } from "./ImageViewerShell";

vi.mock("./DirectPlaneImage", () => ({
  DirectPlaneImage: ({ imageUrl }: { imageUrl: string }) => (
    <div data-testid="direct-plane-image" data-image-url={imageUrl} />
  ),
}));

vi.mock("./SlicePlaneCanvas", () => ({
  SlicePlaneCanvas: ({ imageUrl, title }: { imageUrl: string; title: string }) => (
    <div data-testid="slice-plane-canvas" data-image-url={imageUrl} data-title={title} />
  ),
}));

vi.mock("./SliceStackVolumeCanvas", () => ({
  SliceStackVolumeCanvas: ({
    displayState,
    xIndex,
    yIndex,
    zIndex,
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
    } | null;
    xIndex?: number;
    yIndex?: number;
    zIndex?: number;
  }) => (
    <div
      data-testid="slice-stack-volume-canvas"
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
      data-clip-z-min={displayState?.volume_clip_min == null ? "" : String(displayState.volume_clip_min.z)}
      data-clip-z-max={displayState?.volume_clip_max == null ? "" : String(displayState.volume_clip_max.z)}
      data-x-index={xIndex == null ? "" : String(xIndex)}
      data-y-index={yIndex == null ? "" : String(yIndex)}
      data-z-index={zIndex == null ? "" : String(zIndex)}
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
  }
  if (config?.cacheKey) {
    params.set("cache_key", config.cacheKey);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return `https://ultra.example.org/v2/uploads/${fileId}/slice${suffix}`;
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
  it("uses shadcn selects instead of native browser dropdowns", () => {
    const source = readFileSync(resolve(process.cwd(), "src/components/viewer/ImageViewerShell.tsx"), "utf8");

    expect(source).not.toMatch(/<select\b/);
    expect(source).toMatch(/<SelectTrigger/);
  });

  it("uses upload histograms to drive direct-image intensity windows", async () => {
    const apiClient = {
      getUploadHistogram: vi.fn(async () => histogram),
      uploadDisplayUrl: vi.fn(buildDisplayUrl),
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
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

    await waitFor(() => expect(apiClient.getUploadHistogram).toHaveBeenCalledWith("file-123", { bins: 256 }));
    const centerSlider = await screen.findByLabelText("Window center");
    fireEvent.change(centerSlider, { target: { value: "1002" } });

    await waitFor(() =>
      expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain(
        "enhancement=hounsfield%3A1002.000%3A3.000"
      )
    );
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
        render_policy: "display",
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
      uploadSliceUrl: vi.fn(() => "https://ultra.example.org/v2/uploads/file-123/slice"),
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
    fireEvent.click(screen.getByRole("button", { name: "Channel 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Channel 3" }));

    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("channels=1");
    });
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
        display_capabilities: ["slice_navigation", "channel_visibility", "physical_scale"],
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
    });

    fireEvent.change(screen.getByLabelText("Z slice"), { target: { value: "0" } });
    await waitFor(() => expect(screen.getByTestId("direct-plane-image").dataset.imageUrl).toContain("z=0"));

    fireEvent.click(screen.getByRole("button", { name: "DAPI" }));
    await waitFor(() => {
      const imageUrl = screen.getByTestId("direct-plane-image").dataset.imageUrl ?? "";
      expect(imageUrl).toContain("channels=0");
      expect(imageUrl).not.toContain("channels=0%2C2");
      expect(imageUrl).not.toContain("channels=0,2");
    });
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
        volume_channel: 0,
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
      expect(apiClient.getUploadScalarVolume).toHaveBeenCalledWith("file-123", {
        t: 0,
        channel: 1,
      })
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
      expect(apiClient.getUploadScalarVolume).toHaveBeenCalledWith("file-123", {
        t: 0,
        channel: 0,
      })
    );
    openAdvancedControls();
    await chooseSelectOption("Volume channel", "Channel 2");

    await waitFor(() =>
      expect(apiClient.getUploadScalarVolume).toHaveBeenLastCalledWith("file-123", {
        t: 0,
        channel: 1,
      })
    );
    expect(screen.getAllByTestId("slice-plane-canvas")[0].dataset.imageUrl).toContain("channels=1");
    expect(await screen.findByText("600")).toBeInTheDocument();
  });

  it("uses scalar volume histograms for the selected volume channel", async () => {
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
      getUploadHistogram: vi.fn(async (_fileId: string, config?: { channels?: number[] }) => {
        const channel = config?.channels?.[0] ?? 0;
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
      }),
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

    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenCalledWith("file-123", { bins: 256, channels: [1] })
    );
    openAdvancedControls();
    expect(await screen.findByText("int16 • 8 samples")).toBeInTheDocument();
    expect(screen.getByText("-5-500")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Channel 1" }));

    await waitFor(() =>
      expect(apiClient.getUploadHistogram).toHaveBeenLastCalledWith("file-123", { bins: 256, channels: [0] })
    );
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
    expect(within(volumeSummary).getByText("Volume")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("32 x 16 x 24")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("Spacing")).toBeInTheDocument();
    expect(within(volumeSummary).getByText("0.50 x 0.50 x 2.00")).toBeInTheDocument();
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

    const metadataSummary = screen.getByLabelText("Metadata at a glance");
    expect(metadataSummary).toHaveAttribute("data-viewer-metadata-layout", "facts");
    expect(within(metadataSummary).getByText("Shape")).toBeInTheDocument();
    expect(screen.getByText("36 × 246 × 246")).toBeInTheDocument();
    expect(within(metadataSummary).getByText("Spacing")).toBeInTheDocument();
    expect(screen.getByText("z=5.000 y=0.480 x=0.480")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Technical details" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Raw metadata" })).not.toBeInTheDocument();
    expect(screen.queryByText("Image Header")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Technical details" }));

    expect(await screen.findByText("Image Header")).toBeInTheDocument();
    expect(screen.getByText("scanner")).toBeInTheDocument();
    expect(screen.getByText("CT-1")).toBeInTheDocument();
  });
});
