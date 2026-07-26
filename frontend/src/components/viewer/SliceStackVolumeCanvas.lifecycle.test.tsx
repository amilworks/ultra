import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type * as ThreeModule from "three";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";
import type * as ScalarVolumeModule from "./scalarVolume";
import type { ResolvedScalarRendering } from "./scalarMask";

const {
  rendererConstruction,
  rendererRender,
  textureInitialization,
  scalarPreparation,
  scalarUniformSets,
  multichannelUniformSets,
  controlEventListeners,
  gpuLimits,
} = vi.hoisted(() => ({
  rendererConstruction: vi.fn(),
  rendererRender: vi.fn(),
  textureInitialization: vi.fn(),
  scalarPreparation: vi.fn(),
  scalarUniformSets: [] as Array<Record<string, { value: unknown }>>,
  multichannelUniformSets: [] as Array<Record<string, { value: unknown }>>,
  controlEventListeners: new Map<string, Set<() => void>>(),
  gpuLimits: { max3DTextureSize: 0 },
}));

vi.mock("three", async () => {
  const actual = await vi.importActual<typeof ThreeModule>("three");
  class WebGLRenderer {
    readonly capabilities = { isWebGL2: true };
    readonly domElement = document.createElement("canvas");
    outputColorSpace = actual.SRGBColorSpace;

    constructor() {
      rendererConstruction();
    }

    getContext() {
      return {
        getParameter: () => gpuLimits.max3DTextureSize,
      };
    }
    setPixelRatio() {}
    setClearColor() {}
    setSize() {}
    render() {
      rendererRender();
    }
    initTexture() {
      textureInitialization();
    }
    dispose() {}
  }
  class ShaderMaterial extends actual.ShaderMaterial {
    constructor(parameters?: ThreeModule.ShaderMaterialParameters) {
      super(parameters);
      const uniforms = this.uniforms as Record<string, { value: unknown }>;
      if (uniforms.uWindowLow || uniforms.uMaskThreshold) {
        scalarUniformSets.push(uniforms);
      }
      if (uniforms.uChannelCount) {
        multichannelUniformSets.push(uniforms);
      }
    }
  }
  return { ...actual, ShaderMaterial, WebGLRenderer };
});

vi.mock("./scalarVolume", async () => {
  const actual = await vi.importActual<typeof ScalarVolumeModule>("./scalarVolume");
  return {
    ...actual,
    MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES: 64,
    prepareScalarVolume: (...args: Parameters<typeof actual.prepareScalarVolume>) => {
      scalarPreparation(...args);
      return actual.prepareScalarVolume(...args);
    },
  };
});

vi.mock("three/examples/jsm/controls/TrackballControls.js", async () => {
  const three = await vi.importActual<typeof ThreeModule>("three");
  return {
    TrackballControls: class {
      readonly target = new three.Vector3();
      noPan = false;
      noZoom = false;
      noRotate = false;
      staticMoving = true;
      rotateSpeed = 1;
      zoomSpeed = 1;
      panSpeed = 1;
      minDistance = 0;
      maxDistance = 1;
      minZoom = 0;
      maxZoom = 1;
      addEventListener(name: string, listener: () => void) {
        const listeners = controlEventListeners.get(name) ?? new Set<() => void>();
        listeners.add(listener);
        controlEventListeners.set(name, listeners);
      }
      removeEventListener(name: string, listener: () => void) {
        controlEventListeners.get(name)?.delete(listener);
      }
      update() {}
      handleResize() {}
      dispose() {}
    },
  };
});

import { SliceStackVolumeCanvas } from "./SliceStackVolumeCanvas";

const plane = {
  axis: "z" as const,
  label: "XY plane",
  axes: ["Y", "X"],
  pixel_size: { width: 2, height: 2 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 2, height: 2 },
  aspect_ratio: 1,
};

const viewerInfo = {
  file_id: "file-nifti",
  original_name: "brain.nii",
  modality: "medical",
  backend_mode: "scalar",
  dims_order: "ZYX",
  axis_sizes: { T: 2, C: 1, Z: 2, Y: 2, X: 2 },
  selected_indices: { T: 0, C: 0, Z: 0 },
  is_volume: true,
  is_timeseries: true,
  is_multichannel: false,
  display_defaults: null,
  metadata: {
    reader: "nifti-1",
    dims_order: "ZYX",
    array_shape: [2, 2, 2],
    array_dtype: "uint16",
    scene_count: 1,
    physical_spacing: { x: 1, y: 1, z: 1 },
  },
  viewer: {
    status: "ready",
    backend_mode: "scalar",
    default_surface: "volume",
    available_surfaces: ["volume"],
    default_axis: "z",
    slice_axes: ["z"],
    default_plane: plane,
    planes: { z: plane },
    volume_mode: "scalar",
    render_policy: "scalar",
    texture_policy: "linear",
    service_urls: { scalar_volume: "/v2/uploads/file-nifti/scalar-volume" },
  },
  service_urls: { scalar_volume: "/v2/uploads/file-nifti/scalar-volume" },
} as unknown as UploadViewerInfo;

const payload: ScalarVolumePayload = {
  data: new Uint16Array([0, 1, 2, 3, 4, 5, 6, 7]).buffer,
  width: 2,
  height: 2,
  depth: 2,
  dtype: "uint16",
  bytesPerVoxel: 2,
  rawMin: 0,
  rawMax: 7,
  sclSlope: 1,
  sclInter: -1024,
  channel: 0,
  time: 0,
  sourceWidth: 2,
  sourceHeight: 2,
  sourceDepth: 2,
  downsampleX: 1,
  downsampleY: 1,
  downsampleZ: 1,
  previewPolicy: "exact-v1",
  sampling: "box",
};

const microscopyViewerInfo = ({
  fileId,
  channels = 3,
  times = 1,
  width = 2,
  height = 2,
  depth = 2,
}: {
  fileId: string;
  channels?: number;
  times?: number;
  width?: number;
  height?: number;
  depth?: number;
}): UploadViewerInfo =>
  ({
    ...viewerInfo,
    file_id: fileId,
    original_name: `${fileId}.ome.tiff`,
    modality: "microscopy",
    axis_sizes: { T: times, C: channels, Z: depth, Y: height, X: width },
    selected_indices: { T: 0, C: 0, Z: 0 },
    is_timeseries: times > 1,
    is_multichannel: channels > 1,
    metadata: {
      ...viewerInfo.metadata,
      sha256: `sha-${fileId}`,
      array_shape: [times, channels, depth, height, width],
    },
    viewer: {
      ...viewerInfo.viewer,
      backend_mode: "direct",
      volume_mode: "slice_stack",
      render_policy: "scalar",
      available_surfaces: ["2d", "metadata", "volume"],
    },
  }) as UploadViewerInfo;

const channelPayload = ({
  channel,
  time = 0,
  sourceWidth = 2,
  sourceHeight = 2,
  sourceDepth = 2,
  width = 2,
  height = 2,
  depth = 2,
}: {
  channel: number;
  time?: number;
  sourceWidth?: number;
  sourceHeight?: number;
  sourceDepth?: number;
  width?: number;
  height?: number;
  depth?: number;
}): ScalarVolumePayload => ({
  ...payload,
  data: new Uint16Array(width * height * depth).fill(channel + 1).buffer,
  channel,
  time,
  width,
  height,
  depth,
  sourceWidth,
  sourceHeight,
  sourceDepth,
  downsampleX: Math.ceil(sourceWidth / width),
  downsampleY: Math.ceil(sourceHeight / height),
  downsampleZ: Math.ceil(sourceDepth / depth),
  previewPolicy: "auto-v1",
});

const resolvedIntensity = (
  channel = 0,
  time = 0
): ResolvedScalarRendering => ({
  channel,
  time,
  requestedMode: "intensity",
  effectiveMode: "intensity",
  sampling: "box",
  threshold: null,
});

const resolvedMask = (threshold: number): ResolvedScalarRendering => ({
  channel: 0,
  time: 0,
  requestedMode: "mask",
  effectiveMode: "mask",
  sampling: "nearest",
  threshold,
});

beforeEach(() => {
  rendererConstruction.mockClear();
  rendererRender.mockClear();
  textureInitialization.mockClear();
  scalarPreparation.mockClear();
  scalarUniformSets.length = 0;
  multichannelUniformSets.length = 0;
  controlEventListeners.clear();
  gpuLimits.max3DTextureSize = 0;
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    }
  );
  vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal("WebGL2RenderingContext", { MAX_3D_TEXTURE_SIZE: 0x8073 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SliceStackVolumeCanvas scalar lifecycle", () => {

  it("uses nearest mask delivery and changes the raw threshold without refetching", async () => {
    const maskInfo = {
      ...microscopyViewerInfo({ fileId: "mask-volume", channels: 1 }),
      scalar_mask_capability: {
        version: 1,
        source_authority: "original",
        source_format: "ome-tiff",
        source_sha256: "sha-mask-volume",
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
      data_semantics: {
        kind: "probability_mask",
        basis: "bounded_scalar_profile",
        strength: "suggested",
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
        },
      },
    } as UploadViewerInfo;
    const nearestPayload: ScalarVolumePayload = {
      ...payload,
      rawMax: 255,
      previewPolicy: "mask-native-integer-v1",
      sampling: "nearest",
    };
    let resolveNearestPayload!: (value: ScalarVolumePayload) => void;
    const getUploadScalarVolume = vi.fn(
      () =>
        new Promise<ScalarVolumePayload>((resolve) => {
          resolveNearestPayload = resolve;
        })
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const baseDisplay = {
      enhancement: "d",
      negative: false,
      fusion_method: "a",
      channels: [0],
      scalar_render_mode: "mask",
      scalar_threshold_method: "otsu-256-v1",
      scalar_threshold_value: 120,
    } as NonNullable<UploadViewerInfo["display_defaults"]>;
    const { rerender } = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="mask-volume"
        viewerInfo={maskInfo}
        displayState={baseDisplay}
        zIndex={0}
        tIndex={0}
        resolvedScalarRendering={resolvedMask(120)}
      />
    );

    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));
    expect(document.querySelector("[data-viewer-surface='volume']")).toHaveAttribute(
      "data-viewer-mask-texture-ready",
      "false"
    );
    expect(textureInitialization).not.toHaveBeenCalled();
    await act(async () => resolveNearestPayload(nearestPayload));
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    expect(document.querySelector("[data-viewer-surface='volume']")).toHaveAttribute(
      "data-viewer-mask-texture-ready",
      "true"
    );
    expect(getUploadScalarVolume).toHaveBeenCalledWith(
      "mask-volume",
      expect.objectContaining({ channel: 0, t: 0, sampling: "nearest" })
    );
    expect(scalarUniformSets[0]?.uMaskThreshold.value).toBe(120);
    expect(scalarPreparation).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(AbortSignal),
      "raw-integer"
    );
    const requestAnimationFrameMock = vi.mocked(window.requestAnimationFrame);
    const initialFrame =
      requestAnimationFrameMock.mock.calls[
        requestAnimationFrameMock.mock.calls.length - 1
      ]?.[0];
    expect(initialFrame).toBeDefined();
    act(() => initialFrame?.(0));
    expect(rendererRender).toHaveBeenCalledTimes(1);
    requestAnimationFrameMock.mockClear();
    const timeoutSpy = vi.spyOn(window, "setTimeout");

    rerender(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="mask-volume"
        viewerInfo={maskInfo}
        displayState={{
          ...baseDisplay,
          scalar_threshold_method: "manual",
          scalar_threshold_value: 140,
        }}
        zIndex={0}
        tIndex={0}
        resolvedScalarRendering={resolvedMask(140)}
      />
    );

    await waitFor(() =>
      expect(scalarUniformSets[0]?.uMaskThreshold.value).toBe(140)
    );
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
    expect(rendererConstruction).toHaveBeenCalledTimes(1);
    expect(textureInitialization).toHaveBeenCalledTimes(1);
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(1);
    const thresholdFrame = requestAnimationFrameMock.mock.calls[0]?.[0];
    act(() => thresholdFrame?.(0));
    expect(rendererRender).toHaveBeenCalledTimes(2);
    requestAnimationFrameMock.mockClear();
    act(() => {
      controlEventListeners.get("start")?.forEach((listener) => listener());
      controlEventListeners.get("change")?.forEach((listener) => listener());
      controlEventListeners.get("end")?.forEach((listener) => listener());
    });
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(1);
    const cameraFrame = requestAnimationFrameMock.mock.calls[0]?.[0];
    act(() => cameraFrame?.(0));
    expect(rendererRender).toHaveBeenCalledTimes(3);
    expect(
      timeoutSpy.mock.calls.some(
        ([, delay]) => typeof delay === "number" && delay > 0 && delay < 200
      )
    ).toBe(false);
    timeoutSpy.mockRestore();
  });

  it("keeps one renderer and one scalar load across cutaway, Z, and window uniform changes", async () => {
    const getUploadScalarVolume = vi.fn(async () => payload);
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const baseDisplay = {
      enhancement: "hounsfield:40:80",
      negative: false,
      fusion_method: "a",
      channels: [0],
      volume_clip_min: { x: 0, y: 0, z: 0 },
      volume_clip_max: { x: 1, y: 1, z: 1 },
    } as NonNullable<UploadViewerInfo["display_defaults"]>;
    const { rerender } = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={baseDisplay}
        zIndex={0}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
        volumeCutaway={false}
      />
    );

    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
    expect(rendererConstruction).toHaveBeenCalledTimes(1);

    rerender(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={{
          ...baseDisplay,
          enhancement: "hounsfield:-600:1500",
          negative: true,
          scalar_colormap: "viridis",
          volume_signal_floor: 0.35,
          volume_density: 2.2,
          volume_lighting: true,
          volume_lighting_strength: 0.8,
        }}
        zIndex={1}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
        volumeCutaway
      />
    );

    await waitFor(() => expect(rendererConstruction).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
    expect(textureInitialization).toHaveBeenCalledTimes(1);
    const uniforms = scalarUniformSets[0];
    expect((uniforms?.uVoxelStep.value as ThreeModule.Vector3).x).toBeCloseTo(0.5, 8);
    expect(uniforms?.uColorMap.value).toBe(1);
    expect(uniforms?.uSignalFloor.value).toBe(0.35);
    expect(uniforms?.uDensityScale.value).toBe(2.2);
    expect(uniforms?.uLightingEnabled.value).toBe(true);
    expect(uniforms?.uLightingStrength.value).toBe(0.8);
    expect(uniforms?.uInvert.value).toBe(true);
  });

  it("prevents an obsolete deferred generation from uploading after source B wins", async () => {
    let resolveA!: (value: ScalarVolumePayload) => void;
    const deferredA = new Promise<ScalarVolumePayload>((resolve) => {
      resolveA = resolve;
    });
    const seenSignals: AbortSignal[] = [];
    const getUploadScalarVolume = vi.fn(
      async (_fileId: string, config?: { t?: number | null; signal?: AbortSignal }) => {
        if (config?.signal) {
          seenSignals.push(config.signal);
        }
        return config?.t === 0 ? deferredA : { ...payload, time: config?.t ?? 0 };
      }
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const displayState = {
      enhancement: "hounsfield:40:80",
      negative: false,
      fusion_method: "a",
      channels: [0],
    } as NonNullable<UploadViewerInfo["display_defaults"]>;
    const { rerender } = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={displayState}
        zIndex={0}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));

    rerender(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={displayState}
        zIndex={0}
        tIndex={1}
        resolvedScalarRendering={resolvedIntensity(0, 1)}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    const lastScalarVolumeCall =
      getUploadScalarVolume.mock.calls[getUploadScalarVolume.mock.calls.length - 1];
    expect(lastScalarVolumeCall?.[1]?.t).toBe(1);
    expect(seenSignals[0]?.aborted).toBe(true);
    const sourceBUniforms = scalarUniformSets[scalarUniformSets.length - 1];
    const sourceBWindowLow = sourceBUniforms?.uWindowLow.value;

    rerender(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={{ ...displayState, enhancement: "hounsfield:-800:400" }}
        zIndex={0}
        tIndex={1}
        resolvedScalarRendering={resolvedIntensity(0, 1)}
      />
    );
    await waitFor(() => expect(sourceBUniforms?.uWindowLow.value).not.toBe(sourceBWindowLow));
    expect(rendererConstruction).toHaveBeenCalledTimes(2);
    expect(textureInitialization).toHaveBeenCalledTimes(1);

    resolveA(payload);
    await Promise.resolve();
    await Promise.resolve();

    expect(textureInitialization).toHaveBeenCalledTimes(1);
  });
});

describe("SliceStackVolumeCanvas multichannel admission", () => {
  const displayState = (channels: number[]) =>
    ({
      enhancement: "d",
      negative: false,
      fusion_method: "a",
      channels,
    }) as NonNullable<UploadViewerInfo["display_defaults"]>;

  it("loads sequentially but publishes and binds only after every selected channel validates", async () => {
    const resolvers = new Map<number, (value: ScalarVolumePayload) => void>();
    const getUploadScalarVolume = vi.fn(
      async (_fileId: string, config?: { channel?: number | null }) =>
        new Promise<ScalarVolumePayload>((resolve) => {
          resolvers.set(config?.channel ?? -1, resolve);
        })
    );
    const info = microscopyViewerInfo({ fileId: "sequential", channels: 3 });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;

    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="sequential"
        viewerInfo={info}
        displayState={displayState([0, 1, 2])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );

    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume.mock.calls[0]?.[1]?.channel).toBe(0);
    expect(textureInitialization).not.toHaveBeenCalled();

    await act(async () => resolvers.get(0)?.(channelPayload({ channel: 0 })));
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(2));
    expect(getUploadScalarVolume.mock.calls[1]?.[1]?.channel).toBe(1);
    expect(textureInitialization).not.toHaveBeenCalled();

    await act(async () => resolvers.get(1)?.(channelPayload({ channel: 1 })));
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(3));
    expect(getUploadScalarVolume.mock.calls[2]?.[1]?.channel).toBe(2);
    expect(textureInitialization).not.toHaveBeenCalled();

    await act(async () => resolvers.get(2)?.(channelPayload({ channel: 2 })));
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(3));
    expect(
      multichannelUniformSets[multichannelUniformSets.length - 1]?.uChannelCount.value
    ).toBe(3);
    expect(
      (multichannelUniformSets[multichannelUniformSets.length - 1]?.uVoxelStep.value as ThreeModule.Vector3).x
    ).toBeCloseTo(0.5, 8);
  });

  it("rejects the first-probe aggregate before channel two, cache publication, or texture bind", async () => {
    const getUploadScalarVolume = vi.fn(async (_fileId: string, config?: { channel?: number | null }) =>
      channelPayload({
        channel: config?.channel ?? 0,
        sourceWidth: 4,
        sourceHeight: 4,
        sourceDepth: 1,
        width: 4,
        height: 4,
        depth: 1,
      })
    );
    const info = microscopyViewerInfo({
      fileId: "over-budget",
      channels: 3,
      width: 4,
      height: 4,
      depth: 1,
    });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const { rerender } = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="over-budget"
        viewerInfo={info}
        displayState={displayState([0, 1, 2])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );

    await waitFor(() => expect(screen.getByText(/aggregate limit/i)).toBeTruthy());
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
    expect(scalarPreparation).not.toHaveBeenCalled();
    expect(textureInitialization).not.toHaveBeenCalled();
    expect(multichannelUniformSets[0]?.uChannelCount.value).toBe(0);

    rerender(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="over-budget"
        viewerInfo={info}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(2);
  });

  it("aborts an in-flight later channel without publishing or binding a partial prefix", async () => {
    const resolvers = new Map<number, (value: ScalarVolumePayload) => void>();
    const signals: AbortSignal[] = [];
    const getUploadScalarVolume = vi.fn(
      async (_fileId: string, config?: { channel?: number | null; signal?: AbortSignal }) => {
        if (config?.signal) signals.push(config.signal);
        return new Promise<ScalarVolumePayload>((resolve) => {
          resolvers.set(config?.channel ?? -1, resolve);
        });
      }
    );
    const info = microscopyViewerInfo({ fileId: "abort-later", channels: 3 });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const { unmount } = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="abort-later"
        viewerInfo={info}
        displayState={displayState([0, 1, 2])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(1));
    await act(async () => resolvers.get(0)?.(channelPayload({ channel: 0 })));
    await waitFor(() => expect(getUploadScalarVolume).toHaveBeenCalledTimes(2));

    unmount();
    expect(signals[1]?.aborted).toBe(true);
    await act(async () => resolvers.get(1)?.(channelPayload({ channel: 1 })));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(2);
    expect(textureInitialization).not.toHaveBeenCalled();
  });

  it("rejects a later channel grid before converting it or binding any texture", async () => {
    const getUploadScalarVolume = vi.fn(
      async (_fileId: string, config?: { channel?: number | null }) =>
        config?.channel === 1
          ? channelPayload({ channel: 1, sourceWidth: 2, width: 1 })
          : channelPayload({ channel: 0 })
    );
    const info = microscopyViewerInfo({ fileId: "later-grid-mismatch", channels: 2 });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;

    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="later-grid-mismatch"
        viewerInfo={info}
        displayState={displayState([0, 1])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );

    await waitFor(() => expect(screen.getByText(/inconsistent delivered grids/i)).toBeTruthy());
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(2);
    expect(scalarPreparation).toHaveBeenCalledTimes(1);
    expect(textureInitialization).not.toHaveBeenCalled();
  });

  it("reuses an identity-bound prepared cache hit without another fetch", async () => {
    const getUploadScalarVolume = vi.fn(async () => channelPayload({ channel: 0 }));
    const info = microscopyViewerInfo({ fileId: "cache-hit", channels: 1 });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const first = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="cache-hit"
        viewerInfo={info}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    first.unmount();

    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="cache-hit"
        viewerInfo={info}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(2));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
  });

  it("checks the scalar prepared cache before issuing a remount request", async () => {
    const getUploadScalarVolume = vi.fn(async () => payload);
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const first = render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="scalar-cache-before-network"
        viewerInfo={{
          ...viewerInfo,
          file_id: "scalar-cache-before-network",
          metadata: {
            ...viewerInfo.metadata,
            sha256: "scalar-cache-sha",
          },
        }}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    first.unmount();

    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="scalar-cache-before-network"
        viewerInfo={{
          ...viewerInfo,
          file_id: "scalar-cache-before-network",
          metadata: {
            ...viewerInfo.metadata,
            sha256: "scalar-cache-sha",
          },
        }}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );

    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(2));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
  });

  it("does not reuse prepared data across a same-file SHA change or a new API client", async () => {
    const getUploadScalarVolumeA = vi.fn(async () => channelPayload({ channel: 0 }));
    const apiClientA = {
      getUploadScalarVolume: getUploadScalarVolumeA,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    const infoA = microscopyViewerInfo({ fileId: "identity-refresh", channels: 1 });
    const first = render(
      <SliceStackVolumeCanvas
        apiClient={apiClientA}
        fileId="identity-refresh"
        viewerInfo={infoA}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(getUploadScalarVolumeA).toHaveBeenCalledTimes(1));
    first.unmount();

    const infoB = {
      ...infoA,
      metadata: { ...infoA.metadata, sha256: "sha-identity-refresh-v2" },
    } as UploadViewerInfo;
    const second = render(
      <SliceStackVolumeCanvas
        apiClient={apiClientA}
        fileId="identity-refresh"
        viewerInfo={infoB}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(getUploadScalarVolumeA).toHaveBeenCalledTimes(2));
    second.unmount();

    const getUploadScalarVolumeB = vi.fn(async () => channelPayload({ channel: 0 }));
    const apiClientB = {
      getUploadScalarVolume: getUploadScalarVolumeB,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClientB}
        fileId="identity-refresh"
        viewerInfo={infoB}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(getUploadScalarVolumeB).toHaveBeenCalledTimes(1));
  });

  it("rejects a scalar identity mismatch before texture publication", async () => {
    const getUploadScalarVolume = vi.fn(async () => channelPayload({ channel: 1 }));
    const info = microscopyViewerInfo({ fileId: "identity-mismatch", channels: 2 });
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="identity-mismatch"
        viewerInfo={info}
        displayState={displayState([0])}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );

    await waitFor(() => expect(screen.getByText(/channel mismatch/i)).toBeTruthy());
    expect(textureInitialization).not.toHaveBeenCalled();
  });
});

describe("SliceStackVolumeCanvas delivered GPU admission and recovery", () => {
  const displayState = {
    enhancement: "hounsfield:40:80",
    negative: false,
    fusion_method: "a",
    channels: [0],
  } as NonNullable<UploadViewerInfo["display_defaults"]>;

  const hugeSourceInfo = {
    ...viewerInfo,
    file_id: "huge-source",
    axis_sizes: { T: 1, C: 1, Z: 80, Y: 624, X: 8192 },
    metadata: { ...viewerInfo.metadata, sha256: "sha-huge-source" },
  } as UploadViewerInfo;

  it("accepts native 8192-wide geometry when the delivered texture fits the GPU", async () => {
    gpuLimits.max3DTextureSize = 4;
    const getUploadScalarVolume = vi.fn(async () =>
      channelPayload({
        channel: 0,
        sourceWidth: 8192,
        sourceHeight: 624,
        sourceDepth: 80,
        width: 2,
        height: 2,
        depth: 2,
      })
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="huge-source"
        viewerInfo={hugeSourceInfo}
        displayState={displayState}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(1);
    expect(rendererConstruction).toHaveBeenCalledTimes(1);
    expect(scalarUniformSets[scalarUniformSets.length - 1]?.uSteps.value).toBe(32);
    expect(
      (scalarUniformSets[scalarUniformSets.length - 1]?.uVoxelStep.value as ThreeModule.Vector3).x
    ).toBeCloseTo(0.5, 8);
  });

  it("rejects an oversized delivered grid before conversion or texture upload", async () => {
    gpuLimits.max3DTextureSize = 4;
    const getUploadScalarVolume = vi.fn(async () =>
      channelPayload({
        channel: 0,
        sourceWidth: 8192,
        sourceHeight: 624,
        sourceDepth: 80,
        width: 5,
        height: 2,
        depth: 2,
      })
    );
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="huge-source"
        viewerInfo={hugeSourceInfo}
        displayState={displayState}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(screen.getByText(/delivered volume exceeds/i)).toBeTruthy());
    expect(textureInitialization).not.toHaveBeenCalled();
  });

  it("retries a failed source generation explicitly", async () => {
    const getUploadScalarVolume = vi
      .fn()
      .mockRejectedValueOnce(new Error("transient scalar failure"))
      .mockResolvedValueOnce(payload);
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={displayState}
        tIndex={0}
        resolvedScalarRendering={resolvedIntensity()}
      />
    );
    await waitFor(() => expect(screen.getByRole("button", { name: /retry volume/i })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: /retry volume/i }));
    await waitFor(() => expect(textureInitialization).toHaveBeenCalledTimes(1));
    expect(getUploadScalarVolume).toHaveBeenCalledTimes(2);
  });

  it("rejects invalid time indices before an API request", async () => {
    const getUploadScalarVolume = vi.fn(async () => payload);
    const apiClient = {
      getUploadScalarVolume,
      uploadAtlasUrl: vi.fn(() => "/atlas.png"),
      uploadSliceUrl: vi.fn(() => "/slice.png"),
    } as unknown as ApiClient;
    render(
      <SliceStackVolumeCanvas
        apiClient={apiClient}
        fileId="file-nifti"
        viewerInfo={viewerInfo}
        displayState={displayState}
        tIndex={1.5}
        resolvedScalarRendering={resolvedIntensity(0, 1.5)}
      />
    );
    expect(await screen.findByText(/time index/i)).toBeTruthy();
    expect(getUploadScalarVolume).not.toHaveBeenCalled();
  });
});
