import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

import { getPlaneDescriptor } from "./shared";

type ScalarVolumeSource = {
  kind: "scalar";
  loadScalarVolume: () => Promise<ScalarVolumePayload>;
  fallbackImageUrl: string;
  axisSizes: UploadViewerInfo["axis_sizes"];
  plane: NonNullable<UploadViewerInfo["viewer"]["default_plane"]>;
  physicalSpacing?: UploadViewerInfo["metadata"]["physical_spacing"] | null;
  renderPolicy?: UploadViewerInfo["viewer"]["render_policy"];
  texturePolicy?: UploadViewerInfo["viewer"]["texture_policy"];
};

type AtlasVolumeSource = {
  kind: "atlas";
  atlasUrl: string;
  fallbackImageUrl: string;
  atlasScheme: NonNullable<UploadViewerInfo["viewer"]["atlas_scheme"]>;
  axisSizes: UploadViewerInfo["axis_sizes"];
  plane: NonNullable<UploadViewerInfo["viewer"]["default_plane"]>;
  physicalSpacing?: UploadViewerInfo["metadata"]["physical_spacing"] | null;
  renderPolicy?: UploadViewerInfo["viewer"]["render_policy"];
  texturePolicy?: UploadViewerInfo["viewer"]["texture_policy"];
};

type SliceStackVolumeCanvasProps = {
  apiClient?: ApiClient;
  fileId?: string;
  viewerInfo?: UploadViewerInfo;
  zIndex?: number;
  tIndex?: number;
  className?: string;
  displayState?: UploadViewerInfo["display_defaults"] | null;
  volumeSource?: ScalarVolumeSource | AtlasVolumeSource;
};

const MAX_STEPS = 512;
const DEFAULT_VOLUME_CLEAR = 0x07090d;

const VERTEX_SHADER = `
  varying vec3 vPosition;

  void main() {
    vec4 position4 = vec4(position, 1.0);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * position4;
  }
`;

const ATLAS_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uData;
  uniform int uSteps;
  uniform float uDensity;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uCameraPositionLocal;

  varying vec3 vPosition;

  bool intersectBox(
    vec3 rayOrigin,
    vec3 rayDir,
    vec3 boxMin,
    vec3 boxMax,
    out float tNear,
    out float tFar
  ) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tsmaller = min(t0, t1);
    vec3 tbigger = max(t0, t1);
    tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
    return tFar > max(tNear, 0.0);
  }

  void main() {
    vec3 rayOrigin = uCameraPositionLocal;
    vec3 rayDir = normalize(vPosition - rayOrigin);
    vec3 boxMin = uClipMin - vec3(0.5);
    vec3 boxMax = uClipMax - vec3(0.5);
    float tNear = 0.0;
    float tFar = 0.0;
    if (!intersectBox(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
      discard;
    }

    int steps = min(uSteps, ${MAX_STEPS});
    if (steps < 1) {
      discard;
    }

    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    vec3 stepVector = (back - front) / float(steps);
    vec3 location = front + vec3(0.5);
    vec3 delta = stepVector;

    vec4 accum = vec4(0.0);
    for (int iter = 0; iter < ${MAX_STEPS}; iter++) {
      if (iter >= steps) {
        break;
      }
      vec4 sampleColor = texture(uData, clamp(location, vec3(0.0), vec3(1.0)));
      float alpha = max(max(sampleColor.r, sampleColor.g), sampleColor.b) * uDensity;
      sampleColor.a = alpha;
      accum.rgb += (1.0 - accum.a) * sampleColor.rgb * sampleColor.a;
      accum.a += (1.0 - accum.a) * sampleColor.a;
      if (accum.a >= 0.985) {
        break;
      }
      location += delta;
    }

    if (accum.a < 0.02) {
      discard;
    }
    gl_FragColor = accum;
  }
`;

const SCALAR_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uData;
  uniform int uSteps;
  uniform float uDensity;
  uniform float uWindowLow;
  uniform float uWindowHigh;
  uniform bool uInvert;
  uniform int uProjectionMode;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uCameraPositionLocal;

  varying vec3 vPosition;

  bool intersectBox(
    vec3 rayOrigin,
    vec3 rayDir,
    vec3 boxMin,
    vec3 boxMax,
    out float tNear,
    out float tFar
  ) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tsmaller = min(t0, t1);
    vec3 tbigger = max(t0, t1);
    tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
    return tFar > max(tNear, 0.0);
  }

  float sampleWindowed(vec3 location) {
    float value = texture(uData, clamp(location, vec3(0.0), vec3(1.0))).r;
    float normalized = clamp(
      (value - uWindowLow) / max(0.000001, uWindowHigh - uWindowLow),
      0.0,
      1.0
    );
    return uInvert ? (1.0 - normalized) : normalized;
  }

  void main() {
    vec3 rayOrigin = uCameraPositionLocal;
    vec3 rayDir = normalize(vPosition - rayOrigin);
    vec3 boxMin = uClipMin - vec3(0.5);
    vec3 boxMax = uClipMax - vec3(0.5);
    float tNear = 0.0;
    float tFar = 0.0;
    if (!intersectBox(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
      discard;
    }

    int steps = min(uSteps, ${MAX_STEPS});
    if (steps < 1) {
      discard;
    }

    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    vec3 stepVector = (back - front) / float(steps);
    vec3 location = front + vec3(0.5);
    vec3 delta = stepVector;

    vec4 accum = vec4(0.0);
    float maxValue = 0.0;
    for (int iter = 0; iter < ${MAX_STEPS}; iter++) {
      if (iter >= steps) {
        break;
      }
      float sampleValue = sampleWindowed(location);
      if (uProjectionMode == 1) {
        maxValue = max(maxValue, sampleValue);
        location += delta;
        continue;
      }
      float alpha = sampleValue * uDensity;
      vec3 sampleColor = vec3(sampleValue);
      accum.rgb += (1.0 - accum.a) * sampleColor * alpha;
      accum.a += (1.0 - accum.a) * alpha;
      if (accum.a >= 0.985) {
        break;
      }
      location += delta;
    }

    if (uProjectionMode == 1) {
      if (maxValue < 0.02) {
        discard;
      }
      gl_FragColor = vec4(vec3(maxValue), clamp(maxValue * 1.2, 0.0, 1.0));
      return;
    }

    if (accum.a < 0.02) {
      discard;
    }
    gl_FragColor = accum;
  }
`;

const atlasToVolumeTexture = async (
  atlasUrl: string,
  atlasScheme: NonNullable<UploadViewerInfo["viewer"]["atlas_scheme"]>,
  texturePolicy: "linear" | "nearest"
): Promise<THREE.Data3DTexture> => {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const element = new window.Image();
    element.decoding = "async";
    element.onload = () => resolve(element);
    element.onerror = () => reject(new Error("Atlas image failed to load"));
    element.src = atlasUrl;
  });

  const atlasWidth = Math.max(1, image.naturalWidth || atlasScheme.atlas_width);
  const atlasHeight = Math.max(1, image.naturalHeight || atlasScheme.atlas_height);
  const sliceWidth = Math.max(1, atlasScheme.slice_width);
  const sliceHeight = Math.max(1, atlasScheme.slice_height);
  const sliceCount = Math.max(1, atlasScheme.slice_count);
  const columns = Math.max(1, atlasScheme.columns);

  const canvas = document.createElement("canvas");
  canvas.width = atlasWidth;
  canvas.height = atlasHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("2D canvas unavailable for atlas decoding");
  }
  context.drawImage(image, 0, 0, atlasWidth, atlasHeight);
  const atlasData = context.getImageData(0, 0, atlasWidth, atlasHeight).data;
  const volumeData = new Uint8Array(sliceWidth * sliceHeight * sliceCount * 4);

  for (let sliceIndex = 0; sliceIndex < sliceCount; sliceIndex += 1) {
    const column = sliceIndex % columns;
    const row = Math.floor(sliceIndex / columns);
    const srcX = column * sliceWidth;
    const srcY = row * sliceHeight;
    for (let y = 0; y < sliceHeight; y += 1) {
      const srcStart = ((srcY + y) * atlasWidth + srcX) * 4;
      const srcEnd = srcStart + sliceWidth * 4;
      const dstStart = ((sliceIndex * sliceHeight + y) * sliceWidth) * 4;
      volumeData.set(atlasData.subarray(srcStart, srcEnd), dstStart);
    }
  }

  const texture = new THREE.Data3DTexture(volumeData, sliceWidth, sliceHeight, sliceCount);
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.magFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
};

const scalarToVolumeTexture = async (
  payload: ScalarVolumePayload,
  texturePolicy: "linear" | "nearest"
): Promise<THREE.Data3DTexture> => {
  const width = Math.max(1, payload.width);
  const height = Math.max(1, payload.height);
  const depth = Math.max(1, payload.depth);
  let textureData: Uint8Array;
  if (payload.bytesPerVoxel >= 2) {
    const source = new Uint16Array(payload.data);
    textureData = new Uint8Array(source.length);
    for (let index = 0; index < source.length; index += 1) {
      textureData[index] = source[index] >>> 8;
    }
  } else {
    textureData = new Uint8Array(payload.data);
  }
  const texture = new THREE.Data3DTexture(textureData, width, height, depth);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.minFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.magFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
};

const parseWindowEnhancement = (
  enhancement: string | undefined,
  rawMin: number,
  rawMax: number
): { low: number; high: number } => {
  const safeEnhancement = String(enhancement || "").trim();
  if (safeEnhancement.startsWith("hounsfield:")) {
    const parts = safeEnhancement.split(":");
    const center = Number(parts[1]);
    const width = Number(parts[2]);
    if (Number.isFinite(center) && Number.isFinite(width) && width > 0) {
      return {
        low: center - width / 2,
        high: center + width / 2,
      };
    }
  }
  return { low: rawMin, high: rawMax > rawMin ? rawMax : rawMin + 1 };
};

const normalizeWindowRange = (
  enhancement: string | undefined,
  rawMin: number,
  rawMax: number
): { low: number; high: number } => {
  const { low, high } = parseWindowEnhancement(enhancement, rawMin, rawMax);
  const range = Math.max(1e-6, rawMax - rawMin);
  const lowNorm = Math.max(0, Math.min(1, (low - rawMin) / range));
  const highNorm = Math.max(lowNorm + 1e-4, Math.min(1, (high - rawMin) / range));
  return { low: lowNorm, high: highNorm };
};

export function SliceStackVolumeCanvas({
  apiClient,
  fileId,
  viewerInfo,
  zIndex,
  tIndex,
  className,
  displayState,
  volumeSource,
}: SliceStackVolumeCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const requestRenderRef = useRef<(() => void) | null>(null);
  const scalarUniformsRef = useRef<{
    uWindowLow: { value: number };
    uWindowHigh: { value: number };
    uInvert: { value: boolean };
  } | null>(null);
  const clipUniformsRef = useRef<{
    uClipMin: { value: THREE.Vector3 };
    uClipMax: { value: THREE.Vector3 };
  } | null>(null);
  const scalarRangeRef = useRef<{ rawMin: number; rawMax: number } | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  const plane = useMemo(
    () =>
      volumeSource?.plane ??
      getPlaneDescriptor(
        viewerInfo as UploadViewerInfo,
        "z"
      ),
    [viewerInfo, volumeSource]
  );
  const axisSizes = volumeSource?.axisSizes ?? viewerInfo?.axis_sizes ?? { T: 1, C: 1, Z: 1, Y: 1, X: 1 };
  const spacing = volumeSource?.physicalSpacing ?? viewerInfo?.metadata.physical_spacing ?? null;
  const volumeDepth = Math.max(1, axisSizes.Z);
  const zSpacing = Math.max(1e-6, Number(spacing?.z ?? 1));
  const ySpacing = Math.max(1e-6, Number(spacing?.y ?? 1));
  const xSpacing = Math.max(1e-6, Number(spacing?.x ?? 1));
  const worldWidth = Math.max(1, Number(plane.pixel_size.width) * xSpacing);
  const worldHeight = Math.max(1, Number(plane.pixel_size.height) * ySpacing);
  const worldDepth = Math.max(1, volumeDepth * zSpacing);
  const normalizedScale = useMemo(() => {
    const normalizer = Math.max(worldWidth, worldHeight, worldDepth, 1);
    return {
      x: worldWidth / normalizer,
      y: worldHeight / normalizer,
      z: worldDepth / normalizer,
    };
  }, [worldDepth, worldHeight, worldWidth]);
  const volumeRadius = useMemo(
    () =>
      Math.max(
        0.25,
        Math.sqrt(
          normalizedScale.x * normalizedScale.x +
            normalizedScale.y * normalizedScale.y +
            normalizedScale.z * normalizedScale.z
        ) / 2
      ),
    [normalizedScale.x, normalizedScale.y, normalizedScale.z]
  );

  const scalarChannel = useMemo(() => {
    const explicitVolumeChannel = displayState?.volume_channel;
    if (typeof explicitVolumeChannel === "number" && Number.isFinite(explicitVolumeChannel)) {
      return Math.max(0, Math.floor(explicitVolumeChannel));
    }
    const selected = displayState?.channels ?? [];
    if (Array.isArray(selected) && selected.length === 1 && Number.isFinite(selected[0])) {
      return Math.max(0, Math.floor(selected[0] ?? 0));
    }
    if (viewerInfo) {
      if (viewerInfo.axis_sizes.C <= 1) {
        return 0;
      }
      return Math.max(0, Math.floor(viewerInfo.selected_indices.C ?? 0));
    }
    return null;
  }, [displayState?.channels, displayState?.volume_channel, viewerInfo]);

  const atlasUrl = useMemo(() => {
    if (!apiClient || !fileId) {
      return "";
    }
    return apiClient.uploadAtlasUrl(fileId, {
      enhancement: displayState?.enhancement,
      fusionMethod: displayState?.fusion_method,
      negative: displayState?.negative,
      channels: displayState?.channels,
      channelColors: displayState?.channel_colors,
      t: tIndex,
    });
  }, [
    apiClient,
    displayState?.channel_colors,
    displayState?.channels,
    displayState?.enhancement,
    displayState?.fusion_method,
    displayState?.negative,
    fileId,
    tIndex,
  ]);

  const resolvedSource = useMemo(() => {
    if (volumeSource) {
      return volumeSource;
    }
    if (
      viewerInfo?.viewer.volume_mode === "scalar" &&
      apiClient &&
      fileId
    ) {
      return {
        kind: "scalar" as const,
        loadScalarVolume: () =>
          apiClient.getUploadScalarVolume(fileId, {
            t: tIndex,
            channel: scalarChannel,
          }),
        fallbackImageUrl: "",
        axisSizes,
        plane,
        physicalSpacing: spacing,
      };
    }
    if (!apiClient || !fileId || !viewerInfo?.viewer.atlas_scheme) {
      return null;
    }
    return {
      kind: "atlas" as const,
      atlasUrl,
      fallbackImageUrl: "",
      atlasScheme: viewerInfo.viewer.atlas_scheme,
      axisSizes,
      plane,
      physicalSpacing: spacing,
    };
  }, [
    apiClient,
    atlasUrl,
    axisSizes,
    fileId,
    plane,
    scalarChannel,
    spacing,
    tIndex,
    viewerInfo,
    volumeSource,
  ]);

  const fallbackImageUrl = useMemo(() => {
    if (volumeSource?.fallbackImageUrl) {
      return volumeSource.fallbackImageUrl;
    }
    if (!apiClient || !fileId) {
      return "";
    }
    return apiClient.uploadSliceUrl(fileId, {
      axis: "z",
      z: zIndex,
      t: tIndex,
      enhancement: displayState?.enhancement,
      fusionMethod: displayState?.fusion_method,
      negative: displayState?.negative,
      channels: displayState?.channels,
      channelColors: displayState?.channel_colors,
    });
  }, [
    apiClient,
    displayState?.channel_colors,
    displayState?.channels,
    displayState?.enhancement,
    displayState?.fusion_method,
    displayState?.negative,
    fileId,
    tIndex,
    volumeSource?.fallbackImageUrl,
    zIndex,
  ]);

  const renderPolicy = resolvedSource?.renderPolicy ?? viewerInfo?.viewer.render_policy ?? "scalar";
  const modality = String(viewerInfo?.modality ?? "").trim().toLowerCase();
  const clipBounds = useMemo(() => {
    const rawMin = displayState?.volume_clip_min ?? { x: 0, y: 0, z: 0 };
    const rawMax = displayState?.volume_clip_max ?? { x: 1, y: 1, z: 1 };
    const clamp = (value: number, fallback: number) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return fallback;
      }
      return Math.max(0, Math.min(1, numeric));
    };
    const nextMin = {
      x: clamp(rawMin.x, 0),
      y: clamp(rawMin.y, 0),
      z: clamp(rawMin.z, 0),
    };
    const nextMax = {
      x: clamp(rawMax.x, 1),
      y: clamp(rawMax.y, 1),
      z: clamp(rawMax.z, 1),
    };
    (["x", "y", "z"] as const).forEach((axis) => {
      if (nextMax[axis] - nextMin[axis] < 0.02) {
        if (nextMin[axis] <= 0.98) {
          nextMax[axis] = Math.min(1, nextMin[axis] + 0.02);
        } else {
          nextMin[axis] = Math.max(0, nextMax[axis] - 0.02);
        }
      }
    });
    return { min: nextMin, max: nextMax };
  }, [displayState?.volume_clip_max, displayState?.volume_clip_min]);
  const projectionMode = renderPolicy === "scalar" && modality === "microscopy" ? "mip" : "composite";
  const clearColor =
    renderPolicy === "scalar" || renderPolicy === "categorical" || renderPolicy === "display"
      ? DEFAULT_VOLUME_CLEAR
      : 0xf5f2eb;
  const density =
    renderPolicy === "scalar"
      ? projectionMode === "mip"
        ? 0.9
        : modality === "medical"
          ? 0.24
          : 0.34
      : 0.22;
  const texturePolicy: "linear" | "nearest" =
    resolvedSource?.texturePolicy === "nearest" || resolvedSource?.texturePolicy === "linear"
      ? resolvedSource.texturePolicy
      : viewerInfo?.viewer.texture_policy === "nearest" || viewerInfo?.viewer.texture_policy === "linear"
        ? viewerInfo.viewer.texture_policy
        : renderPolicy === "categorical" || renderPolicy === "analysis"
          ? "nearest"
          : "linear";

  useEffect(() => {
    const scalarRange = scalarRangeRef.current;
    const scalarUniforms = scalarUniformsRef.current;
    if (!scalarRange || !scalarUniforms) {
      return;
    }
    const normalizedWindow = normalizeWindowRange(
      displayState?.enhancement,
      scalarRange.rawMin,
      scalarRange.rawMax
    );
    scalarUniforms.uWindowLow.value = normalizedWindow.low;
    scalarUniforms.uWindowHigh.value = normalizedWindow.high;
    scalarUniforms.uInvert.value = Boolean(displayState?.negative);
    requestRenderRef.current?.();
  }, [displayState?.enhancement, displayState?.negative]);

  useEffect(() => {
    const clipUniforms = clipUniformsRef.current;
    if (!clipUniforms) {
      return;
    }
    clipUniforms.uClipMin.value.set(clipBounds.min.x, clipBounds.min.y, clipBounds.min.z);
    clipUniforms.uClipMax.value.set(clipBounds.max.x, clipBounds.max.y, clipBounds.max.z);
    requestRenderRef.current?.();
  }, [clipBounds]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !resolvedSource) {
      return;
    }

    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      if (renderError) {
        setRenderError(null);
      }
    } catch (error) {
      setRenderError(error instanceof Error ? error.message : "WebGL unavailable");
      return;
    }
    if (!renderer.capabilities.isWebGL2) {
      renderer.dispose();
      setRenderError("WebGL2 unavailable");
      return;
    }

    const gl = renderer.getContext();
    const max3DTextureSize = Number(
      typeof WebGL2RenderingContext !== "undefined" && typeof gl?.getParameter === "function"
        ? (gl as WebGL2RenderingContext).getParameter(WebGL2RenderingContext.MAX_3D_TEXTURE_SIZE)
        : 0
    );
    const largestDimension = Math.max(
      Number(resolvedSource.axisSizes.X ?? 1),
      Number(resolvedSource.axisSizes.Y ?? 1),
      Number(resolvedSource.axisSizes.Z ?? 1)
    );
    if (Number.isFinite(max3DTextureSize) && max3DTextureSize > 0 && largestDimension > max3DTextureSize) {
      renderer.dispose();
      setRenderError(
        `Volume exceeds this browser's 3D texture limit (${largestDimension} > ${max3DTextureSize}).`
      );
      return;
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(clearColor, 1);
    renderer.domElement.className = "viewer-webgl-canvas";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    const controls = new TrackballControls(camera, renderer.domElement);
    controls.noPan = false;
    controls.noZoom = false;
    controls.noRotate = false;
    controls.staticMoving = true;
    controls.rotateSpeed = 5;
    controls.zoomSpeed = 1.5;
    controls.panSpeed = 0.9;
    controls.minDistance = volumeRadius * 1.05;
    controls.maxDistance = Math.max(volumeRadius * 10, 6);

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material =
      resolvedSource.kind === "scalar"
        ? new THREE.ShaderMaterial({
            uniforms: {
              uData: { value: null },
              uSteps: { value: Math.max(128, Math.min(MAX_STEPS, volumeDepth * 2)) },
              uDensity: { value: density },
              uWindowLow: { value: 0.0 },
              uWindowHigh: { value: 1.0 },
              uInvert: { value: Boolean(displayState?.negative) },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(clipBounds.min.x, clipBounds.min.y, clipBounds.min.z) },
              uClipMax: { value: new THREE.Vector3(clipBounds.max.x, clipBounds.max.y, clipBounds.max.z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
            },
            vertexShader: VERTEX_SHADER,
            fragmentShader: SCALAR_FRAGMENT_SHADER,
            side: THREE.DoubleSide,
            transparent: true,
            depthWrite: false,
          })
        : new THREE.ShaderMaterial({
            uniforms: {
              uData: { value: null },
              uSteps: { value: Math.max(96, Math.min(MAX_STEPS, volumeDepth * 2)) },
              uDensity: { value: density },
              uClipMin: { value: new THREE.Vector3(clipBounds.min.x, clipBounds.min.y, clipBounds.min.z) },
              uClipMax: { value: new THREE.Vector3(clipBounds.max.x, clipBounds.max.y, clipBounds.max.z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
            },
            vertexShader: VERTEX_SHADER,
            fragmentShader: ATLAS_FRAGMENT_SHADER,
            side: THREE.DoubleSide,
            transparent: true,
            depthWrite: false,
          });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set(normalizedScale.x, normalizedScale.y, normalizedScale.z);
    scene.add(mesh);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));

    const cameraPositionUniform = (material.uniforms as Record<string, { value: THREE.Vector3 }>).uCameraPositionLocal;

    const render = () => {
      const cameraLocal = mesh.worldToLocal(camera.position.clone());
      cameraPositionUniform.value.copy(cameraLocal);
      renderer.render(scene, camera);
    };
    requestRenderRef.current = render;
    clipUniformsRef.current = {
      uClipMin: (material.uniforms as Record<string, { value: THREE.Vector3 }>).uClipMin,
      uClipMax: (material.uniforms as Record<string, { value: THREE.Vector3 }>).uClipMax,
    };

    const resize = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      camera.lookAt(0, 0, 0);
      controls.handleResize();
      render();
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);

    let texture3D: THREE.Data3DTexture | null = null;
    let animationFrame = 0;
    camera.position.set(1.8, 1.4, 1.8);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    const animate = () => {
      if (disposed) {
        return;
      }
      controls.update();
      render();
      animationFrame = window.requestAnimationFrame(animate);
    };

    const loadPromise =
      resolvedSource.kind === "scalar"
        ? resolvedSource.loadScalarVolume().then(async (payload) => {
            const texture = await scalarToVolumeTexture(payload, texturePolicy);
            scalarRangeRef.current = { rawMin: payload.rawMin, rawMax: payload.rawMax };
            const normalizedWindow = normalizeWindowRange(
              displayState?.enhancement,
              payload.rawMin,
              payload.rawMax
            );
            scalarUniformsRef.current = {
              uWindowLow: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowLow as { value: number },
              uWindowHigh: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowHigh as { value: number },
              uInvert: (material.uniforms as Record<string, { value: number | boolean | null }>).uInvert as { value: boolean },
            };
            scalarUniformsRef.current.uWindowLow.value = normalizedWindow.low;
            scalarUniformsRef.current.uWindowHigh.value = normalizedWindow.high;
            scalarUniformsRef.current.uInvert.value = Boolean(displayState?.negative);
            return texture;
          })
        : atlasToVolumeTexture(resolvedSource.atlasUrl, resolvedSource.atlasScheme, texturePolicy);

    void loadPromise
      .then((decodedTexture) => {
        if (disposed) {
          decodedTexture.dispose();
          return;
        }
        texture3D = decodedTexture;
        if (typeof renderer.initTexture === "function") {
          renderer.initTexture(decodedTexture);
        }
        material.uniforms.uData.value = decodedTexture;
        material.needsUpdate = true;
        resize();
      })
      .catch((error: unknown) => {
        if (disposed) {
          return;
        }
        setRenderError(error instanceof Error ? error.message : "Volume data failed to load");
      });

    resize();
    animate();

    return () => {
      disposed = true;
      requestRenderRef.current = null;
      scalarUniformsRef.current = null;
      clipUniformsRef.current = null;
      scalarRangeRef.current = null;
      observer.disconnect();
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      controls.dispose();
      geometry.dispose();
      material.dispose();
      texture3D?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    renderError,
    resolvedSource,
    texturePolicy,
    clearColor,
    density,
    projectionMode,
    clipBounds.max.x,
    clipBounds.max.y,
    clipBounds.max.z,
    clipBounds.min.x,
    clipBounds.min.y,
    clipBounds.min.z,
    normalizedScale.x,
    normalizedScale.y,
    normalizedScale.z,
    volumeDepth,
    volumeRadius,
    worldDepth,
    worldHeight,
    worldWidth,
  ]);

  const backendLabel = resolvedSource?.kind ?? "atlas";

  if (
    renderError ||
    !resolvedSource ||
    (resolvedSource.kind === "atlas" && !resolvedSource.atlasScheme)
  ) {
    return (
      <div
        className={className ?? "viewer-canvas-root"}
        data-viewer-surface="volume"
        data-viewer-backend={backendLabel}
        data-viewer-aspect={plane.aspect_ratio.toFixed(4)}
        data-viewer-renderer="fallback"
        data-viewer-render-policy={renderPolicy}
        data-viewer-texture-policy={texturePolicy}
        data-viewer-projection-mode={projectionMode}
        data-viewer-clip-x-min={clipBounds.min.x.toFixed(2)}
        data-viewer-clip-x-max={clipBounds.max.x.toFixed(2)}
        data-viewer-clip-y-min={clipBounds.min.y.toFixed(2)}
        data-viewer-clip-y-max={clipBounds.max.y.toFixed(2)}
        data-viewer-clip-z-min={clipBounds.min.z.toFixed(2)}
        data-viewer-clip-z-max={clipBounds.max.z.toFixed(2)}
        data-viewer-volume-channel={scalarChannel == null ? undefined : String(scalarChannel)}
      >
        <div className="viewer-image-fallback" style={{ aspectRatio: `${Math.max(1e-6, plane.aspect_ratio)}` }}>
          <img src={fallbackImageUrl} alt="Volume fallback preview" className="viewer-image-fallback-media" />
        </div>
        <p className="viewer-fallback-note">
          {renderError
            ? `${backendLabel === "scalar" ? "Scalar" : "Atlas"} volume viewer unavailable: ${renderError}. Showing a representative slice preview instead.`
            : "Volume viewer unavailable. Showing a representative slice preview instead."}
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className ?? "viewer-canvas-root"}
      data-viewer-surface="volume"
      data-viewer-backend={resolvedSource.kind}
      data-viewer-aspect={plane.aspect_ratio.toFixed(4)}
      data-viewer-render-policy={renderPolicy}
      data-viewer-texture-policy={texturePolicy}
      data-viewer-projection-mode={projectionMode}
      data-viewer-clip-x-min={clipBounds.min.x.toFixed(2)}
      data-viewer-clip-x-max={clipBounds.max.x.toFixed(2)}
      data-viewer-clip-y-min={clipBounds.min.y.toFixed(2)}
      data-viewer-clip-y-max={clipBounds.max.y.toFixed(2)}
      data-viewer-clip-z-min={clipBounds.min.z.toFixed(2)}
      data-viewer-clip-z-max={clipBounds.max.z.toFixed(2)}
      data-viewer-volume-channel={scalarChannel == null ? undefined : String(scalarChannel)}
    />
  );
}
