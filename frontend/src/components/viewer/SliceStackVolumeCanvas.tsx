import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

import { scalarVolumePayloadToTextureBytes } from "./scalarVolume";
import { getPlaneDescriptor } from "./shared";
import { computePhysicalVolumeGeometry } from "./volumeGeometry";
import {
  resolveScalarVolumeColorMap,
} from "./volumeColorMap";
import { resolveScalarVolumeLighting } from "./volumeLighting";
import { resolveVolumeAxisCue } from "./volumeAxisCue";
import { resolveVolumeClipCue } from "./volumeClipCue";
import { resolveVolumeScaleBar } from "./volumeScaleBar";
import {
  resolveVolumeSliceCursorCue,
  type VolumeSliceCursorCue,
} from "./volumeSliceCursor";
import { resolveScalarVolumeTransferFunction } from "./volumeTransferFunction";
import {
  resolveVolumeCameraMode,
  type VolumeCameraMode,
} from "./volumeCameraMode";
import {
  resolveVolumeViewPreset,
  type VolumeViewPreset,
} from "./volumeViewPreset";
import { resolveVolumeOrientationCue } from "./volumeOrientation";

export { scalarVolumePayloadToTextureBytes, scalarVolumePayloadValueAt } from "./scalarVolume";
export { computePhysicalVolumeGeometry, type PhysicalVolumeGeometry } from "./volumeGeometry";
export { resolveVolumeScaleBar, type VolumeScaleBar } from "./volumeScaleBar";
export {
  resolveVolumeSliceCursorCue,
  type VolumeSliceCursorAxis,
  type VolumeSliceCursorCue,
} from "./volumeSliceCursor";
export {
  SCALAR_VOLUME_COLOR_MAPS,
  resolveScalarVolumeColorMap,
  type ScalarVolumeColorMap,
  type ScalarVolumeColorMapId,
} from "./volumeColorMap";
export { resolveVolumeAxisCue, type VolumeAxisCue, type VolumeAxisCueAxis } from "./volumeAxisCue";
export { resolveVolumeClipCue, type VolumeClipCue, type VolumeClipCueAxis } from "./volumeClipCue";
export { resolveScalarVolumeLighting, type ScalarVolumeLighting } from "./volumeLighting";
export { resolveScalarVolumeTransferFunction, type ScalarVolumeTransferFunction } from "./volumeTransferFunction";
export { resolveVolumeOrientationCue, type VolumeOrientationCue } from "./volumeOrientation";
export {
  VOLUME_VIEW_PRESETS,
  resolveVolumeViewPreset,
  type VolumeViewPreset,
  type VolumeViewPresetId,
} from "./volumeViewPreset";
export {
  VOLUME_CAMERA_MODES,
  resolveVolumeCameraMode,
  type VolumeCameraMode,
  type VolumeCameraModeId,
} from "./volumeCameraMode";

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
  xIndex?: number;
  yIndex?: number;
  zIndex?: number;
  tIndex?: number;
  className?: string;
  displayState?: UploadViewerInfo["display_defaults"] | null;
  volumeSource?: ScalarVolumeSource | AtlasVolumeSource;
};

export const MAX_STEPS = 512;
const MIN_INTERACTIVE_STEPS = 32;
const SAMPLE_RAMP_FACTOR = 1.5;
const DEFAULT_VOLUME_CLEAR = 0x07090d;
const DEFAULT_VOLUME_AXIS_SIZES: UploadViewerInfo["axis_sizes"] = { T: 1, C: 1, Z: 1, Y: 1, X: 1 };
const ORTHOGRAPHIC_FRUSTUM_SCALE = 2.8;
const MIN_ORTHOGRAPHIC_FRUSTUM_HEIGHT = 1.8;

type VolumeCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

export type VolumeSampleBudget = {
  interactiveSteps: number;
  settledSteps: number;
  rampFactor: number;
};

export function computeVolumeSampleBudget({
  sourceKind,
  volumeDepth,
  projectionMode,
}: {
  sourceKind: ScalarVolumeSource["kind"] | AtlasVolumeSource["kind"];
  volumeDepth: number;
  projectionMode: "mip" | "composite";
}): VolumeSampleBudget {
  const safeDepth = Math.max(1, Math.floor(Number(volumeDepth) || 1));
  const floor = sourceKind === "scalar" ? 128 : 96;
  const projectionMultiplier = projectionMode === "mip" ? 2 : 2;
  const settledSteps = Math.max(floor, Math.min(MAX_STEPS, safeDepth * projectionMultiplier));
  const interactiveSteps = Math.max(
    MIN_INTERACTIVE_STEPS,
    Math.min(settledSteps, Math.round(settledSteps / 8))
  );
  return {
    interactiveSteps,
    settledSteps,
    rampFactor: SAMPLE_RAMP_FACTOR,
  };
}

export function advanceProgressiveVolumeSteps(
  currentSteps: number,
  budget: VolumeSampleBudget
): number {
  const current = Math.max(1, Math.floor(Number(currentSteps) || 1));
  if (current >= budget.settledSteps) {
    return budget.settledSteps;
  }
  return Math.min(
    budget.settledSteps,
    Math.max(current + 1, Math.round(current * budget.rampFactor))
  );
}

export function resolveVolumeProjectionMode({
  renderPolicy,
  modality,
  fusionMethod,
}: {
  renderPolicy?: string;
  modality?: string;
  fusionMethod?: string;
}): "mip" | "composite" {
  const normalizedPolicy = String(renderPolicy ?? "").trim().toLowerCase();
  const normalizedModality = String(modality ?? "").trim().toLowerCase();
  const normalizedFusion = String(fusionMethod ?? "").trim().toLowerCase();
  if (normalizedPolicy !== "scalar") {
    return "composite";
  }
  if (normalizedFusion === "m") {
    return "mip";
  }
  if (normalizedFusion === "a") {
    return "composite";
  }
  return normalizedModality === "microscopy" ? "mip" : "composite";
}

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
  uniform vec3 uCameraDirectionLocal;
  uniform bool uOrthographicCamera;

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
    vec3 rayDir = uOrthographicCamera
      ? normalize(uCameraDirectionLocal)
      : normalize(vPosition - uCameraPositionLocal);
    vec3 rayOrigin = uOrthographicCamera
      ? vPosition - rayDir * 2.0
      : uCameraPositionLocal;
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
  uniform int uColorMap;
  uniform float uSignalFloor;
  uniform float uDensityScale;
  uniform bool uLightingEnabled;
  uniform float uLightingStrength;
  uniform vec3 uVoxelStep;
  uniform int uProjectionMode;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uCameraPositionLocal;
  uniform vec3 uCameraDirectionLocal;
  uniform bool uOrthographicCamera;

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

  vec3 ramp5(float value, vec3 c0, vec3 c1, vec3 c2, vec3 c3, vec3 c4) {
    float v = clamp(value, 0.0, 1.0);
    if (v < 0.25) {
      return mix(c0, c1, v / 0.25);
    }
    if (v < 0.5) {
      return mix(c1, c2, (v - 0.25) / 0.25);
    }
    if (v < 0.75) {
      return mix(c2, c3, (v - 0.5) / 0.25);
    }
    return mix(c3, c4, (v - 0.75) / 0.25);
  }

  vec3 scalarColor(float value) {
    if (uColorMap == 1) {
      return ramp5(
        value,
        vec3(0.267, 0.004, 0.329),
        vec3(0.204, 0.286, 0.561),
        vec3(0.129, 0.568, 0.551),
        vec3(0.369, 0.789, 0.383),
        vec3(0.993, 0.906, 0.145)
      );
    }
    if (uColorMap == 2) {
      return ramp5(
        value,
        vec3(0.000, 0.000, 0.016),
        vec3(0.322, 0.071, 0.486),
        vec3(0.714, 0.216, 0.475),
        vec3(0.984, 0.537, 0.384),
        vec3(0.988, 0.992, 0.749)
      );
    }
    if (uColorMap == 3) {
      return ramp5(
        value,
        vec3(0.000, 0.000, 0.016),
        vec3(0.341, 0.063, 0.431),
        vec3(0.737, 0.216, 0.329),
        vec3(0.976, 0.557, 0.035),
        vec3(0.988, 1.000, 0.643)
      );
    }
    return vec3(value);
  }

  float sampleOpacity(float value) {
    return smoothstep(uSignalFloor, 1.0, clamp(value, 0.0, 1.0));
  }

  vec3 applyDepthLighting(vec3 location, vec3 color, float value) {
    if (!uLightingEnabled || uLightingStrength <= 0.0 || value <= uSignalFloor) {
      return color;
    }

    float gx = sampleWindowed(location + vec3(uVoxelStep.x, 0.0, 0.0)) -
      sampleWindowed(location - vec3(uVoxelStep.x, 0.0, 0.0));
    float gy = sampleWindowed(location + vec3(0.0, uVoxelStep.y, 0.0)) -
      sampleWindowed(location - vec3(0.0, uVoxelStep.y, 0.0));
    float gz = sampleWindowed(location + vec3(0.0, 0.0, uVoxelStep.z)) -
      sampleWindowed(location - vec3(0.0, 0.0, uVoxelStep.z));
    vec3 gradient = vec3(gx, gy, gz);
    if (length(gradient) < 0.0001) {
      return color;
    }

    vec3 normal = normalize(gradient);
    vec3 lightDir = normalize(vec3(-0.45, 0.55, 0.72));
    vec3 viewDir = uOrthographicCamera
      ? -normalize(uCameraDirectionLocal)
      : normalize(uCameraPositionLocal - (location - vec3(0.5)));
    float diffuse = max(dot(normal, lightDir), dot(-normal, lightDir));
    float rim = pow(1.0 - clamp(abs(dot(normal, viewDir)), 0.0, 1.0), 2.0);
    float shade = clamp(0.42 + 0.72 * diffuse + 0.18 * rim, 0.35, 1.25);
    return mix(color, color * shade, uLightingStrength);
  }

  void main() {
    vec3 rayDir = uOrthographicCamera
      ? normalize(uCameraDirectionLocal)
      : normalize(vPosition - uCameraPositionLocal);
    vec3 rayOrigin = uOrthographicCamera
      ? vPosition - rayDir * 2.0
      : uCameraPositionLocal;
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
    vec3 maxLocation = location;
    for (int iter = 0; iter < ${MAX_STEPS}; iter++) {
      if (iter >= steps) {
        break;
      }
      float sampleValue = sampleWindowed(location);
      float opacityValue = sampleOpacity(sampleValue);
      if (uProjectionMode == 1) {
        if (opacityValue > 0.0 && sampleValue > maxValue) {
          maxValue = sampleValue;
          maxLocation = location;
        }
        location += delta;
        continue;
      }
      float alpha = opacityValue * uDensity * uDensityScale;
      vec3 sampleColor = applyDepthLighting(location, scalarColor(sampleValue), sampleValue);
      accum.rgb += (1.0 - accum.a) * sampleColor * alpha;
      accum.a += (1.0 - accum.a) * alpha;
      if (accum.a >= 0.985) {
        break;
      }
      location += delta;
    }

    if (uProjectionMode == 1) {
      float maxOpacity = sampleOpacity(maxValue);
      if (maxOpacity < 0.02) {
        discard;
      }
      vec3 maxColor = applyDepthLighting(maxLocation, scalarColor(maxValue), maxValue);
      gl_FragColor = vec4(maxColor, clamp(maxOpacity * uDensityScale * 1.2, 0.0, 1.0));
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
  const textureData = scalarVolumePayloadToTextureBytes(payload);
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

const computeOrthographicFrustumHeight = (volumeRadius: number): number =>
  Math.max(MIN_ORTHOGRAPHIC_FRUSTUM_HEIGHT, volumeRadius * ORTHOGRAPHIC_FRUSTUM_SCALE);

const createVolumeCamera = ({
  mode,
  volumeRadius,
}: {
  mode: VolumeCameraMode;
  volumeRadius: number;
}): VolumeCamera => {
  if (!mode.isOrthographic) {
    return new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  }
  const frustumHeight = computeOrthographicFrustumHeight(volumeRadius);
  return new THREE.OrthographicCamera(
    -frustumHeight / 2,
    frustumHeight / 2,
    frustumHeight / 2,
    -frustumHeight / 2,
    0.01,
    100
  );
};

const configureVolumeCameraProjection = ({
  camera,
  width,
  height,
  volumeRadius,
}: {
  camera: VolumeCamera;
  width: number;
  height: number;
  volumeRadius: number;
}) => {
  const aspect = width / height;
  if (camera instanceof THREE.OrthographicCamera) {
    const frustumHeight = computeOrthographicFrustumHeight(volumeRadius);
    const frustumWidth = frustumHeight * aspect;
    camera.left = -frustumWidth / 2;
    camera.right = frustumWidth / 2;
    camera.top = frustumHeight / 2;
    camera.bottom = -frustumHeight / 2;
  } else {
    camera.aspect = aspect;
  }
  camera.updateProjectionMatrix();
};

const applyVolumeCameraPreset = ({
  camera,
  controls,
  preset,
  volumeRadius,
}: {
  camera: VolumeCamera;
  controls: TrackballControls;
  preset: VolumeViewPreset;
  volumeRadius: number;
}) => {
  const direction = new THREE.Vector3(preset.direction.x, preset.direction.y, preset.direction.z).normalize();
  const distance = Math.max(volumeRadius * 3.4, 2.4);
  camera.position.copy(direction.multiplyScalar(distance));
  camera.up.set(preset.up.x, preset.up.y, preset.up.z);
  controls.target.set(0, 0, 0);
  camera.lookAt(0, 0, 0);
  controls.update();
};

type SliceCursorPlanes = {
  x: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  y: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  z: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
};

const applyVolumeSliceCursorPlanes = ({
  planes,
  cue,
  normalizedScale,
}: {
  planes: SliceCursorPlanes;
  cue: VolumeSliceCursorCue;
  normalizedScale: { x: number; y: number; z: number };
}) => {
  planes.x.visible = cue.visible && cue.x.count > 1;
  planes.y.visible = cue.visible && cue.y.count > 1;
  planes.z.visible = cue.visible && cue.z.count > 1;

  planes.x.scale.set(normalizedScale.z, normalizedScale.y, 1);
  planes.x.position.set(normalizedScale.x * cue.x.local, 0, 0);

  planes.y.scale.set(normalizedScale.x, normalizedScale.z, 1);
  planes.y.position.set(0, normalizedScale.y * cue.y.local, 0);

  planes.z.scale.set(normalizedScale.x, normalizedScale.y, 1);
  planes.z.position.set(0, 0, normalizedScale.z * cue.z.local);
};

const resolveSpatialUnit = (viewerInfo?: UploadViewerInfo | null): string => {
  const coordinates = viewerInfo?.phys?.coordinates;
  const units =
    coordinates && typeof coordinates === "object"
      ? (coordinates as Record<string, unknown>).space_units
      : null;
  const spatial =
    units && typeof units === "object"
      ? (units as Record<string, unknown>).spatial
      : null;
  if (typeof spatial === "string" && spatial.trim()) {
    return spatial.trim();
  }
  return viewerInfo?.metadata.physical_spacing ? "vox" : "units";
};

export function SliceStackVolumeCanvas({
  apiClient,
  fileId,
  viewerInfo,
  xIndex,
  yIndex,
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
    uColorMap: { value: number };
    uSignalFloor: { value: number };
    uDensityScale: { value: number };
    uLightingEnabled: { value: boolean };
    uLightingStrength: { value: number };
  } | null>(null);
  const clipUniformsRef = useRef<{
    uClipMin: { value: THREE.Vector3 };
    uClipMax: { value: THREE.Vector3 };
  } | null>(null);
  const scalarRenderConfigRef = useRef({
    enhancement: undefined as string | undefined,
    negative: false,
    colorMapShaderValue: 0,
    signalFloor: 0,
    densityScale: 1,
    lightingEnabled: false,
    lightingStrength: 0.65,
  });
  const sliceCursorCueRef = useRef<VolumeSliceCursorCue | null>(null);
  const cameraRigRef = useRef<{
    camera: VolumeCamera;
    controls: TrackballControls;
  } | null>(null);
  const sliceCursorPlanesRef = useRef<SliceCursorPlanes | null>(null);
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
  const axisSizes = useMemo(
    () => volumeSource?.axisSizes ?? viewerInfo?.axis_sizes ?? DEFAULT_VOLUME_AXIS_SIZES,
    [viewerInfo?.axis_sizes, volumeSource?.axisSizes]
  );
  const spacing = volumeSource?.physicalSpacing ?? viewerInfo?.metadata.physical_spacing ?? null;
  const volumeDepth = Math.max(1, axisSizes.Z);
  const physicalGeometry = useMemo(
    () =>
      computePhysicalVolumeGeometry({
        planePixelSize: plane.pixel_size,
        volumeDepth,
        physicalSpacing: spacing,
      }),
    [plane.pixel_size, spacing, volumeDepth]
  );
  const normalizedScale = physicalGeometry.normalizedScale;
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
    volumeSource,
    zIndex,
  ]);

  const renderPolicy = resolvedSource?.renderPolicy ?? viewerInfo?.viewer.render_policy ?? "scalar";
  const orientationCue = resolveVolumeOrientationCue(viewerInfo?.viewer.orientation);
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
  const projectionMode = resolveVolumeProjectionMode({
    renderPolicy,
    modality,
    fusionMethod: displayState?.fusion_method,
  });
  const scalarColorMap = useMemo(
    () => resolveScalarVolumeColorMap(displayState?.scalar_colormap),
    [displayState?.scalar_colormap]
  );
  const scalarTransfer = useMemo(
    () =>
      resolveScalarVolumeTransferFunction({
        volume_density: displayState?.volume_density,
        volume_signal_floor: displayState?.volume_signal_floor,
      }),
    [displayState?.volume_density, displayState?.volume_signal_floor]
  );
  const scalarLighting = useMemo(
    () =>
      resolveScalarVolumeLighting({
        volume_lighting: displayState?.volume_lighting,
        volume_lighting_strength: displayState?.volume_lighting_strength,
      }),
    [displayState?.volume_lighting, displayState?.volume_lighting_strength]
  );
  const volumeCameraMode = useMemo(
    () => resolveVolumeCameraMode(displayState?.volume_camera_mode),
    [displayState?.volume_camera_mode]
  );
  const volumeViewPreset = resolveVolumeViewPreset(displayState?.volume_view_preset);
  const spatialUnit = resolveSpatialUnit(viewerInfo);
  const sliceCursorCue = useMemo(
    () =>
      resolveVolumeSliceCursorCue({
        axisSizes,
        indices: {
          x: xIndex ?? Math.floor((Math.max(1, axisSizes.X) - 1) / 2),
          y: yIndex ?? Math.floor((Math.max(1, axisSizes.Y) - 1) / 2),
          z: zIndex ?? Math.floor((Math.max(1, axisSizes.Z) - 1) / 2),
        },
        worldWidth: physicalGeometry.worldWidth,
        worldHeight: physicalGeometry.worldHeight,
        worldDepth: physicalGeometry.worldDepth,
        unit: spatialUnit,
      }),
    [
      axisSizes,
      physicalGeometry.worldDepth,
      physicalGeometry.worldHeight,
      physicalGeometry.worldWidth,
      spatialUnit,
      xIndex,
      yIndex,
      zIndex,
    ]
  );
  const volumeScaleBar = resolveVolumeScaleBar({
    worldWidth: physicalGeometry.worldWidth,
    unit: spatialUnit,
  });
  const volumeAxisCue = resolveVolumeAxisCue({
    worldWidth: physicalGeometry.worldWidth,
    worldHeight: physicalGeometry.worldHeight,
    worldDepth: physicalGeometry.worldDepth,
    unit: spatialUnit,
  });
  const volumeClipCue = resolveVolumeClipCue({
    worldWidth: physicalGeometry.worldWidth,
    worldHeight: physicalGeometry.worldHeight,
    worldDepth: physicalGeometry.worldDepth,
    clipMin: clipBounds.min,
    clipMax: clipBounds.max,
    unit: spatialUnit,
  });
  const scalarVoxelStep = useMemo(
    () => ({
      x: 1 / Math.max(1, axisSizes.X - 1),
      y: 1 / Math.max(1, axisSizes.Y - 1),
      z: 1 / Math.max(1, axisSizes.Z - 1),
    }),
    [axisSizes.X, axisSizes.Y, axisSizes.Z]
  );
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
  const sampleBudget = useMemo(
    () =>
      computeVolumeSampleBudget({
        sourceKind: resolvedSource?.kind ?? "atlas",
        volumeDepth,
        projectionMode,
      }),
    [projectionMode, resolvedSource?.kind, volumeDepth]
  );
  const scalarRenderConfig = useMemo(
    () => ({
      enhancement: displayState?.enhancement,
      negative: Boolean(displayState?.negative),
      colorMapShaderValue: scalarColorMap.shaderValue,
      signalFloor: scalarTransfer.signalFloor,
      densityScale: scalarTransfer.densityScale,
      lightingEnabled: scalarLighting.enabled,
      lightingStrength: scalarLighting.strength,
    }),
    [
      displayState?.enhancement,
      displayState?.negative,
      scalarColorMap.shaderValue,
      scalarLighting.enabled,
      scalarLighting.strength,
      scalarTransfer.densityScale,
      scalarTransfer.signalFloor,
    ]
  );

  useEffect(() => {
    scalarRenderConfigRef.current = scalarRenderConfig;
  }, [scalarRenderConfig]);

  useEffect(() => {
    sliceCursorCueRef.current = sliceCursorCue;
  }, [sliceCursorCue]);

  useEffect(() => {
    const scalarRange = scalarRangeRef.current;
    const scalarUniforms = scalarUniformsRef.current;
    if (!scalarRange || !scalarUniforms) {
      return;
    }
    const normalizedWindow = normalizeWindowRange(
      scalarRenderConfig.enhancement,
      scalarRange.rawMin,
      scalarRange.rawMax
    );
    scalarUniforms.uWindowLow.value = normalizedWindow.low;
    scalarUniforms.uWindowHigh.value = normalizedWindow.high;
    scalarUniforms.uInvert.value = scalarRenderConfig.negative;
    scalarUniforms.uColorMap.value = scalarRenderConfig.colorMapShaderValue;
    scalarUniforms.uSignalFloor.value = scalarRenderConfig.signalFloor;
    scalarUniforms.uDensityScale.value = scalarRenderConfig.densityScale;
    scalarUniforms.uLightingEnabled.value = scalarRenderConfig.lightingEnabled;
    scalarUniforms.uLightingStrength.value = scalarRenderConfig.lightingStrength;
    requestRenderRef.current?.();
  }, [scalarRenderConfig]);

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
    const rig = cameraRigRef.current;
    if (!rig) {
      return;
    }
    applyVolumeCameraPreset({
      camera: rig.camera,
      controls: rig.controls,
      preset: volumeViewPreset,
      volumeRadius,
    });
    requestRenderRef.current?.();
  }, [volumeRadius, volumeViewPreset]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !resolvedSource) {
      return;
    }

    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    const commitRenderError = (message: string | null) => {
      window.setTimeout(() => {
        if (!disposed) {
          setRenderError(message);
        }
      }, 0);
    };
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      if (renderError) {
        commitRenderError(null);
      }
    } catch (error) {
      commitRenderError(error instanceof Error ? error.message : "WebGL unavailable");
      return () => {
        disposed = true;
      };
    }
    if (!renderer.capabilities.isWebGL2) {
      renderer.dispose();
      commitRenderError("WebGL2 unavailable");
      return () => {
        disposed = true;
      };
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
      commitRenderError(
        `Volume exceeds this browser's 3D texture limit (${largestDimension} > ${max3DTextureSize}).`
      );
      return () => {
        disposed = true;
      };
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(clearColor, 1);
    renderer.domElement.className = "viewer-webgl-canvas";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = createVolumeCamera({
      mode: volumeCameraMode,
      volumeRadius,
    });
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
    cameraRigRef.current = { camera, controls };
    applyVolumeCameraPreset({
      camera,
      controls,
      preset: volumeViewPreset,
      volumeRadius,
    });

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material =
      resolvedSource.kind === "scalar"
        ? new THREE.ShaderMaterial({
            uniforms: {
              uData: { value: null },
              uSteps: { value: sampleBudget.interactiveSteps },
              uDensity: { value: density },
              uWindowLow: { value: 0.0 },
              uWindowHigh: { value: 1.0 },
              uInvert: { value: scalarRenderConfigRef.current.negative },
              uColorMap: { value: scalarRenderConfigRef.current.colorMapShaderValue },
              uSignalFloor: { value: scalarRenderConfigRef.current.signalFloor },
              uDensityScale: { value: scalarRenderConfigRef.current.densityScale },
              uLightingEnabled: { value: scalarRenderConfigRef.current.lightingEnabled },
              uLightingStrength: { value: scalarRenderConfigRef.current.lightingStrength },
              uVoxelStep: { value: new THREE.Vector3(scalarVoxelStep.x, scalarVoxelStep.y, scalarVoxelStep.z) },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(clipBounds.min.x, clipBounds.min.y, clipBounds.min.z) },
              uClipMax: { value: new THREE.Vector3(clipBounds.max.x, clipBounds.max.y, clipBounds.max.z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
              uCameraDirectionLocal: { value: new THREE.Vector3(0, 0, -1) },
              uOrthographicCamera: { value: volumeCameraMode.isOrthographic },
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
              uSteps: { value: sampleBudget.interactiveSteps },
              uDensity: { value: density },
              uClipMin: { value: new THREE.Vector3(clipBounds.min.x, clipBounds.min.y, clipBounds.min.z) },
              uClipMax: { value: new THREE.Vector3(clipBounds.max.x, clipBounds.max.y, clipBounds.max.z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
              uCameraDirectionLocal: { value: new THREE.Vector3(0, 0, -1) },
              uOrthographicCamera: { value: volumeCameraMode.isOrthographic },
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

    const edgeGeometry = new THREE.EdgesGeometry(geometry);
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: renderPolicy === "scalar" ? 0x8fb7d9 : 0x334155,
      depthTest: false,
      depthWrite: false,
      opacity: renderPolicy === "scalar" ? 0.78 : 0.55,
      transparent: true,
    });
    const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    edgeLines.scale.set(normalizedScale.x, normalizedScale.y, normalizedScale.z);
    edgeLines.renderOrder = 2;
    scene.add(edgeLines);

    const clipEdgeGeometry = new THREE.EdgesGeometry(geometry);
    const clipEdgeMaterial = new THREE.LineBasicMaterial({
      color: 0xf7c948,
      depthTest: false,
      depthWrite: false,
      opacity: 0.94,
      transparent: true,
    });
    const clipEdgeLines = new THREE.LineSegments(clipEdgeGeometry, clipEdgeMaterial);
    const clipSize = {
      x: Math.max(0.001, clipBounds.max.x - clipBounds.min.x),
      y: Math.max(0.001, clipBounds.max.y - clipBounds.min.y),
      z: Math.max(0.001, clipBounds.max.z - clipBounds.min.z),
    };
    const clipCenter = {
      x: clipBounds.min.x + clipSize.x / 2,
      y: clipBounds.min.y + clipSize.y / 2,
      z: clipBounds.min.z + clipSize.z / 2,
    };
    clipEdgeLines.scale.set(
      normalizedScale.x * clipSize.x,
      normalizedScale.y * clipSize.y,
      normalizedScale.z * clipSize.z
    );
    clipEdgeLines.position.set(
      normalizedScale.x * (clipCenter.x - 0.5),
      normalizedScale.y * (clipCenter.y - 0.5),
      normalizedScale.z * (clipCenter.z - 0.5)
    );
    clipEdgeLines.renderOrder = 3;
    clipEdgeLines.visible = volumeClipCue.active;
    scene.add(clipEdgeLines);

    const slicePlaneGeometry = new THREE.PlaneGeometry(1, 1);
    const createSlicePlane = (color: number, opacity: number) => {
      const sliceMaterial = new THREE.MeshBasicMaterial({
        color,
        depthTest: false,
        depthWrite: false,
        opacity,
        side: THREE.DoubleSide,
        transparent: true,
      });
      const planeMesh = new THREE.Mesh(slicePlaneGeometry, sliceMaterial);
      planeMesh.renderOrder = 4;
      return planeMesh;
    };
    const sliceCursorPlanes: SliceCursorPlanes = {
      x: createSlicePlane(0xf47f72, 0.16),
      y: createSlicePlane(0x66c78d, 0.14),
      z: createSlicePlane(0x74a7ff, 0.15),
    };
    const initialSliceCursorCue = sliceCursorCueRef.current;
    sliceCursorPlanes.x.rotation.y = Math.PI / 2;
    sliceCursorPlanes.y.rotation.x = -Math.PI / 2;
    if (initialSliceCursorCue) {
      applyVolumeSliceCursorPlanes({
        planes: sliceCursorPlanes,
        cue: initialSliceCursorCue,
        normalizedScale,
      });
    }
    sliceCursorPlanesRef.current = sliceCursorPlanes;
    scene.add(sliceCursorPlanes.x, sliceCursorPlanes.y, sliceCursorPlanes.z);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));

    const volumeUniforms = material.uniforms as Record<string, { value: THREE.Vector3 | number | boolean | null }>;
    const cameraPositionUniform = volumeUniforms.uCameraPositionLocal as { value: THREE.Vector3 };
    const cameraDirectionUniform = volumeUniforms.uCameraDirectionLocal as { value: THREE.Vector3 };
    const stepsUniform = (material.uniforms as Record<string, { value: number }>).uSteps;
    let currentSteps = sampleBudget.interactiveSteps;
    let lastInteractionAt = window.performance.now();
    const setSamplingSteps = (steps: number) => {
      const nextSteps = Math.max(1, Math.floor(steps));
      if (nextSteps === currentSteps) {
        return;
      }
      currentSteps = nextSteps;
      stepsUniform.value = nextSteps;
    };
    const resetInteractiveSampling = () => {
      lastInteractionAt = window.performance.now();
      setSamplingSteps(sampleBudget.interactiveSteps);
    };
    const markInteractionSettled = () => {
      lastInteractionAt = window.performance.now();
    };
    controls.addEventListener("start", resetInteractiveSampling);
    controls.addEventListener("change", resetInteractiveSampling);
    controls.addEventListener("end", markInteractionSettled);

    const cameraWorldDirection = new THREE.Vector3();
    const cameraWorldPoint = new THREE.Vector3();
    const cameraLocal = new THREE.Vector3();
    const cameraDirectionLocal = new THREE.Vector3();
    const render = () => {
      cameraLocal.copy(camera.position);
      mesh.worldToLocal(cameraLocal);
      camera.getWorldDirection(cameraWorldDirection);
      cameraWorldPoint.copy(camera.position).add(cameraWorldDirection);
      cameraDirectionLocal.copy(cameraWorldPoint);
      mesh.worldToLocal(cameraDirectionLocal).sub(cameraLocal).normalize();
      cameraPositionUniform.value.copy(cameraLocal);
      cameraDirectionUniform.value.copy(cameraDirectionLocal);
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
      configureVolumeCameraProjection({ camera, width, height, volumeRadius });
      camera.lookAt(0, 0, 0);
      controls.handleResize();
      render();
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);

    let texture3D: THREE.Data3DTexture | null = null;
    let animationFrame = 0;
    const animate = () => {
      if (disposed) {
        return;
      }
      controls.update();
      if (window.performance.now() - lastInteractionAt > 120 && currentSteps < sampleBudget.settledSteps) {
        setSamplingSteps(advanceProgressiveVolumeSteps(currentSteps, sampleBudget));
      }
      render();
      animationFrame = window.requestAnimationFrame(animate);
    };

    const loadPromise =
      resolvedSource.kind === "scalar"
        ? resolvedSource.loadScalarVolume().then(async (payload) => {
            const texture = await scalarToVolumeTexture(payload, texturePolicy);
            scalarRangeRef.current = { rawMin: payload.rawMin, rawMax: payload.rawMax };
            const latestScalarRenderConfig = scalarRenderConfigRef.current;
            const normalizedWindow = normalizeWindowRange(
              latestScalarRenderConfig.enhancement,
              payload.rawMin,
              payload.rawMax
            );
            scalarUniformsRef.current = {
              uWindowLow: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowLow as { value: number },
              uWindowHigh: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowHigh as { value: number },
              uInvert: (material.uniforms as Record<string, { value: number | boolean | null }>).uInvert as { value: boolean },
              uColorMap: (material.uniforms as Record<string, { value: number | boolean | null }>).uColorMap as { value: number },
              uSignalFloor: (material.uniforms as Record<string, { value: number | boolean | null }>).uSignalFloor as { value: number },
              uDensityScale: (material.uniforms as Record<string, { value: number | boolean | null }>).uDensityScale as { value: number },
              uLightingEnabled: (material.uniforms as Record<string, { value: number | boolean | null }>).uLightingEnabled as { value: boolean },
              uLightingStrength: (material.uniforms as Record<string, { value: number | boolean | null }>).uLightingStrength as { value: number },
            };
            scalarUniformsRef.current.uWindowLow.value = normalizedWindow.low;
            scalarUniformsRef.current.uWindowHigh.value = normalizedWindow.high;
            scalarUniformsRef.current.uInvert.value = latestScalarRenderConfig.negative;
            scalarUniformsRef.current.uColorMap.value = latestScalarRenderConfig.colorMapShaderValue;
            scalarUniformsRef.current.uSignalFloor.value = latestScalarRenderConfig.signalFloor;
            scalarUniformsRef.current.uDensityScale.value = latestScalarRenderConfig.densityScale;
            scalarUniformsRef.current.uLightingEnabled.value = latestScalarRenderConfig.lightingEnabled;
            scalarUniformsRef.current.uLightingStrength.value = latestScalarRenderConfig.lightingStrength;
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
      cameraRigRef.current = null;
      sliceCursorPlanesRef.current = null;
      scalarRangeRef.current = null;
      observer.disconnect();
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      controls.removeEventListener("start", resetInteractiveSampling);
      controls.removeEventListener("change", resetInteractiveSampling);
      controls.removeEventListener("end", markInteractionSettled);
      controls.dispose();
      geometry.dispose();
      edgeGeometry.dispose();
      edgeMaterial.dispose();
      clipEdgeGeometry.dispose();
      clipEdgeMaterial.dispose();
      slicePlaneGeometry.dispose();
      sliceCursorPlanes.x.material.dispose();
      sliceCursorPlanes.y.material.dispose();
      sliceCursorPlanes.z.material.dispose();
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
    renderPolicy,
    sampleBudget,
    scalarVoxelStep.x,
    scalarVoxelStep.y,
    scalarVoxelStep.z,
    clipBounds.max.x,
    clipBounds.max.y,
    clipBounds.max.z,
    clipBounds.min.x,
    clipBounds.min.y,
    clipBounds.min.z,
    normalizedScale.x,
    normalizedScale.y,
    normalizedScale.z,
    normalizedScale,
    volumeDepth,
    volumeRadius,
    volumeCameraMode,
    volumeViewPreset,
    physicalGeometry.worldDepth,
    physicalGeometry.worldHeight,
    physicalGeometry.worldWidth,
    volumeClipCue.active,
  ]);

  useEffect(() => {
    const planes = sliceCursorPlanesRef.current;
    if (!planes) {
      return;
    }
    applyVolumeSliceCursorPlanes({
      planes,
      cue: sliceCursorCue,
      normalizedScale,
    });
    requestRenderRef.current?.();
  }, [normalizedScale, sliceCursorCue]);

  const backendLabel = resolvedSource?.kind ?? "atlas";
  const renderVolumeOrientationOverlay = (variant?: "fallback") => (
    <div
      className={[
        "viewer-volume-orientation-triad",
        variant === "fallback" ? "viewer-volume-orientation-triad-fallback" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      aria-label={`Volume orientation ${orientationCue.frame}`}
      data-viewer-volume-orientation="true"
      data-viewer-orientation-frame={orientationCue.frame}
      data-viewer-orientation-x={orientationCue.x.label}
      data-viewer-orientation-y={orientationCue.y.label}
      data-viewer-orientation-z={orientationCue.z.label}
    >
      <span className="viewer-volume-orientation-frame">{orientationCue.frame}</span>
      <span className="viewer-volume-orientation-axis viewer-volume-orientation-axis-x">
        {orientationCue.x.label}
      </span>
      <span className="viewer-volume-orientation-axis viewer-volume-orientation-axis-y">
        {orientationCue.y.label}
      </span>
      <span className="viewer-volume-orientation-axis viewer-volume-orientation-axis-z">
        {orientationCue.z.label}
      </span>
    </div>
  );
  const renderVolumeScaleBar = () =>
    volumeScaleBar.visible ? (
      <div
        className="viewer-volume-scale-bar"
        aria-label={`Volume scale ${volumeScaleBar.label}`}
        data-viewer-volume-scale-bar="true"
        data-viewer-scale-length={volumeScaleBar.length.toFixed(4)}
        data-viewer-scale-label={volumeScaleBar.label}
        data-viewer-scale-fraction={volumeScaleBar.fraction.toFixed(4)}
      >
        <span className="viewer-volume-scale-track">
          <span
            className="viewer-volume-scale-measure"
            style={{ width: `${(volumeScaleBar.fraction * 100).toFixed(2)}%` }}
          />
        </span>
        <span className="viewer-volume-scale-label">{volumeScaleBar.label}</span>
      </div>
    ) : null;
  const renderVolumeAxisCue = () =>
    volumeAxisCue.visible ? (
      <div
        className="viewer-volume-axis-cue"
        aria-label={`Volume axes ${volumeAxisCue.summary}`}
        data-viewer-volume-axis-cue="true"
        data-viewer-axis-unit={volumeAxisCue.unit}
        data-viewer-axis-summary={volumeAxisCue.summary}
        data-viewer-axis-x-label={volumeAxisCue.x.label}
        data-viewer-axis-y-label={volumeAxisCue.y.label}
        data-viewer-axis-z-label={volumeAxisCue.z.label}
        data-viewer-axis-x-length={volumeAxisCue.x.length.toFixed(4)}
        data-viewer-axis-y-length={volumeAxisCue.y.length.toFixed(4)}
        data-viewer-axis-z-length={volumeAxisCue.z.length.toFixed(4)}
      >
        <span className="viewer-volume-axis-chip viewer-volume-axis-chip-x">{volumeAxisCue.x.label}</span>
        <span className="viewer-volume-axis-chip viewer-volume-axis-chip-y">{volumeAxisCue.y.label}</span>
        <span className="viewer-volume-axis-chip viewer-volume-axis-chip-z">{volumeAxisCue.z.label}</span>
      </div>
    ) : null;
  const renderVolumeClipCue = () =>
    volumeClipCue.active ? (
      <div
        className="viewer-volume-clip-cue"
        aria-label={`Active volume cutaway ${volumeClipCue.summary}`}
        data-viewer-volume-clip-cue="true"
        data-viewer-clip-active="true"
        data-viewer-clip-unit={volumeClipCue.unit}
        data-viewer-clip-summary={volumeClipCue.summary}
        data-viewer-clip-x-label={volumeClipCue.x.label}
        data-viewer-clip-y-label={volumeClipCue.y.label}
        data-viewer-clip-z-label={volumeClipCue.z.label}
        data-viewer-clip-x-length={volumeClipCue.x.length.toFixed(4)}
        data-viewer-clip-y-length={volumeClipCue.y.length.toFixed(4)}
        data-viewer-clip-z-length={volumeClipCue.z.length.toFixed(4)}
      >
        <span className="viewer-volume-clip-cue-title">Cutaway</span>
        <span className="viewer-volume-clip-chip">{volumeClipCue.x.label}</span>
        <span className="viewer-volume-clip-chip">{volumeClipCue.y.label}</span>
        <span className="viewer-volume-clip-chip">{volumeClipCue.z.label}</span>
      </div>
    ) : null;
  const renderVolumeSliceCursorCue = () =>
    sliceCursorCue.visible ? (
      <div
        className="viewer-volume-slice-cue"
        aria-label={`Current volume slices ${sliceCursorCue.summary}`}
        data-viewer-volume-slice-cue="true"
        data-viewer-slice-summary={sliceCursorCue.summary}
        data-viewer-slice-unit={sliceCursorCue.unit}
        data-viewer-slice-x-label={sliceCursorCue.x.label}
        data-viewer-slice-y-label={sliceCursorCue.y.label}
        data-viewer-slice-z-label={sliceCursorCue.z.label}
      >
        <span className="viewer-volume-slice-cue-title">Slices</span>
        <span className="viewer-volume-slice-chip viewer-volume-slice-chip-x">
          {sliceCursorCue.x.label}
          <em>{sliceCursorCue.x.positionLabel}</em>
        </span>
        <span className="viewer-volume-slice-chip viewer-volume-slice-chip-y">
          {sliceCursorCue.y.label}
          <em>{sliceCursorCue.y.positionLabel}</em>
        </span>
        <span className="viewer-volume-slice-chip viewer-volume-slice-chip-z">
          {sliceCursorCue.z.label}
          <em>{sliceCursorCue.z.positionLabel}</em>
        </span>
      </div>
    ) : null;

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
        data-viewer-scalar-colormap={scalarColorMap.id}
        data-viewer-scalar-colormap-label={scalarColorMap.label}
        data-viewer-signal-floor={scalarTransfer.signalFloor.toFixed(2)}
        data-viewer-density-scale={scalarTransfer.densityScale.toFixed(2)}
        data-viewer-depth-lighting={scalarLighting.enabled ? "true" : "false"}
        data-viewer-lighting-strength={scalarLighting.strength.toFixed(2)}
        data-viewer-scale-label={volumeScaleBar.label || undefined}
        data-viewer-view-preset={volumeViewPreset.id}
        data-viewer-view-label={volumeViewPreset.label}
        data-viewer-camera-mode={volumeCameraMode.id}
        data-viewer-camera-label={volumeCameraMode.label}
        data-viewer-camera-orthographic={volumeCameraMode.isOrthographic ? "true" : "false"}
        data-viewer-axis-summary={volumeAxisCue.summary || undefined}
        data-viewer-axis-unit={volumeAxisCue.unit}
        data-viewer-axis-x-length={volumeAxisCue.visible ? volumeAxisCue.x.length.toFixed(4) : undefined}
        data-viewer-axis-y-length={volumeAxisCue.visible ? volumeAxisCue.y.length.toFixed(4) : undefined}
        data-viewer-axis-z-length={volumeAxisCue.visible ? volumeAxisCue.z.length.toFixed(4) : undefined}
        data-viewer-slice-summary={sliceCursorCue.summary || undefined}
        data-viewer-slice-unit={sliceCursorCue.unit}
        data-viewer-slice-x-index={String(sliceCursorCue.x.index)}
        data-viewer-slice-y-index={String(sliceCursorCue.y.index)}
        data-viewer-slice-z-index={String(sliceCursorCue.z.index)}
        data-viewer-slice-x-normalized={sliceCursorCue.x.normalized.toFixed(4)}
        data-viewer-slice-y-normalized={sliceCursorCue.y.normalized.toFixed(4)}
        data-viewer-slice-z-normalized={sliceCursorCue.z.normalized.toFixed(4)}
        data-viewer-slice-x-position={sliceCursorCue.x.position.toFixed(4)}
        data-viewer-slice-y-position={sliceCursorCue.y.position.toFixed(4)}
        data-viewer-slice-z-position={sliceCursorCue.z.position.toFixed(4)}
        data-viewer-clip-active={volumeClipCue.active ? "true" : "false"}
        data-viewer-clip-summary={volumeClipCue.summary || undefined}
        data-viewer-clip-unit={volumeClipCue.unit}
        data-viewer-clip-x-length={volumeClipCue.active ? volumeClipCue.x.length.toFixed(4) : undefined}
        data-viewer-clip-y-length={volumeClipCue.active ? volumeClipCue.y.length.toFixed(4) : undefined}
        data-viewer-clip-z-length={volumeClipCue.active ? volumeClipCue.z.length.toFixed(4) : undefined}
        data-viewer-orientation-frame={orientationCue.frame}
        data-viewer-orientation-x={orientationCue.x.label}
        data-viewer-orientation-y={orientationCue.y.label}
        data-viewer-orientation-z={orientationCue.z.label}
        data-viewer-clip-x-min={clipBounds.min.x.toFixed(2)}
        data-viewer-clip-x-max={clipBounds.max.x.toFixed(2)}
        data-viewer-clip-y-min={clipBounds.min.y.toFixed(2)}
        data-viewer-clip-y-max={clipBounds.max.y.toFixed(2)}
        data-viewer-clip-z-min={clipBounds.min.z.toFixed(2)}
        data-viewer-clip-z-max={clipBounds.max.z.toFixed(2)}
        data-viewer-volume-channel={scalarChannel == null ? undefined : String(scalarChannel)}
        data-viewer-physical-width={physicalGeometry.worldWidth.toFixed(4)}
        data-viewer-physical-height={physicalGeometry.worldHeight.toFixed(4)}
        data-viewer-physical-depth={physicalGeometry.worldDepth.toFixed(4)}
        data-viewer-physical-scale-x={physicalGeometry.normalizedScale.x.toFixed(4)}
        data-viewer-physical-scale-y={physicalGeometry.normalizedScale.y.toFixed(4)}
        data-viewer-physical-scale-z={physicalGeometry.normalizedScale.z.toFixed(4)}
        data-viewer-physical-anisotropic={physicalGeometry.isAnisotropic ? "true" : "false"}
        data-viewer-progressive-sampling="true"
        data-viewer-sample-steps-interactive={String(sampleBudget.interactiveSteps)}
        data-viewer-sample-steps-settled={String(sampleBudget.settledSteps)}
      >
        <div className="viewer-image-fallback" style={{ aspectRatio: `${Math.max(1e-6, plane.aspect_ratio)}` }}>
          <img src={fallbackImageUrl} alt="Volume fallback preview" className="viewer-image-fallback-media" />
        </div>
        {renderVolumeOrientationOverlay("fallback")}
        {renderVolumeAxisCue()}
        {renderVolumeSliceCursorCue()}
        {renderVolumeClipCue()}
        {renderVolumeScaleBar()}
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
      data-viewer-scalar-colormap={scalarColorMap.id}
      data-viewer-scalar-colormap-label={scalarColorMap.label}
      data-viewer-signal-floor={scalarTransfer.signalFloor.toFixed(2)}
      data-viewer-density-scale={scalarTransfer.densityScale.toFixed(2)}
      data-viewer-depth-lighting={scalarLighting.enabled ? "true" : "false"}
      data-viewer-lighting-strength={scalarLighting.strength.toFixed(2)}
      data-viewer-scale-label={volumeScaleBar.label || undefined}
      data-viewer-view-preset={volumeViewPreset.id}
      data-viewer-view-label={volumeViewPreset.label}
      data-viewer-camera-mode={volumeCameraMode.id}
      data-viewer-camera-label={volumeCameraMode.label}
      data-viewer-camera-orthographic={volumeCameraMode.isOrthographic ? "true" : "false"}
      data-viewer-axis-summary={volumeAxisCue.summary || undefined}
      data-viewer-axis-unit={volumeAxisCue.unit}
      data-viewer-axis-x-length={volumeAxisCue.visible ? volumeAxisCue.x.length.toFixed(4) : undefined}
      data-viewer-axis-y-length={volumeAxisCue.visible ? volumeAxisCue.y.length.toFixed(4) : undefined}
      data-viewer-axis-z-length={volumeAxisCue.visible ? volumeAxisCue.z.length.toFixed(4) : undefined}
      data-viewer-slice-summary={sliceCursorCue.summary || undefined}
      data-viewer-slice-unit={sliceCursorCue.unit}
      data-viewer-slice-x-index={String(sliceCursorCue.x.index)}
      data-viewer-slice-y-index={String(sliceCursorCue.y.index)}
      data-viewer-slice-z-index={String(sliceCursorCue.z.index)}
      data-viewer-slice-x-normalized={sliceCursorCue.x.normalized.toFixed(4)}
      data-viewer-slice-y-normalized={sliceCursorCue.y.normalized.toFixed(4)}
      data-viewer-slice-z-normalized={sliceCursorCue.z.normalized.toFixed(4)}
      data-viewer-slice-x-position={sliceCursorCue.x.position.toFixed(4)}
      data-viewer-slice-y-position={sliceCursorCue.y.position.toFixed(4)}
      data-viewer-slice-z-position={sliceCursorCue.z.position.toFixed(4)}
      data-viewer-clip-active={volumeClipCue.active ? "true" : "false"}
      data-viewer-clip-summary={volumeClipCue.summary || undefined}
      data-viewer-clip-unit={volumeClipCue.unit}
      data-viewer-clip-x-length={volumeClipCue.active ? volumeClipCue.x.length.toFixed(4) : undefined}
      data-viewer-clip-y-length={volumeClipCue.active ? volumeClipCue.y.length.toFixed(4) : undefined}
      data-viewer-clip-z-length={volumeClipCue.active ? volumeClipCue.z.length.toFixed(4) : undefined}
      data-viewer-orientation-frame={orientationCue.frame}
      data-viewer-orientation-x={orientationCue.x.label}
      data-viewer-orientation-y={orientationCue.y.label}
      data-viewer-orientation-z={orientationCue.z.label}
      data-viewer-clip-x-min={clipBounds.min.x.toFixed(2)}
      data-viewer-clip-x-max={clipBounds.max.x.toFixed(2)}
      data-viewer-clip-y-min={clipBounds.min.y.toFixed(2)}
      data-viewer-clip-y-max={clipBounds.max.y.toFixed(2)}
      data-viewer-clip-z-min={clipBounds.min.z.toFixed(2)}
      data-viewer-clip-z-max={clipBounds.max.z.toFixed(2)}
      data-viewer-volume-channel={scalarChannel == null ? undefined : String(scalarChannel)}
      data-viewer-physical-width={physicalGeometry.worldWidth.toFixed(4)}
      data-viewer-physical-height={physicalGeometry.worldHeight.toFixed(4)}
      data-viewer-physical-depth={physicalGeometry.worldDepth.toFixed(4)}
      data-viewer-physical-scale-x={physicalGeometry.normalizedScale.x.toFixed(4)}
      data-viewer-physical-scale-y={physicalGeometry.normalizedScale.y.toFixed(4)}
      data-viewer-physical-scale-z={physicalGeometry.normalizedScale.z.toFixed(4)}
      data-viewer-physical-anisotropic={physicalGeometry.isAnisotropic ? "true" : "false"}
      data-viewer-progressive-sampling="true"
      data-viewer-sample-steps-interactive={String(sampleBudget.interactiveSteps)}
      data-viewer-sample-steps-settled={String(sampleBudget.settledSteps)}
    >
      {renderVolumeOrientationOverlay()}
      {renderVolumeAxisCue()}
      {renderVolumeSliceCursorCue()}
      {renderVolumeClipCue()}
      {renderVolumeScaleBar()}
    </div>
  );
}
