import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

import { computeScalarVolumeAutoContrast, scalarVolumePayloadToHalfFloat } from "./scalarVolume";
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

export {
  scalarVolumePayloadToTextureBytes,
  scalarVolumePayloadToHalfFloat,
  scalarVolumePayloadValueAt,
} from "./scalarVolume";
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

type MultichannelVolumeSource = {
  kind: "multichannel";
  // Enabled channel indices, in render order. The renderer loads one R16F volume
  // per index and fuses them; changing this set reloads (cached so no re-fetch).
  channelIndices: number[];
  loadChannel: (channel: number) => Promise<ScalarVolumePayload>;
  fallbackImageUrl: string;
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
  volumeSource?: ScalarVolumeSource | AtlasVolumeSource | MultichannelVolumeSource;
};

/**
 * Parse an `#rrggbb` color to a LINEAR-light RGB triplet (0..1). The multichannel
 * shader sums channel emissions in linear space, so channel LUT colors (authored
 * as sRGB hex) must be sRGB->linear decoded first, or overlaps hue-shift/blow out.
 */
export const hexToLinearRgb = (hex: string | undefined): [number, number, number] => {
  const value = String(hex || "").trim().replace(/^#/, "");
  const full =
    value.length === 3
      ? value
          .split("")
          .map((c) => c + c)
          .join("")
      : value.padEnd(6, "0").slice(0, 6);
  const srgb = [0, 1, 2].map((i) => {
    const channel = Number.parseInt(full.slice(i * 2, i * 2 + 2), 16);
    return Number.isFinite(channel) ? channel / 255 : 0;
  });
  const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  return [toLinear(srgb[0]), toLinear(srgb[1]), toLinear(srgb[2])];
};

type MultichannelRenderConfig = {
  channels: Array<{
    index: number;
    color: [number, number, number];
    // The user's explicit window, or null to fall back to the per-channel
    // auto-contrast computed from the loaded volume (ImageJ-style).
    manualWindow: { low: number; high: number } | null;
    invert: boolean;
  }>;
  gammaScale: number;
  brightness: number;
  densityScale: number;
};

// Bounded cache of decoded per-channel volume payloads, keyed file:channel:time,
// so toggling a channel (which rebuilds the renderer) reuses already-fetched
// channels instead of re-downloading ~184 MB each. Capped to the max simultaneous
// channels; the default view loads a single channel.
const channelVolumeCache = new Map<string, ScalarVolumePayload>();
const channelVolumeCacheKey = (fileId: string, channel: number, t: number) => `${fileId}:${channel}:${t}`;
const rememberChannelVolume = (key: string, payload: ScalarVolumePayload): void => {
  channelVolumeCache.delete(key);
  channelVolumeCache.set(key, payload);
  while (channelVolumeCache.size > MAX_VOLUME_CHANNELS) {
    const oldest = channelVolumeCache.keys().next().value as string | undefined;
    if (oldest === undefined) {
      break;
    }
    channelVolumeCache.delete(oldest);
  }
};

type MultichannelUniformSet = {
  uChannelCount: { value: number };
  uChanLow: { value: number[] };
  uChanHigh: { value: number[] };
  uChanColor: { value: THREE.Vector3[] };
  uChanInvert: { value: boolean[] };
  uGammaScale: { value: number };
  uBrightness: { value: number };
  uDensityScale: { value: number };
};

// Push per-channel color/window/invert + gamma into the live uniforms WITHOUT
// touching uChannelCount — the count is owned by the texture-load step so the
// shader never samples a channel slot whose texture has not been bound yet.
const DEFAULT_CHANNEL_WINDOW = { low: 0, high: 1 };
const applyMultichannelChannelUniforms = (
  uniforms: MultichannelUniformSet,
  config: MultichannelRenderConfig,
  autoWindows: Map<number, { low: number; high: number }>
): void => {
  for (let i = 0; i < MAX_VOLUME_CHANNELS; i++) {
    const channel = config.channels[i];
    const window = channel
      ? channel.manualWindow ?? autoWindows.get(channel.index) ?? DEFAULT_CHANNEL_WINDOW
      : DEFAULT_CHANNEL_WINDOW;
    uniforms.uChanLow.value[i] = window.low;
    uniforms.uChanHigh.value[i] = window.high;
    uniforms.uChanColor.value[i].set(
      channel ? channel.color[0] : 0,
      channel ? channel.color[1] : 0,
      channel ? channel.color[2] : 0
    );
    uniforms.uChanInvert.value[i] = channel ? channel.invert : false;
  }
  uniforms.uGammaScale.value = config.gammaScale;
  uniforms.uBrightness.value = config.brightness;
  uniforms.uDensityScale.value = config.densityScale;
};

export const MAX_STEPS = 512;
const MIN_INTERACTIVE_STEPS = 32;
const SAMPLE_RAMP_FACTOR = 1.5;
const DEFAULT_VOLUME_CLEAR = 0x07090d;
const DEFAULT_VOLUME_AXIS_SIZES: UploadViewerInfo["axis_sizes"] = { T: 1, C: 1, Z: 1, Y: 1, X: 1 };
const DEFAULT_VOLUME_CAMERA_FOV = 42;
const MIN_ORTHOGRAPHIC_FRUSTUM_HEIGHT = 1.8;
const VOLUME_CAMERA_FIT_MARGIN = 1.35;
const VOLUME_INTERIOR_MIN_DISTANCE_FACTOR = 0.045;
const VOLUME_INTERIOR_MAX_ZOOM = 8;
const VOLUME_INTERIOR_MIN_LOOK_DISTANCE = 0.18;
const VOLUME_INTERIOR_LOOK_DISTANCE_FACTOR = 0.72;
const DEFAULT_VOLUME_CAMERA_SAFE_INSETS = {
  top: 0.08,
  right: 0.06,
  bottom: 0.22,
  left: 0.06,
} as const;

type VolumeCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

type VolumeCameraSafeInsets = Partial<{
  top: number;
  right: number;
  bottom: number;
  left: number;
}>;

export type VolumeCameraFit = {
  distance: number;
  minDistance: number;
  insideMinDistance: number;
  maxDistance: number;
  inspectMaxZoom: number;
  orthographicFrustumHeight: number;
  safeWidth: number;
  safeHeight: number;
};

type VolumeVector = { x: number; y: number; z: number };

type VolumeClipBounds = {
  min: VolumeVector;
  max: VolumeVector;
};

/**
 * Normalized texture-depth position [0, 1] of the cutaway cut face for a given Z
 * slice. The volume is sliced at the live Z cursor, so scrubbing Z sweeps the
 * cut through the stack. The `(index + 0.5) / depth` mapping samples voxel
 * centers — identical to the backend slice extraction — so the exposed
 * cross-section lines up with the 2D slice at the same index. Out-of-range and
 * missing indices clamp to the volume (a null index centers the cut).
 */
export const resolveVolumeCutawayCutZ = (
  zIndex: number | null | undefined,
  depth: number
): number => {
  const safeDepth = Math.max(1, depth);
  const fallback = Math.floor((safeDepth - 1) / 2);
  const index = Math.max(0, Math.min(safeDepth - 1, Math.floor(Number(zIndex ?? fallback))));
  return Math.min(1, Math.max(0, (index + 0.5) / safeDepth));
};

/**
 * Clip box for the Z-cursor cutaway: keep the volume from z=0 up to the cut face
 * so the exposed interior cross-section faces the overview camera.
 */
export const resolveVolumeCutawayClip = (cutZ: number): VolumeClipBounds => ({
  min: { x: 0, y: 0, z: 0 },
  max: { x: 1, y: 1, z: cutZ },
});

export type VolumeInteriorCameraFrame = {
  position: VolumeVector;
  target: VolumeVector;
  lookDistance: number;
};

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

const normalizeVector = (value: VolumeVector): VolumeVector => {
  const length = Math.sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
  if (!Number.isFinite(length) || length < 1e-6) {
    return { x: 0, y: 0, z: 1 };
  }
  return {
    x: value.x / length,
    y: value.y / length,
    z: value.z / length,
  };
};

const roundSceneCoordinate = (value: number): number => Number(value.toFixed(6));

export function isVolumeInteriorInspectionActive({
  clipActive,
  cameraMode,
}: {
  clipActive: boolean;
  cameraMode: VolumeCameraMode;
}): boolean {
  return clipActive && !cameraMode.isOrthographic;
}

export function shouldShowVolumeSliceCursorPlanes({
  cueVisible,
  interiorInspectionActive,
  cutawayActive = false,
}: {
  cueVisible: boolean;
  interiorInspectionActive: boolean;
  cutawayActive?: boolean;
}): boolean {
  // Hide the flat translucent X/Y/Z cursor quads whenever the view is focused on
  // the interior — both the legacy fly-inside and the Z-cursor cutaway. In
  // cutaway the crisp opaque cut face IS the inspection surface, so the cursor
  // planes only tint/occlude it (the user sees red/green washes over the slice).
  return cueVisible && !interiorInspectionActive && !cutawayActive;
}

export function shouldShowVolumeContextEdges({
  cueVisible,
  interiorInspectionActive,
}: {
  cueVisible: boolean;
  interiorInspectionActive: boolean;
}): boolean {
  return cueVisible && !interiorInspectionActive;
}

export function computeVolumeInteriorCameraFrame({
  clipBounds,
  normalizedScale,
  preset,
  volumeRadius,
}: {
  clipBounds: VolumeClipBounds;
  normalizedScale: VolumeVector;
  preset: VolumeViewPreset;
  volumeRadius: number;
}): VolumeInteriorCameraFrame {
  const centerLocal = {
    x: (clipBounds.min.x + clipBounds.max.x) / 2 - 0.5,
    y: (clipBounds.min.y + clipBounds.max.y) / 2 - 0.5,
    z: (clipBounds.min.z + clipBounds.max.z) / 2 - 0.5,
  };
  const position = {
    x: roundSceneCoordinate(centerLocal.x * normalizedScale.x),
    y: roundSceneCoordinate(centerLocal.y * normalizedScale.y),
    z: roundSceneCoordinate(centerLocal.z * normalizedScale.z),
  };
  const clipSize = {
    x: Math.max(0.001, (clipBounds.max.x - clipBounds.min.x) * normalizedScale.x),
    y: Math.max(0.001, (clipBounds.max.y - clipBounds.min.y) * normalizedScale.y),
    z: Math.max(0.001, (clipBounds.max.z - clipBounds.min.z) * normalizedScale.z),
  };
  const clipRadius = Math.sqrt(
    clipSize.x * clipSize.x +
      clipSize.y * clipSize.y +
      clipSize.z * clipSize.z
  ) / 2;
  const safeRadius = Math.max(0.25, Number.isFinite(volumeRadius) ? volumeRadius : 0.25);
  const lookDistance = roundSceneCoordinate(
    Math.max(
      VOLUME_INTERIOR_MIN_LOOK_DISTANCE,
      Math.min(safeRadius * 0.95, clipRadius * VOLUME_INTERIOR_LOOK_DISTANCE_FACTOR)
    )
  );
  const direction = normalizeVector(preset.direction);
  return {
    position,
    target: {
      x: roundSceneCoordinate(position.x - direction.x * lookDistance),
      y: roundSceneCoordinate(position.y - direction.y * lookDistance),
      z: roundSceneCoordinate(position.z - direction.z * lookDistance),
    },
    lookDistance,
  };
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

  float alphaFromOpacity(float opacityValue, float stepLength) {
    float baseAlpha = clamp(opacityValue * uDensity, 0.0, 0.95);
    float stepScale = max(0.001, stepLength * 128.0);
    return 1.0 - pow(1.0 - baseAlpha, stepScale);
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
      float alpha = alphaFromOpacity(max(max(sampleColor.r, sampleColor.g), sampleColor.b), length(delta));
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
  uniform vec3 uVolumeScale;   // per-axis world EXTENT ratio (box dimensions)
  uniform vec3 uVoxelSpacing;  // per-axis voxel SPACING ratio (mm), max-normalized
  uniform float uEdgeStrength;
  uniform float uInteriorOpacity;
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

  float alphaFromOpacity(float opacityValue, float stepLength) {
    float baseAlpha = clamp(opacityValue * uDensity * uDensityScale, 0.0, 0.95);
    float stepScale = max(0.001, stepLength * 128.0);
    return 1.0 - pow(1.0 - baseAlpha, stepScale);
  }

  // Central-difference gradient of the windowed signal. Components are spaced one
  // voxel apart on each axis, so this is the local rate of change that marks a
  // tissue interface (e.g. the CSF<->parenchyma wall of a ventricle).
  vec3 scalarGradient(vec3 location) {
    float gx = sampleWindowed(location + vec3(uVoxelStep.x, 0.0, 0.0)) -
      sampleWindowed(location - vec3(uVoxelStep.x, 0.0, 0.0));
    float gy = sampleWindowed(location + vec3(0.0, uVoxelStep.y, 0.0)) -
      sampleWindowed(location - vec3(0.0, uVoxelStep.y, 0.0));
    float gz = sampleWindowed(location + vec3(0.0, 0.0, uVoxelStep.z)) -
      sampleWindowed(location - vec3(0.0, 0.0, uVoxelStep.z));
    return vec3(gx, gy, gz);
  }

  // Levoy boundary-emphasis opacity: homogeneous interiors (low gradient) stay
  // translucent so we can see past them, while tissue interfaces (high gradient)
  // become opaque surfaces. This is what makes the ventricle walls and other
  // internal boundaries visible when looking around inside the volume.
  float structuredOpacity(float opacityValue, vec3 gradient) {
    float edge = clamp(length(gradient) * uEdgeStrength, 0.0, 1.0);
    return opacityValue * mix(uInteriorOpacity, 1.0, edge);
  }

  vec3 applyDepthLighting(vec3 location, vec3 color, vec3 gradient) {
    if (!uLightingEnabled || uLightingStrength <= 0.0) {
      return color;
    }

    // Anisotropy correction: a central difference spans one voxel per axis, so
    // the world-space gradient is the per-voxel difference divided by the voxel
    // SPACING (mm), not the full-axis extent. Dividing by spacing makes the
    // surface normal physically correct on thick (e.g. 5 mm) slices; using the
    // extent (count x spacing) previously skewed normals toward the thick axis.
    vec3 worldGradient = gradient / max(vec3(0.06), uVoxelSpacing);
    if (length(worldGradient) < 0.0001) {
      return color;
    }

    vec3 normal = normalize(worldGradient);
    vec3 lightDir = normalize(vec3(-0.45, 0.55, 0.72));
    vec3 viewDir = uOrthographicCamera
      ? -normalize(uCameraDirectionLocal)
      : normalize(uCameraPositionLocal - (location - vec3(0.5)));
    vec3 halfDir = normalize(lightDir + viewDir);
    float diffuse = max(dot(normal, lightDir), dot(-normal, lightDir));
    float specular = pow(max(dot(normal, halfDir), max(dot(-normal, halfDir), 0.0)), 22.0);
    float rim = pow(1.0 - clamp(abs(dot(normal, viewDir)), 0.0, 1.0), 2.0);
    float shade = clamp(0.42 + 0.72 * diffuse + 0.18 * rim, 0.35, 1.25);
    vec3 lit = color * shade + vec3(0.28 * specular);
    return mix(color, lit, uLightingStrength);
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
      vec3 gradient = scalarGradient(location);
      float opacity = structuredOpacity(opacityValue, gradient);
      // Optical depth must be proportional to the PHYSICAL path length per step.
      // The ray marches in normalized cube space, so scale the step by the box
      // extents; otherwise opacity is view-dependent on anisotropic volumes
      // (a ray along the thick axis traverses more material per cube-step).
      float alpha = alphaFromOpacity(opacity, length(delta * uVolumeScale));
      vec3 sampleColor = applyDepthLighting(location, scalarColor(sampleValue), gradient);
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
      vec3 maxColor = applyDepthLighting(maxLocation, scalarColor(maxValue), scalarGradient(maxLocation));
      gl_FragColor = vec4(maxColor, clamp(maxOpacity * uDensityScale * 1.2, 0.0, 1.0));
      return;
    }

    if (accum.a < 0.02) {
      discard;
    }
    gl_FragColor = accum;
  }
`;

// High-resolution cut face for the Z-cursor cutaway. A flat quad placed at the
// cut depth samples the SAME R16F volume texture as the ray-marcher, but as a
// single crisp cross-section (one texel-accurate sample per fragment) instead of
// an accumulated volumetric average. This is what makes the exposed interior
// read at full slice resolution as the user scrubs through Z, while the clipped
// volume behind it still provides 3D context.
const CUTFACE_VERTEX_SHADER = `
  varying vec2 vCutUv;

  void main() {
    // PlaneGeometry local position spans [-0.5, 0.5]; the volume's texcoord is
    // (localPosition + 0.5), so uv maps 1:1 onto the volume's X/Y sampling.
    vCutUv = position.xy + 0.5;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const CUTFACE_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uData;
  uniform float uCutZ;
  uniform float uWindowLow;
  uniform float uWindowHigh;
  uniform bool uInvert;
  uniform int uColorMap;

  varying vec2 vCutUv;

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

  void main() {
    vec3 location = vec3(
      clamp(vCutUv.x, 0.0, 1.0),
      clamp(vCutUv.y, 0.0, 1.0),
      clamp(uCutZ, 0.0, 1.0)
    );
    float raw = texture(uData, location).r;
    float normalized = clamp(
      (raw - uWindowLow) / max(0.000001, uWindowHigh - uWindowLow),
      0.0,
      1.0
    );
    if (uInvert) {
      normalized = 1.0 - normalized;
    }
    gl_FragColor = vec4(scalarColor(normalized), 1.0);
  }
`;

// Max simultaneously-rendered channels. WebGL2 guarantees >= 16 fragment texture
// units; fluorescence is typically 2-5 channels, 8 covers spectral with headroom
// for the cut-face. Enabling beyond this caps to the first MAX_CHANNELS.
const MAX_VOLUME_CHANNELS = 8;

// Multichannel fluorescence volume shader. Adapted from the scalar ray-march and
// the Allen Institute vole-core approach: each enabled channel is its own R16F 3D
// texture; per voxel we window each channel to [0,1], tint by its (linear) color,
// and combine across channels with a per-component MAX ("fuse") so overlapping
// channels keep their hue instead of additively blowing out to white. The fused
// emission is then integrated along the ray (alpha-over composite or MIP) and a
// gamma/brightness tone curve is applied — matching vole-core's final mapping so
// faint structure stays readable. GLSL ES 3.0 forbids dynamic sampler indexing, so
// channels are sampled through a constant-unrolled if-ladder, not an array index.
const MULTICHANNEL_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uChan0;
  uniform sampler3D uChan1;
  uniform sampler3D uChan2;
  uniform sampler3D uChan3;
  uniform sampler3D uChan4;
  uniform sampler3D uChan5;
  uniform sampler3D uChan6;
  uniform sampler3D uChan7;
  uniform int uChannelCount;
  uniform float uChanLow[${MAX_VOLUME_CHANNELS}];
  uniform float uChanHigh[${MAX_VOLUME_CHANNELS}];
  uniform vec3 uChanColor[${MAX_VOLUME_CHANNELS}];
  uniform bool uChanInvert[${MAX_VOLUME_CHANNELS}];

  uniform int uSteps;
  uniform float uDensity;
  uniform float uDensityScale;
  uniform vec3 uVolumeScale;
  uniform int uProjectionMode;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uCameraPositionLocal;
  uniform vec3 uCameraDirectionLocal;
  uniform bool uOrthographicCamera;
  uniform float uGammaMin;
  uniform float uGammaMax;
  uniform float uGammaScale;
  uniform float uBrightness;
  uniform float uSignalFloor;     // below this windowed intensity -> transparent
  uniform bool uLightingEnabled;
  uniform float uLightingStrength;
  uniform vec3 uVoxelStep;        // one-voxel offset per axis (gradient delta)
  uniform vec3 uVoxelSpacing;     // mm spacing ratio (anisotropy-correct normals)

  varying vec3 vPosition;

  bool intersectBox(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tsmaller = min(t0, t1);
    vec3 tbigger = max(t0, t1);
    tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
    return tFar > max(tNear, 0.0);
  }

  float sampleChannelRaw(int c, vec3 p) {
    if (c == 0) return texture(uChan0, p).r;
    if (c == 1) return texture(uChan1, p).r;
    if (c == 2) return texture(uChan2, p).r;
    if (c == 3) return texture(uChan3, p).r;
    if (c == 4) return texture(uChan4, p).r;
    if (c == 5) return texture(uChan5, p).r;
    if (c == 6) return texture(uChan6, p).r;
    return texture(uChan7, p).r;
  }

  float windowChannel(int c, float raw) {
    float n = clamp((raw - uChanLow[c]) / max(0.000001, uChanHigh[c] - uChanLow[c]), 0.0, 1.0);
    return uChanInvert[c] ? (1.0 - n) : n;
  }

  // Combined density (MAX windowed intensity across channels). Used for the
  // opacity transfer + the shading gradient; cheaper than fusing color too.
  float densityAt(vec3 location) {
    vec3 p = clamp(location, vec3(0.0), vec3(1.0));
    float density = 0.0;
    for (int c = 0; c < ${MAX_VOLUME_CHANNELS}; c++) {
      if (c >= uChannelCount) break;
      density = max(density, windowChannel(c, sampleChannelRaw(c, p)));
    }
    return density;
  }

  // Fuse all enabled channels at one voxel: returns (emission RGB, density).
  // emission = per-component MAX of color_c * windowed_c; density = MAX of
  // windowed_c (the strongest channel drives opacity).
  vec4 fuseVoxel(vec3 location) {
    vec3 p = clamp(location, vec3(0.0), vec3(1.0));
    vec3 emission = vec3(0.0);
    float density = 0.0;
    for (int c = 0; c < ${MAX_VOLUME_CHANNELS}; c++) {
      if (c >= uChannelCount) break;
      float n = windowChannel(c, sampleChannelRaw(c, p));
      emission = max(emission, uChanColor[c] * n);
      density = max(density, n);
    }
    return vec4(emission, density);
  }

  // Central-difference gradient of the combined density — the local surface
  // normal at a structure boundary, used to shade flat emission into 3D form.
  vec3 densityGradient(vec3 location) {
    return vec3(
      densityAt(location + vec3(uVoxelStep.x, 0.0, 0.0)) - densityAt(location - vec3(uVoxelStep.x, 0.0, 0.0)),
      densityAt(location + vec3(0.0, uVoxelStep.y, 0.0)) - densityAt(location - vec3(0.0, uVoxelStep.y, 0.0)),
      densityAt(location + vec3(0.0, 0.0, uVoxelStep.z)) - densityAt(location - vec3(0.0, 0.0, uVoxelStep.z))
    );
  }

  // Anisotropy-correct Blinn-Phong shading on the fused emission, so fluorescence
  // structures read as 3D surfaces instead of a flat translucent fog.
  vec3 applyLighting(vec3 location, vec3 color, vec3 gradient) {
    if (!uLightingEnabled || uLightingStrength <= 0.0) {
      return color;
    }
    vec3 worldGradient = gradient / max(vec3(0.06), uVoxelSpacing);
    if (length(worldGradient) < 0.0001) {
      return color;
    }
    vec3 normal = normalize(worldGradient);
    vec3 lightDir = normalize(vec3(-0.45, 0.55, 0.72));
    vec3 viewDir = uOrthographicCamera
      ? -normalize(uCameraDirectionLocal)
      : normalize(uCameraPositionLocal - (location - vec3(0.5)));
    vec3 halfDir = normalize(lightDir + viewDir);
    float diffuse = max(dot(normal, lightDir), dot(-normal, lightDir));
    float specular = pow(max(dot(normal, halfDir), max(dot(-normal, halfDir), 0.0)), 24.0);
    float shade = clamp(0.5 + 0.7 * diffuse, 0.4, 1.3);
    vec3 lit = color * shade + vec3(0.22 * specular);
    return mix(color, lit, uLightingStrength);
  }

  float alphaFromOpacity(float opacityValue, float stepLength) {
    float baseAlpha = clamp(opacityValue * uDensity * uDensityScale, 0.0, 0.95);
    float stepScale = max(0.001, stepLength * 128.0);
    return 1.0 - pow(1.0 - baseAlpha, stepScale);
  }

  vec3 applyTone(vec3 c) {
    c *= max(0.0, uBrightness);
    float range = max(0.0001, uGammaMax - uGammaMin);
    c = clamp((c - uGammaMin) / range, 0.0, 1.0);
    return pow(c, vec3(max(0.0001, uGammaScale)));
  }

  void main() {
    vec3 rayDir = uOrthographicCamera ? normalize(uCameraDirectionLocal) : normalize(vPosition - uCameraPositionLocal);
    vec3 rayOrigin = uOrthographicCamera ? vPosition - rayDir * 2.0 : uCameraPositionLocal;
    vec3 boxMin = uClipMin - vec3(0.5);
    vec3 boxMax = uClipMax - vec3(0.5);
    float tNear = 0.0;
    float tFar = 0.0;
    if (!intersectBox(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
      discard;
    }
    int steps = min(uSteps, ${MAX_STEPS});
    if (steps < 1 || uChannelCount < 1) {
      discard;
    }
    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    vec3 delta = (back - front) / float(steps);
    vec3 location = front + vec3(0.5);
    // Dither the ray start by a per-pixel fraction of one step. Coherent (un-jittered)
    // sampling of a dense volume at interactive step counts produces wood-grain streak
    // artifacts; a stable per-pixel offset (no time term, so it never flickers) trades
    // those streaks for fine, unobjectionable noise. Homogeneous volumes are unaffected
    // (every sample along the ray is identical), so deterministic readback tests hold.
    location += delta * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float stepLen = length(delta * uVolumeScale);

    vec4 accum = vec4(0.0);
    // Per-channel maxima for MIP. MIP must be computed INDEPENDENTLY per channel —
    // taking the single max-density voxel per ray (and its fused color) collapses a
    // multichannel volume to whichever channel is densest, hiding the sparse nuclei.
    float chanMax[${MAX_VOLUME_CHANNELS}];
    for (int c = 0; c < ${MAX_VOLUME_CHANNELS}; c++) {
      chanMax[c] = 0.0;
    }
    for (int iter = 0; iter < ${MAX_STEPS}; iter++) {
      if (iter >= steps) {
        break;
      }
      if (uProjectionMode == 1) {
        vec3 pp = clamp(location, vec3(0.0), vec3(1.0));
        for (int c = 0; c < ${MAX_VOLUME_CHANNELS}; c++) {
          if (c >= uChannelCount) break;
          chanMax[c] = max(chanMax[c], windowChannel(c, sampleChannelRaw(c, pp)));
        }
        location += delta;
        continue;
      }
      vec4 fused = fuseVoxel(location);
      // Suppress background: only intensity above the signal floor becomes opaque,
      // so the volume box stops fogging up into a flat haze.
      float opacity = smoothstep(uSignalFloor, 1.0, fused.a);
      if (opacity <= 0.0) {
        location += delta;
        continue;
      }
      vec3 lit = applyLighting(location, fused.rgb, densityGradient(location));
      float alpha = alphaFromOpacity(opacity, stepLen);
      accum.rgb += (1.0 - accum.a) * lit * alpha;
      accum.a += (1.0 - accum.a) * alpha;
      if (accum.a >= 0.985) {
        break;
      }
      location += delta;
    }

    if (uProjectionMode == 1) {
      // Combine each channel's max in its own color (additive, like the 2D fuse),
      // so every channel's brightest structures show — not just the densest one.
      vec3 emission = vec3(0.0);
      float maxDensity = 0.0;
      for (int c = 0; c < ${MAX_VOLUME_CHANNELS}; c++) {
        if (c >= uChannelCount) break;
        emission += uChanColor[c] * chanMax[c];
        maxDensity = max(maxDensity, chanMax[c]);
      }
      emission = min(emission, vec3(1.0));
      float maxOpacity = smoothstep(uSignalFloor, 1.0, maxDensity);
      if (maxOpacity < 0.02) {
        discard;
      }
      gl_FragColor = vec4(applyTone(emission), clamp(maxOpacity * uDensityScale * 1.2, 0.0, 1.0));
      return;
    }
    if (accum.a < 0.02) {
      discard;
    }
    gl_FragColor = vec4(applyTone(accum.rgb), accum.a);
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
  // Upload as normalized 16-bit half-float (R16F) instead of 8-bit so the brain's
  // narrow soft-tissue band keeps real contrast. R16F supports hardware linear
  // filtering natively in WebGL2, so window/level stays a cheap GPU uniform.
  const textureData = scalarVolumePayloadToHalfFloat(payload);
  const texture = new THREE.Data3DTexture(textureData, width, height, depth);
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.unpackAlignment = 2;
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

export function computeVolumeCameraFit({
  volumeRadius,
  aspect = 1,
  safeInsets = DEFAULT_VOLUME_CAMERA_SAFE_INSETS,
}: {
  volumeRadius: number;
  aspect?: number;
  safeInsets?: VolumeCameraSafeInsets;
}): VolumeCameraFit {
  const radius = Math.max(0.25, Number.isFinite(volumeRadius) ? volumeRadius : 0.25);
  const safeTop = Math.max(0, Math.min(0.45, safeInsets.top ?? DEFAULT_VOLUME_CAMERA_SAFE_INSETS.top));
  const safeRight = Math.max(0, Math.min(0.45, safeInsets.right ?? DEFAULT_VOLUME_CAMERA_SAFE_INSETS.right));
  const safeBottom = Math.max(0, Math.min(0.45, safeInsets.bottom ?? DEFAULT_VOLUME_CAMERA_SAFE_INSETS.bottom));
  const safeLeft = Math.max(0, Math.min(0.45, safeInsets.left ?? DEFAULT_VOLUME_CAMERA_SAFE_INSETS.left));
  const safeHeight = Math.max(0.35, 1 - safeTop - safeBottom);
  const safeWidth = Math.max(0.35, 1 - safeLeft - safeRight);
  const safeAspect = Math.max(0.1, Number.isFinite(aspect) ? aspect : 1);
  const verticalFov = THREE.MathUtils.degToRad(DEFAULT_VOLUME_CAMERA_FOV);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * safeAspect);
  const safeVerticalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * safeHeight);
  const safeHorizontalFov = 2 * Math.atan(Math.tan(horizontalFov / 2) * safeWidth);
  const verticalDistance = radius / Math.max(0.05, Math.sin(safeVerticalFov / 2));
  const horizontalDistance = radius / Math.max(0.05, Math.sin(safeHorizontalFov / 2));
  const distance = Math.max(2.4, Math.max(verticalDistance, horizontalDistance) * VOLUME_CAMERA_FIT_MARGIN);
  const minDistance = Math.max(radius * 2.72, distance / VOLUME_CAMERA_FIT_MARGIN);
  const insideMinDistance = Math.max(0.025, radius * VOLUME_INTERIOR_MIN_DISTANCE_FACTOR);
  const maxDistance = Math.max(distance * 4, radius * 10, 6);
  const orthographicHeightForVerticalFit = (radius * 2 * VOLUME_CAMERA_FIT_MARGIN) / safeHeight;
  const orthographicHeightForHorizontalFit =
    (radius * 2 * VOLUME_CAMERA_FIT_MARGIN) / Math.max(0.35, safeWidth * safeAspect);
  const orthographicFrustumHeight = Math.max(
    MIN_ORTHOGRAPHIC_FRUSTUM_HEIGHT,
    orthographicHeightForVerticalFit,
    orthographicHeightForHorizontalFit
  );
  return {
    distance,
    minDistance,
    insideMinDistance,
    maxDistance,
    inspectMaxZoom: VOLUME_INTERIOR_MAX_ZOOM,
    orthographicFrustumHeight,
    safeWidth,
    safeHeight,
  };
}

const computeOrthographicFrustumHeight = (volumeRadius: number, aspect = 1): number =>
  computeVolumeCameraFit({ volumeRadius, aspect }).orthographicFrustumHeight;

const createVolumeCamera = ({
  mode,
  volumeRadius,
}: {
  mode: VolumeCameraMode;
  volumeRadius: number;
}): VolumeCamera => {
  if (!mode.isOrthographic) {
    return new THREE.PerspectiveCamera(DEFAULT_VOLUME_CAMERA_FOV, 1, 0.01, 100);
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
  const fit = computeVolumeCameraFit({ volumeRadius, aspect });
  if (camera instanceof THREE.OrthographicCamera) {
    const frustumHeight = fit.orthographicFrustumHeight;
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
  aspect = 1,
  interiorFrame = null,
}: {
  camera: VolumeCamera;
  controls: TrackballControls;
  preset: VolumeViewPreset;
  volumeRadius: number;
  aspect?: number;
  interiorFrame?: VolumeInteriorCameraFrame | null;
}) => {
  const direction = new THREE.Vector3(preset.direction.x, preset.direction.y, preset.direction.z).normalize();
  const fit = computeVolumeCameraFit({ volumeRadius, aspect });
  if (interiorFrame && !(camera instanceof THREE.OrthographicCamera)) {
    camera.position.set(interiorFrame.position.x, interiorFrame.position.y, interiorFrame.position.z);
    controls.target.set(interiorFrame.target.x, interiorFrame.target.y, interiorFrame.target.z);
  } else {
    camera.position.copy(direction.multiplyScalar(fit.distance));
    controls.target.set(0, 0, 0);
  }
  camera.up.set(preset.up.x, preset.up.y, preset.up.z);
  if (camera instanceof THREE.OrthographicCamera) {
    camera.zoom = 1;
    camera.updateProjectionMatrix();
  }
  controls.minDistance = fit.insideMinDistance;
  controls.maxDistance = fit.maxDistance;
  controls.minZoom = 0.2;
  controls.maxZoom = fit.inspectMaxZoom;
  camera.lookAt(controls.target);
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
  showPlanes,
}: {
  planes: SliceCursorPlanes;
  cue: VolumeSliceCursorCue;
  normalizedScale: { x: number; y: number; z: number };
  showPlanes: boolean;
}) => {
  planes.x.visible = showPlanes && cue.x.count > 1;
  planes.y.visible = showPlanes && cue.y.count > 1;
  planes.z.visible = showPlanes && cue.z.count > 1;

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
  const canvasHostRef = useRef<HTMLDivElement | null>(null);
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
  const cutFaceRef = useRef<{ mesh: THREE.Mesh; material: THREE.ShaderMaterial } | null>(null);
  const scalarRangeRef = useRef<{ rawMin: number; rawMax: number } | null>(null);
  // Live multichannel uniforms (the THREE uniform objects) so per-channel
  // color/window + gamma update without rebuilding the renderer.
  const multichannelUniformsRef = useRef<MultichannelUniformSet | null>(null);
  const multichannelRenderConfigRef = useRef<MultichannelRenderConfig | null>(null);
  // Per-channel ImageJ-style auto-contrast windows (normalized [0,1]), computed
  // from each loaded volume; used as the default window when the user has not set
  // a manual one, so the background maps to zero opacity (no fog).
  const channelAutoWindowsRef = useRef<Map<number, { low: number; high: number }>>(new Map());
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
  // Max-normalized voxel SPACING ratio (sx:sy:sz) for the lighting gradient —
  // distinct from normalizedScale (the box extent = count x spacing).
  const voxelSpacingRatio = useMemo(() => {
    const s = physicalGeometry.voxelSpacing;
    const maxS = Math.max(s.x, s.y, s.z, 1e-9);
    return { x: s.x / maxS, y: s.y / maxS, z: s.z / maxS };
  }, [physicalGeometry.voxelSpacing]);
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

  // The set of channels the multichannel volume composites: the user's enabled
  // channels (display_defaults.channels), de-duped, capped, defaulting to the
  // single scalar channel. Changing this set reloads the per-channel textures.
  const enabledVolumeChannels = useMemo(() => {
    const raw = Array.isArray(displayState?.channels) ? displayState.channels : [];
    const cleaned = raw
      .filter((value) => Number.isFinite(value))
      .map((value) => Math.max(0, Math.floor(Number(value))))
      .filter((value, index, list) => list.indexOf(value) === index)
      .sort((a, b) => a - b)
      .slice(0, MAX_VOLUME_CHANNELS);
    return cleaned.length > 0 ? cleaned : [Math.max(0, scalarChannel ?? 0)];
  }, [displayState, scalarChannel]);

  // Any NON-MEDICAL 3D volume (single- or multi-channel microscopy / scientific
  // scalar) renders through the full-res per-channel quality path — auto-contrast
  // + density-gradient shading + signal-floor opacity — rather than the downsampled
  // server-fused atlas. Medical volumes (volume_mode "scalar") deliberately stay on
  // the clinical scalar path so their Hounsfield/window presets are not discarded
  // by auto-contrast.
  const isPerChannelVolume = Boolean(
    apiClient &&
      fileId &&
      viewerInfo?.is_volume &&
      viewerInfo?.viewer.volume_mode !== "scalar" &&
      Boolean(viewerInfo?.viewer.available_surfaces?.includes("volume")) &&
      Boolean(viewerInfo?.viewer.service_urls?.scalar_volume ?? viewerInfo?.service_urls?.scalar_volume)
  );

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
    // Multichannel fluorescence z-stack: composite the enabled channels' full-res
    // volumes in-shader (vole-core-style fuse-then-raymarch). Takes priority over
    // the server-fused atlas fallback. Identity changes when the enabled set or
    // time changes (reload); per-channel color/window are applied incrementally.
    if (isPerChannelVolume && apiClient && fileId) {
      return {
        kind: "multichannel" as const,
        channelIndices: enabledVolumeChannels,
        loadChannel: (channel: number) => apiClient.getUploadScalarVolume(fileId, { t: tIndex, channel }),
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
    enabledVolumeChannels,
    isPerChannelVolume,
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
  // Z-cursor cutaway: the volume is cut at the live Z slice so the interior is
  // exposed with the camera kept in overview. The cut position is derived from
  // the slice cursor, so scrubbing Z sweeps the cut through the volume.
  const cutawayActive = Boolean(displayState?.volume_cutaway);
  const cutawayZ = useMemo(() => resolveVolumeCutawayCutZ(zIndex, axisSizes.Z), [axisSizes.Z, zIndex]);
  // Manual box clip only (the legacy "Advanced cutaway" sliders). The Z-cursor
  // cutaway is applied separately via `effectiveClipBounds` so scrubbing Z does
  // NOT change this memo — keeping the renderer effect (which lists clipBounds in
  // its deps to rebuild the clip-edge box) from tearing down on every Z step.
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
  // The clip the shader actually applies: in cutaway mode the volume is sliced at
  // the live Z (overview camera); otherwise it's the manual box clip. Updated
  // incrementally through the clip effect, never by rebuilding the renderer.
  const effectiveClipBounds = useMemo(() => {
    if (cutawayActive) {
      return resolveVolumeCutawayClip(cutawayZ);
    }
    return clipBounds;
  }, [cutawayActive, cutawayZ, clipBounds]);
  // Live cutaway state mirrored into refs so the renderer effect can seed the
  // clip uniforms and cut-face mesh at creation WITHOUT listing them in its deps
  // (which would rebuild the WebGL context on every Z step). The clip effect and
  // the dedicated cut-face effect keep these in sync after creation.
  const effectiveClipBoundsRef = useRef(effectiveClipBounds);
  const cutawayActiveRef = useRef(cutawayActive);
  const cutawayZRef = useRef(cutawayZ);
  // 3D ray-projection. `volume_projection` is the dedicated control (decoupled from
  // the 2D `fusion_method`); when the user has set it, it wins. Otherwise pick a
  // per-source default: multichannel fluorescence z-stacks are dense and space-
  // filling, so a MIP maxes every ray to its brightest voxel and flattens the whole
  // stack into a uniform cloud — composite (front-to-back, occluding) reads as a
  // coherent 3D volume, so default per-channel volumes to composite. Scalar volumes
  // keep the existing modality/fusion-derived default.
  const explicitVolumeProjection =
    displayState?.volume_projection === "mip" || displayState?.volume_projection === "composite"
      ? displayState.volume_projection
      : null;
  const projectionMode =
    explicitVolumeProjection ??
    (isPerChannelVolume
      ? "composite"
      : resolveVolumeProjectionMode({
          renderPolicy,
          modality,
          fusionMethod: displayState?.fusion_method,
        }));
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
  // The cutaway keeps the camera in overview — the fly-inside interior camera is
  // only for the legacy manual box clip.
  const volumeInteriorInspectionActive =
    !cutawayActive &&
    isVolumeInteriorInspectionActive({
      clipActive: volumeClipCue.active,
      cameraMode: volumeCameraMode,
    });
  const volumeInteriorCameraFrame = useMemo(
    () =>
      volumeInteriorInspectionActive
        ? computeVolumeInteriorCameraFrame({
            clipBounds,
            normalizedScale,
            preset: volumeViewPreset,
            volumeRadius,
          })
        : null,
    [
      clipBounds,
      normalizedScale,
      volumeInteriorInspectionActive,
      volumeRadius,
      volumeViewPreset,
    ]
  );
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
  const density = isPerChannelVolume
    ? // Per-channel fluorescence: MIP ignores uDensity (its opacity is the windowed
      // max), so this only matters for composite. The sample is dense and space-
      // filling, so a low density integrates a see-through haze; a higher density
      // makes the front layers occlude into a solid, depth-ordered volume. The
      // Density slider (uDensityScale) tunes around this base.
      projectionMode === "mip"
      ? 0.9
      : 1.1
    : renderPolicy === "scalar"
      ? projectionMode === "mip"
        ? 0.9
        : modality === "medical"
          ? 0.5
          : 0.34
      : 0.22;
  // Boundary-emphasis transfer function (composite scalar volumes only). A higher
  // edge strength + lower interior opacity reveals internal tissue interfaces such
  // as ventricle walls; medical data leans harder on this than generic scalars.
  const volumeEdgeStrength =
    renderPolicy === "scalar" ? (modality === "medical" ? 7.0 : 3.5) : 0.0;
  const volumeInteriorOpacity =
    renderPolicy === "scalar" ? (modality === "medical" ? 0.14 : 0.5) : 1.0;
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
        // Multichannel ray-marches R16F 3D textures like the scalar path, so it
        // shares the scalar (higher) sample budget rather than the atlas one.
        sourceKind: resolvedSource?.kind === "atlas" ? "atlas" : "scalar",
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
    effectiveClipBoundsRef.current = effectiveClipBounds;
    cutawayActiveRef.current = cutawayActive;
    cutawayZRef.current = cutawayZ;
  }, [effectiveClipBounds, cutawayActive, cutawayZ]);

  useEffect(() => {
    scalarRenderConfigRef.current = scalarRenderConfig;
  }, [scalarRenderConfig]);

  const multichannelRenderConfig = useMemo<MultichannelRenderConfig>(
    () => ({
      channels: enabledVolumeChannels.map((index) => {
        const hex =
          displayState?.channel_colors?.[index] ??
          viewerInfo?.phys?.channel_colors?.[index]?.hex ??
          "#ffffff";
        const win = displayState?.volume_channel_windows?.[index];
        let manualWindow: { low: number; high: number } | null = null;
        if (win && (Number.isFinite(win.low) || Number.isFinite(win.high))) {
          const low = Math.min(1, Math.max(0, Number(win.low ?? 0)));
          const high = Math.min(1, Math.max(0, Number(win.high ?? 1)));
          manualWindow = { low, high: high > low ? high : Math.min(1, low + 0.0001) };
        }
        return {
          index,
          color: hexToLinearRgb(hex),
          manualWindow,
          invert: Boolean(displayState?.negative),
        };
      }),
      gammaScale: (() => {
        const gamma = Number(displayState?.volume_gamma);
        return Number.isFinite(gamma) && gamma > 0 ? gamma : 1;
      })(),
      brightness: 1,
      densityScale: scalarTransfer.densityScale,
    }),
    [enabledVolumeChannels, displayState, viewerInfo?.phys?.channel_colors, scalarTransfer.densityScale]
  );

  // Keep the latest config in a ref (read at renderer-creation) and push
  // per-channel color/window/gamma into the live uniforms with no rebuild.
  useEffect(() => {
    multichannelRenderConfigRef.current = multichannelRenderConfig;
    const uniforms = multichannelUniformsRef.current;
    if (!uniforms) {
      return;
    }
    applyMultichannelChannelUniforms(uniforms, multichannelRenderConfig, channelAutoWindowsRef.current);
    requestRenderRef.current?.();
  }, [multichannelRenderConfig]);

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
    // Keep the cutaway cut face on the same window/level + colormap + invert as
    // the volume so the exposed cross-section reads identically to the body.
    const cutFace = cutFaceRef.current;
    if (cutFace) {
      const cutUniforms = cutFace.material.uniforms as Record<string, { value: number | boolean }>;
      cutUniforms.uWindowLow.value = normalizedWindow.low;
      cutUniforms.uWindowHigh.value = normalizedWindow.high;
      cutUniforms.uInvert.value = scalarRenderConfig.negative;
      cutUniforms.uColorMap.value = scalarRenderConfig.colorMapShaderValue;
    }
    requestRenderRef.current?.();
  }, [scalarRenderConfig]);

  useEffect(() => {
    const clipUniforms = clipUniformsRef.current;
    if (!clipUniforms) {
      return;
    }
    clipUniforms.uClipMin.value.set(
      effectiveClipBounds.min.x,
      effectiveClipBounds.min.y,
      effectiveClipBounds.min.z
    );
    clipUniforms.uClipMax.value.set(
      effectiveClipBounds.max.x,
      effectiveClipBounds.max.y,
      effectiveClipBounds.max.z
    );
    requestRenderRef.current?.();
  }, [effectiveClipBounds]);

  // Drive the high-resolution cut face: toggle visibility with the cutaway, and
  // slide + re-sample it as the user scrubs Z so the exposed cross-section
  // tracks the live slice cursor.
  useEffect(() => {
    const cutFace = cutFaceRef.current;
    if (!cutFace) {
      return;
    }
    cutFace.mesh.visible = cutawayActive;
    cutFace.mesh.scale.set(normalizedScale.x, normalizedScale.y, 1);
    cutFace.mesh.position.set(0, 0, (cutawayZ - 0.5) * normalizedScale.z);
    (cutFace.material.uniforms.uCutZ as { value: number }).value = cutawayZ;
    requestRenderRef.current?.();
  }, [cutawayActive, cutawayZ, normalizedScale.x, normalizedScale.y, normalizedScale.z]);

  useEffect(() => {
    const rig = cameraRigRef.current;
    if (!rig) {
      return;
    }
    const width = Math.max(1, containerRef.current?.clientWidth || 1);
    const height = Math.max(1, containerRef.current?.clientHeight || 1);
    applyVolumeCameraPreset({
      camera: rig.camera,
      controls: rig.controls,
      preset: volumeViewPreset,
      volumeRadius,
      aspect: width / height,
      interiorFrame: volumeInteriorCameraFrame,
    });
    requestRenderRef.current?.();
  }, [volumeInteriorCameraFrame, volumeRadius, volumeViewPreset]);

  useEffect(() => {
    const container = containerRef.current;
    const canvasHost = canvasHostRef.current;
    if (!container || !canvasHost || !resolvedSource) {
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
    canvasHost.appendChild(renderer.domElement);

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
    cameraRigRef.current = { camera, controls };
    const initialWidth = Math.max(1, container.clientWidth || 1);
    const initialHeight = Math.max(1, container.clientHeight || 1);
    applyVolumeCameraPreset({
      camera,
      controls,
      preset: volumeViewPreset,
      volumeRadius,
      aspect: initialWidth / initialHeight,
      interiorFrame: volumeInteriorCameraFrame,
    });

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    // Seed the clip uniforms with the cutaway-aware clip so the first frame is
    // already sliced; the clip effect keeps them live as Z scrubs without a
    // renderer rebuild.
    const initialClip = effectiveClipBoundsRef.current;
    const mcConfig = multichannelRenderConfigRef.current;
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
              uVolumeScale: { value: new THREE.Vector3(normalizedScale.x, normalizedScale.y, normalizedScale.z) },
              uVoxelSpacing: { value: new THREE.Vector3(voxelSpacingRatio.x, voxelSpacingRatio.y, voxelSpacingRatio.z) },
              uEdgeStrength: { value: volumeEdgeStrength },
              uInteriorOpacity: { value: volumeInteriorOpacity },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(initialClip.min.x, initialClip.min.y, initialClip.min.z) },
              uClipMax: { value: new THREE.Vector3(initialClip.max.x, initialClip.max.y, initialClip.max.z) },
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
        : resolvedSource.kind === "multichannel"
        ? new THREE.ShaderMaterial({
            uniforms: {
              uChan0: { value: null },
              uChan1: { value: null },
              uChan2: { value: null },
              uChan3: { value: null },
              uChan4: { value: null },
              uChan5: { value: null },
              uChan6: { value: null },
              uChan7: { value: null },
              // uChannelCount stays 0 until the load step binds the textures, so
              // the shader never samples an unbound slot.
              uChannelCount: { value: 0 },
              // Seed manual windows; the load step overwrites with auto-contrast
              // (or manual) once each channel's data is available.
              uChanLow: { value: Array.from({ length: MAX_VOLUME_CHANNELS }, (_v, i) => mcConfig?.channels[i]?.manualWindow?.low ?? 0) },
              uChanHigh: { value: Array.from({ length: MAX_VOLUME_CHANNELS }, (_v, i) => mcConfig?.channels[i]?.manualWindow?.high ?? 1) },
              uChanColor: {
                value: Array.from({ length: MAX_VOLUME_CHANNELS }, (_v, i) => {
                  const color = mcConfig?.channels[i]?.color;
                  return new THREE.Vector3(color?.[0] ?? 0, color?.[1] ?? 0, color?.[2] ?? 0);
                }),
              },
              uChanInvert: { value: Array.from({ length: MAX_VOLUME_CHANNELS }, (_v, i) => mcConfig?.channels[i]?.invert ?? false) },
              uSteps: { value: sampleBudget.interactiveSteps },
              uDensity: { value: density },
              uDensityScale: { value: mcConfig?.densityScale ?? scalarRenderConfigRef.current.densityScale },
              uVolumeScale: { value: new THREE.Vector3(normalizedScale.x, normalizedScale.y, normalizedScale.z) },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(initialClip.min.x, initialClip.min.y, initialClip.min.z) },
              uClipMax: { value: new THREE.Vector3(initialClip.max.x, initialClip.max.y, initialClip.max.z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
              uCameraDirectionLocal: { value: new THREE.Vector3(0, 0, -1) },
              uOrthographicCamera: { value: volumeCameraMode.isOrthographic },
              uGammaMin: { value: 0 },
              uGammaMax: { value: 1 },
              uGammaScale: { value: mcConfig?.gammaScale ?? 1 },
              uBrightness: { value: mcConfig?.brightness ?? 1 },
              uSignalFloor: { value: 0.08 },
              uLightingEnabled: { value: true },
              uLightingStrength: { value: 0.55 },
              uVoxelStep: { value: new THREE.Vector3(scalarVoxelStep.x, scalarVoxelStep.y, scalarVoxelStep.z) },
              uVoxelSpacing: { value: new THREE.Vector3(voxelSpacingRatio.x, voxelSpacingRatio.y, voxelSpacingRatio.z) },
            },
            vertexShader: VERTEX_SHADER,
            fragmentShader: MULTICHANNEL_FRAGMENT_SHADER,
            side: THREE.DoubleSide,
            transparent: true,
            depthWrite: false,
          })
        : new THREE.ShaderMaterial({
            uniforms: {
              uData: { value: null },
              uSteps: { value: sampleBudget.interactiveSteps },
              uDensity: { value: density },
              uClipMin: { value: new THREE.Vector3(initialClip.min.x, initialClip.min.y, initialClip.min.z) },
              uClipMax: { value: new THREE.Vector3(initialClip.max.x, initialClip.max.y, initialClip.max.z) },
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
    edgeLines.visible = shouldShowVolumeContextEdges({
      cueVisible: true,
      interiorInspectionActive: volumeInteriorInspectionActive,
    });
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
    clipEdgeLines.visible = shouldShowVolumeContextEdges({
      cueVisible: volumeClipCue.active,
      interiorInspectionActive: volumeInteriorInspectionActive,
    });
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
        showPlanes: shouldShowVolumeSliceCursorPlanes({
          cueVisible: initialSliceCursorCue.visible,
          interiorInspectionActive: volumeInteriorInspectionActive,
          // Hide the flat cursor quads for the multichannel volume too — they wash
          // out the fluorescence render (the light cross-bands).
          cutawayActive: cutawayActiveRef.current || resolvedSource.kind === "multichannel",
        }),
      });
    }
    sliceCursorPlanesRef.current = sliceCursorPlanes;
    scene.add(sliceCursorPlanes.x, sliceCursorPlanes.y, sliceCursorPlanes.z);

    // High-resolution cut face for the Z-cursor cutaway (scalar volumes only).
    // It samples the same R16F texture as the ray-marcher but renders a single
    // crisp cross-section at the cut depth. Drawn opaque and on top (depthTest
    // off, like the cursor planes) so the exposed interior reads at full slice
    // resolution; the clipped volume behind it supplies 3D context.
    let cutFaceGeometry: THREE.PlaneGeometry | null = null;
    let cutFaceMaterial: THREE.ShaderMaterial | null = null;
    if (resolvedSource.kind === "scalar") {
      const initialCutZ = cutawayZRef.current;
      cutFaceGeometry = new THREE.PlaneGeometry(1, 1);
      cutFaceMaterial = new THREE.ShaderMaterial({
        uniforms: {
          uData: { value: null },
          uCutZ: { value: initialCutZ },
          uWindowLow: { value: 0.0 },
          uWindowHigh: { value: 1.0 },
          uInvert: { value: scalarRenderConfigRef.current.negative },
          uColorMap: { value: scalarRenderConfigRef.current.colorMapShaderValue },
        },
        vertexShader: CUTFACE_VERTEX_SHADER,
        fragmentShader: CUTFACE_FRAGMENT_SHADER,
        side: THREE.DoubleSide,
        // transparent:true so the cut face joins the SAME render bucket as the
        // translucent volume + cursor planes. THREE always draws the whole opaque
        // bucket before the transparent bucket and renderOrder only sorts within a
        // bucket — so an opaque cut face would be painted FIRST and then hazed over
        // by the volume and tinted by the cursor planes. In the transparent bucket
        // with the highest renderOrder it draws LAST; its fragments output alpha=1,
        // so it still fully occludes (src*1 + dst*0 = src) within its footprint.
        transparent: true,
        depthTest: false,
        depthWrite: false,
      });
      const cutFaceMesh = new THREE.Mesh(cutFaceGeometry, cutFaceMaterial);
      // Above the volume (0), context edges (2), clip edges (3), cursor planes (4).
      cutFaceMesh.renderOrder = 6;
      cutFaceMesh.scale.set(normalizedScale.x, normalizedScale.y, 1);
      cutFaceMesh.position.set(0, 0, (initialCutZ - 0.5) * normalizedScale.z);
      cutFaceMesh.visible = cutawayActiveRef.current;
      cutFaceRef.current = { mesh: cutFaceMesh, material: cutFaceMaterial };
      scene.add(cutFaceMesh);
    }

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
    if (resolvedSource.kind === "multichannel") {
      // Point the incremental config effect at this material's live uniforms.
      multichannelUniformsRef.current = material.uniforms as unknown as MultichannelUniformSet;
    }

    const resize = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      const aspect = width / height;
      const fit = computeVolumeCameraFit({ volumeRadius, aspect });
      configureVolumeCameraProjection({ camera, width, height, volumeRadius });
      controls.minDistance = fit.insideMinDistance;
      controls.maxDistance = fit.maxDistance;
      controls.minZoom = 0.2;
      controls.maxZoom = fit.inspectMaxZoom;
      if (camera instanceof THREE.OrthographicCamera && camera.zoom > controls.maxZoom) {
        camera.zoom = controls.maxZoom;
        camera.updateProjectionMatrix();
      }
      if (
        !volumeInteriorInspectionActive &&
        !(camera instanceof THREE.OrthographicCamera) &&
        camera.position.length() < fit.insideMinDistance
      ) {
        camera.position.normalize().multiplyScalar(fit.insideMinDistance);
      }
      camera.lookAt(controls.target);
      controls.handleResize();
      render();
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);

    let texture3D: THREE.Data3DTexture | null = null;
    const channelTextures: THREE.Data3DTexture[] = [];
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

    if (resolvedSource.kind === "multichannel") {
      // Load each enabled channel's full-res volume (cached so a channel toggle
      // reuses already-fetched channels), upload as its own R16F 3D texture, bind
      // to uChan0..N, then publish uChannelCount last so the shader only ever reads
      // bound slots. Per-channel color/window/gamma come from the config.
      const samplerNames = ["uChan0", "uChan1", "uChan2", "uChan3", "uChan4", "uChan5", "uChan6", "uChan7"];
      const channelIndices = resolvedSource.channelIndices.slice(0, MAX_VOLUME_CHANNELS);
      const multichannelSource = resolvedSource;
      const loadOne = async (channel: number): Promise<ScalarVolumePayload> => {
        const key = channelVolumeCacheKey(fileId ?? "", channel, tIndex ?? 0);
        const cached = channelVolumeCache.get(key);
        if (cached) {
          return cached;
        }
        const payload = await multichannelSource.loadChannel(channel);
        rememberChannelVolume(key, payload);
        return payload;
      };
      void Promise.all(
        channelIndices.map(async (channel, slot) => {
          const payload = await loadOne(channel);
          // ImageJ-style auto-contrast from the actual data so background is
          // transparent (used as the default window when no manual one is set).
          channelAutoWindowsRef.current.set(channel, computeScalarVolumeAutoContrast(payload));
          const texture = await scalarToVolumeTexture(payload, texturePolicy);
          return { slot, texture };
        })
      )
        .then((loaded) => {
          if (disposed) {
            loaded.forEach(({ texture }) => texture.dispose());
            return;
          }
          const uniforms = material.uniforms as Record<string, { value: unknown }>;
          loaded
            .sort((a, b) => a.slot - b.slot)
            .forEach(({ slot, texture }) => {
              if (typeof renderer.initTexture === "function") {
                renderer.initTexture(texture);
              }
              channelTextures[slot] = texture;
              uniforms[samplerNames[slot]].value = texture;
            });
          const config = multichannelRenderConfigRef.current;
          if (config) {
            applyMultichannelChannelUniforms(
              material.uniforms as unknown as MultichannelUniformSet,
              config,
              channelAutoWindowsRef.current
            );
          }
          (uniforms.uChannelCount as { value: number }).value = channelTextures.filter(Boolean).length;
          material.needsUpdate = true;
          resize();
        })
        .catch((error: unknown) => {
          if (disposed) {
            return;
          }
          setRenderError(error instanceof Error ? error.message : "Volume channels failed to load");
        });
    } else {
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
            if (cutFaceMaterial) {
              const cutUniforms = cutFaceMaterial.uniforms as Record<string, { value: number | boolean }>;
              cutUniforms.uWindowLow.value = normalizedWindow.low;
              cutUniforms.uWindowHigh.value = normalizedWindow.high;
              cutUniforms.uInvert.value = latestScalarRenderConfig.negative;
              cutUniforms.uColorMap.value = latestScalarRenderConfig.colorMapShaderValue;
            }
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
        if (cutFaceMaterial) {
          cutFaceMaterial.uniforms.uData.value = decodedTexture;
          cutFaceMaterial.needsUpdate = true;
        }
        resize();
      })
      .catch((error: unknown) => {
        if (disposed) {
          return;
        }
        setRenderError(error instanceof Error ? error.message : "Volume data failed to load");
      });
    }

    resize();
    animate();

    return () => {
      disposed = true;
      requestRenderRef.current = null;
      scalarUniformsRef.current = null;
      multichannelUniformsRef.current = null;
      clipUniformsRef.current = null;
      cameraRigRef.current = null;
      sliceCursorPlanesRef.current = null;
      cutFaceRef.current = null;
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
      cutFaceGeometry?.dispose();
      cutFaceMaterial?.dispose();
      material.dispose();
      texture3D?.dispose();
      channelTextures.forEach((texture) => texture?.dispose());
      renderer.dispose();
      renderer.domElement.parentNode?.removeChild(renderer.domElement);
    };
  }, [
    renderError,
    resolvedSource,
    texturePolicy,
    clearColor,
    density,
    volumeEdgeStrength,
    volumeInteriorOpacity,
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
    voxelSpacingRatio,
    volumeDepth,
    volumeRadius,
    volumeCameraMode,
    volumeViewPreset,
    volumeInteriorCameraFrame,
    volumeInteriorInspectionActive,
    physicalGeometry.worldDepth,
    physicalGeometry.worldHeight,
    physicalGeometry.worldWidth,
    volumeClipCue.active,
    fileId,
    tIndex,
    isPerChannelVolume,
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
      showPlanes: shouldShowVolumeSliceCursorPlanes({
        cueVisible: sliceCursorCue.visible,
        interiorInspectionActive: volumeInteriorInspectionActive,
        cutawayActive: cutawayActive || isPerChannelVolume,
      }),
    });
    requestRenderRef.current?.();
  }, [normalizedScale, sliceCursorCue, volumeInteriorInspectionActive, cutawayActive, isPerChannelVolume]);

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
        data-viewer-interior-inspection={volumeInteriorInspectionActive ? "true" : "false"}
        data-viewer-interior-camera={volumeInteriorCameraFrame ? "center" : "overview"}
        data-viewer-interior-camera-x={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.x.toFixed(4) : undefined}
        data-viewer-interior-camera-y={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.y.toFixed(4) : undefined}
        data-viewer-interior-camera-z={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.z.toFixed(4) : undefined}
        data-viewer-slice-cursor-planes={
          shouldShowVolumeSliceCursorPlanes({
            cueVisible: sliceCursorCue.visible,
            interiorInspectionActive: volumeInteriorInspectionActive,
            cutawayActive: cutawayActive || isPerChannelVolume,
          })
            ? "true"
            : "false"
        }
        data-viewer-context-edges={
          shouldShowVolumeContextEdges({
            cueVisible: true,
            interiorInspectionActive: volumeInteriorInspectionActive,
          })
            ? "true"
            : "false"
        }
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
      data-viewer-interior-inspection={volumeInteriorInspectionActive ? "true" : "false"}
      data-viewer-interior-camera={volumeInteriorCameraFrame ? "center" : "overview"}
      data-viewer-interior-camera-x={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.x.toFixed(4) : undefined}
      data-viewer-interior-camera-y={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.y.toFixed(4) : undefined}
      data-viewer-interior-camera-z={volumeInteriorCameraFrame ? volumeInteriorCameraFrame.position.z.toFixed(4) : undefined}
      data-viewer-slice-cursor-planes={
        shouldShowVolumeSliceCursorPlanes({
          cueVisible: sliceCursorCue.visible,
          interiorInspectionActive: volumeInteriorInspectionActive,
        })
          ? "true"
          : "false"
      }
      data-viewer-context-edges={
        shouldShowVolumeContextEdges({
          cueVisible: true,
          interiorInspectionActive: volumeInteriorInspectionActive,
        })
          ? "true"
          : "false"
      }
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
      <div ref={canvasHostRef} className="viewer-webgl-canvas-host" aria-hidden="true" />
      {renderVolumeOrientationOverlay()}
      {renderVolumeAxisCue()}
      {renderVolumeSliceCursorCue()}
      {renderVolumeClipCue()}
      {renderVolumeScaleBar()}
    </div>
  );
}
