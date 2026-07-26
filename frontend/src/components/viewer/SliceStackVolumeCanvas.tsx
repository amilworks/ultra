import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import type { ApiClient, ScalarVolumePayload } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

import {
  MAX_ACTIVE_SCALAR_VOLUME_GPU_BYTES,
  MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES,
  PreparedScalarVolumeResidencyManager,
  SCALAR_MASK_NATIVE_POLICY,
  prepareScalarVolume,
  preparedScalarVolumeByteLength,
  resolveScalarVolumeWindow,
  type PreparedScalarVolume,
  type PreparedScalarVolumeLease,
  type PreparedScalarVolumeReservation,
  type ScalarVolumePreparationEncoding,
  scalarVolumeApiNamespace,
  scalarVolumeIdentityKey,
  scalarVolumeSourceIdentity,
  validateExactMaskSourcePreflight,
  validateExactMaskVolume,
  validateScalarVolumeIdentity,
} from "./scalarVolume";
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
import type { ResolvedScalarRendering } from "./scalarMask";

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
  loadScalarVolume: (signal?: AbortSignal) => Promise<ScalarVolumePayload>;
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
  loadChannel: (channel: number, signal?: AbortSignal) => Promise<ScalarVolumePayload>;
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
  categoricalMode?: CategoricalVolumeMode;
  volumeCutaway?: boolean;
  featureMask?: boolean;
  cameraPersistenceKey?: string;
  resolvedScalarRendering?: ResolvedScalarRendering | null;
};

export type CategoricalVolumeMode = "surface" | "xray";

export type AtlasVolumeRenderMode = {
  id: CategoricalVolumeMode;
  shaderValue: 0 | 1;
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
  lightingEnabled: boolean;
  lightingStrength: number;
};

// Cache only prepared R16F data and its render metadata. Raw response buffers are
// intentionally not retained, so they become collectible after conversion.
const scalarVolumeResidency = new PreparedScalarVolumeResidencyManager();
const SCALAR_PREVIEW_POLICY = "auto-v1";

type MultichannelUniformSet = {
  uChannelCount: { value: number };
  uChanLow: { value: number[] };
  uChanHigh: { value: number[] };
  uChanColor: { value: THREE.Vector3[] };
  uChanInvert: { value: boolean[] };
  uGammaScale: { value: number };
  uBrightness: { value: number };
  uDensityScale: { value: number };
  uLightingEnabled: { value: boolean };
  uLightingStrength: { value: number };
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
  uniforms.uLightingEnabled.value = config.lightingEnabled;
  uniforms.uLightingStrength.value = config.lightingStrength;
};

export const MAX_STEPS = 512;
export const MAX_CATEGORICAL_SURFACE_STEPS = 768;
// Shared with imaging/atlas.py. A native Mask volume is admitted only when its
// worst-case X+Y+Z DDA crossings fit this compile-time shader bound.
export const MAX_MASK_SURFACE_STEPS = 2048;

export const resolveVolumePixelRatio = (
  devicePixelRatio: number,
  maskMode: boolean
): number => {
  const safeRatio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0
    ? devicePixelRatio
    : 1;
  return Math.min(safeRatio, maskMode ? 1 : 2);
};
export const CATEGORICAL_SURFACE_TIE_EPSILON = 1e-7;
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

type VolumeGrid = { width: number; height: number; depth: number };
type VolumeSpacing = { x: number; y: number; z: number };

export const resolveStrictVolumeIndex = (
  value: number,
  count: number,
  field: "channel" | "time" | "z"
): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value >= count) {
    throw new RangeError(`${field} index ${String(value)} is out of range for ${field.toUpperCase()}=${count}.`);
  }
  return value;
};

export const classifyMaskDdaAxisStep = (
  direction: number,
  epsilon = 0.0000001
): -1 | 0 | 1 =>
  Math.abs(direction) <= epsilon ? 0 : direction < 0 ? -1 : 1;

export const validateVolumeTextureGrid = (grid: VolumeGrid, max3DTextureSize: number): void => {
  const dimensions = [grid.width, grid.height, grid.depth];
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError("Delivered volume texture grid must contain positive safe integers.");
  }
  const largestDimension = Math.max(...dimensions);
  if (
    Number.isFinite(max3DTextureSize) &&
    max3DTextureSize > 0 &&
    largestDimension > max3DTextureSize
  ) {
    throw new RangeError(
      `Delivered volume exceeds this browser's 3D texture limit (${largestDimension} > ${max3DTextureSize}).`
    );
  }
};

export const computeDeliveredVoxelSpacing = ({
  sourceGrid,
  sourceSpacing,
  deliveredGrid,
}: {
  sourceGrid: VolumeGrid;
  sourceSpacing: VolumeSpacing;
  deliveredGrid: VolumeGrid;
}): VolumeSpacing => {
  validateVolumeTextureGrid(sourceGrid, 0);
  validateVolumeTextureGrid(deliveredGrid, 0);
  const spacing = [sourceSpacing.x, sourceSpacing.y, sourceSpacing.z];
  if (spacing.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("Source voxel spacing must be finite and positive.");
  }
  return {
    x: (sourceGrid.width * sourceSpacing.x) / deliveredGrid.width,
    y: (sourceGrid.height * sourceSpacing.y) / deliveredGrid.height,
    z: (sourceGrid.depth * sourceSpacing.z) / deliveredGrid.depth,
  };
};

export const computeVolumeVoxelStep = (grid: VolumeGrid): VolumeSpacing => {
  validateVolumeTextureGrid(grid, 0);
  return {
    x: 1 / grid.width,
    y: 1 / grid.height,
    z: 1 / grid.depth,
  };
};

export const validateSelectedChannelPreparedBudget = (
  firstPreparedBytes: number,
  selectedChannelCount: number,
  maxBytes = MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES
): void => {
  if (
    !Number.isSafeInteger(firstPreparedBytes) ||
    firstPreparedBytes <= 0 ||
    !Number.isSafeInteger(selectedChannelCount) ||
    selectedChannelCount <= 0
  ) {
    throw new RangeError("Selected channel prepared-byte budget requires positive safe integers.");
  }
  const aggregateBytes = firstPreparedBytes * selectedChannelCount;
  if (!Number.isSafeInteger(aggregateBytes) || aggregateBytes > maxBytes) {
    throw new RangeError(
      `Selected channel previews require ${aggregateBytes} prepared bytes, exceeding the ${maxBytes} byte aggregate limit.`
    );
  }
};

export const validateActiveScalarGpuBudget = (
  predictedTextureBytes: number,
  selectedChannelCount: number,
  maxBytes = MAX_ACTIVE_SCALAR_VOLUME_GPU_BYTES
): void => {
  if (
    !Number.isSafeInteger(predictedTextureBytes) ||
    predictedTextureBytes <= 0 ||
    !Number.isSafeInteger(selectedChannelCount) ||
    selectedChannelCount <= 0
  ) {
    throw new RangeError("Active scalar GPU budget requires positive safe integers.");
  }
  const activeBytes = predictedTextureBytes * selectedChannelCount;
  if (!Number.isSafeInteger(activeBytes) || activeBytes > maxBytes) {
    throw new RangeError(
      `Selected scalar textures require ${activeBytes} active GPU bytes, exceeding the ${maxBytes} byte GPU limit.`
    );
  }
};

export const MAX_ATLAS_DECODE_WORKING_SET_BYTES = 256 * 1024 * 1024;

export const validateAtlasDecodeWorkingSet = (geometry: {
  atlasWidth: number;
  atlasHeight: number;
  sliceWidth: number;
  sliceHeight: number;
  sliceCount: number;
  columns: number;
  rows: number;
}): { atlasBytes: number; volumeBytes: number; totalBytes: number } => {
  const dimensions = Object.entries(geometry);
  if (dimensions.some(([, value]) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError("Atlas decode dimensions must be positive safe integers.");
  }
  const atlasPixels = geometry.atlasWidth * geometry.atlasHeight;
  const volumeVoxels = geometry.sliceWidth * geometry.sliceHeight * geometry.sliceCount;
  const slotCount = geometry.columns * geometry.rows;
  const atlasBytes = atlasPixels * 4;
  const volumeBytes = volumeVoxels * 4;
  const totalBytes = atlasBytes + volumeBytes;
  if (
    !Number.isSafeInteger(atlasPixels) ||
    !Number.isSafeInteger(volumeVoxels) ||
    !Number.isSafeInteger(slotCount) ||
    !Number.isSafeInteger(atlasBytes) ||
    !Number.isSafeInteger(volumeBytes) ||
    !Number.isSafeInteger(totalBytes)
  ) {
    throw new RangeError("Atlas decode geometry overflows safe integer bounds.");
  }
  const requiredRows = Math.ceil(geometry.sliceCount / geometry.columns);
  if (
    geometry.sliceCount > slotCount ||
    geometry.sliceWidth * geometry.columns > geometry.atlasWidth ||
    geometry.sliceHeight * requiredRows > geometry.atlasHeight
  ) {
    throw new RangeError("Atlas decode geometry does not fit within the delivered atlas.");
  }
  if (totalBytes > MAX_ATLAS_DECODE_WORKING_SET_BYTES) {
    throw new RangeError(
      `Atlas decode requires ${totalBytes} bytes, exceeding the ${MAX_ATLAS_DECODE_WORKING_SET_BYTES} byte working-set limit.`
    );
  }
  return { atlasBytes, volumeBytes, totalBytes };
};

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

export type VolumeClipBounds = {
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

export const volumeCameraSnapshotKey = ({
  persistenceKey,
  cameraModeId,
  viewPresetId,
  interiorFrame,
}: {
  persistenceKey?: string;
  cameraModeId: string;
  viewPresetId: string;
  interiorFrame: VolumeInteriorCameraFrame | null;
}): string | null => {
  if (!persistenceKey) {
    return null;
  }
  const interiorIdentity = interiorFrame
    ? [
        interiorFrame.position.x,
        interiorFrame.position.y,
        interiorFrame.position.z,
        interiorFrame.target.x,
        interiorFrame.target.y,
        interiorFrame.target.z,
        interiorFrame.lookDistance,
      ].join(",")
    : "overview";
  return `${persistenceKey}:${cameraModeId}:${viewPresetId}:${interiorIdentity}`;
};

export type VolumeSampleBudget = {
  interactiveSteps: number;
  settledSteps: number;
  rampFactor: number;
};

export function computeVolumeSampleBudget({
  sourceKind,
  volumeWidth,
  volumeHeight,
  volumeDepth,
  projectionMode,
}: {
  sourceKind: ScalarVolumeSource["kind"] | AtlasVolumeSource["kind"] | MultichannelVolumeSource["kind"];
  volumeWidth?: number;
  volumeHeight?: number;
  volumeDepth: number;
  projectionMode: "mip" | "composite";
}): VolumeSampleBudget {
  const safeDepth = Math.max(1, Math.floor(Number(volumeDepth) || 1));
  const floor = sourceKind === "atlas" ? 96 : 128;
  // A scalar ray can cross intervals on all three axes. Budget from the full
  // delivery grid (bounded by MAX_STEPS), not only Z: shallow thick-slice CTs
  // still have long in-plane rays. Atlas keeps its established depth budget.
  const safeWidth = Math.max(1, Math.floor(Number(volumeWidth) || safeDepth));
  const safeHeight = Math.max(1, Math.floor(Number(volumeHeight) || safeDepth));
  const semanticGridCrossings = safeWidth + safeHeight + safeDepth;
  const atlasProjectionMultiplier: Record<"mip" | "composite", number> = {
    mip: 2,
    composite: 2,
  };
  const atlasDepthBudget = safeDepth * atlasProjectionMultiplier[projectionMode];
  const settledSteps = Math.max(
    floor,
    Math.min(MAX_STEPS, sourceKind === "atlas" ? atlasDepthBudget : semanticGridCrossings)
  );
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
  maskMode = false,
}: {
  cueVisible: boolean;
  interiorInspectionActive: boolean;
  cutawayActive?: boolean;
  maskMode?: boolean;
}): boolean {
  // Hide the flat translucent X/Y/Z cursor quads whenever the view is focused on
  // the interior — both the legacy fly-inside and the Z-cursor cutaway. In
  // cutaway the crisp opaque cut face IS the inspection surface, so the cursor
  // planes only tint/occlude it (the user sees red/green washes over the slice).
  return cueVisible && !interiorInspectionActive && !cutawayActive && !maskMode;
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

export function resolveScalarOpacityPolicy({
  renderPolicy,
  modality,
}: {
  renderPolicy: string;
  modality: string;
}): { edgeStrength: number; interiorOpacity: number } {
  if (renderPolicy !== "scalar") {
    return { edgeStrength: 0, interiorOpacity: 1 };
  }
  if (modality === "medical") {
    // A medical transfer function must remain monotonic in the windowed scalar;
    // gradients may light anatomy, but must not manufacture opacity shells.
    return { edgeStrength: 0, interiorOpacity: 1 };
  }
  return { edgeStrength: 3.5, interiorOpacity: 0.5 };
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

export function resolveAtlasVolumeRenderMode({
  renderPolicy,
  categoricalMode,
}: {
  renderPolicy?: string;
  categoricalMode?: CategoricalVolumeMode;
}): AtlasVolumeRenderMode {
  if (String(renderPolicy ?? "").trim().toLowerCase() !== "categorical") {
    return { id: "xray", shaderValue: 0 };
  }
  return categoricalMode === "xray"
    ? { id: "xray", shaderValue: 0 }
    : { id: "surface", shaderValue: 1 };
}

const VERTEX_SHADER = `
  varying vec3 vPosition;

  void main() {
    vec4 position4 = vec4(position, 1.0);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * position4;
  }
`;

const MASK_VERTEX_SHADER = `
  out vec3 vPosition;

  void main() {
    vec4 position4 = vec4(position, 1.0);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * position4;
  }
`;

const createMaskFragmentShader = (samplerType: "usampler3D" | "isampler3D") => `
  precision highp float;
  precision highp int;
  precision highp ${samplerType};

  uniform ${samplerType} uData;
  uniform int uMaskThreshold;
  uniform vec3 uGridSize;
  uniform vec3 uVoxelSpacing;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uCameraPositionLocal;
  uniform vec3 uCameraDirectionLocal;
  uniform bool uOrthographicCamera;

  in vec3 vPosition;
  out vec4 outColor;

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

  bool occupied(ivec3 cell) {
    ivec3 gridLimit = ivec3(uGridSize);
    if (
      cell.x < 0 || cell.y < 0 || cell.z < 0 ||
      cell.x >= gridLimit.x || cell.y >= gridLimit.y || cell.z >= gridLimit.z
    ) {
      return false;
    }
    vec3 texelCenter = (vec3(cell) + vec3(0.5)) / uGridSize;
    return int(texture(uData, texelCenter).r) > uMaskThreshold;
  }

  vec3 maskSurfaceColor(ivec3 cell, vec3 entryNormal, vec3 rayDir) {
    vec3 occupancyGradient = vec3(
      float(occupied(cell - ivec3(1, 0, 0))) - float(occupied(cell + ivec3(1, 0, 0))),
      float(occupied(cell - ivec3(0, 1, 0))) - float(occupied(cell + ivec3(0, 1, 0))),
      float(occupied(cell - ivec3(0, 0, 1))) - float(occupied(cell + ivec3(0, 0, 1)))
    ) / max(uVoxelSpacing, vec3(0.000001));
    vec3 normal = length(occupancyGradient) > 0.000001
      ? normalize(occupancyGradient)
      : normalize(entryNormal);
    if (length(normal) < 0.000001) {
      normal = -normalize(rayDir);
    }
    vec3 lightDirection = normalize(vec3(-0.42, 0.58, 0.70));
    vec3 viewDirection = -normalize(rayDir);
    float diffuse = max(0.0, dot(normal, lightDirection));
    float facing = max(0.0, dot(normal, viewDirection));
    float shade = clamp(0.62 + 0.30 * diffuse + 0.08 * facing, 0.58, 1.0);
    return vec3(0.90, 0.94, 1.0) * shade;
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

    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    vec3 gridOrigin = clamp(
      (front + vec3(0.5)) * uGridSize,
      vec3(0.0),
      uGridSize - vec3(0.0001)
    );
    vec3 gridDirection = rayDir * uGridSize;
    ivec3 cell = ivec3(floor(gridOrigin));
    ivec3 gridLimit = ivec3(uGridSize);
    ivec3 cellStep = ivec3(
      abs(gridDirection.x) <= 0.0000001 ? 0 : (gridDirection.x < 0.0 ? -1 : 1),
      abs(gridDirection.y) <= 0.0000001 ? 0 : (gridDirection.y < 0.0 ? -1 : 1),
      abs(gridDirection.z) <= 0.0000001 ? 0 : (gridDirection.z < 0.0 ? -1 : 1)
    );
    vec3 safeDirection = vec3(
      cellStep.x == 0 ? 1.0 : gridDirection.x,
      cellStep.y == 0 ? 1.0 : gridDirection.y,
      cellStep.z == 0 ? 1.0 : gridDirection.z
    );
    vec3 nextBoundary = vec3(
      cellStep.x > 0 ? cell.x + 1 : cell.x,
      cellStep.y > 0 ? cell.y + 1 : cell.y,
      cellStep.z > 0 ? cell.z + 1 : cell.z
    );
    vec3 tMax = (nextBoundary - gridOrigin) / safeDirection;
    vec3 tDelta = abs(vec3(1.0) / safeDirection);
    if (cellStep.x == 0) {
      tMax.x = 1e30;
      tDelta.x = 1e30;
    }
    if (cellStep.y == 0) {
      tMax.y = 1e30;
      tDelta.y = 1e30;
    }
    if (cellStep.z == 0) {
      tMax.z = 1e30;
      tDelta.z = 1e30;
    }

    vec3 entryDistance = min(abs(front - boxMin), abs(front - boxMax));
    vec3 entryNormal =
      entryDistance.x <= entryDistance.y && entryDistance.x <= entryDistance.z
        ? vec3(rayDir.x > 0.0 ? -1.0 : 1.0, 0.0, 0.0)
        : entryDistance.y <= entryDistance.z
          ? vec3(0.0, rayDir.y > 0.0 ? -1.0 : 1.0, 0.0)
          : vec3(0.0, 0.0, rayDir.z > 0.0 ? -1.0 : 1.0);

    for (int iter = 0; iter < ${MAX_MASK_SURFACE_STEPS}; iter++) {
      if (
        cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        cell.x >= gridLimit.x || cell.y >= gridLimit.y || cell.z >= gridLimit.z
      ) {
        break;
      }
      if (occupied(cell)) {
        outColor = vec4(maskSurfaceColor(cell, entryNormal, rayDir), 1.0);
        return;
      }
      float nextT = min(tMax.x, min(tMax.y, tMax.z));
      if (nextT > distance(front, back) + 0.000001) {
        break;
      }
      entryNormal = vec3(0.0);
      if (tMax.x <= nextT + 0.0000001) {
        cell.x += cellStep.x;
        tMax.x += tDelta.x;
        entryNormal.x = -float(cellStep.x);
      }
      if (tMax.y <= nextT + 0.0000001) {
        cell.y += cellStep.y;
        tMax.y += tDelta.y;
        entryNormal.y = -float(cellStep.y);
      }
      if (tMax.z <= nextT + 0.0000001) {
        cell.z += cellStep.z;
        tMax.z += tDelta.z;
        entryNormal.z = -float(cellStep.z);
      }
    }
    discard;
  }
`;

const UNSIGNED_MASK_FRAGMENT_SHADER = createMaskFragmentShader("usampler3D");
const SIGNED_MASK_FRAGMENT_SHADER = createMaskFragmentShader("isampler3D");

const ATLAS_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uData;
  uniform int uSteps;
  uniform float uDensity;
  uniform int uAtlasRenderMode;
  uniform bool uFeatureMask;
  uniform vec3 uGridSize;
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

    float entryT = min(tFar, max(tNear, 0.0) + 0.0000001);
    vec3 front = rayOrigin + rayDir * entryT;
    vec3 back = rayOrigin + rayDir * tFar;
    vec3 stepVector = (back - front) / float(steps);
    vec3 location = front + vec3(0.5);
    vec3 delta = stepVector;

    // Categorical Surface mode is an opaque first-hit renderer. Nearest-filtered
    // texel-center samples are returned intact, while voxel DDA visits every
    // intersected cell exactly once. Label 0 is encoded as black and remains empty.
    if (uAtlasRenderMode == 1) {
      vec3 gridSize = max(uGridSize, vec3(1.0));
      ivec3 voxel = ivec3(clamp(floor((front + vec3(0.5)) * gridSize), vec3(0.0), gridSize - vec3(1.0)));
      ivec3 voxelStep = ivec3(sign(rayDir));
      vec3 nextBoundary = (vec3(voxel) + step(vec3(0.0), rayDir)) / gridSize - vec3(0.5);
      vec3 nextCrossing = vec3(1.0e30);
      vec3 crossingDelta = vec3(1.0e30);
      if (abs(rayDir.x) > 0.000000001) {
        nextCrossing.x = (nextBoundary.x - rayOrigin.x) / rayDir.x;
        crossingDelta.x = 1.0 / (gridSize.x * abs(rayDir.x));
      }
      if (abs(rayDir.y) > 0.000000001) {
        nextCrossing.y = (nextBoundary.y - rayOrigin.y) / rayDir.y;
        crossingDelta.y = 1.0 / (gridSize.y * abs(rayDir.y));
      }
      if (abs(rayDir.z) > 0.000000001) {
        nextCrossing.z = (nextBoundary.z - rayOrigin.z) / rayDir.z;
        crossingDelta.z = 1.0 / (gridSize.z * abs(rayDir.z));
      }
      for (int iter = 0; iter < ${MAX_CATEGORICAL_SURFACE_STEPS}; iter++) {
        vec3 texelCenter = (vec3(voxel) + vec3(0.5)) / gridSize;
        vec4 sampleColor = texture(uData, texelCenter);
        bool occupied = uFeatureMask
          ? sampleColor.a > 0.0
          : max(max(sampleColor.r, sampleColor.g), sampleColor.b) > 0.0;
        if (occupied) {
          gl_FragColor = vec4(sampleColor.rgb, 1.0);
          return;
        }
        float crossing = min(nextCrossing.x, min(nextCrossing.y, nextCrossing.z));
        if (crossing > tFar) {
          break;
        }
        float tieTolerance = max(
          ${CATEGORICAL_SURFACE_TIE_EPSILON},
          abs(crossing) * ${CATEGORICAL_SURFACE_TIE_EPSILON}
        );
        if (abs(nextCrossing.x - crossing) <= tieTolerance) {
          voxel.x += voxelStep.x;
          nextCrossing.x += crossingDelta.x;
        }
        if (abs(nextCrossing.y - crossing) <= tieTolerance) {
          voxel.y += voxelStep.y;
          nextCrossing.y += crossingDelta.y;
        }
        if (abs(nextCrossing.z - crossing) <= tieTolerance) {
          voxel.z += voxelStep.z;
          nextCrossing.z += crossingDelta.z;
        }
        if (any(lessThan(voxel, ivec3(0))) || any(greaterThanEqual(voxel, ivec3(gridSize)))) {
          break;
        }
      }
      discard;
    }

    vec4 accum = vec4(0.0);
    for (int iter = 0; iter < ${MAX_STEPS}; iter++) {
      if (iter >= steps) {
        break;
      }
      vec4 sampleColor = texture(uData, clamp(location, vec3(0.0), vec3(1.0)));
      float occupancy = uFeatureMask
        ? sampleColor.a
        : max(max(sampleColor.r, sampleColor.g), sampleColor.b);
      float alpha = alphaFromOpacity(occupancy, length(delta));
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
  uniform vec3 uGridSize;
  uniform vec3 uVoxelStep;
  uniform vec3 uVolumeScale;   // per-axis world EXTENT ratio (box dimensions)
  uniform vec3 uVoxelSpacing;  // per-axis voxel SPACING ratio (mm), max-normalized
  uniform float uEdgeStrength;
  uniform float uInteriorOpacity;
  uniform int uProjectionMode;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform bool uCutawayActive;
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

  float stableRayJitter() {
    // Stable interleaved-gradient noise breaks coherent sampling bands without
    // temporal shimmer. It is screen-space only; no time-varying uniform.
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
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
    vec3 physicalGradient = gradient / max(vec3(0.06), uVoxelSpacing);
    float edge = clamp(length(physicalGradient) * uEdgeStrength, 0.0, 1.0);
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
      ? -normalize(uCameraDirectionLocal * uVolumeScale)
      : normalize((uCameraPositionLocal - (location - vec3(0.5))) * uVolumeScale);
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

    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    // The Z cut surface participates in ray order: emit the exact voxel-center
    // sample only when it is the ray's first clip-box intersection. Rays entering
    // through a retained side march that anatomy normally, and cameras behind the
    // cut enter through z-min, so the slice can never paint through the volume.
    if (uCutawayActive && rayDir.z < 0.0 && abs(front.z - boxMax.z) <= 0.0001) {
      float cutValue = sampleWindowed(front + vec3(0.5));
      gl_FragColor = vec4(scalarColor(cutValue), 1.0);
      return;
    }
    float rayIntervals = dot(abs(back - front) * uGridSize, vec3(1.0));
    int rayStepLimit = max(1, int(ceil(rayIntervals)));
    int steps = min(min(uSteps, ${MAX_STEPS}), rayStepLimit);
    if (steps < 1) {
      discard;
    }

    vec3 stepVector = (back - front) / float(steps);
    vec3 delta = stepVector;
    vec3 location = front + vec3(0.5) + delta * stableRayJitter();

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
      if (opacityValue <= 0.0) {
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

// Max simultaneously-rendered channels. WebGL2 guarantees >= 16 fragment texture
// units; fluorescence is typically 2-5 channels, 8 covers spectral with headroom
// for the cut face. Larger selections fail explicitly instead of silently dropping
// scientific channels.
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
  uniform bool uCutawayActive;
  uniform vec3 uGridSize;
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
      ? -normalize(uCameraDirectionLocal * uVolumeScale)
      : normalize((uCameraPositionLocal - (location - vec3(0.5))) * uVolumeScale);
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
    if (uChannelCount < 1) {
      discard;
    }
    vec3 front = rayOrigin + rayDir * max(tNear, 0.0);
    vec3 back = rayOrigin + rayDir * tFar;
    if (uCutawayActive && rayDir.z < 0.0 && abs(front.z - boxMax.z) <= 0.0001) {
      vec4 cutValue = fuseVoxel(front + vec3(0.5));
      gl_FragColor = vec4(applyTone(cutValue.rgb), 1.0);
      return;
    }
    float rayIntervals = dot(abs(back - front) * uGridSize, vec3(1.0));
    int rayStepLimit = max(1, int(ceil(rayIntervals)));
    int steps = min(min(uSteps, ${MAX_STEPS}), rayStepLimit);
    if (steps < 1) {
      discard;
    }
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
      vec3 lit = fused.rgb;
      if (uLightingEnabled && uLightingStrength > 0.0) {
        lit = applyLighting(location, fused.rgb, densityGradient(location));
      }
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

const throwIfVolumeLoadAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) {
    throw new DOMException("Volume load aborted", "AbortError");
  }
};

type DecodedAtlasImage = CanvasImageSource & {
  width: number;
  height: number;
  close?: () => void;
};

const loadAtlasImage = async (atlasUrl: string, signal?: AbortSignal): Promise<DecodedAtlasImage> => {
  throwIfVolumeLoadAborted(signal);
  const response = await fetch(atlasUrl, { credentials: "include", signal });
  if (!response.ok) {
    throw new Error(`Atlas image failed to load (${response.status})`);
  }
  const blob = await response.blob();
  throwIfVolumeLoadAborted(signal);
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    if (signal?.aborted) {
      bitmap.close();
      throwIfVolumeLoadAborted(signal);
    }
    return bitmap;
  }
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new window.Image();
    const finish = () => {
      signal?.removeEventListener("abort", onAbort);
      URL.revokeObjectURL(objectUrl);
    };
    const onAbort = () => {
      image.src = "";
      finish();
      reject(new DOMException("Volume load aborted", "AbortError"));
    };
    image.decoding = "async";
    image.onload = () => {
      finish();
      resolve(image);
    };
    image.onerror = () => {
      finish();
      reject(new Error("Atlas image failed to decode"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    image.src = objectUrl;
  });
};

export const atlasToVolumeTexture = async (
  atlasUrl: string,
  atlasScheme: NonNullable<UploadViewerInfo["viewer"]["atlas_scheme"]>,
  texturePolicy: "linear" | "nearest",
  signal?: AbortSignal
): Promise<THREE.Data3DTexture> => {
  const image = await loadAtlasImage(atlasUrl, signal);
  throwIfVolumeLoadAborted(signal);
  const atlasWidth = image.width || atlasScheme.atlas_width;
  const atlasHeight = image.height || atlasScheme.atlas_height;
  const sliceWidth = atlasScheme.slice_width;
  const sliceHeight = atlasScheme.slice_height;
  const sliceCount = atlasScheme.slice_count;
  const columns = atlasScheme.columns;
  try {
    validateAtlasDecodeWorkingSet({
      atlasWidth,
      atlasHeight,
      sliceWidth,
      sliceHeight,
      sliceCount,
      columns,
      rows: atlasScheme.rows,
    });
  } catch (error) {
    image.close?.();
    throw error;
  }

  const canvas = document.createElement("canvas");
  canvas.width = atlasWidth;
  canvas.height = atlasHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    image.close?.();
    throw new Error("2D canvas unavailable for atlas decoding");
  }
  try {
    context.drawImage(image, 0, 0, atlasWidth, atlasHeight);
  } finally {
    image.close?.();
  }
  throwIfVolumeLoadAborted(signal);
  const atlasData = context.getImageData(0, 0, atlasWidth, atlasHeight).data;
  throwIfVolumeLoadAborted(signal);
  const volumeData = new Uint8Array(sliceWidth * sliceHeight * sliceCount * 4);

  for (let sliceIndex = 0; sliceIndex < sliceCount; sliceIndex += 1) {
    if (sliceIndex > 0 && sliceIndex % 8 === 0) {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      throwIfVolumeLoadAborted(signal);
    }
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

  throwIfVolumeLoadAborted(signal);
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

const preparedScalarToVolumeTexture = (
  prepared: PreparedScalarVolume,
  texturePolicy: "linear" | "nearest"
): THREE.Data3DTexture => {
  if (
    prepared.textureEncoding !== "normalized-half" &&
    prepared.textureEncoding !== "raw-float"
  ) {
    throw new RangeError("Integer Mask data requires the dedicated integer texture path.");
  }
  const texture = new THREE.Data3DTexture(
    prepared.textureData,
    prepared.width,
    prepared.height,
    prepared.depth
  );
  texture.format = THREE.RedFormat;
  texture.type =
    prepared.textureEncoding === "raw-float"
      ? THREE.FloatType
      : THREE.HalfFloatType;
  texture.unpackAlignment =
    prepared.textureEncoding === "raw-float"
      ? Float32Array.BYTES_PER_ELEMENT
      : Uint16Array.BYTES_PER_ELEMENT;
  texture.generateMipmaps = false;
  texture.minFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.magFilter = texturePolicy === "nearest" ? THREE.NearestFilter : THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
};

const preparedMaskToIntegerTexture = (
  prepared: PreparedScalarVolume
): THREE.Data3DTexture => {
  const texture = new THREE.Data3DTexture(
    prepared.textureData,
    prepared.width,
    prepared.height,
    prepared.depth
  );
  texture.format = THREE.RedIntegerFormat;
  if (
    prepared.textureEncoding === "raw-uint8" &&
    prepared.textureData instanceof Uint8Array
  ) {
    texture.type = THREE.UnsignedByteType;
    texture.internalFormat = "R8UI";
  } else if (
    prepared.textureEncoding === "raw-uint16" &&
    prepared.textureData instanceof Uint16Array
  ) {
    texture.type = THREE.UnsignedShortType;
    texture.internalFormat = "R16UI";
  } else if (
    prepared.textureEncoding === "raw-int16" &&
    prepared.textureData instanceof Int16Array
  ) {
    texture.type = THREE.ShortType;
    texture.internalFormat = "R16I";
  } else {
    throw new RangeError("Prepared Mask texture dtype does not match its integer encoding.");
  }
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
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

const normalizedSpatialUnit = (value: unknown): string | null => {
  if (typeof value !== "string") {
    return null;
  }
  const unit = value.trim();
  if (
    !unit ||
    ["px", "pixel", "pixels", "voxel", "voxels", "unknown"].includes(unit.toLowerCase())
  ) {
    return null;
  }
  return unit;
};

export const resolveSpatialUnit = (viewerInfo?: UploadViewerInfo | null): string => {
  const explicitUnit = normalizedSpatialUnit(viewerInfo?.metadata.physical_spacing_unit);
  if (explicitUnit) {
    return explicitUnit;
  }
  const coordinates = viewerInfo?.phys?.coordinates;
  const units =
    coordinates && typeof coordinates === "object"
      ? (coordinates as Record<string, unknown>).space_units
      : null;
  const spatial =
    units && typeof units === "object"
      ? (units as Record<string, unknown>).spatial
      : null;
  const coordinateUnit = normalizedSpatialUnit(spatial);
  if (coordinateUnit) {
    return coordinateUnit;
  }
  const pixelUnit = normalizedSpatialUnit(viewerInfo?.phys?.pixel_units?.[0]);
  if (pixelUnit) {
    return pixelUnit;
  }
  const spacingUnits = viewerInfo?.metadata.spacing_units;
  const spacingUnitValues = [spacingUnits?.x, spacingUnits?.y, spacingUnits?.z]
    .map(normalizedSpatialUnit)
    .filter((unit): unit is string => unit != null);
  if (
    spacingUnitValues.length > 0 &&
    new Set(spacingUnitValues.map((unit) => unit.toLowerCase())).size === 1
  ) {
    return spacingUnitValues[0] as string;
  }
  return viewerInfo?.metadata.physical_spacing ? "units" : "vox";
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
  categoricalMode,
  volumeCutaway,
  featureMask = false,
  cameraPersistenceKey,
  resolvedScalarRendering,
}: SliceStackVolumeCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasHostRef = useRef<HTMLDivElement | null>(null);
  const requestInteractiveRenderRef = useRef<(() => void) | null>(null);
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
  const maskThresholdUniformRef = useRef<{ value: number } | null>(null);
  const clipUniformsRef = useRef<{
    uClipMin: { value: THREE.Vector3 };
    uClipMax: { value: THREE.Vector3 };
    uCutawayActive?: { value: boolean };
  } | null>(null);
  const atlasRenderModeUniformRef = useRef<{ value: number } | null>(null);
  const scalarRenderConfigRef = useRef({
    enhancement: undefined as string | undefined,
    negative: false,
    colorMapShaderValue: 0,
    signalFloor: 0,
    densityScale: 1,
    lightingEnabled: false,
    lightingStrength: 0.65,
  });
  const scalarMaskConfigRef = useRef({ enabled: false, rawThreshold: 0 });
  const sliceCursorCueRef = useRef<VolumeSliceCursorCue | null>(null);
  const cameraRigRef = useRef<{
    camera: VolumeCamera;
    controls: TrackballControls;
  } | null>(null);
  const cameraSnapshotRef = useRef<{
    key: string;
    position: THREE.Vector3;
    target: THREE.Vector3;
    up: THREE.Vector3;
    zoom: number;
  } | null>(null);
  const sliceCursorPlanesRef = useRef<SliceCursorPlanes | null>(null);
  const scalarRangeRef = useRef<
    Pick<ScalarVolumePayload, "rawMin" | "rawMax" | "sclSlope" | "sclInter"> | null
  >(null);
  // Live multichannel uniforms (the THREE uniform objects) so per-channel
  // color/window + gamma update without rebuilding the renderer.
  const multichannelUniformsRef = useRef<MultichannelUniformSet | null>(null);
  const multichannelRenderConfigRef = useRef<MultichannelRenderConfig | null>(null);
  // Per-channel ImageJ-style auto-contrast windows (normalized [0,1]), computed
  // from each loaded volume; used as the default window when the user has not set
  // a manual one, so the background maps to zero opacity (no fog).
  const channelAutoWindowsRef = useRef<Map<number, { low: number; high: number }>>(new Map());
  const [renderFailure, setRenderFailure] = useState<{
    sourceKey: string;
    message: string;
  } | null>(null);
  const [retryGeneration, setRetryGeneration] = useState(0);
  const [channelLoadProgress, setChannelLoadProgress] = useState<{ loaded: number; total: number } | null>(null);

  const plane = useMemo(
    () =>
      volumeSource?.plane ??
      getPlaneDescriptor(
        viewerInfo as UploadViewerInfo,
        "z"
      ),
    [viewerInfo, volumeSource?.plane]
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

  const displayChannels = displayState?.channels;
  const volumeSelection = useMemo(() => {
    try {
      if (!Number.isSafeInteger(axisSizes.C) || axisSizes.C <= 0) {
        throw new RangeError("Manifest channel count must be a positive safe integer.");
      }
      if (!Number.isSafeInteger(axisSizes.T) || axisSizes.T <= 0) {
        throw new RangeError("Manifest time count must be a positive safe integer.");
      }
      if (
        resolvedScalarRendering === undefined &&
        viewerInfo &&
        !volumeSource &&
        viewerInfo.kind !== "hdf5" &&
        viewerInfo.viewer.render_policy !== "categorical"
      ) {
        throw new RangeError("Resolved scalar rendering identity is required.");
      }
      if (resolvedScalarRendering === null) {
        throw new RangeError("Resolved scalar rendering identity is unavailable.");
      }
      const requestedTime = resolveStrictVolumeIndex(
        resolvedScalarRendering?.time ?? tIndex ?? 0,
        axisSizes.T,
        "time"
      );
      const requestedChannel = resolveStrictVolumeIndex(
        resolvedScalarRendering?.channel ?? 0,
        axisSizes.C,
        "channel"
      );
      if (
        resolvedScalarRendering?.effectiveMode === "mask" &&
        !Number.isFinite(resolvedScalarRendering.threshold)
      ) {
        throw new RangeError("Resolved mask threshold is unavailable.");
      }
      const rawChannels = Array.isArray(displayChannels) && displayChannels.length > 0
        ? displayChannels
        : [requestedChannel];
      const enabledChannels =
        resolvedScalarRendering?.effectiveMode === "mask"
          ? [requestedChannel]
          : rawChannels.map((value) =>
              resolveStrictVolumeIndex(value, axisSizes.C, "channel")
            );
      if (new Set(enabledChannels).size !== enabledChannels.length) {
        throw new RangeError("Duplicate channel indices are not allowed for volume rendering.");
      }
      return {
        requestedTime,
        scalarChannel: requestedChannel,
        enabledChannels,
        error: null as string | null,
      };
    } catch (error) {
      return {
        requestedTime: 0,
        scalarChannel: null,
        enabledChannels: [] as number[],
        error: error instanceof Error ? error.message : "Invalid volume selection.",
      };
    }
  }, [axisSizes.C, axisSizes.T, displayChannels, resolvedScalarRendering, tIndex, viewerInfo, volumeSource]);
  const scalarChannel = volumeSelection.scalarChannel;
  const enabledVolumeChannels = volumeSelection.enabledChannels;
  const requestedTime = volumeSelection.requestedTime;
  const effectiveMaskMode =
    resolvedScalarRendering?.effectiveMode === "mask";
  const rawMaskThreshold = Number(resolvedScalarRendering?.threshold);

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
      !effectiveMaskMode &&
      viewerInfo?.viewer.volume_mode !== "scalar" &&
      Boolean(viewerInfo?.viewer.available_surfaces?.includes("volume")) &&
      Boolean(viewerInfo?.viewer.service_urls?.scalar_volume ?? viewerInfo?.service_urls?.scalar_volume)
  );

  const atlasUrl = useMemo(() => {
    const atlasServiceUrl = viewerInfo?.viewer.service_urls?.atlas ?? viewerInfo?.service_urls?.atlas;
    if (!apiClient || !fileId || !atlasServiceUrl || volumeSelection.error) {
      return "";
    }
    return apiClient.uploadAtlasUrl(fileId, {
      enhancement: displayState?.enhancement,
      fusionMethod: displayState?.fusion_method,
      negative: displayState?.negative,
      channels: effectiveMaskMode
        ? enabledVolumeChannels
        : displayState?.channels,
      channelColors: displayState?.channel_colors,
      t: requestedTime,
    });
  }, [
    apiClient,
    displayState?.channel_colors,
    displayState?.channels,
    displayState?.enhancement,
    displayState?.fusion_method,
    displayState?.negative,
    effectiveMaskMode,
    enabledVolumeChannels,
    fileId,
    requestedTime,
    viewerInfo?.service_urls?.atlas,
    viewerInfo?.viewer.service_urls?.atlas,
    volumeSelection.error,
  ]);

  const nativeScalarSource = useMemo<ScalarVolumeSource | null>(() => {
    if (
      (viewerInfo?.viewer.volume_mode === "scalar" || effectiveMaskMode) &&
      Boolean(viewerInfo?.viewer.service_urls?.scalar_volume ?? viewerInfo?.service_urls?.scalar_volume) &&
      apiClient &&
      fileId &&
      scalarChannel != null &&
      !volumeSelection.error
    ) {
      return {
        kind: "scalar" as const,
        loadScalarVolume: async (signal?: AbortSignal) => {
          const requestedChannel = scalarChannel;
          const payload = await apiClient.getUploadScalarVolume(fileId, {
            t: requestedTime,
            channel: scalarChannel,
            sampling: effectiveMaskMode ? "nearest" : "box",
            signal,
          });
          validateScalarVolumeIdentity(payload, {
            channel: requestedChannel,
            time: requestedTime,
            sourceGrid: { width: axisSizes.X, height: axisSizes.Y, depth: axisSizes.Z },
            policy: payload.previewPolicy,
          });
          return payload;
        },
        fallbackImageUrl: "",
        axisSizes,
        plane,
        physicalSpacing: spacing,
        texturePolicy: effectiveMaskMode ? "nearest" : undefined,
      };
    }
    return null;
  }, [apiClient, axisSizes, effectiveMaskMode, fileId, plane, requestedTime, scalarChannel, spacing, viewerInfo?.service_urls?.scalar_volume, viewerInfo?.viewer.service_urls?.scalar_volume, viewerInfo?.viewer.volume_mode, volumeSelection.error]);

  const nativeMultichannelSource = useMemo<MultichannelVolumeSource | null>(() => {
    // Multichannel fluorescence z-stack: composite the enabled channels' full-res
    // volumes in-shader (vole-core-style fuse-then-raymarch). Takes priority over
    // the server-fused atlas fallback. Identity changes when the enabled set or
    // time changes (reload); per-channel color/window are applied incrementally.
    if (
      isPerChannelVolume &&
      apiClient &&
      fileId &&
      enabledVolumeChannels.length > 0 &&
      !volumeSelection.error
    ) {
      return {
        kind: "multichannel" as const,
        channelIndices: enabledVolumeChannels,
        loadChannel: (channel: number, signal?: AbortSignal) =>
          apiClient.getUploadScalarVolume(fileId, { t: requestedTime, channel, signal }),
        fallbackImageUrl: "",
        axisSizes,
        plane,
        physicalSpacing: spacing,
      };
    }
    return null;
  }, [apiClient, axisSizes, enabledVolumeChannels, fileId, isPerChannelVolume, plane, requestedTime, spacing, volumeSelection.error]);

  const atlasScheme = viewerInfo?.viewer.atlas_scheme;
  const nativeAtlasSource = useMemo<AtlasVolumeSource | null>(() => {
    const hasAtlasAuthority = Boolean(
      viewerInfo?.viewer.service_urls?.atlas ?? viewerInfo?.service_urls?.atlas
    );
    if (!apiClient || !fileId || !atlasScheme || !hasAtlasAuthority || !atlasUrl || volumeSelection.error) {
      return null;
    }
    return {
      kind: "atlas" as const,
      atlasUrl,
      fallbackImageUrl: "",
      atlasScheme,
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
    spacing,
    atlasScheme,
    viewerInfo?.service_urls?.atlas,
    viewerInfo?.viewer.service_urls?.atlas,
    volumeSelection.error,
  ]);
  const resolvedSource = volumeSource ?? nativeScalarSource ?? nativeMultichannelSource ?? nativeAtlasSource;
  const scalarServiceUrl = String(
    viewerInfo?.viewer.service_urls?.scalar_volume ?? viewerInfo?.service_urls?.scalar_volume ?? ""
  );
  const volumeApiNamespace = useMemo(
    () => scalarVolumeApiNamespace(apiClient as object | undefined, scalarServiceUrl),
    [apiClient, scalarServiceUrl]
  );
  const volumeSourceIdentity = scalarVolumeSourceIdentity({
    fileId: fileId ?? viewerInfo?.file_id ?? "external",
    sha256: viewerInfo?.metadata.sha256,
    sizeBytes: viewerInfo?.metadata.size_bytes,
    dtype: viewerInfo?.metadata.array_dtype,
    sourceGrid: { width: axisSizes.X, height: axisSizes.Y, depth: axisSizes.Z },
  });
  const renderSourceKey = JSON.stringify({
    kind: resolvedSource?.kind ?? "none",
    api: volumeApiNamespace,
    file: fileId ?? viewerInfo?.file_id ?? "external",
    source: volumeSourceIdentity,
    time: requestedTime,
    channels:
      resolvedSource?.kind === "multichannel"
        ? resolvedSource.channelIndices
        : scalarChannel == null
          ? []
          : [scalarChannel],
    atlas: resolvedSource?.kind === "atlas" ? resolvedSource.atlasUrl : "",
    sampling: effectiveMaskMode ? "nearest" : "box",
  });
  const renderError = renderFailure?.sourceKey === renderSourceKey ? renderFailure.message : null;
  const activeRenderError = volumeSelection.error ?? renderError;

  const fallbackImageUrl = useMemo(() => {
    if (volumeSource?.fallbackImageUrl) {
      return volumeSource.fallbackImageUrl;
    }
    const sliceServiceUrl = viewerInfo?.viewer.service_urls?.slice ?? viewerInfo?.service_urls?.slice;
    if (!apiClient || !fileId || !sliceServiceUrl || volumeSelection.error) {
      return "";
    }
    return apiClient.uploadSliceUrl(fileId, {
      axis: "z",
      z: zIndex,
      t: requestedTime,
      enhancement: displayState?.enhancement,
      fusionMethod: displayState?.fusion_method,
      negative: displayState?.negative,
      channels: effectiveMaskMode
        ? enabledVolumeChannels
        : displayState?.channels,
      channelColors: displayState?.channel_colors,
      scalarRenderMode: effectiveMaskMode ? "mask" : "intensity",
      scalarThresholdValue: effectiveMaskMode ? rawMaskThreshold : undefined,
      scalarThresholdForeground: effectiveMaskMode ? "above" : undefined,
    });
  }, [
    apiClient,
    displayState?.channel_colors,
    displayState?.channels,
    displayState?.enhancement,
    displayState?.fusion_method,
    displayState?.negative,
    enabledVolumeChannels,
    effectiveMaskMode,
    fileId,
    rawMaskThreshold,
    requestedTime,
    volumeSource,
    volumeSelection.error,
    viewerInfo?.service_urls?.slice,
    viewerInfo?.viewer.service_urls?.slice,
    zIndex,
  ]);

  const renderPolicy = resolvedSource?.renderPolicy ?? viewerInfo?.viewer.render_policy ?? "scalar";
  const atlasRenderMode = resolveAtlasVolumeRenderMode({ renderPolicy, categoricalMode });
  const atlasRenderModeValueRef = useRef<number>(atlasRenderMode.shaderValue);
  const orientationCue = resolveVolumeOrientationCue(viewerInfo?.viewer.orientation);
  const modality = String(viewerInfo?.modality ?? "").trim().toLowerCase();
  // Z-cursor cutaway: the volume is cut at the live Z slice so the interior is
  // exposed with the camera kept in overview. The cut position is derived from
  // the slice cursor, so scrubbing Z sweeps the cut through the volume.
  const cutawayActive = volumeCutaway ?? Boolean(displayState?.volume_cutaway);
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
  const cameraSnapshotKey = volumeCameraSnapshotKey({
    persistenceKey: cameraPersistenceKey,
    cameraModeId: volumeCameraMode.id,
    viewPresetId: volumeViewPreset.id,
    interiorFrame: volumeInteriorCameraFrame,
  });
  const scalarVoxelStep = useMemo(
    () => computeVolumeVoxelStep({ width: axisSizes.X, height: axisSizes.Y, depth: axisSizes.Z }),
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
  const scalarOpacityPolicy = resolveScalarOpacityPolicy({ renderPolicy, modality });
  const volumeEdgeStrength = scalarOpacityPolicy.edgeStrength;
  const volumeInteriorOpacity = scalarOpacityPolicy.interiorOpacity;
  const texturePolicy: "linear" | "nearest" =
    effectiveMaskMode
      ? "nearest"
      : resolvedSource?.texturePolicy === "nearest" || resolvedSource?.texturePolicy === "linear"
      ? resolvedSource.texturePolicy
      : viewerInfo?.viewer.texture_policy === "nearest" || viewerInfo?.viewer.texture_policy === "linear"
        ? viewerInfo.viewer.texture_policy
        : renderPolicy === "categorical" || renderPolicy === "analysis"
        ? "nearest"
        : "linear";
  const sampleBudget = computeVolumeSampleBudget({
    sourceKind: resolvedSource?.kind ?? "atlas",
    volumeWidth: axisSizes.X,
    volumeHeight: axisSizes.Y,
    volumeDepth,
    projectionMode,
  });
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
  }, [effectiveClipBounds, cutawayActive, cutawayZ]);

  useEffect(() => {
    atlasRenderModeValueRef.current = atlasRenderMode.shaderValue;
    if (!atlasRenderModeUniformRef.current) {
      return;
    }
    atlasRenderModeUniformRef.current.value = atlasRenderMode.shaderValue;
    requestInteractiveRenderRef.current?.();
  }, [atlasRenderMode.shaderValue]);

  useEffect(() => {
    scalarRenderConfigRef.current = scalarRenderConfig;
  }, [scalarRenderConfig]);

  useEffect(() => {
    scalarMaskConfigRef.current = {
      enabled: effectiveMaskMode,
      rawThreshold: rawMaskThreshold,
    };
  }, [effectiveMaskMode, rawMaskThreshold]);

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
      lightingEnabled: scalarLighting.enabled,
      lightingStrength: scalarLighting.strength,
    }),
    [enabledVolumeChannels, displayState, viewerInfo?.phys?.channel_colors, scalarLighting.enabled, scalarLighting.strength, scalarTransfer.densityScale]
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
    requestInteractiveRenderRef.current?.();
  }, [multichannelRenderConfig]);

  useEffect(() => {
    sliceCursorCueRef.current = sliceCursorCue;
  }, [sliceCursorCue]);

  useEffect(() => {
    if (effectiveMaskMode) {
      if (maskThresholdUniformRef.current) {
        maskThresholdUniformRef.current.value = Math.trunc(rawMaskThreshold);
        requestRenderRef.current?.();
      }
      return;
    }
    const scalarRange = scalarRangeRef.current;
    const scalarUniforms = scalarUniformsRef.current;
    if (!scalarRange || !scalarUniforms) {
      return;
    }
    const normalizedWindow = resolveScalarVolumeWindow(
      scalarRange,
      scalarRenderConfig.enhancement,
      scalarRenderConfig.negative
    );
    scalarUniforms.uWindowLow.value = normalizedWindow.low;
    scalarUniforms.uWindowHigh.value = normalizedWindow.high;
    scalarUniforms.uInvert.value = normalizedWindow.invert;
    scalarUniforms.uColorMap.value = scalarRenderConfig.colorMapShaderValue;
    scalarUniforms.uSignalFloor.value = scalarRenderConfig.signalFloor;
    scalarUniforms.uDensityScale.value = scalarRenderConfig.densityScale;
    scalarUniforms.uLightingEnabled.value = scalarRenderConfig.lightingEnabled;
    scalarUniforms.uLightingStrength.value = scalarRenderConfig.lightingStrength;
    requestRenderRef.current?.();
  }, [effectiveMaskMode, rawMaskThreshold, scalarRenderConfig]);

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
    if (clipUniforms.uCutawayActive) {
      clipUniforms.uCutawayActive.value = cutawayActive;
    }
    requestInteractiveRenderRef.current?.();
  }, [cutawayActive, effectiveClipBounds]);

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
    requestInteractiveRenderRef.current?.();
  }, [volumeInteriorCameraFrame, volumeRadius, volumeViewPreset]);

  useEffect(() => {
    const container = containerRef.current;
    const canvasHost = canvasHostRef.current;
    if (!container || !canvasHost || !resolvedSource || activeRenderError) {
      return;
    }

    let disposed = false;
    const volumeLoadController = new AbortController();
    const rendererSampleBudget: VolumeSampleBudget = {
      interactiveSteps: sampleBudget.interactiveSteps,
      settledSteps: sampleBudget.settledSteps,
      rampFactor: sampleBudget.rampFactor,
    };
    let renderer: THREE.WebGLRenderer;
    const commitRenderError = (message: string) => {
      window.setTimeout(() => {
        if (!disposed) {
          setRenderFailure({ sourceKey: renderSourceKey, message });
        }
      }, 0);
    };
    if (
      resolvedSource.kind === "multichannel" &&
      (resolvedSource.channelIndices.length < 1 ||
        resolvedSource.channelIndices.length > MAX_VOLUME_CHANNELS)
    ) {
      commitRenderError(
        resolvedSource.channelIndices.length < 1
          ? "At least one channel is required for multichannel volume rendering"
          : `Selected channel count ${resolvedSource.channelIndices.length} exceeds the 8-channel shader limit`
      );
      return () => {
        disposed = true;
        volumeLoadController.abort();
      };
    }
    if (resolvedSource.kind === "multichannel") {
      try {
        const strictChannels = resolvedSource.channelIndices.map((channel) =>
          resolveStrictVolumeIndex(channel, resolvedSource.axisSizes.C, "channel")
        );
        if (new Set(strictChannels).size !== strictChannels.length) {
          throw new RangeError("Duplicate channel indices are not allowed for volume rendering.");
        }
      } catch (error) {
        commitRenderError(error instanceof Error ? error.message : "Invalid volume channel selection");
        return () => {
          disposed = true;
          volumeLoadController.abort();
        };
      }
    }
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
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
    if (resolvedSource.kind === "atlas") {
      try {
        validateVolumeTextureGrid(
          {
            width: resolvedSource.atlasScheme.slice_width,
            height: resolvedSource.atlasScheme.slice_height,
            depth: resolvedSource.atlasScheme.slice_count,
          },
          max3DTextureSize
        );
      } catch (error) {
        renderer.dispose();
        commitRenderError(error instanceof Error ? error.message : "Invalid atlas delivery grid");
        return () => {
          disposed = true;
        };
      }
    }
    if (resolvedSource.kind === "scalar" && effectiveMaskMode) {
      try {
        validateExactMaskSourcePreflight({
          sourceGrid: {
            width: resolvedSource.axisSizes.X,
            height: resolvedSource.axisSizes.Y,
            depth: resolvedSource.axisSizes.Z,
          },
          dtype: String(viewerInfo?.scalar_mask_capability?.dtype ?? ""),
          max3DTextureSize,
        });
      } catch (error) {
        renderer.dispose();
        commitRenderError(error instanceof Error ? error.message : "Exact Mask source is unsupported");
        return () => {
          disposed = true;
          volumeLoadController.abort();
        };
      }
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(
      resolveVolumePixelRatio(window.devicePixelRatio || 1, effectiveMaskMode)
    );
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
    const snapshot = cameraSnapshotRef.current;
    if (cameraSnapshotKey && snapshot?.key === cameraSnapshotKey) {
      camera.position.copy(snapshot.position);
      camera.up.copy(snapshot.up);
      camera.zoom = snapshot.zoom;
      camera.updateProjectionMatrix();
      controls.target.copy(snapshot.target);
      controls.update();
      cameraSnapshotRef.current = null;
    }

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    // Seed the clip uniforms with the cutaway-aware clip so the first frame is
    // already sliced; the clip effect keeps them live as Z scrubs without a
    // renderer rebuild.
    const initialClip = effectiveClipBoundsRef.current;
    const mcConfig = multichannelRenderConfigRef.current;
    const signedMaskTexture =
      String(viewerInfo?.scalar_mask_capability?.dtype ?? "").trim().toLowerCase() ===
      "int16";
    const material =
      resolvedSource.kind === "scalar"
        ? effectiveMaskMode
          ? new THREE.ShaderMaterial({
              uniforms: {
                uData: { value: null },
                uMaskThreshold: { value: Math.trunc(scalarMaskConfigRef.current.rawThreshold) },
                uGridSize: {
                  value: new THREE.Vector3(axisSizes.X, axisSizes.Y, axisSizes.Z),
                },
                uVoxelSpacing: {
                  value: new THREE.Vector3(
                    voxelSpacingRatio.x,
                    voxelSpacingRatio.y,
                    voxelSpacingRatio.z
                  ),
                },
                uClipMin: {
                  value: new THREE.Vector3(
                    initialClip.min.x,
                    initialClip.min.y,
                    initialClip.min.z
                  ),
                },
                uClipMax: {
                  value: new THREE.Vector3(
                    initialClip.max.x,
                    initialClip.max.y,
                    initialClip.max.z
                  ),
                },
                uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
                uCameraDirectionLocal: { value: new THREE.Vector3(0, 0, -1) },
                uOrthographicCamera: { value: volumeCameraMode.isOrthographic },
              },
              vertexShader: MASK_VERTEX_SHADER,
              fragmentShader: signedMaskTexture
                ? SIGNED_MASK_FRAGMENT_SHADER
                : UNSIGNED_MASK_FRAGMENT_SHADER,
              glslVersion: THREE.GLSL3,
              side: THREE.BackSide,
              transparent: true,
              depthWrite: false,
            })
          : new THREE.ShaderMaterial({
            uniforms: {
              uData: { value: null },
              uSteps: { value: rendererSampleBudget.interactiveSteps },
              uDensity: { value: density },
              uWindowLow: { value: 0.0 },
              uWindowHigh: { value: 1.0 },
              uInvert: { value: scalarRenderConfigRef.current.negative },
              uColorMap: { value: scalarRenderConfigRef.current.colorMapShaderValue },
              uSignalFloor: { value: scalarRenderConfigRef.current.signalFloor },
              uDensityScale: { value: scalarRenderConfigRef.current.densityScale },
              uLightingEnabled: { value: scalarRenderConfigRef.current.lightingEnabled },
              uLightingStrength: { value: scalarRenderConfigRef.current.lightingStrength },
              uGridSize: {
                value: new THREE.Vector3(axisSizes.X, axisSizes.Y, axisSizes.Z),
              },
              uVoxelStep: { value: new THREE.Vector3(scalarVoxelStep.x, scalarVoxelStep.y, scalarVoxelStep.z) },
              uVolumeScale: { value: new THREE.Vector3(normalizedScale.x, normalizedScale.y, normalizedScale.z) },
              uVoxelSpacing: { value: new THREE.Vector3(voxelSpacingRatio.x, voxelSpacingRatio.y, voxelSpacingRatio.z) },
              uEdgeStrength: { value: volumeEdgeStrength },
              uInteriorOpacity: { value: volumeInteriorOpacity },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(initialClip.min.x, initialClip.min.y, initialClip.min.z) },
              uClipMax: { value: new THREE.Vector3(initialClip.max.x, initialClip.max.y, initialClip.max.z) },
              uCutawayActive: { value: cutawayActiveRef.current },
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
              uSteps: { value: rendererSampleBudget.interactiveSteps },
              uDensity: { value: density },
              uDensityScale: { value: mcConfig?.densityScale ?? scalarRenderConfigRef.current.densityScale },
              uVolumeScale: { value: new THREE.Vector3(normalizedScale.x, normalizedScale.y, normalizedScale.z) },
              uProjectionMode: { value: projectionMode === "mip" ? 1 : 0 },
              uClipMin: { value: new THREE.Vector3(initialClip.min.x, initialClip.min.y, initialClip.min.z) },
              uClipMax: { value: new THREE.Vector3(initialClip.max.x, initialClip.max.y, initialClip.max.z) },
              uCutawayActive: { value: cutawayActiveRef.current },
              uGridSize: { value: new THREE.Vector3(axisSizes.X, axisSizes.Y, axisSizes.Z) },
              uCameraPositionLocal: { value: new THREE.Vector3(0, 0, 2) },
              uCameraDirectionLocal: { value: new THREE.Vector3(0, 0, -1) },
              uOrthographicCamera: { value: volumeCameraMode.isOrthographic },
              uGammaMin: { value: 0 },
              uGammaMax: { value: 1 },
              uGammaScale: { value: mcConfig?.gammaScale ?? 1 },
              uBrightness: { value: mcConfig?.brightness ?? 1 },
              uSignalFloor: { value: 0.08 },
              uLightingEnabled: { value: mcConfig?.lightingEnabled ?? false },
              uLightingStrength: { value: mcConfig?.lightingStrength ?? 0.65 },
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
              uSteps: { value: rendererSampleBudget.interactiveSteps },
              uDensity: { value: density },
              uAtlasRenderMode: { value: atlasRenderModeValueRef.current },
              uFeatureMask: { value: featureMask },
              uGridSize: {
                value: new THREE.Vector3(
                  resolvedSource.atlasScheme.slice_width,
                  resolvedSource.atlasScheme.slice_height,
                  resolvedSource.atlasScheme.slice_count
                ),
              },
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
    mesh.visible = !effectiveMaskMode;
    container.dataset.viewerMaskTextureReady =
      effectiveMaskMode ? "false" : "not-applicable";
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
          maskMode: effectiveMaskMode,
        }),
      });
    }
    sliceCursorPlanesRef.current = sliceCursorPlanes;
    scene.add(sliceCursorPlanes.x, sliceCursorPlanes.y, sliceCursorPlanes.z);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));

    const volumeUniforms = material.uniforms as Record<string, { value: THREE.Vector3 | number | boolean | null }>;
    const cameraPositionUniform = volumeUniforms.uCameraPositionLocal as { value: THREE.Vector3 };
    const cameraDirectionUniform = volumeUniforms.uCameraDirectionLocal as { value: THREE.Vector3 };
    const stepsUniform = (material.uniforms as Record<string, { value: number }>).uSteps;
    let currentSteps = effectiveMaskMode
      ? rendererSampleBudget.settledSteps
      : rendererSampleBudget.interactiveSteps;
    let lastInteractionAt = window.performance.now();
    let interactionActive = false;
    let animationFrame = 0;
    let rampTimer = 0;
    const setSamplingSteps = (steps: number) => {
      const nextSteps = Math.max(1, Math.floor(steps));
      if (nextSteps === currentSteps) {
        return;
      }
      currentSteps = nextSteps;
      if (stepsUniform) {
        stepsUniform.value = nextSteps;
      }
    };
    if (stepsUniform) {
      stepsUniform.value = currentSteps;
    }
    const updateDeliveredSampleBudget = (grid: VolumeGrid) => {
      const deliveredBudget = computeVolumeSampleBudget({
        sourceKind: resolvedSource.kind,
        volumeWidth: grid.width,
        volumeHeight: grid.height,
        volumeDepth: grid.depth,
        projectionMode,
      });
      rendererSampleBudget.interactiveSteps = deliveredBudget.interactiveSteps;
      rendererSampleBudget.settledSteps = deliveredBudget.settledSteps;
      rendererSampleBudget.rampFactor = deliveredBudget.rampFactor;
      setSamplingSteps(
        effectiveMaskMode ? deliveredBudget.settledSteps : deliveredBudget.interactiveSteps
      );
    };

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
    const renderFrame = () => {
      animationFrame = 0;
      if (disposed) {
        return;
      }
      controls.update();
      if (
        !effectiveMaskMode &&
        window.performance.now() - lastInteractionAt >= 120 &&
        currentSteps < rendererSampleBudget.settledSteps
      ) {
        setSamplingSteps(advanceProgressiveVolumeSteps(currentSteps, rendererSampleBudget));
      }
      render();
      if (interactionActive) {
        scheduleRender();
      } else if (!effectiveMaskMode) {
        scheduleProgressiveRamp();
      }
    };
    const scheduleRender = () => {
      if (disposed || animationFrame) {
        return;
      }
      animationFrame = window.requestAnimationFrame(renderFrame);
    };
    const scheduleProgressiveRamp = () => {
      if (effectiveMaskMode) {
        return;
      }
      if (rampTimer) {
        window.clearTimeout(rampTimer);
        rampTimer = 0;
      }
      if (disposed || currentSteps >= rendererSampleBudget.settledSteps) {
        return;
      }
      const elapsed = window.performance.now() - lastInteractionAt;
      if (elapsed >= 120) {
        scheduleRender();
        return;
      }
      rampTimer = window.setTimeout(() => {
        rampTimer = 0;
        scheduleRender();
      }, Math.max(1, 120 - elapsed));
    };
    const resetInteractiveSampling = () => {
      lastInteractionAt = window.performance.now();
      if (effectiveMaskMode) {
        setSamplingSteps(rendererSampleBudget.settledSteps);
      } else {
        setSamplingSteps(rendererSampleBudget.interactiveSteps);
      }
      scheduleRender();
      if (!effectiveMaskMode) {
        scheduleProgressiveRamp();
      }
    };
    const beginInteraction = () => {
      interactionActive = true;
      resetInteractiveSampling();
    };
    const markInteractionSettled = () => {
      interactionActive = false;
      lastInteractionAt = window.performance.now();
      scheduleRender();
      if (!effectiveMaskMode) {
        scheduleProgressiveRamp();
      }
    };
    controls.addEventListener("start", beginInteraction);
    controls.addEventListener("change", resetInteractiveSampling);
    controls.addEventListener("end", markInteractionSettled);
    requestInteractiveRenderRef.current = resetInteractiveSampling;
    requestRenderRef.current = scheduleRender;
    clipUniformsRef.current = {
      uClipMin: (material.uniforms as Record<string, { value: THREE.Vector3 }>).uClipMin,
      uClipMax: (material.uniforms as Record<string, { value: THREE.Vector3 }>).uClipMax,
      uCutawayActive: (material.uniforms as Record<string, { value: boolean }>).uCutawayActive,
    };
    if (resolvedSource.kind === "multichannel") {
      // Point the incremental config effect at this material's live uniforms.
      multichannelUniformsRef.current = material.uniforms as unknown as MultichannelUniformSet;
    } else if (resolvedSource.kind === "atlas") {
      atlasRenderModeUniformRef.current = (
        material.uniforms as Record<string, { value: number }>
      ).uAtlasRenderMode;
    }

    const resize = ({ resetQuality = false }: { resetQuality?: boolean } = {}) => {
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
      if (resetQuality) {
        resetInteractiveSampling();
      } else {
        scheduleRender();
      }
    };

    const observer = new ResizeObserver(() => resize({ resetQuality: true }));
    observer.observe(container);

    let texture3D: THREE.Data3DTexture | null = null;
    const channelTextures: THREE.Data3DTexture[] = [];
    const preparedResidencyLeases: PreparedScalarVolumeLease[] = [];
    let preparedResidencyReservation: PreparedScalarVolumeReservation | null = null;

    if (resolvedSource.kind === "multichannel") {
      const samplerNames = ["uChan0", "uChan1", "uChan2", "uChan3", "uChan4", "uChan5", "uChan6", "uChan7"];
      const channelIndices = resolvedSource.channelIndices;
      const multichannelSource = resolvedSource;
      const expectedSourceGrid = {
        width: Math.max(1, multichannelSource.axisSizes.X),
        height: Math.max(1, multichannelSource.axisSizes.Y),
        depth: Math.max(1, multichannelSource.axisSizes.Z),
      };
      const channelKeys = channelIndices.map((channel) =>
        scalarVolumeIdentityKey({
          apiNamespace: volumeApiNamespace,
          fileId: fileId ?? viewerInfo?.file_id ?? "external",
          sourceIdentity: volumeSourceIdentity,
          sourceGrid: expectedSourceGrid,
          channel,
          time: requestedTime,
          channelCount: multichannelSource.axisSizes.C,
          timeCount: multichannelSource.axisSizes.T,
          policy: SCALAR_PREVIEW_POLICY,
        })
      );
      const loadPayload = async (channel: number): Promise<ScalarVolumePayload> => {
        const payload = await multichannelSource.loadChannel(channel, volumeLoadController.signal);
        throwIfVolumeLoadAborted(volumeLoadController.signal);
        validateScalarVolumeIdentity(payload, {
          channel,
          time: requestedTime,
          sourceGrid: expectedSourceGrid,
          policy: SCALAR_PREVIEW_POLICY,
        });
        validateVolumeTextureGrid(payload, max3DTextureSize);
        return payload;
      };
      void (async () => {
        const candidates: Array<{
          channel: number;
          key: string;
          prepared: PreparedScalarVolume;
          publish: boolean;
        }> = [];
        if (!disposed) {
          setChannelLoadProgress({ loaded: 0, total: channelIndices.length });
        }
        const firstChannel = channelIndices[0] as number;
        const firstKey = channelKeys[0] as string;
        const firstCached = scalarVolumeResidency.get(firstKey);
        const firstPayload = firstCached ? null : await loadPayload(firstChannel);
        throwIfVolumeLoadAborted(volumeLoadController.signal);
        const firstGeometry = firstCached ?? (firstPayload as ScalarVolumePayload);
        if (firstCached) {
          validateScalarVolumeIdentity(firstCached, {
            channel: firstChannel,
            time: requestedTime,
            sourceGrid: expectedSourceGrid,
            policy: SCALAR_PREVIEW_POLICY,
          });
          validateVolumeTextureGrid(firstCached, max3DTextureSize);
        }
        const predictedPreparedBytes = preparedScalarVolumeByteLength(firstGeometry);
        validateSelectedChannelPreparedBudget(
          predictedPreparedBytes,
          channelIndices.length
        );
        validateActiveScalarGpuBudget(predictedPreparedBytes, channelIndices.length);
        preparedResidencyReservation = scalarVolumeResidency.reserve(
          channelKeys,
          predictedPreparedBytes
        );
        const deliveredGrid: [number, number, number] = [
          firstGeometry.width,
          firstGeometry.height,
          firstGeometry.depth,
        ];
        updateDeliveredSampleBudget({
          width: deliveredGrid[0],
          height: deliveredGrid[1],
          depth: deliveredGrid[2],
        });

        for (let slot = 0; slot < channelIndices.length; slot += 1) {
          const channel = channelIndices[slot] as number;
          const key = channelKeys[slot] as string;
          const cachedLease = scalarVolumeResidency.acquire(key);
          if (cachedLease) {
            validateScalarVolumeIdentity(cachedLease.value, {
              channel,
              time: requestedTime,
              sourceGrid: expectedSourceGrid,
              policy: SCALAR_PREVIEW_POLICY,
            });
            validateVolumeTextureGrid(cachedLease.value, max3DTextureSize);
            if (cachedLease.value.textureData.byteLength !== predictedPreparedBytes) {
              cachedLease.release();
              throw new RangeError("Scalar volume channel previews have inconsistent byte sizes.");
            }
            const cachedGrid = [
              cachedLease.value.width,
              cachedLease.value.height,
              cachedLease.value.depth,
            ];
            if (cachedGrid.some((value, axis) => value !== deliveredGrid[axis])) {
              cachedLease.release();
              throw new RangeError("Scalar volume channel previews have inconsistent delivered grids.");
            }
            preparedResidencyLeases.push(cachedLease);
            candidates.push({ channel, key, prepared: cachedLease.value, publish: false });
            setChannelLoadProgress({ loaded: slot + 1, total: channelIndices.length });
            continue;
          }
          const payload = slot === 0 && firstPayload ? firstPayload : await loadPayload(channel);
          throwIfVolumeLoadAborted(volumeLoadController.signal);
          const payloadBytes = preparedScalarVolumeByteLength(payload);
          const nextGrid = [payload.width, payload.height, payload.depth];
          if (
            payloadBytes !== predictedPreparedBytes ||
            nextGrid.some((value, axis) => value !== deliveredGrid[axis])
          ) {
            throw new RangeError(
              "Scalar volume channel previews have inconsistent delivered grids or byte sizes."
            );
          }
          const prepared = await prepareScalarVolume(payload, volumeLoadController.signal);
          throwIfVolumeLoadAborted(volumeLoadController.signal);
          if (prepared.textureData.byteLength !== predictedPreparedBytes) {
            throw new RangeError("Scalar volume conversion did not honor its prepared-byte reservation.");
          }
          candidates.push({ channel, key, prepared, publish: true });
          setChannelLoadProgress({ loaded: slot + 1, total: channelIndices.length });
        }

        throwIfVolumeLoadAborted(volumeLoadController.signal);
        for (const candidate of candidates) {
          if (candidate.publish) {
            const lease = scalarVolumeResidency.publishAndAcquire(
              preparedResidencyReservation,
              candidate.key,
              candidate.prepared
            );
            candidate.prepared = lease.value;
            preparedResidencyLeases.push(lease);
          }
        }
        preparedResidencyReservation.release();
        preparedResidencyReservation = null;

        const localAutoWindows = new Map<number, { low: number; high: number }>();
        for (const candidate of candidates) {
          localAutoWindows.set(candidate.channel, candidate.prepared.autoWindow);
          const texture = preparedScalarToVolumeTexture(candidate.prepared, texturePolicy);
          channelTextures.push(texture);
          if (typeof renderer.initTexture === "function") {
            renderer.initTexture(texture);
          }
        }
        throwIfVolumeLoadAborted(volumeLoadController.signal);

        const finalGrid = deliveredGrid;
        const deliveredSpacing = computeDeliveredVoxelSpacing({
          sourceGrid: expectedSourceGrid,
          sourceSpacing: physicalGeometry.voxelSpacing,
          deliveredGrid: { width: finalGrid[0], height: finalGrid[1], depth: finalGrid[2] },
        });
        const maxSpacing = Math.max(
          deliveredSpacing.x,
          deliveredSpacing.y,
          deliveredSpacing.z,
          1e-9
        );
        const uniforms = material.uniforms as Record<string, { value: unknown }>;
        const deliveredVoxelStep = computeVolumeVoxelStep({
          width: finalGrid[0],
          height: finalGrid[1],
          depth: finalGrid[2],
        });
        candidates.forEach((candidate, slot) => {
          uniforms[samplerNames[slot]].value = channelTextures[slot];
        });
        (uniforms.uGridSize as { value: THREE.Vector3 }).value.set(...finalGrid);
        (uniforms.uVoxelStep as { value: THREE.Vector3 }).value.set(
          deliveredVoxelStep.x,
          deliveredVoxelStep.y,
          deliveredVoxelStep.z
        );
        (uniforms.uVoxelSpacing as { value: THREE.Vector3 }).value.set(
          deliveredSpacing.x / maxSpacing,
          deliveredSpacing.y / maxSpacing,
          deliveredSpacing.z / maxSpacing
        );
        channelAutoWindowsRef.current = localAutoWindows;
        const config = multichannelRenderConfigRef.current;
        if (config) {
          applyMultichannelChannelUniforms(
            material.uniforms as unknown as MultichannelUniformSet,
            config,
            channelAutoWindowsRef.current
          );
        }
        (uniforms.uChannelCount as { value: number }).value = candidates.length;
        material.needsUpdate = true;
        resize({ resetQuality: true });
        if (!disposed) {
          setChannelLoadProgress(null);
        }
      })()
        .catch((error: unknown) => {
          preparedResidencyReservation?.release();
          preparedResidencyReservation = null;
          preparedResidencyLeases.splice(0).forEach((lease) => lease.release());
          if (disposed || (error instanceof DOMException && error.name === "AbortError")) {
            return;
          }
          setChannelLoadProgress(null);
          setRenderFailure({
            sourceKey: renderSourceKey,
            message: error instanceof Error ? error.message : "Volume channels failed to load",
          });
        });
    } else {
      const loadPromise =
      resolvedSource.kind === "scalar"
        ? (async () => {
            const sourceGrid = {
              width: resolvedSource.axisSizes.X,
              height: resolvedSource.axisSizes.Y,
              depth: resolvedSource.axisSizes.Z,
            };
            if (scalarChannel == null) {
              throw new RangeError("Scalar volume channel identity is unavailable.");
            }
            const textureEncoding: ScalarVolumePreparationEncoding =
              effectiveMaskMode ? "raw-integer" : "normalized-half";
            const expectedPreviewPolicy = effectiveMaskMode
              ? SCALAR_MASK_NATIVE_POLICY
              : null;
            const preparedKey = scalarVolumeIdentityKey({
              apiNamespace: volumeApiNamespace,
              fileId: fileId ?? viewerInfo?.file_id ?? "external",
              sourceIdentity: volumeSourceIdentity,
              sourceGrid,
              channel: scalarChannel,
              time: requestedTime,
              channelCount: resolvedSource.axisSizes.C,
              timeCount: resolvedSource.axisSizes.T,
              policy: `${
                expectedPreviewPolicy ?? SCALAR_PREVIEW_POLICY
              }:${String(viewerInfo?.metadata.array_dtype ?? "")}:${textureEncoding}`,
            });
            let preparedLease = scalarVolumeResidency.acquire(preparedKey);
            if (preparedLease) {
              validateScalarVolumeIdentity(preparedLease.value, {
                channel: scalarChannel,
                time: requestedTime,
                sourceGrid,
                policy: expectedPreviewPolicy ?? preparedLease.value.previewPolicy,
              });
              validateVolumeTextureGrid(preparedLease.value, max3DTextureSize);
              if (effectiveMaskMode) {
                validateExactMaskVolume(preparedLease.value, {
                  sourceGrid,
                  dtype: String(viewerInfo?.scalar_mask_capability?.dtype ?? ""),
                  max3DTextureSize,
                });
              }
              validateSelectedChannelPreparedBudget(
                preparedLease.value.textureData.byteLength,
                1
              );
              validateActiveScalarGpuBudget(
                preparedLease.value.textureData.byteLength,
                1
              );
              updateDeliveredSampleBudget({
                width: preparedLease.value.width,
                height: preparedLease.value.height,
                depth: preparedLease.value.depth,
              });
            } else {
              const payload = await resolvedSource.loadScalarVolume(
                volumeLoadController.signal
              );
              throwIfVolumeLoadAborted(volumeLoadController.signal);
              validateVolumeTextureGrid(payload, max3DTextureSize);
              validateScalarVolumeIdentity(payload, {
                channel: scalarChannel,
                time: requestedTime,
                sourceGrid,
                policy: expectedPreviewPolicy ?? payload.previewPolicy,
              });
              if (effectiveMaskMode) {
                validateExactMaskVolume(payload, {
                  sourceGrid,
                  dtype: String(viewerInfo?.scalar_mask_capability?.dtype ?? ""),
                  max3DTextureSize,
                });
              }
              if (
                effectiveMaskMode &&
                String(payload.dtype).trim().toLowerCase() !==
                  String(viewerInfo?.scalar_mask_capability?.dtype ?? "")
                    .trim()
                    .toLowerCase()
              ) {
                throw new RangeError("Exact Mask volume dtype does not match its advertised capability.");
              }
              const predictedPreparedBytes = preparedScalarVolumeByteLength(
                payload,
                effectiveMaskMode
                  ? payload.bytesPerVoxel
                  : Uint16Array.BYTES_PER_ELEMENT
              );
              validateSelectedChannelPreparedBudget(predictedPreparedBytes, 1);
              validateActiveScalarGpuBudget(predictedPreparedBytes, 1);
              updateDeliveredSampleBudget({
                width: payload.width,
                height: payload.height,
                depth: payload.depth,
              });
              preparedResidencyReservation = scalarVolumeResidency.reserve(
                [preparedKey],
                predictedPreparedBytes
              );
              preparedLease = scalarVolumeResidency.acquire(preparedKey);
              if (preparedLease) {
                validateScalarVolumeIdentity(preparedLease.value, {
                  channel: scalarChannel,
                  time: requestedTime,
                  sourceGrid,
                  policy: expectedPreviewPolicy ?? preparedLease.value.previewPolicy,
                });
                if (effectiveMaskMode) {
                  validateExactMaskVolume(preparedLease.value, {
                    sourceGrid,
                    dtype: String(viewerInfo?.scalar_mask_capability?.dtype ?? ""),
                    max3DTextureSize,
                  });
                }
                if (
                  preparedLease.value.textureData.byteLength !==
                  predictedPreparedBytes
                ) {
                  preparedLease.release();
                  throw new RangeError(
                    "Cached scalar volume byte size does not match its identity."
                  );
                }
              } else {
                const prepared = await prepareScalarVolume(
                  payload,
                  volumeLoadController.signal,
                  textureEncoding
                );
                throwIfVolumeLoadAborted(volumeLoadController.signal);
                if (effectiveMaskMode) {
                  validateExactMaskVolume(prepared, {
                    sourceGrid,
                    dtype: String(viewerInfo?.scalar_mask_capability?.dtype ?? ""),
                    max3DTextureSize,
                  });
                }
                preparedLease = scalarVolumeResidency.publishAndAcquire(
                  preparedResidencyReservation,
                  preparedKey,
                  prepared
                );
              }
              preparedResidencyReservation.release();
              preparedResidencyReservation = null;
            }
            preparedResidencyLeases.push(preparedLease);
            const prepared = preparedLease.value;
            const texture = effectiveMaskMode
              ? preparedMaskToIntegerTexture(prepared)
              : preparedScalarToVolumeTexture(prepared, texturePolicy);
            throwIfVolumeLoadAborted(volumeLoadController.signal);
            const scalarMaterialUniforms = material.uniforms as Record<
              string,
              { value: THREE.Vector3 }
            >;
            scalarMaterialUniforms.uGridSize.value.set(
              prepared.width,
              prepared.height,
              prepared.depth
            );
            const deliveredSpacing = computeDeliveredVoxelSpacing({
              sourceGrid: {
                width: resolvedSource.axisSizes.X,
                height: resolvedSource.axisSizes.Y,
                depth: resolvedSource.axisSizes.Z,
              },
              sourceSpacing: physicalGeometry.voxelSpacing,
              deliveredGrid: {
                width: prepared.width,
                height: prepared.height,
                depth: prepared.depth,
              },
            });
            const maxSpacing = Math.max(
              deliveredSpacing.x,
              deliveredSpacing.y,
              deliveredSpacing.z,
              1e-9
            );
            scalarMaterialUniforms.uVoxelSpacing.value.set(
              deliveredSpacing.x / maxSpacing,
              deliveredSpacing.y / maxSpacing,
              deliveredSpacing.z / maxSpacing
            );
            if (effectiveMaskMode) {
              const thresholdUniform = (
                material.uniforms as Record<string, { value: number }>
              ).uMaskThreshold;
              thresholdUniform.value = Math.trunc(
                scalarMaskConfigRef.current.rawThreshold
              );
              maskThresholdUniformRef.current = thresholdUniform;
              return texture;
            }
            const latestScalarRenderConfig = scalarRenderConfigRef.current;
            const normalizedWindow = resolveScalarVolumeWindow(
              prepared,
              latestScalarRenderConfig.enhancement,
              latestScalarRenderConfig.negative
            );
            const nextScalarUniforms = {
              uWindowLow: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowLow as { value: number },
              uWindowHigh: (material.uniforms as Record<string, { value: number | boolean | null }>).uWindowHigh as { value: number },
              uInvert: (material.uniforms as Record<string, { value: number | boolean | null }>).uInvert as { value: boolean },
              uColorMap: (material.uniforms as Record<string, { value: number | boolean | null }>).uColorMap as { value: number },
              uSignalFloor: (material.uniforms as Record<string, { value: number | boolean | null }>).uSignalFloor as { value: number },
              uDensityScale: (material.uniforms as Record<string, { value: number | boolean | null }>).uDensityScale as { value: number },
              uLightingEnabled: (material.uniforms as Record<string, { value: number | boolean | null }>).uLightingEnabled as { value: boolean },
              uLightingStrength: (material.uniforms as Record<string, { value: number | boolean | null }>).uLightingStrength as { value: number },
            };
            nextScalarUniforms.uWindowLow.value = normalizedWindow.low;
            nextScalarUniforms.uWindowHigh.value = normalizedWindow.high;
            nextScalarUniforms.uInvert.value = normalizedWindow.invert;
            nextScalarUniforms.uColorMap.value = latestScalarRenderConfig.colorMapShaderValue;
            nextScalarUniforms.uSignalFloor.value = latestScalarRenderConfig.signalFloor;
            nextScalarUniforms.uDensityScale.value = latestScalarRenderConfig.densityScale;
            nextScalarUniforms.uLightingEnabled.value = latestScalarRenderConfig.lightingEnabled;
            nextScalarUniforms.uLightingStrength.value = latestScalarRenderConfig.lightingStrength;
            const deliveredVoxelStep = computeVolumeVoxelStep({
              width: prepared.width,
              height: prepared.height,
              depth: prepared.depth,
            });
            scalarMaterialUniforms.uVoxelStep.value.set(
              deliveredVoxelStep.x,
              deliveredVoxelStep.y,
              deliveredVoxelStep.z
            );
            throwIfVolumeLoadAborted(volumeLoadController.signal);
            scalarRangeRef.current = {
              rawMin: prepared.rawMin,
              rawMax: prepared.rawMax,
              sclSlope: prepared.sclSlope,
              sclInter: prepared.sclInter,
            };
            scalarUniformsRef.current = nextScalarUniforms;
            return texture;
          })()
        : atlasToVolumeTexture(
            resolvedSource.atlasUrl,
            resolvedSource.atlasScheme,
            texturePolicy,
            volumeLoadController.signal
          );

    void loadPromise
      .then((decodedTexture) => {
        if (disposed || volumeLoadController.signal.aborted) {
          decodedTexture.dispose();
          return;
        }
        texture3D = decodedTexture;
        if (typeof renderer.initTexture === "function") {
          renderer.initTexture(decodedTexture);
        }
        material.uniforms.uData.value = decodedTexture;
        mesh.visible = true;
        container.dataset.viewerMaskTextureReady =
          effectiveMaskMode ? "true" : "not-applicable";
        material.needsUpdate = true;
        resize({ resetQuality: true });
      })
      .catch((error: unknown) => {
        preparedResidencyReservation?.release();
        preparedResidencyReservation = null;
        preparedResidencyLeases.splice(0).forEach((lease) => lease.release());
        if (disposed || (error instanceof DOMException && error.name === "AbortError")) {
          return;
        }
        setRenderFailure({
          sourceKey: renderSourceKey,
          message: error instanceof Error ? error.message : "Volume data failed to load",
        });
      });
    }

    resize({ resetQuality: true });
    scheduleRender();

    return () => {
      disposed = true;
      volumeLoadController.abort();
      preparedResidencyReservation?.release();
      preparedResidencyReservation = null;
      preparedResidencyLeases.splice(0).forEach((lease) => lease.release());
      cameraSnapshotRef.current = cameraSnapshotKey
        ? {
            key: cameraSnapshotKey,
            position: camera.position.clone(),
            target: controls.target.clone(),
            up: camera.up.clone(),
            zoom: camera.zoom,
          }
        : null;
      requestInteractiveRenderRef.current = null;
      requestRenderRef.current = null;
      scalarUniformsRef.current = null;
      maskThresholdUniformRef.current = null;
      multichannelUniformsRef.current = null;
      clipUniformsRef.current = null;
      atlasRenderModeUniformRef.current = null;
      cameraRigRef.current = null;
      sliceCursorPlanesRef.current = null;
      scalarRangeRef.current = null;
      observer.disconnect();
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      if (rampTimer) {
        window.clearTimeout(rampTimer);
      }
      controls.removeEventListener("start", beginInteraction);
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
      channelTextures.forEach((texture) => texture?.dispose());
      renderer.dispose();
      renderer.domElement.parentNode?.removeChild(renderer.domElement);
    };
  }, [
    activeRenderError,
    effectiveMaskMode,
    resolvedSource,
    texturePolicy,
    clearColor,
    density,
    volumeEdgeStrength,
    volumeInteriorOpacity,
    projectionMode,
    renderPolicy,
    featureMask,
    sampleBudget.interactiveSteps,
    sampleBudget.rampFactor,
    sampleBudget.settledSteps,
    axisSizes.X,
    axisSizes.Y,
    axisSizes.Z,
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
    cameraSnapshotKey,
    volumeViewPreset,
    volumeInteriorCameraFrame,
    volumeInteriorInspectionActive,
    physicalGeometry.worldDepth,
    physicalGeometry.worldHeight,
    physicalGeometry.worldWidth,
    physicalGeometry.voxelSpacing.x,
    physicalGeometry.voxelSpacing.y,
    physicalGeometry.voxelSpacing.z,
    physicalGeometry.voxelSpacing,
    volumeClipCue.active,
    fileId,
    requestedTime,
    scalarChannel,
    volumeApiNamespace,
    volumeSourceIdentity,
    viewerInfo?.file_id,
    viewerInfo?.metadata.array_dtype,
    viewerInfo?.scalar_mask_capability?.dtype,
    isPerChannelVolume,
    retryGeneration,
    renderSourceKey,
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
        maskMode: effectiveMaskMode,
      }),
    });
    requestInteractiveRenderRef.current?.();
  }, [
    normalizedScale,
    sliceCursorCue,
    volumeInteriorInspectionActive,
    cutawayActive,
    isPerChannelVolume,
    effectiveMaskMode,
  ]);

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
    activeRenderError ||
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
        data-viewer-atlas-render-mode={resolvedSource?.kind === "atlas" ? atlasRenderMode.id : undefined}
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
            maskMode: effectiveMaskMode,
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
           {fallbackImageUrl ? (
             <img src={fallbackImageUrl} alt="Volume fallback preview" className="viewer-image-fallback-media" />
           ) : null}
         </div>
        {renderVolumeOrientationOverlay("fallback")}
        {renderVolumeAxisCue()}
        {renderVolumeSliceCursorCue()}
        {renderVolumeClipCue()}
        {renderVolumeScaleBar()}
        <p className="viewer-fallback-note">
          {activeRenderError
            ? `${backendLabel === "scalar" ? "Scalar" : "Atlas"} volume viewer unavailable: ${activeRenderError}. Showing a representative slice preview instead.`
            : "Volume viewer unavailable. Showing a representative slice preview instead."}
        </p>
        {renderError && !volumeSelection.error ? (
          <button
            type="button"
            className="viewer-fallback-retry"
            onClick={() => {
              setRenderFailure(null);
              setRetryGeneration((generation) => generation + 1);
            }}
          >
            Retry volume
          </button>
        ) : null}
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
      data-viewer-atlas-render-mode={resolvedSource.kind === "atlas" ? atlasRenderMode.id : undefined}
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
          maskMode: effectiveMaskMode,
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
      {channelLoadProgress ? (
        <span className="viewer-volume-load-progress" role="status" aria-live="polite">
          Loading channels {channelLoadProgress.loaded}/{channelLoadProgress.total}
        </span>
      ) : null}
      {renderVolumeOrientationOverlay()}
      {renderVolumeAxisCue()}
      {renderVolumeSliceCursorCue()}
      {renderVolumeClipCue()}
      {renderVolumeScaleBar()}
    </div>
  );
}
