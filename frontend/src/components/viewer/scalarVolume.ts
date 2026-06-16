import { DataUtils } from "three";

import type { ScalarVolumePayload } from "@/lib/api";

export type ScalarVolumeVoxelIndex = {
  x: number;
  y: number;
  z: number;
};

const safePositiveInteger = (value: number): number =>
  Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));

const safeVoxelIndex = (value: number): number =>
  Number.isFinite(value) ? Math.floor(value) : -1;

const scalarVolumeOffset = (
  payload: ScalarVolumePayload,
  index: ScalarVolumeVoxelIndex
): { offset: number; bytesPerVoxel: number } | null => {
  const width = safePositiveInteger(payload.width);
  const height = safePositiveInteger(payload.height);
  const depth = safePositiveInteger(payload.depth);
  const x = safeVoxelIndex(index.x);
  const y = safeVoxelIndex(index.y);
  const z = safeVoxelIndex(index.z);
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);

  if (
    width < 1 ||
    height < 1 ||
    depth < 1 ||
    x < 0 ||
    y < 0 ||
    z < 0 ||
    x >= width ||
    y >= height ||
    z >= depth
  ) {
    return null;
  }

  const voxelIndex = (z * height + y) * width + x;
  const offset = voxelIndex * bytesPerVoxel;
  if (offset + bytesPerVoxel > payload.data.byteLength) {
    return null;
  }
  return { offset, bytesPerVoxel };
};

const scalarVolumeSampleValue = (view: DataView, offset: number, payload: ScalarVolumePayload): number => {
  const dtype = String(payload.dtype ?? "").trim().toLowerCase();
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);
  if (dtype === "float32" || bytesPerVoxel === 4) {
    return view.getFloat32(offset, true);
  }
  if (dtype === "int16") {
    return view.getInt16(offset, true);
  }
  if (dtype === "uint16" || bytesPerVoxel === 2) {
    return view.getUint16(offset, true);
  }
  return view.getUint8(offset);
};

export const scalarVolumePayloadValueAt = (
  payload: ScalarVolumePayload,
  index: ScalarVolumeVoxelIndex
): number | null => {
  const location = scalarVolumeOffset(payload, index);
  if (!location) {
    return null;
  }
  const view = new DataView(payload.data);
  return scalarVolumeSampleValue(view, location.offset, payload);
};

const scalarVolumeNormalizationRange = (
  payload: ScalarVolumePayload
): { rawMin: number; range: number } => {
  const rawMin = Number.isFinite(payload.rawMin) ? payload.rawMin : 0;
  const rawMax = Number.isFinite(payload.rawMax) && payload.rawMax > rawMin ? payload.rawMax : rawMin + 1;
  return { rawMin, range: rawMax - rawMin };
};

const scalarVolumeOutputLength = (payload: ScalarVolumePayload): number => {
  const voxelCount =
    safePositiveInteger(payload.width) *
    safePositiveInteger(payload.height) *
    safePositiveInteger(payload.depth);
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);
  const availableVoxels = Math.floor(payload.data.byteLength / bytesPerVoxel);
  return voxelCount > 0 ? Math.min(voxelCount, availableVoxels) : availableVoxels;
};

/**
 * Auto-contrast window for a scalar volume in volume-rendering use, returned in
 * the normalized [0,1] space the GPU texture stores. Unlike a 2D display
 * auto-contrast (which brackets the full data range), a *volume* needs the low
 * cutoff ABOVE the background bulk: even a small per-voxel background opacity
 * accumulates into a flat fog over a deep z-stack. So we take a high low-percentile
 * (the background of fluorescence is the dominant low-intensity mode) and a high
 * high-percentile, computed from a subsampled histogram.
 */
export const computeScalarVolumeAutoContrast = (
  payload: ScalarVolumePayload,
  lowPercentile = 0.78,
  highPercentile = 0.995
): { low: number; high: number } => {
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  const voxelCount = scalarVolumeOutputLength(payload);
  const BINS = 512;
  const histogram = new Float64Array(BINS);
  // Subsample for speed; a histogram is robust to it (cap ~2M samples).
  const stride = Math.max(1, Math.floor(voxelCount / 2_000_000));
  let sampled = 0;
  for (let voxelIndex = 0; voxelIndex < voxelCount; voxelIndex += stride) {
    const value = scalarVolumeSampleValue(view, voxelIndex * bytesPerVoxel, payload);
    const finite = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(0.999999, (finite - rawMin) / range));
    histogram[Math.floor(normalized * BINS)] += 1;
    sampled += 1;
  }
  if (sampled === 0) {
    return { low: 0, high: 1 };
  }
  const percentileBin = (p: number): number => {
    const target = sampled * Math.max(0, Math.min(1, p));
    let cumulative = 0;
    for (let i = 0; i < BINS; i += 1) {
      cumulative += histogram[i];
      if (cumulative >= target) {
        return i;
      }
    }
    return BINS - 1;
  };
  const low = percentileBin(lowPercentile) / BINS;
  const highBin = percentileBin(highPercentile);
  const high = Math.min(1, (highBin + 1) / BINS);
  // Guarantee a usable, non-degenerate window.
  if (high <= low) {
    return { low: Math.min(low, 0.95), high: 1 };
  }
  return { low, high };
};

export const scalarVolumePayloadToTextureBytes = (payload: ScalarVolumePayload): Uint8Array => {
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);
  const output = new Uint8Array(scalarVolumeOutputLength(payload));
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  for (let voxelIndex = 0; voxelIndex < output.length; voxelIndex += 1) {
    const offset = voxelIndex * bytesPerVoxel;
    const value = scalarVolumeSampleValue(view, offset, payload);
    const finiteValue = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(1, (finiteValue - rawMin) / range));
    output[voxelIndex] = Math.max(0, Math.min(255, Math.round(normalized * 255)));
  }
  return output;
};

/**
 * Convert a scalar volume to normalized 16-bit half-float samples for GPU upload.
 *
 * Medical CT/MRI volumes span a huge raw range (air ~-1024 HU through bone/metal
 * >3000 HU). Quantizing that to 8 bits collapses the entire brain-tissue band
 * (CSF ~0-15 HU, parenchyma ~20-45 HU) into ~3 code values, erasing the
 * ventricle/parenchyma boundary before windowing can ever stretch it. Storing the
 * normalized value at half-float precision keeps ~11 bits of mantissa, i.e.
 * sub-HU resolution across the full range, so soft-tissue contrast survives and
 * window/level stays a cheap GPU uniform. R16F linear filtering is core in WebGL2.
 */
export const scalarVolumePayloadToHalfFloat = (payload: ScalarVolumePayload): Uint16Array => {
  const bytesPerVoxel = Math.max(1, safePositiveInteger(payload.bytesPerVoxel) || 1);
  const output = new Uint16Array(scalarVolumeOutputLength(payload));
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  for (let voxelIndex = 0; voxelIndex < output.length; voxelIndex += 1) {
    const offset = voxelIndex * bytesPerVoxel;
    const value = scalarVolumeSampleValue(view, offset, payload);
    const finiteValue = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(1, (finiteValue - rawMin) / range));
    output[voxelIndex] = DataUtils.toHalfFloat(normalized);
  }
  return output;
};
