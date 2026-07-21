import type { ScalarVolumePayload } from "@/lib/api";

import { resolveScalarStoredWindow } from "./scalarVolume";

/**
 * Client-side slice extraction from a loaded scalar volume.
 *
 * This reproduces the backend `/v2/uploads/{id}/slice` PNG pixel-for-pixel so the
 * viewer can scrub slices and re-window with zero network round-trips. The mapping
 * and window math below intentionally mirror the Go handler
 * (`serveNiftiSliceAsPNG` in backend/controlplane/internal/httpapi/handlers.go):
 *
 *   voxel offset  = ((z * Height + y) * Width + x) * bytesPerVoxel   (row-major)
 *   (row, col) -> (x, y, z):
 *     axis "z" (axial):    x = col,        y = row,        z = sliceIndex
 *     axis "y" (coronal):  x = col,        y = sliceIndex, z = row
 *     axis "x" (sagittal): x = sliceIndex, y = col,        z = row
 *   physical   = sclSlope * storedValue + sclInter
 *   normalized = clamp((physical - windowLow) / (windowHigh - windowLow), 0, 1)
 *   pixel      = round(normalized * 255)          (then 255 - pixel if inverted)
 *
 * There are NO axis flips in the backend, so row 0 is the lower y/z and col 0 the
 * lower x/y — exactly the array order produced here. Feeding the result through
 * `createImageBitmap` matches the flipY the PNG path already uses, so crosshairs,
 * measurements and voxel probing stay aligned.
 */

export type ScalarSliceAxis = "x" | "y" | "z";

export type ScalarSliceRequest = {
  axis: ScalarSliceAxis;
  sliceIndex: number;
  /** Window bounds in physical intensity units (center - width/2 and center + width/2). */
  windowLow: number;
  windowHigh: number;
  invert?: boolean;
};

export type ScalarSliceImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

const MAX_SCALAR_SLICE_PIXELS = 16 * 1024 * 1024;

const safeDim = (value: number): number => Math.max(1, Math.floor(Number(value) || 1));

export const scalarSliceDimensions = (
  payload: Pick<ScalarVolumePayload, "width" | "height" | "depth">,
  axis: ScalarSliceAxis
): { width: number; height: number } => {
  const width = safeDim(payload.width);
  const height = safeDim(payload.height);
  const depth = safeDim(payload.depth);
  if (axis === "z") {
    return { width, height };
  }
  if (axis === "y") {
    return { width, height: depth };
  }
  return { width: height, height: depth };
};

export const scalarSliceAxisExtent = (
  payload: Pick<ScalarVolumePayload, "width" | "height" | "depth">,
  axis: ScalarSliceAxis
): number => {
  if (axis === "z") {
    return safeDim(payload.depth);
  }
  if (axis === "y") {
    return safeDim(payload.height);
  }
  return safeDim(payload.width);
};

type ValueReader = (view: DataView, offset: number) => number;

const valueReaderFor = (dtype: string, bytesPerVoxel: number): ValueReader => {
  const normalized = String(dtype ?? "").trim().toLowerCase();
  if (normalized === "float32" || bytesPerVoxel === 4) {
    return (view, offset) => view.getFloat32(offset, true);
  }
  if (normalized === "int16") {
    return (view, offset) => view.getInt16(offset, true);
  }
  if (normalized === "uint16" || bytesPerVoxel === 2) {
    return (view, offset) => view.getUint16(offset, true);
  }
  return (view, offset) => view.getUint8(offset);
};

export const extractScalarSlice = (
  payload: ScalarVolumePayload,
  request: ScalarSliceRequest
): ScalarSliceImage => {
  const volumeWidth = safeDim(payload.width);
  const volumeHeight = safeDim(payload.height);
  const volumeDepth = safeDim(payload.depth);
  const bytesPerVoxel = Math.max(1, Math.floor(Number(payload.bytesPerVoxel) || 1));
  const readValue = valueReaderFor(payload.dtype, bytesPerVoxel);
  const view = new DataView(payload.data);
  const byteLength = payload.data.byteLength;

  const { width, height } = scalarSliceDimensions(payload, request.axis);
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount <= 0 || pixelCount > MAX_SCALAR_SLICE_PIXELS) {
    throw new RangeError(`Scalar slice pixel count exceeds the ${MAX_SCALAR_SLICE_PIXELS} pixel limit.`);
  }
  const axisExtent = scalarSliceAxisExtent(payload, request.axis);
  const sliceIndex = Math.max(0, Math.min(axisExtent - 1, Math.floor(Number(request.sliceIndex) || 0)));
  const storedWindow = resolveScalarStoredWindow(
    payload,
    request.windowLow,
    request.windowHigh,
    request.invert
  );
  const span = Math.max(1e-6, storedWindow.high - storedWindow.low);
  const invert = storedWindow.invert;

  const data = new Uint8ClampedArray(pixelCount * 4);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      let x: number;
      let y: number;
      let z: number;
      if (request.axis === "z") {
        x = col;
        y = row;
        z = sliceIndex;
      } else if (request.axis === "y") {
        x = col;
        y = sliceIndex;
        z = row;
      } else {
        x = sliceIndex;
        y = col;
        z = row;
      }

      let value = 0;
      if (x >= 0 && y >= 0 && z >= 0 && x < volumeWidth && y < volumeHeight && z < volumeDepth) {
        const offset = ((z * volumeHeight + y) * volumeWidth + x) * bytesPerVoxel;
        if (offset + bytesPerVoxel <= byteLength) {
          const raw = readValue(view, offset);
          value = Number.isFinite(raw) ? raw : 0;
        }
      }

      let normalized = (value - storedWindow.low) / span;
      if (normalized < 0) {
        normalized = 0;
      } else if (normalized > 1) {
        normalized = 1;
      }
      let pixel = Math.round(normalized * 255);
      if (invert) {
        pixel = 255 - pixel;
      }

      const offset = (row * width + col) * 4;
      data[offset] = pixel;
      data[offset + 1] = pixel;
      data[offset + 2] = pixel;
      data[offset + 3] = 255;
    }
  }

  return { width, height, data };
};
