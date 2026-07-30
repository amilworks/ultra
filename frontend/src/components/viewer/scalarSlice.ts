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

export type ScalarNearestAxisMapping = {
  deliveryIndex: number;
  sampledSourceIndex: number;
  exact: boolean;
};

export type ScalarNearestVoxelMapping = {
  delivery: { x: number; y: number; z: number };
  sampledSource: { x: number; y: number; z: number };
  exact: boolean;
};

export type ScalarPlanePoint = {
  row: number;
  col: number;
};

export type ScalarNearestPlaneMapping = {
  delivery: ScalarPlanePoint;
  sampledSource: ScalarPlanePoint;
  exact: boolean;
};

const MAX_SCALAR_SLICE_PIXELS = 16 * 1024 * 1024;

const safeDim = (value: number): number => Math.max(1, Math.floor(Number(value) || 1));

const nearestAxisGeometry = (
  payload: ScalarVolumePayload,
  axis: ScalarSliceAxis
): { sourceExtent: number; deliveryExtent: number; factor: number } => {
  if (axis === "x") {
    return {
      sourceExtent: safeDim(payload.sourceWidth),
      deliveryExtent: safeDim(payload.width),
      factor: safeDim(payload.downsampleX),
    };
  }
  if (axis === "y") {
    return {
      sourceExtent: safeDim(payload.sourceHeight),
      deliveryExtent: safeDim(payload.height),
      factor: safeDim(payload.downsampleY),
    };
  }
  return {
    sourceExtent: safeDim(payload.sourceDepth),
    deliveryExtent: safeDim(payload.depth),
    factor: safeDim(payload.downsampleZ),
  };
};

export const mapSourceSliceToNearestDelivery = (
  payload: ScalarVolumePayload,
  axis: ScalarSliceAxis,
  sourceIndex: number
): ScalarNearestAxisMapping => {
  const { sourceExtent, deliveryExtent, factor } = nearestAxisGeometry(payload, axis);
  const clampedSourceIndex = Math.max(
    0,
    Math.min(sourceExtent - 1, Math.floor(Number(sourceIndex) || 0))
  );
  const deliveryIndex = Math.max(
    0,
    Math.min(deliveryExtent - 1, Math.floor(clampedSourceIndex / factor))
  );
  const sampledSourceIndex = Math.min(
    sourceExtent - 1,
    deliveryIndex * factor + Math.floor(factor / 2)
  );
  return {
    deliveryIndex,
    sampledSourceIndex,
    exact: clampedSourceIndex === sampledSourceIndex,
  };
};

export const mapSourceVoxelToNearestDelivery = (
  payload: ScalarVolumePayload,
  source: { x: number; y: number; z: number }
): ScalarNearestVoxelMapping => {
  const x = mapSourceSliceToNearestDelivery(payload, "x", source.x);
  const y = mapSourceSliceToNearestDelivery(payload, "y", source.y);
  const z = mapSourceSliceToNearestDelivery(payload, "z", source.z);
  return {
    delivery: {
      x: x.deliveryIndex,
      y: y.deliveryIndex,
      z: z.deliveryIndex,
    },
    sampledSource: {
      x: x.sampledSourceIndex,
      y: y.sampledSourceIndex,
      z: z.sampledSourceIndex,
    },
    exact: x.exact && y.exact && z.exact,
  };
};

const scalarPlaneAxes = (
  axis: ScalarSliceAxis
): { row: ScalarSliceAxis; col: ScalarSliceAxis } => {
  if (axis === "y") {
    return { row: "z", col: "x" };
  }
  if (axis === "x") {
    return { row: "z", col: "y" };
  }
  return { row: "y", col: "x" };
};

const mapNearestDeliveryIndexToSource = (
  payload: ScalarVolumePayload,
  axis: ScalarSliceAxis,
  deliveryIndex: number
): number => {
  const { sourceExtent, deliveryExtent, factor } = nearestAxisGeometry(
    payload,
    axis
  );
  const clampedDeliveryIndex = Math.max(
    0,
    Math.min(
      deliveryExtent - 1,
      Math.floor(Number(deliveryIndex) || 0)
    )
  );
  return Math.min(
    sourceExtent - 1,
    clampedDeliveryIndex * factor + Math.floor(factor / 2)
  );
};

export const mapSourcePlanePointToNearestDelivery = (
  payload: ScalarVolumePayload,
  axis: ScalarSliceAxis,
  point: ScalarPlanePoint
): ScalarNearestPlaneMapping => {
  const axes = scalarPlaneAxes(axis);
  const row = mapSourceSliceToNearestDelivery(payload, axes.row, point.row);
  const col = mapSourceSliceToNearestDelivery(payload, axes.col, point.col);
  return {
    delivery: {
      row: row.deliveryIndex,
      col: col.deliveryIndex,
    },
    sampledSource: {
      row: row.sampledSourceIndex,
      col: col.sampledSourceIndex,
    },
    exact: row.exact && col.exact,
  };
};

export const mapNearestDeliveryPlanePointToSource = (
  payload: ScalarVolumePayload,
  axis: ScalarSliceAxis,
  point: ScalarPlanePoint
): ScalarPlanePoint => {
  const axes = scalarPlaneAxes(axis);
  return {
    row: mapNearestDeliveryIndexToSource(payload, axes.row, point.row),
    col: mapNearestDeliveryIndexToSource(payload, axes.col, point.col),
  };
};

export const scalarPayloadUsesNativeGrid = (payload: ScalarVolumePayload): boolean =>
  safeDim(payload.downsampleX) === 1 &&
  safeDim(payload.downsampleY) === 1 &&
  safeDim(payload.downsampleZ) === 1 &&
  safeDim(payload.sourceWidth) === safeDim(payload.width) &&
  safeDim(payload.sourceHeight) === safeDim(payload.height) &&
  safeDim(payload.sourceDepth) === safeDim(payload.depth);

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
