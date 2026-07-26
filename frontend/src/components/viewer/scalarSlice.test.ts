import { describe, expect, it } from "vitest";

import type { ScalarVolumePayload } from "@/lib/api";

import {
  extractScalarSlice,
  mapNearestDeliveryPlanePointToSource,
  mapSourcePlanePointToNearestDelivery,
  mapSourceSliceToNearestDelivery,
  mapSourceVoxelToNearestDelivery,
  scalarPayloadUsesNativeGrid,
  scalarSliceAxisExtent,
  scalarSliceDimensions,
} from "./scalarSlice";

// 2x2x2 volume; value at (x,y,z) follows the row-major order (z*H + y)*W + x.
//   (0,0,0)=10 (1,0,0)=20 (0,1,0)=30 (1,1,0)=40
//   (0,0,1)=50 (1,0,1)=60 (0,1,1)=70 (1,1,1)=80
const makePayload = (): ScalarVolumePayload => ({
  data: new Uint16Array([10, 20, 30, 40, 50, 60, 70, 80]).buffer,
  width: 2,
  height: 2,
  depth: 2,
  dtype: "uint16",
  bytesPerVoxel: 2,
  rawMin: 10,
  rawMax: 80,
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
  sclSlope: 1,
  sclInter: 0,
});

const px = (img: { width: number; data: Uint8ClampedArray }, row: number, col: number): number =>
  img.data[(row * img.width + col) * 4];

const windowed = (value: number): number => Math.round((value / 100) * 255);

describe("extractScalarSlice", () => {
  it("maps the axial (z) plane as pixel[row][col] = voxel(col, row, z)", () => {
    const img = extractScalarSlice(makePayload(), { axis: "z", sliceIndex: 0, windowLow: 0, windowHigh: 100 });

    expect([img.width, img.height]).toEqual([2, 2]);
    expect(px(img, 0, 0)).toBe(windowed(10));
    expect(px(img, 0, 1)).toBe(windowed(20));
    expect(px(img, 1, 0)).toBe(windowed(30));
    expect(px(img, 1, 1)).toBe(windowed(40));
    // grayscale + opaque
    expect(img.data[1]).toBe(img.data[0]);
    expect(img.data[2]).toBe(img.data[0]);
    expect(img.data[3]).toBe(255);
  });

  it("maps the coronal (y) plane as pixel[row][col] = voxel(col, y, row), sized W x D", () => {
    const img = extractScalarSlice(makePayload(), { axis: "y", sliceIndex: 0, windowLow: 0, windowHigh: 100 });

    expect([img.width, img.height]).toEqual([2, 2]);
    expect(px(img, 0, 0)).toBe(windowed(10)); // z=0, x=0
    expect(px(img, 1, 0)).toBe(windowed(50)); // z=1, x=0
  });

  it("maps the sagittal (x) plane as pixel[row][col] = voxel(x, col, row), sized H x D", () => {
    const img = extractScalarSlice(makePayload(), { axis: "x", sliceIndex: 0, windowLow: 0, windowHigh: 100 });

    expect([img.width, img.height]).toEqual([2, 2]);
    expect(px(img, 0, 1)).toBe(windowed(30)); // y=1, z=0, x=0
    expect(px(img, 1, 0)).toBe(windowed(50)); // y=0, z=1, x=0
  });

  it("applies the window bounds and inversion", () => {
    const normal = extractScalarSlice(makePayload(), { axis: "z", sliceIndex: 0, windowLow: 0, windowHigh: 100 });
    const inverted = extractScalarSlice(makePayload(), {
      axis: "z",
      sliceIndex: 0,
      windowLow: 0,
      windowHigh: 100,
      invert: true,
    });
    expect(px(inverted, 0, 0)).toBe(255 - px(normal, 0, 0));

    // A narrow window saturates: values at/below low -> 0, at/above high -> 255.
    const narrow = extractScalarSlice(makePayload(), { axis: "z", sliceIndex: 0, windowLow: 20, windowHigh: 30 });
    expect(px(narrow, 0, 0)).toBe(0); // value 10 < low
    expect(px(narrow, 1, 1)).toBe(255); // value 40 > high
  });

  it("renders exact > threshold membership on every MPR axis for the selected C/T payload", () => {
    const payload: ScalarVolumePayload = {
      ...makePayload(),
      data: new Uint16Array([120, 121, 119, 122, 121, 120, 123, 118]).buffer,
      channel: 2,
      time: 2,
      rawMin: 118,
      rawMax: 123,
      sampling: "nearest",
    };
    const renderMask = (axis: "x" | "y" | "z", sliceIndex: number) =>
      extractScalarSlice(payload, {
        axis,
        sliceIndex,
        windowLow: 120,
        windowHigh: 121,
        invert: false,
      });

    expect([payload.channel, payload.time, payload.sampling]).toEqual([
      2,
      2,
      "nearest",
    ]);
    const axial = renderMask("z", 0);
    expect([px(axial, 0, 0), px(axial, 0, 1), px(axial, 1, 0), px(axial, 1, 1)]).toEqual([
      0,
      255,
      0,
      255,
    ]);
    const coronal = renderMask("y", 0);
    expect([px(coronal, 0, 0), px(coronal, 0, 1), px(coronal, 1, 0), px(coronal, 1, 1)]).toEqual([
      0,
      255,
      255,
      0,
    ]);
    const sagittal = renderMask("x", 0);
    expect([px(sagittal, 0, 0), px(sagittal, 0, 1), px(sagittal, 1, 0), px(sagittal, 1, 1)]).toEqual([
      0,
      0,
      255,
      255,
    ]);
  });

  it("applies a non-unit NIfTI rescale before the physical display window", () => {
    const payload = {
      ...makePayload(),
      sclSlope: 2,
      sclInter: -10,
    };
    const img = extractScalarSlice(payload, {
      axis: "z",
      sliceIndex: 0,
      windowLow: 10,
      windowHigh: 70,
    });

    expect(px(img, 0, 0)).toBe(0); // stored 10 -> physical 10
    expect(px(img, 0, 1)).toBe(Math.round((20 / 60) * 255)); // stored 20 -> physical 30
    expect(px(img, 1, 1)).toBe(255); // stored 40 -> physical 70
  });

  it("reverses display polarity for a negative slope and XORs user inversion", () => {
    const payload = {
      ...makePayload(),
      sclSlope: -2,
      sclInter: 100,
    };
    const normal = extractScalarSlice(payload, {
      axis: "z",
      sliceIndex: 0,
      windowLow: 20,
      windowHigh: 80,
    });
    const inverted = extractScalarSlice(payload, {
      axis: "z",
      sliceIndex: 0,
      windowLow: 20,
      windowHigh: 80,
      invert: true,
    });

    expect(px(normal, 0, 0)).toBe(255); // stored 10 -> physical 80
    expect(px(normal, 1, 1)).toBe(0); // stored 40 -> physical 20
    expect(px(inverted, 0, 0)).toBe(0);
    expect(px(inverted, 1, 1)).toBe(255);
  });

  it("rejects zero or non-finite rescale metadata", () => {
    const payload = {
      ...makePayload(),
      sclSlope: 0,
      sclInter: Number.NaN,
    };
    expect(() =>
      extractScalarSlice(payload, {
        axis: "z",
        sliceIndex: 0,
        windowLow: 0,
        windowHigh: 100,
      })
    ).toThrow(/rescale metadata/i);
  });

  it("clamps an out-of-range slice index to the axis extent", () => {
    const img = extractScalarSlice(makePayload(), { axis: "z", sliceIndex: 99, windowLow: 0, windowHigh: 100 });
    expect(px(img, 0, 0)).toBe(windowed(50)); // clamped to z=1 -> voxel(0,0,1)=50
  });

  it("reports slice dimensions and axis extents per axis", () => {
    const payload = { width: 4, height: 3, depth: 2 };
    expect(scalarSliceDimensions(payload, "z")).toEqual({ width: 4, height: 3 });
    expect(scalarSliceDimensions(payload, "y")).toEqual({ width: 4, height: 2 });
    expect(scalarSliceDimensions(payload, "x")).toEqual({ width: 3, height: 2 });
    expect(scalarSliceAxisExtent(payload, "z")).toBe(2);
    expect(scalarSliceAxisExtent(payload, "y")).toBe(3);
    expect(scalarSliceAxisExtent(payload, "x")).toBe(4);
  });

  it("rejects an oversized client slice before allocating its RGBA buffer", () => {
    const payload = {
      ...makePayload(),
      width: 32767,
      height: 32767,
      depth: 1,
    };

    expect(() =>
      extractScalarSlice(payload, {
        axis: "z",
        sliceIndex: 0,
        windowLow: 0,
        windowHigh: 100,
      })
    ).toThrow(/slice.*limit/i);
  });

  it("maps a 1024 source grid onto 512 nearest samples using source-cell centres", () => {
    const payload: ScalarVolumePayload = {
      ...makePayload(),
      width: 512,
      height: 512,
      depth: 2,
      sourceWidth: 1024,
      sourceHeight: 1024,
      sourceDepth: 4,
      downsampleX: 2,
      downsampleY: 2,
      downsampleZ: 2,
      sampling: "nearest",
    };

    expect(mapSourceSliceToNearestDelivery(payload, "x", 0)).toEqual({
      deliveryIndex: 0,
      sampledSourceIndex: 1,
      exact: false,
    });
    expect(mapSourceSliceToNearestDelivery(payload, "x", 1)).toEqual({
      deliveryIndex: 0,
      sampledSourceIndex: 1,
      exact: true,
    });
    expect(mapSourceSliceToNearestDelivery(payload, "x", 1023)).toEqual({
      deliveryIndex: 511,
      sampledSourceIndex: 1023,
      exact: true,
    });
    expect(mapSourceVoxelToNearestDelivery(payload, { x: 513, y: 512, z: 3 })).toEqual({
      delivery: { x: 256, y: 256, z: 1 },
      sampledSource: { x: 513, y: 513, z: 3 },
      exact: false,
    });
    expect(scalarPayloadUsesNativeGrid(payload)).toBe(false);
    expect(scalarPayloadUsesNativeGrid(makePayload())).toBe(true);
  });

  it("maps downsampled MPR overlays and clicks through delivery texel centres", () => {
    const payload: ScalarVolumePayload = {
      ...makePayload(),
      data: new Uint16Array(3 * 3 * 3).buffer,
      width: 3,
      height: 3,
      depth: 3,
      sourceWidth: 9,
      sourceHeight: 9,
      sourceDepth: 9,
      downsampleX: 4,
      downsampleY: 4,
      downsampleZ: 4,
      sampling: "nearest",
    };

    expect(
      mapSourcePlanePointToNearestDelivery(payload, "z", {
        row: 6,
        col: 8,
      })
    ).toEqual({
      delivery: { row: 1, col: 2 },
      sampledSource: { row: 6, col: 8 },
      exact: true,
    });
    expect(
      mapNearestDeliveryPlanePointToSource(payload, "z", {
        row: 1,
        col: 2,
      })
    ).toEqual({ row: 6, col: 8 });
    expect(
      mapNearestDeliveryPlanePointToSource(payload, "y", {
        row: 1,
        col: 2,
      })
    ).toEqual({ row: 6, col: 8 });
    expect(
      mapNearestDeliveryPlanePointToSource(payload, "x", {
        row: 1,
        col: 2,
      })
    ).toEqual({ row: 6, col: 8 });
  });
});
