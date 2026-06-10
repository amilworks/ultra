import { describe, expect, it } from "vitest";

import type { ScalarVolumePayload } from "@/lib/api";

import { extractScalarSlice, scalarSliceAxisExtent, scalarSliceDimensions } from "./scalarSlice";

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
});
