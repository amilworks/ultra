import { describe, expect, it } from "vitest";

import type { ScalarVolumePayload } from "@/lib/api";

import {
  MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES,
  PreparedScalarVolumeCache,
  PreparedScalarVolumeResidencyManager,
  prepareScalarVolume,
  resolveScalarVolumeRescale,
  resolveScalarVolumeWindow,
  scalarVolumeApiNamespace,
  scalarVolumeIdentityKey,
  scalarVolumeSourceIdentity,
  scalarVolumePayloadToHalfFloatAsync,
  scalarVolumePayloadToRawInteger,
  scalarVolumePayloadValueAt,
  validateExactMaskSourcePreflight,
  validateExactMaskVolume,
  validateScalarVolumeIdentity,
} from "./scalarVolume";
import { extractScalarSlice } from "./scalarSlice";

const makePayload = (overrides: Partial<ScalarVolumePayload> = {}): ScalarVolumePayload => ({
  data: new Uint16Array([1024, 1104]).buffer,
  width: 2,
  height: 1,
  depth: 1,
  dtype: "uint16",
  bytesPerVoxel: 2,
  rawMin: 0,
  rawMax: 4095,
  channel: 0,
  time: 0,
  sclSlope: 1,
  sclInter: -1024,
  sourceWidth: 2,
  sourceHeight: 1,
  sourceDepth: 1,
  downsampleX: 1,
  downsampleY: 1,
  downsampleZ: 1,
  previewPolicy: "exact-v1",
  sampling: "box",
  ...overrides,
});

describe("exact Mask volume admission", () => {
  const sourceGrid = { width: 32, height: 24, depth: 8 };
  const exact = makePayload({
    data: new Uint16Array(32 * 24 * 8).buffer,
    width: 32,
    height: 24,
    depth: 8,
    sourceWidth: 32,
    sourceHeight: 24,
    sourceDepth: 8,
    previewPolicy: "mask-native-integer-v1",
    sampling: "nearest",
  });

  it("requires native integer nearest provenance for fresh and prepared data", () => {
    expect(() =>
      validateExactMaskVolume(exact, { sourceGrid, dtype: "uint16", max3DTextureSize: 2048 })
    ).not.toThrow();
    expect(() =>
      validateExactMaskVolume(
        { ...exact, width: 16, downsampleX: 2 },
        { sourceGrid, dtype: "uint16", max3DTextureSize: 2048 }
      )
    ).toThrow(/native integer grid/i);
    expect(() =>
      validateExactMaskVolume(
        { ...exact, textureData: new Uint16Array(1) },
        { sourceGrid, dtype: "uint16", max3DTextureSize: 2048 }
      )
    ).toThrow(/byte size/i);
  });

  it("fails source preflight before loading when traversal, GPU, or 128 MiB bounds are exceeded", () => {
    expect(() =>
      validateExactMaskSourcePreflight({
        sourceGrid: { width: 1025, height: 1, depth: 1 },
        dtype: "uint8",
        max3DTextureSize: 2048,
      })
    ).toThrow(/texture\/traversal limit/i);
    expect(() =>
      validateExactMaskSourcePreflight({
        sourceGrid: { width: 512, height: 512, depth: 512 },
        dtype: "uint16",
        max3DTextureSize: 2048,
      })
    ).toThrow(/byte limit/i);
    expect(() =>
      validateExactMaskSourcePreflight({
        sourceGrid,
        dtype: "uint16",
        max3DTextureSize: 16,
      })
    ).toThrow(/texture\/traversal limit/i);
  });

  it("bounds worst-case oblique DDA crossings by the sum of native axes", () => {
    expect(() =>
      validateExactMaskSourcePreflight({
        sourceGrid: { width: 1024, height: 1023, depth: 1 },
        dtype: "uint8",
        max3DTextureSize: 2048,
      })
    ).not.toThrow();
    expect(() =>
      validateExactMaskSourcePreflight({
        sourceGrid: { width: 1024, height: 1023, depth: 2 },
        dtype: "uint8",
        max3DTextureSize: 2048,
      })
    ).toThrow(/DDA traversal limit/i);
  });
});

describe("prepared scalar volume cache", () => {
  it("scopes identity to API client, source SHA/grid, C/T counts, and preview policy", () => {
    const firstClient = {};
    const secondClient = {};
    const sourceGrid = { width: 4, height: 3, depth: 2 };
    const base = {
      fileId: "same-file",
      sourceGrid,
      channel: 2,
      time: 1,
      channelCount: 3,
      timeCount: 2,
      policy: "auto-v1",
    };
    const sourceA = scalarVolumeSourceIdentity({ ...base, sha256: "sha-a" });
    const sourceB = scalarVolumeSourceIdentity({ ...base, sha256: "sha-b" });
    const keyA = scalarVolumeIdentityKey({
      ...base,
      apiNamespace: scalarVolumeApiNamespace(firstClient, "/scalar-volume"),
      sourceIdentity: sourceA,
    });

    expect(
      scalarVolumeIdentityKey({
        ...base,
        apiNamespace: scalarVolumeApiNamespace(firstClient, "/scalar-volume"),
        sourceIdentity: sourceB,
      })
    ).not.toBe(keyA);
    expect(
      scalarVolumeIdentityKey({
        ...base,
        apiNamespace: scalarVolumeApiNamespace(secondClient, "/scalar-volume"),
        sourceIdentity: sourceA,
      })
    ).not.toBe(keyA);
    expect(scalarVolumeIdentityKey({ ...base, apiNamespace: "api", sourceIdentity: sourceA, policy: "exact-v1" }))
      .not.toBe(scalarVolumeIdentityKey({ ...base, apiNamespace: "api", sourceIdentity: sourceA }));
  });

  it("evicts least-recently-used prepared R16F entries by byte weight", async () => {
    const cache = new PreparedScalarVolumeCache(12);
    const first = await prepareScalarVolume(makePayload({ data: new Uint16Array(3).buffer, width: 3 }));
    const second = await prepareScalarVolume(makePayload({ data: new Uint16Array(3).buffer, width: 3 }));
    const third = await prepareScalarVolume(makePayload({ data: new Uint16Array(3).buffer, width: 3 }));

    cache.set("first", first);
    cache.set("second", second);
    expect(cache.get("first")).toBe(first);
    cache.set("third", third);

    expect(cache.byteSize).toBe(12);
    expect(cache.get("first")).toBe(first);
    expect(cache.get("second")).toBeUndefined();
    expect(cache.get("third")).toBe(third);
    expect(MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES).toBe(128 * 1024 * 1024);
  });

  it("retains only prepared texture data and validates requested C/T plus provenance", async () => {
    const payload = makePayload({ channel: 3, time: 1, previewPolicy: "auto-v1" });
    const prepared = await prepareScalarVolume(payload);

    expect(prepared.textureData).toBeInstanceOf(Uint16Array);
    expect(prepared).not.toHaveProperty("data");
    expect(prepared.autoWindow.low).toBeGreaterThanOrEqual(0);
    expect(() =>
      validateScalarVolumeIdentity(payload, {
        channel: 3,
        time: 1,
        sourceGrid: { width: 2, height: 1, depth: 1 },
        policy: "auto-v1",
      })
    ).not.toThrow();
    expect(() =>
      validateScalarVolumeIdentity(payload, {
        channel: 2,
        time: 1,
        sourceGrid: { width: 2, height: 1, depth: 1 },
        policy: "auto-v1",
      })
    ).toThrow(/channel/i);
    expect(() =>
      validateScalarVolumeIdentity(payload, {
        channel: 3,
        time: 0,
        sourceGrid: { width: 2, height: 1, depth: 1 },
        policy: "auto-v1",
      })
    ).toThrow(/time/i);
    expect(() =>
      validateScalarVolumeIdentity(payload, {
        channel: 3,
        time: 1,
        sourceGrid: { width: 4, height: 1, depth: 1 },
        policy: "auto-v1",
      })
    ).toThrow(/source grid/i);
    expect(() =>
      validateScalarVolumeIdentity(payload, {
        channel: 3,
        time: 1,
        sourceGrid: { width: 2, height: 1, depth: 1 },
        policy: "exact-v1",
      })
    ).toThrow(/policy/i);
  });

  it("counts reserved, cached, and pinned arrays once and evicts only unpinned LRU entries", async () => {
    const manager = new PreparedScalarVolumeResidencyManager(12);
    const first = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );
    const second = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );
    const third = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );

    const firstReservation = manager.reserve(["first"], 6);
    expect(manager.byteSize).toBe(6);
    const firstLease = manager.publishAndAcquire(firstReservation, "first", first);
    firstReservation.release();
    expect(manager.byteSize).toBe(6);

    const secondReservation = manager.reserve(["second"], 6);
    const secondLease = manager.publishAndAcquire(secondReservation, "second", second);
    secondReservation.release();
    secondLease.release();
    expect(manager.byteSize).toBe(12);

    expect(manager.get("first")).toBe(first);
    const thirdReservation = manager.reserve(["third"], 6);
    const thirdLease = manager.publishAndAcquire(thirdReservation, "third", third);
    thirdReservation.release();
    expect(manager.get("second")).toBeUndefined();
    expect(manager.get("first")).toBe(first);
    expect(manager.byteSize).toBe(12);

    expect(() => manager.reserve(["fourth", "fifth"], 6)).toThrow(/residency/i);
    firstLease.release();
    thirdLease.release();
    manager.clear();
    expect(manager.byteSize).toBe(0);
  });

  it("accounts overlapping preparations separately until they converge on one cache entry", async () => {
    const manager = new PreparedScalarVolumeResidencyManager(12);
    const first = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );
    const duplicate = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );
    const firstReservation = manager.reserve(["shared"], 6);
    const duplicateReservation = manager.reserve(["shared"], 6);
    expect(manager.byteSize).toBe(12);

    const firstLease = manager.publishAndAcquire(firstReservation, "shared", first);
    firstReservation.release();
    expect(manager.byteSize).toBe(12);
    const duplicateLease = manager.publishAndAcquire(
      duplicateReservation,
      "shared",
      duplicate
    );
    duplicateReservation.release();

    expect(duplicateLease.value).toBe(first);
    expect(manager.byteSize).toBe(6);
    firstLease.release();
    duplicateLease.release();
    manager.clear();
  });

  it("rejects publication after its staged reservation is released", async () => {
    const manager = new PreparedScalarVolumeResidencyManager(6);
    const prepared = await prepareScalarVolume(
      makePayload({ data: new Uint16Array(3).buffer, width: 3, sourceWidth: 3 })
    );
    const reservation = manager.reserve(["released"], 6);
    reservation.release();

    expect(() => manager.publishAndAcquire(reservation, "released", prepared)).toThrow(
      /no longer active/i
    );
    expect(manager.byteSize).toBe(0);
  });
});

describe("raw mask texture conversion", () => {
  it("preserves uint16 threshold membership in its native integer storage", () => {
    const values = new Uint16Array([0, 120, 121, 65_535]);
    const raw = scalarVolumePayloadToRawInteger(
      makePayload({
        data: values.buffer,
        width: values.length,
        sourceWidth: values.length,
        rawMin: 0,
        rawMax: 65_535,
      })
    );

    expect(raw).toBeInstanceOf(Uint16Array);
    expect(Array.from(raw)).toEqual([0, 120, 121, 65_535]);
    expect(Array.from(raw, (value) => value > 120)).toEqual([
      false,
      false,
      true,
      true,
    ]);
    expect(Array.from(raw, (value) => value > -1)).toEqual([
      true,
      true,
      true,
      true,
    ]);
    expect(Array.from(raw, (value) => value > 65_535)).toEqual([
      false,
      false,
      false,
      false,
    ]);
  });

  it("preserves signed int16 boundaries and uses two bytes per prepared voxel", async () => {
    const values = new Int16Array([-32_768, -1, 0, 32_767]);
    const prepared = await prepareScalarVolume(
      makePayload({
        data: values.buffer,
        width: values.length,
        sourceWidth: values.length,
        dtype: "int16",
        bytesPerVoxel: 2,
        rawMin: -32_768,
        rawMax: 32_767,
      }),
      undefined,
      "raw-integer"
    );
    expect(prepared.textureEncoding).toBe("raw-int16");
    expect(prepared.textureData).toBeInstanceOf(Int16Array);
    expect(Array.from(prepared.textureData)).toEqual(Array.from(values));
    expect(prepared.textureData.byteLength).toBe(
      values.length * Int16Array.BYTES_PER_ELEMENT
    );
  });

  it("keeps uint8 mask preparation at one byte per voxel", async () => {
    const values = new Uint8Array([0, 1, 254, 255]);
    const prepared = await prepareScalarVolume(
      makePayload({
        data: values.buffer,
        width: values.length,
        sourceWidth: values.length,
        dtype: "uint8",
        bytesPerVoxel: 1,
        rawMin: 0,
        rawMax: 255,
      }),
      undefined,
      "raw-integer"
    );

    expect(prepared.textureEncoding).toBe("raw-uint8");
    expect(prepared.textureData).toBeInstanceOf(Uint8Array);
    expect(Array.from(prepared.textureData)).toEqual(Array.from(values));
  });
});

describe("NIfTI scalar rescaling", () => {
  it("maps a physical CT window into normalized stored-code bounds", () => {
    const window = resolveScalarVolumeWindow(makePayload(), "hounsfield:40:80", false);

    expect(window.low).toBeCloseTo(1024 / 4095, 6);
    expect(window.high).toBeCloseTo(1104 / 4095, 6);
    expect(window.invert).toBe(false);
  });

  it("supports non-unit positive slopes and intercepts", () => {
    const payload = makePayload({ rawMin: 0, rawMax: 100, sclSlope: 2, sclInter: 10 });
    const window = resolveScalarVolumeWindow(payload, "hounsfield:30:20", false);

    expect(window.low).toBeCloseTo(5 / 100, 6);
    expect(window.high).toBeCloseTo(15 / 100, 6);
    expect(window.invert).toBe(false);
  });

  it("orders negative-slope code bounds and XORs physical polarity with user inversion", () => {
    const payload = makePayload({ rawMin: 0, rawMax: 100, sclSlope: -2, sclInter: 100 });

    expect(resolveScalarVolumeWindow(payload, "hounsfield:30:20", false)).toEqual({
      low: 0.3,
      high: 0.4,
      invert: true,
    });
    expect(resolveScalarVolumeWindow(payload, "hounsfield:30:20", true).invert).toBe(false);
  });

  it("rejects zero or non-finite rescale metadata instead of guessing identity", () => {
    expect(() =>
      resolveScalarVolumeRescale(makePayload({ sclSlope: 0, sclInter: Number.NaN }))
    ).toThrow(/rescale metadata/i);
    expect(() =>
      resolveScalarVolumeRescale(
        makePayload({ sclSlope: Number.POSITIVE_INFINITY, sclInter: 0 })
      )
    ).toThrow(/rescale metadata/i);
  });

  it("returns physical intensity in voxel probes", () => {
    expect(scalarVolumePayloadValueAt(makePayload(), { x: 0, y: 0, z: 0 })).toBe(0);
    expect(scalarVolumePayloadValueAt(makePayload(), { x: 1, y: 0, z: 0 })).toBe(80);
  });

  it("rejects fractional voxel coordinates instead of flooring them", () => {
    expect(scalarVolumePayloadValueAt(makePayload(), { x: 0.5, y: 0, z: 0 })).toBeNull();
    expect(scalarVolumePayloadValueAt(makePayload(), { x: -0.5, y: 0, z: 0 })).toBeNull();
  });

  it("returns null when a finite stored voxel overflows the physical rescale", () => {
    const payload = makePayload({
      data: new Float32Array([3.4e38]).buffer,
      width: 1,
      dtype: "float32",
      bytesPerVoxel: 4,
      rawMin: 0,
      rawMax: 1,
      sclSlope: Number.MAX_VALUE,
      sclInter: 0,
    });

    expect(scalarVolumePayloadValueAt(payload, { x: 0, y: 0, z: 0 })).toBeNull();
  });

  it.each([
    {
      name: "positive slope",
      payload: makePayload({
        data: new Uint16Array([10, 20, 30, 40]).buffer,
        width: 4,
        rawMin: 10,
        rawMax: 80,
        sclSlope: 2,
        sclInter: -10,
      }),
      windows: [
        { low: -30, high: 50, expectOutside: "below" },
        { low: 170, high: 210, expectOutside: "above" },
      ],
    },
    {
      name: "negative slope",
      payload: makePayload({
        data: new Uint16Array([10, 20, 30, 40]).buffer,
        width: 4,
        rawMin: 10,
        rawMax: 80,
        sclSlope: -2,
        sclInter: 100,
      }),
      windows: [
        { low: 60, high: 100, expectOutside: "below" },
        { low: 100, high: 120, expectOutside: "below" },
      ],
    },
  ])("keeps $name windows outside the observed range aligned with 2D extraction", ({ payload, windows }) => {
    const range = payload.rawMax - payload.rawMin;
    const storedValues = [10, 20, 30, 40];
    windows.forEach(({ low, high, expectOutside }) => {
      const enhancement = `hounsfield:${(low + high) / 2}:${high - low}`;
      const window = resolveScalarVolumeWindow(payload, enhancement, false);
      if (expectOutside === "below") {
        expect(window.low).toBeLessThan(0);
      } else {
        expect(window.low).toBeGreaterThan(1);
      }
      const slice = extractScalarSlice(payload, {
        axis: "z",
        sliceIndex: 0,
        windowLow: low,
        windowHigh: high,
      });
      const gpuPixels = storedValues.map((storedValue) => {
        const textureValue = (storedValue - payload.rawMin) / range;
        const windowed = Math.max(
          0,
          Math.min(1, (textureValue - window.low) / (window.high - window.low))
        );
        return Math.round((window.invert ? 1 - windowed : windowed) * 255);
      });
      const slicePixels = storedValues.map((_, index) => slice.data[index * 4]);
      gpuPixels.forEach((pixel, index) => {
        expect(Math.abs(pixel - (slicePixels[index] ?? 0))).toBeLessThanOrEqual(1);
      });
    });
  });

  it("cancels long half-float conversion before publishing obsolete texture data", async () => {
    const payload = makePayload({
      data: new Uint16Array(600_000).buffer,
      width: 600_000,
      rawMin: 0,
      rawMax: 4095,
    });
    const controller = new AbortController();
    const pending = scalarVolumePayloadToHalfFloatAsync(payload, controller.signal, 64_000);

    window.setTimeout(() => controller.abort(), 0);

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
