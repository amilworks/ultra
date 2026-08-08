import { describe, expect, it } from "vitest";

import {
  clampWithCount,
  linearToSrgb,
  LINEAR_SRGB_KNEE,
  srgbBytesToLinearFloat,
  srgbToLinear,
  SRGB_LINEAR_KNEE,
} from "./sceneColor";

/** The approximation this module deliberately refuses to use. */
const gamma22 = (c: number): number => c ** 2.2;

describe("srgbToLinear", () => {
  it("pins the endpoints exactly", () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBe(1);
  });

  it("matches the IEC 61966-2-1 value at mid grey", () => {
    expect(srgbToLinear(0.5)).toBeCloseTo(0.21404114048223255, 12);
  });

  it("uses the linear segment at and below the 0.04045 knee", () => {
    expect(SRGB_LINEAR_KNEE).toBe(0.04045);
    expect(srgbToLinear(SRGB_LINEAR_KNEE)).toBe(SRGB_LINEAR_KNEE / 12.92);
    expect(srgbToLinear(SRGB_LINEAR_KNEE)).toBeCloseTo(0.003130804953560372, 15);
    expect(srgbToLinear(0.02)).toBe(0.02 / 12.92);
  });

  it("switches to the power segment immediately above the knee, continuously", () => {
    const justAbove = SRGB_LINEAR_KNEE + 1e-10;
    // The branch really did change: the power segment is not c/12.92 any more.
    expect(srgbToLinear(justAbove)).not.toBe(justAbove / 12.92);
    // ...but the curve does not jump. The standard's two pieces meet to ~1e-8.
    expect(Math.abs(srgbToLinear(justAbove) - srgbToLinear(SRGB_LINEAR_KNEE))).toBeLessThan(1e-7);
  });

  it("is NOT a 2.2 gamma — the approximation fails this module's tolerance", () => {
    // Mid grey: the two disagree in the third decimal.
    expect(gamma22(0.5)).toBeCloseTo(0.2176376, 6);
    expect(Math.abs(srgbToLinear(0.5) - gamma22(0.5))).toBeGreaterThan(3e-3);
    expect(srgbToLinear(0.5)).not.toBeCloseTo(gamma22(0.5), 3);

    // In the shadows — where a point cloud's crevices live — it is off by 8x.
    expect(srgbToLinear(0.02) / gamma22(0.02)).toBeGreaterThan(8);
  });

  it("extends below zero by odd symmetry instead of returning NaN", () => {
    // 0.5 + C0*f_dc is unclamped and measured down to -0.511 on the real file.
    expect(srgbToLinear(-0.511)).toBe(-srgbToLinear(0.511));
    expect(Number.isFinite(srgbToLinear(-0.511))).toBe(true);
    expect(srgbToLinear(-0.02)).toBe(-0.02 / 12.92);
  });

  it("handles values above 1 (measured up to 2.704) without saturating", () => {
    expect(srgbToLinear(2.704)).toBeGreaterThan(1);
    expect(Number.isFinite(srgbToLinear(2.704))).toBe(true);
  });

  it("returns 0 for a non-finite input rather than poisoning a buffer", () => {
    expect(srgbToLinear(Number.NaN)).toBe(0);
    expect(srgbToLinear(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe("linearToSrgb", () => {
  it("pins the endpoints and the knee", () => {
    expect(linearToSrgb(0)).toBe(0);
    expect(linearToSrgb(1)).toBeCloseTo(1, 12);
    expect(LINEAR_SRGB_KNEE).toBe(0.0031308);
    expect(linearToSrgb(LINEAR_SRGB_KNEE)).toBe(LINEAR_SRGB_KNEE * 12.92);
  });

  it("reproduces the measured double-encode: linear 0.2 displays as ~0.48", () => {
    // Contract §4.3 — this is what happens when sRGB point colour is handed to three.js
    // as though it were already linear.
    expect(linearToSrgb(0.2)).toBeCloseTo(0.4845, 3);
    expect(linearToSrgb(0.2)).toBeGreaterThan(0.48);
    expect(linearToSrgb(0.2)).toBeLessThan(0.49);
  });

  it("inverts srgbToLinear across the whole range, both branches", () => {
    for (const c of [0, 0.001, 0.01, 0.04, 0.0405, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      expect(linearToSrgb(srgbToLinear(c)), `c=${c}`).toBeCloseTo(c, 10);
      expect(srgbToLinear(linearToSrgb(c)), `c=${c}`).toBeCloseTo(c, 10);
    }
  });

  it("inherits the standard's own ~3e-8 knee inconsistency, and no more", () => {
    // IEC 61966-2-1 rounds its two knee constants independently:
    //   0.04045 / 12.92 = 0.0031308049...  >  0.0031308
    // so a value exactly at the sRGB knee comes back through the INVERSE's power branch
    // rather than its linear branch. This asserts the size of that published quirk so a
    // future "simplification" of either branch cannot hide behind it.
    expect(srgbToLinear(SRGB_LINEAR_KNEE)).toBeGreaterThan(LINEAR_SRGB_KNEE);
    const error = Math.abs(linearToSrgb(srgbToLinear(SRGB_LINEAR_KNEE)) - SRGB_LINEAR_KNEE);
    expect(error).toBeCloseTo(2.96e-8, 10);
    expect(error).toBeLessThan(5e-8);
  });

  it("extends below zero by odd symmetry", () => {
    expect(linearToSrgb(-0.2)).toBe(-linearToSrgb(0.2));
  });
});

describe("srgbBytesToLinearFloat", () => {
  it("converts every byte exactly as srgbToLinear(byte/255) would", () => {
    const src = new Uint8Array(256);
    for (let i = 0; i < 256; i += 1) {
      src[i] = i;
    }
    const out = srgbBytesToLinearFloat(src);
    expect(out).toHaveLength(256);
    expect(out[0]).toBe(0);
    expect(out[255]).toBe(1);
    for (let i = 0; i < 256; i += 1) {
      // The lookup table is Float32, so compare at f32 precision.
      expect(out[i], `byte ${i}`).toBeCloseTo(srgbToLinear(i / 255), 6);
    }
  });

  it("is monotonic — a lookup table with a transposed entry would fail here", () => {
    const src = new Uint8Array(256);
    for (let i = 0; i < 256; i += 1) {
      src[i] = i;
    }
    const out = srgbBytesToLinearFloat(src);
    for (let i = 1; i < 256; i += 1) {
      expect(out[i], `byte ${i}`).toBeGreaterThan(out[i - 1]);
    }
  });

  it("does NOT double-encode: sRGB byte 51 (0.2) lands near linear 0.033, not 0.2", () => {
    const out = srgbBytesToLinearFloat(new Uint8Array([51]));
    expect(out[0]).toBeCloseTo(0.033104766, 6);
    expect(out[0]).toBeLessThan(0.2);
  });

  it("fills a supplied output buffer in place", () => {
    const out = new Float32Array(8).fill(-1);
    const returned = srgbBytesToLinearFloat(new Uint8Array([0, 255, 128]), out);
    expect(returned).toBe(out);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(1);
    expect(out[2]).toBeCloseTo(srgbToLinear(128 / 255), 6);
    // Trailing slots are untouched, so one buffer can be reused across chunks.
    expect(out[3]).toBe(-1);
  });

  it("throws when the supplied buffer is too small", () => {
    expect(() => srgbBytesToLinearFloat(new Uint8Array(4), new Float32Array(3)))
      .toThrow(/holds 3 values, need 4/);
  });
});

describe("clampWithCount", () => {
  it("clamps to [0,1] and counts every component it had to touch", () => {
    // The measured range of 0.5 + C0*f_dc_0 on the real file.
    expect(clampWithCount([-0.511, 0.513, 2.704])).toEqual({ values: [0, 0.513, 1], clamped: 2 });
  });

  it("counts nothing when everything is in gamut", () => {
    expect(clampWithCount([0, 0.25, 1])).toEqual({ values: [0, 0.25, 1], clamped: 0 });
  });

  it("treats a non-finite component as out of gamut", () => {
    expect(clampWithCount([Number.NaN, Number.POSITIVE_INFINITY, 0.5]))
      .toEqual({ values: [0, 0, 0.5], clamped: 2 });
  });

  it("handles an empty input", () => {
    expect(clampWithCount([])).toEqual({ values: [], clamped: 0 });
  });

  it("counts display-boundary clipping without requiring splat-wire clamping", () => {
    const dcColours = [-0.2, 0.1, 0.4, 0.9, 1.4, 0.55, 0.6, 0.7];
    const { clamped } = clampWithCount(dcColours);
    expect(clamped / dcColours.length).toBeCloseTo(0.25, 12);
  });
});
