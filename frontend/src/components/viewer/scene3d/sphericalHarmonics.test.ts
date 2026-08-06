import { describe, expect, it } from "vitest";

import {
  dcToBaseColour,
  degreeFromRestCount,
  deplanarizeSh,
  evaluateSh,
  shBandCount,
  SH_C0,
  SH_C1,
  SH_C2,
  SH_C3,
} from "./sphericalHarmonics";

/**
 * Build an interleaved rest buffer with only the named coefficients populated.
 * Keys are the reference's 1-based `sh[i]` indices, so `{ 1: [...] }` is the first
 * l=1 coefficient.
 */
const restFor = (degree: number, entries: Record<number, [number, number, number]>): Float32Array => {
  const out = new Float32Array(shBandCount(degree) * 3);
  for (const [key, rgb] of Object.entries(entries)) {
    const i = Number(key);
    out[(i - 1) * 3] = rgb[0];
    out[(i - 1) * 3 + 1] = rgb[1];
    out[(i - 1) * 3 + 2] = rgb[2];
  }
  return out;
};

const WHITE: [number, number, number] = [1, 1, 1];
const norm = (v: readonly number[]): [number, number, number] => {
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
};

describe("INRIA constant table", () => {
  it("reproduces the reference rasterizer's constants exactly", () => {
    // cuda_rasterizer/forward.cu, computeColorFromSH.
    expect(SH_C0).toBe(0.28209479177387814);
    expect(SH_C1).toBe(0.4886025119029199);
    expect([...SH_C2]).toEqual([
      1.0925484305920792,
      -1.0925484305920792,
      0.31539156525252005,
      -1.0925484305920792,
      0.5462742152960396,
    ]);
    expect([...SH_C3]).toEqual([
      -0.5900435899266435,
      2.890611442640554,
      -0.4570457994644658,
      0.3731763325901154,
      -0.4570457994644658,
      1.445305721320277,
      -0.5900435899266435,
    ]);
  });

  it("carries negative entries — a table of magnitudes only would be a different basis", () => {
    expect(SH_C2.filter((c) => c < 0)).toHaveLength(2);
    expect(SH_C3.filter((c) => c < 0)).toHaveLength(4);
  });
});

describe("dcToBaseColour", () => {
  it("is 0.5 + C0*dc", () => {
    expect(dcToBaseColour([1, -1, 0])).toEqual([0.5 + SH_C0, 0.5 - SH_C0, 0.5]);
  });

  it("is UNCLAMPED — the real file runs -0.511 to 2.704", () => {
    const [low] = dcToBaseColour([-3.585, 0, 0]);
    const [high] = dcToBaseColour([7.813, 0, 0]);
    expect(low).toBeLessThan(0);
    expect(high).toBeGreaterThan(1);
  });

  it("rejects a short dc vector", () => {
    expect(() => dcToBaseColour([1, 2])).toThrow(/3 components/);
  });
});

describe("shBandCount / degreeFromRestCount", () => {
  it("counts rest coefficients per channel as 0, 3, 8, 15", () => {
    expect([0, 1, 2, 3].map(shBandCount)).toEqual([0, 3, 8, 15]);
  });

  it("maps the total f_rest count back to a degree", () => {
    expect(degreeFromRestCount(0)).toBe(0);
    expect(degreeFromRestCount(9)).toBe(1);
    expect(degreeFromRestCount(24)).toBe(2);
    expect(degreeFromRestCount(45)).toBe(3);
  });

  it("throws rather than guessing when the count matches no degree", () => {
    expect(() => degreeFromRestCount(10)).toThrow(/matches no SH degree/);
    expect(() => degreeFromRestCount(15)).toThrow(/matches no SH degree/);
    expect(() => shBandCount(4)).toThrow(/integer in 0\.\.3/);
    expect(() => shBandCount(-1)).toThrow(/integer in 0\.\.3/);
  });
});

describe("deplanarizeSh", () => {
  it("regroups RRR..GGG..BBB into RGB per coefficient", () => {
    // Per-channel distinct values, so a wrong grouping cannot accidentally match.
    const planar = new Float32Array([1, 2, 3, 10, 20, 30, 100, 200, 300]);
    expect([...deplanarizeSh(planar, 3)]).toEqual([1, 10, 100, 2, 20, 200, 3, 30, 300]);
  });

  it("keeps each channel's coefficients in order", () => {
    const planar = new Float32Array(45);
    for (let i = 0; i < 45; i += 1) {
      planar[i] = i;
    }
    const out = deplanarizeSh(planar, 15);
    expect(out).toHaveLength(45);
    for (let i = 0; i < 15; i += 1) {
      expect(out[i * 3], `coefficient ${i} R`).toBe(i);
      expect(out[i * 3 + 1], `coefficient ${i} G`).toBe(15 + i);
      expect(out[i * 3 + 2], `coefficient ${i} B`).toBe(30 + i);
    }
    // The interleaved buffer is NOT the planar buffer reordered by luck.
    expect([...out]).not.toEqual([...planar]);
  });

  it("throws when the planar buffer is short", () => {
    expect(() => deplanarizeSh(new Float32Array(8), 3)).toThrow(/need 9/);
  });
});

describe("evaluateSh", () => {
  it("degree 0 is exactly dcToBaseColour", () => {
    const dc = [1.234, -0.5, 3.75];
    const empty = new Float32Array(0);
    for (const dir of [[0, 0, 1], [1, 0, 0], [-0.3, 0.9, 0.31]]) {
      expect(evaluateSh(0, dc, empty, dir)).toEqual(dcToBaseColour(dc));
    }
  });

  it("reproduces the l=1 signs: -C1·y, +C1·z, -C1·x", () => {
    const dc = [0, 0, 0];
    // sh[1] -> R only, sh[2] -> G only, sh[3] -> B only, so each channel isolates one term.
    const rest = restFor(1, { 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1] });

    // +z picks out sh[2] with a PLUS sign.
    const alongZ = evaluateSh(1, dc, rest, [0, 0, 1]);
    expect(alongZ[1]).toBeCloseTo(0.5 + SH_C1, 12);
    expect(alongZ[1]).toBeCloseTo(0.9886025119029199, 12);
    expect(alongZ[0]).toBeCloseTo(0.5, 12);
    expect(alongZ[2]).toBeCloseTo(0.5, 12);

    // +y picks out sh[1] with a MINUS sign. A textbook basis without the reference's
    // sign fold would give 0.5 + C1 here, which is what this assertion catches.
    const alongY = evaluateSh(1, dc, rest, [0, 1, 0]);
    expect(alongY[0]).toBeCloseTo(0.5 - SH_C1, 12);
    expect(alongY[0]).toBeCloseTo(0.011397488097080103, 12);
    expect(alongY[0]).toBeLessThan(0.5);

    // +x picks out sh[3], also MINUS.
    const alongX = evaluateSh(1, dc, rest, [1, 0, 0]);
    expect(alongX[2]).toBeCloseTo(0.5 - SH_C1, 12);
    expect(alongX[2]).toBeLessThan(0.5);
  });

  it("reproduces the l=2 constants, signs included", () => {
    const dc = [0, 0, 0];

    // sh[6] with dir=+z: C2[2]·(2z² - x² - y²) = 0.31539156525252005 · 2.
    const onZ = evaluateSh(2, dc, restFor(2, { 6: WHITE }), [0, 0, 1]);
    expect(onZ[0]).toBeCloseTo(1.1307831305050402, 12);

    // sh[4] with dir=(1,1,0)/√2: C2[0]·xy = 1.0925484305920792 · 0.5.
    const onXy = evaluateSh(2, dc, restFor(2, { 4: WHITE }), norm([1, 1, 0]));
    expect(onXy[0]).toBeCloseTo(1.0462742152960396, 12);

    // sh[5] with dir=(0,1,1)/√2: C2[1]·yz is NEGATIVE — the constant carries the sign.
    const onYz = evaluateSh(2, dc, restFor(2, { 5: WHITE }), norm([0, 1, 1]));
    expect(onYz[0]).toBeCloseTo(-0.0462742152960396, 12);
    expect(onYz[0]).toBeLessThan(0);
  });

  it("reproduces the l=3 constants, signs included", () => {
    const dc = [0, 0, 0];

    // sh[12] with dir=+z: C3[3]·z(2z² - 3x² - 3y²) = 0.3731763325901154 · 2.
    const onZ = evaluateSh(3, dc, restFor(3, { 12: WHITE }), [0, 0, 1]);
    expect(onZ[0]).toBeCloseTo(1.2463526651802308, 12);

    // sh[9] with dir=+y: C3[0]·y(3x² - y²) = -0.5900435899266435 · -1.
    const onY = evaluateSh(3, dc, restFor(3, { 9: WHITE }), [0, 1, 0]);
    expect(onY[0]).toBeCloseTo(1.0900435899266434, 12);
  });

  it("negating an ODD band mirrors the result about the DC term (global sign flip guard)", () => {
    const dc = [0.2, -0.4, 1.1];
    const base = dcToBaseColour(dc);
    const coefficients: Record<number, [number, number, number]> = {
      1: [0.7, -0.3, 0.11],
      2: [-0.25, 0.9, -0.6],
      3: [0.05, 0.42, 0.8],
    };
    const negated: Record<number, [number, number, number]> = {};
    for (const [k, v] of Object.entries(coefficients)) {
      negated[Number(k)] = [-v[0], -v[1], -v[2]];
    }
    const dir = norm([0.3, -0.7, 0.65]);

    const positive = evaluateSh(1, dc, restFor(1, coefficients), dir);
    const flipped = evaluateSh(1, dc, restFor(1, negated), dir);

    for (let c = 0; c < 3; c += 1) {
      expect(positive[c]).not.toBeCloseTo(flipped[c], 6);
      expect(flipped[c] - base[c]).toBeCloseTo(-(positive[c] - base[c]), 6);
    }
  });

  it("flips the odd bands when dir is reversed, and leaves the even band alone", () => {
    const dc = [0, 0, 0];
    const dir = norm([0.3, -0.7, 0.65]);
    const back: [number, number, number] = [-dir[0], -dir[1], -dir[2]];

    // l=1 is linear in dir: reversing camera->splat into splat->camera flips its sign.
    const odd = restFor(1, { 1: [0.7, -0.3, 0.11], 2: [-0.25, 0.9, -0.6], 3: [0.05, 0.42, 0.8] });
    const oddForward = evaluateSh(1, dc, odd, dir);
    const oddBack = evaluateSh(1, dc, odd, back);
    expect(oddForward[0]).not.toBeCloseTo(oddBack[0], 6);
    expect(oddBack[0] - 0.5).toBeCloseTo(-(oddForward[0] - 0.5), 6);

    // l=2 is quadratic in dir, so it is genuinely invariant — proving the test above
    // measures band parity and not just "any change".
    const even = restFor(2, { 4: [0.3, 0.2, 0.1], 6: [-0.5, 0.25, 0.9], 8: [0.15, -0.65, 0.4] });
    const evenForward = evaluateSh(2, dc, even, dir);
    const evenBack = evaluateSh(2, dc, even, back);
    for (let c = 0; c < 3; c += 1) {
      expect(evenForward[c]).toBeCloseTo(evenBack[c], 12);
    }

    // l=3 is cubic, so it flips like l=1.
    const l3 = restFor(3, { 10: [0.4, -0.2, 0.7], 14: [0.9, 0.1, -0.3] });
    const l3Forward = evaluateSh(3, dc, l3, dir);
    const l3Back = evaluateSh(3, dc, l3, back);
    expect(l3Back[0] - 0.5).toBeCloseTo(-(l3Forward[0] - 0.5), 6);
  });

  it("normalizes dir, exactly as the reference does", () => {
    const dc = [0.1, 0.2, 0.3];
    const rest = restFor(3, { 2: [0.5, -0.5, 0.25], 6: WHITE, 12: [0.3, 0.3, 0.3] });
    const unit = evaluateSh(3, dc, rest, [0, 0, 1]);
    const long = evaluateSh(3, dc, rest, [0, 0, 137.5]);
    for (let c = 0; c < 3; c += 1) {
      expect(long[c]).toBeCloseTo(unit[c], 12);
    }
  });

  it("falls back to the DC term for a zero-length direction instead of returning NaN", () => {
    const dc = [0.4, 0.5, 0.6];
    const rest = restFor(1, { 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1] });
    expect(evaluateSh(1, dc, rest, [0, 0, 0])).toEqual(dcToBaseColour(dc));
  });

  it("throws when the rest buffer is too short for the degree", () => {
    expect(() => evaluateSh(2, [0, 0, 0], new Float32Array(9), [0, 0, 1])).toThrow(/needs 24/);
    expect(() => evaluateSh(1, [0, 0], new Float32Array(9), [0, 0, 1])).toThrow(/3 components/);
    expect(() => evaluateSh(1, [0, 0, 0], new Float32Array(9), [0, 1])).toThrow(/dir/);
  });

  it("ignores coefficients above the requested degree", () => {
    const dc = [0, 0, 0];
    const full = restFor(3, { 6: WHITE, 12: WHITE });
    const degree2Only = evaluateSh(2, dc, full, [0, 0, 1]);
    const degree3 = evaluateSh(3, dc, full, [0, 0, 1]);
    expect(degree2Only[0]).toBeCloseTo(1.1307831305050402, 12);
    expect(degree3[0]).toBeCloseTo(1.1307831305050402 + 0.7463526651802308, 12);
  });
});
