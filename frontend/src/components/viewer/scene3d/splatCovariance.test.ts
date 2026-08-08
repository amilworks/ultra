import { describe, expect, it } from "vitest";

import { quatToMat3 } from "./sceneFrame";
import {
  covarianceFromScaleRot,
  SPLAT_FOOTPRINT_SIGMAS,
  worldFootprintRadius,
} from "./splatCovariance";

const IDENTITY_QUAT = [0, 0, 0, 1];
const SQRT_HALF = Math.SQRT1_2;
/** 90 degrees about +z, xyzw. */
const Z90_XYZW = [0, 0, SQRT_HALF, SQRT_HALF];

const LOG_2_3_4 = [Math.log(2), Math.log(3), Math.log(4)];

const expectVecClose = (actual: readonly number[], expected: readonly number[], digits = 10) => {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i], `component ${i}`).toBeCloseTo(expected[i], digits);
  }
};

describe("covarianceFromScaleRot", () => {
  it("gives a diagonal covariance of exp(logScale)² for the identity quaternion", () => {
    // exp(ln 2, ln 3, ln 4) = (2, 3, 4); variances are their squares.
    expectVecClose(covarianceFromScaleRot(LOG_2_3_4, IDENTITY_QUAT), [4, 0, 0, 9, 0, 16]);
  });

  it("applies exp — skipping it would give a completely different matrix", () => {
    const cov = covarianceFromScaleRot(LOG_2_3_4, IDENTITY_QUAT);
    // What a converter that copied the log values straight through would produce.
    const withoutExp = [
      Math.log(2) ** 2, 0, 0,
      Math.log(3) ** 2, 0,
      Math.log(4) ** 2,
    ];
    expect(cov[0]).toBeCloseTo(4, 12);
    expect(withoutExp[0]).toBeCloseTo(0.4804530139182014, 12);
    expect(Math.abs(cov[0] - withoutExp[0])).toBeGreaterThan(3);
  });

  it("handles the real file's measured median scale (-4.639 -> 0.00967)", () => {
    const median = -4.6392;
    const cov = covarianceFromScaleRot([median, median, median], IDENTITY_QUAT);
    expect(Math.sqrt(cov[0])).toBeCloseTo(Math.exp(median), 12);
    expect(Math.sqrt(cov[0])).toBeCloseTo(0.00967, 5);
  });

  it("rotates the principal axes: 90 degrees about z swaps the x and y variances", () => {
    expectVecClose(covarianceFromScaleRot(LOG_2_3_4, Z90_XYZW), [9, 0, 0, 4, 0, 16]);
  });

  it("keeps the trace invariant under rotation (it is the sum of the variances)", () => {
    const quats = [
      IDENTITY_QUAT,
      Z90_XYZW,
      [0.5, 0.5, 0.5, 0.5],
      [0.183, -0.4, 0.66, 0.61],
      [-0.2, 0.9, 0.1, -0.35],
    ];
    const expectedTrace = 4 + 9 + 16;
    for (const q of quats) {
      const cov = covarianceFromScaleRot(LOG_2_3_4, q);
      expect(cov[0] + cov[3] + cov[5], JSON.stringify(q)).toBeCloseTo(expectedTrace, 9);
    }
  });

  it("is positive semi-definite: every diagonal entry is non-negative", () => {
    for (const q of [IDENTITY_QUAT, Z90_XYZW, [0.183, -0.4, 0.66, 0.61]]) {
      const cov = covarianceFromScaleRot([-1, -4, 0.5], q);
      expect(cov[0]).toBeGreaterThanOrEqual(0);
      expect(cov[3]).toBeGreaterThanOrEqual(0);
      expect(cov[5]).toBeGreaterThanOrEqual(0);
    }
  });

  it("matches R·diag(v)·Rᵀ computed independently", () => {
    const q = [0.183, -0.4, 0.66, 0.61];
    const r = quatToMat3(q);
    const v = [4, 9, 16];
    const sigma = (i: number, j: number): number =>
      r[i * 3] * v[0] * r[j * 3] +
      r[i * 3 + 1] * v[1] * r[j * 3 + 1] +
      r[i * 3 + 2] * v[2] * r[j * 3 + 2];
    expectVecClose(covarianceFromScaleRot(LOG_2_3_4, q), [
      sigma(0, 0), sigma(0, 1), sigma(0, 2),
      sigma(1, 1), sigma(1, 2),
      sigma(2, 2),
    ], 12);
  });

  it("normalizes the quaternion — an unnormalized one must not fold in an extra scale", () => {
    const unit = covarianceFromScaleRot(LOG_2_3_4, [0.183, -0.4, 0.66, 0.61]);
    const scaled = covarianceFromScaleRot(LOG_2_3_4, [0.549, -1.2, 1.98, 1.83]);
    expectVecClose(scaled, unit, 10);
  });

  it("collapses a non-finite log scale to a degenerate axis rather than a NaN", () => {
    const cov = covarianceFromScaleRot([Number.NaN, Math.log(3), Math.log(4)], IDENTITY_QUAT);
    expect(cov.every((value) => Number.isFinite(value))).toBe(true);
    expect(cov[0]).toBe(0);
    expect(cov[3]).toBeCloseTo(9, 12);
  });

  it("rejects a short log scale", () => {
    expect(() => covarianceFromScaleRot([1, 2], IDENTITY_QUAT)).toThrow(/3 components/);
  });
});

describe("worldFootprintRadius", () => {
  it("is 3 sigma of the largest activated axis", () => {
    expect(SPLAT_FOOTPRINT_SIGMAS).toBe(3);
    expect(worldFootprintRadius(LOG_2_3_4)).toBeCloseTo(12, 12);
  });

  it("does not care which axis is largest", () => {
    expect(worldFootprintRadius([Math.log(4), Math.log(2), Math.log(3)])).toBeCloseTo(12, 12);
    expect(worldFootprintRadius([Math.log(2), Math.log(4), Math.log(3)])).toBeCloseTo(12, 12);
  });

  it("stays small for the real file's typical splat", () => {
    // Measured median scale_0 = -4.6392 -> 0.00967 world units.
    expect(worldFootprintRadius([-4.6392, -4.6392, -4.6392])).toBeCloseTo(3 * Math.exp(-4.6392), 12);
    expect(worldFootprintRadius([-4.6392, -4.6392, -4.6392])).toBeLessThan(0.03);
  });

  it("survives a non-finite axis", () => {
    expect(worldFootprintRadius([Number.NEGATIVE_INFINITY, Math.log(5), Number.NaN]))
      .toBeCloseTo(15, 12);
  });
});
