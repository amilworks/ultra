import { describe, expect, it } from "vitest";

import {
  applyMat3,
  cameraBasisFromColmap,
  cameraCentreFromColmap,
  IDENTITY_MAT3,
  multiplyMat3,
  normalizeQuat,
  quatToMat3,
  quatWxyzToXyzw,
  RDF_TO_RUB,
  transposeMat3,
  type Mat3,
} from "./sceneFrame";

const SQRT_HALF = Math.SQRT1_2;

/** 90 degrees about +z, in COLMAP's wxyz order. */
const QUAT_Z90_WXYZ = [SQRT_HALF, 0, 0, SQRT_HALF];

/** A handful of deterministic, deliberately unnormalized poses. */
const POSES: { q: number[]; centre: [number, number, number] }[] = [
  { q: [1, 0, 0, 0], centre: [0, 0, 0] },
  { q: QUAT_Z90_WXYZ, centre: [1, 2, 3] },
  { q: [0.5, 0.5, 0.5, 0.5], centre: [-7.25, 11.5, 0.125] },
  { q: [0.183, -0.4, 0.66, 0.61], centre: [123.5, -0.75, 1000.25] },
  { q: [-0.9, 0.2, 0.3, -0.1], centre: [1e-4, -1e-4, 5e5] },
];

const rotationFor = (qWxyz: readonly number[]): Mat3 => quatToMat3(quatWxyzToXyzw(qWxyz));

/** COLMAP's t for a given world-to-camera rotation and camera centre: t = -R C. */
const translationFor = (r: Mat3, centre: readonly number[]): [number, number, number] => {
  const rc = applyMat3(r, centre);
  return [-rc[0], -rc[1], -rc[2]];
};

const expectVecClose = (actual: readonly number[], expected: readonly number[], digits = 9) => {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i], digits);
  }
};

describe("quatWxyzToXyzw", () => {
  it("reorders without touching magnitudes", () => {
    expect(quatWxyzToXyzw([1, 2, 3, 4])).toEqual([2, 3, 4, 1]);
  });

  it("rejects a short or non-finite quaternion instead of silently reading undefined", () => {
    expect(() => quatWxyzToXyzw([1, 2, 3])).toThrow(/4 finite numbers/);
    expect(() => quatWxyzToXyzw([1, 2, 3, Number.NaN])).toThrow(/4 finite numbers/);
  });
});

describe("normalizeQuat", () => {
  it("scales to unit length", () => {
    const [x, y, z, w] = normalizeQuat([0, 0, 0, 4]);
    expect([x, y, z, w]).toEqual([0, 0, 0, 1]);
    expect(Math.hypot(...normalizeQuat([1, 2, 3, 4]))).toBeCloseTo(1, 12);
  });

  it("falls back to the xyzw identity for a zero quaternion", () => {
    expect(normalizeQuat([0, 0, 0, 0])).toEqual([0, 0, 0, 1]);
  });
});

describe("quatToMat3", () => {
  it("builds the 90-degree-about-z rotation", () => {
    const m = quatToMat3(quatWxyzToXyzw(QUAT_Z90_WXYZ));
    expectVecClose(m, [0, -1, 0, 1, 0, 0, 0, 0, 1], 12);
    // (1,0,0) rotates onto (0,1,0), i.e. counter-clockwise about +z.
    expectVecClose(applyMat3(m, [1, 0, 0]), [0, 1, 0], 12);
  });

  it("normalizes defensively — INRIA's writer emits unnormalized quaternions", () => {
    // A quaternion of norm 2 would produce a matrix scaled by 4 if fed in raw.
    expectVecClose(quatToMat3([0, 0, 0, 2]), IDENTITY_MAT3, 12);
    expectVecClose(quatToMat3([0, 0, 3 * SQRT_HALF, 3 * SQRT_HALF]), [0, -1, 0, 1, 0, 0, 0, 0, 1], 12);
  });

  it("produces an orthonormal matrix (R Rᵀ = I)", () => {
    for (const { q } of POSES) {
      const r = rotationFor(q);
      expectVecClose(multiplyMat3(r, transposeMat3(r)), IDENTITY_MAT3, 12);
    }
  });
});

describe("cameraCentreFromColmap", () => {
  it("round-trips a known camera-to-world pose to 1e-9", () => {
    for (const { q, centre } of POSES) {
      const r = rotationFor(q);
      const t = translationFor(r, centre);
      expectVecClose(cameraCentreFromColmap(q, t), centre, 9);
    }
  });

  it("satisfies the defining relation R·C + t = 0", () => {
    for (const { q, centre } of POSES) {
      const r = rotationFor(q);
      const t = translationFor(r, centre);
      const c = cameraCentreFromColmap(q, t);
      const projected = applyMat3(r, c);
      expectVecClose([projected[0] + t[0], projected[1] + t[1], projected[2] + t[2]], [0, 0, 0], 9);
    }
  });

  it("computes -Rᵀt, which is NOT t and NOT -t (the classic COLMAP bug)", () => {
    // 90 degrees about z, camera centre (1, 2, 3)  ->  t = -R C = (2, -1, -3).
    const t = translationFor(rotationFor(QUAT_Z90_WXYZ), [1, 2, 3]);
    expectVecClose(t, [2, -1, -3], 12);

    const centre = cameraCentreFromColmap(QUAT_Z90_WXYZ, t);
    expectVecClose(centre, [1, 2, 3], 12);

    // Prove the assertion above can fail: the two shortcuts land somewhere else.
    const usingPlusT = t;
    const usingMinusT = [-t[0], -t[1], -t[2]];
    expect(Math.hypot(centre[0] - usingPlusT[0], centre[1] - usingPlusT[1], centre[2] - usingPlusT[2]))
      .toBeGreaterThan(1);
    expect(Math.hypot(centre[0] - usingMinusT[0], centre[1] - usingMinusT[1], centre[2] - usingMinusT[2]))
      .toBeGreaterThan(1);
  });

  it("rejects a malformed translation", () => {
    expect(() => cameraCentreFromColmap(QUAT_Z90_WXYZ, [1, 2])).toThrow(/tvec/);
    expect(() => cameraCentreFromColmap(QUAT_Z90_WXYZ, [1, 2, Number.POSITIVE_INFINITY])).toThrow(/tvec/);
  });
});

describe("RDF_TO_RUB", () => {
  it("is diag(1, -1, -1)", () => {
    expect(RDF_TO_RUB).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1]);
  });

  it("applied twice is the identity — and applied once is not", () => {
    expectVecClose(multiplyMat3(RDF_TO_RUB, RDF_TO_RUB), IDENTITY_MAT3, 12);

    const v = [3, 5, 7];
    const once = applyMat3(RDF_TO_RUB, v);
    const twice = applyMat3(RDF_TO_RUB, once);
    expect(twice).toEqual([3, 5, 7]);
    // The whole point of contract §3: a doubled flip is silently a no-op, so a viewer
    // that flips twice looks exactly like one that never flipped.
    expect(once).not.toEqual([3, 5, 7]);
    expect(once).toEqual([3, -5, -7]);
    expect(multiplyMat3(RDF_TO_RUB, RDF_TO_RUB)).not.toEqual([...RDF_TO_RUB]);
  });
});

describe("cameraBasisFromColmap", () => {
  it("is diag(1,-1,-1) for the identity pose, i.e. a camera looking down world +z", () => {
    const basis = cameraBasisFromColmap([1, 0, 0, 0], [0, 0, 0]);
    expectVecClose(basis, RDF_TO_RUB, 12);
    // Column 2 is the camera's backward axis; backward = -z means it looks toward +z,
    // which is exactly COLMAP's convention for an identity rotation.
    expect([basis[2], basis[5], basis[8]]).toEqual([0, 0, -1]);
    // Column 1 is up: COLMAP's +y is down, so world up is -y.
    expect([basis[1], basis[4], basis[7]]).toEqual([0, -1, 0]);
  });

  it("equals Rᵀ · RDF_TO_RUB for every pose", () => {
    for (const { q, centre } of POSES) {
      const r = rotationFor(q);
      const t = translationFor(r, centre);
      expectVecClose(
        cameraBasisFromColmap(q, t),
        multiplyMat3(transposeMat3(r), RDF_TO_RUB),
        12
      );
    }
  });

  it("stays orthonormal (it is a rotation composed with a reflection)", () => {
    for (const { q, centre } of POSES) {
      const basis = cameraBasisFromColmap(q, translationFor(rotationFor(q), centre));
      expectVecClose(multiplyMat3(basis, transposeMat3(basis)), IDENTITY_MAT3, 12);
    }
  });

  it("is not Rᵀ alone — the flip is genuinely applied", () => {
    const r = rotationFor(QUAT_Z90_WXYZ);
    const basis = cameraBasisFromColmap(QUAT_Z90_WXYZ, [0, 0, 0]);
    expect(basis).not.toEqual([...transposeMat3(r)]);
  });

  it("validates tvec even though orientation does not depend on it", () => {
    expect(() => cameraBasisFromColmap(QUAT_Z90_WXYZ, [1, 2])).toThrow(/tvec/);
  });
});

describe("applyMat3", () => {
  it("treats the matrix as row-major times a column vector", () => {
    const m: Mat3 = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(applyMat3(m, [1, 0, 0])).toEqual([1, 4, 7]);
    expect(applyMat3(m, [0, 1, 0])).toEqual([2, 5, 8]);
    expect(applyMat3(m, [1, 1, 1])).toEqual([6, 15, 24]);
  });
});
