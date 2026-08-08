import { describe, expect, it } from "vitest";

import {
  focalOf,
  hasDistortion,
  projectionMatrixFor,
  SUPPORTED_COLMAP_MODELS,
  verticalFovDeg,
  type ColmapCamera,
} from "./sceneIntrinsics";

/**
 * The adversarial fixture: fx != fy AND a principal point well away from the image
 * centre. Centre would be (640, 360); this camera's is (700, 300).
 */
const OFF_CENTRE: ColmapCamera = {
  model: "PINHOLE",
  width: 1280,
  height: 720,
  params: [1200, 900, 700, 300],
};

/** Same sensor, principal point exactly centred — the control case. */
const CENTRED: ColmapCamera = {
  model: "PINHOLE",
  width: 1280,
  height: 720,
  params: [1200, 900, 640, 360],
};

/** Project a camera-space point through a column-major GL matrix back to pixels. */
const projectToPixel = (
  m: number[],
  cam: readonly [number, number, number],
  width: number,
  height: number
): { u: number; v: number; ndcZ: number } => {
  const [x, y, z] = cam;
  const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
  const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
  const clipZ = m[2] * x + m[6] * y + m[10] * z + m[14];
  const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];
  return {
    u: ((clipX / clipW + 1) / 2) * width,
    v: ((1 - clipY / clipW) / 2) * height,
    ndcZ: clipZ / clipW,
  };
};

/** Back-project a pixel at depth d into the RUB camera frame. */
const unproject = (
  c: ColmapCamera,
  u: number,
  v: number,
  depth: number
): [number, number, number] => {
  const { fx, fy, cx, cy } = focalOf(c);
  return [((u - cx) * depth) / fx, (-(v - cy) * depth) / fy, -depth];
};

describe("focalOf", () => {
  it("reads the shared focal of the SIMPLE_* models into both axes", () => {
    expect(focalOf({ model: "SIMPLE_PINHOLE", width: 640, height: 480, params: [500, 320, 240] }))
      .toEqual({ fx: 500, fy: 500, cx: 320, cy: 240 });
  });

  it("keeps fx and fy distinct for the two-focal models", () => {
    expect(focalOf(OFF_CENTRE)).toEqual({ fx: 1200, fy: 900, cx: 700, cy: 300 });
  });

  it("reads the pinhole core of every supported model", () => {
    const fixtures: ColmapCamera[] = [
      { model: "SIMPLE_PINHOLE", width: 8, height: 8, params: [7, 3, 4] },
      { model: "PINHOLE", width: 8, height: 8, params: [7, 7, 3, 4] },
      { model: "SIMPLE_RADIAL", width: 8, height: 8, params: [7, 3, 4, 0.1] },
      { model: "RADIAL", width: 8, height: 8, params: [7, 3, 4, 0.1, 0.2] },
      { model: "OPENCV", width: 8, height: 8, params: [7, 7, 3, 4, 0.1, 0.2, 0.3, 0.4] },
      { model: "OPENCV_FISHEYE", width: 8, height: 8, params: [7, 7, 3, 4, 0.1, 0.2, 0.3, 0.4] },
      {
        model: "FULL_OPENCV",
        width: 8,
        height: 8,
        params: [7, 7, 3, 4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      },
      { model: "FOV", width: 8, height: 8, params: [7, 7, 3, 4, 0.9] },
      { model: "SIMPLE_RADIAL_FISHEYE", width: 8, height: 8, params: [7, 3, 4, 0.1] },
      { model: "RADIAL_FISHEYE", width: 8, height: 8, params: [7, 3, 4, 0.1, 0.2] },
      {
        model: "THIN_PRISM_FISHEYE",
        width: 8,
        height: 8,
        params: [7, 7, 3, 4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      },
    ];
    expect(fixtures.map((c) => c.model).sort()).toEqual([...SUPPORTED_COLMAP_MODELS].sort());
    for (const camera of fixtures) {
      expect(focalOf(camera), camera.model).toEqual({ fx: 7, fy: 7, cx: 3, cy: 4 });
    }
  });

  it("throws on an unknown model rather than guessing a layout", () => {
    expect(() => focalOf({ model: "PERSPECTIVE", width: 8, height: 8, params: [1, 2, 3] }))
      .toThrow(/unsupported COLMAP camera model/);
  });

  it("throws when params has the wrong length for the model", () => {
    // OPENCV needs 8. Reading a truncated record positionally would silently return
    // whatever the next field happened to be.
    expect(() => focalOf({ model: "OPENCV", width: 8, height: 8, params: [7, 7, 3, 4] }))
      .toThrow(/expects 8 params, got 4/);
  });
});

describe("hasDistortion", () => {
  it("is false for the pinhole models", () => {
    expect(hasDistortion({ model: "PINHOLE", width: 8, height: 8, params: [7, 7, 3, 4] })).toBe(false);
    expect(hasDistortion({ model: "SIMPLE_PINHOLE", width: 8, height: 8, params: [7, 3, 4] })).toBe(false);
  });

  it("is false when a distortion model carries all-zero coefficients", () => {
    // COLMAP emits SIMPLE_RADIAL with k = 0 for already-undistorted images; claiming
    // distortion there would put a false caveat in the provenance panel.
    expect(hasDistortion({ model: "SIMPLE_RADIAL", width: 8, height: 8, params: [7, 3, 4, 0] })).toBe(false);
    expect(
      hasDistortion({ model: "OPENCV", width: 8, height: 8, params: [7, 7, 3, 4, 0, 0, 0, 0] })
    ).toBe(false);
  });

  it("is true as soon as one coefficient is non-zero", () => {
    expect(hasDistortion({ model: "SIMPLE_RADIAL", width: 8, height: 8, params: [7, 3, 4, -1e-6] })).toBe(true);
    expect(
      hasDistortion({ model: "OPENCV", width: 8, height: 8, params: [7, 7, 3, 4, 0, 0, 0, 0.02] })
    ).toBe(true);
    expect(
      hasDistortion({
        model: "THIN_PRISM_FISHEYE",
        width: 8,
        height: 8,
        params: [7, 7, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3e-4],
      })
    ).toBe(true);
  });
});

describe("projectionMatrixFor", () => {
  it("returns 16 column-major elements with the GL perspective signature", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, 100);
    expect(m).toHaveLength(16);
    expect(m[11]).toBe(-1);
    expect(m[15]).toBe(0);
    // The rows GL leaves empty.
    for (const i of [1, 2, 3, 4, 6, 7, 12, 13]) {
      expect(m[i], `element ${i}`).toBe(0);
    }
  });

  it("scales x and y independently when fx != fy", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, 100);
    // 2n/(r-l) reduces to 2fx/width, and 2n/(t-b) to 2fy/height.
    expect(m[0]).toBeCloseTo((2 * 1200) / 1280, 12);
    expect(m[5]).toBeCloseTo((2 * 900) / 720, 12);
    expect(m[0]).not.toBeCloseTo(m[5], 6);
  });

  it("produces an ASYMMETRIC frustum for an off-centre principal point", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, 100);
    // (r+l)/(r-l) = (width - 2cx)/width, (t+b)/(t-b) = (2cy - height)/height.
    expect(m[8]).toBeCloseTo((1280 - 2 * 700) / 1280, 12);
    expect(m[9]).toBeCloseTo((2 * 300 - 720) / 720, 12);
    expect(m[8]).toBeCloseTo(-0.09375, 12);
    expect(m[9]).toBeCloseTo(-0.16666666666, 9);

    // These two entries ARE the asymmetry. A symmetric frustum has both at zero, so a
    // viewer built on PerspectiveCamera(fov, aspect) cannot reach this matrix at all.
    expect(m[8]).not.toBe(0);
    expect(m[9]).not.toBe(0);
  });

  it("collapses to a symmetric frustum only when the principal point is centred", () => {
    const m = projectionMatrixFor(CENTRED, 0.1, 100);
    expect(m[8]).toBeCloseTo(0, 12);
    expect(m[9]).toBeCloseTo(0, 12);
  });

  it("puts the optical axis at (cx, cy) — where a symmetric viewer would put (w/2, h/2)", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, 100);
    // The ray straight down the camera's -z axis.
    const axis = projectToPixel(m, [0, 0, -5], OFF_CENTRE.width, OFF_CENTRE.height);
    expect(axis.u).toBeCloseTo(700, 9);
    expect(axis.v).toBeCloseTo(300, 9);

    // What a cx/cy-ignoring viewer would produce: the same matrix with the shear zeroed.
    const symmetric = [...m];
    symmetric[8] = 0;
    symmetric[9] = 0;
    const ignored = projectToPixel(symmetric, [0, 0, -5], OFF_CENTRE.width, OFF_CENTRE.height);
    expect(ignored.u).toBeCloseTo(640, 9);
    expect(ignored.v).toBeCloseTo(360, 9);
    // 60 px and 60 px of silent error — this is the assertion that fails for a viewer
    // that ignores the principal point.
    expect(Math.abs(ignored.u - axis.u)).toBeGreaterThan(50);
    expect(Math.abs(ignored.v - axis.v)).toBeGreaterThan(50);
  });

  it("round-trips arbitrary pixels through unproject -> project", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.05, 500);
    for (const [u, v, depth] of [
      [0, 0, 1],
      [1280, 720, 1],
      [700, 300, 12.5],
      [37, 611, 0.25],
      [1279, 3, 400],
    ] as const) {
      const projected = projectToPixel(m, unproject(OFF_CENTRE, u, v, depth), 1280, 720);
      expect(projected.u).toBeCloseTo(u, 6);
      expect(projected.v).toBeCloseTo(v, 6);
    }
  });

  it("maps the near plane to ndc z = -1 and the far plane to +1", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, 100);
    expect(projectToPixel(m, [0, 0, -0.1], 1280, 720).ndcZ).toBeCloseTo(-1, 9);
    expect(projectToPixel(m, [0, 0, -100], 1280, 720).ndcZ).toBeCloseTo(1, 9);
  });

  it("supports an infinite far plane", () => {
    const m = projectionMatrixFor(OFF_CENTRE, 0.1, Number.POSITIVE_INFINITY);
    expect(m[10]).toBe(-1);
    expect(m[14]).toBeCloseTo(-0.2, 12);
    expect(projectToPixel(m, [0, 0, -0.1], 1280, 720).ndcZ).toBeCloseTo(-1, 9);
    expect(projectToPixel(m, [0, 0, -1e9], 1280, 720).ndcZ).toBeCloseTo(1, 6);
    // The shear survives the infinite-far limit.
    expect(m[8]).toBeCloseTo(-0.09375, 12);
  });

  it("rejects a degenerate near/far range", () => {
    expect(() => projectionMatrixFor(OFF_CENTRE, 0, 100)).toThrow(/near/);
    expect(() => projectionMatrixFor(OFF_CENTRE, 1, 1)).toThrow(/far must be greater than near/);
    expect(() => projectionMatrixFor(OFF_CENTRE, 10, 1)).toThrow(/far must be greater than near/);
  });

  it("rejects a camera with no pixel extent", () => {
    expect(() => projectionMatrixFor({ ...OFF_CENTRE, width: 0 }, 0.1, 100)).toThrow(/width/);
    expect(() => projectionMatrixFor({ ...OFF_CENTRE, height: -720 }, 0.1, 100)).toThrow(/height/);
  });
});

describe("verticalFovDeg", () => {
  it("matches 2·atan(h / 2fy) when the principal point is centred", () => {
    expect(verticalFovDeg(CENTRED)).toBeCloseTo((2 * Math.atan(720 / (2 * 900)) * 180) / Math.PI, 12);
    expect(verticalFovDeg(CENTRED)).toBeCloseTo(43.60281897, 6);
  });

  it("reports the true extent when cy is off-centre, not the symmetric approximation", () => {
    const trueExtent = ((Math.atan(300 / 900) + Math.atan(420 / 900)) * 180) / Math.PI;
    expect(verticalFovDeg(OFF_CENTRE)).toBeCloseTo(trueExtent, 12);
    expect(verticalFovDeg(OFF_CENTRE)).toBeCloseTo(43.45183, 4);
    // The symmetric formula is wrong here, and the difference is what this function
    // exists to avoid papering over.
    const symmetric = (2 * Math.atan(720 / (2 * 900)) * 180) / Math.PI;
    expect(Math.abs(verticalFovDeg(OFF_CENTRE) - symmetric)).toBeGreaterThan(0.1);
  });
});
