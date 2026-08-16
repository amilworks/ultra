import { describe, expect, it } from "vitest";

import {
  cameraDistanceToFrameSphere,
  frameSceneBounds,
  maxDistanceToSceneBounds,
  resolveSceneDepthPlan,
} from "./sceneBounds";

const POINT_FULL = [
  -39.299224853515625,
  -345.4450378417969,
  1.0418380498886108,
  1215.3311767578125,
  8.988668441772461,
  3199.109619140625,
];
const POINT_ROBUST = [
  -8.669791412353515,
  -5.132812767028809,
  2.135948781967163,
  15.183669013977099,
  0.91526864528656,
  23.78661300659181,
];
const ESTATE_FULL = [-61.283, -4.622, -50.352, 60.918, 22.482, 70.742];
const ESTATE_ROBUST = [-45.518, 5.127, -30.22, 43.842, 17.463, 54.311];

describe("scene bounds", () => {
  it("frames the robust reconstruction while keeping all measured point outliers in depth", () => {
    const plan = resolveSceneDepthPlan(POINT_ROBUST, POINT_FULL);

    expect(plan.focus.radius).toBeLessThan(20);
    expect(plan.far).toBeGreaterThan(
      maxDistanceToSceneBounds(plan.cameraPosition, POINT_FULL)
    );
    expect(plan.logarithmicDepthBuffer).toBe(true);
  });

  it("does not pay for logarithmic depth when full and robust bounds have comparable scale", () => {
    const plan = resolveSceneDepthPlan([-10, -8, -12, 10, 8, 12], [-12, -10, -14, 12, 10, 14]);

    expect(plan.logarithmicDepthBuffer).toBe(false);
    expect(plan.near).toBeGreaterThan(0);
    expect(plan.far).toBeGreaterThan(plan.near);
  });

  it("frames the complete robust estate sphere while retaining the exact depth range", () => {
    const view = { verticalFovDegrees: 50, aspect: 16 / 9 };
    const plan = resolveSceneDepthPlan(ESTATE_ROBUST, ESTATE_FULL, view);
    const distance = Math.hypot(
      plan.cameraPosition[0] - plan.focus.centre[0],
      plan.cameraPosition[1] - plan.focus.centre[1],
      plan.cameraPosition[2] - plan.focus.centre[2]
    );

    expect(distance).toBeCloseTo(cameraDistanceToFrameSphere(plan.focus.radius, view), 8);
    expect(distance / plan.focus.radius).toBeGreaterThan(2);
    expect(plan.far).toBeGreaterThan(maxDistanceToSceneBounds(plan.cameraPosition, ESTATE_FULL));
  });

  it("moves back when a narrow viewport makes horizontal field of view limiting", () => {
    const radius = 25;
    expect(cameraDistanceToFrameSphere(radius, { aspect: 0.6 })).toBeGreaterThan(
      cameraDistanceToFrameSphere(radius, { aspect: 16 / 9 })
    );
  });

  it("places the initial camera above a source whose signed up direction is negative Y", () => {
    const bounds = [-10, -4, -12, 10, 6, 12];
    // Cast through the current public view type so this regression test remains runnable
    // before the signed-up field is implemented.
    const view = { up: [0, -1, 0] } as unknown as Parameters<
      typeof resolveSceneDepthPlan
    >[2];
    const plan = resolveSceneDepthPlan(bounds, bounds, view);

    expect(plan.cameraPosition[1]).toBeLessThan(plan.focus.centre[1]);
  });

  it("falls back to a finite non-degenerate frame for malformed or flat bounds", () => {
    expect(frameSceneBounds([Number.NaN])).toEqual({ centre: [0, 0, 0], radius: Math.sqrt(3) });
    expect(frameSceneBounds([4, 4, 4, 4, 4, 4])).toEqual({ centre: [4, 4, 4], radius: 1 });
  });
});
