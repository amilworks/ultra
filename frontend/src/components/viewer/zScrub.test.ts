import { describe, expect, it } from "vitest";

import {
  clampZ,
  isScrubbable,
  neighborSliceUrls,
  nextZFromWheel,
  wheelZStep,
  zFromPointerY,
} from "./zScrub";

describe("zScrub helpers", () => {
  it("clamps z into the stack range and rounds", () => {
    expect(clampZ(5, 10)).toBe(5);
    expect(clampZ(-3, 10)).toBe(0);
    expect(clampZ(20, 10)).toBe(9);
    expect(clampZ(2.6, 10)).toBe(3);
    expect(clampZ(4, 0)).toBe(0);
  });

  it("derives a signed plane step from wheel deltas", () => {
    expect(wheelZStep(0, 0)).toBe(0);
    expect(wheelZStep(100, 0)).toBe(2); // ~1 plane / 50px
    expect(wheelZStep(30, 0)).toBe(1); // minimum one plane per tick
    expect(wheelZStep(-100, 0)).toBe(-2);
    expect(wheelZStep(-1, 1)).toBe(-1); // line mode: one plane per tick
    expect(wheelZStep(5, 2)).toBe(1); // page mode: one plane per tick
  });

  it("computes the next z from a wheel event, clamped", () => {
    expect(nextZFromWheel(5, 100, 0, 10)).toBe(7);
    expect(nextZFromWheel(5, -100, 0, 10)).toBe(3);
    expect(nextZFromWheel(9, 100, 0, 10)).toBe(9); // clamp at top
    expect(nextZFromWheel(0, -100, 0, 10)).toBe(0); // clamp at bottom
  });

  it("only marks multi-plane stacks scrubbable", () => {
    expect(isScrubbable(1)).toBe(false);
    expect(isScrubbable(0)).toBe(false);
    expect(isScrubbable(12)).toBe(true);
  });

  it("maps a pointer's vertical offset to a z plane (top=first, bottom=last)", () => {
    expect(zFromPointerY(0, 100, 10)).toBe(0); // top edge -> first plane
    expect(zFromPointerY(100, 100, 10)).toBe(9); // bottom edge -> last plane
    expect(zFromPointerY(50, 100, 10)).toBe(5); // middle
    expect(zFromPointerY(-20, 100, 10)).toBe(0); // clamp above the thumbnail
    expect(zFromPointerY(999, 100, 10)).toBe(9); // clamp below
    expect(zFromPointerY(50, 0, 10)).toBe(0); // zero height is a no-op
    expect(zFromPointerY(50, 100, 0)).toBe(0); // no planes
  });

  it("builds de-duplicated, clamped neighbor URLs", () => {
    const f = (z: number) => `/slice?z=${z}`;
    expect(neighborSliceUrls(5, 10, 2, f)).toEqual(["/slice?z=3", "/slice?z=4", "/slice?z=6", "/slice?z=7"]);
    // At the bottom edge, clamped neighbors collapse onto 0 and are dropped.
    expect(neighborSliceUrls(0, 10, 2, f)).toEqual(["/slice?z=1", "/slice?z=2"]);
  });
});
