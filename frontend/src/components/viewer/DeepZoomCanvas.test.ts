import { describe, expect, it } from "vitest";

import { chooseTileLevel, niceScaleBar, orderTileLevelsForRendering } from "./DeepZoomCanvas";

describe("orderTileLevelsForRendering", () => {
  it("puts the overview level first even when backend metadata lists full resolution first", () => {
    const levels = [
      { level: 0, width: 95174, height: 91416, columns: 372, rows: 358, downsample: 1 },
      { level: 1, width: 47587, height: 45708, columns: 186, rows: 179, downsample: 2 },
      { level: 7, width: 743, height: 714, columns: 3, rows: 3, downsample: 128 },
      { level: 8, width: 371, height: 357, columns: 2, rows: 2, downsample: 256 },
    ];

    expect(orderTileLevelsForRendering(levels).map((level) => level.level)).toEqual([8, 7, 1, 0]);
  });
});

describe("chooseTileLevel", () => {
  // A real ~95k-px orthomosaic pyramid (downsample 1..256).
  const levels = [
    { level: 8, downsample: 256 },
    { level: 7, downsample: 128 },
    { level: 6, downsample: 64 },
    { level: 5, downsample: 32 },
    { level: 4, downsample: 16 },
    { level: 0, downsample: 1 },
  ];
  const WORLD = 95174;
  const FULL = 95174; // world == pixel grid

  it("does NOT force the coarsest overview at the fit view (the blurry-until-zoom bug)", () => {
    // Fit: the whole 95174-px image shown in a ~1900px viewport. The resolution-matched
    // level is ~downsample 64, NOT the coarsest 256 the old shortcut returned.
    const chosen = chooseTileLevel(levels, 1900, WORLD, WORLD, FULL);
    expect(chosen.downsample).toBe(64);
    expect(chosen.downsample).not.toBe(256);
  });

  it("picks progressively finer levels as the visible region shrinks (zoom in)", () => {
    // Visible world width halves each step -> the chosen level should get finer.
    const fit = chooseTileLevel(levels, 1900, WORLD, WORLD, FULL).downsample;
    const half = chooseTileLevel(levels, 1900, WORLD / 4, WORLD, FULL).downsample;
    const deep = chooseTileLevel(levels, 1900, WORLD / 64, WORLD, FULL).downsample;
    expect(half).toBeLessThan(fit);
    expect(deep).toBeLessThan(half);
    // Zoomed in close, it reaches full resolution.
    expect(chooseTileLevel(levels, 1900, WORLD / 256, WORLD, FULL).downsample).toBe(1);
  });
});

describe("niceScaleBar", () => {
  it("returns a round value near the target width", () => {
    // 2 screen px per unit, target 120px => ~60 units -> rounds to 50
    const bar = niceScaleBar(2, "m", 120);
    expect(bar).not.toBeNull();
    expect(bar!.label).toBe("50 m");
    expect(bar!.widthPx).toBeCloseTo(100, 0); // 50 * 2
  });

  it("promotes µm up to mm/m for readability", () => {
    // 0.002 screen px per µm, target 120 => 60000 µm -> promote to 50 mm
    const bar = niceScaleBar(0.002, "um", 120);
    expect(bar).not.toBeNull();
    expect(bar!.label).toBe("50 mm");
  });

  it("returns null for an unusable scale", () => {
    expect(niceScaleBar(0, "m")).toBeNull();
    expect(niceScaleBar(Number.NaN, "m")).toBeNull();
  });
});
