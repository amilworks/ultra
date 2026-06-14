import { describe, expect, it } from "vitest";

import { niceScaleBar, orderTileLevelsForRendering } from "./DeepZoomCanvas";

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
