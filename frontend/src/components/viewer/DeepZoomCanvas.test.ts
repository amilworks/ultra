import { describe, expect, it } from "vitest";

import { computeAdaptiveMaxZoom, orderTileLevelsForRendering } from "./DeepZoomCanvas";

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

describe("computeAdaptiveMaxZoom", () => {
  it("lets a huge image reach (and exceed) native resolution", () => {
    // 95174px-wide image fit to a 1000px viewport (visible ~= worldWidth). A fixed
    // cap of 64 could never reach 1:1 (~95x); the adaptive cap must allow it.
    const worldWidth = 95174;
    const maxZoom = computeAdaptiveMaxZoom(1000, worldWidth, worldWidth, worldWidth);
    // ~2x native: 2 * 95174/1000 ~= 190
    expect(maxZoom).toBeGreaterThan(95); // can reach native 1:1
    expect(maxZoom).toBeCloseTo((2 * 95174) / 1000, 0);
  });

  it("is robust to normalized world units (world_size != pixel_size)", () => {
    // Same image but world normalized to ~1 unit wide; fullWidth still 95174px.
    const maxZoom = computeAdaptiveMaxZoom(1000, 1, 1, 95174);
    expect(maxZoom).toBeCloseTo((2 * 95174) / 1000, 0);
  });

  it("does not over-zoom a small image but still allows some zoom-in", () => {
    // A 400px image already near native at fit -> modest cap, floored at 8.
    const maxZoom = computeAdaptiveMaxZoom(800, 400, 400, 400);
    expect(maxZoom).toBe(8);
  });
});
