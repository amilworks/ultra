import { describe, expect, it } from "vitest";

import {
  describeDecimation,
  maxElementsFor,
  MAX_SCENE_PIXEL_RATIO,
  MAX_SCENE_PIXEL_RATIO_MOBILE,
  POINT_GPU_BYTES,
  resolveScenePixelRatio,
  SCENE_BUDGET_BYTES_DESKTOP,
  SCENE_BUDGET_BYTES_MOBILE,
  SPLAT_GPU_BYTES,
} from "./sceneBudget";

/** Measured on the real files (contract Appendix A). */
const SPLAT_TOTAL = 14_469_103;
const POINT_TOTAL = 2_068_089;

describe("resolveScenePixelRatio", () => {
  it("caps at 2 on desktop, mirroring resolveVolumePixelRatio", () => {
    expect(MAX_SCENE_PIXEL_RATIO).toBe(2);
    expect(resolveScenePixelRatio(3, false)).toBe(2);
    expect(resolveScenePixelRatio(2, false)).toBe(2);
    expect(resolveScenePixelRatio(1, false)).toBe(1);
    expect(resolveScenePixelRatio(1.5, false)).toBe(1.5);
  });

  it("caps at 1.5 on mobile — the splat pass is fill-rate bound", () => {
    expect(MAX_SCENE_PIXEL_RATIO_MOBILE).toBe(1.5);
    expect(resolveScenePixelRatio(3, true)).toBe(1.5);
    expect(resolveScenePixelRatio(2, true)).toBe(1.5);
    expect(resolveScenePixelRatio(1.25, true)).toBe(1.25);
  });

  it("falls back to 1 for a missing or nonsense devicePixelRatio", () => {
    expect(resolveScenePixelRatio(0, false)).toBe(1);
    expect(resolveScenePixelRatio(-2, true)).toBe(1);
    expect(resolveScenePixelRatio(Number.NaN, false)).toBe(1);
    expect(resolveScenePixelRatio(Number.POSITIVE_INFINITY, false)).toBe(1);
  });
});

describe("maxElementsFor", () => {
  it("derives ceilings from an explicit byte budget", () => {
    const desktop = maxElementsFor({ isMobile: false });
    expect(desktop.splats).toBe(Math.floor(SCENE_BUDGET_BYTES_DESKTOP / SPLAT_GPU_BYTES));
    expect(desktop.points).toBe(Math.floor(SCENE_BUDGET_BYTES_DESKTOP / POINT_GPU_BYTES));
    expect(desktop).toEqual({ points: 10_000_000, splats: 4_000_000 });
  });

  it("is far smaller on mobile", () => {
    const mobile = maxElementsFor({ isMobile: true });
    expect(mobile).toEqual({ points: 2_500_000, splats: 1_000_000 });
    expect(mobile.splats).toBe(Math.floor(SCENE_BUDGET_BYTES_MOBILE / SPLAT_GPU_BYTES));
    expect(mobile.splats).toBeLessThan(maxElementsFor({ isMobile: false }).splats);
  });

  it("holds more points than splats — points are 16 B, splats 40 B", () => {
    for (const isMobile of [false, true]) {
      const budget = maxElementsFor({ isMobile });
      expect(budget.points).toBeGreaterThan(budget.splats);
    }
  });

  it("scales with navigator.deviceMemory", () => {
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 1 }).splats).toBe(2_000_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 2 }).splats).toBe(3_000_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 4 }).splats).toBe(4_000_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 8 }).splats).toBe(6_000_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 64 }).splats).toBe(6_000_000);
  });

  it("is monotonic in device memory", () => {
    const ladder = [0.25, 1, 1.9, 2, 3.5, 4, 7.9, 8, 32].map(
      (gb) => maxElementsFor({ isMobile: false, deviceMemoryGb: gb }).splats
    );
    for (let i = 1; i < ladder.length; i += 1) {
      expect(ladder[i]).toBeGreaterThanOrEqual(ladder[i - 1]);
    }
  });

  it("assumes the baseline when deviceMemory is absent or nonsense — never the best case", () => {
    const baseline = maxElementsFor({ isMobile: false });
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: undefined })).toEqual(baseline);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: Number.NaN })).toEqual(baseline);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 0 })).toEqual(baseline);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: -4 })).toEqual(baseline);
    // Safari reports nothing; it must not silently get the 8 GB tier.
    expect(baseline.splats).toBeLessThan(
      maxElementsFor({ isMobile: false, deviceMemoryGb: 8 }).splats
    );
  });

  it("returns whole elements", () => {
    for (const gb of [0.25, 1, 2, 4, 8]) {
      for (const isMobile of [false, true]) {
        const budget = maxElementsFor({ isMobile, deviceMemoryGb: gb });
        expect(Number.isInteger(budget.points)).toBe(true);
        expect(Number.isInteger(budget.splats)).toBe(true);
      }
    }
  });

  it("cannot hold the real 14.5M-splat file on any tier — decimation must be surfaced", () => {
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 64 }).splats).toBeLessThan(SPLAT_TOTAL);
    // The 2M-point file does fit on desktop, so points are usually shown whole.
    expect(maxElementsFor({ isMobile: false }).points).toBeGreaterThan(POINT_TOTAL);
  });
});

describe("describeDecimation", () => {
  it("names both numbers when elements were dropped", () => {
    expect(describeDecimation(2_000_000, SPLAT_TOTAL)).toBe("showing 2,000,000 of 14,469,103");
    expect(describeDecimation(1, 2)).toBe("showing 1 of 2");
  });

  it("says so plainly when nothing was dropped", () => {
    expect(describeDecimation(SPLAT_TOTAL, SPLAT_TOTAL)).toBe("showing all 14,469,103");
    expect(describeDecimation(POINT_TOTAL, POINT_TOTAL)).toBe("showing all 2,068,089");
    expect(describeDecimation(0, 0)).toBe("showing all 0");
  });

  it("never claims more than the source holds", () => {
    expect(describeDecimation(99_000_000, SPLAT_TOTAL)).toBe("showing all 14,469,103");
  });

  it("always states the total, so a reader can tell what they are not seeing", () => {
    for (const shown of [0, 1, 1_000, 14_469_102]) {
      expect(describeDecimation(shown, SPLAT_TOTAL)).toContain("14,469,103");
    }
  });

  it("does not round — 14,469,102 of 14,469,103 is still a decimated scene", () => {
    expect(describeDecimation(14_469_102, SPLAT_TOTAL)).toBe("showing 14,469,102 of 14,469,103");
  });

  it("survives non-finite or fractional inputs", () => {
    expect(describeDecimation(Number.NaN, SPLAT_TOTAL)).toBe("showing 0 of 14,469,103");
    expect(describeDecimation(-5, SPLAT_TOTAL)).toBe("showing 0 of 14,469,103");
    expect(describeDecimation(1_500.9, 3_000)).toBe("showing 1,500 of 3,000");
    expect(describeDecimation(10, Number.NaN)).toBe("showing all 0");
  });
});
