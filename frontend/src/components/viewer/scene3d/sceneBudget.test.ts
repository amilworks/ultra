import { describe, expect, it } from "vitest";

import {
  describeDecimation,
  hasPagedLodWork,
  INTERACTIVE_LOD_SCALE,
  INTERACTIVE_LOD_RENDER_SCALE,
  maxElementsFor,
  MAX_SCENE_PIXEL_RATIO,
  MAX_SCENE_PIXEL_RATIO_MOBILE,
  PAGED_LOD_BOOTSTRAP_MS,
  POINT_GPU_BYTES,
  resolvePagedSplatPool,
  resolveScenePixelRatio,
  resolveSplatLodBudget,
  SCENE_BUDGET_BYTES_DESKTOP,
  SCENE_BUDGET_BYTES_MOBILE,
  SETTLED_SPLAT_LIMIT_DESKTOP,
  SETTLED_LOD_RENDER_SCALE,
  SPLAT_GPU_BYTES,
  splatResidentBytesForSh,
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
    expect(desktop).toEqual({ points: 3_600_000, splats: 1_500_000 });
  });

  it("is far smaller on mobile", () => {
    const mobile = maxElementsFor({ isMobile: true });
    expect(mobile).toEqual({ points: 300_000, splats: 125_000 });
    expect(mobile.splats).toBe(Math.floor(SCENE_BUDGET_BYTES_MOBILE / SPLAT_GPU_BYTES));
    expect(mobile.splats).toBeLessThan(maxElementsFor({ isMobile: false }).splats);
  });

  it("holds more points than splats after accounting for CPU, GPU and sort copies", () => {
    for (const isMobile of [false, true]) {
      const budget = maxElementsFor({ isMobile });
      expect(budget.points).toBeGreaterThan(budget.splats);
    }
  });

  it("scales with navigator.deviceMemory", () => {
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 1 }).splats).toBe(750_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 2 }).splats).toBe(1_125_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 4 }).splats).toBe(1_500_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 8 }).splats).toBe(3_000_000);
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 64 }).splats).toBe(3_000_000);
  });

  it("budgets the exact SH planes retained by Spark's paged representation", () => {
    expect(splatResidentBytesForSh(0)).toBe(64);
    expect(splatResidentBytesForSh(1)).toBe(96);
    expect(splatResidentBytesForSh(2)).toBe(128);
    expect(splatResidentBytesForSh(3)).toBe(192);
    expect(splatResidentBytesForSh(99)).toBe(SPLAT_GPU_BYTES);
    expect(splatResidentBytesForSh(undefined)).toBe(SPLAT_GPU_BYTES);

    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 8, splatShDegree: 0 }).splats).toBe(
      9_000_000
    );
    expect(maxElementsFor({ isMobile: false, deviceMemoryGb: 8, splatShDegree: 3 }).splats).toBe(
      3_000_000
    );
  });

  it("admits the estate fixture's first recognisable complete refinement on desktop", () => {
    expect(maxElementsFor({ isMobile: false }).splats).toBeGreaterThanOrEqual(527_577);
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

  it("uses a bounded interactive fraction before restoring the settled budget", () => {
    expect(INTERACTIVE_LOD_SCALE).toBe(0.25);
    expect(INTERACTIVE_LOD_SCALE).toBeGreaterThan(0);
    expect(INTERACTIVE_LOD_SCALE).toBeLessThan(1);
    expect(SETTLED_LOD_RENDER_SCALE).toBe(0.5);
    expect(INTERACTIVE_LOD_RENDER_SCALE).toBe(1);
    expect(SETTLED_LOD_RENDER_SCALE).toBeLessThan(INTERACTIVE_LOD_RENDER_SCALE);
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
    // The 2M-point fixture fits the resident-memory model at the desktop baseline.
    expect(maxElementsFor({ isMobile: false }).points).toBeGreaterThan(POINT_TOTAL);
  });
});

describe("resolveSplatLodBudget", () => {
  it("caps the measured SH0 estate scene for responsive settled and interactive views", () => {
    const hardCeiling = maxElementsFor({
      isMobile: false,
      deviceMemoryGb: 8,
      splatShDegree: 0,
    }).splats;

    expect(hardCeiling).toBe(9_000_000);
    expect(resolveSplatLodBudget({ hardCeiling, isMobile: false })).toEqual({
      settled: SETTLED_SPLAT_LIMIT_DESKTOP,
      interactive: 1_000_000,
      interactiveScale: 0.25,
    });
  });

  it("reserves paged-neighbour headroom inside the hard resident-memory ceiling", () => {
    const hardCeiling = maxElementsFor({
      isMobile: false,
      deviceMemoryGb: 8,
      splatShDegree: 3,
    }).splats;
    const budget = resolveSplatLodBudget({ hardCeiling, isMobile: false });
    const pagedPool = resolvePagedSplatPool(budget.settled, hardCeiling);

    expect(budget.settled).toBeLessThan(SETTLED_SPLAT_LIMIT_DESKTOP);
    expect(budget.interactive).toBe(Math.floor(budget.settled * 0.25));
    expect(pagedPool).toBeGreaterThanOrEqual(budget.settled);
    expect(pagedPool).toBeLessThanOrEqual(hardCeiling);
  });

  it("never raises a low device ceiling and returns finite ratios for empty input", () => {
    const low = resolveSplatLodBudget({ hardCeiling: 100_000, isMobile: true });
    expect(low.settled).toBeLessThanOrEqual(100_000);
    expect(low.interactive).toBeLessThan(low.settled);
    expect(low.interactiveScale).toBeGreaterThan(0);
    expect(low.interactiveScale).toBeLessThan(1);

    expect(resolveSplatLodBudget({ hardCeiling: 0, isMobile: false })).toEqual({
      settled: 0,
      interactive: 0,
      interactiveScale: 1,
    });
  });

  it("bounds the estate page pool far below the previous 11.27M-slot allocation", () => {
    const budget = resolveSplatLodBudget({ hardCeiling: 9_000_000, isMobile: false });
    expect(resolvePagedSplatPool(budget.settled, 9_000_000)).toBe(5_046_272);
  });
});

describe("hasPagedLodWork", () => {
  const idle = {
    fetchers: 0,
    fetched: 0,
    newUploads: 0,
    readyUploads: 0,
    lodTreeUpdates: 0,
  };

  it("stops the render pump once every asynchronous pager queue is empty", () => {
    expect(hasPagedLodWork(idle)).toBe(false);
    expect(PAGED_LOD_BOOTSTRAP_MS).toBe(30_000);
  });

  it.each(Object.keys(idle) as Array<keyof typeof idle>)(
    "keeps rendering while %s still has work",
    (queue) => {
      expect(hasPagedLodWork({ ...idle, [queue]: 1 })).toBe(true);
    }
  );
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
