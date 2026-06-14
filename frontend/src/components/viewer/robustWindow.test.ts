import { describe, expect, it } from "vitest";

import { computeRobustHistogramWindow } from "./ImageViewerShell";

describe("computeRobustHistogramWindow", () => {
  it("excludes a single hot-voxel outlier from the default window (MR robustness)", () => {
    // Bulk of the signal is in [0, 100]; one bright outlier sits near 1000.
    const bins = new Array(100).fill(0);
    for (let i = 0; i < 10; i += 1) bins[i] = 1000; // dense low-intensity tissue
    bins[99] = 1; // a single hot voxel at the top of the range
    const window = computeRobustHistogramWindow(
      { histogram: { min: 0, max: 1000, bins } },
      0.01,
      0.99
    );
    expect(window).not.toBeNull();
    // p99 must land in the dense region (~<=110), not be dragged to ~1000.
    expect(window!.max).toBeLessThan(150);
    expect(window!.min).toBeLessThan(window!.max);
  });

  it("returns null for empty or degenerate histograms", () => {
    expect(computeRobustHistogramWindow(null, 0.01, 0.99)).toBeNull();
    expect(computeRobustHistogramWindow({ histogram: { min: 0, max: 0, bins: [] } }, 0.01, 0.99)).toBeNull();
    expect(computeRobustHistogramWindow({ histogram: { min: 5, max: 5, bins: [1, 2] } }, 0.01, 0.99)).toBeNull();
  });
});
