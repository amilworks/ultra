import { describe, expect, it } from "vitest";

import { ZOOM_STEP, formatZoomReadout, isAtFitScale, nativeZoomLevel } from "./viewerZoom";

describe("viewer zoom helpers", () => {
  it("uses one shared zoom step", () => {
    expect(ZOOM_STEP).toBeGreaterThan(1);
    expect(ZOOM_STEP).toBeLessThan(2);
  });

  it("formats the zoom readout as % of native (shared by both 2D surfaces)", () => {
    expect(formatZoomReadout(1, true)).toBe("Fit");
    expect(formatZoomReadout(1, false)).toBe("100%");
    expect(formatZoomReadout(0.5, false)).toBe("50%");
    expect(formatZoomReadout(2.5, false)).toBe("250%");
    // Fine detail below 10% keeps one decimal so the readout never reads "0%".
    expect(formatZoomReadout(0.05, false)).toBe("5.0%");
    // Mid values round to whole percent.
    expect(formatZoomReadout(0.182, false)).toBe("18%");
    // Very high zoom rounds to the nearest 100% to stay short.
    expect(formatZoomReadout(12, false)).toBe("1200%");
  });

  it("converts world-unit viewport scale to '% of native' identically across surfaces", () => {
    // Isotropic image (world == pixel grid): scale is already screen-px per source-px.
    expect(nativeZoomLevel(1, 6000, 6000)).toBe(1);
    expect(nativeZoomLevel(0.5, 6000, 6000)).toBe(0.5);
    // Anisotropic / physical spacing (0.25 units per source px): a 6000px image with a
    // 1500-unit world. At scale 4 (px per world unit) that is exactly 100% of native.
    expect(nativeZoomLevel(4, 1500, 6000)).toBeCloseTo(1, 10);
    expect(formatZoomReadout(nativeZoomLevel(4, 1500, 6000), false)).toBe("100%");
    // Guard against a zero/invalid pixel width (falls back to the raw scale).
    expect(nativeZoomLevel(2, 1500, 0)).toBe(2);
  });

  it("treats scales within tolerance of the fit scale as 'Fit'", () => {
    expect(isAtFitScale(0.18, 0.18)).toBe(true);
    expect(isAtFitScale(0.1805, 0.18)).toBe(true); // within 0.025 tolerance
    expect(isAtFitScale(0.36, 0.18)).toBe(false); // 2x fit
    expect(isAtFitScale(Number.NaN, 0.18)).toBe(true); // non-finite defaults to Fit
  });
});
