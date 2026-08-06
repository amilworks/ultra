/**
 * Device budgets for the `scene3d` Lens modality: how many pixels we render into, how
 * many elements we are willing to hold, and the sentence we owe the user when the second
 * number is smaller than the source.
 *
 * The real file is 14,469,103 splats at 32 B each — 463 MB of GPU memory before Spark's
 * own sort buffers. Some ceiling exists on every device. Contract §5 is unambiguous
 * about what that means: **no silent decimation, ever.** If we show fewer elements than
 * the source holds, the readout says so, the manifest says so, and an exported
 * screenshot says so. `describeDecimation` is that sentence, and it is the reason these
 * ceilings are a pure function rather than a number buried in a render loop.
 *
 * Pure arithmetic — no three.js, no Spark, no DOM.
 */

/**
 * Device-pixel-ratio cap, mirroring `resolveVolumePixelRatio` in
 * `SliceStackVolumeCanvas.tsx`. A 3x phone panel costs 9x the fragments for a splat
 * pass that is already fill-rate bound, so mobile caps lower than desktop.
 */
export const MAX_SCENE_PIXEL_RATIO = 2;
export const MAX_SCENE_PIXEL_RATIO_MOBILE = 1.5;

export const resolveScenePixelRatio = (dpr: number, isMobile: boolean): number => {
  const safeRatio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  return Math.min(safeRatio, isMobile ? MAX_SCENE_PIXEL_RATIO_MOBILE : MAX_SCENE_PIXEL_RATIO);
};

/**
 * GPU bytes per element, so the ceilings below are derived from a memory budget rather
 * than picked out of the air.
 *
 * A splat is 32 B of `ExtSplats` payload (contract §4.2) plus Spark's index and sort-key
 * buffers, which is where the extra 8 B comes from. A point is 12 B of f32 position plus
 * a 4 B rgba colour.
 */
export const SPLAT_GPU_BYTES = 40;
export const POINT_GPU_BYTES = 16;

/** Element-memory budget before the device-memory factor. */
export const SCENE_BUDGET_BYTES_DESKTOP = 160_000_000;
export const SCENE_BUDGET_BYTES_MOBILE = 40_000_000;

/**
 * `navigator.deviceMemory` is a coarse, deliberately-rounded hint (0.25 … 8), and it is
 * absent on Safari entirely. Absent means "assume the baseline" — never "assume the
 * best", which would OOM the machines that withhold the number.
 */
const memoryFactor = (deviceMemoryGb?: number): number => {
  if (typeof deviceMemoryGb !== "number" || !Number.isFinite(deviceMemoryGb) || deviceMemoryGb <= 0) {
    return 1;
  }
  if (deviceMemoryGb < 2) return 0.5;
  if (deviceMemoryGb < 4) return 0.75;
  if (deviceMemoryGb < 8) return 1;
  return 1.5;
};

/**
 * Element ceilings for this device. Both species are reported even though points and
 * splats are mutually exclusive by default (contract §9), because the caller decides
 * which layer it is about to load.
 */
export const maxElementsFor = (opts: {
  isMobile: boolean;
  deviceMemoryGb?: number;
}): { points: number; splats: number } => {
  const budget =
    (opts.isMobile ? SCENE_BUDGET_BYTES_MOBILE : SCENE_BUDGET_BYTES_DESKTOP) *
    memoryFactor(opts.deviceMemoryGb);
  return {
    points: Math.floor(budget / POINT_GPU_BYTES),
    splats: Math.floor(budget / SPLAT_GPU_BYTES),
  };
};

// Explicit locale: this string is provenance. It goes in the readout and gets burned
// into exported screenshots, so it must not change shape with the viewer's locale.
const count = (value: number): string =>
  Math.max(0, Math.floor(Number.isFinite(value) ? value : 0)).toLocaleString("en-US");

/**
 * The honesty sentence. `"showing 2,000,000 of 14,469,103"` when elements were dropped,
 * `"showing all 14,469,103"` when none were.
 *
 * It never omits the total, and it never rounds — a reader has to be able to tell
 * whether they are looking at the whole scene.
 */
export const describeDecimation = (shown: number, total: number): string => {
  const safeTotal = Math.max(0, Math.floor(Number.isFinite(total) ? total : 0));
  const safeShown = Math.min(Math.max(0, Math.floor(Number.isFinite(shown) ? shown : 0)), safeTotal);
  if (safeShown >= safeTotal) {
    return `showing all ${count(safeTotal)}`;
  }
  return `showing ${count(safeShown)} of ${count(safeTotal)}`;
};
