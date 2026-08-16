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
 * These are resident-memory estimates, not wire sizes. A splat arrives as 32 B but is
 * also represented by Spark input textures, sort keys/indices, accumulation targets,
 * readback buffers, and transient upload copies. A point arrives as 16 B but is retained
 * by the fetched buffer and WebGL attributes and may be copied by the driver. Budgeting
 * only the wire payload was the reason multi-million-element scenes killed the app.
 */
/**
 * Resident bytes for a degree-3 paged `ExtSplats` pool. Kept as the conservative
 * public/default estimate for callers that do not know the retained SH degree.
 *
 * Spark allocates two RGBA32UI planes for every extended splat, then allocates SH
 * planes lazily: one for degree 1, one more for degree 2, and two more for degree 3.
 * Each plane is 16 B on the CPU and 16 B on the GPU. The resulting exact pool cost is
 * therefore 64/96/128/192 B for degrees 0/1/2/3. Using the degree-3 cost for this
 * Postshot asset (whose 45 declared SH properties are all measured zero) unnecessarily
 * discarded two thirds of the available detail and made the estate look melted.
 */
export const SPLAT_GPU_BYTES = 192;
const SPLAT_RESIDENT_BYTES_BY_SH = [64, 96, 128, SPLAT_GPU_BYTES] as const;

export const splatResidentBytesForSh = (degree?: number): number => {
  if (typeof degree !== "number" || !Number.isFinite(degree)) {
    return SPLAT_GPU_BYTES;
  }
  const retainedDegree = Math.max(0, Math.min(3, Math.floor(degree)));
  return SPLAT_RESIDENT_BYTES_BY_SH[retainedDegree];
};
export const POINT_GPU_BYTES = 80;

/** Element-memory budget before the device-memory factor. */
// The estate fixture becomes structurally clear around one million active reconstructed
// nodes. The conservative degree-3 baseline admits 1.5M resident slots when
// `navigator.deviceMemory` is unavailable; Chrome's 8 GB hint admits 3M. Measured lower
// SH degrees may admit more resident slots, but the performance target below separately
// bounds visible work. Spark's native RAD traversal spends either budget on visible
// spatial detail rather than a uniform subset of source rows.
export const SCENE_BUDGET_BYTES_DESKTOP = 288_000_000;
export const SCENE_BUDGET_BYTES_MOBILE = 24_000_000;

/**
 * A resident-memory ceiling is necessary but not sufficient for an interactive splat
 * viewer. Sorting and blending nine million SH0 splats can fit in memory and still
 * monopolize the renderer. Spark's own desktop default is 2.5M; four million retains
 * the estate fixture's architectural detail after motion while remaining a bounded
 * exception for high-resolution displays. Lower-memory and higher-SH assets remain
 * constrained by their stricter resident-byte ceiling below.
 */
export const SETTLED_SPLAT_LIMIT_DESKTOP = 4_000_000;
export const SETTLED_SPLAT_LIMIT_MOBILE = 750_000;

/** Spark's paged RAD cache is allocated in fixed 65,536-splat pages. */
export const SPARK_PAGE_SPLATS = 65_536;

/**
 * Reserve neighbouring pages so a small orbit does not evict and immediately refetch
 * the visible hierarchy. This headroom is included inside the hard resident ceiling;
 * it is not an extra allocation added after budgeting.
 */
export const PAGED_SPLAT_POOL_HEADROOM = 1.25;

/**
 * Keep camera manipulation light, then restore the full scientific-detail budget when
 * the pointer is released. Spark reuses the same paged hierarchy for both passes, so
 * this changes only the selected LoD nodes; it never substitutes a different asset.
 */
export const INTERACTIVE_LOD_SCALE = 0.25;

/**
 * Spark's default stops traversal near one LoD node per screen pixel. The estate still
 * visibly merges roof and facade structure at that threshold, even with memory left.
 * Half-pixel settled nodes spend the available budget on native detail; interaction
 * returns to Spark's one-pixel threshold until the camera stops.
 */
export const SETTLED_LOD_RENDER_SCALE = 0.5;
export const INTERACTIVE_LOD_RENDER_SCALE = 1;

/**
 * Maximum time to keep asking Spark for frames while the first paged LoD node is
 * bootstrapping. A failed RAD header is surfaced before this pump starts; this bound is
 * only a guard against an unexpected worker/pager stall consuming the GPU forever.
 */
export const PAGED_LOD_BOOTSTRAP_MS = 30_000;

export type PagedLodQueueState = {
  fetchers: number;
  fetched: number;
  newUploads: number;
  readyUploads: number;
  lodTreeUpdates: number;
};

/**
 * Spark's pager fetches and decodes outside the render call, but consumes decoded pages
 * on a later render. Its queues therefore are the authoritative answer to whether an
 * otherwise on-demand canvas still owes the user another frame.
 */
export const hasPagedLodWork = (queues: PagedLodQueueState): boolean =>
  Object.values(queues).some((length) => Number.isFinite(length) && length > 0);

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
  return 2;
};

/**
 * Element ceilings for this device. Both species are reported even though points and
 * splats are mutually exclusive by default (contract §9), because the caller decides
 * which layer it is about to load.
 */
export const maxElementsFor = (opts: {
  isMobile: boolean;
  deviceMemoryGb?: number;
  /** Highest SH band retained by the rendered artifact, not merely declared by PLY. */
  splatShDegree?: number;
}): { points: number; splats: number } => {
  const budget =
    (opts.isMobile ? SCENE_BUDGET_BYTES_MOBILE : SCENE_BUDGET_BYTES_DESKTOP) *
    memoryFactor(opts.deviceMemoryGb);
  return {
    points: Math.floor(budget / POINT_GPU_BYTES),
    splats: Math.floor(budget / splatResidentBytesForSh(opts.splatShDegree)),
  };
};

export type SplatLodBudget = {
  /** Maximum reconstructed nodes used after navigation settles. */
  settled: number;
  /** Maximum reconstructed nodes used while the camera is moving. */
  interactive: number;
  /** Spark's interactive `lodSplatScale`, derived from the two explicit counts. */
  interactiveScale: number;
};

const safeElementCount = (value: number): number =>
  Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));

/**
 * Split a device's hard resident ceiling into an active LoD target and pager headroom.
 * Both targets traverse the same source-bound RAD hierarchy; this function changes
 * presentation detail only and never samples or rewrites source rows.
 */
export const resolveSplatLodBudget = (opts: {
  hardCeiling: number;
  isMobile: boolean;
}): SplatLodBudget => {
  const hardCeiling = safeElementCount(opts.hardCeiling);
  if (hardCeiling === 0) {
    return { settled: 0, interactive: 0, interactiveScale: 1 };
  }

  // PagedSplats rounds its allocation to whole pages. Floor the usable resident
  // capacity first so the normal case cannot cross the byte-derived hard ceiling.
  const residentCapacity =
    hardCeiling >= SPARK_PAGE_SPLATS
      ? Math.floor(hardCeiling / SPARK_PAGE_SPLATS) * SPARK_PAGE_SPLATS
      : hardCeiling;
  const residencyLimited = Math.max(
    1,
    Math.floor(residentCapacity / PAGED_SPLAT_POOL_HEADROOM)
  );
  const performanceLimit = opts.isMobile
    ? SETTLED_SPLAT_LIMIT_MOBILE
    : SETTLED_SPLAT_LIMIT_DESKTOP;
  const settled = Math.min(hardCeiling, residencyLimited, performanceLimit);
  const interactive = Math.max(1, Math.floor(settled * INTERACTIVE_LOD_SCALE));

  return {
    settled,
    interactive,
    interactiveScale: interactive / settled,
  };
};

/**
 * Page-aligned Spark cache capacity for a settled LoD target. Except for devices whose
 * entire declared ceiling is smaller than Spark's indivisible page, this stays inside
 * the caller's byte-derived hard ceiling.
 */
export const resolvePagedSplatPool = (settled: number, hardCeiling: number): number => {
  const safeSettled = safeElementCount(settled);
  const safeHardCeiling = safeElementCount(hardCeiling);
  if (safeSettled === 0 || safeHardCeiling === 0) {
    return 0;
  }

  const desired =
    Math.ceil((safeSettled * PAGED_SPLAT_POOL_HEADROOM) / SPARK_PAGE_SPLATS) *
    SPARK_PAGE_SPLATS;
  const hardPageCapacity =
    Math.floor(safeHardCeiling / SPARK_PAGE_SPLATS) * SPARK_PAGE_SPLATS;
  return Math.min(desired, Math.max(SPARK_PAGE_SPLATS, hardPageCapacity));
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

/** Honest readout for a reconstructed Gaussian LoD tree, whose active nodes are not a
 * literal subset of source rows and therefore must not be described as “showing N of M”. */
export const describeAdaptiveLod = (active: number, sourceTotal: number): string =>
  `adaptive LoD · ${count(active)} active · ${count(sourceTotal)} source`;
