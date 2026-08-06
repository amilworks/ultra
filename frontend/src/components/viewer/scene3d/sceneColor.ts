/**
 * sRGB ↔ linear conversion and clamp accounting for the `scene3d` Lens modality.
 *
 * Two colour paths meet here and they are not the same (contract §4):
 *
 *   - splats (`USX1`) carry **linear** colour, converted once in the derive;
 *   - points (`UPC1`) carry **sRGB** bytes straight from source photographs, and three.js
 *     assumes a vertex-colour attribute is already in the linear working space. Handing
 *     it sRGB double-encodes: measured, sRGB 0.2 renders at ≈0.48.
 *
 * The transfer function is the **exact IEC 61966-2-1 piecewise curve**, not a 2.2-power
 * approximation. The two differ by up to ~0.01 in the mid-tones and by a factor of ~2 in
 * the deep shadows, which is exactly where a point cloud's shaded crevices live.
 *
 * Pure math over plain arrays — no three.js, no Spark, no DOM.
 */

/** Piecewise knee of the sRGB encoding curve. Below it the curve is a pure 1/12.92 line. */
export const SRGB_LINEAR_KNEE = 0.04045;

/**
 * Piecewise knee of the inverse (linear → sRGB) curve.
 *
 * The standard rounds its two knee constants independently, and they do not quite agree:
 * `0.04045 / 12.92 = 0.0031308049…`, which sits just *above* `0.0031308`. So a value
 * exactly at the sRGB knee round-trips through the inverse's power branch and comes back
 * ~3e-8 low. That is the published standard's inconsistency, not this module's — we
 * reproduce the standard rather than "fixing" it, because a corrected knee would no
 * longer match what every other renderer in the pipeline does.
 */
export const LINEAR_SRGB_KNEE = 0.0031308;

/**
 * sRGB → linear, IEC 61966-2-1:
 *
 *   c ≤ 0.04045 → c / 12.92
 *   else        → ((c + 0.055) / 1.055)^2.4
 *
 * Extended below 0 by odd symmetry (`f(−c) = −f(c)`). That branch is not decoration:
 * `0.5 + C0·f_dc` is unclamped and measured down to −0.511 on the real file, and the
 * naive formula would raise a negative base to 2.4 and return NaN.
 */
export const srgbToLinear = (c: number): number => {
  if (!Number.isFinite(c)) {
    return 0;
  }
  if (c < 0) {
    return -srgbToLinear(-c);
  }
  if (c <= SRGB_LINEAR_KNEE) {
    return c / 12.92;
  }
  return ((c + 0.055) / 1.055) ** 2.4;
};

/**
 * linear → sRGB, the exact inverse of `srgbToLinear`:
 *
 *   c ≤ 0.0031308 → c · 12.92
 *   else          → 1.055 · c^(1/2.4) − 0.055
 */
export const linearToSrgb = (c: number): number => {
  if (!Number.isFinite(c)) {
    return 0;
  }
  if (c < 0) {
    return -linearToSrgb(-c);
  }
  if (c <= LINEAR_SRGB_KNEE) {
    return c * 12.92;
  }
  return 1.055 * c ** (1 / 2.4) - 0.055;
};

// A byte can only take 256 values, so the transfer function is a table lookup rather
// than a `pow` per component — exact, and it keeps a multi-million-point conversion off
// the main thread's critical path.
const BYTE_TO_LINEAR = (() => {
  const table = new Float32Array(256);
  for (let i = 0; i < 256; i += 1) {
    table[i] = srgbToLinear(i / 255);
  }
  return table;
})();

/**
 * Convert sRGB-encoded bytes to linear floats, componentwise.
 *
 * **Componentwise means componentwise.** Alpha is not sRGB-encoded, so a caller holding
 * interleaved RGBA (as `UPC1` point colour is) must pass the colour components only, or
 * restore alpha afterwards. This signature carries no stride and cannot do it for you.
 *
 * `out` is filled in place when supplied — the point path reuses one buffer across
 * chunks — and must be at least as long as `src`.
 */
export const srgbBytesToLinearFloat = (src: Uint8Array, out?: Float32Array): Float32Array => {
  const target = out ?? new Float32Array(src.length);
  if (target.length < src.length) {
    throw new Error(`output buffer holds ${target.length} values, need ${src.length}`);
  }
  for (let i = 0; i < src.length; i += 1) {
    target[i] = BYTE_TO_LINEAR[src[i]];
  }
  return target;
};

/**
 * Clamp to [0, 1] and count how many components needed it.
 *
 * The count is the point: `0.5 + C0·f_dc` runs out of gamut on real files, and the
 * manifest reports `clamped_color_fraction` so the provenance panel can say how much of
 * the scene's colour was altered rather than pretending the clamp was free.
 *
 * A non-finite component is not in [0, 1] either — it becomes 0 and counts as clamped.
 */
export const clampWithCount = (v: readonly number[]): { values: number[]; clamped: number } => {
  const values = new Array<number>(v.length);
  let clamped = 0;
  for (let i = 0; i < v.length; i += 1) {
    const value = v[i];
    if (!Number.isFinite(value)) {
      values[i] = 0;
      clamped += 1;
    } else if (value < 0) {
      values[i] = 0;
      clamped += 1;
    } else if (value > 1) {
      values[i] = 1;
      clamped += 1;
    } else {
      values[i] = value;
    }
  }
  return { values, clamped };
};
