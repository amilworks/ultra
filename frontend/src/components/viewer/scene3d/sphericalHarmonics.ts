/**
 * Spherical harmonics for 3D Gaussian splats, transcribed from INRIA's reference
 * rasterizer (`diff-gaussian-rasterization`, `cuda_rasterizer/forward.cu`,
 * `computeColorFromSH`).
 *
 * The constants and per-term signs below are the reference's, **not** a textbook real-SH
 * basis. They differ: the reference folds the l=1 band's sign into the evaluation
 * (`−C1·y·sh[1] + C1·z·sh[2] − C1·x·sh[3]`) and orders the l=2/l=3 coefficients with
 * signs baked into the constant table. Substituting a "correct" generic basis gives a
 * scene whose view-dependent highlights sit on the wrong side of every surface — the
 * error is invisible in a still frame and obvious while orbiting. Trained coefficients
 * only mean anything under the basis they were trained in, so we reproduce that basis
 * exactly and document each number.
 *
 * `dir` is **camera → splat** (`pos − campos` in the reference), normalized. Reversing it
 * flips every odd band.
 *
 * Pure math over plain arrays — no three.js, no Spark, no DOM.
 */

/** `SH_C0` — the l=0 (DC) normalization, `1/(2·sqrt(pi))`. */
export const SH_C0 = 0.28209479177387814;

/** `SH_C1` — the shared l=1 magnitude, `sqrt(3/(4·pi))`. Signs live in `evaluateSh`. */
export const SH_C1 = 0.4886025119029199;

/**
 * `SH_C2[0..4]` — the l=2 band, signs included:
 *   [0]  1.0925484305920792   xy
 *   [1] -1.0925484305920792   yz
 *   [2]  0.31539156525252005  2z² − x² − y²
 *   [3] -1.0925484305920792   xz
 *   [4]  0.5462742152960396   x² − y²
 */
export const SH_C2: readonly number[] = [
  1.0925484305920792,
  -1.0925484305920792,
  0.31539156525252005,
  -1.0925484305920792,
  0.5462742152960396,
];

/**
 * `SH_C3[0..6]` — the l=3 band, signs included:
 *   [0] -0.5900435899266435   y(3x² − y²)
 *   [1]  2.890611442640554    xyz
 *   [2] -0.4570457994644658   y(4z² − x² − y²)
 *   [3]  0.3731763325901154   z(2z² − 3x² − 3y²)
 *   [4] -0.4570457994644658   x(4z² − x² − y²)
 *   [5]  1.445305721320277    z(x² − y²)
 *   [6] -0.5900435899266435   x(x² − 3y²)
 */
export const SH_C3: readonly number[] = [
  -0.5900435899266435,
  2.890611442640554,
  -0.4570457994644658,
  0.3731763325901154,
  -0.4570457994644658,
  1.445305721320277,
  -0.5900435899266435,
];

/** Highest SH degree the INRIA layout (and therefore this module) defines. */
export const MAX_SH_DEGREE = 3;

/**
 * View-independent base colour from the DC coefficients: `0.5 + C0·f_dc`.
 *
 * **Unclamped on purpose.** Measured on the real 14.5M-splat file, `0.5 + C0·f_dc_0`
 * spans −0.511 … 2.704 (contract Appendix A). Clamping here would hide how much of the
 * scene is out of gamut; clamping and *counting* is `sceneColor.clampWithCount`, and the
 * clamped fraction goes in the manifest.
 */
export const dcToBaseColour = (dc: readonly number[]): [number, number, number] => {
  if (dc.length < 3) {
    throw new Error(`f_dc must have 3 components, got ${dc.length}`);
  }
  return [0.5 + SH_C0 * dc[0], 0.5 + SH_C0 * dc[1], 0.5 + SH_C0 * dc[2]];
};

/**
 * Rest coefficients per colour channel for a degree: `(degree+1)² − 1`, i.e. the full
 * coefficient count minus the DC term. 0 → 0, 1 → 3, 2 → 8, 3 → 15.
 */
export const shBandCount = (degree: number): number => {
  if (!Number.isInteger(degree) || degree < 0 || degree > MAX_SH_DEGREE) {
    throw new Error(`SH degree must be an integer in 0..${MAX_SH_DEGREE}, got ${String(degree)}`);
  }
  return (degree + 1) * (degree + 1) - 1;
};

/**
 * Degree implied by the **total** `f_rest_*` property count in a PLY header (all three
 * channels): 0 → 0, 9 → 1, 24 → 2, 45 → 3.
 *
 * Anything else means the header was misread, and guessing a degree from it is precisely
 * the silent-misinterpretation failure this modality exists to avoid — so it throws.
 */
export const degreeFromRestCount = (n: number): number => {
  for (let degree = 0; degree <= MAX_SH_DEGREE; degree += 1) {
    if (n === shBandCount(degree) * 3) {
      return degree;
    }
  }
  throw new Error(
    `f_rest count ${String(n)} matches no SH degree (expected 0, 9, 24 or 45 across 3 channels)`
  );
};

/**
 * PLY planar order → interleaved RGB.
 *
 * INRIA's writer emits `f_rest_*` channel-major: all of R's rest coefficients, then all
 * of G's, then all of B's (`RRR…GGG…BBB`). The evaluator wants one RGB triple per
 * coefficient (`sh[i]` is a `vec3` in the reference), so coefficient `i` of channel `c`
 * moves from `planar[c·restPerChannel + i]` to `out[i·3 + c]`.
 *
 * Reading these two layouts as though they were the same produces a scene whose
 * view-dependent term is a colour-swapped scramble of the trained one.
 */
export const deplanarizeSh = (planar: Float32Array, restPerChannel: number): Float32Array => {
  if (!Number.isInteger(restPerChannel) || restPerChannel < 0) {
    throw new Error(`restPerChannel must be a non-negative integer, got ${String(restPerChannel)}`);
  }
  const needed = restPerChannel * 3;
  if (planar.length < needed) {
    throw new Error(`planar SH buffer holds ${planar.length} values, need ${needed}`);
  }
  const out = new Float32Array(needed);
  for (let channel = 0; channel < 3; channel += 1) {
    const base = channel * restPerChannel;
    for (let i = 0; i < restPerChannel; i += 1) {
      out[i * 3 + channel] = planar[base + i];
    }
  }
  return out;
};

/**
 * Evaluate the SH colour for one splat along one view direction.
 *
 * @param degree  measured degree, 0..3. Bands above it are never touched.
 * @param dc      `f_dc_0..2`.
 * @param rest    **interleaved** RGB rest coefficients — the output of `deplanarizeSh`.
 *                Coefficient `i` (1-based, matching the reference's `sh[i]`) lives at
 *                `rest[(i−1)*3 + channel]`.
 * @param dir     camera → splat. Normalized here, exactly as the reference does.
 * @returns       linear-ish colour, **unclamped**; the reference adds 0.5 and clamps at
 *                the call site while recording which components were clamped.
 */
export const evaluateSh = (
  degree: number,
  dc: readonly number[],
  rest: Float32Array,
  dir: readonly number[]
): [number, number, number] => {
  const bands = shBandCount(degree);
  if (rest.length < bands * 3) {
    throw new Error(`degree ${degree} needs ${bands * 3} interleaved rest values, got ${rest.length}`);
  }
  if (dc.length < 3) {
    throw new Error(`f_dc must have 3 components, got ${dc.length}`);
  }
  if (dir.length < 3) {
    throw new Error(`dir must have 3 components, got ${dir.length}`);
  }

  // result = SH_C0 * sh[0]
  let r = SH_C0 * dc[0];
  let g = SH_C0 * dc[1];
  let b = SH_C0 * dc[2];

  if (degree > 0) {
    const length = Math.hypot(dir[0], dir[1], dir[2]);
    // A zero-length direction has no band structure; fall back to the DC term rather
    // than propagating NaN through the whole splat.
    if (Number.isFinite(length) && length > 0) {
      const x = dir[0] / length;
      const y = dir[1] / length;
      const z = dir[2] / length;

      // result = result - C1*y*sh[1] + C1*z*sh[2] - C1*x*sh[3]
      const c1y = -SH_C1 * y;
      const c1z = SH_C1 * z;
      const c1x = -SH_C1 * x;
      r += c1y * rest[0] + c1z * rest[3] + c1x * rest[6];
      g += c1y * rest[1] + c1z * rest[4] + c1x * rest[7];
      b += c1y * rest[2] + c1z * rest[5] + c1x * rest[8];

      if (degree > 1) {
        const xx = x * x;
        const yy = y * y;
        const zz = z * z;
        const xy = x * y;
        const yz = y * z;
        const xz = x * z;

        const k4 = SH_C2[0] * xy;
        const k5 = SH_C2[1] * yz;
        const k6 = SH_C2[2] * (2 * zz - xx - yy);
        const k7 = SH_C2[3] * xz;
        const k8 = SH_C2[4] * (xx - yy);
        r += k4 * rest[9] + k5 * rest[12] + k6 * rest[15] + k7 * rest[18] + k8 * rest[21];
        g += k4 * rest[10] + k5 * rest[13] + k6 * rest[16] + k7 * rest[19] + k8 * rest[22];
        b += k4 * rest[11] + k5 * rest[14] + k6 * rest[17] + k7 * rest[20] + k8 * rest[23];

        if (degree > 2) {
          const k9 = SH_C3[0] * y * (3 * xx - yy);
          const k10 = SH_C3[1] * xy * z;
          const k11 = SH_C3[2] * y * (4 * zz - xx - yy);
          const k12 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
          const k13 = SH_C3[4] * x * (4 * zz - xx - yy);
          const k14 = SH_C3[5] * z * (xx - yy);
          const k15 = SH_C3[6] * x * (xx - 3 * yy);
          r +=
            k9 * rest[24] + k10 * rest[27] + k11 * rest[30] + k12 * rest[33] +
            k13 * rest[36] + k14 * rest[39] + k15 * rest[42];
          g +=
            k9 * rest[25] + k10 * rest[28] + k11 * rest[31] + k12 * rest[34] +
            k13 * rest[37] + k14 * rest[40] + k15 * rest[43];
          b +=
            k9 * rest[26] + k10 * rest[29] + k11 * rest[32] + k12 * rest[35] +
            k13 * rest[38] + k14 * rest[41] + k15 * rest[44];
        }
      }
    }
  }

  // result += 0.5 — the reference's final step, and the reason degree 0 is exactly
  // dcToBaseColour.
  return [r + 0.5, g + 0.5, b + 0.5];
};
