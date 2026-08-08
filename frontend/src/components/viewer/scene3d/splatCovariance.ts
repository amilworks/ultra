/**
 * 3D Gaussian covariance from the stored splat parameters.
 *
 * A splat's shape is stored as a rotation quaternion plus three **log** scales; the
 * covariance the rasterizer wants is
 *
 *   Σ = R · diag(exp(s)²) · Rᵀ
 *
 * matching INRIA's `computeCov3D` (`cuda_rasterizer/forward.cu`), which builds
 * `M = S·R` and returns `MᵀM` — the same matrix, written the other way round.
 *
 * The `exp` is not optional. PLY stores raw model parameters (contract §3): the measured
 * median `scale_0` on the real file is −4.639, i.e. a true extent of 0.0097. Feeding the
 * log values in directly produces negative "scales" and a field of giant blurs.
 *
 * Pure math over plain arrays — no three.js, no Spark, no DOM.
 */

import { quatToMat3 } from "./sceneFrame";

/**
 * How many standard deviations of the Gaussian count as its footprint. INRIA's
 * rasterizer uses `3·sqrt(λ_max)` for the screen-space radius, so 3σ is the convention
 * this codebase matches — anything else silently changes culling and tile bounds.
 */
export const SPLAT_FOOTPRINT_SIGMAS = 3;

const readLogScale = (logScale: readonly number[]): [number, number, number] => {
  if (logScale.length < 3) {
    throw new Error(`log scale must have 3 components, got ${logScale.length}`);
  }
  return [logScale[0], logScale[1], logScale[2]];
};

/** `exp` of a log scale, with non-finite input collapsing to a degenerate 0 extent. */
const activate = (value: number): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  const scale = Math.exp(value);
  return Number.isFinite(scale) ? scale : 0;
};

/**
 * Covariance as the 6 upper-triangular entries in INRIA's order:
 * `[Σxx, Σxy, Σxz, Σyy, Σyz, Σzz]`.
 *
 * `quatXyzw` is normalized by `quatToMat3` first — INRIA's writer does not normalize
 * (contract Appendix A), and an unnormalized quaternion would fold an extra scale
 * factor into the covariance.
 */
export const covarianceFromScaleRot = (
  logScale: readonly number[],
  quatXyzw: readonly number[]
): number[] => {
  const [lx, ly, lz] = readLogScale(logScale);
  const sx = activate(lx);
  const sy = activate(ly);
  const sz = activate(lz);
  // Variance along each principal axis: the square of the activated scale.
  const vx = sx * sx;
  const vy = sy * sy;
  const vz = sz * sz;

  const r = quatToMat3(quatXyzw);
  // Σ_ij = Σ_k R_ik · v_k · R_jk, with R row-major.
  const a0 = r[0];
  const a1 = r[1];
  const a2 = r[2];
  const b0 = r[3];
  const b1 = r[4];
  const b2 = r[5];
  const c0 = r[6];
  const c1 = r[7];
  const c2 = r[8];

  return [
    a0 * a0 * vx + a1 * a1 * vy + a2 * a2 * vz, // xx
    a0 * b0 * vx + a1 * b1 * vy + a2 * b2 * vz, // xy
    a0 * c0 * vx + a1 * c1 * vy + a2 * c2 * vz, // xz
    b0 * b0 * vx + b1 * b1 * vy + b2 * b2 * vz, // yy
    b0 * c0 * vx + b1 * c1 * vy + b2 * c2 * vz, // yz
    c0 * c0 * vx + c1 * c1 * vy + c2 * c2 * vz, // zz
  ];
};

/**
 * World-space radius of a splat's footprint: `3σ` along its largest axis.
 *
 * Rotation-independent by construction — the largest axis of an ellipsoid is the same
 * length whichever way it points — which is what makes this usable as a cheap bounding
 * radius for chunk bboxes and culling without touching the quaternion.
 */
export const worldFootprintRadius = (logScale: readonly number[]): number => {
  const [lx, ly, lz] = readLogScale(logScale);
  return SPLAT_FOOTPRINT_SIGMAS * Math.max(activate(lx), activate(ly), activate(lz));
};
