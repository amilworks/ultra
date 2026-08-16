/**
 * COLMAP pose algebra for the `scene3d` Lens modality.
 *
 * COLMAP stores a **world-to-camera** pose: `x_cam = R * x_world + t`, with `R` given
 * as a quaternion in **wxyz** order (`QW QX QY QZ` in `images.txt`, the same field
 * order in `images.bin`). Everything a renderer wants is the inverse of that:
 *
 *   camera centre in world  C = -Rᵀ t          <- NOT t, and not -t
 *   camera orientation      Rᵀ · diag(1,−1,−1) <- RDF axes flipped to RUB
 *
 * Getting `C` wrong is the classic COLMAP bug: `t` alone is the world origin expressed
 * in camera coordinates, which only coincides with the camera centre when `R` is the
 * identity. Frusta drawn at `t` look plausible on a turntable and are wrong everywhere.
 *
 * The RDF→RUB flip is applied **here and nowhere else** (contract §2). COLMAP cameras
 * look down +Z with +X right and +Y down (RDF); OpenGL/three.js cameras look down −Z
 * with +Y up (RUB). Flipping the *scene* instead would rotate splat geometry without
 * rotating its spherical-harmonic coefficients, sign-flipping every odd SH band.
 *
 * Pure math over plain arrays — no three.js, no Spark, no DOM.
 */

/** Row-major 3x3. `m[row * 3 + col]`, so `m[1]` is row 0 column 1. */
export type Mat3 = readonly [number, number, number, number, number, number, number, number, number];

/** Identity, exported for callers that need a neutral basis. */
export const IDENTITY_MAT3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * RDF (COLMAP: x right, y down, z forward) → RUB (OpenGL: x right, y up, z backward).
 *
 * `diag(1, −1, −1)`. It is an involution: applying it twice returns to RDF, which is
 * exactly why "applying the flip twice is as broken as never applying it" (contract §3).
 */
export const RDF_TO_RUB: Mat3 = [1, 0, 0, 0, -1, 0, 0, 0, -1];

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

/** Read a 3-vector defensively; a malformed pose must fail loudly, not render askew. */
const readVec3 = (v: readonly number[], label: string): [number, number, number] => {
  if (v.length < 3 || !isFiniteNumber(v[0]) || !isFiniteNumber(v[1]) || !isFiniteNumber(v[2])) {
    throw new Error(`${label} must be 3 finite numbers, got ${JSON.stringify(v)}`);
  }
  return [v[0], v[1], v[2]];
};

const readQuat = (q: readonly number[], label: string): [number, number, number, number] => {
  if (
    q.length < 4 ||
    !isFiniteNumber(q[0]) ||
    !isFiniteNumber(q[1]) ||
    !isFiniteNumber(q[2]) ||
    !isFiniteNumber(q[3])
  ) {
    throw new Error(`${label} must be 4 finite numbers, got ${JSON.stringify(q)}`);
  }
  return [q[0], q[1], q[2], q[3]];
};

/**
 * COLMAP quaternion order (w, x, y, z) → the (x, y, z, w) order three.js, glTF and
 * Spark all use. Pure reordering; no normalization, no sign convention change.
 */
export const quatWxyzToXyzw = (q: readonly number[]): [number, number, number, number] => {
  const [w, x, y, z] = readQuat(q, "quaternion (wxyz)");
  return [x, y, z, w];
};

/**
 * Scale a quaternion to unit length. Order-agnostic — it only divides by the norm — so
 * it may be applied to either ordering, but note the degenerate fallback below.
 *
 * Postshot writes normalized quaternions; INRIA's writer does not (contract Appendix A),
 * so every quaternion entering the renderer is normalized defensively.
 *
 * A zero or non-finite quaternion carries no rotation at all. There is no order-agnostic
 * identity, so the fallback is the **xyzw** identity `[0, 0, 0, 1]`: normalize *after*
 * converting to xyzw, which is what every call site in this module does.
 */
export const normalizeQuat = (q: readonly number[]): [number, number, number, number] => {
  const [a, b, c, d] = readQuat(q, "quaternion");
  const norm = Math.hypot(a, b, c, d);
  if (!Number.isFinite(norm) || norm <= 0) {
    return [0, 0, 0, 1];
  }
  return [a / norm, b / norm, c / norm, d / norm];
};

/**
 * Unit quaternion (xyzw) → row-major rotation matrix. The input is normalized first,
 * so an unnormalized INRIA quaternion produces a true rotation rather than a matrix
 * that also scales.
 */
export const quatToMat3 = (xyzw: readonly number[]): Mat3 => {
  const [x, y, z, w] = normalizeQuat(xyzw);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
  ];
};

/** Row-major matrix times column vector. */
export const applyMat3 = (m: Mat3, v: readonly number[]): [number, number, number] => {
  const [x, y, z] = readVec3(v, "vector");
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
};

/** Row-major 3x3 product. */
export const multiplyMat3 = (a: Mat3, b: Mat3): Mat3 => {
  const out = new Array<number>(9);
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      out[row * 3 + col] =
        a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
    }
  }
  return out as unknown as Mat3;
};

/** Transpose of a row-major 3x3. For a rotation this is also its inverse. */
export const transposeMat3 = (m: Mat3): Mat3 => [
  m[0], m[3], m[6],
  m[1], m[4], m[7],
  m[2], m[5], m[8],
];

/**
 * Camera centre in world coordinates: `C = -Rᵀ t`.
 *
 * `t` is the world origin expressed in the camera frame. Using `t` or `-t` directly is
 * only correct when `R` is the identity; for any real pose it places the frustum
 * somewhere else entirely.
 */
export const cameraCentreFromColmap = (
  qvecWxyz: readonly number[],
  tvec: readonly number[]
): [number, number, number] => {
  const r = quatToMat3(quatWxyzToXyzw(qvecWxyz));
  const t = readVec3(tvec, "tvec");
  // Rᵀ t, then negate. Rᵀ's row i is R's column i.
  return [
    -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
    -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
    -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
  ];
};

/**
 * Camera orientation basis in world coordinates: `Rᵀ · diag(1, −1, −1)`.
 *
 * Columns are the camera's right / up / backward axes in world space, ready to drop
 * into a three.js basis. `Rᵀ` undoes COLMAP's world-to-camera rotation; the right
 * factor is the single RDF→RUB flip permitted by contract §2.
 *
 * `tvec` does not affect orientation. It is accepted so both COLMAP pose helpers share
 * one call shape, and it is validated here so a malformed pose fails at whichever
 * helper the caller reaches first.
 */
export const cameraBasisFromColmap = (
  qvecWxyz: readonly number[],
  tvec: readonly number[]
): Mat3 => {
  readVec3(tvec, "tvec");
  const r = quatToMat3(quatWxyzToXyzw(qvecWxyz));
  return multiplyMat3(transposeMat3(r), RDF_TO_RUB);
};
