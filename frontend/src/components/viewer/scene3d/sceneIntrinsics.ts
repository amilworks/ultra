/**
 * COLMAP camera intrinsics → OpenGL projection, for the `scene3d` Lens modality.
 *
 * COLMAP's eleven camera models all pack their parameters into one flat `params` array
 * whose meaning is positional and model-dependent. The first three or four entries are
 * always the pinhole core (`f` or `fx, fy`, then `cx, cy`); everything after is
 * distortion. This module reads the core, reports whether distortion is present, and
 * builds the projection matrix.
 *
 * The projection matrix is built as an **asymmetric** frustum. A principal point that is
 * not at the image centre — which is the normal case for a real calibration, and the
 * usual case for a cropped or undistorted render — shifts the frustum sideways. A
 * symmetric `PerspectiveCamera(fov, aspect)` is structurally incapable of representing
 * that (contract §9): it would silently re-centre every frustum, drawing camera pyramids
 * that point in subtly wrong directions and misaligning any reprojection.
 *
 * Pure math over plain arrays — no three.js, no Spark, no DOM.
 */

export type ColmapCamera = {
  model: string;
  width: number;
  height: number;
  params: readonly number[];
};

export type ColmapFocal = {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
};

/**
 * `params` layout per COLMAP `src/colmap/sensor/models.h`.
 *
 * `sharedFocal` means the model stores a single `f` used for both axes, so the layout is
 * `f, cx, cy, ...` rather than `fx, fy, cx, cy, ...`. `distortion` is the number of
 * trailing distortion parameters; `size` is the total, and is checked on every read
 * because a params array of the wrong length means the camera record was misparsed and
 * every number after it is garbage.
 */
type ColmapModelSpec = {
  sharedFocal: boolean;
  distortion: number;
  size: number;
};

const spec = (sharedFocal: boolean, distortion: number): ColmapModelSpec => ({
  sharedFocal,
  distortion,
  size: (sharedFocal ? 3 : 4) + distortion,
});

const COLMAP_MODELS: Record<string, ColmapModelSpec> = {
  // f, cx, cy
  SIMPLE_PINHOLE: spec(true, 0),
  // fx, fy, cx, cy
  PINHOLE: spec(false, 0),
  // f, cx, cy, k
  SIMPLE_RADIAL: spec(true, 1),
  // f, cx, cy, k1, k2
  RADIAL: spec(true, 2),
  // fx, fy, cx, cy, k1, k2, p1, p2
  OPENCV: spec(false, 4),
  // fx, fy, cx, cy, k1, k2, k3, k4
  OPENCV_FISHEYE: spec(false, 4),
  // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
  FULL_OPENCV: spec(false, 8),
  // fx, fy, cx, cy, omega
  FOV: spec(false, 1),
  // f, cx, cy, k
  SIMPLE_RADIAL_FISHEYE: spec(true, 1),
  // f, cx, cy, k1, k2
  RADIAL_FISHEYE: spec(true, 2),
  // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
  THIN_PRISM_FISHEYE: spec(false, 8),
};

/** The models this module knows, for error messages and callers that want to gate UI. */
export const SUPPORTED_COLMAP_MODELS: readonly string[] = Object.keys(COLMAP_MODELS);

const specFor = (camera: ColmapCamera): ColmapModelSpec => {
  const found = COLMAP_MODELS[camera.model];
  if (!found) {
    throw new Error(
      `unsupported COLMAP camera model ${JSON.stringify(camera.model)}; supported: ${SUPPORTED_COLMAP_MODELS.join(", ")}`
    );
  }
  if (camera.params.length !== found.size) {
    throw new Error(
      `COLMAP camera model ${camera.model} expects ${found.size} params, got ${camera.params.length}`
    );
  }
  return found;
};

const finiteOrThrow = (value: number, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite, got ${String(value)}`);
  }
  return value;
};

/**
 * Pinhole core of any supported model. `SIMPLE_*` models share one focal length across
 * both axes; the rest carry `fx` and `fy` separately and they are genuinely unequal for
 * anamorphic or non-square-pixel sensors, so they are never collapsed to one number.
 */
export const focalOf = (c: ColmapCamera): ColmapFocal => {
  const model = specFor(c);
  const p = c.params;
  if (model.sharedFocal) {
    const f = finiteOrThrow(p[0], `${c.model} f`);
    return { fx: f, fy: f, cx: finiteOrThrow(p[1], "cx"), cy: finiteOrThrow(p[2], "cy") };
  }
  return {
    fx: finiteOrThrow(p[0], "fx"),
    fy: finiteOrThrow(p[1], "fy"),
    cx: finiteOrThrow(p[2], "cx"),
    cy: finiteOrThrow(p[3], "cy"),
  };
};

/**
 * Whether this camera carries **effective** distortion — the model declares distortion
 * slots *and* at least one of them is non-zero.
 *
 * The model name alone is not the answer: a `SIMPLE_RADIAL` with `k = 0` is exactly a
 * `SIMPLE_PINHOLE`, and COLMAP emits that constantly for already-undistorted images.
 * Reporting distortion there would put a false caveat in the provenance panel; the
 * caveat only earns its place when the numbers say the pinhole projection is an
 * approximation.
 */
export const hasDistortion = (c: ColmapCamera): boolean => {
  const model = specFor(c);
  if (model.distortion === 0) {
    return false;
  }
  const first = model.sharedFocal ? 3 : 4;
  for (let i = first; i < c.params.length; i += 1) {
    const value = c.params[i];
    if (Number.isFinite(value) && value !== 0) {
      return true;
    }
  }
  return false;
};

const positiveDimension = (value: number, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive number, got ${String(value)}`);
  }
  return value;
};

/**
 * Column-major 4x4 OpenGL projection matrix (the layout three.js `Matrix4.elements`
 * uses), built from the intrinsics as an asymmetric frustum.
 *
 * Frustum edges at the near plane, in the RUB camera frame (x right, y up, looking down
 * −z). A pixel column `u` sits at camera x = `(u − cx) · near / fx`; a pixel row `v`
 * sits at camera y = `−(v − cy) · near / fy`, the sign flip being COLMAP's y-down image
 * convention meeting OpenGL's y-up camera:
 *
 *   left   = −cx · near / fx           right = (width  − cx) · near / fx
 *   top    =  cy · near / fy           bottom = −(height − cy) · near / fy
 *
 * With `cx = width/2` and `cy = height/2` this collapses to the familiar symmetric
 * frustum; with any other principal point it does not, and that asymmetry is the whole
 * point of building the matrix by hand.
 *
 * `far = Infinity` is supported and yields the standard infinite-far limit.
 */
export const projectionMatrixFor = (c: ColmapCamera, near: number, far: number): number[] => {
  const { fx, fy, cx, cy } = focalOf(c);
  const width = positiveDimension(c.width, "camera width");
  const height = positiveDimension(c.height, "camera height");
  positiveDimension(fx, "fx");
  positiveDimension(fy, "fy");
  positiveDimension(near, "near");
  if (typeof far !== "number" || Number.isNaN(far) || far <= near) {
    throw new Error(`far must be greater than near, got near=${String(near)} far=${String(far)}`);
  }

  const left = (-cx * near) / fx;
  const right = ((width - cx) * near) / fx;
  const top = (cy * near) / fy;
  const bottom = (-(height - cy) * near) / fy;

  const rl = right - left;
  const tb = top - bottom;
  if (rl === 0 || tb === 0) {
    throw new Error("degenerate frustum: principal point places an edge on the optical axis");
  }

  const m = new Array<number>(16).fill(0);
  m[0] = (2 * near) / rl;
  m[5] = (2 * near) / tb;
  m[8] = (right + left) / rl; // x shear — non-zero exactly when cx is off-centre
  m[9] = (top + bottom) / tb; // y shear — non-zero exactly when cy is off-centre
  m[11] = -1;
  if (Number.isFinite(far)) {
    const fn = far - near;
    m[10] = -(far + near) / fn;
    m[14] = (-2 * far * near) / fn;
  } else {
    // lim far->inf of the two entries above.
    m[10] = -1;
    m[14] = -2 * near;
  }
  return m;
};

/**
 * Total vertical angular extent of the frustum, in degrees — a readout and fallback
 * value only. The renderer projects with `projectionMatrixFor`; a single FOV number
 * cannot carry the principal-point offset (contract §9).
 *
 * Computed as the true extent `atan(cy/fy) + atan((height−cy)/fy)` rather than the
 * symmetric `2·atan(height / 2fy)`, so it degrades gracefully instead of lying: the two
 * agree exactly when `cy = height/2` and diverge as the principal point moves.
 */
export const verticalFovDeg = (c: ColmapCamera): number => {
  const { fy, cy } = focalOf(c);
  const height = positiveDimension(c.height, "camera height");
  positiveDimension(fy, "fy");
  const above = Math.atan(cy / fy);
  const below = Math.atan((height - cy) / fy);
  return ((above + below) * 180) / Math.PI;
};
