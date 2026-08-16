/** Pure scene framing/depth arithmetic for the Scene3D viewer. */

export type SceneFrame = {
  centre: [number, number, number];
  radius: number;
};

export type SceneDepthPlan = {
  focus: SceneFrame;
  cameraPosition: [number, number, number];
  near: number;
  far: number;
  logarithmicDepthBuffer: boolean;
};

export type SceneViewGeometry = {
  verticalFovDegrees?: number;
  aspect?: number;
  padding?: number;
  /** Signed source-world up direction used only to place the initial camera. */
  up?: readonly [number, number, number];
};

const FALLBACK_BOUNDS = [-1, -1, -1, 1, 1, 1] as const;
const DEFAULT_UP = [0, 1, 0] as const;
// A stable source-world diagonal from which the horizontal part of the initial view is
// derived. Projecting it onto the plane perpendicular to `up` keeps the camera above
// either +Y, -Y, or +Z scenes without baking a transform into the geometry.
const INITIAL_HORIZONTAL_REFERENCE = [1, 1, 1] as const;
const DEFAULT_VERTICAL_FOV_DEGREES = 50;
const DEFAULT_FRAME_PADDING = 1.08;
const LOG_DEPTH_RATIO = 10_000;

const normalizedBounds = (raw: readonly number[]): [number, number, number, number, number, number] => {
  if (
    raw.length < 6 ||
    raw.slice(0, 6).some((value) => !Number.isFinite(value)) ||
    raw[3] < raw[0] ||
    raw[4] < raw[1] ||
    raw[5] < raw[2]
  ) {
    return [...FALLBACK_BOUNDS];
  }
  return [raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]];
};

export const frameSceneBounds = (raw: readonly number[]): SceneFrame => {
  const bounds = normalizedBounds(raw);
  const centre: [number, number, number] = [
    (bounds[0] + bounds[3]) / 2,
    (bounds[1] + bounds[4]) / 2,
    (bounds[2] + bounds[5]) / 2,
  ];
  const radius = Math.hypot(
    bounds[3] - bounds[0],
    bounds[4] - bounds[1],
    bounds[5] - bounds[2]
  ) / 2;
  return { centre, radius: Number.isFinite(radius) && radius > 0 ? radius : 1 };
};

export const maxDistanceToSceneBounds = (
  position: readonly [number, number, number],
  raw: readonly number[]
): number => {
  const bounds = normalizedBounds(raw);
  let farthest = 0;
  for (const x of [bounds[0], bounds[3]]) {
    for (const y of [bounds[1], bounds[4]]) {
      for (const z of [bounds[2], bounds[5]]) {
        farthest = Math.max(
          farthest,
          Math.hypot(x - position[0], y - position[1], z - position[2])
        );
      }
    }
  }
  return farthest;
};

/**
 * Distance from a perspective camera to the centre of a bounding sphere that keeps the
 * complete sphere inside both viewport dimensions.
 *
 * `radius / tan(fov / 2)` is a common box approximation, but it clips a sphere near the
 * corners. The tangent construction for a sphere is `radius / sin(fov / 2)`. The
 * smaller of the horizontal and vertical fields of view is the limiting dimension, so
 * portrait and split-view layouts remain honest without a magic distance multiplier.
 */
export const cameraDistanceToFrameSphere = (
  radius: number,
  view: SceneViewGeometry = {}
): number => {
  const safeRadius = Number.isFinite(radius) && radius > 0 ? radius : 1;
  const verticalFov = Math.max(
    1,
    Math.min(179, view.verticalFovDegrees ?? DEFAULT_VERTICAL_FOV_DEGREES)
  );
  const aspect =
    typeof view.aspect === "number" && Number.isFinite(view.aspect) && view.aspect > 0
      ? view.aspect
      : 1;
  const padding =
    typeof view.padding === "number" && Number.isFinite(view.padding) && view.padding >= 1
      ? view.padding
      : DEFAULT_FRAME_PADDING;
  const verticalHalfFov = (verticalFov * Math.PI) / 360;
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * aspect);
  const limitingHalfFov = Math.min(verticalHalfFov, horizontalHalfFov);
  return (safeRadius / Math.sin(limitingHalfFov)) * padding;
};

export const cameraPositionForSceneFrame = (
  frame: SceneFrame,
  view: SceneViewGeometry = {}
): [number, number, number] => {
  const rawUp = view.up ?? DEFAULT_UP;
  const upNorm = Math.hypot(...rawUp);
  const up: [number, number, number] =
    Number.isFinite(upNorm) && upNorm > Number.EPSILON
      ? [rawUp[0] / upNorm, rawUp[1] / upNorm, rawUp[2] / upNorm]
      : [...DEFAULT_UP];

  const referenceDot =
    INITIAL_HORIZONTAL_REFERENCE[0] * up[0] +
    INITIAL_HORIZONTAL_REFERENCE[1] * up[1] +
    INITIAL_HORIZONTAL_REFERENCE[2] * up[2];
  let horizontal: [number, number, number] = [
    INITIAL_HORIZONTAL_REFERENCE[0] - referenceDot * up[0],
    INITIAL_HORIZONTAL_REFERENCE[1] - referenceDot * up[1],
    INITIAL_HORIZONTAL_REFERENCE[2] - referenceDot * up[2],
  ];
  let horizontalNorm = Math.hypot(...horizontal);
  if (!(horizontalNorm > Number.EPSILON)) {
    // Only reachable for a non-axis-aligned future up vector parallel to [1,1,1].
    const fallbackReference: [number, number, number] = [1, 0, 0];
    const fallbackDot =
      fallbackReference[0] * up[0] +
      fallbackReference[1] * up[1] +
      fallbackReference[2] * up[2];
    horizontal = [
      fallbackReference[0] - fallbackDot * up[0],
      fallbackReference[1] - fallbackDot * up[1],
      fallbackReference[2] - fallbackDot * up[2],
    ];
    horizontalNorm = Math.hypot(...horizontal);
  }
  horizontal = horizontal.map((value) => value / horizontalNorm) as [number, number, number];

  // Equal normalized horizontal and vertical contributions form a calm 45-degree
  // three-quarter view. Most importantly, dot(direction, up) is positive: Reset view
  // always returns to the physically "above" side of the declared source frame.
  const direction = [
    horizontal[0] + up[0],
    horizontal[1] + up[1],
    horizontal[2] + up[2],
  ] as const;
  const norm = Math.hypot(...direction);
  const distance = cameraDistanceToFrameSphere(frame.radius, view);
  return [
    frame.centre[0] + (direction[0] / norm) * distance,
    frame.centre[1] + (direction[1] / norm) * distance,
    frame.centre[2] + (direction[2] / norm) * distance,
  ];
};

/**
 * Frame the middle-98% focus box, but keep the exact full bounds inside the depth range.
 * A logarithmic buffer is requested only when that honest range would exceed ordinary
 * depth precision. This keeps far-field reconstruction points discoverable without
 * shrinking the scientifically relevant scene to a speck.
 */
export const resolveSceneDepthPlan = (
  focusBounds: readonly number[],
  fullBounds: readonly number[],
  view: SceneViewGeometry = {}
): SceneDepthPlan => {
  const focus = frameSceneBounds(focusBounds);
  const cameraPosition = cameraPositionForSceneFrame(focus, view);
  const near = Math.max(focus.radius / 100, Number.EPSILON);
  const far = Math.max(
    focus.radius * 6,
    maxDistanceToSceneBounds(cameraPosition, fullBounds) * 1.05
  );
  return {
    focus,
    cameraPosition,
    near,
    far,
    logarithmicDepthBuffer: far / near >= LOG_DEPTH_RATIO,
  };
};
