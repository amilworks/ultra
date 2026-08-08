import type { Scene3dManifest } from "@/types";

export type SceneUpVector = readonly [number, number, number];

const DEFAULT_UP: SceneUpVector = [0, 1, 0];

/**
 * Resolve the signed world-up direction used by the camera without rotating source
 * geometry.
 *
 * `world.up_axis` deliberately names only the dominant axis. Gaussian-splat PLY and
 * COLMAP use the common right/down/forward convention, so a heuristic Y-axis hint means
 * physical up is -Y. Z-up PLYs retain +Z, authoritative source/user hints win, and
 * formats outside that family retain Three's +Y convention.
 */
export const resolveSceneUpVector = (manifest: Scene3dManifest): SceneUpVector => {
  const axis = manifest.world.up_axis.trim().toLowerCase();
  if (axis === "z") {
    return [0, 0, 1];
  }
  if (axis !== "y") {
    return DEFAULT_UP;
  }

  const format = manifest.source.format.trim().toLowerCase();
  const basis = manifest.world.up_axis_basis.trim().toLowerCase();
  // A declared or user-calibrated +Y axis is authoritative. Only unsigned heuristic
  // hints need the legacy file-family convention to resolve their sign.
  const hasAuthoritativePositiveAxis = basis === "declared" || basis === "user";
  const usesRdfConvention =
    !hasAuthoritativePositiveAxis &&
    (format === "ply" ||
      format === "colmap" ||
      manifest.scene_kind.trim().toLowerCase() === "colmap" ||
      manifest.layers.some((layer) => layer.source_frame.trim().toLowerCase() === "rdf"));
  return usesRdfConvention ? [0, -1, 0] : DEFAULT_UP;
};

export const describeSceneUpDirection = (manifest: Scene3dManifest): string => {
  const axis = manifest.world.up_axis.trim().toLowerCase();
  if (axis !== "y" && axis !== "z") {
    return "unknown";
  }
  const [x, y, z] = resolveSceneUpVector(manifest);
  if (Math.abs(x) > 0.5) {
    return `${x < 0 ? "−" : "+"}X`;
  }
  if (Math.abs(y) > 0.5) {
    return `${y < 0 ? "−" : "+"}Y`;
  }
  return `${z < 0 ? "−" : "+"}Z`;
};
