import type { UploadViewerInfo } from "@/types";

/**
 * Which axis a gallery thumbnail should scrub through, and how many planes it has.
 * Prefer Z (a focal/depth stack); fall back to T for a Z-less time-series so movie
 * thumbnails advance through time instead of a single clamped plane.
 */
export function thumbnailScrubAxis(
  axisSizes: Pick<UploadViewerInfo["axis_sizes"], "Z" | "T"> | null | undefined
): { axis: "z" | "t"; count: number } {
  const z = Math.max(1, Math.floor(Number(axisSizes?.Z ?? 1)) || 1);
  const t = Math.max(1, Math.floor(Number(axisSizes?.T ?? 1)) || 1);
  if (z > 1) {
    return { axis: "z", count: z };
  }
  if (t > 1) {
    return { axis: "t", count: t };
  }
  return { axis: "z", count: 1 };
}
