import type { UploadViewerInfo } from "@/types";

type ViewerOrientation = NonNullable<UploadViewerInfo["viewer"]["orientation"]>;

export type VolumeOrientationAxisCue = {
  negative: string;
  positive: string;
  label: string;
};

export type VolumeOrientationCue = {
  frame: string;
  x: VolumeOrientationAxisCue;
  y: VolumeOrientationAxisCue;
  z: VolumeOrientationAxisCue;
};

const fallbackAxisLabels = {
  x: { positive: "X", negative: "-X" },
  y: { positive: "Y", negative: "-Y" },
  z: { positive: "Z", negative: "-Z" },
};

const cleanLabel = (value: unknown, fallback: string): string => {
  const label = String(value ?? "").trim();
  return label.length > 0 ? label : fallback;
};

const axisCue = (
  axis: "x" | "y" | "z",
  axisLabels: ViewerOrientation["axis_labels"] | undefined
): VolumeOrientationAxisCue => {
  const fallback = fallbackAxisLabels[axis];
  const source = axisLabels?.[axis] ?? fallback;
  const negative = cleanLabel(source.negative, fallback.negative);
  const positive = cleanLabel(source.positive, fallback.positive);
  return {
    negative,
    positive,
    label: `${axis.toUpperCase()} ${negative}->${positive}`,
  };
};

export function resolveVolumeOrientationCue(
  orientation?: ViewerOrientation | null
): VolumeOrientationCue {
  const frame = cleanLabel(orientation?.frame, "voxel").toLowerCase();
  const axisLabels = orientation?.axis_labels;
  return {
    frame,
    x: axisCue("x", axisLabels),
    y: axisCue("y", axisLabels),
    z: axisCue("z", axisLabels),
  };
}
