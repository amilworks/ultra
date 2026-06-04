import type { UploadViewerInfo } from "@/types";

type AxisSizes = Pick<UploadViewerInfo["axis_sizes"], "X" | "Y" | "Z">;

export type VolumeSliceCursorAxis = {
  axis: "X" | "Y" | "Z";
  index: number;
  count: number;
  normalized: number;
  local: number;
  position: number;
  label: string;
  positionLabel: string;
};

export type VolumeSliceCursorCue = {
  visible: boolean;
  unit: string;
  summary: string;
  x: VolumeSliceCursorAxis;
  y: VolumeSliceCursorAxis;
  z: VolumeSliceCursorAxis;
};

const formatSliceCursorNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Number.isInteger(value)) {
    return value.toFixed(0);
  }
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
};

const positiveCount = (value: unknown): number => Math.max(1, Math.floor(Number(value) || 1));

const clampIndex = (value: unknown, count: number): number =>
  Math.max(0, Math.min(count - 1, Math.floor(Number(value) || 0)));

const axisCue = ({
  axis,
  index,
  count,
  length,
  unit,
}: {
  axis: VolumeSliceCursorAxis["axis"];
  index: unknown;
  count: unknown;
  length: unknown;
  unit: string;
}): VolumeSliceCursorAxis => {
  const safeCount = positiveCount(count);
  const safeIndex = clampIndex(index, safeCount);
  const normalized = safeCount > 1 ? (safeIndex + 0.5) / safeCount : 0.5;
  const safeLength = Math.max(0, Number(length) || 0);
  const position = normalized * safeLength;
  return {
    axis,
    index: safeIndex,
    count: safeCount,
    normalized,
    local: normalized - 0.5,
    position,
    label: `${axis} ${safeIndex + 1}/${safeCount}`,
    positionLabel: `${formatSliceCursorNumber(position)} ${unit}`,
  };
};

export function resolveVolumeSliceCursorCue({
  axisSizes,
  indices,
  worldWidth,
  worldHeight,
  worldDepth,
  unit,
}: {
  axisSizes: AxisSizes;
  indices: { x?: number; y?: number; z?: number };
  worldWidth: number;
  worldHeight: number;
  worldDepth: number;
  unit?: string | null;
}): VolumeSliceCursorCue {
  const safeUnit = String(unit || "units").trim() || "units";
  const x = axisCue({
    axis: "X",
    index: indices.x,
    count: axisSizes.X,
    length: worldWidth,
    unit: safeUnit,
  });
  const y = axisCue({
    axis: "Y",
    index: indices.y,
    count: axisSizes.Y,
    length: worldHeight,
    unit: safeUnit,
  });
  const z = axisCue({
    axis: "Z",
    index: indices.z,
    count: axisSizes.Z,
    length: worldDepth,
    unit: safeUnit,
  });
  const visible = x.count > 1 || y.count > 1 || z.count > 1;
  return {
    visible,
    unit: safeUnit,
    summary: visible ? `${x.label} / ${y.label} / ${z.label}` : "",
    x,
    y,
    z,
  };
}
