type ClipAxis = "X" | "Y" | "Z";

type ClipBounds = {
  x: number;
  y: number;
  z: number;
};

export type VolumeClipCueAxis = {
  axis: ClipAxis;
  start: number;
  end: number;
  length: number;
  label: string;
};

export type VolumeClipCue = {
  active: boolean;
  unit: string;
  summary: string;
  x: VolumeClipCueAxis;
  y: VolumeClipCueAxis;
  z: VolumeClipCueAxis;
};

const emptyAxis = (axis: ClipAxis): VolumeClipCueAxis => ({
  axis,
  start: 0,
  end: 0,
  length: 0,
  label: "",
});

const clampUnit = (value: number, fallback: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(0, Math.min(1, numeric));
};

const formatLength = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 100 || Number.isInteger(value)) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  return value.toFixed(2);
};

const formatRangeValue = (value: number, clippedAxis: boolean): string =>
  clippedAxis ? value.toFixed(2) : formatLength(value);

const roundPhysicalValue = (value: number): number => Number(value.toFixed(4));

const axisCue = (
  axis: ClipAxis,
  extent: number,
  minFraction: number,
  maxFraction: number,
  unit: string
): VolumeClipCueAxis => {
  const start = roundPhysicalValue(extent * minFraction);
  const end = roundPhysicalValue(extent * maxFraction);
  const clippedAxis = Math.abs(minFraction) > 0.0001 || Math.abs(1 - maxFraction) > 0.0001;
  return {
    axis,
    start,
    end,
    length: roundPhysicalValue(Math.max(0, end - start)),
    label: `${axis} ${formatRangeValue(start, clippedAxis)}-${formatRangeValue(end, clippedAxis)} ${unit}`,
  };
};

export function resolveVolumeClipCue({
  worldWidth,
  worldHeight,
  worldDepth,
  clipMin,
  clipMax,
  unit,
}: {
  worldWidth: number;
  worldHeight: number;
  worldDepth: number;
  clipMin?: ClipBounds | null;
  clipMax?: ClipBounds | null;
  unit?: string | null;
}): VolumeClipCue {
  const safeUnit = String(unit || "units").trim() || "units";
  const width = Number(worldWidth);
  const height = Number(worldHeight);
  const depth = Number(worldDepth);

  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    !Number.isFinite(depth) ||
    width <= 0 ||
    height <= 0 ||
    depth <= 0
  ) {
    return {
      active: false,
      unit: safeUnit,
      summary: "",
      x: emptyAxis("X"),
      y: emptyAxis("Y"),
      z: emptyAxis("Z"),
    };
  }

  const min = {
    x: clampUnit(clipMin?.x ?? 0, 0),
    y: clampUnit(clipMin?.y ?? 0, 0),
    z: clampUnit(clipMin?.z ?? 0, 0),
  };
  const max = {
    x: clampUnit(clipMax?.x ?? 1, 1),
    y: clampUnit(clipMax?.y ?? 1, 1),
    z: clampUnit(clipMax?.z ?? 1, 1),
  };

  (["x", "y", "z"] as const).forEach((axis) => {
    if (max[axis] < min[axis]) {
      const previous = min[axis];
      min[axis] = max[axis];
      max[axis] = previous;
    }
  });

  const active =
    Math.abs(min.x) > 0.0001 ||
    Math.abs(min.y) > 0.0001 ||
    Math.abs(min.z) > 0.0001 ||
    Math.abs(1 - max.x) > 0.0001 ||
    Math.abs(1 - max.y) > 0.0001 ||
    Math.abs(1 - max.z) > 0.0001;

  const x = axisCue("X", width, min.x, max.x, safeUnit);
  const y = axisCue("Y", height, min.y, max.y, safeUnit);
  const z = axisCue("Z", depth, min.z, max.z, safeUnit);

  return {
    active,
    unit: safeUnit,
    summary: active ? `Clip ${x.label} / ${y.label} / ${z.label}` : "",
    x,
    y,
    z,
  };
}
