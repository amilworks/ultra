export type VolumeAxisCueAxis = {
  axis: "X" | "Y" | "Z";
  length: number;
  label: string;
};

export type VolumeAxisCue = {
  visible: boolean;
  unit: string;
  summary: string;
  x: VolumeAxisCueAxis;
  y: VolumeAxisCueAxis;
  z: VolumeAxisCueAxis;
};

const emptyAxis = (axis: VolumeAxisCueAxis["axis"]): VolumeAxisCueAxis => ({
  axis,
  length: 0,
  label: "",
});

const formatLength = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 100 || Number.isInteger(value)) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
};

const axisCue = (
  axis: VolumeAxisCueAxis["axis"],
  length: number,
  unit: string
): VolumeAxisCueAxis => ({
  axis,
  length,
  label: `${axis} ${formatLength(length)} ${unit}`,
});

export function resolveVolumeAxisCue({
  worldWidth,
  worldHeight,
  worldDepth,
  unit,
}: {
  worldWidth: number;
  worldHeight: number;
  worldDepth: number;
  unit?: string | null;
}): VolumeAxisCue {
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
      visible: false,
      unit: safeUnit,
      summary: "",
      x: emptyAxis("X"),
      y: emptyAxis("Y"),
      z: emptyAxis("Z"),
    };
  }

  const x = axisCue("X", width, safeUnit);
  const y = axisCue("Y", height, safeUnit);
  const z = axisCue("Z", depth, safeUnit);

  return {
    visible: true,
    unit: safeUnit,
    summary: `${x.label} / ${y.label} / ${z.label}`,
    x,
    y,
    z,
  };
}
