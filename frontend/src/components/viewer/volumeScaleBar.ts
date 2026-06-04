export type VolumeScaleBar = {
  visible: boolean;
  length: number;
  label: string;
  fraction: number;
};

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

const niceLength = (target: number): number => {
  if (!Number.isFinite(target) || target <= 0) {
    return 0;
  }
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const normalized = target / magnitude;
  const step = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return step * magnitude;
};

export function resolveVolumeScaleBar({
  worldWidth,
  unit,
}: {
  worldWidth: number;
  unit?: string | null;
}): VolumeScaleBar {
  const safeWidth = Number(worldWidth);
  if (!Number.isFinite(safeWidth) || safeWidth <= 0) {
    return { visible: false, length: 0, label: "", fraction: 0 };
  }

  const length = niceLength(safeWidth / 3);
  if (length <= 0) {
    return { visible: false, length: 0, label: "", fraction: 0 };
  }

  const safeUnit = String(unit || "units").trim() || "units";
  return {
    visible: true,
    length,
    label: `${formatLength(length)} ${safeUnit}`,
    fraction: Math.max(0.08, Math.min(0.5, length / safeWidth)),
  };
}
