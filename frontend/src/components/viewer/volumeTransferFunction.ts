type ScalarVolumeTransferInput = {
  volume_signal_floor?: number | null;
  volume_density?: number | null;
};

export type ScalarVolumeTransferFunction = {
  signalFloor: number;
  densityScale: number;
  signalFloorLabel: string;
  densityLabel: string;
};

const clamp = (value: unknown, fallback: number, min: number, max: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, numeric));
};

export function resolveScalarVolumeTransferFunction(
  value?: ScalarVolumeTransferInput | null
): ScalarVolumeTransferFunction {
  const signalFloor = clamp(value?.volume_signal_floor, 0, 0, 0.95);
  const densityScale = clamp(value?.volume_density, 1, 0.1, 3);
  return {
    signalFloor,
    densityScale,
    signalFloorLabel: `${Math.round(signalFloor * 100)}%`,
    densityLabel: `${densityScale.toFixed(2)}x`,
  };
}
