type ScalarVolumeLightingInput = {
  volume_lighting?: boolean | null;
  volume_lighting_strength?: number | null;
};

export type ScalarVolumeLighting = {
  enabled: boolean;
  strength: number;
  strengthLabel: string;
};

const clamp = (value: unknown, fallback: number, min: number, max: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, numeric));
};

export function resolveScalarVolumeLighting(
  value?: ScalarVolumeLightingInput | null
): ScalarVolumeLighting {
  const strength = clamp(value?.volume_lighting_strength, 0.65, 0, 1);
  return {
    enabled: Boolean(value?.volume_lighting ?? false),
    strength,
    strengthLabel: `${Math.round(strength * 100)}%`,
  };
}
