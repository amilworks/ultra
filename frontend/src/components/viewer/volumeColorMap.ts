export type ScalarVolumeColorMapId = "grayscale" | "viridis" | "magma" | "inferno";

export type ScalarVolumeColorMap = {
  id: ScalarVolumeColorMapId;
  label: string;
  shaderValue: number;
};

export const SCALAR_VOLUME_COLOR_MAPS: ScalarVolumeColorMap[] = [
  { id: "grayscale", label: "Grayscale", shaderValue: 0 },
  { id: "viridis", label: "Viridis", shaderValue: 1 },
  { id: "magma", label: "Magma", shaderValue: 2 },
  { id: "inferno", label: "Inferno", shaderValue: 3 },
];

export function resolveScalarVolumeColorMap(value?: unknown): ScalarVolumeColorMap {
  const normalized = String(value ?? "").trim().toLowerCase();
  return (
    SCALAR_VOLUME_COLOR_MAPS.find((colorMap) => colorMap.id === normalized) ??
    SCALAR_VOLUME_COLOR_MAPS[0]
  );
}
