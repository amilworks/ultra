export type VolumeViewPresetId = "iso" | "xy" | "xz" | "yz";

export type VolumeViewPreset = {
  id: VolumeViewPresetId;
  label: string;
  direction: { x: number; y: number; z: number };
  up: { x: number; y: number; z: number };
};

export const VOLUME_VIEW_PRESETS: VolumeViewPreset[] = [
  {
    id: "iso",
    label: "3D",
    direction: { x: 1.8, y: 1.4, z: 1.8 },
    up: { x: 0, y: 1, z: 0 },
  },
  {
    id: "xy",
    label: "XY",
    direction: { x: 0, y: 0, z: 1 },
    up: { x: 0, y: 1, z: 0 },
  },
  {
    id: "xz",
    label: "XZ",
    direction: { x: 0, y: 1, z: 0 },
    up: { x: 0, y: 0, z: 1 },
  },
  {
    id: "yz",
    label: "YZ",
    direction: { x: 1, y: 0, z: 0 },
    up: { x: 0, y: 0, z: 1 },
  },
];

export function resolveVolumeViewPreset(value?: unknown): VolumeViewPreset {
  const normalized = String(value ?? "").trim().toLowerCase();
  return VOLUME_VIEW_PRESETS.find((preset) => preset.id === normalized) ?? VOLUME_VIEW_PRESETS[0];
}
