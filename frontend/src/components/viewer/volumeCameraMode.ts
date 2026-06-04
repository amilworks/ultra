export type VolumeCameraModeId = "perspective" | "orthographic";

export type VolumeCameraMode = {
  id: VolumeCameraModeId;
  label: string;
  isOrthographic: boolean;
};

export const VOLUME_CAMERA_MODES: VolumeCameraMode[] = [
  { id: "perspective", label: "Perspective", isOrthographic: false },
  { id: "orthographic", label: "Orthographic", isOrthographic: true },
];

export function resolveVolumeCameraMode(value?: unknown): VolumeCameraMode {
  const normalized = String(value ?? "").trim().toLowerCase();
  return VOLUME_CAMERA_MODES.find((mode) => mode.id === normalized) ?? VOLUME_CAMERA_MODES[0];
}
