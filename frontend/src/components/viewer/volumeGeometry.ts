import type { UploadViewerInfo } from "@/types";

type PhysicalSpacing = UploadViewerInfo["metadata"]["physical_spacing"] | null | undefined;

export type PhysicalVolumeGeometry = {
  voxelSpacing: {
    x: number;
    y: number;
    z: number;
  };
  worldWidth: number;
  worldHeight: number;
  worldDepth: number;
  normalizedScale: {
    x: number;
    y: number;
    z: number;
  };
  aspectLabel: string;
  spacingLabel: string;
  isAnisotropic: boolean;
};

const positiveNumber = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
};

const formatExtentValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 100 || Number.isInteger(value)) {
    return value.toFixed(0);
  }
  return value.toFixed(2);
};

const formatSpacingValue = (value: number): string =>
  Number.isFinite(value) ? value.toFixed(2) : "0.00";

export function computePhysicalVolumeGeometry({
  planePixelSize,
  volumeDepth,
  physicalSpacing,
}: {
  planePixelSize: { width?: number | null; height?: number | null };
  volumeDepth: number;
  physicalSpacing?: PhysicalSpacing;
}): PhysicalVolumeGeometry {
  const xSpacing = positiveNumber(physicalSpacing?.x, 1);
  const ySpacing = positiveNumber(physicalSpacing?.y, 1);
  const zSpacing = positiveNumber(physicalSpacing?.z, 1);
  const width = positiveNumber(planePixelSize.width, 1);
  const height = positiveNumber(planePixelSize.height, 1);
  const depth = positiveNumber(volumeDepth, 1);
  const worldWidth = Math.max(1, width * xSpacing);
  const worldHeight = Math.max(1, height * ySpacing);
  const worldDepth = Math.max(1, depth * zSpacing);
  const normalizer = Math.max(worldWidth, worldHeight, worldDepth, 1);
  const spacingValues = [xSpacing, ySpacing, zSpacing];
  const minSpacing = Math.min(...spacingValues);
  const maxSpacing = Math.max(...spacingValues);

  return {
    voxelSpacing: {
      x: xSpacing,
      y: ySpacing,
      z: zSpacing,
    },
    worldWidth,
    worldHeight,
    worldDepth,
    normalizedScale: {
      x: worldWidth / normalizer,
      y: worldHeight / normalizer,
      z: worldDepth / normalizer,
    },
    aspectLabel: [worldWidth, worldHeight, worldDepth].map(formatExtentValue).join(" x "),
    spacingLabel: [xSpacing, ySpacing, zSpacing].map(formatSpacingValue).join(" x "),
    isAnisotropic: maxSpacing / Math.max(minSpacing, 1e-9) > 1.01,
  };
}
