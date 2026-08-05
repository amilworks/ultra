import type { UploadViewerInfo } from "@/types";

export type SliceChannelSelection = {
  c?: number;
  channels?: number[];
  channelColors?: string[];
};

/** Native scalar slice contract advertised by the backend (currently NIfTI). */
export const usesStrictScalarSliceContract = (
  viewerInfo: Pick<UploadViewerInfo, "viewer"> | null | undefined
): boolean => viewerInfo?.viewer.display_capabilities?.includes("strict_scalar_slice") === true;

export const supportsSliceChannelColor = (
  viewerInfo: Pick<UploadViewerInfo, "viewer"> | null | undefined
): boolean => viewerInfo?.viewer.display_capabilities?.includes("channel_lut_transport") === true;

export const sliceChannelSelection = (
  viewerInfo: Pick<UploadViewerInfo, "viewer" | "selected_indices"> | null | undefined,
  channels: number[] | null | undefined,
  channelColors: string[] | null | undefined,
  scalarChannel?: number | null
): SliceChannelSelection => {
  if (usesStrictScalarSliceContract(viewerInfo)) {
    return { c: scalarChannel ?? channels?.[0] ?? viewerInfo?.selected_indices.C ?? 0 };
  }
  return {
    channels: channels ?? undefined,
    channelColors: supportsSliceChannelColor(viewerInfo) ? channelColors ?? undefined : undefined,
  };
};

export const sliceAxisCoordinates = (
  viewerInfo: Pick<UploadViewerInfo, "viewer"> | null | undefined,
  axis: "x" | "y" | "z",
  indices: { x: number; y: number; z: number }
): { x?: number; y?: number; z?: number } => {
  if (!usesStrictScalarSliceContract(viewerInfo)) {
    return indices;
  }
  return { [axis]: indices[axis] };
};
