import type { UploadViewerInfo } from "@/types";

export type ThumbnailScrubConfig = {
  axis: "z" | "t";
  count: number;
  channels: number[];
  /** Source-channel-indexed palette; ApiClient projects it into selected order. */
  channelColors: string[];
  timeIndex: number;
  zIndex: number;
  enhancement: string;
  fusionMethod: string;
  negative: boolean;
};

export type ThumbnailScrubSliceRequest = {
  axis: "z";
  z: number;
  t: number;
  enhancement: string;
  fusionMethod: string;
  negative: boolean;
  channels: number[];
  channelColors: string[];
  fullResolution: false;
  cacheKey: string;
};

type ThumbnailScrubViewerInfo = Pick<
  UploadViewerInfo,
  "axis_sizes" | "selected_indices" | "display_defaults" | "phys"
>;

const boundedIndex = (value: unknown, count: number, fallback = 0): number => {
  const numeric = Number(value);
  const candidate = Number.isFinite(numeric) ? Math.floor(numeric) : fallback;
  return Math.max(0, Math.min(Math.max(1, count) - 1, candidate));
};

const validSourceIndex = (value: unknown, count: number): number | null => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  const index = Math.floor(numeric);
  return index >= 0 && index < count ? index : null;
};

/**
 * Which axis a gallery thumbnail should scrub through, and how many planes it has.
 * Prefer Z (a focal/depth stack); fall back to T for a Z-less time-series so movie
 * thumbnails advance through time instead of a single clamped plane.
 */
export function thumbnailScrubAxis(
  axisSizes: Pick<UploadViewerInfo["axis_sizes"], "Z" | "T"> | null | undefined
): { axis: "z" | "t"; count: number } {
  const z = Math.max(1, Math.floor(Number(axisSizes?.Z ?? 1)) || 1);
  const t = Math.max(1, Math.floor(Number(axisSizes?.T ?? 1)) || 1);
  if (z > 1) {
    return { axis: "z", count: z };
  }
  if (t > 1) {
    return { axis: "t", count: t };
  }
  return { axis: "z", count: 1 };
}

/**
 * Resolve the complete display contract for gallery scrubbing from authoritative
 * viewer metadata. A scrubbed frame must not silently fall back to source channel
 * zero: it should be the same composite, LUT, enhancement, and fixed time/depth as
 * the static resource thumbnail, with only the scrub axis changing.
 */
export function thumbnailScrubConfig(info: ThumbnailScrubViewerInfo): ThumbnailScrubConfig {
  const { axis, count } = thumbnailScrubAxis(info.axis_sizes);
  const channelCount = Math.max(1, Math.floor(Number(info.axis_sizes.C)) || 1);
  const defaults = info.display_defaults;
  const seenChannels = new Set<number>();
  const channels = (defaults?.channels ?? [])
    .map((value) => validSourceIndex(value, channelCount))
    .filter((value): value is number => {
      if (value === null || seenChannels.has(value)) {
        return false;
      }
      seenChannels.add(value);
      return true;
    });
  if (channels.length === 0) {
    channels.push(boundedIndex(info.selected_indices.C, channelCount));
  }

  const channelColors = Array.from({ length: channelCount }, () => "");
  for (const entry of info.phys?.channel_colors ?? []) {
    const index = validSourceIndex(entry.index, channelCount);
    const color = String(entry.hex ?? "").trim();
    if (index !== null && color) {
      channelColors[index] = color;
    }
  }
  const defaultColors = (defaults?.channel_colors ?? []).map((value) => String(value ?? "").trim());
  if (defaultColors.length >= channelCount) {
    defaultColors.slice(0, channelCount).forEach((color, index) => {
      if (color) {
        channelColors[index] = color;
      }
    });
  } else if (defaultColors.length === channels.length) {
    // Tolerate older manifests that serialized colors in selected-channel order.
    channels.forEach((channel, index) => {
      if (defaultColors[index]) {
        channelColors[channel] = defaultColors[index];
      }
    });
  }

  return {
    axis,
    count,
    channels,
    channelColors,
    timeIndex: boundedIndex(defaults?.time_index, info.axis_sizes.T, info.selected_indices.T),
    zIndex: boundedIndex(defaults?.z_index, info.axis_sizes.Z, info.selected_indices.Z),
    enhancement: String(defaults?.enhancement ?? "d"),
    fusionMethod: String(defaults?.fusion_method ?? "m"),
    negative: Boolean(defaults?.negative ?? false),
  };
}

/** Build one bounded XY-plane request while varying only the selected scrub axis. */
export function thumbnailScrubSliceRequest(
  config: ThumbnailScrubConfig,
  index: number
): ThumbnailScrubSliceRequest {
  const scrubIndex = boundedIndex(index, config.count);
  return {
    axis: "z",
    z: config.axis === "z" ? scrubIndex : config.zIndex,
    t: config.axis === "t" ? scrubIndex : config.timeIndex,
    enhancement: config.enhancement,
    fusionMethod: config.fusionMethod,
    negative: config.negative,
    channels: config.channels,
    channelColors: config.channelColors,
    fullResolution: false,
    cacheKey: "metadata-zscrub-v1",
  };
}
