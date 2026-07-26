import type { UploadViewerInfo } from "@/types";

export type EffectiveScalarRenderMode = "intensity" | "mask";

export type ResolvedScalarRendering = {
  channel: number;
  time: number;
  requestedMode: "auto" | "intensity" | "mask";
  effectiveMode: EffectiveScalarRenderMode;
  sampling: "box" | "nearest";
  threshold: number | null;
};

type ExactMaskDtype = "uint8" | "uint16" | "int16";

const exactMaskDtype = (dtype: unknown): ExactMaskDtype | null => {
  const normalized = String(dtype ?? "").trim().toLowerCase();
  if (normalized === "uint8" || /^[<>=|]?u1$/.test(normalized)) {
    return "uint8";
  }
  if (normalized === "uint16" || /^[<>=|]?u2$/.test(normalized)) {
    return "uint16";
  }
  if (normalized === "int16" || /^[<>=|]?i2$/.test(normalized)) {
    return "int16";
  }
  return null;
};

export const isExactMaskDtype = (dtype: unknown): boolean =>
  exactMaskDtype(dtype) !== null;

export const canonicalMaskThreshold = (
  value: unknown,
  dtype: unknown
): number | null => {
  if (value == null || (typeof value === "string" && value.trim() === "")) {
    return null;
  }
  const threshold = Number(value);
  const membershipDtype = exactMaskDtype(dtype);
  if (!Number.isFinite(threshold) || membershipDtype === null) {
    return null;
  }
  const [minimum, maximum] =
    membershipDtype === "uint8"
      ? [-1, 255]
      : membershipDtype === "uint16"
        ? [-1, 65_535]
        : [-32_769, 32_767];
  return Math.min(maximum, Math.max(minimum, Math.floor(threshold)));
};

export const isRawMaskForeground = (sample: number, threshold: number): boolean =>
  Number.isFinite(sample) && Number.isFinite(threshold) && sample > threshold;

export const isAuthoritativeBinaryMaskSemantics = (
  semantics: UploadViewerInfo["data_semantics"] | null | undefined,
  channel?: number,
  time?: number
): boolean => {
  const threshold = semantics?.threshold;
  return Boolean(
    semantics?.kind === "binary_mask" &&
      (semantics.strength === "exact" || semantics.strength === "authoritative") &&
      semantics.recommended_view === "mask" &&
      threshold?.method === "otsu-256-v1" &&
      threshold.domain === "raw" &&
      threshold.foreground === "above" &&
      Number.isFinite(threshold.value) &&
      Number.isSafeInteger(threshold.channel) &&
      threshold.channel >= 0 &&
      (channel == null || threshold.channel === channel) &&
      Number.isSafeInteger(threshold.t) &&
      threshold.t >= 0 &&
      (time == null || threshold.t === time) &&
      threshold.sample_scope === "volume" &&
      Number.isSafeInteger(threshold.sample_count) &&
      threshold.sample_count > 0 &&
      String(threshold.sampling_algorithm ?? "").trim()
  );
};

export const isScalarMaskCapability = (
  viewerInfo:
    | Pick<
        UploadViewerInfo,
        "scalar_mask_capability" | "metadata" | "viewer" | "is_volume"
      >
    | null
    | undefined
): boolean => {
  const capability = viewerInfo?.scalar_mask_capability;
  if (!capability || !viewerInfo?.is_volume) {
    return false;
  }
  const expectedSurfaces = viewerInfo.viewer.available_surfaces.filter(
    (surface) => surface === "2d" || surface === "mpr" || surface === "volume"
  );
  const requiredSurfaces = ["2d", "mpr", "volume"];
  const sourceSha = String(viewerInfo.metadata.sha256 ?? "").trim();
  return Boolean(
    sourceSha &&
    capability.version === 1 &&
      capability.source_authority === "original" &&
      (capability.source_format === "tiff" ||
        capability.source_format === "ome-tiff") &&
      isExactMaskDtype(capability.dtype) &&
      String(viewerInfo.metadata.array_dtype).trim().toLowerCase() ===
        capability.dtype &&
      String(capability.source_sha256 ?? "").trim() === sourceSha &&
      capability.threshold_domain === "raw" &&
      capability.threshold_foreground === "above" &&
      capability.slice_delivery === "thresholded_png" &&
      capability.volume_delivery === "raw_scalar" &&
      capability.volume_sampling === "nearest" &&
      capability.channel_selection === "single" &&
      capability.time_selection === "single" &&
      expectedSurfaces.length === requiredSurfaces.length &&
      expectedSurfaces.every(
        (surface, index) => surface === requiredSurfaces[index]
      ) &&
      capability.surfaces.length === requiredSurfaces.length &&
      capability.surfaces.every(
        (surface, index) => surface === requiredSurfaces[index]
      )
  );
};

export const isMaskCapableDataSemantics = (
  semantics: UploadViewerInfo["data_semantics"] | null | undefined
): boolean => {
  void semantics;
  return false;
};

export const isMaskCapableScalarVolume = (
  viewerInfo: Pick<
    UploadViewerInfo,
    "scalar_mask_capability" | "metadata" | "viewer" | "is_volume"
  >
): boolean => isScalarMaskCapability(viewerInfo);

const validAxisIndex = (value: unknown, count: number): number | null => {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isSafeInteger(numeric) && numeric >= 0 && numeric < count
    ? numeric
    : null;
};

export const resolveScalarRendering = ({
  viewerInfo,
  displayState,
  settledTime,
  requestedMode,
  threshold,
}: {
  viewerInfo: Pick<
    UploadViewerInfo,
    | "axis_sizes"
    | "selected_indices"
    | "data_semantics"
    | "scalar_mask_capability"
    | "metadata"
    | "viewer"
    | "is_volume"
  >;
  displayState:
    | Pick<
        NonNullable<UploadViewerInfo["display_defaults"]>,
        "channels" | "volume_channel"
      >
    | null
    | undefined;
  settledTime: number;
  requestedMode: "auto" | "intensity" | "mask" | null | undefined;
  threshold: unknown;
}): ResolvedScalarRendering | null => {
  const channelCount = Number(viewerInfo.axis_sizes.C);
  const timeCount = Number(viewerInfo.axis_sizes.T);
  if (
    !Number.isSafeInteger(channelCount) ||
    channelCount <= 0 ||
    !Number.isSafeInteger(timeCount) ||
    timeCount <= 0
  ) {
    return null;
  }
  const time = validAxisIndex(settledTime, timeCount);
  if (time === null) {
    return null;
  }
  const volumeChannel = validAxisIndex(
    displayState?.volume_channel,
    channelCount
  );
  const selectedChannel = validAxisIndex(
    displayState?.channels?.[0],
    channelCount
  );
  const sourceChannel =
    validAxisIndex(viewerInfo.selected_indices.C, channelCount) ?? 0;
  const channel = volumeChannel ?? selectedChannel ?? sourceChannel;
  const normalizedRequested =
    requestedMode === "mask" || requestedMode === "intensity"
      ? requestedMode
      : "auto";
  const capable = isScalarMaskCapability(viewerInfo);
  const effectiveMode: EffectiveScalarRenderMode =
    capable && normalizedRequested === "mask"
      ? "mask"
      : capable &&
          normalizedRequested === "auto" &&
          isAuthoritativeBinaryMaskSemantics(
            viewerInfo.data_semantics,
            channel,
            time
          )
        ? "mask"
        : "intensity";
  const canonicalThreshold =
    effectiveMode === "mask"
      ? canonicalMaskThreshold(
          threshold,
          viewerInfo.scalar_mask_capability?.dtype
        )
      : null;
  if (effectiveMode === "mask" && canonicalThreshold === null) {
    return null;
  }
  return {
    channel,
    time,
    requestedMode: normalizedRequested,
    effectiveMode,
    sampling: effectiveMode === "mask" ? "nearest" : "box",
    threshold: canonicalThreshold,
  };
};

export const resolveEffectiveScalarRenderMode = (
  viewerInfo: Pick<
    UploadViewerInfo,
    | "axis_sizes"
    | "selected_indices"
    | "data_semantics"
    | "scalar_mask_capability"
    | "metadata"
    | "viewer"
    | "is_volume"
  >,
  requested: "auto" | "intensity" | "mask" | null | undefined
): EffectiveScalarRenderMode =>
  resolveScalarRendering({
    viewerInfo,
    displayState: null,
    settledTime: viewerInfo.selected_indices.T,
    requestedMode: requested,
    threshold: viewerInfo.data_semantics?.threshold?.value,
  })?.effectiveMode ?? "intensity";
