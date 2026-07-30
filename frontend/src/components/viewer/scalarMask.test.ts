import { describe, expect, it } from "vitest";

import type { UploadViewerInfo } from "@/types";

import {
  canonicalMaskThreshold,
  isExactMaskDtype,
  isMaskCapableScalarVolume,
  isRawMaskForeground,
  resolveEffectiveScalarRenderMode,
  resolveScalarRendering,
} from "./scalarMask";

type ScalarViewer = Pick<
  UploadViewerInfo,
  | "axis_sizes"
  | "selected_indices"
  | "is_volume"
  | "data_semantics"
  | "scalar_mask_capability"
  | "metadata"
  | "viewer"
>;

const maskViewer: ScalarViewer = {
  is_volume: true,
  axis_sizes: { X: 32, Y: 24, Z: 8, C: 3, T: 2 },
  selected_indices: { C: 2, T: 0, Z: 0 },
  metadata: {
    sha256: "source-sha",
    array_dtype: "uint8",
  } as UploadViewerInfo["metadata"],
  viewer: {
    available_surfaces: ["2d", "mpr", "volume", "metadata"],
  } as UploadViewerInfo["viewer"],
  scalar_mask_capability: {
    version: 1,
    source_authority: "original",
    source_format: "tiff",
    source_sha256: "source-sha",
    dtype: "uint8",
    threshold_domain: "raw",
    threshold_foreground: "above",
    slice_delivery: "thresholded_png",
    volume_delivery: "raw_scalar",
    volume_sampling: "nearest",
    channel_selection: "single",
    time_selection: "single",
    surfaces: ["2d", "mpr", "volume"],
  },
  data_semantics: {
    kind: "probability_mask",
    basis: "bounded_scalar_profile",
    strength: "suggested",
    supported_modes: ["intensity", "mask"],
    recommended_view: "mask",
    threshold: {
      method: "otsu-256-v1",
      value: 120,
      domain: "raw",
      foreground: "above",
      sample_scope: "stratified_z",
      sample_count: 100,
      z_samples: [0],
      channel: 1,
      t: 1,
      sampling_algorithm: "scalar-profile-otsu-256-v1",
    },
  },
};

const resolve = (
  overrides: Partial<Parameters<typeof resolveScalarRendering>[0]> = {}
) =>
  resolveScalarRendering({
    viewerInfo: maskViewer,
    displayState: { channels: [1, 2], volume_channel: 0 },
    settledTime: 1,
    requestedMode: "mask",
    threshold: 120,
    ...overrides,
  });

describe("scalar mask policy", () => {
  it("gates manual mask rendering on the explicit capability record", () => {
    expect(isMaskCapableScalarVolume(maskViewer)).toBe(true);
    expect(
      isMaskCapableScalarVolume({
        ...maskViewer,
        scalar_mask_capability: undefined,
      })
    ).toBe(false);
    expect(
      isMaskCapableScalarVolume({
        ...maskViewer,
        scalar_mask_capability: {
          ...maskViewer.scalar_mask_capability!,
          surfaces: ["volume"],
        },
      })
    ).toBe(false);
    expect(
      isMaskCapableScalarVolume({
        ...maskViewer,
        metadata: {
          ...maskViewer.metadata,
          array_dtype: "float32",
        },
      })
    ).toBe(false);
  });

  it("resolves one canonical channel/time identity with stable precedence", () => {
    expect(resolve()).toEqual({
      channel: 0,
      time: 1,
      requestedMode: "mask",
      effectiveMode: "mask",
      sampling: "nearest",
      threshold: 120,
    });
    expect(
      resolve({
        displayState: { channels: [1, 2], volume_channel: undefined },
      })?.channel
    ).toBe(1);
    expect(resolve({ displayState: null })?.channel).toBe(2);
    expect(resolve({ settledTime: 2 })).toBeNull();
    expect(
      resolve({
        displayState: { channels: [99], volume_channel: 99 },
      })?.channel
    ).toBe(2);
  });

  it("auto-selects mask only for an exact matching binary recommendation", () => {
    expect(
      resolve({
        displayState: { channels: [1], volume_channel: undefined },
        requestedMode: "auto",
      })?.effectiveMode
    ).toBe("intensity");

    const authoritativeViewer: ScalarViewer = {
      ...maskViewer,
      data_semantics: {
        ...maskViewer.data_semantics!,
        kind: "binary_mask",
        strength: "exact",
        threshold: {
          ...maskViewer.data_semantics!.threshold!,
          sample_scope: "volume",
        },
      },
    };
    expect(
      resolve({
        viewerInfo: authoritativeViewer,
        displayState: { channels: [1], volume_channel: undefined },
        requestedMode: "auto",
      })?.effectiveMode
    ).toBe("mask");
    expect(
      resolve({
        viewerInfo: authoritativeViewer,
        displayState: { channels: [0], volume_channel: undefined },
        requestedMode: "auto",
      })?.effectiveMode
    ).toBe("intensity");
    expect(resolveEffectiveScalarRenderMode(maskViewer, "auto")).toBe(
      "intensity"
    );
    expect(resolveEffectiveScalarRenderMode(maskViewer, "intensity")).toBe(
      "intensity"
    );
    expect(resolveEffectiveScalarRenderMode(maskViewer, "mask")).toBe("mask");
  });

  it("fails closed when mask threshold or capability is unavailable", () => {
    expect(resolve({ threshold: Number.NaN })).toBeNull();
    expect(
      resolve({
        viewerInfo: {
          ...maskViewer,
          scalar_mask_capability: undefined,
        },
      })
    ).toEqual(
      expect.objectContaining({
        effectiveMode: "intensity",
        sampling: "box",
        threshold: null,
      })
    );
  });

  it("uses strict source-unit membership without clamping outside the raw range", () => {
    expect(isRawMaskForeground(121, 120)).toBe(true);
    expect(isRawMaskForeground(120, 120)).toBe(false);
    expect(isRawMaskForeground(0, -1)).toBe(true);
    expect(isRawMaskForeground(65_535, 65_535)).toBe(false);
  });

  it.each([
    ["uint8", 120.9, 120],
    ["uint8", -99, -1],
    ["uint8", 999, 255],
    ["uint16", 65_534.9, 65_534],
    ["uint16", -99, -1],
    ["uint16", 99_999, 65_535],
    ["int16", -32_768.1, -32_769],
    ["int16", 12.9, 12],
    ["int16", 99_999, 32_767],
  ] as const)(
    "canonicalizes %s threshold %s membership-equivalently for CPU and GLSL",
    (dtype, threshold, expected) => {
      const canonical = canonicalMaskThreshold(threshold, dtype);
      expect(canonical).toBe(expected);
      expect(Math.fround(canonical!)).toBe(canonical);
    }
  );

  it("withholds float32 and lossy mask dtypes instead of changing membership", () => {
    expect(canonicalMaskThreshold(null, "uint8")).toBeNull();
    expect(canonicalMaskThreshold("", "uint8")).toBeNull();
    expect(canonicalMaskThreshold(0.9999999701976776, "float32")).toBeNull();
    expect(canonicalMaskThreshold(120, "float64")).toBeNull();
    expect(canonicalMaskThreshold(120, "uint32")).toBeNull();
    expect(isExactMaskDtype("<u2")).toBe(true);
    expect(isExactMaskDtype("float32")).toBe(false);
    expect(isExactMaskDtype("int8")).toBe(false);
  });

  it("requires the capability and metadata to share one non-empty source SHA", () => {
    expect(
      isMaskCapableScalarVolume({
        ...maskViewer,
        metadata: { ...maskViewer.metadata, sha256: undefined },
      })
    ).toBe(false);
    expect(
      isMaskCapableScalarVolume({
        ...maskViewer,
        scalar_mask_capability: {
          ...maskViewer.scalar_mask_capability!,
          source_sha256: undefined as unknown as string,
        },
      })
    ).toBe(false);
  });
});
