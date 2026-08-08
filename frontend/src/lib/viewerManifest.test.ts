import { describe, expect, it } from "vitest";
import type { UploadViewerInfo } from "@/types";

import {
  thumbnailScrubAxis,
  thumbnailScrubConfig,
  thumbnailScrubSliceRequest,
} from "./thumbnailScrubAxis";
import { ApiClient } from "./api";
import { normalizeUploadViewerInfo } from "./viewerManifest";

describe("thumbnailScrubAxis", () => {
  it("scrubs Z for a focal/depth stack", () => {
    expect(thumbnailScrubAxis({ Z: 32, T: 1 })).toEqual({ axis: "z", count: 32 });
  });
  it("scrubs T for a Z-less time-series (e.g. a 61-frame movie)", () => {
    // The real prod regression: axes (t,c,y,x), Z=1 — must scrub time, not the single Z plane.
    expect(thumbnailScrubAxis({ Z: 1, T: 61 })).toEqual({ axis: "t", count: 61 });
  });
  it("prefers Z when an image has both depth and time", () => {
    expect(thumbnailScrubAxis({ Z: 10, T: 5 })).toEqual({ axis: "z", count: 10 });
  });
  it("is non-scrubbable (count 1) for a flat single-plane image", () => {
    expect(thumbnailScrubAxis({ Z: 1, T: 1 })).toEqual({ axis: "z", count: 1 });
  });
  it("tolerates missing or invalid axis sizes", () => {
    expect(thumbnailScrubAxis(null)).toEqual({ axis: "z", count: 1 });
    expect(thumbnailScrubAxis({ Z: 0, T: 0 } as never)).toEqual({ axis: "z", count: 1 });
  });

  it("keeps multichannel display metadata on every scrubbed z slice", () => {
    const channelColors = [
      "#1e90ff",
      "#00ff66",
      "#ff3b3b",
      "#ff00ff",
      "#ffd400",
      "#00e5ff",
      "#1e90ff",
    ];
    const config = thumbnailScrubConfig({
      axis_sizes: { T: 1, C: 7, Z: 80, Y: 624, X: 924 },
      selected_indices: { T: 0, C: 1, Z: 40 },
      display_defaults: {
        enhancement: "d",
        negative: false,
        rotate: 0,
        fusion_method: "m",
        channel_mode: "composite",
        channels: [1, 3, 5],
        channel_colors: channelColors,
        time_index: 0,
        z_index: 40,
      },
      phys: {
        channel_colors: channelColors.map((hex, index) => ({
          index,
          hex,
          rgb: [255, 255, 255],
        })),
      },
      viewer: {
        display_capabilities: ["channel_color", "channel_lut_transport"],
      } as UploadViewerInfo["viewer"],
    });

    expect(config).toMatchObject({
      axis: "z",
      count: 80,
      channels: [1, 3, 5],
      channelColors,
      timeIndex: 0,
      zIndex: 40,
    });
    expect(thumbnailScrubSliceRequest(config, 63)).toEqual({
      axis: "z",
      z: 63,
      t: 0,
      enhancement: "d",
      fusionMethod: "m",
      negative: false,
      channels: [1, 3, 5],
      channelColors,
      fullResolution: false,
      cacheKey: "metadata-zscrub-v1",
    });
  });

  it("uses the strict scalar NIfTI slice contract for metadata hover scrubbing", () => {
    const info = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-nifti",
      original_name: "brain.nii",
      backend_mode: "scalar",
      axis_sizes: { T: 2, C: 2, Z: 8, Y: 16, X: 24 },
      selected_indices: { T: 1, C: 1, Z: 3 },
      display_defaults: {
        channels: [1],
        channel_colors: ["#0000ff", "#ff0000"],
        time_index: 1,
        z_index: 3,
      },
      viewer: {
        backend_mode: "scalar",
        delivery_mode: "scalar",
        volume_mode: "scalar",
        render_policy: "scalar",
        display_capabilities: ["strict_scalar_slice", "channel_visibility", "time_navigation"],
      },
    });
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const request = thumbnailScrubSliceRequest(thumbnailScrubConfig(info), 5);
    const parsed = new URL(client.uploadSliceUrl(info.file_id, request));

    expect(parsed.searchParams.get("axis")).toBe("z");
    expect(parsed.searchParams.get("z")).toBe("5");
    expect(parsed.searchParams.get("t")).toBe("1");
    expect(parsed.searchParams.get("c")).toBe("1");
    expect(parsed.searchParams.has("channels")).toBe(false);
    expect(parsed.searchParams.has("channel_colors")).toBe(false);
    expect(parsed.searchParams.has("x")).toBe(false);
    expect(parsed.searchParams.has("y")).toBe(false);
  });
});

describe("normalizeUploadViewerInfo Scene3D admission", () => {
  it("preserves an unsupported PLY identity instead of relabeling it as a point cloud", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "scene3d",
      file_id: "file-ascii",
      original_name: "ascii.ply",
      decodable: false,
      scene_kind: "unknown",
      status: "failed",
      message: "ASCII PLY is not supported; re-export as binary PLY.",
      service_urls: {},
    });

    expect(viewer.decodable).toBe(false);
    expect(viewer.scene3d?.scene_kind).toBe("unknown");
    expect(viewer.scene3d?.status).toBe("failed");
    expect(viewer.message).toContain("ASCII PLY");
  });
});

describe("normalizeUploadViewerInfo scalar medical defaults", () => {
  it("does not repair malformed mask provenance into authoritative semantics", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "malformed-mask",
      original_name: "malformed.tif",
      axis_sizes: { T: 1, C: 1, Z: 2, Y: 2, X: 2 },
      is_volume: true,
      data_semantics: {
        kind: "binary_mask",
        basis: "exact",
        strength: "exact",
        supported_modes: ["intensity", "mask"],
        recommended_view: "mask",
        threshold: {
          method: "otsu-256-v1",
          value: 120,
          domain: "normalized",
          foreground: "below",
          sample_scope: "volume",
          sample_count: 8,
          channel: -1,
          t: 0,
          sampling_algorithm: "scalar-profile-otsu-256-v1",
        },
      },
      metadata: { array_dtype: "uint8" },
      viewer: { volume_mode: "slice_stack", render_policy: "scalar" },
    });

    expect(viewer.data_semantics?.threshold).toBeUndefined();
  });

  it("preserves mask semantics, raw Otsu provenance, SHA, and restored calibration defaults", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-mask",
      original_name: "tomm-mask.tif",
      dims_order: "ZYX",
      axis_sizes: { T: 3, C: 1, Z: 65, Y: 312, X: 462 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: true,
      data_semantics: {
        kind: "probability_mask",
        basis: "bounded_scalar_profile",
        strength: "suggested",
        supported_modes: ["intensity", "mask"],
        recommended_view: "intensity",
        threshold: {
          method: "otsu-256-v1",
          value: 120,
          domain: "raw",
          foreground: "above",
          sample_scope: "stratified_z",
          sample_count: 1048576,
          z_samples: [0, 32, 64],
          channel: 0,
          t: 0,
          sampling_algorithm: "scalar-profile-otsu-256-v1",
        },
      },
      viewer_calibrations: {
        version: 1,
        source_sha256: "source-sha",
        selections: {
          "c0:t0": {
            revision: 1,
            channel: 0,
            t: 0,
            render_mode: "mask",
            threshold_method: "manual",
            threshold_value: 133,
            threshold_foreground: "above",
            threshold_provenance: {
              method: "otsu-256-v1",
              value: 120,
              domain: "raw",
              foreground: "above",
              channel: 0,
              t: 0,
              sample_scope: "stratified_z",
              sample_count: 1048576,
              sampling_algorithm: "scalar-profile-otsu-256-v1",
              sampling_strategy: "stratified-z-spatial",
              z_samples: [0, 32, 64],
              source_sha256: "source-sha",
              bins: 256,
            },
          },
          "c0:t2": {
            revision: 1,
            channel: 0,
            t: 2,
            render_mode: "mask",
            threshold_method: "manual",
            threshold_value: 231,
            threshold_foreground: "above",
            threshold_provenance: {
              method: "otsu-256-v1",
              value: 220,
              domain: "raw",
              foreground: "above",
              channel: 0,
              t: 2,
              sample_scope: "stratified_z",
              sample_count: 1048576,
              sampling_algorithm: "scalar-profile-otsu-256-v1",
              sampling_strategy: "stratified-z-spatial",
              z_samples: [0, 32, 64],
              source_sha256: "source-sha",
              bins: 256,
            },
          },
        },
      },
      display_defaults: {
        scalar_render_mode: "mask",
        scalar_threshold_method: "manual",
        scalar_threshold_value: 133,
        scalar_threshold_foreground: "above",
      },
      metadata: {
        array_dtype: "uint8",
        sha256: "source-sha",
        size_bytes: 12345,
      },
      viewer: {
        volume_mode: "slice_stack",
        render_policy: "scalar",
        default_surface: "volume",
      },
    });

    expect(viewer.data_semantics).toMatchObject({
      kind: "probability_mask",
      basis: "bounded_scalar_profile",
      strength: "suggested",
      recommended_view: "intensity",
      threshold: { value: 120, domain: "raw", foreground: "above" },
    });
    expect(viewer.display_defaults).toMatchObject({
      scalar_render_mode: "mask",
      scalar_threshold_method: "manual",
      scalar_threshold_value: 133,
    });
    expect(viewer.metadata.sha256).toBe("source-sha");
    expect(viewer.metadata.size_bytes).toBe(12345);
    expect(viewer.viewer_calibrations?.selections["c0:t0"].threshold_value).toBe(133);
    expect(viewer.viewer_calibrations?.selections["c0:t2"]).toMatchObject({
      t: 2,
      threshold_value: 231,
      threshold_provenance: { value: 220, t: 2 },
    });
  });

  it("preserves the backend physical spacing unit for derived volume extents", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-nifti",
      original_name: "timeseries.nii.gz",
      modality: "medical",
      backend_mode: "scalar",
      dims_order: "TZYX",
      axis_sizes: { T: 405, C: 1, Z: 72, Y: 104, X: 90 },
      selected_indices: { T: 259, C: 0, Z: 43 },
      is_volume: true,
      metadata: {
        array_dtype: "int16",
        physical_spacing: { x: 2, y: 2, z: 2 },
        physical_spacing_unit: "mm",
      },
      viewer: { volume_mode: "scalar", default_surface: "volume" },
    });

    expect(viewer.metadata.physical_spacing).toEqual({ x: 2, y: 2, z: 2 });
    expect(viewer.metadata.physical_spacing_unit).toBe("mm");
  });

  it("reconstructs preferred-reader phys from top-level anisotropic metadata", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-bioio",
      original_name: "markers.nd2",
      modality: "microscopy",
      dims_order: "TCZYX",
      axis_sizes: { T: 2, C: 3, Z: 4, Y: 80, X: 120 },
      channel_names: ["DAPI", "FITC", "Cy5"],
      physical_spacing: { x: 0.11, y: 0.22, z: 1.7 },
      metadata: {
        reader: "bioio",
        spacing_units: { x: "um", y: "um", z: "um" },
      },
      display_defaults: { channels: [2, 0] },
      viewer: { render_policy: "scalar" },
    });

    expect(viewer.metadata.physical_spacing).toEqual({ x: 0.11, y: 0.22, z: 1.7 });
    expect(viewer.metadata.spacing_units).toEqual({ x: "um", y: "um", z: "um" });
    expect(viewer.phys).toMatchObject({
      x: 120,
      y: 80,
      z: 4,
      t: 2,
      ch: 3,
      pixel_size: [0.11, 0.22, 1.7, 1],
      pixel_units: ["um", "um", "um", "frame"],
      channel_names: ["DAPI", "FITC", "Cy5"],
      display_channels: [2, 0],
    });
  });

  it("promotes CT-like scalar volumes to legible scientific 3D defaults", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-ct",
      original_name: "ct-head.nii.gz",
      modality: "medical",
      backend_mode: "scalar",
      dims_order: "ZYX",
      axis_sizes: { T: 1, C: 1, Z: 32, Y: 512, X: 512 },
      selected_indices: { T: 0, C: 0, Z: 16 },
      is_volume: true,
      service_urls: {
        scalar_volume: "/v2/uploads/file-ct/scalar-volume",
      },
      metadata: {
        array_dtype: "float32",
        array_min: -1023.9,
        array_max: 1823,
        physical_spacing: { x: 0.439, y: 0.439, z: 5 },
      },
      viewer: {
        volume_mode: "scalar",
        render_policy: "scalar",
        default_surface: "volume",
      },
    });

    expect(viewer.display_defaults?.enhancement).toBe("hounsfield:40.000:80.000");
    expect(viewer.display_defaults?.volume_signal_floor).toBe(0.12);
    expect(viewer.display_defaults?.volume_density).toBe(1.75);
    expect(viewer.display_defaults?.volume_lighting).toBe(true);
    expect(viewer.display_defaults?.volume_lighting_strength).toBe(0.72);
    expect(viewer.display_defaults?.volume_view_preset).toBe("iso");
    expect(viewer.display_defaults?.volume_camera_mode).toBe("orthographic");
  });

  it("honors sub-millimeter and anisotropic in-plane spacing without clamping to 1mm", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-aniso",
      original_name: "aniso.nii",
      modality: "medical",
      backend_mode: "scalar",
      dims_order: "ZYX",
      axis_sizes: { T: 1, C: 1, Z: 32, Y: 512, X: 512 },
      selected_indices: { T: 0, C: 0, Z: 16 },
      is_volume: true,
      service_urls: { scalar_volume: "/v2/uploads/file-aniso/scalar-volume" },
      // In-plane spacing differs (x=0.5, y=2.0): the XY plane is physically
      // anisotropic. Pre-fix both were clamped to >=1mm -> a wrong 1:1 aspect.
      metadata: { array_dtype: "int16", physical_spacing: { x: 0.5, y: 2.0, z: 5 } },
      viewer: { volume_mode: "scalar", default_surface: "2d" },
    });

    const plane = viewer.viewer.default_plane;
    // XY world size = pixels x spacing: 512*0.5 x 512*2.0.
    expect(plane?.world_size.width).toBeCloseTo(256, 3);
    expect(plane?.world_size.height).toBeCloseTo(1024, 3);
    expect(plane?.aspect_ratio).toBeCloseTo(0.25, 5);
    // The clamped-to-1 bug would have produced a square 512x512 (aspect 1).
    expect(plane?.aspect_ratio).not.toBeCloseTo(1, 2);
  });

  it("keeps generic scalar volume defaults when the range is not CT-like", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-generic",
      original_name: "normalized-volume.nii",
      modality: "medical",
      backend_mode: "scalar",
      axis_sizes: { T: 1, C: 1, Z: 8, Y: 32, X: 32 },
      selected_indices: { T: 0, C: 0, Z: 4 },
      is_volume: true,
      metadata: {
        array_dtype: "float32",
        array_min: -1,
        array_max: 1,
      },
      viewer: {
        volume_mode: "scalar",
        render_policy: "scalar",
      },
    });

    expect(viewer.display_defaults?.enhancement).toBe("d");
    expect(viewer.display_defaults?.volume_signal_floor).toBe(0);
    expect(viewer.display_defaults?.volume_density).toBe(1);
    expect(viewer.display_defaults?.volume_lighting).toBe(false);
    expect(viewer.display_defaults?.volume_camera_mode).toBeUndefined();
  });
});

describe("normalizeUploadViewerInfo unsupported formats", () => {
  it("preserves the undecodable signal so the viewer can show a download card", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "unsupported",
      decodable: false,
      file_id: "file-lif",
      original_name: "Training_20240812-czQC.lif",
      modality: "image",
      backend_mode: "none",
      dims_order: "YX",
      axis_sizes: { T: 1, C: 1, Z: 1, Y: 0, X: 0 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: false,
      service_urls: { download: "/v2/resources/file-lif/download" },
      message: "LIF files can't be previewed by the image engine yet.",
    });
    expect(viewer.kind).toBe("unsupported");
    expect(viewer.decodable).toBe(false);
    expect(viewer.message).toContain("LIF");
  });

  it("treats a normal decodable image as decodable (no false positive)", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-png",
      original_name: "plate.png",
      dims_order: "YX",
      axis_sizes: { T: 1, C: 3, Z: 1, Y: 256, X: 256 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: false,
      service_urls: {},
    });
    expect(viewer.kind).toBe("image");
    expect(viewer.decodable).toBeUndefined();
  });
});

describe("normalizeUploadViewerInfo HDF5 geometry", () => {
  it("preserves geometry-consistency evidence", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "hdf5",
      file_id: "file-hdf5-evidence",
      original_name: "qualified.h5",
      hdf5: {
        enabled: true,
        supported: true,
        summary: {
          geometry: {
            path: "/DataContainers/Image/_SIMPL_GEOMETRY",
            dimensions: [4, 3, 2],
            spacing: [0.5, 0.5, 1],
            origin: [0, 0, 0],
            cell_data_path: "/DataContainers/Image/CellData",
            cell_data_consistent: true,
            complete: true,
          },
        },
      },
    });

    expect(viewer.hdf5?.summary.geometry).toMatchObject({
      cell_data_path: "/DataContainers/Image/CellData",
      cell_data_consistent: true,
      complete: true,
    });
  });
});

describe("normalizeUploadViewerInfo NGFF channel identity", () => {
  it("rejects malformed channel indices and deduplicates valid identities", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "malformed-channels",
      original_name: "channels.ome.tif",
      axis_sizes: { T: 1, C: 3, Z: 1, Y: 8, X: 8 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      phys: {
        ch: 3,
        display_channels: ["malformed", -1, 2, 2, 99, 1.5],
      },
      display_defaults: {
        channels: ["malformed", -1, 2, 2, 99, 1.5],
      },
      viewer: { render_policy: "scalar" },
    });

    expect(viewer.phys?.display_channels).toEqual([2]);
    expect(viewer.display_defaults?.channels).toEqual([2]);
  });

  it("round-trips canonical OME colors into sparse slice and tile requests", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "image",
      file_id: "file-ngff",
      original_name: "markers.ome.zarr",
      modality: "microscopy",
      backend_mode: "pyramid",
      dims_order: "TCZYX",
      axis_sizes: { T: 1, C: 3, Z: 1, Y: 64, X: 64 },
      selected_indices: { T: 0, C: 0, Z: 0 },
      is_volume: false,
      is_multichannel: true,
      phys: {
        channel_names: ["DAPI", "FITC", "TRITC"],
        channel_colors: [
          { index: 0, hex: "#0000FF", rgb: [0, 0, 255] },
          { index: 1, hex: "#00FF00", rgb: [0, 255, 0] },
          { index: 2, hex: "#FF0000", rgb: [255, 0, 0] },
        ],
      },
      display_defaults: {
        enhancement: "d",
        negative: false,
        rotate: 0,
        fusion_method: "m",
        channel_mode: "composite",
        channels: [0, 2],
        channel_colors: ["#0000FF", "#00FF00", "#FF0000"],
        time_index: 0,
        z_index: 0,
      },
      viewer: {
        backend_mode: "pyramid",
        delivery_mode: "deferred_multiscale",
        render_policy: "scalar",
        channel_mode: "composite",
        tile_scheme: {
          tile_size: 512,
          format: "png",
          levels: [{ level: 0, width: 64, height: 64, columns: 1, rows: 1, downsample: 1 }],
        },
      },
    });
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const channels = viewer.display_defaults?.channels;
    const colors = viewer.display_defaults?.channel_colors;
    const urls = [
      client.uploadSliceUrl(viewer.file_id, { axis: "z", z: 0, channels, channelColors: colors }),
      client.uploadTileUrl(viewer.file_id, {
        axis: "z",
        level: 0,
        tileX: 0,
        tileY: 0,
        channels,
        channelColors: colors,
      }),
    ];

    expect(viewer.phys?.channel_colors).toEqual([
      { index: 0, hex: "#0000FF", rgb: [0, 0, 255] },
      { index: 1, hex: "#00FF00", rgb: [0, 255, 0] },
      { index: 2, hex: "#FF0000", rgb: [255, 0, 0] },
    ]);
    urls.forEach((value) => {
      const parsed = new URL(value);
      expect(parsed.searchParams.get("channels")).toBe("0,2");
      expect(parsed.searchParams.get("channel_colors")).toBe("#0000FF,#FF0000");
    });
  });
});
