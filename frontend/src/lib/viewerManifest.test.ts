import { describe, expect, it } from "vitest";

import { thumbnailScrubAxis } from "./thumbnailScrubAxis";
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
});

describe("normalizeUploadViewerInfo scalar medical defaults", () => {
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

describe("normalizeUploadViewerInfo DREAM.3D phase metadata", () => {
  it("preserves the backend source and provenance instead of implying phase detection", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "hdf5",
      file_id: "file-dream3d",
      original_name: "synthetic-volume.dream3d",
      hdf5: {
        enabled: true,
        supported: true,
        materials: {
          detected: true,
          schema: "dream3d",
          capabilities: ["grain_metrics"],
          roles: {},
          phase_names: ["Primary"],
          phase_names_source: "stored_metadata",
          phase_names_provenance:
            "Read from stored DREAM.3D PhaseName metadata; no phase-identification algorithm was run.",
          recommended_view: "materials",
        },
      },
    });

    expect(viewer.hdf5?.materials).toMatchObject({
      phase_names: ["Primary"],
      phase_names_source: "stored_metadata",
      phase_names_provenance:
        "Read from stored DREAM.3D PhaseName metadata; no phase-identification algorithm was run.",
    });
  });

  it("preserves geometry-consistency and reserved-feature-zero evidence", () => {
    const viewer = normalizeUploadViewerInfo({
      kind: "hdf5",
      file_id: "file-dream3d-evidence",
      original_name: "qualified.dream3d",
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
        materials: {
          detected: true,
          schema: "dream3d",
          capabilities: ["grain_metrics"],
          roles: {},
          phase_names: [],
          declared_feature_tuple_count: 42,
          referenced_positive_feature_count: 0,
          feature_id_scan_complete: true,
          feature_id_consistency: false,
          feature_zero_reserved: true,
          recommended_view: "materials",
        },
      },
    });

    expect(viewer.hdf5?.summary.geometry).toMatchObject({
      cell_data_path: "/DataContainers/Image/CellData",
      cell_data_consistent: true,
      complete: true,
    });
    expect(viewer.hdf5?.materials?.feature_zero_reserved).toBe(true);
    expect(viewer.hdf5?.materials).toMatchObject({
      declared_feature_tuple_count: 42,
      referenced_positive_feature_count: 0,
      feature_id_scan_complete: true,
      feature_id_consistency: false,
      grain_count: null,
    });
  });
});
