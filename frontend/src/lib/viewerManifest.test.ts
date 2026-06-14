import { describe, expect, it } from "vitest";

import { normalizeUploadViewerInfo } from "./viewerManifest";

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
