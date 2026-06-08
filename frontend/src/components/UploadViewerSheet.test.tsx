import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { UploadedFileRecord, UploadViewerInfo } from "@/types";

import { UploadViewerWorkspace } from "./UploadViewerSheet";

vi.mock("./viewer/ImageViewerShell", () => ({
  ImageViewerShell: ({ selectedDisplayState }: { selectedDisplayState: UploadViewerInfo["display_defaults"] }) => (
    <div
      data-testid="image-viewer-shell"
      data-enhancement={selectedDisplayState?.enhancement}
      data-fusion={selectedDisplayState?.fusion_method}
      data-signal-floor={selectedDisplayState?.volume_signal_floor}
      data-density={selectedDisplayState?.volume_density}
      data-lighting={selectedDisplayState?.volume_lighting ? "true" : "false"}
      data-lighting-strength={selectedDisplayState?.volume_lighting_strength}
      data-view-preset={selectedDisplayState?.volume_view_preset}
      data-camera-mode={selectedDisplayState?.volume_camera_mode}
    />
  ),
}));

const pdfUpload: UploadedFileRecord = {
  file_id: "file_pdf",
  original_name: "bright-4b.pdf",
  content_type: "application/pdf",
  size_bytes: 4_250_000,
  sha256: "sha-pdf",
  created_at: "2026-06-07T12:30:00Z",
};

describe("UploadViewerWorkspace PDF uploads", () => {
  const renderWorkspace = (file: UploadedFileRecord) => {
    const getUploadViewer = vi.fn(async () => {
      throw new Error("PDF uploads should not load scientific image metadata");
    });
    const uploadDisplayUrl = vi.fn(
      (fileId: string) => `https://ultra.example.org/v2/uploads/${fileId}/display`
    );
    const apiClient = {
      getUploadViewer,
      uploadDisplayUrl,
    } as unknown as ApiClient;

    render(
      <UploadViewerWorkspace
        uploadedFiles={[file]}
        bisqueLinksByFileId={{}}
        apiClient={apiClient}
        active
      />
    );

    return { getUploadViewer };
  };

  it("renders a PDF reader without hydrating the scientific image viewer", async () => {
    const { getUploadViewer } = renderWorkspace(pdfUpload);

    expect(await screen.findByRole("region", { name: "PDF reader" })).toBeInTheDocument();
    expect(screen.getByText("bright-4b.pdf")).toBeInTheDocument();
    expect(screen.getByText("Document · PDF · 4.3 MB")).toBeInTheDocument();
    expect(screen.getByTitle("PDF viewer for bright-4b.pdf")).toHaveAttribute(
      "src",
      "https://ultra.example.org/v2/uploads/file_pdf/display"
    );
    expect(getUploadViewer).not.toHaveBeenCalled();
  });

  it("uses the PDF reader for legacy records with a PDF extension", async () => {
    const { getUploadViewer } = renderWorkspace({
      ...pdfUpload,
      content_type: "application/octet-stream",
    });

    expect(await screen.findByRole("region", { name: "PDF reader" })).toBeInTheDocument();
    expect(getUploadViewer).not.toHaveBeenCalled();
  });
});

describe("UploadViewerWorkspace volume defaults", () => {
  it("propagates scientific volume display defaults into the image viewer shell", async () => {
    const getUploadViewer = vi.fn(async (): Promise<UploadViewerInfo> => ({
      kind: "image",
      file_id: "file_ct",
      original_name: "ct-head.nii.gz",
      modality: "medical",
      backend_mode: "scalar",
      dims_order: "ZYX",
      axis_sizes: { T: 1, C: 1, Z: 32, Y: 512, X: 512 },
      selected_indices: { T: 0, C: 0, Z: 16 },
      is_volume: true,
      is_timeseries: false,
      is_multichannel: false,
      display_defaults: {
        enhancement: "hounsfield:350.000:1800.000",
        negative: false,
        rotate: 0,
        fusion_method: "a",
        channel_mode: "single",
        channels: [0],
        channel_colors: ["#ffffff"],
        time_index: 0,
        z_index: 16,
        scalar_colormap: "grayscale",
        volume_signal_floor: 0.12,
        volume_density: 1.75,
        volume_lighting: true,
        volume_lighting_strength: 0.72,
        volume_channel: 0,
        volume_view_preset: "iso",
        volume_camera_mode: "orthographic",
        volume_clip_min: { x: 0, y: 0, z: 0 },
        volume_clip_max: { x: 1, y: 1, z: 1 },
      },
      service_urls: {
        preview: "/v2/uploads/file_ct/preview",
        slice: "/v2/uploads/file_ct/slice",
        scalar_volume: "/v2/uploads/file_ct/scalar-volume",
      },
      metadata: {
        reader: "nifti-1",
        dims_order: "ZYX",
        array_shape: [32, 512, 512],
        array_dtype: "float32",
        array_min: -1024,
        array_max: 1823,
        physical_spacing: { z: 5, y: 0.439, x: 0.439 },
        scene_count: 1,
        header: {},
        filename_hints: {},
        warnings: [],
      },
      viewer: {
        status: "ready",
        warmup_mode: "lazy",
        backend_mode: "scalar",
        default_surface: "volume",
        available_surfaces: ["2d", "mpr", "volume", "metadata"],
        default_axis: "z",
        slice_axes: ["z", "y", "x"],
        channel_mode: "single",
        volume_mode: "scalar",
        render_policy: "scalar",
        delivery_mode: "scalar",
        diagnostic_surface: "mpr",
        first_paint_mode: "webgl",
        measurement_policy: "spacing-aware",
        texture_policy: "linear",
        display_capabilities: ["slice_navigation", "volume_context", "window_level"],
        viewer_capabilities: ["webgl_first_paint", "scalar_volume_delivery"],
        default_plane: {
          axis: "z",
          label: "XY plane",
          axes: ["Y", "X"],
          pixel_size: { width: 512, height: 512 },
          world_size: { width: 225, height: 225 },
          aspect_ratio: 1,
          spacing: { row: 0.439, col: 0.439 },
        },
        planes: {},
        tile_scheme: { tile_size: 256, format: "png", levels: [] },
        asset_preparation: {
          status: "ready",
          native_supported: true,
          tile_pyramid: "deferred",
          volume_representation: "scalar",
        },
      },
    }));
    const apiClient = {
      getUploadViewer,
      uploadDisplayUrl: vi.fn((fileId: string) => `/v2/uploads/${fileId}/display`),
    } as unknown as ApiClient;

    render(
      <UploadViewerWorkspace
        uploadedFiles={[
          {
            file_id: "file_ct",
            original_name: "ct-head.nii.gz",
            content_type: "application/gzip",
            size_bytes: 1024,
            sha256: "sha-ct",
            created_at: "2026-06-08T00:00:00Z",
          },
        ]}
        bisqueLinksByFileId={{}}
        apiClient={apiClient}
        active
      />
    );

    const shell = await screen.findByTestId("image-viewer-shell");
    expect(shell).toHaveAttribute("data-enhancement", "hounsfield:350.000:1800.000");
    expect(shell).toHaveAttribute("data-fusion", "a");
    expect(shell).toHaveAttribute("data-signal-floor", "0.12");
    expect(shell).toHaveAttribute("data-density", "1.75");
    expect(shell).toHaveAttribute("data-lighting", "true");
    expect(shell).toHaveAttribute("data-lighting-strength", "0.72");
    expect(shell).toHaveAttribute("data-view-preset", "iso");
    expect(shell).toHaveAttribute("data-camera-mode", "orthographic");
  });
});
