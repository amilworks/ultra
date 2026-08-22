import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ApiClient, UploadVideoExportResponse } from "@/lib/api";

import { VolumeVideoExportDialog } from "./VolumeVideoExportDialog";

const queuedExport: UploadVideoExportResponse = {
  render_id: "a".repeat(64),
  status: "queued",
  mode: "time_series",
  profile: "complete",
  fps: 24,
  source_frame_count: 405,
  frames_total: 405,
  frames_completed: 0,
  sampled: false,
};

const renderDialog = (
  overrides: Partial<ComponentProps<typeof VolumeVideoExportDialog>> = {}
) => {
  const apiClient = {
    createUploadVideoExport: vi.fn().mockResolvedValue(queuedExport),
    getUploadVideoExport: vi.fn().mockResolvedValue(queuedExport),
    downloadUploadVideoExport: vi.fn(),
  } as unknown as ApiClient;
  render(
    <VolumeVideoExportDialog
      apiClient={apiClient}
      fileId="file_stack"
      originalName="brain.ome.tiff"
      zCount={300}
      tCount={405}
      currentZ={11}
      currentT={8}
      channels={[1, 3]}
      channelColors={["#000000", "#ff0000", "#000000", "#00ff00"]}
      strictScalarSlice={false}
      scalarRenderMode="intensity"
      {...overrides}
    />
  );
  return apiClient;
};

describe("VolumeVideoExportDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("freezes the active scientific selectors into a complete time-series recipe", async () => {
    const apiClient = renderDialog();

    fireEvent.click(screen.getByRole("button", { name: /export video/i }));
    fireEvent.click(screen.getByRole("button", { name: /time series/i }));
    fireEvent.click(screen.getByRole("button", { name: /complete/i }));
    fireEvent.click(screen.getByRole("button", { name: /create mp4/i }));

    await waitFor(() => {
      expect(apiClient.createUploadVideoExport).toHaveBeenCalledWith(
        "file_stack",
        expect.objectContaining({
          mode: "time_series",
          profile: "complete",
          fixed_z: 11,
          fixed_t: 8,
          channels: [1, 3],
          channel_colors: ["#ff0000", "#00ff00"],
          scalar_render_mode: "intensity",
        })
      );
    });
    expect(screen.getByText("405 of 405")).toBeInTheDocument();
  });

  it("keeps oversized axes on the bounded endpoint-preserving preview", () => {
    renderDialog({ zCount: 2401, tCount: 1 });

    fireEvent.click(screen.getByRole("button", { name: /export video/i }));

    expect(screen.getByRole("dialog", { name: "Export video" })).toHaveClass(
      "viewer-video-export-dialog"
    );
    expect(screen.queryByRole("button", { name: /z sweep/i })).not.toBeInTheDocument();
    expect(screen.getByText("240 frames")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /complete/i })).toBeDisabled();
    expect(screen.getByText("240 of 2401")).toBeInTheDocument();
    expect(screen.getByText(/first and last source plane/i)).toBeInTheDocument();
  });
});
