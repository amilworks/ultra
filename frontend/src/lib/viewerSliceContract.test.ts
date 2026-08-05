import { describe, expect, it } from "vitest";

import { ApiClient } from "./api";
import { normalizeUploadViewerInfo } from "./viewerManifest";
import { sliceAxisCoordinates, sliceChannelSelection } from "./viewerSliceContract";

describe("strict scalar slice request shaping", () => {
  const strictViewer = normalizeUploadViewerInfo({
    kind: "image",
    file_id: "file-nifti",
    original_name: "brain.nii",
    axis_sizes: { T: 2, C: 2, Z: 8, Y: 16, X: 24 },
    selected_indices: { T: 1, C: 1, Z: 3 },
    viewer: {
      display_capabilities: ["strict_scalar_slice", "channel_visibility"],
      volume_mode: "scalar",
      delivery_mode: "scalar",
    },
  });

  it.each([
    ["direct 2D", "z", { x: 7, y: 6, z: 5 }],
    ["MPR", "y", { x: 7, y: 6, z: 5 }],
    ["volume fallback", "z", { x: 7, y: 6, z: 5 }],
  ] as const)("builds a production ApiClient URL for %s", (_surface, axis, indices) => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const url = client.uploadSliceUrl(strictViewer.file_id, {
      axis,
      ...sliceAxisCoordinates(strictViewer, axis, indices),
      ...sliceChannelSelection(strictViewer, [1], ["#0000ff", "#ff0000"], 1),
      t: 1,
    });
    const params = new URL(url).searchParams;

    expect(params.get(axis)).toBe(String(indices[axis]));
    for (const otherAxis of ["x", "y", "z"].filter((candidate) => candidate !== axis)) {
      expect(params.has(otherAxis)).toBe(false);
    }
    expect(params.get("c")).toBe("1");
    expect(params.has("channels")).toBe(false);
    expect(params.has("channel_colors")).toBe(false);
  });

  it("sends channel colors only when the viewer advertises that capability", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const capable = normalizeUploadViewerInfo({
      ...strictViewer,
      viewer: {
        ...strictViewer.viewer,
        display_capabilities: ["channel_color", "channel_lut_transport"],
      },
    });
    const incapable = normalizeUploadViewerInfo({
      ...capable,
      viewer: { ...capable.viewer, display_capabilities: [] },
    });
    const editableOnly = normalizeUploadViewerInfo({
      ...capable,
      viewer: { ...capable.viewer, display_capabilities: ["channel_color"] },
    });
    const build = (viewer: typeof capable) =>
      new URL(
        client.uploadSliceUrl(viewer.file_id, {
          axis: "z",
          z: 0,
          ...sliceChannelSelection(viewer, [1], ["#0000ff", "#ff0000"]),
        })
      ).searchParams;

    expect(build(capable).get("channel_colors")).toBe("#ff0000");
    expect(build(incapable).has("channel_colors")).toBe(false);
    expect(build(editableOnly).has("channel_colors")).toBe(false);
  });
});
