import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { Hdf5DatasetHistogramResponse, Hdf5DatasetSummary } from "@/types";

import { Hdf5DatasetPreview } from "./Hdf5DatasetPreview";
import { Hdf5OverlayContainerProvider } from "./Hdf5OverlayContainer";

const { volumeCanvasSpy } = vi.hoisted(() => ({
  volumeCanvasSpy: vi.fn(),
}));

vi.mock("../SlicePlaneCanvas", () => ({
  SlicePlaneCanvas: () => <div data-testid="slice-plane-canvas" />,
}));

vi.mock("../SliceStackVolumeCanvas", () => ({
  SliceStackVolumeCanvas: (props: Record<string, unknown>) => {
    volumeCanvasSpy(props);
    return <div data-testid="volume-canvas" />;
  },
}));

const plane = (axis: "z" | "y" | "x") => ({
  axis,
  label: axis === "z" ? "XY plane" : axis === "y" ? "XZ plane" : "YZ plane",
  axes: axis === "z" ? ["Y", "X"] : axis === "y" ? ["Z", "X"] : ["Z", "Y"],
  pixel_size: { width: 96, height: 64 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 96, height: 64 },
  aspect_ratio: 1.5,
});

const volumeSummary: Hdf5DatasetSummary = {
  file_id: "file-hdf5",
  dataset_path: "/Data/FeatureIds",
  dataset_name: "FeatureIds",
  preview_kind: "scalar_volume",
  semantic_role: "feature_ids",
  units_hint: null,
  materials_domain_tags: ["microstructure"],
  dtype: "uint32",
  shape: [12, 64, 96],
  rank: 3,
  element_count: 73_728,
  estimated_bytes: 294_912,
  dimension_summary: { z: 12, y: 64, x: 96 },
  capabilities: ["slice", "volume", "histogram"],
  render_policy: "scalar",
  delivery_mode: "scalar",
  diagnostic_surface: "mpr",
  first_paint_mode: "image",
  measurement_policy: "spacing-aware",
  texture_policy: "nearest",
  display_capabilities: ["slice_navigation", "volume_context"],
  viewer_capabilities: ["scalar_volume_delivery"],
  volume_eligible: true,
  volume_reason: "Eligible for bounded native volume rendering.",
  axis_sizes: { T: 1, C: 1, Z: 12, Y: 64, X: 96 },
  physical_spacing: { z: 1, y: 1, x: 1 },
  atlas_scheme: null,
  attributes: {},
  geometry: { dimensions: [96, 64, 12], spacing: [1, 1, 1], complete: true },
  structured_fields: [],
  component_count: 1,
  component_labels: [],
  slice_axes: ["z", "y", "x"],
  preview_planes: { z: plane("z"), y: plane("y"), x: plane("x") },
  sample_shape: [4, 4, 4],
  sample_values: [0, 1, 2],
  sample_statistics: { sample_count: 64, min: 0, max: 12, mean: 4.5, unique_values: 13 },
};

const categoricalSummary: Hdf5DatasetSummary = {
  ...volumeSummary,
  preview_kind: "label_volume",
  render_policy: "categorical",
  delivery_mode: "atlas",
  texture_policy: "nearest",
  atlas_scheme: {
    slice_count: 12,
    columns: 4,
    rows: 3,
    slice_width: 96,
    slice_height: 64,
    atlas_width: 384,
    atlas_height: 192,
    downsample: 1,
    format: "png",
  },
  feature_filter: {
    supported: true,
    source_dataset_path: "/Data/FeatureIds",
    max_ids: 64,
    background_id: 0,
    provenance: "co_registered_raw_integer_feature_ids",
    registration_key: "/Data/FeatureIds|12x64x96|1x1x1",
    target_role: "feature_ids",
    native_shape: [12, 64, 96],
    preview_shape: [12, 64, 96],
    preview_stride: { z: 1, y: 1, x: 1 },
  },
};

const buildApiClient = () => {
  const getHdf5ScalarVolume = vi.fn();
  const getHdf5DatasetHistogram = vi.fn(
    async (): Promise<Hdf5DatasetHistogramResponse> => ({
      file_id: volumeSummary.file_id,
      dataset_path: volumeSummary.dataset_path,
      preview_kind: volumeSummary.preview_kind,
      sample_count: 64,
      discrete: true,
      min: 0,
      max: 12,
      bins: [],
    })
  );
  return {
    apiClient: {
      hdf5SlicePreviewUrl: vi.fn(() => "/preview.png"),
      hdf5AtlasPreviewUrl: vi.fn(() => "/atlas.png"),
      getHdf5ScalarVolume,
      getHdf5DatasetHistogram,
    } as unknown as ApiClient,
    getHdf5DatasetHistogram,
    getHdf5ScalarVolume,
  };
};

const openFeatureFilter = () => {
  const trigger = screen.getByRole("button", { name: /^Filter grains/ });
  if (trigger.getAttribute("aria-expanded") !== "true") {
    fireEvent.click(trigger);
  }
  return trigger;
};

describe("Hdf5DatasetPreview first paint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("opens a volume-capable dataset on Slice and exposes accessible slice controls", () => {
    const { apiClient, getHdf5ScalarVolume } = buildApiClient();

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={volumeSummary} />);

    expect(screen.getByRole("tab", { name: "Slice" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("slice-plane-canvas")).toBeInTheDocument();
    expect(screen.queryByTestId("volume-canvas")).not.toBeInTheDocument();
    expect(getHdf5ScalarVolume).not.toHaveBeenCalled();
    expect(screen.getByRole("group", { name: "Slice orientation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "XY" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("slider", { name: "XY plane slice" })).toHaveAttribute(
      "aria-valuetext",
      "Slice 7 of 12"
    );
  });

  it("forwards the renderer generation signal into HDF scalar delivery", async () => {
    const { apiClient, getHdf5ScalarVolume } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={volumeSummary} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });
    const lastCanvasCall = volumeCanvasSpy.mock.calls[volumeCanvasSpy.mock.calls.length - 1];
    const source = lastCanvasCall?.[0]?.volumeSource as {
      loadScalarVolume: (signal: AbortSignal) => Promise<unknown>;
    };
    const controller = new AbortController();

    await source.loadScalarVolume(controller.signal);

    expect(getHdf5ScalarVolume).toHaveBeenCalledWith(volumeSummary.file_id, {
      datasetPath: volumeSummary.dataset_path,
      signal: controller.signal,
    });
  });

  it("passes preview axis sizes and calibrated physical spacing to the volume canvas unchanged", () => {
    const { apiClient } = buildApiClient();
    const axisSizes = { T: 1, C: 1, Z: 4, Y: 5, X: 4 };
    const physicalSpacing = { x: 0.6475, y: 1.098, z: 3.1075 };
    const previewPlane = {
      ...plane("z"),
      pixel_size: { width: 4, height: 5 },
      spacing: { row: physicalSpacing.y, col: physicalSpacing.x },
      world_size: {
        width: 4 * physicalSpacing.x,
        height: 5 * physicalSpacing.y,
      },
      aspect_ratio: (4 * physicalSpacing.x) / (5 * physicalSpacing.y),
    };
    const summary = {
      ...volumeSummary,
      axis_sizes: axisSizes,
      physical_spacing: physicalSpacing,
      preview_planes: { ...volumeSummary.preview_planes, z: previewPlane },
    };

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={summary} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });

    const lastCanvasCall = volumeCanvasSpy.mock.calls[volumeCanvasSpy.mock.calls.length - 1];
    expect(lastCanvasCall?.[0]?.volumeSource.axisSizes).toBe(axisSizes);
    expect(lastCanvasCall?.[0]?.volumeSource.physicalSpacing).toBe(physicalSpacing);
    expect(lastCanvasCall?.[0]?.volumeSource.plane).toBe(previewPlane);
    expect(previewPlane.pixel_size).toEqual({ width: axisSizes.X, height: axisSizes.Y });
    expect(previewPlane.spacing).toEqual({
      row: physicalSpacing.y,
      col: physicalSpacing.x,
    });
    expect(previewPlane.aspect_ratio).toBeCloseTo(
      previewPlane.world_size.width / previewPlane.world_size.height
    );
    expect(previewPlane.world_size.width).toBeCloseTo(0.37 * 7);
    expect(previewPlane.world_size.height).toBeCloseTo(0.61 * 9);
    expect(axisSizes.Z * physicalSpacing.z).toBeCloseTo(1.13 * 11);
  });

  it("does not request a histogram until Distribution is active", async () => {
    const { apiClient, getHdf5DatasetHistogram } = buildApiClient();

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={volumeSummary} />);

    expect(getHdf5DatasetHistogram).not.toHaveBeenCalled();
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Distribution" }), {
      button: 0,
      ctrlKey: false,
    });

    await waitFor(() => {
      expect(getHdf5DatasetHistogram).toHaveBeenCalledTimes(1);
    });
    expect(getHdf5DatasetHistogram).toHaveBeenCalledWith(
      volumeSummary.file_id,
      volumeSummary.dataset_path,
      { bins: 24, component: undefined }
    );
  });

  it("opens categorical volumes as a crisp surface and wires X-ray plus preview-grid cutaway depth", () => {
    const { apiClient } = buildApiClient();

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });

    const renderControls = screen.getByRole("group", { name: "Categorical volume rendering" });
    expect(renderControls).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Surface" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "X-ray" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Cutaway" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("slider", { name: "Cutaway Z depth" })).not.toBeInTheDocument();
    expect(screen.queryByRole("note")).not.toBeInTheDocument();

    expect(volumeCanvasSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        categoricalMode: "surface",
        volumeCutaway: false,
        zIndex: 5,
        volumeSource: expect.objectContaining({
          kind: "atlas",
          atlasUrl: "/atlas.png",
          renderPolicy: "categorical",
          texturePolicy: "nearest",
        }),
      })
    );

    fireEvent.click(screen.getByRole("button", { name: "X-ray" }));
    expect(screen.getByRole("button", { name: "X-ray" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("note")).toHaveTextContent(
      "X-ray blends labels for depth context; blended colors do not represent feature IDs."
    );
    expect(volumeCanvasSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ categoricalMode: "xray", volumeCutaway: false, zIndex: 5 })
    );

    fireEvent.click(screen.getByRole("button", { name: "Cutaway" }));
    const depth = screen.getByRole("slider", { name: "Cutaway Z depth" });
    expect(depth).toHaveAttribute("min", "0");
    expect(depth).toHaveAttribute("max", "11");
    expect(depth).toHaveAttribute("aria-valuetext", "Preview-grid Z 6 of 12");
    fireEvent.change(depth, { target: { value: "8" } });
    expect(volumeCanvasSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ categoricalMode: "xray", volumeCutaway: true, zIndex: 8 })
    );
  });

  it("keeps the supported grain filter closed by default and exposes it as an accessible disclosure", () => {
    const { apiClient } = buildApiClient();

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);

    const trigger = screen.getByRole("button", { name: "Filter grains" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveAttribute("aria-controls");
    expect(screen.queryByRole("textbox", { name: "Feature IDs" })).not.toBeInTheDocument();
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("textbox", { name: "Feature IDs" })).toHaveAccessibleDescription(
      "Raw Feature IDs; background 0 excluded."
    );
  });

  it("keeps one compact grain disclosure mounted with its state across preview tabs", () => {
    const { apiClient } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} compactLayout />);
    const trigger = openFeatureFilter();
    const input = screen.getByRole("textbox", { name: "Feature IDs" });
    fireEvent.change(input, { target: { value: "25" } });
    trigger.focus();

    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });

    expect(screen.getByRole("button", { name: "Filter grains" })).toBe(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("textbox", { name: "Feature IDs" })).toHaveValue("25");
    expect(trigger).toHaveFocus();

    fireEvent.mouseDown(screen.getByRole("tab", { name: "Distribution" }), {
      button: 0,
      ctrlKey: false,
    });
    expect(screen.getByRole("button", { name: "Filter grains" })).toBe(trigger);
    expect(trigger).toHaveFocus();
  });

  it("validates manual IDs, filters every preview URL, and preserves selection across compatible maps", () => {
    const { apiClient } = buildApiClient();
    const { rerender } = render(
      <Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />
    );

    openFeatureFilter();
    fireEvent.change(screen.getByRole("textbox", { name: "Feature IDs" }), {
      target: { value: "25, 7, 25" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByRole("button", { name: /ID 7/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ID 25/ })).toBeInTheDocument();
    expect(apiClient.hdf5SlicePreviewUrl).toHaveBeenLastCalledWith(
      categoricalSummary.file_id,
      expect.objectContaining({ featureIds: ["7", "25"] })
    );
    expect(apiClient.hdf5AtlasPreviewUrl).toHaveBeenLastCalledWith(
      categoricalSummary.file_id,
      expect.objectContaining({ featureIds: ["7", "25"] })
    );

    const compatibleEuler: Hdf5DatasetSummary = {
      ...categoricalSummary,
      dataset_path: "/Data/EulerAngles",
      dataset_name: "EulerAngles",
      semantic_role: "euler_angles",
      feature_filter: { ...categoricalSummary.feature_filter!, target_role: "euler_angles" },
    };
    rerender(<Hdf5DatasetPreview apiClient={apiClient} summary={compatibleEuler} />);
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    openFeatureFilter();
    expect(screen.getByRole("button", { name: /ID 7/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ID 25/ })).toBeInTheDocument();
    expect(apiClient.hdf5AtlasPreviewUrl).toHaveBeenLastCalledWith(
      compatibleEuler.file_id,
      expect.objectContaining({ datasetPath: compatibleEuler.dataset_path, featureIds: ["7", "25"] })
    );

    const compatibleIpf: Hdf5DatasetSummary = {
      ...categoricalSummary,
      dataset_path: "/Data/IPFColor",
      dataset_name: "IPFColor",
      semantic_role: "ipf_colors",
      feature_filter: { ...categoricalSummary.feature_filter!, target_role: "ipf_colors" },
    };
    rerender(<Hdf5DatasetPreview apiClient={apiClient} summary={compatibleIpf} />);
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    openFeatureFilter();
    expect(screen.getByRole("button", { name: /ID 7/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ID 25/ })).toBeInTheDocument();
    expect(apiClient.hdf5AtlasPreviewUrl).toHaveBeenLastCalledWith(
      compatibleIpf.file_id,
      expect.objectContaining({ datasetPath: compatibleIpf.dataset_path, featureIds: ["7", "25"] })
    );
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });
    expect(screen.queryByRole("button", { name: "Pick grain" })).not.toBeInTheDocument();
    const canvasProps = volumeCanvasSpy.mock.calls[volumeCanvasSpy.mock.calls.length - 1]?.[0] as Record<string, unknown>;
    expect(canvasProps).toMatchObject({ featureMask: true });
    expect(canvasProps).not.toHaveProperty("featureIdsVolume");
    expect(canvasProps).not.toHaveProperty("pickFeatureActive");
    expect(canvasProps).not.toHaveProperty("onPickFeature");
    expect(canvasProps).not.toHaveProperty("onPickFeatureMiss");
  });

  it("keeps draft separate, unions successive manual applies, and treats a canonical no-op as inert", () => {
    const { apiClient } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);
    openFeatureFilter();
    const input = screen.getByRole("textbox", { name: "Feature IDs" });

    fireEvent.change(input, { target: { value: "25, 7" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByRole("button", { name: "Remove Feature ID 7" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove Feature ID 25" })).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByRole("button", { name: "Remove Feature ID 7" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove Feature ID 25" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove Feature ID 9" })).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "09,9" } });
    vi.mocked(apiClient.hdf5AtlasPreviewUrl).mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(apiClient.hdf5AtlasPreviewUrl).not.toHaveBeenCalled();
    expect(input).toHaveValue("");
  });

  it("keeps the active count and Clear discoverable while collapsed and preserves draft and applied IDs", () => {
    const { apiClient } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);
    const trigger = openFeatureFilter();
    const input = screen.getByRole("textbox", { name: "Feature IDs" });

    fireEvent.change(input, { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    fireEvent.change(input, { target: { value: "25" } });
    fireEvent.click(trigger);

    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getByText("1 selected")).toHaveAttribute("aria-live", "polite");
    expect(trigger).toHaveAccessibleName("Filter grains, 1 selected");
    expect(screen.getByRole("button", { name: "Clear grain filter" })).toHaveTextContent("Clear");
    expect(screen.queryByRole("textbox", { name: "Feature IDs" })).not.toBeInTheDocument();

    fireEvent.click(trigger);
    expect(screen.getByRole("textbox", { name: "Feature IDs" })).toHaveValue("25");
    expect(screen.getByRole("button", { name: "Remove Feature ID 7" })).toBeInTheDocument();

    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole("button", { name: "Clear grain filter" }));
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument();
    expect(trigger).toHaveAccessibleName("Filter grains");
    expect(screen.queryByRole("button", { name: "Clear grain filter" })).not.toBeInTheDocument();
  });

  it("keeps every allowed Feature ID chip keyboard reachable inside the bounded list", () => {
    const { apiClient } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);
    const trigger = openFeatureFilter();
    const featureIds = Array.from({ length: 64 }, (_, index) => String(index + 1));

    fireEvent.change(screen.getByRole("textbox", { name: "Feature IDs" }), {
      target: { value: featureIds.join(",") },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    expect(trigger).toHaveAccessibleName("Filter grains, 64 selected");
    expect(screen.getAllByRole("button", { name: /Remove Feature ID/ })).toHaveLength(64);
    const lastChip = screen.getByRole("button", { name: "Remove Feature ID 64" });
    lastChip.focus();
    expect(lastChip).toHaveFocus();
    expect(lastChip.closest("[aria-label='Selected Feature IDs']")).not.toBeNull();
  });

  it("does not rebuild preview URLs or replace the volume canvas source when the disclosure toggles", () => {
    const { apiClient } = buildApiClient();
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={categoricalSummary} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });
    const canvas = screen.getByTestId("volume-canvas");
    const initialCanvasProps = volumeCanvasSpy.mock.calls[volumeCanvasSpy.mock.calls.length - 1]?.[0] as Record<
      string,
      unknown
    >;
    const initialVolumeSource = initialCanvasProps.volumeSource;
    vi.mocked(apiClient.hdf5SlicePreviewUrl).mockClear();
    vi.mocked(apiClient.hdf5AtlasPreviewUrl).mockClear();

    const trigger = screen.getByRole("button", { name: "Filter grains" });
    fireEvent.click(trigger);
    fireEvent.click(trigger);

    expect(apiClient.hdf5SlicePreviewUrl).not.toHaveBeenCalled();
    expect(apiClient.hdf5AtlasPreviewUrl).not.toHaveBeenCalled();
    expect(screen.getByTestId("volume-canvas")).toBe(canvas);
    const finalCanvasProps = volumeCanvasSpy.mock.calls[volumeCanvasSpy.mock.calls.length - 1]?.[0] as Record<
      string,
      unknown
    >;
    expect(finalCanvasProps.volumeSource).toBe(initialVolumeSource);
  });

  it("uses the advertised maximum and exposes validation accessibly", () => {
    const { apiClient } = buildApiClient();
    const summary = {
      ...categoricalSummary,
      feature_filter: { ...categoricalSummary.feature_filter!, max_ids: 2 },
    };
    render(<Hdf5DatasetPreview apiClient={apiClient} summary={summary} />);
    const trigger = openFeatureFilter();
    const input = screen.getByRole("textbox", { name: "Feature IDs" });
    fireEvent.change(input, { target: { value: "1,2,3" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(input).toHaveAttribute("aria-invalid", "true");
    const alert = screen.getByRole("alert");
    expect(input.getAttribute("aria-describedby")?.split(" ")).toContain(alert.id);
    expect(alert).toHaveTextContent("Select at most 2 unique Feature IDs.");
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
  });

  it("retains a compatible applied selection through loading and unsupported summaries without leaking filter URLs", () => {
    const { apiClient } = buildApiClient();
    const unsupported = { ...categoricalSummary, feature_filter: undefined };
    const compatibleIpf = {
      ...categoricalSummary,
      dataset_path: "/Data/IPFColor",
      semantic_role: "ipf_colors",
      feature_filter: { ...categoricalSummary.feature_filter!, target_role: "ipf_colors" },
    } as Hdf5DatasetSummary;
    function Harness({ summary }: { summary: Hdf5DatasetSummary | null }) {
      const [selection, setSelection] = useState<Parameters<typeof Hdf5DatasetPreview>[0]["featureSelection"]>(null);
      return summary ? (
        <Hdf5DatasetPreview
          apiClient={apiClient}
          summary={summary}
          featureSelection={selection}
          onFeatureSelectionChange={setSelection}
        />
      ) : <div>Loading selected map</div>;
    }
    const { rerender } = render(<Harness summary={categoricalSummary} />);
    openFeatureFilter();
    fireEvent.change(screen.getByRole("textbox", { name: "Feature IDs" }), { target: { value: "7,25" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    rerender(<Harness summary={null} />);
    expect(screen.getByText("Loading selected map")).toBeInTheDocument();
    vi.mocked(apiClient.hdf5AtlasPreviewUrl).mockClear();
    vi.mocked(apiClient.hdf5SlicePreviewUrl).mockClear();
    rerender(<Harness summary={unsupported} />);
    expect(apiClient.hdf5AtlasPreviewUrl).toHaveBeenLastCalledWith(
      unsupported.file_id,
      expect.objectContaining({ featureIds: [] })
    );
    expect(apiClient.hdf5SlicePreviewUrl).toHaveBeenLastCalledWith(
      unsupported.file_id,
      expect.objectContaining({ featureIds: [] })
    );
    rerender(<Harness summary={compatibleIpf} />);
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    openFeatureFilter();
    expect(screen.getByRole("button", { name: "Remove Feature ID 7" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove Feature ID 25" })).toBeInTheDocument();
  });

  it("does not offer the grain disclosure for an unsupported summary", () => {
    const { apiClient } = buildApiClient();
    render(
      <Hdf5DatasetPreview
        apiClient={apiClient}
        summary={{ ...categoricalSummary, feature_filter: undefined }}
      />
    );

    expect(screen.queryByRole("button", { name: "Filter grains" })).not.toBeInTheDocument();
    expect(screen.queryByText(/selected$/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Pick grain" })).not.toBeInTheDocument();
  });

  it("does not show categorical rendering controls for scalar volumes", () => {
    const { apiClient } = buildApiClient();

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={volumeSummary} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Volume" }), {
      button: 0,
      ctrlKey: false,
    });

    expect(screen.queryByRole("group", { name: "Categorical volume rendering" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Surface" })).not.toBeInTheDocument();
  });

  it("reuses an in-flight histogram request when Distribution is reopened", async () => {
    let resolveHistogram!: (value: Awaited<ReturnType<ApiClient["getHdf5DatasetHistogram"]>>) => void;
    const histogramRequest = new Promise<
      Awaited<ReturnType<ApiClient["getHdf5DatasetHistogram"]>>
    >((resolve) => {
      resolveHistogram = resolve;
    });
    const { apiClient, getHdf5DatasetHistogram } = buildApiClient();
    getHdf5DatasetHistogram.mockReturnValue(histogramRequest);

    render(<Hdf5DatasetPreview apiClient={apiClient} summary={volumeSummary} />);
    const activateTab = (name: string) => {
      fireEvent.mouseDown(screen.getByRole("tab", { name }), { button: 0, ctrlKey: false });
    };

    activateTab("Distribution");
    await waitFor(() => expect(getHdf5DatasetHistogram).toHaveBeenCalledTimes(1));
    activateTab("Slice");
    activateTab("Distribution");
    expect(getHdf5DatasetHistogram).toHaveBeenCalledTimes(1);

    resolveHistogram({
      file_id: volumeSummary.file_id,
      dataset_path: volumeSummary.dataset_path,
      preview_kind: volumeSummary.preview_kind,
      sample_count: 64,
      discrete: true,
      min: 0,
      max: 12,
      bins: [],
    });
    await waitFor(() => expect(screen.getByText("No histogram data available for this dataset.")).toBeInTheDocument());
  });

  it("portals the component picker into the HDF5 overlay container", async () => {
    const overlayContainer = document.createElement("div");
    document.body.append(overlayContainer);
    const vectorSummary: Hdf5DatasetSummary = {
      ...volumeSummary,
      preview_kind: "vector_volume",
      component_count: 2,
      component_labels: ["x", "y"],
    };
    const { apiClient } = buildApiClient();

    render(
      <Hdf5OverlayContainerProvider container={overlayContainer}>
        <Hdf5DatasetPreview apiClient={apiClient} summary={vectorSummary} />
      </Hdf5OverlayContainerProvider>
    );
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "ArrowDown" });

    const option = await screen.findByRole("option", { name: "y" });
    expect(overlayContainer).toContainElement(option);
    overlayContainer.remove();
  });
});
