import { useState } from "react";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { Hdf5DatasetSummary, Hdf5MaterialsDashboardResponse } from "@/types";

import { MaterialsHdf5Dashboard } from "./MaterialsHdf5Dashboard";
import { Hdf5OverlayContainerProvider } from "./Hdf5OverlayContainer";

const { datasetPreviewSpy } = vi.hoisted(() => ({ datasetPreviewSpy: vi.fn() }));

vi.mock("./Hdf5DatasetPreview", () => ({
  Hdf5DatasetPreview: (props: { summary: Hdf5DatasetSummary }) => {
    datasetPreviewSpy(props);
    return <div data-testid="materials-map-preview">{props.summary.dataset_path}</div>;
  },
}));

const selectedSummary = {
  file_id: "file-materials",
  dataset_path: "/Data/FeatureIds",
  dataset_name: "FeatureIds",
  preview_kind: "label_volume",
  semantic_role: "feature_ids",
  units_hint: "label",
} as Hdf5DatasetSummary;

const eulerSummary = {
  ...selectedSummary,
  dataset_path: "/Data/EulerAngles",
  dataset_name: "EulerAngles",
  preview_kind: "vector_volume",
  semantic_role: "euler_angles",
} as Hdf5DatasetSummary;

const dashboard: Hdf5MaterialsDashboardResponse = {
  file_id: "file-materials",
  schema: "dream3d",
  overview: {
    phase_names: ["Ferrite"],
    feature_id_scan_complete: true,
    capabilities: ["maps"],
    recommended_map_dataset_path: selectedSummary.dataset_path,
  },
  maps: [
    {
      title: "Feature IDs",
      description: "Segmented grain labels",
      dataset_path: selectedSummary.dataset_path,
      semantic_role: "feature_ids",
      preview_kind: "label_volume",
    },
    {
      title: "Euler angles",
      description: "Crystal orientation map",
      dataset_path: "/Data/EulerAngles",
      semantic_role: "euler_angles",
      preview_kind: "vector_volume",
    },
  ],
  grain_charts: [],
  orientation_charts: [],
  synthetic_stats: [],
  dataset_links: [],
};

function StatefulMaterialsHarness({
  onSelectedDatasetPathChange,
  onOpenDatasetInExplorer,
  onUseDatasetInChat,
}: {
  onSelectedDatasetPathChange: (datasetPath: string) => void;
  onOpenDatasetInExplorer: (datasetPath: string) => void;
  onUseDatasetInChat: (fileId: string, datasetPaths: string[]) => void;
}) {
  const [selectedDatasetPath, setSelectedDatasetPath] = useState(selectedSummary.dataset_path);
  const summary =
    selectedDatasetPath === eulerSummary.dataset_path ? eulerSummary : selectedSummary;

  return (
    <MaterialsHdf5Dashboard
      apiClient={{} as ApiClient}
      dashboard={dashboard}
      section="maps"
      selectedDatasetPath={selectedDatasetPath}
      onSelectedDatasetPathChange={(datasetPath) => {
        setSelectedDatasetPath(datasetPath);
        onSelectedDatasetPathChange(datasetPath);
      }}
      onOpenDatasetInExplorer={onOpenDatasetInExplorer}
      selectedDatasetSummary={summary}
      onUseDatasetInChat={onUseDatasetInChat}
    />
  );
}

describe("MaterialsHdf5Dashboard map workspace", () => {
  it("owns feature selection above the cold selected-summary loading boundary", () => {
    const commonProps = {
      apiClient: {} as ApiClient,
      dashboard,
      section: "maps" as const,
      selectedDatasetPath: selectedSummary.dataset_path,
      onSelectedDatasetPathChange: vi.fn(),
      onOpenDatasetInExplorer: vi.fn(),
    };
    const { rerender } = render(
      <MaterialsHdf5Dashboard {...commonProps} selectedDatasetSummary={selectedSummary} />
    );
    const firstProps = datasetPreviewSpy.mock.calls[datasetPreviewSpy.mock.calls.length - 1]?.[0] as {
      onFeatureSelectionChange: (selection: unknown) => void;
    };
    act(() =>
      firstProps.onFeatureSelectionChange({
        fileId: dashboard.file_id,
        registrationKey: "registered",
        appliedFeatureIds: ["7", "25"],
        draftFeatureIds: "",
        error: null,
      })
    );
    rerender(<MaterialsHdf5Dashboard {...commonProps} selectedDatasetSummary={null} />);
    expect(screen.getByText("Loading the selected materials map…")).toBeInTheDocument();
    rerender(<MaterialsHdf5Dashboard {...commonProps} selectedDatasetSummary={selectedSummary} />);
    expect(datasetPreviewSpy.mock.calls[datasetPreviewSpy.mock.calls.length - 1]?.[0]).toEqual(
      expect.objectContaining({
        featureSelection: expect.objectContaining({ appliedFeatureIds: ["7", "25"] }),
      })
    );
  });

  it("keeps map rows selectable and the selected-map actions actionable", () => {
    const onSelectedDatasetPathChange = vi.fn();
    const onOpenDatasetInExplorer = vi.fn();
    const onUseDatasetInChat = vi.fn();

    render(
      <StatefulMaterialsHarness
        onSelectedDatasetPathChange={onSelectedDatasetPathChange}
        onOpenDatasetInExplorer={onOpenDatasetInExplorer}
        onUseDatasetInChat={onUseDatasetInChat}
      />
    );

    expect(document.querySelector("[data-hdf5-material-section='maps']")).not.toBeNull();
    const selectedMap = screen.getByRole("region", { name: "Selected materials map" });
    const mapRail = screen.getByRole("complementary", { name: "Available materials maps" });
    expect(selectedMap.compareDocumentPosition(mapRail) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING
    );
    expect(screen.queryByText("Canonical spatial datasets")).not.toBeInTheDocument();
    expect(within(selectedMap).queryByText(/^Feature IDs$/i)).not.toBeInTheDocument();
    expect(within(selectedMap).getByText("Label Volume • label")).toBeInTheDocument();

    expect(within(mapRail).getByRole("button", { name: /Feature IDs/ })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    const eulerMap = screen.getByRole("button", { name: /Euler angles/ });
    expect(eulerMap).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(eulerMap);
    expect(onSelectedDatasetPathChange).toHaveBeenCalledWith("/Data/EulerAngles");
    expect(eulerMap).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /Feature IDs/ })).toHaveAttribute(
      "aria-pressed",
      "false"
    );
    expect(screen.getByTestId("materials-map-preview")).toHaveTextContent("/Data/EulerAngles");

    fireEvent.click(screen.getByRole("button", { name: "Open in Explorer" }));
    expect(onOpenDatasetInExplorer).toHaveBeenCalledWith(eulerSummary.dataset_path);

    fireEvent.click(screen.getByRole("button", { name: "Use in chat" }));
    expect(onUseDatasetInChat).toHaveBeenCalledWith(eulerSummary.file_id, [
      eulerSummary.dataset_path,
    ]);
  });

  it("portals chart Source details into the HDF5 overlay container", async () => {
    const overlayContainer = document.createElement("div");
    document.body.append(overlayContainer);
    const chartDashboard: Hdf5MaterialsDashboardResponse = {
      ...dashboard,
      grain_charts: [
        {
          kind: "histogram",
          title: "Grain size distribution",
          x_key: "diameter",
          y_key: "count",
          data: [{ diameter: 1, count: 2 }],
          source_paths: ["/Data/EquivalentDiameters"],
          provenance: "Bounded sample",
        },
      ],
    };

    const { unmount } = render(
      <Hdf5OverlayContainerProvider container={overlayContainer}>
        <MaterialsHdf5Dashboard
          apiClient={{} as ApiClient}
          dashboard={chartDashboard}
          section="grains"
          selectedDatasetPath={selectedSummary.dataset_path}
          onSelectedDatasetPathChange={vi.fn()}
          onOpenDatasetInExplorer={vi.fn()}
          selectedDatasetSummary={selectedSummary}
        />
      </Hdf5OverlayContainerProvider>
    );

    fireEvent.pointerEnter(screen.getByRole("button", { name: "Source" }));
    const sourceOverlay = await screen.findByText("Source datasets");
    expect(overlayContainer).toContainElement(sourceOverlay);
    expect(sourceOverlay.closest("[data-hdf5-overlay='material-source']")).not.toBeNull();

    unmount();
    overlayContainer.remove();
  });
});
