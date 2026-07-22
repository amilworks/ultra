import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { Hdf5DatasetSummary, UploadViewerInfo } from "@/types";

import { Hdf5ViewerShell } from "./Hdf5Overview";

vi.mock("./Hdf5Navigator", () => ({
  Hdf5Navigator: () => <div data-testid="hdf5-navigator" />,
}));

vi.mock("./Hdf5Inspector", () => ({
  Hdf5Inspector: () => <div data-testid="hdf5-inspector" />,
}));

vi.mock("./MaterialsHdf5Dashboard", () => ({
  MaterialsHdf5Dashboard: () => <div data-testid="materials-dashboard" />,
}));

vi.mock("./PhaseMetadataSummary", () => ({
  PhaseMetadataSummary: () => null,
}));

const datasetPath = "/Data/FeatureIds";
const datasetSummary = {
  file_id: "file-hdf5",
  dataset_path: datasetPath,
  dataset_name: "FeatureIds",
  preview_kind: "label_volume",
} as Hdf5DatasetSummary;

const viewerInfo = {
  kind: "hdf5",
  file_id: "file-hdf5",
  original_name: "microstructure.dream3d",
  hdf5: {
    enabled: true,
    supported: true,
    error: null,
    default_dataset_path: datasetPath,
    tree: [
      {
        node_type: "dataset",
        name: "FeatureIds",
        path: datasetPath,
        children: [],
      },
    ],
    summary: {
      group_count: 1,
      dataset_count: 1,
      truncated: false,
      dataset_kinds: { label_volume: 1 },
      geometry: { dimensions: [96, 64, 12] },
    },
    root_attributes: {},
    limitations: [],
    materials: {
      detected: true,
      schema: "dream3d",
      capabilities: [],
      phase_names: [],
      feature_id_scan_complete: true,
      recommended_view: "explorer",
    },
  },
} as unknown as UploadViewerInfo;

describe("Hdf5ViewerShell fullscreen", () => {
  it("owns native fullscreen focus and restores the trigger after native exit", () => {
    let fullscreenElement: Element | null = null;
    const fullscreenElementDescriptor = Object.getOwnPropertyDescriptor(
      document,
      "fullscreenElement"
    );
    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => fullscreenElement,
    });

    try {
      const apiClient = { getHdf5MaterialsDashboard: vi.fn() } as unknown as ApiClient;
      const { container } = render(
        <Hdf5ViewerShell
          viewerInfo={viewerInfo}
          apiClient={apiClient}
          selectedDatasetPath={datasetPath}
          onSelectedDatasetPathChange={vi.fn()}
          selectedDatasetSummary={datasetSummary}
          cacheDatasetSummary={vi.fn()}
        />
      );
      const root = container.querySelector<HTMLElement>("[data-hdf5-workspace-root='true']")!;
      const trigger = screen.getByRole("button", { name: "Enter HDF5 fullscreen" });
      Object.defineProperty(root, "requestFullscreen", {
        configurable: true,
        value: vi.fn(async () => undefined),
      });

      trigger.focus();
      fireEvent.click(trigger);
      fullscreenElement = root;
      fireEvent(document, new Event("fullscreenchange"));
      expect(root).toHaveAttribute("data-hdf5-fullscreen", "true");
      expect(root).toHaveFocus();

      fullscreenElement = null;
      fireEvent(document, new Event("fullscreenchange"));
      expect(root).toHaveAttribute("data-hdf5-fullscreen", "false");
      expect(trigger).toHaveFocus();
    } finally {
      if (fullscreenElementDescriptor) {
        Object.defineProperty(document, "fullscreenElement", fullscreenElementDescriptor);
      } else {
        Reflect.deleteProperty(document, "fullscreenElement");
      }
    }
  });

  it("uses CSS fullscreen when the native API is unavailable and exits on Escape", () => {
    const getHdf5MaterialsDashboard = vi.fn();
    const apiClient = { getHdf5MaterialsDashboard } as unknown as ApiClient;
    const { container } = render(
      <Hdf5ViewerShell
        viewerInfo={viewerInfo}
        apiClient={apiClient}
        selectedDatasetPath={datasetPath}
        onSelectedDatasetPathChange={vi.fn()}
        selectedDatasetSummary={datasetSummary}
        cacheDatasetSummary={vi.fn()}
      />
    );

    const root = container.querySelector<HTMLElement>("[data-hdf5-workspace-root='true']");
    expect(root).toHaveAttribute("data-hdf5-fullscreen", "false");
    expect(getHdf5MaterialsDashboard).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Enter HDF5 fullscreen" }));
    expect(root).toHaveAttribute("data-hdf5-fullscreen", "true");
    expect(screen.getByRole("button", { name: "Exit HDF5 fullscreen" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );

    fireEvent.keyDown(window, { key: "Escape" });
    expect(root).toHaveAttribute("data-hdf5-fullscreen", "false");
  });
});
