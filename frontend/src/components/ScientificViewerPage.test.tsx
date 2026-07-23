import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";

vi.mock("./UploadViewerSheet", () => ({
  UploadViewerWorkspace: () => <div data-testid="viewer-workspace" />,
}));

import { ScientificViewerPage } from "./ScientificViewerPage";

describe("ScientificViewerPage", () => {
  it("exposes a breadcrumb and a compact Lens page heading", () => {
    render(
      <ScientificViewerPage
        uploadedFiles={[]}
        bisqueLinksByFileId={{}}
        apiClient={{} as ApiClient}
      />
    );

    const breadcrumb = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(within(breadcrumb).getByRole("img", { name: "BisQue Ultra" })).toBeInTheDocument();
    expect(
      within(breadcrumb).getByRole("heading", { level: 1, name: "Lens" })
    ).toBeInTheDocument();
    expect(
      within(breadcrumb).getByRole("listitem", { current: "page" })
    ).toBeInTheDocument();
    expect(screen.getByTestId("viewer-workspace")).toBeInTheDocument();
  });
});
