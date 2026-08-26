import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { UploadedFileRecord } from "../types";

vi.mock("./UploadViewerSheet", () => ({
  UploadViewerWorkspace: () => <div data-testid="viewer-workspace" />,
}));

import { ScientificViewerPage } from "./ScientificViewerPage";

const file = (id: string): UploadedFileRecord => ({
  file_id: id,
  original_name: `${id}.tif`,
  content_type: "image/tiff",
  size_bytes: 1,
  sha256: "0",
  created_at: "2026-01-01T00:00:00Z",
});

// jsdom starts every test with history.length === 1 (a fresh tab). Stubbing the
// instance property shadows the prototype getter; deleting it restores the getter.
const stubHistoryLength = (length: number): void => {
  Object.defineProperty(window.history, "length", { configurable: true, value: length });
};

const renderPage = (props: Partial<React.ComponentProps<typeof ScientificViewerPage>> = {}) =>
  render(
    <ScientificViewerPage
      uploadedFiles={[]}
      bisqueLinksByFileId={{}}
      apiClient={{} as ApiClient}
      {...props}
    />
  );

describe("ScientificViewerPage", () => {
  afterEach(() => {
    Reflect.deleteProperty(window.history, "length");
  });

  it("exposes a breadcrumb and a compact Lens page heading", () => {
    renderPage();

    const breadcrumb = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(within(breadcrumb).getByRole("img", { name: "BisQue Ultra" })).toBeInTheDocument();
    expect(
      within(breadcrumb).getByRole("heading", { level: 1, name: "Lens" })
    ).toBeInTheDocument();
    expect(
      within(breadcrumb).getByRole("listitem", { current: "page" })
    ).toBeInTheDocument();
    expect(screen.getByTestId("viewer-workspace")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  describe("empty-state notice", () => {
    it("renders the unavailable variant with Go back when there is history to return to", () => {
      stubHistoryLength(3);
      const onOpenResources = vi.fn();
      const back = vi.spyOn(window.history, "back").mockImplementation(() => undefined);
      renderPage({ unavailableFileIds: ["missing-1"], onOpenResources });

      expect(
        screen.getByRole("heading", { level: 2, name: "This resource isn't available" })
      ).toBeInTheDocument();
      expect(
        screen.getByText("It may have been removed, or it isn't shared with you.")
      ).toBeInTheDocument();
      expect(screen.queryByTestId("viewer-workspace")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      // Unavailable is final: a Retry would only re-ask a question already answered.
      expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Open chat" })).not.toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Open Resources" }));
      expect(onOpenResources).toHaveBeenCalledTimes(1);
      fireEvent.click(screen.getByRole("button", { name: "Go back" }));
      expect(back).toHaveBeenCalledTimes(1);
      back.mockRestore();
    });

    it("renders the failed variant with a Retry that re-asks for the files", () => {
      stubHistoryLength(2);
      const onRetry = vi.fn();
      renderPage({ failedFileIds: ["flaky-1"], onRetry });

      expect(
        screen.getByRole("heading", { level: 2, name: "This resource couldn't be loaded" })
      ).toBeInTheDocument();
      expect(screen.getByText("Check your connection and try again.")).toBeInTheDocument();
      expect(screen.queryByTestId("viewer-workspace")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Open Resources" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Go back" })).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it("prefers the failed variant when nothing opened and both kinds of miss occurred", () => {
      renderPage({ unavailableFileIds: ["missing-1"], failedFileIds: ["flaky-1"] });
      expect(
        screen.getByRole("heading", { level: 2, name: "This resource couldn't be loaded" })
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    });

    it("offers Open chat instead of Go back in a fresh tab with no history", () => {
      expect(window.history.length).toBe(1);
      const onOpenChat = vi.fn();
      renderPage({ unavailableFileIds: ["missing-1"], onOpenChat });

      expect(screen.queryByRole("button", { name: "Go back" })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
      expect(onOpenChat).toHaveBeenCalledTimes(1);
    });
  });

  describe("partial availability", () => {
    it("keeps the workspace and names the unavailable share in one quiet line", () => {
      renderPage({ uploadedFiles: [file("ok-1")], unavailableFileIds: ["missing-1"] });

      expect(screen.getByTestId("viewer-workspace")).toBeInTheDocument();
      expect(
        screen.queryByRole("heading", { name: "This resource isn't available" })
      ).not.toBeInTheDocument();
      const status = screen.getByRole("status");
      expect(status).toHaveTextContent("1 of 2 files isn't available");
      expect(within(status).queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("pluralizes when more than one file is unavailable", () => {
      renderPage({
        uploadedFiles: [file("ok-1")],
        unavailableFileIds: ["missing-1", "missing-2"],
      });
      expect(screen.getByRole("status")).toHaveTextContent("2 of 3 files aren't available");
    });

    it("names the failed share with a Retry that calls onRetry", () => {
      const onRetry = vi.fn();
      renderPage({
        uploadedFiles: [file("ok-1"), file("ok-2")],
        failedFileIds: ["flaky-1"],
        onRetry,
      });

      expect(screen.getByTestId("viewer-workspace")).toBeInTheDocument();
      const status = screen.getByRole("status");
      expect(status).toHaveTextContent("1 of 3 files couldn't be loaded");
      fireEvent.click(within(status).getByRole("button", { name: "Retry" }));
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it("reports both shares against the same total when both occurred", () => {
      renderPage({
        uploadedFiles: [file("ok-1")],
        unavailableFileIds: ["missing-1"],
        failedFileIds: ["flaky-1", "flaky-2"],
      });
      const status = screen.getByRole("status");
      expect(status).toHaveTextContent("1 of 4 files isn't available");
      expect(status).toHaveTextContent("2 of 4 files couldn't be loaded");
      expect(within(status).getByRole("button", { name: "Retry" })).toBeInTheDocument();
    });
  });

  it("keeps the workspace when nothing is unavailable and nothing is open", () => {
    renderPage();
    expect(screen.getByTestId("viewer-workspace")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "This resource isn't available" })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
