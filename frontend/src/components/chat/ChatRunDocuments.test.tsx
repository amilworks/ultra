import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatRunDocuments, type ChatRunDocument } from "./ChatRunDocuments";

const reportDoc: ChatRunDocument = {
  path: "outputs/toeplitz_report.md",
  title: "Toeplitz report",
  downloadUrl: "/v2/artifacts/report-1/download",
  kind: "report",
  mimeType: "text/markdown",
  sizeBytes: 16606,
};

const codeDoc: ChatRunDocument = {
  path: "outputs/toeplitz_plots.py",
  title: "toeplitz_plots.py",
  downloadUrl: "/v2/artifacts/code-1/download",
  kind: "code",
  mimeType: "text/x-python",
  sizeBytes: 20769,
};

describe("ChatRunDocuments", () => {
  it("renders nothing when there are no documents", () => {
    const { container } = render(
      <ChatRunDocuments documents={[]} loadDocumentText={vi.fn()} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("reads the markdown report inline and exposes a download link", async () => {
    const loadDocumentText = vi
      .fn()
      .mockResolvedValue("# Toeplitz\n\nThe FFT embedding trick reduces cost.");

    render(<ChatRunDocuments documents={[reportDoc]} loadDocumentText={loadDocumentText} />);

    expect(loadDocumentText).toHaveBeenCalledWith("/v2/artifacts/report-1/download");
    await waitFor(() =>
      expect(screen.getByText(/FFT embedding trick reduces cost/)).toBeInTheDocument()
    );

    const downloadLink = screen.getByLabelText("Download Toeplitz report");
    expect(downloadLink).toHaveAttribute("href", "/v2/artifacts/report-1/download");
    expect(downloadLink).toHaveAttribute("download");
  });

  it("rewrites the report's own figure references to served artifact URLs", async () => {
    const loadDocumentText = vi
      .fn()
      .mockResolvedValue("![Figure 1](outputs/fig1.png)");

    const { container } = render(
      <ChatRunDocuments
        documents={[reportDoc]}
        imageArtifacts={[
          {
            path: "outputs/fig1.png",
            url: "/v2/artifacts/fig-1/download",
            downloadUrl: "/v2/artifacts/fig-1/download",
          },
        ]}
        loadDocumentText={loadDocumentText}
      />
    );

    await waitFor(() => {
      const image = container.querySelector("img");
      expect(image).not.toBeNull();
      expect(image?.getAttribute("src")).toBe("/v2/artifacts/fig-1/download");
    });
  });

  it("renders non-report outputs as download chips without fetching them", () => {
    const loadDocumentText = vi.fn();
    render(<ChatRunDocuments documents={[codeDoc]} loadDocumentText={loadDocumentText} />);

    const chip = screen.getByText("toeplitz_plots.py").closest("a");
    expect(chip).toHaveAttribute("href", "/v2/artifacts/code-1/download");
    expect(chip).toHaveAttribute("download");
    expect(loadDocumentText).not.toHaveBeenCalled();
  });

  it("falls back to a download prompt when the report fails to load", async () => {
    const loadDocumentText = vi.fn().mockRejectedValue(new Error("network down"));
    render(<ChatRunDocuments documents={[reportDoc]} loadDocumentText={loadDocumentText} />);

    await waitFor(() =>
      expect(screen.getByText(/download instead/)).toBeInTheDocument()
    );
  });
});
