import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatRunDocuments, type ChatRunDocument } from "./ChatRunDocuments";
import { runReportPathKey } from "@/features/chat/run-artifact-hydration";

const reportDoc: ChatRunDocument = {
  path: "outputs/toeplitz_report.md",
  title: "Toeplitz report",
  downloadUrl: "/v2/artifacts/report-1/download",
  kind: "report",
  mimeType: "text/markdown",
  sizeBytes: 16606,
};

const htmlReportDoc: ChatRunDocument = {
  path: "outputs/benchmark.html",
  title: "benchmark.html",
  downloadUrl: "/v2/artifacts/report-2/download",
  kind: "report",
  mimeType: "text/html",
  sizeBytes: 530432,
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
    const { container } = render(<ChatRunDocuments documents={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a report as a card that opens the canvas, not an inline reader", () => {
    const onOpenReport = vi.fn();
    render(<ChatRunDocuments documents={[reportDoc]} onOpenReport={onOpenReport} />);

    const card = screen.getByRole("button", { name: /Toeplitz report/ });
    expect(card).toHaveAttribute("aria-expanded", "false");
    expect(card).toHaveAttribute("aria-controls", "report-canvas");
    /* The card never fetches or expands content into the transcript. */
    expect(screen.queryByText(/Loading report/)).not.toBeInTheDocument();

    fireEvent.click(card);
    expect(onOpenReport).toHaveBeenCalledWith(reportDoc);
  });

  it("marks the card open and says so in the meta line", () => {
    render(
      <ChatRunDocuments
        documents={[htmlReportDoc]}
        onOpenReport={vi.fn()}
        openReportPathKey={runReportPathKey(htmlReportDoc.path)}
      />
    );

    const card = screen.getByRole("button", { name: /benchmark/ });
    expect(card).toHaveAttribute("data-open", "true");
    expect(card).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/open in canvas/)).toBeInTheDocument();
  });

  it("shows the conversation-level version on the card", () => {
    render(
      <ChatRunDocuments
        documents={[reportDoc]}
        onOpenReport={vi.fn()}
        reportVersionCounts={{ [runReportPathKey(reportDoc.path)]: 3 }}
      />
    );
    expect(screen.getByText(/Report · v3/)).toBeInTheDocument();
  });

  it("degrades to a download link when no canvas handler is wired", () => {
    render(<ChatRunDocuments documents={[reportDoc]} />);
    const fallback = screen.getByText("Toeplitz report").closest("a");
    expect(fallback).toHaveAttribute("href", "/v2/artifacts/report-1/download");
    expect(fallback).toHaveAttribute("download");
  });

  it("opens non-report outputs in the canvas and keeps download as a separate action", () => {
    const onOpenReport = vi.fn();
    render(<ChatRunDocuments documents={[codeDoc]} onOpenReport={onOpenReport} />);

    const preview = screen.getByRole("button", { name: "Preview toeplitz_plots.py" });
    expect(preview).toHaveAttribute("aria-expanded", "false");
    expect(preview).toHaveAttribute("aria-controls", "report-canvas");

    const download = screen.getByRole("link", { name: "Download toeplitz_plots.py" });
    expect(download).toHaveAttribute("href", "/v2/artifacts/code-1/download");
    expect(download).toHaveAttribute("download");

    fireEvent.click(preview);
    expect(onOpenReport).toHaveBeenCalledWith(codeDoc);
  });

  it("marks a non-report chip as open without turning its download into the trigger", () => {
    render(
      <ChatRunDocuments
        documents={[codeDoc]}
        onOpenReport={vi.fn()}
        openReportPathKey={runReportPathKey(codeDoc.path)}
      />
    );

    expect(screen.getByRole("button", { name: "Preview toeplitz_plots.py" })).toHaveAttribute(
      "aria-expanded",
      "true"
    );
    expect(screen.getByText(/Code · 20.3 KB · open in canvas/)).toBeInTheDocument();
  });
});
