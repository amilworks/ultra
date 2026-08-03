import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  MAX_INLINE_REPORT_BYTES,
  ReportCanvas,
  prepareHtmlReportDocument,
  type ReportCanvasVersion,
} from "./ReportCanvas";

const markdownVersion: ReportCanvasVersion = {
  messageId: "m1",
  runId: "run_9f3c2a",
  document: {
    path: "outputs/report.md",
    title: "report.md",
    downloadUrl: "/v2/runs/run_9f3c2a/artifacts/download?path=outputs%2Freport.md",
    mimeType: "text/markdown",
    sizeBytes: 4200,
  },
  imageArtifacts: [],
};

const htmlVersion: ReportCanvasVersion = {
  messageId: "m2",
  runId: "run_b71d44",
  document: {
    path: "outputs/benchmark.html",
    title: "benchmark.html",
    downloadUrl: "/v2/runs/run_b71d44/artifacts/download?path=outputs%2Fbenchmark.html",
    mimeType: "text/html",
    sizeBytes: 128000,
  },
  imageArtifacts: [],
};

describe("prepareHtmlReportDocument", () => {
  it("injects the CSP first, strips <base>, and inlines artifact images as data: URIs", async () => {
    /* data:, NOT blob: — a blob URL is scoped to the origin that minted it,
       and the sandboxed frame's opaque origin cannot read the host's blobs
       (verified live: parent-minted blob renders broken, data: renders). */
    const fetchImageDataUrl = vi.fn().mockResolvedValue("data:image/png;base64,MOCK1");
    const prepared = await prepareHtmlReportDocument(
      `<!doctype html><html><head><base href="https://evil.example/"><title>Bench</title></head>` +
        `<body><img src="outputs/fig1.png"><img src="https://cdn.example/x.png"></body></html>`,
      [{ path: "outputs/fig1.png", url: "/v2/artifacts/fig-1/download" }],
      fetchImageDataUrl
    );

    expect(prepared.title).toBe("Bench");
    expect(prepared.srcdoc).not.toContain("<base");
    /* The CSP must be the head's FIRST child so nothing loads outside it. */
    expect(prepared.srcdoc).toMatch(/<head><meta http-equiv="Content-Security-Policy"/);
    expect(prepared.srcdoc).toContain("default-src 'none'");
    /* Artifact-backed figure inlined; unknown external reference left for the
       CSP to block rather than silently fetched. */
    expect(fetchImageDataUrl).toHaveBeenCalledTimes(1);
    expect(fetchImageDataUrl).toHaveBeenCalledWith("/v2/artifacts/fig-1/download");
    expect(prepared.srcdoc).toContain('src="data:image/png;base64,MOCK1"');
    expect(prepared.srcdoc).toContain('src="https://cdn.example/x.png"');
  });

  it("keeps data: images untouched", async () => {
    const fetchImageDataUrl = vi.fn();
    const prepared = await prepareHtmlReportDocument(
      `<html><body><img src="data:image/png;base64,AAAA"></body></html>`,
      [],
      fetchImageDataUrl
    );
    expect(fetchImageDataUrl).not.toHaveBeenCalled();
    expect(prepared.srcdoc).toContain("data:image/png;base64,AAAA");
  });

  it("injects the fragment-navigation shim after the report's own content", async () => {
    /* Fragment links can't navigate in a sandboxed srcdoc frame — the click
       either dies silently or replaces the report with an error page (a
       delivered report's TOC blanked the canvas exactly this way). The shim
       turns them into same-document scrolling. */
    const prepared = await prepareHtmlReportDocument(
      `<html><body><a href="#sec">TOC</a><h2 id="sec">Section</h2></body></html>`,
      [],
      vi.fn()
    );
    expect(prepared.srcdoc).toContain("a[href^=");
    expect(prepared.srcdoc).toContain("scrollIntoView");
    /* preventDefault unconditionally: a missing target must no-op, never
       attempt the navigation that blanks the frame. */
    expect(prepared.srcdoc).toContain("event.preventDefault()");
    expect(prepared.srcdoc.indexOf("scrollIntoView")).toBeGreaterThan(
      prepared.srcdoc.indexOf('id="sec"')
    );
    /* Reduced motion is honored inside the frame too. */
    expect(prepared.srcdoc).toContain("prefers-reduced-motion");
  });
});

describe("ReportCanvas", () => {
  it("renders a markdown report at reading quality and titles the chrome from its H1", async () => {
    const loadDocumentText = vi
      .fn()
      .mockResolvedValue("# Prairie-dog detection\n\nCheckpoint B clears A.");

    render(
      <ReportCanvas
        versions={[markdownVersion]}
        mode="split"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
      />
    );

    await waitFor(() =>
      expect(screen.getByText(/Checkpoint B clears A/)).toBeInTheDocument()
    );
    expect(screen.getAllByText("Prairie-dog detection").length).toBeGreaterThan(0);
    expect(screen.getByText(/run_9f3c2a/)).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders an HTML report inside a sandboxed opaque-origin frame", async () => {
    const loadDocumentText = vi
      .fn()
      .mockResolvedValue("<html><head><title>Bench</title></head><body>hi</body></html>");

    const { container } = render(
      <ReportCanvas
        versions={[htmlVersion]}
        mode="split"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
      />
    );

    await waitFor(() => expect(container.querySelector("iframe")).not.toBeNull());
    const frame = container.querySelector("iframe");
    /* The security boundary: scripts may run, but the frame gets an opaque
       origin — allow-same-origin would hand it the user's authenticated
       origin, so its absence is load-bearing. */
    expect(frame?.getAttribute("sandbox")).toBe("allow-scripts");
    expect(frame?.getAttribute("srcdoc")).toContain("Content-Security-Policy");
    expect(frame?.getAttribute("referrerpolicy")).toBe("no-referrer");
  });

  it("closes from the header and offers the artifact download", async () => {
    const onClose = vi.fn();
    const loadDocumentText = vi.fn().mockResolvedValue("# T\n\nbody");
    render(
      <ReportCanvas
        versions={[markdownVersion]}
        mode="split"
        onClose={onClose}
        loadDocumentText={loadDocumentText}
      />
    );

    const download = await screen.findByLabelText(/Download/);
    expect(download).toHaveAttribute("href", markdownVersion.document.downloadUrl);
    expect(download).toHaveAttribute("download");

    fireEvent.click(screen.getByLabelText("Close report canvas"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("shows the version chip only when a second registration exists, newest active", async () => {
    const loadDocumentText = vi.fn().mockResolvedValue("# v2 body");
    const olderVersion: ReportCanvasVersion = {
      ...markdownVersion,
      messageId: "m0",
      runId: "run_older",
      document: {
        ...markdownVersion.document,
        downloadUrl: "/v2/runs/run_older/artifacts/download?path=outputs%2Freport.md",
      },
    };

    render(
      <ReportCanvas
        versions={[olderVersion, markdownVersion]}
        mode="split"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
      />
    );

    /* Latest registration wins by default… */
    await waitFor(() =>
      expect(loadDocumentText).toHaveBeenCalledWith(markdownVersion.document.downloadUrl)
    );
    expect(screen.getByRole("button", { name: /Version 2 of 2/ })).toBeInTheDocument();
  });

  it("never pulls an oversized report into memory — offers the download instead", async () => {
    const loadDocumentText = vi.fn();
    render(
      <ReportCanvas
        versions={[
          {
            ...htmlVersion,
            document: {
              ...htmlVersion.document,
              sizeBytes: MAX_INLINE_REPORT_BYTES + 1,
            },
          },
        ]}
        mode="split"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
      />
    );

    expect(await screen.findByText(/too large to render inline/)).toBeInTheDocument();
    expect(screen.getByText("Download it")).toHaveAttribute(
      "href",
      htmlVersion.document.downloadUrl
    );
    expect(loadDocumentText).not.toHaveBeenCalled();
  });

  it("becomes a full-screen sheet on the phone regime with a back affordance", async () => {
    const onClose = vi.fn();
    const loadDocumentText = vi.fn().mockResolvedValue("# T\n\nbody");
    render(
      <ReportCanvas
        versions={[markdownVersion]}
        mode="sheet"
        onClose={onClose}
        loadDocumentText={loadDocumentText}
      />
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Back to chat"));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByLabelText("Close report canvas")).not.toBeInTheDocument();
  });
});

describe("ReportCanvas split resize", () => {
  it("exposes a keyboard-operable separator that commits clamped widths", async () => {
    const onSplitWidthCommit = vi.fn();
    const onSplitWidthReset = vi.fn();
    const loadDocumentText = vi.fn().mockResolvedValue("# T\n\nbody");

    render(
      <ReportCanvas
        versions={[markdownVersion]}
        mode="split"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
        splitWidth={648}
        splitWidthBounds={{ min: 320, max: 700 }}
        onSplitWidthCommit={onSplitWidthCommit}
        onSplitWidthReset={onSplitWidthReset}
      />
    );

    const separator = screen.getByRole("separator", { name: "Resize report panel" });
    expect(separator).toHaveAttribute("aria-valuenow", "648");
    /* The panel hangs on the right, so ArrowLeft grows it; bounds clamp. */
    fireEvent.keyDown(separator, { key: "ArrowLeft" });
    expect(onSplitWidthCommit).toHaveBeenCalledWith(664);
    fireEvent.keyDown(separator, { key: "End" });
    expect(onSplitWidthCommit).toHaveBeenCalledWith(700);
    fireEvent.keyDown(separator, { key: "Home" });
    expect(onSplitWidthCommit).toHaveBeenCalledWith(320);
    fireEvent.doubleClick(separator);
    expect(onSplitWidthReset).toHaveBeenCalledTimes(1);
  });

  it("keeps the divider out of the sheet regime", () => {
    const loadDocumentText = vi.fn().mockResolvedValue("# T\n\nbody");
    render(
      <ReportCanvas
        versions={[markdownVersion]}
        mode="sheet"
        onClose={vi.fn()}
        loadDocumentText={loadDocumentText}
      />
    );
    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
  });
});
