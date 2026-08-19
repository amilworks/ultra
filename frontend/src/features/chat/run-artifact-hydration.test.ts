import { describe, expect, it } from "vitest";

import {
  classifyRunDocumentKind,
  isHydratableRunArtifactDocument,
  isHydratableRunArtifactVisual,
  isReportRunDocument,
  resolveRunOutputArtifactUrl,
  runDocumentCodeLanguage,
  runDocumentPreviewFormat,
  rewriteArtifactMarkdownImageUrls,
  runReportDocumentFormat,
  runReportPathKey,
  shouldHydrateRunArtifacts,
} from "./run-artifact-hydration";

describe("runDocumentPreviewFormat", () => {
  it("previews reports, source artifacts, and PDFs through explicit safe paths", () => {
    expect(runDocumentPreviewFormat("report.md", "text/markdown")).toBe("markdown");
    expect(runDocumentPreviewFormat("report.html", "text/html")).toBe("html");
    expect(runDocumentPreviewFormat("analysis.py", "text/x-python")).toBe("source");
    expect(runDocumentPreviewFormat("results.csv", "text/csv")).toBe("source");
    expect(runDocumentPreviewFormat("notes.txt", "text/plain")).toBe("source");
    expect(runDocumentPreviewFormat("paper.pdf", "application/pdf")).toBe("pdf");
  });

  it("never sends binary scientific data through the text decoder", () => {
    expect(runDocumentPreviewFormat("volume.h5", "application/x-hdf5")).toBeNull();
    expect(runDocumentPreviewFormat("table.parquet", "application/octet-stream")).toBeNull();
    expect(runDocumentPreviewFormat("array.npy", "application/octet-stream")).toBeNull();
  });

  it("maps source files to stable syntax-highlighting languages", () => {
    expect(runDocumentCodeLanguage("analysis.py", "text/plain")).toBe("python");
    expect(runDocumentCodeLanguage("notebook.ipynb", "application/json")).toBe("json");
    expect(runDocumentCodeLanguage("results.tsv", "text/tab-separated-values")).toBe("csv");
    expect(runDocumentCodeLanguage("config.yml", "application/yaml")).toBe("yaml");
    expect(runDocumentCodeLanguage("README.txt", "text/plain")).toBe("text");
  });
});

describe("isHydratableRunArtifactVisual", () => {
  it("keeps code artifacts out of the image preview grid while preserving plots", () => {
    expect(
      isHydratableRunArtifactVisual({
        path: "plot_xcubed.py",
        mime_type: "text/x-python",
      })
    ).toBe(false);

    expect(
      isHydratableRunArtifactVisual({
        path: "plot_xcubed.png",
        mime_type: "image/png",
      })
    ).toBe(true);
  });

  it("uses image file extensions when workers emit a generic MIME type", () => {
    expect(
      isHydratableRunArtifactVisual({
        path: "figures/result.png",
        mime_type: "application/octet-stream",
      })
    ).toBe(true);
    expect(
      isHydratableRunArtifactVisual({
        path: "reports/result.md",
        mime_type: "text/markdown",
      })
    ).toBe(false);
  });
});

describe("shouldHydrateRunArtifacts", () => {
  it("skips hydration when figure urls are already artifact-backed", () => {
    const shouldHydrate = shouldHydrateRunArtifacts(
      {
        role: "assistant",
        runId: "run_123",
        runArtifacts: [
          {
            path: "tool_outputs/example_overlay_mip.png",
            url: "https://example.invalid/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
            downloadUrl:
              "https://example.invalid/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
          },
        ],
        responseMetadata: {
          tool_invocations: [
            {
              tool: "segment_image_megaseg",
            },
          ],
        },
      },
      [
        {
          tool: "segment_image_megaseg",
        },
      ]
    );

    expect(shouldHydrate).toBe(false);
  });

  it("hydrates when the run event stream reports created artifacts", () => {
    const shouldHydrate = shouldHydrateRunArtifacts(
      {
        role: "assistant",
        runId: "run_paper",
        runArtifacts: [],
        runEvents: [{ event_type: "artifact.created" }],
      },
      []
    );

    expect(shouldHydrate).toBe(true);
  });

  it("hydrates persisted assistant messages that only have a run id", () => {
    const shouldHydrate = shouldHydrateRunArtifacts(
      {
        role: "assistant",
        runId: "run_historical",
        runArtifacts: [],
        content: "I saved the plot and code as durable outputs.",
      },
      []
    );

    expect(shouldHydrate).toBe(true);
  });

  it("hydrates when assistant text references an output image path", () => {
    const shouldHydrate = shouldHydrateRunArtifacts(
      {
        role: "assistant",
        runId: "run_paper",
        runArtifacts: [],
        content: "Rendered page: /outputs/paper_pages/arxiv_1706.03762_page_003.png",
      },
      []
    );

    expect(shouldHydrate).toBe(true);
  });

  it("rewrites run-relative paper page markdown images to hydrated artifact urls", () => {
    const rewritten = rewriteArtifactMarkdownImageUrls(
      "See ![Figure 2](/paper_pages/arxiv_1706.03762_page_004.png) for the architecture.",
      [
        {
          path: "paper_pages/arxiv_1706.03762_page_004.png",
          url: "https://ultra.example.org/v2/artifacts/artifact-paper/download",
          downloadUrl: "https://ultra.example.org/v2/artifacts/artifact-paper/download",
        },
      ]
    );

    expect(rewritten).toBe(
      "See ![Figure 2](https://ultra.example.org/v2/artifacts/artifact-paper/download) for the architecture."
    );
  });

  it("rewrites /outputs-prefixed markdown image paths to matching artifacts", () => {
    const rewritten = rewriteArtifactMarkdownImageUrls(
      "Rendered ![page](/outputs/paper_pages/page_004.png \"attention figure\")",
      [
        {
          path: "paper_pages/page_004.png",
          url: "/v2/artifacts/artifact-page/download",
        },
      ]
    );

    expect(rewritten).toBe(
      'Rendered ![page](/v2/artifacts/artifact-page/download "attention figure")'
    );
  });
});

describe("classifyRunDocumentKind", () => {
  it("classifies reports, code, and data by extension", () => {
    expect(classifyRunDocumentKind("outputs/report.md")).toBe("report");
    expect(classifyRunDocumentKind("outputs/analysis.py")).toBe("code");
    expect(classifyRunDocumentKind("outputs/results.csv")).toBe("data");
    expect(classifyRunDocumentKind("outputs/notes.txt")).toBe("document");
  });

  it("falls back to the MIME type when the extension is ambiguous", () => {
    expect(classifyRunDocumentKind("outputs/report", "text/markdown")).toBe("report");
    expect(classifyRunDocumentKind("outputs/data", "application/json")).toBe("data");
  });

  it("returns null for images and unknown binaries", () => {
    expect(classifyRunDocumentKind("outputs/fig1.png", "image/png")).toBeNull();
    expect(classifyRunDocumentKind("outputs/model.bin", "application/octet-stream")).toBeNull();
  });

  it("identifies markdown reports", () => {
    expect(isReportRunDocument("outputs/toeplitz_report.md")).toBe(true);
    expect(isReportRunDocument("outputs/script.py")).toBe(false);
  });

  it("classifies HTML artifacts as reports, not plain documents", () => {
    /* text/html previously fell through to the generic text/* "document"
       branch, which left an HTML report without a reading surface. The
       report check must win before that fallback. */
    expect(classifyRunDocumentKind("outputs/benchmark.html")).toBe("report");
    expect(classifyRunDocumentKind("outputs/index.htm")).toBe("report");
    expect(classifyRunDocumentKind("outputs/report", "text/html")).toBe("report");
    expect(isReportRunDocument("outputs/benchmark.html")).toBe(true);
  });
});

describe("runReportDocumentFormat", () => {
  it("separates the two report renderers by extension and MIME", () => {
    expect(runReportDocumentFormat("outputs/report.md")).toBe("markdown");
    expect(runReportDocumentFormat("outputs/report.html")).toBe("html");
    expect(runReportDocumentFormat("outputs/report", "text/html")).toBe("html");
    expect(runReportDocumentFormat("outputs/report", "text/markdown")).toBe("markdown");
    expect(runReportDocumentFormat("outputs/script.py")).toBeNull();
  });
});

describe("runReportPathKey", () => {
  it("gives re-registrations of the same logical output one identity", () => {
    expect(runReportPathKey("outputs/report.html")).toBe(
      runReportPathKey("/srv/runs/run_b/outputs/report.html")
    );
    expect(runReportPathKey("outputs/Report.HTML")).toBe(
      runReportPathKey("outputs/report.html")
    );
    expect(runReportPathKey("outputs/report.html")).not.toBe(
      runReportPathKey("outputs/appendix.html")
    );
  });
});

describe("resolveRunOutputArtifactUrl", () => {
  it("resolves report-relative references to served artifact urls", () => {
    const artifacts = [
      { path: "outputs/fig1.png", url: "/v2/artifacts/fig-1/download" },
    ];
    expect(resolveRunOutputArtifactUrl("outputs/fig1.png", artifacts)).toBe(
      "/v2/artifacts/fig-1/download"
    );
    expect(resolveRunOutputArtifactUrl("fig1.png", artifacts)).toBe(
      "/v2/artifacts/fig-1/download"
    );
    expect(resolveRunOutputArtifactUrl("outputs/missing.png", artifacts)).toBeNull();
  });
});

describe("isHydratableRunArtifactDocument", () => {
  it("surfaces the markdown report and supporting code from a run's outputs", () => {
    expect(
      isHydratableRunArtifactDocument({
        path: "outputs/toeplitz_fft_embedding_report.md",
        mime_type: "text/markdown",
      })
    ).toBe(true);
    expect(
      isHydratableRunArtifactDocument({
        path: "outputs/toeplitz_plots.py",
        mime_type: "text/x-python",
      })
    ).toBe(true);
  });

  it("never duplicates figures already in the image strip", () => {
    expect(
      isHydratableRunArtifactDocument({
        path: "outputs/fig1_toeplitz_structure.png",
        mime_type: "image/png",
      })
    ).toBe(false);
  });

  it("surfaces HTML reports from a run's outputs", () => {
    expect(
      isHydratableRunArtifactDocument({
        path: "outputs/benchmark.html",
        mime_type: "text/html",
      })
    ).toBe(true);
  });

  it("accepts the REAL registration shape: paths relative to the outputs root", () => {
    /* Traced live (2026-08-01): the control plane registers artifact paths
       WITHOUT an outputs/ prefix — "toeplitz_matrix_report.html",
       "toeplitz_figs/make_figures.py". The old prefix-requiring gate
       dark-holed every such report while the backend registered it
       perfectly. The prefixed shape (agent nests a literal outputs/ dir)
       stays accepted too. */
    expect(
      isHydratableRunArtifactDocument({
        path: "toeplitz_matrix_report.html",
        mime_type: "text/html",
      })
    ).toBe(true);
    expect(
      isHydratableRunArtifactDocument({
        path: "toeplitz_figs/make_figures.py",
        mime_type: "text/x-python",
      })
    ).toBe(true);
    expect(
      isHydratableRunArtifactDocument({
        path: "rarespot_run/report.md",
        mime_type: "text/markdown",
      })
    ).toBe(true);
  });

  it("still denies non-output roots even without the prefix convention", () => {
    for (const path of [
      "uploads/source_notes.md",
      "staged_artifacts/run_prev/data.csv",
      "tool_outputs/tile_004.csv",
      "diagnostics/report_preview/report.console.json",
      ".deepagents/state.json",
    ]) {
      expect(isHydratableRunArtifactDocument({ path, mime_type: "text/plain" })).toBe(
        false
      );
    }
  });

  it("ignores uploads and staged inputs that are not run outputs", () => {
    expect(
      isHydratableRunArtifactDocument({
        path: "uploads/source_notes.md",
        mime_type: "text/markdown",
      })
    ).toBe(false);
    expect(
      isHydratableRunArtifactDocument({
        path: "staged_artifacts/run_prev/data.csv",
        mime_type: "text/csv",
      })
    ).toBe(false);
  });
});
