import { describe, expect, it } from "vitest";

import {
  isHydratableRunArtifactVisual,
  rewriteArtifactMarkdownImageUrls,
  shouldHydrateRunArtifacts,
} from "./run-artifact-hydration";

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
  it("rehydrates megaseg runs that still point figures at local filesystem paths", () => {
    const shouldHydrate = shouldHydrateRunArtifacts(
      {
        role: "assistant",
        runId: "run_123",
        runArtifacts: [],
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
          megasegInsights: {
            heroFigure: {
              url: "/srv/ultra/shared/science/megaseg_results/example_overlay_mip.png",
            },
            secondaryFigures: [],
          },
        },
      ]
    );

    expect(shouldHydrate).toBe(true);
  });

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
          megasegInsights: {
            heroFigure: {
              url: "https://example.invalid/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
            },
            secondaryFigures: [],
          },
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
