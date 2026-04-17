import { describe, expect, it } from "vitest";

import { shouldHydrateRunArtifacts } from "./run-artifact-hydration";

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
            url: "https://ultra.ece.ucsb.edu/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
            downloadUrl:
              "https://ultra.ece.ucsb.edu/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
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
              url: "https://ultra.ece.ucsb.edu/v1/artifacts/run_123/download?path=tool_outputs%2Fexample_overlay_mip.png",
            },
            secondaryFigures: [],
          },
        },
      ]
    );

    expect(shouldHydrate).toBe(false);
  });
});
