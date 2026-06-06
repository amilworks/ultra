import { describe, expect, it } from "vitest";

import {
  fallbackConversationTitleFromText,
  resolveConversationTitle,
} from "./conversation-title";

describe("conversation title fallback", () => {
  it("keeps a meaningful stored title", () => {
    expect(resolveConversationTitle("DPI bubble sort smoke fixed", "ignored fallback")).toBe(
      "DPI bubble sort smoke fixed"
    );
  });

  it("derives a calm short title when the stored title is default", () => {
    expect(
      resolveConversationTitle(
        "New conversation",
        "Run a durable RareSpot analysis on the latest prairie dog imagery"
      )
    ).toBe("RareSpot Analysis Prairie Dog Imagery");
  });

  it("normalizes quoted and long fallback text", () => {
    expect(fallbackConversationTitleFromText("  `Analyze   CT scan slice alignment`  ")).toBe(
      "CT Scan Slice Alignment"
    );
  });

  it("uses keywords across varied scientific prompts instead of the first few words", () => {
    expect(
      fallbackConversationTitleFromText(
        "Please compare the attention paper with a transformer baseline"
      )
    ).toBe("Attention Paper Transformer Baseline");
    expect(
      fallbackConversationTitleFromText("Create a matplotlib y = x^2 plot with labels")
    ).toBe("Matplotlib Y X 2 Plot Labels");
    expect(
      fallbackConversationTitleFromText("Train a small UNet for cell segmentation masks")
    ).toBe("Small UNet Cell Segmentation Masks");
  });
});
