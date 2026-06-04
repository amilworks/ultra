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
    ).toBe("Run a durable RareSpot");
  });

  it("normalizes quoted and long fallback text", () => {
    expect(fallbackConversationTitleFromText("  `Analyze   CT scan slice alignment`  ")).toBe(
      "Analyze CT scan slice"
    );
  });
});
