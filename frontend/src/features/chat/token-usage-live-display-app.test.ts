import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("assistant token usage live display wiring", () => {
  it("does not hide token usage metadata behind the completed-message branch", () => {
    const tokenUsageStart = appSource.indexOf("const tokenUsage = useMemo");
    expect(tokenUsageStart).toBeGreaterThan(-1);

    const assistantReturnStart = appSource.indexOf(
      "return (\n      <Message",
      tokenUsageStart
    );
    const leadingCardsStart = appSource.indexOf(
      "{showLeadingToolResultCards ?",
      assistantReturnStart
    );
    expect(assistantReturnStart).toBeGreaterThan(tokenUsageStart);
    expect(leadingCardsStart).toBeGreaterThan(assistantReturnStart);

    const assistantHeader = appSource.slice(assistantReturnStart, leadingCardsStart);
    // Token usage (now shown alongside the elapsed time) must render directly,
    // not behind a completed-message-only branch.
    expect(assistantHeader).toContain("{tokenUsage || elapsedLabel ? (");
    expect(assistantHeader).toContain("isStreamingAssistant ? (");
    expect(assistantHeader).toContain("<AnimatedTokenCount value={tokenUsage.total_tokens} />");
    expect(assistantHeader).toContain("formatTokens(tokenUsage.total_tokens)");
    expect(assistantHeader).not.toContain(
      ") : reasonedDurationLabel || summaryModeLabel || tokenUsage ? ("
    );
  });

  it("keeps the live token ticker out of the composer loader", () => {
    const composerStart = appSource.indexOf('<div className="composer-running">');
    const composerEnd = appSource.indexOf("<PromptInputTextarea", composerStart);
    expect(composerStart).toBeGreaterThan(-1);
    expect(composerEnd).toBeGreaterThan(composerStart);

    const composerRunning = appSource.slice(composerStart, composerEnd);
    expect(composerRunning).toContain("formatTokens(activeStreamingTokenUsage.total_tokens)");
    expect(composerRunning).not.toContain("AnimatedTokenCount");
  });
});
