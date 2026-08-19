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
    // This is inline prose metadata, not an aligned numeric column. Tabular
    // figures put the narrow `1` inside the same wide cell as `9`, which makes
    // a value such as 91.0K look as though it contains an accidental space.
    expect(assistantHeader).toContain('className="proportional-nums"');
    expect(assistantHeader).not.toContain('className="tabular-nums"');
    expect(assistantHeader).not.toContain(
      ") : reasonedDurationLabel || summaryModeLabel || tokenUsage ? ("
    );
  });

  it("keeps the live token ticker in the composer, and keeps it calm", () => {
    const composerStart = appSource.search(/<div className="composer-running"/);
    const composerEnd = appSource.indexOf("<PromptInputTextarea", composerStart);
    expect(composerStart).toBeGreaterThan(-1);
    expect(composerEnd).toBeGreaterThan(composerStart);

    const composerRunning = appSource.slice(composerStart, composerEnd);
    // The composer renders the composed label and never the animated counter:
    // a number animating next to the stop button is motion the user did not ask
    // for, in the one place they are already waiting.
    expect(composerRunning).toContain("composerRunningLabel");
    expect(composerRunning).not.toContain("AnimatedTokenCount");
    // The breakdown tooltip is what stops a cumulative total like "616K tokens"
    // from reading as conversation size.
    expect(composerRunning).toContain("title={composerRunningTitle}");

    const labelStart = appSource.indexOf("const composerRunningLabel = useMemo");
    expect(labelStart).toBeGreaterThan(-1);
    const labelEnd = appSource.indexOf("}, [activeElapsedSeconds", labelStart);
    expect(labelEnd).toBeGreaterThan(labelStart);
    const label = appSource.slice(labelStart, labelEnd);

    // Metrics ADD to the status line rather than replacing it. Regressing to a
    // bare token count is the original bug: the plain-language reassurance
    // disappeared exactly when a turn ran long enough to need it.
    expect(label).toContain("BisQue Ultra is processing");
    expect(label).toContain("formatElapsedDuration(activeElapsedSeconds)");
    expect(label).toContain("formatTokens(activeStreamingTokenUsage.total_tokens)");
  });

  it("ticks the composer clock once a second, anchored so it cannot restart", () => {
    const effectStart = appSource.indexOf("const activeRunStartedAtRef");
    expect(effectStart).toBeGreaterThan(-1);
    const effect = appSource.slice(effectStart, effectStart + 1200);

    // 1s is the calm cadence — fast enough to read as live, slow enough not to
    // pull the eye. A sub-second interval would also re-render the composer
    // several times a second for a number nobody is reading that closely.
    expect(effect).toContain("window.setInterval(tick, 1000)");
    expect(effect).toContain("window.clearInterval(timer)");
    // Anchoring on a ref keeps a mid-run re-render (new tokens, new events)
    // from resetting the clock back to zero.
    expect(effect).toContain("activeRunStartedAtRef.current === null");
  });
});

describe("composer running line stays quiet", () => {
  const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

  it("renders the running telemetry muted, small, and in tabular figures", () => {
    const start = styles.indexOf(".composer-running {");
    expect(start).toBeGreaterThan(-1);
    const rule = styles.slice(start, styles.indexOf("}", start));

    // This line sits behind a running turn; it must never be the loudest thing
    // on screen. Full-brightness body text here is the bug this guards.
    expect(rule).toContain("color: var(--text-muted)");
    expect(rule).not.toMatch(/color:\s*var\(--(text-primary|foreground|sidebar-foreground)\)/);
    // Both the elapsed seconds and the token total tick while running, so
    // proportional digits would reflow the line on every update.
    expect(rule).toContain("font-variant-numeric: tabular-nums");
  });
});
