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
    expect(assistantHeader).toContain("{tokenUsage ? (");
    expect(assistantHeader).not.toContain(
      ") : reasonedDurationLabel || summaryModeLabel || tokenUsage ? ("
    );
  });
});
