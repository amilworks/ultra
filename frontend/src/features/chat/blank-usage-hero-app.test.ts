import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const heatmapSource = readFileSync(
  path.join(process.cwd(), "src/components/TokenActivityHeatmap.tsx"),
  "utf8"
);
const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("blank chat usage hero wiring", () => {
  it("renders usage in the empty chat branch instead of the welcome headline", () => {
    const emptyBranchStart = appSource.indexOf("messages.length === 0 ? (");
    expect(emptyBranchStart).toBeGreaterThan(-1);
    const messagesBranchStart = appSource.indexOf("messages.map((message", emptyBranchStart);
    expect(messagesBranchStart).toBeGreaterThan(emptyBranchStart);

    const emptyBranch = appSource.slice(emptyBranchStart, messagesBranchStart);
    expect(emptyBranch).toContain("<UserTokenUsagePanel");
    expect(emptyBranch).toContain('density="compact"');
    expect(emptyBranch).not.toContain("welcomeHeadline");
    expect(emptyBranch).not.toContain("welcomeSubtitle");
  });

  it("loads token usage for the blank active chat and passes it to the transcript", () => {
    expect(appSource).toContain("const shouldShowBlankChatUsage");
    expect(appSource).toContain("loadCurrentUserTokenUsage(365)");
    expect(appSource).toContain("blankChatTokenUsage={blankChatTokenUsage}");
  });

  it("keeps the empty-chat usage module within the chat text width", () => {
    expect(stylesSource).toMatch(
      /\.blank-chat-usage-panel\s*\{[^}]*width:\s*min\(100%,\s*var\(--user-chat-width\)\);/s
    );
    expect(stylesSource).not.toMatch(/\.blank-chat-usage-panel\s*\{[^}]*70rem/s);
    expect(stylesSource).toMatch(/\.token-heatmap-visual\s*\{/);
    expect(stylesSource).toMatch(
      /\.blank-chat-usage-panel\s+\.token-heatmap\s*\{[^}]*--token-heatmap-cell-size:\s*clamp\(10px,\s*1\.1vw,\s*14px\);/s
    );
  });

  it("uses shadcn tooltips for per-cell token counts", () => {
    expect(heatmapSource).toContain('from "@/components/ui/tooltip"');
    expect(heatmapSource).toContain("<TooltipProvider");
    expect(heatmapSource).toContain("<TooltipTrigger asChild>");
    expect(heatmapSource).toContain("<TooltipContent");
    expect(heatmapSource).toContain("formatExactTokens(cell.value)");
    expect(stylesSource).toMatch(
      /\.token-heatmap-cell:not\(\.is-future\):hover\s*\{[^}]*transform:\s*scale\(1\.08\);/s
    );
    expect(heatmapSource).not.toContain("title={`${formatDayLabel");
  });
});
