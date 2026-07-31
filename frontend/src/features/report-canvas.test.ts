/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const canvasSource = readFileSync(
  path.join(process.cwd(), "src/components/canvas/ReportCanvas.tsx"),
  "utf8"
);
const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

/** The report-canvas styles block, from its banner to the rule set that follows it. */
const canvasStyles = (() => {
  const start = stylesSource.indexOf("REPORT CANVAS");
  const end = stylesSource.indexOf("@keyframes rise-in", start);
  return start === -1 || end === -1 ? "" : stylesSource.slice(start, end);
})();

describe("report canvas source contract", () => {
  it("has a styles block to guard", () => {
    expect(canvasStyles.length).toBeGreaterThan(0);
  });

  it("animates the split on the house curve inside the motion law's band", () => {
    // One curve everywhere (brand-motion law): the grid column and every
    // canvas transition ride var(--motion-ease); geometry sits at 220ms.
    expect(canvasStyles).toMatch(
      /transition:\s*grid-template-columns 220ms var\(--motion-ease\)/
    );
    // Forbid the BARE `ease` keyword (whitespace-delimited), not the substring —
    // every var(--motion-ease) reference contains "ease" behind a hyphen.
    expect(canvasStyles).not.toMatch(/transition:[^;]*\sease[\s,;]/);
    expect(canvasStyles).not.toMatch(/cubic-bezier/);
  });

  it("keeps canvas chrome on theme tokens with zero hue literals", () => {
    // The calm law: chrome is neutral, color enters only as data inside the
    // report. Any hex in this block would be a theme-drift seed.
    expect(canvasStyles).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
    expect(canvasStyles).toMatch(/\.chat-report-card\s*\{[^}]*var\(--bg-panel-strong\)/);
    expect(canvasStyles).toMatch(/\.report-canvas-frame\s*\{[^}]*var\(--line\)/);
  });

  it("honors prefers-reduced-motion for the split, the frame, and the chevron", () => {
    const reduced = canvasStyles.slice(
      canvasStyles.indexOf("@media (prefers-reduced-motion: reduce)")
    );
    expect(reduced).toContain(".app-main-shell[data-report-canvas]");
    expect(reduced).toContain(".report-canvas-frame");
    expect(reduced).toContain(".chat-report-card-chevron");
  });

  it("keeps the sandbox an opaque origin: allow-scripts only, never allow-same-origin", () => {
    // Load-bearing security invariant. allow-same-origin would hand
    // model-generated HTML the user's authenticated origin — cookies, /v2,
    // everything. If this assertion fails, stop and think, do not update it.
    expect(canvasSource).toMatch(/REPORT_FRAME_SANDBOX = "allow-scripts"/);
    // Assert against CODE lines only — the comment explaining this rule
    // mentions the forbidden token by name, and asserting a removed feature's
    // name against the whole file is the light-theme-ink ss02 lesson.
    const codeLines = canvasSource
      .split("\n")
      .filter((line) => !/^\s*(\/\/|\/?\*)/.test(line));
    expect(codeLines.some((line) => line.includes("allow-same-origin"))).toBe(false);
    expect(codeLines.some((line) => line.includes("allow-top-navigation"))).toBe(false);
    expect(canvasSource).toMatch(/default-src 'none'/);
    expect(canvasSource).toMatch(/referrerPolicy="no-referrer"/);
  });

  it("auto-opens only from live completion hydration, never from load-time backfill", () => {
    const liveCalls = appSource.match(/autoOpenReport:\s*true/g) ?? [];
    expect(liveCalls).toHaveLength(1);
    // The backfill path re-hydrates persisted conversations on load; if it
    // gained the flag, every old report would pop the canvas on open.
    expect(appSource).toMatch(
      /await hydrateRunArtifacts\(target\.conversationId, target\.messageId, target\.runId\);/
    );
  });

  it("keeps the sidebar controlled so the canvas can trade the rail for measure", () => {
    expect(appSource).toMatch(/open=\{sidebarOpen\}/);
    expect(appSource).toMatch(/onOpenChange=\{setSidebarOpen\}/);
  });

  it("lists the canvas props in EVERY narrow memo comparator on the card's path", () => {
    // ConversationMessageRow AND ConversationTranscript both memoize with
    // EXPLICIT prop lists; a new prop missing from either is silently frozen
    // at its first value. The card's open state shipped stale exactly this
    // way in live verification — the transcript passed the key while the row
    // comparator bailed. If a third comparator ever appears on this path, it
    // joins this loop.
    const comparatorStarts = [...appSource.matchAll(/\(previousProps, nextProps\) =>/g)].map(
      (match) => match.index ?? -1
    );
    expect(comparatorStarts.length).toBeGreaterThanOrEqual(2);
    for (const start of comparatorStarts) {
      const comparator = appSource.slice(start, appSource.indexOf(");", start));
      expect(comparator).toContain(
        "previousProps.openReportPathKey === nextProps.openReportPathKey"
      );
      expect(comparator).toContain(
        "previousProps.reportVersionCounts === nextProps.reportVersionCounts"
      );
    }
  });
});
