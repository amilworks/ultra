/**
 * Welcome stage — the empty-chat composition contract.
 *
 * On an empty, hydrated DESKTOP chat, hero + composer + starter chips render
 * as one centered cluster: the primary action sits where the eye lands, not
 * docked a viewport away. The first send drops the flag (the optimistic
 * message makes messages.length > 0) and the same composer instance re-docks
 * by reflow — never by remount, so the draft and focus survive. Phones keep
 * their own composer-forward hero (thumb reach wants the dock).
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

const blockFrom = (start: string, end: string): string => {
  const startIndex = appSource.indexOf(start);
  expect(startIndex, `missing block: ${start.slice(0, 60)}`).toBeGreaterThan(-1);
  const endIndex = appSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated block: ${start.slice(0, 60)}`).toBeGreaterThan(startIndex);
  return appSource.slice(startIndex, endIndex + end.length);
};

describe("the welcome stage flag", () => {
  it("is desktop-only, chat-only, and drops the moment a message exists", () => {
    const flag = blockFrom("const welcomeStageActive =", ";");
    expect(flag).toContain('activePanel === "chat"');
    expect(flag).toContain("!isPhoneView");
    expect(flag).toContain("activeConversationHydrated");
    expect(flag).toContain("activeMessages.length === 0");
  });

  it("rides the main shell as a data attribute the CSS keys on", () => {
    expect(appSource).toContain(
      'data-welcome-stage={welcomeStageActive ? "true" : undefined}'
    );
  });

  it("keeps the composer expanded on the welcome stage — no idle pill under the hero", () => {
    // Both collapse attributes must be suppressed while the composer is the
    // welcome stage's primary element.
    const slim = blockFrom("data-composer-slim={", "}");
    expect(slim).toContain("!welcomeStageActive &&");
    const idle = blockFrom("data-composer-idle={", "}");
    expect(idle).toContain("!welcomeStageActive &&");
  });
});

describe("the centered-cluster layout", () => {
  it("centers with `safe` and never fights the report-canvas grid", () => {
    expect(styles).toMatch(
      /\.app-main-shell\[data-welcome-stage="true"\]:not\(\[data-report-canvas\]\)\s*\{[^}]*justify-content:\s*safe center;/s
    );
    expect(styles).toMatch(
      /\.app-main-shell\[data-welcome-stage="true"\]:not\(\[data-report-canvas\]\)\s+\.chat-stage-scroller\s*\{[^}]*flex:\s*0 1 auto;/s
    );
  });

  it("styles starter chips with tokens only — hairline ghosts, ink label, muted icon", () => {
    const chip = styles.match(/\.welcome-starter-chip\s*\{[^}]*\}/s)?.[0];
    expect(chip).toBeTruthy();
    expect(chip).toContain("border: 1px solid var(--line);");
    expect(chip).toContain("color: var(--text-main);");
    expect(chip).not.toMatch(/#[0-9a-fA-F]{3,8}/);
    const icon = styles.match(/\.welcome-starter-chip svg\s*\{[^}]*\}/s)?.[0];
    expect(icon).toContain("color: var(--text-muted);");
    // Keyboard users get a visible ring.
    expect(styles).toMatch(/\.welcome-starter-chip:focus-visible\s*\{[^}]*var\(--ring\)/s);
  });
});

describe("starter chips", () => {
  it("renders only on the welcome stage, capped at three quiet suggestions", () => {
    const starters = blockFrom(
      "{welcomeStageActive ? (\n            <div className=\"welcome-starters\">",
      "\n          ) : null}"
    );
    const chipCount = (starters.match(/className="welcome-starter-chip"/g) || []).length;
    expect(chipCount).toBeGreaterThanOrEqual(2);
    expect(chipCount).toBeLessThanOrEqual(3);
  });

  it("wires every chip to a REAL destination — live state, not canned copy", () => {
    const starters = blockFrom(
      "{welcomeStageActive ? (\n            <div className=\"welcome-starters\">",
      "\n          ) : null}"
    );
    // Continue <most recent real conversation> — same handler the sidebar uses.
    expect(starters).toContain("openHistoryItem(welcomeStarterConversation)");
    // Dashboard chip drafts into the composer and focuses it.
    expect(starters).toContain("startDashboardDraft");
    // Lens chip is the same navigation the sidebar performs.
    expect(starters).toContain("openScientificViewerPanel");
  });

  it("only offers Continue when a real prior conversation exists", () => {
    const source = blockFrom("const welcomeStarterConversation = useMemo(", ");");
    expect(source).toContain("item.id !== activeConversationId");
    expect(source).toContain("item.title.trim().length > 0");
    // The chip itself is conditional on the lookup.
    expect(appSource).toContain("{welcomeStarterConversation ? (");
  });

  it("drafts the dashboard prompt instead of auto-sending anything", () => {
    const draft = blockFrom("const startDashboardDraft = useCallback", "]);");
    expect(draft).toContain("setActivePromptValue(");
    expect(draft).toContain("focusComposerTextarea()");
    expect(draft).not.toContain("handleSubmit");
  });
});
