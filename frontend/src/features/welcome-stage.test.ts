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
  it("shares empty-chat identity across layouts while the centered stage stays desktop-only", () => {
    const landingFlag = blockFrom("const newChatLandingActive =", ";");
    expect(landingFlag).toContain('activePanel === "chat"');
    expect(landingFlag).toContain("activeConversationHydrated");
    expect(landingFlag).toContain("activeMessages.length === 0");

    const welcomeFlag = blockFrom("const welcomeStageActive =", ";");
    expect(welcomeFlag).toContain("newChatLandingActive");
    expect(welcomeFlag).toContain("!isPhoneView");
  });

  it("rides the main shell as a data attribute the CSS keys on", () => {
    expect(appSource).toContain(
      'data-welcome-stage={welcomeStageActive ? "true" : undefined}'
    );
  });

  it("keeps the composer expanded on the welcome stage — no idle pill under the hero", () => {
    // Both collapse attributes must be suppressed while the composer is the
    // welcome stage's primary element.
    // The composer learns about the stage as a prop, and its state model
    // treats the welcome stage as "composing" — never the resting line.
    expect(appSource).toMatch(/welcomeStage=\{welcomeStageActive\}/);
    const model = readFileSync(
      path.join(process.cwd(), "src/components/composer/composerModel.ts"),
      "utf8"
    );
    expect(model).toMatch(/inputs\.welcomeStage\s*\) \{\s*return "composing";/);
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

  it("keeps the scientific field subordinate to the question", () => {
    expect(styles).toMatch(
      /\.meridian-field\s*\{[^}]*width:\s*min\(39rem, 88%\);[^}]*height:\s*clamp\(82px, 13vh, 142px\);[^}]*opacity:\s*0\.82;/s
    );
  });

  it("uses a restrained typographic hierarchy for the welcome prompt", () => {
    expect(styles).toMatch(
      /\.blank-chat-welcome-hero\s*\{[^}]*font-size:\s*1\.625rem;[^}]*font-weight:\s*var\(--font-weight-desktop-invitation\);/s
    );
    expect(styles).toMatch(
      /\.welcome-starting-points-summary\s*\{[^}]*font-weight:\s*400;/s
    );
  });

  it("quiets history only while the desktop welcome stage is resting", () => {
    expect(appSource).toContain(
      'data-welcome-stage={welcomeStageActive ? "true" : undefined}'
    );
    expect(styles).toMatch(
      /\.app-sidebar\[data-welcome-stage="true"\]\s+\.app-sidebar-history-scroll\s*\{[^}]*opacity:\s*0\.62;/s
    );
    expect(styles).toMatch(
      /\.app-sidebar\[data-welcome-stage="true"\]\s+\.app-sidebar-history-scroll:(?:hover|focus-within)/s
    );
  });
});

describe("starter chips", () => {
  it("shows one contextual resume-or-continue action and tucks generic starts behind disclosure", () => {
    const starters = blockFrom(
      "{welcomeStageActive ? (\n            <div className=\"welcome-starters\">",
      "\n          ) : null}"
    );
    const chipCount = (starters.match(/className="welcome-starter-chip"/g) || []).length;
    expect(chipCount).toBe(1);
    expect(starters).toContain("welcomePrimaryAction");
    expect(starters).toContain('data-kind={welcomePrimaryAction.kind}');
    expect(starters).toContain('className="welcome-starting-points"');
    expect(starters).toContain('className="welcome-starting-points-summary"');
    expect(starters.match(/className="welcome-starting-point"/g)).toHaveLength(3);
  });

  it("wires every chip to a REAL destination — live state, not canned copy", () => {
    const starters = blockFrom(
      "{welcomeStageActive ? (\n            <div className=\"welcome-starters\">",
      "\n          ) : null}"
    );
    // Resume or Continue uses the same conversation-opening path as history.
    expect(starters).toContain("openConversationById(welcomePrimaryAction.conversationId)");
    // Dashboard chip drafts into the composer and focuses it.
    expect(starters).toContain("startDashboardDraft");
    // Lens chip is the same navigation the sidebar performs.
    expect(starters).toContain("openScientificViewerPanel");
    // First-use accounts can connect their scientific library in place.
    expect(starters).toContain('openSettings("bisque")');
  });

  it("only offers Continue when a real prior conversation exists", () => {
    const source = blockFrom("const welcomeStarterConversation = useMemo(", ");");
    expect(source).toContain("item.id !== activeConversationId");
    expect(source).toContain("item.title.trim().length > 0");
    // Continue is the fallback only when no resumable run or draft outranks it.
    const primary = blockFrom(
      "const welcomePrimaryAction = useMemo(",
      ");\n  const [welcomeStartingPointsOpen"
    );
    expect(primary).toContain("if (welcomeResumeTarget)");
    expect(primary).toContain("return welcomeStarterConversation");
    expect(primary).toContain('kind: "continue" as const');
  });

  it("expands first-use starting points without an onboarding modal", () => {
    expect(appSource).toContain("setWelcomeStartingPointsOpen(historyItems.length === 0)");
    expect(appSource).toContain("open={welcomeStartingPointsOpen}");
    expect(appSource).not.toContain("WelcomeOnboardingDialog");
  });

  it("drafts the dashboard prompt instead of auto-sending anything", () => {
    const draft = blockFrom("const startDashboardDraft = useCallback", "]);");
    expect(draft).toContain("setActivePromptValue(");
    expect(draft).toContain("focusComposerTextarea()");
    expect(draft).not.toContain("handleSubmit");
  });
});

describe("the task-first welcome contract", () => {
  it("keeps account analytics out of the New Chat canvas", () => {
    const welcome = blockFrom('<div className="blank-chat-welcome">', "\n          </div>");
    const starters = blockFrom(
      "{welcomeStageActive ? (\n            <div className=\"welcome-starters\">",
      "\n          ) : null}"
    );
    expect(welcome).not.toContain("UserTokenUsagePanel");
    expect(starters).not.toContain("welcome-usage");
    expect(starters).not.toContain("<UserTokenUsagePanel");
    expect(appSource).not.toContain("useBlankChatTokenUsage");
  });

  it("does not retain dead welcome-usage chrome", () => {
    expect(styles).not.toContain(".welcome-usage-disclosure");
    expect(styles).not.toContain(".welcome-usage-link");
    expect(styles).not.toContain(".blank-chat-usage-panel");
  });
});
