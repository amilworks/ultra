/**
 * ⌘F find-within-conversation wiring.
 *
 * Browser find is defeated in both transcript modes (Virtuoso virtualization;
 * the windowed tail behind "Show earlier messages"), so find runs over message
 * DATA and drives the scroller. These guards pin the wiring that makes that
 * true — and the interception boundaries that keep native find everywhere else.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const bar = readFileSync(
  path.join(process.cwd(), "src/components/chat/TranscriptFindBar.tsx"),
  "utf8"
);

describe("⌘F interception", () => {
  const shortcut = () =>
    appSource.slice(
      appSource.indexOf("const handleFindShortcut"),
      appSource.indexOf('window.addEventListener("keydown", handleFindShortcut)')
    );

  it("intercepts plain ⌘F/^F only — ⌘⇧F and ⌥⌘F pass through", () => {
    expect(shortcut()).toMatch(/!event\.altKey/);
    expect(shortcut()).toMatch(/!event\.shiftKey/);
    expect(shortcut()).toMatch(/event\.key\.toLowerCase\(\) === "f"/);
  });

  it("stands down outside the chat panel, in the viewer, and under dialogs", () => {
    // Resources and the viewer render plain DOM where native find works.
    const effect = appSource.slice(
      appSource.indexOf("/* ⌘F/^F opens (or refocuses) the bar."),
      appSource.indexOf("[activePanel, authStatus, openTranscriptFind, viewerOpen]")
    );
    expect(effect).toMatch(/activePanel !== "chat" \|\| viewerOpen/);
    expect(effect).toMatch(/hasBlockingOverlay\(\)/);
  });

  it("closes when the conversation switches — stale match lists lie", () => {
    expect(appSource).toMatch(/\}, \[activeConversationIdForFind\]\);/);
  });
});

describe("navigation defeats virtualization", () => {
  it("searches message data, not the DOM", () => {
    expect(appSource).toMatch(/computeTranscriptFindMatches\(activeMessages, transcriptFindQuery\)/);
  });

  it("drives Virtuoso by index for long chats", () => {
    expect(appSource).toMatch(/virtuosoRef\.current\?\.scrollToIndex\(\{ index, align: "center" \}\)/);
    expect(appSource).toMatch(/ref=\{virtuosoRef\}/);
  });

  it("widens the message window when the match is behind the fold", () => {
    expect(appSource).toMatch(/setMessageWindow\(messages\.length - index\)/);
  });

  it("keys the scroll target on primitives so streaming deltas cannot yank the scroll", () => {
    const memo = appSource.slice(
      appSource.indexOf("const transcriptFindTarget = useMemo"),
      appSource.indexOf("]", appSource.indexOf("[currentFindMessageId, transcriptFindNonce, transcriptFindOpen]"))
    );
    expect(memo).toMatch(/currentFindMessageId, transcriptFindNonce, transcriptFindOpen/);
  });

  it("clamps the index while matches shift under a streaming answer", () => {
    expect(appSource).toMatch(/Math\.min\(transcriptFindIndex, transcriptFindMatches\.length - 1\)/);
  });

  it("participates in the transcript's memo comparator", () => {
    expect(appSource).toMatch(/previousProps\.findTarget === nextProps\.findTarget/);
  });

  it("both message row variants carry the identity the painter needs", () => {
    const tags = appSource.match(/data-message-id=\{message\.id\}/g) ?? [];
    expect(tags.length).toBe(2);
  });
});

describe("highlighting", () => {
  it("retries paint until the scrolled-to row has mounted", () => {
    const effect = appSource.slice(
      appSource.indexOf("/* Paint highlights over mounted rows."),
      appSource.indexOf("transcriptFindQuery,\n  ]);")
    );
    expect(effect).toMatch(/!currentLocated && attempts < 12/);
    expect(effect).toMatch(/window\.setTimeout\(apply, 90\)/);
  });

  it("uses the Custom Highlight API tints on the brand precedent", () => {
    expect(styles).toMatch(/::highlight\(ultra-find-match\) \{\s*background-color: color-mix\(in oklab, var\(--brand\) 20%, transparent\)/);
    expect(styles).toMatch(/::highlight\(ultra-find-current\)/);
  });

  it("clears highlights on close and on conversation switch", () => {
    const closes = appSource.match(/clearTranscriptFindHighlights\(\)/g) ?? [];
    expect(closes.length).toBeGreaterThanOrEqual(3);
  });
});

describe("the bar itself", () => {
  it("is a labelled search landmark with labelled controls", () => {
    expect(bar).toContain('role="search"');
    expect(bar).toContain('aria-label="Find in conversation"');
    for (const label of ["Previous match", "Next match", "Close find"]) {
      expect(bar).toContain(`aria-label="${label}"`);
    }
  });

  it("reuses the shared icon-button class — no bespoke button styling", () => {
    const buttons = bar.match(/className="chat-message-action"/g) ?? [];
    expect(buttons.length).toBe(3);
  });

  it("Enter advances, Shift+Enter retreats, Escape closes, IME is respected", () => {
    expect(bar).toMatch(/event\.nativeEvent\.isComposing/);
    expect(bar).toMatch(/event\.shiftKey/);
    expect(bar).toMatch(/onPrevious\(\)/);
    expect(bar).toMatch(/event\.key === "Escape"/);
  });

  it("announces the count politely, in tabular figures", () => {
    expect(bar).toContain('aria-live="polite"');
    expect(styles).toMatch(/\.chat-find-count \{[^}]*font-variant-numeric: tabular-nums/s);
  });

  it("is the slim composer's sibling: centered stadium pill on the panel surface", () => {
    const rule = styles.slice(
      styles.indexOf(".chat-find-bar {"),
      styles.indexOf("}", styles.indexOf(".chat-find-bar {"))
    );
    expect(rule).toMatch(/border-radius: 999px/);
    expect(rule).toMatch(/left: 50%/);
    expect(rule).toMatch(/translateX\(-50%\)/);
    expect(rule).toMatch(/background: var\(--bg-panel-strong\)/);
  });

  it("carries focus on the pill border and suppresses every inner ring", () => {
    // The review screenshot showed a stray blue UA focus ring on the input.
    expect(styles).toMatch(/\.chat-find-bar:focus-within \{\s*border-color: color-mix/);
    expect(styles).toMatch(/\.chat-find-bar input:focus,\s*\.chat-find-bar input:focus-visible \{\s*outline: none;\s*box-shadow: none;/);
  });

  it("anchors OUTSIDE the scroll container so it cannot ride away with content", () => {
    // ChatContainerRoot is the scroller (overflow-y-auto); the bar must be its
    // sibling under the non-scrolling relative wrapper, rendered before it.
    const wrapper = appSource.indexOf('className="relative min-h-0 flex-1 overflow-hidden"');
    const barAt = appSource.indexOf("<TranscriptFindBar", wrapper);
    const rootAt = appSource.indexOf("<ChatContainerRoot", wrapper);
    expect(barAt).toBeGreaterThan(wrapper);
    expect(barAt).toBeLessThan(rootAt);
  });
});
