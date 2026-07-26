/**
 * Queued follow-ups — Phase 0 of "double texting" (enqueue flavour).
 *
 * Text composed while a run is in flight queues and dispatches as the NEXT
 * turn on clean completion. The risk profile here is entirely about cost and
 * consent: an agentic turn costs minutes and money, so most of these guards
 * pin when dispatch must NOT happen.
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

describe("queueing", () => {
  it("Enter during a run queues — in its own branch, leaving the send path pinned", () => {
    const branch = blockFrom(
      "// Enter during a run queues the draft as a follow-up",
      "queueFollowup();"
    );
    expect(branch).toMatch(/activeSending/);
    expect(branch).toMatch(/!event\.nativeEvent\.isComposing/);
    expect(branch).toMatch(/!event\.shiftKey/);
  });

  it("grows ONE queued message instead of stacking a run-per-message queue", () => {
    // N queued messages would auto-fire N sequential agentic runs — a cost
    // surprise. Repeat sends append a paragraph to the single queued message.
    const queue = blockFrom("const queueFollowup = useCallback", "setActivePromptValue(\"\");");
    expect(queue).toMatch(/current\.queuedFollowup\s*\?\s*`\$\{current\.queuedFollowup\}\\n\\n\$\{text\}`/);
  });

  it("honours the slash-menu and picker contracts the send path enforces", () => {
    const queue = blockFrom("const queueFollowup = useCallback", "setActivePromptValue(\"\");");
    expect(queue).toMatch(/slashMenuOpen \|\| composerResourcePickerOpen/);
    expect(appSource).toMatch(/activePrompt\.trim\(\) && !slashMenuOpen && !composerResourcePickerOpen \? \(/);
  });

  it("ArrowUp with a pending queue un-queues instead of recalling history", () => {
    expect(appSource).toMatch(/if \(activeConversation\?\.queuedFollowup\) \{\s*event\.preventDefault\(\);\s*cancelQueuedFollowup\(\);/);
  });

  it("offers a visible queue button beside Stop, with Stop anchored in place", () => {
    const running = blockFrom("{activeSending ? (", 'aria-label="Stop response"');
    expect(running).toMatch(/aria-label="Queue follow-up"/);
    expect(running).toMatch(/variant="ghost"/);
    // Queue renders BEFORE Stop in source = left of it visually; Stop's slot
    // never shifts (the send-position jump was a bug once already).
    expect(running.indexOf('aria-label="Queue follow-up"')).toBeLessThan(
      running.indexOf('aria-label="Stop response"')
    );
  });
});

describe("the composer is typable during a run — the whole point", () => {
  it("does not fold activeSending into the textarea's disabled state", () => {
    // Review-critical: isLoading={activeSending || ...} disabled the textarea
    // for the entire run, making mid-run follow-ups unreachable for real
    // keyboards. Only script-dispatched events ever reached it.
    expect(appSource).toMatch(/isLoading=\{!activeConversationHydrated\}/);
    expect(appSource).not.toMatch(/isLoading=\{activeSending/);
  });
});

describe("dispatch — when spending a run is allowed", () => {
  const dispatch = () =>
    blockFrom("/* Dispatch: the enqueue contract.", "slashMenuOpen,\n    updateConversation,\n  ]);");

  it("fires only when the conversation has fully settled", () => {
    const effect = dispatch();
    expect(effect).toMatch(/conversation\.sending \|\| conversation\.streamingMessageId/);
    expect(effect).toMatch(/!conversation\.hydrated/);
  });

  it("never auto-spends a run into a stopped or failed turn", () => {
    // The user stopped for a reason. Queued text returns to the draft and the
    // user's own Enter is the consent to spend the next run.
    const effect = dispatch();
    expect(effect).toMatch(/lastMessage\.status !== "stopped"/);
    expect(effect).toMatch(/lastMessage\.status !== "failed"/);
    expect(effect).toMatch(/!conversation\.chatError/);
    expect(effect).toMatch(/setActivePromptValue\(\(previous\) =>/);
  });

  it("only a completion witnessed THIS SESSION may auto-spend a run", () => {
    // A reload can hydrate a failed — or transiently even a still-active — run
    // as settled-and-clean; unarmed conversations take the draft-return arm.
    const effect = dispatch();
    expect(effect).toMatch(/dispatchArmedConversationsRef\.current\.has\(conversation\.id\)/);
    expect(appSource).toMatch(/dispatchArmedConversationsRef\.current\.add\(activeConversation\.id\)/);
  });

  it("defers — without clearing — while the send path would refuse", () => {
    // Clearing first and letting handleSubmit silently decline destroyed the
    // queued text (a "/"-prefixed draft hydrating alongside a queue sufficed).
    const effect = dispatch();
    const defer = effect.indexOf("slashMenuOpen || composerResourcePickerOpen || conversation.selectionImportPending");
    const clear = effect.indexOf('queuedFollowup: "",');
    expect(defer).toBeGreaterThan(-1);
    expect(defer).toBeLessThan(clear);
  });

  it("is StrictMode-safe via an in-flight guard, not just clear-before-submit", () => {
    expect(dispatch()).toMatch(/dispatchInFlightRef\.current\.has\(conversation\.id\)/);
  });

  it("clears the queue BEFORE submitting so a double-fire is impossible", () => {
    const effect = dispatch();
    const clearAt = effect.indexOf('queuedFollowup: "",');
    const submitAt = effect.indexOf("handleSubmitRef.current(queued)");
    expect(clearAt).toBeGreaterThan(-1);
    expect(submitAt).toBeGreaterThan(clearAt);
  });

  it("submits through the same stable handle as the retry path", () => {
    expect(dispatch()).toMatch(/handleSubmitRef\.current\(queued\)/);
  });
});

describe("the queued thought is never destroyed", () => {
  it("cancel returns the text to the composer with the append rule", () => {
    const cancel = blockFrom("const cancelQueuedFollowup = useCallback", "focusComposerTextarea();");
    expect(cancel).toMatch(/queuedFollowup: "",/);
    expect(cancel).toMatch(/previous\.trim\(\) \? `\$\{previous\.replace\(\/\\s\+\$\/, ""\)\}\\n\\n\$\{queued\}` : queued/);
  });

  it("survives a reload — persisted with the snapshot and hydrated back", () => {
    expect(appSource).toMatch(/queuedFollowup: conversation\.queuedFollowup \?\? "",/);
    expect(appSource).toMatch(/typeof state\.queuedFollowup === "string" \? state\.queuedFollowup : ""/);
  });
});

describe("the queued bubble", () => {
  it("states the contract and offers a labelled cancel", () => {
    expect(appSource).toContain("Queued — sends when this run finishes");
    expect(appSource).toContain('aria-label="Cancel queued follow-up"');
  });

  it("renders outside ConversationTranscript's narrow memo", () => {
    const wrapper = appSource.indexOf('className="chat-queued-followup chat-width-frame');
    const transcript = appSource.indexOf("<ConversationTranscript");
    const anchor = appSource.indexOf("<ChatContainerScrollAnchor />");
    expect(wrapper).toBeGreaterThan(transcript);
    expect(wrapper).toBeLessThan(anchor);
  });

  it("uses the non-submit tooltip class and a WCAG-passing eyebrow", () => {
    const running = blockFrom("{activeSending ? (", 'aria-label="Stop response"');
    expect(running).toContain('className="app-composer-tooltip"');
    expect(styles).toMatch(/\.chat-queued-followup-eyebrow \{[^}]*color: color-mix\(in oklab, var\(--text-muted\) 72%, var\(--text-main\) 28%\)/s);
  });

  it("dresses at reduced presence with the brand motion and its opt-out", () => {
    const bubble = styles.slice(
      styles.indexOf(".chat-queued-followup-bubble {"),
      styles.indexOf("}", styles.indexOf(".chat-queued-followup-bubble {"))
    );
    expect(bubble).toMatch(/var\(--motion-ease\)/);
    expect(bubble).toMatch(/border: 1px solid var\(--line\)/);
    expect(styles).toMatch(/\.chat-queued-followup-text \{[^}]*white-space: pre-wrap/s);
    const reduced = styles.indexOf(".chat-queued-followup-bubble {", styles.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(reduced).toBeGreaterThan(-1);
  });
});
