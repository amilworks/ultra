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

import { mergeQueuedNoteSearchScopeOverride } from "./queued-followup";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const composerSource = readFileSync(
  path.join(process.cwd(), "src/components/composer/Composer.tsx"),
  "utf8"
);
const editorSource = readFileSync(
  path.join(process.cwd(), "src/components/composer/ComposerEditor.tsx"),
  "utf8"
);
const blockFromComposer = (start: string, end: string): string => {
  const startIndex = composerSource.indexOf(start);
  expect(startIndex, `missing composer block: ${start.slice(0, 60)}`).toBeGreaterThan(-1);
  const endIndex = composerSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated composer block: ${start.slice(0, 60)}`).toBeGreaterThan(startIndex);
  return composerSource.slice(startIndex, endIndex + end.length);
};

const blockFrom = (start: string, end: string): string => {
  const startIndex = appSource.indexOf(start);
  expect(startIndex, `missing block: ${start.slice(0, 60)}`).toBeGreaterThan(-1);
  const endIndex = appSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated block: ${start.slice(0, 60)}`).toBeGreaterThan(startIndex);
  return appSource.slice(startIndex, endIndex + end.length);
};

describe("queueing", () => {
  it("Enter during a run queues — in its own branch, leaving the send path pinned", () => {
    // Plain Enter reaches the composer only outside IME composition and without
    // Shift (the editor's gate); the composer's own branch queues during a run.
    const enter = blockFromComposer("const handleEnter = useCallback", "}, []);");
    expect(enter).toMatch(/if \(state\.running\) \{\s*state\.onQueue\(\);\s*return true;/);
    expect(editorSource).toMatch(/!event\.shiftKey &&\s*!event\.altKey &&\s*!event\.isComposing/);
    expect(appSource).toMatch(/running=\{activeSending\}/);
    expect(appSource).toMatch(/onQueue=\{queueFollowup\}/);
  });

  it("grows ONE queued message instead of stacking a run-per-message queue", () => {
    // N queued messages would auto-fire N sequential agentic runs — a cost
    // surprise. Repeat sends append a paragraph to the single queued message.
    const queue = blockFrom("const queueFollowup = useCallback", "setActivePromptValue(\"\");");
    expect(queue).toMatch(/current\.queuedFollowup\s*\?\s*`\$\{current\.queuedFollowup\}\\n\\n\$\{text\}`/);
  });

  it("seals Note refs and pasted-reference provenance into that queued turn", () => {
    const queue = blockFrom("const queueFollowup = useCallback", "setActivePromptValue(\"\");");
    expect(queue).toMatch(/queuedFollowupNotes: queuedNotes/);
    expect(queue).toMatch(/queuedFollowupExcludedNoteIntentText/);
    expect(queue).toMatch(/selectedNotes: \[\]/);
    expect(queue).toMatch(/withoutNoteAccess\(current\.activeSelectionContext\)/);
  });

  it("uses one precedence rule for normal queueing and steering-closed fallback", () => {
    expect(appSource.match(/mergeQueuedNoteSearchScopeOverride\(\{/g)).toHaveLength(2);
    expect(
      mergeQueuedNoteSearchScopeOverride({
        existingOverride: false,
        incomingOverride: null,
        incomingText: "ordinary follow-up",
        incomingExcludedReferenceText: [],
      })
    ).toBe(false);
    expect(
      mergeQueuedNoteSearchScopeOverride({
        existingOverride: false,
        incomingOverride: null,
        incomingText: "Search my notes for calibration",
        incomingExcludedReferenceText: [],
      })
    ).toBe(true);
    expect(
      mergeQueuedNoteSearchScopeOverride({
        existingOverride: true,
        incomingOverride: false,
        incomingText: "Search my notes for calibration",
        incomingExcludedReferenceText: [],
      })
    ).toBe(false);
  });

  it("does not let a paste-only request silently supersede a removed scope", () => {
    const pasted = "Search my notes for calibration";
    expect(
      mergeQueuedNoteSearchScopeOverride({
        existingOverride: false,
        incomingOverride: null,
        incomingText: pasted,
        incomingExcludedReferenceText: [pasted],
      })
    ).toBe(false);
    expect(
      mergeQueuedNoteSearchScopeOverride({
        existingOverride: false,
        incomingOverride: true,
        incomingText: pasted,
        incomingExcludedReferenceText: [pasted],
      })
    ).toBe(true);
  });

  it("honours the slash-menu and picker contracts the send path enforces", () => {
    const queue = blockFrom("const queueFollowup = useCallback", "setActivePromptValue(\"\");");
    expect(queue).toMatch(
      /slashMenuOpen\s+\|\|\s+composerResourcePickerOpen\s+\|\|\s+composerNotePickerOpen/
    );
    expect(appSource).toMatch(
      /const composerCanSteer =\s*activePrompt\.trim\(\)\.length > 0 &&\s*!slashMenuOpen &&\s*!composerResourcePickerOpen &&\s*!composerNotePickerOpen;/
    );
  });

  it("ArrowUp with a pending queue un-queues instead of recalling history", () => {
    expect(appSource).toMatch(/if \(activeConversation\?\.queuedFollowup\) \{\s*event\.preventDefault\(\);\s*cancelQueuedFollowup\(\);/);
  });

  it("offers a visible queue button beside Stop, with Stop anchored in place", () => {
    const running = blockFromComposer("{running ? (", 'aria-label="Stop response"');
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
    // Hydration is the ONLY thing that disables typing; the run state reaches
    // the composer as its own prop and never touches the editor's disabled flag.
    expect(composerSource).toMatch(/disabled: !hydrated,/);
    expect(composerSource).not.toMatch(/disabled: [^\n]*running/);
    expect(appSource).toMatch(/hydrated=\{activeConversationHydrated\}/);
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
    const defer = effect.search(
      /slashMenuOpen\s+\|\|\s+composerResourcePickerOpen\s+\|\|\s+composerNotePickerOpen\s+\|\|\s+conversation\.selectionImportPending/
    );
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
    const submitAt = effect.indexOf("handleSubmitRef.current(queued,");
    expect(clearAt).toBeGreaterThan(-1);
    expect(submitAt).toBeGreaterThan(clearAt);
  });

  it("submits through the same stable handle as the retry path", () => {
    expect(dispatch()).toMatch(/handleSubmitRef\.current\(queued,/);
  });

  it("submits the sealed Notes snapshot, never mutable composer chips", () => {
    const effect = dispatch();
    expect(effect).toMatch(/selectedNotes: queuedNotes/);
    expect(effect).toMatch(/excludedNoteIntentText: queuedExcludedNoteIntentText/);
    expect(effect).toMatch(/preserveComposerScope: true/);
    expect(effect).toMatch(/onNoteScopeFailure: restoreQueuedToDraft/);
  });
});

describe("the queued thought is never destroyed", () => {
  it("cancel returns the text to the composer with the append rule", () => {
    const cancel = blockFrom("const cancelQueuedFollowup = useCallback", "focusComposerTextarea();");
    expect(cancel).toMatch(/queuedFollowup: "",/);
    expect(cancel).toMatch(/queuedFollowupNotes: \[\]/);
    expect(cancel).toMatch(/selectedNotes: restoredNotes/);
    expect(cancel).toMatch(/previous\.trim\(\) \? `\$\{previous\.replace\(\/\\s\+\$\/, ""\)\}\\n\\n\$\{queued\}` : queued/);
  });

  it("survives a reload — persisted with the snapshot and hydrated back", () => {
    expect(appSource).toMatch(/queuedFollowup: conversation\.queuedFollowup \?\? "",/);
    expect(appSource).toMatch(/typeof state\.queuedFollowup === "string" \? state\.queuedFollowup : ""/);
    expect(appSource).toMatch(/queuedFollowupNotes: conversation\.queuedFollowupNotes \?\? \[\]/);
    expect(appSource).toMatch(/toSelectedNoteChips\(state\.queuedFollowupNotes\)/);
  });
});

describe("the queued bubble", () => {
  it("states the contract and offers a labelled cancel", () => {
    expect(appSource).toContain("Queued — sends when this run finishes");
    expect(appSource).toContain('aria-label="Cancel queued follow-up"');
    expect(appSource).toContain('aria-label="Notes for queued message"');
  });

  it("shows sealed Notes search as a calm removable queued-message chip", () => {
    expect(appSource).toContain("const activeQueuedNoteSearchScope");
    expect(appSource).toContain("activeConversation.queuedFollowupExcludedNoteIntentText");
    expect(appSource).toContain("activeConversation.queuedFollowupNoteSearchScopeOverride");
    expect(appSource).toContain("Search Notes");
    expect(appSource).toContain("· this message");
    expect(appSource).toContain('aria-label="Don’t search Notes for queued message"');
    const remove = blockFrom(
      "const removeQueuedNoteSearchScope = useCallback",
      "}, [activeConversation, updateConversation]);"
    );
    expect(remove).toContain("queuedFollowupNoteSearchScopeOverride: false");
    expect(appSource).toContain("inline-flex min-h-11 max-w-full");
    expect(appSource).toContain("inline-flex size-11 items-center justify-center");
  });

  it("renders outside ConversationTranscript's narrow memo", () => {
    const wrapper = appSource.indexOf('className="chat-queued-followup chat-width-frame');
    const transcript = appSource.indexOf("<ConversationTranscript");
    const anchor = appSource.indexOf("<ChatContainerScrollAnchor />");
    expect(wrapper).toBeGreaterThan(transcript);
    expect(wrapper).toBeLessThan(anchor);
  });

  it("uses the non-submit tooltip class and a WCAG-passing eyebrow", () => {
    const running = blockFromComposer("{running ? (", 'aria-label="Stop response"');
    // Steer and Queue take the plain composer tooltip (ComposerTooltip's
    // default), never the send button's own class.
    expect(running).not.toContain("app-composer-submit-tooltip");
    const tooltipSource = readFileSync(
      path.join(process.cwd(), "src/components/composer/ComposerTooltip.tsx"),
      "utf8"
    );
    expect(tooltipSource).toMatch(/className = "app-composer-tooltip"/);
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
