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

describe("dispatch — when spending a run is allowed", () => {
  const dispatch = () =>
    blockFrom("/* Dispatch: the enqueue contract.", "}, [activeConversation, setActivePromptValue, updateConversation]);");

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
