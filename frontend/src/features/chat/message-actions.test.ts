/**
 * Per-message action buttons: reveal, reach, and honesty.
 *
 * The defect this file exists to prevent was not cosmetic. Tailwind compiles
 * `group-hover:opacity-100` inside `@media (hover:hover)`, so on a touch device
 * the action row never became visible — while the coarse-pointer block grew it
 * to a 44px target. Every user message on a phone carried an invisible,
 * unlabelled, unconfirmed Delete that also removes the assistant's reply.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const codeBlock = readFileSync(
  path.join(process.cwd(), "src/components/prompt-kit/code-block.tsx"),
  "utf8"
);

const ruleFor = (selector: string): string => {
  const start = styles.indexOf(selector);
  expect(start, `missing CSS rule: ${selector}`).toBeGreaterThan(-1);
  return styles.slice(start, styles.indexOf("}", start));
};

/**
 * Absence assertions must read code, not prose. Several of these guards forbid a
 * string that the explanatory comment right above the fix necessarily quotes —
 * without stripping, the comment explaining why emerald is gone would itself
 * fail the "no emerald" check.
 */
const stripComments = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

const appCode = stripComments(appSource);
const codeBlockCode = stripComments(codeBlock);

describe("message actions reveal", () => {
  it("is visible on touch devices, where there is no hover to wait for", () => {
    // The whole point. Without this the row is permanently opacity:0 on phones.
    expect(styles).toMatch(
      /@media \(pointer: coarse\) \{\s*\.chat-message-actions \{\s*opacity: 1;/
    );
  });

  it("reveals on keyboard focus, not only pointer hover", () => {
    // Tabbing into an opacity-0 button previously opened a tooltip anchored to
    // nothing, with no visible control behind it.
    expect(styles).toMatch(/\.group:focus-within > \.chat-message-actions/);
  });

  it("no longer reveals via a Tailwind group-hover utility", () => {
    // That utility is the media-query trap. The reveal must stay in CSS we own.
    expect(appCode).not.toMatch(/group-hover:opacity-100/);
  });

  it("keeps the last turn's actions pinned without requiring a hover", () => {
    expect(styles).toMatch(/\.chat-message-actions\[data-pinned="true"\]/);
    expect(appSource).toMatch(/data-pinned=\{isLastMessage \|\| undefined\}/);
  });

  it("still grows the buttons to a 44px target on coarse pointers", () => {
    // Several coarse-pointer blocks exist; find the one that owns these buttons
    // rather than whichever happens to appear first in the file.
    const anchor = styles.indexOf(".chat-message-action,\n  .app-settings-close-button");
    expect(anchor, "coarse-pointer block no longer lists .chat-message-action").toBeGreaterThan(-1);
    const block = styles.slice(anchor, styles.indexOf("}", anchor));
    expect(block).toMatch(/min-height: 44px/);
  });
});

describe("message actions interaction", () => {
  it("uses the house raised-hover pattern rather than a wash nobody can see", () => {
    const hover = ruleFor(".chat-message-action:hover:not(:disabled) {");
    expect(hover).toMatch(/transform: translateY\(-1px\)/);
    expect(hover).toMatch(/box-shadow: 0 6px 16px/);
    // The border is what actually carries the signal; a 5% wash alone measures
    // barely above the surface it sits on.
    expect(hover).toMatch(/border-color: var\(--line\)/);
  });

  it("settles the lift on press — the lift is the invitation, not the receipt", () => {
    const active = ruleFor(".chat-message-action:active:not(:disabled) {");
    expect(active).toMatch(/transform: translateY\(0\)/);
    expect(active).toMatch(/box-shadow: none/);
  });

  it("gives dark mode its own shadow, since the light one is invisible there", () => {
    const dark = ruleFor(".dark .chat-message-action:hover:not(:disabled) {");
    expect(dark).toMatch(/rgba\(0, 0, 0, 0\.45\)/);
  });

  it("locates on focus without lifting", () => {
    const focus = ruleFor(".chat-message-action:focus-visible {");
    expect(focus).toMatch(/outline:/);
    expect(focus).not.toMatch(/translateY\(-1px\)/);
  });

  it("stays on the brand curve and off Tailwind's", () => {
    const rest = ruleFor(".chat-message-action {");
    expect(rest).toMatch(/var\(--motion-ease\)/);
    // Tailwind's transition-colors ships cubic-bezier(0.4, 0, 0.2, 1) — the one
    // off-brand curve that lived in this row, and one the motion guard cannot
    // see because it only parses `transition:` shorthands in styles.css.
    expect(appCode).not.toMatch(/transition-colors duration-150/);
  });
});

describe("message actions are labelled and honest", () => {
  it("gives every icon-only button an accessible name", () => {
    // A Radix tooltip supplies aria-describedby — a description, never a name.
    // Without these a screen reader announced anonymous buttons per message.
    // Edit and Delete carry literal labels; Copy's is an expression that flips
    // with state, so assert the strings themselves rather than the attribute.
    expect(appCode).toContain('aria-label="Edit message"');
    expect(appCode).toContain('aria-label="Delete message"');
    for (const label of ["Copy message", "Copy response"]) {
      expect(appCode).toContain(`"${label}"`);
    }
    // Copy flips its own label so the confirmation is announced, not just drawn.
    expect(appSource).toMatch(/aria-label=\{isCopied \? "Message copied"/);
    expect(appSource).toMatch(/aria-label=\{isCopied \? "Response copied"/);
  });

  it("marks Delete destructive by intent, with no red at rest", () => {
    expect(appSource).toMatch(/data-tone="destructive"/);
    const rest = ruleFor(".chat-message-action {");
    expect(rest).not.toMatch(/--danger|--destructive/);
    expect(styles).toMatch(
      /\.chat-message-action\[data-tone="destructive"\]:hover:not\(:disabled\)/
    );
  });

  it("routes Delete through a confirmation that names the blast radius", () => {
    // Deleting a user message also removes every consecutive assistant reply.
    expect(appSource).toMatch(/onRequestDeleteUserMessage\(message\.id\)/);
    expect(appSource).toContain("This also removes the reply it produced.");
    expect(appSource).toContain("Delete this message?");
  });

  it("promises exactly the deletion the backend performs", () => {
    // The old copy said "remove its messages from storage" while the handler ran
    // UPDATE control_threads SET status='deleted' and removed no row at all.
    // The handler now hard-deletes, so the strong wording is honest again — but
    // this pair must move together. If deletion ever softens back to a status
    // flip, this assertion is the thing that should fail.
    expect(appCode).not.toContain("remove its messages from storage");
    expect(appCode).toContain("This cannot be undone.");
  });

  it("has no unwired decoration left in the row", () => {
    // Upvote/Downvote had no onClick, no state, and no endpoint at any layer.
    expect(appCode).not.toMatch(/ThumbsUp|ThumbsDown/);
  });

  it("confirms copy with an ink step, never emerald", () => {
    // emerald-500/emerald-600/... class usages, not the word in prose.
    for (const source of [appCode, codeBlockCode]) {
      expect(source).not.toMatch(/emerald-\d/);
    }
    expect(styles).toMatch(/\.chat-message-action\[data-state="copied"\]/);
  });

  it("disables Copy while the answer is still streaming", () => {
    // Otherwise Copy captures a truncated message.
    expect(appSource).toMatch(/disabled=\{isStreamingAssistant\}/);
  });
});

describe("edit means one thing", () => {
  it("truncates the turn instead of only writing the composer draft", () => {
    // It used to be wired straight to setActivePromptValue: the original stayed,
    // the reply stayed, nothing re-ran, and sending produced a duplicate turn.
    expect(appCode).not.toMatch(/onEditUserMessage: setActivePromptValue/);
    expect(appSource).toMatch(/onEditUserMessage: handleEditUserMessage/);
    const start = appSource.indexOf("const handleEditUserMessage");
    expect(start).toBeGreaterThan(-1);
    const handler = appSource.slice(start, start + 5000);
    expect(handler).toMatch(/handleDeleteUserMessage\(messageId\)/);
    expect(handler).toMatch(/setActivePromptValue\(content\)/);
    // Focusing is what makes the edit visible as an edit.
    expect(handler).toMatch(/focusComposerTextarea\(\)/);
    expect(handler).toMatch(/messageToEdit\?\.noteReferences/);
    expect(handler).toMatch(/messageToEdit\?\.excludedNoteIntentText/);
    expect(handler).toMatch(/selectedNotes: noteReferences/);
    expect(handler).toMatch(/activeSelectionContext: noteSelectionContextForChips/);
    expect(handler).toMatch(/await resealTurnNotes\(conversationId, historicalReferences\)/);
    expect(handler.indexOf("await resealTurnNotes")).toBeLessThan(
      handler.indexOf("handleDeleteUserMessage(messageId)")
    );
    expect(handler).toMatch(
      /setPastedComposerTextForConversation\([\s\S]*historicalExcludedNoteIntentText/
    );
  });

  it("is undoable, and restores the draft it replaced", () => {
    const start = appSource.indexOf("const handleEditUserMessage");
    const handler = appSource.slice(start, start + 5000);
    expect(handler).toMatch(/showUndoToast/);
    // Editing silently clobbered whatever was already typed, then persisted it.
    expect(handler).toMatch(/previousDraft/);
    expect(handler).toMatch(/setActivePromptValue\(previousDraft\)/);
  });

  it("restores an exact snapshot on undo, not the last N messages", () => {
    const start = appSource.indexOf("const confirmDeleteUserMessage");
    expect(start).toBeGreaterThan(-1);
    const handler = appSource.slice(start, start + 1200);
    expect(handler).toMatch(/previousMessages/);
    expect(handler).toMatch(/showUndoToast/);
  });
});
