/**
 * Type-to-focus: start typing anywhere in a chat and the composer takes it.
 *
 * The pain: you look at a chat, start typing your next prompt, and discover
 * several words later that nothing was focused and the text went nowhere.
 *
 * Almost all the risk here is in NOT stealing keystrokes that meant something
 * else, so most of this file is about what must keep working.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

/**
 * Absence assertions must read code, not prose: the comment explaining why this
 * code does NOT call preventDefault would otherwise fail the "no preventDefault"
 * check.
 */
const stripComments = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

const predicateBody = (): string => {
  const start = appSource.indexOf("const shouldTypingFocusComposer");
  expect(start, "shouldTypingFocusComposer is missing").toBeGreaterThan(-1);
  return appSource.slice(start, appSource.indexOf("\n};", start));
};

const effectBody = (): string => {
  const start = appSource.indexOf("  // Type-to-focus: start typing anywhere in a chat");
  expect(start, "type-to-focus effect is missing").toBeGreaterThan(-1);
  const end = appSource.indexOf("  }, [activePanel, authStatus, viewerOpen]);", start);
  expect(end, "type-to-focus effect end marker moved").toBeGreaterThan(start);
  return appSource.slice(start, end);
};

describe("type-to-focus captures the keystroke", () => {
  it("focuses synchronously and never preventDefaults", () => {
    // This is the whole mechanism. The keystroke's default action inserts into
    // whatever holds focus when it fires — after this handler returns — so a
    // synchronous focus means the character lands in the composer by itself.
    // preventDefault would drop it; manually inserting would type it twice.
    const effect = effectBody();
    expect(effect).toMatch(/textarea\.focus\(\{ preventScroll: true \}\)/);
    const code = stripComments(effect);
    expect(code).not.toMatch(/preventDefault/);
    expect(code).not.toMatch(/textarea\.value\s*[+]?=/);
  });

  it("does not route through focusComposerTextarea", () => {
    // That helper focuses inside a requestAnimationFrame — a whole frame too
    // late to catch the keystroke that triggered it.
    expect(stripComments(effectBody())).not.toMatch(/focusComposerTextarea\(\)/);
  });

  it("puts the caret at the end so it appends to an existing draft", () => {
    expect(effectBody()).toMatch(/setSelectionRange\(caret, caret\)/);
  });
});

describe("type-to-focus keeps its hands off everything else", () => {
  it("ignores every modifier combination", () => {
    // ⌘K, ⌘C, ⌘⇧E and friends must reach their handlers untouched. Plain Shift
    // is deliberately allowed — capitals start sentences.
    const body = predicateBody();
    expect(body).toMatch(/event\.metaKey \|\| event\.ctrlKey \|\| event\.altKey/);
    expect(stripComments(body)).not.toMatch(/event\.shiftKey/);
  });

  it("only reacts to printable characters, and never to Space", () => {
    // key.length === 1 admits printable characters and nothing else: Enter,
    // Tab, Escape, Backspace, arrows and F-keys all report longer names.
    // Space is excluded on purpose — it scrolls the transcript, and a prompt
    // essentially never starts with one.
    const body = predicateBody();
    expect(body).toMatch(/event\.key\.length !== 1 \|\| event\.key === " "/);
  });

  it("stands down during IME composition", () => {
    // Stealing focus mid-composition destroys in-progress CJK text.
    expect(predicateBody()).toMatch(/event\.isComposing/);
  });

  it("respects an already-handled event", () => {
    expect(predicateBody()).toMatch(/event\.defaultPrevented/);
  });

  it("never steals from another field, or from itself", () => {
    const effect = effectBody();
    expect(effect).toMatch(/isEditableEventTarget\(event\.target\)/);
    expect(effect).toMatch(/textarea === document\.activeElement/);
  });

  it("stands down while a dialog, menu or popover is open", () => {
    expect(effectBody()).toMatch(/hasBlockingOverlay\(\)/);
    const start = appSource.indexOf("const hasBlockingOverlay");
    expect(start).toBeGreaterThan(-1);
    const overlay = appSource.slice(start, appSource.indexOf("\n\n", start));
    expect(overlay).toMatch(/role="dialog"/);
    expect(overlay).toMatch(/role="alertdialog"/);
    expect(overlay).toMatch(/data-scroll-locked/);
  });

  it("does not fire on a disabled composer", () => {
    // A conversation that has not hydrated yet.
    expect(effectBody()).toMatch(/textarea\.disabled/);
  });

  it("only runs on the chat panel, when signed in", () => {
    // Resources, Training and Lens have their own inputs and their own
    // keyboard behaviour; the composer is not on screen there.
    expect(effectBody()).toMatch(/activePanel !== "chat"/);
    expect(effectBody()).toMatch(/authStatus !== "authenticated"/);
    // The Lens/file viewer overlays the chat; typing there must not warp
    // focus down into a composer the user cannot see.
    expect(effectBody()).toMatch(/viewerOpen/);
  });

  it("unsubscribes on cleanup", () => {
    expect(effectBody()).toMatch(/removeEventListener\("keydown", handleTypeToFocus\)/);
  });
});
