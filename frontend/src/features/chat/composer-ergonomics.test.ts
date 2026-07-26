/**
 * Composer ergonomics: paste-to-focus, huge-paste-to-attachment, ArrowUp
 * prompt recall, and the ask-about-selection chip.
 *
 * As with type-to-focus, most of the risk is in stealing an interaction that
 * meant something else, so most of these guards pin what must NOT happen.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

const stripComments = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

const blockFrom = (start: string, end: string): string => {
  const startIndex = appSource.indexOf(start);
  expect(startIndex, `missing block: ${start.slice(0, 60)}`).toBeGreaterThan(-1);
  const endIndex = appSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated block: ${start.slice(0, 60)}`).toBeGreaterThan(startIndex);
  return appSource.slice(startIndex, endIndex + end.length);
};

describe("paste-to-focus", () => {
  const effect = () =>
    blockFrom("/* Paste-to-focus: ⌘V anywhere in a chat", 'window.addEventListener("paste", handleChatPaste);');

  it("never intercepts a paste aimed at a real input", () => {
    // When the composer is focused its own onPaste owns the event; when any
    // other field is focused the paste belongs to that field.
    expect(effect()).toMatch(/isEditableEventTarget\(event\.target\)/);
    expect(effect()).toMatch(/hasBlockingOverlay\(\)/);
  });

  it("routes files through the same attach path as the composer's own paste", () => {
    expect(effect()).toMatch(/filesFromClipboard\(event\.clipboardData\)/);
    expect(effect()).toMatch(/attachFilesToActiveConversation\(pastedFiles\)/);
  });

  it("appends pasted text below the draft, never glued onto the last word", () => {
    expect(effect()).toMatch(/\\s\$\/\.test\(previous\)/);
    expect(effect()).toMatch(/`\$\{previous\}\\n\$\{pastedText\}`/);
  });

  it("stays inside the chat panel and out of the viewer", () => {
    expect(effect()).toMatch(/activePanel !== "chat" \|\| viewerOpen/);
  });

  it("does not run against a composer that cannot accept it", () => {
    expect(effect()).toMatch(/textarea\.disabled/);
  });
});

describe("huge paste becomes an attachment", () => {
  it("applies in BOTH paste paths — focused composer and global", () => {
    // The conversion must not depend on where focus happened to be.
    const occurrences = appSource.match(/shouldAttachPastedText\(/g) ?? [];
    expect(occurrences.length).toBeGreaterThanOrEqual(3); // both paste paths + selection
  });

  it("reads plain text, never rich markup", () => {
    expect(appSource).toMatch(/getData\("text\/plain"\)/);
    expect(stripComments(appSource)).not.toMatch(/getData\("text\/html"\)/);
  });

  it("converts silently — the chip is the feedback, not a toast", () => {
    const helper = blockFrom("const attachPastedText = useCallback", "[attachFilesToActiveConversation]");
    expect(stripComments(helper)).not.toMatch(/showSuccessToast|showErrorToast/);
    expect(helper).toMatch(/pastedTextFile\(text\)/);
  });
});

describe("ArrowUp recalls the last prompt", () => {
  const branch = () =>
    blockFrom("// ArrowUp in an EMPTY composer recalls the last prompt", "setActivePromptValue(lastPrompt);");

  it("fires only in a genuinely empty composer", () => {
    // Any drafted text means the user is cursoring, not recalling.
    expect(branch()).toMatch(/!activePrompt\.trim\(\)/);
  });

  it("stays away from modifiers, IME, and the resource picker", () => {
    const body = branch();
    expect(body).toMatch(/!event\.nativeEvent\.isComposing/);
    expect(body).toMatch(/!event\.metaKey/);
    expect(body).toMatch(/!event\.shiftKey/);
    expect(body).toMatch(/!composerResourcePickerOpen/);
  });

  it("recalls the text only — never the destructive edit machinery", () => {
    // handleEditUserMessage removes a turn and its reply. Far too much
    // consequence to hang off an arrow key.
    expect(stripComments(branch())).not.toMatch(/handleEditUserMessage|handleDeleteUserMessage/);
    expect(branch()).toMatch(/setActivePromptValue\(lastPrompt\)/);
  });

  it("walks back to the latest USER message, not the assistant's answer", () => {
    expect(branch()).toMatch(/role === "user"/);
  });
});

describe("ask-about-selection", () => {
  it("captures the text at show time, before the click can collapse it", () => {
    const effect = blockFrom("/* Ask-about-selection: highlight transcript text", "useEffect(() => {");
    expect(effect).toMatch(/text: string;/);
    // And the chip's mousedown is neutralised as the second line of defence.
    expect(appSource).toMatch(/onMouseDown=\{\(event\) => \{\s*event\.preventDefault\(\);\s*\}\}/);
  });

  it("only offers itself on transcript text", () => {
    expect(appSource).toMatch(/closest\("\.pk-message"\)/);
  });

  it("shows on mouseup, hides on collapse — never flickers mid-drag", () => {
    const effect = blockFrom("const measureSelection = ", 'window.addEventListener("mouseup", reveal);');
    expect(effect).toMatch(/selectionchange/);
    expect(appSource).toMatch(/window\.addEventListener\("mouseup", reveal\)/);
  });

  it("quotes as a visible markdown blockquote the user can edit", () => {
    expect(appSource).toMatch(/draftWithQuotedSelection\(previous, ask\.text\)/);
  });

  it("captures selections KaTeX-aware — formulas quote as TeX, not glyph soup", () => {
    // selection.toString() on rendered math yields the visible layer's glue
    // characters; textFromSelection swaps each full render for the original
    // TeX in its annotation node. Unit-tested in lib/selection-capture.test.ts.
    expect(appSource).toMatch(/const text = textFromSelection\(selection\)/);
    expect(stripComments(appSource)).not.toMatch(/const text = selection\.toString\(\)/);
  });

  it("routes an oversized selection through the attachment path", () => {
    const callback = blockFrom("const askAboutSelection = useCallback", "focusComposerTextarea();");
    expect(callback).toMatch(/shouldAttachPastedText\(ask\.text\)/);
    expect(callback).toMatch(/attachPastedText\(ask\.text\)/);
  });

  it("is portaled, named by its visible text, and dismissed by Escape", () => {
    expect(appSource).toMatch(/createPortal\(/);
    // WCAG 2.5.3: the visible "Ask about this" IS the accessible name — an
    // aria-label overriding visible text breaks voice-control users.
    expect(stripComments(appSource)).not.toContain('aria-label="Ask about the selected text"');
    expect(appSource).toMatch(/event\.key === "Escape" \|\| isEditableEventTarget\(event\.target\)/);
  });

  it("collapses the selection when consumed — the resurrection fix", () => {
    // Review-confirmed: without this, Escape's own keyup (and the chip click's
    // own mouseup) re-measured the still-live selection and re-showed the chip
    // one frame after every dismissal.
    const callback = blockFrom("const askAboutSelection = useCallback", "focusComposerTextarea();");
    expect(callback).toMatch(/window\.getSelection\(\)\?\.removeAllRanges\(\)/);
  });

  it("guards the keyup reveal symmetrically with the keydown dismiss", () => {
    const reveal = blockFrom("const reveal = (event: Event): void =>", "window.requestAnimationFrame");
    expect(reveal).toMatch(/event\.key === "Escape" \|\| isEditableEventTarget\(event\.target\)/);
    expect(reveal).toMatch(/closest\("\.chat-selection-ask"\)/);
  });

  it("offers itself on fine pointers only — touch keeps the native callout", () => {
    expect(appSource).toMatch(/!window\.matchMedia\("\(pointer: fine\)"\)\.matches/);
  });

  it("stands down off-viewport, under overlays, and against a disabled composer", () => {
    const measure = blockFrom("const measureSelection = ", "const reveal =");
    expect(measure).toMatch(/rect\.bottom < 0/);
    expect(measure).toMatch(/hasBlockingOverlay\(\)/);
    expect(measure).toMatch(/composerTextareaRef\.current\?\.disabled/);
  });

  it("sits below every Radix overlay and selects only KaTeX's visible layer", () => {
    const chipRule = styles.slice(
      styles.indexOf(".chat-selection-ask {"),
      styles.indexOf("}", styles.indexOf(".chat-selection-ask {"))
    );
    expect(chipRule).toMatch(/z-index: 40/);
    expect(styles).toMatch(/\.katex-mathml \{\s*user-select: none/);
  });

  it("dresses in the house language: quiet rest, brand motion, dark shadow, reduced-motion opt-out", () => {
    const rule = styles.slice(
      styles.indexOf(".chat-selection-ask {"),
      styles.indexOf("@media (prefers-reduced-motion: reduce) {", styles.indexOf(".chat-selection-ask {"))
    );
    expect(rule).toMatch(/color: var\(--text-muted\)/);
    expect(rule).toMatch(/var\(--motion-ease\)/);
    expect(rule).toMatch(/border-radius: 10px/);
    expect(styles).toMatch(/\.dark \.chat-selection-ask \{\s*box-shadow: 0 6px 16px rgba\(0, 0, 0, 0\.45\)/);
    const reducedIndex = styles.indexOf("@media (prefers-reduced-motion: reduce) {", styles.indexOf(".chat-selection-ask {"));
    expect(styles.slice(reducedIndex, reducedIndex + 300)).toMatch(/\.chat-selection-ask \{\s*animation: none/);
  });
});
