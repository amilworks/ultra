/**
 * Notes dual-mode — two modes, one file.
 *
 * The contract this suite pins (see the design mock):
 * - Markdown mode edits like a doc over a pure-markdown data structure; the
 *   styling layer is rendering, never hidden data Ultra can't see.
 * - Plaintext mode is the raw mono surface, untouched.
 * - The mode is per-note and sticky (editor_mode on the record, every layer).
 * - The ribbon is pinned to the basics — every button writes plain markdown.
 * - The one highlight is ==content==, shared with the chat renderer.
 * - Mobile is first-class: finger-sized ribbon, no iOS focus-zoom, and the
 *   list stays reachable behind a back chip.
 *
 * Engine fidelity (zero-edit ⇒ zero-diff) lives in its own runtime suite:
 * src/components/notes/markdownFidelity.test.ts.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const read = (relative: string): string =>
  readFileSync(path.join(process.cwd(), relative), "utf8");

const pageSource = read("src/components/NotesPage.tsx");
const editorSource = read("src/components/notes/MarkdownNoteEditor.tsx");
const highlightPluginSource = read("src/components/notes/notesHighlight.ts");
const dialectSource = read("src/components/notes/notesDialect.ts");
const remarkHighlightSource = read("src/lib/remarkHighlight.ts");
const markdownSource = read("src/components/prompt-kit/markdown.tsx");
const apiSource = read("src/lib/api.ts");
const styles = read("src/styles.css");
const mockApi = read("scripts/mock-api.mjs");
const openapi = read("../backend/controlplane/api/openapi.yaml");
const schema = read("../backend/controlplane/internal/store/schema.sql");
const notesHandlers = read("../backend/controlplane/internal/httpapi/notes.go");

describe("two modes, one file", () => {
  it("offers exactly Markdown and Plaintext, as a segmented control", () => {
    expect(pageSource).toContain('aria-label="Editor mode"');
    expect(pageSource).toMatch(/switchEditorMode\("markdown"\)/);
    expect(pageSource).toMatch(/switchEditorMode\("plaintext"\)/);
    // No third mode, no preview toggle — the styled surface IS the note.
    expect(pageSource).not.toContain("Toggle preview");
  });

  it("⌘⇧E flips the mode; ⌘E belongs to inline code now", () => {
    expect(pageSource).toMatch(
      /event\.shiftKey && event\.key\.toLowerCase\(\) === "e"[\s\S]{0,240}switchEditorMode/
    );
    // The old page-level plain-⌘E preview handler is gone.
    expect(pageSource).not.toMatch(/!event\.shiftKey && event\.key\.toLowerCase\(\) === "e"/);
    expect(pageSource).toContain('title="Code (⌘E)"');
  });

  it("the mode is per-note and sticky at every layer", () => {
    // Draft carries it; the single save channel persists it.
    expect(pageSource).toMatch(/editor_mode: draft\.editorMode/);
    expect(pageSource).toMatch(/editorMode: record\.editor_mode/);
    // API type, mock harness, HTTP validation, and column all agree.
    expect(apiSource).toContain('NoteEditorMode = "markdown" | "plaintext"');
    expect(mockApi).toMatch(/editor_mode: "plaintext"/);
    expect(notesHandlers).toContain("NoteEditorModePlaintext");
    expect(schema).toContain(
      "ALTER TABLE control_notes ADD COLUMN IF NOT EXISTS editor_mode text NOT NULL DEFAULT 'markdown';"
    );
    expect(openapi).toContain("editor_mode: { type: string, enum: [markdown, plaintext] }");
  });

  it("a deliberate mode flip persists immediately, not on the debounce", () => {
    expect(pageSource).toMatch(
      /const switchEditorMode = useCallback\([\s\S]{0,800}void flushSave\(\)/
    );
  });
});

describe("the zero-rewrite law", () => {
  it("markdown leaves the editor only on real doc changes", () => {
    expect(editorSource).toMatch(/markdown !== prevMarkdown/);
  });

  it("the page trims the serializer's trailing newline so round-trips are byte-stable", () => {
    expect(pageSource).toContain('markdown.replace(/\\n$/, "")');
  });

  it("serialization goes through the shared house dialect", () => {
    expect(editorSource).toContain("withNotesDialect");
    expect(dialectSource).toContain('bullet: "-"');
    expect(dialectSource).toContain("highlight: highlightToMarkdown");
    // Intraword underscores are scientific vocabulary, not emphasis.
    expect(dialectSource).toContain("INTRAWORD_ESCAPED_UNDERSCORE");
  });
});

describe("the ribbon — pinned to the basics", () => {
  it("carries exactly the formalized controls, labeled with their shortcuts", () => {
    for (const [label, shortcut] of [
      ["Bold", "⌘B"],
      ["Italic", "⌘I"],
      ["Strikethrough", "⌘⇧X"],
      ["Highlight", "⌘⇧H"],
      ["Link", "⌘K"],
      ["Quote", "⌘⇧9"],
      ["Code", "⌘E"],
      ["Bulleted list", "⌘⇧8"],
      ["Numbered list", "⌘⇧7"],
    ] as const) {
      expect(pageSource).toContain(`aria-label="${label}"`);
      expect(pageSource).toContain(`(${shortcut})`);
    }
    expect(pageSource).toContain('aria-label="Text size"');
    expect(pageSource).toContain('aria-label="Table"');
    expect(pageSource).toContain('aria-label="Attach image or video"');
    // Said no to, on purpose: no color palette, no alignment buttons.
    expect(pageSource).not.toMatch(/aria-label="Align/);
    expect(pageSource).not.toMatch(/color-palette|text-color/i);
  });

  it("toolbar clicks keep the editor's focus and selection", () => {
    expect(pageSource).toContain("onMouseDown={keepEditorFocus}");
  });

  it("the GDocs muscle-memory shortcuts are bound in the editor", () => {
    for (const shortcut of ["Mod-Shift-x", "Mod-Shift-7", "Mod-Shift-8", "Mod-Shift-9"]) {
      expect(editorSource).toContain(`"${shortcut}"`);
    }
    // Tab walks table cells, GROWS the table by a row from the last cell
    // (Docs/Word basic), and nests list items — one chained binding.
    expect(editorSource).toMatch(
      /goToNextTableCellCommand\.key\) \|\|[\s\S]{0,300}addRowAfterCommand\.key\)[\s\S]{0,200}sinkListItemCommand\.key\)/
    );
  });

  it("LaTeX is first-class: math plugin loaded, chat-dialect storage, KaTeX css in the chunk", () => {
    expect(editorSource).toContain('.use(notesMath)');
    expect(editorSource).toContain('import "katex/dist/katex.min.css"');
    const mathSource = read("src/components/notes/notesMath.ts");
    expect(mathSource).toContain('"inlineMath"');
    expect(mathSource).toContain('=== "math"');
    expect(mathSource).toContain("remarkMath");
    // Click-to-edit: the atom flips to a raw-TeX field in place.
    expect(mathSource).toContain("startEditing");
  });

  it("pastes are sanitized before ProseMirror parses them", () => {
    expect(editorSource).toContain("transformPastedHTML: sanitizePastedHtml");
    const pasteSource = read("src/components/notes/notesPaste.ts");
    expect(pasteSource).toContain('annotation[encoding="application/x-tex"]');
    expect(pasteSource).toContain("data-math-block");
    expect(pasteSource).toContain('input[type="checkbox"]');
    expect(pasteSource).toContain(".pk-code-render");
  });
});

describe("==highlight== is content, shared everywhere", () => {
  it("one remark plugin feeds both the editor and the chat renderer", () => {
    expect(highlightPluginSource).toContain('from "@/lib/remarkHighlight"');
    expect(markdownSource).toMatch(/remarkGfm, remarkHighlight/);
  });

  it("the wash is one amber token, themed for both modes", () => {
    expect(styles).toMatch(/:root \{\s*--highlight-wash/);
    expect(styles).toMatch(/\.dark \{\s*--highlight-wash/);
    const markRule = styles.match(/\.pk-message-content mark\s*\{[^}]*\}/s)?.[0];
    expect(markRule).toContain("var(--highlight-wash)");
  });

  it("flanking rules stay conservative — comparisons never light up", () => {
    expect(remarkHighlightSource).toContain("(^|[^=\\\\w])==");
    // The renderer-side pattern avoids lookbehind for older Safari.
    expect(remarkHighlightSource).not.toMatch(/HIGHLIGHT_SOURCE = "[^"]*\(\?<!/);
  });
});

describe("bundle discipline", () => {
  it("the editor chunk is lazy; the page imports only types from it", () => {
    expect(pageSource).toContain(
      'lazyNamedWithRetry(\n  () => import("@/components/notes/MarkdownNoteEditor"),\n  "MarkdownNoteEditor"\n)'
    );
    const staticImports = pageSource.match(
      /^import (?!type)[^;]*from "@\/components\/notes\/MarkdownNoteEditor"/gm
    );
    expect(staticImports).toBeNull();
  });
});

describe("mobile is first-class", () => {
  it("the notes list stays reachable on phones via the back chip", () => {
    expect(pageSource).toContain('className="notes-mobile-back"');
    expect(pageSource).toContain("data-mobile-list=");
    expect(styles).toMatch(
      /\.notes-page\[data-mobile-list="true"\] \.notes-list \{ display: flex; \}/
    );
    expect(styles).toMatch(
      /\.notes-page\[data-mobile-list="true"\] \.notes-editor \{ display: none; \}/
    );
  });

  it("no iOS focus-zoom: both editing surfaces hold 16px on phones", () => {
    const mobile = styles.slice(styles.indexOf('.notes-page { grid-template-columns: 1fr; }'));
    expect(mobile).toMatch(/\.notes-body-input \{ font-size: 16px; \}/);
    expect(mobile).toMatch(/\.notes-md-prose \{ font-size: 16px; \}/);
  });

  it("the ribbon becomes a finger-sized, sideways-scrolling toolbar", () => {
    const mobile = styles.slice(styles.indexOf('.notes-page { grid-template-columns: 1fr; }'));
    expect(mobile).toMatch(/\.notes-ribbon \{[^}]*overflow-x: auto;/s);
    expect(mobile).toMatch(/\.notes-ribbon-btn \{ min-width: 40px; height: 40px;/);
    // Double-tap zoom is disarmed on the buttons themselves.
    const ribbonBtn = styles.match(/\.notes-ribbon-btn\s*\{[^}]*\}/s)?.[0];
    expect(ribbonBtn).toContain("touch-action: manipulation;");
  });

  it("uploads keep working where drag-drop does not exist — the attach button rides the same picker", () => {
    expect(pageSource).toMatch(/aria-label="Attach image or video"[\s\S]{0,200}filePickerRef\.current\?\.\(\)/);
  });
});
