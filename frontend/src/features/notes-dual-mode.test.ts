/**
 * Notes writing surface — progressive disclosure over one durable file.
 *
 * The contract this suite pins (see the design mock):
 * - Markdown mode edits like a doc over a pure-markdown data structure; the
 *   styling layer is rendering, never hidden data Ultra can't see.
 * - Plaintext mode is the raw mono surface, untouched.
 * - The mode is per-note and sticky (editor_mode on the record, every layer).
 * - Formatting is contextual; the document is not permanently crowded by a
 *   ribbon. Every command still writes plain markdown.
 * - The one highlight is ==content==, shared with the chat renderer.
 * - Mobile is first-class: finger-sized actions, no iOS focus-zoom, and the
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

describe("one file, progressively disclosed source mode", () => {
  it("keeps formatted writing primary and raw Markdown in the overflow", () => {
    expect(pageSource).toContain('aria-label="More note actions"');
    expect(pageSource).toContain("Edit Markdown source");
    expect(pageSource).toContain("Return to formatted editor");
    expect(pageSource).not.toContain('aria-label="Editor mode"');
    expect(pageSource).not.toContain("Toggle preview");
  });

  it("keeps raw source in the explicit menu and leaves the shell Resources shortcut alone", () => {
    expect(pageSource).not.toMatch(
      /event\.shiftKey && event\.key\.toLowerCase\(\) === "e"[\s\S]{0,240}switchEditorMode/
    );
    expect(pageSource).not.toMatch(/!event\.shiftKey && event\.key\.toLowerCase\(\) === "e"/);
    expect(pageSource).toContain('aria-label="Inline code"');
  });

  it("the mode is per-note and sticky at every layer", () => {
    // Draft carries it; the single save channel persists it.
    expect(pageSource).toMatch(/editor_mode: snapshot\.editorMode/);
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

describe("contextual writing controls", () => {
  it("puts inline formatting at the selection and blocks behind slash", () => {
    for (const label of ["Bold", "Italic", "Highlight", "Inline code", "Link"] as const) {
      expect(pageSource).toContain(`aria-label="${label}"`);
    }
    expect(pageSource).toContain('className="notes-selection-toolbar"');
    expect(pageSource).not.toContain('className="notes-ribbon"');
    for (const block of ["Heading", "Quote", "Bulleted list", "Table", "Code"]) {
      expect(pageSource).toContain(`label: "${block}"`);
    }
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

  it("wires the advertised highlight shortcut on the formatted surface", () => {
    expect(pageSource).toMatch(
      /event\.shiftKey &&[\s\S]{0,80}event\.key\.toLowerCase\(\) === "h"[\s\S]{0,180}exec\("highlight"\)/
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

describe("editor polish — tables, exits, checkboxes, code chip (user findings)", () => {
  it("the caret in a table reveals labeled row/column controls", () => {
    for (const label of ["+ Row", "− Row", "+ Column", "− Column", "Delete table"]) {
      expect(pageSource).toContain(`>${label}</button>`);
    }
    expect(pageSource).toContain("editorActive.inTable ?");
    for (const action of ["rowBelow", "rowDelete", "colRight", "colDelete", "tableDelete"]) {
      expect(editorSource).toContain(`case "${action}"`);
    }
  });

  it("there is always a way out of a code block — trailing paragraph, gap cursor, ⌘⏎", () => {
    expect(editorSource).toContain(".use(cursor)");
    expect(editorSource).toContain(".use(trailing)");
    expect(editorSource).toMatch(/"Mod-Enter"[\s\S]{0,300}exitCode/);
    expect(styles).toContain(".notes-md-prose .ProseMirror-gapcursor");
  });

  it("task checkboxes are semantic, keyboard-operable, and draw with crisp borders", () => {
    expect(editorSource).toContain("handleClick");
    expect(editorSource).toContain("handleKeyDown");
    expect(editorSource).toContain("checked: !item.attrs.checked");
    expect(editorSource).toContain('role: "checkbox"');
    expect(editorSource).toContain('"aria-checked"');
    expect(editorSource).toContain('tabindex: "0"');
    expect(editorSource).toMatch(/event\.key !== " " && event\.key !== "Enter"/);
    // Gutter clicks resolve BETWEEN items — the clicked item is nodeAfter.
    expect(editorSource).toContain("$pos.nodeAfter");
    expect(styles).toContain(
      '.notes-md-prose li[data-item-type="task"][role="checkbox"]:focus-visible'
    );
    const tick = styles.match(/data-checked="true"\]::after\s*\{[^}]*\}/s)?.[0];
    expect(tick).toContain("border-right");
    expect(tick).toContain("rotate(");
    expect(tick).not.toContain('content: "✓"');
  });

  it("inline code wears the violet chip in both voices, from one token pair", () => {
    expect(styles).toMatch(/:root \{\s*--inline-code-bg/);
    expect(styles).toMatch(/\.dark \{\s*--inline-code-bg/);
    const pk = styles.match(/^\.pk-inline-code\s*\{[^}]*\}/ms)?.[0];
    expect(pk).toContain("var(--inline-code-bg)");
    expect(pk).toContain("var(--inline-code-ink)");
    const notes = styles.match(/\.notes-md-prose :not\(pre\) > code\s*\{[^}]*\}/s)?.[0];
    expect(notes).toContain("var(--inline-code-ink)");
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

  it("keeps visible mobile actions finger-sized without restoring a permanent ribbon", () => {
    const mobile = styles.slice(styles.indexOf('.notes-page { grid-template-columns: 1fr; }'));
    expect(mobile).toMatch(/\.notes-icon-button \{[^}]*width: 44px;[^}]*height: 44px;/s);
    expect(mobile).toMatch(/\.notes-mobile-back \{[^}]*min-height: 44px;/s);
    expect(styles).not.toContain(".notes-ribbon");
  });

  it("uploads keep working where drag-drop does not exist — overflow rides the same picker", () => {
    expect(pageSource).toContain("<FilePickerBridge openRef={filePickerRef} />");
    expect(pageSource).toContain("filePickerRef.current?.()");
    expect(pageSource).toContain("Upload a file");
  });
});
