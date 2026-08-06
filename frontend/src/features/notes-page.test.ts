/**
 * Notes Phase 1 — the frictionless contract.
 *
 * Notes exists to sit next to the work and help it. Phase 1 is the page:
 * owner-scoped CRUD, markdown as the source of truth, and a surface where
 * nothing stands between "opened Notes" and "writing" — no dialogs, no save
 * buttons, autosave that flushes on every exit path.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const pageSource = readFileSync(
  path.join(process.cwd(), "src/components/NotesPage.tsx"),
  "utf8"
);
const apiSource = readFileSync(path.join(process.cwd(), "src/lib/api.ts"), "utf8");
const navSource = readFileSync(path.join(process.cwd(), "src/lib/navUrl.ts"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const mockApi = readFileSync(path.join(process.cwd(), "scripts/mock-api.mjs"), "utf8");
const markdownSource = readFileSync(
  path.join(process.cwd(), "src/components/prompt-kit/markdown.tsx"),
  "utf8"
);
const editorSource = readFileSync(
  path.join(process.cwd(), "src/components/notes/MarkdownNoteEditor.tsx"),
  "utf8"
);
const ultraLibSource = readFileSync(
  path.join(process.cwd(), "src/lib/ultraResource.ts"),
  "utf8"
);

describe("navigation", () => {
  it("sits directly below Resources in the sidebar with its own shortcut", () => {
    const resourcesIndex = appSource.indexOf("<span>Resources</span>");
    const notesIndex = appSource.indexOf("<span>Notes</span>");
    const trainingIndex = appSource.indexOf("Training dashboard (⌘+Shift+T)");
    expect(resourcesIndex).toBeGreaterThan(-1);
    expect(notesIndex).toBeGreaterThan(resourcesIndex);
    expect(notesIndex).toBeLessThan(trainingIndex);
    expect(appSource).toContain('const NOTES_SHORTCUT_KEY = "u";');
    expect(appSource).toContain('title="Notes (⌘+Shift+U)"');
  });

  it("participates in URL-as-state so Back/refresh keep the panel", () => {
    expect(navSource).toContain('notes: "notes",');
    expect(navSource).toMatch(/NavPanel = "chat" \| "resources" \| "notes"/);
  });

  it("lazy-loads the page like every other panel", () => {
    expect(appSource).toContain(
      'const LazyNotesPage = lazyNamed(() => import("./components/NotesPage"), "NotesPage");'
    );
  });
});

describe("frictionless editing", () => {
  it("creates a note and lands focus in the title — no dialogs anywhere", () => {
    expect(pageSource).toContain("titleRef.current?.focus()");
    expect(pageSource).not.toMatch(/<Dialog/);
  });

  it("auto-creates the very first note instead of an empty lecture", () => {
    expect(pageSource).toContain("autoCreatedRef");
    expect(pageSource).toMatch(/items\.length === 0[\s\S]{0,120}void createNote\(\)/);
  });

  it("title Enter/Tab drops straight into the body — whichever surface is active", () => {
    expect(pageSource).toMatch(
      /event\.key === "Enter" \|\| event\.key === "Tab"[\s\S]{0,320}bodyRef\.current\?\.focus\(\)/
    );
    expect(pageSource).toMatch(
      /event\.key === "Enter" \|\| event\.key === "Tab"[\s\S]{0,320}editorApiRef\.current\?\.focus\(\)/
    );
  });

  it("autosaves on debounce AND flushes on blur, note switch, and unmount", () => {
    expect(pageSource).toContain("AUTOSAVE_DEBOUNCE_MS = 800");
    expect(pageSource).toContain('onBlur={() => void flushSave()}');
    // Switch: openNote flushes the previous draft before loading the next.
    expect(pageSource).toMatch(
      /draftRef\.current\.noteId !== noteId[\s\S]{0,80}await flushSave\(\)/
    );
    // Unmount: the cleanup effect flushes.
    expect(pageSource).toMatch(/return \(\) => \{[\s\S]{0,200}void flushSave\(\);\s*\};\s*\}, \[flushSave\]\);/);
  });

  it("has no explicit save button — autosave IS the save", () => {
    expect(pageSource).not.toMatch(/>\s*Save\s*</);
  });

  it("the Markdown surface wears the chat reading voice — no separate preview exists", () => {
    // The editable ProseMirror root carries pk-message-content, so a note's
    // tables and code read exactly like an answer's while being edited.
    expect(editorSource).toContain('class: "pk-message-content pk-markdown notes-md-prose"');
    // Preview retired: the page no longer renders through the react-markdown
    // preview path. (LazyMarkdownNoteEditor is the editor, not a preview.)
    expect(pageSource).not.toContain('from "@/components/prompt-kit/lazy-markdown"');
    expect(pageSource).not.toContain("notes-preview");
  });

  it("opens the slash menu only at line starts and inserts markdown blocks", () => {
    expect(pageSource).toContain('if (event.key === "/")');
    expect(pageSource).toContain("caret === lineStart");
    for (const block of ["Heading", "To-do list", "Table", "Code", "Divider"]) {
      expect(pageSource).toContain(`label: "${block}"`);
    }
  });

  it("deletes with an inline two-step, never a browser confirm", () => {
    expect(pageSource).toContain("Really delete");
    expect(pageSource).not.toContain("window.confirm");
  });
});

describe("plumbing", () => {
  it("client methods cover the whole owner-scoped surface", () => {
    for (const method of ["listNotes", "createNote", "getNote", "updateNote", "deleteNote"]) {
      expect(apiSource).toContain(`async ${method}(`);
    }
  });

  it("styles stay in tokens, and the page collapses to the editor on phones", () => {
    const chip = styles.match(/\.notes-row-title\s*\{[^}]*\}/s)?.[0];
    expect(chip).toContain("var(--sidebar-nav-foreground)");
    expect(styles).toMatch(/@media \(max-width: 720px\)[\s\S]{0,200}\.notes-page \{ grid-template-columns: 1fr; \}/);
  });

  it("the harness serves notes so the page can be driven end to end", () => {
    expect(mockApi).toContain('url.pathname === "/v2/notes"');
    expect(mockApi).toContain("note_seed_protocol");
  });
});

describe("media in notes — one pipeline, one catalog", () => {
  it("drops, pastes, and slash-picks all ride apiClient.uploadFiles — the SAME pipeline as chat, so files land in Resources", () => {
    expect(pageSource).toContain("apiClient.uploadFiles(files)");
    expect(pageSource).toContain("<FileUpload");
    expect(pageSource).toContain("onPaste={handleBodyPaste}");
    expect(pageSource).toContain('id: "media", label: "Image or video"');
    // No parallel upload endpoint, no note-private storage.
    expect(pageSource).not.toContain("uploadNoteMedia");
  });

  it("stores portable ultra:// references, never absolute URLs", () => {
    expect(ultraLibSource).toContain("ultra://resource/");
    expect(pageSource).toMatch(/markdownForUpload/);
    expect(pageSource).not.toMatch(/insertAtCaret\(`\\n!\[.*http/);
  });

  it("renders video references as a native player and images inline, resolved through the Resources download URL", () => {
    expect(editorSource).toContain("VIDEO_EXTENSION_PATTERN.test(name)");
    expect(editorSource).toContain('"notes-media-video"');
    expect(editorSource).toContain("video.controls = true");
    expect(pageSource).toContain("apiClient.resourceDownloadUrl(fileId)");
  });

  it("whispers upload progress in the same voice as autosave", () => {
    expect(pageSource).toMatch(/Uploading \{uploadingCount\} file/);
  });

  it("styles media with tokens — hairline border, house radius", () => {
    const img = styles.match(/\.notes-media-img\s*\{[^}]*\}/s)?.[0];
    const video = styles.match(/\.notes-media-video\s*\{[^}]*\}/s)?.[0];
    expect(img).toContain("var(--line)");
    expect(img).toContain("var(--radius)");
    expect(video).toContain("var(--line)");
    expect(video).toContain("max-width: 100%");
  });
});

describe("the ultra:// scheme in shared markdown", () => {
  it("passes through the first-party scheme instead of blanking it, everything else stays sanitized", () => {
    expect(markdownSource).toContain("defaultUrlTransform");
    expect(markdownSource).toMatch(/url\.startsWith\("ultra:\/\/"\) \? url : defaultUrlTransform\(url\)/);
    expect(markdownSource).toContain("urlTransform={ultraUrlTransform}");
  });
});

describe("plaintext mode is the raw source — and the type says so", () => {
  it("body edits in the house mono; Markdown mode flips to the reading face", () => {
    const body = styles.match(/\.notes-body-input\s*\{[^}]*\}/s)?.[0];
    expect(body).toContain('"JetBrains Mono"');
    expect(body).toContain("font-variant-ligatures: none;");
    expect(body).toContain("tab-size: 2;");
    // The typographic flip IS the mode signal: the other surface reads like
    // a chat answer.
    expect(editorSource).toContain('class: "pk-message-content pk-markdown notes-md-prose"');
  });

  it("Tab indents inside the body instead of escaping the editor", () => {
    expect(pageSource).toMatch(/event\.key === "Tab" && !event\.shiftKey[\s\S]{0,600}setSelectionRange\(start \+ 2/);
  });

  it("text pastes pass straight through — only FILE pastes are intercepted", () => {
    expect(pageSource).toMatch(/clipboardData\?\.files[\s\S]{0,120}files\.length > 0[\s\S]{0,60}preventDefault/);
  });
});
