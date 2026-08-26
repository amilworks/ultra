/**
 * Notes — the writing-first contract.
 *
 * Notes exists to sit next to the work and help it. Phase 1 is the page:
 * owner-scoped CRUD, markdown as the source of truth, and a surface where
 * nothing stands between "new note" and writing. Blank drafts stay local,
 * autosave is serialized, and sync failures remain explicit and retryable.
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
  it("starts a body-focused local draft without creating a server note", () => {
    expect(pageSource).toContain('LOCAL_DRAFT_ID = "__ultra_local_note_draft__"');
    expect(pageSource).toContain("pendingBodyFocusRef.current = true");
    expect(pageSource).toContain("editorApiRef.current?.focus()");
    expect(pageSource).not.toContain("apiClient.createNote({});");
  });

  it("never creates blank list clutter; the empty state offers one direct action", () => {
    expect(pageSource).not.toContain("autoCreatedRef");
    expect(pageSource).toContain("!meaningfulDraft(currentDraft)");
    expect(pageSource).toContain("Write your first note");
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
    expect(pageSource).toContain("AUTOSAVE_DEBOUNCE_MS = 700");
    expect(pageSource).toContain("onBlur={handleEditorBlur}");
    // Switch: openNote flushes the previous draft before loading the next.
    expect(pageSource).toMatch(
      /draftRef\.current\.noteId !== noteId[\s\S]{0,260}await flushSave\(\)/
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

  it("confirms permanent deletion with an accessible alert dialog", () => {
    expect(pageSource).toContain("<AlertDialog");
    expect(pageSource).toContain("Deletion is permanent");
    expect(pageSource).not.toContain("window.confirm");
  });
});

describe("plumbing", () => {
  it("client methods cover the whole owner-scoped surface", () => {
    for (const method of ["listNotes", "createNote", "getNote", "updateNote", "deleteNote"]) {
      expect(apiSource).toContain(`async ${method}(`);
    }
  });

  it("uses revision-aware writes so autosave cannot overwrite a newer browser", () => {
    expect(apiSource).toContain("expected_revision?: number");
    expect(pageSource).toContain("expected_revision: snapshot.revision");
    expect(pageSource).toContain("isNoteRevisionConflict(error)");
    expect(mockApi).toContain("payload.expected_revision !== note.revision");
    expect(mockApi).toContain('code: "note_revision_conflict"');
  });

  it("styles stay in tokens, and the page collapses to the editor on phones", () => {
    const chip = styles.match(/\.notes-row-title\s*\{[^}]*\}/s)?.[0];
    expect(chip).toContain("var(--sidebar-nav-foreground)");
    expect(styles).toMatch(/@media \(max-width: 960px\)[\s\S]{0,200}\.notes-page \{ grid-template-columns: 1fr; \}/);
  });

  it("the harness serves notes so the page can be driven end to end", () => {
    expect(mockApi).toContain('url.pathname === "/v2/notes"');
    expect(mockApi).toContain("note_seed_protocol");
  });

  it("re-seals exact Note ids immediately before every run and fails before mutating the turn", () => {
    const start = appSource.indexOf("const handleSubmit = async (");
    const end = appSource.indexOf("const handleSubmitRef = useRef(handleSubmit)", start);
    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    const submit = appSource.slice(start, end);
    const reseal = submit.indexOf("await resealTurnNotes(");
    const turnMutation = submit.indexOf("setViewerOpen(false)");
    const createRun = submit.indexOf("apiClient.chatStream(chatRequest");
    expect(reseal).toBeGreaterThan(-1);
    expect(reseal).toBeLessThan(turnMutation);
    expect(reseal).toBeLessThan(createRun);
    expect(submit).toMatch(/if \(!selectedNotesForTurn\) \{\s*turnOverride\?\.onNoteScopeFailure\?\.\(\);\s*return;/);
    expect(submit).toMatch(/noteAccessForTurn\(\s*text,\s*selectedNotesForTurn,\s*excludedNoteIntentTextForTurn/);
    expect(submit).toMatch(/excludedNoteIntentText:\s*excludedNoteIntentTextForTurn\.length > 0/);
  });
});

describe("media in notes — one pipeline, one catalog", () => {
  it("drops, pastes, and slash-picks all ride apiClient.uploadFiles — the SAME pipeline as chat, so files land in Resources", () => {
    expect(pageSource).toContain("apiClient.uploadFiles(files)");
    expect(pageSource).toContain("<FileUpload");
    expect(pageSource).toContain("onPaste={handleBodyPaste}");
    expect(pageSource).toContain('id: "media", label: "Upload file"');
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
    expect(pageSource).toContain("Uploading ${uploadingCount} file");
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
    expect(body).toContain("font-family: var(--font-mono);");
    expect(body).toContain("font-weight: var(--font-weight-mono);");
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
