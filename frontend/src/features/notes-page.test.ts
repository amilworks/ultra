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

  it("title Enter/Tab drops straight into the body", () => {
    expect(pageSource).toMatch(
      /event\.key === "Enter" \|\| event\.key === "Tab"[\s\S]{0,120}bodyRef\.current\?\.focus\(\)/
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

  it("previews through the SAME markdown pipeline as chat answers", () => {
    expect(pageSource).toContain(
      'import { LazyMarkdown } from "@/components/prompt-kit/lazy-markdown"'
    );
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
