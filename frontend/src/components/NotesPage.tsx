import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import {
  Check,
  Eye,
  Loader2,
  Pencil,
  Pin,
  PinOff,
  Plus,
  Search,
  Trash,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { LazyMarkdown } from "@/components/prompt-kit/lazy-markdown";
import type { ApiClient, NoteListItem, NoteRecord } from "@/lib/api";
import { Suspense } from "react";

/* eslint react-hooks/set-state-in-effect: "off" -- This page is a flow
   state machine: its effects ARE the drivers (initial list fetch, debounced
   search, auto-open of the latest note, first-note auto-create), and state
   advancing inside them is the mechanism, not derived state. The rule
   reports positionally across traced calls, so per-line disables are
   whack-a-mole (same call as GoogleDriveImport). */

/* Notes — the personal layer of the workbench (Phase 1).
 *
 * The whole surface is tuned for zero friction:
 * - New note → the title is focused and you are typing immediately; Enter or
 *   Tab drops into the body. No dialogs, no save button anywhere.
 * - Autosave: 800ms debounce + flush on blur/unmount/note-switch. The meta
 *   line whispers Saving…/Saved; failures surface once and keep the draft.
 * - Write is the default mode (a scratchpad is written far more than read);
 *   ⌘E or the toggle flips to Preview, rendered by the SAME markdown pipeline
 *   chat answers use — tables, code, and math match the rest of the app.
 * - "/" at the start of a line opens a small block menu (house slash-menu
 *   pattern) that inserts markdown skeletons. Ultra-object embeds arrive in
 *   the next phase; the menu is built to grow into them.
 */

const AUTOSAVE_DEBOUNCE_MS = 800;

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

type SlashBlock = {
  id: string;
  label: string;
  hint: string;
  insert: string;
  cursorOffset?: number;
};

const SLASH_BLOCKS: SlashBlock[] = [
  { id: "heading", label: "Heading", hint: "## ", insert: "## " },
  { id: "todo", label: "To-do list", hint: "- [ ]", insert: "- [ ] " },
  { id: "bullets", label: "Bulleted list", hint: "- ", insert: "- " },
  {
    id: "table",
    label: "Table",
    hint: "3 columns",
    insert: "| Column | Column | Column |\n| --- | --- | --- |\n|  |  |  |\n",
  },
  {
    id: "code",
    label: "Code",
    hint: "```",
    insert: "```python\n\n```",
    cursorOffset: 10,
  },
  { id: "divider", label: "Divider", hint: "---", insert: "---\n" },
  { id: "quote", label: "Quote", hint: "> ", insert: "> " },
];

const relativeTime = (iso: string): string => {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) {
    return "";
  }
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

const listGroupFor = (iso: string): string => {
  const then = new Date(iso);
  const now = new Date();
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const dayDelta = Math.round((startOfDay(now) - startOfDay(then)) / 86400000);
  if (dayDelta <= 0) return "Today";
  if (dayDelta === 1) return "Yesterday";
  return "Earlier";
};

const cleanSnippet = (snippet: string): string =>
  snippet
    .replace(/^#+\s+/gm, "")
    .replace(/[*_`>]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 160);

export type NotesPageProps = {
  apiClient: ApiClient;
};

export function NotesPage({ apiClient }: NotesPageProps) {
  const [items, setItems] = useState<NoteListItem[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [activeNote, setActiveNote] = useState<NoteRecord | null>(null);
  const [noteLoading, setNoteLoading] = useState(false);
  const [mode, setMode] = useState<"write" | "preview">("write");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashIndex, setSlashIndex] = useState(0);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const titleRef = useRef<HTMLInputElement | null>(null);
  const bodyRef = useRef<HTMLTextAreaElement | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  const draftRef = useRef<{ noteId: string; title: string; body: string; pinned: boolean } | null>(null);
  const savedRef = useRef<{ title: string; body: string; pinned: boolean } | null>(null);
  const searchGenerationRef = useRef(0);

  const refreshList = useCallback(
    async (query: string) => {
      const generation = ++searchGenerationRef.current;
      setListError(null);
      try {
        const page = await apiClient.listNotes({ query: query || undefined, limit: 200 });
        if (generation !== searchGenerationRef.current) {
          return;
        }
        setItems(page.notes);
        setListLoading(false);
      } catch (error) {
        if (generation !== searchGenerationRef.current) {
          return;
        }
        setListError(error instanceof Error ? error.message : String(error));
        setListLoading(false);
      }
    },
    [apiClient]
  );

  useEffect(() => {
    void refreshList("");
  }, [refreshList]);

  // Search-as-you-type with a light debounce; the generation counter keeps a
  // slow earlier response from clobbering a fresher one.
  useEffect(() => {
    const handle = window.setTimeout(() => {
      void refreshList(searchQuery.trim());
    }, 180);
    return () => window.clearTimeout(handle);
  }, [searchQuery, refreshList]);

  const flushSave = useCallback(async (): Promise<void> => {
    const draft = draftRef.current;
    if (!draft) {
      return;
    }
    const saved = savedRef.current;
    if (saved && saved.title === draft.title && saved.body === draft.body && saved.pinned === draft.pinned) {
      return;
    }
    setSaveState("saving");
    try {
      const record = await apiClient.updateNote(draft.noteId, {
        title: draft.title,
        body_markdown: draft.body,
        pinned: draft.pinned,
      });
      savedRef.current = { title: record.title, body: record.body_markdown, pinned: record.pinned };
      setSaveState("saved");
      setItems((current) =>
        current.map((item) =>
          item.note_id === record.note_id
            ? {
                ...item,
                title: record.title,
                snippet: record.body_markdown.slice(0, 300),
                pinned: record.pinned,
                updated_at: record.updated_at,
              }
            : item
        )
      );
      setActiveNote((current) =>
        current && current.note_id === record.note_id ? { ...current, updated_at: record.updated_at } : current
      );
    } catch (error) {
      setSaveState("error");
      setListError(error instanceof Error ? `Autosave failed — ${error.message}` : "Autosave failed.");
    }
  }, [apiClient]);

  const scheduleSave = useCallback(() => {
    setSaveState("dirty");
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null;
      void flushSave();
    }, AUTOSAVE_DEBOUNCE_MS);
  }, [flushSave]);

  // The draft always flushes on unmount or note switch — leaving the page is
  // never how a keystroke gets lost.
  useEffect(() => {
    return () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
      }
      void flushSave();
    };
  }, [flushSave]);

  const openNote = useCallback(
    async (noteId: string) => {
      if (draftRef.current && draftRef.current.noteId !== noteId) {
        await flushSave();
      }
      setActiveNoteId(noteId);
      setNoteLoading(true);
      setMode("write");
      setSlashOpen(false);
      try {
        const record = await apiClient.getNote(noteId);
        draftRef.current = {
          noteId: record.note_id,
          title: record.title,
          body: record.body_markdown,
          pinned: record.pinned,
        };
        savedRef.current = { title: record.title, body: record.body_markdown, pinned: record.pinned };
        setActiveNote(record);
        setSaveState("idle");
      } catch (error) {
        setListError(error instanceof Error ? error.message : String(error));
        setActiveNoteId(null);
        setActiveNote(null);
      } finally {
        setNoteLoading(false);
      }
    },
    [apiClient, flushSave]
  );

  const createNote = useCallback(async () => {
    try {
      const record = await apiClient.createNote({});
      setItems((current) => [
        {
          note_id: record.note_id,
          title: record.title,
          snippet: "",
          pinned: record.pinned,
          updated_at: record.updated_at,
        },
        ...current,
      ]);
      draftRef.current = {
        noteId: record.note_id,
        title: record.title,
        body: record.body_markdown,
        pinned: record.pinned,
      };
      savedRef.current = { title: record.title, body: record.body_markdown, pinned: record.pinned };
      setActiveNoteId(record.note_id);
      setActiveNote(record);
      setMode("write");
      setSaveState("idle");
      // The whole point: New note → typing the title with zero further clicks.
      window.requestAnimationFrame(() => titleRef.current?.focus());
    } catch (error) {
      setListError(error instanceof Error ? error.message : String(error));
    }
  }, [apiClient]);

  // First visit with zero notes: create one instead of showing an empty
  // lecture — the fastest possible path from "opened Notes" to "writing".
  const autoCreatedRef = useRef(false);
  useEffect(() => {
    if (
      !listLoading &&
      !listError &&
      items.length === 0 &&
      !searchQuery.trim() &&
      !autoCreatedRef.current
    ) {
      autoCreatedRef.current = true;
      void createNote();
    }
  }, [listLoading, listError, items.length, searchQuery, createNote]);

  // Opening the page lands on the most recent note — reading position, ready
  // to type, no blank pane.
  useEffect(() => {
    if (!listLoading && !activeNoteId && items.length > 0) {
      void openNote(items[0].note_id);
    }
  }, [listLoading, activeNoteId, items, openNote]);

  const updateDraft = useCallback(
    (patch: Partial<{ title: string; body: string; pinned: boolean }>) => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      draftRef.current = { ...draft, ...patch };
      scheduleSave();
    },
    [scheduleSave]
  );

  const togglePinned = useCallback(async () => {
    const draft = draftRef.current;
    if (!draft) {
      return;
    }
    const nextPinned = !draft.pinned;
    draftRef.current = { ...draft, pinned: nextPinned };
    setActiveNote((current) => (current ? { ...current, pinned: nextPinned } : current));
    setItems((current) =>
      current.map((item) => (item.note_id === draft.noteId ? { ...item, pinned: nextPinned } : item))
    );
    await flushSave();
  }, [flushSave]);

  const deleteActiveNote = useCallback(async () => {
    const noteId = activeNoteId;
    if (!noteId) {
      return;
    }
    try {
      await apiClient.deleteNote(noteId);
      draftRef.current = null;
      savedRef.current = null;
      setPendingDeleteId(null);
      setActiveNoteId(null);
      setActiveNote(null);
      setItems((current) => current.filter((item) => item.note_id !== noteId));
    } catch (error) {
      setListError(error instanceof Error ? error.message : String(error));
    }
  }, [activeNoteId, apiClient]);

  const insertSlashBlock = useCallback(
    (block: SlashBlock) => {
      const textarea = bodyRef.current;
      const draft = draftRef.current;
      if (!textarea || !draft) {
        return;
      }
      const start = textarea.selectionStart;
      // Replace the "/" that opened the menu.
      const before = textarea.value.slice(0, Math.max(0, start - 1));
      const after = textarea.value.slice(start);
      const nextValue = before + block.insert + after;
      const caret = before.length + (block.cursorOffset ?? block.insert.length);
      textarea.value = nextValue;
      textarea.setSelectionRange(caret, caret);
      updateDraft({ body: nextValue });
      setActiveNote((current) => (current ? { ...current, body_markdown: nextValue } : current));
      setSlashOpen(false);
      textarea.focus();
    },
    [updateDraft]
  );

  const handleBodyKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      const textarea = event.currentTarget;
      if (slashOpen) {
        if (event.key === "Escape") {
          event.preventDefault();
          setSlashOpen(false);
          return;
        }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          setSlashIndex((current) => {
            const delta = event.key === "ArrowDown" ? 1 : -1;
            return (current + delta + SLASH_BLOCKS.length) % SLASH_BLOCKS.length;
          });
          return;
        }
        if (event.key === "Enter" || event.key === "Tab") {
          event.preventDefault();
          insertSlashBlock(SLASH_BLOCKS[slashIndex]);
          return;
        }
        if (event.key.length === 1 || event.key === "Backspace") {
          // Any real typing dismisses the menu and lands in the note.
          setSlashOpen(false);
        }
        return;
      }
      if (event.key === "/") {
        const caret = textarea.selectionStart;
        const lineStart = textarea.value.lastIndexOf("\n", caret - 1) + 1;
        if (caret === lineStart) {
          setSlashOpen(true);
          setSlashIndex(0);
        }
      }
    },
    [insertSlashBlock, slashIndex, slashOpen]
  );

  // ⌘E flips Write/Preview from anywhere on the page.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "e") {
        event.preventDefault();
        setMode((current) => (current === "write" ? "preview" : "write"));
        setSlashOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const groupedItems = useMemo(() => {
    const pinned = items.filter((item) => item.pinned);
    const rest = items.filter((item) => !item.pinned);
    const groups: Array<{ label: string; rows: NoteListItem[] }> = [];
    if (pinned.length > 0) {
      groups.push({ label: "Pinned", rows: pinned });
    }
    const byPeriod = new Map<string, NoteListItem[]>();
    for (const item of rest) {
      const label = listGroupFor(item.updated_at);
      const rows = byPeriod.get(label) ?? [];
      rows.push(item);
      byPeriod.set(label, rows);
    }
    for (const label of ["Today", "Yesterday", "Earlier"]) {
      const rows = byPeriod.get(label);
      if (rows && rows.length > 0) {
        groups.push({ label, rows });
      }
    }
    return groups;
  }, [items]);

  const saveLabel =
    saveState === "saving"
      ? "Saving…"
      : saveState === "dirty"
        ? "Unsaved changes"
        : saveState === "error"
          ? "Autosave failed — retrying on next edit"
          : activeNote
            ? `Saved ${relativeTime(activeNote.updated_at)}`
            : "";

  return (
    <div className="notes-page" data-testid="notes-page">
      <aside className="notes-list">
        <div className="notes-list-head">
          <h2>Notes</h2>
          <Button type="button" variant="outline" size="sm" className="notes-new-button" onClick={() => void createNote()}>
            <Plus data-icon="inline-start" aria-hidden="true" />
            New note
          </Button>
        </div>
        <label className="notes-search">
          <Search aria-hidden="true" />
          <input
            type="search"
            value={searchQuery}
            placeholder="Search notes"
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </label>
        {listLoading ? (
          <div className="notes-list-state" role="status">
            <Loader2 className="animate-spin" aria-hidden="true" /> Loading notes…
          </div>
        ) : listError ? (
          <div className="notes-list-state" role="alert">{listError}</div>
        ) : items.length === 0 ? (
          <div className="notes-list-state">No notes match “{searchQuery.trim()}”.</div>
        ) : (
          <div className="notes-list-scroll">
            {groupedItems.map((group) => (
              <div key={group.label}>
                <div className="notes-group-label">{group.label}</div>
                {group.rows.map((item) => (
                  <button
                    key={item.note_id}
                    type="button"
                    className="notes-row"
                    data-active={item.note_id === activeNoteId ? "true" : undefined}
                    onClick={() => void openNote(item.note_id)}
                  >
                    <span className="notes-row-title">
                      {item.pinned ? <Pin className="notes-row-pin" aria-label="Pinned" /> : null}
                      {item.title.trim() || "Untitled"}
                    </span>
                    <span className="notes-row-snippet">{cleanSnippet(item.snippet) || "Empty note"}</span>
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
        <div className="notes-list-foot">
          Notes are private to you. Ultra will read them for context in your own chats — coming next.
        </div>
      </aside>

      <section className="notes-editor">
        {activeNote ? (
          <>
            <div className="notes-editor-bar">
              <span className="notes-save-state" data-state={saveState} role="status">
                {saveState === "saving" ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
                {saveState === "saved" || saveState === "idle" ? <Check aria-hidden="true" /> : null}
                {saveLabel}
              </span>
              <div className="notes-editor-actions">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="notes-mode-toggle"
                  aria-pressed={mode === "preview"}
                  title="Toggle preview (⌘E)"
                  onClick={() => setMode((current) => (current === "write" ? "preview" : "write"))}
                >
                  {mode === "write" ? <Eye data-icon="inline-start" aria-hidden="true" /> : <Pencil data-icon="inline-start" aria-hidden="true" />}
                  {mode === "write" ? "Preview" : "Write"}
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => void togglePinned()}>
                  {activeNote.pinned ? <PinOff data-icon="inline-start" aria-hidden="true" /> : <Pin data-icon="inline-start" aria-hidden="true" />}
                  {activeNote.pinned ? "Unpin" : "Pin"}
                </Button>
                {pendingDeleteId === activeNote.note_id ? (
                  <Button type="button" variant="destructive" size="sm" onClick={() => void deleteActiveNote()}>
                    <Trash data-icon="inline-start" aria-hidden="true" />
                    Really delete
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setPendingDeleteId(activeNote.note_id)}
                    onBlur={() => setPendingDeleteId(null)}
                  >
                    <Trash data-icon="inline-start" aria-hidden="true" />
                    Delete
                  </Button>
                )}
              </div>
            </div>

            <input
              ref={titleRef}
              className="notes-title-input"
              value={activeNote.title}
              placeholder="Untitled"
              aria-label="Note title"
              onChange={(event) => {
                const title = event.target.value;
                setActiveNote((current) => (current ? { ...current, title } : current));
                updateDraft({ title });
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === "Tab") {
                  event.preventDefault();
                  bodyRef.current?.focus();
                }
              }}
            />

            {mode === "write" ? (
              <div className="notes-body-shell">
                <textarea
                  ref={bodyRef}
                  className="notes-body-input"
                  value={activeNote.body_markdown}
                  placeholder={'Write in markdown — press "/" on an empty line for blocks'}
                  aria-label="Note body"
                  onChange={(event) => {
                    const body = event.target.value;
                    setActiveNote((current) => (current ? { ...current, body_markdown: body } : current));
                    updateDraft({ body });
                  }}
                  onKeyDown={handleBodyKeyDown}
                  onBlur={() => void flushSave()}
                />
                {slashOpen ? (
                  <div className="notes-slash" role="listbox" aria-label="Insert block">
                    {SLASH_BLOCKS.map((block, index) => (
                      <button
                        key={block.id}
                        type="button"
                        role="option"
                        aria-selected={index === slashIndex}
                        className="notes-slash-item"
                        data-highlighted={index === slashIndex ? "true" : undefined}
                        onMouseEnter={() => setSlashIndex(index)}
                        onMouseDown={(event) => {
                          event.preventDefault();
                          insertSlashBlock(block);
                        }}
                      >
                        <span>{block.label}</span>
                        <small>{block.hint}</small>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="notes-preview" data-testid="notes-preview">
                <Suspense fallback={null}>
                  <LazyMarkdown className="notes-preview-markdown">
                    {activeNote.body_markdown.trim() || "*Nothing here yet.*"}
                  </LazyMarkdown>
                </Suspense>
              </div>
            )}
          </>
        ) : noteLoading ? (
          <div className="notes-editor-empty" role="status">
            <Loader2 className="animate-spin" aria-hidden="true" /> Opening…
          </div>
        ) : (
          <div className="notes-editor-empty">Select a note, or create one.</div>
        )}
      </section>
    </div>
  );
}
