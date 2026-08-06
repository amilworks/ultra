import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent as ReactClipboardEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type MutableRefObject,
} from "react";
import {
  Bold,
  Check,
  ChevronDown,
  ChevronLeft,
  Code,
  Highlighter,
  Italic,
  Link as LinkIcon,
  List,
  ListOrdered,
  Loader2,
  Paperclip,
  Pin,
  PinOff,
  Plus,
  Quote,
  Search,
  Strikethrough,
  Table as TableIcon,
  Trash,
  Type,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { FileUpload, useFileUploadContext } from "@/components/prompt-kit";
import type { ApiClient, NoteEditorMode, NoteListItem, NoteRecord } from "@/lib/api";
import { lazyNamedWithRetry } from "@/lib/lazy-retry";
import { markdownForUpload } from "@/lib/ultraResource";
/* Types only — the editor chunk (ProseMirror + remark) must never ride the
   main bundle; the value import below is the lazy one. */
import type {
  NotesActiveStates,
  NotesEditorAction,
  NotesEditorHandle,
} from "@/components/notes/MarkdownNoteEditor";

const LazyMarkdownNoteEditor = lazyNamedWithRetry(
  () => import("@/components/notes/MarkdownNoteEditor"),
  "MarkdownNoteEditor"
);

/* eslint react-hooks/set-state-in-effect: "off", react-hooks/immutability: "off" --
   This page is a flow state machine: its effects ARE the drivers (initial
   list fetch, debounced search, auto-open of the latest note, first-note
   auto-create), and state advancing inside them is the mechanism, not
   derived state. It is also an imperative bridge hub: the file picker and
   the markdown editor hand their APIs up through bind callbacks that write
   refs (never during render), and plaintext caret splices write the
   textarea DOM directly. The compiler's immutability inference flags those
   ref bridges positionally (the same write pattern passes elsewhere), so
   per-line disables are whack-a-mole. */

/* Notes — the personal layer of the workbench.
 *
 * Two modes, one file. Every note is body_markdown; the mode is per-note and
 * sticky (editor_mode rides the record):
 * - Markdown (default): edit like a doc. The styled surface IS the rendered
 *   note — ribbon + the shortcuts everyone knows, format-as-you-type, media
 *   inline. No separate preview exists anymore.
 * - Plaintext: the raw mono dump surface (JetBrains Mono, Tab indents,
 *   pastes land untouched, "/" block menu).
 * ⌘⇧E flips modes; ⌘E belongs to inline code now, like every other editor.
 *
 * The frictionless contract is unchanged: New note → typing immediately;
 * Enter/Tab from the title into the body; 800ms autosave flushed on
 * blur/unmount/note-switch; no dialogs, no save button.
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
  // Sentinel: handled by the page (opens the file picker), not by insertion.
  { id: "media", label: "Image or video", hint: "upload", insert: "" },
];

/* Media references are stored as portable ultra:// URIs (shared helpers in
   lib/ultraResource): notes stay exportable, environments stay swappable,
   and both editing surfaces resolve the same scheme at render time. */

/* Ribbon idle snapshot. Deliberately a local literal — importing a value
   from MarkdownNoteEditor would pull the editor chunk into the main bundle. */
const RIBBON_IDLE: NotesActiveStates = {
  bold: false,
  italic: false,
  strike: false,
  code: false,
  highlight: false,
  link: false,
  linkHref: null,
  block: "body",
  inTable: false,
};

/* Captures the FileUpload context's picker opener for the slash menu, which
   renders outside the place the hook can be called. */
function FilePickerBridge({
  openRef,
}: {
  openRef: MutableRefObject<(() => void) | null>;
}) {
  const { openFilePicker } = useFileUploadContext();
  // The ref write lives in an effect, where mutation is legal — the previous
  // bind-callback shape did the same write but through a useCallback argument,
  // which react-hooks/immutability rightly cannot prove safe.
  useEffect(() => {
    openRef.current = openFilePicker;
    return () => {
      openRef.current = null;
    };
  }, [openRef, openFilePicker]);
  return null;
}

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
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashIndex, setSlashIndex] = useState(0);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [editorActive, setEditorActive] = useState<NotesActiveStates>(RIBBON_IDLE);
  const [linkOpen, setLinkOpen] = useState(false);
  const [linkValue, setLinkValue] = useState("");
  // Phones collapse to the editor; this flips the list back in front.
  const [mobileListOpen, setMobileListOpen] = useState(false);

  const titleRef = useRef<HTMLInputElement | null>(null);
  const bodyRef = useRef<HTMLTextAreaElement | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  const draftRef = useRef<{
    noteId: string;
    title: string;
    body: string;
    pinned: boolean;
    editorMode: NoteEditorMode;
  } | null>(null);
  const savedRef = useRef<{
    title: string;
    body: string;
    pinned: boolean;
    editorMode: NoteEditorMode;
  } | null>(null);
  const searchGenerationRef = useRef(0);
  const editorApiRef = useRef<NotesEditorHandle | null>(null);
  const editorActiveRef = useRef<NotesActiveStates>(RIBBON_IDLE);

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
    if (
      saved &&
      saved.title === draft.title &&
      saved.body === draft.body &&
      saved.pinned === draft.pinned &&
      saved.editorMode === draft.editorMode
    ) {
      return;
    }
    setSaveState("saving");
    try {
      const record = await apiClient.updateNote(draft.noteId, {
        title: draft.title,
        body_markdown: draft.body,
        pinned: draft.pinned,
        editor_mode: draft.editorMode,
      });
      savedRef.current = {
        title: record.title,
        body: record.body_markdown,
        pinned: record.pinned,
        editorMode: record.editor_mode,
      };
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
      setSlashOpen(false);
      setLinkOpen(false);
      setEditorActive(RIBBON_IDLE);
      setMobileListOpen(false);
      try {
        const record = await apiClient.getNote(noteId);
        draftRef.current = {
          noteId: record.note_id,
          title: record.title,
          body: record.body_markdown,
          pinned: record.pinned,
          editorMode: record.editor_mode === "plaintext" ? "plaintext" : "markdown",
        };
        savedRef.current = {
          title: record.title,
          body: record.body_markdown,
          pinned: record.pinned,
          editorMode: draftRef.current.editorMode,
        };
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
        editorMode: record.editor_mode === "plaintext" ? "plaintext" : "markdown",
      };
      savedRef.current = {
        title: record.title,
        body: record.body_markdown,
        pinned: record.pinned,
        editorMode: draftRef.current.editorMode,
      };
      setActiveNoteId(record.note_id);
      setActiveNote(record);
      setMobileListOpen(false);
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
    (patch: Partial<{ title: string; body: string; pinned: boolean; editorMode: NoteEditorMode }>) => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      draftRef.current = { ...draft, ...patch };
      scheduleSave();
    },
    [scheduleSave]
  );

  /* The mode is a deliberate, sticky choice — flip the surface immediately
     and persist without waiting for the debounce. */
  const switchEditorMode = useCallback(
    (nextMode: NoteEditorMode) => {
      const draft = draftRef.current;
      if (!draft || draft.editorMode === nextMode) {
        return;
      }
      draftRef.current = { ...draft, editorMode: nextMode };
      setActiveNote((current) => (current ? { ...current, editor_mode: nextMode } : current));
      setEditorActive(RIBBON_IDLE);
      editorActiveRef.current = RIBBON_IDLE;
      setLinkOpen(false);
      setSlashOpen(false);
      void flushSave();
    },
    [flushSave]
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
      if (block.id === "media") {
        // Remove the "/" that opened the menu, then hand off to the picker;
        // the upload path inserts the reference at this caret.
        const caret = textarea.selectionStart;
        const value = textarea.value.slice(0, Math.max(0, caret - 1)) + textarea.value.slice(caret);
        textarea.value = value;
        textarea.setSelectionRange(Math.max(0, caret - 1), Math.max(0, caret - 1));
        updateDraft({ body: value });
        setActiveNote((current) => (current ? { ...current, body_markdown: value } : current));
        setSlashOpen(false);
        filePickerRef.current?.();
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

  const [uploadingCount, setUploadingCount] = useState(0);
  const filePickerRef = useRef<(() => void) | null>(null);

  const insertAtCaret = useCallback(
    (text: string) => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      if (draft.editorMode === "markdown") {
        const api = editorApiRef.current;
        if (api) {
          // The editor's markdownUpdated listener syncs the draft.
          api.insertMarkdown(text);
          return;
        }
      }
      const textarea = bodyRef.current;
      let nextValue: string;
      if (textarea && draft.editorMode === "plaintext") {
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        nextValue = textarea.value.slice(0, start) + text + textarea.value.slice(end);
        const caret = start + text.length;
        textarea.value = nextValue;
        textarea.setSelectionRange(caret, caret);
        textarea.focus();
      } else {
        // Editor chunk still loading (or surface unmounted): append — an
        // upload that raced the mount is never lost.
        const body = draft.body;
        nextValue = body.length === 0 || body.endsWith("\n") ? body + text : body + "\n" + text;
      }
      updateDraft({ body: nextValue });
      setActiveNote((current) => (current ? { ...current, body_markdown: nextValue } : current));
    },
    [updateDraft]
  );

  /* Dropped/pasted/picked files ride the SAME upload pipeline as chat
     attachments, so every file cataloged here appears in Resources — one
     central place to find real data. The note stores only the reference. */
  const handleNoteFilesAdded = useCallback(
    async (files: File[]) => {
      if (!draftRef.current || files.length === 0) {
        return;
      }
      setUploadingCount((count) => count + files.length);
      try {
        const response = await apiClient.uploadFiles(files);
        if (!draftRef.current || response.uploaded.length === 0) {
          return;
        }
        const block = response.uploaded.map(markdownForUpload).join("\n");
        insertAtCaret(`\n${block}\n`);
      } catch (error) {
        setListError(
          error instanceof Error ? `Upload failed — ${error.message}` : "Upload failed."
        );
      } finally {
        setUploadingCount((count) => Math.max(0, count - files.length));
      }
    },
    [apiClient, insertAtCaret]
  );

  const handleBodyPaste = useCallback(
    (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const files = Array.from(event.clipboardData?.files ?? []);
      if (files.length > 0) {
        event.preventDefault();
        void handleNoteFilesAdded(files);
      }
    },
    [handleNoteFilesAdded]
  );

  /* The Markdown surface resolves ultra://resource media through the same
     download endpoint Resources uses (node views inside the editor). */
  const resolveResourceUrl = useCallback(
    (fileId: string) => apiClient.resourceDownloadUrl(fileId),
    [apiClient]
  );

  const bindEditorApi = useCallback((api: NotesEditorHandle | null) => {
    editorApiRef.current = api;
  }, []);

  const handleActiveStates = useCallback((states: NotesActiveStates) => {
    editorActiveRef.current = states;
    setEditorActive(states);
  }, []);

  const handleMarkdownChange = useCallback(
    (markdown: string) => {
      // The serializer always appends one newline; trimming it here is what
      // makes open → edit → open round-trips byte-stable (fidelity suite).
      const body = markdown.replace(/\n$/, "");
      setActiveNote((current) => (current ? { ...current, body_markdown: body } : current));
      updateDraft({ body });
    },
    [updateDraft]
  );

  const handleEditorBlur = useCallback(() => {
    void flushSave();
  }, [flushSave]);

  /* File pastes in Markdown mode are caught before ProseMirror sees them and
     ride the one upload pipeline; text pastes pass straight through to the
     editor's markdown-aware clipboard handling. */
  const handleMarkdownShellPaste = useCallback(
    (event: ReactClipboardEvent<HTMLDivElement>) => {
      const files = Array.from(event.clipboardData?.files ?? []);
      if (files.length > 0) {
        event.preventDefault();
        event.stopPropagation();
        void handleNoteFilesAdded(files);
      }
    },
    [handleNoteFilesAdded]
  );

  const handleMarkdownShellKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        event.stopPropagation();
        setLinkValue(editorActiveRef.current.linkHref ?? "");
        setLinkOpen(true);
      }
    },
    []
  );

  const execRibbon = useCallback((action: NotesEditorAction) => {
    editorApiRef.current?.exec(action);
  }, []);

  /* Toolbar convention: eat mousedown so the editor keeps focus and the
     selection the command should apply to. Click still fires. */
  const keepEditorFocus = useCallback((event: ReactMouseEvent) => {
    event.preventDefault();
  }, []);

  const applyLink = useCallback(() => {
    const raw = linkValue.trim();
    if (raw.length === 0) {
      setLinkOpen(false);
      return;
    }
    const href =
      /^[a-z][a-z0-9+.-]*:/i.test(raw) || raw.startsWith("/") || raw.startsWith("#")
        ? raw
        : `https://${raw}`;
    editorApiRef.current?.applyLink(href);
    setLinkOpen(false);
  }, [linkValue]);

  const removeLink = useCallback(() => {
    editorApiRef.current?.removeLink();
    setLinkOpen(false);
  }, []);

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
      if (event.key === "Tab" && !event.shiftKey) {
        // Sublime-grade plaintext: Tab indents, it never escapes the editor.
        // (⌘E, Escape-then-Tab, and every toolbar button remain the exits.)
        event.preventDefault();
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const nextValue = textarea.value.slice(0, start) + "  " + textarea.value.slice(end);
        textarea.value = nextValue;
        textarea.setSelectionRange(start + 2, start + 2);
        updateDraft({ body: nextValue });
        setActiveNote((current) => (current ? { ...current, body_markdown: nextValue } : current));
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
    [insertSlashBlock, slashIndex, slashOpen, updateDraft]
  );

  // ⌘⇧E flips Markdown/Plaintext from anywhere on the page. (⌘E now writes
  // inline code inside the Markdown surface — the editors-everywhere default.)
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "e") {
        event.preventDefault();
        const draft = draftRef.current;
        if (!draft) {
          return;
        }
        switchEditorMode(draft.editorMode === "markdown" ? "plaintext" : "markdown");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [switchEditorMode]);

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
    <div
      className="notes-page"
      data-testid="notes-page"
      data-mobile-list={mobileListOpen ? "true" : undefined}
    >
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
        <FileUpload
          onFilesAdded={(files) => void handleNoteFilesAdded(files)}
          multiple
          className="notes-editor-drop"
        >
          <FilePickerBridge openRef={filePickerRef} />
        {activeNote ? (
          <>
            <div className="notes-editor-bar">
              <div className="notes-editor-bar-lead">
                <button
                  type="button"
                  className="notes-mobile-back"
                  onClick={() => setMobileListOpen(true)}
                >
                  <ChevronLeft aria-hidden="true" />
                  Notes
                </button>
                <span className="notes-save-state" data-state={saveState} role="status">
                  {uploadingCount > 0 ? (
                    <>
                      <Loader2 className="animate-spin" aria-hidden="true" />
                      Uploading {uploadingCount} file{uploadingCount === 1 ? "" : "s"}…
                    </>
                  ) : (
                    <>
                      {saveState === "saving" ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
                      {saveState === "saved" || saveState === "idle" ? <Check aria-hidden="true" /> : null}
                      {saveLabel}
                    </>
                  )}
                </span>
              </div>
              <div className="notes-editor-actions">
                <div className="notes-mode-seg" role="tablist" aria-label="Editor mode">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeNote.editor_mode !== "plaintext"}
                    data-active={activeNote.editor_mode !== "plaintext" ? "true" : undefined}
                    title="Edit like a doc (⌘⇧E)"
                    onClick={() => switchEditorMode("markdown")}
                  >
                    Markdown
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeNote.editor_mode === "plaintext"}
                    data-active={activeNote.editor_mode === "plaintext" ? "true" : undefined}
                    title="Raw text, mono (⌘⇧E)"
                    onClick={() => switchEditorMode("plaintext")}
                  >
                    Plaintext
                  </button>
                </div>
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
                  if (draftRef.current?.editorMode === "markdown") {
                    editorApiRef.current?.focus();
                  } else {
                    bodyRef.current?.focus();
                  }
                }
              }}
            />

            {activeNote.editor_mode === "plaintext" ? (
              <div className="notes-body-shell">
                <textarea
                  ref={bodyRef}
                  className="notes-body-input"
                  value={activeNote.body_markdown}
                  placeholder={'Write in markdown — "/" for blocks; drop or paste images and videos'}
                  aria-label="Note body"
                  onPaste={handleBodyPaste}
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
              <>
                <div className="notes-ribbon" role="toolbar" aria-label="Formatting">
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Bold"
                    title="Bold (⌘B)"
                    aria-pressed={editorActive.bold}
                    data-active={editorActive.bold ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("bold")}
                  >
                    <Bold aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Italic"
                    title="Italic (⌘I)"
                    aria-pressed={editorActive.italic}
                    data-active={editorActive.italic ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("italic")}
                  >
                    <Italic aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Strikethrough"
                    title="Strikethrough (⌘⇧X)"
                    aria-pressed={editorActive.strike}
                    data-active={editorActive.strike ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("strike")}
                  >
                    <Strikethrough aria-hidden="true" />
                  </button>
                  <span className="notes-ribbon-sep" aria-hidden="true" />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        className="notes-ribbon-btn notes-ribbon-tt"
                        aria-label="Text size"
                        title="Text size (⌘⌥1 / ⌘⌥2 / ⌘⌥0)"
                        onMouseDown={keepEditorFocus}
                      >
                        <Type aria-hidden="true" />
                        <ChevronDown className="notes-ribbon-caret" aria-hidden="true" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="notes-ribbon-menu">
                      <DropdownMenuItem
                        data-current={editorActive.block === "h2" ? "true" : undefined}
                        onSelect={() => execRibbon("h2")}
                      >
                        Heading
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        data-current={editorActive.block === "h3" ? "true" : undefined}
                        onSelect={() => execRibbon("h3")}
                      >
                        Subheading
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        data-current={editorActive.block === "body" ? "true" : undefined}
                        onSelect={() => execRibbon("body")}
                      >
                        Body
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <span className="notes-ribbon-sep" aria-hidden="true" />
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Highlight"
                    title="Highlight (⌘⇧H)"
                    aria-pressed={editorActive.highlight}
                    data-active={editorActive.highlight ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("highlight")}
                  >
                    <Highlighter aria-hidden="true" />
                  </button>
                  <span className="notes-ribbon-sep" aria-hidden="true" />
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Link"
                    title="Link (⌘K)"
                    aria-pressed={editorActive.link}
                    data-active={editorActive.link ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => {
                      setLinkValue(editorActiveRef.current.linkHref ?? "");
                      setLinkOpen((current) => !current);
                    }}
                  >
                    <LinkIcon aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Quote"
                    title="Quote (⌘⇧9)"
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("quote")}
                  >
                    <Quote aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Code"
                    title="Code (⌘E)"
                    aria-pressed={editorActive.code}
                    data-active={editorActive.code ? "true" : undefined}
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("code")}
                  >
                    <Code aria-hidden="true" />
                  </button>
                  <span className="notes-ribbon-sep" aria-hidden="true" />
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Bulleted list"
                    title="Bulleted list (⌘⇧8)"
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("bullet")}
                  >
                    <List aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Numbered list"
                    title="Numbered list (⌘⇧7)"
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("ordered")}
                  >
                    <ListOrdered aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Table"
                    title="Insert table"
                    onMouseDown={keepEditorFocus}
                    onClick={() => execRibbon("table")}
                  >
                    <TableIcon aria-hidden="true" />
                  </button>
                  {editorActive.inTable ? (
                    <>
                      <span className="notes-ribbon-sep" aria-hidden="true" />
                      <button
                        type="button"
                        className="notes-ribbon-btn notes-ribbon-text"
                        aria-label="Add row below"
                        title="Add row below"
                        onMouseDown={keepEditorFocus}
                        onClick={() => execRibbon("rowBelow")}
                      >
                        + Row
                      </button>
                      <button
                        type="button"
                        className="notes-ribbon-btn notes-ribbon-text"
                        aria-label="Delete row"
                        title="Delete row"
                        onMouseDown={keepEditorFocus}
                        onClick={() => execRibbon("rowDelete")}
                      >
                        − Row
                      </button>
                      <button
                        type="button"
                        className="notes-ribbon-btn notes-ribbon-text"
                        aria-label="Add column right"
                        title="Add column right"
                        onMouseDown={keepEditorFocus}
                        onClick={() => execRibbon("colRight")}
                      >
                        + Col
                      </button>
                      <button
                        type="button"
                        className="notes-ribbon-btn notes-ribbon-text"
                        aria-label="Delete column"
                        title="Delete column"
                        onMouseDown={keepEditorFocus}
                        onClick={() => execRibbon("colDelete")}
                      >
                        − Col
                      </button>
                      <button
                        type="button"
                        className="notes-ribbon-btn"
                        aria-label="Delete table"
                        title="Delete table"
                        onMouseDown={keepEditorFocus}
                        onClick={() => execRibbon("tableDelete")}
                      >
                        <Trash aria-hidden="true" />
                      </button>
                    </>
                  ) : null}
                  <span className="notes-ribbon-spacer" aria-hidden="true" />
                  <button
                    type="button"
                    className="notes-ribbon-btn"
                    aria-label="Attach image or video"
                    title="Attach image or video"
                    onMouseDown={keepEditorFocus}
                    onClick={() => filePickerRef.current?.()}
                  >
                    <Paperclip aria-hidden="true" />
                  </button>
                </div>
                {linkOpen ? (
                  <div className="notes-link-pop">
                    <input
                      autoFocus
                      type="text"
                      value={linkValue}
                      placeholder="Paste or type a link"
                      aria-label="Link URL"
                      onChange={(event) => setLinkValue(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          applyLink();
                        }
                        if (event.key === "Escape") {
                          event.preventDefault();
                          setLinkOpen(false);
                          editorApiRef.current?.focus();
                        }
                      }}
                    />
                    <Button type="button" variant="outline" size="sm" onClick={applyLink}>
                      Apply
                    </Button>
                    {editorActive.link ? (
                      <Button type="button" variant="outline" size="sm" onClick={removeLink}>
                        Remove
                      </Button>
                    ) : null}
                  </div>
                ) : null}
                <div
                  className="notes-md-shell"
                  onPasteCapture={handleMarkdownShellPaste}
                  onKeyDownCapture={handleMarkdownShellKeyDown}
                >
                  <Suspense
                    fallback={
                      <div className="notes-editor-empty" role="status">
                        <Loader2 className="animate-spin" aria-hidden="true" /> Loading editor…
                      </div>
                    }
                  >
                    <LazyMarkdownNoteEditor
                      key={activeNote.note_id}
                      defaultMarkdown={activeNote.body_markdown}
                      resourceUrl={resolveResourceUrl}
                      onMarkdownChange={handleMarkdownChange}
                      onBlur={handleEditorBlur}
                      onActiveStatesChange={handleActiveStates}
                      bindApi={bindEditorApi}
                    />
                  </Suspense>
                </div>
              </>
            )}
          </>
        ) : noteLoading ? (
          <div className="notes-editor-empty" role="status">
            <Loader2 className="animate-spin" aria-hidden="true" /> Opening…
          </div>
        ) : (
          <div className="notes-editor-empty">Select a note, or create one.</div>
        )}
        </FileUpload>
      </section>
    </div>
  );
}
