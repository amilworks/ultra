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
  AlertCircle,
  Bold,
  Check,
  ChevronLeft,
  Code,
  Expand,
  FileText,
  Highlighter,
  Italic,
  Link as LinkIcon,
  Loader2,
  Minimize2,
  MoreHorizontal,
  Paperclip,
  Pin,
  Plus,
  RotateCcw,
  Search,
  Trash,
  Upload,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { FileUpload, useFileUploadContext } from "@/components/prompt-kit";
import type { ApiClient, NoteEditorMode, NoteListItem, NoteRecord } from "@/lib/api";
import { lazyNamedWithRetry } from "@/lib/lazy-retry";
import { markdownForUpload } from "@/lib/ultraResource";
import type { ResourceRecord } from "@/types";
/* Types only — the editor chunk (ProseMirror + remark) must never ride the
   main bundle; the value import below is the lazy one. */
import type {
  NotesActiveStates,
  NotesEditorAnchor,
  NotesEditorAction,
  NotesEditorHandle,
  NotesEditorMenuRequest,
} from "@/components/notes/MarkdownNoteEditor";

const LazyMarkdownNoteEditor = lazyNamedWithRetry(
  () => import("@/components/notes/MarkdownNoteEditor"),
  "MarkdownNoteEditor"
);

/* eslint react-hooks/set-state-in-effect: "off", react-hooks/immutability: "off" --
   This page is a flow state machine: its effects ARE the drivers (initial
   list fetch, debounced search, auto-open of the latest note), and state
   advancing inside them is the mechanism, not
   derived state. It is also an imperative bridge hub: the file picker and
   the markdown editor hand their APIs up through bind callbacks that write
   refs (never during render), and plaintext caret splices write the
   textarea DOM directly. The compiler's immutability inference flags those
   ref bridges positionally (the same write pattern passes elsewhere), so
   per-line disables are whack-a-mole. */

/* Notes — the personal writing layer of the workbench.
 *
 * Markdown remains the only durable body format. The default surface behaves
 * like a quiet document; raw Markdown is a per-note expert preference behind
 * the overflow menu. Formatting appears at the selection, blocks appear on
 * "/", and Ultra resources appear on "@" or the paperclip.
 *
 * New note is a client-only, body-focused draft. It reaches the API only after
 * meaningful input, then autosaves through one serialized channel. Blank
 * drafts disappear without creating list clutter; failures remain visible and
 * retryable instead of being described as saved.
 */

const AUTOSAVE_DEBOUNCE_MS = 700;
const LOCAL_DRAFT_ID = "__ultra_local_note_draft__";
const COMPACT_NOTES_MEDIA = "(max-width: 960px)";

type SaveState = "idle" | "draft" | "dirty" | "saving" | "saved" | "error";

type SlashBlock = {
  id: string;
  label: string;
  hint: string;
  insert: string;
  cursorOffset?: number;
};

const SLASH_BLOCKS: SlashBlock[] = [
  {
    id: "observation",
    label: "Observation",
    hint: "Scientific block",
    // NBSP survives Markdown parsing so the writer's first word never welds
    // itself to the label before the next serialization pass.
    insert: "> **Observation:**\u00a0",
  },
  {
    id: "decision",
    label: "Decision",
    hint: "Scientific block",
    insert: "> **Decision:**\u00a0",
  },
  {
    id: "hypothesis",
    label: "Hypothesis",
    hint: "Scientific block",
    insert: "> **Hypothesis:**\u00a0",
  },
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
  // Sentinels: handled by the page, not by direct insertion.
  { id: "resource", label: "Ultra resource", hint: "Link existing", insert: "" },
  { id: "media", label: "Upload file", hint: "Add to Resources", insert: "" },
];

const matchingSlashBlocks = (query: string): SlashBlock[] => {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) {
    return SLASH_BLOCKS;
  }
  return SLASH_BLOCKS.filter((block) =>
    `${block.label} ${block.hint} ${block.id}`.toLocaleLowerCase().includes(normalized)
  );
};

/* Media references are stored as portable ultra:// URIs (shared helpers in
   lib/ultraResource): notes stay exportable, environments stay swappable,
   and both editing surfaces resolve the same scheme at render time. */

/* Editor-state idle snapshot. Deliberately a local literal — importing a value
   from MarkdownNoteEditor would pull the editor chunk into the main bundle. */
const EDITOR_IDLE: NotesActiveStates = {
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

type NoteDraft = {
  noteId: string;
  title: string;
  body: string;
  pinned: boolean;
  editorMode: NoteEditorMode;
};

type SavedDraft = Omit<NoteDraft, "noteId">;

const meaningfulDraft = (draft: NoteDraft): boolean =>
  draft.title.trim().length > 0 || draft.body.trim().length > 0;

const draftMatchesSaved = (draft: NoteDraft, saved: SavedDraft | null): boolean =>
  Boolean(
    saved &&
      saved.title === draft.title &&
      saved.body === draft.body &&
      saved.pinned === draft.pinned &&
      saved.editorMode === draft.editorMode
  );

const titleFromBody = (markdown: string): string => {
  const firstLine = markdown
    .split("\n")
    .map((line) => {
      const cleaned = line
        .replace(/^\s*```.*$/, "")
        .replace(/^#{1,6}\s+/, "")
        .replace(/^>\s*/, "")
        .replace(/^[-*+]\s+(?:\[[ xX]\]\s*)?/, "")
        .replace(/^\d+[.)]\s+/, "")
        .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
        .replace(/==([^=\n]+)==/g, "$1")
        .replace(/[*_`~]/g, "")
        .trim();
      return cleaned.replace(/^(?:Observation|Decision|Hypothesis)\s*:\s*/i, "").trim();
    })
    .find(Boolean);
  if (!firstLine) {
    return "";
  }
  return firstLine.length > 64 ? `${firstLine.slice(0, 63).trimEnd()}…` : firstLine;
};

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum);

const floatingPanelPosition = (
  anchor: NotesEditorAnchor,
  preferredWidth: number,
  preferredHeight: number
): { left: number; top: number } => {
  if (typeof window === "undefined") {
    return { left: anchor.left, top: anchor.bottom + 8 };
  }
  const width = Math.min(preferredWidth, Math.max(0, window.innerWidth - 24));
  const left = clamp(anchor.left, 12, Math.max(12, window.innerWidth - width - 12));
  const below = anchor.bottom + 8;
  const top =
    below + preferredHeight <= window.innerHeight - 12
      ? below
      : Math.max(12, anchor.top - preferredHeight - 8);
  return { left, top };
};

const centeredFloatingPosition = (
  anchor: NotesEditorAnchor,
  preferredWidth: number,
  height: number
): { left: number; top: number } => {
  if (typeof window === "undefined") {
    return { left: anchor.left, top: Math.max(8, anchor.top - height - 8) };
  }
  const halfWidth = Math.min(preferredWidth, Math.max(0, window.innerWidth - 16)) / 2;
  const left = clamp(anchor.left, halfWidth + 8, Math.max(halfWidth + 8, window.innerWidth - halfWidth - 8));
  const top = anchor.top >= height + 16 ? anchor.top - height - 8 : anchor.bottom + 8;
  return { left, top };
};

const noteRecordForLocalDraft = (): NoteRecord => {
  const now = new Date().toISOString();
  return {
    note_id: LOCAL_DRAFT_ID,
    title: "",
    body_markdown: "",
    pinned: false,
    editor_mode: "markdown",
    created_at: now,
    updated_at: now,
  };
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
    .replace(/^\s*[-*+]\s+(?:\[[ xX]\]\s*)?/gm, "")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/==([^=\n]+)==/g, "$1")
    .replace(/[*_`~>]/g, "")
    .replace(/\|/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 160);

const listTitle = (item: Pick<NoteListItem, "title" | "snippet">): string =>
  item.title.trim() || titleFromBody(item.snippet) || "Untitled";

const listSnippet = (item: Pick<NoteListItem, "title" | "snippet">): string => {
  const lines = item.snippet.split("\n");
  const firstMeaningfulLine = lines.findIndex((line) => Boolean(titleFromBody(line)));
  if (item.title.trim()) {
    if (
      firstMeaningfulLine >= 0 &&
      titleFromBody(lines[firstMeaningfulLine]).toLocaleLowerCase() ===
        item.title.trim().toLocaleLowerCase()
    ) {
      return cleanSnippet(lines.slice(firstMeaningfulLine + 1).join("\n"));
    }
    return cleanSnippet(item.snippet) || "Empty note";
  }
  if (firstMeaningfulLine < 0) {
    return "";
  }
  return cleanSnippet(lines.slice(firstMeaningfulLine + 1).join("\n"));
};

const resourceKindLabel = (kind: string): string => {
  const normalized = kind.trim().toLocaleLowerCase();
  if (normalized === "table") return "DATA";
  if (normalized === "document") return "DOC";
  if (normalized === "image") return "IMG";
  if (normalized === "video") return "VID";
  if (normalized === "file") return "FILE";
  return normalized.slice(0, 4).toLocaleUpperCase() || "FILE";
};

export type NotesPageProps = {
  apiClient: ApiClient;
};

export function NotesPage({ apiClient }: NotesPageProps) {
  const [items, setItems] = useState<NoteListItem[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [activeNote, setActiveNote] = useState<NoteRecord | null>(null);
  const [titleEditing, setTitleEditing] = useState(false);
  const [editorSessionKey, setEditorSessionKey] = useState(0);
  const [noteLoading, setNoteLoading] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashIndex, setSlashIndex] = useState(0);
  const [slashQuery, setSlashQuery] = useState("");
  const [slashAnchor, setSlashAnchor] = useState<NotesEditorAnchor | null>(null);
  const [selectionAnchor, setSelectionAnchor] = useState<NotesEditorAnchor | null>(null);
  const [caretAnchor, setCaretAnchor] = useState<NotesEditorAnchor | null>(null);
  const [editorActive, setEditorActive] = useState<NotesActiveStates>(EDITOR_IDLE);
  const [linkOpen, setLinkOpen] = useState(false);
  const [linkValue, setLinkValue] = useState("");
  const [linkAnchor, setLinkAnchor] = useState<NotesEditorAnchor | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [resourcePickerOpen, setResourcePickerOpen] = useState(false);
  const [resourceQuery, setResourceQuery] = useState("");
  const [resourceItems, setResourceItems] = useState<ResourceRecord[]>([]);
  const [resourceLoading, setResourceLoading] = useState(false);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [resourceRefreshKey, setResourceRefreshKey] = useState(0);
  // Phones begin with navigation instead of dropping people into an
  // arbitrary recent note. Selecting or creating a note swaps panes.
  const [mobileListOpen, setMobileListOpen] = useState(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(COMPACT_NOTES_MEDIA).matches
      : false
  );

  const titleRef = useRef<HTMLInputElement | null>(null);
  const bodyRef = useRef<HTMLTextAreaElement | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  const saveInFlightRef = useRef<Promise<boolean> | null>(null);
  const saveQueuedRef = useRef(false);
  const draftRef = useRef<NoteDraft | null>(null);
  const savedRef = useRef<SavedDraft | null>(null);
  const searchGenerationRef = useRef(0);
  const noteGenerationRef = useRef(0);
  const resourceGenerationRef = useRef(0);
  const initialOpenHandledRef = useRef(false);
  const editorApiRef = useRef<NotesEditorHandle | null>(null);
  const editorActiveRef = useRef<NotesActiveStates>(EDITOR_IDLE);
  const pendingBodyFocusRef = useRef(false);

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

  // Search-as-you-type with a light debounce. The first list request starts
  // immediately; subsequent queries wait just long enough to avoid a request
  // per keystroke. The generation counter rejects stale responses.
  useEffect(() => {
    const delay = searchGenerationRef.current === 0 ? 0 : 180;
    const handle = window.setTimeout(() => {
      void refreshList(searchQuery.trim());
    }, delay);
    return () => window.clearTimeout(handle);
  }, [searchQuery, refreshList]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }
    const compact = window.matchMedia(COMPACT_NOTES_MEDIA);
    const handleChange = (event: MediaQueryListEvent) => {
      if (event.matches && !draftRef.current) {
        setMobileListOpen(true);
      } else if (!event.matches) {
        setMobileListOpen(false);
      }
    };
    compact.addEventListener("change", handleChange);
    return () => compact.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    if (titleEditing) {
      titleRef.current?.focus();
    }
  }, [titleEditing]);

  useEffect(() => {
    if (!resourcePickerOpen) {
      return;
    }
    const generation = ++resourceGenerationRef.current;
    setResourceLoading(true);
    setResourceError(null);
    const handle = window.setTimeout(() => {
      void apiClient
        .listResources({ limit: 10, query: resourceQuery.trim() || undefined })
        .then((page) => {
          if (generation !== resourceGenerationRef.current) {
            return;
          }
          setResourceItems(page.resources);
          setResourceLoading(false);
        })
        .catch((error: unknown) => {
          if (generation !== resourceGenerationRef.current) {
            return;
          }
          setResourceError(error instanceof Error ? error.message : String(error));
          setResourceLoading(false);
        });
    }, 160);
    return () => window.clearTimeout(handle);
  }, [apiClient, resourcePickerOpen, resourceQuery, resourceRefreshKey]);

  const flushSave = useCallback(async (): Promise<boolean> => {
    if (saveInFlightRef.current) {
      saveQueuedRef.current = true;
      return await saveInFlightRef.current;
    }

    const run = async (): Promise<boolean> => {
      do {
        saveQueuedRef.current = false;
        const currentDraft = draftRef.current;
        if (!currentDraft) {
          return true;
        }
        if (currentDraft.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(currentDraft)) {
          setSaveState("draft");
          return true;
        }
        if (currentDraft.noteId !== LOCAL_DRAFT_ID && draftMatchesSaved(currentDraft, savedRef.current)) {
          setSaveState("saved");
          return true;
        }

        const snapshot = { ...currentDraft };
        setSaveState("saving");
        setSaveError(null);
        try {
          if (snapshot.noteId === LOCAL_DRAFT_ID) {
            const record = await apiClient.createNote({
              title: snapshot.title,
              body_markdown: snapshot.body,
              pinned: snapshot.pinned,
              editor_mode: snapshot.editorMode,
            });
            const latest = draftRef.current;
            if (!latest || latest.noteId !== LOCAL_DRAFT_ID) {
              return true;
            }
            const persistedLatest: NoteDraft = { ...latest, noteId: record.note_id };
            draftRef.current = persistedLatest;
            savedRef.current = {
              title: record.title,
              body: record.body_markdown,
              pinned: record.pinned,
              editorMode: record.editor_mode,
            };
            setActiveNoteId(record.note_id);
            setActiveNote((current) =>
              current && current.note_id === LOCAL_DRAFT_ID
                ? {
                    ...record,
                    title: persistedLatest.title,
                    body_markdown: persistedLatest.body,
                    pinned: persistedLatest.pinned,
                    editor_mode: persistedLatest.editorMode,
                  }
                : current
            );
            setItems((current) => [
              {
                note_id: record.note_id,
                title: persistedLatest.title,
                snippet: persistedLatest.body.slice(0, 300),
                pinned: persistedLatest.pinned,
                updated_at: record.updated_at,
              },
              ...current.filter((item) => item.note_id !== record.note_id),
            ]);
            if (!draftMatchesSaved(persistedLatest, savedRef.current)) {
              saveQueuedRef.current = true;
            }
          } else {
            const record = await apiClient.updateNote(snapshot.noteId, {
              title: snapshot.title,
              body_markdown: snapshot.body,
              pinned: snapshot.pinned,
              editor_mode: snapshot.editorMode,
            });
            const latest = draftRef.current;
            if (latest && latest.noteId === record.note_id) {
              savedRef.current = {
                title: record.title,
                body: record.body_markdown,
                pinned: record.pinned,
                editorMode: record.editor_mode,
              };
              setItems((current) => {
                const existing = current.find((item) => item.note_id === record.note_id);
                if (!existing) {
                  return current;
                }
                const updated: NoteListItem = {
                  ...existing,
                  title: latest.title,
                  snippet: latest.body.slice(0, 300),
                  pinned: latest.pinned,
                  updated_at: record.updated_at,
                };
                return [updated, ...current.filter((item) => item.note_id !== record.note_id)];
              });
              setActiveNote((current) =>
                current && current.note_id === record.note_id
                  ? { ...current, updated_at: record.updated_at }
                  : current
              );
              if (!draftMatchesSaved(latest, savedRef.current)) {
                saveQueuedRef.current = true;
              }
            }
          }
        } catch (error) {
          setSaveState("error");
          setSaveError(error instanceof Error ? error.message : "Ultra could not save this note.");
          return false;
        }
      } while (saveQueuedRef.current);

      setSaveState("saved");
      setSaveError(null);
      return true;
    };

    const promise = run();
    saveInFlightRef.current = promise;
    try {
      return await promise;
    } finally {
      if (saveInFlightRef.current === promise) {
        saveInFlightRef.current = null;
      }
    }
  }, [apiClient]);

  const scheduleSave = useCallback(() => {
    setSaveState(draftRef.current?.noteId === LOCAL_DRAFT_ID ? "draft" : "dirty");
    setSaveError(null);
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null;
      void flushSave();
    }, AUTOSAVE_DEBOUNCE_MS);
  }, [flushSave]);

  // Note switches await this same save channel. Unmount gets a best-effort
  // flush too, without claiming offline or browser-shutdown durability.
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
      if (draftRef.current?.noteId === noteId) {
        setMobileListOpen(false);
        return;
      }
      if (draftRef.current && draftRef.current.noteId !== noteId) {
        const localBlank =
          draftRef.current.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(draftRef.current);
        if (!localBlank && !(await flushSave())) {
          return;
        }
      }
      const generation = ++noteGenerationRef.current;
      draftRef.current = null;
      savedRef.current = null;
      setActiveNoteId(noteId);
      setActiveNote(null);
      setNoteLoading(true);
      setSlashOpen(false);
      setSlashQuery("");
      setSlashAnchor(null);
      setSelectionAnchor(null);
      setCaretAnchor(null);
      setLinkOpen(false);
      setLinkAnchor(null);
      setResourcePickerOpen(false);
      setEditorActive(EDITOR_IDLE);
      setEditorError(null);
      setMobileListOpen(false);
      try {
        const record = await apiClient.getNote(noteId);
        if (generation !== noteGenerationRef.current) {
          return;
        }
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
        setTitleEditing(false);
        setEditorSessionKey((current) => current + 1);
        setSaveState("idle");
        setSaveError(null);
      } catch (error) {
        if (generation !== noteGenerationRef.current) {
          return;
        }
        setEditorError(error instanceof Error ? error.message : String(error));
        setActiveNote(null);
      } finally {
        if (generation === noteGenerationRef.current) {
          setNoteLoading(false);
        }
      }
    },
    [apiClient, flushSave]
  );

  const startNewNote = useCallback(async () => {
    const current = draftRef.current;
    const disposableBlank =
      current?.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(current);
    if (current && !disposableBlank && !(await flushSave())) {
      return;
    }
    ++noteGenerationRef.current;
    const record = noteRecordForLocalDraft();
    draftRef.current = {
      noteId: LOCAL_DRAFT_ID,
      title: "",
      body: "",
      pinned: false,
      editorMode: "markdown",
    };
    savedRef.current = null;
    pendingBodyFocusRef.current = true;
    setActiveNoteId(LOCAL_DRAFT_ID);
    setActiveNote(record);
    setTitleEditing(false);
    setEditorSessionKey((value) => value + 1);
    setNoteLoading(false);
    setSaveState("draft");
    setSaveError(null);
    setEditorError(null);
    setSlashOpen(false);
    setSlashQuery("");
    setSlashAnchor(null);
    setSelectionAnchor(null);
    setCaretAnchor(null);
    setLinkOpen(false);
    setLinkAnchor(null);
    setResourcePickerOpen(false);
    setEditorActive(EDITOR_IDLE);
    setSearchQuery("");
    setMobileListOpen(false);
    window.requestAnimationFrame(() => editorApiRef.current?.focus());
  }, [flushSave]);

  // Desktop opens the most recent note. Phones stay on the list until the
  // user chooses a note, so navigation is never hidden behind an arbitrary
  // recent document.
  useEffect(() => {
    if (listLoading || initialOpenHandledRef.current) {
      return;
    }
    initialOpenHandledRef.current = true;
    const compact =
      typeof window.matchMedia === "function" && window.matchMedia(COMPACT_NOTES_MEDIA).matches;
    if (compact) {
      setMobileListOpen(true);
    } else if (!activeNoteId && items.length > 0) {
      void openNote(items[0].note_id);
    }
  }, [listLoading, activeNoteId, items, openNote]);

  const updateDraft = useCallback(
    (patch: Partial<Omit<NoteDraft, "noteId">>) => {
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
      setEditorActive(EDITOR_IDLE);
      editorActiveRef.current = EDITOR_IDLE;
      setLinkOpen(false);
      setLinkAnchor(null);
      setSlashOpen(false);
      setSlashQuery("");
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
    const requestedNoteId = draftRef.current?.noteId ?? activeNoteId;
    if (!requestedNoteId) {
      return;
    }

    // A delete must be ordered after any mutation already in flight. This is
    // especially important for a local draft whose create request may be
    // resolving: deleting the temporary id first would strand the newly
    // created server note. Pending debounce work is intentionally discarded.
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    const generation = noteGenerationRef.current;
    if (saveInFlightRef.current) {
      await saveInFlightRef.current;
    }
    if (generation !== noteGenerationRef.current) {
      return;
    }
    const noteId = draftRef.current?.noteId ?? requestedNoteId;
    ++noteGenerationRef.current;

    if (noteId === LOCAL_DRAFT_ID) {
      draftRef.current = null;
      savedRef.current = null;
      setDeleteDialogOpen(false);
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
      setSaveError(null);
      setEditorError(null);
      setMobileListOpen(true);
      return;
    }
    try {
      await apiClient.deleteNote(noteId);
      draftRef.current = null;
      savedRef.current = null;
      setDeleteDialogOpen(false);
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
      setSaveError(null);
      setEditorError(null);
      setItems((current) => current.filter((item) => item.note_id !== noteId));
      setMobileListOpen(true);
    } catch (error) {
      setDeleteDialogOpen(false);
      setEditorError(
        error instanceof Error ? `Couldn’t delete this note — ${error.message}` : "Couldn’t delete this note."
      );
    }
  }, [activeNoteId, apiClient]);

  const insertSlashBlock = useCallback(
    (block: SlashBlock) => {
      const textarea = bodyRef.current;
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      if (block.id === "media" || block.id === "resource") {
        // Remove the "/" that opened the menu, then hand off to the picker;
        // the upload path inserts the reference at this caret.
        if (textarea && draft.editorMode === "plaintext") {
          const caret = textarea.selectionStart;
          const value =
            textarea.value.slice(0, Math.max(0, caret - 1)) + textarea.value.slice(caret);
          textarea.value = value;
          textarea.setSelectionRange(Math.max(0, caret - 1), Math.max(0, caret - 1));
          updateDraft({ body: value });
          setActiveNote((current) => (current ? { ...current, body_markdown: value } : current));
        }
        setSlashOpen(false);
        setSlashQuery("");
        setSlashAnchor(null);
        if (block.id === "resource") {
          setResourceQuery("");
          setResourcePickerOpen(true);
        } else {
          filePickerRef.current?.();
        }
        return;
      }
      if (draft.editorMode === "markdown") {
        editorApiRef.current?.insertMarkdown(block.insert);
        setSlashOpen(false);
        setSlashQuery("");
        setSlashAnchor(null);
        return;
      }
      if (!textarea) {
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
      setSlashQuery("");
      setSlashAnchor(null);
      textarea.focus();
    },
    [updateDraft]
  );

  const handleRichMenuKeyDown = useCallback(
    (event: KeyboardEvent): boolean => {
      if (!slashOpen) {
        return false;
      }
      const matches = matchingSlashBlocks(slashQuery);
      if (event.key === "Escape") {
        editorApiRef.current?.insertMarkdown(`/${slashQuery}`);
        setSlashOpen(false);
        setSlashQuery("");
        setSlashAnchor(null);
        return true;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        if (matches.length > 0) {
          const delta = event.key === "ArrowDown" ? 1 : -1;
          setSlashIndex((current) => (current + delta + matches.length) % matches.length);
        }
        return true;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        const block = matches[slashIndex];
        if (block) {
          insertSlashBlock(block);
        }
        return true;
      }
      if (event.key === "Backspace") {
        if (slashQuery.length === 0) {
          setSlashOpen(false);
          setSlashAnchor(null);
        } else {
          setSlashQuery((current) => current.slice(0, -1));
          setSlashIndex(0);
        }
        return true;
      }
      if (!event.metaKey && !event.ctrlKey && !event.altKey && event.key.length === 1) {
        setSlashQuery((current) => current + event.key);
        setSlashIndex(0);
        return true;
      }
      return false;
    },
    [insertSlashBlock, slashIndex, slashOpen, slashQuery]
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

  const insertResource = useCallback(
    (resource: ResourceRecord) => {
      insertAtCaret(`\n${markdownForUpload(resource)}\n`);
      setResourcePickerOpen(false);
      setResourceQuery("");
      setSlashAnchor(null);
    },
    [insertAtCaret]
  );

  /* Dropped/pasted/picked files ride the SAME upload pipeline as chat
     attachments, so every file cataloged here appears in Resources — one
     central place to find real data. The note stores only the reference. */
  const handleNoteFilesAdded = useCallback(
    async (files: File[]) => {
      if (!draftRef.current || files.length === 0) {
        return;
      }
      const noteGeneration = noteGenerationRef.current;
      setUploadingCount((count) => count + files.length);
      try {
        const response = await apiClient.uploadFiles(files);
        if (
          noteGeneration !== noteGenerationRef.current ||
          !draftRef.current ||
          response.uploaded.length === 0
        ) {
          return;
        }
        const block = response.uploaded.map(markdownForUpload).join("\n");
        insertAtCaret(`\n${block}\n`);
      } catch (error) {
        if (noteGeneration === noteGenerationRef.current) {
          setEditorError(
            error instanceof Error ? `Upload failed — ${error.message}` : "Upload failed."
          );
        }
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
    if (api && pendingBodyFocusRef.current) {
      pendingBodyFocusRef.current = false;
      window.requestAnimationFrame(() => api.focus());
    }
  }, []);

  const handleActiveStates = useCallback((states: NotesActiveStates) => {
    editorActiveRef.current = states;
    setEditorActive(states);
  }, []);

  const handleSelectionAnchor = useCallback((anchor: NotesEditorAnchor | null) => {
    setSelectionAnchor(anchor);
    if (anchor) {
      setSlashOpen(false);
      setSlashAnchor(null);
    }
  }, []);

  const handleCaretAnchor = useCallback((anchor: NotesEditorAnchor | null) => {
    setCaretAnchor(anchor);
  }, []);

  const handleEditorMenuRequest = useCallback((request: NotesEditorMenuRequest) => {
    setSelectionAnchor(null);
    setLinkOpen(false);
    setLinkAnchor(null);
    setSlashAnchor(request.anchor);
    if (request.kind === "resources") {
      setSlashOpen(false);
      setResourceQuery("");
      setResourcePickerOpen(true);
    } else {
      setResourcePickerOpen(false);
      setSlashIndex(0);
      setSlashQuery("");
      setSlashOpen(true);
    }
  }, []);

  const handleMarkdownChange = useCallback(
    (markdown: string) => {
      // The serializer always appends one newline; trimming it here is what
      // makes open → edit → open round-trips byte-stable (fidelity suite).
      const body = markdown.replace(/\n$/, "");
      setActiveNote((current) =>
        current ? { ...current, body_markdown: body } : current
      );
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
        setLinkAnchor(selectionAnchor ?? caretAnchor);
        setLinkOpen(true);
      }
    },
    [caretAnchor, selectionAnchor]
  );

  const execEditorAction = useCallback((action: NotesEditorAction) => {
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
      setLinkAnchor(null);
      window.requestAnimationFrame(() => editorApiRef.current?.focus());
      return;
    }
    const href =
      /^[a-z][a-z0-9+.-]*:/i.test(raw) || raw.startsWith("/") || raw.startsWith("#")
        ? raw
        : `https://${raw}`;
    editorApiRef.current?.applyLink(href);
    setLinkOpen(false);
    setLinkAnchor(null);
  }, [linkValue]);

  const removeLink = useCallback(() => {
    editorApiRef.current?.removeLink();
    setLinkOpen(false);
    setLinkAnchor(null);
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

  const returnToList = useCallback(async () => {
    const draft = draftRef.current;
    if (draft?.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(draft)) {
      draftRef.current = null;
      savedRef.current = null;
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
    } else if (draft && !(await flushSave())) {
      return;
    }
    setSlashOpen(false);
    setSlashQuery("");
    setSlashAnchor(null);
    setSelectionAnchor(null);
    setCaretAnchor(null);
    setLinkOpen(false);
    setLinkAnchor(null);
    setResourcePickerOpen(false);
    setMobileListOpen(true);
  }, [flushSave]);

  const closeResourcePicker = useCallback(
    (restoreMention: boolean) => {
      const shouldRestoreMention = restoreMention && Boolean(slashAnchor);
      setResourcePickerOpen(false);
      setResourceQuery("");
      setSlashAnchor(null);
      if (shouldRestoreMention) {
        editorApiRef.current?.insertMarkdown("@");
      }
      editorApiRef.current?.focus();
    },
    [slashAnchor]
  );

  const dismissFloatingTools = useCallback(() => {
    setSelectionAnchor(null);
    setCaretAnchor(null);
    setSlashOpen(false);
    setSlashQuery("");
    setSlashAnchor(null);
    setLinkOpen(false);
    setLinkAnchor(null);
    setResourcePickerOpen(false);
  }, []);

  useEffect(() => {
    window.addEventListener("resize", dismissFloatingTools);
    return () => window.removeEventListener("resize", dismissFloatingTools);
  }, [dismissFloatingTools]);

  // Keyboard paths stay complete without turning the UI into a shortcut
  // legend. Raw Markdown remains an explicit expert preference.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      const modifier = event.metaKey || event.ctrlKey;
      if (modifier && !event.shiftKey && event.key.toLowerCase() === "n") {
        event.preventDefault();
        void startNewNote();
        return;
      }
      if (modifier && !event.shiftKey && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (saveTimerRef.current !== null) {
          window.clearTimeout(saveTimerRef.current);
          saveTimerRef.current = null;
        }
        void flushSave();
        return;
      }
      if (modifier && event.shiftKey && event.key.toLowerCase() === "e") {
        event.preventDefault();
        const draft = draftRef.current;
        if (!draft) {
          return;
        }
        switchEditorMode(draft.editorMode === "markdown" ? "plaintext" : "markdown");
        return;
      }
      if (event.key === "Escape") {
        if (resourcePickerOpen) {
          closeResourcePicker(true);
          return;
        }
        if (slashOpen) {
          editorApiRef.current?.insertMarkdown(`/${slashQuery}`);
          setSlashOpen(false);
          setSlashQuery("");
          setSlashAnchor(null);
          editorApiRef.current?.focus();
          return;
        }
        if (linkOpen) {
          setLinkOpen(false);
          setLinkAnchor(null);
          editorApiRef.current?.focus();
          return;
        }
        if (focusMode) {
          setFocusMode(false);
          return;
        }
        const draft = draftRef.current;
        if (draft?.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(draft)) {
          void returnToList();
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    flushSave,
    closeResourcePicker,
    focusMode,
    linkOpen,
    resourcePickerOpen,
    returnToList,
    slashOpen,
    slashQuery,
    startNewNote,
    switchEditorMode,
  ]);

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

  const richSlashBlocks = useMemo(() => matchingSlashBlocks(slashQuery), [slashQuery]);
  const selectionPosition = selectionAnchor
    ? centeredFloatingPosition(selectionAnchor, 190, 40)
    : null;
  const linkPosition = linkAnchor
    ? centeredFloatingPosition(linkAnchor, 480, 48)
    : null;
  const tablePosition = caretAnchor
    ? centeredFloatingPosition(caretAnchor, 430, 40)
    : null;
  const slashPosition = slashAnchor
    ? floatingPanelPosition(slashAnchor, 310, 440)
    : null;
  const resourcePosition = slashAnchor
    ? floatingPanelPosition(slashAnchor, 350, 430)
    : null;
  const localDraftHasContent = Boolean(
    activeNote &&
      activeNote.note_id === LOCAL_DRAFT_ID &&
      (activeNote.title.trim() || activeNote.body_markdown.trim())
  );
  const showTitleInput = Boolean(activeNote?.title.trim()) || titleEditing;

  const saveLabel =
    saveState === "saving"
      ? "Saving…"
      : saveState === "draft"
        ? "Draft · Not saved yet"
      : saveState === "dirty"
        ? "Saving soon…"
        : saveState === "error"
          ? "Couldn’t sync"
          : activeNote
            ? `Saved ${relativeTime(activeNote.updated_at)}`
            : "";

  return (
    <div
      className="notes-page"
      data-testid="notes-page"
      data-mobile-list={mobileListOpen ? "true" : undefined}
      data-focus-mode={focusMode ? "true" : undefined}
    >
      <aside className="notes-list">
        <div className="notes-list-head">
          <h2>Notes</h2>
          <Button type="button" variant="outline" size="sm" className="notes-new-button" onClick={() => void startNewNote()}>
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
            aria-label="Search notes"
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </label>
        {listLoading ? (
          <div className="notes-list-state" role="status">
            <Loader2 className="animate-spin" aria-hidden="true" /> Loading notes…
          </div>
        ) : listError ? (
          <div className="notes-list-state notes-list-error" role="alert">
            <span>{listError}</span>
            <button type="button" onClick={() => void refreshList(searchQuery.trim())}>
              Try again
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="notes-list-state notes-list-empty">
            {searchQuery.trim() ? (
              <>No notes match “{searchQuery.trim()}”.</>
            ) : (
              <>
                <span>No notes yet.</span>
                <button type="button" onClick={() => void startNewNote()}>Write your first note</button>
              </>
            )}
          </div>
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
                    aria-label={`${item.pinned ? "Pinned, " : ""}${listTitle(item)}, ${relativeTime(item.updated_at)}`}
                    onClick={() => void openNote(item.note_id)}
                  >
                    <span className="notes-row-title">
                      {item.pinned ? <Pin className="notes-row-pin" aria-label="Pinned" /> : null}
                      {listTitle(item)}
                    </span>
                    <span className="notes-row-snippet">{listSnippet(item)}</span>
                    <span className="notes-row-time">{relativeTime(item.updated_at)}</span>
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
        <div className="notes-list-foot">
          Notes are private to you.
        </div>
      </aside>

      <section className="notes-editor" onScroll={dismissFloatingTools}>
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
                  onClick={() => void returnToList()}
                >
                  <ChevronLeft aria-hidden="true" />
                  Notes
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      className="notes-save-state"
                      data-state={saveState}
                      aria-label={`Save status: ${saveLabel}`}
                    >
                      {uploadingCount > 0 || saveState === "saving" ? (
                        <Loader2 className="animate-spin" aria-hidden="true" />
                      ) : saveState === "error" ? (
                        <AlertCircle aria-hidden="true" />
                      ) : saveState === "saved" || saveState === "idle" ? (
                        <Check aria-hidden="true" />
                      ) : null}
                      <span role="status" aria-live="polite">
                        {uploadingCount > 0
                          ? `Uploading ${uploadingCount} file${uploadingCount === 1 ? "" : "s"} to Resources…`
                          : saveLabel}
                      </span>
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="notes-save-menu">
                    {saveState === "error" ? (
                      <>
                        <div className="notes-save-detail" role="alert">
                          <strong>Ultra couldn’t sync this note.</strong>
                          <span>{saveError || "Your text is still open in this editor."}</span>
                        </div>
                        <DropdownMenuItem onSelect={() => void flushSave()}>
                          <RotateCcw aria-hidden="true" /> Retry sync
                        </DropdownMenuItem>
                      </>
                    ) : activeNote.note_id === LOCAL_DRAFT_ID ? (
                      <div className="notes-save-detail">
                        <strong>
                          {localDraftHasContent ? "Waiting to sync" : "Not created yet"}
                        </strong>
                        <span>
                          {localDraftHasContent
                            ? "Ultra will create this note after you pause."
                            : "This blank draft disappears if you leave it. Typing creates the note."}
                        </span>
                      </div>
                    ) : (
                      <div className="notes-save-detail">
                        <strong>Synced to Ultra</strong>
                        <span>Autosave is on. Press ⌘S anytime to save immediately.</span>
                      </div>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div className="notes-editor-actions">
                <button
                  type="button"
                  className="notes-icon-button"
                  aria-label={activeNote.pinned ? "Unpin note" : "Pin note"}
                  title={activeNote.pinned ? "Unpin note" : "Pin note"}
                  aria-pressed={activeNote.pinned}
                  onClick={() => void togglePinned()}
                >
                  <Pin aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="notes-icon-button"
                  aria-label="Link an Ultra resource"
                  title="Link an Ultra resource"
                  onClick={() => {
                    setSlashAnchor(null);
                    if (resourcePickerOpen) {
                      closeResourcePicker(false);
                    } else {
                      setResourceQuery("");
                      setResourcePickerOpen(true);
                    }
                  }}
                >
                  <Paperclip aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="notes-icon-button notes-focus-button"
                  aria-label={focusMode ? "Show note list" : "Focus on this note"}
                  title={focusMode ? "Show note list" : "Focus on this note"}
                  aria-pressed={focusMode}
                  onClick={() => setFocusMode((current) => !current)}
                >
                  {focusMode ? <Minimize2 aria-hidden="true" /> : <Expand aria-hidden="true" />}
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      className="notes-icon-button"
                      aria-label="More note actions"
                      title="More note actions"
                    >
                      <MoreHorizontal aria-hidden="true" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onSelect={() =>
                        switchEditorMode(activeNote.editor_mode === "plaintext" ? "markdown" : "plaintext")
                      }
                    >
                      <FileText aria-hidden="true" />
                      {activeNote.editor_mode === "plaintext"
                        ? "Return to formatted editor"
                        : "Edit Markdown source"}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => {
                        setResourcePickerOpen(false);
                        filePickerRef.current?.();
                      }}
                    >
                      <Upload aria-hidden="true" /> Upload a file
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="notes-delete-menu-item" onSelect={() => setDeleteDialogOpen(true)}>
                      <Trash aria-hidden="true" />
                      {activeNote.note_id === LOCAL_DRAFT_ID ? "Discard draft" : "Delete note"}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            {editorError ? (
              <div className="notes-editor-error" role="alert">
                <AlertCircle aria-hidden="true" />
                <span>{editorError}</span>
                <button type="button" aria-label="Dismiss error" onClick={() => setEditorError(null)}>
                  <X aria-hidden="true" />
                </button>
              </div>
            ) : null}

            {showTitleInput ? (
              <input
                ref={titleRef}
                className="notes-title-input"
                value={activeNote.title}
                placeholder="Add a title"
                aria-label="Note title, optional"
                onFocus={() => setTitleEditing(true)}
                onBlur={() => {
                  setTitleEditing(false);
                  void flushSave();
                }}
                onChange={(event) => {
                  const title = event.target.value;
                  setActiveNote((current) => (current ? { ...current, title } : current));
                  updateDraft({ title });
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === "Tab") {
                    event.preventDefault();
                    setTitleEditing(false);
                    if (draftRef.current?.editorMode === "markdown") {
                      editorApiRef.current?.focus();
                    } else {
                      bodyRef.current?.focus();
                    }
                  }
                }}
              />
            ) : null}

            <div className="notes-document-context">
              <span>Personal note</span>
              <span>
                {activeNote.note_id === LOCAL_DRAFT_ID
                  ? "Created when you start writing"
                  : `Edited ${relativeTime(activeNote.updated_at)}`}
              </span>
              {!showTitleInput ? (
                <button
                  type="button"
                  aria-label="Add a title"
                  onClick={() => setTitleEditing(true)}
                >
                  Add title
                </button>
              ) : null}
            </div>

            {activeNote.editor_mode === "plaintext" ? (
              <div className="notes-body-shell">
                <div className="notes-source-banner">
                  <span>Markdown source</span>
                  <button type="button" onClick={() => switchEditorMode("markdown")}>
                    Return to formatted editor
                  </button>
                </div>
                <textarea
                  ref={bodyRef}
                  className="notes-body-input"
                  value={activeNote.body_markdown}
                  placeholder={'Write in markdown — "/" for blocks; drop or paste images and videos'}
                  aria-label="Note body"
                  onPaste={handleBodyPaste}
                  onChange={(event) => {
                    const body = event.target.value;
                    setActiveNote((current) =>
                      current ? { ...current, body_markdown: body } : current
                    );
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
                {editorActive.inTable ? (
                  <div
                    className="notes-table-tools"
                    role="toolbar"
                    aria-label="Table actions"
                    style={
                      tablePosition
                        ? { left: `${tablePosition.left}px`, top: `${tablePosition.top}px` }
                        : undefined
                    }
                  >
                    <button type="button" onMouseDown={keepEditorFocus} onClick={() => execEditorAction("rowBelow")}>+ Row</button>
                    <button type="button" onMouseDown={keepEditorFocus} onClick={() => execEditorAction("rowDelete")}>− Row</button>
                    <button type="button" onMouseDown={keepEditorFocus} onClick={() => execEditorAction("colRight")}>+ Column</button>
                    <button type="button" onMouseDown={keepEditorFocus} onClick={() => execEditorAction("colDelete")}>− Column</button>
                    <button type="button" onMouseDown={keepEditorFocus} onClick={() => execEditorAction("tableDelete")}>Delete table</button>
                  </div>
                ) : null}
                {linkOpen ? (
                  <div
                    className="notes-link-pop notes-link-pop-contextual"
                    style={
                      linkPosition
                        ? { left: `${linkPosition.left}px`, top: `${linkPosition.top}px` }
                        : { left: "50%", top: "72px" }
                    }
                  >
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
                          setLinkAnchor(null);
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
                  onPointerDownCapture={dismissFloatingTools}
                >
                  {activeNote.body_markdown.trim() === "" ? (
                    <div className="notes-editor-placeholder" aria-hidden="true">
                      Start writing…
                    </div>
                  ) : null}
                  <Suspense
                    fallback={
                      <div className="notes-editor-empty" role="status">
                        <Loader2 className="animate-spin" aria-hidden="true" /> Loading editor…
                      </div>
                    }
                  >
                    <LazyMarkdownNoteEditor
                      key={editorSessionKey}
                      defaultMarkdown={activeNote.body_markdown}
                      resourceUrl={resolveResourceUrl}
                      onMarkdownChange={handleMarkdownChange}
                      onBlur={handleEditorBlur}
                      onActiveStatesChange={handleActiveStates}
                      onSelectionAnchorChange={handleSelectionAnchor}
                      onCaretAnchorChange={handleCaretAnchor}
                      onMenuRequest={handleEditorMenuRequest}
                      onMenuKeyDown={handleRichMenuKeyDown}
                      bindApi={bindEditorApi}
                    />
                  </Suspense>
                  {activeNote.body_markdown.trim() === "" ? (
                    <div className="notes-editor-hint" aria-hidden="true">
                      <kbd>/</kbd> blocks <span>·</span> <kbd>@</kbd> Ultra resources
                    </div>
                  ) : null}
                </div>
              </>
            )}

            {selectionAnchor && activeNote.editor_mode !== "plaintext" ? (
              <div
                className="notes-selection-toolbar"
                role="toolbar"
                aria-label="Text formatting"
                style={
                  selectionPosition
                    ? { left: `${selectionPosition.left}px`, top: `${selectionPosition.top}px` }
                    : undefined
                }
              >
                <button
                  type="button"
                  aria-label="Bold"
                  title="Bold (⌘B)"
                  aria-pressed={editorActive.bold}
                  data-active={editorActive.bold ? "true" : undefined}
                  onMouseDown={keepEditorFocus}
                  onClick={() => execEditorAction("bold")}
                >
                  <Bold aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="Italic"
                  title="Italic (⌘I)"
                  aria-pressed={editorActive.italic}
                  data-active={editorActive.italic ? "true" : undefined}
                  onMouseDown={keepEditorFocus}
                  onClick={() => execEditorAction("italic")}
                >
                  <Italic aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="Highlight"
                  title="Highlight (⌘⇧H)"
                  aria-pressed={editorActive.highlight}
                  data-active={editorActive.highlight ? "true" : undefined}
                  onMouseDown={keepEditorFocus}
                  onClick={() => execEditorAction("highlight")}
                >
                  <Highlighter aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="Inline code"
                  title="Inline code (⌘E)"
                  aria-pressed={editorActive.code}
                  data-active={editorActive.code ? "true" : undefined}
                  onMouseDown={keepEditorFocus}
                  onClick={() => execEditorAction("code")}
                >
                  <Code aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="Link"
                  title="Link (⌘K)"
                  aria-pressed={editorActive.link}
                  data-active={editorActive.link ? "true" : undefined}
                  onMouseDown={keepEditorFocus}
                  onClick={() => {
                    setLinkValue(editorActiveRef.current.linkHref ?? "");
                    setLinkAnchor(selectionAnchor);
                    setLinkOpen(true);
                  }}
                >
                  <LinkIcon aria-hidden="true" />
                </button>
              </div>
            ) : null}

            {slashOpen && activeNote.editor_mode !== "plaintext" ? (
              <div
                className="notes-slash notes-slash-floating"
                role="listbox"
                aria-label="Insert block"
                style={
                  slashPosition
                    ? { left: `${slashPosition.left}px`, top: `${slashPosition.top}px` }
                    : undefined
                }
              >
                <div className="notes-slash-heading">
                  {slashQuery ? `Insert /${slashQuery}` : "Insert a block"}
                </div>
                {richSlashBlocks.length > 0 ? (
                  richSlashBlocks.map((block, index) => (
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
                  ))
                ) : (
                  <div className="notes-slash-empty">No matching block</div>
                )}
              </div>
            ) : null}

            {resourcePickerOpen ? (
              <div
                className="notes-resource-picker"
                role="dialog"
                aria-label="Link an Ultra resource"
                data-floating={slashAnchor ? "true" : undefined}
                style={
                  resourcePosition
                    ? { left: `${resourcePosition.left}px`, top: `${resourcePosition.top}px` }
                    : undefined
                }
              >
                <div className="notes-resource-picker-head">
                  <strong>Link an Ultra resource</strong>
                  <button
                    type="button"
                    aria-label="Close resource picker"
                    onClick={() => closeResourcePicker(true)}
                  >
                    <X aria-hidden="true" />
                  </button>
                </div>
                <label className="notes-resource-search">
                  <Search aria-hidden="true" />
                  <input
                    autoFocus
                    type="search"
                    value={resourceQuery}
                    placeholder="Search recent resources"
                    aria-label="Search Ultra resources"
                    onChange={(event) => setResourceQuery(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        closeResourcePicker(true);
                      }
                    }}
                  />
                </label>
                <div className="notes-resource-results">
                  {resourceLoading ? (
                    <div className="notes-resource-state" role="status">
                      <Loader2 className="animate-spin" aria-hidden="true" /> Loading resources…
                    </div>
                  ) : resourceError ? (
                    <div className="notes-resource-state" role="alert">
                      <span>{resourceError}</span>
                      <button type="button" onClick={() => setResourceRefreshKey((value) => value + 1)}>
                        Try again
                      </button>
                    </div>
                  ) : resourceItems.length === 0 ? (
                    <div className="notes-resource-state">No matching resources.</div>
                  ) : (
                    resourceItems.map((resource) => (
                      <button
                        key={resource.file_id}
                        type="button"
                        className="notes-resource-row"
                        onClick={() => insertResource(resource)}
                      >
                        <span className="notes-resource-kind">
                          {resourceKindLabel(resource.resource_kind)}
                        </span>
                        <span className="notes-resource-copy">
                          <strong>{resource.original_name}</strong>
                          <small>{relativeTime(resource.created_at)}</small>
                        </span>
                        <Plus aria-hidden="true" />
                      </button>
                    ))
                  )}
                </div>
                <button
                  type="button"
                  className="notes-resource-upload"
                  onClick={() => {
                    setResourcePickerOpen(false);
                    setResourceQuery("");
                    setSlashAnchor(null);
                    filePickerRef.current?.();
                  }}
                >
                  <Upload aria-hidden="true" /> Upload a new file
                </button>
              </div>
            ) : null}
          </>
        ) : noteLoading ? (
          <div className="notes-editor-empty" role="status">
            <Loader2 className="animate-spin" aria-hidden="true" /> Opening…
          </div>
        ) : editorError && activeNoteId ? (
          <div className="notes-editor-empty notes-editor-load-error" role="alert">
            <AlertCircle aria-hidden="true" />
            <strong>Couldn’t open this note.</strong>
            <span>{editorError}</span>
            <div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="notes-load-back"
                onClick={() => setMobileListOpen(true)}
              >
                <ChevronLeft data-icon="inline-start" aria-hidden="true" /> Back to notes
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={() => void openNote(activeNoteId)}>
                <RotateCcw data-icon="inline-start" aria-hidden="true" /> Try again
              </Button>
            </div>
          </div>
        ) : (
          <div className="notes-editor-empty notes-editor-welcome">
            <span>Select a note or start a new one.</span>
            <Button type="button" variant="outline" size="sm" onClick={() => void startNewNote()}>
              <Plus data-icon="inline-start" aria-hidden="true" /> New note
            </Button>
          </div>
        )}
        </FileUpload>
      </section>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {activeNote?.note_id === LOCAL_DRAFT_ID ? "Discard this draft?" : "Delete this note?"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {activeNote?.note_id === LOCAL_DRAFT_ID
                ? "This draft has not been created in Ultra. Its current text will be discarded."
                : "Deletion is permanent in the current Notes service and cannot be undone."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => void deleteActiveNote()}>
              {activeNote?.note_id === LOCAL_DRAFT_ID ? "Discard draft" : "Delete note"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
