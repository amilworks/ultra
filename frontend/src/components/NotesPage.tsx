import {
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
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
  MessageSquare,
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
import {
  ApiError,
  isDefinitiveNoteWriteReplayRejection,
  isDeterministicNoteWriteRejection,
  isNoteRevisionConflict,
  type ApiClient,
  type NoteEditorMode,
  type NoteListItem,
  type NoteRecord,
} from "@/lib/api";
import { lazyNamedWithRetry } from "@/lib/lazy-retry";
import {
  clearNoteDraftRecovery,
  enableNoteDraftRecoveryScope,
  readLatestNoteDraftRecovery,
  readNoteDraftRecovery,
  resolveBrowserLocalStorage,
  writeNoteDraftRecovery,
  type NoteDraftRecoveryWriteResult,
  type NoteDraftRecoveryRecord,
} from "@/lib/noteDraftRecovery";
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
const NOTES_LIST_PAGE_SIZE = 50;
const LOCAL_DRAFT_ID = "__ultra_local_note_draft__";
const COMPACT_NOTES_MEDIA = "(max-width: 960px)";

type SaveState = "idle" | "draft" | "dirty" | "saving" | "saved" | "conflict" | "error";
type DeviceRecoveryState = "available" | Exclude<NoteDraftRecoveryWriteResult, "stored" | "scope_disabled">;

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
  revision: number;
  createKey: string | null;
  createAttempt: {
    title: string;
    body: string;
    pinned: boolean;
    editorMode: NoteEditorMode;
  } | null;
};

type SavedDraft = Pick<NoteDraft, "title" | "body" | "pinned" | "editorMode" | "revision">;

const meaningfulDraft = (draft: NoteDraft): boolean =>
  draft.title.trim().length > 0 || draft.body.trim().length > 0;

const freshNoteCreateKey = (): string =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? `note-create:${crypto.randomUUID()}`
    : `note-create:${Date.now()}:${Math.random().toString(16).slice(2)}`;

const draftMatchesSaved = (draft: NoteDraft, saved: SavedDraft | null): boolean =>
  Boolean(
    saved &&
      saved.title === draft.title &&
      saved.body === draft.body &&
      saved.pinned === draft.pinned &&
      saved.editorMode === draft.editorMode
  );

const noteRecordMatchesDraftContent = (record: NoteRecord, draft: NoteDraft): boolean =>
  record.title === draft.title &&
  record.body_markdown === draft.body &&
  record.pinned === draft.pinned &&
  record.editor_mode === draft.editorMode;

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
    revision: 0,
    content_digest: "",
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

const noteContentUpdatedAt = (
  note: Pick<NoteListItem, "content_updated_at" | "updated_at">
): string => note.content_updated_at || note.updated_at;

const sortBrowseNotes = (notes: readonly NoteListItem[]): NoteListItem[] =>
  [...notes].sort((left, right) => {
    if (left.pinned !== right.pinned) return left.pinned ? -1 : 1;
    const byContentTime =
      new Date(noteContentUpdatedAt(right)).getTime() -
      new Date(noteContentUpdatedAt(left)).getTime();
    return byContentTime || left.note_id.localeCompare(right.note_id);
  });

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
  recoveryScope?: string | null;
  initialNoteId?: string | null;
  refreshVersion?: number;
  listRequestVersion?: number;
  initialDraft?: { key: string; bodyMarkdown: string } | null;
  onInitialDraftConsumed?: (key: string) => void;
  onActiveNoteChange?: (noteId: string | null) => void;
  onUseInChat?: (note: { note_id: string; title: string; revision: number }) => void;
  onLogoutFlushReady?: (flush: (() => Promise<boolean>) | null) => void;
};

export function NotesPage({
  apiClient,
  recoveryScope = null,
  initialNoteId = null,
  refreshVersion = 0,
  listRequestVersion = 0,
  initialDraft = null,
  onInitialDraftConsumed,
  onActiveNoteChange,
  onUseInChat,
  onLogoutFlushReady,
}: NotesPageProps) {
  const [items, setItems] = useState<NoteListItem[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listLoadingMore, setListLoadingMore] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [listNextOffset, setListNextOffset] = useState(0);
  const [listHasMore, setListHasMore] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [listMoreError, setListMoreError] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [activeNote, setActiveNote] = useState<NoteRecord | null>(null);
  const [titleEditing, setTitleEditing] = useState(false);
  const [editorSessionKey, setEditorSessionKey] = useState(0);
  const [noteLoading, setNoteLoading] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [recoveredChanges, setRecoveredChanges] = useState(false);
  const [recoveredMissingOriginal, setRecoveredMissingOriginal] = useState(false);
  const [deviceRecoveryState, setDeviceRecoveryState] =
    useState<DeviceRecoveryState>("available");
  const [conflictRecord, setConflictRecord] = useState<NoteRecord | null>(null);
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
  const plaintextTabExitArmedRef = useRef(false);
  const itemsRef = useRef(items);
  useLayoutEffect(() => {
    itemsRef.current = items;
  }, [items]);
  const saveTimerRef = useRef<number | null>(null);
  const saveInFlightRef = useRef<Promise<boolean> | null>(null);
  const saveQueuedRef = useRef(false);
  const recoveryTimerRef = useRef<number | null>(null);
  const pendingRecoveryRef = useRef<NoteDraft | null>(null);
  // State drives the banner; this synchronous guard drives the write channel.
  // It prevents blur/unmount flushes from retrying a known-stale revision
  // before the writer has explicitly chosen which version to keep.
  const saveBlockedByConflictRef = useRef(false);
  // Once a create response is uncertain, passive blur/unmount flushes must not
  // keep replaying it behind the writer's back. Recovery stays durable until
  // an explicit retry, discard, or fresh edit resumes the same idempotent
  // attempt.
  const createReconciliationRequiredRef = useRef(false);
  // A typed client rejection proves that create did not commit, but passive
  // blur/unmount flushes must not resubmit the same rejected draft forever.
  // Editing or an explicit Retry sync reopens the write channel; Discard stays
  // local because there is no uncertain server-side create to reconcile.
  const createRetryBlockedUntilEditRef = useRef(false);
  const deleteReconciliationNoteIdRef = useRef<string | null>(null);
  const draftRef = useRef<NoteDraft | null>(null);
  const savedRef = useRef<SavedDraft | null>(null);
  const searchGenerationRef = useRef(0);
  const searchQueryRef = useRef("");
  const noteGenerationRef = useRef(0);
  const resourceGenerationRef = useRef(0);
  const initialOpenHandledRef = useRef(false);
  const editorApiRef = useRef<NotesEditorHandle | null>(null);
  const editorActiveRef = useRef<NotesActiveStates>(EDITOR_IDLE);
  const pendingBodyFocusRef = useRef(false);
  const refreshVersionRef = useRef(refreshVersion);
  const listRequestVersionRef = useRef(listRequestVersion);

  useEffect(() => {
    enableNoteDraftRecoveryScope(recoveryScope);
  }, [recoveryScope]);

  const cancelPendingRecovery = useCallback((noteId?: string): void => {
    if (noteId && pendingRecoveryRef.current?.noteId !== noteId) return;
    pendingRecoveryRef.current = null;
    if (recoveryTimerRef.current !== null) {
      window.clearTimeout(recoveryTimerRef.current);
      recoveryTimerRef.current = null;
    }
  }, []);

  const readRecovery = useCallback(
    (noteId: string): NoteDraftRecoveryRecord | null => {
      const storage = resolveBrowserLocalStorage();
      if (!storage) {
        setDeviceRecoveryState("unavailable");
        return null;
      }
      const result = readNoteDraftRecovery(storage, recoveryScope, noteId);
      if (result.status === "unavailable") setDeviceRecoveryState("unavailable");
      return result.record;
    },
    [recoveryScope]
  );

  const readLatestRecovery = useCallback((): NoteDraftRecoveryRecord | null => {
    const storage = resolveBrowserLocalStorage();
    if (!storage) {
      setDeviceRecoveryState("unavailable");
      return null;
    }
    const result = readLatestNoteDraftRecovery(storage, recoveryScope);
    if (result.status === "unavailable") setDeviceRecoveryState("unavailable");
    return result.record;
  }, [recoveryScope]);

  const clearRecovery = useCallback(
    (noteId: string): void => {
      cancelPendingRecovery(noteId);
      const storage = resolveBrowserLocalStorage();
      if (!storage) return;
      clearNoteDraftRecovery(storage, recoveryScope, noteId);
    },
    [cancelPendingRecovery, recoveryScope]
  );

  const persistRecovery = useCallback(
    (draft: NoteDraft | null): void => {
      if (!draft) return;
      cancelPendingRecovery(draft.noteId);
      // A pristine local draft is disposable. Emptying an existing Note (or
      // changing only pin/mode) is a real unsaved edit and must remain durable.
      if (
        draft.noteId === LOCAL_DRAFT_ID &&
        !meaningfulDraft(draft) &&
        !draft.createAttempt
      ) {
        clearRecovery(draft.noteId);
        return;
      }
      const storage = resolveBrowserLocalStorage();
      if (!storage) {
        setDeviceRecoveryState("unavailable");
        return;
      }
      const result = writeNoteDraftRecovery(storage, recoveryScope, {
        note_id: draft.noteId,
        title: draft.title,
        body_markdown: draft.body,
        pinned: draft.pinned,
        editor_mode: draft.editorMode,
        expected_revision: draft.revision,
        ...(draft.createKey ? { create_key: draft.createKey } : {}),
        ...(draft.createAttempt
          ? {
              create_attempt: {
                title: draft.createAttempt.title,
                body_markdown: draft.createAttempt.body,
                pinned: draft.createAttempt.pinned,
                editor_mode: draft.createAttempt.editorMode,
              },
            }
          : {}),
      });
      if (result === "stored") setDeviceRecoveryState("available");
      else if (result !== "scope_disabled") setDeviceRecoveryState(result);
    },
    [cancelPendingRecovery, clearRecovery, recoveryScope]
  );

  const scheduleRecovery = useCallback(
    (draft: NoteDraft): void => {
      pendingRecoveryRef.current = {
        ...draft,
        createAttempt: draft.createAttempt ? { ...draft.createAttempt } : null,
      };
      if (recoveryTimerRef.current !== null) return;
      recoveryTimerRef.current = window.setTimeout(() => {
        recoveryTimerRef.current = null;
        const pending = pendingRecoveryRef.current;
        pendingRecoveryRef.current = null;
        if (pending) persistRecovery(pending);
      }, 400);
    },
    [persistRecovery]
  );

  const flushRecoveryNow = useCallback((): void => {
    const pending = pendingRecoveryRef.current;
    if (!pending) return;
    pendingRecoveryRef.current = null;
    if (recoveryTimerRef.current !== null) {
      window.clearTimeout(recoveryTimerRef.current);
      recoveryTimerRef.current = null;
    }
    persistRecovery(pending);
  }, [persistRecovery]);

  const refreshList = useCallback(
    async (query: string, offset = 0, quiet = false) => {
      const generation = ++searchGenerationRef.current;
      if (offset === 0 && !quiet) setListLoading(true);
      if (offset > 0) setListLoadingMore(true);
      if (offset === 0) {
        setListError(null);
        setListMoreError(null);
      }
      else setListMoreError(null);
      try {
        const page = await apiClient.listNotes({
          query: query || undefined,
          limit: NOTES_LIST_PAGE_SIZE,
          offset,
        });
        if (generation !== searchGenerationRef.current) {
          return;
        }
        const consumed = offset + page.notes.length;
        setItems((current) => {
          if (offset === 0) return page.notes;
          const seen = new Set(current.map((note) => note.note_id));
          return [...current, ...page.notes.filter((note) => !seen.has(note.note_id))];
        });
        setTotalCount(page.total_count);
        setListNextOffset(consumed);
        setListHasMore(page.notes.length > 0 && consumed < page.total_count);
        if (!quiet) setListLoading(false);
        if (offset > 0) setListLoadingMore(false);
      } catch (error) {
        if (generation !== searchGenerationRef.current) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        if (offset === 0) {
          setItems([]);
          setListError(message);
        }
        else setListMoreError(message);
        if (!quiet) setListLoading(false);
        if (offset > 0) setListLoadingMore(false);
      }
    },
    [apiClient]
  );

  // Search-as-you-type with a light debounce. The first list request starts
  // immediately; subsequent queries wait just long enough to avoid a request
  // per keystroke. The generation counter rejects stale responses.
  useEffect(() => {
    searchQueryRef.current = searchQuery.trim();
    const delay = searchGenerationRef.current === 0 ? 0 : 180;
    // Invalidate any old request as soon as the visible query changes, not
    // after the debounce. Rows from browse/query A must never appear clickable
    // as if they matched query B.
    ++searchGenerationRef.current;
    setItems([]);
    setListError(null);
    setListMoreError(null);
    setListLoading(true);
    const handle = window.setTimeout(() => {
      void refreshList(searchQuery.trim(), 0);
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

  const flushSave = useCallback(async (
    options: { reconcileCreate?: boolean } = {}
  ): Promise<boolean> => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    if (saveBlockedByConflictRef.current) {
      return false;
    }
    if (
      createRetryBlockedUntilEditRef.current &&
      draftRef.current?.noteId === LOCAL_DRAFT_ID &&
      !options.reconcileCreate
    ) {
      return false;
    }
    if (
      createReconciliationRequiredRef.current &&
      draftRef.current?.noteId === LOCAL_DRAFT_ID &&
      draftRef.current.createAttempt &&
      !options.reconcileCreate
    ) {
      return false;
    }
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
        if (
          currentDraft.noteId === LOCAL_DRAFT_ID &&
          !meaningfulDraft(currentDraft) &&
          !currentDraft.createAttempt
        ) {
          clearRecovery(currentDraft.noteId);
          setRecoveredChanges(false);
          setSaveState("draft");
          return true;
        }
        if (currentDraft.noteId !== LOCAL_DRAFT_ID && draftMatchesSaved(currentDraft, savedRef.current)) {
          clearRecovery(currentDraft.noteId);
          setRecoveredChanges(false);
          setSaveState("saved");
          return true;
        }

        const snapshot = { ...currentDraft };
        setSaveState("saving");
        setSaveError(null);
        try {
          if (snapshot.noteId === LOCAL_DRAFT_ID) {
            const createKey = snapshot.createKey ?? freshNoteCreateKey();
            const createAttempt = snapshot.createAttempt ?? {
              title: snapshot.title,
              body: snapshot.body,
              pinned: snapshot.pinned,
              editorMode: snapshot.editorMode,
            };
            if (!snapshot.createKey || !snapshot.createAttempt) {
              const latest = draftRef.current;
              if (latest?.noteId === LOCAL_DRAFT_ID) {
                draftRef.current = { ...latest, createKey, createAttempt };
                persistRecovery(draftRef.current);
              }
            }
            const record = await apiClient.createNote({
              title: createAttempt.title,
              body_markdown: createAttempt.body,
              pinned: createAttempt.pinned,
              editor_mode: createAttempt.editorMode,
            }, createKey);
            const latest = draftRef.current;
            if (!latest || latest.noteId !== LOCAL_DRAFT_ID) {
              return true;
            }
            const persistedLatest: NoteDraft = {
              ...latest,
              noteId: record.note_id,
              revision: record.revision,
              createKey: null,
              createAttempt: null,
            };
            createReconciliationRequiredRef.current = false;
            createRetryBlockedUntilEditRef.current = false;
            clearRecovery(LOCAL_DRAFT_ID);
            draftRef.current = persistedLatest;
            savedRef.current = {
              title: record.title,
              body: record.body_markdown,
              pinned: record.pinned,
              editorMode: record.editor_mode,
              revision: record.revision,
            };
            setActiveNoteId(record.note_id);
            onActiveNoteChange?.(record.note_id);
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
            if (searchQueryRef.current) {
              void refreshList(searchQueryRef.current, 0, true);
            } else {
              setItems((current) =>
                sortBrowseNotes([
                  {
                    note_id: record.note_id,
                    title: persistedLatest.title,
                    snippet: persistedLatest.body.slice(0, 300),
                    pinned: persistedLatest.pinned,
                    revision: record.revision,
                    content_updated_at: record.content_updated_at ?? record.updated_at,
                    updated_at: record.updated_at,
                  },
                  ...current.filter((item) => item.note_id !== record.note_id),
                ])
              );
              setTotalCount((current) => current + 1);
              setListNextOffset((current) => current + 1);
            }
            if (!draftMatchesSaved(persistedLatest, savedRef.current)) {
              persistRecovery(persistedLatest);
              saveQueuedRef.current = true;
            } else {
              clearRecovery(record.note_id);
            }
          } else {
            const record = await apiClient.updateNote(snapshot.noteId, {
              title: snapshot.title,
              body_markdown: snapshot.body,
              pinned: snapshot.pinned,
              editor_mode: snapshot.editorMode,
              expected_revision: snapshot.revision,
            });
            const latest = draftRef.current;
            if (latest && latest.noteId === record.note_id) {
              const revisedLatest = { ...latest, revision: record.revision };
              draftRef.current = revisedLatest;
              savedRef.current = {
                title: record.title,
                body: record.body_markdown,
                pinned: record.pinned,
                editorMode: record.editor_mode,
                revision: record.revision,
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
                  revision: record.revision,
                  content_updated_at: record.content_updated_at ?? record.updated_at,
                  updated_at: record.updated_at,
                };
                const next = current.map((item) =>
                  item.note_id === record.note_id ? updated : item
                );
                return searchQueryRef.current ? next : sortBrowseNotes(next);
              });
              setActiveNote((current) =>
                current && current.note_id === record.note_id
                  ? {
                      ...current,
                      revision: record.revision,
                      content_digest: record.content_digest,
                      content_updated_at: record.content_updated_at ?? record.updated_at,
                      updated_at: record.updated_at,
                    }
                  : current
              );
              if (searchQueryRef.current) {
                void refreshList(searchQueryRef.current, 0, true);
              }
              if (!draftMatchesSaved(revisedLatest, savedRef.current)) {
                persistRecovery(revisedLatest);
                saveQueuedRef.current = true;
              } else {
                clearRecovery(record.note_id);
              }
            }
          }
        } catch (error) {
          if (snapshot.noteId === LOCAL_DRAFT_ID) {
            const replayingUncertainCreate = Boolean(
              snapshot.createKey && snapshot.createAttempt
            );
            if (
              (!replayingUncertainCreate &&
                isDeterministicNoteWriteRejection(error)) ||
              (replayingUncertainCreate &&
                isDefinitiveNoteWriteReplayRejection(error, "create"))
            ) {
              // A current-call deterministic rejection, or the endpoint's
              // receipt-first post-lookup terminal proof, establishes that
              // this exact create did not commit. Retire its frozen request so
              // a corrected draft gets a fresh key and payload.
              const latest = draftRef.current;
              if (latest?.noteId === LOCAL_DRAFT_ID) {
                const retryableDraft: NoteDraft = {
                  ...latest,
                  createKey: freshNoteCreateKey(),
                  createAttempt: null,
                };
                draftRef.current = retryableDraft;
                persistRecovery(retryableDraft);
              }
              createReconciliationRequiredRef.current = false;
              createRetryBlockedUntilEditRef.current = true;
            } else {
              // A transport failure, malformed success response, or 5xx may
              // follow a committed transaction. Preserve the exact key and
              // payload until an idempotent replay resolves that ambiguity.
              createReconciliationRequiredRef.current = true;
              createRetryBlockedUntilEditRef.current = false;
            }
          }
          if (
            snapshot.noteId !== LOCAL_DRAFT_ID &&
            error instanceof ApiError &&
            error.status === 404
          ) {
            const latest = draftRef.current;
            if (latest?.noteId === snapshot.noteId) {
              const recoveredAsNew: NoteDraft = {
                ...latest,
                noteId: LOCAL_DRAFT_ID,
                revision: 0,
                createKey: freshNoteCreateKey(),
                createAttempt: null,
              };
              clearRecovery(snapshot.noteId);
              draftRef.current = recoveredAsNew;
              savedRef.current = null;
              persistRecovery(recoveredAsNew);
              createReconciliationRequiredRef.current = false;
              createRetryBlockedUntilEditRef.current = false;
              saveBlockedByConflictRef.current = false;
              setConflictRecord(null);
              setActiveNoteId(LOCAL_DRAFT_ID);
              onActiveNoteChange?.(null);
              setActiveNote((current) =>
                current
                  ? {
                      ...noteRecordForLocalDraft(),
                      title: recoveredAsNew.title,
                      body_markdown: recoveredAsNew.body,
                      pinned: recoveredAsNew.pinned,
                      editor_mode: recoveredAsNew.editorMode,
                    }
                  : current
              );
              const removedLoadedNote = itemsRef.current.some(
                (item) => item.note_id === snapshot.noteId
              );
              setItems((current) =>
                current.filter((item) => item.note_id !== snapshot.noteId)
              );
              if (removedLoadedNote) {
                setTotalCount((current) => Math.max(0, current - 1));
                setListNextOffset((current) => Math.max(0, current - 1));
              }
              setRecoveredChanges(true);
              setRecoveredMissingOriginal(true);
              saveQueuedRef.current = true;
              continue;
            }
          }
          if (snapshot.noteId !== LOCAL_DRAFT_ID && isNoteRevisionConflict(error)) {
            try {
              const latestRecord = await apiClient.getNote(snapshot.noteId);
              if (noteRecordMatchesDraftContent(latestRecord, snapshot)) {
                const latest = draftRef.current;
                if (latest?.noteId === snapshot.noteId) {
                  const revisedLatest = { ...latest, revision: latestRecord.revision };
                  draftRef.current = revisedLatest;
                  savedRef.current = {
                    title: latestRecord.title,
                    body: latestRecord.body_markdown,
                    pinned: latestRecord.pinned,
                    editorMode: latestRecord.editor_mode,
                    revision: latestRecord.revision,
                  };
                  setItems((current) => {
                    const next = current.map((item) =>
                      item.note_id === latestRecord.note_id
                        ? {
                            ...item,
                            title: latestRecord.title,
                            snippet: latestRecord.body_markdown.slice(0, 300),
                            pinned: latestRecord.pinned,
                            revision: latestRecord.revision,
                            content_updated_at:
                              latestRecord.content_updated_at ?? latestRecord.updated_at,
                            updated_at: latestRecord.updated_at,
                          }
                        : item
                    );
                    return searchQueryRef.current ? next : sortBrowseNotes(next);
                  });
                  setActiveNote((current) =>
                    current?.note_id === latestRecord.note_id ? latestRecord : current
                  );
                  saveBlockedByConflictRef.current = false;
                  setConflictRecord(null);
                  if (draftMatchesSaved(revisedLatest, savedRef.current)) {
                    clearRecovery(latestRecord.note_id);
                  } else {
                    persistRecovery(revisedLatest);
                    saveQueuedRef.current = true;
                  }
                }
                continue;
              }
              saveBlockedByConflictRef.current = true;
              if (draftRef.current?.noteId === snapshot.noteId) {
                setConflictRecord(latestRecord);
                setSaveState("conflict");
                setSaveError(
                  "This note changed elsewhere. Your version is still open and has not been overwritten."
                );
              }
            } catch {
              saveBlockedByConflictRef.current = true;
              setSaveState("conflict");
              setSaveError(
                "This note changed elsewhere. Your version is still open; reload the note before saving again."
              );
            }
            return false;
          }
          setSaveState("error");
          setSaveError(error instanceof Error ? error.message : "Ultra could not save this note.");
          return false;
        }
      } while (saveQueuedRef.current);

      setSaveState("saved");
      setSaveError(null);
      setRecoveredChanges(false);
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
  }, [apiClient, clearRecovery, onActiveNoteChange, persistRecovery, refreshList]);

  useEffect(() => {
    if (!onLogoutFlushReady) return;
    const flushForLogout = async (): Promise<boolean> => {
      flushRecoveryNow();
      return await flushSave({ reconcileCreate: true });
    };
    onLogoutFlushReady(flushForLogout);
    return () => onLogoutFlushReady(null);
  }, [flushRecoveryNow, flushSave, onLogoutFlushReady]);

  const scheduleSave = useCallback(() => {
    if (saveBlockedByConflictRef.current) {
      setSaveState("conflict");
      return;
    }
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
      flushRecoveryNow();
      void flushSave();
    };
  }, [flushRecoveryNow, flushSave]);

  useEffect(() => {
    const handlePageHide = () => {
      flushRecoveryNow();
      void flushSave();
    };
    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [flushRecoveryNow, flushSave]);

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
      onActiveNoteChange?.(noteId);
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
      setRecoveredMissingOriginal(false);
      setMobileListOpen(false);
      const recovery = readRecovery(noteId);
      try {
        const record = await apiClient.getNote(noteId);
        if (generation !== noteGenerationRef.current) {
          return;
        }
        const serverDraft: NoteDraft = {
          noteId: record.note_id,
          title: record.title,
          body: record.body_markdown,
          pinned: record.pinned,
          editorMode: record.editor_mode === "plaintext" ? "plaintext" : "markdown",
          revision: record.revision,
          createKey: null,
          createAttempt: null,
        };
        const recoveredDraft: NoteDraft | null = recovery
          ? {
              noteId: record.note_id,
              title: recovery.title,
              body: recovery.body_markdown,
              pinned: recovery.pinned,
              editorMode: recovery.editor_mode,
              revision: recovery.expected_revision,
              createKey: null,
              createAttempt: null,
            }
          : null;
        const hasRecoveredChanges = Boolean(
          recoveredDraft && !draftMatchesSaved(recoveredDraft, {
            title: serverDraft.title,
            body: serverDraft.body,
            pinned: serverDraft.pinned,
            editorMode: serverDraft.editorMode,
            revision: serverDraft.revision,
          })
        );
        if (recovery && !hasRecoveredChanges) clearRecovery(noteId);
        draftRef.current = hasRecoveredChanges && recoveredDraft ? recoveredDraft : serverDraft;
        savedRef.current = {
          title: record.title,
          body: record.body_markdown,
          pinned: record.pinned,
          editorMode: serverDraft.editorMode,
          revision: record.revision,
        };
        setActiveNote(
          hasRecoveredChanges && recoveredDraft
            ? {
                ...record,
                title: recoveredDraft.title,
                body_markdown: recoveredDraft.body,
                pinned: recoveredDraft.pinned,
                editor_mode: recoveredDraft.editorMode,
              }
            : record
        );
        setTitleEditing(false);
        setEditorSessionKey((current) => current + 1);
        setRecoveredChanges(hasRecoveredChanges);
        if (hasRecoveredChanges && recoveredDraft?.revision !== record.revision) {
          saveBlockedByConflictRef.current = true;
          setConflictRecord(record);
          setSaveState("conflict");
          setSaveError(
            "Recovered unsaved changes, but this note also changed elsewhere. Your version is still open."
          );
        } else {
          setSaveError(null);
          saveBlockedByConflictRef.current = false;
          setConflictRecord(null);
          if (hasRecoveredChanges) scheduleSave();
          else setSaveState("idle");
        }
      } catch (error) {
        if (generation !== noteGenerationRef.current) {
          return;
        }
        if (recovery && error instanceof ApiError && error.status === 404) {
          const recoveredDraft: NoteDraft = {
            noteId: LOCAL_DRAFT_ID,
            title: recovery.title,
            body: recovery.body_markdown,
            pinned: recovery.pinned,
            editorMode: recovery.editor_mode,
            revision: 0,
            createKey: freshNoteCreateKey(),
            createAttempt: null,
          };
          clearRecovery(noteId);
          draftRef.current = recoveredDraft;
          savedRef.current = null;
          persistRecovery(recoveredDraft);
          createReconciliationRequiredRef.current = false;
          createRetryBlockedUntilEditRef.current = false;
          saveBlockedByConflictRef.current = false;
          setConflictRecord(null);
          setActiveNoteId(LOCAL_DRAFT_ID);
          onActiveNoteChange?.(null);
          setActiveNote({
            ...noteRecordForLocalDraft(),
            title: recoveredDraft.title,
            body_markdown: recoveredDraft.body,
            pinned: recoveredDraft.pinned,
            editor_mode: recoveredDraft.editorMode,
          });
          setItems((current) => current.filter((item) => item.note_id !== noteId));
          setTotalCount((current) => Math.max(0, current - 1));
          setListNextOffset((current) => Math.max(0, current - 1));
          setEditorSessionKey((current) => current + 1);
          setRecoveredChanges(true);
          setRecoveredMissingOriginal(true);
          setSaveState("draft");
          setSaveError(null);
          setEditorError(null);
          scheduleSave();
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
    [apiClient, clearRecovery, flushSave, onActiveNoteChange, persistRecovery, readRecovery, scheduleSave]
  );

  const startNewNote = useCallback(async (
    initialBody = "",
    recovery: NoteDraftRecoveryRecord | null = null
  ): Promise<boolean> => {
    const current = draftRef.current;
    const disposableBlank =
      current?.noteId === LOCAL_DRAFT_ID && !meaningfulDraft(current);
    if (current && !disposableBlank && !(await flushSave())) {
      return false;
    }
    ++noteGenerationRef.current;
    const nextDraft: NoteDraft = {
      noteId: LOCAL_DRAFT_ID,
      title: recovery?.title ?? "",
      body: recovery?.body_markdown ?? initialBody,
      pinned: recovery?.pinned ?? false,
      editorMode: recovery?.editor_mode ?? "markdown",
      revision: 0,
      createKey: recovery?.create_key ?? freshNoteCreateKey(),
      createAttempt: recovery?.create_attempt
        ? {
            title: recovery.create_attempt.title,
            body: recovery.create_attempt.body_markdown,
            pinned: recovery.create_attempt.pinned,
            editorMode: recovery.create_attempt.editor_mode,
          }
        : null,
    };
    const record = {
      ...noteRecordForLocalDraft(),
      title: nextDraft.title,
      body_markdown: nextDraft.body,
      pinned: nextDraft.pinned,
      editor_mode: nextDraft.editorMode,
    };
    draftRef.current = nextDraft;
    // Installing a recovered/prefilled draft is a fresh, visible resume point.
    // Give its stable create key one automatic attempt; a failure flips the
    // reconciliation guard so blur/unmount cannot hot-loop it afterward.
    createReconciliationRequiredRef.current = false;
    createRetryBlockedUntilEditRef.current = false;
    persistRecovery(nextDraft);
    savedRef.current = null;
    pendingBodyFocusRef.current = true;
    setActiveNoteId(LOCAL_DRAFT_ID);
    onActiveNoteChange?.(null);
    setActiveNote(record);
    setTitleEditing(false);
    setEditorSessionKey((value) => value + 1);
    setNoteLoading(false);
    setSaveState("draft");
    setSaveError(null);
    setRecoveredChanges(Boolean(recovery));
    setRecoveredMissingOriginal(false);
    saveBlockedByConflictRef.current = false;
    setConflictRecord(null);
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
    if (meaningfulDraft(nextDraft) || nextDraft.createAttempt) {
      scheduleSave();
    }
    window.requestAnimationFrame(() => editorApiRef.current?.focus());
    return true;
  }, [flushSave, onActiveNoteChange, persistRecovery, scheduleSave]);

  const consumedInitialDraftKeyRef = useRef<string | null>(null);
  const installingInitialDraftKeyRef = useRef<string | null>(null);
  const failedInitialDraftKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (
      !initialDraft ||
      consumedInitialDraftKeyRef.current === initialDraft.key ||
      installingInitialDraftKeyRef.current === initialDraft.key
    ) return;
    if (failedInitialDraftKeyRef.current === initialDraft.key) {
      if (saveState !== "saved" && saveState !== "idle" && saveState !== "dirty") {
        return;
      }
      failedInitialDraftKeyRef.current = null;
    }
    installingInitialDraftKeyRef.current = initialDraft.key;
    void startNewNote(initialDraft.bodyMarkdown).then((installed) => {
      if (installed) {
        consumedInitialDraftKeyRef.current = initialDraft.key;
        failedInitialDraftKeyRef.current = null;
        onInitialDraftConsumed?.(initialDraft.key);
      } else {
        failedInitialDraftKeyRef.current = initialDraft.key;
      }
    }).finally(() => {
      if (installingInitialDraftKeyRef.current === initialDraft.key) {
        installingInitialDraftKeyRef.current = null;
      }
    });
  }, [initialDraft, onInitialDraftConsumed, saveState, startNewNote]);

  // Desktop opens the most recently edited content. Browse ordering is pinned
  // first, so resolve this with the explicit recent endpoint rather than
  // guessing from the first (possibly paginated) browse page. Phones stay on
  // the list until the user chooses a note.
  useEffect(() => {
    if (listLoading || initialDraft || initialOpenHandledRef.current) {
      return;
    }
    initialOpenHandledRef.current = true;
    const compact =
      typeof window.matchMedia === "function" && window.matchMedia(COMPACT_NOTES_MEDIA).matches;
    if (initialNoteId?.trim()) {
      void openNote(initialNoteId.trim());
    } else if (!activeNoteId) {
      const recovery = readLatestRecovery();
      if (recovery?.note_id === LOCAL_DRAFT_ID) {
        void startNewNote("", recovery);
      } else if (recovery) {
        void openNote(recovery.note_id);
      } else if (compact) {
        setMobileListOpen(true);
      } else if (items.length > 0) {
        const navigationGeneration = noteGenerationRef.current;
        void apiClient
          .listNotes({ sort: "recent", limit: 1, offset: 0 })
          .then((page) => {
            const mostRecent = page.notes[0];
            if (
              mostRecent &&
              navigationGeneration === noteGenerationRef.current &&
              !draftRef.current
            ) {
              void openNote(mostRecent.note_id);
            }
          })
          .catch(() => {
            // The browse list remains fully usable if this convenience request
            // fails; do not replace it with a second, unrelated error state.
          });
      }
    }
  }, [
    activeNoteId,
    apiClient,
    initialDraft,
    initialNoteId,
    items.length,
    listLoading,
    openNote,
    readLatestRecovery,
    startNewNote,
  ]);

  useEffect(() => {
    const requestedNoteId = initialNoteId?.trim();
    if (!requestedNoteId || listLoading) {
      return;
    }
    if (requestedNoteId === activeNoteId) {
      // Forward navigation may target the Note that remains warm behind the
      // mobile list. Reopen the pane without refetching or remounting it.
      setMobileListOpen(false);
      return;
    }
    void openNote(requestedNoteId);
  }, [activeNoteId, initialNoteId, listLoading, openNote]);

  const updateDraft = useCallback(
    (patch: Partial<Omit<NoteDraft, "noteId">>) => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      const nextDraft = { ...draft, ...patch };
      draftRef.current = nextDraft;
      createReconciliationRequiredRef.current = false;
      createRetryBlockedUntilEditRef.current = false;
      scheduleRecovery(nextDraft);
      scheduleSave();
    },
    [scheduleRecovery, scheduleSave]
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
      createReconciliationRequiredRef.current = false;
      createRetryBlockedUntilEditRef.current = false;
      persistRecovery(draftRef.current);
      setActiveNote((current) => (current ? { ...current, editor_mode: nextMode } : current));
      setEditorActive(EDITOR_IDLE);
      editorActiveRef.current = EDITOR_IDLE;
      setLinkOpen(false);
      setLinkAnchor(null);
      setSlashOpen(false);
      setSlashQuery("");
      void flushSave();
    },
    [flushSave, persistRecovery]
  );

  const togglePinned = useCallback(async () => {
    const draft = draftRef.current;
    if (!draft) {
      return;
    }
    const nextPinned = !draft.pinned;
    draftRef.current = { ...draft, pinned: nextPinned };
    createReconciliationRequiredRef.current = false;
    createRetryBlockedUntilEditRef.current = false;
    persistRecovery(draftRef.current);
    setActiveNote((current) => (current ? { ...current, pinned: nextPinned } : current));
    setItems((current) => {
      const next = current.map((item) =>
        item.note_id === draft.noteId ? { ...item, pinned: nextPinned } : item
      );
      return searchQueryRef.current ? next : sortBrowseNotes(next);
    });
    const saved = await flushSave();
    if (saved && !searchQueryRef.current) {
      // Pinning changes the browse sort tuple and can move a row across the
      // current offset boundary. Restart browse pagination from the server so
      // a later Load more cannot skip or duplicate shifted rows.
      void refreshList("", 0, true);
    }
  }, [flushSave, persistRecovery, refreshList]);

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
    if (
      draftRef.current?.noteId === LOCAL_DRAFT_ID &&
      draftRef.current.createAttempt &&
      !(await flushSave({ reconcileCreate: true }))
    ) {
      setDeleteDialogOpen(false);
      setEditorError(
        "Couldn’t confirm whether this draft was already created. It is still here; try discarding again when the connection returns."
      );
      return;
    }
    const noteId = draftRef.current?.noteId ?? requestedNoteId;
    ++noteGenerationRef.current;

    if (noteId === LOCAL_DRAFT_ID) {
      clearRecovery(noteId);
      createReconciliationRequiredRef.current = false;
      createRetryBlockedUntilEditRef.current = false;
      draftRef.current = null;
      savedRef.current = null;
      setDeleteDialogOpen(false);
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
      setSaveError(null);
      setRecoveredChanges(false);
      saveBlockedByConflictRef.current = false;
      setConflictRecord(null);
      setEditorError(null);
      setMobileListOpen(true);
      onActiveNoteChange?.(null);
      return;
    }
    try {
      await apiClient.deleteNote(noteId);
    } catch (error) {
      const reconciledMissing = error instanceof ApiError && error.status === 404;
      if (!reconciledMissing) {
        deleteReconciliationNoteIdRef.current = isDeterministicNoteWriteRejection(error)
          ? null
          : noteId;
        setDeleteDialogOpen(false);
        setEditorError(
          error instanceof Error ? `Couldn’t delete this note — ${error.message}` : "Couldn’t delete this note."
        );
        return;
      }
    }
    deleteReconciliationNoteIdRef.current = null;
    try {
      clearRecovery(noteId);
      createReconciliationRequiredRef.current = false;
      createRetryBlockedUntilEditRef.current = false;
      draftRef.current = null;
      savedRef.current = null;
      setDeleteDialogOpen(false);
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
      setSaveError(null);
      setRecoveredChanges(false);
      saveBlockedByConflictRef.current = false;
      setConflictRecord(null);
      setEditorError(null);
      const removedLoadedNote = items.some((item) => item.note_id === noteId);
      setItems((current) => current.filter((item) => item.note_id !== noteId));
      setTotalCount((current) => Math.max(0, current - 1));
      if (removedLoadedNote) {
        setListNextOffset((current) => Math.max(0, current - 1));
      }
      if (searchQueryRef.current) {
        void refreshList(searchQueryRef.current, 0, true);
      }
      setMobileListOpen(true);
      onActiveNoteChange?.(null);
    } catch (error) {
      // Local cleanup should not normally throw, but retain a visible failure
      // rather than allowing a render exception to strand the page.
      setDeleteDialogOpen(false);
      setEditorError(
        error instanceof Error ? `Couldn’t delete this note — ${error.message}` : "Couldn’t delete this note."
      );
    }
  }, [activeNoteId, apiClient, clearRecovery, flushSave, items, onActiveNoteChange, refreshList]);

  const adoptNoteRecord = useCallback((record: NoteRecord) => {
    clearRecovery(record.note_id);
    createReconciliationRequiredRef.current = false;
    createRetryBlockedUntilEditRef.current = false;
    const editorMode = record.editor_mode === "plaintext" ? "plaintext" : "markdown";
    draftRef.current = {
      noteId: record.note_id,
      title: record.title,
      body: record.body_markdown,
      pinned: record.pinned,
      editorMode,
      revision: record.revision,
      createKey: null,
      createAttempt: null,
    };
    savedRef.current = {
      title: record.title,
      body: record.body_markdown,
      pinned: record.pinned,
      editorMode,
      revision: record.revision,
    };
    setActiveNote(record);
    setItems((current) => {
      const next = current.map((item) =>
        item.note_id === record.note_id
          ? {
              ...item,
              title: record.title,
              snippet: record.body_markdown.slice(0, 300),
              pinned: record.pinned,
              revision: record.revision,
              content_updated_at: record.content_updated_at ?? record.updated_at,
              updated_at: record.updated_at,
            }
          : item
      );
      return searchQueryRef.current ? next : sortBrowseNotes(next);
    });
    setEditorSessionKey((current) => current + 1);
    saveBlockedByConflictRef.current = false;
    setConflictRecord(null);
    setSaveError(null);
    setRecoveredChanges(false);
    setSaveState("saved");
  }, [clearRecovery]);

  const useLatestConflictVersion = useCallback(() => {
    const record = conflictRecord;
    if (!record || draftRef.current?.noteId !== record.note_id) {
      return;
    }
    adoptNoteRecord(record);
  }, [adoptNoteRecord, conflictRecord]);

  const saveConflictVersion = useCallback(async () => {
    const record = conflictRecord;
    const draft = draftRef.current;
    if (!record || !draft || draft.noteId !== record.note_id) {
      return;
    }
    draftRef.current = { ...draft, revision: record.revision };
    persistRecovery(draftRef.current);
    setActiveNote((current) =>
      current?.note_id === record.note_id ? { ...current, revision: record.revision } : current
    );
    saveBlockedByConflictRef.current = false;
    setConflictRecord(null);
    setSaveState("dirty");
    setSaveError(null);
    await flushSave();
  }, [conflictRecord, flushSave, persistRecovery]);

  const retryConflictLoad = useCallback(async () => {
    const noteId = draftRef.current?.noteId;
    if (!noteId || noteId === LOCAL_DRAFT_ID) {
      return;
    }
    setSaveError(null);
    try {
      const latest = await apiClient.getNote(noteId);
      if (draftRef.current?.noteId === noteId) {
        setConflictRecord(latest);
      }
    } catch (error) {
      setSaveError(
        error instanceof Error
          ? `Couldn’t load the latest version — ${error.message}`
          : "Couldn’t load the latest version."
      );
    }
  }, [apiClient]);

  const sendActiveNoteToChat = useCallback(async () => {
    if (!onUseInChat || !(await flushSave())) {
      return;
    }
    const draft = draftRef.current;
    if (!draft || draft.noteId === LOCAL_DRAFT_ID || draft.revision < 1) {
      return;
    }
    onUseInChat({
      note_id: draft.noteId,
      title: draft.title.trim() || titleFromBody(draft.body) || "Untitled",
      revision: draft.revision,
    });
  }, [flushSave, onUseInChat]);

  // A committed proposal or Undo may update the Note while this page remains
  // mounted. Refresh only a clean editor; a local draft must never disappear
  // because another surface changed the same Note.
  useEffect(() => {
    if (refreshVersionRef.current === refreshVersion) {
      return;
    }
    refreshVersionRef.current = refreshVersion;
    const draft = draftRef.current;
    if (
      !draft ||
      draft.noteId === LOCAL_DRAFT_ID ||
      !draftMatchesSaved(draft, savedRef.current) ||
      saveBlockedByConflictRef.current ||
      conflictRecord
    ) {
      return;
    }
    let cancelled = false;
    void apiClient
      .getNote(draft.noteId)
      .then((record) => {
        if (!cancelled && draftRef.current?.noteId === record.note_id) {
          adoptNoteRecord(record);
        }
      })
      .catch(() => {
        // The normal open/save paths surface actionable errors. A background
        // freshness check stays silent so it cannot interrupt writing.
      });
    return () => {
      cancelled = true;
    };
  }, [adoptNoteRecord, apiClient, conflictRecord, refreshVersion]);

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
    flushRecoveryNow();
    void flushSave();
  }, [flushRecoveryNow, flushSave]);

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
      if (
        (event.metaKey || event.ctrlKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "h"
      ) {
        event.preventDefault();
        event.stopPropagation();
        editorApiRef.current?.exec("highlight");
        return;
      }
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
      if (event.key === "Escape") {
        plaintextTabExitArmedRef.current = true;
        return;
      }
      if (event.key === "Tab" && plaintextTabExitArmedRef.current) {
        // One explicit Escape hands the next Tab back to the browser so a
        // keyboard user can leave the raw-source editor without a pointer.
        plaintextTabExitArmedRef.current = false;
        return;
      }
      plaintextTabExitArmedRef.current = false;
      if (event.key === "Tab" && !event.shiftKey) {
        // Sublime-grade plaintext: Tab indents until the user explicitly
        // arms the browser's normal focus traversal with Escape.
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
      clearRecovery(draft.noteId);
      draftRef.current = null;
      savedRef.current = null;
      setActiveNoteId(null);
      setActiveNote(null);
      setSaveState("idle");
      setRecoveredChanges(false);
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
    onActiveNoteChange?.(null);
  }, [clearRecovery, flushSave, onActiveNoteChange]);

  // App-level Back/Forward owns the URL, while this component owns the mobile
  // split-pane state. A monotonic request bridges “?view=notes” back to the
  // list without remounting (and therefore without dropping an unsaved draft).
  useEffect(() => {
    if (listRequestVersionRef.current === listRequestVersion) {
      return;
    }
    listRequestVersionRef.current = listRequestVersion;
    const compact =
      typeof window.matchMedia === "function" && window.matchMedia(COMPACT_NOTES_MEDIA).matches;
    if (compact) {
      void returnToList();
    }
  }, [listRequestVersion, returnToList]);

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
        flushRecoveryNow();
        void flushSave();
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
    flushRecoveryNow,
    flushSave,
    closeResourcePicker,
    focusMode,
    linkOpen,
    resourcePickerOpen,
    returnToList,
    slashOpen,
    slashQuery,
    startNewNote,
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
      const label = listGroupFor(noteContentUpdatedAt(item));
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
  const searchActive = Boolean(searchQuery.trim());
  const displayedGroups = useMemo(
    () =>
      searchActive
        ? [{ label: null, rows: items }]
        : groupedItems.map((group) => ({ ...group, label: group.label as string | null })),
    [groupedItems, items, searchActive]
  );

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
      : recoveredChanges && (saveState === "draft" || saveState === "dirty")
        ? "Recovered unsaved changes"
      : saveState === "draft"
        ? "Draft · Not saved yet"
      : saveState === "dirty"
        ? "Saving soon…"
        : saveState === "conflict"
          ? "Review changes"
        : saveState === "error"
          ? "Couldn’t sync"
          : activeNote
            ? `Saved ${relativeTime(activeNote.updated_at)}`
            : "";
  const deviceRecoveryMessage =
    deviceRecoveryState === "unavailable"
      ? "Device recovery unavailable. Server autosave is still on."
      : deviceRecoveryState === "too_large"
        ? "This note is too large for device recovery. Server autosave is still on."
        : deviceRecoveryState === "budget_exceeded"
          ? "Device recovery is full. Server autosave is still on."
          : null;

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
            {displayedGroups.map((group, groupIndex) => (
              <div key={group.label ?? `results-${groupIndex}`}>
                {group.label ? <div className="notes-group-label">{group.label}</div> : null}
                {group.rows.map((item) => (
                  <button
                    key={item.note_id}
                    type="button"
                    className="notes-row"
                    data-active={item.note_id === activeNoteId ? "true" : undefined}
                    aria-label={`${item.pinned ? "Pinned, " : ""}${listTitle(item)}, ${relativeTime(noteContentUpdatedAt(item))}`}
                    onClick={() => void openNote(item.note_id)}
                  >
                    <span className="notes-row-title">
                      {item.pinned ? <Pin className="notes-row-pin" aria-label="Pinned" /> : null}
                      {listTitle(item)}
                    </span>
                    <span className="notes-row-snippet">{listSnippet(item)}</span>
                    <span className="notes-row-time">{relativeTime(noteContentUpdatedAt(item))}</span>
                  </button>
                ))}
              </div>
            ))}
            {listHasMore && listNextOffset < totalCount ? (
              listMoreError ? (
                <div className="notes-list-state notes-list-error" role="alert">
                  <span>Couldn’t load more notes.</span>
                  <button
                    type="button"
                    onClick={() => void refreshList(searchQuery.trim(), listNextOffset)}
                  >
                    Try again
                  </button>
                </div>
              ) : (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="notes-load-more"
                  disabled={listLoadingMore}
                  onClick={() => void refreshList(searchQuery.trim(), listNextOffset)}
                >
                  {listLoadingMore ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
                  Load more
                </Button>
              )
            ) : null}
          </div>
        )}
        <div className="notes-list-foot">
          Saved Notes live in your Ultra account. Unsynced edits are also kept on this device for
          up to 30 days and cleared when you sign out. {onUseInChat
            ? "Ultra reads Notes only when you use one or ask it to search."
            : "Using Notes in chat is currently unavailable."}
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
                      ) : saveState === "error" || saveState === "conflict" ? (
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
                    {saveState === "conflict" ? (
                      <div className="notes-save-detail" role="alert">
                        <strong>This note changed elsewhere.</strong>
                        <span>Your edits are still open. Choose which version to keep in the editor.</span>
                      </div>
                    ) : saveState === "error" ? (
                      <>
                        <div className="notes-save-detail" role="alert">
                          <strong>Ultra couldn’t sync this note.</strong>
                          <span>{saveError || "Your text is still open in this editor."}</span>
                        </div>
                        <DropdownMenuItem
                          onSelect={() => void flushSave({ reconcileCreate: true })}
                        >
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
                    {onUseInChat ? (
                      <DropdownMenuItem
                        disabled={activeNote.note_id === LOCAL_DRAFT_ID && !localDraftHasContent}
                        onSelect={() => void sendActiveNoteToChat()}
                      >
                        <MessageSquare aria-hidden="true" /> Use in chat
                      </DropdownMenuItem>
                    ) : null}
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

            {deviceRecoveryMessage ? (
              <div className="notes-editor-error" role="status">
                <AlertCircle aria-hidden="true" />
                <span>{deviceRecoveryMessage}</span>
              </div>
            ) : null}

            {recoveredMissingOriginal ? (
              <div
                className="border-border bg-muted/45 mx-5 mt-3 rounded-xl border px-4 py-3 text-sm sm:mx-8"
                role="status"
              >
                <strong className="block font-medium">Recovered edits are ready as a new note.</strong>
                <span className="text-muted-foreground">
                  The original note is no longer available. Ultra kept your device copy and will save it as a new note.
                </span>
              </div>
            ) : null}

            {saveState === "conflict" ? (
              <div
                className="mx-5 mt-3 flex flex-col gap-3 rounded-xl border border-border bg-muted/45 px-4 py-3 text-sm sm:mx-8 sm:flex-row sm:items-center sm:justify-between"
                role="alert"
              >
                <div className="min-w-0">
                  <strong className="block font-medium">This note changed elsewhere.</strong>
                  <span className="text-muted-foreground">
                    {conflictRecord
                      ? "Your edits are safe in this editor. Keep the latest saved note or save your version over it."
                      : saveError || "Your edits are safe. Load the latest saved note before choosing a version."}
                  </span>
                </div>
                <div className="flex shrink-0 flex-wrap gap-2">
                  {conflictRecord ? (
                    <>
                      <Button type="button" variant="outline" size="sm" onClick={useLatestConflictVersion}>
                        Use latest
                      </Button>
                      <Button type="button" size="sm" onClick={() => void saveConflictVersion()}>
                        Save my version
                      </Button>
                    </>
                  ) : (
                    <Button type="button" variant="outline" size="sm" onClick={() => void retryConflictLoad()}>
                      Load latest
                    </Button>
                  )}
                </div>
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
                  flushRecoveryNow();
                  void flushSave();
                }}
                onChange={(event) => {
                  const title = event.target.value;
                  setActiveNote((current) => (current ? { ...current, title } : current));
                  updateDraft({ title });
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || (event.key === "Tab" && !event.shiftKey)) {
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
              <span>
                {activeNote.note_id === LOCAL_DRAFT_ID
                  ? "Created when you start writing"
                  : `Edited ${relativeTime(activeNote.content_updated_at ?? activeNote.updated_at)}`}
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
                  onBlur={() => {
                    flushRecoveryNow();
                    void flushSave();
                  }}
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
                onClick={() => void returnToList()}
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
                : "This permanently deletes this Note and cannot be undone."}
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
