import { useEffect, useRef, useState } from "react";
import { AlertCircle, Check, Loader2, NotebookPen, Plus } from "lucide-react";

import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  isDefinitiveNoteWriteReplayRejection,
  isDeterministicNoteWriteRejection,
  isNoteRevisionConflict,
  type ApiClient,
  type NoteDirectAppendReceipt,
  type NoteListItem,
} from "@/lib/api";
import { Button } from "@/components/ui/button";

/* eslint react-hooks/set-state-in-effect: "off" -- opening/query changes intentionally drive a remote Notes list. */

export type SelectedNoteChip = {
  note_id: string;
  title: string;
  revision: number;
};

const noteLabel = (note: Pick<NoteListItem, "title" | "snippet">): string => {
  const title = note.title.trim();
  if (title) return title;
  const firstLine = note.snippet
    .split("\n")
    .map((line) => line.replace(/^#{1,6}\s+/, "").replace(/[*_`~>]/g, "").trim())
    .find(Boolean);
  return firstLine?.slice(0, 72) || "Untitled";
};

const appendUniqueNotes = (
  current: readonly NoteListItem[],
  incoming: readonly NoteListItem[]
): NoteListItem[] => {
  const seen = new Set(current.map((note) => note.note_id));
  return [
    ...current,
    ...incoming.filter((note) => {
      if (seen.has(note.note_id)) return false;
      seen.add(note.note_id);
      return true;
    }),
  ];
};

export function NoteContextPicker({
  apiClient,
  open,
  selectedNoteIds,
  onOpenChange,
  onSelect,
}: {
  apiClient: ApiClient;
  open: boolean;
  selectedNoteIds: readonly string[];
  onOpenChange: (open: boolean) => void;
  onSelect: (note: SelectedNoteChip) => void;
}) {
  const [query, setQuery] = useState("");
  const [notes, setNotes] = useState<NoteListItem[]>([]);
  const [nextOffset, setNextOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const generationRef = useRef(0);
  const selected = new Set(selectedNoteIds);

  useEffect(() => {
    if (!open) return;
    const generation = ++generationRef.current;
    setLoading(true);
    setError(null);
    const timer = window.setTimeout(() => {
      void apiClient
        .listNotes({
          query: query.trim() || undefined,
          sort: query.trim() ? undefined : "recent",
          limit: 20,
          offset: 0,
        })
        .then((page) => {
          if (generation !== generationRef.current) return;
          const consumed = page.notes.length;
          setNotes(appendUniqueNotes([], page.notes));
          setNextOffset(consumed);
          setHasMore(consumed > 0 && consumed < page.total_count);
          setLoading(false);
        })
        .catch((requestError: unknown) => {
          if (generation !== generationRef.current) return;
          setError(requestError instanceof Error ? requestError.message : "Couldn’t load Notes.");
          setLoading(false);
        });
    }, query ? 180 : 0);
    return () => window.clearTimeout(timer);
  }, [apiClient, open, query, refreshKey]);

  const loadMore = async (): Promise<void> => {
    if (loadingMore || !hasMore) return;
    const generation = generationRef.current;
    setLoadingMore(true);
    try {
      const page = await apiClient.listNotes({
        query: query.trim() || undefined,
        sort: query.trim() ? undefined : "recent",
        limit: 20,
        offset: nextOffset,
      });
      if (generation === generationRef.current) {
        const consumed = nextOffset + page.notes.length;
        setNotes((current) => appendUniqueNotes(current, page.notes));
        setNextOffset(consumed);
        setHasMore(page.notes.length > 0 && consumed < page.total_count);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Couldn’t load more Notes.");
    } finally {
      setLoadingMore(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        onOpenChange(nextOpen);
        if (!nextOpen) setQuery("");
      }}
    >
      <DialogContent className="gap-3 overflow-hidden p-0 sm:max-w-md">
        <DialogHeader className="px-5 pt-5 text-left">
          <DialogTitle className="text-base">Use a note</DialogTitle>
          <DialogDescription>
            Ultra can read it for this message. Content used in chat becomes part of this conversation.
          </DialogDescription>
        </DialogHeader>
        <Command shouldFilter={false} className="rounded-none border-t">
          <CommandInput
            value={query}
            onValueChange={setQuery}
            placeholder="Search Notes"
            aria-label="Search Notes to use"
          />
          <CommandList className="max-h-[min(22rem,55vh)] p-2">
            {loading ? (
              <div className="text-muted-foreground flex items-center justify-center gap-2 py-8 text-sm" role="status">
                <Loader2 className="size-4 animate-spin" aria-hidden="true" /> Loading Notes…
              </div>
            ) : error ? (
              <div className="px-3 py-8 text-center text-sm" role="alert">
                <p>{error}</p>
                <button
                  type="button"
                  className="text-muted-foreground mt-2 underline underline-offset-4"
                  onClick={() => setRefreshKey((value) => value + 1)}
                >
                  Try again
                </button>
              </div>
            ) : (
              <>
                <CommandEmpty>No matching Notes.</CommandEmpty>
                <CommandGroup heading={query.trim() ? "Results" : "Recent"}>
                  {notes.map((note) => {
                    const isSelected = selected.has(note.note_id);
                    const selectionFull = selected.size >= 8 && !isSelected;
                    const title = noteLabel(note);
                    return (
                      <CommandItem
                        key={note.note_id}
                        value={note.note_id}
                        className="min-h-11"
                        disabled={isSelected || selectionFull}
                        onSelect={() => {
                          onSelect({ note_id: note.note_id, title, revision: note.revision });
                          onOpenChange(false);
                          setQuery("");
                        }}
                      >
                        <NotebookPen className="text-muted-foreground mt-0.5 size-4" aria-hidden="true" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-medium">{title}</span>
                          <span className="text-muted-foreground block truncate text-xs">
                            {note.snippet.trim() || "Empty note"}
                          </span>
                        </span>
                        {isSelected ? <Check className="size-4" aria-label="Already added" /> : null}
                      </CommandItem>
                    );
                  })}
                </CommandGroup>
                {hasMore ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="mx-auto mt-1 min-h-10 sm:min-h-8"
                    onClick={() => void loadMore()}
                    disabled={loadingMore}
                  >
                    {loadingMore ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
                    Load more
                  </Button>
                ) : null}
              </>
            )}
          </CommandList>
          {selected.size >= 8 ? (
            <p className="text-muted-foreground border-t px-4 py-2 text-xs" role="status">
              Eight Notes are attached. Remove one before adding another.
            </p>
          ) : null}
        </Command>
      </DialogContent>
    </Dialog>
  );
}

export type NoteSelectionCapture = {
  text: string;
  idempotencyKey: string;
  attempt?: {
    note_id: string;
    note_title: string;
    expected_revision: number;
    idempotency_key: string;
    /**
     * `pending` and `uncertain` both require an exact replay before this
     * attempt can be forgotten. `rejected` is a typed 4xx that proves no
     * append committed, so Discard can remain local.
     */
    status?: "pending" | "uncertain" | "rejected";
  };
};

const appendAttemptKey = (
  captureKey: string,
  noteId: string,
  revision: number
): string => {
  const value = `${captureKey}\u0000${noteId}\u0000${revision}`;
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  return `note-capture:${(first >>> 0).toString(36)}:${(second >>> 0).toString(36)}:${revision.toString(36)}`;
};

export function AddSelectionToNoteDialog({
  apiClient,
  open,
  capture,
  onOpenChange,
  onAttemptChange,
  onDiscardAttempt,
  onAdded,
  onNewNote,
}: {
  apiClient: ApiClient;
  open: boolean;
  capture: NoteSelectionCapture | null;
  onOpenChange: (open: boolean) => void;
  onAttemptChange: (
    attempt: NonNullable<NoteSelectionCapture["attempt"]>
  ) => boolean | void;
  onDiscardAttempt: () => void | Promise<void>;
  onAdded: (
    receipt: NoteDirectAppendReceipt,
    attempt: NonNullable<NoteSelectionCapture["attempt"]>
  ) => void;
  onNewNote: (text: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [notes, setNotes] = useState<NoteListItem[]>([]);
  const [nextOffset, setNextOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failedTarget, setFailedTarget] = useState<NoteListItem | null>(null);
  const [revisionConflict, setRevisionConflict] = useState(false);
  const [savingNoteId, setSavingNoteId] = useState<string | null>(null);
  const generationRef = useRef(0);
  const dialogInitializedRef = useRef(false);
  const writeInFlightRef = useRef(false);

  useEffect(() => {
    if (!open) {
      dialogInitializedRef.current = false;
      return;
    }
    if (!capture || dialogInitializedRef.current) return;
    dialogInitializedRef.current = true;
    setFailedTarget(null);
    setRevisionConflict(false);
    if (capture.attempt) {
      setFailedTarget({
        note_id: capture.attempt.note_id,
        title: capture.attempt.note_title,
        snippet: "",
        pinned: false,
        revision: capture.attempt.expected_revision,
        updated_at: "",
      });
      setError(
        capture.attempt.status === "rejected"
          ? "This add was rejected. Correct the selection or discard this attempt before choosing another Note."
          : "This add may already have completed. Retry safely to confirm it before doing anything else."
      );
    } else {
      setError(null);
    }
  }, [capture, open]);

  useEffect(() => {
    if (!open) return;
    const generation = ++generationRef.current;
    if (capture?.attempt) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    // The query shown in the input and the targets shown below it must always
    // describe the same completed request. Never leave prior-query rows
    // clickable while a new search is pending or after it fails.
    setNotes([]);
    setNextOffset(0);
    setHasMore(false);
    const handle = window.setTimeout(() => {
      void apiClient
        .listNotes({
          query: query.trim() || undefined,
          sort: query.trim() ? undefined : "recent",
          limit: 20,
          offset: 0,
        })
        .then((page) => {
          if (generation !== generationRef.current) return;
          const consumed = page.notes.length;
          setNotes(appendUniqueNotes([], page.notes));
          setNextOffset(consumed);
          setHasMore(consumed > 0 && consumed < page.total_count);
          setLoading(false);
        })
        .catch((requestError: unknown) => {
          if (generation !== generationRef.current) return;
          setNotes([]);
          setNextOffset(0);
          setHasMore(false);
          setError(requestError instanceof Error ? requestError.message : "Couldn’t load Notes.");
          setLoading(false);
        });
    }, query ? 180 : 0);
    return () => window.clearTimeout(handle);
  }, [apiClient, capture?.attempt, open, query]);

  const commit = async (target: NoteListItem, forceFreshAttempt = false): Promise<void> => {
    if (!capture || writeInFlightRef.current) return;
    const existingAttempt = forceFreshAttempt ? undefined : capture.attempt;
    const replayingUncertainAttempt = existingAttempt?.status === "uncertain";
    if (capture.attempt && capture.attempt.note_id !== target.note_id) {
      return;
    }
    const attempt =
      existingAttempt && existingAttempt.note_id === target.note_id
        ? existingAttempt
        : {
            note_id: target.note_id,
            note_title: noteLabel(target),
            expected_revision: target.revision,
            idempotency_key: appendAttemptKey(
              capture.idempotencyKey,
              target.note_id,
              target.revision
            ),
            status: "pending" as const,
          };
    const pendingAttempt = { ...attempt, status: "pending" as const };
    if (onAttemptChange(pendingAttempt) === false) {
      setFailedTarget(target);
      setError(
        "Ultra can’t protect this add across a reload in this browser. Keep the selection as a new Note instead."
      );
      return;
    }
    writeInFlightRef.current = true;
    setSavingNoteId(target.note_id);
    setFailedTarget(null);
    setRevisionConflict(false);
    setError(null);
    try {
      const receipt = await apiClient.appendToNote(
        attempt.note_id,
        { body_markdown: capture.text, expected_revision: attempt.expected_revision },
        attempt.idempotency_key
      );
      onAdded(receipt, pendingAttempt);
    } catch (requestError) {
      if (isNoteRevisionConflict(requestError)) {
        onAttemptChange({ ...attempt, status: "rejected" });
        try {
          const latest = await apiClient.getNote(target.note_id);
          setFailedTarget({
            ...target,
            title: latest.title,
            snippet: latest.body_markdown.slice(0, 300),
            pinned: latest.pinned,
            revision: latest.revision,
            content_updated_at: latest.content_updated_at,
            updated_at: latest.updated_at,
          });
          setRevisionConflict(true);
          setError("This note changed since the list loaded. Review the target, then retry against the latest version.");
        } catch {
          setFailedTarget(target);
          setRevisionConflict(true);
          setError("This note changed, and Ultra couldn’t refresh it. Your selection is still here.");
        }
      } else {
        const definitelyRejected = replayingUncertainAttempt
          ? isDefinitiveNoteWriteReplayRejection(requestError, "append")
          : isDeterministicNoteWriteRejection(requestError);
        onAttemptChange({
          ...attempt,
          status: definitelyRejected ? "rejected" : "uncertain",
        });
        setFailedTarget(target);
        setError(requestError instanceof Error ? requestError.message : "Couldn’t add this selection.");
      }
    } finally {
      writeInFlightRef.current = false;
      setSavingNoteId(null);
    }
  };

  const discardAttempt = async (): Promise<void> => {
    const attempt = capture?.attempt;
    if (!capture || !attempt || writeInFlightRef.current) return;
    writeInFlightRef.current = true;
    setSavingNoteId(attempt.note_id);
    setError(null);
    try {
      await onDiscardAttempt();
      setFailedTarget(null);
      setRevisionConflict(false);
    } catch (requestError) {
      setFailedTarget({
        note_id: attempt.note_id,
        title: attempt.note_title,
        snippet: "",
        pinned: false,
        revision: attempt.expected_revision,
        updated_at: "",
      });
      setError(
        requestError instanceof Error
          ? `Couldn’t safely discard this add — ${requestError.message}`
          : "Couldn’t safely discard this add. The exact attempt is still here."
      );
    } finally {
      writeInFlightRef.current = false;
      setSavingNoteId(null);
    }
  };

  const loadMore = async (): Promise<void> => {
    if (loadingMore || !hasMore) return;
    const generation = generationRef.current;
    setLoadingMore(true);
    try {
      const page = await apiClient.listNotes({
        query: query.trim() || undefined,
        sort: query.trim() ? undefined : "recent",
        limit: 20,
        offset: nextOffset,
      });
      if (generation === generationRef.current) {
        const consumed = nextOffset + page.notes.length;
        setNotes((current) => appendUniqueNotes(current, page.notes));
        setNextOffset(consumed);
        setHasMore(page.notes.length > 0 && consumed < page.total_count);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Couldn’t load more Notes.");
    } finally {
      setLoadingMore(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        onOpenChange(nextOpen);
        if (!nextOpen) setQuery("");
      }}
    >
      <DialogContent className="gap-3 overflow-hidden p-0 sm:max-w-md">
        <DialogHeader className="px-5 pt-5 text-left">
          <DialogTitle className="text-base">Add to note</DialogTitle>
          <DialogDescription>Choose where to keep this exact selection.</DialogDescription>
        </DialogHeader>
        {capture ? (
          <blockquote className="border-border bg-muted/45 mx-5 line-clamp-3 whitespace-pre-wrap rounded-lg border px-3 py-2 text-sm">
            {capture.text}
          </blockquote>
        ) : null}
        {error ? (
          <div className="border-border bg-muted/35 mx-5 rounded-lg border px-3 py-2 text-sm" role="alert">
            <div className="flex items-start gap-2">
              <AlertCircle className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <span>{error}</span>
            </div>
            {failedTarget ? (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span className="text-muted-foreground min-w-0 flex-1 truncate text-xs">
                  {noteLabel(failedTarget)}
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="min-h-10 sm:min-h-8"
                  onClick={() => {
                    void commit(failedTarget, revisionConflict);
                  }}
                >
                  {revisionConflict ? "Retry with latest" : "Try again"}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="min-h-10 sm:min-h-8"
                  disabled={savingNoteId !== null}
                  onClick={() => void discardAttempt()}
                >
                  {savingNoteId !== null ? "Discarding…" : "Discard"}
                </Button>
              </div>
            ) : null}
          </div>
        ) : null}
        {savingNoteId && !error ? (
          <div
            className="border-border bg-muted/35 mx-5 flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
            role="status"
          >
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            <span>Adding to {capture?.attempt?.note_title || "Note"}…</span>
          </div>
        ) : null}
        {!capture?.attempt ? <Command shouldFilter={false} className="rounded-none border-t">
          <CommandInput
            value={query}
            onValueChange={setQuery}
            placeholder="Search Notes"
            aria-label="Search Notes to add to"
          />
          <CommandList className="max-h-[min(22rem,48vh)] p-2">
            <CommandGroup>
              <CommandItem
                value="__new_note__"
                className="min-h-11"
                disabled={Boolean(capture?.attempt)}
                onSelect={() => {
                  if (!capture) return;
                  onNewNote(capture.text);
                  onOpenChange(false);
                }}
              >
                <Plus className="text-muted-foreground size-4" aria-hidden="true" />
                <span className="font-medium">New note</span>
              </CommandItem>
            </CommandGroup>
            {loading ? (
              <div className="text-muted-foreground flex items-center justify-center gap-2 py-8 text-sm" role="status">
                <Loader2 className="size-4 animate-spin" aria-hidden="true" /> Loading Notes…
              </div>
            ) : (
              <>
                <CommandEmpty>No matching Notes.</CommandEmpty>
                <CommandGroup heading={query.trim() ? "Results" : "Recent"}>
                  {notes.map((note) => (
                    <CommandItem
                      key={note.note_id}
                      value={note.note_id}
                      className="min-h-11"
                      disabled={Boolean(savingNoteId || capture?.attempt)}
                      onSelect={() => {
                        void commit(note);
                      }}
                    >
                      {savingNoteId === note.note_id ? (
                        <Loader2 className="text-muted-foreground size-4 animate-spin" aria-hidden="true" />
                      ) : (
                        <NotebookPen className="text-muted-foreground size-4" aria-hidden="true" />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-medium">{noteLabel(note)}</span>
                        <span className="text-muted-foreground block truncate text-xs">
                          {note.snippet.trim() || "Empty note"}
                        </span>
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
                {hasMore ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="mx-auto mt-1 min-h-10 sm:min-h-8"
                    onClick={() => void loadMore()}
                    disabled={loadingMore}
                  >
                    {loadingMore ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
                    Load more
                  </Button>
                ) : null}
              </>
            )}
          </CommandList>
        </Command> : null}
      </DialogContent>
    </Dialog>
  );
}
