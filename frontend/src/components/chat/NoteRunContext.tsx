import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, ChevronDown, ChevronRight, Loader2, NotebookPen, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  ApiError,
  type ApiClient,
  type NoteAppendOperationReceipt,
  type NoteAppendProposal,
} from "@/lib/api";
import type { RunEvent } from "@/types";

/* eslint react-hooks/set-state-in-effect: "off" -- persisted proposal IDs intentionally hydrate browser-authorized state on mount. */

export type ReadNoteEvent = {
  note_id: string;
  revision: number;
};

export type NoteRunMetadata = {
  readNotes: ReadNoteEvent[];
  proposalIds: string[];
};

export const noteRunMetadataFromEvents = (events: readonly RunEvent[]): NoteRunMetadata => {
  const reads = new Map<string, ReadNoteEvent>();
  const proposals = new Set<string>();
  for (const event of events) {
    if (String(event.event_type || "").trim() !== "tool_call.completed") continue;
    const payload = event.payload ?? {};
    const toolName = String(payload.tool_name ?? "").trim();
    if (toolName === "read_note") {
      const noteId = String(payload.note_id ?? "").trim();
      const revision = Number(payload.revision);
      if (noteId && Number.isSafeInteger(revision) && revision > 0) {
        reads.set(`${noteId}:${revision}`, {
          note_id: noteId,
          revision,
        });
      }
    } else if (toolName === "propose_note_append") {
      const proposalId = String(payload.proposal_id ?? "").trim();
      if (proposalId) proposals.add(proposalId);
    }
  }
  return { readNotes: Array.from(reads.values()), proposalIds: Array.from(proposals) };
};

const errorCopy = (error: unknown, operation: "load" | "commit" | "undo"): string => {
  if (error instanceof ApiError) {
    const detail =
      error.detail && typeof error.detail === "object"
        ? (error.detail as Record<string, unknown>)
        : null;
    const code = String(detail?.code ?? "");
    if (error.status === 404 || error.status === 410) {
      return operation === "load"
        ? "This note update is no longer available."
        : "This note update expired before it could be applied.";
    }
    if (error.status === 409) {
      if (code === "note_proposal_expired") {
        return "This note update expired before it could be applied. Your proposed text remains available to copy.";
      }
      return operation === "undo"
        ? "The note changed again, so Ultra won’t undo over newer writing."
        : "The note changed since this update was prepared. Open the note to review it; your proposed text remains available to copy.";
    }
  }
  return error instanceof Error ? error.message : "Ultra couldn’t update this note.";
};

function NoteAppendProposalCard({
  proposalId,
  apiClient,
  onOpenNote,
  onNoteChanged,
}: {
  proposalId: string;
  apiClient: ApiClient;
  onOpenNote: (noteId: string) => void;
  onNoteChanged: (noteId: string) => void;
}) {
  const [proposal, setProposal] = useState<NoteAppendProposal | null>(null);
  const [receipt, setReceipt] = useState<NoteAppendOperationReceipt | null>(null);
  const [reviewedBody, setReviewedBody] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"commit" | "undo" | null>(null);
  const [commitBlocked, setCommitBlocked] = useState(false);
  const [undoBlocked, setUndoBlocked] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    void apiClient
      .getNoteAppendProposal(proposalId)
      .then((record) => {
        setProposal(record);
        setReceipt(record.operation ?? null);
        setReviewedBody(record.body_markdown ?? "");
        setCommitBlocked(false);
        setUndoBlocked(false);
      })
      .catch((requestError: unknown) => setError(errorCopy(requestError, "load")))
      .finally(() => setLoading(false));
  }, [apiClient, proposalId]);

  useEffect(load, [load]);

  const commit = async () => {
    if (!proposal || !reviewedBody.trim()) return;
    setBusy("commit");
    setError(null);
    try {
      const operation = await apiClient.commitNoteAppendProposal(proposal.proposal_id, {
        body_markdown: reviewedBody,
      });
      setReceipt(operation);
      onNoteChanged(operation.note_id);
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 409) {
        setCommitBlocked(true);
      }
      setError(errorCopy(requestError, "commit"));
    } finally {
      setBusy(null);
    }
  };

  const undo = async () => {
    const operationId = receipt?.operation_id ?? proposal?.operation_id;
    if (!operationId || receipt?.undone_at || undoBlocked) return;
    setBusy("undo");
    setError(null);
    try {
      const operation = await apiClient.undoNoteAppendOperation(operationId);
      setReceipt(operation);
      onNoteChanged(operation.note_id);
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 409) {
        // A revision-fenced Undo cannot become valid on a later retry: once
        // newer writing exists, repeating the same operation would only create
        // a frustrating loop. Keep Open note as the recovery path.
        setUndoBlocked(true);
      }
      setError(errorCopy(requestError, "undo"));
    } finally {
      setBusy(null);
    }
  };

  if (loading) {
    return (
      <div className="border-border bg-muted/25 flex items-center gap-2 rounded-xl border px-4 py-3 text-sm" role="status">
        <Loader2 className="text-muted-foreground size-4 animate-spin" aria-hidden="true" />
        Loading proposed note update…
      </div>
    );
  }

  if (!proposal) {
    return (
      <div className="border-border bg-muted/25 rounded-xl border px-4 py-3 text-sm" role="status">
        <div className="flex items-start gap-2">
          <AlertCircle className="text-muted-foreground mt-0.5 size-4" aria-hidden="true" />
          <span className="flex-1">{error ?? "This note update is unavailable."}</span>
          <Button type="button" variant="ghost" size="xs" onClick={load}>Retry</Button>
        </div>
      </div>
    );
  }

  if (receipt || proposal.status === "committed") {
    const undone = Boolean(receipt?.undone_at);
    const operationId = receipt?.operation_id ?? proposal.operation_id;
    return (
      <section className="border-border bg-muted/25 rounded-xl border px-4 py-3" aria-label="Note update">
        <div className="flex flex-wrap items-center gap-2">
          <span className="bg-background flex size-7 items-center justify-center rounded-full border" aria-hidden="true">
            {undone ? <RotateCcw className="size-3.5" /> : <Check className="size-3.5" />}
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">
              {undone ? "Update undone" : `Added to ${receipt?.note_title || proposal.note_title || "note"}`}
            </p>
            <p className="text-muted-foreground text-xs">
              {undone ? "The exact addition was removed." : "Saved to Notes."}
            </p>
          </div>
          <Button type="button" variant="outline" size="xs" onClick={() => onOpenNote(receipt?.note_id ?? proposal.note_id)}>
            Open note
          </Button>
          {!undone && operationId ? (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              disabled={busy !== null || undoBlocked}
              onClick={() => void undo()}
            >
              {busy === "undo" ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
              Undo
            </Button>
          ) : null}
        </div>
        {error ? <p className="text-destructive mt-2 text-xs" role="alert">{error}</p> : null}
      </section>
    );
  }

  // Expired proposals intentionally no longer carry their private body. Do
  // not render an empty editor that implies recoverable text; keep the exact
  // target and a direct path to the Note instead.
  if (proposal.status === "expired") {
    return (
      <section className="border-border bg-muted/25 rounded-xl border px-4 py-3" aria-label="Expired note update">
        <div className="flex flex-wrap items-center gap-2">
          <span className="bg-background flex size-7 items-center justify-center rounded-full border" aria-hidden="true">
            <AlertCircle className="text-muted-foreground size-3.5" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">Update to {proposal.note_title || "note"} expired</p>
            <p className="text-muted-foreground text-xs">This proposal is no longer available.</p>
          </div>
          <Button type="button" variant="outline" size="xs" onClick={() => onOpenNote(proposal.note_id)}>
            Open note
          </Button>
        </div>
      </section>
    );
  }

  return (
    <section className="border-border bg-muted/25 rounded-xl border p-4" aria-label="Proposed note update">
      <div className="flex items-start gap-3">
        <span className="bg-background flex size-8 shrink-0 items-center justify-center rounded-full border" aria-hidden="true">
          <NotebookPen className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">Add to {proposal.note_title || "note"}</p>
          <p className="text-muted-foreground text-xs">Review the exact text before it is saved.</p>
        </div>
        <Button type="button" variant="ghost" size="xs" onClick={() => onOpenNote(proposal.note_id)}>
          Open
        </Button>
      </div>
      <label className="mt-3 block">
        <span className="sr-only">Text to add</span>
        <Textarea
          value={reviewedBody}
          onChange={(event) => setReviewedBody(event.target.value)}
          disabled={busy !== null}
          className="bg-background min-h-24 max-h-64 resize-y font-mono text-xs leading-relaxed"
          aria-label="Text to add to note"
        />
      </label>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-muted-foreground min-w-0 flex-1 text-xs" role={error ? "alert" : undefined}>
          {error ??
            (commitBlocked
                ? "Open the note to review newer writing. You can still copy this text."
                : "You can edit this text before adding it.")}
        </p>
        <Button
          type="button"
          size="sm"
          disabled={commitBlocked || busy !== null || !reviewedBody.trim()}
          onClick={() => void commit()}
        >
          {busy === "commit" ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
          Add to note
        </Button>
      </div>
    </section>
  );
}

function NotesUsedFooter({
  reads,
  apiClient,
  onOpenNote,
}: {
  reads: ReadNoteEvent[];
  apiClient: ApiClient;
  onOpenNote: (noteId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [titles, setTitles] = useState<Record<string, string>>({});
  const noteIds = useMemo(() => Array.from(new Set(reads.map((read) => read.note_id))), [reads]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void Promise.all(
      noteIds.map(async (noteId) => {
        try {
          const note = await apiClient.getNote(noteId);
          return [noteId, note.title.trim() || "Untitled"] as const;
        } catch {
          return [noteId, "Note unavailable"] as const;
        }
      })
    ).then((entries) => {
      if (!cancelled) setTitles(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, [apiClient, noteIds, open]);

  if (noteIds.length === 0) return null;
  return (
    <div className="pt-1 text-xs">
      <button
        type="button"
        className="text-muted-foreground hover:text-foreground focus-visible:ring-ring inline-flex items-center gap-1 rounded-md py-1 transition-colors focus-visible:ring-2 focus-visible:outline-none"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? <ChevronDown className="size-3.5" aria-hidden="true" /> : <ChevronRight className="size-3.5" aria-hidden="true" />}
        Notes used · {noteIds.length}
      </button>
      {open ? (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {noteIds.map((noteId) => (
            <button
              key={noteId}
              type="button"
              className="border-border bg-background hover:bg-accent max-w-full truncate rounded-full border px-2.5 py-1"
              onClick={() => onOpenNote(noteId)}
            >
              {titles[noteId] ?? "Loading note…"}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function NoteRunContext({
  runEvents,
  apiClient,
  onOpenNote,
  onNoteChanged,
}: {
  runEvents: readonly RunEvent[];
  apiClient: ApiClient;
  onOpenNote: (noteId: string) => void;
  onNoteChanged: (noteId: string) => void;
}) {
  const metadata = useMemo(() => noteRunMetadataFromEvents(runEvents), [runEvents]);
  if (metadata.readNotes.length === 0 && metadata.proposalIds.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      {metadata.proposalIds.map((proposalId) => (
        <NoteAppendProposalCard
          key={proposalId}
          proposalId={proposalId}
          apiClient={apiClient}
          onOpenNote={onOpenNote}
          onNoteChanged={onNoteChanged}
        />
      ))}
      <NotesUsedFooter reads={metadata.readNotes} apiClient={apiClient} onOpenNote={onOpenNote} />
    </div>
  );
}
