import { useEffect, useRef, useState } from "react";
import { Check, Loader2, NotebookPen } from "lucide-react";

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
import type { ApiClient, NoteListItem } from "@/lib/api";

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
  const [loading, setLoading] = useState(false);
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
        .listNotes({ query: query.trim() || undefined, limit: 50 })
        .then((page) => {
          if (generation !== generationRef.current) return;
          setNotes(page.notes);
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
          <DialogTitle className="text-base">Add a Note</DialogTitle>
          <DialogDescription>
            Ultra can read it for this message. Content used in chat becomes part of this conversation.
          </DialogDescription>
        </DialogHeader>
        <Command shouldFilter={false} className="rounded-none border-t">
          <CommandInput
            value={query}
            onValueChange={setQuery}
            placeholder="Search Notes"
            aria-label="Search Notes to add"
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
                <CommandGroup>
                  {notes.map((note) => {
                    const isSelected = selected.has(note.note_id);
                    const selectionFull = selected.size >= 8 && !isSelected;
                    const title = noteLabel(note);
                    return (
                      <CommandItem
                        key={note.note_id}
                        value={note.note_id}
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
