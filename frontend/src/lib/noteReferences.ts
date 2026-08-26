import { ApiError, type ApiClient } from "@/lib/api";

export type NoteReferenceChip = {
  note_id: string;
  title: string;
  revision: number;
};

export class NoteReferenceResealError extends Error {
  readonly noteId: string;
  readonly unavailable: boolean;

  constructor(note: NoteReferenceChip, cause: unknown) {
    const unavailable = cause instanceof ApiError && cause.status === 404;
    const title = note.title.trim() || "Untitled";
    super(
      unavailable
        ? `The Note “${title}” is no longer available. Nothing was sent or changed.`
        : `Ultra couldn’t refresh the Note “${title}”. Check your connection and try again; nothing was sent or changed.`
    );
    this.name = "NoteReferenceResealError";
    this.noteId = note.note_id;
    this.unavailable = unavailable;
  }
}

/**
 * Re-authorize the same exact Note IDs through the browser session immediately
 * before a new run. Only title/revision survive this boundary; historical run
 * provenance remains untouched, while a stale composer chip cannot turn an
 * ordinary retry into a create-run 409.
 */
export const resealNoteReferences = async (
  apiClient: Pick<ApiClient, "getNote">,
  references: readonly NoteReferenceChip[]
): Promise<NoteReferenceChip[]> => {
  const seen = new Set<string>();
  const unique = references.filter((reference) => {
    const noteId = reference.note_id.trim();
    if (!noteId || seen.has(noteId)) return false;
    seen.add(noteId);
    return true;
  });
  return await Promise.all(
    unique.map(async (reference) => {
      try {
        const note = await apiClient.getNote(reference.note_id);
        return {
          note_id: note.note_id,
          title: note.title.trim() || "Untitled",
          revision: note.revision,
        };
      } catch (error) {
        throw new NoteReferenceResealError(reference, error);
      }
    })
  );
};
