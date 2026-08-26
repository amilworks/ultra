import { describe, expect, it, vi } from "vitest";

import { ApiError, type ApiClient, type NoteRecord } from "@/lib/api";
import { resealNoteReferences } from "./noteReferences";
import type { NoteReferenceResealError } from "./noteReferences";

const note = (noteId: string, revision: number, title: string): NoteRecord => ({
  note_id: noteId,
  title,
  body_markdown: "private body is not retained by the helper",
  pinned: false,
  editor_mode: "markdown",
  revision,
  content_digest: "digest",
  created_at: "2026-08-25T12:00:00Z",
  updated_at: "2026-08-25T12:00:00Z",
});

describe("resealNoteReferences", () => {
  it("keeps exact IDs while refreshing title and revision for a new run", async () => {
    const apiClient = {
      getNote: vi.fn().mockResolvedValue(note("note_1", 7, "Renamed protocol")),
    } as unknown as ApiClient;

    await expect(
      resealNoteReferences(apiClient, [
        { note_id: "note_1", title: "Old protocol", revision: 2 },
      ])
    ).resolves.toEqual([{ note_id: "note_1", title: "Renamed protocol", revision: 7 }]);
    expect(apiClient.getNote).toHaveBeenCalledWith("note_1");
  });

  it("fails clearly and leaves the caller in control when a Note was deleted", async () => {
    const apiClient = {
      getNote: vi.fn().mockRejectedValue(new ApiError("not found", 404, null)),
    } as unknown as ApiClient;

    await expect(
      resealNoteReferences(apiClient, [
        { note_id: "note_1", title: "Field protocol", revision: 2 },
      ])
    ).rejects.toMatchObject({
      unavailable: true,
      message: "The Note “Field protocol” is no longer available. Nothing was sent or changed.",
    } satisfies Partial<NoteReferenceResealError>);
  });

  it("deduplicates IDs before browser authorization", async () => {
    const apiClient = {
      getNote: vi.fn().mockResolvedValue(note("note_1", 4, "Protocol")),
    } as unknown as ApiClient;

    await resealNoteReferences(apiClient, [
      { note_id: "note_1", title: "Protocol", revision: 2 },
      { note_id: "note_1", title: "Protocol", revision: 3 },
    ]);
    expect(apiClient.getNote).toHaveBeenCalledTimes(1);
  });
});
