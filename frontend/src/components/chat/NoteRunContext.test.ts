import { describe, expect, it } from "vitest";

import { noteRunMetadataFromEvents } from "./NoteRunContext";

describe("Note run metadata", () => {
  it("extracts only redacted completed Note tool metadata", () => {
    expect(
      noteRunMetadataFromEvents([
        {
          event_type: "tool_call.completed",
          payload: {
            tool_name: "read_note",
            note_id: "note_1",
            revision: 4,
            returned_bytes: 500,
          },
        },
        {
          event_type: "tool_call.completed",
          payload: { tool_name: "propose_note_append", proposal_id: "proposal_1", note_id: "note_1" },
        },
        { event_type: "tool_call.started", payload: { tool_name: "read_note", note_id: "ignored" } },
      ])
    ).toEqual({
      readNotes: [{ note_id: "note_1", revision: 4 }],
      proposalIds: ["proposal_1"],
    });
  });

  it("deduplicates retries without consuming private output previews", () => {
    const event = {
      event_type: "tool_call.completed",
      payload: {
        tool_name: "read_note",
        note_id: "note_1",
        revision: 4,
        output_preview: "must not be used",
      },
    };
    expect(noteRunMetadataFromEvents([event, event]).readNotes).toHaveLength(1);
  });
});
