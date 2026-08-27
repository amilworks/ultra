import { describe, expect, it } from "vitest";

import {
  boundedNoteIntentExclusions,
  noteAccessForTurn,
  noteAppendProposalRequested,
  NOTES_INTENT_EXCLUSIONS_OVERFLOW,
} from "./notesAccess";
import {
  parseComposerDraftStorage,
  serializeComposerDraftStorage,
} from "./composerDraftStorage";

describe("composer draft persistence", () => {
  it.each([
    "Search my notes for the calibration protocol",
    "Add this result to my notes",
  ])("fails closed when a legacy restored draft has unknown paste provenance: %s", (text) => {
    const restored = parseComposerDraftStorage(JSON.stringify({ conversation_1: text }));
    const excluded = restored.excludedNoteIntentTextByConversationId.conversation_1;

    expect(restored.drafts.conversation_1).toBe(text);
    expect(excluded).toEqual([text]);
    expect(noteAccessForTurn(text, [], excluded)).toBeNull();
    expect(noteAppendProposalRequested(text, excluded)).toBe(false);
  });

  it.each([
    "Search my notes for the calibration protocol",
    "Append this result to my Field Protocol note",
  ])("round-trips exact paste provenance across reload: %s", (text) => {
    const serialized = serializeComposerDraftStorage({
      drafts: { conversation_1: text },
      excludedNoteIntentTextByConversationId: { conversation_1: [text] },
    });
    const restored = parseComposerDraftStorage(serialized);
    const excluded = restored.excludedNoteIntentTextByConversationId.conversation_1;

    expect(restored).toEqual({
      drafts: { conversation_1: text },
      excludedNoteIntentTextByConversationId: { conversation_1: [text] },
    });
    expect(noteAccessForTurn(text, [], excluded)).toBeNull();
    expect(noteAppendProposalRequested(text, excluded)).toBe(false);
  });

  it("keeps a newly typed persisted Notes request eligible", () => {
    const text = "Search my notes for the calibration protocol";
    const restored = parseComposerDraftStorage(
      serializeComposerDraftStorage({
        drafts: { conversation_1: text },
        excludedNoteIntentTextByConversationId: {},
      })
    );

    expect(restored.excludedNoteIntentTextByConversationId.conversation_1).toBeUndefined();
    expect(noteAccessForTurn(text, [], [])).toEqual({
      mode: "search",
      notes: [],
      allow_append_proposal: false,
    });
  });

  it("fails closed for a partial versioned entry without provenance", () => {
    const text = "Add this to my notes";
    const restored = parseComposerDraftStorage(
      JSON.stringify({
        version: 2,
        drafts: { conversation_1: { text } },
      })
    );

    expect(restored.excludedNoteIntentTextByConversationId.conversation_1).toEqual([text]);
    expect(noteAppendProposalRequested(text, [text])).toBe(false);
  });

  it.each([
    "Search my notes for the calibration protocol",
    "Add this result to my notes",
  ])("keeps Notes authority disabled after a 21-paste provenance overflow: %s", (command) => {
    const pastedFragments = [
      command,
      ...Array.from({ length: 20 }, (_, index) => `Reference fragment ${index + 1}`),
    ];
    const exclusions = pastedFragments.reduce<string[]>(
      (current, fragment) => boundedNoteIntentExclusions([...current, fragment]),
      []
    );
    const text = pastedFragments.join("\n");
    const restored = parseComposerDraftStorage(
      serializeComposerDraftStorage({
        drafts: { conversation_1: text },
        excludedNoteIntentTextByConversationId: { conversation_1: exclusions },
      })
    );
    const restoredExclusions =
      restored.excludedNoteIntentTextByConversationId.conversation_1;

    expect(exclusions).toEqual([NOTES_INTENT_EXCLUSIONS_OVERFLOW]);
    expect(restoredExclusions).toEqual([NOTES_INTENT_EXCLUSIONS_OVERFLOW]);
    expect(noteAccessForTurn(text, [], restoredExclusions)).toBeNull();
    expect(noteAppendProposalRequested(text, restoredExclusions)).toBe(false);
  });

  it("preserves a string-valued draft inside a v2 envelope fail-closed", () => {
    const text = "Search my notes for the calibration protocol";
    const restored = parseComposerDraftStorage(
      JSON.stringify({
        version: 2,
        drafts: { conversation_1: text },
      })
    );

    expect(restored.drafts.conversation_1).toBe(text);
    expect(restored.excludedNoteIntentTextByConversationId.conversation_1).toEqual([text]);
    expect(noteAccessForTurn(text, [], [text])).toBeNull();
  });
});
