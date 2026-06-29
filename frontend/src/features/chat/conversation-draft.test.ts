import { describe, expect, it } from "vitest";

import {
  findReusableBlankDraftConversation,
  isBlankDraftConversation,
  shouldExposeConversationInUrl,
  shouldShowConversationInHistory,
  shouldPersistConversationSnapshot,
} from "./conversation-draft";

const draft = (overrides: Record<string, unknown> = {}) => ({
  id: "draft-1",
  hydrated: true,
  messages: [],
  pendingFiles: [],
  uploadedFiles: [],
  stagedUploadFileIds: [],
  activeSelectionContext: null,
  selectionImportPending: false,
  sending: false,
  chatError: null,
  streamingMessageId: null,
  ...overrides,
});

describe("conversation draft persistence", () => {
  it("treats a hydrated, empty new chat as a reusable blank draft", () => {
    expect(isBlankDraftConversation(draft())).toBe(true);
    expect(shouldPersistConversationSnapshot(draft())).toBe(false);
  });

  it("does not treat conversations with user-visible work as blank drafts", () => {
    expect(
      isBlankDraftConversation(
        draft({
          messages: [{ id: "msg-user", role: "user", content: "hello" }],
        })
      )
    ).toBe(false);
    expect(
      isBlankDraftConversation(
        draft({
          pendingFiles: [{ id: "file-1", name: "image.png" }],
        })
      )
    ).toBe(false);
    expect(
      isBlankDraftConversation(
        draft({
          activeSelectionContext: { resource_uris: ["bisque://image/1"] },
        })
      )
    ).toBe(false);
  });

  it("reuses the active blank draft before looking for another blank draft", () => {
    const activeBlank = draft({ id: "draft-active" });
    const olderBlank = draft({ id: "draft-older" });
    const reusable = findReusableBlankDraftConversation(
      [olderBlank, activeBlank],
      "draft-active"
    );

    expect(reusable?.id).toBe("draft-active");
  });

  it("reuses an existing blank draft instead of creating another one", () => {
    const reusable = findReusableBlankDraftConversation(
      [
        draft({
          id: "conversation-with-work",
          messages: [{ id: "msg-user", role: "user", content: "hello" }],
        }),
        draft({ id: "blank-draft" }),
      ],
      "conversation-with-work"
    );

    expect(reusable?.id).toBe("blank-draft");
  });

  it("hides blank drafts from sidebar history while keeping real conversations", () => {
    expect(shouldShowConversationInHistory(draft())).toBe(false);
    expect(
      shouldShowConversationInHistory(
        draft({
          id: "conversation-with-work",
          messages: [{ id: "msg-user", role: "user", content: "hello" }],
        })
      )
    ).toBe(true);
    expect(shouldShowConversationInHistory(draft({ hydrated: false }))).toBe(true);
  });

  it("does not expose blank local draft ids in the browser URL", () => {
    expect(shouldExposeConversationInUrl(draft())).toBe(false);
    expect(
      shouldExposeConversationInUrl(
        draft({
          id: "conversation-with-work",
          messages: [{ id: "msg-user", role: "user", content: "hello" }],
        })
      )
    ).toBe(true);
    expect(shouldExposeConversationInUrl(draft({ hydrated: false }))).toBe(true);
  });
});
