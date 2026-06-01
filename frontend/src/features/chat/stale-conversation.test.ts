import { describe, expect, it } from "vitest";

import { shouldKeepOptimisticConversationAfterHydration } from "./stale-conversation";

describe("shouldKeepOptimisticConversationAfterHydration", () => {
  it("drops the URL-requested conversation when the backend reports it missing", () => {
    expect(
      shouldKeepOptimisticConversationAfterHydration({
        conversationId: "thread_stale",
        incomingConversationIds: new Set(["thread_latest"]),
        missingRequestedConversationId: "thread_stale",
      })
    ).toBe(false);
  });

  it("keeps unrelated optimistic local conversations", () => {
    expect(
      shouldKeepOptimisticConversationAfterHydration({
        conversationId: "local_unsent",
        incomingConversationIds: new Set(["thread_latest"]),
        missingRequestedConversationId: "thread_stale",
      })
    ).toBe(true);
  });

  it("does not duplicate conversations already returned by the backend", () => {
    expect(
      shouldKeepOptimisticConversationAfterHydration({
        conversationId: "thread_latest",
        incomingConversationIds: new Set(["thread_latest"]),
        missingRequestedConversationId: null,
      })
    ).toBe(false);
  });
});
