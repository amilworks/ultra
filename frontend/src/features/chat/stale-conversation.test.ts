import { describe, expect, it } from "vitest";

import {
  prependResolvedConversation,
  shouldKeepOptimisticConversationAfterHydration,
} from "./stale-conversation";

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

describe("prependResolvedConversation", () => {
  it("replaces an existing entry that shares the resolved id instead of duplicating it", () => {
    const existing = [
      { id: "local_a", updatedAt: 30 },
      { id: "local_b", updatedAt: 20 },
    ];
    const result = prependResolvedConversation({ id: "local_b", updatedAt: 50 }, existing);

    expect(result.map((item) => item.id)).toEqual(["local_b", "local_a"]);
    expect(result.filter((item) => item.id === "local_b")).toHaveLength(1);
    expect(result[0].updatedAt).toBe(50);
  });

  it("prepends a genuinely new conversation and keeps the list sorted by recency", () => {
    const existing = [
      { id: "local_a", updatedAt: 30 },
      { id: "local_b", updatedAt: 40 },
    ];
    const result = prependResolvedConversation({ id: "local_c", updatedAt: 35 }, existing);

    expect(result.map((item) => item.id)).toEqual(["local_b", "local_c", "local_a"]);
  });
});
