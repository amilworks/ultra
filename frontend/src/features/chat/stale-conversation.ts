/**
 * Prepend a conversation resolved from a deep link, replacing any existing
 * entry that shares its id. A conversation can already be present under its
 * resolved id even when the URL referenced it by a different identifier (e.g.
 * a backend thread id vs the local conversation id it maps to), so deduping by
 * the resolved id prevents duplicate React keys in the history list.
 */
export const prependResolvedConversation = <T extends { id: string; updatedAt: number }>(
  conversation: T,
  existing: readonly T[]
): T[] => {
  const resolvedId = conversation.id.trim();
  const deduped = existing.filter((item) => item.id.trim() !== resolvedId);
  return [conversation, ...deduped].sort((a, b) => b.updatedAt - a.updatedAt);
};

export const shouldKeepOptimisticConversationAfterHydration = ({
  conversationId,
  incomingConversationIds,
  missingRequestedConversationId,
}: {
  conversationId: string;
  incomingConversationIds: ReadonlySet<string>;
  missingRequestedConversationId?: string | null;
}): boolean => {
  const normalizedConversationId = conversationId.trim();
  if (!normalizedConversationId || incomingConversationIds.has(normalizedConversationId)) {
    return false;
  }

  const missingId = String(missingRequestedConversationId ?? "").trim();
  if (missingId && normalizedConversationId === missingId) {
    return false;
  }

  return true;
};
