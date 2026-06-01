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
