type CollectionLike = {
  length: number;
};

export type DraftConversationLike = {
  id: string;
  hydrated?: boolean;
  messages?: CollectionLike | null;
  pendingFiles?: CollectionLike | null;
  uploadedFiles?: CollectionLike | null;
  stagedUploadFileIds?: CollectionLike | null;
  activeSelectionContext?: unknown;
  selectionImportPending?: boolean | null;
  sending?: boolean | null;
  chatError?: unknown;
  streamingMessageId?: string | null;
};

const hasItems = (value: CollectionLike | null | undefined): boolean =>
  Boolean(value && value.length > 0);

export const isBlankDraftConversation = (
  conversation: DraftConversationLike | null | undefined
): boolean => {
  if (!conversation?.hydrated) {
    return false;
  }
  return (
    !hasItems(conversation.messages) &&
    !hasItems(conversation.pendingFiles) &&
    !hasItems(conversation.uploadedFiles) &&
    !hasItems(conversation.stagedUploadFileIds) &&
    !conversation.activeSelectionContext &&
    !conversation.selectionImportPending &&
    !conversation.sending &&
    !conversation.chatError &&
    !conversation.streamingMessageId
  );
};

export const shouldPersistConversationSnapshot = (
  conversation: DraftConversationLike | null | undefined
): boolean => Boolean(conversation?.hydrated && !isBlankDraftConversation(conversation));

export const shouldShowConversationInHistory = (
  conversation: DraftConversationLike | null | undefined
): boolean => Boolean(conversation && !isBlankDraftConversation(conversation));

export const shouldExposeConversationInUrl = (
  conversation: DraftConversationLike | null | undefined
): boolean => shouldShowConversationInHistory(conversation);

export const findReusableBlankDraftConversation = <
  Conversation extends DraftConversationLike,
>(
  conversations: readonly Conversation[],
  activeConversationId: string | null | undefined
): Conversation | null => {
  const activeId = String(activeConversationId ?? "").trim();
  if (activeId) {
    const active = conversations.find(
      (conversation) => conversation.id === activeId && isBlankDraftConversation(conversation)
    );
    if (active) {
      return active;
    }
  }
  return conversations.find(isBlankDraftConversation) ?? null;
};
