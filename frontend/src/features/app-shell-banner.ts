type ActivePanel = "chat" | "resources" | "notes" | "admin" | "training" | "scientific-viewer";

export const MISSING_REQUESTED_CONVERSATION_MESSAGE =
  "That chat is no longer available. We opened your latest chat instead.";

export const MISSING_REQUESTED_CONVERSATION_NEW_MESSAGE =
  "That chat is no longer available. We started a new chat instead.";

export const isStaleChatRecoveryMessage = (message: string | null): boolean =>
  message === MISSING_REQUESTED_CONVERSATION_MESSAGE ||
  message === MISSING_REQUESTED_CONVERSATION_NEW_MESSAGE;

export const appShellBannerVariant = (message: string | null): "action" | "error" =>
  isStaleChatRecoveryMessage(message) ? "action" : "error";

export const selectLatestAvailableConversation = <
  Conversation extends { id: string; hydrated: boolean; updatedAt: number },
>(
  conversations: Conversation[],
  missingConversationId: string,
  isAvailable: (conversation: Conversation) => boolean
): Conversation | undefined =>
  conversations.reduce<Conversation | undefined>((latest, conversation) => {
    if (
      conversation.id === missingConversationId ||
      !conversation.hydrated ||
      !isAvailable(conversation)
    ) {
      return latest;
    }
    return !latest || conversation.updatedAt > latest.updatedAt ? conversation : latest;
  }, undefined);

export const shouldShowAppShellBanner = (
  activePanel: ActivePanel,
  message: string | null
): boolean => {
  if (!message) {
    return false;
  }
  return activePanel === "chat" || !isStaleChatRecoveryMessage(message);
};
