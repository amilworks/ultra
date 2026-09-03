export type AuthenticatedLandingResumeKind = "active-run" | "draft";

type LandingConversationLike = {
  id: string;
  updatedAt?: number | null;
  hydrated?: boolean | null;
  sending?: boolean | null;
  streamingMessageId?: string | null;
  historyRunning?: boolean | null;
  prompt?: string | null;
};

export type AuthenticatedLandingResumeTarget = {
  conversationId: string;
  kind: AuthenticatedLandingResumeKind;
};

const orderedNonActiveConversations = <Conversation extends LandingConversationLike>(
  conversations: readonly Conversation[],
  activeConversationId: string | null | undefined
): Conversation[] => {
  const activeId = String(activeConversationId ?? "").trim();
  return conversations
    .filter((conversation) => conversation.id.trim() && conversation.id !== activeId)
    .sort((left, right) => Number(right.updatedAt ?? 0) - Number(left.updatedAt ?? 0));
};

/**
 * Choose the single piece of work worth putting above generic welcome starts.
 * A live run wins over a draft because it is changing without the user; within
 * either class, the most recently touched conversation wins.
 */
export const findAuthenticatedLandingResumeTarget = <
  Conversation extends LandingConversationLike,
>(
  conversations: readonly Conversation[],
  draftsByConversationId: Readonly<Record<string, string>>,
  activeConversationId: string | null | undefined
): AuthenticatedLandingResumeTarget | null => {
  const candidates = orderedNonActiveConversations(conversations, activeConversationId);
  const running = candidates.find((conversation) =>
    Boolean(
      conversation.sending ||
        conversation.streamingMessageId ||
        conversation.historyRunning
    )
  );
  if (running) {
    return { conversationId: running.id, kind: "active-run" };
  }

  const drafted = candidates.find((conversation) => {
    const locallyDrafted = draftsByConversationId[conversation.id];
    const prompt = typeof locallyDrafted === "string" ? locallyDrafted : conversation.prompt;
    return Boolean(prompt?.trim());
  });
  return drafted ? { conversationId: drafted.id, kind: "draft" } : null;
};

/**
 * Composer text is local-first. If its blank conversation never reached the
 * server, keep one recoverable local shell so the new landing can offer Resume
 * instead of either reopening it or throwing it away.
 */
export const findOrphanComposerDraft = (
  draftsByConversationId: Readonly<Record<string, string>>,
  knownConversationIds: ReadonlySet<string>
): { conversationId: string; prompt: string } | null => {
  for (const [conversationId, prompt] of Object.entries(draftsByConversationId)) {
    const normalizedId = conversationId.trim();
    if (!normalizedId || knownConversationIds.has(normalizedId) || !prompt.trim()) {
      continue;
    }
    return { conversationId: normalizedId, prompt };
  }
  return null;
};
