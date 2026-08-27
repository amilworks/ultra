import { boundedNoteIntentExclusions } from "./notesAccess";

const COMPOSER_DRAFT_STORAGE_VERSION = 2 as const;

export type ComposerDraftStorageState = {
  drafts: Record<string, string>;
  excludedNoteIntentTextByConversationId: Record<string, string[]>;
};

type StoredComposerDraft = {
  text: string;
  excluded_note_intent_text: string[];
};

type StoredComposerDraftEnvelope = {
  version: typeof COMPOSER_DRAFT_STORAGE_VERSION;
  drafts: Record<string, StoredComposerDraft>;
};

const emptyComposerDraftStorageState = (): ComposerDraftStorageState => ({
  drafts: {},
  excludedNoteIntentTextByConversationId: {},
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const normalizedConversationId = (value: string): string => value.trim();

const normalizedExcludedText = (value: unknown): string[] | null => {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    return null;
  }
  return boundedNoteIntentExclusions(value);
};

/**
 * Restores unsent composer text and its paste provenance from one storage
 * record. Legacy drafts did not carry provenance, so a restored legacy draft
 * is treated entirely as reference text for Notes authority. It stays fully
 * editable/sendable; only private Notes search/append capability fails closed.
 */
export const parseComposerDraftStorage = (raw: string | null): ComposerDraftStorageState => {
  if (!raw) {
    return emptyComposerDraftStorageState();
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyComposerDraftStorageState();
  }
  if (!isRecord(parsed)) {
    return emptyComposerDraftStorageState();
  }

  const result = emptyComposerDraftStorageState();
  const candidateDrafts = parsed.drafts;
  const versionedDrafts =
    parsed.version === COMPOSER_DRAFT_STORAGE_VERSION && isRecord(candidateDrafts)
      ? candidateDrafts
      : null;
  const isVersionedEnvelope = versionedDrafts !== null;
  const entries = Object.entries(versionedDrafts ?? parsed);

  entries.forEach(([rawConversationId, value]) => {
    const conversationId = normalizedConversationId(rawConversationId);
    if (!conversationId) {
      return;
    }

    // Version 1 was simply { [conversationId]: draftText }. Because its paste
    // provenance is unknowable after reload, exclude the whole restored draft
    // from capability detection rather than guessing that it was typed.
    if (!isVersionedEnvelope) {
      if (typeof value !== "string") {
        return;
      }
      result.drafts[conversationId] = value;
      if (value.length > 0) {
        result.excludedNoteIntentTextByConversationId[conversationId] = [value];
      }
      return;
    }

    // An interrupted/partial migration may place a legacy string inside the
    // v2 envelope. Preserve its visible draft and fail closed just like v1.
    if (typeof value === "string") {
      result.drafts[conversationId] = value;
      if (value.length > 0) {
        result.excludedNoteIntentTextByConversationId[conversationId] = [value];
      }
      return;
    }
    if (!isRecord(value) || typeof value.text !== "string") {
      return;
    }
    result.drafts[conversationId] = value.text;
    const excludedText = normalizedExcludedText(value.excluded_note_intent_text);
    if (excludedText === null) {
      // A partial/corrupt versioned entry must not silently become authority.
      if (value.text.length > 0) {
        result.excludedNoteIntentTextByConversationId[conversationId] = [value.text];
      }
      return;
    }
    if (excludedText.length > 0) {
      result.excludedNoteIntentTextByConversationId[conversationId] = excludedText;
    }
  });

  return result;
};

/** Serializes draft text and paste provenance together for one atomic write. */
export const serializeComposerDraftStorage = ({
  drafts,
  excludedNoteIntentTextByConversationId,
}: ComposerDraftStorageState): string => {
  const storedDrafts: Record<string, StoredComposerDraft> = {};
  Object.entries(drafts).forEach(([rawConversationId, text]) => {
    const conversationId = normalizedConversationId(rawConversationId);
    if (!conversationId || typeof text !== "string") {
      return;
    }
    storedDrafts[conversationId] = {
      text,
      excluded_note_intent_text:
        normalizedExcludedText(excludedNoteIntentTextByConversationId[conversationId] ?? []) ??
        (text.length > 0 ? [text] : []),
    };
  });
  const envelope: StoredComposerDraftEnvelope = {
    version: COMPOSER_DRAFT_STORAGE_VERSION,
    drafts: storedDrafts,
  };
  return JSON.stringify(envelope);
};
