import type { BriefFileToken } from "@/features/chat/brief-tokens";

/**
 * Where a conversation's brief tokens live between page loads.
 *
 * A token is only meaningful next to the draft text it decorates, so this
 * rides beside the composer draft store — but in its OWN key, with its own
 * tiny codec, so the draft codec (which has a versioned envelope, provenance
 * semantics and its own tests) never learns about tokens and neither writer
 * can clobber the other's fields. Everything read back is untrusted: every
 * entry is validated and bounded before it is trusted to map a label to a
 * file id.
 */

export const BRIEF_TOKEN_STORAGE_KEY = "bisque.frontend.composerBriefTokens";
const BRIEF_TOKEN_STORAGE_VERSION = 1 as const;
const MAX_CONVERSATIONS = 200;
const MAX_TOKENS_PER_CONVERSATION = 64;
const MAX_LABEL_LENGTH = 80;

export type BriefTokenStorageState = Record<string, BriefFileToken[]>;

type StoredBriefToken = { label: string; file_id: string };
type StoredBriefTokenEnvelope = {
  version: typeof BRIEF_TOKEN_STORAGE_VERSION;
  tokens: Record<string, StoredBriefToken[]>;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const normalizeTokens = (value: unknown): BriefFileToken[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const seenIds = new Set<string>();
  const seenLabels = new Set<string>();
  const tokens: BriefFileToken[] = [];
  for (const entry of value) {
    if (!isRecord(entry)) {
      continue;
    }
    const label = typeof entry.label === "string" ? entry.label.trim() : "";
    const fileId = typeof entry.file_id === "string" ? entry.file_id.trim() : "";
    if (
      !label ||
      !fileId ||
      label.length > MAX_LABEL_LENGTH ||
      seenIds.has(fileId) ||
      seenLabels.has(label)
    ) {
      continue;
    }
    seenIds.add(fileId);
    seenLabels.add(label);
    tokens.push({ label, fileId });
    if (tokens.length >= MAX_TOKENS_PER_CONVERSATION) {
      break;
    }
  }
  return tokens;
};

export const parseBriefTokenStorage = (raw: string | null): BriefTokenStorageState => {
  if (!raw) {
    return {};
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }
  if (!isRecord(parsed) || parsed.version !== BRIEF_TOKEN_STORAGE_VERSION || !isRecord(parsed.tokens)) {
    return {};
  }
  const state: BriefTokenStorageState = {};
  let count = 0;
  for (const [rawConversationId, value] of Object.entries(parsed.tokens)) {
    const conversationId = rawConversationId.trim();
    if (!conversationId) {
      continue;
    }
    const tokens = normalizeTokens(value);
    if (tokens.length === 0) {
      continue;
    }
    state[conversationId] = tokens;
    count += 1;
    if (count >= MAX_CONVERSATIONS) {
      break;
    }
  }
  return state;
};

export const serializeBriefTokenStorage = (state: BriefTokenStorageState): string => {
  const tokens: Record<string, StoredBriefToken[]> = {};
  Object.entries(state).forEach(([rawConversationId, entries]) => {
    const conversationId = rawConversationId.trim();
    if (!conversationId || !Array.isArray(entries) || entries.length === 0) {
      return;
    }
    tokens[conversationId] = entries
      .slice(0, MAX_TOKENS_PER_CONVERSATION)
      .map((token) => ({ label: token.label, file_id: token.fileId }));
  });
  const envelope: StoredBriefTokenEnvelope = { version: BRIEF_TOKEN_STORAGE_VERSION, tokens };
  return JSON.stringify(envelope);
};

export const readBriefTokenStorage = (): BriefTokenStorageState => {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    return parseBriefTokenStorage(window.localStorage.getItem(BRIEF_TOKEN_STORAGE_KEY));
  } catch {
    return {};
  }
};
