import type { NoteEditorMode } from "@/lib/api";

const NOTE_DRAFT_RECOVERY_PREFIX = "ultra.notes.unsaved.v1";
export const NOTE_DRAFT_RECOVERY_TTL_MS = 30 * 24 * 60 * 60 * 1000;
export const MAX_NOTE_DRAFT_RECOVERY_RECORD_BYTES = 512 * 1024;
export const MAX_NOTE_DRAFT_RECOVERY_SCOPE_BYTES = 2 * 1024 * 1024;
export const MAX_RECOVERED_NOTES_PER_ACCOUNT = 20;

export type NoteDraftRecoveryRecord = {
  note_id: string;
  title: string;
  body_markdown: string;
  pinned: boolean;
  editor_mode: NoteEditorMode;
  expected_revision: number;
  create_key?: string;
  create_attempt?: {
    title: string;
    body_markdown: string;
    pinned: boolean;
    editor_mode: NoteEditorMode;
  };
  stored_at: number;
};

export type NoteDraftRecoveryReadResult =
  | { status: "ok"; record: NoteDraftRecoveryRecord | null }
  | { status: "unavailable" | "scope_disabled"; record: null };

export type NoteDraftRecoveryWriteResult =
  | "stored"
  | "too_large"
  | "budget_exceeded"
  | "unavailable"
  | "scope_disabled";

const disabledScopes = new Set<string>();
const utf8 = new TextEncoder();

export const normalizedNoteDraftRecoveryScope = (
  scope: string | null | undefined
): string => String(scope ?? "").trim();

const scopePrefix = (scope: string): string =>
  `${NOTE_DRAFT_RECOVERY_PREFIX}:${encodeURIComponent(scope)}:`;

const recordKey = (scope: string, noteId: string): string =>
  `${scopePrefix(scope)}${encodeURIComponent(noteId)}`;

const storageBytes = (key: string, raw: string): number =>
  utf8.encode(key).byteLength + utf8.encode(raw).byteLength;

const parseRecord = (raw: string | null): NoteDraftRecoveryRecord | null => {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Record<string, unknown>;
    const noteId = typeof value.note_id === "string" ? value.note_id.trim() : "";
    const editorMode = value.editor_mode === "plaintext" ? "plaintext" : "markdown";
    const revision = Number(value.expected_revision);
    const storedAt = Number(value.stored_at);
    const rawCreateAttempt =
      value.create_attempt && typeof value.create_attempt === "object"
        ? (value.create_attempt as Record<string, unknown>)
        : null;
    const createAttempt =
      rawCreateAttempt &&
      typeof rawCreateAttempt.title === "string" &&
      typeof rawCreateAttempt.body_markdown === "string" &&
      typeof rawCreateAttempt.pinned === "boolean"
        ? {
            title: rawCreateAttempt.title,
            body_markdown: rawCreateAttempt.body_markdown,
            pinned: rawCreateAttempt.pinned,
            editor_mode:
              rawCreateAttempt.editor_mode === "plaintext"
                ? ("plaintext" as const)
                : ("markdown" as const),
          }
        : undefined;
    if (
      !noteId ||
      typeof value.title !== "string" ||
      typeof value.body_markdown !== "string" ||
      typeof value.pinned !== "boolean" ||
      !Number.isSafeInteger(revision) ||
      revision < 0 ||
      !Number.isFinite(storedAt) ||
      storedAt <= 0
    ) {
      return null;
    }
    return {
      note_id: noteId,
      title: value.title,
      body_markdown: value.body_markdown,
      pinned: value.pinned,
      editor_mode: editorMode,
      expected_revision: revision,
      ...(typeof value.create_key === "string" && value.create_key.trim()
        ? { create_key: value.create_key.trim() }
        : {}),
      ...(createAttempt ? { create_attempt: createAttempt } : {}),
      stored_at: storedAt,
    };
  } catch {
    return null;
  }
};

type StoredRecovery = {
  key: string;
  raw: string;
  record: NoteDraftRecoveryRecord;
  bytes: number;
};

const recordsForScope = (
  storage: Storage,
  scope: string,
  now: number
): StoredRecovery[] => {
  const prefix = scopePrefix(scope);
  const keys: string[] = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (key?.startsWith(prefix)) keys.push(key);
  }

  const records: StoredRecovery[] = [];
  for (const key of keys) {
    const raw = storage.getItem(key);
    const record = parseRecord(raw);
    if (!raw || !record || now - record.stored_at > NOTE_DRAFT_RECOVERY_TTL_MS) {
      storage.removeItem(key);
      continue;
    }
    records.push({ key, raw, record, bytes: storageBytes(key, raw) });
  }
  return records;
};

export const resolveBrowserLocalStorage = (): Storage | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
};

export const readNoteDraftRecovery = (
  storage: Storage,
  scopeValue: string | null | undefined,
  noteIdValue: string,
  now = Date.now()
): NoteDraftRecoveryReadResult => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  const noteId = noteIdValue.trim();
  if (!scope || !noteId) return { status: "ok", record: null };
  if (disabledScopes.has(scope)) return { status: "scope_disabled", record: null };
  try {
    const record =
      recordsForScope(storage, scope, now).find(
        (candidate) => candidate.record.note_id === noteId
      )?.record ?? null;
    return { status: "ok", record };
  } catch {
    return { status: "unavailable", record: null };
  }
};

export const readLatestNoteDraftRecovery = (
  storage: Storage,
  scopeValue: string | null | undefined,
  now = Date.now()
): NoteDraftRecoveryReadResult => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  if (!scope) return { status: "ok", record: null };
  if (disabledScopes.has(scope)) return { status: "scope_disabled", record: null };
  try {
    const record =
      recordsForScope(storage, scope, now).sort(
        (left, right) => right.record.stored_at - left.record.stored_at
      )[0]?.record ?? null;
    return { status: "ok", record };
  } catch {
    return { status: "unavailable", record: null };
  }
};

export const writeNoteDraftRecovery = (
  storage: Storage,
  scopeValue: string | null | undefined,
  draft: Omit<NoteDraftRecoveryRecord, "stored_at">,
  now = Date.now()
): NoteDraftRecoveryWriteResult => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  const noteId = draft.note_id.trim();
  if (!scope || !noteId) return "unavailable";
  if (disabledScopes.has(scope)) return "scope_disabled";
  const record: NoteDraftRecoveryRecord = {
    ...draft,
    note_id: noteId,
    stored_at: now,
  };
  const key = recordKey(scope, noteId);
  const raw = JSON.stringify(record);
  const candidateBytes = storageBytes(key, raw);
  if (candidateBytes > MAX_NOTE_DRAFT_RECOVERY_RECORD_BYTES) return "too_large";

  try {
    const existing = recordsForScope(storage, scope, now).filter(
      (candidate) => candidate.key !== key
    );
    if (existing.length >= MAX_RECOVERED_NOTES_PER_ACCOUNT) {
      return "budget_exceeded";
    }
    const totalBytes = existing.reduce((total, candidate) => total + candidate.bytes, 0);
    if (totalBytes + candidateBytes > MAX_NOTE_DRAFT_RECOVERY_SCOPE_BYTES) {
      return "budget_exceeded";
    }
    storage.setItem(key, raw);
    return "stored";
  } catch {
    return "unavailable";
  }
};

export const clearNoteDraftRecovery = (
  storage: Storage,
  scopeValue: string | null | undefined,
  noteIdValue: string
): void => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  const noteId = noteIdValue.trim();
  if (!scope || !noteId) return;
  try {
    storage.removeItem(recordKey(scope, noteId));
  } catch {
    // Device recovery is best effort; API persistence remains authoritative.
  }
};

export const purgeAndDisableNoteDraftRecovery = (
  storage: Storage | null,
  scopeValue: string | null | undefined
): void => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  if (!scope) return;
  // Disable first: an editor unmount flush racing logout cannot recreate data.
  disabledScopes.add(scope);
  if (!storage) return;
  try {
    const prefix = scopePrefix(scope);
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (key?.startsWith(prefix)) keys.push(key);
    }
    keys.forEach((key) => storage.removeItem(key));
  } catch {
    // The in-memory disabled guard still prevents resurrection this session.
  }
};

export const enableNoteDraftRecoveryScope = (
  scopeValue: string | null | undefined
): void => {
  const scope = normalizedNoteDraftRecoveryScope(scopeValue);
  if (scope) disabledScopes.delete(scope);
};
