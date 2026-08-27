import type { NoteSelectionCapture } from "@/components/chat/NoteContextPicker";

const STORAGE_PREFIX = "ultra.notes.selection-capture.v1";
export const NOTE_SELECTION_CAPTURE_MAX_BYTES = 96 * 1024;
const utf8 = new TextEncoder();

const normalizedScope = (scope: string | null | undefined): string =>
  String(scope ?? "").trim();

const storageKey = (scope: string): string =>
  `${STORAGE_PREFIX}:${encodeURIComponent(scope)}`;

type StoredCapture = {
  text: string;
  idempotency_key: string;
  note_id: string;
  note_title: string;
  expected_revision: number;
  append_idempotency_key: string;
  stored_at: number;
};

export const resolveBrowserSessionStorage = (): Storage | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
};

export const clearNoteSelectionCaptureRecovery = (
  storage: Storage | null,
  scopeValue: string | null | undefined
): void => {
  const scope = normalizedScope(scopeValue);
  if (!storage || !scope) return;
  try {
    storage.removeItem(storageKey(scope));
  } catch {
    // Session recovery is best effort; the server idempotency receipt remains authoritative.
  }
};

export const clearNoteSelectionCaptureRecoveryIfMatches = (
  storage: Storage | null,
  scopeValue: string | null | undefined,
  expected: { captureKey: string; appendKey?: string }
): boolean => {
  const scope = normalizedScope(scopeValue);
  if (!storage || !scope) return false;
  const key = storageKey(scope);
  try {
    const raw = storage.getItem(key);
    if (!raw) return true;
    const value = JSON.parse(raw) as Partial<StoredCapture>;
    if (
      value.idempotency_key !== expected.captureKey ||
      (expected.appendKey !== undefined &&
        value.append_idempotency_key !== expected.appendKey)
    ) {
      return false;
    }
    storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
};

export const persistNoteSelectionCaptureRecovery = (
  storage: Storage | null,
  scopeValue: string | null | undefined,
  capture: NoteSelectionCapture,
  now = Date.now()
): boolean => {
  const scope = normalizedScope(scopeValue);
  const attempt = capture.attempt;
  if (!storage || !scope) return false;
  if (!attempt || attempt.status === "rejected") {
    return clearNoteSelectionCaptureRecoveryIfMatches(storage, scope, {
      captureKey: capture.idempotencyKey,
      ...(attempt ? { appendKey: attempt.idempotency_key } : {}),
    });
  }

  const record: StoredCapture = {
    text: capture.text,
    idempotency_key: capture.idempotencyKey,
    note_id: attempt.note_id,
    note_title: attempt.note_title,
    expected_revision: attempt.expected_revision,
    append_idempotency_key: attempt.idempotency_key,
    stored_at: now,
  };
  const raw = JSON.stringify(record);
  if (utf8.encode(storageKey(scope)).byteLength + utf8.encode(raw).byteLength > NOTE_SELECTION_CAPTURE_MAX_BYTES) {
    return false;
  }
  try {
    storage.setItem(storageKey(scope), raw);
    return true;
  } catch {
    return false;
  }
};

export const readNoteSelectionCaptureRecovery = (
  storage: Storage | null,
  scopeValue: string | null | undefined,
  _now = Date.now()
): NoteSelectionCapture | null => {
  // Deliberately session-lifetime, not TTL-based: an unresolved mutation may
  // still have committed and must remain reconcilable until this tab closes.
  void _now;
  const scope = normalizedScope(scopeValue);
  if (!storage || !scope) return null;
  const key = storageKey(scope);
  try {
    const raw = storage.getItem(key);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<StoredCapture>;
    const revision = Number(value.expected_revision);
    const storedAt = Number(value.stored_at);
    const valid =
      typeof value.text === "string" &&
      typeof value.idempotency_key === "string" &&
      Boolean(value.idempotency_key.trim()) &&
      typeof value.note_id === "string" &&
      Boolean(value.note_id.trim()) &&
      typeof value.note_title === "string" &&
      Number.isSafeInteger(revision) &&
      revision >= 0 &&
      typeof value.append_idempotency_key === "string" &&
      Boolean(value.append_idempotency_key.trim()) &&
      Number.isFinite(storedAt) &&
      storedAt > 0;
    if (!valid) {
      storage.removeItem(key);
      return null;
    }
    return {
      text: value.text!,
      idempotencyKey: value.idempotency_key!.trim(),
      attempt: {
        note_id: value.note_id!.trim(),
        note_title: value.note_title!,
        expected_revision: revision,
        idempotency_key: value.append_idempotency_key!.trim(),
        // A page transition can happen after the request reached the server but
        // before the browser observed its receipt. Always reconcile on resume.
        status: "uncertain",
      },
    };
  } catch {
    try {
      storage.removeItem(key);
    } catch {
      // Ignore a second storage failure.
    }
    return null;
  }
};
