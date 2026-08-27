import { beforeEach, describe, expect, it } from "vitest";

import {
  clearNoteDraftRecovery,
  enableNoteDraftRecoveryScope,
  MAX_NOTE_DRAFT_RECOVERY_RECORD_BYTES,
  MAX_RECOVERED_NOTES_PER_ACCOUNT,
  NOTE_DRAFT_RECOVERY_TTL_MS,
  purgeAndDisableNoteDraftRecovery,
  readLatestNoteDraftRecovery,
  readNoteDraftRecovery,
  resolveBrowserLocalStorage,
  writeNoteDraftRecovery,
} from "./noteDraftRecovery";

const draft = (noteId: string, body = "Exact unsaved text") => ({
  note_id: noteId,
  title: `Title ${noteId}`,
  body_markdown: body,
  pinned: false,
  editor_mode: "markdown" as const,
  expected_revision: 1,
});

describe("Note draft recovery", () => {
  let values: Map<string, string>;
  let storage: Storage;

  beforeEach(() => {
    values = new Map<string, string>();
    storage = {
      get length() {
        return values.size;
      },
      clear: () => values.clear(),
      getItem: (key: string) => values.get(key) ?? null,
      key: (index: number) => [...values.keys()][index] ?? null,
      removeItem: (key: string) => values.delete(key),
      setItem: (key: string, value: string) => values.set(key, value),
    } as unknown as Storage;
  });

  it("keeps exact drafts isolated by stable account scope and Note", () => {
    expect(
      writeNoteDraftRecovery(storage, "workos:user-1:org-1", {
        ...draft("note_1", "Exact unsaved\ntext"),
        title: "Protocol",
        pinned: true,
        expected_revision: 7,
      })
    ).toBe("stored");

    expect(
      readNoteDraftRecovery(storage, "workos:user-1:org-1", "note_1").record
    ).toMatchObject({
      title: "Protocol",
      body_markdown: "Exact unsaved\ntext",
      pinned: true,
      expected_revision: 7,
    });
    expect(
      readNoteDraftRecovery(storage, "WORKOS:USER-1:ORG-1", "note_1").record
    ).toBeNull();
    expect(readNoteDraftRecovery(storage, "other", "note_1").record).toBeNull();
    expect(readNoteDraftRecovery(storage, "workos:user-1:org-1", "note_2").record).toBeNull();
  });

  it("keeps the exact TTL boundary and removes the record one millisecond later", () => {
    const now = 1_800_000_000_000;
    expect(writeNoteDraftRecovery(storage, "ttl-user", draft("note_1"), now)).toBe("stored");
    expect(
      readNoteDraftRecovery(storage, "ttl-user", "note_1", now + NOTE_DRAFT_RECOVERY_TTL_MS)
        .record?.note_id
    ).toBe("note_1");
    expect(
      readNoteDraftRecovery(storage, "ttl-user", "note_1", now + NOTE_DRAFT_RECOVERY_TTL_MS + 1)
        .record
    ).toBeNull();
    expect(values.size).toBe(0);
  });

  it("measures multibyte UTF-8 and preserves an older record when overwrite is too large", () => {
    expect(writeNoteDraftRecovery(storage, "size-user", draft("note_1", "safe"))).toBe("stored");
    const oversized = "🧬".repeat(Math.ceil(MAX_NOTE_DRAFT_RECOVERY_RECORD_BYTES / 4));
    expect(writeNoteDraftRecovery(storage, "size-user", draft("note_1", oversized))).toBe(
      "too_large"
    );
    expect(readNoteDraftRecovery(storage, "size-user", "note_1").record?.body_markdown).toBe(
      "safe"
    );
  });

  it("rejects count and aggregate budget breaches without evicting valid drafts", () => {
    for (let index = 0; index < MAX_RECOVERED_NOTES_PER_ACCOUNT; index += 1) {
      expect(writeNoteDraftRecovery(storage, "count-user", draft(`note_${index}`))).toBe("stored");
    }
    expect(writeNoteDraftRecovery(storage, "count-user", draft("overflow"))).toBe(
      "budget_exceeded"
    );
    expect(readLatestNoteDraftRecovery(storage, "count-user").record).not.toBeNull();

    for (let index = 0; index < 4; index += 1) {
      expect(
        writeNoteDraftRecovery(
          storage,
          "byte-user",
          draft(`large_${index}`, "x".repeat(480 * 1024))
        )
      ).toBe("stored");
    }
    expect(
      writeNoteDraftRecovery(
        storage,
        "byte-user",
        draft("large_4", "x".repeat(480 * 1024))
      )
    ).toBe("budget_exceeded");
    expect(readNoteDraftRecovery(storage, "byte-user", "large_0").record).not.toBeNull();
  });

  it("re-enables device recovery when an account departure is rejected", () => {
    expect(writeNoteDraftRecovery(storage, "account-a", draft("note_1"))).toBe("stored");
    purgeAndDisableNoteDraftRecovery(storage, "account-a");
    expect(writeNoteDraftRecovery(storage, "account-a", draft("note_2"))).toBe(
      "scope_disabled"
    );

    enableNoteDraftRecoveryScope("account-a");
    expect(writeNoteDraftRecovery(storage, "account-a", draft("note_2"))).toBe("stored");
    expect(readNoteDraftRecovery(storage, "account-a", "note_2").record).toMatchObject({
      note_id: "note_2",
      body_markdown: "Exact unsaved text",
    });
  });

  it("prunes expired and malformed owned entries without touching other scopes or keys", () => {
    const now = 1_800_000_000_000;
    writeNoteDraftRecovery(storage, "owner", draft("expired"), now - NOTE_DRAFT_RECOVERY_TTL_MS - 1);
    writeNoteDraftRecovery(storage, "other", draft("keep"), now);
    values.set("ultra.notes.unsaved.v1:owner:malformed", "not-json");
    values.set("unrelated", "keep-me");

    expect(readLatestNoteDraftRecovery(storage, "owner", now).record).toBeNull();
    expect([...values.keys()].some((key) => key.includes("owner"))).toBe(false);
    expect(readNoteDraftRecovery(storage, "other", "keep", now).record?.note_id).toBe("keep");
    expect(values.get("unrelated")).toBe("keep-me");
  });

  it("reports storage getter and operation failures without throwing", () => {
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      get: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    });
    expect(resolveBrowserLocalStorage()).toBeNull();
    if (original) Object.defineProperty(window, "localStorage", original);

    const inaccessible = {
      get length() {
        throw new DOMException("blocked", "SecurityError");
      },
    } as unknown as Storage;
    expect(readLatestNoteDraftRecovery(inaccessible, "failure-user").status).toBe("unavailable");
    expect(writeNoteDraftRecovery(inaccessible, "failure-user", draft("note_1"))).toBe(
      "unavailable"
    );

    const quota = { ...storage, setItem: () => { throw new DOMException("full", "QuotaExceededError"); } } as Storage;
    expect(writeNoteDraftRecovery(quota, "quota-user", draft("note_1"))).toBe("unavailable");
  });

  it("purges only the departing scope and disables unmount resurrection until re-enabled", () => {
    writeNoteDraftRecovery(storage, "account-a", draft("note_a"));
    writeNoteDraftRecovery(storage, "account-b", draft("note_b"));
    purgeAndDisableNoteDraftRecovery(storage, "account-a");

    expect(readNoteDraftRecovery(storage, "account-a", "note_a").status).toBe("scope_disabled");
    expect(readNoteDraftRecovery(storage, "account-b", "note_b").record?.note_id).toBe("note_b");
    expect(writeNoteDraftRecovery(storage, "account-a", draft("resurrect"))).toBe("scope_disabled");

    enableNoteDraftRecoveryScope("account-a");
    expect(writeNoteDraftRecovery(storage, "account-a", draft("new-session"))).toBe("stored");
    clearNoteDraftRecovery(storage, "account-a", "new-session");
    expect(readLatestNoteDraftRecovery(storage, "account-a").record).toBeNull();
  });
});
