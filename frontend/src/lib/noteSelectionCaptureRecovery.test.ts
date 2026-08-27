import { describe, expect, it } from "vitest";

import {
  NOTE_SELECTION_CAPTURE_MAX_BYTES,
  clearNoteSelectionCaptureRecovery,
  clearNoteSelectionCaptureRecoveryIfMatches,
  persistNoteSelectionCaptureRecovery,
  readNoteSelectionCaptureRecovery,
} from "./noteSelectionCaptureRecovery";

const capture = {
  text: "Exact selected text",
  idempotencyKey: "selection:key",
  attempt: {
    note_id: "note_1",
    note_title: "Protocol",
    expected_revision: 7,
    idempotency_key: "append:key",
    status: "pending" as const,
  },
};

describe("note selection capture session recovery", () => {
  it("restores only the matching owner and marks an in-flight write uncertain", () => {
    expect(persistNoteSelectionCaptureRecovery(sessionStorage, "owner-a", capture, 1_000)).toBe(
      true
    );

    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner-b", 1_001)).toBeNull();
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner-a", 1_001)).toEqual({
      ...capture,
      attempt: { ...capture.attempt, status: "uncertain" },
    });
  });

  it("retains unresolved responsibility for the lifetime of sessionStorage", () => {
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", capture, 1_000);
    expect(
      readNoteSelectionCaptureRecovery(sessionStorage, "owner", Number.MAX_SAFE_INTEGER)
    ).not.toBeNull();
  });

  it("does not retain rejected or targetless attempts", () => {
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", capture, 1_000);
    expect(
      persistNoteSelectionCaptureRecovery(
        sessionStorage,
        "owner",
        { ...capture, attempt: { ...capture.attempt, status: "rejected" } },
        1_001
      )
    ).toBe(true);
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner", 1_002)).toBeNull();

    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", capture, 1_003);
    persistNoteSelectionCaptureRecovery(
      sessionStorage,
      "owner",
      { text: capture.text, idempotencyKey: capture.idempotencyKey },
      1_004
    );
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner", 1_005)).toBeNull();
  });

  it("rejects oversized plaintext without replacing a prior recoverable attempt", () => {
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", capture, 1_000);
    const oversized = { ...capture, text: "🧬".repeat(NOTE_SELECTION_CAPTURE_MAX_BYTES) };

    expect(
      persistNoteSelectionCaptureRecovery(sessionStorage, "owner", oversized, 1_001)
    ).toBe(false);
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner", 1_002)?.text).toBe(
      capture.text
    );
  });

  it("clears only the requested owner", () => {
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner-a", capture, 1_000);
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner-b", capture, 1_000);
    clearNoteSelectionCaptureRecovery(sessionStorage, "owner-a");

    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner-a", 1_001)).toBeNull();
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner-b", 1_001)).not.toBeNull();
  });

  it("compare-and-clear cannot erase a newer same-owner attempt", () => {
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", capture, 1_000);
    const newer = {
      ...capture,
      idempotencyKey: "selection:newer",
      attempt: { ...capture.attempt, idempotency_key: "append:newer" },
    };
    persistNoteSelectionCaptureRecovery(sessionStorage, "owner", newer, 1_001);

    expect(
      clearNoteSelectionCaptureRecoveryIfMatches(sessionStorage, "owner", {
        captureKey: capture.idempotencyKey,
        appendKey: capture.attempt.idempotency_key,
      })
    ).toBe(false);
    expect(readNoteSelectionCaptureRecovery(sessionStorage, "owner", 1_002)).toMatchObject({
      idempotencyKey: newer.idempotencyKey,
      attempt: { idempotency_key: newer.attempt.idempotency_key },
    });
    expect(
      clearNoteSelectionCaptureRecoveryIfMatches(sessionStorage, "owner", {
        captureKey: newer.idempotencyKey,
        appendKey: newer.attempt.idempotency_key,
      })
    ).toBe(true);
  });
});
