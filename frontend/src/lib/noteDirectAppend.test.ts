import { describe, expect, it, vi } from "vitest";

import { ApiError, type ApiClient } from "./api";
import { reconcileAndUndoNoteSelectionAppend } from "./noteDirectAppend";

const receipt = {
  operation_id: "operation_1",
  note_id: "note_1",
  note_title: "Calibration log",
  before_revision: 3,
  after_revision: 4,
  appended_bytes: 12,
  before_content_digest: "before",
  after_content_digest: "after",
  created_at: "2026-08-27T12:00:00Z",
};

const snapshot = {
  text: "Exact selected text",
  attempt: {
    note_id: "note_1",
    expected_revision: 3,
    idempotency_key: "capture-key",
    status: "uncertain" as const,
  },
};

describe("reconcileAndUndoNoteSelectionAppend", () => {
  it("replays the exact uncertain append and undoes its stable receipt", async () => {
    const appendToNote = vi.fn().mockResolvedValue(receipt);
    const undoDirectNoteAppendOperation = vi.fn().mockResolvedValue({
      ...receipt,
      undo_revision: 5,
      undone_at: "2026-08-27T12:01:00Z",
    });
    const apiClient = { appendToNote, undoDirectNoteAppendOperation } as unknown as ApiClient;

    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).resolves.toEqual(receipt);
    expect(appendToNote).toHaveBeenCalledWith(
      "note_1",
      { body_markdown: "Exact selected text", expected_revision: 3 },
      "capture-key"
    );
    expect(undoDirectNoteAppendOperation).toHaveBeenCalledWith("operation_1");
  });

  it("clears a typed deterministic rejection without replaying it", async () => {
    const appendToNote = vi.fn();
    const undoDirectNoteAppendOperation = vi.fn();
    const apiClient = { appendToNote, undoDirectNoteAppendOperation } as unknown as ApiClient;

    await expect(
      reconcileAndUndoNoteSelectionAppend(apiClient, {
        ...snapshot,
        attempt: { ...snapshot.attempt, status: "rejected" },
      })
    ).resolves.toBeNull();
    expect(appendToNote).not.toHaveBeenCalled();
    expect(undoDirectNoteAppendOperation).not.toHaveBeenCalled();
  });

  it("retains responsibility when an uncertain replay is rejected by current auth", async () => {
    const appendToNote = vi.fn().mockRejectedValue(
      new ApiError("sign in again", 401, { error: "unauthorized" })
    );
    const undoDirectNoteAppendOperation = vi.fn();
    const apiClient = { appendToNote, undoDirectNoteAppendOperation } as unknown as ApiClient;

    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).rejects.toMatchObject({
      status: 401,
    });
    expect(undoDirectNoteAppendOperation).not.toHaveBeenCalled();
  });

  it("releases an uncertain replay after stable pre-receipt validation", async () => {
    const appendToNote = vi.fn().mockRejectedValue(
      new ApiError("append did not commit", 400, { code: "note_append_not_committed" })
    );
    const undoDirectNoteAppendOperation = vi.fn();
    const apiClient = { appendToNote, undoDirectNoteAppendOperation } as unknown as ApiClient;

    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).resolves.toBeNull();
    expect(appendToNote).toHaveBeenCalledOnce();
    expect(undoDirectNoteAppendOperation).not.toHaveBeenCalled();
  });

  it("keeps generic proxy validation ambiguous without the terminal replay code", async () => {
    const failure = new ApiError("payload too large", 413, { error: "payload too large" });
    const appendToNote = vi.fn().mockRejectedValue(failure);
    const apiClient = {
      appendToNote,
      undoDirectNoteAppendOperation: vi.fn(),
    } as unknown as ApiClient;

    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).rejects.toBe(
      failure
    );
  });

  it("retains responsibility when replay or undo remains uncertain", async () => {
    const replayFailure = new ApiError("gateway timeout", 408, undefined);
    const appendToNote = vi
      .fn()
      .mockRejectedValueOnce(replayFailure)
      .mockResolvedValue(receipt);
    const undoFailure = new TypeError("undo response lost");
    const undoDirectNoteAppendOperation = vi
      .fn()
      .mockRejectedValueOnce(undoFailure)
      .mockResolvedValueOnce({ ...receipt, undone_at: "2026-08-27T12:01:00Z" });
    const apiClient = { appendToNote, undoDirectNoteAppendOperation } as unknown as ApiClient;

    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).rejects.toBe(replayFailure);
    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).rejects.toBe(undoFailure);
    await expect(reconcileAndUndoNoteSelectionAppend(apiClient, snapshot)).resolves.toEqual(receipt);

    expect(appendToNote).toHaveBeenCalledTimes(3);
    expect(appendToNote.mock.calls[0]).toEqual(appendToNote.mock.calls[1]);
    expect(appendToNote.mock.calls[1]).toEqual(appendToNote.mock.calls[2]);
    expect(undoDirectNoteAppendOperation).toHaveBeenCalledTimes(2);
  });
});
