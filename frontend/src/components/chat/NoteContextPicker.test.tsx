import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { ApiError, type ApiClient } from "@/lib/api";
import { reconcileAndUndoNoteSelectionAppend } from "@/lib/noteDirectAppend";
import {
  AddSelectionToNoteDialog,
  NoteContextPicker,
  type NoteSelectionCapture,
} from "./NoteContextPicker";

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

const noteItem = {
  note_id: "note_1",
  title: "Calibration log",
  snippet: "Daily drift observations",
  pinned: false,
  revision: 3,
  updated_at: "2026-08-25T12:00:00Z",
};

describe("NoteContextPicker", () => {
  it("explains the conversation boundary and attaches the chosen revision", async () => {
    const onSelect = vi.fn();
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
    } as unknown as ApiClient;
    render(
      <NoteContextPicker
        apiClient={apiClient}
        open
        selectedNoteIds={[]}
        onOpenChange={vi.fn()}
        onSelect={onSelect}
      />
    );

    expect(
      screen.getByText("Ultra can read it for this message. Content used in chat becomes part of this conversation.")
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Use a note" })).toBeInTheDocument();
    fireEvent.click(await screen.findByText("Calibration log"));
    expect(onSelect).toHaveBeenCalledWith({
      note_id: "note_1",
      title: "Calibration log",
      revision: 3,
    });
  });

  it("makes the eight-Note cap visible", async () => {
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
    } as unknown as ApiClient;
    render(
      <NoteContextPicker
        apiClient={apiClient}
        open
        selectedNoteIds={Array.from({ length: 8 }, (_, index) => `note_${index + 2}`)}
        onOpenChange={vi.fn()}
        onSelect={vi.fn()}
      />
    );

    expect(await screen.findByText("Eight Notes are attached. Remove one before adding another.")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Calibration log").closest("[cmdk-item]")).toHaveAttribute("data-disabled", "true"));
  });

  it("loads recent Notes in pages without changing the server order", async () => {
    const second = { ...noteItem, note_id: "note_2", title: "Older note", revision: 2 };
    const listNotes = vi
      .fn()
      .mockResolvedValueOnce({ notes: [noteItem], total_count: 3 })
      .mockResolvedValueOnce({ notes: [noteItem, second], total_count: 3 });
    const apiClient = { listNotes } as unknown as ApiClient;
    render(
      <NoteContextPicker
        apiClient={apiClient}
        open
        selectedNoteIds={[]}
        onOpenChange={vi.fn()}
        onSelect={vi.fn()}
      />
    );

    await screen.findByText("Calibration log");
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Older note")).toBeInTheDocument();
    expect(screen.getAllByText("Calibration log")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
    expect(listNotes).toHaveBeenNthCalledWith(1, {
      query: undefined,
      sort: "recent",
      limit: 20,
      offset: 0,
    });
    expect(listNotes).toHaveBeenNthCalledWith(2, {
      query: undefined,
      sort: "recent",
      limit: 20,
      offset: 1,
    });
  });
});

const directReceipt = {
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

describe("AddSelectionToNoteDialog", () => {
  it("deduplicates a terminal page while advancing by rows consumed", async () => {
    const listNotes = vi
      .fn()
      .mockResolvedValueOnce({ notes: [noteItem], total_count: 2 })
      .mockResolvedValueOnce({ notes: [noteItem], total_count: 2 });
    const apiClient = { listNotes } as unknown as ApiClient;
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={{ text: "Exact selected text", idempotencyKey: "capture_1" }}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );

    await screen.findByText("Calibration log");
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await waitFor(() => expect(listNotes).toHaveBeenCalledTimes(2));
    expect(listNotes).toHaveBeenLastCalledWith({
      query: undefined,
      sort: "recent",
      limit: 20,
      offset: 1,
    });
    expect(screen.getAllByText("Calibration log")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });

  it("shows the exact preview, recent Notes, and a local New note path", async () => {
    const onNewNote = vi.fn();
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
    } as unknown as ApiClient;
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={{ text: "Exact selected text", idempotencyKey: "capture_1" }}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={onNewNote}
      />
    );

    expect(screen.getByRole("heading", { name: "Add to note" })).toBeInTheDocument();
    expect(screen.getByText("Exact selected text")).toBeInTheDocument();
    await waitFor(() =>
      expect(apiClient.listNotes).toHaveBeenCalledWith({
        query: undefined,
        sort: "recent",
        limit: 20,
        offset: 0,
      })
    );
    fireEvent.click(screen.getByText("New note"));
    expect(onNewNote).toHaveBeenCalledWith("Exact selected text");
  });

  it("removes stale targets while a new search is pending and after it fails", async () => {
    let rejectSearch: ((reason: unknown) => void) | null = null;
    const listNotes = vi
      .fn()
      .mockResolvedValueOnce({ notes: [noteItem], total_count: 1 })
      .mockImplementationOnce(
        () =>
          new Promise((_, reject) => {
            rejectSearch = reject;
          })
      );
    const apiClient = { listNotes } as unknown as ApiClient;
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={{ text: "Exact selected text", idempotencyKey: "capture_1" }}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );

    expect(await screen.findByText("Calibration log")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Search Notes to add to"), {
      target: { value: "different query" },
    });
    await waitFor(() => expect(listNotes).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("Calibration log")).not.toBeInTheDocument();

    expect(rejectSearch).not.toBeNull();
    (rejectSearch as unknown as (reason: unknown) => void)(
      new Error("search unavailable")
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("search unavailable");
    expect(screen.queryByText("Calibration log")).not.toBeInTheDocument();
  });

  it("keeps the exact target visible while an append is in flight", async () => {
    let resolveAppend: ((value: typeof directReceipt) => void) | null = null;
    const appendToNote = vi.fn(
      () =>
        new Promise<typeof directReceipt>((resolve) => {
          resolveAppend = resolve;
        })
    );
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
      appendToNote,
    } as unknown as ApiClient;
    let capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
    };
    const dialog = () => (
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={(attempt) => {
          capture = { ...capture, attempt };
          view.rerender(dialog());
        }}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );
    const view = render(dialog());

    fireEvent.click(await screen.findByText("Calibration log"));
    expect(await screen.findByRole("status")).toHaveTextContent("Adding to Calibration log…");
    expect(screen.queryByLabelText("Search Notes to add to")).not.toBeInTheDocument();

    expect(resolveAppend).not.toBeNull();
    (resolveAppend as unknown as (value: typeof directReceipt) => void)(directReceipt);
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(1));
  });

  it("reuses one key for a transport retry and keeps target-specific keys", async () => {
    const second = { ...noteItem, note_id: "note_2", title: "Field notes", revision: 8 };
    const appendToNote = vi
      .fn()
      .mockRejectedValueOnce(new Error("Connection lost"))
      .mockRejectedValueOnce(new Error("Still offline"))
      .mockResolvedValueOnce({ ...directReceipt, note_id: "note_2", note_title: "Field notes" });
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem, second], total_count: 2 }),
      appendToNote,
    } as unknown as ApiClient;
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={{ text: "Exact selected text", idempotencyKey: "capture_1" }}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );

    fireEvent.click(await screen.findByText("Calibration log"));
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(2));
    expect(appendToNote.mock.calls[0][2]).toBe(appendToNote.mock.calls[1][2]);

    fireEvent.click(screen.getByText("Field notes"));
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(3));
    expect(appendToNote.mock.calls[2][2]).not.toBe(appendToNote.mock.calls[0][2]);
  });

  it("requires an explicit new-key retry after refreshing a revision conflict", async () => {
    const appendToNote = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError("changed", 409, { code: "note_revision_conflict" })
      )
      .mockResolvedValueOnce(directReceipt);
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
      appendToNote,
      getNote: vi.fn().mockResolvedValue({
        ...noteItem,
        body_markdown: "Latest body",
        editor_mode: "markdown",
        content_digest: "digest",
        revision: 4,
        created_at: noteItem.updated_at,
      }),
    } as unknown as ApiClient;
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={{ text: "Exact selected text", idempotencyKey: "capture_1" }}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );

    fireEvent.click(await screen.findByText("Calibration log"));
    fireEvent.click(await screen.findByRole("button", { name: "Retry with latest" }));
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(2));
    expect(appendToNote.mock.calls[0][1].expected_revision).toBe(3);
    expect(appendToNote.mock.calls[1][1].expected_revision).toBe(4);
    expect(appendToNote.mock.calls[1][2]).not.toBe(appendToNote.mock.calls[0][2]);
  });

  it("reconciles a response-lost attempt with the same target, revision, and key after reopen", async () => {
    const second = { ...noteItem, note_id: "note_2", title: "Field notes", revision: 8 };
    const appendToNote = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockResolvedValueOnce(directReceipt);
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem, second], total_count: 2 }),
      appendToNote,
    } as unknown as ApiClient;
    const onAdded = vi.fn();
    let capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
    };
    const dialog = (open: boolean) => (
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open={open}
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={(attempt) => {
          capture = { ...capture, attempt };
          view.rerender(dialog(true));
        }}
        onDiscardAttempt={vi.fn()}
        onAdded={onAdded}
        onNewNote={vi.fn()}
      />
    );
    const view = render(dialog(true));

    fireEvent.click(await screen.findByText("Calibration log"));
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(1));
    expect(capture.attempt).toMatchObject({ note_id: "note_1", expected_revision: 3 });
    expect(screen.queryByText("Field notes")).not.toBeInTheDocument();
    expect(screen.queryByText("New note")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Discard" })).toBeInTheDocument();

    view.rerender(dialog(false));
    view.rerender(dialog(true));
    expect(
      await screen.findByText(/may already have completed/i)
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(onAdded).toHaveBeenCalledWith(
        directReceipt,
        expect.objectContaining({
          note_id: "note_1",
          expected_revision: noteItem.revision,
          status: "pending",
        })
      )
    );

    expect(appendToNote.mock.calls[1][1].expected_revision).toBe(
      appendToNote.mock.calls[0][1].expected_revision
    );
    expect(appendToNote.mock.calls[1][2]).toBe(appendToNote.mock.calls[0][2]);
  });

  it("keeps an uncertain append and its exact key after a replay meets a 401", async () => {
    const appendToNote = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockRejectedValueOnce(
        new ApiError("sign in again", 401, { error: "unauthorized" })
      );
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
      appendToNote,
    } as unknown as ApiClient;
    let capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
    };
    const dialog = () => (
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={(attempt) => {
          capture = { ...capture, attempt };
          view.rerender(dialog());
        }}
        onDiscardAttempt={vi.fn()}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );
    const view = render(dialog());

    fireEvent.click(await screen.findByText("Calibration log"));
    await waitFor(() => expect(capture.attempt?.status).toBe("uncertain"));
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(appendToNote).toHaveBeenCalledTimes(2));

    expect(capture.attempt).toMatchObject({
      note_id: "note_1",
      expected_revision: 3,
      status: "uncertain",
    });
    expect(appendToNote.mock.calls[1]).toEqual(appendToNote.mock.calls[0]);
    expect(screen.getByRole("button", { name: "Discard" })).toBeInTheDocument();
    expect(screen.queryByText("New note")).not.toBeInTheDocument();
  });

  it("releases a restored uncertain append only after stable validation and Discard", async () => {
    const appendToNote = vi.fn().mockRejectedValue(
      new ApiError("append did not commit", 400, { code: "note_append_not_committed" })
    );
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
      appendToNote,
      undoDirectNoteAppendOperation: vi.fn(),
    } as unknown as ApiClient;
    let capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
      attempt: {
        note_id: "note_1",
        note_title: "Calibration log",
        expected_revision: 3,
        idempotency_key: "capture-key",
        status: "uncertain",
      },
    };
    const dialog = () => (
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={(attempt) => {
          capture = { ...capture, attempt };
          view.rerender(dialog());
        }}
        onDiscardAttempt={async () => {
          await reconcileAndUndoNoteSelectionAppend(apiClient, {
            text: capture.text,
            attempt: capture.attempt!,
          });
          capture = { ...capture, attempt: undefined };
          view.rerender(dialog());
        }}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );
    const view = render(dialog());

    expect(screen.queryByText("New note")).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    await waitFor(() => expect(capture.attempt?.status).toBe("rejected"));
    expect(screen.queryByText("New note")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(await screen.findByText("New note")).toBeInTheDocument();
    expect(appendToNote).toHaveBeenCalledOnce();
    expect(apiClient.undoDirectNoteAppendOperation).not.toHaveBeenCalled();
  });

  it("replays and undoes an uncertain append before Discard clears its target", async () => {
    const appendToNote = vi.fn().mockResolvedValue(directReceipt);
    const undoDirectNoteAppendOperation = vi.fn().mockResolvedValue({
      ...directReceipt,
      undo_revision: 5,
      undone_at: "2026-08-27T12:01:00Z",
    });
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
      appendToNote,
      undoDirectNoteAppendOperation,
    } as unknown as ApiClient;
    let capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
      attempt: {
        note_id: "note_1",
        note_title: "Calibration log",
        expected_revision: 3,
        idempotency_key: "capture-key",
        status: "uncertain",
      },
    };
    const dialog = () => (
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={async () => {
          await reconcileAndUndoNoteSelectionAppend(apiClient, {
            text: capture.text,
            attempt: capture.attempt!,
          });
          capture = { ...capture, attempt: undefined };
          view.rerender(dialog());
        }}
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );
    const view = render(dialog());

    fireEvent.click(await screen.findByRole("button", { name: "Discard" }));
    await waitFor(() => expect(undoDirectNoteAppendOperation).toHaveBeenCalledWith("operation_1"));

    expect(appendToNote).toHaveBeenCalledWith(
      "note_1",
      { body_markdown: "Exact selected text", expected_revision: 3 },
      "capture-key"
    );
    expect(capture.attempt).toBeUndefined();
    expect(await screen.findByText("Calibration log")).toBeInTheDocument();
  });

  it("retains and locks an uncertain target when Discard reconciliation fails", async () => {
    const failure = new TypeError("connection lost again");
    const appendToNote = vi.fn().mockRejectedValue(failure);
    const undoDirectNoteAppendOperation = vi.fn();
    const apiClient = {
      appendToNote,
      undoDirectNoteAppendOperation,
    } as unknown as ApiClient;
    const capture: NoteSelectionCapture = {
      text: "Exact selected text",
      idempotencyKey: "capture_1",
      attempt: {
        note_id: "note_1",
        note_title: "Calibration log",
        expected_revision: 3,
        idempotency_key: "capture-key",
        status: "uncertain",
      },
    };
    render(
      <AddSelectionToNoteDialog
        apiClient={apiClient}
        open
        capture={capture}
        onOpenChange={vi.fn()}
        onAttemptChange={vi.fn()}
        onDiscardAttempt={() =>
          reconcileAndUndoNoteSelectionAppend(apiClient, {
            text: capture.text,
            attempt: capture.attempt!,
          }).then(() => undefined)
        }
        onAdded={vi.fn()}
        onNewNote={vi.fn()}
      />
    );

    fireEvent.click(await screen.findByRole("button", { name: "Discard" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn’t safely discard this add");
    expect(capture.attempt).toBeDefined();
    expect(undoDirectNoteAppendOperation).not.toHaveBeenCalled();
    expect(screen.queryByText("New note")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Search Notes")).not.toBeInTheDocument();
  });
});
