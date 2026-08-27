import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError, type ApiClient, type NoteAppendOperationReceipt } from "@/lib/api";
import type { RunEvent } from "@/types";
import { NoteRunContext } from "./NoteRunContext";

const proposalEvent: RunEvent = {
  event_type: "tool_call.completed",
  payload: {
    tool_name: "propose_note_append",
    proposal_id: "proposal_1",
    note_id: "note_1",
  },
};

const receipt = (overrides: Partial<NoteAppendOperationReceipt> = {}): NoteAppendOperationReceipt => ({
  operation_id: "operation_1",
  proposal_id: "proposal_1",
  run_id: "run_1",
  note_id: "note_1",
  note_title: "Field protocol",
  before_revision: 3,
  after_revision: 4,
  appended_bytes: 19,
  before_content_digest: "before",
  after_content_digest: "after",
  created_at: "2026-08-25T12:00:00Z",
  ...overrides,
});

const pendingProposal = {
  proposal_id: "proposal_1",
  note_id: "note_1",
  note_title: "Field protocol",
  body_markdown: "## Result\n\nOriginal text",
  expected_revision: 3,
  status: "pending" as const,
  expires_at: "2026-08-25T12:15:00Z",
  created_at: "2026-08-25T12:00:00Z",
};

describe("Note append proposal card", () => {
  it("shows exact text, commits the reviewed edit, and offers a safe Undo", async () => {
    const commit = vi.fn().mockResolvedValue(receipt());
    const undo = vi.fn().mockResolvedValue(
      receipt({ undo_revision: 5, undone_at: "2026-08-25T12:02:00Z" })
    );
    const onChanged = vi.fn();
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue(pendingProposal),
      commitNoteAppendProposal: commit,
      undoNoteAppendOperation: undo,
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={onChanged}
      />
    );

    const editor = await screen.findByLabelText("Text to add to note");
    expect(editor).toHaveValue("## Result\n\nOriginal text");
    fireEvent.change(editor, { target: { value: "## Result\n\nReviewed text" } });
    fireEvent.click(screen.getByRole("button", { name: "Add to note" }));

    await waitFor(() =>
      expect(commit).toHaveBeenCalledWith("proposal_1", {
        body_markdown: "## Result\n\nReviewed text",
      })
    );
    expect(await screen.findByText("Added to Field protocol")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(undo).toHaveBeenCalledWith("operation_1"));
    expect(await screen.findByText("Update undone")).toBeInTheDocument();
    expect(onChanged).toHaveBeenCalledTimes(2);
  });

  it("preserves copyable reviewed text and disables a stale proposal after conflict", async () => {
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue(pendingProposal),
      commitNoteAppendProposal: vi.fn().mockRejectedValue(
        new ApiError("conflict", 409, { code: "note_revision_conflict" })
      ),
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={vi.fn()}
      />
    );

    const editor = await screen.findByLabelText("Text to add to note");
    fireEvent.change(editor, { target: { value: "Keep this exact text" } });
    fireEvent.click(screen.getByRole("button", { name: "Add to note" }));

    expect(await screen.findByText(/note changed since this update was prepared/i)).toBeInTheDocument();
    expect(editor).toHaveValue("Keep this exact text");
    expect(editor).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Add to note" })).toBeDisabled();
  });

  it("renders an expired proposal even when the server correctly omits its body", async () => {
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue({
        ...pendingProposal,
        status: "expired",
        body_markdown: undefined,
      }),
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={vi.fn()}
      />
    );

    expect(await screen.findByText("Update to Field protocol expired")).toBeInTheDocument();
    expect(screen.getByText("This proposal is no longer available.")).toBeInTheDocument();
    expect(screen.queryByLabelText("Text to add to note")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open note" })).toBeInTheDocument();
  });

  it("keeps exact text copyable when a loaded proposal expires during commit", async () => {
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue(pendingProposal),
      commitNoteAppendProposal: vi.fn().mockRejectedValue(
        new ApiError("expired", 409, { code: "note_proposal_expired" })
      ),
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={vi.fn()}
      />
    );

    const editor = await screen.findByLabelText("Text to add to note");
    fireEvent.change(editor, { target: { value: "Keep this expired proposal text" } });
    fireEvent.click(screen.getByRole("button", { name: "Add to note" }));

    expect(await screen.findByText(/expired before it could be applied/i)).toBeInTheDocument();
    expect(editor).toHaveValue("Keep this expired proposal text");
    expect(editor).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Add to note" })).toBeDisabled();
  });

  it("keeps the exact reviewed text retryable when a commit receipt is incomplete", async () => {
    const commit = vi
      .fn()
      .mockRejectedValueOnce(
        new Error(
          "Ultra received an incomplete Note append receipt. The result is uncertain; retry the exact request before continuing."
        )
      )
      .mockResolvedValueOnce(receipt());
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue(pendingProposal),
      commitNoteAppendProposal: commit,
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={vi.fn()}
      />
    );

    const editor = await screen.findByLabelText("Text to add to note");
    fireEvent.change(editor, { target: { value: "Keep this exact reviewed text" } });
    fireEvent.click(screen.getByRole("button", { name: "Add to note" }));

    expect(await screen.findByText(/result is uncertain/i)).toBeInTheDocument();
    expect(editor).toHaveValue("Keep this exact reviewed text");
    expect(screen.getByRole("button", { name: "Add to note" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Add to note" }));
    expect(await screen.findByText("Added to Field protocol")).toBeInTheDocument();
    expect(commit).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenLastCalledWith("proposal_1", {
      body_markdown: "Keep this exact reviewed text",
    });
  });

  it("stops retrying an Undo that newer note writing made permanently unsafe", async () => {
    const undo = vi.fn().mockRejectedValue(
      new ApiError("conflict", 409, { code: "note_undo_conflict" })
    );
    const apiClient = {
      getNoteAppendProposal: vi.fn().mockResolvedValue({
        ...pendingProposal,
        status: "committed",
        body_markdown: undefined,
        operation_id: "operation_1",
        operation: receipt(),
      }),
      undoNoteAppendOperation: undo,
    } as unknown as ApiClient;

    render(
      <NoteRunContext
        runEvents={[proposalEvent]}
        apiClient={apiClient}
        onOpenNote={vi.fn()}
        onNoteChanged={vi.fn()}
      />
    );

    const undoButton = await screen.findByRole("button", { name: "Undo" });
    fireEvent.click(undoButton);

    expect(await screen.findByText(/won’t undo over newer writing/i)).toBeInTheDocument();
    expect(undo).toHaveBeenCalledTimes(1);
    expect(undoButton).toBeDisabled();
  });
});
