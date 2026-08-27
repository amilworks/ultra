import {
  isDefinitiveNoteWriteReplayRejection,
  type ApiClient,
  type NoteDirectAppendReceipt,
} from "@/lib/api";

export type NoteSelectionAppendSnapshot = {
  text: string;
  attempt: {
    note_id: string;
    expected_revision: number;
    idempotency_key: string;
    status?: "pending" | "uncertain" | "rejected";
  };
};

/**
 * Resolve an append before forgetting it. A rejected attempt is known not to
 * have committed. Every other attempt is replayed with its exact request and,
 * if a receipt exists, conditionally undone before this function resolves.
 * Callers must retain the snapshot whenever this throws.
 */
export const reconcileAndUndoNoteSelectionAppend = async (
  apiClient: ApiClient,
  snapshot: NoteSelectionAppendSnapshot
): Promise<NoteDirectAppendReceipt | null> => {
  const { attempt } = snapshot;
  if (attempt.status === "rejected") return null;

  let receipt: NoteDirectAppendReceipt;
  try {
    receipt = await apiClient.appendToNote(
      attempt.note_id,
      {
        body_markdown: snapshot.text,
        expected_revision: attempt.expected_revision,
      },
      attempt.idempotency_key
    );
  } catch (error) {
    // Once an earlier response was lost, ordinary replay failures cannot
    // rewrite history. Only the server's receipt-first, post-lookup terminal
    // codes prove that no operation exists; authority, conflict, rate-limit,
    // proxy validation, network, and server failures remain ambiguous.
    if (isDefinitiveNoteWriteReplayRejection(error, "append")) return null;
    throw error;
  }

  if (!receipt.undone_at) {
    await apiClient.undoDirectNoteAppendOperation(receipt.operation_id);
  }
  return receipt;
};
