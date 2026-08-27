import { readFileSync } from "node:fs";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  noteAccessForTurn,
  noteAppendProposalRequested,
} from "@/lib/notesAccess";
import { createConversationSnapshotWriter } from "./conversation-snapshot-writer";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

type NotesSnapshot = {
  label: "edit" | "retry";
  text: string;
  excludedNoteIntentText: string[];
};

type Deferred = {
  promise: Promise<void>;
  resolve: () => void;
};

const deferred = (): Deferred => {
  let resolve = (): void => undefined;
  const promise = new Promise<void>((onResolve) => {
    resolve = onResolve;
  });
  return { promise, resolve };
};

const settleMicrotasks = async (): Promise<void> => {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve();
  }
};

afterEach(() => {
  vi.useRealTimers();
});

describe("conversation snapshot writer", () => {
  it("routes every current snapshot through the writer and hashes only acknowledgements", () => {
    const start = appSource.indexOf("const flushConversationSnapshots = useCallback");
    const end = appSource.indexOf("const flushConversationSnapshotsRef", start);
    const flush = appSource.slice(start, end);
    const writerStart = appSource.indexOf("const conversationSnapshotWriter = useMemo");
    const writerEnd = appSource.indexOf("void hydrateResourceUploadProgressFromQueueStore", writerStart);
    const wiring = appSource.slice(writerStart, writerEnd);

    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    expect(flush).toContain("conversationSnapshotWriter.enqueue({");
    expect(flush).not.toContain("Promise.allSettled(");
    expect(flush).not.toMatch(/persistedConversationHashesRef\.current\[[^\]]+\] === fingerprint/);
    expect(wiring).toMatch(
      /onAcknowledged:[\s\S]*persistedConversationHashesRef\.current\[conversationId\] = fingerprint/
    );
  });

  it("retries a rejected Notes edit without acknowledging or losing paste provenance", async () => {
    vi.useFakeTimers();
    const command = "Search my notes for the calibration protocol";
    const snapshot: NotesSnapshot = {
      label: "edit",
      text: command,
      excludedNoteIntentText: [command],
    };
    const writes: NotesSnapshot[] = [];
    const acknowledged: string[] = [];
    let attempt = 0;
    const writer = createConversationSnapshotWriter<NotesSnapshot>({
      retryBaseDelayMs: 25,
      retryMaxDelayMs: 25,
      write: async (record) => {
        writes.push(record);
        attempt += 1;
        if (attempt === 1) {
          throw new Error("temporary snapshot failure");
        }
      },
      onAcknowledged: (_conversationId, fingerprint) => {
        acknowledged.push(fingerprint);
      },
    });

    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "edit-with-provenance",
      snapshot,
    });
    await settleMicrotasks();

    expect(writes).toHaveLength(1);
    expect(acknowledged).toEqual([]);

    await vi.runOnlyPendingTimersAsync();
    await settleMicrotasks();

    expect(writes).toEqual([snapshot, snapshot]);
    expect(acknowledged).toEqual(["edit-with-provenance"]);
    expect(
      noteAccessForTurn(
        writes[1].text,
        [],
        writes[1].excludedNoteIntentText
      )
    ).toBeNull();
    writer.dispose();
  });

  it("keeps the retry snapshot last when completions are resolved in reverse order", async () => {
    const command = "Add this result to my notes";
    const edit: NotesSnapshot = {
      label: "edit",
      text: command,
      excludedNoteIntentText: [],
    };
    const retry: NotesSnapshot = {
      label: "retry",
      text: command,
      excludedNoteIntentText: [command],
    };
    const editWrite = deferred();
    const retryWrite = deferred();
    const calls: NotesSnapshot[] = [];
    const acknowledged: string[] = [];
    let persisted: NotesSnapshot | null = null;
    const writer = createConversationSnapshotWriter<NotesSnapshot>({
      write: (snapshot) => {
        calls.push(snapshot);
        const gate = snapshot.label === "edit" ? editWrite : retryWrite;
        return gate.promise.then(() => {
          persisted = snapshot;
        });
      },
      onAcknowledged: (_conversationId, fingerprint) => {
        acknowledged.push(fingerprint);
      },
    });

    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "edit-before-provenance",
      snapshot: edit,
    });
    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "retry-with-provenance",
      snapshot: retry,
    });

    retryWrite.resolve();
    await settleMicrotasks();
    expect(calls).toEqual([edit]);
    expect(acknowledged).toEqual([]);

    editWrite.resolve();
    await settleMicrotasks();

    expect(calls).toEqual([edit, retry]);
    expect(acknowledged).toEqual([
      "edit-before-provenance",
      "retry-with-provenance",
    ]);
    expect(persisted).toEqual(retry);
    const finalSnapshot = persisted as NotesSnapshot | null;
    expect(
      noteAppendProposalRequested(
        finalSnapshot?.text ?? "",
        finalSnapshot?.excludedNoteIntentText ?? []
      )
    ).toBe(false);
    writer.dispose();
  });

  it("cancels a pending edit when current state reverts to the in-flight retry", async () => {
    const command = "Search my notes for the calibration protocol";
    const retry: NotesSnapshot = {
      label: "retry",
      text: command,
      excludedNoteIntentText: [command],
    };
    const edit: NotesSnapshot = {
      label: "edit",
      text: `${command}\nDraft edit`,
      excludedNoteIntentText: [],
    };
    const retryWrite = deferred();
    const calls: NotesSnapshot[] = [];
    const writer = createConversationSnapshotWriter<NotesSnapshot>({
      write: (snapshot) => {
        calls.push(snapshot);
        return retryWrite.promise;
      },
      onAcknowledged: () => undefined,
    });

    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "retry-safe",
      snapshot: retry,
    });
    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "edit-unsafe",
      snapshot: edit,
    });
    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "retry-safe",
      snapshot: retry,
    });

    retryWrite.resolve();
    await settleMicrotasks();

    expect(calls).toEqual([retry]);
    expect(noteAccessForTurn(retry.text, [], retry.excludedNoteIntentText)).toBeNull();
    writer.dispose();
  });

  it("queues an acknowledged-state revert behind a different in-flight write", async () => {
    const command = "Add this result to my notes";
    const safe: NotesSnapshot = {
      label: "retry",
      text: command,
      excludedNoteIntentText: [command],
    };
    const unsafe: NotesSnapshot = {
      label: "edit",
      text: command,
      excludedNoteIntentText: [],
    };
    const firstSafeWrite = deferred();
    const unsafeWrite = deferred();
    const revertedSafeWrite = deferred();
    const gates = [firstSafeWrite, unsafeWrite, revertedSafeWrite];
    const calls: NotesSnapshot[] = [];
    let persisted: NotesSnapshot | null = null;
    const writer = createConversationSnapshotWriter<NotesSnapshot>({
      write: (snapshot) => {
        const gate = gates[calls.length];
        calls.push(snapshot);
        return gate.promise.then(() => {
          persisted = snapshot;
        });
      },
      onAcknowledged: () => undefined,
    });

    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "safe",
      snapshot: safe,
    });
    firstSafeWrite.resolve();
    await settleMicrotasks();

    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "unsafe",
      snapshot: unsafe,
    });
    writer.enqueue({
      conversationId: "conversation_1",
      fingerprint: "safe",
      snapshot: safe,
    });
    unsafeWrite.resolve();
    await settleMicrotasks();

    expect(calls).toEqual([safe, unsafe, safe]);
    revertedSafeWrite.resolve();
    await settleMicrotasks();

    expect(persisted).toEqual(safe);
    const finalSnapshot = persisted as NotesSnapshot | null;
    expect(
      noteAppendProposalRequested(
        finalSnapshot?.text ?? "",
        finalSnapshot?.excludedNoteIntentText ?? []
      )
    ).toBe(false);
    writer.dispose();
  });
});
