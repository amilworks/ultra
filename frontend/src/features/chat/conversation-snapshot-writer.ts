export type ConversationSnapshotWrite<TSnapshot> = {
  conversationId: string;
  fingerprint: string;
  snapshot: TSnapshot;
};

type RetryHandle = ReturnType<typeof globalThis.setTimeout>;

type ConversationSnapshotWriterOptions<TSnapshot> = {
  write: (snapshot: TSnapshot) => Promise<unknown>;
  onAcknowledged: (conversationId: string, fingerprint: string) => void;
  retryBaseDelayMs?: number;
  retryMaxDelayMs?: number;
  scheduleRetry?: (callback: () => void, delayMs: number) => RetryHandle;
  cancelRetry?: (handle: RetryHandle) => void;
};

type WriteLane<TSnapshot> = {
  acknowledgedFingerprint: string | null;
  inFlight: ConversationSnapshotWrite<TSnapshot> | null;
  pending: ConversationSnapshotWrite<TSnapshot> | null;
  retryAttempt: number;
  retryHandle: RetryHandle | null;
};

export type ConversationSnapshotWriter<TSnapshot> = {
  enqueue: (entry: ConversationSnapshotWrite<TSnapshot>) => void;
  cancelPending: (conversationId: string) => void;
  seedAcknowledged: (conversationId: string, fingerprint: string) => void;
  reset: (conversationId?: string) => void;
  dispose: () => void;
};

/**
 * A latest-value write lane per conversation.
 *
 * Only one request for a conversation can be in flight. While it runs, newer
 * snapshots replace the pending snapshot rather than creating an overlapping
 * request. A rejected latest snapshot is retried with bounded backoff and is
 * never acknowledged until the server accepts it.
 */
export const createConversationSnapshotWriter = <TSnapshot>({
  write,
  onAcknowledged,
  retryBaseDelayMs = 500,
  retryMaxDelayMs = 10_000,
  scheduleRetry = (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
  cancelRetry = (handle) => globalThis.clearTimeout(handle),
}: ConversationSnapshotWriterOptions<TSnapshot>): ConversationSnapshotWriter<TSnapshot> => {
  const lanes = new Map<string, WriteLane<TSnapshot>>();
  let disposed = false;

  const normalizedDelay = (attempt: number): number =>
    Math.min(
      Math.max(0, retryMaxDelayMs),
      Math.max(0, retryBaseDelayMs) * 2 ** Math.max(0, attempt - 1)
    );

  const clearRetry = (lane: WriteLane<TSnapshot>): void => {
    if (lane.retryHandle === null) {
      return;
    }
    cancelRetry(lane.retryHandle);
    lane.retryHandle = null;
  };

  const drain = (conversationId: string, lane: WriteLane<TSnapshot>): void => {
    if (
      disposed ||
      lanes.get(conversationId) !== lane ||
      lane.inFlight ||
      lane.retryHandle !== null ||
      !lane.pending
    ) {
      return;
    }

    const entry = lane.pending;
    lane.pending = null;
    lane.inFlight = entry;

    let request: Promise<unknown>;
    try {
      request = Promise.resolve(write(entry.snapshot));
    } catch (error) {
      request = Promise.reject(error);
    }
    void request.then(
      () => {
        if (disposed || lanes.get(conversationId) !== lane || lane.inFlight !== entry) {
          return;
        }
        lane.inFlight = null;
        lane.retryAttempt = 0;
        lane.acknowledgedFingerprint = entry.fingerprint;
        onAcknowledged(conversationId, entry.fingerprint);
        drain(conversationId, lane);
      },
      () => {
        if (disposed || lanes.get(conversationId) !== lane || lane.inFlight !== entry) {
          return;
        }
        lane.inFlight = null;
        // A rejected response may still have followed a committed server
        // write. Until a retry succeeds, no prior fingerprint is certain.
        lane.acknowledgedFingerprint = null;
        // A newer state makes retrying this rejected snapshot unnecessary.
        // Drain it immediately; otherwise retain this exact snapshot for retry.
        if (lane.pending && lane.pending.fingerprint !== entry.fingerprint) {
          lane.retryAttempt = 0;
          drain(conversationId, lane);
          return;
        }
        lane.pending = entry;
        lane.retryAttempt += 1;
        lane.retryHandle = scheduleRetry(() => {
          if (disposed || lanes.get(conversationId) !== lane) {
            return;
          }
          lane.retryHandle = null;
          drain(conversationId, lane);
        }, normalizedDelay(lane.retryAttempt));
      }
    );
  };

  const enqueue = (entry: ConversationSnapshotWrite<TSnapshot>): void => {
    if (disposed) {
      return;
    }
    const conversationId = String(entry.conversationId || "").trim();
    const fingerprint = String(entry.fingerprint || "");
    if (!conversationId || !fingerprint) {
      return;
    }
    let lane = lanes.get(conversationId);
    if (!lane) {
      lane = {
        acknowledgedFingerprint: null,
        inFlight: null,
        pending: null,
        retryAttempt: 0,
        retryHandle: null,
      };
      lanes.set(conversationId, lane);
    }
    const normalizedEntry = { ...entry, conversationId, fingerprint };
    if (lane.inFlight) {
      if (lane.inFlight.fingerprint === fingerprint) {
        // The UI reverted to (or reaffirmed) the in-flight state. Any different
        // pending snapshot is no longer desired and must not land afterward.
        lane.pending = null;
      } else {
        // Even an already-acknowledged fingerprint must be queued here: the
        // different in-flight request may commit, so the revert has to follow it.
        lane.pending = normalizedEntry;
      }
      return;
    }
    if (lane.pending?.fingerprint === fingerprint) {
      return;
    }
    const hadUnsettledWrite = lane.pending !== null || lane.retryHandle !== null;
    if (!hadUnsettledWrite && lane.acknowledgedFingerprint === fingerprint) {
      return;
    }

    lane.pending = normalizedEntry;
    // A fresh, newer snapshot should not wait behind an old retry timer.
    if (lane.retryHandle !== null) {
      clearRetry(lane);
      lane.retryAttempt = 0;
    }
    drain(conversationId, lane);
  };

  const reset = (conversationId?: string): void => {
    const normalizedConversationId = String(conversationId || "").trim();
    if (normalizedConversationId) {
      const lane = lanes.get(normalizedConversationId);
      if (lane) {
        clearRetry(lane);
        lanes.delete(normalizedConversationId);
      }
      return;
    }
    lanes.forEach(clearRetry);
    lanes.clear();
  };

  const cancelPending = (conversationId: string): void => {
    const normalizedConversationId = String(conversationId || "").trim();
    const lane = lanes.get(normalizedConversationId);
    if (!lane) {
      return;
    }
    lane.pending = null;
    lane.retryAttempt = 0;
    clearRetry(lane);
  };

  const seedAcknowledged = (conversationId: string, fingerprint: string): void => {
    if (disposed) {
      return;
    }
    const normalizedConversationId = String(conversationId || "").trim();
    const normalizedFingerprint = String(fingerprint || "");
    if (!normalizedConversationId || !normalizedFingerprint) {
      return;
    }
    let lane = lanes.get(normalizedConversationId);
    if (!lane) {
      lane = {
        acknowledgedFingerprint: null,
        inFlight: null,
        pending: null,
        retryAttempt: 0,
        retryHandle: null,
      };
      lanes.set(normalizedConversationId, lane);
    }
    // Hydration seeds a known server state before local writes begin. Never
    // use a late seed to erase knowledge of work already queued/in flight.
    if (!lane.inFlight && !lane.pending && lane.retryHandle === null) {
      lane.acknowledgedFingerprint = normalizedFingerprint;
    }
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    reset();
    disposed = true;
  };

  return { enqueue, cancelPending, seedAcknowledged, reset, dispose };
};
