// Multi-tab stream presence (S5 owner-election, advisory). When the same conversation is open in
// several tabs, each tab otherwise opens its own SSE recovery/poll stream for an in-flight run — N
// duplicate firehoses. This lets a tab that is actively driving a run's stream broadcast a heartbeat
// over a BroadcastChannel so peers can DEFER their own redundant recovery. It is advisory, never a
// hard lock: if the owner stops heart-beating (tab closed/crashed/navigated), its heartbeat goes
// stale after HEARTBEAT_TTL_MS and a peer takes over — so a dead owner can never wedge streaming.
//
// The staleness core (PresenceRegistry) is pure + time-injected for deterministic tests; the
// BroadcastChannel wiring is a thin, SSR-safe shell around it.

const HEARTBEAT_INTERVAL_MS = 2000;
export const HEARTBEAT_TTL_MS = 5000; // > 2x interval: tolerate a missed beat before takeover

type HeartbeatMessage = { type: "stream-heartbeat"; runId: string; tabId: string; at: number };
type ReleaseMessage = { type: "stream-release"; runId: string; tabId: string };
type PresenceMessage = HeartbeatMessage | ReleaseMessage;

// Pure registry of peer heartbeats keyed by runId. `now` is injected so staleness is testable.
export class PresenceRegistry {
  private readonly peers = new Map<string, { tabId: string; at: number }>();

  record(runId: string, tabId: string, at: number): void {
    this.peers.set(runId, { tabId, at });
  }

  release(runId: string, tabId: string): void {
    const current = this.peers.get(runId);
    if (current && current.tabId === tabId) {
      this.peers.delete(runId);
    }
  }

  // True when a DIFFERENT tab has a fresh (non-stale) heartbeat for this run, i.e. a live owner
  // exists, so the caller should defer its own redundant stream.
  hasLiveOwner(runId: string, selfTabId: string, now: number): boolean {
    const current = this.peers.get(runId);
    if (!current || current.tabId === selfTabId) {
      return false;
    }
    if (now - current.at > HEARTBEAT_TTL_MS) {
      this.peers.delete(runId); // stale → the owner is gone; allow takeover
      return false;
    }
    return true;
  }
}

export type StreamPresence = {
  /** Begin (and keep) broadcasting ownership of this run's stream until stop() is called. */
  start: (runId: string) => void;
  /** Whether another live tab currently owns this run's stream (so this tab should defer). */
  isOwnedElsewhere: (runId: string) => boolean;
  stop: () => void;
};

type BroadcastChannelLike = {
  postMessage: (message: unknown) => void;
  close: () => void;
  onmessage: ((event: { data: unknown }) => void) | null;
};

const noopPresence: StreamPresence = {
  start: () => {},
  isOwnedElsewhere: () => false,
  stop: () => {},
};

// Create a presence coordinator. Dependencies (channel factory, clock, tab id) are injectable for
// tests; in the browser they default to BroadcastChannel + Date.now + a random tab id.
export const createStreamPresence = (options?: {
  channelFactory?: (name: string) => BroadcastChannelLike | null;
  now?: () => number;
  tabId?: string;
}): StreamPresence => {
  const now = options?.now ?? (() => Date.now());
  const tabId =
    options?.tabId ??
    `tab-${now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;

  const makeChannel =
    options?.channelFactory ??
    ((name: string): BroadcastChannelLike | null => {
      if (typeof BroadcastChannel === "undefined") {
        return null;
      }
      return new BroadcastChannel(name) as unknown as BroadcastChannelLike;
    });

  const channel = makeChannel("ultra-stream-presence");
  if (!channel) {
    return noopPresence; // no BroadcastChannel (SSR/old browser): every tab streams independently
  }

  const registry = new PresenceRegistry();
  let ownedRunId: string | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  channel.onmessage = (event) => {
    const message = event.data as PresenceMessage | undefined;
    if (!message || typeof message !== "object") {
      return;
    }
    if (message.type === "stream-heartbeat") {
      registry.record(message.runId, message.tabId, message.at);
    } else if (message.type === "stream-release") {
      registry.release(message.runId, message.tabId);
    }
  };

  const beat = () => {
    if (!ownedRunId) {
      return;
    }
    channel.postMessage({ type: "stream-heartbeat", runId: ownedRunId, tabId, at: now() } satisfies HeartbeatMessage);
  };

  return {
    start: (runId: string) => {
      ownedRunId = runId;
      beat();
      if (!heartbeatTimer) {
        heartbeatTimer = setInterval(beat, HEARTBEAT_INTERVAL_MS);
      }
    },
    isOwnedElsewhere: (runId: string) => registry.hasLiveOwner(runId, tabId, now()),
    stop: () => {
      if (ownedRunId) {
        channel.postMessage({ type: "stream-release", runId: ownedRunId, tabId } satisfies ReleaseMessage);
      }
      ownedRunId = null;
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    },
  };
};
