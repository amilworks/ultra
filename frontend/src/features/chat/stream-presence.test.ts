import { describe, expect, it } from "vitest";

import { createStreamPresence, HEARTBEAT_TTL_MS, PresenceRegistry } from "./stream-presence";

describe("PresenceRegistry", () => {
  it("reports a live owner only for a fresh heartbeat from a DIFFERENT tab", () => {
    const reg = new PresenceRegistry();
    reg.record("run1", "tabA", 0);
    expect(reg.hasLiveOwner("run1", "tabB", 1000)).toBe(true); // fresh peer
    expect(reg.hasLiveOwner("run1", "tabA", 1000)).toBe(false); // self is never "elsewhere"
  });

  it("expires a stale heartbeat so a peer can take over a dead owner's run", () => {
    const reg = new PresenceRegistry();
    reg.record("run1", "tabA", 0);
    expect(reg.hasLiveOwner("run1", "tabB", HEARTBEAT_TTL_MS + 1)).toBe(false); // stale → no owner
    // The stale entry was cleared; a fresh heartbeat re-establishes ownership.
    reg.record("run1", "tabA", 100);
    expect(reg.hasLiveOwner("run1", "tabB", 200)).toBe(true);
  });

  it("release clears ownership", () => {
    const reg = new PresenceRegistry();
    reg.record("run1", "tabA", 0);
    reg.release("run1", "tabA");
    expect(reg.hasLiveOwner("run1", "tabB", 100)).toBe(false);
  });
});

describe("createStreamPresence (multi-tab owner election)", () => {
  it("lets a peer tab see a live owner and defer, then reclaim after release", () => {
    // A synchronous in-memory BroadcastChannel bus shared by the simulated tabs.
    type Ch = {
      onmessage: ((e: { data: unknown }) => void) | null;
      postMessage: (data: unknown) => void;
      close: () => void;
    };
    const channels: Ch[] = [];
    const factory = (): Ch => {
      const ch: Ch = {
        onmessage: null,
        postMessage: (data: unknown) => {
          for (const other of channels) {
            if (other !== ch && other.onmessage) {
              other.onmessage({ data });
            }
          }
        },
        close: () => {},
      };
      channels.push(ch);
      return ch;
    };
    const now = () => 1000;

    const tabA = createStreamPresence({ channelFactory: factory, now, tabId: "A" });
    const tabB = createStreamPresence({ channelFactory: factory, now, tabId: "B" });

    expect(tabB.isOwnedElsewhere("run1")).toBe(false); // nobody streaming yet

    tabA.start("run1"); // broadcasts a heartbeat synchronously
    expect(tabB.isOwnedElsewhere("run1")).toBe(true); // B sees A's heartbeat → would defer
    expect(tabA.isOwnedElsewhere("run1")).toBe(false); // A is the owner, not "elsewhere"

    tabA.stop(); // broadcasts release
    expect(tabB.isOwnedElsewhere("run1")).toBe(false); // B can now take over

    tabB.stop();
  });

  it("degrades to independent streaming when BroadcastChannel is unavailable", () => {
    const presence = createStreamPresence({ channelFactory: () => null, now: () => 0, tabId: "A" });
    presence.start("run1");
    expect(presence.isOwnedElsewhere("run1")).toBe(false); // no coordination → never defers
    presence.stop();
  });
});
