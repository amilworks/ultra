import { describe, expect, it } from "vitest";

import { windowTailMessages } from "./message-window";

const ids = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `m${i}` }));

describe("windowTailMessages", () => {
  it("returns everything when at or under the window", () => {
    const msgs = ids(10);
    const { visible, hiddenCount } = windowTailMessages(msgs, 50);
    expect(visible).toBe(msgs); // same reference, no copy
    expect(hiddenCount).toBe(0);
  });

  it("keeps the most-recent window and reports the hidden count for long threads", () => {
    const msgs = ids(120);
    const { visible, hiddenCount } = windowTailMessages(msgs, 50);
    expect(visible).toHaveLength(50);
    expect(hiddenCount).toBe(70);
    // tail-anchored: the newest (and streaming) message is always included.
    expect(visible[visible.length - 1].id).toBe("m119");
    expect(visible[0].id).toBe("m70");
  });

  it("expands to show more as the window grows", () => {
    const msgs = ids(120);
    expect(windowTailMessages(msgs, 100).visible).toHaveLength(100);
    expect(windowTailMessages(msgs, 100).hiddenCount).toBe(20);
    expect(windowTailMessages(msgs, 200).hiddenCount).toBe(0);
  });
});
