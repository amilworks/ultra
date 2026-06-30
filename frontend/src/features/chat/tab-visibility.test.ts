import { describe, expect, it, vi } from "vitest";

import { isTabHidden, onVisibilityChange } from "./tab-visibility";

describe("tab-visibility", () => {
  it("reflects document.visibilityState", () => {
    const spy = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    expect(isTabHidden()).toBe(true);
    spy.mockReturnValue("visible");
    expect(isTabHidden()).toBe(false);
    spy.mockRestore();
  });

  it("subscribes and unsubscribes to visibilitychange", () => {
    const handler = vi.fn();
    const off = onVisibilityChange(handler);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(handler).toHaveBeenCalledTimes(1);
    off();
    document.dispatchEvent(new Event("visibilitychange"));
    expect(handler).toHaveBeenCalledTimes(1); // no further calls after unsubscribe
  });
});
