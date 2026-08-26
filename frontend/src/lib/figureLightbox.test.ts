import { afterEach, describe, expect, it, vi } from "vitest";

import {
  closeFigureLightbox,
  getFigureLightboxState,
  openFigureLightbox,
  subscribeFigureLightbox,
} from "./figureLightbox";

afterEach(() => {
  closeFigureLightbox();
});

describe("figureLightbox store", () => {
  it("opens with figures + clamped index and notifies subscribers", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeFigureLightbox(listener);
    openFigureLightbox(
      [
        { url: "/a.png", title: "A" },
        { url: "/b.png", title: "B" },
      ],
      5
    );
    expect(listener).toHaveBeenCalledTimes(1);
    const state = getFigureLightboxState();
    expect(state?.figures).toHaveLength(2);
    expect(state?.index).toBe(1); // clamped to last
    unsubscribe();
  });

  it("drops figures without a url and is a no-op when empty", () => {
    openFigureLightbox([{ url: "", title: "X" }]);
    expect(getFigureLightboxState()).toBeNull();
    openFigureLightbox([{ url: "/ok.png", title: "ok" }, { url: "", title: "skip" }], 0);
    expect(getFigureLightboxState()?.figures).toHaveLength(1);
  });

  it("close clears state", () => {
    openFigureLightbox([{ url: "/a.png", title: "A" }]);
    expect(getFigureLightboxState()).not.toBeNull();
    closeFigureLightbox();
    expect(getFigureLightboxState()).toBeNull();
  });
});
