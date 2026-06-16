import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CHROME_IDLE_THRESHOLD_MS,
  createChromeFadeController,
  resolveChromeVisibility,
} from "./chromeVisibility";

describe("resolveChromeVisibility", () => {
  it("forces visible under reduced motion regardless of idle", () => {
    expect(
      resolveChromeVisibility({ focusWithin: false, idleMs: 99999, reducedMotion: true })
    ).toBe("visible");
  });

  it("forces visible while keyboard focus is within chrome", () => {
    expect(
      resolveChromeVisibility({ focusWithin: true, idleMs: 99999, reducedMotion: false })
    ).toBe("visible");
  });

  it("stays visible before the idle threshold", () => {
    expect(
      resolveChromeVisibility({ focusWithin: false, idleMs: 100, reducedMotion: false })
    ).toBe("visible");
  });

  it("fades once idle past the threshold", () => {
    expect(
      resolveChromeVisibility({
        focusWithin: false,
        idleMs: CHROME_IDLE_THRESHOLD_MS + 1,
        reducedMotion: false,
      })
    ).toBe("faded");
  });
});

describe("createChromeFadeController", () => {
  afterEach(() => vi.useRealTimers());

  it("reveals immediately and fades after the idle threshold", () => {
    vi.useFakeTimers();
    const root = document.createElement("div");
    const ctrl = createChromeFadeController(root, { idleThreshold: 1000 });
    ctrl.reveal();
    expect(root.dataset.chromeFaded).toBe("false");
    vi.advanceTimersByTime(1001);
    expect(root.dataset.chromeFaded).toBe("true");
    ctrl.dispose();
  });

  it("never fades under reduced motion", () => {
    vi.useFakeTimers();
    const root = document.createElement("div");
    const ctrl = createChromeFadeController(root, { idleThreshold: 1000, reducedMotion: true });
    ctrl.reveal();
    vi.advanceTimersByTime(5000);
    expect(root.dataset.chromeFaded).toBe("false");
    ctrl.fadeNow();
    expect(root.dataset.chromeFaded).toBe("false"); // fadeNow is a no-op under reduced motion
    ctrl.dispose();
  });

  it("fadeNow fades immediately and cancels the pending timer", () => {
    vi.useFakeTimers();
    const root = document.createElement("div");
    const ctrl = createChromeFadeController(root, { idleThreshold: 1000 });
    ctrl.reveal();
    ctrl.fadeNow();
    expect(root.dataset.chromeFaded).toBe("true");
    vi.advanceTimersByTime(2000);
    expect(root.dataset.chromeFaded).toBe("true"); // no late flip back
    ctrl.dispose();
  });
});
