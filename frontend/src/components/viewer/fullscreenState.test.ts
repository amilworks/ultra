import { describe, expect, it } from "vitest";

import { isTypingTarget, keyToFullscreenAction, nextFullscreen } from "./fullscreenState";

describe("nextFullscreen", () => {
  it("toggles from the current flag", () => {
    expect(nextFullscreen(false, "toggle")).toBe(true);
    expect(nextFullscreen(true, "toggle")).toBe(false);
  });

  it("always resolves to false on exit", () => {
    expect(nextFullscreen(true, "exit")).toBe(false);
    expect(nextFullscreen(false, "exit")).toBe(false);
  });
});

describe("keyToFullscreenAction", () => {
  it("maps F (either case) to toggle", () => {
    expect(keyToFullscreenAction("f", false)).toBe("toggle");
    expect(keyToFullscreenAction("F", false)).toBe("toggle");
  });

  it("maps Escape to exit", () => {
    expect(keyToFullscreenAction("Escape", false)).toBe("exit");
  });

  it("ignores keys while typing in a field", () => {
    expect(keyToFullscreenAction("f", true)).toBeNull();
    expect(keyToFullscreenAction("Escape", true)).toBeNull();
  });

  it("ignores F when a modifier is held (do not hijack browser shortcuts)", () => {
    expect(keyToFullscreenAction("f", false, true)).toBeNull();
  });

  it("ignores unrelated keys", () => {
    expect(keyToFullscreenAction("g", false)).toBeNull();
  });
});

describe("isTypingTarget", () => {
  it("detects input / textarea / select", () => {
    for (const tag of ["input", "textarea", "select"] as const) {
      expect(isTypingTarget(document.createElement(tag))).toBe(true);
    }
  });

  it("is false for non-typing elements", () => {
    expect(isTypingTarget(document.createElement("div"))).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });

  it("detects contenteditable", () => {
    const el = document.createElement("div");
    el.setAttribute("contenteditable", "true");
    // jsdom does not derive isContentEditable from the attribute; emulate the DOM.
    Object.defineProperty(el, "isContentEditable", { value: true });
    expect(isTypingTarget(el)).toBe(true);
  });
});
