import { describe, expect, it } from "vitest";

import { measureTextareaCaret } from "./textareaCaret";

describe("measureTextareaCaret", () => {
  it("never throws and always cleans up its mirror", () => {
    const textarea = document.createElement("textarea");
    textarea.value = "Register the EBSD map in @fus";
    document.body.appendChild(textarea);
    const before = document.body.childElementCount;
    const result = measureTextareaCaret(textarea, 29);
    expect(result === null || typeof result.left === "number").toBe(true);
    if (result) {
      expect(Number.isFinite(result.top)).toBe(true);
      expect(result.height).toBeGreaterThan(0);
    }
    expect(document.body.childElementCount).toBe(before);
    textarea.remove();
  });

  it("clamps an out-of-range position instead of failing", () => {
    const textarea = document.createElement("textarea");
    textarea.value = "abc";
    document.body.appendChild(textarea);
    expect(() => measureTextareaCaret(textarea, 999)).not.toThrow();
    expect(() => measureTextareaCaret(textarea, -5)).not.toThrow();
    textarea.remove();
  });
});
