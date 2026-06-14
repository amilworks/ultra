import { describe, expect, it } from "vitest";

import { classifyWheelGesture } from "./wheelGesture";

describe("classifyWheelGesture", () => {
  const base = { deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, metaKey: false };

  it("treats a pinch (ctrlKey) as zoom", () => {
    expect(classifyWheelGesture({ ...base, deltaY: 3.2, ctrlKey: true })).toBe("zoom");
  });

  it("treats ⌘/meta + scroll as zoom", () => {
    expect(classifyWheelGesture({ ...base, deltaY: 20, metaKey: true })).toBe("zoom");
  });

  it("treats a classic mouse wheel (line mode / coarse step) as zoom", () => {
    expect(classifyWheelGesture({ ...base, deltaY: 3, deltaMode: 1 })).toBe("zoom");
    expect(classifyWheelGesture({ ...base, deltaY: 100 })).toBe("zoom");
    expect(classifyWheelGesture({ ...base, deltaY: -120 })).toBe("zoom");
  });

  it("treats a trackpad two-finger scroll as pan", () => {
    expect(classifyWheelGesture({ ...base, deltaX: 12, deltaY: 0 })).toBe("pan"); // horizontal
    expect(classifyWheelGesture({ ...base, deltaY: 8.5 })).toBe("pan"); // fine/fractional
    expect(classifyWheelGesture({ ...base, deltaY: 12 })).toBe("pan"); // small step
  });
});
