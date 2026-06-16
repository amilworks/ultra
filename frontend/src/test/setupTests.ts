import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver, which Radix UI primitives (e.g. the Slider thumb's
// size hook) require. A no-op stub is enough for rendering/interaction tests.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
}
