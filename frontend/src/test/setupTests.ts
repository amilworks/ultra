import "@testing-library/jest-dom/vitest";

// Node 25 exposes an experimental global localStorage whose methods are
// unavailable unless the process receives a storage-file path. jsdom tests
// need an ordinary browser Storage regardless of that host-level flag.
const testStorageValues = new Map<string, string>();
const testLocalStorage = {
  get length() {
    return testStorageValues.size;
  },
  clear: () => testStorageValues.clear(),
  getItem: (key: string) => testStorageValues.get(key) ?? null,
  key: (index: number) => [...testStorageValues.keys()][index] ?? null,
  removeItem: (key: string) => {
    testStorageValues.delete(key);
  },
  setItem: (key: string, value: string) => {
    testStorageValues.set(key, String(value));
  },
} satisfies Storage;
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: testLocalStorage,
});
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: testLocalStorage,
});

// jsdom has no ResizeObserver, which Radix UI primitives (e.g. the Slider thumb's
// size hook) require. A no-op stub is enough for rendering/interaction tests.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
}
