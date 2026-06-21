import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { retryDynamicImport } from "./lazy-retry";

describe("retryDynamicImport", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the module on first success", async () => {
    const factory = vi.fn().mockResolvedValue({ value: 1 });
    await expect(retryDynamicImport(factory)).resolves.toEqual({ value: 1 });
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it("retries a transient chunk-load failure then succeeds", async () => {
    const chunkError = new Error("Failed to fetch dynamically imported module");
    const factory = vi
      .fn()
      .mockRejectedValueOnce(chunkError)
      .mockResolvedValueOnce({ value: 2 });
    await expect(retryDynamicImport(factory, { delayMs: 1 })).resolves.toEqual({ value: 2 });
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it("does not retry ordinary module errors and rethrows immediately", async () => {
    const realError = new Error("ReferenceError in module");
    const factory = vi.fn().mockRejectedValue(realError);
    await expect(retryDynamicImport(factory, { delayMs: 1 })).rejects.toBe(realError);
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it("recovers from a persistent stale-chunk failure with a single reload", async () => {
    const reloadSpy = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadSpy },
    });
    const chunkError = new Error("Loading chunk app failed");
    const factory = vi.fn().mockRejectedValue(chunkError);

    // The reload path returns a never-settling promise; race it against a tick
    // so the test does not hang, and assert the reload was triggered.
    const result = retryDynamicImport(factory, { delayMs: 1, label: "app" });
    const settled = await Promise.race([
      result.then(() => "settled").catch(() => "rejected"),
      new Promise((resolve) => window.setTimeout(() => resolve("pending"), 20)),
    ]);

    expect(settled).toBe("pending");
    expect(reloadSpy).toHaveBeenCalledTimes(1);
    expect(factory).toHaveBeenCalledTimes(2); // initial + 1 retry before reload
  });
});
