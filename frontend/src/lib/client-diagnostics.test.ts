import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getClientErrorLog,
  isChunkLoadError,
  reloadOnceForChunkError,
  reportClientError,
} from "./client-diagnostics";

describe("isChunkLoadError", () => {
  it("detects ChunkLoadError by name", () => {
    const error = new Error("boom");
    error.name = "ChunkLoadError";
    expect(isChunkLoadError(error)).toBe(true);
  });

  it("detects failed dynamic import / stale chunk messages", () => {
    expect(isChunkLoadError(new Error("Failed to fetch dynamically imported module: /assets/x.js"))).toBe(true);
    expect(isChunkLoadError(new Error("Loading chunk vendor-ui failed"))).toBe(true);
    expect(isChunkLoadError(new Error("error loading dynamically imported module"))).toBe(true);
    expect(isChunkLoadError("Importing a module script failed.")).toBe(true);
  });

  it("does not flag ordinary errors", () => {
    expect(isChunkLoadError(new Error("Cannot read properties of undefined"))).toBe(false);
    expect(isChunkLoadError(null)).toBe(false);
    expect(isChunkLoadError(undefined)).toBe(false);
  });
});

describe("reportClientError", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("captures an inspectable log entry with chunk classification", () => {
    reportClientError(new Error("Loading chunk 12 failed"), { source: "lazy-import" });
    const log = getClientErrorLog();
    expect(log).toHaveLength(1);
    expect(log[0]).toMatchObject({
      source: "lazy-import",
      message: "Loading chunk 12 failed",
      chunkLoad: true,
      recovered: false,
    });
  });

  it("retains only the most recent errors", () => {
    for (let i = 0; i < 40; i += 1) {
      reportClientError(new Error(`err-${i}`), { source: "react-render" });
    }
    const log = getClientErrorLog();
    expect(log.length).toBeLessThanOrEqual(25);
    expect(log[log.length - 1].message).toBe("err-39");
  });
});

describe("reloadOnceForChunkError", () => {
  let reloadSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    window.sessionStorage.clear();
    vi.spyOn(console, "error").mockImplementation(() => {});
    reloadSpy = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadSpy },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reloads once, then suppresses repeats within the cooldown", () => {
    expect(reloadOnceForChunkError("Markdown")).toBe(true);
    expect(reloadSpy).toHaveBeenCalledTimes(1);
    // Second call inside the cooldown must not loop.
    expect(reloadOnceForChunkError("Markdown")).toBe(false);
    expect(reloadSpy).toHaveBeenCalledTimes(1);
  });
});
