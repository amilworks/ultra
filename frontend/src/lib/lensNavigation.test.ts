import { afterEach, describe, expect, it, vi } from "vitest";

import { getLensOpener, registerLensOpener } from "./lensNavigation";

afterEach(() => {
  registerLensOpener(null);
});

describe("lensNavigation registry", () => {
  it("starts empty", () => {
    expect(getLensOpener()).toBeNull();
  });

  it("returns the registered opener and forwards file ids", () => {
    const opener = vi.fn();
    registerLensOpener(opener);
    expect(getLensOpener()).toBe(opener);
    getLensOpener()?.(["file-1", "file-2"]);
    expect(opener).toHaveBeenCalledWith(["file-1", "file-2"]);
  });

  it("replaces a previous opener and clears on null", () => {
    const first = vi.fn();
    const second = vi.fn();
    registerLensOpener(first);
    registerLensOpener(second);
    expect(getLensOpener()).toBe(second);
    registerLensOpener(null);
    expect(getLensOpener()).toBeNull();
  });
});
