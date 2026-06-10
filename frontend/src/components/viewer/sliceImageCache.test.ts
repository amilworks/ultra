import * as THREE from "three";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearSliceCache,
  loadSliceBitmap,
  prefetchSliceBitmaps,
  sliceBitmapToTexture,
  sliceCacheSize,
} from "./sliceImageCache";

const makeBitmap = (): ImageBitmap => ({ width: 4, height: 4, close: vi.fn() }) as unknown as ImageBitmap;
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("sliceImageCache", () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  let createBitmapMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    clearSliceCache();
    fetchMock = vi.fn(async () => ({ ok: true, blob: async () => new Blob() }) as unknown as Response);
    createBitmapMock = vi.fn(async () => makeBitmap());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("createImageBitmap", createBitmapMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    clearSliceCache();
  });

  it("decodes a slice once and serves repeat visits from cache", async () => {
    const url = "/v2/uploads/file/slice?z=1";
    const first = await loadSliceBitmap(url);
    const second = await loadSliceBitmap(url);

    expect(second).toBe(first);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(createBitmapMock).toHaveBeenCalledTimes(1);
  });

  it("deduplicates concurrent in-flight requests for the same slice", async () => {
    const url = "/v2/uploads/file/slice?z=2";
    const [a, b] = await Promise.all([loadSliceBitmap(url), loadSliceBitmap(url)]);

    expect(a).toBe(b);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("warms uncached neighbours via prefetch but skips cached ones", async () => {
    await loadSliceBitmap("/slice/a");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    prefetchSliceBitmaps(["/slice/a", "/slice/b", "/slice/c"]);
    await flush();

    // "/slice/a" was already cached, so only b and c are fetched.
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(sliceCacheSize()).toBe(3);
  });

  it("rejects but does not crash when a slice request fails", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 404 } as unknown as Response);
    await expect(loadSliceBitmap("/slice/missing")).rejects.toThrow(/404/);
    // A failed load must not poison the cache.
    expect(sliceCacheSize()).toBe(0);
  });

  it("bounds the cache with LRU eviction", async () => {
    for (let index = 0; index < 90; index += 1) {
      await loadSliceBitmap(`/slice/${index}`);
    }
    expect(sliceCacheSize()).toBeLessThanOrEqual(80);
  });

  it("builds an sRGB, mip-free texture from a decoded slice", () => {
    const texture = sliceBitmapToTexture(makeBitmap());

    expect(texture.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(texture.minFilter).toBe(THREE.LinearFilter);
    expect(texture.generateMipmaps).toBe(false);
    expect(texture.image).toBeTruthy();
  });
});
