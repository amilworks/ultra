import * as THREE from "three";

import type { ScalarVolumePayload } from "@/lib/api";

import { extractScalarSlice, type ScalarSliceAxis } from "./scalarSlice";

/**
 * Shared, bounded cache of decoded slice images.
 *
 * Moving through a stack used to re-fetch and re-decode a PNG from the backend
 * for every slice (~150ms each, with no reuse even when scrubbing back over a
 * visited slice). This cache keeps recently decoded `ImageBitmap`s keyed by their
 * full slice URL and lets the viewer prefetch neighbours, so revisited and
 * look-ahead slices render with no network round-trip.
 *
 * The URL already encodes everything that changes a slice's pixels (axis, index,
 * window/level, channels, cache key), so it is a complete cache key.
 */

const MAX_ENTRIES = 80;

// Insertion order doubles as recency (Map preserves it), so the first key is the
// least-recently-used entry.
const bitmapCache = new Map<string, ImageBitmap>();
const inflight = new Map<string, Promise<ImageBitmap>>();

const promote = (url: string, bitmap: ImageBitmap): void => {
  bitmapCache.delete(url);
  bitmapCache.set(url, bitmap);
  while (bitmapCache.size > MAX_ENTRIES) {
    const oldest = bitmapCache.keys().next().value as string | undefined;
    if (oldest === undefined) {
      break;
    }
    // Drop the reference and let GC reclaim it. We deliberately do not call
    // bitmap.close(): a THREE.Texture built from the bitmap still references it,
    // and closing underneath a not-yet-uploaded texture would blank the slice.
    bitmapCache.delete(oldest);
  }
};

export const sliceCacheSize = (): number => bitmapCache.size;

export const clearSliceCache = (): void => {
  bitmapCache.clear();
  inflight.clear();
};

/**
 * onProgress maps to REAL load phases so a slow open looks like it is working, not
 * broken: `null` while the backend is still decoding (request sent, no bytes yet —
 * the viewer shows an indeterminate bar), then 0..1 as the plane's bytes stream in,
 * then 1 when the response is fully received.
 */
export type SliceLoadProgress = (fraction: number | null) => void;

export const loadSliceBitmap = (url: string, onProgress?: SliceLoadProgress): Promise<ImageBitmap> => {
  const cached = bitmapCache.get(url);
  if (cached) {
    promote(url, cached);
    onProgress?.(1);
    return Promise.resolve(cached);
  }
  const pending = inflight.get(url);
  if (pending) {
    return pending;
  }
  const request = (async () => {
    // Bound the request so a hung image-service read fails the z-scrub frame instead of
    // leaving the slider spinning forever; the viewer surfaces the rejection as an error.
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 45000);
    let response: Response;
    onProgress?.(null); // request in flight, backend decoding: indeterminate
    try {
      response = await fetch(url, { credentials: "include", signal: controller.signal });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw Object.assign(new Error("Slice request timed out"), { cause: error });
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      throw new Error(`Slice request failed: ${response.status}`);
    }
    let blob: Blob;
    const headers = response.headers;
    const total = Number(headers?.get("content-length") || 0);
    if (response.body && total > 0) {
      // Stream the body so the bar reflects the real bytes received for this plane.
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.length;
          onProgress?.(Math.min(0.99, received / total));
        }
      }
      blob = new Blob(chunks as BlobPart[], { type: headers?.get("content-type") || "image/png" });
    } else {
      blob = await response.blob();
    }
    onProgress?.(1);
    // WebGL ignores texture.flipY for ImageBitmap sources, so bake the vertical
    // flip in here — otherwise slices render upside-down vs the old TextureLoader
    // (HTMLImage) path, which had flipY honoured.
    const bitmap = await createImageBitmap(blob, { imageOrientation: "flipY" });
    promote(url, bitmap);
    return bitmap;
  })();
  inflight.set(url, request);
  void request.then(
    () => inflight.delete(url),
    () => inflight.delete(url)
  );
  return request;
};

/**
 * Warm the cache for slices the user is likely to reach next. Already-cached or
 * in-flight URLs are skipped, and failures are swallowed (a prefetch that loses a
 * race with navigation must never surface an error).
 */
export const prefetchSliceBitmaps = (urls: Iterable<string>): void => {
  for (const url of urls) {
    if (!url || bitmapCache.has(url) || inflight.has(url)) {
      continue;
    }
    void loadSliceBitmap(url).catch(() => {});
  }
};

export type ScalarSliceSource = {
  /** Stable identity of the slice's pixels: file + axis + index + window + invert. */
  cacheKey: string;
  payload: ScalarVolumePayload;
  axis: ScalarSliceAxis;
  sliceIndex: number;
  windowLow: number;
  windowHigh: number;
  invert: boolean;
};

/**
 * Build a {@link ScalarSliceSource} with a canonical cache key. The key captures
 * everything that changes the slice's pixels so identical requests share a cached
 * bitmap and any change (slice index, window, invert) misses cleanly.
 */
export const buildScalarSliceSource = (params: {
  fileId: string;
  payload: ScalarVolumePayload;
  axis: ScalarSliceAxis;
  sliceIndex: number;
  windowLow: number;
  windowHigh: number;
  invert: boolean;
}): ScalarSliceSource => ({
  cacheKey: [
    "scalar",
    params.fileId,
    params.payload.channel ?? 0,
    params.axis,
    params.sliceIndex,
    params.windowLow.toFixed(3),
    params.windowHigh.toFixed(3),
    params.invert ? 1 : 0,
  ].join(":"),
  payload: params.payload,
  axis: params.axis,
  sliceIndex: params.sliceIndex,
  windowLow: params.windowLow,
  windowHigh: params.windowHigh,
  invert: params.invert,
});

/**
 * Render a slice directly from the loaded volume (no network) and cache the
 * resulting bitmap alongside the network slices. The extraction matches the
 * backend PNG pixel-for-pixel, so the result is interchangeable with a fetched
 * slice — same display, same crosshair/measurement alignment.
 */
export const loadScalarSliceBitmap = (source: ScalarSliceSource): Promise<ImageBitmap> => {
  const cached = bitmapCache.get(source.cacheKey);
  if (cached) {
    promote(source.cacheKey, cached);
    return Promise.resolve(cached);
  }
  const pending = inflight.get(source.cacheKey);
  if (pending) {
    return pending;
  }
  const request = (async () => {
    const image = extractScalarSlice(source.payload, {
      axis: source.axis,
      sliceIndex: source.sliceIndex,
      windowLow: source.windowLow,
      windowHigh: source.windowHigh,
      invert: source.invert,
    });
    const imageData = new ImageData(image.width, image.height);
    imageData.data.set(image.data);
    // Match the network-slice path: bake the vertical flip in (WebGL ignores
    // texture.flipY for ImageBitmap), so row 0 of the extracted slice renders
    // at the top exactly like the backend PNG.
    const bitmap = await createImageBitmap(imageData, { imageOrientation: "flipY" });
    promote(source.cacheKey, bitmap);
    return bitmap;
  })();
  inflight.set(source.cacheKey, request);
  void request.then(
    () => inflight.delete(source.cacheKey),
    () => inflight.delete(source.cacheKey)
  );
  return request;
};

export const prefetchScalarSliceBitmaps = (sources: Iterable<ScalarSliceSource>): void => {
  for (const source of sources) {
    if (!source || bitmapCache.has(source.cacheKey) || inflight.has(source.cacheKey)) {
      continue;
    }
    void loadScalarSliceBitmap(source).catch(() => {});
  }
};

export const sliceBitmapToTexture = (bitmap: ImageBitmap): THREE.Texture => {
  const texture = new THREE.Texture(bitmap);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  // The vertical flip is already baked into the bitmap (createImageBitmap with
  // imageOrientation: "flipY"); don't let the GPU flip it a second time.
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
};
