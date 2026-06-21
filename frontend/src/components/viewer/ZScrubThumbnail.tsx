import { useCallback, useRef, useState, type MouseEvent, type WheelEvent } from "react";

import { isScrubbable, neighborSliceUrls, nextZFromWheel, zFromPointerY } from "./zScrub";

export type ZScrubThumbnailProps = {
  fileId: string;
  alt: string;
  /** The default (static) thumbnail shown when not scrubbing. */
  staticThumbnailUrl: string;
  /** Builds the image URL for a given z plane (e.g. apiClient.uploadSliceUrl). */
  sliceUrlFor: (z: number) => string;
  /** Lazily resolves the number of z planes (e.g. via getUploadViewer). */
  loadZCount: (fileId: string) => Promise<number>;
  className?: string;
  /** How many neighbor planes to prefetch on each step. */
  prefetchRadius?: number;
  /** Injectable prefetch (defaults to the shared slice cache); overridable in tests. */
  prefetch?: (urls: string[]) => void;
  /** Called if the displayed image fails to load (e.g. to mark a failed thumbnail). */
  onError?: () => void;
};

type SliceImageCacheModule = {
  prefetchSliceBitmaps: (urls: Iterable<string>) => void;
};

let sliceImageCacheModulePromise: Promise<SliceImageCacheModule> | null = null;

const loadSliceImageCacheModule = (): Promise<SliceImageCacheModule> => {
  sliceImageCacheModulePromise ??= import("./sliceImageCache").catch((error: unknown) => {
    sliceImageCacheModulePromise = null;
    throw error;
  });
  return sliceImageCacheModulePromise;
};

const prefetchSliceBitmapsLazily = (urls: string[]): void => {
  if (urls.length === 0) {
    return;
  }
  void loadSliceImageCacheModule()
    .then(({ prefetchSliceBitmaps }) => {
      prefetchSliceBitmaps(urls);
    })
    .catch(() => undefined);
};

/**
 * A thumbnail that becomes a live z-scrubber on hover: the first hover lazily
 * loads the z-plane count, then moving the cursor up/down over the thumbnail
 * scrubs through planes (top = first plane, bottom = last) — and the scroll wheel
 * steps too. Each plane is fetched from the libbioimage image service. Flat
 * (single-plane) images render as an ordinary thumbnail. The component is purely
 * additive — drop it into a card in place of a plain `<img>`.
 */
export function ZScrubThumbnail({
  fileId,
  alt,
  staticThumbnailUrl,
  sliceUrlFor,
  loadZCount,
  className,
  prefetchRadius = 2,
  prefetch = prefetchSliceBitmapsLazily,
  onError,
}: ZScrubThumbnailProps) {
  const [zCount, setZCount] = useState<number | null>(null);
  const [z, setZ] = useState(0);
  const [src, setSrc] = useState(staticThumbnailUrl);
  const loadingRef = useRef(false);

  const ensureZCount = useCallback(async () => {
    if (zCount !== null || loadingRef.current) {
      return;
    }
    loadingRef.current = true;
    try {
      setZCount(await loadZCount(fileId));
    } catch {
      setZCount(1); // treat as non-scrubbable on failure
    } finally {
      loadingRef.current = false;
    }
  }, [zCount, loadZCount, fileId]);

  const handleMouseEnter = useCallback(() => {
    void ensureZCount();
  }, [ensureZCount]);

  // Current plane mirrored in a ref so the move/wheel handlers read it without a
  // stale closure and without putting side effects inside a state updater.
  const zRef = useRef(0);

  const handleMouseLeave = useCallback(() => {
    zRef.current = 0;
    setZ(0);
    setSrc(staticThumbnailUrl);
  }, [staticThumbnailUrl]);

  const applyZ = useCallback(
    (next: number, count: number) => {
      if (next === zRef.current) {
        return;
      }
      zRef.current = next;
      setZ(next);
      setSrc(sliceUrlFor(next));
      prefetch(neighborSliceUrls(next, count, prefetchRadius, sliceUrlFor));
    },
    [sliceUrlFor, prefetch, prefetchRadius],
  );

  const handleMouseMove = useCallback(
    (event: MouseEvent<HTMLImageElement>) => {
      const count = zCount ?? 0;
      if (!isScrubbable(count)) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      applyZ(zFromPointerY(event.clientY - rect.top, rect.height, count), count);
    },
    [zCount, applyZ],
  );

  const handleWheel = useCallback(
    (event: WheelEvent<HTMLImageElement>) => {
      const count = zCount ?? 0;
      if (!isScrubbable(count)) {
        return;
      }
      event.preventDefault();
      applyZ(nextZFromWheel(zRef.current, event.deltaY, event.deltaMode, count), count);
    },
    [zCount, applyZ],
  );

  const scrubbable = zCount !== null && isScrubbable(zCount);

  return (
    <img
      className={className}
      alt={alt}
      src={src}
      draggable={false}
      onError={onError}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseMove={handleMouseMove}
      onWheel={handleWheel}
      data-zscrub={scrubbable ? "true" : "false"}
      data-z={z}
    />
  );
}
