// Pure helpers for live z-scrub thumbnails: hovering a z-stack thumbnail and
// scrolling up/down moves through Z planes, each served by the libbioimage image
// service. The logic is factored here so it is unit-testable without the DOM; the
// `ZScrubThumbnail` component wires it to wheel events and the slice cache.

/** Clamp a (possibly fractional) z index into [0, zCount-1]. */
export const clampZ = (z: number, zCount: number): number => {
  if (zCount <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(zCount - 1, Math.round(z)));
};

/**
 * Translate a wheel delta into a signed number of z planes. Pixel-mode deltas
 * (deltaMode 0) advance ~1 plane per 50px (min 1 per tick); line/page deltas
 * advance one plane per tick. deltaY > 0 (scroll down) advances z.
 */
export const wheelZStep = (deltaY: number, deltaMode: number): number => {
  if (!deltaY) {
    return 0;
  }
  const direction = Math.sign(deltaY);
  if (deltaMode === 0) {
    return direction * Math.max(1, Math.round(Math.abs(deltaY) / 50));
  }
  return direction;
};

/** Next z index for a wheel event, clamped to the stack. */
export const nextZFromWheel = (
  currentZ: number,
  deltaY: number,
  deltaMode: number,
  zCount: number,
): number => clampZ(currentZ + wheelZStep(deltaY, deltaMode), zCount);

/**
 * Map a pointer's vertical offset within the thumbnail to a z plane: the top edge
 * (offsetY 0) is plane 0 and the bottom edge (offsetY = height) is the last plane.
 * This is the "hover and move the cursor up/down to scrub" interaction. Offsets
 * outside [0, height] clamp to the ends.
 */
export const zFromPointerY = (offsetY: number, height: number, zCount: number): number => {
  if (zCount <= 0 || height <= 0) {
    return 0;
  }
  const fraction = Math.max(0, Math.min(1, offsetY / height));
  const index = Math.floor(fraction * zCount);
  return Math.max(0, Math.min(zCount - 1, index));
};

/** Only multi-plane images are scrubbable. */
export const isScrubbable = (zCount: number): boolean => zCount > 1;

/**
 * Slice URLs for the neighbors of z within `radius`, clamped and de-duplicated
 * (so edges of the stack do not refetch the current plane). Used to warm the
 * cache so scrubbing stays smooth.
 */
export const neighborSliceUrls = (
  z: number,
  zCount: number,
  radius: number,
  sliceUrlFor: (z: number) => string,
): string[] => {
  const urls: string[] = [];
  const seen = new Set<number>([clampZ(z, zCount)]);
  for (let offset = -radius; offset <= radius; offset += 1) {
    if (offset === 0) {
      continue;
    }
    const neighbor = clampZ(z + offset, zCount);
    if (!seen.has(neighbor)) {
      seen.add(neighbor);
      urls.push(sliceUrlFor(neighbor));
    }
  }
  return urls;
};
