// 256-entry RGB lookup tables for the CIFTI views. The carpet z-scores per row
// and the connectivity matrix centres correlation at 0, so a perceptually even
// diverging map (RdBu_r) is the default; grayscale and viridis are offered too.

export type ColormapKey = "rdbu" | "gray" | "viridis";

const hexes = (arr: string[]): [number, number, number][] =>
  arr.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);

const buildLut = (stops: [number, number, number][]): Uint8ClampedArray => {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const lo = Math.floor(x);
    const f = x - lo;
    const a = stops[lo];
    const b = stops[Math.min(lo + 1, stops.length - 1)];
    out[i * 3] = a[0] + (b[0] - a[0]) * f;
    out[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
    out[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
  }
  return out;
};

export const COLORMAPS: Record<ColormapKey, Uint8ClampedArray> = {
  rdbu: buildLut(
    hexes([
      "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
      "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
    ])
  ),
  gray: buildLut(hexes(["#000000", "#ffffff"])),
  viridis: buildLut(
    hexes([
      "#440154", "#482878", "#3e4a89", "#31688e", "#26828e",
      "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ])
  ),
};

export const COLORMAP_LABELS: Record<ColormapKey, string> = {
  rdbu: "Diverging",
  gray: "Grayscale",
  viridis: "Viridis",
};

/** rgb() string for a colormap sample in [0, 1]. */
export const sampleColor = (map: ColormapKey, t: number): string => {
  const lut = COLORMAPS[map];
  const k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  return `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
};

/** Turn "CIFTI_STRUCTURE_CORTEX_LEFT" into "Cortex Left". */
export const prettyStructure = (name: string): string =>
  name
    .replace(/^CIFTI_STRUCTURE_/i, "")
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());

/**
 * Compact structure name for the narrow carpet gutter: hemisphere → L/R and the
 * few long anatomical terms abbreviated, so "Diencephalon Ventral Left" fits as
 * "Dienceph Vent L" without truncating. The full name still shows on hover.
 */
export const shortStructure = (name: string): string =>
  prettyStructure(name)
    .replace(/\bDiencephalon\b/g, "Dienceph")
    .replace(/\bVentral\b/g, "Vent")
    .replace(/\bLeft\b/g, "L")
    .replace(/\bRight\b/g, "R");

/** Decode a base64 payload into a Uint8Array (uint8 matrix). */
export const decodeBase64Bytes = (b64: string): Uint8Array => {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};

/** Decode a base64 little-endian float32 payload. */
export const decodeBase64Float32 = (b64: string): Float32Array => {
  const bytes = decodeBase64Bytes(b64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
};
