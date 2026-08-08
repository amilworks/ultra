/**
 * `USX1` / `UPC1` chunk header parsing and zero-copy typed-array views (contract §4).
 *
 * Both wire formats are **planar**, behind a 64-byte header. 64 is chosen so every array
 * that follows starts 4- and 8-byte aligned, which is what lets the browser construct
 * `Uint32Array` / `Float32Array` views directly over the fetched `ArrayBuffer` — no copy,
 * no per-element JavaScript, no second allocation of a 100 MB chunk.
 *
 * Every view below is a window onto the caller's buffer. Mutating a view mutates the
 * fetched bytes, and that is deliberate: it is the proof the copy did not happen.
 *
 * Pure buffer arithmetic — no three.js, no Spark, no DOM.
 */

export type ChunkMagic = "USX1" | "UPC1";

export type ChunkHeader = {
  magic: ChunkMagic;
  version: number;
  flags: number;
  count: number;
  shDegree: number;
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
  origin: [number, number, number];
};

/** Fixed header size. Also the payload offset, and the reason alignment works out. */
export const CHUNK_HEADER_BYTES = 64;

/** The only wire version this build understands. */
export const CHUNK_VERSION = 1;

/** `flags` bit 0 on a `UPC1` chunk: point alpha is meaningful (otherwise it is 255). */
export const UPC1_FLAG_ALPHA = 1;

/** Bytes per splat, summed across both `ExtSplats` word arrays (16 + 16). */
export const USX1_BYTES_PER_SPLAT = 32;

/** Bytes per point: 12 for xyz f32 + 4 for rgba u8. */
export const UPC1_BYTES_PER_POINT = 16;

const MAGICS: readonly string[] = ["USX1", "UPC1"];

const readMagic = (bytes: Uint8Array): string =>
  String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);

/**
 * Parse the 64-byte header. Throws on a truncated buffer, an unknown magic, or a version
 * this build does not understand — a chunk we cannot read exactly is not a chunk we
 * render approximately.
 */
export const parseChunkHeader = (buf: ArrayBuffer): ChunkHeader => {
  if (!(buf instanceof ArrayBuffer)) {
    throw new Error("chunk must be an ArrayBuffer");
  }
  if (buf.byteLength < CHUNK_HEADER_BYTES) {
    throw new Error(
      `chunk is truncated: ${buf.byteLength} bytes, header needs ${CHUNK_HEADER_BYTES}`
    );
  }

  const magic = readMagic(new Uint8Array(buf, 0, 4));
  if (!MAGICS.includes(magic)) {
    throw new Error(`bad chunk magic ${JSON.stringify(magic)}; expected USX1 or UPC1`);
  }

  const view = new DataView(buf);
  const version = view.getUint16(4, true);
  if (version !== CHUNK_VERSION) {
    throw new Error(`unsupported chunk version ${version}; this build reads ${CHUNK_VERSION}`);
  }

  return {
    magic: magic as ChunkMagic,
    version,
    flags: view.getUint16(6, true),
    count: view.getUint32(8, true),
    shDegree: view.getUint32(12, true),
    bboxMin: [view.getFloat32(16, true), view.getFloat32(20, true), view.getFloat32(24, true)],
    bboxMax: [view.getFloat32(28, true), view.getFloat32(32, true), view.getFloat32(36, true)],
    origin: [view.getFloat32(40, true), view.getFloat32(44, true), view.getFloat32(48, true)],
  };
};

/**
 * Guard a view before constructing it. A misaligned `byteOffset` makes the typed-array
 * constructor throw a `RangeError` whose message says nothing useful; a payload that
 * overruns the buffer throws too, but only sometimes. Both get a message naming the
 * chunk field that was wrong.
 */
const assertView = (
  buf: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
  bytesPerElement: number,
  label: string
): void => {
  if (byteOffset % bytesPerElement !== 0) {
    throw new Error(
      `${label} would start at byte ${byteOffset}, not a multiple of ${bytesPerElement} — a zero-copy view is impossible`
    );
  }
  if (byteLength % bytesPerElement !== 0) {
    throw new Error(`${label} spans ${byteLength} bytes, not a multiple of ${bytesPerElement}`);
  }
  if (byteOffset + byteLength > buf.byteLength) {
    throw new Error(
      `${label} needs bytes ${byteOffset}..${byteOffset + byteLength} but the chunk holds ${buf.byteLength}`
    );
  }
};

const expectMagic = (h: ChunkHeader, magic: ChunkMagic): void => {
  if (h.magic !== magic) {
    throw new Error(`expected a ${magic} chunk, got ${h.magic}`);
  }
};

/**
 * Spark `ExtSplats` word arrays, as zero-copy views:
 *
 *   `extA` at 64,                4 u32 per splat — x, y, z (full f32 bits), half(opacity)
 *   `extB` at 64 + count*16,     4 u32 per splat — packed colour, log scales, oct quat
 *
 * Handed straight to `new ExtSplats({ extArrays: [extA, extB], numSplats })`.
 */
export const splatViews = (
  buf: ArrayBuffer,
  h: ChunkHeader
): { extA: Uint32Array; extB: Uint32Array } => {
  expectMagic(h, "USX1");
  const half = h.count * 16;
  const offsetA = CHUNK_HEADER_BYTES;
  const offsetB = CHUNK_HEADER_BYTES + half;
  assertView(buf, offsetA, half, 4, "USX1 extA");
  assertView(buf, offsetB, half, 4, "USX1 extB");
  return {
    extA: new Uint32Array(buf, offsetA, h.count * 4),
    extB: new Uint32Array(buf, offsetB, h.count * 4),
  };
};

export type SplatChunkPart = {
  extA: Uint32Array;
  extB: Uint32Array;
  /** World-space origin used by the chunk-local positions in `extA`. */
  origin: [number, number, number];
};

export type MergedSplatParts = {
  extA: Uint32Array;
  extB: Uint32Array;
  count: number;
  origin: [number, number, number];
};

/**
 * Rebase several USX1 parts into one Spark `ExtSplats` pair.
 *
 * Gaussian alpha blending is order dependent. Giving Spark one `SplatMesh` per source
 * chunk makes it sort each random whole-scene subset independently, then composite the
 * subsets in object order — an attractive but scientifically false image. One merged
 * source gives Spark one global back-to-front ordering domain.
 *
 * Only xyz changes. Opacity, colour, log-scale and quaternion words are copied bit for
 * bit. The supplied common origin should be near the robust scene centre so the rebased
 * float32 locals retain useful precision.
 */
export const mergeSplatParts = (
  parts: readonly SplatChunkPart[],
  origin: [number, number, number]
): MergedSplatParts => {
  if (!origin.every(Number.isFinite)) {
    throw new Error("merged splat origin must be finite");
  }

  let count = 0;
  for (const part of parts) {
    if (
      part.extA.length % 4 !== 0 ||
      part.extB.length !== part.extA.length ||
      !part.origin.every(Number.isFinite)
    ) {
      throw new Error("splat part arrays and origin are inconsistent");
    }
    count += part.extA.length / 4;
  }

  const extA = new Uint32Array(count * 4);
  const extB = new Uint32Array(count * 4);
  const positions = new Float32Array(extA.buffer);
  let offset = 0;
  for (const part of parts) {
    const partCount = part.extA.length / 4;
    extA.set(part.extA, offset * 4);
    extB.set(part.extB, offset * 4);

    const dx = part.origin[0] - origin[0];
    const dy = part.origin[1] - origin[1];
    const dz = part.origin[2] - origin[2];
    if (dx !== 0 || dy !== 0 || dz !== 0) {
      const local = new Float32Array(
        part.extA.buffer,
        part.extA.byteOffset,
        part.extA.length
      );
      for (let index = 0; index < partCount; index += 1) {
        const source = index * 4;
        const target = (offset + index) * 4;
        positions[target] = local[source] + dx;
        positions[target + 1] = local[source + 1] + dy;
        positions[target + 2] = local[source + 2] + dz;
        // target + 3 is packed opacity and was already copied as a uint32 word.
      }
    }
    offset += partCount;
  }

  return { extA, extB, count, origin: [...origin] as [number, number, number] };
};

/**
 * `UPC1` point arrays, as zero-copy views:
 *
 *   `positions` at 64,               3 f32 per point, chunk-local
 *   `colors`    at 64 + count*12,    4 u8 per point, **sRGB** (see `sceneColor`)
 */
export const pointViews = (
  buf: ArrayBuffer,
  h: ChunkHeader
): { positions: Float32Array; colors: Uint8Array } => {
  expectMagic(h, "UPC1");
  const positionBytes = h.count * 12;
  const colorBytes = h.count * 4;
  const offsetColors = CHUNK_HEADER_BYTES + positionBytes;
  assertView(buf, CHUNK_HEADER_BYTES, positionBytes, 4, "UPC1 positions");
  assertView(buf, offsetColors, colorBytes, 1, "UPC1 colors");
  return {
    positions: new Float32Array(buf, CHUNK_HEADER_BYTES, h.count * 3),
    colors: new Uint8Array(buf, offsetColors, colorBytes),
  };
};

/**
 * Chunk indices needed to render at detail `level` — the **cumulative union** of tiers
 * 0..level, in tier order and then first-appearance order within a tier.
 *
 * Tiers are additive by construction (contract §5): tier 0 is complete spatial coverage
 * at reduced density, and each later tier refines it. Returning `tiers[level]` alone
 * would drop the coverage the earlier tiers provide and render a spatial subset —
 * exactly the silent decimation the contract forbids.
 *
 * The order is the load order, so a progressive fetch walks it front to back. `level` is
 * clamped rather than throwing: a level past the end means "everything".
 */
export const selectTier = (tiers: number[][], level: number): number[] => {
  if (!Array.isArray(tiers) || tiers.length === 0) {
    return [];
  }
  const safeLevel = Number.isFinite(level) ? Math.floor(level) : 0;
  const last = Math.min(Math.max(safeLevel, 0), tiers.length - 1);
  const seen = new Set<number>();
  const out: number[] = [];
  for (let tier = 0; tier <= last; tier += 1) {
    const indices = tiers[tier];
    if (!Array.isArray(indices)) {
      continue;
    }
    for (const index of indices) {
      if (!seen.has(index)) {
        seen.add(index);
        out.push(index);
      }
    }
  }
  return out;
};

export type SceneChunkCount = { index: number; count: number };

export type TierSelection = {
  indices: number[];
  count: number;
  level: number;
};

/**
 * Select the densest *complete* cumulative tier that fits an element budget.
 *
 * A refinement tier is an indivisible scientific view: loading only the first chunks
 * would bias the scene toward their spatial regions. Tier 0 is always kept intact even
 * when a caller supplies a smaller nominal budget; the producer deliberately bounds
 * that whole-source preview, and breaking it would destroy its coverage guarantee.
 */
export const selectTierForBudget = (
  tiers: number[][],
  chunks: SceneChunkCount[],
  maxElements: number
): TierSelection => {
  if (!Array.isArray(tiers) || tiers.length === 0) {
    return { indices: [], count: 0, level: -1 };
  }

  const countByIndex = new Map<number, number>();
  for (const chunk of chunks) {
    if (
      Number.isInteger(chunk.index) &&
      chunk.index >= 0 &&
      Number.isFinite(chunk.count) &&
      chunk.count >= 0
    ) {
      countByIndex.set(chunk.index, Math.floor(chunk.count));
    }
  }

  const budget = Math.max(0, Math.floor(Number.isFinite(maxElements) ? maxElements : 0));
  let selected = selectTier(tiers, 0);
  const countElements = (indices: number[]): number =>
    indices.reduce((total, index) => {
      const count = countByIndex.get(index);
      if (count === undefined) {
        throw new Error(`tier references missing chunk ${index}`);
      }
      return total + count;
    }, 0);
  let selectedCount = countElements(selected);
  let selectedLevel = 0;

  for (let level = 1; level < tiers.length; level += 1) {
    const candidate = selectTier(tiers, level);
    const candidateCount = countElements(candidate);
    if (candidateCount > budget) {
      break;
    }
    selected = candidate;
    selectedCount = candidateCount;
    selectedLevel = level;
  }

  return { indices: selected, count: selectedCount, level: selectedLevel };
};
