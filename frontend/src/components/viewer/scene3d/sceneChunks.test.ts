import { describe, expect, it } from "vitest";

import {
  CHUNK_HEADER_BYTES,
  CHUNK_VERSION,
  mergeSplatParts,
  parseChunkHeader,
  pointViews,
  selectTierForBudget,
  selectTier,
  splatViews,
  UPC1_BYTES_PER_POINT,
  UPC1_FLAG_ALPHA,
  USX1_BYTES_PER_SPLAT,
  type ChunkHeader,
} from "./sceneChunks";

type HeaderFields = {
  magic?: string;
  version?: number;
  flags?: number;
  count?: number;
  shDegree?: number;
  bboxMin?: [number, number, number];
  bboxMax?: [number, number, number];
  origin?: [number, number, number];
};

const writeHeader = (buf: ArrayBuffer, fields: HeaderFields): ArrayBuffer => {
  const magic = fields.magic ?? "USX1";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < 4; i += 1) {
    bytes[i] = magic.charCodeAt(i);
  }
  const view = new DataView(buf);
  view.setUint16(4, fields.version ?? CHUNK_VERSION, true);
  view.setUint16(6, fields.flags ?? 0, true);
  view.setUint32(8, fields.count ?? 0, true);
  view.setUint32(12, fields.shDegree ?? 0, true);
  const bboxMin = fields.bboxMin ?? [0, 0, 0];
  const bboxMax = fields.bboxMax ?? [0, 0, 0];
  const origin = fields.origin ?? [0, 0, 0];
  for (let i = 0; i < 3; i += 1) {
    view.setFloat32(16 + i * 4, bboxMin[i], true);
    view.setFloat32(28 + i * 4, bboxMax[i], true);
    view.setFloat32(40 + i * 4, origin[i], true);
  }
  return buf;
};

const splatChunk = (count: number, fields: HeaderFields = {}): ArrayBuffer => {
  const buf = new ArrayBuffer(CHUNK_HEADER_BYTES + count * USX1_BYTES_PER_SPLAT);
  writeHeader(buf, { magic: "USX1", count, ...fields });
  const words = new Uint32Array(buf, CHUNK_HEADER_BYTES, count * 8);
  for (let i = 0; i < words.length; i += 1) {
    words[i] = 0x1000_0000 + i;
  }
  return buf;
};

const pointChunk = (count: number, fields: HeaderFields = {}): ArrayBuffer => {
  const buf = new ArrayBuffer(CHUNK_HEADER_BYTES + count * UPC1_BYTES_PER_POINT);
  writeHeader(buf, { magic: "UPC1", count, ...fields });
  const positions = new Float32Array(buf, CHUNK_HEADER_BYTES, count * 3);
  for (let i = 0; i < positions.length; i += 1) {
    positions[i] = i * 0.5;
  }
  const colors = new Uint8Array(buf, CHUNK_HEADER_BYTES + count * 12, count * 4);
  for (let i = 0; i < colors.length; i += 1) {
    colors[i] = (i * 7) % 256;
  }
  return buf;
};

describe("parseChunkHeader", () => {
  it("reads every field of the 64-byte header", () => {
    const buf = splatChunk(3, {
      flags: 0,
      shDegree: 2,
      bboxMin: [-1.5, -2.5, -3.5],
      bboxMax: [1.5, 2.5, 3.5],
      origin: [100.25, -200.5, 0.75],
    });
    const header = parseChunkHeader(buf);
    expect(header).toEqual<ChunkHeader>({
      magic: "USX1",
      version: 1,
      flags: 0,
      count: 3,
      shDegree: 2,
      bboxMin: [-1.5, -2.5, -3.5],
      bboxMax: [1.5, 2.5, 3.5],
      origin: [100.25, -200.5, 0.75],
    });
  });

  it("reads the UPC1 alpha flag", () => {
    expect(parseChunkHeader(pointChunk(2, { flags: UPC1_FLAG_ALPHA })).flags & UPC1_FLAG_ALPHA)
      .toBe(UPC1_FLAG_ALPHA);
    expect(parseChunkHeader(pointChunk(2)).flags & UPC1_FLAG_ALPHA).toBe(0);
  });

  it("reports the MEASURED sh degree the derive wrote, whatever the source declared", () => {
    // The real 14.5M-splat file declares degree 3 and measures 0.
    expect(parseChunkHeader(splatChunk(1, { shDegree: 0 })).shDegree).toBe(0);
    expect(parseChunkHeader(splatChunk(1, { shDegree: 3 })).shDegree).toBe(3);
  });

  it("throws on a bad magic", () => {
    const buf = splatChunk(1, { magic: "PLY\n" });
    expect(() => parseChunkHeader(buf)).toThrow(/bad chunk magic/);
    expect(() => parseChunkHeader(splatChunk(1, { magic: "usx1" }))).toThrow(/bad chunk magic/);
  });

  it("throws on a truncated buffer", () => {
    expect(() => parseChunkHeader(new ArrayBuffer(0))).toThrow(/truncated/);
    expect(() => parseChunkHeader(new ArrayBuffer(CHUNK_HEADER_BYTES - 1))).toThrow(/truncated/);
    // Exactly the header, no payload, is a legal (empty) chunk.
    expect(parseChunkHeader(splatChunk(0)).count).toBe(0);
  });

  it("throws on an unsupported version rather than misreading a future layout", () => {
    expect(() => parseChunkHeader(splatChunk(1, { version: 2 }))).toThrow(/unsupported chunk version 2/);
    expect(() => parseChunkHeader(splatChunk(1, { version: 0 }))).toThrow(/unsupported chunk version 0/);
  });

  it("throws when handed something that is not an ArrayBuffer", () => {
    expect(() => parseChunkHeader(new Uint8Array(64) as unknown as ArrayBuffer))
      .toThrow(/must be an ArrayBuffer/);
  });
});

describe("splatViews", () => {
  it("lays extA and extB back to back after the header", () => {
    const count = 5;
    const buf = splatChunk(count);
    const { extA, extB } = splatViews(buf, parseChunkHeader(buf));

    expect(extA.byteOffset).toBe(CHUNK_HEADER_BYTES);
    expect(extA).toHaveLength(count * 4);
    expect(extB.byteOffset).toBe(CHUNK_HEADER_BYTES + count * 16);
    expect(extB).toHaveLength(count * 4);
    expect(extA[0]).toBe(0x1000_0000);
    expect(extB[0]).toBe(0x1000_0000 + count * 4);
  });

  it("is ZERO-COPY — the views alias the source buffer", () => {
    const buf = splatChunk(4);
    const { extA, extB } = splatViews(buf, parseChunkHeader(buf));
    expect(extA.buffer).toBe(buf);
    expect(extB.buffer).toBe(buf);

    // Mutating the view must be visible in the underlying bytes. If splatViews copied,
    // this read would still see the original word.
    extA[0] = 0xdead_beef;
    extB[3] = 0x0bad_f00d;
    const raw = new DataView(buf);
    expect(raw.getUint32(CHUNK_HEADER_BYTES, true)).toBe(0xdead_beef);
    expect(raw.getUint32(CHUNK_HEADER_BYTES + 4 * 16 + 3 * 4, true)).toBe(0x0bad_f00d);

    // ...and the reverse direction too.
    raw.setUint32(CHUNK_HEADER_BYTES + 4, 0x1234_5678, true);
    expect(extA[1]).toBe(0x1234_5678);
  });

  it("handles an empty chunk", () => {
    const buf = splatChunk(0);
    const { extA, extB } = splatViews(buf, parseChunkHeader(buf));
    expect(extA).toHaveLength(0);
    expect(extB).toHaveLength(0);
  });

  it("refuses a UPC1 chunk", () => {
    const buf = pointChunk(3);
    expect(() => splatViews(buf, parseChunkHeader(buf))).toThrow(/expected a USX1 chunk, got UPC1/);
  });

  it("throws when the payload does not fit the declared count", () => {
    const buf = new ArrayBuffer(CHUNK_HEADER_BYTES);
    writeHeader(buf, { magic: "USX1", count: 10 });
    expect(() => splatViews(buf, parseChunkHeader(buf))).toThrow(/USX1 extA needs bytes/);

    // extA fits but extB runs off the end — the half-truncated download.
    const halfBuf = new ArrayBuffer(CHUNK_HEADER_BYTES + 10 * 16);
    writeHeader(halfBuf, { magic: "USX1", count: 10 });
    expect(() => splatViews(halfBuf, parseChunkHeader(halfBuf))).toThrow(/USX1 extB needs bytes/);
  });
});

describe("mergeSplatParts", () => {
  it("rebases positions into one origin while preserving every packed non-position word", () => {
    const firstBuffer = splatChunk(2, { origin: [10, 20, 30] });
    const secondBuffer = splatChunk(1, { origin: [-5, 4, 8] });
    const first = splatViews(firstBuffer, parseChunkHeader(firstBuffer));
    const second = splatViews(secondBuffer, parseChunkHeader(secondBuffer));
    const firstFloat = new Float32Array(first.extA.buffer, first.extA.byteOffset, first.extA.length);
    const secondFloat = new Float32Array(second.extA.buffer, second.extA.byteOffset, second.extA.length);
    firstFloat.set([1, 2, 3], 0);
    firstFloat.set([-1, -2, -3], 4);
    secondFloat.set([0.25, 0.5, 0.75], 0);

    const firstOpacity = first.extA[3];
    const secondOpacity = second.extA[3];
    const firstExtB = [...first.extB];
    const secondExtB = [...second.extB];
    const merged = mergeSplatParts(
      [
        { ...first, origin: [10, 20, 30] },
        { ...second, origin: [-5, 4, 8] },
      ],
      [2, 3, 4]
    );
    const xyz = new Float32Array(merged.extA.buffer);

    expect(merged.count).toBe(3);
    expect(merged.origin).toEqual([2, 3, 4]);
    expect([...xyz.slice(0, 3)]).toEqual([9, 19, 29]);
    expect([...xyz.slice(4, 7)]).toEqual([7, 15, 23]);
    expect([...xyz.slice(8, 11)]).toEqual([-6.75, 1.5, 4.75]);
    expect(merged.extA[3]).toBe(firstOpacity);
    expect(merged.extA[11]).toBe(secondOpacity);
    expect([...merged.extB]).toEqual([...firstExtB, ...secondExtB]);
  });

  it("copies an already-rebased part bit for bit when the origin is unchanged", () => {
    const buffer = splatChunk(2, { origin: [4, 5, 6] });
    const views = splatViews(buffer, parseChunkHeader(buffer));
    const merged = mergeSplatParts([{ ...views, origin: [4, 5, 6] }], [4, 5, 6]);
    expect([...merged.extA]).toEqual([...views.extA]);
    expect([...merged.extB]).toEqual([...views.extB]);
    expect(merged.extA.buffer).not.toBe(views.extA.buffer);
  });

  it("rejects malformed parts before allocating a misleading scene", () => {
    expect(() =>
      mergeSplatParts(
        [{ extA: new Uint32Array(4), extB: new Uint32Array(3), origin: [0, 0, 0] }],
        [0, 0, 0]
      )
    ).toThrow(/inconsistent/);
    expect(() => mergeSplatParts([], [Number.NaN, 0, 0])).toThrow(/finite/);
  });
});

describe("pointViews", () => {
  it("lays positions then rgba after the header", () => {
    const count = 4;
    const buf = pointChunk(count);
    const { positions, colors } = pointViews(buf, parseChunkHeader(buf));

    expect(positions.byteOffset).toBe(CHUNK_HEADER_BYTES);
    expect(positions).toHaveLength(count * 3);
    expect([...positions.slice(0, 3)]).toEqual([0, 0.5, 1]);

    expect(colors.byteOffset).toBe(CHUNK_HEADER_BYTES + count * 12);
    expect(colors).toHaveLength(count * 4);
    expect([...colors.slice(0, 4)]).toEqual([0, 7, 14, 21]);
  });

  it("is ZERO-COPY — the views alias the source buffer", () => {
    const buf = pointChunk(3);
    const { positions, colors } = pointViews(buf, parseChunkHeader(buf));
    expect(positions.buffer).toBe(buf);
    expect(colors.buffer).toBe(buf);

    positions[0] = -12.25;
    colors[0] = 200;
    const raw = new DataView(buf);
    expect(raw.getFloat32(CHUNK_HEADER_BYTES, true)).toBe(-12.25);
    expect(raw.getUint8(CHUNK_HEADER_BYTES + 3 * 12)).toBe(200);
  });

  it("refuses a USX1 chunk", () => {
    const buf = splatChunk(3);
    expect(() => pointViews(buf, parseChunkHeader(buf))).toThrow(/expected a UPC1 chunk, got USX1/);
  });

  it("throws when the colour block runs off the end", () => {
    const buf = new ArrayBuffer(CHUNK_HEADER_BYTES + 6 * 12);
    writeHeader(buf, { magic: "UPC1", count: 6 });
    expect(() => pointViews(buf, parseChunkHeader(buf))).toThrow(/UPC1 colors needs bytes/);
  });
});

describe("selectTier", () => {
  const TIERS = [[0, 1, 2], [3, 4, 5, 6], [7]];

  it("returns tier 0 alone at level 0", () => {
    expect(selectTier(TIERS, 0)).toEqual([0, 1, 2]);
  });

  it("is CUMULATIVE — level 1 keeps tier 0's spatial coverage", () => {
    expect(selectTier(TIERS, 1)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    // Returning tiers[level] alone would render a spatial subset, which contract §5
    // calls out by name as forbidden.
    expect(selectTier(TIERS, 1)).not.toEqual([3, 4, 5, 6]);
  });

  it("clamps a level past the end to everything, and a negative level to tier 0", () => {
    expect(selectTier(TIERS, 2)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(selectTier(TIERS, 99)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(selectTier(TIERS, -1)).toEqual([0, 1, 2]);
    expect(selectTier(TIERS, Number.NaN)).toEqual([0, 1, 2]);
    expect(selectTier(TIERS, 1.7)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it("de-duplicates chunks that appear in more than one tier", () => {
    expect(selectTier([[0, 1], [1, 2], [0, 3]], 2)).toEqual([0, 1, 2, 3]);
  });

  it("preserves tier order, which is also the progressive load order", () => {
    expect(selectTier([[9, 4], [1]], 1)).toEqual([9, 4, 1]);
  });

  it("handles empty and malformed tier lists", () => {
    expect(selectTier([], 0)).toEqual([]);
    expect(selectTier([[]], 0)).toEqual([]);
    expect(selectTier([[0], undefined as unknown as number[], [2]], 2)).toEqual([0, 2]);
  });
});

describe("selectTierForBudget", () => {
  const chunks = [
    { index: 0, count: 70 },
    { index: 1, count: 30 },
    { index: 2, count: 120 },
    { index: 3, count: 260 },
  ];
  const tiers = [[0, 1], [2], [3]];

  it("chooses the highest complete cumulative tier that fits", () => {
    expect(selectTierForBudget(tiers, chunks, 250)).toEqual({
      indices: [0, 1, 2],
      count: 220,
      level: 1,
    });
  });

  it("never starts a refinement tier it cannot finish", () => {
    expect(selectTierForBudget(tiers, chunks, 219)).toEqual({
      indices: [0, 1],
      count: 100,
      level: 0,
    });
  });

  it("keeps the preview tier intact even on an undersized budget", () => {
    expect(selectTierForBudget(tiers, chunks, 50)).toEqual({
      indices: [0, 1],
      count: 100,
      level: 0,
    });
  });
});
