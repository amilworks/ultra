import { describe, expect, it } from "vitest";

import { parseBriefTokenStorage, serializeBriefTokenStorage } from "./briefTokenStorage";

describe("brief token storage", () => {
  it("round-trips a registry through its own versioned envelope", () => {
    const state = {
      "conv-1": [
        { label: "scan.tif", fileId: "f1" },
        { label: "EBSD map.h5", fileId: "f2" },
      ],
    };
    const raw = serializeBriefTokenStorage(state);
    expect(JSON.parse(raw)).toMatchObject({ version: 1 });
    expect(parseBriefTokenStorage(raw)).toEqual(state);
  });

  it("treats anything malformed as empty rather than trusting it", () => {
    expect(parseBriefTokenStorage(null)).toEqual({});
    expect(parseBriefTokenStorage("not json")).toEqual({});
    expect(parseBriefTokenStorage(JSON.stringify({ version: 99, tokens: {} }))).toEqual({});
    expect(parseBriefTokenStorage(JSON.stringify({ version: 1, tokens: [] }))).toEqual({});
  });

  it("drops entries that could map a label to the wrong file", () => {
    const raw = JSON.stringify({
      version: 1,
      tokens: {
        "conv-1": [
          { label: "a.tif", file_id: "f1" },
          { label: "a.tif", file_id: "f2" },
          { label: "", file_id: "f3" },
          { label: "b.tif", file_id: "" },
          { label: "c.tif", file_id: "f1" },
          "junk",
          { label: "x".repeat(200), file_id: "f9" },
        ],
        " ": [{ label: "orphan", file_id: "f7" }],
        "conv-2": [],
      },
    });
    expect(parseBriefTokenStorage(raw)).toEqual({ "conv-1": [{ label: "a.tif", fileId: "f1" }] });
  });

  it("omits empty conversations when writing", () => {
    const raw = serializeBriefTokenStorage({ "conv-1": [], "conv-2": [{ label: "a", fileId: "f" }] });
    expect(JSON.parse(raw).tokens).toEqual({ "conv-2": [{ label: "a", file_id: "f" }] });
  });
});
