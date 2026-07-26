import { describe, expect, it } from "vitest";

import { collectFindRanges, computeTranscriptFindMatches } from "./transcript-find";

describe("computeTranscriptFindMatches", () => {
  const messages = [
    { id: "m1", content: "The Hungarian algorithm solves assignment." },
    { id: "m2", content: "Assignment costs: the assignment matrix has zeros." },
    { id: "m3", content: "Unrelated." },
  ];

  it("finds matches across every message, mounted or not", () => {
    const matches = computeTranscriptFindMatches(messages, "assignment");
    expect(matches).toEqual([
      { messageId: "m1", messageIndex: 0, occurrence: 0 },
      { messageId: "m2", messageIndex: 1, occurrence: 0 },
      { messageId: "m2", messageIndex: 1, occurrence: 1 },
    ]);
  });

  it("is case-insensitive in both directions", () => {
    expect(computeTranscriptFindMatches(messages, "HUNGARIAN")).toHaveLength(1);
    expect(computeTranscriptFindMatches([{ id: "m", content: "ALL CAPS" }], "caps")).toHaveLength(1);
  });

  it("returns nothing for empty or whitespace queries", () => {
    expect(computeTranscriptFindMatches(messages, "")).toEqual([]);
    expect(computeTranscriptFindMatches(messages, "   ")).toEqual([]);
  });

  it("counts non-overlapping occurrences, matching browser find", () => {
    expect(computeTranscriptFindMatches([{ id: "m", content: "aaa" }], "aa")).toHaveLength(1);
  });

  it("finds text inside code fences — source is searched, not rendering", () => {
    const withCode = [{ id: "m", content: "Run this:\n```bash\nultra-control migrate\n```" }];
    expect(computeTranscriptFindMatches(withCode, "ultra-control")).toHaveLength(1);
  });
});

describe("collectFindRanges", () => {
  const mount = (html: string): HTMLElement => {
    const root = document.createElement("div");
    root.innerHTML = html;
    document.body.appendChild(root);
    return root;
  };

  it("finds a match confined to one text node", () => {
    const root = mount("<p>plain sentence here</p>");
    const ranges = collectFindRanges(root, "sentence");
    expect(ranges).toHaveLength(1);
    expect(ranges[0].toString()).toBe("sentence");
    root.remove();
  });

  it("finds a match SPANNING element boundaries — the markdown case", () => {
    // Rendered `**bo**ld` is two text nodes; a per-node search would miss it.
    const root = mount("<p><b>bo</b>ld move</p>");
    const ranges = collectFindRanges(root, "bold");
    expect(ranges).toHaveLength(1);
    expect(ranges[0].toString()).toBe("bold");
    root.remove();
  });

  it("finds every occurrence, case-insensitively, across siblings", () => {
    const root = mount("<p>Alpha beta</p><p>ALPHA gamma alpha</p>");
    const ranges = collectFindRanges(root, "alpha");
    expect(ranges).toHaveLength(3);
    expect(ranges.map((r) => r.toString().toLowerCase())).toEqual(["alpha", "alpha", "alpha"]);
    root.remove();
  });

  it("handles a match ending exactly at a node boundary", () => {
    const root = mount("<p><em>end</em>start</p>");
    const ranges = collectFindRanges(root, "end");
    expect(ranges).toHaveLength(1);
    expect(ranges[0].toString()).toBe("end");
    root.remove();
  });

  it("returns nothing for empty roots and empty queries", () => {
    const root = mount("");
    expect(collectFindRanges(root, "anything")).toEqual([]);
    expect(collectFindRanges(mount("<p>text</p>"), " ")).toEqual([]);
    root.remove();
  });
});
