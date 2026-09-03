import { describe, expect, it } from "vitest";

import {
  briefBackspaceTarget,
  briefCaretAfterArrow,
  briefDeleteTarget,
  briefFileTokensInText,
  briefMentionQueryAtCaret,
  briefSummary,
  insertBriefToken,
  normalizeBriefLabel,
  parseBriefSegments,
  removeBriefSegment,
  syncBriefRegistryWithText,
  uniqueBriefLabel,
} from "./brief-tokens";

const registry = [
  { label: "scan.tif", fileId: "f-scan" },
  { label: "EBSD map 3.h5", fileId: "f-ebsd" },
  { label: "scan.tif (2)", fileId: "f-scan2" },
];

describe("parseBriefSegments", () => {
  it("recognises registered labels at boundaries, longest label first", () => {
    const segments = parseBriefSegments("Compare @scan.tif (2) with @scan.tif.", registry);
    expect(segments.map((segment) => segment.kind)).toEqual([
      "text",
      "file",
      "text",
      "file",
      "text",
    ]);
    const files = segments.filter((segment) => segment.kind === "file");
    expect(files.map((segment) => segment.kind === "file" && segment.fileId)).toEqual([
      "f-scan2",
      "f-scan",
    ]);
    // Trailing sentence punctuation is prose, not part of the token.
    expect(segments[segments.length - 1]).toMatchObject({ kind: "text", text: "." });
  });

  it("allows spaces inside a label", () => {
    const segments = parseBriefSegments("Register @EBSD map 3.h5 to the cloud", registry);
    expect(segments[1]).toMatchObject({ kind: "file", fileId: "f-ebsd", start: 9, end: 23 });
  });

  it("never matches a longer word that merely starts with a label", () => {
    const segments = parseBriefSegments("@scan.tiff is not the file", registry);
    expect(segments).toHaveLength(1);
    expect(segments[0].kind).toBe("text");
  });

  it("ignores an @ glued to a word, so handles and emails stay prose", () => {
    const segments = parseBriefSegments("mail amil@scan.tif today", registry);
    expect(segments.every((segment) => segment.kind === "text")).toBe(true);
  });

  it("accepts a token wrapped in brackets or followed by a comma", () => {
    const segments = parseBriefSegments("(@scan.tif), then", registry);
    expect(segments[1]).toMatchObject({ kind: "file", fileId: "f-scan", start: 1, end: 10 });
  });

  it("treats an unregistered @ run as prose", () => {
    expect(parseBriefSegments("see @nothing here", [])).toEqual([
      { kind: "text", text: "see @nothing here", start: 0, end: 17 },
    ]);
  });
});

describe("labels", () => {
  it("normalises names into single-line labels without a nested prefix", () => {
    expect(normalizeBriefLabel("  @@ my   scan\n.tif ")).toBe("my scan .tif");
  });

  it("bounds a pathological name", () => {
    expect(normalizeBriefLabel("x".repeat(500)).length).toBe(80);
  });

  it("reuses the label a file already has and numbers a colliding newcomer", () => {
    expect(uniqueBriefLabel("scan.tif", "f-scan", registry)).toBe("scan.tif");
    expect(uniqueBriefLabel("scan.tif", "f-new", registry)).toBe("scan.tif (3)");
    expect(uniqueBriefLabel("", "f-blank", [])).toBe("file");
  });
});

describe("insertBriefToken", () => {
  it("pads the token so it lands on its own boundaries", () => {
    expect(insertBriefToken("Register", 8, 8, "scan.tif")).toEqual({
      text: "Register @scan.tif ",
      caret: 19,
    });
  });

  it("does not double spaces that already exist", () => {
    expect(insertBriefToken("in  to", 3, 3, "scan.tif")).toEqual({
      text: "in @scan.tif to",
      caret: 13,
    });
  });

  it("replaces the selection and works at the start of the text", () => {
    expect(insertBriefToken("@sca rest", 0, 4, "scan.tif")).toEqual({
      text: "@scan.tif rest",
      caret: 10,
    });
  });
});

describe("briefMentionQueryAtCaret", () => {
  it("reports the run being typed after an @ on a boundary", () => {
    expect(briefMentionQueryAtCaret("Register @sca", 13, registry)).toEqual({
      start: 9,
      query: "sca",
    });
    expect(briefMentionQueryAtCaret("@", 1, registry)).toEqual({ start: 0, query: "" });
  });

  it("allows a space inside the query so multi-word names can be found", () => {
    expect(briefMentionQueryAtCaret("in @EBSD ma", 11, registry)).toEqual({
      start: 3,
      query: "EBSD ma",
    });
  });

  it("does not open for an @ followed by a space, glued to a word, or across a line", () => {
    expect(briefMentionQueryAtCaret("@ hello", 3, registry)).toBeNull();
    expect(briefMentionQueryAtCaret("amil@x", 6, registry)).toBeNull();
    expect(briefMentionQueryAtCaret("@a\nb", 4, registry)).toBeNull();
  });

  it("does not reopen on a token that is already registered", () => {
    expect(briefMentionQueryAtCaret("see @scan.tif", 13, registry)).toBeNull();
  });

  it("gives up on a run longer than the query bound", () => {
    expect(briefMentionQueryAtCaret(`@${"a".repeat(60)}`, 61, registry)).toBeNull();
  });
});

describe("caret and deletion around tokens", () => {
  const text = "in @scan.tif to";
  const segments = parseBriefSegments(text, registry);

  it("steps over a token with the arrow keys", () => {
    expect(briefCaretAfterArrow(segments, 3, 1)).toBe(12);
    expect(briefCaretAfterArrow(segments, 12, -1)).toBe(3);
    expect(briefCaretAfterArrow(segments, 7, 1)).toBe(12);
    expect(briefCaretAfterArrow(segments, 7, -1)).toBe(3);
    expect(briefCaretAfterArrow(segments, 1, 1)).toBeNull();
  });

  it("targets a whole token for Backspace and Delete only at its edges", () => {
    expect(briefBackspaceTarget(segments, 12)?.fileId).toBe("f-scan");
    expect(briefBackspaceTarget(segments, 11)).toBeNull();
    expect(briefDeleteTarget(segments, 3)?.fileId).toBe("f-scan");
    expect(briefDeleteTarget(segments, 4)).toBeNull();
  });

  it("removes a token with one padding space, collapsing cleanly", () => {
    const target = briefBackspaceTarget(segments, 12)!;
    expect(removeBriefSegment(text, target)).toEqual({ text: "in to", caret: 3 });
    // At the end of the text the space BEFORE the token goes instead.
    const tail = "in @scan.tif";
    const tailSegments = parseBriefSegments(tail, registry);
    expect(removeBriefSegment(tail, briefBackspaceTarget(tailSegments, 12)!)).toEqual({
      text: "in",
      caret: 2,
    });
  });
});

describe("registry sync", () => {
  it("drops entries whose token was edited away, and keeps identity otherwise", () => {
    expect(syncBriefRegistryWithText("in @scan.tif to", registry)).toEqual([registry[0]]);
    const full = "@scan.tif @EBSD map 3.h5 @scan.tif (2)";
    expect(syncBriefRegistryWithText(full, registry)).toBe(registry);
    expect(syncBriefRegistryWithText("", [])).toEqual([]);
  });

  it("lists tokens present in the text once each, in order", () => {
    expect(
      briefFileTokensInText("@scan.tif and @scan.tif and @EBSD map 3.h5", registry).map(
        (token) => token.fileId
      )
    ).toEqual(["f-scan", "f-ebsd"]);
  });
});

describe("briefSummary", () => {
  it("names the count, the workflow, and the mode, and nothing when empty", () => {
    expect(briefSummary({ fileCount: 2, workflowLabel: "Image analysis", modeLabel: "Pro" })).toBe(
      "2 files · Image analysis · Pro"
    );
    expect(briefSummary({ fileCount: 1 })).toBe("1 file");
    expect(briefSummary({ fileCount: 0, workflowLabel: " ", modeLabel: null })).toBe("");
  });
});
