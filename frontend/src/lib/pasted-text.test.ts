import { describe, expect, it } from "vitest";

import {
  PASTE_ATTACH_ALWAYS_CHARS,
  PASTE_ATTACH_STRUCTURED_CHARS,
  PASTE_ATTACH_STRUCTURED_LINES,
  draftWithQuotedSelection,
  pastedTextFile,
  pastedTextFileName,
  quoteForComposer,
  shouldAttachPastedText,
} from "./pasted-text";

describe("shouldAttachPastedText", () => {
  it("leaves an ordinary prompt alone", () => {
    expect(shouldAttachPastedText("what is the sample size?")).toBe(false);
  });

  it("leaves a LONG prose prompt alone — length by itself is not data", () => {
    // Someone drafted a careful prompt elsewhere and pasted it in. It is over
    // the structured-chars bar but has almost no newlines: exactly the text a
    // user wants inline and editable, not filed away as an attachment.
    const prose = ("A long, considered paragraph about methodology. ").repeat(60);
    expect(prose.length).toBeGreaterThan(PASTE_ATTACH_STRUCTURED_CHARS);
    expect(shouldAttachPastedText(prose)).toBe(false);
  });

  it("leaves a short many-lined snippet alone — lines by themselves are not data", () => {
    const snippet = Array.from({ length: 30 }, (_, i) => `l${i}`).join("\n");
    expect(snippet.length).toBeLessThan(PASTE_ATTACH_STRUCTURED_CHARS);
    expect(shouldAttachPastedText(snippet)).toBe(false);
  });

  it("converts moderately long many-lined text — the log/table/FASTA shape", () => {
    const log = Array.from(
      { length: PASTE_ATTACH_STRUCTURED_LINES + 5 },
      (_, i) =>
        `2026-07-26T08:00:${String(i).padStart(2, "0")}.123Z INFO ultra-worker-1 ` +
        `nats_worker.py:412 lease renewed run_id=run_e82bf0594d8396e4 partition=3 attempt=1`
    ).join("\n");
    expect(log.length).toBeGreaterThan(PASTE_ATTACH_STRUCTURED_CHARS);
    expect(shouldAttachPastedText(log)).toBe(true);
  });

  it("converts anything enormous regardless of shape", () => {
    expect(shouldAttachPastedText("x".repeat(PASTE_ATTACH_ALWAYS_CHARS))).toBe(true);
  });
});

describe("pastedTextFile", () => {
  it("names the file with a full timestamp so consecutive pastes never collide", () => {
    const name = pastedTextFileName(new Date(2026, 6, 26, 8, 5, 9));
    expect(name).toBe("pasted-2026-07-26-080509.txt");
  });

  it("produces a plain-text File carrying the exact pasted bytes", async () => {
    const file = pastedTextFile("line one\nline two", new Date(2026, 0, 2, 3, 4, 5));
    expect(file.type).toBe("text/plain");
    expect(file.name).toBe("pasted-2026-01-02-030405.txt");
    expect(await file.text()).toBe("line one\nline two");
  });
});

describe("quoteForComposer", () => {
  it("prefixes every line, keeping one selection one connected block", () => {
    expect(quoteForComposer("first\nsecond")).toBe("> first\n> second");
  });

  it("keeps interior blank lines quoted so the block does not split", () => {
    expect(quoteForComposer("first\n\nsecond")).toBe("> first\n>\n> second");
  });

  it("trims the selection's ragged edges", () => {
    expect(quoteForComposer("\n\n  padded  \n\n")).toBe(">   padded");
  });

  it("returns nothing for whitespace-only selections", () => {
    expect(quoteForComposer("   \n \n")).toBe("");
  });
});

describe("draftWithQuotedSelection", () => {
  it("starts an empty draft with the quote and a fresh line for the question", () => {
    expect(draftWithQuotedSelection("", "the finding")).toBe("> the finding\n\n");
  });

  it("appends below an existing draft as a separate markdown block", () => {
    expect(draftWithQuotedSelection("So far so good", "the finding")).toBe(
      "So far so good\n\n> the finding\n\n"
    );
  });

  it("leaves the draft untouched when the selection is only whitespace", () => {
    expect(draftWithQuotedSelection("keep me", "   ")).toBe("keep me");
  });
});
