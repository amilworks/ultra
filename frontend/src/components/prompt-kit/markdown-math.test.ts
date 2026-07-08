import { describe, expect, it } from "vitest";

import {
  normalizeMathMarkdown,
  parseMarkdownIntoBlocks,
} from "./markdown";

describe("parseMarkdownIntoBlocks — math-aware block splitting", () => {
  it("keeps a multi-line $$ display block with `- ` lines as a single block", () => {
    // This is the exact failure from the screenshot: marked.lexer used to split
    // this into a broken `$$\\boxed{...` paragraph (red KaTeX error) plus a
    // markdown list (raw-LaTeX bullets).
    const source = [
      "$$",
      "\\boxed{S = A_{\\Gamma\\Gamma}",
      "- A_{\\Gamma 1}A_{11}^{-1}A_{1\\Gamma}",
      "- A_{\\Gamma 2}A_{22}^{-1}A_{2\\Gamma}}",
      "$$",
    ].join("\n");

    const blocks = parseMarkdownIntoBlocks(source);
    const mathBlocks = blocks.filter((b) => b.includes("\\boxed"));

    expect(mathBlocks).toHaveLength(1);
    // The whole display block is intact: opening + body + closing all present,
    // with the subtraction terms still inside the same `$$ ... $$` span.
    expect(mathBlocks[0]).toContain("$$");
    expect(mathBlocks[0]).toContain("- A_{\\Gamma 1}A_{11}^{-1}A_{1\\Gamma}");
    expect(mathBlocks[0]).toContain("- A_{\\Gamma 2}A_{22}^{-1}A_{2\\Gamma}}");
    // The `- ` lines must NOT have become a markdown list token in their own block.
    const listBlocks = blocks.filter(
      (b) => /^\s*-\s/.test(b) && !b.includes("$$")
    );
    expect(listBlocks).toHaveLength(0);
  });

  it("still splits a genuine markdown list (no math present)", () => {
    const source = "Intro paragraph.\n\n- first\n- second\n- third";
    const blocks = parseMarkdownIntoBlocks(source);
    // The list stays its own block and keeps its dashes.
    const listBlock = blocks.find((b) => b.trim().startsWith("- first"));
    expect(listBlock).toBeDefined();
    expect(listBlock).toContain("- second");
  });

  it("leaves the private-use sentinels out of the returned blocks", () => {
    const source = "before\n\n$$\na - b\n$$\n\nafter";
    const blocks = parseMarkdownIntoBlocks(source);
    const joined = blocks.join("");
    expect(joined).not.toContain(String.fromCodePoint(0xe000));
    expect(joined).not.toContain(String.fromCodePoint(0xe001));
    expect(joined).toContain("$$\na - b\n$$");
  });
});

describe("normalizeMathMarkdown — bare environment fencing", () => {
  it("wraps a bare \\begin{bmatrix} environment in $$", () => {
    const source =
      "The vector is \\begin{bmatrix} b_1 \\\\ b_2 \\\\ b_\\Gamma \\end{bmatrix}.";
    const out = normalizeMathMarkdown(source);
    expect(out).toContain("$$");
    expect(out).toMatch(/\$\$[\s\S]*\\begin\{bmatrix\}[\s\S]*\\end\{bmatrix\}[\s\S]*\$\$/);
  });

  it("wraps a bare \\begin{aligned} environment", () => {
    const source = "\\begin{aligned} x &= 1 \\\\ y &= 2 \\end{aligned}";
    const out = normalizeMathMarkdown(source);
    expect(out).toMatch(/\$\$[\s\S]*\\begin\{aligned\}/);
  });

  it("does NOT double-wrap an already-fenced environment", () => {
    const source = "$$\n\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}\n$$";
    const out = normalizeMathMarkdown(source);
    // Exactly one opening + one closing `$$` — no nested wrapping.
    const fenceCount = (out.match(/\$\$/g) ?? []).length;
    expect(fenceCount).toBe(2);
  });

  it("still converts \\[ ... \\] and \\( ... \\) delimiters", () => {
    expect(normalizeMathMarkdown("\\[ x^2 \\]")).toContain("$$");
    expect(normalizeMathMarkdown("value \\( y \\) here")).toContain("$y$");
  });

  it("leaves ordinary prose and currency untouched", () => {
    const source = "It costs $5 and $10, and array indexing a[i] is fine.";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });
});
