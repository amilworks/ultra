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

  it("does NOT fence an environment inside an open inline `$…$` span", () => {
    // The live failure: a smallmatrix inside a single-dollar span. Fencing it
    // severed the span, desynced every later `$` in the paragraph, and pushed
    // prose into math mode ("has a" rendered as "hasa"). The span must stay
    // intact so remark-math renders the whole thing as one inline formula.
    const source =
      "with $t = e_1$, $E_s = \\tfrac12\\left[\\begin{smallmatrix} 0 & s \\\\ s & 0 \\end{smallmatrix}\\right]$ has a $-s^2$ leading minor.";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("still fences a bare environment in a LATER paragraph after currency", () => {
    // A stray unpaired dollar two paragraphs up must not poison fencing:
    // inline spans cannot cross blank lines, so parity is per-paragraph.
    const source =
      "It costs $5 today.\n\nThe system is \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix} as shown.";
    const out = normalizeMathMarkdown(source);
    expect(out).toMatch(/\$\$[\s\S]*\\begin\{bmatrix\}[\s\S]*\$\$/);
  });

  it("ignores dollars inside code when judging span parity", () => {
    // `echo $PATH` contributes one dollar of noise; without masking it would
    // flip parity and wrongly suppress the fence.
    const source =
      "Run `echo $PATH` first. Then \\begin{bmatrix} x \\end{bmatrix} holds.";
    const out = normalizeMathMarkdown(source);
    expect(out).toMatch(/\$\$[\s\S]*\\begin\{bmatrix\}[\s\S]*\$\$/);
  });

  it("never rewrites math-like text inside fenced code blocks", () => {
    const source = [
      "Example LaTeX source:",
      "",
      "```latex",
      "\\[ x^2 \\]",
      "$E = mc^2$",
      "\\begin{bmatrix} 1 \\end{bmatrix}",
      "```",
      "",
      "And that is the syntax.",
    ].join("\n");
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("never rewrites math-like text inside inline code spans", () => {
    const source = "Write `\\( y \\)` or `\\[ z \\]` to open math mode.";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });
});

describe("normalizeMathMarkdown — lone-formula display promotion", () => {
  it("promotes a paragraph that is exactly one inline formula", () => {
    const source = "so the reflection is\n\n$\\rho(X) = QX + q$\n\nas claimed.";
    const out = normalizeMathMarkdown(source);
    expect(out).toBe(
      "so the reflection is\n\n$$\n\\rho(X) = QX + q\n$$\n\nas claimed."
    );
  });

  it("moves trailing sentence punctuation inside the promoted display", () => {
    const out = normalizeMathMarkdown("$C_2 = QC + q$,");
    expect(out).toBe("$$\nC_2 = QC + q,\n$$");
  });

  it("promotes a single-line $$…$$ paragraph (remark-math parses it inline)", () => {
    const out = normalizeMathMarkdown("$$E = mc^2$$");
    expect(out).toBe("$$\nE = mc^2\n$$");
  });

  it("leaves a multi-line $$ flow block byte-identical", () => {
    const source = "$$\nE = mc^2\n$$";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("does not promote a formula woven into a sentence", () => {
    const source = "The energy $E = mc^2$ is invariant.";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("does not promote a paragraph holding two separate spans", () => {
    const source = "$a = 1$ $b = 2$";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("does not promote list items, blockquotes, or headings", () => {
    const source = "- $x = 1$\n\n> $y = 2$\n\n## $z = 3$";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("does not promote inside indented code blocks", () => {
    const source = "Literally type:\n\n    $x = 1$\n\ndone.";
    expect(normalizeMathMarkdown(source)).toBe(source);
  });

  it("promotes a lone \\( … \\) paragraph through the delimiter conversion", () => {
    const out = normalizeMathMarkdown("\\( E = mc^2 \\).");
    expect(out).toBe("$$\nE = mc^2.\n$$");
  });

  it("leaves lone currency amounts alone", () => {
    // One dollar sign — no closing delimiter, so no span, so no promotion.
    expect(normalizeMathMarkdown("$5.")).toBe("$5.");
  });
});

describe("normalizeMathMarkdown — punctuation folding on fenced environments", () => {
  it("folds a trailing period inside the fenced display", () => {
    const out = normalizeMathMarkdown(
      "The vector is \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}."
    );
    // Punctuation ends up inside the `$$` block, and no orphan "." paragraph
    // remains after the closing fence.
    expect(out).toMatch(/\\end\{bmatrix\}\.\n\$\$/);
    expect(out).not.toMatch(/\$\$\n\n+\s*\.\s*$/);
  });

  it("folds a comma and lets the sentence resume as prose", () => {
    const out = normalizeMathMarkdown(
      "Given \\begin{bmatrix} 1 \\end{bmatrix}, the result follows."
    );
    expect(out).toMatch(/\\end\{bmatrix\},\n\$\$/);
    expect(out).toContain("the result follows.");
  });

  it("does not fold punctuation glued to non-space text", () => {
    // `.5` is content, not sentence punctuation — leave it outside.
    const out = normalizeMathMarkdown("\\begin{bmatrix} 1 \\end{bmatrix}.5 scale");
    expect(out).toMatch(/\\end\{bmatrix\}\n\$\$/);
    expect(out).toContain(".5 scale");
  });
});

describe("normalizeMathMarkdown — streaming tail hold", () => {
  it("holds promotion of the final paragraph while streaming", () => {
    const source = "so the reflection is\n\n$\\rho(X) = QX + q$";
    expect(normalizeMathMarkdown(source, { streamingTail: true })).toBe(source);
  });

  it("promotes the same tail once streaming ends", () => {
    const source = "so the reflection is\n\n$\\rho(X) = QX + q$";
    expect(normalizeMathMarkdown(source, { streamingTail: false })).toContain(
      "$$\n\\rho(X) = QX + q\n$$"
    );
  });

  it("still promotes an earlier paragraph mid-stream", () => {
    const source = "intro\n\n$E = mc^2$\n\nand the tail keeps stream";
    const out = normalizeMathMarkdown(source, { streamingTail: true });
    expect(out).toContain("$$\nE = mc^2\n$$");
  });

  it("promotes a tail formula already closed by a blank line", () => {
    const source = "so we get\n\n$E = mc^2$\n\n";
    const out = normalizeMathMarkdown(source, { streamingTail: true });
    expect(out).toContain("$$\nE = mc^2\n$$");
  });
});
