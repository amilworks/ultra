import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Markdown } from "./markdown";

const cases: Array<{ name: string; md: string; expectTable: boolean }> = [
  {
    name: "01 well-formed (control)",
    md: "| A | B |\n|---|---|\n| 1 | 2 |",
    expectTable: true,
  },
  {
    name: "02 no outer pipes",
    md: "A | B\n---|---\n1 | 2",
    expectTable: true,
  },
  {
    name: "03 unescaped | in cell + count mismatch (SCREENSHOT) — stays raw by design (ambiguous)",
    md: "| Estimate | |Δ|/σ | Class |\n|---|---|---|\n| x | 0.84 | MARGINAL |",
    expectTable: false,
  },
  {
    name: "03b screenshot with pipes escaped (model-side fix) + wrong delimiter — repaired",
    md: "| Estimate | \\|Δ\\|/σ | Class |\n|---|---|---|---|\n| x | 0.84 | MARGINAL |",
    expectTable: true,
  },
  {
    name: "04 delimiter count != header count",
    md: "| A | B | C |\n|---|---|\n| 1 | 2 | 3 |",
    expectTable: true,
  },
  {
    name: "05 no blank line before table (paragraph above)",
    md: "Here is the summary:\n| A | B |\n|---|---|\n| 1 | 2 |",
    expectTable: true,
  },
  {
    name: "06 alignment colons",
    md: "| A | B | C |\n|:--|:-:|--:|\n| 1 | 2 | 3 |",
    expectTable: true,
  },
  {
    name: "07 table then text no blank line after",
    md: "| A | B |\n|---|---|\n| 1 | 2 |\nAnd more prose.",
    expectTable: true,
  },
  {
    name: "08 table inside blockquote",
    md: "> | A | B |\n> |---|---|\n> | 1 | 2 |",
    expectTable: true,
  },
  {
    name: "09 indented 2 spaces",
    md: "  | A | B |\n  |---|---|\n  | 1 | 2 |",
    expectTable: true,
  },
  {
    name: "10 ragged data rows",
    md: "| A | B | C |\n|---|---|---|\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |",
    expectTable: true,
  },
  {
    name: "11 inline math in cell",
    md: "| Sym | Val |\n|---|---|\n| $x^2$ | 4 |",
    expectTable: true,
  },
  {
    name: "12 bold + code in cells",
    md: "| A | B |\n|---|---|\n| **x** | `y` |",
    expectTable: true,
  },
  {
    name: "13 escaped pipes in cell",
    md: "| Expr | Val |\n|---|---|\n| \\|Δ\\|/σ | 0.84 |",
    expectTable: true,
  },
  {
    name: "14 header after heading no blank line",
    md: "### Results\n| A | B |\n|---|---|\n| 1 | 2 |",
    expectTable: true,
  },
  {
    name: "15 leading spaces before pipes (1-3)",
    md: " | A | B |\n |---|---|\n | 1 | 2 |",
    expectTable: true,
  },
  {
    name: "16 delimiter too LONG (4) vs header/data (3) — repaired",
    md: "| A | B | C |\n|---|---|---|---|\n| 1 | 2 | 3 |",
    expectTable: true,
  },
  {
    name: "17 header-only table, wrong delimiter — NOT repaired (no data row; too often a setext heading / HR / prose)",
    md: "| A | B | C |\n|---|---|",
    expectTable: false,
  },
  {
    name: "18 blockquote table with bad delimiter — left as-is (not rewritten)",
    md: "> | A | B | C |\n> |---|---|\n> | 1 | 2 | 3 |",
    // We intentionally skip blockquote-nested repair; remark-gfm's own leniency
    // decides. Assert only that it does not crash and renders *something*.
    expectTable: false,
  },
  {
    name: "19 thematic break after a pipe line — NOT a delimiter",
    md: "Options a | b are fine.\n\n---\n\nNext section.",
    expectTable: false,
  },
  {
    name: "20 inline code |---| in prose — not touched",
    md: "Write the separator as `|---|` in your table.",
    expectTable: false,
  },
];

// Alignment preservation on repair: header 3, delimiter 2 with colons.
import { parseMarkdownIntoBlocks, repairTableDelimiters } from "./markdown";
describe("repairTableDelimiters — alignment preservation", () => {
  it("keeps :-- / :-: alignment and defaults the added column", () => {
    const out = repairTableDelimiters(
      "| A | B | C |\n|:--|:-:|\n| 1 | 2 | 3 |"
    );
    expect(out.split("\n")[1]).toBe("| :-- | :-: | --- |");
  });
});

describe("code fences are never rewritten by table repair", () => {
  // Rendering a code block needs window.matchMedia (jsdom lacks it), so assert
  // protection at the block level: the bad delimiter inside the fence must be
  // returned verbatim, not rebuilt.
  it("leaves a bad pipe-table inside a code fence untouched", () => {
    const md = "```\n| A | B | C |\n|---|---|\n| 1 | 2 | 3 |\n```";
    const blocks = parseMarkdownIntoBlocks(md);
    const joined = blocks.join("\n");
    expect(joined).toContain("|---|---|");
    expect(joined).not.toContain("| --- | --- | --- |");
  });
});

// Numeric columns right-align automatically (remarkNumericColumnAlign) so
// units digits line up under tabular-nums; explicit delimiter markers and
// prose columns are untouched. Alignment reaches the DOM as an inline
// `text-align` style (react-markdown's tableCellAlignToStyle), not a class.
describe("numeric column auto-alignment", () => {
  const alignsOf = (container: HTMLElement, selector: "th" | "td") =>
    Array.from(container.querySelectorAll(selector)).map(
      (cell) => (cell as HTMLElement).style.textAlign || "left"
    );

  it("right-aligns integer and signed-delta columns, leaves label columns left", () => {
    const md =
      "| Class | Count | Change |\n|---|---|---|\n| `burrow` | **118** | +91 |\n| prairie_dog | 30 | -26 |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "th")).toEqual(["left", "right", "right"]);
    expect(alignsOf(container, "td")).toEqual([
      "left",
      "right",
      "right",
      "left",
      "right",
      "right",
    ]);
  });

  it("never overrides explicit delimiter alignment", () => {
    const md = "| A | B |\n|:--|---|\n| 1 | 2 |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "td")).toEqual(["left", "right"]);
  });

  it("keeps mixed text/number columns left-aligned", () => {
    const md = "| A | B |\n|---|---|\n| 1 | 2 |\n| note | 3 |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "td")).toEqual([
      "left",
      "right",
      "left",
      "right",
    ]);
  });

  it("treats dash/empty/n-a cells as neutral but requires one real number", () => {
    const md =
      "| A | B | C |\n|---|---|---|\n| 118 | — | 1 |\n| — | n/a | 2 |\n|  | — | 3 |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "th")).toEqual(["right", "left", "right"]);
  });

  it("accepts decimals, thousands commas, percents, scientific notation", () => {
    const md =
      "| a | b | c | d |\n|---|---|---|---|\n| 3.14 | 1,234 | 43% | 1.2e-3 |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "td")).toEqual([
      "right",
      "right",
      "right",
      "right",
    ]);
  });

  it("leaves unit-suffixed values and formulas as prose", () => {
    const md = "| lat | expr |\n|---|---|\n| 137ms | $x^2$ |";
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(alignsOf(container, "td")).toEqual(["left", "left"]);
  });
});

describe("Markdown table repair — render matrix", () => {
  for (const c of cases) {
    it(c.name, () => {
      const { container } = render(<Markdown>{c.md}</Markdown>);
      const hasTable = container.querySelector("table") !== null;
      expect(hasTable, `expected table render for: ${c.name}`).toBe(
        c.expectTable
      );
    });
  }
});
