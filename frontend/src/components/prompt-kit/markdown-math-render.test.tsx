import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Markdown } from "./markdown";

// End-to-end proof through the real pipeline: normalizeMathMarkdown ->
// parseMarkdownIntoBlocks (marked) -> react-markdown + remark-math +
// rehype-katex (lazily imported). Asserts the two failing screenshot cases now
// render as KaTeX rather than a red error over raw-LaTeX bullets.
describe("Markdown math rendering (full pipeline)", () => {
  it("renders a multi-line $$\\boxed{...}$$ block with `- ` lines as one KaTeX node", async () => {
    const content = [
      "The Schur complement is",
      "",
      "$$",
      "\\boxed{S = A_{\\Gamma\\Gamma}",
      "- A_{\\Gamma 1}A_{11}^{-1}A_{1\\Gamma}",
      "- A_{\\Gamma 2}A_{22}^{-1}A_{2\\Gamma}}",
      "$$",
    ].join("\n");

    const { container } = render(<Markdown>{content}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".katex")).not.toBeNull(),
      { timeout: 4000 }
    );

    // No KaTeX render error (rehype-katex marks failures with .katex-error),
    // and the `- ` terms did NOT leak out as a markdown list.
    expect(container.querySelector(".katex-error")).toBeNull();
    expect(container.querySelector("li")).toBeNull();
    // The raw LaTeX must be consumed by KaTeX, not shown as visible prose. KaTeX
    // keeps the source in a hidden <annotation>, so check only the visible HTML
    // layer (.katex-html), which never contains the raw \boxed command.
    const visible = container.querySelector(".katex-html")?.textContent ?? "";
    expect(visible).not.toContain("\\boxed");
  });

  it("renders a bare \\begin{bmatrix} environment (no $$ fence from the model) as KaTeX", async () => {
    const content =
      "The right-hand side is \\begin{bmatrix} b_1 \\\\ b_2 \\\\ b_\\Gamma \\end{bmatrix}.";

    const { container } = render(<Markdown>{content}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".katex")).not.toBeNull(),
      { timeout: 4000 }
    );
    expect(container.querySelector(".katex-error")).toBeNull();
    const visible = container.querySelector(".katex-html")?.textContent ?? "";
    expect(visible).not.toContain("\\begin{bmatrix}");
  });
});
