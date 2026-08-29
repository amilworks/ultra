import { fireEvent, render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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

  it("keeps a smallmatrix inside an inline `$…$` span intact — prose stays prose", async () => {
    // The live desync failure: auto-fencing the environment out of the open
    // span re-paired every later dollar, so "has a" rendered inside math mode
    // with its space collapsed ("hasa") and real math fell out as raw text.
    const content =
      "with $t = e_1$, $E_s = \\left[\\begin{smallmatrix} 0 & s \\\\ s & 0 \\end{smallmatrix}\\right]$ has a $-s^2$ leading minor; at $\\theta = \\pi$, $E$ is symmetric.";

    const { container } = render(<Markdown>{content}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".katex")).not.toBeNull(),
      { timeout: 4000 }
    );
    expect(container.querySelector(".katex-error")).toBeNull();

    // Prose survives OUTSIDE math: strip every KaTeX node and the connective
    // words must remain in the paragraph text, spaces intact.
    const paragraph = container.querySelector("p");
    expect(paragraph).not.toBeNull();
    const prose = Array.from(paragraph?.childNodes ?? [])
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent)
      .join("");
    expect(prose).toContain(" has a ");
    expect(prose).toContain(" leading minor; at ");
    // And no raw LaTeX leaked into the visible prose.
    expect(prose).not.toContain("\\begin");
    expect(prose).not.toContain("$");
  });

  it("promotes a lone-formula paragraph to a true display block", async () => {
    const content = "so the reflection is\n\n$\\rho(X) = QX + q$.\n\nas claimed.";

    const { container } = render(<Markdown>{content}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".katex-display")).not.toBeNull(),
      { timeout: 4000 }
    );
    expect(container.querySelector(".katex-error")).toBeNull();
    // The surrounding prose paragraphs are untouched.
    expect(container.textContent).toContain("so the reflection is");
    expect(container.textContent).toContain("as claimed.");
  });

  it("renders a single-line $$…$$ paragraph as display math, not inline", async () => {
    const { container } = render(<Markdown>{"$$E = mc^2$$"}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".katex-display")).not.toBeNull(),
      { timeout: 4000 }
    );
    expect(container.querySelector(".katex-error")).toBeNull();
  });

  it("offers a copy-LaTeX button on display equations that copies the source", async () => {
    const writeText = vi.fn<(text: string) => Promise<void>>(() =>
      Promise.resolve()
    );
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    const { container } = render(<Markdown>{"$$E = mc^2$$"}</Markdown>);

    await waitFor(
      () => expect(container.querySelector(".pk-math-copy")).not.toBeNull(),
      { timeout: 4000 }
    );
    fireEvent.click(container.querySelector(".pk-math-copy")!);
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(String(writeText.mock.calls[0][0])).toContain("E = mc^2");
    // Feedback: the label flips while the copied state is active.
    await waitFor(() =>
      expect(
        container
          .querySelector(".pk-math-copy")
          ?.getAttribute("aria-label")
      ).toBe("Copied LaTeX source")
    );
  });

  it("does not attach the copy affordance to inline math", async () => {
    const { container } = render(
      <Markdown>{"the energy $E = mc^2$ is inline"}</Markdown>
    );
    await waitFor(
      () => expect(container.querySelector(".katex")).not.toBeNull(),
      { timeout: 4000 }
    );
    expect(container.querySelector(".pk-math-copy")).toBeNull();
  });
});
