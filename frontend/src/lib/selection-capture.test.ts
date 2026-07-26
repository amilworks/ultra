import { afterEach, describe, expect, it } from "vitest";

import { textFromSelection } from "./selection-capture";

/**
 * Minimal faithful KaTeX DOM: the real render is
 *   <span class="katex">
 *     <span class="katex-mathml"><math>…<annotation encoding="application/x-tex">TEX</annotation>…</math></span>
 *     <span class="katex-html">glyph soup</span>
 *   </span>
 * with display math wrapped in <span class="katex-display">.
 */
const katexMarkup = (tex: string, glyphs: string, display = false): string => {
  const inner =
    `<span class="katex">` +
    `<span class="katex-mathml"><math><semantics><mrow></mrow>` +
    `<annotation encoding="application/x-tex">${tex}</annotation>` +
    `</semantics></math></span>` +
    `<span class="katex-html" aria-hidden="true">${glyphs}</span>` +
    `</span>`;
  return display ? `<span class="katex-display">${inner}</span>` : inner;
};

const selectAll = (html: string): Selection => {
  const host = document.createElement("div");
  host.id = "capture-host";
  host.innerHTML = html;
  document.body.appendChild(host);
  const range = document.createRange();
  range.selectNodeContents(host);
  const selection = window.getSelection();
  if (!selection) {
    throw new Error("jsdom selection unavailable");
  }
  selection.removeAllRanges();
  selection.addRange(range);
  return selection;
};

afterEach(() => {
  document.getElementById("capture-host")?.remove();
  window.getSelection()?.removeAllRanges();
});

describe("textFromSelection", () => {
  it("passes plain prose through untouched", () => {
    const selection = selectAll("<p>The algorithm runs in cubic time.</p>");
    expect(textFromSelection(selection)).toBe("The algorithm runs in cubic time.");
  });

  it("swaps an inline formula for its TeX source, dollar-fenced", () => {
    const selection = selectAll(
      `<p>The algorithm runs in ${katexMarkup("O(n^3)", "O(n3)")} time.</p>`
    );
    expect(textFromSelection(selection)).toBe(
      "The algorithm runs in $O(n^3)$ time."
    );
  });

  it("fences display math with double dollars", () => {
    const selection = selectAll(
      `<p>Recall:</p>${katexMarkup("Q^{\\pi}_g(s,a)", "Qgπ(s,a)", true)}<p>as defined.</p>`
    );
    // Blank lines around display math: proper markdown block separation.
    expect(textFromSelection(selection)).toBe(
      "Recall:\n\n$$Q^{\\pi}_g(s,a)$$\n\nas defined."
    );
  });

  it("never doubles a formula that lacks its annotation", () => {
    // A partially selected formula clones without the TeX source; the MathML
    // half must be dropped so the visible glyphs appear exactly once.
    const noAnnotation =
      `<span class="katex">` +
      `<span class="katex-mathml"><math><mrow>O(n3)</mrow></math></span>` +
      `<span class="katex-html">O(n3)</span>` +
      `</span>`;
    const selection = selectAll(`<p>runs in ${noAnnotation} time</p>`);
    expect(textFromSelection(selection)).toBe("runs in O(n3) time");
  });

  it("separates paragraphs with a blank line — faithful markdown", () => {
    const selection = selectAll("<p>First point.</p><p>Second point.</p>");
    expect(textFromSelection(selection)).toBe("First point.\n\nSecond point.");
  });

  it("honours explicit line breaks", () => {
    const selection = selectAll("<p>above<br>below</p>");
    expect(textFromSelection(selection)).toBe("above\nbelow");
  });

  it("preserves indentation inside code blocks and re-fences them", () => {
    // The whitespace normalizer dedented PRE content — quoted Python arrived
    // syntactically broken, worse than native toString.
    const selection = selectAll(
      "<p>Try:</p><pre><code>def fit(x):\n    if x:\n        return 1</code></pre>"
    );
    // The blank line before the fence is proper markdown block separation.
    expect(textFromSelection(selection)).toBe(
      "Try:\n\n```\ndef fit(x):\n    if x:\n        return 1\n```"
    );
  });

  it("separates table cells instead of gluing data together", () => {
    const selection = selectAll(
      "<table><tbody><tr><td>SRR300D</td><td>≈2 nm</td><td>High ledge density</td></tr>" +
      "<tr><td>SRR300E</td><td>≈4 nm</td><td>Low</td></tr></tbody></table>"
    );
    expect(textFromSelection(selection)).toBe(
      "SRR300D | ≈2 nm | High ledge density\nSRR300E | ≈4 nm | Low"
    );
  });

  it("gives consecutive display equations their own lines", () => {
    const selection = selectAll(
      katexMarkup("a=1", "a=1", true) + katexMarkup("b=2", "b=2", true)
    );
    expect(textFromSelection(selection)).toBe("$$a=1$$\n\n$$b=2$$");
  });

  it("restores backticks around inline code, making its dollars inert", () => {
    const selection = selectAll("<p>run <code>fit($x)</code> now</p>");
    expect(textFromSelection(selection)).toBe("run `fit($x)` now");
  });

  it("captures every range of a multi-range selection", () => {
    // Firefox produces these for Ctrl+select and table columns.
    const host = document.createElement("div");
    host.id = "capture-host";
    host.innerHTML = "<p id='a'>alpha text</p><p id='b'>beta text</p>";
    document.body.appendChild(host);
    const r1 = document.createRange();
    r1.selectNodeContents(host.querySelector("#a") as Element);
    const r2 = document.createRange();
    r2.selectNodeContents(host.querySelector("#b") as Element);
    const fakeSelection = {
      rangeCount: 2,
      isCollapsed: false,
      getRangeAt: (index: number) => (index === 0 ? r1 : r2),
    };
    expect(textFromSelection(fakeSelection as unknown as Selection)).toBe("alpha text\nbeta text");
  });

  it("returns empty for a collapsed selection", () => {
    const host = document.createElement("div");
    host.id = "capture-host";
    document.body.appendChild(host);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    expect(textFromSelection(selection as Selection)).toBe("");
  });
});
