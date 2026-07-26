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
    expect(textFromSelection(selection)).toBe(
      "Recall:\n$$Q^{\\pi}_g(s,a)$$\nas defined."
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

  it("returns empty for a collapsed selection", () => {
    const host = document.createElement("div");
    host.id = "capture-host";
    document.body.appendChild(host);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    expect(textFromSelection(selection as Selection)).toBe("");
  });
});
