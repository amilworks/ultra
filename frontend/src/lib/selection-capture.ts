/**
 * Turn a transcript selection into composer-ready text.
 *
 * `selection.toString()` is wrong for a scientific transcript in two ways:
 *
 * 1. KaTeX renders every formula twice — an accessibility MathML layer and the
 *    visible HTML layer — and the HTML layer's text is glyph soup ("O(n3)",
 *    spacing glue, invisible operators). But KaTeX also embeds the ORIGINAL
 *    TeX source in `<annotation encoding="application/x-tex">`, so a quoted
 *    formula can carry real, model-readable LaTeX: `$O(n^3)$`.
 * 2. toString flattens block structure inconsistently across browsers.
 *
 * So: clone the range, swap each fully-captured KaTeX render for its TeX
 * source (display math gets `$$…$$`), strip any leftover MathML (a partially
 * selected formula has no annotation in range — it degrades to the visible
 * glyphs rather than doubling), then serialize with newlines at block edges.
 */

const BLOCK_TAGS = new Set([
  "ADDRESS",
  "ARTICLE",
  "BLOCKQUOTE",
  "DIV",
  "FIGCAPTION",
  "FIGURE",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "LI",
  "OL",
  "P",
  "PRE",
  "SECTION",
  "TABLE",
  "TR",
  "UL",
]);

export const textFromSelection = (selection: Selection): string => {
  if (selection.rangeCount === 0 || selection.isCollapsed) {
    return "";
  }
  const fragment = selection.getRangeAt(0).cloneContents();

  // Full KaTeX renders → their TeX source, fenced for the composer.
  fragment.querySelectorAll(".katex").forEach((katex) => {
    const tex = katex
      .querySelector('annotation[encoding="application/x-tex"]')
      ?.textContent?.trim();
    if (!tex) {
      return;
    }
    const display =
      Boolean(katex.closest(".katex-display")) ||
      katex.parentElement?.classList.contains("katex-display");
    katex.replaceWith(
      document.createTextNode(display ? `$$${tex}$$` : `$${tex}$`)
    );
  });
  // Partially selected formulas carry no annotation; drop their MathML half so
  // the visible glyphs appear once instead of twice.
  fragment.querySelectorAll(".katex-mathml").forEach((node) => node.remove());

  const pieces: string[] = [];
  const walk = (node: Node): void => {
    if (node.nodeType === Node.TEXT_NODE) {
      pieces.push((node as Text).data);
      return;
    }
    if (!(node instanceof Element)) {
      return;
    }
    if (node.tagName === "BR") {
      pieces.push("\n");
      return;
    }
    const isBlock = BLOCK_TAGS.has(node.tagName);
    if (isBlock) {
      pieces.push("\n");
    }
    node.childNodes.forEach(walk);
    if (isBlock) {
      pieces.push("\n");
    }
  };
  fragment.childNodes.forEach(walk);

  return pieces
    .join("")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/^\n+|\n+$/g, "");
};
