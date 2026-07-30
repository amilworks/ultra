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
  "UL",
]);

type Piece = { text: string; verbatim: boolean };

const serializeFragment = (fragment: DocumentFragment): Piece[] => {
  // Full KaTeX renders → their TeX source, fenced for the composer. Display
  // math gets its own line: two adjacent display equations otherwise glue into
  // "$$a$$$$b$$" because .katex-display is a SPAN with no whitespace between.
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
      document.createTextNode(display ? `\n$$${tex}$$\n` : `$${tex}$`)
    );
  });
  // Partially selected formulas carry no annotation; drop their MathML half so
  // the visible glyphs appear once instead of twice.
  fragment.querySelectorAll(".katex-mathml").forEach((node) => node.remove());

  const pieces: Piece[] = [];
  let preDepth = 0;
  const push = (text: string, verbatim = false): void => {
    pieces.push({ text, verbatim });
  };
  const walk = (node: Node): void => {
    if (node.nodeType === Node.TEXT_NODE) {
      // Inside PRE the whitespace IS the content — indentation-mangled Python
      // presented as a faithful quote is worse than no quote.
      push((node as Text).data, preDepth > 0);
      return;
    }
    if (!(node instanceof Element)) {
      return;
    }
    if (node.tagName === "BR") {
      push("\n", preDepth > 0);
      return;
    }
    if (node.tagName === "PRE") {
      // Re-fence so the quote survives as code downstream.
      push("\n", false);
      push("```\n", true);
      preDepth += 1;
      node.childNodes.forEach(walk);
      preDepth -= 1;
      push("\n```", true);
      push("\n", false);
      return;
    }
    if (node.tagName === "CODE" && preDepth === 0) {
      // Backticks make inline code inert to any math parser downstream — a
      // literal $ inside code can never read as a TeX fence.
      push("`", true);
      node.childNodes.forEach(walk);
      push("`", true);
      return;
    }
    if (node.tagName === "TR") {
      // Newline AFTER only: a block treatment's leading newline would put a
      // blank line between rows, which breaks a markdown table.
      node.childNodes.forEach(walk);
      push("\n", false);
      return;
    }
    if (node.tagName === "TD" || node.tagName === "TH") {
      node.childNodes.forEach(walk);
      // Cells glued without separators mutilate data rows; the trailing
      // separator before a row break is cleaned up in normalization.
      push(" | ", false);
      return;
    }
    const isBlock = BLOCK_TAGS.has(node.tagName);
    if (isBlock) {
      push("\n", preDepth > 0);
    }
    node.childNodes.forEach(walk);
    if (isBlock) {
      push("\n", preDepth > 0);
    }
  };
  fragment.childNodes.forEach(walk);
  return pieces;
};

/** Normalize ONLY outside verbatim (PRE/fence) spans, where whitespace is content. */
const assemble = (pieces: Piece[]): string => {
  const out: string[] = [];
  let run = "";
  const flushRun = (): void => {
    if (!run) {
      return;
    }
    out.push(
      run
        .replace(/[ \t]*\n[ \t]*/g, "\n")
        // Trailing cell separator at a row break; runs AFTER the normalizer,
        // which has already eaten the separator's trailing space.
        .replace(/ \|\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
    );
    run = "";
  };
  for (const piece of pieces) {
    if (piece.verbatim) {
      flushRun();
      out.push(piece.text);
    } else {
      run += piece.text;
    }
  }
  flushRun();
  return out.join("").replace(/^\s*\n/, "").replace(/\n\s*$/, "");
};

export const textFromSelection = (selection: Selection): string => {
  if (selection.rangeCount === 0 || selection.isCollapsed) {
    return "";
  }
  // Firefox produces multi-range selections (Ctrl+select, table columns);
  // reading only range 0 would silently drop regions the user sees selected.
  const captures: string[] = [];
  for (let index = 0; index < selection.rangeCount; index += 1) {
    const text = assemble(serializeFragment(selection.getRangeAt(index).cloneContents()));
    if (text) {
      captures.push(text);
    }
  }
  return captures.join("\n");
};
