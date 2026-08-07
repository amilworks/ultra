/* The house serialization dialect for Markdown mode.
 *
 * One merger, used by the editor AND the fidelity suite, so the gate always
 * tests exactly what ships:
 * - `-` bullets and `---` rules (every seed and agent-written doc on the
 *   platform speaks this dialect);
 * - ==highlight== via the shared handler;
 * - intraword underscores stay raw: `survey_2026_final.csv` is scientific
 *   vocabulary, and CommonMark's flanking rules guarantee an underscore
 *   between alphanumerics can never open emphasis — remark-stringify's
 *   blanket `\_` escape is noise our zero-rewrite law can't accept;
 * - image refs serialize verbatim (alt escaped only for []\), so
 *   `![pooled_grid.png](ultra://…)` round-trips byte-stable.
 */

import { highlightToMarkdown } from "@/lib/remarkHighlight";

type StringifyState = {
  safe: (value: string, config: Record<string, unknown>) => string;
};

const INTRAWORD_ESCAPED_UNDERSCORE = /([\p{L}\p{N}])\\_(?=[\p{L}\p{N}])/gu;

const textWithIntrawordUnderscores = (
  node: { value?: string },
  _parent: unknown,
  state: StringifyState,
  info: Record<string, unknown>
): string =>
  state
    .safe(String(node.value ?? ""), info)
    .replace(INTRAWORD_ESCAPED_UNDERSCORE, "$1_");

/* Milkdown preserves an empty paragraph (an empty table cell, an abandoned
   list item) by serializing the html sentinel `<br />`. Notes are plain
   markdown — an empty cell is written as emptiness, not as HTML. Only the
   exact sentinel is blanked; any other raw html a note carries passes
   through byte-for-byte. */
const EMPTY_LINE_SENTINEL = "<br />";

const plainEmptyHtml = (node: { value?: string }): string => {
  const value = String(node.value ?? "");
  return value === EMPTY_LINE_SENTINEL ? "" : value;
};

const verbatimImage = (node: {
  alt?: string | null;
  url?: string | null;
  title?: string | null;
}): string => {
  const alt = String(node.alt ?? "")
    .replace(/\n+/g, " ")
    .replace(/([[\]\\])/g, "\\$1");
  const url = String(node.url ?? "");
  const wrappedUrl = /[\s()<>]/.test(url) ? `<${url}>` : url;
  const title = node.title ? ` "${String(node.title).replace(/"/g, '\\"')}"` : "";
  return `![${alt}](${wrappedUrl}${title})`;
};

/* Generic over the ctx's own Options type — remark-stringify is a transitive
   dependency (via Milkdown), so its types are not importable here and the
   merger stays structural on purpose. */
export const withNotesDialect = <T extends object>(options: T): T =>
  ({
    ...options,
    bullet: "-",
    rule: "-",
    handlers: {
      ...(options as { handlers?: Record<string, unknown> }).handlers,
      highlight: highlightToMarkdown,
      text: textWithIntrawordUnderscores,
      image: verbatimImage,
      html: plainEmptyHtml,
    },
  }) as T;
