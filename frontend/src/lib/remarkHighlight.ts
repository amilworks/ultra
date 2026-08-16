/* ==highlight== — the one styling mark Notes adds on top of GFM.
 *
 * The law behind it: markdown is the data structure; the styling layer is
 * rendering, never hidden data. A highlight means "this matters", which is
 * content — so it lives in the text as ==…== where it survives plaintext
 * edits and reaches the agent, instead of in a side-channel neither can see.
 *
 * Flanking rules are deliberately conservative so dumped prose/code never
 * lights up by accident: the opener must not touch a word character or
 * another '=', the inner text starts and ends with non-space/non-'=', a
 * literal '==' can never appear inside (single '=' is fine — E=mc²), the
 * closer must not run into a word character, and spans stay on one line.
 * Anything ambiguous stays literal text.
 *
 * No lookbehind: the leading context is a capture group instead, so the
 * pattern runs on every Safari the platform supports.
 */

type MdastNode = {
  type: string;
  value?: string;
  children?: MdastNode[];
  data?: Record<string, unknown>;
};

const HIGHLIGHT_SOURCE = "(^|[^=\\w])==(?![\\s=])((?:[^=\\n]|=(?!=))*?[^\\s=])==(?!\\w)";

export const HIGHLIGHT_PATTERN = new RegExp(HIGHLIGHT_SOURCE);

const splitTextNode = (value: string): MdastNode[] | null => {
  const pattern = new RegExp(HIGHLIGHT_SOURCE, "g");
  const out: MdastNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value)) !== null) {
    const prefix = match[1] ?? "";
    const inner = match[2];
    const start = match.index + prefix.length;
    if (start > last) {
      out.push({ type: "text", value: value.slice(last, start) });
    }
    out.push({
      type: "highlight",
      children: [{ type: "text", value: inner }],
      // mdast-util-to-hast renders unknown nodes through data.hName, so the
      // chat renderer gets a real <mark> with zero component wiring.
      data: { hName: "mark" },
    });
    last = start + inner.length + 4;
  }
  if (out.length === 0) {
    return null;
  }
  if (last < value.length) {
    out.push({ type: "text", value: value.slice(last) });
  }
  return out;
};

const walk = (node: MdastNode): void => {
  if (!node.children || node.children.length === 0) {
    return;
  }
  const next: MdastNode[] = [];
  let changed = false;
  for (const child of node.children) {
    if (child.type === "text" && typeof child.value === "string") {
      const pieces = splitTextNode(child.value);
      if (pieces) {
        next.push(...pieces);
        changed = true;
        continue;
      }
    }
    walk(child);
    next.push(child);
  }
  if (changed) {
    node.children = next;
  }
};

/* Remark transformer: text nodes containing ==…== become `highlight` nodes.
   Code/inlineCode carry values, not children, so they are skipped by
   construction — a highlighter never reaches into code. */
export function remarkHighlight() {
  return (tree: unknown): void => {
    walk(tree as MdastNode);
  };
}

/* mdast-util-to-markdown handler for the editor's serializer: a highlight
   node stringifies back to ==…== — byte-stable round trips. */
type PhrasingState = {
  containerPhrasing: (node: unknown, info: Record<string, unknown>) => string;
};

export const highlightToMarkdown = (
  node: unknown,
  _parent: unknown,
  state: PhrasingState,
  info: Record<string, unknown>
): string => `==${state.containerPhrasing(node, { ...info, before: "=", after: "=" })}==`;
