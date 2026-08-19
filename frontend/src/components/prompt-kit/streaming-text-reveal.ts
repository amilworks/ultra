type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

type RevealCandidate = {
  parent: HastNode;
  index: number;
  value: string;
  start: number;
  end: number;
  phase: 0 | 1;
};

const PROSE_TAGS = new Set(["p", "li", "blockquote"]);
const EXCLUDED_TAGS = new Set([
  "pre",
  "code",
  "table",
  "thead",
  "tbody",
  "tfoot",
  "tr",
  "th",
  "td",
  "math",
  "svg",
  "script",
  "style",
]);
const WHITESPACE = /\s/u;

const classNamesFor = (node: HastNode): string[] => {
  const className = node.properties?.className;
  if (Array.isArray(className)) {
    return className.map(String);
  }
  return typeof className === "string" ? className.split(/\s+/u) : [];
};

const isExcludedElement = (node: HastNode): boolean => {
  if (node.type !== "element") {
    return false;
  }
  if (node.tagName && EXCLUDED_TAGS.has(node.tagName)) {
    return true;
  }
  return classNamesFor(node).some(
    (className) => className === "katex" || className === "katex-display"
  );
};

const finalWordRange = (
  value: string
): { start: number; end: number; phase: 0 | 1 } | null => {
  let end = value.length;
  while (end > 0 && WHITESPACE.test(value[end - 1] ?? "")) {
    end -= 1;
  }
  if (end === 0) {
    return null;
  }

  let start = end;
  while (start > 0 && !WHITESPACE.test(value[start - 1] ?? "")) {
    start -= 1;
  }

  // Alternate two identical animation names at each local word boundary. React
  // keeps this single span mounted while the current word grows, then the class
  // switch restarts the focus transition exactly once for the next word.
  let wordCount = 0;
  let insideWord = false;
  for (let index = 0; index < end; index += 1) {
    const isWhitespace = WHITESPACE.test(value[index] ?? "");
    if (!isWhitespace && !insideWord) {
      wordCount += 1;
    }
    insideWord = !isWhitespace;
  }

  return { start, end, phase: (wordCount % 2) as 0 | 1 };
};

/**
 * Rehype transform used only for the actively streaming Markdown block.
 *
 * It wraps one thing: the final visible word when that word belongs to prose.
 * Existing words return to ordinary text nodes as the stream advances, keeping
 * the DOM bounded to one animated span even for very long responses. The
 * wrapper does not duplicate or hide text from assistive technology.
 */
export function rehypeStreamingTextReveal() {
  return (tree: HastNode): void => {
    let candidate: RevealCandidate | null = null;

    const visit = (
      node: HastNode,
      parent: HastNode | null,
      index: number,
      insideProse: boolean,
      excluded: boolean
    ): void => {
      const nextExcluded = excluded || isExcludedElement(node);
      const nextInsideProse =
        insideProse ||
        (node.type === "element" && Boolean(node.tagName && PROSE_TAGS.has(node.tagName)));

      if (node.type === "text" && typeof node.value === "string") {
        if (!node.value.trim()) {
          return;
        }
        // This is now the final visible source text encountered. Clear a prior
        // prose candidate when the real tail belongs to code, math, a table, or
        // another deliberately stable surface.
        candidate = null;
        if (!parent || nextExcluded || !nextInsideProse) {
          return;
        }
        const range = finalWordRange(node.value);
        if (range) {
          candidate = {
            parent,
            index,
            value: node.value,
            ...range,
          };
        }
        return;
      }

      node.children?.forEach((child, childIndex) => {
        visit(child, node, childIndex, nextInsideProse, nextExcluded);
      });
    };

    visit(tree, null, -1, false, false);

    // TypeScript does not model assignments made inside the recursive closure,
    // so re-establish the post-traversal union before narrowing it.
    const resolvedCandidate = candidate as RevealCandidate | null;
    if (!resolvedCandidate?.parent.children) {
      return;
    }

    const before = resolvedCandidate.value.slice(0, resolvedCandidate.start);
    const word = resolvedCandidate.value.slice(
      resolvedCandidate.start,
      resolvedCandidate.end
    );
    const after = resolvedCandidate.value.slice(resolvedCandidate.end);
    const replacement: HastNode[] = [];

    if (before) {
      replacement.push({ type: "text", value: before });
    }
    replacement.push({
      type: "element",
      tagName: "span",
      properties: {
        className: [
          "pk-stream-tail",
          `pk-stream-tail-phase-${resolvedCandidate.phase}`,
        ],
      },
      children: [{ type: "text", value: word }],
    });
    if (after) {
      replacement.push({ type: "text", value: after });
    }

    resolvedCandidate.parent.children.splice(
      resolvedCandidate.index,
      1,
      ...replacement
    );
  };
}
