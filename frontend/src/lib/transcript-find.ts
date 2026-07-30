/**
 * ⌘F find-within-conversation.
 *
 * Browser find is defeated in BOTH transcript modes: long chats virtualize
 * through react-virtuoso, and short chats window their tail behind a "Show
 * earlier messages" button — either way most of the conversation has no DOM.
 * So matching runs over the conversation DATA (every message, mounted or not),
 * and only the visual highlight touches the DOM that happens to exist.
 *
 * Matching is against each message's source content. That is a deliberate
 * choice, not a limitation: source is what the user typed and what the model
 * produced, it is stable while rows mount and unmount, and it makes text
 * inside code fences findable byte-for-byte. The consequence is that a query
 * matching markdown syntax itself ("**", "```") counts occurrences the
 * rendered view does not show — an accepted, documented edge.
 */

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * One matcher for counting AND painting, so the two can never disagree.
 *
 * Case-insensitivity comes from the regex engine, not toLowerCase — this is a
 * correctness requirement, not style. Lowercasing can CHANGE STRING LENGTH
 * (Turkish İ becomes i + a combining dot, two code units), so offsets found in
 * a lowered haystack misalign against the original string: highlights shift,
 * and a match near a node boundary produces an end offset past the node's
 * length, which makes Range.setEnd throw — inside App's paint effect, where
 * the error boundary would take down the whole app. Regex case-insensitive
 * matching reports indices native to the original string.
 */
export const buildFindMatcher = (query: string): RegExp | null => {
  const needle = query.trim();
  if (!needle) {
    return null;
  }
  return new RegExp(escapeRegExp(needle), "giu");
};

export type TranscriptFindMatch = {
  messageId: string;
  messageIndex: number;
  /** Zero-based occurrence of the query within THIS message's content. */
  occurrence: number;
};

type FindableMessage = { id: string; content: string };

export const computeTranscriptFindMatches = (
  messages: readonly FindableMessage[],
  query: string
): TranscriptFindMatch[] => {
  const matcher = buildFindMatcher(query);
  if (!matcher) {
    return [];
  }
  const matches: TranscriptFindMatch[] = [];
  messages.forEach((message, messageIndex) => {
    matcher.lastIndex = 0;
    let occurrence = 0;
    let hit: RegExpExecArray | null;
    // exec's lastIndex advance is non-overlapping, matching browser find
    // semantics ("aa" in "aaa" is one).
    while ((hit = matcher.exec(message.content)) !== null) {
      matches.push({ messageId: message.id, messageIndex, occurrence });
      occurrence += 1;
      if (hit.index === matcher.lastIndex) {
        matcher.lastIndex += 1;
      }
    }
  });
  return matches;
};

/**
 * Collect DOM Ranges for every occurrence of `query` under `root`.
 *
 * Rendered markdown splits text across element boundaries (`**bo**ld` is two
 * text nodes), so a per-node indexOf would miss any match spanning a boundary.
 * Instead: flatten every text node into one string recording where each node
 * starts, search the flat string, then map global offsets back to
 * (node, offset) pairs for the Range endpoints.
 */
export const collectFindRanges = (root: Node, query: string): Range[] => {
  const matcher = buildFindMatcher(query);
  if (!matcher) {
    return [];
  }
  /* Walk only text the USER CAN SEE. Two invisible layers live inside message
     rows and both shifted ordinals until the current-match tint landed on the
     wrong (or an invisible) instance — review-measured: "gamma" counted 15,
     all 15 painted ranges inside KaTeX's 1×1px clipped MathML layer.
     - .katex-mathml: the accessibility double of every formula.
     - closed <details> (the reasoning trace): body text is in the DOM while
       collapsed; its <summary> stays visible and stays searchable.
     Matches that exist only in invisible source (\gamma) remain COUNTED by
     the data layer — the documented source-vs-render edge — but they no
     longer corrupt which visible occurrence is painted as current. */
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      const parent = node.parentElement;
      if (!parent) {
        return NodeFilter.FILTER_ACCEPT;
      }
      if (parent.closest(".katex-mathml")) {
        return NodeFilter.FILTER_REJECT;
      }
      const closedDetails = parent.closest("details:not([open])");
      if (closedDetails && !parent.closest("summary")) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes: Text[] = [];
  const starts: number[] = [];
  let flat = "";
  while (walker.nextNode()) {
    const textNode = walker.currentNode as Text;
    nodes.push(textNode);
    starts.push(flat.length);
    flat += textNode.data;
  }
  if (!nodes.length) {
    return [];
  }

  // Binary search: which node owns this global offset?
  const locate = (globalIndex: number): { node: Text; offset: number } => {
    let low = 0;
    let high = nodes.length - 1;
    while (low < high) {
      const mid = (low + high + 1) >> 1;
      if (starts[mid] <= globalIndex) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    return { node: nodes[low], offset: globalIndex - starts[low] };
  };

  const ranges: Range[] = [];
  matcher.lastIndex = 0;
  let hit: RegExpExecArray | null;
  while ((hit = matcher.exec(flat)) !== null) {
    const at = hit.index;
    const length = hit[0].length;
    const start = locate(at);
    // Locate the LAST character then extend by one: an end offset equal to a
    // node's length is valid, but locating `at + length` directly would jump
    // into the next node and produce a zero-width tail there.
    const last = locate(at + length - 1);
    const range = document.createRange();
    range.setStart(start.node, start.offset);
    range.setEnd(last.node, last.offset + 1);
    ranges.push(range);
    if (hit.index === matcher.lastIndex) {
      matcher.lastIndex += 1;
    }
  }
  return ranges;
};

const FIND_MATCH_HIGHLIGHT = "ultra-find-match";
const FIND_CURRENT_HIGHLIGHT = "ultra-find-current";

/**
 * CSS Custom Highlight API registry, or null where unsupported. Highlights are
 * pure paint — no DOM mutation — so React never sees them and streaming
 * re-renders cannot orphan wrapper spans (the classic mark.js failure).
 */
const highlightRegistry = (): HighlightRegistry | null =>
  typeof CSS !== "undefined" && "highlights" in CSS ? CSS.highlights : null;

export const clearTranscriptFindHighlights = (): void => {
  const registry = highlightRegistry();
  if (!registry) {
    return;
  }
  registry.delete(FIND_MATCH_HIGHLIGHT);
  registry.delete(FIND_CURRENT_HIGHLIGHT);
};

/**
 * Paint every visible occurrence of the query across the MOUNTED message rows,
 * with the current match painted distinctly. Counts and navigation come from
 * the data layer; this only decorates whatever rows exist right now.
 *
 * Returns whether the current match's row was found in the DOM — false means
 * the virtualized scroll has not mounted it yet and the caller should retry.
 * Without highlight support there is nothing to wait for.
 *
 * The current occurrence is located by ordinal within the rendered text, which
 * matches the source ordinal except when the query also hits markdown syntax —
 * the same accepted edge as counting.
 */
export const applyTranscriptFindHighlights = (input: {
  query: string;
  currentMessageId: string | null;
  currentOccurrence: number;
}): { currentLocated: boolean } => {
  const registry = highlightRegistry();
  if (!registry) {
    return { currentLocated: true };
  }
  registry.delete(FIND_MATCH_HIGHLIGHT);
  registry.delete(FIND_CURRENT_HIGHLIGHT);
  const needle = input.query.trim();
  if (!needle) {
    return { currentLocated: true };
  }
  const matchRanges: Range[] = [];
  let currentRange: Range | null = null;
  for (const root of document.querySelectorAll("[data-message-id]")) {
    let ranges: Range[];
    try {
      ranges = collectFindRanges(root, needle);
    } catch {
      // One pathological row degrades to "no tint there", never to an
      // exception inside App's paint effect — the error boundary above it
      // would replace the entire app.
      continue;
    }
    if (
      input.currentMessageId &&
      root.getAttribute("data-message-id") === input.currentMessageId &&
      ranges.length > 0
    ) {
      // The current match lives ONLY in its own highlight set. Registered in
      // both, the two tints stack — and with --brand ≡ the ink colour, the
      // stacked wash dropped the current match below WCAG AA in both themes.
      const currentIndex = Math.min(input.currentOccurrence, ranges.length - 1);
      currentRange = ranges[currentIndex];
      ranges.splice(currentIndex, 1);
    }
    matchRanges.push(...ranges);
  }
  if (matchRanges.length > 0) {
    registry.set(FIND_MATCH_HIGHLIGHT, new Highlight(...matchRanges));
  }
  if (currentRange) {
    registry.set(FIND_CURRENT_HIGHLIGHT, new Highlight(currentRange));
  }
  return { currentLocated: Boolean(currentRange) || !input.currentMessageId };
};
