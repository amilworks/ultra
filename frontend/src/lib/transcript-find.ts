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
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return [];
  }
  const matches: TranscriptFindMatch[] = [];
  messages.forEach((message, messageIndex) => {
    const haystack = message.content.toLowerCase();
    let from = 0;
    let occurrence = 0;
    while (true) {
      const at = haystack.indexOf(needle, from);
      if (at === -1) {
        break;
      }
      matches.push({ messageId: message.id, messageIndex, occurrence });
      occurrence += 1;
      // Non-overlapping, matching browser find semantics ("aa" in "aaa" is one).
      from = at + needle.length;
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
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return [];
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
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
  const haystack = flat.toLowerCase();

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
  let from = 0;
  while (true) {
    const at = haystack.indexOf(needle, from);
    if (at === -1) {
      break;
    }
    const start = locate(at);
    // Locate the LAST character then extend by one: an end offset equal to a
    // node's length is valid, but locating `at + length` directly would jump
    // into the next node and produce a zero-width tail there.
    const last = locate(at + needle.length - 1);
    const range = document.createRange();
    range.setStart(start.node, start.offset);
    range.setEnd(last.node, last.offset + 1);
    ranges.push(range);
    from = at + needle.length;
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
    const ranges = collectFindRanges(root, needle);
    if (
      input.currentMessageId &&
      root.getAttribute("data-message-id") === input.currentMessageId &&
      ranges.length > 0
    ) {
      currentRange = ranges[Math.min(input.currentOccurrence, ranges.length - 1)];
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
