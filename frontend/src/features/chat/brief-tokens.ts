/**
 * The brief grammar: files referenced inside the composer's prose.
 *
 * The composer stays a plain <textarea> — it is the only widget whose drafts,
 * IME composition, paste handling, autosize, undo, screen-reader semantics and
 * phone keyboards are all already right. A file the user brings in becomes a
 * short run of ordinary text, `@<label>`, and a registry beside the draft maps
 * each label to the resource it stands for. Everything here is pure text math
 * over that pair: recognising tokens, keeping the registry honest when text is
 * edited underneath it, and the caret behaviour that makes a token feel atomic
 * (arrows step over it, Backspace removes it whole).
 *
 * Labels are matched EXACTLY against the registry, longest first, so a label
 * may contain spaces ("EBSD map 3.h5") without any escaping and without ever
 * swallowing prose that merely follows an `@`.
 */

export type BriefFileToken = {
  label: string;
  fileId: string;
};

export type BriefTextSegment = {
  kind: "text";
  text: string;
  start: number;
  end: number;
};

export type BriefFileSegment = {
  kind: "file";
  text: string;
  label: string;
  fileId: string;
  start: number;
  end: number;
};

export type BriefSegment = BriefTextSegment | BriefFileSegment;

export const BRIEF_TOKEN_PREFIX = "@";
export const BRIEF_LABEL_MAX_LENGTH = 80;
export const BRIEF_MENTION_QUERY_MAX_LENGTH = 48;

const OPENING_PUNCTUATION = /[([{"'“‘]/;
const CLOSING_PUNCTUATION = /[.,;:!?)\]}"'”’]/;

const isWhitespace = (char: string | undefined): boolean =>
  char !== undefined && /\s/.test(char);

/** A token may begin at the start of the text, after whitespace, or after an
 *  opening bracket or quote — never glued to the end of a word, so an email
 *  address or a handle in the prose is not mistaken for a file. */
const isTokenBoundaryBefore = (text: string, index: number): boolean =>
  index === 0 || isWhitespace(text[index - 1]) || OPENING_PUNCTUATION.test(text[index - 1] ?? "");

/** A token must end at the end of the text, before whitespace, or before
 *  closing punctuation, so "@scan.tif," and "(@scan.tif)" both read as tokens
 *  while "@scan.tiff" never matches the label "scan.tif". */
const isTokenBoundaryAfter = (text: string, index: number): boolean =>
  index >= text.length || isWhitespace(text[index]) || CLOSING_PUNCTUATION.test(text[index] ?? "");

/** Collapse a file name into label form: one line, single spaces, no leading
 *  `@` (which would nest the prefix), bounded so a pathological name cannot
 *  turn the line into a wall of chip. */
export const normalizeBriefLabel = (name: string): string => {
  const collapsed = String(name ?? "")
    .replace(/\s+/g, " ")
    .replace(/^[@\s]+/, "")
    .trim();
  return collapsed.length > BRIEF_LABEL_MAX_LENGTH
    ? collapsed.slice(0, BRIEF_LABEL_MAX_LENGTH).trimEnd()
    : collapsed;
};

/** The label a resource gets in this brief. A file already in the registry
 *  keeps the label it has; a new file whose name collides with another file's
 *  label gets a numbered suffix, so two "scan.tif" uploads stay distinguishable
 *  in the prose and never map to the wrong resource. */
export const uniqueBriefLabel = (
  name: string,
  fileId: string,
  registry: readonly BriefFileToken[]
): string => {
  const existing = registry.find((token) => token.fileId === fileId);
  if (existing) {
    return existing.label;
  }
  const base = normalizeBriefLabel(name) || "file";
  const taken = new Set(registry.map((token) => token.label));
  if (!taken.has(base)) {
    return base;
  }
  for (let suffix = 2; suffix < 1000; suffix += 1) {
    const candidate = `${base} (${suffix})`;
    if (!taken.has(candidate)) {
      return candidate;
    }
  }
  return `${base} (${fileId.slice(0, 6)})`;
};

export const briefTokenText = (label: string): string => `${BRIEF_TOKEN_PREFIX}${label}`;

/** Split the draft into prose and file tokens. Labels are tried longest first
 *  at every `@` that sits on a token boundary; anything that does not match a
 *  registered label is prose, including a bare `@` mid-sentence. */
export const parseBriefSegments = (
  text: string,
  registry: readonly BriefFileToken[]
): BriefSegment[] => {
  const source = String(text ?? "");
  const labels = [...registry]
    .filter((token) => token.label.length > 0)
    .sort((a, b) => b.label.length - a.label.length);
  const segments: BriefSegment[] = [];
  let textStart = 0;
  let index = 0;
  const flushText = (end: number) => {
    if (end > textStart) {
      segments.push({ kind: "text", text: source.slice(textStart, end), start: textStart, end });
    }
  };
  while (index < source.length) {
    if (source[index] !== BRIEF_TOKEN_PREFIX || !isTokenBoundaryBefore(source, index)) {
      index += 1;
      continue;
    }
    const labelStart = index + 1;
    let matched: BriefFileToken | null = null;
    for (const token of labels) {
      if (
        source.startsWith(token.label, labelStart) &&
        isTokenBoundaryAfter(source, labelStart + token.label.length)
      ) {
        matched = token;
        break;
      }
    }
    if (!matched) {
      index += 1;
      continue;
    }
    flushText(index);
    const end = labelStart + matched.label.length;
    segments.push({
      kind: "file",
      text: source.slice(index, end),
      label: matched.label,
      fileId: matched.fileId,
      start: index,
      end,
    });
    textStart = end;
    index = end;
  }
  flushText(source.length);
  return segments;
};

/** Tokens present in the text, in order of first appearance, one per file. */
export const briefFileTokensInText = (
  text: string,
  registry: readonly BriefFileToken[]
): BriefFileToken[] => {
  const seen = new Set<string>();
  const present: BriefFileToken[] = [];
  parseBriefSegments(text, registry).forEach((segment) => {
    if (segment.kind === "file" && !seen.has(segment.fileId)) {
      seen.add(segment.fileId);
      present.push({ label: segment.label, fileId: segment.fileId });
    }
  });
  return present;
};

/** The registry, minus every entry whose token no longer appears in the text.
 *  Deleting a token's characters IS removing the file — the registry follows
 *  the prose, never the other way round. Returns the same array when nothing
 *  changed so callers can bail on identity. */
export const syncBriefRegistryWithText = (
  text: string,
  registry: readonly BriefFileToken[]
): readonly BriefFileToken[] => {
  if (registry.length === 0) {
    return registry;
  }
  const presentIds = new Set(briefFileTokensInText(text, registry).map((token) => token.fileId));
  if (presentIds.size === registry.length) {
    return registry;
  }
  return registry.filter((token) => presentIds.has(token.fileId));
};

export type BriefTextEdit = {
  text: string;
  caret: number;
};

/** Replace the selection with a token, padded with the single spaces that keep
 *  it on its own boundaries: a space before unless the token begins the text or
 *  follows whitespace, and a space after unless whitespace already follows. The
 *  caret lands after the trailing space so typing continues naturally. */
export const insertBriefToken = (
  text: string,
  selectionStart: number,
  selectionEnd: number,
  label: string
): BriefTextEdit => {
  const source = String(text ?? "");
  const start = Math.max(0, Math.min(selectionStart, selectionEnd, source.length));
  const end = Math.max(start, Math.min(Math.max(selectionStart, selectionEnd), source.length));
  const before = source.slice(0, start);
  const after = source.slice(end);
  const needsSpaceBefore = before.length > 0 && !isWhitespace(before[before.length - 1]);
  const needsSpaceAfter = !isWhitespace(after[0]);
  const insertion = `${needsSpaceBefore ? " " : ""}${briefTokenText(label)}${needsSpaceAfter ? " " : ""}`;
  const nextText = `${before}${insertion}${after}`;
  // Land after the space that follows the token — the one just inserted, or
  // the one that was already there — so typing continues on a fresh word.
  const caret =
    before.length + insertion.length + (!needsSpaceAfter && after[0] === " " ? 1 : 0);
  return { text: nextText, caret };
};

export type BriefMentionQuery = {
  /** Index of the `@` that opened the query. */
  start: number;
  /** Text typed after the `@`, up to the caret. */
  query: string;
};

/** The `@…` run the caret is currently typing, if any. The `@` must sit on a
 *  token boundary, the run cannot cross a line break, cannot already be a
 *  registered token, and is bounded in length so a stray `@` far back in a
 *  paragraph does not keep a picker alive across everything typed since. */
export const briefMentionQueryAtCaret = (
  text: string,
  caret: number,
  registry: readonly BriefFileToken[]
): BriefMentionQuery | null => {
  const source = String(text ?? "");
  const position = Math.max(0, Math.min(caret, source.length));
  const windowStart = Math.max(0, position - BRIEF_MENTION_QUERY_MAX_LENGTH - 1);
  for (let index = position - 1; index >= windowStart; index -= 1) {
    const char = source[index];
    if (char === "\n") {
      return null;
    }
    if (char !== BRIEF_TOKEN_PREFIX) {
      continue;
    }
    if (!isTokenBoundaryBefore(source, index)) {
      return null;
    }
    const query = source.slice(index + 1, position);
    if (query.length > 0 && isWhitespace(query[0])) {
      return null;
    }
    const segments = parseBriefSegments(source, registry);
    const alreadyToken = segments.some(
      (segment) => segment.kind === "file" && segment.start === index
    );
    if (alreadyToken) {
      return null;
    }
    return { start: index, query };
  }
  return null;
};

export const briefFileSegmentEndingAt = (
  segments: readonly BriefSegment[],
  position: number
): BriefFileSegment | null =>
  (segments.find(
    (segment) => segment.kind === "file" && segment.end === position
  ) as BriefFileSegment | undefined) ?? null;

export const briefFileSegmentStartingAt = (
  segments: readonly BriefSegment[],
  position: number
): BriefFileSegment | null =>
  (segments.find(
    (segment) => segment.kind === "file" && segment.start === position
  ) as BriefFileSegment | undefined) ?? null;

export const briefFileSegmentContaining = (
  segments: readonly BriefSegment[],
  position: number
): BriefFileSegment | null =>
  (segments.find(
    (segment) => segment.kind === "file" && segment.start < position && position < segment.end
  ) as BriefFileSegment | undefined) ?? null;

/** Where a collapsed caret lands after an arrow press, when a token is in the
 *  way: stepping right from a token's start lands after it, stepping left from
 *  its end lands before it, and a caret that somehow sits inside one is pushed
 *  out in the direction of travel. `null` means the browser's own caret motion
 *  is the right answer. */
export const briefCaretAfterArrow = (
  segments: readonly BriefSegment[],
  caret: number,
  direction: -1 | 1
): number | null => {
  const inside = briefFileSegmentContaining(segments, caret);
  if (inside) {
    return direction === 1 ? inside.end : inside.start;
  }
  if (direction === 1) {
    const ahead = briefFileSegmentStartingAt(segments, caret);
    return ahead ? ahead.end : null;
  }
  const behind = briefFileSegmentEndingAt(segments, caret);
  return behind ? behind.start : null;
};

/** The token Backspace should remove whole: the one that ends exactly at a
 *  collapsed caret. Backspace anywhere else edits characters as usual. */
export const briefBackspaceTarget = (
  segments: readonly BriefSegment[],
  caret: number
): BriefFileSegment | null => briefFileSegmentEndingAt(segments, caret);

/** The token Delete (forward) should remove whole: the one that starts exactly
 *  at a collapsed caret. */
export const briefDeleteTarget = (
  segments: readonly BriefSegment[],
  caret: number
): BriefFileSegment | null => briefFileSegmentStartingAt(segments, caret);

/** Remove a token run plus one padding space (the one after it by preference,
 *  else the one before), so "in @scan.tif to" collapses to "in to" rather than
 *  "in  to". Caret lands where the token began. */
export const removeBriefSegment = (text: string, segment: BriefFileSegment): BriefTextEdit => {
  const source = String(text ?? "");
  let start = segment.start;
  let end = segment.end;
  if (isWhitespace(source[end]) && source[end] !== "\n") {
    end += 1;
  } else if (start > 0 && isWhitespace(source[start - 1]) && source[start - 1] !== "\n") {
    start -= 1;
  }
  return { text: `${source.slice(0, start)}${source.slice(end)}`, caret: start };
};

/** The whisper under a ready brief: what will run, in five words or fewer. */
export const briefSummary = ({
  fileCount,
  workflowLabel,
  modeLabel,
}: {
  fileCount: number;
  workflowLabel?: string | null;
  modeLabel?: string | null;
}): string => {
  const parts: string[] = [];
  if (fileCount > 0) {
    parts.push(`${fileCount} ${fileCount === 1 ? "file" : "files"}`);
  }
  if (workflowLabel && workflowLabel.trim()) {
    parts.push(workflowLabel.trim());
  }
  if (modeLabel && modeLabel.trim()) {
    parts.push(modeLabel.trim());
  }
  return parts.join(" · ");
};
