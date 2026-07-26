/**
 * Pasted-text ergonomics: when does a paste stop being a prompt and start
 * being a document?
 *
 * People paste logs, stack traces, CSV rows and sequence data into a research
 * tool. Dumped inline, a 300-line log makes the prompt unreadable and buries
 * whatever the user types after it. Converted to an attachment it becomes a
 * removable chip, the composer stays a place for the actual question, and the
 * agent reads the data through the same resource path as any uploaded file.
 */

/**
 * Above this size the paste is an attachment no matter what it looks like —
 * nobody is editing 10k characters in a chat composer.
 */
export const PASTE_ATTACH_ALWAYS_CHARS = 10_000;

/**
 * The structured-data heuristic: moderately long AND many-lined.
 *
 * Both conditions, deliberately. Length alone would convert a long prose
 * prompt someone drafted elsewhere and pasted in — exactly the text a user
 * wants inline and editable. Line count alone would convert a short shell
 * snippet. Logs, tables and FASTA hit both; prose hits neither.
 */
export const PASTE_ATTACH_STRUCTURED_CHARS = 2_000;
export const PASTE_ATTACH_STRUCTURED_LINES = 25;

export const shouldAttachPastedText = (text: string): boolean => {
  if (text.length >= PASTE_ATTACH_ALWAYS_CHARS) {
    return true;
  }
  if (text.length < PASTE_ATTACH_STRUCTURED_CHARS) {
    return false;
  }
  let lines = 1;
  for (let i = 0; i < text.length; i += 1) {
    if (text.charCodeAt(i) === 10) {
      lines += 1;
      if (lines >= PASTE_ATTACH_STRUCTURED_LINES) {
        return true;
      }
    }
  }
  return false;
};

/**
 * Timestamped so consecutive pastes never collide in the chip row or the
 * resource catalog; date included because these outlive the conversation as
 * uploaded resources.
 */
export const pastedTextFileName = (now: Date): string => {
  const pad = (value: number): string => String(value).padStart(2, "0");
  return (
    `pasted-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.txt`
  );
};

export const pastedTextFile = (text: string, now: Date = new Date()): File =>
  new File([text], pastedTextFileName(now), { type: "text/plain" });

/**
 * Quote a transcript selection into the composer as a markdown blockquote.
 *
 * Markdown, not hidden metadata: the user sees exactly what the model will see
 * and can edit or delete any line of it. Trailing whitespace is trimmed so the
 * quote ends cleanly; interior blank lines keep their `>` so one selection
 * stays one visually connected quote block.
 */
export const quoteForComposer = (selection: string): string => {
  const trimmed = selection.replace(/\s+$/, "").replace(/^\n+/, "");
  if (!trimmed) {
    return "";
  }
  return trimmed
    .split("\n")
    .map((line) => (line.trim() ? `> ${line}` : ">"))
    .join("\n");
};

/**
 * Append a quoted selection to an existing draft without mangling either:
 * a blank line on each side keeps the blockquote a separate markdown block,
 * and the trailing newlines put the caret on a fresh line for the question.
 */
export const draftWithQuotedSelection = (draft: string, selection: string): string => {
  const quoted = quoteForComposer(selection);
  if (!quoted) {
    return draft;
  }
  const base = draft.replace(/\s+$/, "");
  return base ? `${base}\n\n${quoted}\n\n` : `${quoted}\n\n`;
};
