import { marked } from "marked";
import { Check, Copy, ExternalLink, Layers3 } from "lucide-react";
import {
  type HTMLAttributes,
  lazy,
  memo,
  type ReactNode,
  Suspense,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";

// First-party scheme: `ultra://resource/<id>/<name>` lets markdown reference
// platform objects (notes today; anywhere markdown renders tomorrow). The
// default transform strips unknown schemes to empty strings, which silently
// blanks those references before a custom component can resolve them. Passing
// the scheme through is inert by itself — browsers do not fetch ultra:// —
// so rendering only happens where a caller supplies a resolving component.
const ultraUrlTransform = (url: string): string =>
  url.startsWith("ultra://") ? url : defaultUrlTransform(url);
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { remarkHighlight } from "@/lib/remarkHighlight";
import { Button } from "@/components/ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { DEFAULT_BISQUE_BROWSER_URL } from "@/lib/config";
import { getLensOpener } from "@/lib/lensNavigation";
import { resolveLensLink } from "@/lib/navUrl";
import { cn } from "@/lib/utils";
import { reportClientError } from "@/lib/client-diagnostics";
import { CodeBlock, CodeBlockCode } from "./code-block";
import { rehypeStreamingTextReveal } from "./streaming-text-reveal";

// Lazy so recharts (bundled) only loads when a chart actually appears in chat.
const ChatChart = lazy(() => import("@/components/chat/ChatChart"));

export type MarkdownProps = {
  children: string;
  id?: string;
  className?: string;
  components?: Partial<Components>;
  streamingReveal?: boolean;
};

// A pathological unbroken paragraph should not pay for a decorative traversal
// on every stream frame. Ordinary prose stays far below this bound; beyond it,
// legibility and catch-up throughput win and the response renders normally.
const STREAMING_REVEAL_MAX_BLOCK_CHARACTERS = 12_000;

// Sentinels from the Unicode private-use area. marked treats them as ordinary
// inline text, so a masked span is never split across block boundaries.
const MATH_MASK_OPEN = "";
const MATH_MASK_CLOSE = "";
const MATH_MASK_PATTERN = /(\d+)/g;
const DISPLAY_MATH_PATTERN = /\$\$[\s\S]*?\$\$/g;

// --- GFM pipe-table repair ------------------------------------------------
// remark-gfm follows the spec strictly: a pipe table is only recognized when
// the header row and the delimiter row have the *same* number of cells. Models
// routinely miscount the `---|---` delimiter (or add/lose a column), and the
// whole table then renders as raw `| ... |` text. When a block clearly attempts
// a table but only the delimiter count is off, rebuild the delimiter to match
// the header so the table renders. We deliberately do NOT touch blocks whose
// header disagrees with the data rows (e.g. an unescaped `|` inside a cell): the
// intended columns are ambiguous there and guessing would produce a misaligned,
// misleading table — that case is handled model-side by escaping pipes.
const DELIMITER_CELL_PATTERN = /^\s*:?-+:?\s*$/;
const DELIMITER_ROW_SHAPE = /^[\s|:-]+$/;

function splitTableCells(row: string): string[] {
  let cells = row.trim();
  if (cells.startsWith("|")) {
    cells = cells.slice(1);
  }
  if (cells.endsWith("|")) {
    cells = cells.slice(0, -1);
  }
  // Split on unescaped pipes only, matching how GFM delimits cells.
  return cells.split(/(?<!\\)\|/);
}

function isDelimiterRow(row: string): boolean {
  const trimmed = row.trim();
  // Require an actual pipe and dash so a thematic break (`---`) or ordinary
  // prose is never mistaken for a one-column delimiter row.
  if (!trimmed.includes("|") || !trimmed.includes("-")) {
    return false;
  }
  if (!DELIMITER_ROW_SHAPE.test(trimmed)) {
    return false;
  }
  const cells = splitTableCells(row);
  return cells.length >= 1 && cells.every((cell) => DELIMITER_CELL_PATTERN.test(cell));
}

function delimiterCellForAlignment(cell: string): string {
  const trimmed = cell.trim();
  const left = trimmed.startsWith(":");
  const right = trimmed.endsWith(":");
  if (left && right) {
    return ":-:";
  }
  if (right) {
    return "--:";
  }
  if (left) {
    return ":--";
  }
  return "---";
}

function buildDelimiterRow(columns: number, existing: string[]): string {
  const cells: string[] = [];
  for (let index = 0; index < columns; index += 1) {
    const source = existing[index];
    cells.push(source ? delimiterCellForAlignment(source) : "---");
  }
  return `| ${cells.join(" | ")} |`;
}

export function repairTableDelimiters(block: string): string {
  if (!block.includes("|") || !block.includes("-")) {
    return block;
  }
  const lines = block.split("\n");
  let changed = false;
  for (let index = 1; index < lines.length; index += 1) {
    if (!isDelimiterRow(lines[index])) {
      continue;
    }
    const header = lines[index - 1];
    if (!header.includes("|") || header.trim() === "") {
      continue;
    }
    // Blockquote / list-nested tables carry a line prefix (`>`, `- `) that our
    // rebuilt delimiter would drop, breaking the container. Leave those alone.
    if (/^\s*(?:>|[-*+]\s|\d+[.)]\s)/.test(header) || /^\s*>/.test(lines[index])) {
      continue;
    }
    const headerColumns = splitTableCells(header).length;
    const delimiterColumns = splitTableCells(lines[index]).length;
    if (headerColumns === delimiterColumns) {
      continue;
    }
    // Only repair when the header agrees with EVERY contiguous data row. A lone
    // header+delimiter with no data rows is too often a setext heading, a
    // thematic rule, or prose describing table syntax; a disagreeing data row
    // means stray/ambiguous pipes. In both cases a rebuild would fabricate or
    // mangle a table, so we leave the block raw and rely on the model to escape
    // pipes and keep column counts consistent.
    const dataCounts: number[] = [];
    for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
      const row = lines[cursor];
      if (row.trim() === "" || !row.includes("|")) {
        break;
      }
      dataCounts.push(splitTableCells(row).length);
    }
    if (
      dataCounts.length === 0 ||
      !dataCounts.every((count) => count === headerColumns)
    ) {
      continue;
    }
    lines[index] = buildDelimiterRow(headerColumns, splitTableCells(lines[index]));
    changed = true;
  }
  return changed ? lines.join("\n") : block;
}

// --- Numeric column alignment ----------------------------------------------
// GFM leaves a column left-aligned unless the delimiter row carries an explicit
// `--:` marker, which models rarely emit. Left-aligned numbers defeat the
// table's `tabular-nums`: magnitudes only compare at a glance when units digits
// line up. Right-align any column whose body cells are all numeric (empty,
// dash, and n/a cells are neutral), leaving explicit model-authored alignment
// untouched.
type MdastNode = {
  type: string;
  value?: string;
  children?: MdastNode[];
  align?: Array<string | null>;
};

// Signed integers/decimals with optional thousands commas, scientific notation,
// or a percent suffix. Deliberately conservative: unit suffixes ("137ms"),
// currency, and formulas stay left-aligned prose.
const NUMERIC_CELL_PATTERN =
  /^[+\-±]?(\d[\d,]*(\.\d+)?|\.\d+)([eE][+-]?\d+)?%?$/;
const NEUTRAL_CELL_PATTERN = /^([-–—·]|n\/a)?$/i;

function flattenMdastValue(node: MdastNode): string {
  // Leaf values cover text and inlineCode, but also inlineMath/html — flatten
  // those too so a `$x^2$` cell reads as "$x^2$" (non-numeric), not as empty
  // (neutral), which would let a formula column right-align.
  if (typeof node.value === "string") {
    return node.value;
  }
  if (Array.isArray(node.children)) {
    return node.children.map((child) => flattenMdastValue(child)).join("");
  }
  return "";
}

function applyNumericColumnAlignment(table: MdastNode): void {
  const align = table.align;
  const bodyRows = (table.children ?? []).slice(1);
  if (!align || bodyRows.length === 0) {
    return;
  }
  for (let column = 0; column < align.length; column += 1) {
    if (align[column] != null) {
      continue;
    }
    let sawNumber = false;
    let qualifies = true;
    for (const row of bodyRows) {
      const cell = row.children?.[column];
      const content = cell ? flattenMdastValue(cell).trim() : "";
      if (NUMERIC_CELL_PATTERN.test(content)) {
        sawNumber = true;
      } else if (!NEUTRAL_CELL_PATTERN.test(content)) {
        qualifies = false;
        break;
      }
    }
    if (qualifies && sawNumber) {
      align[column] = "right";
    }
  }
}

export function remarkNumericColumnAlign() {
  return (tree: MdastNode): void => {
    const visit = (node: MdastNode): void => {
      if (node.type === "table") {
        applyNumericColumnAlignment(node);
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

export function parseMarkdownIntoBlocks(markdown: string): string[] {
  try {
    // marked's lexer has no math awareness. A multi-line `$$ ... $$` display
    // block whose body contains markdown-significant lines — `- term`, `# 2`,
    // a blank line — gets split into a broken paragraph plus a list, which then
    // renders as a red KaTeX error over raw-LaTeX bullets. Mask each display
    // span as an opaque placeholder before lexing so the block stays intact,
    // then restore the spans in every token's raw text after splitting.
    const displaySpans: string[] = [];
    const masked = markdown.replace(DISPLAY_MATH_PATTERN, (span) => {
      const index = displaySpans.push(span) - 1;
      return `${MATH_MASK_OPEN}${index}${MATH_MASK_CLOSE}`;
    });
    const tokens = marked.lexer(masked);
    const restore = (raw: string): string =>
      displaySpans.length === 0
        ? raw
        : raw.replace(
            MATH_MASK_PATTERN,
            (whole, index: string) => displaySpans[Number(index)] ?? whole
          );
    return tokens.map((token) => {
      // Never rewrite fenced/indented code — pipes and dashes there are literal.
      // Table repair runs on the masked text so display math stays invisible.
      if (token.type === "code") {
        return restore(token.raw);
      }
      return restore(repairTableDelimiters(token.raw));
    });
  } catch (error) {
    // marked's lexer can throw on pathological model output (e.g. extreme
    // nesting). This runs in a render-path useMemo, so an uncaught throw would
    // crash the whole app via the top-level boundary. Fall back to rendering the
    // text as a single block so the message still shows.
    reportClientError(error, { source: "markdown-lexer" });
    return [markdown];
  }
}

// KaTeX display environments that models frequently emit *without* any `$$`
// fence. remark-math only recognizes math inside delimiters, so a bare
// `\begin{bmatrix} ... \end{bmatrix}` renders as raw LaTeX prose. We auto-fence
// these so the equation renders as math.
const MATH_ENVIRONMENTS =
  "bmatrix|pmatrix|vmatrix|Vmatrix|Bmatrix|smallmatrix|matrix|" +
  "cases|aligned|alignedat|align\\*?|alignat\\*?|gather\\*?|gathered|" +
  "equation\\*?|multline\\*?|split|array|eqnarray\\*?";
// The optional trailing group captures sentence punctuation sitting right
// after the environment (followed by whitespace or end): fencing pulls it
// INSIDE the display, where print mathematics sets it — otherwise the `.` of
// "The vector is \begin{bmatrix}…\end{bmatrix}." is stranded between the two
// inserted blank lines and renders as a one-character orphan paragraph.
const BARE_ENVIRONMENT_PATTERN = new RegExp(
  String.raw`(?<![$\\])(\\begin\{(${MATH_ENVIRONMENTS})\}[\s\S]*?\\end\{\2\})(?:[ \t]*([.,;:!?])(?=\s|$))?`,
  "g"
);

// Fenced code blocks are opaque to every math transform: their dollars and
// TeX-lookalike text are literal. CommonMark, approximated: a fence is 0–3
// spaces of indent plus three or more backticks/tildes; a backtick fence's
// info string cannot itself contain a backtick; the closing fence uses the
// same character, at least as long, alone on its line. An unclosed fence
// (mid-stream) swallows the rest of the source as code, which matches how
// the markdown parser will treat it.
type SourceSegment = { text: string; isCode: boolean };

const FENCE_CLOSE_PATTERN = /^ {0,3}(`{3,}|~{3,})[ \t]*$/;
const FENCE_OPEN_PATTERN = /^ {0,3}(`{3,}|~{3,})(.*)$/;

const splitFencedCode = (source: string): SourceSegment[] => {
  const segments: SourceSegment[] = [];
  let buffer: string[] = [];
  let bufferIsCode = false;
  let openFence: { char: string; length: number } | null = null;
  const flush = () => {
    if (buffer.length > 0) {
      segments.push({ text: buffer.join("\n"), isCode: bufferIsCode });
      buffer = [];
    }
  };
  for (const line of source.split("\n")) {
    if (openFence) {
      buffer.push(line);
      const close = FENCE_CLOSE_PATTERN.exec(line);
      if (
        close &&
        close[1][0] === openFence.char &&
        close[1].length >= openFence.length
      ) {
        openFence = null;
        flush();
        bufferIsCode = false;
      }
      continue;
    }
    const open = FENCE_OPEN_PATTERN.exec(line);
    if (open && !(open[1][0] === "`" && open[2].includes("`"))) {
      flush();
      bufferIsCode = true;
      openFence = { char: open[1][0], length: open[1].length };
      buffer.push(line);
      continue;
    }
    buffer.push(line);
  }
  flush();
  return segments;
};

// Inline code spans keep their contents literal too. Masking them to
// same-length blanks (newlines kept, so offsets AND paragraph boundaries stay
// aligned) lets each transform test, by offset, whether its match overlaps a
// span — and keeps their dollars (`echo $PATH`) out of the delimiter parity
// counts below. Runs of 1–2 backticks cover real model output; a span that
// needs longer runs to escape inner backticks is not modeled.
const INLINE_CODE_SPAN_PATTERN =
  /(?<!`)(`{1,2})(?!`)(?:[^`\n]|\n(?![ \t]*\n))+?\1(?!`)/g;

const maskInlineCode = (text: string): string =>
  text.replace(INLINE_CODE_SPAN_PATTERN, (span) => span.replace(/[^\n]/g, " "));

// `\$` never delimits math; `$$` pairs delimit display blocks (which may span
// blank lines); lone `$` pairs delimit inline spans (which never cross one).
const countDollarDelimiters = (
  text: string
): { singles: number; doubles: number } => {
  let singles = 0;
  let doubles = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\\") {
      i += 1;
      continue;
    }
    if (ch !== "$") {
      continue;
    }
    if (text[i + 1] === "$") {
      doubles += 1;
      i += 1;
    } else {
      singles += 1;
    }
  }
  return { singles, doubles };
};

// Whether `maskedPrefix` (everything before a candidate match, inline code
// already masked) ends inside an open math span. Display parity is counted
// over the whole prefix; inline parity only within the current paragraph,
// since a `$…$` span cannot cross a blank line — so a stray currency dollar
// in an earlier paragraph cannot poison later fencing. Getting this wrong is
// worse than leaving raw LaTeX visible: fencing a matrix that sits INSIDE an
// open `$…$` span severs the span and re-pairs every later dollar in the
// paragraph — prose lands in math mode with its spaces collapsed, and real
// math falls out as raw text (observed live with a smallmatrix inside an
// inline span, which garbled the entire rest of its paragraph).
const isInsideMathSpan = (maskedPrefix: string): boolean => {
  if (countDollarDelimiters(maskedPrefix).doubles % 2 === 1) {
    return true;
  }
  const paragraphBreaks = /\n[ \t]*\n/g;
  let paragraphStart = 0;
  for (
    let breakMatch = paragraphBreaks.exec(maskedPrefix);
    breakMatch;
    breakMatch = paragraphBreaks.exec(maskedPrefix)
  ) {
    paragraphStart = breakMatch.index + breakMatch[0].length;
  }
  return (
    countDollarDelimiters(maskedPrefix.slice(paragraphStart)).singles % 2 === 1
  );
};

// Applies `pattern` to `text`, rewriting only matches that land wholly outside
// inline code: the mask is offset-aligned, so a match whose masked slice
// differs from itself overlaps a code span and must stay literal.
const replaceOutsideInlineCode = (
  text: string,
  pattern: RegExp,
  build: (context: {
    match: string;
    groups: Array<string | undefined>;
    offset: number;
    masked: string;
  }) => string
): string => {
  const masked = maskInlineCode(text);
  return text.replace(pattern, (...args) => {
    const match = args[0] as string;
    const offset = args[args.length - 2] as number;
    if (masked.slice(offset, offset + match.length) !== match) {
      return match;
    }
    const groups = args.slice(1, -2) as Array<string | undefined>;
    return build({ match, groups, offset, masked });
  });
};

// A paragraph that is exactly one inline formula is a display equation the
// model failed to mark up: it renders left-aligned at inline size (side-set
// limits, `\tfrac`/`smallmatrix` contortions) and never gets the
// `.katex-display` centering, displaystyle sizing, or overflow scrollport.
// Promote it to a real `$$` flow block. The single-line `$$…$$` form needs
// the same lift: remark-math parses it as INLINE math (math-text with a
// two-dollar run), not display. Trailing sentence punctuation moves inside
// the display, where print mathematics sets it. During streaming the tail
// paragraph may promote and then demote when more of the sentence arrives —
// the same class of one-delta settle as raw→rendered math.
const LONE_DISPLAY_MATH_PARAGRAPH =
  /^\$\$((?:\\[\s\S]|[^$\\])+)\$\$([.,;:!?])?$/;
const LONE_INLINE_MATH_PARAGRAPH = /^\$((?:\\[\s\S]|[^$\\])+)\$([.,;:!?])?$/;

const promoteParagraph = (paragraph: string): string => {
  if (/^(?: {4,}|\t)/.test(paragraph)) {
    return paragraph; // indented code block, not prose
  }
  const trimmed = paragraph.trim();
  const match =
    LONE_DISPLAY_MATH_PARAGRAPH.exec(trimmed) ??
    LONE_INLINE_MATH_PARAGRAPH.exec(trimmed);
  if (!match) {
    return paragraph;
  }
  const body = (match[1] ?? "").trim();
  if (!body) {
    return paragraph;
  }
  return `$$\n${body}${match[2] ?? ""}\n$$`;
};

const promoteLoneMathParagraphs = (
  prose: string,
  holdFinalParagraph: boolean
): string => {
  const parts = prose.split(/(\n[ \t]*\n+)/);
  for (let i = 0; i < parts.length; i += 2) {
    // The captured-separator split always ends on a paragraph slot (empty when
    // the source ends with a blank line, in which case skipping is a no-op and
    // the real final paragraph — already closed by that blank line — promotes).
    if (holdFinalParagraph && i === parts.length - 1) {
      continue;
    }
    parts[i] = promoteParagraph(parts[i]);
  }
  return parts.join("");
};

const normalizeProseMath = (
  prose: string,
  holdFinalParagraph: boolean
): string => {
  let normalized = prose;

  // Normalize only explicit TeX delimiters so we do not accidentally turn
  // ordinary bracketed prose into math. Strip leading blockquote markers from the
  // captured display body: models frequently emit `> \[ ... > \]`, and leaving the
  // `> ` prefixes inside the converted `$$ ... $$` block makes remark-math fail to
  // parse it, so the equation renders as raw LaTeX instead of math.
  normalized = replaceOutsideInlineCode(
    normalized,
    /\\\[([\s\S]*?)\\\]/g,
    ({ groups }) =>
      `\n$$\n${String(groups[0] ?? "")
        .replace(/^[ \t]*>[ \t]?/gm, "")
        .trim()}\n$$\n`
  );
  normalized = replaceOutsideInlineCode(
    normalized,
    /\\\((.+?)\\\)/g,
    ({ groups }) =>
      `$${String(groups[0] ?? "")
        .replace(/^[ \t]*>[ \t]?/gm, "")
        .trim()}$`
  );

  // Auto-fence bare display environments — except inside an already-open math
  // span (`$$` block or inline `$…$`, see isInsideMathSpan), where the
  // environment is already delimited and fencing would sever the span. Trailing
  // sentence punctuation (group 3) folds inside the display; the prose after it
  // resumes as its own paragraph, which is exactly how print sets it.
  normalized = replaceOutsideInlineCode(
    normalized,
    BARE_ENVIRONMENT_PATTERN,
    ({ match, groups, offset, masked }) =>
      isInsideMathSpan(masked.slice(0, offset))
        ? match
        : `\n\n$$\n${groups[0] ?? match}${groups[2] ?? ""}\n$$\n\n`
  );

  return promoteLoneMathParagraphs(normalized, holdFinalParagraph);
};

export type NormalizeMathOptions = {
  /** True while the message is still streaming. The final paragraph may be an
   *  unfinished sentence, so lone-formula promotion holds off on it: a span
   *  promoted to a display and demoted one delta later is a visible layout
   *  jump, while promoting once at completion is the same settle the
   *  raw→rendered math swap already has. */
  streamingTail?: boolean;
};

export const normalizeMathMarkdown = (
  source: string,
  options?: NormalizeMathOptions
): string => {
  const segments = splitFencedCode(source);
  return segments
    .map((segment, index) =>
      segment.isCode
        ? segment.text
        : normalizeProseMath(
            segment.text,
            options?.streamingTail === true && index === segments.length - 1
          )
    )
    .join("\n");
};

const hasMathMarkdownSyntax = (source: string): boolean => {
  if (!source.trim()) {
    return false;
  }
  return (
    /\\\(|\\\[|\$\$/m.test(source) ||
    /(^|[^\\])\$(?!\$)([^$\n]|\\\$)+\$(?!\$)/m.test(source)
  );
};

function extractLanguage(className?: string): string {
  if (!className) return "plaintext";
  const match = className.match(/language-([\w-]+)/);
  return match ? match[1] : "plaintext";
}

function flattenNodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map((entry) => flattenNodeText(entry)).join("");
  if (typeof node === "object" && "props" in node) {
    const props = (node as { props?: { children?: ReactNode } }).props;
    return flattenNodeText(props?.children);
  }
  return "";
}

function shouldConstrainTableCell(children: ReactNode): boolean {
  const content = flattenNodeText(children).trim();
  if (!content) return false;
  if (content.length >= 120) return true;
  return /\S{56,}/.test(content);
}

type BisqueLinkMeta = {
  clientViewUrl: string;
  imageServiceUrl: string | null;
  resourceId: string | null;
};

const BISQUE_LINK_FALLBACK_IMAGE_URL = "/bq-bg8.webp";

const decodeSafe = (value: string): string => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const configuredBisqueOrigin = (() => {
  const candidate = String(DEFAULT_BISQUE_BROWSER_URL || "").trim();
  if (!candidate) {
    return null;
  }
  try {
    const parsed = new URL(candidate);
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return null;
  }
})();

const resolveBisqueLinkMeta = (href: string): BisqueLinkMeta | null => {
  let parsed: URL;
  try {
    parsed = new URL(href);
  } catch {
    return null;
  }

  const path = parsed.pathname;
  const origin = configuredBisqueOrigin || `${parsed.protocol}//${parsed.host}`;
  const resourceUri = (() => {
    if (/\/client_service\/view$/i.test(path)) {
      const resourceRaw = parsed.searchParams.get("resource");
      if (!resourceRaw) {
        return null;
      }
      return decodeSafe(resourceRaw);
    }
    if (/\/data_service\//i.test(path)) {
      return parsed.toString();
    }
    if (/\/image_service\//i.test(path)) {
      return parsed.toString().replace("/image_service/", "/data_service/");
    }
    return null;
  })();

  if (!resourceUri) {
    return null;
  }
  const normalizedResourceUri = resourceUri.replace("/image_service/", "/data_service/");
  const resourceUniq =
    normalizedResourceUri.split("/").filter(Boolean).pop() ?? null;
  const imageServiceUrl = resourceUniq ? `${origin}/image_service/${resourceUniq}` : null;
  return {
    clientViewUrl: `${origin}/client_service/view?resource=${normalizedResourceUri}`,
    imageServiceUrl,
    resourceId: resourceUniq,
  };
};

function BisqueMarkdownLink({
  href,
  children,
  className,
  ...props
}: React.ComponentPropsWithoutRef<"a">) {
  const bisqueMeta = useMemo(
    () => (typeof href === "string" ? resolveBisqueLinkMeta(href) : null),
    [href]
  );
  const [failedPreviewUrl, setFailedPreviewUrl] = useState<string | null>(null);
  const previewUrl = bisqueMeta?.imageServiceUrl ?? null;
  const canShowPreviewImage = Boolean(previewUrl && failedPreviewUrl !== previewUrl);

  if (!href || !bisqueMeta) {
    return (
      <a
        href={href}
        className={cn("pk-link", className)}
        target="_blank"
        rel="noreferrer"
        {...props}
      >
        {children}
      </a>
    );
  }

  return (
    <HoverCard openDelay={120} closeDelay={120}>
      <HoverCardTrigger asChild>
        <span className="bisque-link-wrap">
          <a
            href={bisqueMeta.clientViewUrl}
            className={cn("pk-link", className)}
            target="_blank"
            rel="noreferrer"
            {...props}
          >
            {children}
          </a>
          <a
            href={bisqueMeta.clientViewUrl}
            className="bisque-link-open"
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink data-icon="inline-start" />
            Open viewer
          </a>
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        sideOffset={8}
        className="bisque-link-preview-card"
      >
        {canShowPreviewImage && previewUrl ? (
          <img
            src={previewUrl}
            alt="BisQue preview"
            loading="lazy"
            className="bisque-link-preview-image"
            onError={() => setFailedPreviewUrl(previewUrl)}
          />
        ) : (
          <div className="bisque-link-preview-fallback">
            <img
              src={BISQUE_LINK_FALLBACK_IMAGE_URL}
              alt=""
              loading="lazy"
              className="bisque-link-preview-image bisque-link-preview-image-fallback"
            />
            <div className="bisque-link-preview-copy">
              <strong>Open this resource in BisQue</strong>
              <p>
                Launch the BisQue viewer for the full interactive view, tools,
                and permissions-aware access.
              </p>
              {bisqueMeta.resourceId ? <span>{bisqueMeta.resourceId}</span> : null}
            </div>
            <Button asChild variant="outline" size="sm" className="bisque-link-preview-action">
              <a href={bisqueMeta.clientViewUrl} target="_blank" rel="noreferrer">
                <ExternalLink data-icon="inline-start" />
                Open viewer
              </a>
            </Button>
          </div>
        )}
      </HoverCardContent>
    </HoverCard>
  );
}

// Ultra resource citations open OUR Lens in-app. Both anchors keep a real href
// so a cold load, a middle-click, or a modifier-click still reaches the deep
// link through the browser; only a plain left-click is intercepted, and only
// when App has registered an opener (read at click time — this component is
// module-level and the block memo would freeze anything captured at render).
function LensMarkdownLink({
  href,
  fileIds,
  children,
  className,
  ...props
}: React.ComponentPropsWithoutRef<"a"> & { href: string; fileIds: string[] }) {
  const handleClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }
    const opener = getLensOpener();
    if (!opener) {
      return;
    }
    event.preventDefault();
    opener(fileIds);
  };
  return (
    <span className="ultra-link-wrap">
      <a href={href} className={cn("pk-link", className)} onClick={handleClick} {...props}>
        {children}
      </a>
      <a href={href} className="ultra-link-open" onClick={handleClick}>
        <Layers3 data-icon="inline-start" />
        Open in Lens
      </a>
    </span>
  );
}

// Copy-LaTeX affordance on display equations. KaTeX keeps the source in a
// hidden MathML <annotation encoding="application/x-tex">, so the button reads
// it from its own subtree — no state threading through the markdown pipeline.
// The shell wraps the KaTeX block so the button does not ride the equation's
// own horizontal scrollport, and so a wide equation scrolls under it.
const MATH_TEX_ANNOTATION_SELECTOR = 'annotation[encoding="application/x-tex"]';

function MathDisplayBlock({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement>) {
  const shellRef = useRef<HTMLSpanElement>(null);
  const resetTimerRef = useRef<number | null>(null);
  const [copied, setCopied] = useState(false);
  useEffect(
    () => () => {
      if (resetTimerRef.current !== null) {
        window.clearTimeout(resetTimerRef.current);
      }
    },
    []
  );
  const copySource = () => {
    const tex =
      shellRef.current?.querySelector(MATH_TEX_ANNOTATION_SELECTOR)
        ?.textContent ?? "";
    if (!tex || !navigator.clipboard) {
      return;
    }
    navigator.clipboard
      .writeText(tex)
      .then(() => {
        setCopied(true);
        if (resetTimerRef.current !== null) {
          window.clearTimeout(resetTimerRef.current);
        }
        resetTimerRef.current = window.setTimeout(
          () => setCopied(false),
          1600
        );
      })
      .catch(() => {
        // Clipboard permission denied: the button simply stays quiet.
      });
  };
  return (
    <span className="pk-math-display-shell" ref={shellRef}>
      <span className={className} {...props}>
        {children}
      </span>
      <button
        type="button"
        className="pk-math-copy"
        aria-label={copied ? "Copied LaTeX source" : "Copy LaTeX source"}
        title="Copy LaTeX source"
        onClick={copySource}
      >
        {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
      </button>
    </span>
  );
}

const BASE_COMPONENTS: Partial<Components> = {
  // Display math is the only span variety that gets intercepted: rehype-katex
  // emits `<span class="katex-display">` for block math, and every other span
  // (including KaTeX's thousands of internal ones) passes straight through.
  span: function SpanComponent({ className, ...props }) {
    if (className?.includes("katex-display")) {
      return <MathDisplayBlock className={className} {...props} />;
    }
    return <span className={className} {...props} />;
  },
  code: function CodeComponent({ className, children, ...props }) {
    const isInline =
      !props.node?.position?.start.line ||
      props.node?.position?.start.line === props.node?.position?.end.line;

    if (isInline) {
      return (
        <code className={cn("pk-inline-code", className)} {...props}>
          {children}
        </code>
      );
    }

    const language = extractLanguage(className);
    const source = String(children);
    if (language === "chart") {
      // Declarative JSON chart spec → fixed, validated recharts renderer.
      // No code executes; an invalid/streaming spec falls back to a code block.
      return (
        <Suspense
          fallback={
            <CodeBlock className={className}>
              <CodeBlockCode code={source} language="json" />
            </CodeBlock>
          }
        >
          <ChatChart source={source} />
        </Suspense>
      );
    }
    return (
      <CodeBlock className={className}>
        <CodeBlockCode code={source} language={language} />
      </CodeBlock>
    );
  },
  a: function LinkComponent({ href, children, ...props }) {
    const lens = typeof href === "string" ? resolveLensLink(href, window.location.origin) : null;
    if (lens) {
      return (
        <LensMarkdownLink href={lens.href} fileIds={lens.fileIds} {...props}>
          {children}
        </LensMarkdownLink>
      );
    }
    return (
      <BisqueMarkdownLink href={href} {...props}>
        {children}
      </BisqueMarkdownLink>
    );
  },
  pre: function PreComponent({ children }) {
    return <>{children}</>;
  },
  table: function TableComponent({ className, children, ...props }) {
    return (
      <div className="pk-table-wrap">
        <table className={cn("pk-table", className)} {...props}>
          {children}
        </table>
      </div>
    );
  },
  thead: function TableHeadComponent({ className, children, ...props }) {
    return (
      <thead className={cn("pk-table-head", className)} {...props}>
        {children}
      </thead>
    );
  },
  tbody: function TableBodyComponent({ className, children, ...props }) {
    return (
      <tbody className={cn("pk-table-body", className)} {...props}>
        {children}
      </tbody>
    );
  },
  tr: function TableRowComponent({ className, children, ...props }) {
    return (
      <tr className={cn("pk-table-row", className)} {...props}>
        {children}
      </tr>
    );
  },
  // Column alignment needs no handling here: react-markdown converts the GFM
  // align attribute into an inline `text-align` style on th/td
  // (hast-util-to-jsx-runtime's tableCellAlignToStyle), which the props spread
  // applies and the cell-content span inherits. It is sourced from explicit
  // `--:` delimiter markers or from remarkNumericColumnAlign.
  th: function TableHeaderCellComponent({ className, children, ...props }) {
    const shouldConstrain = shouldConstrainTableCell(children);
    return (
      <th
        className={cn(
          "pk-table-head-cell",
          shouldConstrain && "pk-table-cell-long",
          className
        )}
        {...props}
      >
        <span className="pk-table-cell-content">{children}</span>
      </th>
    );
  },
  td: function TableCellComponent({ className, children, ...props }) {
    const shouldConstrain = shouldConstrainTableCell(children);
    return (
      <td
        className={cn(
          "pk-table-cell",
          shouldConstrain && "pk-table-cell-long",
          className
        )}
        {...props}
      >
        <span className="pk-table-cell-content">{children}</span>
      </td>
    );
  },
};

const MemoizedMarkdownBlock = memo(
  function MarkdownBlock({
    content,
    components = BASE_COMPONENTS,
    remarkPlugins,
    rehypePlugins,
  }: {
    content: string;
    components?: Partial<Components>;
    remarkPlugins: Array<unknown>;
    rehypePlugins: Array<unknown>;
    pluginKey: string;
  }) {
    return (
      <ReactMarkdown
        remarkPlugins={remarkPlugins as []}
        rehypePlugins={rehypePlugins as []}
        components={components}
        urlTransform={ultraUrlTransform}
      >
        {content}
      </ReactMarkdown>
    );
  },
  (prevProps, nextProps) =>
    prevProps.content === nextProps.content &&
    prevProps.pluginKey === nextProps.pluginKey
);

MemoizedMarkdownBlock.displayName = "MemoizedMarkdownBlock";

function MarkdownComponent({
  children,
  id,
  className,
  components = BASE_COMPONENTS,
  streamingReveal = false,
}: MarkdownProps) {
  const [mathPlugins, setMathPlugins] = useState<{
    rehypeKatex: unknown;
    remarkMath: unknown;
  } | null>(null);
  const generatedId = useId();
  const blockId = id ?? generatedId;
  const normalizedMarkdown = useMemo(
    () => normalizeMathMarkdown(children, { streamingTail: streamingReveal }),
    [children, streamingReveal]
  );
  const needsMathEnhancement = useMemo(
    () => hasMathMarkdownSyntax(normalizedMarkdown),
    [normalizedMarkdown]
  );
  const blocks = useMemo(
    () => parseMarkdownIntoBlocks(normalizedMarkdown),
    [normalizedMarkdown]
  );
  useEffect(() => {
    if (!needsMathEnhancement || mathPlugins) {
      return;
    }
    let cancelled = false;

    void Promise.all([
      import("remark-math"),
      import("rehype-katex"),
      import("katex/dist/katex.min.css"),
    ])
      .then(([remarkMathModule, rehypeKatexModule]) => {
        if (cancelled) {
          return;
        }
        setMathPlugins({
          remarkMath: remarkMathModule.default,
          rehypeKatex: rehypeKatexModule.default,
        });
      })
      .catch(() => {
        if (!cancelled) {
          setMathPlugins(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mathPlugins, needsMathEnhancement]);
  const remarkPlugins = useMemo<Array<unknown>>(
    () =>
      mathPlugins
        ? [remarkGfm, remarkHighlight, remarkNumericColumnAlign, remarkBreaks, mathPlugins.remarkMath]
        : [remarkGfm, remarkHighlight, remarkNumericColumnAlign, remarkBreaks],
    [mathPlugins]
  );
  const rehypePlugins = useMemo<Array<unknown>>(
    () => (mathPlugins ? [mathPlugins.rehypeKatex] : []),
    [mathPlugins]
  );
  const streamingRehypePlugins = useMemo<Array<unknown>>(
    () => [...rehypePlugins, rehypeStreamingTextReveal],
    [rehypePlugins]
  );
  const pluginKey = mathPlugins ? "math" : "base";

  return (
    <div className={cn("pk-markdown", className)}>
      {blocks.map((block, index) => {
        const revealsStreamingTail =
          streamingReveal &&
          index === blocks.length - 1 &&
          block.length <= STREAMING_REVEAL_MAX_BLOCK_CHARACTERS &&
          !hasMathMarkdownSyntax(block);
        return (
          <MemoizedMarkdownBlock
            key={`${blockId}-block-${index}`}
            content={block}
            components={components}
            remarkPlugins={remarkPlugins}
            rehypePlugins={
              revealsStreamingTail ? streamingRehypePlugins : rehypePlugins
            }
            pluginKey={`${pluginKey}-${revealsStreamingTail ? "streaming" : "stable"}`}
          />
        );
      })}
    </div>
  );
}

export const Markdown = memo(MarkdownComponent);
Markdown.displayName = "Markdown";
