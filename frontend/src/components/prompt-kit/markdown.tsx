import { marked } from "marked";
import { ExternalLink } from "lucide-react";
import { lazy, memo, type ReactNode, Suspense, useEffect, useId, useMemo, useState } from "react";
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
import { Button } from "@/components/ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { DEFAULT_BISQUE_BROWSER_URL } from "@/lib/config";
import { cn } from "@/lib/utils";
import { reportClientError } from "@/lib/client-diagnostics";
import { CodeBlock, CodeBlockCode } from "./code-block";

// Lazy so recharts (bundled) only loads when a chart actually appears in chat.
const ChatChart = lazy(() => import("@/components/chat/ChatChart"));

export type MarkdownProps = {
  children: string;
  id?: string;
  className?: string;
  components?: Partial<Components>;
};

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
const BARE_ENVIRONMENT_PATTERN = new RegExp(
  String.raw`(?<![$\\])\\begin\{(${MATH_ENVIRONMENTS})\}([\s\S]*?)\\end\{\1\}`,
  "g"
);

export const normalizeMathMarkdown = (source: string): string => {
  let normalized = source;

  // Normalize only explicit TeX delimiters so we do not accidentally turn
  // ordinary bracketed prose into math. Strip leading blockquote markers from the
  // captured display body: models frequently emit `> \[ ... > \]`, and leaving the
  // `> ` prefixes inside the converted `$$ ... $$` block makes remark-math fail to
  // parse it, so the equation renders as raw LaTeX instead of math.
  normalized = normalized.replace(
    /\\\[([\s\S]*?)\\\]/g,
    (_match, expr: string) =>
      `\n$$\n${String(expr).replace(/^[ \t]*>[ \t]?/gm, "").trim()}\n$$\n`
  );
  normalized = normalized.replace(
    /\\\((.+?)\\\)/g,
    (_match, expr: string) => `$${String(expr).replace(/^[ \t]*>[ \t]?/gm, "").trim()}$`
  );

  // Auto-fence bare display environments, but skip any that already sit inside
  // an open `$$ ... $$` block (an odd number of `$$` before the match means we
  // are inside one) so we never double-wrap already-delimited math.
  normalized = normalized.replace(
    BARE_ENVIRONMENT_PATTERN,
    (match, _env: string, _body: string, offset: number, full: string) => {
      const dollarDollarBefore = (full.slice(0, offset).match(/\$\$/g) ?? [])
        .length;
      if (dollarDollarBefore % 2 === 1) {
        return match;
      }
      return `\n\n$$\n${match}\n$$\n\n`;
    }
  );

  return normalized;
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

const BASE_COMPONENTS: Partial<Components> = {
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
}: MarkdownProps) {
  const [mathPlugins, setMathPlugins] = useState<{
    rehypeKatex: unknown;
    remarkMath: unknown;
  } | null>(null);
  const generatedId = useId();
  const blockId = id ?? generatedId;
  const normalizedMarkdown = useMemo(
    () => normalizeMathMarkdown(children),
    [children]
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
        ? [remarkGfm, remarkNumericColumnAlign, remarkBreaks, mathPlugins.remarkMath]
        : [remarkGfm, remarkNumericColumnAlign, remarkBreaks],
    [mathPlugins]
  );
  const rehypePlugins = useMemo<Array<unknown>>(
    () => (mathPlugins ? [mathPlugins.rehypeKatex] : []),
    [mathPlugins]
  );
  const pluginKey = mathPlugins ? "math" : "base";

  return (
    <div className={cn("pk-markdown", className)}>
      {blocks.map((block, index) => (
        <MemoizedMarkdownBlock
          key={`${blockId}-block-${index}`}
          content={block}
          components={components}
          remarkPlugins={remarkPlugins}
          rehypePlugins={rehypePlugins}
          pluginKey={pluginKey}
        />
      ))}
    </div>
  );
}

export const Markdown = memo(MarkdownComponent);
Markdown.displayName = "Markdown";
