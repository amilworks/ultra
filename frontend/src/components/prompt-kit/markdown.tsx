import { marked } from "marked";
import { memo, type ReactNode, useEffect, useId, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { CodeBlock, CodeBlockCode } from "./code-block";

export type MarkdownProps = {
  children: string;
  id?: string;
  className?: string;
  components?: Partial<Components>;
};

function parseMarkdownIntoBlocks(markdown: string): string[] {
  const tokens = marked.lexer(markdown);
  return tokens.map((token) => token.raw);
}

const normalizeMathMarkdown = (source: string): string => {
  let normalized = source;

  // Normalize only explicit TeX delimiters so we do not accidentally turn
  // ordinary bracketed prose into math.
  normalized = normalized.replace(
    /\\\[([\s\S]*?)\\\]/g,
    (_match, expr: string) => `\n$$\n${expr.trim()}\n$$\n`
  );
  normalized = normalized.replace(
    /\\\((.+?)\\\)/g,
    (_match, expr: string) => `$${String(expr).trim()}$`
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

function tableAlignClass(align?: string): string {
  const normalized = String(align || "").toLowerCase();
  if (normalized === "center") return "text-center";
  if (normalized === "right") return "text-right";
  return "text-left";
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
};

const decodeSafe = (value: string): string => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const resolveBisqueLinkMeta = (href: string): BisqueLinkMeta | null => {
  let parsed: URL;
  try {
    parsed = new URL(href);
  } catch {
    return null;
  }

  const path = parsed.pathname;
  const origin = `${parsed.protocol}//${parsed.host}`;
  let resourceUri: string | null = null;

  if (/\/client_service\/view$/i.test(path)) {
    const resourceRaw = parsed.searchParams.get("resource");
    if (!resourceRaw) {
      return null;
    }
    resourceUri = decodeSafe(resourceRaw);
  } else if (/\/data_service\//i.test(path)) {
    resourceUri = parsed.toString();
  } else if (/\/image_service\//i.test(path)) {
    resourceUri = parsed.toString().replace("/image_service/", "/data_service/");
  } else {
    return null;
  }

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
  const [previewFailed, setPreviewFailed] = useState(false);

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
        Open viewer
      </a>
      <span className="bisque-link-hover-preview" role="tooltip">
        {!previewFailed && bisqueMeta.imageServiceUrl ? (
          <img
            src={bisqueMeta.imageServiceUrl}
            alt="BisQue preview"
            loading="lazy"
            onError={() => setPreviewFailed(true)}
          />
        ) : (
          <span>Preview unavailable</span>
        )}
      </span>
    </span>
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
    return (
      <CodeBlock className={className}>
        <CodeBlockCode code={String(children)} language={language} />
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
  th: function TableHeaderCellComponent({
    className,
    children,
    align,
    ...props
  }) {
    const shouldConstrain = shouldConstrainTableCell(children);
    return (
      <th
        className={cn(
          "pk-table-head-cell",
          shouldConstrain && "pk-table-cell-long",
          tableAlignClass(align),
          className
        )}
        {...props}
      >
        <span className="pk-table-cell-content">{children}</span>
      </th>
    );
  },
  td: function TableCellComponent({ className, children, align, ...props }) {
    const shouldConstrain = shouldConstrainTableCell(children);
    return (
      <td
        className={cn(
          "pk-table-cell",
          shouldConstrain && "pk-table-cell-long",
          tableAlignClass(align),
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
    pluginKey,
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
        ? [remarkGfm, remarkBreaks, mathPlugins.remarkMath]
        : [remarkGfm, remarkBreaks],
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
