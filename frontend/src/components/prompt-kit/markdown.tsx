import { marked } from "marked";
import { ExternalLink } from "lucide-react";
import { memo, type ReactNode, useEffect, useId, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
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
  resourceId: string | null;
};

const BISQUE_LINK_FALLBACK_IMAGE_URL = "/bq-bg8.png";

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
