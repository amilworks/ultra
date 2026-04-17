import {
  Braces,
  Check,
  Code2,
  Copy,
  SquareTerminal,
  type LucideIcon,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import {
  isShellLikeLanguage,
  normalizeCodeForClipboard,
  normalizeCodeForDisplay,
  writeClipboardText,
} from "@/lib/clipboard";
import { cn } from "@/lib/utils";

export type CodeBlockProps = {
  children?: React.ReactNode;
  className?: string;
} & React.HTMLProps<HTMLDivElement>;

function CodeBlock({ children, className, ...props }: CodeBlockProps) {
  return (
    <div
      className={cn(
        "not-prose group/codeblock flex w-full flex-col overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--bg-panel-strong)] shadow-[var(--shadow)]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export type CodeBlockCodeProps = {
  code: string;
  language?: string;
  theme?: "auto" | "github-light" | "github-dark";
  showToolbar?: boolean;
  showLanguage?: boolean;
  showCopyButton?: boolean;
  className?: string;
} & React.HTMLProps<HTMLDivElement>;

const formatLanguageLabel = (language: string): string => {
  const trimmed = language.trim().toLowerCase();
  if (!trimmed) {
    return "Plain text";
  }
  const canonicalLabels: Record<string, string> = {
    bash: "Bash",
    console: "Terminal",
    csv: "CSV",
    css: "CSS",
    dockerfile: "Dockerfile",
    html: "HTML",
    ini: "INI",
    javascript: "JavaScript",
    json: "JSON",
    jsonc: "JSONC",
    jsx: "JSX",
    markdown: "Markdown",
    plaintext: "Plain text",
    shell: "Shell",
    shellscript: "Shell",
    sh: "Shell",
    sql: "SQL",
    text: "Plain text",
    toml: "TOML",
    tsx: "TSX",
    typescript: "TypeScript",
    xml: "XML",
    yaml: "YAML",
    yml: "YAML",
    zsh: "Zsh",
  };
  if (canonicalLabels[trimmed]) {
    return canonicalLabels[trimmed];
  }
  return trimmed
    .split(/[-_]/g)
    .map((chunk) => (chunk ? chunk[0].toUpperCase() + chunk.slice(1) : chunk))
    .join(" ");
};

const bracesLanguageIds = new Set([
  "csv",
  "dockerfile",
  "ini",
  "json",
  "jsonc",
  "markdown",
  "toml",
  "xml",
  "yaml",
  "yml",
]);

type LanguageMeta = {
  copyLabel: string;
  detail: string;
  icon: LucideIcon;
  label: string;
};

const buildLanguageMeta = (
  language: string,
  lineCount: number
): LanguageMeta => {
  const normalized = language.trim().toLowerCase();
  const label = formatLanguageLabel(language);
  if (isShellLikeLanguage(normalized)) {
    return {
      copyLabel: "Copy command",
      detail: "Terminal-safe copy",
      icon: SquareTerminal,
      label,
    };
  }
  return {
    copyLabel: "Copy code",
    detail:
      lineCount > 0 ? `${lineCount} ${lineCount === 1 ? "line" : "lines"}` : "Empty snippet",
    icon: bracesLanguageIds.has(normalized) ? Braces : Code2,
    label,
  };
};

const escapeHtml = (input: string): string =>
  input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const useResolvedTheme = (): "light" | "dark" => {
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(() => {
    if (typeof document === "undefined") {
      return "light";
    }
    return document.documentElement.classList.contains("dark") ? "dark" : "light";
  });

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const updateTheme = () => {
      setResolvedTheme(
        document.documentElement.classList.contains("dark") ? "dark" : "light"
      );
    };
    updateTheme();

    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onMediaChange = () => updateTheme();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", onMediaChange);
    } else {
      mediaQuery.addListener(onMediaChange);
    }

    return () => {
      observer.disconnect();
      if (typeof mediaQuery.removeEventListener === "function") {
        mediaQuery.removeEventListener("change", onMediaChange);
      } else {
        mediaQuery.removeListener(onMediaChange);
      }
    };
  }, []);

  return resolvedTheme;
};

function CodeBlockCode({
  code,
  language = "text",
  theme = "auto",
  showToolbar = true,
  showLanguage = true,
  showCopyButton = true,
  className,
  ...props
}: CodeBlockCodeProps) {
  const resolvedTheme = useResolvedTheme();
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null);
  const [isCopied, setIsCopied] = useState(false);

  const codeText = useMemo(() => String(code ?? ""), [code]);
  const displayCodeText = useMemo(
    () => normalizeCodeForDisplay(codeText),
    [codeText]
  );
  const languageText = useMemo(() => String(language || "text"), [language]);
  const lineCount = useMemo(
    () => (displayCodeText ? displayCodeText.split("\n").length : 0),
    [displayCodeText]
  );
  const languageMeta = useMemo(
    () => buildLanguageMeta(languageText, lineCount),
    [languageText, lineCount]
  );
  const themeName = useMemo<"github-light" | "github-dark">(() => {
    if (theme === "github-light" || theme === "github-dark") {
      return theme;
    }
    return resolvedTheme === "dark" ? "github-dark" : "github-light";
  }, [resolvedTheme, theme]);
  const LanguageIcon = languageMeta.icon;

  useEffect(() => {
    let cancelled = false;
    const source = displayCodeText;

    const renderHighlight = async () => {
      if (!source) {
        if (!cancelled) {
          setHighlightedHtml("<pre><code></code></pre>");
        }
        return;
      }
      try {
        const { codeToHtml } = await import("@/lib/shiki");
        const html = await codeToHtml({
          code: source,
          lang: languageText,
          theme: themeName,
        });
        if (!cancelled) {
          setHighlightedHtml(html);
        }
        return;
      } catch {
        try {
          const { codeToHtml } = await import("@/lib/shiki");
          const fallback = await codeToHtml({
            code: source,
            lang: "text",
            theme: themeName,
          });
          if (!cancelled) {
            setHighlightedHtml(fallback);
          }
        } catch {
          if (!cancelled) {
            setHighlightedHtml(`<pre><code>${escapeHtml(source)}</code></pre>`);
          }
        }
      }
    };

    void renderHighlight();
    return () => {
      cancelled = true;
    };
  }, [displayCodeText, languageText, themeName]);

  const onCopy = async () => {
    const source = normalizeCodeForClipboard(codeText, languageText);
    if (!source) {
      return;
    }
    try {
      await writeClipboardText(source);
      setIsCopied(true);
      window.setTimeout(() => setIsCopied(false), 1500);
    } catch {
      setIsCopied(false);
    }
  };

  const contentClassName = cn(
    "pk-code-render w-full overflow-auto overscroll-contain text-[13px] leading-[1.2]",
    "[&_pre]:m-0 [&_pre]:min-w-full",
    "[&_code]:font-[\"JetBrains_Mono\",\"SF_Mono\",\"Menlo\",monospace] [&_code]:text-[13px] [&_code]:leading-[1.2]",
    className
  );

  return (
    <div className="relative">
      {(showToolbar || showCopyButton) ? (
        <div className="flex items-center justify-between gap-3 border-b border-[color-mix(in_oklab,var(--line)_88%,transparent)] bg-[color-mix(in_oklab,var(--bg-panel)_94%,transparent)] px-4 py-3">
          {showLanguage ? (
            <div className="flex min-w-0 items-center gap-3">
              <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-[0.95rem] border border-[var(--line)] bg-[color-mix(in_oklab,var(--bg-main)_72%,var(--bg-panel-strong)_28%)] text-[var(--text-main)] shadow-[inset_0_1px_0_rgba(255,255,255,0.18)]">
                <LanguageIcon className="h-4 w-4" />
              </span>
              <div className="min-w-0">
                <span className="block truncate text-sm font-medium text-[var(--text-main)]">
                  {languageMeta.label}
                </span>
                <span className="block truncate text-[11px] text-[var(--text-muted)]">
                  {languageMeta.detail}
                </span>
              </div>
            </div>
          ) : (
            <span />
          )}
          {showCopyButton ? (
            <button
              type="button"
              aria-label={languageMeta.copyLabel}
              title={languageMeta.copyLabel}
              onClick={onCopy}
              className={cn(
                "inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-[var(--line)] bg-[color-mix(in_oklab,var(--bg-panel-strong)_84%,var(--bg-main)_16%)] px-3 text-[12px] font-medium text-[var(--text-muted)] transition-all duration-200 hover:border-[color-mix(in_oklab,var(--text-main)_18%,var(--line))] hover:bg-[color-mix(in_oklab,var(--bg-panel-strong)_94%,var(--bg-main)_6%)] hover:text-[var(--text-main)]",
                isCopied &&
                  "border-emerald-500/25 bg-emerald-500/10 text-emerald-600 hover:border-emerald-500/30 hover:bg-emerald-500/15 dark:text-emerald-400"
              )}
            >
              {isCopied ? (
                <Check className="h-3.5 w-3.5 animate-in zoom-in-50 fade-in-0 duration-200" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              <span className="hidden sm:inline">{isCopied ? "Copied" : "Copy"}</span>
            </button>
          ) : null}
        </div>
      ) : null}
      {highlightedHtml ? (
        <div
          className={contentClassName}
          dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          {...props}
        />
      ) : (
        <div className={contentClassName} {...props}>
          <pre>
            <code>{displayCodeText}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

export type CodeBlockGroupProps = React.HTMLAttributes<HTMLDivElement>;

function CodeBlockGroup({
  children,
  className,
  ...props
}: CodeBlockGroupProps) {
  return (
    <div
      className={cn("flex items-center justify-between", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export { CodeBlockGroup, CodeBlockCode, CodeBlock };
