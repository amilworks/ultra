import type { Highlighter } from "shiki/bundle/web";

let highlighterPromise: Promise<Highlighter> | null = null;
let shikiBundlePromise: Promise<typeof import("shiki/bundle/web")> | null = null;

const preferredLanguageIds = [
  "text",
  "plaintext",
  "bash",
  "shellscript",
  "python",
  "json",
  "jsonc",
  "yaml",
  "toml",
  "csv",
  "typescript",
  "javascript",
  "tsx",
  "jsx",
  "sql",
  "r",
  "markdown",
  "html",
  "css",
  "xml",
  "diff",
  "ini",
  "dockerfile",
] as const;

const loadShikiBundle = async (): Promise<typeof import("shiki/bundle/web")> => {
  if (!shikiBundlePromise) {
    shikiBundlePromise = import("shiki/bundle/web");
  }
  return shikiBundlePromise;
};

const getHighlighter = async (): Promise<Highlighter> => {
  if (!highlighterPromise) {
    highlighterPromise = loadShikiBundle().then(
      ({ bundledLanguages, createHighlighter }) => {
        const supportedLanguageIds = preferredLanguageIds.filter(
          (language): language is (typeof preferredLanguageIds)[number] =>
            language in bundledLanguages
        );
        return createHighlighter({
          themes: ["github-light", "github-dark"],
          langs: supportedLanguageIds,
        });
      }
    );
  }
  return highlighterPromise;
};

export const codeToHtml = async ({
  code,
  lang,
  theme,
}: {
  code: string;
  lang: string;
  theme: "github-light" | "github-dark";
}): Promise<string> => {
  const highlighter = await getHighlighter();
  const source = code || "";
  if (!source) {
    return "<pre><code></code></pre>";
  }
  return highlighter.codeToHtml(source, {
    lang,
    theme,
  });
};
