import { createBundledHighlighter } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

const languageLoaders = {
  bash: () => import("shiki/langs/bash"),
  csv: () => import("shiki/langs/csv"),
  css: () => import("shiki/langs/css"),
  diff: () => import("shiki/langs/diff"),
  dockerfile: () => import("shiki/langs/dockerfile"),
  html: () => import("shiki/langs/html"),
  ini: () => import("shiki/langs/ini"),
  javascript: () => import("shiki/langs/javascript"),
  json: () => import("shiki/langs/json"),
  jsx: () => import("shiki/langs/jsx"),
  markdown: () => import("shiki/langs/markdown"),
  python: () => import("shiki/langs/python"),
  r: () => import("shiki/langs/r"),
  sql: () => import("shiki/langs/sql"),
  toml: () => import("shiki/langs/toml"),
  tsx: () => import("shiki/langs/tsx"),
  typescript: () => import("shiki/langs/typescript"),
  xml: () => import("shiki/langs/xml"),
  yaml: () => import("shiki/langs/yaml"),
} as const;

type SupportedLanguageId = keyof typeof languageLoaders;

const languageAliases: Record<string, SupportedLanguageId | "text"> = {
  console: "bash",
  js: "javascript",
  jsonc: "json",
  plaintext: "text",
  py: "python",
  shell: "bash",
  shellscript: "bash",
  sh: "bash",
  text: "text",
  ts: "typescript",
  yml: "yaml",
  zsh: "bash",
};

const createAppHighlighter = createBundledHighlighter({
  engine: createJavaScriptRegexEngine,
  langs: languageLoaders,
  themes: {
    "github-dark": () => import("shiki/themes/github-dark"),
    "github-light": () => import("shiki/themes/github-light"),
  },
});

type Highlighter = Awaited<ReturnType<typeof createAppHighlighter>>;

let highlighterPromise: Promise<Highlighter> | null = null;

const preferredLanguageIds = Object.keys(languageLoaders) as SupportedLanguageId[];

const resolveLanguage = (language: string): SupportedLanguageId | "text" => {
  const normalized = language.trim().toLowerCase();
  if (!normalized) {
    return "text";
  }
  if (normalized in languageLoaders) {
    return normalized as SupportedLanguageId;
  }
  return languageAliases[normalized] ?? "text";
};

const getHighlighter = async (): Promise<Highlighter> => {
  if (highlighterPromise) {
    return highlighterPromise;
  }
  highlighterPromise = createAppHighlighter({
    langAlias: languageAliases,
    langs: preferredLanguageIds,
    themes: ["github-light", "github-dark"],
  });
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
    lang: resolveLanguage(lang),
    theme,
  });
};
