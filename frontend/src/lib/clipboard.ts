const SHELL_LANGUAGE_IDS = new Set([
  "bash",
  "console",
  "shell",
  "shellscript",
  "sh",
  "terminal",
  "zsh",
]);

const SHELL_PROMPT_PREFIX = /^(\s*)(?:[$>%]|\.{3})\s/;

const normalizeLineEndings = (value: string): string =>
  value.replace(/\r\n?/g, "\n");

const stripShellPrompts = (value: string): string => {
  const lines = value.split("\n");
  const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
  if (
    nonEmptyLines.length === 0 ||
    !nonEmptyLines.every((line) => SHELL_PROMPT_PREFIX.test(line))
  ) {
    return value;
  }
  return lines
    .map((line) =>
      line.trim().length === 0 ? line : line.replace(SHELL_PROMPT_PREFIX, "$1")
    )
    .join("\n");
};

export const isShellLikeLanguage = (language?: string): boolean =>
  SHELL_LANGUAGE_IDS.has(String(language || "").trim().toLowerCase());

export const normalizeCodeForDisplay = (code: string): string =>
  normalizeLineEndings(String(code ?? "")).replace(/\n+$/, "");

export const normalizeCodeForClipboard = (
  code: string,
  language?: string
): string => {
  const normalized = normalizeLineEndings(String(code ?? ""));
  if (!isShellLikeLanguage(language)) {
    return normalized;
  }
  return stripShellPrompts(normalized).replace(/\n+$/, "");
};

export async function writeClipboardText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  if (typeof document === "undefined" || !document.body) {
    throw new Error("Clipboard unavailable");
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  if (!copied) {
    throw new Error("Clipboard unavailable");
  }
}
