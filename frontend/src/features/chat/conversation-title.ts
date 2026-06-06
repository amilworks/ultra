const TITLE_STOP_WORDS = new Set([
  "a",
  "about",
  "also",
  "analyze",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "build",
  "by",
  "calculate",
  "can",
  "compare",
  "compute",
  "could",
  "create",
  "durable",
  "explain",
  "find",
  "for",
  "from",
  "generate",
  "how",
  "i",
  "in",
  "into",
  "is",
  "it",
  "latest",
  "list",
  "make",
  "me",
  "of",
  "on",
  "or",
  "our",
  "please",
  "produce",
  "real",
  "run",
  "show",
  "that",
  "the",
  "this",
  "to",
  "train",
  "using",
  "visualize",
  "we",
  "what",
  "with",
  "would",
  "write",
  "you",
]);

const normalizeTitleWord = (word: string): string => {
  const trimmed = word.replace(/^[^\w]+|[^\w]+$/g, "");
  if (!trimmed) {
    return "";
  }
  if (/[A-Z]/.test(trimmed.slice(1)) || /\d|[.+/#-]/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1).toLowerCase()}`;
};

export const fallbackConversationTitleFromText = (value: string, maxWords = 6): string => {
  const singleLine = value.replace(/\s+/g, " ").trim().replace(/^["'`]+|["'`]+$/g, "");
  if (!singleLine) {
    return "New conversation";
  }
  const candidates = singleLine
    .match(/[A-Za-z0-9][A-Za-z0-9.+/#-]*/g)
    ?.map(normalizeTitleWord)
    .filter(Boolean) ?? [];
  const keywords = candidates.filter(
    (word) =>
      !TITLE_STOP_WORDS.has(word.toLowerCase()) &&
      (word.length > 1 || /^[A-Z]$/.test(word) || /\d/.test(word))
  );
  const words = (keywords.length > 0 ? keywords : candidates).slice(0, Math.max(1, maxWords));
  const title = words.join(" ").trim();
  if (!title) {
    return "New conversation";
  }
  return title.length <= 52 ? title : `${title.slice(0, 51)}...`;
};

export const resolveConversationTitle = (
  storedTitle: string | null | undefined,
  fallbackSeed: string
): string => {
  const normalized = String(storedTitle || "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^["'`]+|["'`]+$/g, "");
  const resolved =
    normalized && normalized !== "New conversation"
      ? normalized
      : fallbackConversationTitleFromText(fallbackSeed);
  if (resolved.length <= 120) {
    return resolved;
  }
  return `${resolved.slice(0, 119)}...`;
};
