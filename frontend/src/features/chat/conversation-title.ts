export const fallbackConversationTitleFromText = (value: string, maxWords = 4): string => {
  const singleLine = value.replace(/\s+/g, " ").trim().replace(/^["'`]+|["'`]+$/g, "");
  if (!singleLine) {
    return "New conversation";
  }
  const words = singleLine.split(" ").filter(Boolean).slice(0, Math.max(1, maxWords));
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
      : fallbackConversationTitleFromText(fallbackSeed, 4);
  if (resolved.length <= 120) {
    return resolved;
  }
  return `${resolved.slice(0, 119)}...`;
};
