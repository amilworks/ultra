import type { ResourceRecord } from "@/types";

const WORKSPACE_PATH_PATTERN =
  /(?:^|[/\\])(?:workspace|outputs|artifacts|data[/\\](?:artifacts|uploads)|deepagents)(?:[/\\]|$)/i;

const stripPathSegments = (value: string): string => {
  const normalized = String(value || "").trim().replace(/\\/g, "/");
  const parts = normalized.split("/").filter((part) => part.length > 0);
  return parts.length > 0 ? parts[parts.length - 1] : normalized;
};

export const hasInternalResourcePath = (value: string | null | undefined): boolean =>
  WORKSPACE_PATH_PATTERN.test(String(value ?? "").trim());

export const resourceDisplayName = (resource: Pick<ResourceRecord, "original_name" | "file_id">): string => {
  const originalName = stripPathSegments(resource.original_name);
  if (originalName && !hasInternalResourcePath(originalName)) {
    return originalName;
  }
  const fileId = String(resource.file_id || "").trim();
  return fileId ? `Resource ${fileId.slice(0, 12)}` : "Untitled resource";
};

export const resourceOriginLabel = (
  resource: Pick<ResourceRecord, "source_type" | "resource_kind" | "source_uri" | "client_view_url">
): string => {
  const sourceType = String(resource.source_type || "").trim().toLowerCase();
  const kind = String(resource.resource_kind || "file").trim().toLowerCase() || "file";
  if (sourceType === "bisque_import") {
    return `Imported BisQue ${kind}`;
  }
  if (sourceType === "upload") {
    return `Uploaded ${kind}`;
  }
  if (sourceType) {
    return `${sourceType.replace(/_/g, " ")} ${kind}`;
  }
  if (resource.client_view_url || resource.source_uri) {
    return `Linked ${kind}`;
  }
  return "Managed resource";
};

/** Chat-pasted text uploads: `pasted-YYYY-MM-DD-HHMMSS-mmm.txt` (see
 * lib/pasted-text.ts; the millisecond suffix is optional because files from
 * before that fix still exist in the catalog). These are the one class of
 * name where the filename carries no identity at all — every card reads
 * identically and the embedded timestamp duplicates the date row — so the
 * browser swaps in a title derived from the file's own first words. */
const PASTED_TEXT_NAME_PATTERN = /^pasted-\d{4}-\d{2}-\d{2}-\d{6}(?:-\d{3})?\.txt$/i;

export const isPastedTextName = (name: string): boolean =>
  PASTED_TEXT_NAME_PATTERN.test(String(name ?? "").trim());

const PASTED_TITLE_MAX_LENGTH = 72;
const PASTED_TITLE_SCAN_LINES = 12;

/** First content line of a pasted file's text head, de-markdowned into a card
 * title. Returns null when nothing usable surfaces in the opening lines
 * (fences, dividers, bare TeX) — callers fall back to the filename. */
export const derivePastedTitle = (headText: string): string | null => {
  const lines = String(headText ?? "").split(/\r?\n/).slice(0, PASTED_TITLE_SCAN_LINES);
  for (const rawLine of lines) {
    let line = rawLine.trim();
    if (!line) {
      continue;
    }
    // Fence and divider lines carry no words.
    if (/^(?:```|~~~|\$\$|-{3,}|={3,}|\*{3,})/.test(line)) {
      continue;
    }
    // Strip ONE block marker — a line has a single block role, so "### 1. Foo"
    // keeps its meaningful "1." after the heading marker goes — then unwrap
    // emphasis/code spans to their inner text. Quote markers strip first and
    // repeatedly ("> > - item" is a quoted list item).
    line = line.replace(/^(?:>\s*)+/, "");
    const heading = line.replace(/^#{1,6}\s+/, "");
    if (heading !== line) {
      line = heading;
    } else {
      line = line
        .replace(/^(?:[-*+]|\d{1,3}[.)])\s+/, "")
        .replace(/^\[[ xX]\]\s+/, "");
    }
    line = line
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/\*([^*]+)\*/g, "$1")
      .replace(/==([^=]+)==/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .trim();
    if (!line || line.startsWith("\\")) {
      // Empty after stripping, or a line of bare TeX — not a title.
      continue;
    }
    if (line.length > PASTED_TITLE_MAX_LENGTH) {
      const cut = line.slice(0, PASTED_TITLE_MAX_LENGTH);
      const lastSpace = cut.lastIndexOf(" ");
      line = `${(lastSpace > 40 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
    }
    return line;
  }
  return null;
};

export type ResourceDateSection<T> = { label: string; items: T[] };

/** Buckets speak the sidebar history's calendar language — Today / Yesterday /
 * Last 7 days — then months (bare for the current year, with the year once it
 * differs), so the two surfaces group time the same way. Items arrive
 * date-sorted from the API; grouping preserves their order and only merges
 * ADJACENT runs, never re-sorts. */
export const groupResourcesByDateSection = <T extends { created_at: string }>(
  items: readonly T[],
  now: Date = new Date()
): Array<ResourceDateSection<T>> => {
  const dayMs = 24 * 60 * 60 * 1000;
  const startOfDay = (date: Date): number =>
    new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const today = startOfDay(now);
  const sections: Array<ResourceDateSection<T>> = [];
  for (const item of items) {
    const created = new Date(item.created_at);
    const createdDay = Number.isNaN(created.getTime())
      ? Number.NaN
      : startOfDay(created);
    let label: string;
    if (Number.isNaN(createdDay)) {
      label = "Undated";
    } else if (createdDay >= today) {
      // A future-dated clock skew lands in Today rather than a nonsense bucket.
      label = "Today";
    } else if (createdDay >= today - dayMs) {
      label = "Yesterday";
    } else if (createdDay >= today - 7 * dayMs) {
      label = "Last 7 days";
    } else {
      label = created.toLocaleDateString(
        [],
        created.getFullYear() === now.getFullYear()
          ? { month: "long" }
          : { month: "long", year: "numeric" }
      );
    }
    const last = sections[sections.length - 1];
    if (last && last.label === label) {
      last.items.push(item);
    } else {
      sections.push({ label, items: [item] });
    }
  }
  return sections;
};

const QUERY_MATCH_LIMIT = 20;

/** Case-insensitive occurrences of the trimmed query in `text`, as index
 * ranges. Pure so the browser's JSX wrapper stays a dumb renderer. */
export const findQueryMatches = (
  text: string,
  query: string
): Array<{ start: number; end: number }> => {
  const needle = String(query ?? "").trim().toLowerCase();
  if (!needle) {
    return [];
  }
  const haystack = String(text ?? "").toLowerCase();
  const ranges: Array<{ start: number; end: number }> = [];
  let from = 0;
  while (ranges.length < QUERY_MATCH_LIMIT) {
    const at = haystack.indexOf(needle, from);
    if (at === -1) {
      break;
    }
    ranges.push({ start: at, end: at + needle.length });
    from = at + needle.length;
  }
  return ranges;
};
