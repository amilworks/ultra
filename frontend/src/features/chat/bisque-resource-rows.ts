import type { ToolResourceRow } from "@/components/chat/ToolResultCards";

// Single BisQue resource-row dedupe (previously three drifting Map-dedupe
// copies). Key cascade: canonical resource URI, then client view URL, then raw
// URI, then a name|owner|created|type fingerprint. First occurrence wins;
// `limit` optionally caps the merged list (card surfaces use 12).
export const dedupeBisqueResourceRows = (
  rows: ToolResourceRow[],
  limit?: number
): ToolResourceRow[] => {
  const merged = new Map<string, ToolResourceRow>();
  rows.forEach((row) => {
    const key =
      row.resourceUri?.toLowerCase() ||
      row.clientViewUrl?.toLowerCase() ||
      row.uri?.toLowerCase() ||
      `${row.name.toLowerCase()}|${String(row.owner ?? "").toLowerCase()}|${String(
        row.created ?? ""
      ).toLowerCase()}|${String(row.resourceType ?? "").toLowerCase()}`;
    if (!merged.has(key)) {
      merged.set(key, row);
    }
  });
  const deduped = Array.from(merged.values());
  return typeof limit === "number" ? deduped.slice(0, limit) : deduped;
};
