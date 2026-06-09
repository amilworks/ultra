import type { ResourceMetadataFilter } from "@/types";

const METADATA_FILTER_OPERATORS = new Set(["eq", "contains", "exists", "lt", "lte", "gt", "gte"]);

export const parseResourceMetadataFilterInput = (value: string): ResourceMetadataFilter[] => {
  const filters: ResourceMetadataFilter[] = [];
  value
    .split(/[,\n]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .forEach((token) => {
      const [rawPath, rawOperator, ...rawValueParts] = token.split(":");
      const path = String(rawPath ?? "").trim();
      const operator = String(rawOperator ?? "").trim().toLowerCase();
      const rawValue = rawValueParts.join(":");
      const value = rawValue.trim();
      if (!path || !operator || !METADATA_FILTER_OPERATORS.has(operator)) {
        return;
      }
      if (operator !== "exists" && !value) {
        return;
      }
      filters.push({ path, operator, value });
    });
  return filters;
};
