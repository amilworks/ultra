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
