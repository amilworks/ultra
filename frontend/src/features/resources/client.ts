import type { ApiClient } from "@/lib/api";
import type { ResourceKindFilter, ResourceSourceFilter } from "@/components/ResourceBrowser";
import type { ResourceListResponse } from "@/types";

export type ResourceLibraryQuery = {
  limit?: number;
  offset?: number;
  query?: string;
  kind?: ResourceKindFilter;
  source?: ResourceSourceFilter;
};

export const loadLibraryResources = (
  apiClient: ApiClient,
  query: ResourceLibraryQuery = {}
): Promise<ResourceListResponse> =>
  apiClient.listResources({
    limit: query.limit ?? 500,
    offset: query.offset ?? 0,
    query: query.query,
    kind: query.kind === "all" ? undefined : query.kind,
    source: query.source === "all" ? undefined : query.source,
  });

export const loadComposerResources = (
  apiClient: ApiClient,
  query: Pick<ResourceLibraryQuery, "limit" | "query"> = {}
): Promise<ResourceListResponse> =>
  apiClient.listResources({
    limit: query.limit ?? 200,
    query: query.query,
  });
