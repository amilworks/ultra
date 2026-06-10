import type { ApiClient } from "@/lib/api";
import type {
  ResourceKindFilter,
  ResourceSharingFilter,
  ResourceSourceFilter,
  ResourceStatusFilter,
} from "@/components/ResourceBrowser";

/**
 * Data Agent processing-status filter accepted by the resource APIs.
 * The default Resources UI no longer surfaces this; it is preserved here for
 * the backend contract and future advanced workflows.
 */
export type ResourceProcessingFilter =
  | "all"
  | "caption_ready"
  | "metadata_ready"
  | "tags_ready"
  | "qc_complete"
  | "dedupe_checked"
  | "organization_ready"
  | "data_agent_ready"
  | "needs_caption"
  | "needs_metadata"
  | "data_agent_failed";
import type {
  DataAgentJobCreateRequest,
  DataAgentJobListResponse,
  DataAgentJobResponse,
  DatasetSnapshotEventListResponse,
  DatasetSnapshotListResponse,
  DatasetSnapshotResponse,
  DatasetSnapshotShareGrantListResponse,
  DatasetSnapshotShareGrantResponse,
  ResourceCollectionListResponse,
  ResourceCollectionRemoveResourcesResponse,
  ResourceCollectionResponse,
  ResourceCollectionShareGrantsCreateResponse,
  ResourceBulkLifecycleResponse,
  ResourceBulkTagResponse,
  ResourceListResponse,
  ResourceMetadataFilter,
  ResourceResponse,
  ResourceShareGrantListResponse,
  ResourceShareGrantResponse,
  ResourceShareGrantsCreateResponse,
} from "@/types";

export type ResourceLibraryQuery = {
  limit?: number;
  offset?: number;
  query?: string;
  kind?: ResourceKindFilter;
  sharing?: ResourceSharingFilter;
  processingStatus?: ResourceProcessingFilter;
  source?: ResourceSourceFilter;
  status?: ResourceStatusFilter;
  tags?: string[];
  descriptors?: string[];
  metadataFilters?: ResourceMetadataFilter[];
  createdAfter?: string;
  createdBefore?: string;
};

export type DataAgentJobQuery = {
  limit?: number;
  offset?: number;
  status?: string;
  jobType?: string;
  projectId?: string;
};

export type DatasetSnapshotQuery = {
  limit?: number;
  offset?: number;
  query?: string;
  projectId?: string;
  sourceCollectionId?: string;
  status?: ResourceStatusFilter | string;
};

export type ResourceDatasetSnapshotQuery = {
  query?: string;
  kind?: ResourceKindFilter;
  source?: ResourceSourceFilter;
  sharing?: ResourceSharingFilter;
  processingStatus?: ResourceProcessingFilter;
  tags?: string[];
  descriptors?: string[];
  metadataFilters?: ResourceMetadataFilter[];
  createdAfter?: string;
  createdBefore?: string;
  projectId?: string;
  sourceCollectionId?: string;
};

export type ResourceDataAgentJobLaunchRequest = {
  jobType: DataAgentJobCreateRequest["job_type"];
  resourceIds?: string[];
  sourceCollectionId?: string | null;
  resourceQuery?: ResourceDatasetSnapshotQuery | null;
  projectId?: string | null;
  inputSelector?: Record<string, unknown> | null;
  tags?: string[];
  snapshotName?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type ResourceShareGrantDraft = {
  granteeUserId?: string | null;
  granteeOrgId?: string | null;
  public?: boolean | null;
  role?: "read" | string;
  metadata?: Record<string, unknown> | null;
};

const uniqueTrimmedStrings = (values: readonly string[] | undefined): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  values?.forEach((value) => {
    const trimmed = String(value ?? "").trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    result.push(trimmed);
  });
  return result;
};

export const loadLibraryResources = (
  apiClient: ApiClient,
  query: ResourceLibraryQuery = {}
): Promise<ResourceListResponse> =>
  apiClient.listResources({
    limit: query.limit ?? 50,
    offset: query.offset ?? 0,
    query: query.query,
    kind: query.kind === "all" ? undefined : query.kind,
    source: query.source === "all" ? undefined : query.source,
    sharing: query.sharing === "all" ? undefined : query.sharing,
    processingStatus: query.processingStatus === "all" ? undefined : query.processingStatus,
    status: query.status,
    tags: uniqueTrimmedStrings(query.tags),
    descriptors: uniqueTrimmedStrings(query.descriptors),
    metadataFilters: query.metadataFilters,
    createdAfter: String(query.createdAfter ?? "").trim() || undefined,
    createdBefore: String(query.createdBefore ?? "").trim() || undefined,
  });

export const loadComposerResources = (
  apiClient: ApiClient,
  query: Pick<ResourceLibraryQuery, "limit" | "query"> = {}
): Promise<ResourceListResponse> =>
  apiClient.listResources({
    limit: query.limit ?? 200,
    query: query.query,
  });

export const loadResourceFolders = (
  apiClient: ApiClient,
  query: Pick<ResourceLibraryQuery, "limit" | "offset" | "query" | "status"> = {}
): Promise<ResourceCollectionListResponse> =>
  apiClient.listResourceCollections({
    limit: query.limit ?? 200,
    offset: query.offset ?? 0,
    query: query.query,
    collectionType: "folder",
    status: query.status,
  });

export const deleteResourceCollection = (
  apiClient: ApiClient,
  collectionId: string
): Promise<ResourceCollectionResponse> => apiClient.deleteResourceCollection(collectionId.trim());

export const renameResourceCollection = (
  apiClient: ApiClient,
  collectionId: string,
  name: string
): Promise<ResourceCollectionResponse> =>
  apiClient.patchResourceCollection(collectionId.trim(), { name: name.trim() });

export const restoreResourceCollection = (
  apiClient: ApiClient,
  collectionId: string
): Promise<ResourceCollectionResponse> => apiClient.restoreResourceCollection(collectionId.trim());

export const loadResourceFolderResources = (
  apiClient: ApiClient,
  collectionId: string,
  query: ResourceLibraryQuery = {}
): Promise<ResourceListResponse> =>
  apiClient.listResourceCollectionResources(collectionId, {
    limit: query.limit ?? 50,
    offset: query.offset ?? 0,
    query: query.query,
    kind: query.kind === "all" ? undefined : query.kind,
    source: query.source === "all" ? undefined : query.source,
    sharing: query.sharing === "all" ? undefined : query.sharing,
    processingStatus: query.processingStatus === "all" ? undefined : query.processingStatus,
    status: query.status,
    tags: uniqueTrimmedStrings(query.tags),
    descriptors: uniqueTrimmedStrings(query.descriptors),
    metadataFilters: query.metadataFilters,
    createdAfter: String(query.createdAfter ?? "").trim() || undefined,
    createdBefore: String(query.createdBefore ?? "").trim() || undefined,
  });

export const loadResourceShareGrants = (
  apiClient: ApiClient,
  resourceId: string,
  query: { limit?: number; status?: string } = {}
): Promise<ResourceShareGrantListResponse> =>
  apiClient.listResourceShareGrants(resourceId.trim(), {
    limit: query.limit ?? 200,
    status: query.status,
  });

export const createResourceShareGrant = (
  apiClient: ApiClient,
  resourceId: string,
  request: ResourceShareGrantDraft
): Promise<ResourceShareGrantResponse> => {
  const granteeUserId = String(request.granteeUserId ?? "").trim();
  const granteeOrgId = String(request.granteeOrgId ?? "").trim();
  return apiClient.createResourceShareGrant(resourceId.trim(), {
    grantee_user_id: granteeUserId || undefined,
    grantee_org_id: granteeOrgId || undefined,
    public: request.public === true ? true : undefined,
    role: request.role ?? "read",
    metadata: {
      ...(request.metadata ?? {}),
      source: "resources_share_panel",
    },
  });
};

export const createBulkResourceShareGrants = (
  apiClient: ApiClient,
  resourceIds: string[],
  request: ResourceShareGrantDraft
): Promise<ResourceShareGrantsCreateResponse> => {
  const normalizedResourceIds = Array.from(
    new Set(resourceIds.map((resourceId) => resourceId.trim()).filter(Boolean))
  );
  const granteeUserId = String(request.granteeUserId ?? "").trim();
  const granteeOrgId = String(request.granteeOrgId ?? "").trim();
  return apiClient.createResourceShareGrants({
    resource_ids: normalizedResourceIds,
    grantee_user_id: granteeUserId || undefined,
    grantee_org_id: granteeOrgId || undefined,
    public: request.public === true ? true : undefined,
    role: request.role ?? "read",
    metadata: {
      ...(request.metadata ?? {}),
      source: "resources_bulk_share_panel",
    },
  });
};

export const createResourceCollectionShareGrants = (
  apiClient: ApiClient,
  collectionId: string,
  request: ResourceShareGrantDraft
): Promise<ResourceCollectionShareGrantsCreateResponse> => {
  const granteeUserId = String(request.granteeUserId ?? "").trim();
  const granteeOrgId = String(request.granteeOrgId ?? "").trim();
  return apiClient.createResourceCollectionShareGrants(collectionId.trim(), {
    grantee_user_id: granteeUserId || undefined,
    grantee_org_id: granteeOrgId || undefined,
    public: request.public === true ? true : undefined,
    role: request.role ?? "read",
    metadata: {
      ...(request.metadata ?? {}),
      source: "resources_folder_share_panel",
    },
  });
};

export const loadDatasetSnapshotShareGrants = (
  apiClient: ApiClient,
  snapshotId: string,
  query: { limit?: number; status?: string } = {}
): Promise<DatasetSnapshotShareGrantListResponse> =>
  apiClient.listDatasetSnapshotShareGrants(snapshotId.trim(), {
    limit: query.limit ?? 200,
    status: query.status,
  });

export const loadDatasetSnapshotEvents = (
  apiClient: ApiClient,
  snapshotId: string,
  query: { limit?: number; offset?: number; eventType?: string; actorUserId?: string } = {}
): Promise<DatasetSnapshotEventListResponse> =>
  apiClient.listDatasetSnapshotEvents(snapshotId.trim(), {
    limit: query.limit ?? 50,
    offset: query.offset,
    eventType: String(query.eventType ?? "").trim() || undefined,
    actorUserId: String(query.actorUserId ?? "").trim() || undefined,
  });

export const loadDatasetSnapshots = (
  apiClient: ApiClient,
  query: DatasetSnapshotQuery = {}
): Promise<DatasetSnapshotListResponse> =>
  apiClient.listDatasetSnapshots({
    limit: query.limit,
    offset: query.offset,
    query: String(query.query ?? "").trim() || undefined,
    projectId: String(query.projectId ?? "").trim() || undefined,
    sourceCollectionId: String(query.sourceCollectionId ?? "").trim() || undefined,
    status: String(query.status ?? "").trim() || undefined,
  });

export const deleteDatasetSnapshot = (
  apiClient: ApiClient,
  snapshotId: string
): Promise<DatasetSnapshotResponse> => apiClient.deleteDatasetSnapshot(snapshotId.trim());

export const restoreDatasetSnapshot = (
  apiClient: ApiClient,
  snapshotId: string
): Promise<DatasetSnapshotResponse> => apiClient.restoreDatasetSnapshot(snapshotId.trim());

export const createDatasetSnapshotShareGrant = (
  apiClient: ApiClient,
  snapshotId: string,
  request: ResourceShareGrantDraft
): Promise<DatasetSnapshotShareGrantResponse> => {
  const granteeUserId = String(request.granteeUserId ?? "").trim();
  const granteeOrgId = String(request.granteeOrgId ?? "").trim();
  return apiClient.createDatasetSnapshotShareGrant(snapshotId.trim(), {
    grantee_user_id: granteeUserId || undefined,
    grantee_org_id: granteeOrgId || undefined,
    role: request.role ?? "read",
    metadata: {
      ...(request.metadata ?? {}),
      source: "resources_dataset_share_panel",
    },
  });
};

export const revokeDatasetSnapshotShareGrant = (
  apiClient: ApiClient,
  snapshotId: string,
  grantId: string
): Promise<DatasetSnapshotShareGrantResponse> =>
  apiClient.revokeDatasetSnapshotShareGrant(snapshotId.trim(), grantId.trim());

export const createBulkResourceTags = (
  apiClient: ApiClient,
  resourceIds: string[],
  tags: string[]
): Promise<ResourceBulkTagResponse> =>
  apiClient.bulkTagResources({
    resource_ids: uniqueTrimmedStrings(resourceIds),
    tags: uniqueTrimmedStrings(tags),
    metadata: {
      source: "resources_bulk_tag_panel",
    },
  });

export const deleteBulkResources = (
  apiClient: ApiClient,
  resourceIds: string[]
): Promise<ResourceBulkLifecycleResponse> =>
  apiClient.deleteResources({
    resource_ids: uniqueTrimmedStrings(resourceIds),
  });

export const restoreBulkResources = (
  apiClient: ApiClient,
  resourceIds: string[]
): Promise<ResourceBulkLifecycleResponse> =>
  apiClient.restoreResources({
    resource_ids: uniqueTrimmedStrings(resourceIds),
  });

export const restoreResource = (
  apiClient: ApiClient,
  resourceId: string
): Promise<ResourceResponse> => apiClient.restoreResource(resourceId.trim());

export const renameResource = (
  apiClient: ApiClient,
  resourceId: string,
  name: string
): Promise<ResourceResponse> => apiClient.renameResource(resourceId.trim(), name.trim());

export const removeResourceFromCollection = (
  apiClient: ApiClient,
  collectionId: string,
  resourceId: string
): Promise<ResourceCollectionRemoveResourcesResponse> =>
  apiClient.removeResourceFromCollection(collectionId.trim(), resourceId.trim());

export const patchResourceMetadata = (
  apiClient: ApiClient,
  resourceId: string,
  metadata: Record<string, unknown>
): Promise<ResourceResponse> =>
  apiClient.patchResourceMetadata(resourceId.trim(), {
    ...metadata,
    edit_source: "resources_metadata_panel",
  });

export const revokeResourceShareGrant = (
  apiClient: ApiClient,
  resourceId: string,
  grantId: string
): Promise<ResourceShareGrantResponse> =>
  apiClient.revokeResourceShareGrant(resourceId.trim(), grantId.trim());

export const loadDataAgentJobs = (
  apiClient: ApiClient,
  query: DataAgentJobQuery = {}
): Promise<DataAgentJobListResponse> =>
  apiClient.listDataAgentJobs({
    limit: query.limit ?? 8,
    offset: query.offset ?? 0,
    status: query.status,
    jobType: query.jobType,
    projectId: query.projectId,
  });

export const startResourceDataAgentJob = (
  apiClient: ApiClient,
  request: ResourceDataAgentJobLaunchRequest
): Promise<DataAgentJobResponse> => {
  const resourceIds = Array.from(
    new Set((request.resourceIds ?? []).map((fileId) => fileId.trim()).filter(Boolean))
  );
  const sourceCollectionId = String(request.sourceCollectionId ?? "").trim() || undefined;
  const projectId = String(request.projectId ?? "").trim() || undefined;
  const tags = uniqueTrimmedStrings(request.tags);
  const snapshotName = String(request.snapshotName ?? "").trim();
  const resourceQueryTags = uniqueTrimmedStrings(request.resourceQuery?.tags);
  const resourceQueryDescriptors = uniqueTrimmedStrings(request.resourceQuery?.descriptors);
  const resourceQueryMetadataFilters = (request.resourceQuery?.metadataFilters ?? [])
    .map((filter) => ({
      path: String(filter.path ?? "").trim(),
      operator: String(filter.operator ?? "").trim().toLowerCase(),
      value: String(filter.value ?? "").trim(),
    }))
    .filter((filter) => filter.path && filter.operator && (filter.operator === "exists" || filter.value));
  const resourceQuery = request.resourceQuery
    ? {
        q: String(request.resourceQuery.query ?? "").trim() || undefined,
        kind:
          request.resourceQuery.kind && request.resourceQuery.kind !== "all"
            ? String(request.resourceQuery.kind).trim()
            : undefined,
        source:
          request.resourceQuery.source && request.resourceQuery.source !== "all"
            ? String(request.resourceQuery.source).trim()
            : undefined,
        sharing:
          request.resourceQuery.sharing && request.resourceQuery.sharing !== "all"
            ? String(request.resourceQuery.sharing).trim()
            : undefined,
        processing_status:
          request.resourceQuery.processingStatus && request.resourceQuery.processingStatus !== "all"
            ? String(request.resourceQuery.processingStatus).trim()
            : undefined,
        project_id: String(request.resourceQuery.projectId ?? "").trim() || undefined,
        tags: resourceQueryTags.length > 0 ? resourceQueryTags : undefined,
        descriptors: resourceQueryDescriptors.length > 0 ? resourceQueryDescriptors : undefined,
        metadata_filters:
          resourceQueryMetadataFilters.length > 0 ? resourceQueryMetadataFilters : undefined,
        created_after: String(request.resourceQuery.createdAfter ?? "").trim() || undefined,
        created_before: String(request.resourceQuery.createdBefore ?? "").trim() || undefined,
      }
    : null;
  const hasResourceQuery = Boolean(
    resourceQuery &&
      (resourceQuery.q ||
        resourceQuery.kind ||
        resourceQuery.source ||
        resourceQuery.sharing ||
        resourceQuery.processing_status ||
        resourceQuery.project_id ||
        (resourceQuery.tags?.length ?? 0) > 0 ||
        (resourceQuery.descriptors?.length ?? 0) > 0 ||
        (resourceQuery.metadata_filters?.length ?? 0) > 0 ||
        resourceQuery.created_after ||
        resourceQuery.created_before)
  );
  const inputSelector = {
    ...(request.inputSelector ?? {}),
    ...(tags.length > 0 ? { tags } : {}),
    ...(snapshotName.length > 0 ? { snapshot_name: snapshotName } : {}),
  };
  return apiClient.createDataAgentJob({
    job_type: request.jobType,
    resource_ids: resourceIds.length > 0 ? resourceIds : undefined,
    source_collection_id: sourceCollectionId,
    project_id: projectId,
    resource_query: hasResourceQuery ? resourceQuery : undefined,
    input_selector: Object.keys(inputSelector).length > 0 ? inputSelector : undefined,
    metadata: {
      ...(request.metadata ?? {}),
      selected_resource_count: resourceIds.length,
      source: "resources_data_agent_launcher",
    },
  });
};
