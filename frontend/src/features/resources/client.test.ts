import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
  createDatasetSnapshotShareGrant,
  createResourceCollectionShareGrants,
  createBulkResourceTags,
  createBulkResourceShareGrants,
  createResourceShareGrant,
  deleteResourceCollection,
  deleteDatasetSnapshot,
  deleteBulkResources,
  loadDatasetSnapshotEvents,
  loadDatasetSnapshotShareGrants,
  loadDatasetSnapshots,
  loadComposerResources,
  loadDataAgentJobs,
  loadLibraryResources,
  loadResourceFolders,
  loadResourceShareGrants,
  patchResourceMetadata,
  removeResourceFromCollection,
  renameResource,
  renameResourceCollection,
  restoreBulkResources,
  restoreDatasetSnapshot,
  restoreResourceCollection,
  restoreResource,
  revokeDatasetSnapshotShareGrant,
  revokeResourceShareGrant,
  startResourceDataAgentJob,
} from "./client";
import { parseResourceMetadataFilterInput } from "./filters";
import {
  hasInternalResourcePath,
  resourceDisplayName,
  resourceOriginLabel,
} from "./presentation";

describe("resource slice client", () => {
  it("parses structured resource metadata filters from compact input", () => {
    expect(
      parseResourceMetadataFilterInput("label:eq:NPH, subject_age:lt:70, note:contains:ventricle")
    ).toEqual([
      { path: "label", operator: "eq", value: "NPH" },
      { path: "subject_age", operator: "lt", value: "70" },
      { path: "note", operator: "contains", value: "ventricle" },
    ]);
    expect(parseResourceMetadataFilterInput("subject_age:lt:, label:bad:NPH")).toEqual([]);
  });

  it("normalizes all-filters into backend-compatible query params", async () => {
    const apiClient = {
      listResources: vi.fn().mockResolvedValue({ count: 0, offset: 0, resources: [] }),
    } as unknown as ApiClient;

    await loadLibraryResources(apiClient, {
      query: "mitochondria",
      kind: "all",
      source: "all",
      sharing: "shared_by_me",
      processingStatus: "caption_ready",
      status: "deleted",
      tags: [" NPH ", "under 70", "NPH", ""],
      descriptors: [" NPH ", "ventriculomegaly", "NPH", ""],
      metadataFilters: [
        { path: "label", operator: "eq", value: "NPH" },
        { path: "subject_age", operator: "lt", value: "70" },
      ],
      createdAfter: " 2026-06-02 ",
      createdBefore: "2026-06-04",
    });

    expect(apiClient.listResources).toHaveBeenCalledWith({
      kind: undefined,
      limit: 50,
      offset: 0,
      query: "mitochondria",
      processingStatus: "caption_ready",
      sharing: "shared_by_me",
      source: undefined,
      status: "deleted",
      tags: ["NPH", "under 70"],
      descriptors: ["NPH", "ventriculomegaly"],
      metadataFilters: [
        { path: "label", operator: "eq", value: "NPH" },
        { path: "subject_age", operator: "lt", value: "70" },
      ],
      createdAfter: "2026-06-02",
      createdBefore: "2026-06-04",
    });
  });

  it("keeps composer resource lookups lightweight by default", async () => {
    const apiClient = {
      listResources: vi.fn().mockResolvedValue({ count: 0, offset: 0, resources: [] }),
    } as unknown as ApiClient;

    await loadComposerResources(apiClient, { query: "atlas" });

    expect(apiClient.listResources).toHaveBeenCalledWith({
      limit: 200,
      query: "atlas",
    });
  });

  it("loads deleted folders and normalizes folder lifecycle helper calls", async () => {
    const apiClient = {
      listResourceCollections: vi.fn().mockResolvedValue({ count: 0, collections: [] }),
      deleteResourceCollection: vi.fn().mockResolvedValue({
        collection: { collection_id: "collection_nph", status: "deleted" },
      }),
      restoreResourceCollection: vi.fn().mockResolvedValue({
        collection: { collection_id: "collection_nph", status: "active" },
      }),
    } as unknown as ApiClient;

    await loadResourceFolders(apiClient, {
      limit: 25,
      offset: 5,
      query: " NPH ",
      status: "deleted",
    });
    await deleteResourceCollection(apiClient, " collection_nph ");
    await restoreResourceCollection(apiClient, " collection_nph ");

    expect(apiClient.listResourceCollections).toHaveBeenCalledWith({
      collectionType: "folder",
      limit: 25,
      offset: 5,
      query: " NPH ",
      status: "deleted",
    });
    expect(apiClient.deleteResourceCollection).toHaveBeenCalledWith("collection_nph");
    expect(apiClient.restoreResourceCollection).toHaveBeenCalledWith("collection_nph");
  });

  it("loads recent Data Agent jobs for the Resources monitor", async () => {
    const apiClient = {
      listDataAgentJobs: vi.fn().mockResolvedValue({ count: 0, jobs: [] }),
    } as unknown as ApiClient;

    await loadDataAgentJobs(apiClient, {
      status: "running",
      jobType: "extract_metadata",
    });

    expect(apiClient.listDataAgentJobs).toHaveBeenCalledWith({
      jobType: "extract_metadata",
      limit: 8,
      offset: 0,
      projectId: undefined,
      status: "running",
    });
  });

  it("starts a normalized Data Agent job from selected Resources", async () => {
    const apiClient = {
      createDataAgentJob: vi.fn().mockResolvedValue({ job: { job_id: "job_metadata" }, events: [] }),
    } as unknown as ApiClient;

    await startResourceDataAgentJob(apiClient, {
      jobType: "extract_metadata",
      resourceIds: [" file_a ", "file_b", "file_a", ""],
    });

    expect(apiClient.createDataAgentJob).toHaveBeenCalledWith({
      job_type: "extract_metadata",
      resource_ids: ["file_a", "file_b"],
      source_collection_id: undefined,
      project_id: undefined,
      input_selector: undefined,
      metadata: {
        selected_resource_count: 2,
        source: "resources_data_agent_launcher",
      },
    });
  });

  it("starts a batch-tag Data Agent job with normalized tag input", async () => {
    const apiClient = {
      createDataAgentJob: vi.fn().mockResolvedValue({ job: { job_id: "job_batch_tag" }, events: [] }),
    } as unknown as ApiClient;

    await startResourceDataAgentJob(apiClient, {
      jobType: "batch_tag_resources",
      resourceIds: [" file_a ", "file_b", "file_a", ""],
      tags: [" NPH ", "Under 70", "NPH", ""],
    });

    expect(apiClient.createDataAgentJob).toHaveBeenCalledWith({
      job_type: "batch_tag_resources",
      resource_ids: ["file_a", "file_b"],
      source_collection_id: undefined,
      project_id: undefined,
      input_selector: {
        tags: ["NPH", "Under 70"],
      },
      metadata: {
        selected_resource_count: 2,
        source: "resources_data_agent_launcher",
      },
    });
  });

  it("starts a dataset snapshot Data Agent job with a normalized snapshot name", async () => {
    const apiClient = {
      createDataAgentJob: vi.fn().mockResolvedValue({ job: { job_id: "job_snapshot" }, events: [] }),
    } as unknown as ApiClient;

    await startResourceDataAgentJob(apiClient, {
      jobType: "create_dataset_snapshot",
      resourceIds: [" file_a ", "file_b"],
      snapshotName: " NPH training cohort v1 ",
    });

    expect(apiClient.createDataAgentJob).toHaveBeenCalledWith({
      job_type: "create_dataset_snapshot",
      resource_ids: ["file_a", "file_b"],
      source_collection_id: undefined,
      project_id: undefined,
      input_selector: {
        snapshot_name: "NPH training cohort v1",
      },
      metadata: {
        selected_resource_count: 2,
        source: "resources_data_agent_launcher",
      },
    });
  });

  it("starts a dataset snapshot Data Agent job from normalized resource query filters", async () => {
    const apiClient = {
      createDataAgentJob: vi.fn().mockResolvedValue({ job: { job_id: "job_query_snapshot" }, events: [] }),
    } as unknown as ApiClient;

    await startResourceDataAgentJob(apiClient, {
      jobType: "create_dataset_snapshot",
      resourceIds: [],
      snapshotName: " NPH under 70 query cohort ",
      resourceQuery: {
        query: " NPH ",
        kind: "file",
        source: "upload",
        sharing: "private",
        processingStatus: "caption_ready",
        tags: [" Under 70 ", "Under 70"],
        descriptors: [" NPH ", "ventriculomegaly", "NPH"],
        createdAfter: "2026-06-02",
        createdBefore: "2026-06-04",
      },
      metadata: {
        query_result_count: 2,
      },
    });

    expect(apiClient.createDataAgentJob).toHaveBeenCalledWith({
      job_type: "create_dataset_snapshot",
      resource_ids: undefined,
      source_collection_id: undefined,
      project_id: undefined,
      resource_query: {
        q: "NPH",
        kind: "file",
        source: "upload",
        sharing: "private",
        processing_status: "caption_ready",
        tags: ["Under 70"],
        descriptors: ["NPH", "ventriculomegaly"],
        created_after: "2026-06-02",
        created_before: "2026-06-04",
      },
      input_selector: {
        snapshot_name: "NPH under 70 query cohort",
      },
      metadata: {
        query_result_count: 2,
        selected_resource_count: 0,
        source: "resources_data_agent_launcher",
      },
    });
  });

  it("starts a normalized Data Agent job from an active Resources folder", async () => {
    const apiClient = {
      createDataAgentJob: vi.fn().mockResolvedValue({ job: { job_id: "job_folder" }, events: [] }),
    } as unknown as ApiClient;

    await startResourceDataAgentJob(apiClient, {
      jobType: "caption_resources",
      sourceCollectionId: " collection_nph ",
      metadata: {
        active_folder_name: "NPH review folder",
      },
    });

    expect(apiClient.createDataAgentJob).toHaveBeenCalledWith({
      job_type: "caption_resources",
      resource_ids: undefined,
      source_collection_id: "collection_nph",
      project_id: undefined,
      input_selector: undefined,
      metadata: {
        active_folder_name: "NPH review folder",
        selected_resource_count: 0,
        source: "resources_data_agent_launcher",
      },
    });
  });

  it("normalizes resource share grant helper calls", async () => {
    const apiClient = {
      listResourceShareGrants: vi.fn().mockResolvedValue({ resource_id: "file_a", count: 0, grants: [] }),
      createResourceShareGrant: vi.fn().mockResolvedValue({
        grant: { grant_id: "grant_a", resource_id: "file_a", role: "read", status: "active" },
      }),
      createResourceShareGrants: vi.fn().mockResolvedValue({
        count: 2,
        grants: [
          { grant_id: "grant_a", resource_id: "file_a", role: "read", status: "active" },
          { grant_id: "grant_b", resource_id: "file_b", role: "read", status: "active" },
        ],
      }),
      revokeResourceShareGrant: vi.fn().mockResolvedValue({
        grant: { grant_id: "grant_a", resource_id: "file_a", role: "read", status: "revoked" },
      }),
    } as unknown as ApiClient;

    await loadResourceShareGrants(apiClient, " file_a ");
    await createResourceShareGrant(apiClient, " file_a ", {
      granteeUserId: " bob ",
      granteeOrgId: " org-b ",
    });
    await createBulkResourceShareGrants(apiClient, [" file_a ", "file_b", "file_a"], {
      granteeUserId: " charlie ",
      granteeOrgId: " org-c ",
    });
    await revokeResourceShareGrant(apiClient, " file_a ", " grant_a ");

    expect(apiClient.listResourceShareGrants).toHaveBeenCalledWith("file_a", {
      limit: 200,
      status: undefined,
    });
    expect(apiClient.createResourceShareGrant).toHaveBeenCalledWith("file_a", {
      grantee_user_id: "bob",
      grantee_org_id: "org-b",
      role: "read",
      metadata: {
        source: "resources_share_panel",
      },
    });
    expect(apiClient.createResourceShareGrants).toHaveBeenCalledWith({
      resource_ids: ["file_a", "file_b"],
      grantee_user_id: "charlie",
      grantee_org_id: "org-c",
      role: "read",
      metadata: {
        source: "resources_bulk_share_panel",
      },
    });
    expect(apiClient.revokeResourceShareGrant).toHaveBeenCalledWith("file_a", "grant_a");
  });

  it("normalizes selected-resource bulk delete helper calls", async () => {
    const apiClient = {
      deleteResources: vi.fn().mockResolvedValue({
        count: 2,
        resources: [],
        events: [],
      }),
    } as unknown as ApiClient;

    await deleteBulkResources(apiClient, [" file_a ", "file_b", "file_a", " "]);

    expect(apiClient.deleteResources).toHaveBeenCalledWith({
      resource_ids: ["file_a", "file_b"],
    });
  });

  it("normalizes selected-resource bulk restore helper calls", async () => {
    const apiClient = {
      restoreResources: vi.fn().mockResolvedValue({
        count: 2,
        resources: [],
        events: [],
      }),
    } as unknown as ApiClient;

    await restoreBulkResources(apiClient, [" file_a ", "file_b", "file_a", " "]);

    expect(apiClient.restoreResources).toHaveBeenCalledWith({
      resource_ids: ["file_a", "file_b"],
    });
  });

  it("normalizes resource restore helper calls", async () => {
    const apiClient = {
      restoreResource: vi.fn().mockResolvedValue({
        resource: { file_id: "file_a", status: "active" },
      }),
    } as unknown as ApiClient;

    await restoreResource(apiClient, " file_a ");

    expect(apiClient.restoreResource).toHaveBeenCalledWith("file_a");
  });

  it("normalizes resource and folder file-manager helper calls", async () => {
    const apiClient = {
      renameResource: vi.fn().mockResolvedValue({
        resource: { file_id: "file_a", original_name: "nph-a-reviewed.nii.gz" },
      }),
      patchResourceCollection: vi.fn().mockResolvedValue({
        collection: { collection_id: "collection_nph", name: "NPH reviewed" },
      }),
      removeResourceFromCollection: vi.fn().mockResolvedValue({
        collection: { collection_id: "collection_nph", resource_count: 1 },
        removed_count: 1,
        memberships: [],
      }),
    } as unknown as ApiClient;

    await renameResource(apiClient, " file_a ", " nph-a-reviewed.nii.gz ");
    await renameResourceCollection(apiClient, " collection_nph ", " NPH reviewed ");
    await removeResourceFromCollection(apiClient, " collection_nph ", " file_a ");

    expect(apiClient.renameResource).toHaveBeenCalledWith("file_a", "nph-a-reviewed.nii.gz");
    expect(apiClient.patchResourceCollection).toHaveBeenCalledWith("collection_nph", {
      name: "NPH reviewed",
    });
    expect(apiClient.removeResourceFromCollection).toHaveBeenCalledWith(
      "collection_nph",
      "file_a"
    );
  });

  it("normalizes folder share helper calls", async () => {
    const apiClient = {
      createResourceCollectionShareGrants: vi.fn().mockResolvedValue({
        count: 2,
        collection: { collection_id: "collection_nph" },
        grants: [
          { grant_id: "grant_a", resource_id: "file_a", role: "read", status: "active" },
          { grant_id: "grant_b", resource_id: "file_b", role: "read", status: "active" },
        ],
      }),
    } as unknown as ApiClient;

    await createResourceCollectionShareGrants(apiClient, " collection_nph ", {
      granteeUserId: " dana ",
      granteeOrgId: " org-d ",
      metadata: { reason: "folder review" },
    });

    expect(apiClient.createResourceCollectionShareGrants).toHaveBeenCalledWith(
      "collection_nph",
      {
        grantee_user_id: "dana",
        grantee_org_id: "org-d",
        role: "read",
        metadata: {
          reason: "folder review",
          source: "resources_folder_share_panel",
        },
      }
    );
  });

  it("normalizes dataset snapshot share helper calls", async () => {
    const apiClient = {
      listDatasetSnapshotShareGrants: vi.fn().mockResolvedValue({ count: 0, grants: [] }),
      createDatasetSnapshotShareGrant: vi.fn().mockResolvedValue({
        grant: {
          grant_id: "dataset_snapshot_grant_bob",
          snapshot_id: "dataset_snapshot_nph_v1",
          role: "read",
          status: "active",
        },
      }),
      revokeDatasetSnapshotShareGrant: vi.fn().mockResolvedValue({
        grant: {
          grant_id: "dataset_snapshot_grant_bob",
          snapshot_id: "dataset_snapshot_nph_v1",
          role: "read",
          status: "revoked",
        },
      }),
    } as unknown as ApiClient;

    await loadDatasetSnapshotShareGrants(apiClient, " dataset_snapshot_nph_v1 ", {
      limit: 25,
      status: "active",
    });
    await createDatasetSnapshotShareGrant(apiClient, " dataset_snapshot_nph_v1 ", {
      granteeUserId: " bob ",
      granteeOrgId: " org-b ",
      metadata: { reason: "cohort review" },
    });
    await revokeDatasetSnapshotShareGrant(
      apiClient,
      " dataset_snapshot_nph_v1 ",
      " dataset_snapshot_grant_bob "
    );

    expect(apiClient.listDatasetSnapshotShareGrants).toHaveBeenCalledWith(
      "dataset_snapshot_nph_v1",
      {
        limit: 25,
        status: "active",
      }
    );
    expect(apiClient.createDatasetSnapshotShareGrant).toHaveBeenCalledWith(
      "dataset_snapshot_nph_v1",
      {
        grantee_user_id: "bob",
        grantee_org_id: "org-b",
        role: "read",
        metadata: {
          reason: "cohort review",
          source: "resources_dataset_share_panel",
        },
      }
    );
    expect(apiClient.revokeDatasetSnapshotShareGrant).toHaveBeenCalledWith(
      "dataset_snapshot_nph_v1",
      "dataset_snapshot_grant_bob"
    );
  });

  it("normalizes dataset snapshot event helper calls", async () => {
    const apiClient = {
      listDatasetSnapshotEvents: vi.fn().mockResolvedValue({
        snapshot_id: "dataset_snapshot_nph_v1",
        count: 0,
        total_count: 0,
        limit: 50,
        offset: 0,
        events: [],
      }),
    } as unknown as ApiClient;

    await loadDatasetSnapshotEvents(apiClient, " dataset_snapshot_nph_v1 ", {
      limit: 50,
      eventType: "dataset_snapshot.shared",
      actorUserId: " user_qa ",
    });

    expect(apiClient.listDatasetSnapshotEvents).toHaveBeenCalledWith(
      "dataset_snapshot_nph_v1",
      {
        limit: 50,
        eventType: "dataset_snapshot.shared",
        actorUserId: "user_qa",
      }
    );
  });

  it("loads deleted dataset snapshots and normalizes dataset lifecycle helper calls", async () => {
    const apiClient = {
      listDatasetSnapshots: vi.fn().mockResolvedValue({
        count: 1,
        snapshots: [{ snapshot_id: "dataset_snapshot_deleted_nph", status: "deleted" }],
      }),
      deleteDatasetSnapshot: vi.fn().mockResolvedValue({
        snapshot: { snapshot_id: "dataset_snapshot_deleted_nph", status: "deleted" },
        resources: [],
      }),
      restoreDatasetSnapshot: vi.fn().mockResolvedValue({
        snapshot: { snapshot_id: "dataset_snapshot_deleted_nph", status: "active" },
        resources: [],
      }),
    } as unknown as ApiClient;

    await loadDatasetSnapshots(apiClient, {
      limit: 25,
      offset: 5,
      projectId: " nph-study ",
      status: "deleted",
    });
    await deleteDatasetSnapshot(apiClient, " dataset_snapshot_deleted_nph ");
    await restoreDatasetSnapshot(apiClient, " dataset_snapshot_deleted_nph ");

    expect(apiClient.listDatasetSnapshots).toHaveBeenCalledWith({
      limit: 25,
      offset: 5,
      projectId: "nph-study",
      status: "deleted",
    });
    expect(apiClient.deleteDatasetSnapshot).toHaveBeenCalledWith("dataset_snapshot_deleted_nph");
    expect(apiClient.restoreDatasetSnapshot).toHaveBeenCalledWith("dataset_snapshot_deleted_nph");
  });

  it("normalizes bulk resource tag helper calls", async () => {
    const apiClient = {
      bulkTagResources: vi.fn().mockResolvedValue({
        count: 2,
        resources: [
          { file_id: "file_a", tags: ["NPH"] },
          { file_id: "file_b", tags: ["NPH"] },
        ],
        events: [],
      }),
    } as unknown as ApiClient;

    await createBulkResourceTags(apiClient, [" file_a ", "file_b", "file_a", ""], [
      " NPH ",
      "under 70",
      "NPH",
      "",
    ]);

    expect(apiClient.bulkTagResources).toHaveBeenCalledWith({
      resource_ids: ["file_a", "file_b"],
      tags: ["NPH", "under 70"],
      metadata: {
        source: "resources_bulk_tag_panel",
      },
    });
  });

  it("normalizes resource metadata patch helper calls", async () => {
    const apiClient = {
      patchResourceMetadata: vi.fn().mockResolvedValue({
        resource: { file_id: "file_a", metadata: { cohort: "NPH" } },
      }),
    } as unknown as ApiClient;

    await patchResourceMetadata(apiClient, " file_a ", {
      cohort: "NPH",
      review: { status: "checked" },
    });

    expect(apiClient.patchResourceMetadata).toHaveBeenCalledWith("file_a", {
      cohort: "NPH",
      review: { status: "checked" },
      edit_source: "resources_metadata_panel",
    });
  });

  it("keeps internal workspace paths out of resource display labels", () => {
    expect(
      resourceDisplayName({
        file_id: "file_1234567890abcdef",
        original_name: "/workspace/outputs/plot.png",
      })
    ).toBe("plot.png");
    expect(hasInternalResourcePath("/workspace/outputs/plot.png")).toBe(true);
  });

  it("summarizes resource origin without exposing raw URIs", () => {
    expect(
      resourceOriginLabel({
        source_type: "bisque_import",
        resource_kind: "image",
        source_uri: "https://bisque.example/data_service/image/123",
        client_view_url: "https://bisque.example/client_service/view?resource=123",
      })
    ).toBe("Imported BisQue image");
  });
});
