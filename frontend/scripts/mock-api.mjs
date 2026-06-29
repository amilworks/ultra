import http from "node:http";
import { createHash } from "node:crypto";

const port = Number(process.env.MOCK_API_PORT || "8000");
const guestCookieName = "bisque_ultra_session";
const bisqueRoot = "https://bisque2.ece.ucsb.edu";

const navLinks = {
  home: `${bisqueRoot}/client_service/`,
  datasets: `${bisqueRoot}/client_service/browser?resource=/data_service/dataset`,
  images: `${bisqueRoot}/client_service/browser?resource=/data_service/image`,
  tables: `${bisqueRoot}/client_service/browser?resource=/data_service/table`,
};
const nowIso = new Date("2026-06-07T12:00:00Z").toISOString();
const datasetSnapshotFixture = {
  snapshot_id: "dataset_snapshot_nph_v1",
  owner_user_id: "user_mobile_smoke",
  owner_org_id: "org_research",
  owner_role: "owner",
  project_id: "project_nph",
  source_collection_id: "collection_nph",
  name: "NPH training cohort v1",
  description: "Immutable NPH training cohort fixture.",
  status: "active",
  resource_count: 2,
  total_bytes: 9956000,
  created_by_user_id: "user_mobile_smoke",
  created_at: nowIso,
  metadata: { source: "mock_api" },
};
const datasetSnapshotResourcesFixture = [
  {
    snapshot_id: datasetSnapshotFixture.snapshot_id,
    resource_id: "file_image",
    position: 1,
    original_name: "prairie-cell-image.png",
    content_type: "image/png",
    size_bytes: 156000,
    sha256: "sha-image",
    source_type: "bisque_import",
    resource_kind: "image",
    source_uri: "/data_service/image/file_image",
    project_id: "project_nph",
    resource_created_at: nowIso,
    metadata: {},
  },
  {
    snapshot_id: datasetSnapshotFixture.snapshot_id,
    resource_id: "file_volume",
    position: 2,
    original_name: "NPH_shunt_002_70yo.nii.gz",
    content_type: "application/gzip",
    size_bytes: 9800000,
    sha256: "sha-volume",
    source_type: "upload",
    resource_kind: "file",
    storage_uri: "file:///mock/NPH_shunt_002_70yo.nii.gz",
    project_id: "project_nph",
    resource_created_at: nowIso,
    metadata: {},
  },
];
const datasetSnapshotShareGrantFixture = {
  grant_id: "dataset_snapshot_grant_bob",
  snapshot_id: datasetSnapshotFixture.snapshot_id,
  owner_user_id: "user_mobile_smoke",
  owner_org_id: "org_research",
  owner_role: "owner",
  grantee_user_id: "bob",
  grantee_org_id: "org_research",
  role: "read",
  status: "active",
  created_by_user_id: "user_mobile_smoke",
  created_at: nowIso,
  updated_at: nowIso,
  metadata: { source: "mock_api" },
};
const datasetSnapshotEventFixture = {
  event_id: "dataset_snapshot_event_created",
  snapshot_id: datasetSnapshotFixture.snapshot_id,
  actor_user_id: "user_mobile_smoke",
  actor_org_id: "org_research",
  event_type: "dataset_snapshot.created",
  ts: nowIso,
  metadata: {
    snapshot_name: datasetSnapshotFixture.name,
    resource_count: datasetSnapshotFixture.resource_count,
    total_bytes: datasetSnapshotFixture.total_bytes,
    project_id: datasetSnapshotFixture.project_id,
    source_collection_id: datasetSnapshotFixture.source_collection_id,
    source: "mock_api",
  },
};
const resourceFixtures = [
  {
    file_id: "file_csv_demo",
    original_name: "survey_2026.csv",
    content_type: "text/csv",
    size_bytes: 2_410_000_000,
    sha256: "sha-csv-demo",
    source_type: "upload",
    resource_kind: "table",
    project_id: "project_nph",
    created_at: nowIso,
    has_thumbnail: false,
    tags: ["survey"],
    metadata: {},
    share_summary: { share_status: "private", active_grant_count: 0, shared_by_me: false, shared_with_me: false },
  },
  {
    file_id: "file_json_demo",
    original_name: "agent_config.json",
    content_type: "application/json",
    size_bytes: 3200,
    sha256: "sha-json-demo",
    source_type: "upload",
    resource_kind: "document",
    project_id: "project_nph",
    created_at: nowIso,
    has_thumbnail: false,
    tags: ["config"],
    metadata: {},
    share_summary: { share_status: "private", active_grant_count: 0, shared_by_me: false, shared_with_me: false },
  },
  {
    file_id: "file_md_demo",
    original_name: "README.md",
    content_type: "text/markdown",
    size_bytes: 6800,
    sha256: "sha-md-demo",
    source_type: "upload",
    resource_kind: "document",
    project_id: "project_nph",
    created_at: nowIso,
    has_thumbnail: false,
    tags: ["docs"],
    metadata: {},
    share_summary: { share_status: "private", active_grant_count: 0, shared_by_me: false, shared_with_me: false },
  },
  {
    file_id: "file_query_dataset_b",
    original_name: "subject-b-nph-under70.nii.gz",
    content_type: "application/gzip",
    size_bytes: 256000,
    sha256: "sha-query-b",
    source_type: "upload",
    resource_kind: "file",
    project_id: "project_nph",
    created_at: nowIso,
    has_thumbnail: false,
    tags: ["NPH", "Under 70"],
    metadata: {
      subject_age: 64,
      label: "NPH",
      scientific_descriptors: ["ventriculomegaly", "shunt review"],
      data_agent: {
        caption_resources: {
          status: "succeeded",
          caption: "NPH under-70 NIfTI volume with shunt-review metadata.",
          completed_at: nowIso,
        },
        extract_metadata: {
          status: "succeeded",
          completed_at: nowIso,
          descriptors: ["Evans index high", "lateral ventricle enlargement"],
        },
      },
    },
    share_summary: {
      share_status: "private",
      active_grant_count: 0,
      shared_by_me: false,
      shared_with_me: false,
    },
  },
  {
    file_id: "file_query_dataset_a",
    original_name: "subject-a-nph-under70.nii.gz",
    content_type: "application/gzip",
    size_bytes: 128000,
    sha256: "sha-query-a",
    source_type: "upload",
    resource_kind: "file",
    project_id: "project_nph",
    created_at: nowIso,
    has_thumbnail: false,
    tags: ["NPH", "Under 70"],
    metadata: {
      subject_age: 68,
      label: "NPH",
      scientific_descriptors: ["ventriculomegaly", "training candidate"],
      data_agent: {
        extract_metadata: {
          status: "succeeded",
          completed_at: nowIso,
          descriptors: ["Evans index high", "NIfTI volume"],
        },
      },
    },
    share_summary: {
      share_status: "private",
      active_grant_count: 0,
      shared_by_me: false,
      shared_with_me: false,
    },
  },
];
const resourceCollectionFixtures = [
  {
    collection_id: "collection_nph",
    owner_user_id: "user_mobile_smoke",
    owner_org_id: "org_research",
    owner_role: "owner",
    project_id: "project_nph",
    parent_collection_id: "",
    name: "NPH review folder",
    description: "Curated NPH review resources.",
    collection_type: "folder",
    status: "active",
    resource_count: resourceFixtures.length,
    created_at: "2026-06-08T00:00:00Z",
    updated_at: "2026-06-08T00:00:00Z",
    metadata: { source: "mock_api" },
  },
  {
    collection_id: "collection_deleted_nph",
    owner_user_id: "user_mobile_smoke",
    owner_org_id: "org_research",
    owner_role: "owner",
    project_id: "project_nph",
    parent_collection_id: "",
    name: "NPH archived review folder",
    description: "Deleted folder fixture for restore QA.",
    collection_type: "folder",
    status: "deleted",
    resource_count: 7,
    created_at: "2026-06-08T00:00:00Z",
    updated_at: "2026-06-09T00:00:00Z",
    metadata: { source: "mock_api" },
  },
];
const mockViewerRasterPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAFnklEQVR42u3VoQ0AAAgEsd9/WCQWdsDSpBOcuVQPAA9FAgADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAADUAHAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADADAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADADAAFQAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAAA5AAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAwABUADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADADAAFQAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAMAAADAAAAwAAAMAwAAAMAAADAAAAwDAAAAwAAAMAAADAMAAADAAAAwAAAMAwAAAMAAAbhYktq01xYecMAAAAABJRU5ErkJggg==",
  "base64"
);
const createdResourceCollections = [];
const resourceCollectionStatusOverrides = new Map();
const resourceCollectionNameOverrides = new Map();
const resourceCollectionCountOverrides = new Map();
const removedCollectionResourceIds = new Map();
const createdDatasetSnapshots = [];
const createdDatasetSnapshotResources = new Map();
const datasetSnapshotStatusOverrides = new Map();
const datasetSnapshotShareGrants = new Map([
  [datasetSnapshotFixture.snapshot_id, [datasetSnapshotShareGrantFixture]],
]);
const datasetSnapshotEvents = new Map([
  [datasetSnapshotFixture.snapshot_id, [datasetSnapshotEventFixture]],
]);
const createdDataAgentJobs = [];
const createdUploadSessions = new Map();
const createdUploadSessionIdsByIdempotency = new Map();
const createdUploadResources = [];
const createdUploadChunkBytes = new Map();
const createdUploadResourceBytes = new Map();

const readJsonBody = async (request) => {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
  }
  return body ? JSON.parse(body) : {};
};

const readRequestBuffer = async (request) => {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
};

const sha256Hex = (bytes) => createHash("sha256").update(bytes).digest("hex");

const mockUserProfile = {
  display_name: "Dr. Ada Lovelace",
  title: "Principal Investigator",
  institution: "UCSB Vision Research Lab",
  research_interests:
    "Cell segmentation, multiplexed imaging, reproducible analysis pipelines",
  bio: "I run a microscopy lab focused on quantitative cell biology. I value careful, reproducible analysis and publication-quality figures.",
};

const mockCurrentUserResponse = () => ({
  user: {
    user_id: "mock-user",
    email: "ada@example.org",
    display_name: mockUserProfile.display_name,
    role: "researcher",
    org_id: "mock-org",
  },
  profile: { ...mockUserProfile },
});

const DAY_MS = 86400000;
const mockDayKey = (ms) => new Date(ms).toISOString().slice(0, 10);

const buildMockTokenUsage = (days = 365) => {
  const now = new Date();
  const todayUTC = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const window = Math.max(1, Math.min(730, Number(days) || 365));
  const daily = [];
  const activeKeys = [];
  let lifetimeInput = 0;
  let lifetimeOutput = 0;
  let lifetimeTotal = 0;
  let peak = 0;
  for (let offset = Math.min(window, 140) - 1; offset >= 0; offset -= 1) {
    const date = new Date(todayUTC - offset * DAY_MS);
    const dow = date.getUTCDay();
    const seed = (offset * 2654435761 + 97) >>> 0;
    const active = dow !== 0 && (offset <= 4 || seed % 5 !== 0);
    if (!active) {
      continue;
    }
    const input = 1500 + (seed % 9000);
    const output = Math.round(input * 0.35) + (seed % 800);
    const total = input + output;
    lifetimeInput += input;
    lifetimeOutput += output;
    lifetimeTotal += total;
    if (total > peak) {
      peak = total;
    }
    const key = mockDayKey(todayUTC - offset * DAY_MS);
    activeKeys.push(key);
    daily.push({
      day: key,
      input_tokens: input,
      output_tokens: output,
      total_tokens: total,
      run_count: 1 + (seed % 4),
    });
  }
  const activeSet = new Set(activeKeys);
  let current = 0;
  let cursor = todayUTC;
  if (!activeSet.has(mockDayKey(cursor))) {
    cursor -= DAY_MS;
  }
  while (activeSet.has(mockDayKey(cursor))) {
    current += 1;
    cursor -= DAY_MS;
  }
  let longest = 0;
  let run = 0;
  let prev = null;
  for (const key of [...activeSet].sort()) {
    const ms = Date.parse(`${key}T00:00:00Z`);
    run = prev !== null && ms - prev === DAY_MS ? run + 1 : 1;
    if (run > longest) {
      longest = run;
    }
    prev = ms;
  }
  return {
    days: window,
    summary: {
      lifetime_input_tokens: lifetimeInput,
      lifetime_output_tokens: lifetimeOutput,
      lifetime_total_tokens: lifetimeTotal,
      peak_daily_total: peak,
      longest_task_seconds: 38580,
      current_streak_days: current,
      longest_streak_days: longest,
      active_days: activeKeys.length,
      last_active_day: activeKeys[activeKeys.length - 1],
    },
    daily,
  };
};

const uploadSessionLimits = () => ({
  max_parallel_files: 4,
  max_parallel_chunks: 4,
  max_files_per_session: 10000,
});

const uploadSessionPayload = (sessionRecord) => ({
  session: sessionRecord.session,
  files: sessionRecord.files,
  chunks: Array.from(sessionRecord.chunks.values()).sort((left, right) => {
    if (left.file_token === right.file_token) {
      return left.chunk_index - right.chunk_index;
    }
    return left.file_token.localeCompare(right.file_token);
  }),
  events: sessionRecord.events,
  limits: uploadSessionLimits(),
});

const recomputeUploadSessionCounters = (sessionRecord) => {
  const chunks = Array.from(sessionRecord.chunks.values());
  const verifiedBytes = chunks
    .filter((chunk) => chunk.status === "verified")
    .reduce((total, chunk) => total + Number(chunk.size_bytes || 0), 0);
  const committedBytes = sessionRecord.files
    .filter((file) => file.status === "completed")
    .reduce((total, file) => total + Number(file.size_bytes || 0), 0);
  sessionRecord.session.bytes_received = verifiedBytes;
  sessionRecord.session.bytes_verified = verifiedBytes;
  sessionRecord.session.bytes_committed = committedBytes;
  if (sessionRecord.files.length > 0 && sessionRecord.files.every((file) => file.status === "completed")) {
    sessionRecord.session.status = "completed";
    sessionRecord.session.completed_at ||= new Date().toISOString();
  }
  sessionRecord.session.updated_at = new Date().toISOString();
};

const datasetSnapshotBaseById = (snapshotId) => {
  if (snapshotId === datasetSnapshotFixture.snapshot_id) {
    return datasetSnapshotFixture;
  }
  return createdDatasetSnapshots.find((snapshot) => snapshot.snapshot_id === snapshotId) || null;
};

const datasetSnapshotWithLifecycle = (snapshot) => {
  const status = datasetSnapshotStatusOverrides.get(snapshot.snapshot_id) || snapshot.status || "active";
  return { ...snapshot, status };
};

const datasetSnapshotResourcesById = (snapshotId) => {
  if (snapshotId === datasetSnapshotFixture.snapshot_id) {
    return datasetSnapshotResourcesFixture;
  }
  return createdDatasetSnapshotResources.get(snapshotId) || [];
};

const metadataValueAtPath = (metadata, path) => {
  const parts = String(path || "")
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);
  let current = metadata || {};
  for (const part of parts) {
    if (!current || typeof current !== "object" || !(part in current)) {
      return undefined;
    }
    current = current[part];
  }
  return current;
};

const metadataFilterObjects = (queryInput = {}) => {
  const rawFilters = [
    ...(Array.isArray(queryInput.metadata_filter) ? queryInput.metadata_filter : []),
    ...(Array.isArray(queryInput.metadata_filters) ? queryInput.metadata_filters : []),
  ];
  return rawFilters
    .map((filter) => {
      if (typeof filter === "string") {
        const [path, operator, ...valueParts] = filter.split(":");
        return {
          path: String(path || "").trim(),
          operator: String(operator || "").trim().toLowerCase(),
          value: valueParts.join(":").trim(),
        };
      }
      return {
        path: String(filter?.path || "").trim(),
        operator: String(filter?.operator || "").trim().toLowerCase(),
        value: String(filter?.value ?? "").trim(),
      };
    })
    .filter((filter) => filter.path && filter.operator && (filter.operator === "exists" || filter.value));
};

const resourceMatchesMetadataFilters = (resource, queryInput = {}) =>
  metadataFilterObjects(queryInput).every((filter) => {
    const actual = metadataValueAtPath(resource.metadata, filter.path);
    if (filter.operator === "exists") {
      return actual !== undefined;
    }
    if (actual === undefined) {
      return false;
    }
    if (filter.operator === "eq") {
      return String(actual).trim().toLowerCase() === filter.value.toLowerCase();
    }
    if (filter.operator === "contains") {
      return JSON.stringify(actual).toLowerCase().includes(filter.value.toLowerCase());
    }
    const actualNumber = Number(actual);
    const expectedNumber = Number(filter.value);
    if (!Number.isFinite(actualNumber) || !Number.isFinite(expectedNumber)) {
      return false;
    }
    if (filter.operator === "lt") return actualNumber < expectedNumber;
    if (filter.operator === "lte") return actualNumber <= expectedNumber;
    if (filter.operator === "gt") return actualNumber > expectedNumber;
    if (filter.operator === "gte") return actualNumber >= expectedNumber;
    return false;
  });

const createdDateBound = (value, endOfDay = false) => {
  const text = String(value || "").trim();
  if (!text) {
    return null;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    const date = new Date(`${text}T00:00:00.000Z`);
    if (Number.isNaN(date.getTime())) {
      return null;
    }
    if (endOfDay) {
      date.setUTCDate(date.getUTCDate() + 1);
      date.setUTCMilliseconds(date.getUTCMilliseconds() - 1);
    }
    return date;
  }
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
};

const resourceMatchesCreatedRange = (resource, queryInput = {}) => {
  const createdAt = new Date(resource.created_at || 0);
  if (Number.isNaN(createdAt.getTime())) {
    return false;
  }
  const createdAfter = createdDateBound(queryInput.created_after);
  const createdBefore = createdDateBound(queryInput.created_before, true);
  if (createdAfter && createdAt < createdAfter) {
    return false;
  }
  if (createdBefore && createdAt > createdBefore) {
    return false;
  }
  return true;
};

const readyJobTypes = [
  "caption_resources",
  "extract_metadata",
  "batch_tag_resources",
  "quality_check_resources",
  "deduplicate_resources",
  "organize_resources",
];

const resourceDataAgentStatus = (resource, jobType) =>
  String(resource.metadata?.data_agent?.[jobType]?.status || "").trim().toLowerCase();

const resourceDataAgentSucceeded = (resource, jobType) =>
  ["succeeded", "completed"].includes(resourceDataAgentStatus(resource, jobType));

const resourceDataAgentFailed = (resource, jobType) =>
  ["failed", "error"].includes(resourceDataAgentStatus(resource, jobType));

const resourceMatchesProcessingStatus = (resource, queryInput = {}) => {
  const status = String(queryInput.processing_status || queryInput.processingStatus || "")
    .trim()
    .toLowerCase();
  if (!status || status === "all") return true;
  if (status === "caption_ready") return resourceDataAgentSucceeded(resource, "caption_resources");
  if (status === "metadata_ready") return resourceDataAgentSucceeded(resource, "extract_metadata");
  if (status === "tags_ready") return resourceDataAgentSucceeded(resource, "batch_tag_resources");
  if (status === "qc_complete") return resourceDataAgentSucceeded(resource, "quality_check_resources");
  if (status === "dedupe_checked") return resourceDataAgentSucceeded(resource, "deduplicate_resources");
  if (status === "organization_ready") return resourceDataAgentSucceeded(resource, "organize_resources");
  if (status === "data_agent_ready") {
    return readyJobTypes.some((jobType) => resourceDataAgentSucceeded(resource, jobType));
  }
  if (status === "needs_caption") return !resourceDataAgentSucceeded(resource, "caption_resources");
  if (status === "needs_metadata") return !resourceDataAgentSucceeded(resource, "extract_metadata");
  if (status === "data_agent_failed") {
    return readyJobTypes.some((jobType) => resourceDataAgentFailed(resource, jobType));
  }
  return false;
};

const descriptorValuesForResource = (resource) => [
  ...(resource.tags || []),
  resource.metadata?.label,
  ...(Array.isArray(resource.metadata?.labels) ? resource.metadata.labels : []),
  resource.metadata?.descriptor,
  ...(Array.isArray(resource.metadata?.descriptors) ? resource.metadata.descriptors : []),
  resource.metadata?.scientific_descriptor,
  ...(Array.isArray(resource.metadata?.scientific_descriptors)
    ? resource.metadata.scientific_descriptors
    : []),
  resource.metadata?.diagnosis,
  ...(Array.isArray(resource.metadata?.diagnoses) ? resource.metadata.diagnoses : []),
  resource.metadata?.modality,
  resource.metadata?.organism,
  resource.metadata?.species,
  resource.metadata?.data_agent?.caption_resources?.caption,
  resource.metadata?.data_agent?.caption_resources?.summary,
  resource.metadata?.data_agent?.extract_metadata?.caption,
  resource.metadata?.data_agent?.extract_metadata?.summary,
  resource.metadata?.data_agent?.extract_metadata?.label,
  ...(Array.isArray(resource.metadata?.data_agent?.extract_metadata?.labels)
    ? resource.metadata.data_agent.extract_metadata.labels
    : []),
  resource.metadata?.data_agent?.extract_metadata?.descriptor,
  ...(Array.isArray(resource.metadata?.data_agent?.extract_metadata?.descriptors)
    ? resource.metadata.data_agent.extract_metadata.descriptors
    : []),
  resource.metadata?.data_agent?.extract_metadata?.scientific_descriptor,
  ...(Array.isArray(resource.metadata?.data_agent?.extract_metadata?.scientific_descriptors)
    ? resource.metadata.data_agent.extract_metadata.scientific_descriptors
    : []),
  resource.metadata?.data_agent?.quality_check_resources?.summary,
  resource.metadata?.data_agent?.organize_resources?.summary,
].filter((value) => String(value || "").trim());

const descriptorFiltersFromQuery = (queryInput = {}) => {
  const values = [];
  const appendValue = (value) => {
    String(value || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .forEach((item) => values.push(item));
  };
  if (Array.isArray(queryInput.descriptor)) {
    queryInput.descriptor.forEach(appendValue);
  } else {
    appendValue(queryInput.descriptor);
  }
  if (Array.isArray(queryInput.descriptors)) {
    queryInput.descriptors.forEach(appendValue);
  } else {
    appendValue(queryInput.descriptors);
  }
  return Array.from(new Set(values));
};

const resourceMatchesDescriptors = (resource, queryInput = {}) => {
  const descriptors = descriptorFiltersFromQuery(queryInput);
  if (descriptors.length === 0) {
    return true;
  }
  const searchable = descriptorValuesForResource(resource).join(" ").toLowerCase();
  return descriptors.every((descriptor) => searchable.includes(descriptor.toLowerCase()));
};

const metadataScalarValues = (value) => {
  if (value == null) {
    return [];
  }
  if (["string", "number", "boolean"].includes(typeof value)) {
    return [String(value)];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => metadataScalarValues(item));
  }
  if (typeof value === "object") {
    return Object.values(value).flatMap((item) => metadataScalarValues(item));
  }
  return [];
};

const normalizeSearchField = (value) =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

const flattenMetadataNumbers = (value, prefix = "") => {
  if (value == null) {
    return [];
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return [[prefix, value]];
  }
  if (typeof value === "string") {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) && value.trim() !== "" ? [[prefix, numberValue]] : [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => flattenMetadataNumbers(item, `${prefix}.${index}`));
  }
  if (typeof value === "object") {
    return Object.entries(value).flatMap(([key, child]) =>
      flattenMetadataNumbers(child, prefix ? `${prefix}.${key}` : key)
    );
  }
  return [];
};

const filenameAge = (resource) => {
  const name = String(resource.original_name || "");
  const match = name.match(/(?:^|[_\-\s])(\d{1,3})\s*(?:yo|yrs?|years?)(?=$|[_\-\s.])/i);
  if (!match) {
    return null;
  }
  const age = Number(match[1]);
  return Number.isFinite(age) ? age : null;
};

const numericFactsForResource = (resource) => {
  const facts = new Map();
  const addFact = (key, value) => {
    const normalized = normalizeSearchField(key);
    const numberValue = Number(value);
    if (!normalized || !Number.isFinite(numberValue)) {
      return;
    }
    if (!facts.has(normalized)) {
      facts.set(normalized, []);
    }
    facts.get(normalized).push(numberValue);
  };
  flattenMetadataNumbers(resource.metadata || {}).forEach(([path, value]) => {
    addFact(path, value);
    const segments = String(path || "").split(".").filter(Boolean);
    if (segments.length > 0) {
      addFact(segments[segments.length - 1], value);
    }
  });
  const age = filenameAge(resource);
  if (age != null) {
    addFact("age", age);
    addFact("subject_age", age);
  }
  if (facts.has("subject_age") && !facts.has("age")) {
    facts.set("age", [...facts.get("subject_age")]);
  }
  if (facts.has("focal_length_mm") && !facts.has("focal_length")) {
    facts.set("focal_length", [...facts.get("focal_length_mm")]);
  }
  return facts;
};

const compareNumber = (actual, operator, expected) => {
  if (operator === "<") return actual < expected;
  if (operator === "<=") return actual <= expected;
  if (operator === ">") return actual > expected;
  if (operator === ">=") return actual >= expected;
  if (operator === "=" || operator === "==") return actual === expected;
  return false;
};

const parseResourceSearchQuery = (rawQuery) => {
  let residual = String(rawQuery || "").trim();
  const numericPredicates = [];
  residual = residual.replace(
    /\b([a-zA-Z][\w.-]*)\s*(<=|>=|==|=|<|>)\s*(-?\d+(?:\.\d+)?)\b/g,
    (_match, field, operator, value) => {
      numericPredicates.push({
        field: normalizeSearchField(field),
        operator,
        value: Number(value),
      });
      return " ";
    }
  );
  const filePatterns = [];
  residual = residual.replace(/(^|\s)(\*\.?[^\s,]+)(?=$|\s|,)/g, (match, prefix, pattern) => {
    filePatterns.push(String(pattern || "").toLowerCase());
    return prefix || " ";
  });
  return {
    numericPredicates,
    filePatterns,
    residual: residual.replace(/\s+/g, " ").trim(),
  };
};

const resourceMatchesNumericPredicates = (resource, predicates) => {
  if (predicates.length === 0) {
    return true;
  }
  const facts = numericFactsForResource(resource);
  return predicates.every((predicate) => {
    const values = facts.get(predicate.field) || [];
    return values.some((value) => compareNumber(value, predicate.operator, predicate.value));
  });
};

const resourceMatchesFilePatterns = (resource, patterns) => {
  if (patterns.length === 0) {
    return true;
  }
  const filename = String(resource.original_name || "").toLowerCase();
  return patterns.every((pattern) => {
    if (pattern.startsWith("*.")) {
      const extension = pattern.slice(2);
      return filename.endsWith(`.${extension}`) || filename.includes(`.${extension}.`);
    }
    const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
    return new RegExp(`^${escaped}$`).test(filename);
  });
};

const resourceMatchesQuery = (resource, queryInput = {}) => {
  const rawQuery = String(queryInput.q || queryInput.query || "").trim();
  const parsedQuery = parseResourceSearchQuery(rawQuery);
  const query = parsedQuery.residual.toLowerCase();
  const kind = String(queryInput.kind || "").trim();
  const source = String(queryInput.source || "").trim();
  const projectId = String(queryInput.project_id || queryInput.projectId || "").trim();
  const sharing = String(queryInput.sharing || "").trim();
  const tags = Array.isArray(queryInput.tags)
    ? queryInput.tags.map((tag) => String(tag).trim()).filter(Boolean)
    : [];
  if (!resourceMatchesNumericPredicates(resource, parsedQuery.numericPredicates)) {
    return false;
  }
  if (!resourceMatchesFilePatterns(resource, parsedQuery.filePatterns)) {
    return false;
  }
  if (query) {
    const searchable = [
      resource.file_id,
      resource.original_name,
      resource.content_type,
      resource.resource_kind,
      resource.source_type,
      resource.project_id,
      resource.sha256,
      ...(resource.tags || []),
      ...metadataScalarValues(resource.metadata || {}),
    ]
      .join(" ")
      .toLowerCase();
    if (!searchable.includes(query)) {
      return false;
    }
  }
  if (kind && kind !== resource.resource_kind) {
    return false;
  }
  if (source && source !== resource.source_type) {
    return false;
  }
  if (projectId && projectId !== resource.project_id) {
    return false;
  }
  if (sharing && sharing !== "all") {
    const shareStatus = String(resource.share_summary?.share_status || "private");
    if (sharing === "shared") {
      if (!["shared_by_me", "shared_with_me", "public"].includes(shareStatus)) {
        return false;
      }
    } else if (sharing !== shareStatus) {
      return false;
    }
  }
  return (
    tags.every((tag) => resource.tags?.includes(tag)) &&
    resourceMatchesDescriptors(resource, queryInput) &&
    resourceMatchesMetadataFilters(resource, queryInput) &&
    resourceMatchesCreatedRange(resource, queryInput) &&
    resourceMatchesProcessingStatus(resource, queryInput)
  );
};

const deletedResourceIds = new Set();
const resourceNameOverrides = new Map();
const resourceDownloadBytes = (resource) =>
  createdUploadResourceBytes.get(resource.file_id) ||
  Buffer.from(
    [
      `BisQue Ultra mock resource: ${resource.original_name}`,
      `file_id: ${resource.file_id}`,
      `sha256: ${resource.sha256 || "mock"}`,
      "",
    ].join("\n"),
    "utf8"
  );
const attachmentFilename = (name) =>
  String(name || "resource")
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"');
const resourceWithLifecycle = (resource) => {
  const deleted = deletedResourceIds.has(resource.file_id);
  const renamed = resourceNameOverrides.get(resource.file_id);
  return {
    ...resource,
    ...(renamed ? { original_name: renamed } : {}),
    status: deleted ? "deleted" : "active",
  };
};
const mockViewerPlane = {
  axis: "z",
  label: "XY plane",
  axes: ["Y", "X"],
  pixel_size: { width: 512, height: 512 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 512, height: 512 },
  aspect_ratio: 1,
};
const mockUploadViewerInfo = (resource) => {
  const encodedFileId = encodeURIComponent(resource.file_id);
  return {
    kind: "image",
    file_id: resource.file_id,
    original_name: resource.original_name,
    modality: String(resource.content_type || "").includes("gzip") ? "nifti" : "microscopy",
    dims_order: "YX",
    backend_mode: "direct",
    axis_sizes: { T: 1, C: 1, Z: 1, Y: 512, X: 512 },
    selected_indices: { T: 0, C: 0, Z: 0 },
    is_volume: false,
    is_timeseries: false,
    is_multichannel: false,
    display_defaults: {
      enhancement: "d",
      negative: false,
      rotate: 0,
      fusion_method: "m",
      channel_mode: "single",
      channels: [0],
      channel_colors: ["#ffffff"],
      time_index: 0,
      z_index: 0,
    },
    service_urls: {
      preview: `/v2/uploads/${encodedFileId}/preview`,
      display: `/v2/uploads/${encodedFileId}/display`,
      slice: `/v2/uploads/${encodedFileId}/slice`,
      histogram: `/v2/uploads/${encodedFileId}/histogram`,
    },
    metadata: {
      reader: "mock-api",
      dims_order: "YX",
      array_shape: [512, 512],
      array_dtype: "uint8",
      sha256: resource.sha256 || resource.file_id,
      scene_count: 1,
      warnings: [],
      resource_metadata: resource.metadata || {},
    },
    viewer: {
      status: "preview-ready",
      warmup_mode: "deferred",
      backend_mode: "direct",
      default_surface: "2d",
      available_surfaces: ["2d", "metadata"],
      default_axis: "z",
      slice_axes: ["z"],
      channel_mode: "single",
      tile_scheme: { tile_size: 256, format: "png", levels: [] },
      default_plane: mockViewerPlane,
      planes: { z: mockViewerPlane },
      volume_mode: "none",
      render_policy: "scalar",
      delivery_mode: "direct",
      diagnostic_surface: "none",
      first_paint_mode: "image",
      measurement_policy: "pixel-only",
      texture_policy: "linear",
      display_capabilities: ["intensity_window", "histogram"],
      viewer_capabilities: ["2d", "metadata"],
      orientation: {
        frame: "pixel",
        row_axis: "Y",
        col_axis: "X",
        slice_axis: null,
      },
    },
  };
};
const allResources = () => [...createdUploadResources, ...resourceFixtures];
const resourcesMatchingQuery = (queryInput = {}) => {
  const status = String(queryInput.status || "active").trim().toLowerCase() || "active";
  return allResources()
    .map(resourceWithLifecycle)
    .filter(
      (resource) =>
        String(resource.status || "active").toLowerCase() === status &&
        resourceMatchesQuery(resource, queryInput)
    );
};

const allResourceCollections = () => [...createdResourceCollections, ...resourceCollectionFixtures];

const resourceCollectionWithLifecycle = (collection) => {
  const override = resourceCollectionStatusOverrides.get(collection.collection_id);
  const renamed = resourceCollectionNameOverrides.get(collection.collection_id);
  const countOverride = resourceCollectionCountOverrides.get(collection.collection_id);
  const status = String(override || collection.status || "active").trim().toLowerCase() || "active";
  return {
    ...collection,
    ...(renamed ? { name: renamed } : {}),
    ...(typeof countOverride === "number" ? { resource_count: countOverride } : {}),
    status,
  };
};

const findResourceCollection = (collectionId) =>
  allResourceCollections()
    .map(resourceCollectionWithLifecycle)
    .find((collection) => collection.collection_id === collectionId);

const resourceCollectionsMatchingQuery = (queryInput = {}) => {
  const status = String(queryInput.status || "active").trim().toLowerCase() || "active";
  const collectionType = String(queryInput.collection_type || queryInput.collectionType || "")
    .trim()
    .toLowerCase();
  const query = String(queryInput.q || queryInput.query || "").trim().toLowerCase();
  const projectId = String(queryInput.project_id || queryInput.projectId || "").trim();
  return allResourceCollections()
    .map(resourceCollectionWithLifecycle)
    .filter((collection) => {
      if (String(collection.status || "active").toLowerCase() !== status) {
        return false;
      }
      if (collectionType && String(collection.collection_type || "").toLowerCase() !== collectionType) {
        return false;
      }
      if (projectId && String(collection.project_id || "") !== projectId) {
        return false;
      }
      if (query) {
        const searchable = [
          collection.collection_id,
          collection.name,
          collection.description,
          collection.collection_type,
          collection.project_id,
          JSON.stringify(collection.metadata || {}),
        ]
          .join(" ")
          .toLowerCase();
        if (!searchable.includes(query)) {
          return false;
        }
      }
      return true;
    });
};

const sendJson = (response, statusCode, payload, headers = {}) => {
  response.writeHead(statusCode, {
    "Content-Type": "application/json",
    ...headers,
  });
  response.end(JSON.stringify(payload));
};

const browserLogoutRedirectUrl = (value) => {
  const candidate = String(value || "").trim();
  if (!candidate) {
    return "/";
  }
  try {
    const parsed = new URL(candidate, "http://localhost");
    if (
      ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname) ||
      candidate.startsWith("/")
    ) {
      return candidate;
    }
  } catch {
    // Fall through to local app root.
  }
  return "/";
};

const workosSession = () => ({
  authenticated: true,
  provider: "workos",
  mode: "workos",
  username: "mobile.smoke@example.com",
  user: {
    id: "user_mobile_smoke",
    email: "mobile.smoke@example.com",
    first_name: "Mobile",
    last_name: "Smoke",
  },
  bisque_root: bisqueRoot,
  bisque_linked: true,
  expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
  is_admin: false,
});

const readSessionMode = (request) =>
  String(request.headers.cookie || "")
    .split(";")
    .map((value) => value.trim())
    .find((value) => value.startsWith(`${guestCookieName}=`))
    ?.slice(guestCookieName.length + 1) || null;

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "127.0.0.1"}`);

  if (request.method === "GET" && url.pathname === "/v1/health") {
    sendJson(response, 200, { status: "ok", ts: new Date().toISOString() });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/config/public") {
    sendJson(response, 200, {
      bisque_root: bisqueRoot,
      bisque_browser_url: navLinks.images,
      bisque_urls: navLinks,
      bisque_auth_enabled: true,
      bisque_guest_enabled: true,
      auth_provider: "local",
      admin_enabled: false,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/config/public") {
    sendJson(response, 200, {
      bisque_root: bisqueRoot,
      bisque_browser_url: navLinks.images,
      bisque_urls: navLinks,
      bisque_auth_enabled: true,
      bisque_guest_enabled: false,
      auth_provider: "workos",
      admin_enabled: false,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/auth/session") {
    sendJson(response, 200, workosSession());
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/login") {
    sendJson(response, 200, {
      authenticated: false,
      provider: "workos",
      mode: "workos",
      authorization_url: "/",
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/logout") {
    sendJson(response, 200, {
      authenticated: false,
      provider: "workos",
      mode: "workos",
      logout_url: "/",
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/guest") {
    sendJson(response, 200, workosSession());
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/me") {
    sendJson(response, 200, mockCurrentUserResponse());
    return;
  }

  if (request.method === "PATCH" && url.pathname === "/v2/me") {
    const payload = await readJsonBody(request);
    for (const key of [
      "display_name",
      "title",
      "institution",
      "research_interests",
      "bio",
    ]) {
      if (typeof payload[key] === "string") {
        mockUserProfile[key] = payload[key].trim();
      }
    }
    sendJson(response, 200, mockCurrentUserResponse());
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/me/token-usage") {
    const days = Number(url.searchParams.get("days")) || 365;
    sendJson(response, 200, buildMockTokenUsage(days));
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/threads") {
    sendJson(response, 200, {
      threads: [],
      count: 0,
      total_count: 0,
      offset: Number(url.searchParams.get("offset") || "0"),
      limit: Number(url.searchParams.get("limit") || "50"),
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/upload-sessions") {
    const payload = await readJsonBody(request);
    const files = Array.isArray(payload.files) ? payload.files : [];
    if (files.length === 0) {
      sendJson(response, 400, { error: "upload session must include at least one file" });
      return;
    }
    const idempotencyKey = String(payload.idempotency_key || "").trim();
    const existingSessionId = idempotencyKey ? createdUploadSessionIdsByIdempotency.get(idempotencyKey) : null;
    if (existingSessionId && createdUploadSessions.has(existingSessionId)) {
      sendJson(response, 200, uploadSessionPayload(createdUploadSessions.get(existingSessionId)));
      return;
    }
    const createdAt = new Date().toISOString();
    const sessionId = `upload_session_mock_${createdUploadSessions.size + 1}`;
    const normalizedFiles = files.map((file, index) => ({
      session_id: sessionId,
      file_token: String(file.file_token || `file-${index}`).trim(),
      original_name: String(file.original_name || `upload-${index + 1}.bin`).trim(),
      relative_path: String(file.relative_path || "").trim() || undefined,
      content_type: String(file.content_type || "application/octet-stream").trim(),
      size_bytes: Math.max(0, Number(file.size_bytes) || 0),
      declared_sha256: String(file.declared_sha256 || "").trim() || undefined,
      status: "pending",
      created_at: createdAt,
      updated_at: createdAt,
      metadata: { source: "mock_api" },
    }));
    const totalBytes =
      Number.isFinite(Number(payload.total_bytes)) && Number(payload.total_bytes) >= 0
        ? Math.floor(Number(payload.total_bytes))
        : normalizedFiles.reduce((total, file) => total + file.size_bytes, 0);
    const sessionRecord = {
      session: {
        session_id: sessionId,
        owner_user_id: "user_mobile_smoke",
        owner_org_id: "org_research",
        owner_role: "owner",
        project_id: String(payload.project_id || "").trim(),
        source_type: "upload",
        status: "active",
        total_bytes: totalBytes,
        bytes_received: 0,
        bytes_verified: 0,
        bytes_committed: 0,
        idempotency_key: idempotencyKey || undefined,
        browser_fingerprint: String(payload.browser_fingerprint || "").trim() || undefined,
        created_at: createdAt,
        updated_at: createdAt,
        metadata: { source: "mock_api", file_count: normalizedFiles.length },
      },
      files: normalizedFiles,
      chunks: new Map(),
      events: [
        {
          event_id: `upload_session_event_mock_created_${createdUploadSessions.size + 1}`,
          session_id: sessionId,
          actor_user_id: "user_mobile_smoke",
          actor_org_id: "org_research",
          event_type: "upload_session.created",
          ts: createdAt,
          metadata: { file_count: normalizedFiles.length },
        },
      ],
    };
    createdUploadSessions.set(sessionId, sessionRecord);
    if (idempotencyKey) {
      createdUploadSessionIdsByIdempotency.set(idempotencyKey, sessionId);
    }
    sendJson(response, 201, uploadSessionPayload(sessionRecord));
    return;
  }

  const uploadSessionStatusMatch = url.pathname.match(/^\/v2\/upload-sessions\/([^/]+)$/);
  if (request.method === "GET" && uploadSessionStatusMatch) {
    const sessionId = decodeURIComponent(uploadSessionStatusMatch[1] || "").trim();
    const sessionRecord = createdUploadSessions.get(sessionId);
    if (!sessionRecord) {
      sendJson(response, 404, { error: "upload session not found" });
      return;
    }
    sendJson(response, 200, uploadSessionPayload(sessionRecord));
    return;
  }

  const uploadSessionControlMatch = url.pathname.match(/^\/v2\/upload-sessions\/([^/]+)\/(pause|resume|cancel)$/);
  if (request.method === "POST" && uploadSessionControlMatch) {
    const sessionId = decodeURIComponent(uploadSessionControlMatch[1] || "").trim();
    const action = uploadSessionControlMatch[2];
    const sessionRecord = createdUploadSessions.get(sessionId);
    if (!sessionRecord) {
      sendJson(response, 404, { error: "upload session not found" });
      return;
    }
    if (action === "pause") {
      sessionRecord.session.status = "paused";
      sessionRecord.session.error = "paused by user";
    } else if (action === "resume") {
      sessionRecord.session.status = "active";
      sessionRecord.session.error = "";
    } else {
      sessionRecord.session.status = "canceled";
      sessionRecord.session.error = "canceled by user";
    }
    sessionRecord.session.updated_at = new Date().toISOString();
    sendJson(response, 200, uploadSessionPayload(sessionRecord));
    return;
  }

  const uploadSessionChunkMatch = url.pathname.match(
    /^\/v2\/upload-sessions\/([^/]+)\/files\/([^/]+)\/chunks\/(\d+)$/
  );
  if (request.method === "PUT" && uploadSessionChunkMatch) {
    const sessionId = decodeURIComponent(uploadSessionChunkMatch[1] || "").trim();
    const fileToken = decodeURIComponent(uploadSessionChunkMatch[2] || "").trim();
    const chunkIndex = Number(uploadSessionChunkMatch[3]);
    const sessionRecord = createdUploadSessions.get(sessionId);
    if (!sessionRecord) {
      sendJson(response, 404, { error: "upload session not found" });
      return;
    }
    if (sessionRecord.session.status !== "active") {
      sendJson(response, 409, { error: `upload session is ${sessionRecord.session.status}` });
      return;
    }
    const file = sessionRecord.files.find((item) => item.file_token === fileToken);
    if (!file) {
      sendJson(response, 404, { error: "upload session file not found" });
      return;
    }
    if (file.status === "completed") {
      sendJson(response, 409, { error: "upload session file is already completed" });
      return;
    }
    const body = await readRequestBuffer(request);
    const offset = Math.max(0, Math.floor(Number(request.headers["x-upload-offset"] || url.searchParams.get("offset") || 0)));
    const actualSHA = sha256Hex(body);
    const declaredSHA = String(request.headers["x-upload-chunk-sha256"] || "").trim().toLowerCase();
    const chunkKey = `${fileToken}:${chunkIndex}`;
    const failedChunk = (message) => {
      const chunk = {
        session_id: sessionId,
        file_token: fileToken,
        chunk_index: chunkIndex,
        offset,
        size_bytes: body.length,
        sha256: actualSHA,
        status: "failed",
        received_at: new Date().toISOString(),
        error: message,
        metadata: { source: "mock_api", failure_stage: "chunk_validation" },
      };
      sessionRecord.chunks.set(chunkKey, chunk);
      recomputeUploadSessionCounters(sessionRecord);
      return chunk;
    };
    if (!/^[a-f0-9]{64}$/.test(declaredSHA)) {
      sendJson(response, 400, { error: "X-Upload-Chunk-Sha256 must be a sha256 hex digest" });
      return;
    }
    if (actualSHA !== declaredSHA) {
      failedChunk("chunk checksum mismatch");
      sendJson(response, 400, { error: "chunk checksum mismatch" });
      return;
    }
    if (offset + body.length > file.size_bytes) {
      failedChunk("chunk exceeds declared file size");
      sendJson(response, 400, { error: "chunk exceeds declared file size" });
      return;
    }
    const existing = sessionRecord.chunks.get(chunkKey);
    if (
      existing?.status === "verified" &&
      (existing.offset !== offset || existing.size_bytes !== body.length || existing.sha256 !== actualSHA)
    ) {
      sendJson(response, 409, { error: "verified upload chunk cannot be replaced with different bytes" });
      return;
    }
    const verifiedAt = new Date().toISOString();
    const chunk = {
      session_id: sessionId,
      file_token: fileToken,
      chunk_index: chunkIndex,
      offset,
      size_bytes: body.length,
      sha256: actualSHA,
      status: "verified",
      storage_uri: `mock://upload-sessions/${sessionId}/${fileToken}/${chunkIndex}`,
      received_at: verifiedAt,
      verified_at: verifiedAt,
      metadata: { source: "mock_api" },
    };
    sessionRecord.chunks.set(chunkKey, chunk);
    createdUploadChunkBytes.set(`${sessionId}:${chunkKey}`, body);
    if (file.status === "pending") {
      file.status = "uploading";
      file.updated_at = verifiedAt;
    }
    recomputeUploadSessionCounters(sessionRecord);
    sendJson(response, 200, { session: sessionRecord.session, file, chunk });
    return;
  }

  const uploadSessionCompleteMatch = url.pathname.match(
    /^\/v2\/upload-sessions\/([^/]+)\/files\/([^/]+)\/complete$/
  );
  if (request.method === "POST" && uploadSessionCompleteMatch) {
    const sessionId = decodeURIComponent(uploadSessionCompleteMatch[1] || "").trim();
    const fileToken = decodeURIComponent(uploadSessionCompleteMatch[2] || "").trim();
    const sessionRecord = createdUploadSessions.get(sessionId);
    if (!sessionRecord) {
      sendJson(response, 404, { error: "upload session not found" });
      return;
    }
    if (!["active", "completed"].includes(sessionRecord.session.status)) {
      sendJson(response, 409, { error: `upload session is ${sessionRecord.session.status}` });
      return;
    }
    const file = sessionRecord.files.find((item) => item.file_token === fileToken);
    if (!file) {
      sendJson(response, 404, { error: "upload session file not found" });
      return;
    }
    if (file.status === "completed" && file.resource_id) {
      const existingResource = createdUploadResources.find((resource) => resource.file_id === file.resource_id);
      sendJson(response, 200, { session: sessionRecord.session, file, resource: existingResource });
      return;
    }
    const chunks = Array.from(sessionRecord.chunks.values())
      .filter((chunk) => chunk.file_token === fileToken)
      .sort((left, right) => left.chunk_index - right.chunk_index);
    let completeBytes = Buffer.alloc(0);
    if (!(file.size_bytes === 0 && chunks.length === 0)) {
      let expectedOffset = 0;
      const parts = [];
      for (let index = 0; index < chunks.length; index += 1) {
        const chunk = chunks[index];
        if (
          chunk.chunk_index !== index ||
          chunk.status !== "verified" ||
          chunk.offset !== expectedOffset ||
          chunk.size_bytes <= 0
        ) {
          sendJson(response, 400, { error: "upload chunks are incomplete" });
          return;
        }
        parts.push(
          createdUploadChunkBytes.get(`${sessionId}:${fileToken}:${chunk.chunk_index}`) ||
            Buffer.alloc(chunk.size_bytes, 0)
        );
        expectedOffset += chunk.size_bytes;
      }
      if (expectedOffset !== file.size_bytes) {
        sendJson(response, 400, { error: "upload chunks are incomplete" });
        return;
      }
      completeBytes = Buffer.concat(parts);
    }
    const computedSHA = sha256Hex(completeBytes);
    if (file.declared_sha256 && file.declared_sha256 !== computedSHA) {
      sendJson(response, 400, { error: "completed file checksum mismatch" });
      return;
    }
    const completedAt = new Date().toISOString();
    const resource = {
      file_id: `file_mock_upload_${createdUploadResources.length + 1}`,
      original_name: file.original_name,
      content_type: file.content_type || "application/octet-stream",
      size_bytes: file.size_bytes,
      sha256: computedSHA,
      source_type: "upload",
      resource_kind: "file",
      source_uri: `upload-session://${sessionId}/${fileToken}`,
      storage_uri: `mock://resources/${fileToken}`,
      project_id: sessionRecord.session.project_id || "",
      created_at: completedAt,
      has_thumbnail: false,
      tags: [],
      metadata: { source: "mock_api", upload_session_id: sessionId, file_token: fileToken },
      share_summary: {
        share_status: "private",
        active_grant_count: 0,
        shared_by_me: false,
        shared_with_me: false,
      },
    };
    createdUploadResources.unshift(resource);
    createdUploadResourceBytes.set(resource.file_id, completeBytes);
    file.resource_id = resource.file_id;
    file.computed_sha256 = computedSHA;
    file.status = "completed";
    file.updated_at = completedAt;
    file.completed_at = completedAt;
    recomputeUploadSessionCounters(sessionRecord);
    sendJson(response, 200, { session: sessionRecord.session, file, resource });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/resources") {
    const query = String(url.searchParams.get("q") || "").trim();
    const kind = String(url.searchParams.get("kind") || "").trim();
    const source = String(url.searchParams.get("source") || "").trim();
    const sharing = String(url.searchParams.get("sharing") || "").trim();
    const status = String(url.searchParams.get("status") || "active").trim();
    const tags = String(url.searchParams.get("tags") || "")
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    const descriptors = [
      ...url.searchParams.getAll("descriptor"),
      ...String(url.searchParams.get("descriptors") || "")
        .split(",")
        .map((descriptor) => descriptor.trim())
        .filter(Boolean),
    ];
    const metadataFilter = url.searchParams.getAll("metadata_filter");
    const createdAfter = String(url.searchParams.get("created_after") || "").trim();
    const createdBefore = String(url.searchParams.get("created_before") || "").trim();
    const processingStatus = String(url.searchParams.get("processing_status") || "").trim();
    const resources = resourcesMatchingQuery({
      q: query,
      kind,
      source,
      sharing,
      status,
      tags,
      descriptors,
      metadata_filter: metadataFilter,
      created_after: createdAfter,
      created_before: createdBefore,
      processing_status: processingStatus,
    });
    sendJson(response, 200, { count: resources.length, resources });
    return;
  }

  const uploadPreviewMatch = url.pathname.match(/^\/v2\/uploads\/([^/]+)\/preview$/);
  if (request.method === "GET" && uploadPreviewMatch) {
    const resourceId = decodeURIComponent(uploadPreviewMatch[1] || "").trim();
    if (!allResources().some((resource) => resource.file_id === resourceId)) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    response.writeHead(200, {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    });
    response.end(mockViewerRasterPng);
    return;
  }

  const uploadViewerMatch = url.pathname.match(/^\/v2\/uploads\/([^/]+)\/viewer$/);
  if (request.method === "GET" && uploadViewerMatch) {
    const resourceId = decodeURIComponent(uploadViewerMatch[1] || "").trim();
    const resource = allResources().map(resourceWithLifecycle).find(
      (fixture) => fixture.file_id === resourceId && fixture.status !== "deleted"
    );
    if (!resource) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    sendJson(response, 200, mockUploadViewerInfo(resource));
    return;
  }

  const uploadRasterMatch = url.pathname.match(/^\/v2\/uploads\/([^/]+)\/(display|slice)$/);
  if (request.method === "GET" && uploadRasterMatch) {
    const resourceId = decodeURIComponent(uploadRasterMatch[1] || "").trim();
    const resource = allResources().map(resourceWithLifecycle).find(
      (fixture) => fixture.file_id === resourceId && fixture.status !== "deleted"
    );
    if (!resource) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    response.writeHead(200, {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    });
    response.end(mockViewerRasterPng);
    return;
  }

  const uploadHistogramMatch = url.pathname.match(/^\/v2\/uploads\/([^/]+)\/histogram$/);
  if (request.method === "GET" && uploadHistogramMatch) {
    const resourceId = decodeURIComponent(uploadHistogramMatch[1] || "").trim();
    const resource = allResources().map(resourceWithLifecycle).find(
      (fixture) => fixture.file_id === resourceId && fixture.status !== "deleted"
    );
    if (!resource) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    sendJson(response, 200, {
      file_id: resource.file_id,
      bins: 8,
      dtype: "uint8",
      source: "mock_api",
      sample_count: 4096,
      histogram: {
        bins: [3, 8, 18, 42, 64, 42, 18, 8],
        edges: [0, 32, 64, 96, 128, 160, 192, 224, 255],
        min: 0,
        max: 255,
        channel_indices: [0],
        time_index: 0,
      },
    });
    return;
  }

  const textHeadMatch = url.pathname.match(/^\/v2\/resources\/([^/]+)\/text-head$/);
  if (request.method === "GET" && textHeadMatch) {
    const resourceId = decodeURIComponent(textHeadMatch[1] || "").trim();
    const jsonText =
      '{\n  "model": "deepseek_v4",\n  "max_input_tokens": 786432,\n  "enable_thinking": true,\n' +
      '  "subagents": ["code-runner", "vision-reasoner"],\n  "sandbox": {\n    "memory_gb": 64,\n    "gpus": 1,\n    "network": "none"\n  }\n}\n';
    const mdText =
      "# Prairie-dog survey pipeline\n\nBatch detection over aerial transects using `MegaSeg` (GPU) and\n`RareSpot` (CPU), aggregated to the Resources catalog.\n\n## Inputs\n\n- GeoTIFF transects, 1–3 GB each\n- A `survey_2026.csv` manifest of sites\n\n## Run\n\nOpen the file in Lens, or hand it to the data agent for full-file analytics.\n";
    const csvText =
      "site_id,species,lat,lon,count,observed_at,confidence,notes\n" +
      "A-014,C. ludovicianus,41.1402,-104.8203,34,2026-05-12,0.97,burrow cluster A\n" +
      "B-002,C. gunnisoni,38.8339,-104.8214,52,2026-05-13,0.99,dense aggregation\n" +
      "C-101,C. leucurus,40.0150,-105.2705,21,2026-05-15,0.95,near road\n" +
      "D-077,C. ludovicianus,41.5868,-109.2029,88,2026-05-18,0.98,largest colony\n";
    const text = resourceId === "file_md_demo" ? mdText : resourceId === "file_csv_demo" ? csvText : jsonText;
    const format = resourceId === "file_md_demo" ? "markdown" : resourceId === "file_csv_demo" ? "csv" : "json";
    sendJson(response, 200, {
      file_id: resourceId,
      original_name: resourceId === "file_md_demo" ? "README.md" : "agent_config.json",
      content_type: resourceId === "file_md_demo" ? "text/markdown" : "application/json",
      format,
      total_size_bytes: text.length,
      offset: 0,
      returned_bytes: text.length,
      next_offset: text.length,
      truncated: false,
      encoding: "utf-8",
      eol: "lf",
      line_count: text.split("\n").length,
      approx_total_lines: text.split("\n").length,
      text,
    });
    return;
  }

  const csvRowsMatch = url.pathname.match(/^\/v2\/resources\/([^/]+)\/csv\/rows$/);
  if (request.method === "GET" && csvRowsMatch) {
    const resourceId = decodeURIComponent(csvRowsMatch[1] || "").trim();
    const offsetBytes = Number(url.searchParams.get("offset_bytes") || "0");
    const columns = ["site_id", "species", "lat", "lon", "count", "observed_at", "confidence", "notes"];
    const species = ["C. ludovicianus", "C. gunnisoni", "C. leucurus"];
    const rows = Array.from({ length: 200 }, (_, i) => {
      const n = offsetBytes / 64 + i;
      return [
        `A-${String((n % 900) + 14).padStart(3, "0")}`,
        species[Math.floor(n) % species.length],
        (41.14 + (n % 13) * 0.001).toFixed(4),
        (-104.82 - (n % 17) * 0.001).toFixed(4),
        String(((Math.floor(n) * 7) % 90) + 3),
        "2026-05-12",
        (0.6 + ((Math.floor(n) * 3) % 40) / 100).toFixed(2),
        "burrow cluster",
      ];
    });
    sendJson(response, 200, {
      file_id: resourceId,
      original_name: "survey_2026.csv",
      delimiter: ",",
      columns: offsetBytes === 0 ? columns : undefined,
      rows,
      offset_bytes: offsetBytes,
      next_offset_bytes: offsetBytes + 200 * 64,
      returned_rows: rows.length,
      has_more: offsetBytes < 64 * 2000,
      approx_total_rows: 2_400_000,
      total_size_bytes: 2_410_000_000,
    });
    return;
  }

  const resourceDownloadMatch = url.pathname.match(/^\/v2\/resources\/([^/]+)\/download$/);
  if (request.method === "GET" && resourceDownloadMatch) {
    const resourceId = decodeURIComponent(resourceDownloadMatch[1] || "").trim();
    const resource = allResources().find((fixture) => fixture.file_id === resourceId);
    if (!resource || deletedResourceIds.has(resourceId)) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    const currentResource = resourceWithLifecycle(resource);
    const body = resourceDownloadBytes(currentResource);
    response.writeHead(200, {
      "Content-Type": currentResource.content_type || "application/octet-stream",
      "Content-Disposition": `attachment; filename="${attachmentFilename(
        currentResource.original_name
      )}"`,
      "Content-Length": String(body.length),
      "Cache-Control": "no-store",
    });
    response.end(body);
    return;
  }

  const restoreResourceMatch = url.pathname.match(/^\/v2\/resources\/([^/]+)\/restore$/);
  if (request.method === "POST" && restoreResourceMatch) {
    const resourceId = decodeURIComponent(restoreResourceMatch[1] || "").trim();
    const resource = allResources().find((fixture) => fixture.file_id === resourceId);
    if (!resource) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    deletedResourceIds.delete(resourceId);
    sendJson(response, 200, { resource: resourceWithLifecycle(resource) });
    return;
  }

  const resourceLifecycleMatch = url.pathname.match(/^\/v2\/resources\/([^/]+)$/);
  if (resourceLifecycleMatch) {
    const resourceId = decodeURIComponent(resourceLifecycleMatch[1] || "").trim();
    const resource = allResources().find((fixture) => fixture.file_id === resourceId);
    if (!resource) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    if (request.method === "GET") {
      sendJson(response, 200, { resource: resourceWithLifecycle(resource) });
      return;
    }
    if (request.method === "PATCH") {
      let body = "";
      for await (const chunk of request) {
        body += chunk;
      }
      const payload = body ? JSON.parse(body) : {};
      const nextName = String(payload.original_name || "").trim();
      if (nextName) {
        resourceNameOverrides.set(resourceId, nextName);
      }
      sendJson(response, 200, {
        resource: {
          ...resourceWithLifecycle(resource),
          updated_at: new Date().toISOString(),
        },
      });
      return;
    }
    if (request.method === "DELETE") {
      deletedResourceIds.add(resourceId);
      sendJson(response, 200, { deleted: true, file_id: resourceId });
      return;
    }
  }

  if (request.method === "POST" && url.pathname === "/v2/resources/restore/bulk") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const resourceIds = Array.from(
      new Set(
        (Array.isArray(payload.resource_ids) ? payload.resource_ids : [])
          .map((value) => String(value || "").trim())
          .filter(Boolean)
      )
    );
    if (resourceIds.length === 0) {
      sendJson(response, 400, { error: "resource_ids must include at least one resource" });
      return;
    }
    const missing = resourceIds.find(
      (resourceId) => !allResources().some((resource) => resource.file_id === resourceId)
    );
    if (missing) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    const restoredAt = new Date().toISOString();
    resourceIds.forEach((resourceId) => deletedResourceIds.delete(resourceId));
    const resources = resourceFixtures
      .filter((resource) => resourceIds.includes(resource.file_id))
      .map((resource) => ({
        ...resource,
        status: "active",
        deleted_at: null,
        retention_expires_at: null,
        updated_at: restoredAt,
      }));
    const events = resources.map((resource, index) => ({
      event_id: `resource_event_mock_restored_${Date.now()}_${index}`,
      resource_id: resource.file_id,
      actor_user_id: "user_mobile_smoke",
      actor_org_id: "org_research",
      event_type: "resource.restored",
      ts: restoredAt,
      metadata: {
        source: "resources_bulk_restore",
        batch_count: resources.length,
      },
    }));
    sendJson(response, 200, { count: resources.length, resources, events });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/resources/delete/bulk") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const resourceIds = Array.from(
      new Set(
        (Array.isArray(payload.resource_ids) ? payload.resource_ids : [])
          .map((value) => String(value || "").trim())
          .filter(Boolean)
      )
    );
    if (resourceIds.length === 0) {
      sendJson(response, 400, { error: "resource_ids must include at least one resource" });
      return;
    }
    const missing = resourceIds.find((resourceId) => !allResources().some((resource) => resource.file_id === resourceId));
    if (missing) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    const deletedAt = new Date().toISOString();
    resourceIds.forEach((resourceId) => deletedResourceIds.add(resourceId));
    const resources = resourceFixtures
      .filter((resource) => resourceIds.includes(resource.file_id))
      .map((resource) => ({
        ...resource,
        status: "deleted",
        deleted_at: deletedAt,
        updated_at: deletedAt,
      }));
    const events = resources.map((resource, index) => ({
      event_id: `resource_event_mock_deleted_${Date.now()}_${index}`,
      resource_id: resource.file_id,
      actor_user_id: "user_mobile_smoke",
      actor_org_id: "org_research",
      event_type: "resource.deleted",
      ts: deletedAt,
      metadata: {
        source: "resources_bulk_delete",
        batch_count: resources.length,
      },
    }));
    sendJson(response, 200, { count: resources.length, resources, events });
    return;
  }

  if (url.pathname === "/v2/resource-collections") {
    if (request.method === "GET") {
      const collections = resourceCollectionsMatchingQuery({
        q: url.searchParams.get("q"),
        collection_type: url.searchParams.get("collection_type"),
        project_id: url.searchParams.get("project_id"),
        status: url.searchParams.get("status") || "active",
      });
      sendJson(response, 200, { count: collections.length, collections });
      return;
    }
    if (request.method === "POST") {
      let body = "";
      for await (const chunk of request) {
        body += chunk;
      }
      const payload = body ? JSON.parse(body) : {};
      const createdAt = new Date().toISOString();
      const collection = {
        collection_id: `collection_mock_${createdResourceCollections.length + 1}`,
        owner_user_id: "user_mobile_smoke",
        owner_org_id: "org_research",
        owner_role: "owner",
        project_id: payload.project_id || "",
        parent_collection_id: payload.parent_collection_id || "",
        name: String(payload.name || "Untitled folder"),
        description: payload.description || "",
        collection_type: payload.collection_type || "folder",
        status: "active",
        resource_count: 0,
        created_at: createdAt,
        updated_at: createdAt,
        metadata: payload.metadata || {},
      };
      createdResourceCollections.unshift(collection);
      sendJson(response, 201, { collection });
      return;
    }
  }

  const collectionResourcesMatch = url.pathname.match(
    /^\/v2\/resource-collections\/([^/]+)\/resources$/
  );
  if (collectionResourcesMatch) {
    const collectionId = decodeURIComponent(collectionResourcesMatch[1] || "").trim();
    const collection = findResourceCollection(collectionId);
    if (!collection || collection.status === "deleted") {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    if (request.method === "GET") {
      const query = String(url.searchParams.get("q") || "").trim();
      const kind = String(url.searchParams.get("kind") || "").trim();
      const source = String(url.searchParams.get("source") || "").trim();
      const sharing = String(url.searchParams.get("sharing") || "").trim();
      const status = String(url.searchParams.get("status") || "active").trim();
      const tags = String(url.searchParams.get("tags") || "")
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
      const descriptors = [
        ...url.searchParams.getAll("descriptor"),
        ...String(url.searchParams.get("descriptors") || "")
          .split(",")
          .map((descriptor) => descriptor.trim())
          .filter(Boolean),
      ];
      const metadataFilter = url.searchParams.getAll("metadata_filter");
      const createdAfter = String(url.searchParams.get("created_after") || "").trim();
      const createdBefore = String(url.searchParams.get("created_before") || "").trim();
      const processingStatus = String(url.searchParams.get("processing_status") || "").trim();
      const removedIds = removedCollectionResourceIds.get(collectionId) || new Set();
      const resources = resourcesMatchingQuery({
        q: query,
        kind,
        source,
        sharing,
        status,
        tags,
        descriptors,
        metadata_filter: metadataFilter,
        created_after: createdAfter,
        created_before: createdBefore,
        processing_status: processingStatus,
      }).filter((resource) => !removedIds.has(resource.file_id));
      sendJson(response, 200, {
        count: resources.length,
        total_count: resources.length,
        limit: Number(url.searchParams.get("limit") || "50"),
        offset: Number(url.searchParams.get("offset") || "0"),
        collection,
        resources,
      });
      return;
    }
    if (request.method === "POST") {
      let body = "";
      for await (const chunk of request) {
        body += chunk;
      }
      const payload = body ? JSON.parse(body) : {};
      const resourceIds = Array.from(
        new Set(
          (Array.isArray(payload.resource_ids) ? payload.resource_ids : [])
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        )
      );
      const updatedAt = new Date().toISOString();
      const updatedCollection = {
        ...collection,
        resource_count: Math.max(collection.resource_count, resourceIds.length),
        updated_at: updatedAt,
      };
      const createdIndex = createdResourceCollections.findIndex(
        (item) => item.collection_id === collectionId
      );
      if (createdIndex >= 0) {
        createdResourceCollections[createdIndex] = updatedCollection;
      }
      sendJson(response, 200, {
        collection: updatedCollection,
        memberships: resourceIds.map((resourceId, index) => ({
          collection_id: collectionId,
          resource_id: resourceId,
          position: index + 1,
          added_by_user_id: "user_mobile_smoke",
          added_at: updatedAt,
          metadata: payload.metadata || {},
        })),
      });
      return;
    }
  }

  const collectionResourceLifecycleMatch = url.pathname.match(
    /^\/v2\/resource-collections\/([^/]+)\/resources\/([^/]+)$/
  );
  if (request.method === "DELETE" && collectionResourceLifecycleMatch) {
    const collectionId = decodeURIComponent(collectionResourceLifecycleMatch[1] || "").trim();
    const resourceId = decodeURIComponent(collectionResourceLifecycleMatch[2] || "").trim();
    const collection = findResourceCollection(collectionId);
    if (!collection || collection.status === "deleted") {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    const removedIds = removedCollectionResourceIds.get(collectionId) || new Set();
    removedIds.add(resourceId);
    removedCollectionResourceIds.set(collectionId, removedIds);
    const nextCount = Math.max(0, Number(collection.resource_count || 0) - 1);
    resourceCollectionCountOverrides.set(collectionId, nextCount);
    const updatedAt = new Date().toISOString();
    const updatedCollection = {
      ...collection,
      resource_count: nextCount,
      updated_at: updatedAt,
    };
    sendJson(response, 200, {
      collection: updatedCollection,
      removed_count: 1,
      memberships: [
        {
          collection_id: collectionId,
          resource_id: resourceId,
          position: 0,
          added_by_user_id: "user_mobile_smoke",
          added_at: updatedAt,
          metadata: {},
        },
      ],
    });
    return;
  }

  const collectionRestoreMatch = url.pathname.match(
    /^\/v2\/resource-collections\/([^/]+)\/restore$/
  );
  if (request.method === "POST" && collectionRestoreMatch) {
    const collectionId = decodeURIComponent(collectionRestoreMatch[1] || "").trim();
    const collection = findResourceCollection(collectionId);
    if (!collection) {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    resourceCollectionStatusOverrides.set(collectionId, "active");
    sendJson(response, 200, {
      collection: {
        ...collection,
        status: "active",
        updated_at: new Date().toISOString(),
      },
    });
    return;
  }

  const collectionLifecycleMatch = url.pathname.match(/^\/v2\/resource-collections\/([^/]+)$/);
  if (request.method === "PATCH" && collectionLifecycleMatch) {
    const collectionId = decodeURIComponent(collectionLifecycleMatch[1] || "").trim();
    const collection = findResourceCollection(collectionId);
    if (!collection || collection.status === "deleted") {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const nextName = String(payload.name || "").trim();
    if (nextName) {
      resourceCollectionNameOverrides.set(collectionId, nextName);
    }
    sendJson(response, 200, {
      collection: {
        ...collection,
        ...(nextName ? { name: nextName } : {}),
        updated_at: new Date().toISOString(),
      },
    });
    return;
  }

  if (request.method === "DELETE" && collectionLifecycleMatch) {
    const collectionId = decodeURIComponent(collectionLifecycleMatch[1] || "").trim();
    const collection = findResourceCollection(collectionId);
    if (!collection || collection.status === "deleted") {
      sendJson(response, 404, { error: "not found" });
      return;
    }
    resourceCollectionStatusOverrides.set(collectionId, "deleted");
    sendJson(response, 200, {
      collection: {
        ...collection,
        status: "deleted",
        updated_at: new Date().toISOString(),
      },
    });
    return;
  }

  const collectionShareMatch = url.pathname.match(
    /^\/v2\/resource-collections\/([^/]+)\/shares$/
  );
  if (request.method === "POST" && collectionShareMatch) {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const collectionId = decodeURIComponent(collectionShareMatch[1]);
    const resources = resourcesMatchingQuery({});
    sendJson(response, 201, {
      count: resources.length,
      collection: {
        collection_id: collectionId,
        owner_user_id: "user_mobile_smoke",
        owner_org_id: "org_research",
        owner_role: "owner",
        name: "NPH review folder",
        collection_type: "folder",
        status: "active",
        resource_count: resources.length,
        created_at: "2026-06-08T00:00:00Z",
        updated_at: new Date().toISOString(),
        metadata: {},
      },
      grants: resources.map((resource, index) => ({
        grant_id: `resource_grant_mock_folder_${index + 1}`,
        resource_id: resource.file_id,
        owner_user_id: resource.owner_user_id || "user_mobile_smoke",
        owner_org_id: resource.owner_org_id || "org_research",
        grantee_user_id: payload.grantee_user_id || "",
        grantee_org_id: payload.grantee_org_id || "",
        role: payload.role || "read",
        status: "active",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        metadata: payload.metadata || {},
      })),
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/data-agent/jobs") {
    sendJson(response, 200, { count: createdDataAgentJobs.length, jobs: createdDataAgentJobs });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/data-agent/jobs") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const resourceIds = Array.isArray(payload.resource_ids) ? payload.resource_ids : [];
    const resourceCount =
      resourceIds.length > 0
        ? resourceIds.length
        : payload.resource_query
          ? resourcesMatchingQuery(payload.resource_query).length
          : Number(payload.metadata?.query_result_count || 0);
    const inputSelector = {
      ...(payload.input_selector || {}),
      ...(resourceIds.length > 0 ? { resource_ids: resourceIds } : {}),
      ...(payload.source_collection_id ? { source_collection_id: payload.source_collection_id } : {}),
      ...(payload.resource_query ? { resource_query: payload.resource_query } : {}),
    };
    const job = {
      job_id: `data_agent_job_mock_${createdDataAgentJobs.length + 1}`,
      owner_user_id: "user_mobile_smoke",
      owner_org_id: "org_research",
      owner_role: "owner",
      project_id: payload.project_id || "project_nph",
      job_type: payload.job_type || "create_dataset_snapshot",
      status: "queued",
      resource_count: resourceCount,
      progress_completed: 0,
      progress_total: resourceCount,
      created_by_user_id: "user_mobile_smoke",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      input_selector: inputSelector,
      output_summary: {},
      metadata: payload.metadata || {},
    };
    const event = {
      event_id: `data_agent_event_mock_${createdDataAgentJobs.length + 1}`,
      job_id: job.job_id,
      sequence: 1,
      event_type: "data_agent.job.created",
      actor_user_id: "user_mobile_smoke",
      actor_org_id: "org_research",
      ts: job.created_at,
      message: "Data Agent job queued.",
      metadata: {},
    };
    createdDataAgentJobs.unshift(job);
    sendJson(response, 202, { job, events: [event] });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/dataset-snapshots") {
    const status = String(url.searchParams.get("status") || "active").trim();
    const snapshots = [...createdDatasetSnapshots, datasetSnapshotFixture]
      .map(datasetSnapshotWithLifecycle)
      .filter((snapshot) => String(snapshot.status || "active") === status);
    sendJson(response, 200, { count: snapshots.length, snapshots });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/dataset-snapshots") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const snapshotId = `dataset_snapshot_mock_${createdDatasetSnapshots.length + 1}`;
    const resources = resourceFixtures.map((resource, index) => ({
      snapshot_id: snapshotId,
      resource_id: resource.file_id,
      position: index,
      original_name: resource.original_name,
      content_type: resource.content_type,
      size_bytes: resource.size_bytes,
      sha256: resource.sha256,
      source_type: resource.source_type,
      resource_kind: resource.resource_kind,
      project_id: resource.project_id,
      resource_created_at: resource.created_at,
      metadata: resource.metadata,
    }));
    const snapshot = {
      snapshot_id: snapshotId,
      owner_user_id: "user_mobile_smoke",
      owner_org_id: "org_research",
      owner_role: "owner",
      project_id: "project_nph",
      name: String(payload.name || "Query dataset snapshot"),
      description: payload.description || null,
      status: "active",
      resource_count: resources.length,
      total_bytes: resources.reduce((total, resource) => total + resource.size_bytes, 0),
      created_by_user_id: "user_mobile_smoke",
      created_at: new Date().toISOString(),
      metadata: payload.metadata || { source: "mock_api" },
    };
    createdDatasetSnapshots.unshift(snapshot);
    createdDatasetSnapshotResources.set(snapshotId, resources);
    datasetSnapshotShareGrants.set(snapshotId, []);
    datasetSnapshotEvents.set(snapshotId, [
      {
        event_id: `dataset_snapshot_event_mock_created_${createdDatasetSnapshots.length}`,
        snapshot_id: snapshotId,
        actor_user_id: "user_mobile_smoke",
        actor_org_id: "org_research",
        event_type: "dataset_snapshot.created",
        ts: snapshot.created_at,
        metadata: {
          snapshot_name: snapshot.name,
          resource_count: snapshot.resource_count,
          total_bytes: snapshot.total_bytes,
          project_id: snapshot.project_id,
          source_collection_id: snapshot.source_collection_id || null,
          source: snapshot.metadata?.source || "mock_api",
        },
      },
    ]);
    sendJson(response, 201, { snapshot, resources });
    return;
  }

  const datasetSnapshotLifecycleMatch = url.pathname.match(
    /^\/v2\/dataset-snapshots\/([^/]+)(?:\/restore)?$/
  );
  if (
    datasetSnapshotLifecycleMatch &&
    (request.method === "DELETE" || request.method === "POST")
  ) {
    const snapshotId = decodeURIComponent(datasetSnapshotLifecycleMatch[1] || "").trim();
    const restore = url.pathname.endsWith("/restore");
    if ((restore && request.method !== "POST") || (!restore && request.method !== "DELETE")) {
      sendJson(response, 405, { error: "method not allowed" });
      return;
    }
    const baseSnapshot = datasetSnapshotBaseById(snapshotId);
    if (!baseSnapshot) {
      sendJson(response, 404, { error: "dataset snapshot not found" });
      return;
    }
    const status = restore ? "active" : "deleted";
    const eventType = restore ? "dataset_snapshot.restored" : "dataset_snapshot.deleted";
    const timestamp = new Date().toISOString();
    datasetSnapshotStatusOverrides.set(snapshotId, status);
    const snapshot = datasetSnapshotWithLifecycle(baseSnapshot);
    const resources = datasetSnapshotResourcesById(snapshotId);
    datasetSnapshotEvents.set(snapshotId, [
      {
        event_id: `dataset_snapshot_event_mock_${restore ? "restored" : "deleted"}_${Date.now()}`,
        snapshot_id: snapshotId,
        actor_user_id: "user_mobile_smoke",
        actor_org_id: "org_research",
        event_type: eventType,
        ts: timestamp,
        metadata: {
          snapshot_id: snapshot.snapshot_id,
          snapshot_name: snapshot.name,
          resource_count: snapshot.resource_count,
          total_bytes: snapshot.total_bytes,
          project_id: snapshot.project_id,
          source_collection_id: snapshot.source_collection_id,
          source: "dataset_snapshot_lifecycle",
          [restore ? "restored_at" : "deleted_at"]: timestamp,
        },
      },
      ...(datasetSnapshotEvents.get(snapshotId) || []),
    ]);
    sendJson(response, 200, { snapshot, resources });
    return;
  }

  const datasetEventsMatch = url.pathname.match(/^\/v2\/dataset-snapshots\/([^/]+)\/events$/);
  if (datasetEventsMatch && request.method === "GET") {
    const snapshotId = decodeURIComponent(datasetEventsMatch[1]);
    const eventType = String(url.searchParams.get("event_type") || "").trim();
    const actorUserId = String(url.searchParams.get("actor_user_id") || "").trim();
    const limit = Math.max(1, Math.min(1000, Number(url.searchParams.get("limit") || "200")));
    const offset = Math.max(0, Number(url.searchParams.get("offset") || "0"));
    const events = (datasetSnapshotEvents.get(snapshotId) || []).filter((event) => {
      if (eventType && event.event_type !== eventType) {
        return false;
      }
      return !actorUserId || event.actor_user_id === actorUserId;
    });
    const page = events.slice(offset, offset + limit);
    sendJson(response, 200, {
      snapshot_id: snapshotId,
      count: page.length,
      total_count: events.length,
      limit,
      offset,
      events: page,
    });
    return;
  }

  const datasetShareMatch = url.pathname.match(/^\/v2\/dataset-snapshots\/([^/]+)\/shares(?:\/([^/]+))?$/);
  if (datasetShareMatch) {
    const snapshotId = decodeURIComponent(datasetShareMatch[1]);
    const grantId = datasetShareMatch[2] ? decodeURIComponent(datasetShareMatch[2]) : "";
    const existing = datasetSnapshotShareGrants.get(snapshotId) || [];
    if (request.method === "GET" && !grantId) {
      const status = String(url.searchParams.get("status") || "").trim();
      const grants = status ? existing.filter((grant) => grant.status === status) : existing;
      sendJson(response, 200, { count: grants.length, grants });
      return;
    }
    if (request.method === "POST" && !grantId) {
      let body = "";
      for await (const chunk of request) {
        body += chunk;
      }
      const payload = body ? JSON.parse(body) : {};
      const createdAt = new Date().toISOString();
      const grant = {
        grant_id: `dataset_snapshot_grant_mock_${existing.length + 1}`,
        snapshot_id: snapshotId,
        owner_user_id: "user_mobile_smoke",
        owner_org_id: "org_research",
        owner_role: "owner",
        grantee_user_id: payload.grantee_user_id || null,
        grantee_org_id: payload.grantee_org_id || null,
        role: payload.role || "read",
        status: "active",
        created_by_user_id: "user_mobile_smoke",
        created_at: createdAt,
        updated_at: createdAt,
        metadata: payload.metadata || { source: "mock_api" },
      };
      datasetSnapshotShareGrants.set(snapshotId, [grant, ...existing]);
      datasetSnapshotEvents.set(snapshotId, [
        {
          event_id: `dataset_snapshot_event_mock_shared_${existing.length + 1}`,
          snapshot_id: snapshotId,
          actor_user_id: "user_mobile_smoke",
          actor_org_id: "org_research",
          event_type: "dataset_snapshot.shared",
          ts: createdAt,
          metadata: {
            grant_id: grant.grant_id,
            grantee_user_id: grant.grantee_user_id,
            grantee_org_id: grant.grantee_org_id,
            role: grant.role,
          },
        },
        ...(datasetSnapshotEvents.get(snapshotId) || []),
      ]);
      sendJson(response, 201, { grant });
      return;
    }
    if (request.method === "DELETE" && grantId) {
      const updatedAt = new Date().toISOString();
      const grants = existing.map((grant) =>
        grant.grant_id === grantId
          ? { ...grant, status: "revoked", revoked_at: updatedAt, updated_at: updatedAt }
          : grant
      );
      datasetSnapshotShareGrants.set(snapshotId, grants);
      const grant = grants.find((item) => item.grant_id === grantId);
      if (!grant) {
        sendJson(response, 404, { error: "dataset snapshot share grant not found" });
        return;
      }
      datasetSnapshotEvents.set(snapshotId, [
        {
          event_id: `dataset_snapshot_event_mock_revoked_${grantId}`,
          snapshot_id: snapshotId,
          actor_user_id: "user_mobile_smoke",
          actor_org_id: "org_research",
          event_type: "dataset_snapshot.share_revoked",
          ts: updatedAt,
          metadata: {
            grant_id: grant.grant_id,
            grantee_user_id: grant.grantee_user_id,
            grantee_org_id: grant.grantee_org_id,
            role: grant.role,
          },
        },
        ...(datasetSnapshotEvents.get(snapshotId) || []),
      ]);
      sendJson(response, 200, { grant });
      return;
    }
  }

  if (
    request.method === "GET" &&
    url.pathname === `/v2/dataset-snapshots/${datasetSnapshotFixture.snapshot_id}`
  ) {
    const snapshot = datasetSnapshotWithLifecycle(datasetSnapshotFixture);
    if (snapshot.status === "deleted") {
      sendJson(response, 404, { error: "dataset snapshot not found" });
      return;
    }
    sendJson(response, 200, {
      snapshot,
      resources: datasetSnapshotResourcesFixture,
    });
    return;
  }

  for (const [snapshotId, resources] of createdDatasetSnapshotResources.entries()) {
    if (request.method === "GET" && url.pathname === `/v2/dataset-snapshots/${snapshotId}`) {
      const baseSnapshot = createdDatasetSnapshots.find((item) => item.snapshot_id === snapshotId);
      const snapshot = baseSnapshot ? datasetSnapshotWithLifecycle(baseSnapshot) : null;
      if (!snapshot || snapshot.status === "deleted") {
        sendJson(response, 404, { error: "dataset snapshot not found" });
        return;
      }
      sendJson(response, 200, { snapshot, resources });
      return;
    }
  }

  if (request.method === "POST" && url.pathname === "/v2/bisque/search") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const resourceType = String(payload.resource_type || "image").toLowerCase();
    const counts = { image: 142, dataset: 12, table: 8 };
    sendJson(response, 200, {
      count: counts[resourceType] ?? 0,
      results: [],
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/bisque/push") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const fileIds = Array.isArray(payload.file_ids) ? payload.file_ids : [];
    const collectionIds = Array.isArray(payload.collection_ids) ? payload.collection_ids : [];
    const uploads = fileIds.map((fileId, index) => ({
      file_id: fileId,
      resource_uri: `${bisqueRoot}/data_service/00-MOCKPUSH${index + 1}`,
      name: `mock-upload-${index + 1}.png`,
      resource_uniq: `00-MOCKPUSH${index + 1}`,
      client_view_url: `${bisqueRoot}/client_service/view?resource=00-MOCKPUSH${index + 1}`,
    }));
    const datasets = collectionIds.map((collectionId, index) => ({
      collection_id: collectionId,
      name: String(payload.dataset_name || `Mock dataset ${index + 1}`),
      resource_uri: `${bisqueRoot}/data_service/00-MOCKDATASET${index + 1}`,
      resource_uniq: `00-MOCKDATASET${index + 1}`,
      member_count: 2,
      client_view_url: `${bisqueRoot}/client_service/view?resource=00-MOCKDATASET${index + 1}`,
    }));
    collectionIds.forEach((collectionId, collectionIndex) => {
      uploads.push(
        ...[1, 2].map((memberIndex) => ({
          file_id: `mock-member-${collectionIndex + 1}-${memberIndex}`,
          resource_uri: `${bisqueRoot}/data_service/00-MOCKMEMBER${collectionIndex + 1}${memberIndex}`,
          name: `mock-member-${memberIndex}.png`,
          resource_uniq: `00-MOCKMEMBER${collectionIndex + 1}${memberIndex}`,
          client_view_url: `${bisqueRoot}/client_service/view?resource=00-MOCKMEMBER${collectionIndex + 1}${memberIndex}`,
        }))
      );
    });
    sendJson(response, 200, {
      count: uploads.length,
      uploads,
      datasets,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/models") {
    sendJson(response, 200, {
      count: 1,
      models: [
        {
          key: "prairie_yolo",
          name: "Prairie YOLO",
          framework: "yolov5",
          task_type: "object_detection",
          description: "Mock prairie detection model for frontend performance tests.",
          supports_training: true,
          supports_finetune: true,
          supports_inference: true,
          dimensions: ["image"],
          default_config: {},
        },
      ],
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/prairie/status") {
    sendJson(response, 200, {
      dataset_name: "Prairie_Dog_Active_Learning",
      dataset_id: "dataset_prairie",
      last_sync_at: nowIso,
      next_sync_at: nowIso,
      active_model_version: "version_active",
      model_health: "healthy",
      reviewed_images: 24,
      unreviewed_images: 3,
      class_counts: { prairie_dog: 48, burrow: 19 },
      unsupported_class_counts: {},
      detection_counts: {},
      latest_metrics: { map50: 0.91 },
      benchmark_baseline: {},
      benchmark_latest_candidate: {},
      last_benchmark_at: nowIso,
      benchmark_ready: true,
      canonical_benchmark_ready: true,
      promotion_benchmark_ready: true,
      retrain_gate: true,
      retrain_gate_reasons: [],
      retrain_gate_counts: {},
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/prairie/retrain-requests") {
    sendJson(response, 200, { count: 0, requests: [] });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/domains") {
    sendJson(response, 200, {
      count: 1,
      domains: [
        {
          domain_id: "domain_prairie",
          name: "Prairie",
          description: "Mock training domain",
          owner_scope: "shared",
          owner_user_id: "user_mobile_smoke",
          metadata: {},
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
    });
    return;
  }

  if (
    request.method === "GET" &&
    url.pathname === "/v2/training/domains/domain_prairie/lineages"
  ) {
    sendJson(response, 200, {
      count: 1,
      lineages: [
        {
          lineage_id: "lineage_prairie",
          domain_id: "domain_prairie",
          scope: "shared",
          owner_user_id: "user_mobile_smoke",
          model_key: "prairie_yolo",
          parent_lineage_id: null,
          active_version_id: "version_active",
          metadata: {},
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
    });
    return;
  }

  if (
    request.method === "GET" &&
    url.pathname === "/v2/training/lineages/lineage_prairie/versions"
  ) {
    sendJson(response, 200, {
      count: 1,
      versions: [
        {
          version_id: "version_active",
          lineage_id: "lineage_prairie",
          source_job_id: null,
          artifact_run_id: null,
          status: "active",
          metrics: { benchmark_ready: true, promotion_benchmark_ready: true },
          metadata: { guardrails: { passed: true, reasons: [] } },
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/auth/session") {
    const sessionMode = readSessionMode(request);
    if (!sessionMode) {
      sendJson(response, 200, { authenticated: false });
      return;
    }
    sendJson(response, 200, {
      authenticated: true,
      username: sessionMode === "guest" ? "Mobile Smoke" : "Mock BisQue User",
      bisque_root: bisqueRoot,
      expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
      mode: sessionMode === "guest" ? "guest" : "bisque",
      guest_profile:
        sessionMode === "guest"
          ? {
              name: "Mobile Smoke",
              email: "mobile.smoke@example.com",
              affiliation: "BisQue Ultra QA",
            }
          : null,
      is_admin: false,
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/login") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    sendJson(
      response,
      200,
      {
        authenticated: true,
        username: String(payload.username || "Mock BisQue User"),
        bisque_root: bisqueRoot,
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        mode: "bisque",
        guest_profile: null,
        is_admin: false,
      },
      {
        "Set-Cookie": `${guestCookieName}=bisque; Path=/; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/logout") {
    sendJson(
      response,
      200,
      { authenticated: false },
      {
        "Set-Cookie": `${guestCookieName}=; Path=/; Max-Age=0; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/auth/logout/browser") {
    response.writeHead(302, {
      Location: browserLogoutRedirectUrl(url.searchParams.get("next")),
      "Set-Cookie": `${guestCookieName}=; Path=/; Max-Age=0; SameSite=Lax`,
    });
    response.end();
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/guest") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    sendJson(
      response,
      200,
      {
        authenticated: true,
        username: String(payload.name || "Mobile Smoke"),
        bisque_root: bisqueRoot,
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        mode: "guest",
        guest_profile: {
          name: String(payload.name || "Mobile Smoke"),
          email: String(payload.email || "mobile.smoke@example.com"),
          affiliation: String(payload.affiliation || "BisQue Ultra QA"),
        },
        is_admin: false,
      },
      {
        "Set-Cookie": `${guestCookieName}=guest; Path=/; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/resources") {
    sendJson(response, 200, { count: 0, resources: [] });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/conversations") {
    sendJson(response, 200, {
      count: 0,
      conversations: [],
      offset: 0,
      limit: 50,
      has_more: false,
      next_offset: null,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/fun/weather/santa-barbara") {
    sendJson(response, 200, {
      success: false,
      location: "Santa Barbara, CA",
      micro_location: "Campus Point",
      blip: "Weather is unavailable in smoke mode.",
      summary: "Weather is unavailable in smoke mode.",
      source: "mock",
    });
    return;
  }

  sendJson(response, 404, { detail: `Unhandled mock endpoint: ${request.method} ${url.pathname}` });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Mock API listening on http://127.0.0.1:${port}`);
});
