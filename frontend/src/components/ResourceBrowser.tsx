import {
  type DragEvent,
  type KeyboardEvent,
  type MouseEvent,
  type UIEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  collectDroppedFiles,
  isOsFileDrag,
  normalizePickedFiles,
  snapshotDropPayload,
  summarizeDropIssues,
} from "@/lib/dropTraversal";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useBreakpoint } from "@/hooks/use-breakpoint";
import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/format";
import { VideoThumbnail } from "./viewer/VideoThumbnail";
import { ZScrubThumbnail } from "./viewer/ZScrubThumbnail";
import { ResourceQuickPeek } from "./ResourceQuickPeek";
import { ResourceThumbnailPreview } from "./ResourceThumbnailPreview";
import { classifyTextResource } from "@/lib/textFormat";
import { resourceDisplayName } from "@/features/resources/presentation";
import {
  ArrowLeft,
  ChevronDown,
  CloudUpload,
  Download,
  Eye,
  File,
  FileText,
  Folder,
  FolderMinus,
  Film,
  FolderPlus,
  FolderUp,
  ImageIcon,
  LayoutGrid,
  Loader2,
  Pause,
  Pencil,
  RefreshCw,
  RotateCcw,
  Search,
  Share2,
  SlidersHorizontal,
  Table2,
  Trash2,
  Upload,
  UserPlus,
  XCircle,
} from "lucide-react";
import type {
  ResourceCollectionRecord,
  ResourceRecord,
  ResourceShareGrantRecord,
  ResourceCollectionShareGrantRecord,
  ShareTargetRecord,
  ResourceTextHead,
} from "../types";

export type ResourceKindFilter = "all" | "image" | "video" | "table" | "file";
export type ResourceSourceFilter = "all" | "upload" | "bisque_import";
export type ResourceSharingFilter = "all" | "private" | "shared_by_me" | "shared_with_me" | "public";
export type ResourceStatusFilter = "active" | "deleted";
type ResourceViewMode = "cards" | "table";

export type ResourceUploadProgress = {
  id: string;
  fingerprint?: string | null;
  sessionId?: string | null;
  fileToken?: string | null;
  fileName: string;
  relativePath?: string | null;
  status: "queued" | "creating" | "uploading" | "verifying" | "completed" | "failed" | string;
  totalBytes: number;
  bytesVerified: number;
  error?: string | null;
};

export type ResourceUploadReselectionContext = {
  resumeFrom?: ResourceUploadProgress;
  uploadTargetCollection?: ResourceCollectionRecord;
};

export type ResourceCollectionSelectionRequest = {
  collectionType: "folder";
  name: string;
  resourceIds: string[];
};

export type ResourceCollectionAddSelectionRequest = {
  collectionId: string;
  resourceIds: string[];
};

export type ResourceShareGrantRequest = {
  granteeUserId?: string;
  granteeOrgId?: string;
  role: "read" | string;
};

type ResourceBrowserProps = {
  resources: ResourceRecord[];
  resourceCollections?: ResourceCollectionRecord[];
  resourceCollectionsLoading?: boolean;
  totalCount: number;
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  query: string;
  kindFilter: ResourceKindFilter;
  sourceFilter: ResourceSourceFilter;
  sharingFilter?: ResourceSharingFilter;
  statusFilter?: ResourceStatusFilter;
  tagFilter?: string;
  deletingFileIds: Record<string, boolean>;
  restoringFileIds?: Record<string, boolean>;
  onQueryChange: (value: string) => void;
  /** Bumped by the mobile nav bar's search action to reveal + focus the field. */
  focusSearchSignal?: number;
  onKindFilterChange: (value: ResourceKindFilter) => void;
  onSourceFilterChange: (value: ResourceSourceFilter) => void;
  onSharingFilterChange?: (value: ResourceSharingFilter) => void;
  onStatusFilterChange?: (value: ResourceStatusFilter) => void;
  onTagFilterChange?: (value: string) => void;
  onRefresh: () => void;
  onLoadMore: () => void;
  onUploadFiles?: (files: File[], context?: ResourceUploadReselectionContext) => void;
  uploading?: boolean;
  uploadProgress?: ResourceUploadProgress[];
  onDismissUploadProgress?: (item: ResourceUploadProgress) => void;
  onPauseUploadProgress?: (item: ResourceUploadProgress) => void;
  onCancelUploadProgress?: (item: ResourceUploadProgress) => void;
  onOpenResource: (resource: ResourceRecord) => void;
  onUseInChat: (resource: ResourceRecord) => void;
  onDeleteResource: (resource: ResourceRecord) => void;
  onRenameResource?: (resource: ResourceRecord, name: string) => Promise<void> | void;
  onRestoreResource?: (resource: ResourceRecord) => void;
  onRemoveResourceFromCollection?: (resource: ResourceRecord) => Promise<void> | void;
  onLoadResourceShareGrants?: (
    resource: ResourceRecord
  ) => Promise<ResourceShareGrantRecord[]> | ResourceShareGrantRecord[];
  onCreateResourceShareGrant?: (
    resource: ResourceRecord,
    request: ResourceShareGrantRequest
  ) => Promise<ResourceShareGrantRecord> | ResourceShareGrantRecord;
  onCreateBulkResourceShareGrants?: (
    resources: ResourceRecord[],
    request: ResourceShareGrantRequest
  ) => Promise<ResourceShareGrantRecord[]> | ResourceShareGrantRecord[];
  onCreateResourceCollectionShareGrants?: (
    collection: ResourceCollectionRecord,
    request: ResourceShareGrantRequest
  ) => Promise<ResourceShareGrantRecord[]> | ResourceShareGrantRecord[];
  onDeleteSelectedResources?: (resources: ResourceRecord[]) => Promise<void> | void;
  onRestoreSelectedResources?: (resources: ResourceRecord[]) => Promise<void> | void;
  onRevokeResourceShareGrant?: (
    resource: ResourceRecord,
    grant: ResourceShareGrantRecord
  ) => Promise<ResourceShareGrantRecord> | ResourceShareGrantRecord;
  onCreateCollectionFromSelection?: (
    request: ResourceCollectionSelectionRequest
  ) => Promise<void> | void;
  onAddSelectionToCollection?: (
    request: ResourceCollectionAddSelectionRequest
  ) => Promise<void> | void;
  activeResourceCollection?: ResourceCollectionRecord | null;
  onOpenCollection?: (collection: ResourceCollectionRecord) => void;
  onRenameCollection?: (collection: ResourceCollectionRecord, name: string) => Promise<void> | void;
  onDeleteCollection?: (collection: ResourceCollectionRecord) => Promise<void> | void;
  onRestoreCollection?: (collection: ResourceCollectionRecord) => Promise<void> | void;
  restoringCollectionIds?: Record<string, boolean>;
  onClearActiveCollection?: () => void;
  thumbnailUrlFor: (resource: ResourceRecord) => string;
  downloadUrlFor?: (resource: ResourceRecord) => string;
  onSearchShareTargets?: (query: string) => Promise<ShareTargetRecord[]>;
  onLoadCollectionShareGrants?: (
    collection: ResourceCollectionRecord
  ) => Promise<ResourceCollectionShareGrantRecord[]>;
  onRevokeCollectionShareGrant?: (
    collection: ResourceCollectionRecord,
    grant: ResourceCollectionShareGrantRecord
  ) => Promise<ResourceCollectionShareGrantRecord>;
  collectionDownloadUrlFor?: (collection: ResourceCollectionRecord) => string;
  quickPeekFetch?: (fileId: string, maxBytes: number) => Promise<ResourceTextHead>;
  onPushResourceToBisque?: (resource: ResourceRecord) => Promise<void> | void;
  onPushCollectionToBisque?: (collection: ResourceCollectionRecord) => Promise<void> | void;
  zScrubThumbnail?: {
    sliceUrlFor: (fileId: string, z: number) => string;
    loadZCount: (fileId: string) => Promise<number>;
  };
};

const kindFilters: Array<{ value: ResourceKindFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "image", label: "Images" },
  { value: "video", label: "Videos" },
  { value: "table", label: "Tables" },
  { value: "file", label: "Files" },
];

const sourceFilters: Array<{ value: ResourceSourceFilter; label: string }> = [
  { value: "all", label: "All sources" },
  { value: "upload", label: "Uploads" },
  { value: "bisque_import", label: "BisQue" },
];

const sharingFilters: Array<{ value: ResourceSharingFilter; label: string }> = [
  { value: "all", label: "All sharing" },
  { value: "private", label: "Private" },
  { value: "shared_by_me", label: "Shared by me" },
  { value: "shared_with_me", label: "Shared with me" },
  { value: "public", label: "Public" },
];

const statusFilters: Array<{ value: ResourceStatusFilter; label: string }> = [
  { value: "active", label: "Active" },
  { value: "deleted", label: "Deleted" },
];

const RESOURCE_SKELETON_COUNT = 8;
const RESOURCE_TABLE_VIRTUALIZATION_THRESHOLD = 80;
const RESOURCE_TABLE_ROW_HEIGHT = 68;
const RESOURCE_TABLE_OVERSCAN_ROWS = 8;
const RESOURCE_TABLE_FALLBACK_VIEWPORT_ROWS = 12;
const RESOURCE_VIEW_MODE_STORAGE_KEY = "bisque.resources.view_mode";
const RESOURCE_DRAG_ID_MIME = "application/x-bisque-resource-id";
const RESOURCE_DRAG_IDS_MIME = "application/x-bisque-resource-ids";

const isResourceViewMode = (value: string | null): value is ResourceViewMode =>
  value === "cards" || value === "table";

const readStoredResourceViewMode = (): ResourceViewMode => {
  try {
    const value = globalThis.localStorage?.getItem(RESOURCE_VIEW_MODE_STORAGE_KEY) ?? null;
    return isResourceViewMode(value) ? value : "cards";
  } catch {
    return "cards";
  }
};

const persistResourceViewMode = (value: ResourceViewMode): void => {
  try {
    globalThis.localStorage?.setItem(RESOURCE_VIEW_MODE_STORAGE_KEY, value);
  } catch {
    // Storage access can fail in private browsing or restricted test environments.
  }
};

const parseDraggedResourceIds = (value: string): string[] => {
  if (!value) {
    return [];
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      return [];
    }
    const seen = new Set<string>();
    const ids: string[] = [];
    parsed.forEach((item) => {
      const id = String(item ?? "").trim();
      if (id && !seen.has(id)) {
        seen.add(id);
        ids.push(id);
      }
    });
    return ids;
  } catch {
    return [];
  }
};

const formatResourceDate = (value: string): string => {
  try {
    return new Date(value).toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return value;
  }
};

const formatResourceAuditDate = (value: string): string => {
  try {
    return new Date(value).toLocaleString([], {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return value;
  }
};

const resourceKindLabel = (kind: string): string => {
  const normalized = String(kind || "").toLowerCase();
  if (!normalized) {
    return "File";
  }
  return normalized[0].toUpperCase() + normalized.slice(1);
};

const sourceLabel = (value: string): string => {
  if (value === "bisque_import") {
    return "BisQue";
  }
  if (value === "upload") {
    return "Upload";
  }
  return value || "Source";
};

const uploadProgressStatusLabel = (value: string): string => {
  switch (String(value || "").toLowerCase()) {
    case "queued":
      return "Queued";
    case "creating":
      return "Preparing";
    case "verifying":
      return "Verifying";
    case "paused":
      return "Paused";
    case "completed":
      return "Completed";
    case "needs_file":
      return "Ready to resume";
    case "failed":
      return "Failed";
    case "uploading":
    default:
      return "Uploading";
  }
};

const uploadProgressRecoveryLabel = (value: string): string | null => {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "needs_file" || normalized === "paused") {
    return "Resume";
  }
  if (normalized === "failed") {
    return "Retry";
  }
  return null;
};

const uploadProgressErrorLabel = (value: string | null | undefined): string | null => {
  const text = String(value ?? "").trim();
  if (!text) {
    return null;
  }
  if (/request failed with status (404|410)/i.test(text)) {
    return "Upload session expired. Start this upload again.";
  }
  return text;
};

const isExpiredUploadProgress = (item: ResourceUploadProgress): boolean =>
  String(item.status || "").toLowerCase() === "failed" &&
  uploadProgressErrorLabel(item.error) === "Upload session expired. Start this upload again.";

const canDismissUploadProgress = (value: string): boolean => {
  const normalized = String(value || "").toLowerCase();
  return normalized === "failed" || normalized === "needs_file" || normalized === "canceled";
};

const canCancelUploadProgress = (item: ResourceUploadProgress): boolean => {
  const status = String(item.status || "").toLowerCase();
  return Boolean(item.sessionId) && status !== "completed" && status !== "canceled";
};

const canPauseUploadProgress = (item: ResourceUploadProgress): boolean => {
  const status = String(item.status || "").toLowerCase();
  return Boolean(item.sessionId) && ["queued", "creating", "uploading", "verifying"].includes(status);
};

const normalizeUploadRelativePath = (value: unknown): string | null => {
  const path = String(value ?? "")
    .replace(/\\/g, "/")
    .split("/")
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0 && segment !== "." && segment !== "..")
    .join("/");
  return path.length > 0 ? path : null;
};

type ParsedUploadFingerprint = {
  identityName: string;
  sizeBytes: number;
  lastModified: number;
  contentType: string;
};

const parseUploadFingerprint = (value: unknown): ParsedUploadFingerprint | null => {
  const text = String(value ?? "").trim();
  if (!text) {
    return null;
  }
  const segments = text.split(":");
  if (/^[a-f0-9]{64}$/i.test(segments[segments.length - 1] ?? "")) {
    segments.pop();
  }
  if (segments.length < 4) {
    return null;
  }
  const contentType = String(segments.pop() ?? "").trim();
  const lastModified = Number(segments.pop());
  const sizeBytes = Number(segments.pop());
  const identityName = segments.join(":").trim();
  if (!identityName || !contentType || !Number.isFinite(sizeBytes) || !Number.isFinite(lastModified)) {
    return null;
  }
  return {
    identityName,
    sizeBytes: Math.max(0, Math.floor(sizeBytes)),
    lastModified: Math.max(0, Math.floor(lastModified)),
    contentType,
  };
};

const fileMatchesUploadProgress = (file: File, item: ResourceUploadProgress): boolean => {
  const parsedFingerprint = parseUploadFingerprint(item.fingerprint);
  if (parsedFingerprint) {
    const selectedIdentity =
      normalizeUploadRelativePath((file as File & { webkitRelativePath?: string }).webkitRelativePath) ??
      (file.name || "upload.bin");
    if (selectedIdentity !== parsedFingerprint.identityName) {
      return false;
    }
    if (file.size !== parsedFingerprint.sizeBytes) {
      return false;
    }
    if (Math.max(0, Math.floor(Number(file.lastModified) || 0)) !== parsedFingerprint.lastModified) {
      return false;
    }
    if ((file.type || "application/octet-stream") !== parsedFingerprint.contentType) {
      return false;
    }
  }
  const expectedName = item.fileName.trim();
  if (expectedName && (file.name || "upload.bin") !== expectedName) {
    return false;
  }
  const expectedBytes = Math.max(0, Math.floor(Number(item.totalBytes) || 0));
  if (expectedBytes > 0 && file.size !== expectedBytes) {
    return false;
  }
  const expectedPath = normalizeUploadRelativePath(item.relativePath);
  if (!expectedPath) {
    return true;
  }
  const selectedPath = normalizeUploadRelativePath(
    (file as File & { webkitRelativePath?: string }).webkitRelativePath
  );
  return selectedPath === expectedPath;
};

const pluralSummary = (count: number, singular: string, plural = `${singular}s`): string =>
  `${count.toLocaleString()} ${count === 1 ? singular : plural}`;

type ResourceShareStatusChip = {
  key: string;
  label: string;
  title: string;
};

const resourceShareStatusChips = (resource: ResourceRecord): ResourceShareStatusChip[] => {
  const summary = resource.share_summary;
  if (!summary) {
    return [];
  }
  const status = String(summary.share_status || "").toLowerCase();
  const grantCount = Math.max(0, Math.floor(Number(summary.active_grant_count) || 0));
  if (status === "shared_by_me") {
    const label =
      grantCount > 0
        ? `${grantCount.toLocaleString()} active ${grantCount === 1 ? "grant" : "grants"}`
        : "Shared by me";
    return [{ key: "shared_by_me", label, title: "Shared by me" }];
  }
  if (status === "shared_with_me") {
    return [{ key: "shared_with_me", label: "Shared with me", title: "Shared with me" }];
  }
  if (status === "public") {
    return [{ key: "public", label: "Public", title: "Public resource" }];
  }
  return [];
};

const resourceLifecycleStatus = (resource: ResourceRecord): ResourceStatusFilter =>
  String(resource.status || "").toLowerCase() === "deleted" ? "deleted" : "active";

const isDeletedResource = (resource: ResourceRecord): boolean =>
  resourceLifecycleStatus(resource) === "deleted";

const normalizeResourceTags = (tags: readonly unknown[] | undefined): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  tags?.forEach((tag) => {
    const trimmed = String(tag ?? "").trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    result.push(trimmed);
  });
  return result;
};

const resourceShareGrantStatusLabel = (value: string): string => {
  switch (String(value || "").toLowerCase()) {
    case "active":
      return "Active";
    case "revoked":
      return "Revoked";
    default:
      return String(value || "Unknown")
        .replace(/[_-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/^\w/, (first) => first.toUpperCase());
  }
};

const resourceShareGrantIdentity = (grant: ResourceShareGrantRecord): string => {
  const userId = String(grant.grantee_user_id ?? "").trim();
  const orgId = String(grant.grantee_org_id ?? "").trim();
  return userId || orgId || grant.grant_id;
};

const resourceShareGrantRoleLabel = (value: string): string => {
  const role = String(value || "").trim() || "read";
  return role
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\w/, (first) => first.toUpperCase());
};

const iconForKind = (kind: string) => {
  switch (String(kind || "").toLowerCase()) {
    case "image":
      return ImageIcon;
    case "video":
      return Film;
    case "table":
      return Table2;
    case "document":
      return FileText;
    default:
      return File;
  }
};

const isVideoResource = (resource: ResourceRecord): boolean =>
  String(resource.resource_kind || "").toLowerCase() === "video" ||
  String(resource.content_type || "").toLowerCase().startsWith("video/");

const shouldRequestThumbnail = (resource: ResourceRecord, failedThumbnailIds: Record<string, true>): boolean => {
  if (failedThumbnailIds[resource.file_id]) {
    return false;
  }
  return Boolean(
    resource.has_thumbnail ||
      resource.thumbnail_url ||
      resource.preview_url ||
      String(resource.resource_kind || "").toLowerCase() === "image"
  );
};

// Share-target picker: grantees are chosen from real principals (same-org
// people + the org itself), never typed as raw ids — a typed id that matches
// nobody is sharing's worst silent failure. Falls back to nothing when the
// search capability is absent; callers keep their legacy inputs in that case.
function ShareTargetPicker({
  idPrefix,
  onSearch,
  selectedLabel,
  hasSelection,
  disabled,
  onPick,
  onClear,
}: {
  idPrefix: string;
  onSearch: (query: string) => Promise<ShareTargetRecord[]>;
  selectedLabel: string;
  hasSelection: boolean;
  disabled?: boolean;
  onPick: (target: ShareTargetRecord) => void;
  onClear: () => void;
}) {
  const [query, setQuery] = useState("");
  const [targets, setTargets] = useState<ShareTargetRecord[]>([]);
  const [searching, setSearching] = useState(false);
  useEffect(() => {
    if (hasSelection) {
      return;
    }
    let cancelled = false;
    setSearching(true);
    const handle = window.setTimeout(() => {
      onSearch(query)
        .then((results) => {
          if (!cancelled) {
            setTargets(results);
            setSearching(false);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setTargets([]);
            setSearching(false);
          }
        });
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [query, onSearch, hasSelection]);

  if (hasSelection) {
    return (
      <div className="resource-browser-share-picked" data-testid={`${idPrefix}-picked`}>
        <span>{selectedLabel || "Selected"}</span>
        <Button type="button" variant="ghost" size="sm" onClick={onClear} disabled={disabled}>
          Change
        </Button>
      </div>
    );
  }
  return (
    <div className="resource-browser-share-picker">
      <label htmlFor={`${idPrefix}-target`}>Share with</label>
      <Input
        id={`${idPrefix}-target`}
        value={query}
        placeholder="Search people in your organization"
        onChange={(event) => setQuery(event.target.value)}
        autoComplete="off"
        disabled={disabled}
      />
      <div className="resource-browser-share-picker-results" role="listbox" aria-label="Share targets">
        {searching && targets.length === 0 ? (
          <p className="resource-browser-share-picker-empty">Searching…</p>
        ) : null}
        {!searching && targets.length === 0 ? (
          <p className="resource-browser-share-picker-empty">No matches in your organization</p>
        ) : null}
        {targets.map((target) => (
          <button
            key={`${target.kind}:${target.grantee_user_id ?? ""}:${target.grantee_org_id ?? ""}`}
            type="button"
            role="option"
            aria-selected="false"
            data-share-target-kind={target.kind}
            onClick={() => onPick(target)}
            disabled={disabled}
          >
            <span className="resource-browser-share-picker-label">{target.label}</span>
            {target.detail ? (
              <span className="resource-browser-share-picker-detail">{target.detail}</span>
            ) : null}
          </button>
        ))}
      </div>
    </div>
  );
}

// "People with access" for a folder: lists collection-level grants and lets
// the owner revoke one — a single revoke cascades to every inherited member
// grant server-side.
function CollectionAccessList({
  collection,
  load,
  revoke,
}: {
  collection: ResourceCollectionRecord;
  load: (collection: ResourceCollectionRecord) => Promise<ResourceCollectionShareGrantRecord[]>;
  revoke?: (
    collection: ResourceCollectionRecord,
    grant: ResourceCollectionShareGrantRecord
  ) => Promise<ResourceCollectionShareGrantRecord>;
}) {
  const [grants, setGrants] = useState<ResourceCollectionShareGrantRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load(collection)
      .then((records) => {
        if (!cancelled) {
          setGrants(records.filter((grant) => grant.status === "active"));
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setGrants([]);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [collection, load, revision]);
  const describeGrantee = (grant: ResourceCollectionShareGrantRecord): string => {
    const user = String(grant.grantee_user_id ?? "").trim();
    if (user === "__public__") {
      return "Anyone on the platform";
    }
    if (user) {
      return user;
    }
    const org = String(grant.grantee_org_id ?? "").trim();
    return org ? `Everyone in ${org}` : "Unknown";
  };
  if (loading) {
    return <p className="resource-browser-share-access-empty">Loading access…</p>;
  }
  return (
    <div className="resource-browser-share-access" aria-label="People with access">
      <p className="resource-browser-share-access-title">People with access</p>
      {grants.length === 0 ? (
        <p className="resource-browser-share-access-empty">Only you</p>
      ) : (
        grants.map((grant) => (
          <div key={grant.grant_id} className="resource-browser-share-access-row">
            <span>{describeGrantee(grant)}</span>
            {revoke ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  void revoke(collection, grant).then(() => setRevision((value) => value + 1));
                }}
              >
                Revoke
              </Button>
            ) : null}
          </div>
        ))
      )}
    </div>
  );
}

export function ResourceBrowser({
  resources,
  resourceCollections = [],
  resourceCollectionsLoading = false,
  totalCount,
  loading,
  loadingMore,
  hasMore,
  error,
  query,
  kindFilter,
  sourceFilter,
  sharingFilter = "all",
  statusFilter = "active",
  tagFilter = "",
  deletingFileIds,
  restoringFileIds = {},
  onQueryChange,
  focusSearchSignal = 0,
  onKindFilterChange,
  onSourceFilterChange,
  onSharingFilterChange,
  onStatusFilterChange,
  onTagFilterChange,
  onRefresh,
  onLoadMore,
  onUploadFiles,
  uploading = false,
  uploadProgress = [],
  onDismissUploadProgress,
  onPauseUploadProgress,
  onCancelUploadProgress,
  onOpenResource,
  onUseInChat,
  onDeleteResource,
  onRenameResource,
  onRestoreResource,
  onRemoveResourceFromCollection,
  onLoadResourceShareGrants,
  onCreateResourceShareGrant,
  onCreateBulkResourceShareGrants,
  onCreateResourceCollectionShareGrants,
  onDeleteSelectedResources,
  onRestoreSelectedResources,
  onRevokeResourceShareGrant,
  onSearchShareTargets,
  onLoadCollectionShareGrants,
  onRevokeCollectionShareGrant,
  onCreateCollectionFromSelection,
  onAddSelectionToCollection,
  activeResourceCollection = null,
  onOpenCollection,
  onRenameCollection,
  onDeleteCollection,
  onRestoreCollection,
  restoringCollectionIds = {},
  onClearActiveCollection,
  thumbnailUrlFor,
  zScrubThumbnail,
  downloadUrlFor,
  collectionDownloadUrlFor,
  quickPeekFetch,
  onPushResourceToBisque,
  onPushCollectionToBisque,
}: ResourceBrowserProps) {
  const isMobileView = useBreakpoint(721);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  // Seeded with the mount-time value, so a signal that was already spent before
  // this mount is not replayed. The counter lives in App and never resets, and
  // this panel unmounts on every panel switch — without the latch, one tap of the
  // nav-bar search action would silently steal focus into the field on every
  // later return to Resources.
  const handledSearchSignalRef = useRef(focusSearchSignal);
  // The mobile header scrolls away with the list, so by the time you want to
  // search the field is usually off-screen. The nav bar's search action bumps
  // this signal: bring the field back into view and focus it.
  useEffect(() => {
    if (focusSearchSignal === handledSearchSignalRef.current) return;
    handledSearchSignalRef.current = focusSearchSignal;
    const input = searchInputRef.current;
    if (!input) return;
    input.scrollIntoView({ block: "nearest" });
    input.focus();
  }, [focusSearchSignal]);
  const [resourceFiltersOpen, setResourceFiltersOpen] = useState(false);
  const [expiredUploadDetailsOpen, setExpiredUploadDetailsOpen] = useState(false);
  const [resourceViewMode, setResourceViewMode] = useState<ResourceViewMode>(
    readStoredResourceViewMode
  );
  const [failedThumbnailIds, setFailedThumbnailIds] = useState<Record<string, true>>({});
  const [selectedResourceIds, setSelectedResourceIds] = useState<Record<string, true>>({});
  const [selectionAnchorResourceId, setSelectionAnchorResourceId] = useState("");
  const [newFolderDialogOpen, setNewFolderDialogOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [newFolderSaving, setNewFolderSaving] = useState(false);
  const [newFolderError, setNewFolderError] = useState<string | null>(null);
  const [organizingSelection, setOrganizingSelection] = useState(false);
  const [shareSheetResourceId, setShareSheetResourceId] = useState("");
  const [shareGrantsByResourceId, setShareGrantsByResourceId] = useState<
    Record<string, ResourceShareGrantRecord[]>
  >({});
  const [shareGrantsLoading, setShareGrantsLoading] = useState(false);
  const [shareGrantsError, setShareGrantsError] = useState<string | null>(null);
  const [shareGranteeUserId, setShareGranteeUserId] = useState("");
  const [shareGranteeOrgId, setShareGranteeOrgId] = useState("");
  const [shareGranteeLabel, setShareGranteeLabel] = useState("");
  const [bulkShareGranteeLabel, setBulkShareGranteeLabel] = useState("");
  const [collectionShareGranteeLabel, setCollectionShareGranteeLabel] = useState("");
  const [shareSaving, setShareSaving] = useState(false);
  const [shareRevokingById, setShareRevokingById] = useState<Record<string, true>>({});
  const [bulkShareOpen, setBulkShareOpen] = useState(false);
  const [bulkShareGranteeUserId, setBulkShareGranteeUserId] = useState("");
  const [bulkShareGranteeOrgId, setBulkShareGranteeOrgId] = useState("");
  const [bulkShareSaving, setBulkShareSaving] = useState(false);
  const [bulkShareError, setBulkShareError] = useState<string | null>(null);
  const [collectionShareTarget, setCollectionShareTarget] =
    useState<ResourceCollectionRecord | null>(null);
  const [collectionShareGranteeUserId, setCollectionShareGranteeUserId] = useState("");
  const [collectionShareGranteeOrgId, setCollectionShareGranteeOrgId] = useState("");
  const [collectionShareSaving, setCollectionShareSaving] = useState(false);
  const [collectionShareError, setCollectionShareError] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<
    | { kind: "resource"; resource: ResourceRecord }
    | { kind: "collection"; collection: ResourceCollectionRecord }
    | null
  >(null);
  const [renameInput, setRenameInput] = useState("");
  const [renameSaving, setRenameSaving] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [removingFromCollectionIds, setRemovingFromCollectionIds] = useState<Record<string, true>>(
    {}
  );
  const [deletingCollectionIds, setDeletingCollectionIds] = useState<Record<string, true>>({});
  const [deletingSelection, setDeletingSelection] = useState(false);
  const [restoringSelection, setRestoringSelection] = useState(false);
  const [draggedResourceId, setDraggedResourceId] = useState("");
  const [fileDropTargetId, setFileDropTargetId] = useState("");
  const [tableScrollTop, setTableScrollTop] = useState(0);
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  const tableShellRef = useRef<HTMLDivElement | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const folderUploadInputRef = useRef<HTMLInputElement | null>(null);
  const pendingUploadReselectionRef = useRef<ResourceUploadProgress | null>(null);
  const [uploadReselectionError, setUploadReselectionError] = useState<string | null>(null);
  const cardResources = useMemo(() => resources, [resources]);
  const effectiveResourceViewMode: ResourceViewMode =
    isMobileView && resourceViewMode === "table" ? "cards" : resourceViewMode;
  const tableVirtualized =
    effectiveResourceViewMode === "table" &&
    cardResources.length > RESOURCE_TABLE_VIRTUALIZATION_THRESHOLD;
  const tableVisibleRowCount = tableVirtualized
    ? Math.min(
        cardResources.length,
        RESOURCE_TABLE_FALLBACK_VIEWPORT_ROWS + RESOURCE_TABLE_OVERSCAN_ROWS * 2
      )
    : cardResources.length;
  const tableStartIndex = tableVirtualized
    ? Math.min(
        Math.max(0, cardResources.length - tableVisibleRowCount),
        Math.max(
          0,
          Math.floor(tableScrollTop / RESOURCE_TABLE_ROW_HEIGHT) -
            RESOURCE_TABLE_OVERSCAN_ROWS
        )
      )
    : 0;
  const tableEndIndex = tableVirtualized
    ? Math.min(cardResources.length, tableStartIndex + tableVisibleRowCount)
    : cardResources.length;
  const tableResources = tableVirtualized
    ? cardResources.slice(tableStartIndex, tableEndIndex)
    : cardResources;
  const tableTopSpacerHeight = tableVirtualized
    ? tableStartIndex * RESOURCE_TABLE_ROW_HEIGHT
    : 0;
  const tableBottomSpacerHeight = tableVirtualized
    ? Math.max(0, (cardResources.length - tableEndIndex) * RESOURCE_TABLE_ROW_HEIGHT)
    : 0;
  // Duplicate awareness over the loaded page: identical content = same
  // sha256 + size. Deliberately pagination-blind (a server-side count is the
  // follow-up); empty checksums (bisque imports, bundles) never group.
  const duplicateShaCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const resource of cardResources) {
      const sha = String(resource.sha256 ?? "").trim().toLowerCase();
      if (!sha || resource.status === "deleted") {
        continue;
      }
      const key = `${sha}:${resource.size_bytes ?? 0}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }, [cardResources]);
  const duplicateCountFor = (resource: ResourceRecord): number => {
    const sha = String(resource.sha256 ?? "").trim().toLowerCase();
    if (!sha) {
      return 0;
    }
    return duplicateShaCounts.get(`${sha}:${resource.size_bytes ?? 0}`) ?? 0;
  };

  const selectedResourcesInOrder = useMemo(
    () =>
      cardResources.filter((resource) => selectedResourceIds[resource.file_id]),
    [cardResources, selectedResourceIds]
  );
  const selectedResourceIdsInOrder = useMemo(
    () => selectedResourcesInOrder.map((resource) => resource.file_id),
    [selectedResourcesInOrder]
  );
  const availableFolderCollections = useMemo(
    () =>
      resourceCollections.filter(
        (collection) =>
          String(collection.status || "").toLowerCase() !== "deleted" &&
          String(collection.collection_type || "").toLowerCase() === "folder"
      ),
    [resourceCollections]
  );
  const deletedFolderCollections = useMemo(
    () =>
      resourceCollections.filter(
        (collection) =>
          String(collection.status || "").toLowerCase() === "deleted" &&
          String(collection.collection_type || "").toLowerCase() === "folder"
      ),
    [resourceCollections]
  );
  // Nested browsing: the strip shows roots at the top level and children of
  // the open folder inside it. A child whose parent is not in the loaded list
  // (200-cap / filter scoping) surfaces at root rather than vanishing.
  const loadedCollectionIds = useMemo(
    () => new Set(resourceCollections.map((collection) => collection.collection_id)),
    [resourceCollections]
  );
  const visibleFolderCollections = useMemo(() => {
    const activeId = activeResourceCollection?.collection_id ?? "";
    return availableFolderCollections.filter((collection) => {
      const parentId = String(collection.parent_collection_id ?? "").trim();
      if (activeId) {
        return parentId === activeId;
      }
      return parentId === "" || !loadedCollectionIds.has(parentId);
    });
  }, [availableFolderCollections, activeResourceCollection, loadedCollectionIds]);
  const safeTotalCount = Math.max(0, Math.floor(Number(totalCount) || 0));
  const visibleCount = cardResources.length;
  const activeCollectionName = String(activeResourceCollection?.name ?? "").trim();
  const activeFolderName = activeCollectionName || "Untitled folder";
  const activeParentCollectionId = String(activeResourceCollection?.parent_collection_id ?? "").trim();
  const activeParentCollection = useMemo(
    () =>
      activeParentCollectionId
        ? (resourceCollections.find(
            (collection) => collection.collection_id === activeParentCollectionId
          ) ?? null)
        : null,
    [activeParentCollectionId, resourceCollections]
  );
  const activeParentCollectionName = String(activeParentCollection?.name ?? "").trim();
  const activeFolderNavigationLabel =
    activeParentCollection && onOpenCollection
      ? `Go up from folder ${activeFolderName}`
      : `Leave folder ${activeFolderName}`;
  const canNavigateFromActiveFolder = Boolean(
    (activeParentCollection && onOpenCollection) || onClearActiveCollection
  );
  const activeTagFilter = tagFilter.trim();
  const visibleFilterCount =
    Number(kindFilter !== "all") +
    Number(sourceFilter !== "all") +
    Number(sharingFilter !== "all") +
    Number(statusFilter !== "active") +
    Number(activeTagFilter.length > 0);
  const resultSummary = loading
    ? activeCollectionName
      ? `Loading ${activeCollectionName}...`
      : "Searching resources..."
    : activeCollectionName
      ? `${visibleCount.toLocaleString()} of ${safeTotalCount.toLocaleString()} resources in ${activeCollectionName}`
      : `${visibleCount.toLocaleString()} of ${safeTotalCount.toLocaleString()} resources`;
  const visibleUploadProgress = uploadProgress.filter(
    (item) => item.status !== "completed" && item.status !== "canceled"
  );
  const expiredUploadProgress = visibleUploadProgress.filter(isExpiredUploadProgress);
  const shouldCollapseExpiredUploads = expiredUploadProgress.length > 1;
  const renderedUploadProgress =
    shouldCollapseExpiredUploads && !expiredUploadDetailsOpen
      ? visibleUploadProgress.filter((item) => !isExpiredUploadProgress(item))
      : visibleUploadProgress;
  const shareSheetResource =
    cardResources.find((resource) => resource.file_id === shareSheetResourceId) ?? null;
  const shareSheetResourceName = shareSheetResource ? resourceDisplayName(shareSheetResource) : "";
  const renameTargetName =
    renameTarget?.kind === "resource"
      ? resourceDisplayName(renameTarget.resource)
      : renameTarget?.kind === "collection"
        ? renameTarget.collection.name
        : "";
  const trimmedRenameInput = renameInput.trim();
  const shareSheetGrants = shareSheetResource
    ? (shareGrantsByResourceId[shareSheetResource.file_id] ?? [])
    : [];
  const isDeletedResourceView = statusFilter === "deleted";
  const canUseSharing = Boolean(
    onLoadResourceShareGrants && onCreateResourceShareGrant && onRevokeResourceShareGrant
  );
  const canUseBulkShare =
    !isDeletedResourceView && Boolean(onCreateBulkResourceShareGrants || onCreateResourceShareGrant);
  const canUseBulkFolderMove = !isDeletedResourceView && Boolean(onAddSelectionToCollection);
  const selectionEnabled = Boolean(
    isDeletedResourceView
      ? onRestoreSelectedResources
      : onAddSelectionToCollection ||
          onDeleteSelectedResources ||
          onCreateBulkResourceShareGrants ||
          onCreateResourceShareGrant
  );
  const selectedResourceCount = selectedResourceIdsInOrder.length;
  const trimmedNewFolderName = newFolderName.trim();
  const trimmedShareGranteeUserId = shareGranteeUserId.trim();
  const trimmedShareGranteeOrgId = shareGranteeOrgId.trim();
  const trimmedBulkShareGranteeUserId = bulkShareGranteeUserId.trim();
  const trimmedBulkShareGranteeOrgId = bulkShareGranteeOrgId.trim();
  const trimmedCollectionShareGranteeUserId = collectionShareGranteeUserId.trim();
  const trimmedCollectionShareGranteeOrgId = collectionShareGranteeOrgId.trim();
  const canGrantShare =
    Boolean(onCreateResourceShareGrant) &&
    Boolean(shareSheetResource) &&
    (trimmedShareGranteeUserId.length > 0 || trimmedShareGranteeOrgId.length > 0) &&
    !shareSaving;
  const canGrantBulkShare =
    canUseBulkShare &&
    selectedResourceCount > 0 &&
    (trimmedBulkShareGranteeUserId.length > 0 || trimmedBulkShareGranteeOrgId.length > 0) &&
    !bulkShareSaving;
  const canGrantCollectionShare =
    Boolean(onCreateResourceCollectionShareGrants) &&
    Boolean(collectionShareTarget) &&
    (trimmedCollectionShareGranteeUserId.length > 0 ||
      trimmedCollectionShareGranteeOrgId.length > 0) &&
    !collectionShareSaving;
  const canSaveRename =
    Boolean(renameTarget) &&
    trimmedRenameInput.length > 0 &&
    trimmedRenameInput !== renameTargetName &&
    !renameSaving;
  const canDeleteSelectedResources =
    Boolean(onDeleteSelectedResources) &&
    !isDeletedResourceView &&
    selectedResourceCount > 0 &&
    !deletingSelection &&
    !organizingSelection;
  const canRestoreSelectedResources =
    Boolean(onRestoreSelectedResources) &&
    isDeletedResourceView &&
    selectedResourceCount > 0 &&
    !restoringSelection &&
    !organizingSelection;
  const canCreateEmptyFolder =
    Boolean(onCreateCollectionFromSelection) &&
    trimmedNewFolderName.length > 0 &&
    !newFolderSaving;
  const canShowBulkMoveToFolder =
    canUseBulkFolderMove && (resourceCollectionsLoading || availableFolderCollections.length > 0);
  const canMoveSelectionToFolder =
    canUseBulkFolderMove &&
    selectedResourceCount > 0 &&
    availableFolderCollections.length > 0 &&
    !resourceCollectionsLoading &&
    !organizingSelection;

  const resetTableScroll = (): void => {
    setTableScrollTop(0);
    if (tableShellRef.current) {
      tableShellRef.current.scrollTop = 0;
    }
  };

  const requestUploadReselection = (item: ResourceUploadProgress): void => {
    pendingUploadReselectionRef.current = item;
    setUploadReselectionError(null);
    if (item.relativePath) {
      folderUploadInputRef.current?.click();
      return;
    }
    uploadInputRef.current?.click();
  };

  const handleUploadInputSelection = (files: File[]): void => {
    const resumeFrom = pendingUploadReselectionRef.current;
    pendingUploadReselectionRef.current = null;
    if (files.length === 0) {
      return;
    }
    if (!resumeFrom) {
      // Same funnel as drops: junk filtering, nested-zarr re-rooting, cap
      // with truncated-bundle withholding. Resume-reselects skip it — they
      // must match the interrupted upload's recorded paths byte for byte.
      const normalized = normalizePickedFiles(files);
      const message = summarizeDropIssues(normalized);
      setUploadReselectionError(message);
      if (normalized.files.length > 0) {
        onUploadFiles?.(normalized.files);
      }
      return;
    }
    const matchingFile = files.find((file) => fileMatchesUploadProgress(file, resumeFrom)) ?? null;
    if (!matchingFile) {
      setUploadReselectionError(
        `Select ${resumeFrom.fileName} to resume this upload. The selected file did not match the interrupted upload.`
      );
      return;
    }
    setUploadReselectionError(null);
    onUploadFiles?.(resumeFrom.relativePath ? files : [matchingFile], { resumeFrom });
  };

  const isExternalFileDrag = (event: DragEvent<HTMLElement>): boolean => {
    if (draggedResourceId) {
      return false;
    }
    // isOsFileDrag also rejects in-page image/link drags, which carry "Files"
    // alongside text/html or text/uri-list.
    return isOsFileDrag(event.dataTransfer);
  };

  const uploadDroppedFiles = (
    event: DragEvent<HTMLElement>,
    collection?: ResourceCollectionRecord | null
  ): boolean => {
    if (!onUploadFiles || !isExternalFileDrag(event)) {
      return false;
    }
    event.preventDefault();
    event.stopPropagation();
    // Entries are only readable during the drop dispatch — snapshot before the
    // async directory walk. Dropped folders traverse recursively and their
    // files carry synthesized relative paths, so zarr stores bundle and plain
    // folders get filed as folders upstream.
    const payload = snapshotDropPayload(event.dataTransfer);
    setFileDropTargetId("");
    setUploadReselectionError(null);
    void collectDroppedFiles(payload).then((dropped) => {
      const message = summarizeDropIssues(dropped);
      if (message) {
        setUploadReselectionError(message);
      }
      if (dropped.files.length === 0) {
        return;
      }
      onUploadFiles(
        dropped.files,
        collection ? { uploadTargetCollection: collection } : undefined
      );
    });
    return true;
  };

  const handleExternalFileDragOver = (
    event: DragEvent<HTMLElement>,
    targetId: string
  ): boolean => {
    if (!onUploadFiles || !isExternalFileDrag(event)) {
      return false;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setFileDropTargetId(targetId);
    return true;
  };

  useEffect(() => {
    persistResourceViewMode(resourceViewMode);
  }, [resourceViewMode]);

  useEffect(() => {
    if (!hasMore || loading || loadingMore) {
      return;
    }
    const node = loadMoreRef.current;
    if (!node) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          onLoadMore();
        }
      },
      { root: null, rootMargin: "560px 0px 560px 0px", threshold: 0.01 }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, onLoadMore]);

  const handleTableScroll = (event: UIEvent<HTMLDivElement>): void => {
    const scrollNode = event.currentTarget;
    const nextScrollTop = Math.max(0, scrollNode.scrollTop);
    if (tableVirtualized) {
      setTableScrollTop((current) => (current === nextScrollTop ? current : nextScrollTop));
    }
    if (!hasMore || loading || loadingMore) {
      return;
    }
    const remainingScroll =
      scrollNode.scrollHeight - nextScrollTop - Math.max(1, scrollNode.clientHeight);
    if (remainingScroll <= RESOURCE_TABLE_ROW_HEIGHT * 6) {
      onLoadMore();
    }
  };

  const toggleResourceSelection = (fileId: string, checked: boolean): void => {
    if (!selectionEnabled) {
      return;
    }
    setSelectionAnchorResourceId((current) => {
      if (checked) {
        return fileId;
      }
      return current === fileId ? "" : current;
    });
    setSelectedResourceIds((current) => {
      if (checked) {
        return { ...current, [fileId]: true };
      }
      if (!current[fileId]) {
        return current;
      }
      const next = { ...current };
      delete next[fileId];
      return next;
    });
  };

  const selectResourceRange = (
    current: Record<string, true>,
    anchorResourceId: string,
    targetResourceId: string
  ): Record<string, true> => {
    const anchorIndex = cardResources.findIndex(
      (candidate) => candidate.file_id === anchorResourceId
    );
    const targetIndex = cardResources.findIndex(
      (candidate) => candidate.file_id === targetResourceId
    );
    if (anchorIndex < 0 || targetIndex < 0) {
      return { ...current, [targetResourceId]: true };
    }
    const startIndex = Math.min(anchorIndex, targetIndex);
    const endIndex = Math.max(anchorIndex, targetIndex);
    const next = { ...current };
    for (const candidate of cardResources.slice(startIndex, endIndex + 1)) {
      next[candidate.file_id] = true;
    }
    return next;
  };

  const clickStartedOnResourceControl = (
    target: EventTarget | null,
    currentTarget: HTMLElement
  ): boolean => {
    if (!(target instanceof Element)) {
      return false;
    }
    const control = target.closest(
      "button,a,input,label,select,textarea,[role='button'],[role='menuitem']"
    );
    return Boolean(control && currentTarget.contains(control));
  };

  const keyboardStartedInResourceControl = (
    target: EventTarget | null,
    currentTarget: HTMLElement
  ): boolean => {
    if (!(target instanceof Element)) {
      return false;
    }
    if (target instanceof HTMLElement && target.isContentEditable) {
      return true;
    }
    const control = target.closest(
      "button,a,input,label,select,textarea,[role='button'],[role='menuitem'],[contenteditable='true']"
    );
    return Boolean(control && currentTarget.contains(control));
  };

  const selectResourceFromSurfaceClick = (
    event: MouseEvent<HTMLElement>,
    resource: ResourceRecord
  ): void => {
    if (
      event.defaultPrevented ||
      clickStartedOnResourceControl(event.target, event.currentTarget)
    ) {
      return;
    }
    const fileId = resource.file_id;
    const isDeleted = isDeletedResource(resource);
    const isSelectionGesture = event.shiftKey || event.metaKey || event.ctrlKey;
    if (!selectionEnabled || (!isSelectionGesture && !isDeleted)) {
      if (!isDeleted) {
        onOpenResource(resource);
      }
      return;
    }
    if (event.shiftKey) {
      const anchorResourceId =
        selectionAnchorResourceId &&
        cardResources.some((candidate) => candidate.file_id === selectionAnchorResourceId)
          ? selectionAnchorResourceId
          : selectedResourceIdsInOrder[0] ?? fileId;
      setSelectionAnchorResourceId(anchorResourceId);
      setSelectedResourceIds((current) =>
        selectResourceRange(current, anchorResourceId, fileId)
      );
      return;
    }
    setSelectionAnchorResourceId(fileId);
    setSelectedResourceIds((current) => {
      if (event.metaKey || event.ctrlKey) {
        if (!current[fileId]) {
          return { ...current, [fileId]: true };
        }
        const next = { ...current };
        delete next[fileId];
        return next;
      }
      if (current[fileId] && Object.keys(current).length === 1) {
        return current;
      }
      return { [fileId]: true };
    });
  };

  const clearSelection = (): void => {
    setSelectedResourceIds({});
    setSelectionAnchorResourceId("");
    setBulkShareOpen(false);
    setBulkShareGranteeUserId("");
    setBulkShareGranteeOrgId("");
    setBulkShareGranteeLabel("");
    setBulkShareError(null);
  };

  const navigateToAllResources = (): void => {
    clearSelection();
    onClearActiveCollection?.();
  };

  const navigateFromActiveFolder = (): void => {
    clearSelection();
    if (activeParentCollection && onOpenCollection) {
      onOpenCollection(activeParentCollection);
      return;
    }
    onClearActiveCollection?.();
  };

  const deleteSelectedResources = async (): Promise<void> => {
    if (!onDeleteSelectedResources || selectedResourcesInOrder.length === 0 || deletingSelection) {
      return;
    }
    const resourcesToDelete = selectedResourcesInOrder;
    setDeletingSelection(true);
    try {
      await onDeleteSelectedResources(resourcesToDelete);
      clearSelection();
    } finally {
      setDeletingSelection(false);
    }
  };

  const restoreSelectedResources = async (): Promise<void> => {
    if (
      !onRestoreSelectedResources ||
      selectedResourcesInOrder.length === 0 ||
      restoringSelection
    ) {
      return;
    }
    const resourcesToRestore = selectedResourcesInOrder;
    setRestoringSelection(true);
    try {
      await onRestoreSelectedResources(resourcesToRestore);
      clearSelection();
    } finally {
      setRestoringSelection(false);
    }
  };

  const selectAllVisibleResources = (): void => {
    if (!selectionEnabled || cardResources.length === 0) {
      return;
    }
    setSelectionAnchorResourceId(cardResources[0]?.file_id ?? "");
    setSelectedResourceIds(
      Object.fromEntries(cardResources.map((resource) => [resource.file_id, true]))
    );
  };

  const handleResourceBrowserKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (keyboardStartedInResourceControl(event.target, event.currentTarget)) {
      return;
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
      if (!selectionEnabled || cardResources.length === 0) {
        return;
      }
      event.preventDefault();
      selectAllVisibleResources();
      return;
    }
    if (event.key === "Escape" && selectedResourceCount > 0) {
      event.preventDefault();
      clearSelection();
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && selectedResourceCount > 0) {
      if (canDeleteSelectedResources) {
        event.preventDefault();
        void deleteSelectedResources();
      }
    }
  };

  const handleResourceItemKeyDown = (
    event: KeyboardEvent<HTMLElement>,
    resource: ResourceRecord,
    isDeleted: boolean
  ): void => {
    if (event.target !== event.currentTarget) {
      return;
    }
    if (event.key === "F2" && onRenameResource && !isDeleted) {
      event.preventDefault();
      event.stopPropagation();
      openResourceRenameDialog(resource);
      return;
    }
    if (event.key !== "Enter") {
      return;
    }
    if (isDeleted) {
      onRestoreResource?.(resource);
    } else {
      onOpenResource(resource);
    }
  };

  const handleCollectionKeyDown = (
    event: KeyboardEvent<HTMLElement>,
    collection: ResourceCollectionRecord
  ): void => {
    if (event.target !== event.currentTarget) {
      return;
    }
    if (event.key === "F2" && onRenameCollection) {
      event.preventDefault();
      event.stopPropagation();
      openCollectionRenameDialog(collection);
      return;
    }
    if (
      (event.key === "Delete" || event.key === "Backspace") &&
      onDeleteCollection &&
      collection.status !== "deleted"
    ) {
      event.preventDefault();
      event.stopPropagation();
      void deleteCollection(collection);
    }
  };

  const openNewFolderDialog = (): void => {
    if (!onCreateCollectionFromSelection) {
      return;
    }
    setNewFolderName("");
    setNewFolderError(null);
    setNewFolderDialogOpen(true);
  };

  const closeNewFolderDialog = (): void => {
    if (newFolderSaving) {
      return;
    }
    setNewFolderDialogOpen(false);
    setNewFolderName("");
    setNewFolderError(null);
  };

  const createEmptyFolder = async (): Promise<void> => {
    if (!onCreateCollectionFromSelection || !canCreateEmptyFolder) {
      return;
    }
    setNewFolderSaving(true);
    setNewFolderError(null);
    try {
      await onCreateCollectionFromSelection({
        collectionType: "folder",
        name: trimmedNewFolderName,
        resourceIds: [],
      });
      setNewFolderDialogOpen(false);
      setNewFolderName("");
    } catch (folderError) {
      setNewFolderError(
        folderError instanceof Error ? folderError.message : "Unable to create folder."
      );
    } finally {
      setNewFolderSaving(false);
    }
  };

  const addSelectionToExistingFolder = async (collectionId: string): Promise<void> => {
    const targetCollectionId = collectionId.trim();
    const resourceIds = selectedResourceIdsInOrder;
    if (
      !onAddSelectionToCollection ||
      !canMoveSelectionToFolder ||
      !targetCollectionId ||
      resourceIds.length === 0
    ) {
      return;
    }
    setOrganizingSelection(true);
    try {
      await onAddSelectionToCollection({
        collectionId: targetCollectionId,
        resourceIds,
      });
      clearSelection();
    } catch {
      // Parent surfaces the API error in the Resources error region.
    } finally {
      setOrganizingSelection(false);
    }
  };

  const handleResourceDragStart = (
    event: DragEvent<HTMLElement>,
    resource: ResourceRecord
  ): void => {
    const canDragResource =
      Boolean(onAddSelectionToCollection) ||
      Boolean(activeResourceCollection && onRemoveResourceFromCollection);
    if (!canDragResource || isDeletedResource(resource)) {
      event.preventDefault();
      return;
    }
    const selectedDragResourceIds =
      selectedResourceIds[resource.file_id] && selectedResourceIdsInOrder.length > 0
        ? selectedResourceIdsInOrder
        : [resource.file_id];
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData(RESOURCE_DRAG_ID_MIME, resource.file_id);
    event.dataTransfer.setData(RESOURCE_DRAG_IDS_MIME, JSON.stringify(selectedDragResourceIds));
    event.dataTransfer.setData(
      "text/plain",
      selectedDragResourceIds.length === 1
        ? resourceDisplayName(resource)
        : pluralSummary(selectedDragResourceIds.length, "resource")
    );
    setDraggedResourceId(resource.file_id);
  };

  const draggedResourceIdsFromEvent = (event: DragEvent<HTMLElement>): string[] => {
    const resourceIds = parseDraggedResourceIds(event.dataTransfer.getData(RESOURCE_DRAG_IDS_MIME));
    if (resourceIds.length > 0) {
      return resourceIds;
    }
    const resourceId = event.dataTransfer.getData(RESOURCE_DRAG_ID_MIME) || draggedResourceId;
    return resourceId ? [resourceId] : [];
  };

  const handleAllResourcesDrop = async (event: DragEvent<HTMLElement>): Promise<void> => {
    if (!activeResourceCollection || !onRemoveResourceFromCollection) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const resourceIds = draggedResourceIdsFromEvent(event);
    const resources = resourceIds
      .map((resourceId) => cardResources.find((item) => item.file_id === resourceId))
      .filter((resource): resource is ResourceRecord => Boolean(resource));
    if (resources.length === 0 || organizingSelection) {
      setDraggedResourceId("");
      return;
    }
    for (const resource of resources) {
      await removeResourceFromCurrentFolder(resource);
    }
    setDraggedResourceId("");
  };

  const handleFolderDrop = async (
    event: DragEvent<HTMLElement>,
    collection: ResourceCollectionRecord
  ): Promise<void> => {
    if (!onAddSelectionToCollection) {
      return;
    }
    event.preventDefault();
    const resourceIds = draggedResourceIdsFromEvent(event);
    if (resourceIds.length === 0 || organizingSelection) {
      setDraggedResourceId("");
      return;
    }
    setOrganizingSelection(true);
    try {
      await onAddSelectionToCollection({
        collectionId: collection.collection_id,
        resourceIds,
      });
      clearSelection();
    } catch {
      // Parent surfaces the API error in the Resources error region.
    } finally {
      setDraggedResourceId("");
      setOrganizingSelection(false);
    }
  };

  const applyTagFilter = (tag: string): void => {
    if (!onTagFilterChange) {
      return;
    }
    clearSelection();
    resetTableScroll();
    onTagFilterChange(tag);
  };

  const openResourceRenameDialog = (resource: ResourceRecord): void => {
    setRenameTarget({ kind: "resource", resource });
    setRenameInput(resourceDisplayName(resource));
    setRenameError(null);
  };

  const openCollectionRenameDialog = (collection: ResourceCollectionRecord): void => {
    setRenameTarget({ kind: "collection", collection });
    setRenameInput(collection.name);
    setRenameError(null);
  };

  const closeRenameDialog = (): void => {
    if (renameSaving) {
      return;
    }
    setRenameTarget(null);
    setRenameInput("");
    setRenameError(null);
  };

  const saveRename = async (): Promise<void> => {
    if (!renameTarget || !canSaveRename) {
      return;
    }
    setRenameSaving(true);
    setRenameError(null);
    try {
      if (renameTarget.kind === "resource") {
        await onRenameResource?.(renameTarget.resource, trimmedRenameInput);
      } else {
        await onRenameCollection?.(renameTarget.collection, trimmedRenameInput);
      }
      setRenameTarget(null);
      setRenameInput("");
    } catch (renameSaveError) {
      setRenameError(
        renameSaveError instanceof Error ? renameSaveError.message : "Unable to rename item."
      );
    } finally {
      setRenameSaving(false);
    }
  };

  const removeResourceFromCurrentFolder = async (resource: ResourceRecord): Promise<void> => {
    if (!activeResourceCollection || !onRemoveResourceFromCollection) {
      return;
    }
    const fileId = resource.file_id;
    setRemovingFromCollectionIds((current) => ({ ...current, [fileId]: true }));
    try {
      await onRemoveResourceFromCollection(resource);
      toggleResourceSelection(fileId, false);
    } catch {
      // Parent owns the Resources error region.
    } finally {
      setRemovingFromCollectionIds((current) => {
        const next = { ...current };
        delete next[fileId];
        return next;
      });
    }
  };

  const renderResourceContextMenuContent = (
    resource: ResourceRecord,
    state: {
      isDeleted: boolean;
      isRestoring: boolean;
      isDeleting: boolean;
      isRemovingFromCollection: boolean;
    }
  ) => {
    if (state.isDeleted) {
      return (
        <ContextMenuItem
          onSelect={() => onRestoreResource?.(resource)}
          disabled={state.isRestoring || !onRestoreResource}
        >
          {state.isRestoring ? (
            <Loader2 data-icon="inline-start" className="animate-spin" />
          ) : (
            <RotateCcw data-icon="inline-start" />
          )}
          Restore
        </ContextMenuItem>
      );
    }
    return (
      <>
        <ContextMenuItem onSelect={() => onOpenResource(resource)}>
          <Eye data-icon="inline-start" />
          View
        </ContextMenuItem>
        {downloadUrlFor ? (
          <ContextMenuItem asChild>
            <a href={downloadUrlFor(resource)} download={resourceDisplayName(resource)}>
              <Download data-icon="inline-start" />
              Download
            </a>
          </ContextMenuItem>
        ) : null}
        <ContextMenuItem onSelect={() => onUseInChat(resource)}>
          <Upload data-icon="inline-start" />
          Use in chat
        </ContextMenuItem>
        {onPushResourceToBisque ? (
          <ContextMenuItem
            onSelect={() => {
              void onPushResourceToBisque(resource);
            }}
          >
            <CloudUpload data-icon="inline-start" />
            Push to BisQue
          </ContextMenuItem>
        ) : null}
        {onRenameResource ? (
          <ContextMenuItem onSelect={() => openResourceRenameDialog(resource)}>
            <Pencil data-icon="inline-start" />
            Rename
          </ContextMenuItem>
        ) : null}
        {canUseSharing ? (
          <ContextMenuItem onSelect={() => openShareSheet(resource)}>
            <Share2 data-icon="inline-start" />
            Share
          </ContextMenuItem>
        ) : null}
        {activeResourceCollection && onRemoveResourceFromCollection ? (
          <ContextMenuItem
            onSelect={() => {
              void removeResourceFromCurrentFolder(resource);
            }}
            disabled={state.isRemovingFromCollection}
          >
            {state.isRemovingFromCollection ? (
              <Loader2 data-icon="inline-start" className="animate-spin" />
            ) : (
              <FolderMinus data-icon="inline-start" />
            )}
            Remove from folder
          </ContextMenuItem>
        ) : null}
        <ContextMenuSeparator />
        <ContextMenuItem
          variant="destructive"
          onSelect={() => onDeleteResource(resource)}
          disabled={state.isDeleting}
        >
          {state.isDeleting ? (
            <Loader2 data-icon="inline-start" className="animate-spin" />
          ) : (
            <Trash2 data-icon="inline-start" />
          )}
          Move to trash
        </ContextMenuItem>
      </>
    );
  };

  const renderCollectionContextMenuContent = (
    collection: ResourceCollectionRecord,
    options: { includeOpen: boolean }
  ) => {
    const deletingCollection = Boolean(deletingCollectionIds[collection.collection_id]);
    return (
      <>
        {options.includeOpen && onOpenCollection ? (
          <ContextMenuItem
            onSelect={() => {
              clearSelection();
              onOpenCollection(collection);
            }}
          >
            <Folder data-icon="inline-start" />
            Open
          </ContextMenuItem>
        ) : null}
        {onPushCollectionToBisque ? (
          <ContextMenuItem
            onSelect={() => {
              void onPushCollectionToBisque(collection);
            }}
          >
            <CloudUpload data-icon="inline-start" />
            Push to BisQue
          </ContextMenuItem>
        ) : null}
        {onRenameCollection ? (
          <ContextMenuItem onSelect={() => openCollectionRenameDialog(collection)}>
            <Pencil data-icon="inline-start" />
            Rename
          </ContextMenuItem>
        ) : null}
        {onCreateResourceCollectionShareGrants ? (
          <ContextMenuItem onSelect={() => openCollectionShareSheet(collection)}>
            <Share2 data-icon="inline-start" />
            Share
          </ContextMenuItem>
        ) : null}
        {collectionDownloadUrlFor ? (
          <ContextMenuItem asChild>
            {/* Server streams the folder as a zip (bundles included as trees). */}
            <a
              href={collectionDownloadUrlFor(collection)}
              download={`${String(collection.name ?? "").trim() || collection.collection_id}.zip`}
            >
              <Download data-icon="inline-start" />
              Download as zip
            </a>
          </ContextMenuItem>
        ) : null}
        {onDeleteCollection ? (
          <>
            <ContextMenuSeparator />
            <ContextMenuItem
              variant="destructive"
              onSelect={() => {
                void deleteCollection(collection);
              }}
              disabled={deletingCollection}
            >
              {deletingCollection ? (
                <Loader2 data-icon="inline-start" className="animate-spin" />
              ) : (
                <Trash2 data-icon="inline-start" />
              )}
              Move folder to trash
            </ContextMenuItem>
          </>
        ) : null}
      </>
    );
  };

  const deleteCollection = async (collection: ResourceCollectionRecord): Promise<void> => {
    if (!onDeleteCollection) {
      return;
    }
    const collectionId = collection.collection_id;
    setDeletingCollectionIds((current) => ({ ...current, [collectionId]: true }));
    try {
      await onDeleteCollection(collection);
      clearSelection();
    } catch {
      // Parent owns the Resources error region.
    } finally {
      setDeletingCollectionIds((current) => {
        const next = { ...current };
        delete next[collectionId];
        return next;
      });
    }
  };

  const openBulkShareSheet = (): void => {
    setBulkShareOpen(true);
    setBulkShareError(null);
  };

  const closeBulkShareSheet = (): void => {
    if (bulkShareSaving) {
      return;
    }
    setBulkShareOpen(false);
    setBulkShareGranteeUserId("");
    setBulkShareGranteeOrgId("");
    setBulkShareGranteeLabel("");
    setBulkShareError(null);
  };

  const createBulkShareGrants = async (): Promise<void> => {
    if ((!onCreateBulkResourceShareGrants && !onCreateResourceShareGrant) || !canGrantBulkShare) {
      return;
    }
    const targetResources = selectedResourcesInOrder;
    if (targetResources.length === 0) {
      return;
    }
    const request: ResourceShareGrantRequest = {
      granteeUserId: trimmedBulkShareGranteeUserId || undefined,
      granteeOrgId: trimmedBulkShareGranteeOrgId || undefined,
      role: "read",
    };
    setBulkShareSaving(true);
    setBulkShareError(null);
    try {
      if (onCreateBulkResourceShareGrants) {
        await onCreateBulkResourceShareGrants(targetResources, request);
      } else if (onCreateResourceShareGrant) {
        for (const resource of targetResources) {
          await onCreateResourceShareGrant(resource, request);
        }
      }
      clearSelection();
    } catch (shareError) {
      setBulkShareError(
        shareError instanceof Error ? shareError.message : "Unable to share selected resources."
      );
    } finally {
      setBulkShareSaving(false);
    }
  };

  const loadShareGrantsForResource = async (resource: ResourceRecord): Promise<void> => {
    if (!onLoadResourceShareGrants) {
      return;
    }
    setShareGrantsLoading(true);
    setShareGrantsError(null);
    try {
      const grants = await onLoadResourceShareGrants(resource);
      setShareGrantsByResourceId((current) => ({
        ...current,
        [resource.file_id]: grants,
      }));
    } catch (shareError) {
      setShareGrantsError(
        shareError instanceof Error ? shareError.message : "Unable to load resource shares."
      );
    } finally {
      setShareGrantsLoading(false);
    }
  };

  const openShareSheet = (resource: ResourceRecord): void => {
    setShareSheetResourceId(resource.file_id);
    setShareGranteeUserId("");
    setShareGranteeOrgId("");
    setShareGranteeLabel("");
    setShareGrantsError(null);
    void loadShareGrantsForResource(resource);
  };

  const closeShareSheet = (): void => {
    setShareSheetResourceId("");
    setShareGranteeUserId("");
    setShareGranteeOrgId("");
    setShareGranteeLabel("");
    setShareGrantsError(null);
    setShareSaving(false);
    setShareRevokingById({});
  };

  const openCollectionShareSheet = (collection: ResourceCollectionRecord): void => {
    setCollectionShareTarget(collection);
    setCollectionShareGranteeUserId("");
    setCollectionShareGranteeOrgId("");
    setCollectionShareGranteeLabel("");
    setCollectionShareError(null);
    setCollectionShareSaving(false);
  };

  const closeCollectionShareSheet = (): void => {
    if (collectionShareSaving) {
      return;
    }
    setCollectionShareTarget(null);
    setCollectionShareGranteeUserId("");
    setCollectionShareGranteeOrgId("");
    setCollectionShareGranteeLabel("");
    setCollectionShareError(null);
  };

  const createCollectionShareGrantsForSheet = async (): Promise<void> => {
    if (
      !onCreateResourceCollectionShareGrants ||
      !collectionShareTarget ||
      !canGrantCollectionShare
    ) {
      return;
    }
    setCollectionShareSaving(true);
    setCollectionShareError(null);
    try {
      await onCreateResourceCollectionShareGrants(collectionShareTarget, {
        granteeUserId: trimmedCollectionShareGranteeUserId || undefined,
        granteeOrgId: trimmedCollectionShareGranteeOrgId || undefined,
        role: "read",
      });
      setCollectionShareTarget(null);
      setCollectionShareGranteeUserId("");
      setCollectionShareGranteeOrgId("");
    setCollectionShareGranteeLabel("");
    } catch (shareError) {
      setCollectionShareError(
        shareError instanceof Error ? shareError.message : "Unable to share this folder."
      );
    } finally {
      setCollectionShareSaving(false);
    }
  };

  const createShareGrantForSheet = async (): Promise<void> => {
    if (!onCreateResourceShareGrant || !shareSheetResource || !canGrantShare) {
      return;
    }
    setShareSaving(true);
    setShareGrantsError(null);
    try {
      const grant = await onCreateResourceShareGrant(shareSheetResource, {
        granteeUserId: trimmedShareGranteeUserId,
        granteeOrgId: trimmedShareGranteeOrgId || undefined,
        role: "read",
      });
      setShareGrantsByResourceId((current) => {
        const existing = current[shareSheetResource.file_id] ?? [];
        return {
          ...current,
          [shareSheetResource.file_id]: [
            grant,
            ...existing.filter((item) => item.grant_id !== grant.grant_id),
          ],
        };
      });
      setShareGranteeUserId("");
      setShareGranteeOrgId("");
    setShareGranteeLabel("");
    } catch (shareError) {
      setShareGrantsError(
        shareError instanceof Error ? shareError.message : "Unable to grant resource access."
      );
    } finally {
      setShareSaving(false);
    }
  };

  const revokeShareGrantForSheet = async (
    grant: ResourceShareGrantRecord
  ): Promise<void> => {
    if (!onRevokeResourceShareGrant || !shareSheetResource) {
      return;
    }
    setShareGrantsError(null);
    setShareRevokingById((current) => ({ ...current, [grant.grant_id]: true }));
    try {
      const revokedGrant = await onRevokeResourceShareGrant(shareSheetResource, grant);
      setShareGrantsByResourceId((current) => {
        const existing = current[shareSheetResource.file_id] ?? [];
        const updatedExisting = existing.map((item) =>
          item.grant_id === revokedGrant.grant_id ? revokedGrant : item
        );
        return {
          ...current,
          [shareSheetResource.file_id]: existing.some(
            (item) => item.grant_id === revokedGrant.grant_id
          )
            ? updatedExisting
            : [revokedGrant, ...existing],
        };
      });
    } catch (shareError) {
      setShareGrantsError(
        shareError instanceof Error ? shareError.message : "Unable to revoke resource access."
      );
    } finally {
      setShareRevokingById((current) => {
        const next = { ...current };
        delete next[grant.grant_id];
        return next;
      });
    }
  };

  return (
    <section className="resource-browser mx-auto flex-1 overflow-y-auto px-3 py-6 sm:px-6 sm:py-8">
      <Card className="resource-browser-shell">
        <CardHeader className="resource-browser-header">
          <div className="resource-browser-header-row">
            <div className="resource-browser-heading">
              <CardTitle className="resource-browser-title">Resources</CardTitle>
              <p className="resource-browser-result-summary">{resultSummary}</p>
            </div>
            <div className="resource-browser-header-actions">
              <input
                ref={uploadInputRef}
                aria-hidden="true"
                className="sr-only"
                data-testid="resource-upload-files-input"
                hidden
                multiple
                tabIndex={-1}
                type="file"
                onChange={(event) => {
                  const files = Array.from(event.currentTarget.files ?? []);
                  event.currentTarget.value = "";
                  handleUploadInputSelection(files);
                }}
              />
              <input
                ref={folderUploadInputRef}
                aria-hidden="true"
                className="sr-only"
                data-testid="resource-upload-folder-input"
                hidden
                multiple
                tabIndex={-1}
                type="file"
                {...{ directory: "", webkitdirectory: "" }}
                onChange={(event) => {
                  const files = Array.from(event.currentTarget.files ?? []);
                  event.currentTarget.value = "";
                  handleUploadInputSelection(files);
                }}
              />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="resource-browser-new-menu"
                    aria-label="New"
                    disabled={uploading || loading || newFolderSaving}
                  >
                    {uploading ? (
                      <Loader2 data-icon="inline-start" className="animate-spin" />
                    ) : (
                      <FolderPlus data-icon="inline-start" />
                    )}
                    <span className="resource-browser-upload-label">
                      {uploading ? "Uploading..." : "New"}
                    </span>
                    <ChevronDown data-icon="inline-end" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="resource-browser-create-menu">
                  <DropdownMenuGroup>
                    {onCreateCollectionFromSelection ? (
                      <DropdownMenuItem onSelect={openNewFolderDialog}>
                        <FolderPlus data-icon="inline-start" />
                        New folder
                      </DropdownMenuItem>
                    ) : null}
                    <DropdownMenuItem
                      disabled={uploading || loading}
                      onSelect={() => {
                        pendingUploadReselectionRef.current = null;
                        setUploadReselectionError(null);
                        uploadInputRef.current?.click();
                      }}
                    >
                      <Upload data-icon="inline-start" />
                      Upload files
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={uploading || loading}
                      onSelect={() => {
                        pendingUploadReselectionRef.current = null;
                        setUploadReselectionError(null);
                        folderUploadInputRef.current?.click();
                      }}
                    >
                      <FolderUp data-icon="inline-start" />
                      Upload folder
                    </DropdownMenuItem>
                  </DropdownMenuGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <div className="resource-browser-controls">
            <div className="resource-browser-toolbar">
              <div className="resource-browser-search-field">
                <Search data-icon="inline-start" aria-hidden="true" />
                <Input
                  ref={searchInputRef}
                  value={query}
                  onChange={(event) => {
                    clearSelection();
                    resetTableScroll();
                    onQueryChange(event.target.value);
                  }}
                  // The nav bar's search action points here, and the magnifier is
                  // aria-hidden, so give the control a name of its own rather than
                  // leaning on the placeholder.
                  aria-label="Search resources"
                  placeholder="Search resources"
                  className="resource-browser-search"
                />
              </div>
              <Button
                type="button"
                variant="outline"
                className="resource-browser-filter-trigger"
                aria-label="More filters"
                onClick={() => setResourceFiltersOpen(true)}
              >
                <SlidersHorizontal data-icon="inline-start" />
                <span className="resource-browser-filter-label">More filters</span>
                {visibleFilterCount > 0 ? (
                  <span className="resource-browser-filter-count">{visibleFilterCount}</span>
                ) : null}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="resource-browser-refresh"
                aria-label="Refresh resources"
                onClick={() => {
                  clearSelection();
                  onRefresh();
                }}
                disabled={loading || loadingMore}
              >
                <RefreshCw data-icon="icon" className={cn((loading || loadingMore) && "animate-spin")} />
              </Button>
            </div>
            {activeTagFilter && onTagFilterChange ? (
              <div className="resource-browser-active-filters" aria-label="Active filters">
                <button
                  type="button"
                  className="resource-browser-active-filter-chip"
                  aria-label={`Clear tag filter ${activeTagFilter}`}
                  onClick={() => applyTagFilter("")}
                >
                  <span>Tag: {activeTagFilter}</span>
                  <XCircle data-icon="inline-end" />
                </button>
              </div>
            ) : null}
          </div>
        </CardHeader>
        <CardContent
          className="resource-browser-content"
          data-file-drop-active={fileDropTargetId === "resources" ? "true" : undefined}
          onDragLeave={(event) => {
            if (event.currentTarget === event.target && fileDropTargetId === "resources") {
              setFileDropTargetId("");
            }
          }}
          onDragOver={(event) => {
            handleExternalFileDragOver(event, "resources");
          }}
          onDrop={(event) => {
            uploadDroppedFiles(event, activeResourceCollection ?? null);
          }}
          onKeyDown={handleResourceBrowserKeyDown}
        >
          {error ? <p className="resource-browser-error">{error}</p> : null}
          {activeResourceCollection ? (
            <ContextMenu>
              <ContextMenuTrigger asChild>
                <div
                  className="resource-browser-active-collection"
                  aria-label="Active folder"
                  tabIndex={0}
                  onKeyDown={(event) => handleCollectionKeyDown(event, activeResourceCollection)}
                >
                  <div className="resource-browser-active-collection-copy">
                    <div className="resource-browser-active-collection-path" aria-label="Folder path">
                      {onClearActiveCollection ? (
                        <button
                          type="button"
                          className="resource-browser-active-collection-root"
                          aria-label="Go to all resources"
                          onClick={navigateToAllResources}
                        >
                          Resources
                        </button>
                      ) : (
                        <span className="resource-browser-active-collection-root">Resources</span>
                      )}
                      <span className="resource-browser-active-collection-separator" aria-hidden="true">
                        &gt;
                      </span>
                      {activeParentCollection && onOpenCollection ? (
                        <>
                          <button
                            type="button"
                            className="resource-browser-active-collection-parent"
                            aria-label={`Go up to folder ${
                              activeParentCollectionName || "Parent folder"
                            }`}
                            onClick={navigateFromActiveFolder}
                          >
                            {activeParentCollectionName || "Parent folder"}
                          </button>
                          <span
                            className="resource-browser-active-collection-separator"
                            aria-hidden="true"
                          >
                            &gt;
                          </span>
                        </>
                      ) : null}
                      {canNavigateFromActiveFolder ? (
                        <button
                          type="button"
                          className="resource-browser-active-collection-current"
                          aria-label={activeFolderNavigationLabel}
                          onClick={navigateFromActiveFolder}
                        >
                          <Folder data-icon="inline-start" />
                          <strong>{activeFolderName}</strong>
                        </button>
                      ) : (
                        <span
                          className="resource-browser-active-collection-current"
                          data-static="true"
                        >
                          <Folder data-icon="inline-start" />
                          <strong>{activeFolderName}</strong>
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="resource-browser-active-collection-actions">
                    {onClearActiveCollection ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="resource-browser-active-collection-clear"
                        aria-label="All resources"
                        data-drop-active={draggedResourceId ? "true" : undefined}
                        onDragOver={(event) => {
                          if (onRemoveResourceFromCollection && draggedResourceId) {
                            event.preventDefault();
                            event.dataTransfer.dropEffect = "move";
                          }
                        }}
                        onDrop={(event) => {
                          void handleAllResourcesDrop(event);
                        }}
                        onClick={navigateToAllResources}
                      >
                        <ArrowLeft data-icon="inline-start" />
                        <span className="resource-browser-active-collection-clear-label">
                          All resources
                        </span>
                      </Button>
                    ) : null}
                  </div>
                </div>
              </ContextMenuTrigger>
              <ContextMenuContent className="resource-browser-context-menu">
                {renderCollectionContextMenuContent(activeResourceCollection, {
                  includeOpen: false,
                })}
              </ContextMenuContent>
            </ContextMenu>
          ) : null}
          {onOpenCollection && visibleFolderCollections.length > 0 ? (
            <div className="resource-browser-collection-strip" aria-label="Resource folders">
              {visibleFolderCollections.map((collection) => {
                const resourceCount = Math.max(0, Number(collection.resource_count) || 0);
                return (
                  <ContextMenu key={collection.collection_id}>
                    <ContextMenuTrigger asChild>
                      <div
                        className="resource-browser-collection-item"
                        data-drop-active={
                          draggedResourceId || fileDropTargetId === collection.collection_id
                            ? "true"
                            : undefined
                        }
                        onDragLeave={(event) => {
                          if (
                            event.currentTarget === event.target &&
                            fileDropTargetId === collection.collection_id
                          ) {
                            setFileDropTargetId("");
                          }
                        }}
                        onDragOver={(event) => {
                          if (handleExternalFileDragOver(event, collection.collection_id)) {
                            return;
                          }
                          if (onAddSelectionToCollection && draggedResourceId) {
                            event.preventDefault();
                            event.dataTransfer.dropEffect = "move";
                          }
                        }}
                        onDrop={(event) => {
                          if (uploadDroppedFiles(event, collection)) {
                            return;
                          }
                          void handleFolderDrop(event, collection);
                        }}
                      >
                        <Button
                          type="button"
                          variant="ghost"
                          className="resource-browser-collection-button"
                          aria-label={`Open folder ${collection.name}`}
                          onKeyDown={(event) => handleCollectionKeyDown(event, collection)}
                          onClick={() => {
                            clearSelection();
                            onOpenCollection(collection);
                          }}
                        >
                          <Folder data-icon="inline-start" />
                          <span className="resource-browser-collection-copy">
                            <span className="resource-browser-collection-name">{collection.name}</span>
                            <span className="resource-browser-collection-count">
                              {resourceCount.toLocaleString()}{" "}
                              {resourceCount === 1 ? "resource" : "resources"}
                            </span>
                          </span>
                        </Button>
                      </div>
                    </ContextMenuTrigger>
                    <ContextMenuContent className="resource-browser-context-menu">
                      {renderCollectionContextMenuContent(collection, {
                        includeOpen: true,
                      })}
                    </ContextMenuContent>
                  </ContextMenu>
                );
              })}
            </div>
          ) : null}
          {!activeResourceCollection &&
          statusFilter === "deleted" &&
          onRestoreCollection &&
          deletedFolderCollections.length > 0 ? (
            <div
              className="resource-browser-collection-strip resource-browser-collection-strip-deleted"
              aria-label="Deleted resource folders"
            >
              {deletedFolderCollections.map((collection) => {
                const resourceCount = Math.max(0, Number(collection.resource_count) || 0);
                const restoringCollection = Boolean(restoringCollectionIds[collection.collection_id]);
                return (
                  <div key={collection.collection_id} className="resource-browser-collection-button">
                    <Folder data-icon="inline-start" />
                    <span className="resource-browser-collection-copy">
                      <span className="resource-browser-collection-name">{collection.name}</span>
                      <span className="resource-browser-collection-count">
                        {resourceCount.toLocaleString()} {resourceCount === 1 ? "resource" : "resources"}
                      </span>
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="resource-browser-collection-restore"
                      aria-label={`Restore folder ${collection.name}`}
                      disabled={restoringCollection}
                      onClick={() => {
                        clearSelection();
                        void onRestoreCollection(collection);
                      }}
                    >
                      {restoringCollection ? (
                        <Loader2 data-icon="inline-start" className="animate-spin" />
                      ) : (
                        <RotateCcw data-icon="inline-start" />
                      )}
                      Restore
                    </Button>
                  </div>
                );
              })}
            </div>
          ) : null}
          {visibleUploadProgress.length > 0 ? (
            <div className="resource-browser-upload-progress" aria-label="Resource upload progress">
              {uploadReselectionError ? (
                <p className="resource-browser-upload-progress-error" role="status">
                  {uploadReselectionError}
                </p>
              ) : null}
              {shouldCollapseExpiredUploads ? (
                <div className="resource-browser-upload-progress-summary">
                  <div className="resource-browser-upload-progress-copy">
                    <p className="resource-browser-upload-progress-name">
                      {expiredUploadProgress.length.toLocaleString()} uploads need attention
                    </p>
                    <p className="resource-browser-upload-progress-detail">
                      Expired sessions can be restarted when needed.
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="resource-browser-upload-progress-action"
                    aria-label={
                      expiredUploadDetailsOpen
                        ? "Hide upload recovery details"
                        : "Show upload recovery details"
                    }
                    onClick={() => setExpiredUploadDetailsOpen((open) => !open)}
                  >
                    {expiredUploadDetailsOpen ? "Hide details" : "Show details"}
                  </Button>
                </div>
              ) : null}
              {renderedUploadProgress.map((item) => {
                const totalBytes = Math.max(0, Number(item.totalBytes) || 0);
                const bytesVerified = Math.min(totalBytes, Math.max(0, Number(item.bytesVerified) || 0));
                const percent = totalBytes > 0 ? Math.round((bytesVerified / totalBytes) * 100) : 0;
                const statusLabel = uploadProgressStatusLabel(item.status);
                const recoveryLabel = uploadProgressRecoveryLabel(item.status);
                const errorLabel = uploadProgressErrorLabel(item.error);
                const dismissable = Boolean(onDismissUploadProgress) && canDismissUploadProgress(item.status);
                const pausable = Boolean(onPauseUploadProgress) && canPauseUploadProgress(item);
                const cancelable = Boolean(onCancelUploadProgress) && canCancelUploadProgress(item);
                return (
                  <div key={item.id} className="resource-browser-upload-progress-item">
                    <div className="resource-browser-upload-progress-copy">
                      <p className="resource-browser-upload-progress-name" title={item.fileName}>
                        {item.fileName}
                      </p>
                      <p className="resource-browser-upload-progress-detail">
                        <span>{statusLabel}</span>
                        <span>{formatBytes(bytesVerified)} verified</span>
                        {totalBytes > 0 ? <span>{formatBytes(totalBytes)} total</span> : null}
                      </p>
                      {errorLabel ? <p className="resource-browser-upload-progress-error">{errorLabel}</p> : null}
                    </div>
                    <div className="resource-browser-upload-progress-side">
                      <div
                        className="resource-browser-upload-progress-track"
                        role="progressbar"
                        aria-label={`${item.fileName} upload progress`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={percent}
                      >
                        <span style={{ width: `${percent}%` }} />
                      </div>
                      {recoveryLabel || pausable || dismissable || cancelable ? (
                        <div className="resource-browser-upload-progress-actions">
                          {recoveryLabel ? (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="resource-browser-upload-progress-action"
                              aria-label={`${recoveryLabel} ${item.fileName} upload`}
                              title={
                                item.relativePath
                                  ? "Select the same folder to continue this upload"
                                  : "Select the same file to continue this upload"
                              }
                              onClick={() => requestUploadReselection(item)}
                            >
                              <RotateCcw data-icon="inline-start" />
                              {recoveryLabel}
                            </Button>
                          ) : null}
                          {pausable ? (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="resource-browser-upload-progress-action"
                              aria-label={`Pause ${item.fileName} upload`}
                              onClick={() => onPauseUploadProgress?.(item)}
                            >
                              <Pause data-icon="inline-start" />
                              Pause
                            </Button>
                          ) : null}
                          {cancelable ? (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="resource-browser-upload-progress-action"
                              aria-label={`Cancel ${item.fileName} upload`}
                              onClick={() => onCancelUploadProgress?.(item)}
                            >
                              <XCircle data-icon="inline-start" />
                              Cancel
                            </Button>
                          ) : null}
                          {dismissable ? (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="resource-browser-upload-progress-action"
                              aria-label={`Dismiss ${item.fileName} upload`}
                              onClick={() => onDismissUploadProgress?.(item)}
                            >
                              <XCircle data-icon="inline-start" />
                              Dismiss
                            </Button>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
          {selectionEnabled && selectedResourceCount > 0 ? (
            <div className="resource-browser-bulk-toolbar" aria-label="Bulk resource actions">
              <div className="resource-browser-bulk-summary">
                <p>{selectedResourceCount.toLocaleString()} selected</p>
              </div>
              <div className="resource-browser-bulk-controls">
                {canUseBulkShare ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="resource-browser-bulk-action resource-browser-bulk-share-action"
                    onClick={openBulkShareSheet}
                    disabled={bulkShareSaving}
                    aria-label="Share"
                  >
                    {bulkShareSaving ? (
                      <Loader2 data-icon="inline-start" className="animate-spin" />
                    ) : (
                      <Share2 data-icon="inline-start" />
                    )}
                    <span className="resource-browser-bulk-action-label">Share</span>
                  </Button>
                ) : null}
                {onRestoreSelectedResources && statusFilter === "deleted" ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="resource-browser-bulk-action resource-browser-bulk-restore-action"
                    onClick={() => {
                      void restoreSelectedResources();
                    }}
                    disabled={!canRestoreSelectedResources}
                    aria-label="Restore"
                  >
                    {restoringSelection ? (
                      <Loader2 data-icon="inline-start" className="animate-spin" />
                    ) : (
                      <RotateCcw data-icon="inline-start" />
                    )}
                    <span className="resource-browser-bulk-action-label">Restore</span>
                  </Button>
                ) : null}
                {canShowBulkMoveToFolder ? (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="resource-browser-bulk-action resource-browser-bulk-move-action"
                        disabled={!canMoveSelectionToFolder}
                        aria-label="Move to"
                      >
                        {organizingSelection || resourceCollectionsLoading ? (
                          <Loader2 data-icon="inline-start" className="animate-spin" />
                        ) : (
                          <FolderPlus data-icon="inline-start" />
                        )}
                        <span className="resource-browser-bulk-action-label">Move to</span>
                        <ChevronDown data-icon="inline-end" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      align="start"
                      className="resource-browser-bulk-move-menu"
                    >
                      <DropdownMenuGroup>
                        {availableFolderCollections.map((collection) => (
                          <DropdownMenuItem
                            key={collection.collection_id}
                            onSelect={() => {
                              void addSelectionToExistingFolder(collection.collection_id);
                            }}
                            disabled={!canMoveSelectionToFolder}
                          >
                            <Folder data-icon="inline-start" />
                            {collection.name} ·{" "}
                            {Math.max(0, Number(collection.resource_count) || 0).toLocaleString()} resources
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuGroup>
                    </DropdownMenuContent>
                  </DropdownMenu>
                ) : null}
                {onDeleteSelectedResources && statusFilter !== "deleted" ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="resource-browser-bulk-action resource-browser-bulk-delete-action"
                    onClick={() => {
                      void deleteSelectedResources();
                    }}
                    disabled={!canDeleteSelectedResources}
                    aria-label="Move to trash"
                  >
                    {deletingSelection ? (
                      <Loader2 data-icon="inline-start" className="animate-spin" />
                    ) : (
                      <Trash2 data-icon="inline-start" />
                    )}
                    <span className="resource-browser-bulk-action-label">Move to trash</span>
                  </Button>
                ) : null}
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="resource-browser-bulk-clear"
                  onClick={clearSelection}
                  disabled={organizingSelection || deletingSelection || restoringSelection}
                  aria-label="Clear"
                >
                  <XCircle data-icon="inline-start" />
                  <span className="resource-browser-bulk-action-label">Clear</span>
                </Button>
              </div>
            </div>
          ) : null}
          {loading ? (
            <div className="resource-browser-grid" aria-label="Loading resources">
              {Array.from({ length: RESOURCE_SKELETON_COUNT }).map((_value, index) => (
                <article key={`resource-skeleton-${index}`} className="resource-browser-card resource-browser-skeleton-card">
                  <Skeleton className="resource-browser-skeleton-preview" />
                  <div className="resource-browser-meta">
                    <Skeleton className="resource-browser-skeleton-title" />
                    <Skeleton className="resource-browser-skeleton-line" />
                    <Skeleton className="resource-browser-skeleton-line resource-browser-skeleton-line-short" />
                  </div>
                </article>
              ))}
            </div>
          ) : cardResources.length === 0 ? (
            <p className="resource-browser-empty">No resources match the current filters.</p>
          ) : (
            <>
              {effectiveResourceViewMode === "table" ? (
                <div
                  ref={tableShellRef}
                  className="resource-browser-table-shell"
                  data-virtualized={tableVirtualized ? "true" : undefined}
                  onScroll={handleTableScroll}
                >
                  <table
                    className="resource-browser-table"
                    aria-label="Resources table"
                    aria-rowcount={cardResources.length + 1}
                  >
                    <thead>
                      <tr aria-rowindex={1}>
                        <th scope="col">Name</th>
                        <th scope="col">Type</th>
                        <th scope="col">Source</th>
                        <th scope="col">Size</th>
                        <th scope="col">Sharing</th>
                        <th scope="col">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableTopSpacerHeight > 0 ? (
                        <tr className="resource-browser-table-spacer" aria-hidden="true">
                          <td colSpan={6} style={{ height: tableTopSpacerHeight }} />
                        </tr>
                      ) : null}
                      {tableResources.map((resource, rowOffset) => {
                        const KindIcon = iconForKind(resource.resource_kind);
                        const displayName = resourceDisplayName(resource);
                        const isDeleting = Boolean(deletingFileIds[resource.file_id]);
                        const isDeleted = isDeletedResource(resource);
                        const isRestoring = Boolean(restoringFileIds[resource.file_id]);
                        const isSelected = Boolean(selectedResourceIds[resource.file_id]);
                        const isRemovingFromCollection = Boolean(
                          removingFromCollectionIds[resource.file_id]
                        );
                        const shareStatusChips = resourceShareStatusChips(resource);
                        const resourceTags = normalizeResourceTags(resource.tags);
                        const subtitle = resource.content_type || resource.file_id;
                        const rowIndex = tableStartIndex + rowOffset + 2;
                        return (
                          <ContextMenu key={resource.file_id}>
                            <ContextMenuTrigger asChild>
                              <tr
                                aria-label={`Resource ${displayName}`}
                                aria-rowindex={rowIndex}
                                data-selected={isSelected ? "true" : undefined}
                                draggable={
                                  !isDeleted &&
                                  (Boolean(onAddSelectionToCollection) ||
                                    Boolean(
                                      activeResourceCollection && onRemoveResourceFromCollection
                                    ))
                                }
                                tabIndex={0}
                                onClick={(event) => selectResourceFromSurfaceClick(event, resource)}
                                onDragStart={(event) => handleResourceDragStart(event, resource)}
                                onDragEnd={() => setDraggedResourceId("")}
                                onDoubleClick={() => {
                                  if (isDeleted) {
                                    onRestoreResource?.(resource);
                                  }
                                }}
                                onKeyDown={(event) =>
                                  handleResourceItemKeyDown(event, resource, isDeleted)
                                }
                              >
                                <td>
                                  <div className="resource-browser-table-name-cell">
                                    {selectionEnabled ? (
                                      <Checkbox
                                        aria-label={`Select ${displayName}`}
                                        checked={isSelected}
                                        className="resource-browser-table-select"
                                        onCheckedChange={(checked) =>
                                          toggleResourceSelection(resource.file_id, checked === true)
                                        }
                                      />
                                    ) : null}
                                    <span className="resource-browser-table-kind" aria-hidden="true">
                                      <KindIcon data-icon="icon" />
                                    </span>
                                    <span className="resource-browser-table-primary">
                                      <span className="resource-browser-table-title" title={displayName}>
                                        {displayName}
                                      </span>
                                      <span className="resource-browser-table-subtitle" title={subtitle}>
                                        {subtitle}
                                      </span>
                                      {resourceTags.length > 0 ? (
                                        <span
                                          className="resource-browser-resource-tags resource-browser-table-tags"
                                          aria-label={`Tags for ${displayName}`}
                                        >
                                          {resourceTags.slice(0, 4).map((tag) =>
                                            onTagFilterChange ? (
                                              <button
                                                key={tag}
                                                type="button"
                                                onClick={() => applyTagFilter(tag)}
                                                aria-label={`Filter by tag ${tag}`}
                                              >
                                                {tag}
                                              </button>
                                            ) : (
                                              <span key={tag}>{tag}</span>
                                            )
                                          )}
                                        </span>
                                      ) : null}
                                    </span>
                                  </div>
                                </td>
                                <td>{resourceKindLabel(resource.resource_kind)}</td>
                                <td>{sourceLabel(resource.source_type)}</td>
                                <td>{formatBytes(resource.size_bytes)}</td>
                                <td>
                                  {isDeleted ? (
                                    <div
                                      className="resource-browser-table-status resource-browser-lifecycle-chip"
                                      aria-label={`Lifecycle status for ${displayName}`}
                                    >
                                      <span>Deleted</span>
                                    </div>
                                  ) : shareStatusChips.length > 0 ? (
                                    <div
                                      className="resource-browser-table-status resource-browser-share-status-chips"
                                      aria-label={`Sharing status for ${displayName}`}
                                    >
                                      {shareStatusChips.map((chip) => (
                                        <span key={chip.key} title={chip.title}>
                                          {chip.label}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <span className="resource-browser-table-muted">Private</span>
                                  )}
                                </td>
                                <td>{formatResourceDate(resource.created_at)}</td>
                              </tr>
                            </ContextMenuTrigger>
                            <ContextMenuContent className="resource-browser-context-menu">
                              {renderResourceContextMenuContent(resource, {
                                isDeleted,
                                isRestoring,
                                isDeleting,
                                isRemovingFromCollection,
                              })}
                            </ContextMenuContent>
                          </ContextMenu>
                        );
                      })}
                      {tableBottomSpacerHeight > 0 ? (
                        <tr className="resource-browser-table-spacer" aria-hidden="true">
                          <td colSpan={6} style={{ height: tableBottomSpacerHeight }} />
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="resource-browser-grid">
                  {cardResources.map((resource) => {
                    const KindIcon = iconForKind(resource.resource_kind);
                    const displayName = resourceDisplayName(resource);
                    const thumbnailReady = shouldRequestThumbnail(resource, failedThumbnailIds);
                    // Text/data resources get a content-preview thumbnail (filling the
                    // full preview area like image cards) instead of a tiny icon chip.
                    const textPreviewKind =
                      quickPeekFetch && !thumbnailReady && !isVideoResource(resource)
                        ? classifyTextResource(resource)
                        : null;
                    const hasPreviewSurface = thumbnailReady || textPreviewKind !== null;
                    const isDeleting = Boolean(deletingFileIds[resource.file_id]);
                    const isDeleted = isDeletedResource(resource);
                    const isRestoring = Boolean(restoringFileIds[resource.file_id]);
                    const isSelected = Boolean(selectedResourceIds[resource.file_id]);
                    const isRemovingFromCollection = Boolean(
                      removingFromCollectionIds[resource.file_id]
                    );
                    const shareStatusChips = resourceShareStatusChips(resource);
                    const resourceTags = normalizeResourceTags(resource.tags);
                    const secondaryLine = [
                      sourceLabel(resource.source_type),
                      resourceKindLabel(resource.resource_kind),
                      formatBytes(resource.size_bytes),
                    ].join(" · ");
                    // Single-slot status chips. The meta box fits exactly three
                    // text rows (name, type/size, date), so on a tile that has a
                    // thumbnail these ride the preview's free bottom-left corner
                    // instead — otherwise a chip row could only be paid for by
                    // crushing the primary lines. Compact tiles have vertical
                    // slack, so there they stay in the meta flow.
                    const statusChipNodes = [
                      isDeleted ? (
                        <div
                          key="lifecycle"
                          className="resource-browser-lifecycle-chip"
                          aria-label={`Lifecycle status for ${displayName}`}
                        >
                          <span>Deleted</span>
                        </div>
                      ) : null,
                      !isDeleted && duplicateCountFor(resource) > 1 ? (
                        <div
                          key="duplicate"
                          className="resource-browser-dup-chip"
                          aria-label={`${displayName} has identical copies in view`}
                          title="Identical content (same checksum and size) appears more than once in this view"
                        >
                          <span>Duplicate ×{duplicateCountFor(resource)}</span>
                        </div>
                      ) : null,
                      shareStatusChips.length > 0 ? (
                        <div
                          key="share"
                          className="resource-browser-share-status-chips"
                          aria-label={`Sharing status for ${displayName}`}
                        >
                          {shareStatusChips.map((chip) => (
                            <span key={chip.key} title={chip.title}>
                              {chip.label}
                            </span>
                          ))}
                        </div>
                      ) : null,
                    ].filter(Boolean);
                    return (
                      <ContextMenu key={resource.file_id}>
                        <ContextMenuTrigger asChild>
                          <article
                            className="resource-browser-card group/resource"
                            data-preview={hasPreviewSurface ? "true" : "false"}
                            data-selected={isSelected ? "true" : undefined}
                            draggable={
                              !isDeleted &&
                              (Boolean(onAddSelectionToCollection) ||
                                Boolean(activeResourceCollection && onRemoveResourceFromCollection))
                            }
                            tabIndex={0}
                            aria-label={`Resource ${displayName}`}
                            onClick={(event) => selectResourceFromSurfaceClick(event, resource)}
                            onDragStart={(event) => handleResourceDragStart(event, resource)}
                            onDragEnd={() => setDraggedResourceId("")}
                            onDoubleClick={() => {
                              if (isDeleted) {
                                onRestoreResource?.(resource);
                              }
                            }}
                            onKeyDown={(event) =>
                              handleResourceItemKeyDown(event, resource, isDeleted)
                            }
                          >
                        <div className="resource-browser-preview">
                          {quickPeekFetch ? (
                            <div className="resource-quick-peek-slot">
                              <ResourceQuickPeek resource={resource} fetchHead={quickPeekFetch} />
                            </div>
                          ) : null}
                          {selectionEnabled ? (
                            <Checkbox
                              aria-label={`Select ${displayName}`}
                              checked={isSelected}
                              className="resource-browser-select"
                              onCheckedChange={(checked) =>
                                toggleResourceSelection(resource.file_id, checked === true)
                              }
                            />
                          ) : null}
                          {isVideoResource(resource) &&
                          downloadUrlFor &&
                          !failedThumbnailIds[resource.file_id] ? (
                            // Server ffmpeg poster by default; previews on hover.
                            <VideoThumbnail
                              posterUrl={thumbnailUrlFor(resource)}
                              videoUrl={downloadUrlFor(resource)}
                              alt={displayName}
                              className="resource-browser-video-thumb"
                              onError={() =>
                                setFailedThumbnailIds((previous) => ({
                                  ...previous,
                                  [resource.file_id]: true,
                                }))
                              }
                            />
                          ) : thumbnailReady ? (
                            zScrubThumbnail ? (
                              <ZScrubThumbnail
                                fileId={resource.file_id}
                                alt={displayName}
                                staticThumbnailUrl={thumbnailUrlFor(resource)}
                                sliceUrlFor={(z) => zScrubThumbnail!.sliceUrlFor(resource.file_id, z)}
                                loadZCount={zScrubThumbnail.loadZCount}
                                onError={() =>
                                  setFailedThumbnailIds((previous) => ({
                                    ...previous,
                                    [resource.file_id]: true,
                                  }))
                                }
                              />
                            ) : (
                              <img
                                src={thumbnailUrlFor(resource)}
                                alt={displayName}
                                loading="lazy"
                                onError={() =>
                                  setFailedThumbnailIds((previous) => ({
                                    ...previous,
                                    [resource.file_id]: true,
                                  }))
                                }
                              />
                            )
                          ) : textPreviewKind && quickPeekFetch ? (
                            <ResourceThumbnailPreview
                              resource={resource}
                              kind={textPreviewKind}
                              fetchHead={quickPeekFetch}
                              fallbackIcon={KindIcon}
                              fallbackLabel={resourceKindLabel(resource.resource_kind)}
                            />
                          ) : (
                            <div className="resource-browser-preview-fallback">
                              <KindIcon className="size-6" aria-hidden="true" />
                              <span>{resourceKindLabel(resource.resource_kind)}</span>
                            </div>
                          )}
                          {hasPreviewSurface && statusChipNodes.length > 0 ? (
                            <div className="resource-browser-status-overlay">{statusChipNodes}</div>
                          ) : null}
                        </div>
                        <div className="resource-browser-meta">
                          <p className="resource-browser-name" title={displayName}>
                            {displayName}
                          </p>
                          <p className="resource-browser-details">{secondaryLine}</p>
                          <p className="resource-browser-date">{formatResourceDate(resource.created_at)}</p>
                          {hasPreviewSurface ? null : statusChipNodes}
                          {resourceTags.length > 0 ? (
                            <div
                              className="resource-browser-resource-tags"
                              aria-label={`Tags for ${displayName}`}
                            >
                              {resourceTags.slice(0, 6).map((tag) =>
                                onTagFilterChange ? (
                                  <button
                                    key={tag}
                                    type="button"
                                    onClick={() => applyTagFilter(tag)}
                                    aria-label={`Filter by tag ${tag}`}
                                  >
                                    {tag}
                                  </button>
                                ) : (
                                  <span key={tag}>{tag}</span>
                                )
                              )}
                            </div>
                          ) : null}
                          {resource.sync_error ? (
                            <p className="resource-browser-sync-error" title={resource.sync_error}>
                              {resource.sync_error}
                            </p>
                          ) : null}
                        </div>
                          </article>
                        </ContextMenuTrigger>
                        <ContextMenuContent className="resource-browser-context-menu">
                          {renderResourceContextMenuContent(resource, {
                            isDeleted,
                            isRestoring,
                            isDeleting,
                            isRemovingFromCollection,
                          })}
                        </ContextMenuContent>
                      </ContextMenu>
                    );
                  })}
                </div>
              )}
              <div ref={loadMoreRef} className="resource-browser-load-more" aria-live="polite">
                {hasMore ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={onLoadMore}
                    disabled={loadingMore}
                  >
                    {loadingMore ? <Loader2 data-icon="inline-start" className="animate-spin" /> : null}
                    {loadingMore ? "Loading more..." : "Load more"}
                  </Button>
                ) : safeTotalCount > 0 ? (
                  <span>End of results</span>
                ) : null}
              </div>
            </>
          )}
        </CardContent>
      </Card>
      <Dialog
        open={newFolderDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            closeNewFolderDialog();
          } else {
            openNewFolderDialog();
          }
        }}
      >
        <DialogContent className="resource-browser-rename-dialog sm:max-w-md">
          <DialogHeader>
            <DialogTitle>New folder</DialogTitle>
            <DialogDescription className="sr-only">Create a folder in Resources.</DialogDescription>
          </DialogHeader>
          <div className="resource-browser-rename-form">
            <label htmlFor="resource-browser-new-folder-input">Folder name</label>
            <Input
              id="resource-browser-new-folder-input"
              value={newFolderName}
              onChange={(event) => setNewFolderName(event.target.value)}
              disabled={newFolderSaving}
              autoFocus
            />
            {newFolderError ? <p className="resource-browser-share-error">{newFolderError}</p> : null}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={closeNewFolderDialog}
              disabled={newFolderSaving}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => {
                void createEmptyFolder();
              }}
              disabled={!canCreateEmptyFolder}
            >
              {newFolderSaving ? (
                <Loader2 data-icon="inline-start" className="animate-spin" />
              ) : (
                <FolderPlus data-icon="inline-start" />
              )}
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={Boolean(renameTarget)}
        onOpenChange={(open) => {
          if (!open) {
            closeRenameDialog();
          }
        }}
      >
        <DialogContent className="resource-browser-rename-dialog sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Rename {renameTarget?.kind === "collection" ? "folder" : "resource"}
            </DialogTitle>
            <DialogDescription>{renameTargetName || "Choose a clear name."}</DialogDescription>
          </DialogHeader>
          <div className="resource-browser-rename-form">
            <label htmlFor="resource-browser-rename-input">Name</label>
            <Input
              id="resource-browser-rename-input"
              value={renameInput}
              onChange={(event) => setRenameInput(event.target.value)}
              disabled={renameSaving}
              autoFocus
            />
            {renameError ? <p className="resource-browser-share-error">{renameError}</p> : null}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={closeRenameDialog}
              disabled={renameSaving}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => {
                void saveRename();
              }}
              disabled={!canSaveRename}
            >
              {renameSaving ? (
                <Loader2 data-icon="inline-start" className="animate-spin" />
              ) : (
                <Pencil data-icon="inline-start" />
              )}
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Sheet
        open={bulkShareOpen}
        onOpenChange={(open) => {
          if (!open) {
            closeBulkShareSheet();
          } else {
            setBulkShareOpen(true);
          }
        }}
      >
        <SheetContent
          side="right"
          className="resource-browser-share-sheet w-[min(100vw,38rem)] sm:max-w-lg"
        >
          <SheetHeader className="resource-browser-share-header">
            <SheetTitle className="resource-browser-share-title">
              Share {selectedResourceCount.toLocaleString()} selected{" "}
              {selectedResourceCount === 1 ? "resource" : "resources"}
            </SheetTitle>
            <SheetDescription className="resource-browser-share-description">
              {pluralSummary(selectedResourceCount, "resource")} selected. Add a person or team
              that can read these resources.
            </SheetDescription>
          </SheetHeader>
          <div className="resource-browser-share-body">
            <div className="resource-browser-share-form">
              {onSearchShareTargets ? (
                <ShareTargetPicker
                  idPrefix="resource-bulk-share"
                  onSearch={onSearchShareTargets}
                  selectedLabel={bulkShareGranteeLabel}
                  hasSelection={Boolean(bulkShareGranteeUserId.trim() || bulkShareGranteeOrgId.trim())}
                  disabled={bulkShareSaving}
                  onPick={(target) => {
                    setBulkShareGranteeUserId(target.grantee_user_id ?? "");
                    setBulkShareGranteeOrgId(target.grantee_org_id ?? "");
                    setBulkShareGranteeLabel(target.label);
                  }}
                  onClear={() => {
                    setBulkShareGranteeUserId("");
                    setBulkShareGranteeOrgId("");
                    setBulkShareGranteeLabel("");
                  }}
                />
              ) : (
              <div className="resource-browser-share-grid">
                <div>
                  <label htmlFor="resource-bulk-share-user-id">Person or user ID</label>
                  <Input
                    id="resource-bulk-share-user-id"
                    value={bulkShareGranteeUserId}
                    onChange={(event) => setBulkShareGranteeUserId(event.target.value)}
                    autoComplete="off"
                    disabled={bulkShareSaving}
                  />
                </div>
                <div>
                  <label htmlFor="resource-bulk-share-org-id">Team or organization ID</label>
                  <Input
                    id="resource-bulk-share-org-id"
                    value={bulkShareGranteeOrgId}
                    onChange={(event) => setBulkShareGranteeOrgId(event.target.value)}
                    autoComplete="off"
                    disabled={bulkShareSaving}
                  />
                </div>
              </div>
              )}
              <Button
                type="button"
                variant="default"
                size="sm"
                className="resource-browser-share-submit"
                onClick={() => {
                  void createBulkShareGrants();
                }}
                disabled={!canGrantBulkShare}
              >
                {bulkShareSaving ? (
                  <Loader2 data-icon="inline-start" className="animate-spin" />
                ) : (
                  <UserPlus data-icon="inline-start" />
                )}
                Share selected
              </Button>
            </div>
            {bulkShareError ? (
              <p className="resource-browser-share-error">{bulkShareError}</p>
            ) : null}
          </div>
        </SheetContent>
      </Sheet>
      <Sheet
        open={Boolean(collectionShareTarget)}
        onOpenChange={(open) => {
          if (!open) {
            closeCollectionShareSheet();
          }
        }}
      >
        {collectionShareTarget ? (
          <SheetContent
            side="right"
            className="resource-browser-share-sheet w-[min(100vw,38rem)] sm:max-w-lg"
          >
            <SheetHeader className="resource-browser-share-header">
              <SheetTitle className="resource-browser-share-title">
                Share {collectionShareTarget.name}
              </SheetTitle>
              <SheetDescription className="resource-browser-share-description">
                Add a person or team that can read every resource in this folder.
              </SheetDescription>
            </SheetHeader>
            <div className="resource-browser-share-body">
              <div className="resource-browser-share-form">
                {onSearchShareTargets ? (
                  <ShareTargetPicker
                    idPrefix="resource-folder-share"
                    onSearch={onSearchShareTargets}
                    selectedLabel={collectionShareGranteeLabel}
                    hasSelection={Boolean(
                      collectionShareGranteeUserId.trim() || collectionShareGranteeOrgId.trim()
                    )}
                    disabled={collectionShareSaving}
                    onPick={(target) => {
                      setCollectionShareGranteeUserId(target.grantee_user_id ?? "");
                      setCollectionShareGranteeOrgId(target.grantee_org_id ?? "");
                      setCollectionShareGranteeLabel(target.label);
                    }}
                    onClear={() => {
                      setCollectionShareGranteeUserId("");
                      setCollectionShareGranteeOrgId("");
                      setCollectionShareGranteeLabel("");
                    }}
                  />
                ) : (
                <div className="resource-browser-share-grid">
                  <div>
                    <label htmlFor="resource-folder-share-user-id">Person or user ID</label>
                    <Input
                      id="resource-folder-share-user-id"
                      value={collectionShareGranteeUserId}
                      onChange={(event) => setCollectionShareGranteeUserId(event.target.value)}
                      autoComplete="off"
                      disabled={collectionShareSaving}
                    />
                  </div>
                  <div>
                    <label htmlFor="resource-folder-share-org-id">
                      Team or organization ID
                    </label>
                    <Input
                      id="resource-folder-share-org-id"
                      value={collectionShareGranteeOrgId}
                      onChange={(event) => setCollectionShareGranteeOrgId(event.target.value)}
                      autoComplete="off"
                      disabled={collectionShareSaving}
                    />
                  </div>
                </div>
                )}
                {collectionShareTarget && onLoadCollectionShareGrants ? (
                  <CollectionAccessList
                    collection={collectionShareTarget}
                    load={onLoadCollectionShareGrants}
                    revoke={onRevokeCollectionShareGrant}
                  />
                ) : null}
                <Button
                  type="button"
                  variant="default"
                  size="sm"
                  className="resource-browser-share-submit"
                  onClick={() => {
                    void createCollectionShareGrantsForSheet();
                  }}
                  disabled={!canGrantCollectionShare}
                >
                  {collectionShareSaving ? (
                    <Loader2 data-icon="inline-start" className="animate-spin" />
                  ) : (
                    <UserPlus data-icon="inline-start" />
                  )}
                  Share folder
                </Button>
              </div>
              {collectionShareError ? (
                <p className="resource-browser-share-error">{collectionShareError}</p>
              ) : null}
            </div>
          </SheetContent>
        ) : null}
      </Sheet>
      <Sheet
        open={Boolean(shareSheetResource)}
        onOpenChange={(open) => {
          if (!open) {
            closeShareSheet();
          }
        }}
      >
        {shareSheetResource ? (
          <SheetContent
            side="right"
            className="resource-browser-share-sheet w-[min(100vw,38rem)] sm:max-w-lg"
          >
            <SheetHeader className="resource-browser-share-header">
              <SheetTitle className="resource-browser-share-title">
                Share {shareSheetResourceName}
              </SheetTitle>
              <SheetDescription className="resource-browser-share-description">
                Add a person or team that can read this resource.
              </SheetDescription>
            </SheetHeader>
            <div className="resource-browser-share-body">
              <div className="resource-browser-share-form">
                {onSearchShareTargets ? (
                  <ShareTargetPicker
                    idPrefix="resource-share"
                    onSearch={onSearchShareTargets}
                    selectedLabel={shareGranteeLabel}
                    hasSelection={Boolean(shareGranteeUserId.trim() || shareGranteeOrgId.trim())}
                    onPick={(target) => {
                      setShareGranteeUserId(target.grantee_user_id ?? "");
                      setShareGranteeOrgId(target.grantee_org_id ?? "");
                      setShareGranteeLabel(target.label);
                    }}
                    onClear={() => {
                      setShareGranteeUserId("");
                      setShareGranteeOrgId("");
                      setShareGranteeLabel("");
                    }}
                  />
                ) : (
                <div className="resource-browser-share-grid">
                  <div>
                    <label htmlFor="resource-share-user-id">Person or user ID</label>
                    <Input
                      id="resource-share-user-id"
                      value={shareGranteeUserId}
                      onChange={(event) => setShareGranteeUserId(event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                  <div>
                    <label htmlFor="resource-share-org-id">Team or organization ID</label>
                    <Input
                      id="resource-share-org-id"
                      value={shareGranteeOrgId}
                      onChange={(event) => setShareGranteeOrgId(event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                </div>
                )}
                <Button
                  type="button"
                  variant="default"
                  size="sm"
                  className="resource-browser-share-submit"
                  onClick={() => {
                    void createShareGrantForSheet();
                  }}
                  disabled={!canGrantShare}
                >
                  {shareSaving ? (
                    <Loader2 data-icon="inline-start" className="animate-spin" />
                  ) : (
                    <UserPlus data-icon="inline-start" />
                  )}
                  Share
                </Button>
              </div>

              {shareGrantsError ? (
                <p className="resource-browser-share-error">{shareGrantsError}</p>
              ) : null}

              <section className="resource-browser-share-section" aria-label="People with access">
                <div className="resource-browser-share-section-header">
                  <h3>People with access</h3>
                  {shareGrantsLoading ? (
                    <span>
                      <Loader2 data-icon="inline-start" className="animate-spin" />
                      Loading
                    </span>
                  ) : (
                    <span>{shareSheetGrants.length.toLocaleString()}</span>
                  )}
                </div>
                {shareGrantsLoading && shareSheetGrants.length === 0 ? (
                  <div className="resource-browser-share-list" aria-label="Loading share grants">
                    {Array.from({ length: 3 }).map((_item, index) => (
                      <div key={`share-skeleton-${index}`} className="resource-browser-share-row">
                        <Skeleton className="resource-browser-share-skeleton-title" />
                        <Skeleton className="resource-browser-share-skeleton-line" />
                      </div>
                    ))}
                  </div>
                ) : shareSheetGrants.length === 0 ? (
                  <p className="resource-browser-share-empty">No share grants yet.</p>
                ) : (
                  <div className="resource-browser-share-list">
                    {shareSheetGrants.map((grant) => {
                      const identity = resourceShareGrantIdentity(grant);
                      const status = String(grant.status || "").toLowerCase();
                      const isRevoking = Boolean(shareRevokingById[grant.grant_id]);
                      const canRevoke =
                        status === "active" && Boolean(onRevokeResourceShareGrant) && !isRevoking;
                      const grantDetails = [
                        grant.grantee_user_id && grant.grantee_user_id !== identity
                          ? grant.grantee_user_id
                          : "",
                        grant.grantee_org_id && grant.grantee_org_id !== identity
                          ? grant.grantee_org_id
                          : "",
                        resourceShareGrantRoleLabel(grant.role),
                      ].filter((detail): detail is string => Boolean(detail));
                      return (
                        <article
                          key={grant.grant_id}
                          className="resource-browser-share-row"
                          data-status={status}
                        >
                          <div className="resource-browser-share-row-main">
                            <div className="resource-browser-share-avatar" aria-hidden="true">
                              {identity.slice(0, 2).toUpperCase()}
                            </div>
                            <div className="resource-browser-share-copy">
                              <p>{identity}</p>
                              <div>
                                {grantDetails.map((detail) => (
                                  <span key={detail}>{detail}</span>
                                ))}
                              </div>
                            </div>
                          </div>
                          <div className="resource-browser-share-row-side">
                            <span className="resource-browser-share-status">
                              {resourceShareGrantStatusLabel(grant.status)}
                            </span>
                            <span>{formatResourceAuditDate(grant.updated_at || grant.created_at)}</span>
                            {status === "active" ? (
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="resource-browser-share-revoke"
                                onClick={() => {
                                  void revokeShareGrantForSheet(grant);
                                }}
                                disabled={!canRevoke}
                                aria-label={`Revoke read access for ${identity}`}
                              >
                                {isRevoking ? (
                                  <Loader2 data-icon="inline-start" className="animate-spin" />
                                ) : null}
                                Revoke
                              </Button>
                            ) : null}
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>
            </div>
          </SheetContent>
        ) : null}
      </Sheet>
      <Sheet open={resourceFiltersOpen} onOpenChange={setResourceFiltersOpen}>
        <SheetContent
          side={isMobileView ? "bottom" : "right"}
          className={cn(
            "resource-browser-filter-sheet",
            isMobileView
              ? "rounded-t-[1.75rem] px-4 pb-[calc(1.2rem+env(safe-area-inset-bottom,0px))] pt-3"
              : "w-[min(100vw,32rem)] px-5 py-5 sm:max-w-md"
          )}
        >
          <SheetHeader className="gap-2 text-left">
            <SheetTitle className="text-base">Resource filters and view</SheetTitle>
            <SheetDescription className="text-sm leading-6 text-muted-foreground">
              Choose a view, then narrow by type, source, sharing, or lifecycle.
            </SheetDescription>
          </SheetHeader>
          <div className="resource-browser-sheet-section">
            <p className="resource-browser-sheet-label">View</p>
            <ToggleGroup
              type="single"
              value={effectiveResourceViewMode}
              onValueChange={(value) => {
                if (value === "cards" || value === "table") {
                  resetTableScroll();
                  setResourceViewMode(value);
                }
              }}
              variant="outline"
              size="sm"
              className="resource-browser-sheet-view-toggle"
              aria-label="Resource view mode"
            >
              <ToggleGroupItem value="cards" aria-label="Card view">
                <LayoutGrid data-icon="inline-start" />
                <span>Cards</span>
              </ToggleGroupItem>
              {!isMobileView ? (
                <ToggleGroupItem value="table" aria-label="Table view">
                  <Table2 data-icon="inline-start" />
                  <span>Table</span>
                </ToggleGroupItem>
              ) : null}
            </ToggleGroup>
          </div>
          <div className="resource-browser-sheet-section">
            <p className="resource-browser-sheet-label">Type</p>
            <div className="resource-browser-sheet-options">
              {kindFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={kindFilter === item.value ? "secondary" : "outline"}
                  size="sm"
                  onClick={() => {
                    clearSelection();
                    resetTableScroll();
                    onKindFilterChange(item.value);
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="resource-browser-sheet-section">
            <p className="resource-browser-sheet-label">Source</p>
            <div className="resource-browser-sheet-options">
              {sourceFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={sourceFilter === item.value ? "secondary" : "outline"}
                  size="sm"
                  onClick={() => {
                    clearSelection();
                    resetTableScroll();
                    onSourceFilterChange(item.value);
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="resource-browser-sheet-section">
            <p className="resource-browser-sheet-label">Sharing</p>
            <div className="resource-browser-sheet-options">
              {sharingFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={sharingFilter === item.value ? "secondary" : "outline"}
                  size="sm"
                  onClick={() => {
                    clearSelection();
                    resetTableScroll();
                    onSharingFilterChange?.(item.value);
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="resource-browser-sheet-section">
            <p className="resource-browser-sheet-label">Lifecycle</p>
            <div className="resource-browser-sheet-options">
              {statusFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={statusFilter === item.value ? "secondary" : "outline"}
                  size="sm"
                  onClick={() => {
                    clearSelection();
                    resetTableScroll();
                    onStatusFilterChange?.(item.value);
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="resource-browser-sheet-actions">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                clearSelection();
                resetTableScroll();
                onKindFilterChange("all");
                onSourceFilterChange("all");
                onSharingFilterChange?.("all");
                onStatusFilterChange?.("active");
                onTagFilterChange?.("");
              }}
            >
              Reset
            </Button>
            <Button type="button" onClick={() => setResourceFiltersOpen(false)}>
              Done
            </Button>
          </div>
        </SheetContent>
      </Sheet>
    </section>
  );
}
