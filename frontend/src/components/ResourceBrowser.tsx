import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/format";
import {
  ExternalLink,
  Eye,
  File,
  Film,
  ImageIcon,
  Link2,
  Loader2,
  RefreshCw,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";
import type { ResourceRecord } from "../types";

export type ResourceKindFilter = "all" | "image" | "video" | "table" | "file";
export type ResourceSourceFilter = "all" | "upload" | "bisque_import";

type ResourceBrowserProps = {
  resources: ResourceRecord[];
  loading: boolean;
  error: string | null;
  query: string;
  kindFilter: ResourceKindFilter;
  sourceFilter: ResourceSourceFilter;
  deletingFileIds: Record<string, boolean>;
  onQueryChange: (value: string) => void;
  onKindFilterChange: (value: ResourceKindFilter) => void;
  onSourceFilterChange: (value: ResourceSourceFilter) => void;
  onRefresh: () => void;
  onOpenResource: (resource: ResourceRecord) => void;
  onUseInChat: (resource: ResourceRecord) => void;
  onDeleteResource: (resource: ResourceRecord) => void;
  thumbnailUrlFor: (resource: ResourceRecord) => string;
};

const kindFilters: Array<{ value: ResourceKindFilter; label: string }> = [
  { value: "all", label: "All types" },
  { value: "image", label: "Images" },
  { value: "video", label: "Videos" },
  { value: "table", label: "Tables" },
  { value: "file", label: "Files" },
];

const sourceFilters: Array<{ value: ResourceSourceFilter; label: string }> = [
  { value: "all", label: "All sources" },
  { value: "upload", label: "Local uploads" },
  { value: "bisque_import", label: "BisQue imports" },
];

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
    return "Browser upload";
  }
  return value || "Source";
};

const syncStatusLabel = (value: string | null | undefined): string | null => {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === "bisque_sync_succeeded") {
    return "Cataloged on BisQue";
  }
  if (normalized === "bisque_sync_running" || normalized === "bisque_sync_queued") {
    return "Syncing to BisQue";
  }
  if (normalized === "bisque_sync_failed") {
    return "BisQue sync failed";
  }
  if (normalized === "local_complete") {
    return "Local staging ready";
  }
  return normalized.replace(/_/g, " ");
};

const iconForKind = (kind: string) => {
  switch (String(kind || "").toLowerCase()) {
    case "image":
      return ImageIcon;
    case "video":
      return Film;
    case "table":
      return Table2;
    default:
      return File;
  }
};

export function ResourceBrowser({
  resources,
  loading,
  error,
  query,
  kindFilter,
  sourceFilter,
  deletingFileIds,
  onQueryChange,
  onKindFilterChange,
  onSourceFilterChange,
  onRefresh,
  onOpenResource,
  onUseInChat,
  onDeleteResource,
  thumbnailUrlFor,
}: ResourceBrowserProps) {
  const [failedThumbnailIds, setFailedThumbnailIds] = useState<Record<string, true>>({});
  const cardResources = useMemo(() => resources, [resources]);

  return (
    <section className="resource-browser mx-auto flex-1 overflow-y-auto px-3 py-6 sm:px-6 sm:py-8">
      <Card className="resource-browser-shell">
        <CardHeader className="resource-browser-header">
          <div className="resource-browser-header-row">
            <div>
              <CardTitle className="resource-browser-title">Resource Browser</CardTitle>
              <p className="resource-browser-subtitle">
                Browse uploads and BisQue imports, open previews, and remove files you no longer need.
              </p>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="resource-browser-refresh"
              onClick={onRefresh}
              disabled={loading}
            >
              <RefreshCw className={cn("size-4", loading && "animate-spin")} />
              Refresh
            </Button>
          </div>
          <div className="resource-browser-controls">
            <Input
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="Search files, BisQue IDs, or URLs"
              className="resource-browser-search"
            />
            <div className="resource-browser-filter-row">
              {kindFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={kindFilter === item.value ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => onKindFilterChange(item.value)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
            <div className="resource-browser-filter-row">
              {sourceFilters.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  variant={sourceFilter === item.value ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => onSourceFilterChange(item.value)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="resource-browser-content">
          {error ? <p className="resource-browser-error">{error}</p> : null}
          {loading ? (
            <p className="resource-browser-empty">Loading resources…</p>
          ) : cardResources.length === 0 ? (
            <p className="resource-browser-empty">No resources match the current filters.</p>
          ) : (
            <div className="resource-browser-grid">
              {cardResources.map((resource) => {
                const KindIcon = iconForKind(resource.resource_kind);
                const canShowThumbnail =
                  resource.resource_kind === "image" &&
                  !failedThumbnailIds[resource.file_id];
                const isDeleting = Boolean(deletingFileIds[resource.file_id]);
                return (
                  <article key={resource.file_id} className="resource-browser-card group/resource">
                    <div className="resource-browser-preview">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="resource-browser-delete"
                        onClick={() => onDeleteResource(resource)}
                        disabled={isDeleting}
                        aria-label={isDeleting ? "Deleting resource" : "Delete resource"}
                      >
                        {isDeleting ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Trash2 className="size-4" />
                        )}
                      </Button>
                      {canShowThumbnail ? (
                        <img
                          src={thumbnailUrlFor(resource)}
                          alt={resource.original_name}
                          loading="lazy"
                          onError={() =>
                            setFailedThumbnailIds((previous) => ({
                              ...previous,
                              [resource.file_id]: true,
                            }))
                          }
                        />
                      ) : (
                        <div className="resource-browser-preview-fallback">
                          <KindIcon className="size-5" />
                          <span>{resourceKindLabel(resource.resource_kind)}</span>
                        </div>
                      )}
                    </div>
                    <div className="resource-browser-meta">
                      <p className="resource-browser-name" title={resource.original_name}>
                        {resource.original_name}
                      </p>
                      <p className="resource-browser-details">
                        {formatBytes(resource.size_bytes)} • {formatResourceDate(resource.created_at)}
                      </p>
                      <div className="resource-browser-badges">
                        <Badge variant="outline" className="resource-browser-tag">
                          {sourceLabel(resource.source_type)}
                        </Badge>
                        <Badge variant="outline" className="resource-browser-tag">
                          {resourceKindLabel(resource.resource_kind)}
                        </Badge>
                        {syncStatusLabel(resource.sync_status) ? (
                          <Badge variant="outline" className="resource-browser-tag">
                            {syncStatusLabel(resource.sync_status)}
                          </Badge>
                        ) : null}
                      </div>
                      {resource.sync_error ? (
                        <p className="resource-browser-uri text-amber-700" title={resource.sync_error}>
                          <Loader2 className="size-3.5" />
                          <span>{resource.sync_error}</span>
                        </p>
                      ) : null}
                      {resource.source_uri ? (
                        <p className="resource-browser-uri" title={resource.source_uri}>
                          <Link2 className="size-3.5" />
                          <span>{resource.source_uri}</span>
                        </p>
                      ) : null}
                      {resource.client_view_url || resource.image_service_url ? (
                        <div className="resource-browser-links">
                          {resource.client_view_url ? (
                            <a href={resource.client_view_url} target="_blank" rel="noreferrer">
                              <ExternalLink className="size-3.5" />
                              BisQue viewer
                            </a>
                          ) : null}
                          {resource.image_service_url ? (
                            <a href={resource.image_service_url} target="_blank" rel="noreferrer">
                              <ExternalLink className="size-3.5" />
                              Image service
                            </a>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                    <CardFooter className="resource-browser-actions">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="resource-browser-action-button"
                        onClick={() => onOpenResource(resource)}
                      >
                        <Eye className="size-4" />
                        View
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="resource-browser-action-button"
                        onClick={() => onUseInChat(resource)}
                      >
                        <Upload className="size-4" />
                        Use in chat
                      </Button>
                    </CardFooter>
                  </article>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
