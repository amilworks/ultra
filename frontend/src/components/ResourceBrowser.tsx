import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
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
import { resourceDisplayName } from "@/features/resources/presentation";
import {
  Eye,
  File,
  Film,
  ImageIcon,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";
import type { ResourceRecord } from "../types";

export type ResourceKindFilter = "all" | "image" | "video" | "table" | "file";
export type ResourceSourceFilter = "all" | "upload" | "bisque_import";

type ResourceBrowserProps = {
  resources: ResourceRecord[];
  totalCount: number;
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  query: string;
  kindFilter: ResourceKindFilter;
  sourceFilter: ResourceSourceFilter;
  deletingFileIds: Record<string, boolean>;
  onQueryChange: (value: string) => void;
  onKindFilterChange: (value: ResourceKindFilter) => void;
  onSourceFilterChange: (value: ResourceSourceFilter) => void;
  onRefresh: () => void;
  onLoadMore: () => void;
  onOpenResource: (resource: ResourceRecord) => void;
  onUseInChat: (resource: ResourceRecord) => void;
  onDeleteResource: (resource: ResourceRecord) => void;
  thumbnailUrlFor: (resource: ResourceRecord) => string;
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

const RESOURCE_SKELETON_COUNT = 8;

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
    return "Upload";
  }
  return value || "Source";
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

export function ResourceBrowser({
  resources,
  totalCount,
  loading,
  loadingMore,
  hasMore,
  error,
  query,
  kindFilter,
  sourceFilter,
  deletingFileIds,
  onQueryChange,
  onKindFilterChange,
  onSourceFilterChange,
  onRefresh,
  onLoadMore,
  onOpenResource,
  onUseInChat,
  onDeleteResource,
  thumbnailUrlFor,
}: ResourceBrowserProps) {
  const isMobileView = useBreakpoint(721);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [failedThumbnailIds, setFailedThumbnailIds] = useState<Record<string, true>>({});
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  const cardResources = useMemo(() => resources, [resources]);
  const safeTotalCount = Math.max(0, Math.floor(Number(totalCount) || 0));
  const visibleCount = cardResources.length;
  const activeFilterCount =
    Number(kindFilter !== "all") + Number(sourceFilter !== "all");
  const resultSummary = loading
    ? "Searching resources..."
    : `${visibleCount.toLocaleString()} of ${safeTotalCount.toLocaleString()} resources`;

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

  return (
    <section className="resource-browser mx-auto flex-1 overflow-y-auto px-3 py-6 sm:px-6 sm:py-8">
      <Card className="resource-browser-shell">
        <CardHeader className="resource-browser-header">
          <div className="resource-browser-header-row">
            <div className="resource-browser-heading">
              <CardTitle className="resource-browser-title">Resources</CardTitle>
              <p className="resource-browser-result-summary">{resultSummary}</p>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="resource-browser-refresh"
              onClick={onRefresh}
              disabled={loading || loadingMore}
            >
              <RefreshCw data-icon="inline-start" className={cn((loading || loadingMore) && "animate-spin")} />
              <span className="resource-browser-refresh-label">Refresh</span>
            </Button>
          </div>
          <div className="resource-browser-controls">
            <div className="resource-browser-toolbar">
              <Input
                value={query}
                onChange={(event) => onQueryChange(event.target.value)}
                placeholder="Search files, BisQue IDs, or URLs"
                className="resource-browser-search"
              />
              {isMobileView ? (
                <Button
                  type="button"
                  variant="outline"
                  className="resource-browser-filter-trigger"
                  onClick={() => setMobileFiltersOpen(true)}
                >
                  <SlidersHorizontal data-icon="inline-start" />
                  <span>Filters</span>
                  {activeFilterCount > 0 ? (
                    <span className="resource-browser-filter-count">{activeFilterCount}</span>
                  ) : null}
                </Button>
              ) : null}
            </div>
            {isMobileView ? null : (
              <div className="resource-browser-filter-groups">
                <div className="resource-browser-filter-row" aria-label="Resource type filters">
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
                <div className="resource-browser-filter-row" aria-label="Resource source filters">
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
            )}
          </div>
        </CardHeader>
        <CardContent className="resource-browser-content">
          {error ? <p className="resource-browser-error">{error}</p> : null}
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
                  <div className="resource-browser-actions">
                    <Skeleton className="resource-browser-skeleton-action" />
                    <Skeleton className="resource-browser-skeleton-action" />
                  </div>
                </article>
              ))}
            </div>
          ) : cardResources.length === 0 ? (
            <p className="resource-browser-empty">No resources match the current filters.</p>
          ) : (
            <>
              <div className="resource-browser-grid">
                {cardResources.map((resource) => {
                  const KindIcon = iconForKind(resource.resource_kind);
                  const displayName = resourceDisplayName(resource);
                  const thumbnailReady = shouldRequestThumbnail(resource, failedThumbnailIds);
                  const isDeleting = Boolean(deletingFileIds[resource.file_id]);
                  const secondaryLine = [
                    sourceLabel(resource.source_type),
                    resourceKindLabel(resource.resource_kind),
                    formatBytes(resource.size_bytes),
                  ].join(" · ");
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
                            <Loader2 data-icon="icon" className="animate-spin" />
                          ) : (
                            <Trash2 data-icon="icon" />
                          )}
                        </Button>
                        {thumbnailReady ? (
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
                        ) : (
                          <div className="resource-browser-preview-fallback">
                            <KindIcon className="size-6" aria-hidden="true" />
                            <span>{resourceKindLabel(resource.resource_kind)}</span>
                          </div>
                        )}
                      </div>
                      <div className="resource-browser-meta">
                        <p className="resource-browser-name" title={displayName}>
                          {displayName}
                        </p>
                        <p className="resource-browser-details">{secondaryLine}</p>
                        <p className="resource-browser-date">{formatResourceDate(resource.created_at)}</p>
                        {resource.sync_error ? (
                          <p className="resource-browser-sync-error" title={resource.sync_error}>
                            {resource.sync_error}
                          </p>
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
                          <Eye data-icon="inline-start" />
                          View
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="resource-browser-action-button"
                          onClick={() => onUseInChat(resource)}
                        >
                          <Upload data-icon="inline-start" />
                          Use in chat
                        </Button>
                      </CardFooter>
                    </article>
                  );
                })}
              </div>
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
      {isMobileView ? (
        <Sheet open={mobileFiltersOpen} onOpenChange={setMobileFiltersOpen}>
          <SheetContent
            side="bottom"
            className="resource-browser-filter-sheet rounded-t-[1.75rem] px-4 pb-[calc(1.2rem+env(safe-area-inset-bottom,0px))] pt-3"
          >
            <SheetHeader className="gap-2 text-left">
              <SheetTitle className="text-base">Filter resources</SheetTitle>
              <SheetDescription className="text-sm leading-6 text-muted-foreground">
                Narrow by type and source.
              </SheetDescription>
            </SheetHeader>
            <div className="resource-browser-sheet-section">
              <p className="resource-browser-sheet-label">Type</p>
              <div className="resource-browser-sheet-options">
                {kindFilters.map((item) => (
                  <Button
                    key={item.value}
                    type="button"
                    variant={kindFilter === item.value ? "secondary" : "outline"}
                    size="sm"
                    onClick={() => onKindFilterChange(item.value)}
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
                    onClick={() => onSourceFilterChange(item.value)}
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
                  onKindFilterChange("all");
                  onSourceFilterChange("all");
                }}
              >
                Reset
              </Button>
              <Button type="button" onClick={() => setMobileFiltersOpen(false)}>
                Done
              </Button>
            </div>
          </SheetContent>
        </Sheet>
      ) : null}
    </section>
  );
}
