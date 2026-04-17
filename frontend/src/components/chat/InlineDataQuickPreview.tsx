import { useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Eye,
  FileImage,
  Layers3,
  Table2,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { buildBisqueThumbnailUrl } from "@/lib/bisquePreview";
import { cn } from "@/lib/utils";
import type {
  Hdf5DatasetSummary,
  Hdf5DatasetTablePreviewResponse,
  UploadViewerInfo,
  UploadedFileRecord,
} from "@/types";

import type { ApiClient } from "../../lib/api";

type PreviewBisqueLink = {
  clientViewUrl: string;
  resourceUri?: string | null;
  imageServiceUrl?: string | null;
};

type InlineDataQuickPreviewProps = {
  fileIds: string[];
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, PreviewBisqueLink>;
  apiClient: ApiClient;
  onOpenInViewer: (fileIds: string[]) => void;
};

type PreviewItem = {
  file: UploadedFileRecord;
  bisqueLink?: PreviewBisqueLink;
};

const HDF5_EXTENSIONS = [".h5", ".hdf5", ".hdf"];
const IMAGE_EXTENSIONS = [
  ".png",
  ".jpg",
  ".jpeg",
  ".bmp",
  ".gif",
  ".webp",
  ".tif",
  ".tiff",
  ".ome.tif",
  ".ome.tiff",
  ".nii",
  ".nii.gz",
  ".nrrd",
  ".mha",
  ".mhd",
];

const looksLikeHdf5 = (file: UploadedFileRecord): boolean => {
  const loweredName = String(file.original_name ?? "").trim().toLowerCase();
  if (HDF5_EXTENSIONS.some((suffix) => loweredName.endsWith(suffix))) {
    return true;
  }
  return String(file.content_type ?? "").trim().toLowerCase().includes("hdf5");
};

const looksLikeImage = (file: UploadedFileRecord): boolean => {
  const contentType = String(file.content_type ?? "").trim().toLowerCase();
  if (contentType.startsWith("image/")) {
    return true;
  }
  const loweredName = String(file.original_name ?? "").trim().toLowerCase();
  return IMAGE_EXTENSIONS.some((suffix) => loweredName.endsWith(suffix));
};

const formatShape = (shape: number[] | null | undefined): string => {
  if (!Array.isArray(shape) || shape.length === 0) {
    return "unknown shape";
  }
  return shape.map((value) => `${Math.max(0, Number(value) || 0)}`).join(" × ");
};

function Hdf5PreviewTile({
  file,
  bisqueLink,
  apiClient,
  onOpenInViewer,
}: {
  file: UploadedFileRecord;
  bisqueLink?: PreviewBisqueLink;
  apiClient: ApiClient;
  onOpenInViewer: (fileIds: string[]) => void;
}) {
  const [viewerInfo, setViewerInfo] = useState<UploadViewerInfo | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<Hdf5DatasetSummary | null>(null);
  const [tablePreview, setTablePreview] = useState<Hdf5DatasetTablePreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setViewerInfo(null);
    setDatasetSummary(null);
    setTablePreview(null);
    setError(null);

    void apiClient
      .getUploadViewer(file.file_id)
      .then((response) => {
        if (cancelled) {
          return;
        }
        setViewerInfo(response);
        const defaultDatasetPath =
          response.kind === "hdf5" ? response.hdf5?.default_dataset_path ?? null : null;
        if (!defaultDatasetPath) {
          return;
        }
        void apiClient
          .getHdf5DatasetSummary(file.file_id, defaultDatasetPath)
          .then((summary) => {
            if (cancelled) {
              return;
            }
            setDatasetSummary(summary);
            if (summary.preview_kind === "table" || summary.preview_kind === "series") {
              void apiClient
                .getHdf5DatasetTablePreview(file.file_id, defaultDatasetPath, {
                  offset: 0,
                  limit: 5,
                })
                .then((preview) => {
                  if (cancelled) {
                    return;
                  }
                  setTablePreview(preview);
                })
                .catch((previewError) => {
                  if (cancelled) {
                    return;
                  }
                  const message =
                    previewError instanceof Error ? previewError.message : String(previewError);
                  setError(message);
                });
            }
          })
          .catch((summaryError) => {
            if (cancelled) {
              return;
            }
            const message =
              summaryError instanceof Error ? summaryError.message : String(summaryError);
            setError(message);
          });
      })
      .catch((viewerError) => {
        if (cancelled) {
          return;
        }
        const message = viewerError instanceof Error ? viewerError.message : String(viewerError);
        setError(message);
      });

    return () => {
      cancelled = true;
    };
  }, [apiClient, file.file_id]);

  const summaryBadges = useMemo(() => {
    if (!datasetSummary) {
      return [];
    }
    return [
      `Kind: ${datasetSummary.preview_kind ?? "dataset"}`,
      `Shape: ${formatShape(datasetSummary.shape)}`,
      `Dtype: ${datasetSummary.dtype}`,
    ];
  }, [datasetSummary]);

  return (
    <article className="rounded-xl border border-border/70 bg-background/80 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Table2 className="size-4 text-muted-foreground" />
            <p className="text-sm font-medium">{file.original_name}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">HDF5</Badge>
            {summaryBadges.map((value) => (
              <Badge key={value} variant="secondary">
                {value}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <Button type="button" size="sm" variant="outline" onClick={() => onOpenInViewer([file.file_id])}>
            <Eye className="mr-2 size-3.5" />
            Open viewer
          </Button>
          {bisqueLink?.clientViewUrl ? (
            <Button type="button" size="sm" variant="ghost" asChild>
              <a href={bisqueLink.clientViewUrl} target="_blank" rel="noreferrer">
                <ExternalLink className="mr-2 size-3.5" />
                BisQue
              </a>
            </Button>
          ) : null}
        </div>
      </div>
      {datasetSummary?.dataset_path ? (
        <p className="mt-3 text-xs text-muted-foreground">
          Default dataset: <code>{datasetSummary.dataset_path}</code>
        </p>
      ) : null}
      {viewerInfo?.kind === "hdf5" && (viewerInfo.hdf5?.root_keys?.length ?? 0) > 0 ? (
        <div className="mt-3 text-xs text-muted-foreground">
          <span className="font-medium text-foreground">Top-level keys:</span>{" "}
          {viewerInfo.hdf5?.root_keys?.map((key) => (
            <code key={`${file.file_id}-${key}`} className="mr-1">
              {key}
            </code>
          ))}
        </div>
      ) : null}
      {tablePreview?.rows.length ? (
        <div className="mt-3 overflow-x-auto rounded-lg border border-border/70">
          <table className="min-w-full text-sm">
            <thead className="bg-muted/60">
              <tr>
                {tablePreview.columns.map((column) => (
                  <th key={column.key} className="px-3 py-2 text-left font-medium">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tablePreview.rows.map((row, rowIndex) => (
                <tr key={`${file.file_id}-row-${rowIndex}`} className="border-t border-border/60">
                  {tablePreview.columns.map((column) => (
                    <td key={`${file.file_id}-${rowIndex}-${column.key}`} className="px-3 py-2 align-top">
                      {String(row[column.key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {datasetSummary && !tablePreview?.rows.length ? (
        <p className="mt-3 text-sm text-muted-foreground">
          Quick view ready. Open the scientific viewer for the full HDF5 navigator and dataset tools.
        </p>
      ) : null}
      {error ? <p className="mt-3 text-sm text-amber-700">{error}</p> : null}
    </article>
  );
}

function PreviewTile({
  item,
  apiClient,
  onOpenInViewer,
}: {
  item: PreviewItem;
  apiClient: ApiClient;
  onOpenInViewer: (fileIds: string[]) => void;
}) {
  const { file, bisqueLink } = item;
  const isHdf5 = looksLikeHdf5(file);
  const localPreviewUrl = looksLikeImage(file) ? apiClient.uploadPreviewUrl(file.file_id) : null;
  const bisquePreviewUrl = buildBisqueThumbnailUrl(bisqueLink?.imageServiceUrl);
  const previewCandidates = useMemo(
    () =>
      Array.from(
        new Set(
          [localPreviewUrl, bisquePreviewUrl, bisqueLink?.imageServiceUrl]
            .map((value) => String(value ?? "").trim())
            .filter(Boolean)
        )
      ),
    [bisqueLink?.imageServiceUrl, bisquePreviewUrl, localPreviewUrl]
  );
  const [previewIndex, setPreviewIndex] = useState(() => (previewCandidates.length > 0 ? 0 : -1));

  useEffect(() => {
    setPreviewIndex(previewCandidates.length > 0 ? 0 : -1);
  }, [previewCandidates]);

  const previewUrl =
    previewIndex >= 0 && previewIndex < previewCandidates.length
      ? previewCandidates[previewIndex]
      : null;

  if (isHdf5) {
    return (
      <Hdf5PreviewTile
        file={file}
        bisqueLink={bisqueLink}
        apiClient={apiClient}
        onOpenInViewer={onOpenInViewer}
      />
    );
  }

  return (
    <article className="overflow-hidden rounded-xl border border-border/70 bg-background/80 shadow-sm">
      <div className="aspect-[4/3] bg-muted/40">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={file.original_name}
            loading="lazy"
            className="h-full w-full object-cover"
            onError={() => {
              setPreviewIndex((current) => {
                if (current < 0) {
                  return current;
                }
                const next = current + 1;
                return next < previewCandidates.length ? next : -1;
              });
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <FileImage className="size-8" />
          </div>
        )}
      </div>
      <div className="space-y-3 p-4">
        <div className="space-y-1">
          <p className="text-sm font-medium">{file.original_name}</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">
              {looksLikeImage(file) ? "Image preview" : "Imported file"}
            </Badge>
            {bisqueLink?.resourceUri ? (
              <Badge variant="secondary">BisQue-linked</Badge>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" variant="outline" onClick={() => onOpenInViewer([file.file_id])}>
            <Eye className="mr-2 size-3.5" />
            Open viewer
          </Button>
          {bisqueLink?.clientViewUrl ? (
            <Button type="button" size="sm" variant="ghost" asChild>
              <a href={bisqueLink.clientViewUrl} target="_blank" rel="noreferrer">
                <ExternalLink className="mr-2 size-3.5" />
                BisQue
              </a>
            </Button>
          ) : null}
        </div>
      </div>
    </article>
  );
}

export function InlineDataQuickPreview({
  fileIds,
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  onOpenInViewer,
}: InlineDataQuickPreviewProps) {
  const items = useMemo<PreviewItem[]>(() => {
    const byId = new Map(uploadedFiles.map((file) => [file.file_id, file] as const));
    return fileIds
      .flatMap((fileId) => {
        const file = byId.get(fileId);
        if (!file) {
          return [];
        }
        return [
          {
            file,
            bisqueLink: bisqueLinksByFileId[fileId],
          } satisfies PreviewItem,
        ];
      });
  }, [bisqueLinksByFileId, fileIds, uploadedFiles]);

  const [open, setOpen] = useState(true);

  useEffect(() => {
    setOpen(true);
  }, [fileIds.join("|")]);

  if (items.length === 0) {
    return null;
  }

  return (
    <Card
      className="mt-3 border border-border/70 bg-card/95 shadow-sm supports-[backdrop-filter]:bg-card/90"
      data-testid="chat-inline-quick-preview"
    >
      <Collapsible open={open} onOpenChange={setOpen}>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2 text-base">
                <Layers3 className="size-4" />
                Quick preview
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Resolved {items.length} resource{items.length === 1 ? "" : "s"} from the recent BisQue selection.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <CollapsibleTrigger asChild>
                <Button type="button" size="sm" variant="ghost" className="gap-2">
                  {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
                  {open ? "Collapse" : "Expand"}
                </Button>
              </CollapsibleTrigger>
            </div>
          </div>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className={cn("grid gap-4 pb-5", items.length === 1 ? "grid-cols-1" : "grid-cols-1 xl:grid-cols-2")}>
            {items.map((item) => (
              <PreviewTile
                key={item.file.file_id}
                item={item}
                apiClient={apiClient}
                onOpenInViewer={onOpenInViewer}
              />
            ))}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
