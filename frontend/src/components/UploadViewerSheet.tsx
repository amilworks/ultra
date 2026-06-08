import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ExternalLink, FileText, Layers3, RefreshCw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import type { ApiClient } from "../lib/api";
import type { Hdf5DatasetSummary, UploadViewerInfo, UploadedFileRecord } from "../types";

import {
  buildInitialViewerIndices,
  clampViewerIndex,
  type ViewerIndices,
  type ViewerSurface,
} from "./viewer/shared";

const LazyImageViewerShell = lazy(() =>
  import("./viewer/ImageViewerShell").then((module) => ({
    default: module.ImageViewerShell,
  }))
);

const LazyHdf5ViewerShell = lazy(() =>
  import("./viewer/hdf5/Hdf5Overview").then((module) => ({
    default: module.Hdf5ViewerShell,
  }))
);

function ViewerPanelFallback() {
  return (
    <div className="viewer-stage viewer-stage-loading min-h-[28rem] items-center justify-center rounded-xl border border-border/70 bg-background text-sm font-medium text-muted-foreground">
      Loading viewer...
    </div>
  );
}

export type UploadViewerWorkspaceProps = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
  onUseHdf5DatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
  active?: boolean;
  className?: string;
};

type UploadViewerSheetProps = Omit<UploadViewerWorkspaceProps, "active" | "className"> & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export type BisqueViewerLink = {
  clientViewUrl: string;
  resourceUri?: string | null;
  imageServiceUrl?: string | null;
  inputUrl?: string;
};

type ViewerDisplayState = NonNullable<UploadViewerInfo["display_defaults"]>;

type ViewerSurfaceMode = "native" | "bisque";

function useDebouncedNumber(value: number, delayMs = 120): number {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => window.clearTimeout(timeoutId);
  }, [delayMs, value]);

  return debouncedValue;
}

const normalizeSurface = (viewerInfo: UploadViewerInfo | null, current?: string | null): ViewerSurface => {
  const fallback = (viewerInfo?.viewer?.default_surface ?? "2d") as ViewerSurface;
  const available = new Set((viewerInfo?.viewer?.available_surfaces ?? [fallback]).map((value) => String(value)));
  if (current && available.has(current)) {
    return current as ViewerSurface;
  }
  if (available.has(fallback)) {
    return fallback;
  }
  return "2d";
};

const buildViewerDisplayState = (
  viewerInfo: UploadViewerInfo,
  override?: Partial<ViewerDisplayState>
): ViewerDisplayState => ({
  enhancement: viewerInfo.display_defaults?.enhancement ?? "d",
  negative: Boolean(viewerInfo.display_defaults?.negative ?? false),
  rotate: Number(viewerInfo.display_defaults?.rotate ?? 0),
  fusion_method: viewerInfo.display_defaults?.fusion_method ?? "m",
  channel_mode: viewerInfo.display_defaults?.channel_mode ?? "composite",
  channels: Array.isArray(viewerInfo.display_defaults?.channels)
    ? viewerInfo.display_defaults.channels
    : [0, 1, 2],
  channel_colors: Array.isArray(viewerInfo.display_defaults?.channel_colors)
    ? viewerInfo.display_defaults.channel_colors
    : [],
  time_index: viewerInfo.display_defaults?.time_index ?? 0,
  z_index: viewerInfo.display_defaults?.z_index ?? 0,
  scalar_colormap: viewerInfo.display_defaults?.scalar_colormap ?? "grayscale",
  volume_signal_floor: viewerInfo.display_defaults?.volume_signal_floor ?? undefined,
  volume_density: viewerInfo.display_defaults?.volume_density ?? undefined,
  volume_lighting: viewerInfo.display_defaults?.volume_lighting ?? false,
  volume_lighting_strength: viewerInfo.display_defaults?.volume_lighting_strength ?? 0.65,
  volume_channel: viewerInfo.display_defaults?.volume_channel ?? viewerInfo.selected_indices.C ?? 0,
  volume_view_preset: viewerInfo.display_defaults?.volume_view_preset ?? undefined,
  volume_camera_mode: viewerInfo.display_defaults?.volume_camera_mode ?? undefined,
  volume_clip_min: {
    x: Number(viewerInfo.display_defaults?.volume_clip_min?.x ?? 0),
    y: Number(viewerInfo.display_defaults?.volume_clip_min?.y ?? 0),
    z: Number(viewerInfo.display_defaults?.volume_clip_min?.z ?? 0),
  },
  volume_clip_max: {
    x: Number(viewerInfo.display_defaults?.volume_clip_max?.x ?? 1),
    y: Number(viewerInfo.display_defaults?.volume_clip_max?.y ?? 1),
    z: Number(viewerInfo.display_defaults?.volume_clip_max?.z ?? 1),
  },
  ...override,
});

const formatStageNumber = (value: number): string =>
  Number.isFinite(value) ? Math.max(0, Math.floor(value)).toLocaleString() : "0";

const formatStageLabel = (value: string | null | undefined): string => {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    return "";
  }
  return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}`;
};

const buildViewerStageMeta = (viewerInfo: UploadViewerInfo): string => {
  const size = viewerInfo.is_volume
    ? `${formatStageNumber(viewerInfo.axis_sizes.X)} x ${formatStageNumber(
        viewerInfo.axis_sizes.Y
      )} x ${formatStageNumber(viewerInfo.axis_sizes.Z)} voxels`
    : `${formatStageNumber(viewerInfo.axis_sizes.X)} x ${formatStageNumber(viewerInfo.axis_sizes.Y)} px`;
  return [
    formatStageLabel(viewerInfo.modality),
    viewerInfo.is_volume ? "Volume" : "Image",
    size,
    viewerInfo.dims_order ? `Axes ${viewerInfo.dims_order}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
};

const isPdfUpload = (file: UploadedFileRecord | null | undefined): boolean => {
  const contentType = String(file?.content_type ?? "")
    .split(";")[0]
    .trim()
    .toLowerCase();
  const originalName = String(file?.original_name ?? "")
    .trim()
    .toLowerCase();

  return (
    contentType === "application/pdf" ||
    contentType === "application/x-pdf" ||
    contentType.endsWith("+pdf") ||
    originalName.endsWith(".pdf")
  );
};

const formatDocumentFileSize = (bytes: number | null | undefined): string | null => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) {
    return null;
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1000 && unitIndex < units.length - 1) {
    value /= 1000;
    unitIndex += 1;
  }

  if (unitIndex === 0) {
    return `${Math.round(value)} ${units[unitIndex]}`;
  }

  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
};

const buildPdfStageMeta = (file: UploadedFileRecord): string =>
  ["Document", "PDF", formatDocumentFileSize(file.size_bytes)].filter(Boolean).join(" · ");

function PdfViewerPanel({ file, apiClient }: { file: UploadedFileRecord; apiClient: ApiClient }) {
  const pdfUrl = apiClient.uploadDisplayUrl(file.file_id);

  return (
    <section className="viewer-pdf-reader" aria-label="PDF reader" role="region">
      <div className="viewer-stage-header viewer-pdf-header">
        <div className="viewer-stage-heading">
          <div className="viewer-pdf-title-row">
            <FileText aria-hidden="true" />
            <div className="viewer-stage-title viewer-pdf-title">{file.original_name}</div>
            <Badge variant="secondary" className="viewer-pdf-badge">
              PDF
            </Badge>
          </div>
          <div className="viewer-stage-meta">{buildPdfStageMeta(file)}</div>
        </div>
        <div className="viewer-pdf-actions">
          <Button asChild type="button" size="sm" variant="outline">
            <a href={pdfUrl} target="_blank" rel="noreferrer">
              <ExternalLink data-icon="inline-start" />
              Open in tab
            </a>
          </Button>
        </div>
      </div>

      <div className="viewer-pdf-shell">
        <iframe
          src={pdfUrl}
          title={`PDF viewer for ${file.original_name}`}
          className="viewer-pdf-frame"
          loading="lazy"
        />
      </div>
    </section>
  );
}

export function UploadViewerWorkspace({
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  onUseHdf5DatasetInChat,
  active = true,
  className,
}: UploadViewerWorkspaceProps) {
  const [requestedSelectedFileId, setRequestedSelectedFileId] = useState<string | null>(null);
  const [viewerInfoById, setViewerInfoById] = useState<Record<string, UploadViewerInfo>>({});
  const [viewerErrorById, setViewerErrorById] = useState<Record<string, string>>({});
  const [viewerIndicesById, setViewerIndicesById] = useState<Record<string, ViewerIndices>>({});
  const [viewerSurfaceById, setViewerSurfaceById] = useState<Record<string, ViewerSurface>>({});
  const [viewerDisplayById, setViewerDisplayById] = useState<Record<string, ViewerDisplayState>>({});
  const [viewerSurfaceModeById, setViewerSurfaceModeById] = useState<Record<string, ViewerSurfaceMode>>({});
  const [viewerHdf5SelectedDatasetById, setViewerHdf5SelectedDatasetById] = useState<
    Record<string, string | null>
  >({});
  const [viewerHdf5DatasetSummaryByKey, setViewerHdf5DatasetSummaryByKey] = useState<
    Record<string, Hdf5DatasetSummary>
  >({});
  const [loadingFileId, setLoadingFileId] = useState<string | null>(null);
  const loadingFileIdRef = useRef<string | null>(null);
  const selectedFileId = useMemo(() => {
    if (uploadedFiles.length === 0) {
      return null;
    }
    if (
      requestedSelectedFileId &&
      uploadedFiles.some((file) => file.file_id === requestedSelectedFileId)
    ) {
      return requestedSelectedFileId;
    }
    return uploadedFiles[0].file_id;
  }, [requestedSelectedFileId, uploadedFiles]);

  const selectedSurfaceMode: ViewerSurfaceMode = selectedFileId
    ? viewerSurfaceModeById[selectedFileId] ?? "native"
    : "native";
  const selectedBisqueLink = selectedFileId ? bisqueLinksByFileId[selectedFileId] ?? null : null;
  const showBisqueFrame = Boolean(
    selectedFileId && selectedSurfaceMode === "bisque" && selectedBisqueLink?.clientViewUrl
  );
  const selectedFile = selectedFileId
    ? uploadedFiles.find((file) => file.file_id === selectedFileId) ?? null
    : null;
  const selectedFileIsPdf = isPdfUpload(selectedFile);
  const selectedViewerInfo = selectedFileId ? viewerInfoById[selectedFileId] ?? null : null;
  const selectedViewerError = selectedFileId ? viewerErrorById[selectedFileId] ?? null : null;

  useEffect(() => {
    if (
      !active ||
      !selectedFileId ||
      selectedFileIsPdf ||
      selectedViewerInfo ||
      selectedViewerError ||
      loadingFileIdRef.current === selectedFileId
    ) {
      return;
    }
    let cancelled = false;
    const activeFileId = selectedFileId;
    loadingFileIdRef.current = activeFileId;
    setLoadingFileId(activeFileId);

    void apiClient
      .getUploadViewer(activeFileId)
      .then((viewerInfo) => {
        if (cancelled) {
          return;
        }
        setViewerInfoById((previous) => ({ ...previous, [activeFileId]: viewerInfo }));
        setViewerIndicesById((previous) => {
          if (previous[activeFileId]) {
            return previous;
          }
          return {
            ...previous,
            [activeFileId]: buildInitialViewerIndices(viewerInfo),
          };
        });
        setViewerSurfaceById((previous) => {
          if (previous[activeFileId]) {
            return previous;
          }
          return {
            ...previous,
            [activeFileId]: normalizeSurface(viewerInfo),
          };
        });
        setViewerDisplayById((previous) => {
          if (previous[activeFileId]) {
            return previous;
          }
          return {
            ...previous,
            [activeFileId]: buildViewerDisplayState(viewerInfo),
          };
        });
        if (viewerInfo.kind === "hdf5") {
          setViewerHdf5SelectedDatasetById((previous) => {
            if (previous[activeFileId] !== undefined) {
              return previous;
            }
            return {
              ...previous,
              [activeFileId]: viewerInfo.hdf5?.default_dataset_path ?? null,
            };
          });
        }
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        setViewerErrorById((previous) => ({ ...previous, [activeFileId]: message }));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        if (loadingFileIdRef.current === activeFileId) {
          loadingFileIdRef.current = null;
        }
        setLoadingFileId((current) => (current === activeFileId ? null : current));
      });

    return () => {
      cancelled = true;
      if (loadingFileIdRef.current === activeFileId) {
        loadingFileIdRef.current = null;
      }
      setLoadingFileId((current) => (current === activeFileId ? null : current));
    };
  }, [active, apiClient, selectedFileId, selectedFileIsPdf, selectedViewerError, selectedViewerInfo]);

  const selectedIndices = selectedFileId ? viewerIndicesById[selectedFileId] ?? null : null;

  const xAxisSize = Math.max(1, Number(selectedViewerInfo?.axis_sizes.X ?? 1));
  const yAxisSize = Math.max(1, Number(selectedViewerInfo?.axis_sizes.Y ?? 1));
  const zAxisSize = Math.max(1, Number(selectedViewerInfo?.axis_sizes.Z ?? 1));
  const tAxisSize = Math.max(1, Number(selectedViewerInfo?.axis_sizes.T ?? 1));
  const clampedIndices = useMemo<ViewerIndices>(() => {
    if (!selectedIndices) {
      return { x: 0, y: 0, z: 0, t: 0 };
    }
    return {
      x: clampViewerIndex(selectedIndices.x, xAxisSize),
      y: clampViewerIndex(selectedIndices.y, yAxisSize),
      z: clampViewerIndex(selectedIndices.z, zAxisSize),
      t: clampViewerIndex(selectedIndices.t, tAxisSize),
    };
  }, [selectedIndices, tAxisSize, xAxisSize, yAxisSize, zAxisSize]);
  const debouncedX = useDebouncedNumber(clampedIndices.x);
  const debouncedY = useDebouncedNumber(clampedIndices.y);
  const debouncedZ = useDebouncedNumber(clampedIndices.z);
  const debouncedT = useDebouncedNumber(clampedIndices.t);

  const selectedSurface = selectedFileId
    ? normalizeSurface(selectedViewerInfo, viewerSurfaceById[selectedFileId] ?? null)
    : "2d";
  const selectedDisplayState = selectedFileId ? viewerDisplayById[selectedFileId] ?? null : null;
  const isHdf5Viewer = selectedViewerInfo?.kind === "hdf5";
  const selectedDatasetPath = selectedFileId ? viewerHdf5SelectedDatasetById[selectedFileId] ?? null : null;
  const selectedDatasetSummaryKey =
    selectedFileId && selectedDatasetPath ? `${selectedFileId}:${selectedDatasetPath}` : null;
  const selectedDatasetSummary = selectedDatasetSummaryKey
    ? viewerHdf5DatasetSummaryByKey[selectedDatasetSummaryKey] ?? null
    : null;
  const showFileStrip = uploadedFiles.length !== 1;

  const retrySelected = (): void => {
    if (!selectedFileId) {
      return;
    }
    setViewerErrorById((previous) => {
      const next = { ...previous };
      delete next[selectedFileId];
      return next;
    });
    setViewerInfoById((previous) => {
      const next = { ...previous };
      delete next[selectedFileId];
      return next;
    });
    setViewerDisplayById((previous) => {
      const next = { ...previous };
      delete next[selectedFileId];
      return next;
    });
  };

  const setSelectedIndex = (axis: keyof ViewerIndices, value: number): void => {
    if (!selectedFileId || !selectedViewerInfo) {
      return;
    }
    setViewerIndicesById((previous) => ({
      ...previous,
      [selectedFileId]: {
        ...(previous[selectedFileId] ?? buildInitialViewerIndices(selectedViewerInfo)),
        [axis]: value,
      },
    }));
  };

  const updateSelectedDisplay = (patch: Partial<ViewerDisplayState>): void => {
    if (!selectedFileId || !selectedViewerInfo) {
      return;
    }
    setViewerDisplayById((previous) => ({
      ...previous,
      [selectedFileId]: buildViewerDisplayState(selectedViewerInfo, {
        ...(previous[selectedFileId] ?? {}),
        ...patch,
      }),
    }));
  };

  const cacheDatasetSummary = useCallback((summary: Hdf5DatasetSummary): void => {
    const cacheKey = `${summary.file_id}:${summary.dataset_path}`;
    setViewerHdf5DatasetSummaryByKey((previous) => ({ ...previous, [cacheKey]: summary }));
  }, []);

  return (
    <section
      className={cn(
        "viewer-workspace",
        !showFileStrip && "viewer-workspace-single-file",
        selectedFileIsPdf && "viewer-workspace-document",
        selectedViewerInfo && !isHdf5Viewer && `viewer-workspace-surface-${selectedSurface}`,
        className
      )}
      aria-label="Scientific viewer workspace"
    >
      <div className="viewer-body">
        {showFileStrip ? (
          <aside className="viewer-file-strip">
            {uploadedFiles.length === 0 ? (
              <p className="viewer-empty">No uploaded files yet.</p>
            ) : (
              uploadedFiles.map((file) => {
                const isActive = file.file_id === selectedFileId;
                return (
                  <Button
                    key={file.file_id}
                    type="button"
                    variant={isActive ? "secondary" : "outline"}
                    size="sm"
                    className={cn("viewer-file-chip", isActive && "viewer-file-chip-active")}
                    onClick={() => setRequestedSelectedFileId(file.file_id)}
                  >
                    {file.original_name}
                  </Button>
                );
              })
            )}
          </aside>
        ) : null}

        <section className="viewer-stage">
          {!selectedFileId ? (
            <div className="viewer-empty">Select a file to start.</div>
          ) : selectedFileIsPdf && selectedFile ? (
            <PdfViewerPanel file={selectedFile} apiClient={apiClient} />
          ) : showBisqueFrame ? (
            <>
              <div className="viewer-stage-header">
                <div className="viewer-stage-heading">
                  <div className="viewer-stage-title">{selectedFile?.original_name ?? "BisQue Viewer"}</div>
                  <div className="viewer-stage-meta">BisQue fallback surface</div>
                </div>
              </div>
              <div className="viewer-mode-toggle">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    selectedFileId &&
                    setViewerSurfaceModeById((previous) => ({
                      ...previous,
                      [selectedFileId]: "native",
                    }))
                  }
                >
                  Native viewer
                </Button>
                <Button type="button" size="sm" variant="secondary">
                  BisQue iframe
                </Button>
              </div>
              <div className="viewer-canvas-shell">
                <iframe
                  src={selectedBisqueLink?.clientViewUrl}
                  title={`BisQue resource ${selectedFile?.original_name ?? selectedFileId}`}
                  className="viewer-bisque-iframe"
                  loading="lazy"
                  referrerPolicy="no-referrer"
                />
              </div>
              <div className="viewer-bisque-footer">
                <a
                  href={selectedBisqueLink?.clientViewUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="viewer-bisque-open-link"
                >
                  <ExternalLink className="size-3.5" />
                  Open in BisQue tab
                </a>
                {selectedBisqueLink?.clientViewUrl ? (
                  <p className="viewer-meta-line">view={selectedBisqueLink.clientViewUrl}</p>
                ) : null}
              </div>
            </>
          ) : loadingFileId === selectedFileId && !selectedViewerInfo ? (
            <div className="viewer-empty">Loading viewer metadata…</div>
          ) : selectedViewerError ? (
            <div className="viewer-error">
              <p>{selectedViewerError}</p>
              <div className="flex items-center gap-2">
                <Button type="button" variant="outline" size="sm" onClick={retrySelected}>
                  <RefreshCw className="mr-2 size-3.5" />
                  Retry
                </Button>
                {selectedBisqueLink?.clientViewUrl ? (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() =>
                      selectedFileId &&
                      setViewerSurfaceModeById((previous) => ({
                        ...previous,
                        [selectedFileId]: "bisque",
                      }))
                    }
                  >
                    Switch to BisQue iframe
                  </Button>
                ) : null}
              </div>
            </div>
          ) : selectedViewerInfo ? (
            <>
              {!isHdf5Viewer ? (
                <div className="viewer-stage-header">
                  <div className="viewer-stage-heading">
                    <div className="viewer-stage-title">{selectedViewerInfo.original_name}</div>
                    <div className="viewer-stage-meta">{buildViewerStageMeta(selectedViewerInfo)}</div>
                  </div>
                </div>
              ) : null}

              {selectedBisqueLink?.clientViewUrl ? (
                <div className="viewer-mode-toggle">
                  <Button type="button" size="sm" variant="secondary">
                    Native viewer
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      selectedFileId &&
                      setViewerSurfaceModeById((previous) => ({
                        ...previous,
                        [selectedFileId]: "bisque",
                      }))
                    }
                  >
                    BisQue iframe
                  </Button>
                </div>
              ) : null}

              <Suspense fallback={<ViewerPanelFallback />}>
                {isHdf5Viewer ? (
	                  <LazyHdf5ViewerShell
	                    key={selectedViewerInfo.file_id}
	                    viewerInfo={selectedViewerInfo}
                    apiClient={apiClient}
                    selectedDatasetPath={selectedDatasetPath}
                    onSelectedDatasetPathChange={(path) =>
                      selectedFileId &&
                      setViewerHdf5SelectedDatasetById((previous) => ({
                        ...previous,
                        [selectedFileId]: path,
                      }))
                    }
                    selectedDatasetSummary={selectedDatasetSummary}
                    cacheDatasetSummary={cacheDatasetSummary}
                    onUseDatasetInChat={onUseHdf5DatasetInChat}
                  />
                ) : (
                  <LazyImageViewerShell
                    viewerInfo={selectedViewerInfo}
                    apiClient={apiClient}
                    selectedSurface={selectedSurface}
                    onSurfaceChange={(surface) =>
                      selectedFileId &&
                      setViewerSurfaceById((previous) => ({
                        ...previous,
                        [selectedFileId]: normalizeSurface(selectedViewerInfo, surface),
                      }))
                    }
                    selectedDisplayState={selectedDisplayState}
                    updateSelectedDisplay={updateSelectedDisplay}
                    clampedIndices={clampedIndices}
                    debouncedX={debouncedX}
                    debouncedY={debouncedY}
                    debouncedZ={debouncedZ}
                    debouncedT={debouncedT}
                    xAxisSize={xAxisSize}
                    yAxisSize={yAxisSize}
                    zAxisSize={zAxisSize}
                    tAxisSize={tAxisSize}
                    setSelectedIndex={setSelectedIndex}
                    selectedCaption=""
                    captionLoading={false}
                  />
                )}
              </Suspense>
            </>
          ) : null}
        </section>
      </div>
    </section>
  );
}

export function UploadViewerSheet({
  open,
  onOpenChange,
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  onUseHdf5DatasetInChat,
}: UploadViewerSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="viewer-sheet w-full p-0 sm:max-w-none">
        <SheetHeader className="viewer-sheet-header border-b px-4 pb-3">
          <SheetTitle className="viewer-sheet-title flex items-center gap-2">
            <Layers3 className="size-4" />
            Scientific Viewer
          </SheetTitle>
          <SheetDescription className="viewer-sheet-description">
            Native scientific imaging workspace for 2D planes, orthogonal slices, metadata, and Three.js volume
            rendering.
          </SheetDescription>
        </SheetHeader>

        <UploadViewerWorkspace
          active={open}
          uploadedFiles={uploadedFiles}
          bisqueLinksByFileId={bisqueLinksByFileId}
          apiClient={apiClient}
          onUseHdf5DatasetInChat={onUseHdf5DatasetInChat}
          className="viewer-workspace-sheet"
        />
      </SheetContent>
    </Sheet>
  );
}
