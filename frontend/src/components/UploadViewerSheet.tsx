import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Download, ExternalLink, FileText, FileVideo, FileWarning, Layers3, RefreshCw } from "lucide-react";

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
import { classifyTextResource } from "@/lib/textFormat";
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

const LazyTextResourceViewer = lazy(() =>
  import("./viewer/TextResourceViewer").then((module) => ({
    default: module.TextResourceViewer,
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
};

type ViewerDisplayState = NonNullable<UploadViewerInfo["display_defaults"]>;

type ViewerSurfaceMode = "native" | "bisque";

function useDebouncedNumber(value: number, delayMs = 70): number {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => window.clearTimeout(timeoutId);
  }, [delayMs, value]);

  return debouncedValue;
}

export const normalizeSurface = (viewerInfo: UploadViewerInfo | null, current?: string | null): ViewerSurface => {
  // Volumes (NIfTI/DICOM medical + microscopy z-stacks) open on the 2D tab by default
  // — it's the fastest and most measurement-accurate surface. This only sets the
  // FIRST-LOAD surface (applied when `current` is absent); the user can still switch to
  // Slice Views/Volume, and that choice is preserved per file via `current`.
  const fallback = ((viewerInfo?.is_volume ? "2d" : viewerInfo?.viewer?.default_surface) ?? "2d") as ViewerSurface;
  const available = new Set((viewerInfo?.viewer?.available_surfaces ?? [fallback]).map((value) => String(value)));
  if (current && available.has(current)) {
    return current as ViewerSurface;
  }
  if (available.has(fallback)) {
    return fallback;
  }
  return "2d";
};

const CT_BRAIN_WINDOW_ENHANCEMENT = "hounsfield:40.000:80.000";

// Hounsfield/CT volumes include air near -1000 HU plus tissue and bone well above
// zero. MRI and most other scalars start near zero, so this gate keeps the brain
// window from being applied where Hounsfield units do not exist.
const looksLikeHounsfieldCt = (viewerInfo: UploadViewerInfo): boolean => {
  if (String(viewerInfo.modality ?? "").trim().toLowerCase() !== "medical") {
    return false;
  }
  const min = Number(viewerInfo.metadata.array_min ?? viewerInfo.metadata.intensity_stats?.min ?? Number.NaN);
  const max = Number(viewerInfo.metadata.array_max ?? viewerInfo.metadata.intensity_stats?.max ?? Number.NaN);
  return Number.isFinite(min) && Number.isFinite(max) && min <= -200 && max >= 200;
};

// A real DICOM acquisition window (carried on the scan) is honoured; a backend
// heuristic default window is not — that is what the brain window replaces.
const hasDicomWindow = (viewerInfo: UploadViewerInfo): boolean => {
  const center = Number(viewerInfo.metadata.dicom?.wnd_center ?? Number.NaN);
  const width = Number(viewerInfo.metadata.dicom?.wnd_width ?? Number.NaN);
  return Number.isFinite(center) && Number.isFinite(width) && width > 0;
};

// Brain CT volumes (DICOM/NIfTI) open on the brain window so soft tissue and the
// ventricles are visible immediately, instead of a generic wide default. A genuine
// DICOM window from the acquisition is respected.
export const resolveDefaultEnhancement = (viewerInfo: UploadViewerInfo): string => {
  if (looksLikeHounsfieldCt(viewerInfo) && !hasDicomWindow(viewerInfo)) {
    return CT_BRAIN_WINDOW_ENHANCEMENT;
  }
  return viewerInfo.display_defaults?.enhancement ?? "d";
};

const buildViewerDisplayState = (
  viewerInfo: UploadViewerInfo,
  override?: Partial<ViewerDisplayState>
): ViewerDisplayState => ({
  enhancement: resolveDefaultEnhancement(viewerInfo),
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

const isVideoUpload = (file: UploadedFileRecord | null | undefined): boolean => {
  const contentType = String(file?.content_type ?? "").split(";")[0].trim().toLowerCase();
  if (contentType.startsWith("video/")) {
    return true;
  }
  const originalName = String(file?.original_name ?? "").trim().toLowerCase();
  return /\.(mp4|mov|avi|mkv|webm|m4v|mpg|mpeg|wmv|flv|ogv)$/.test(originalName);
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

const buildVideoStageMeta = (file: UploadedFileRecord): string =>
  ["Video", formatDocumentFileSize(file.size_bytes)].filter(Boolean).join(" · ");

function VideoViewerPanel({ file, apiClient }: { file: UploadedFileRecord; apiClient: ApiClient }) {
  // Raw file streamed via HTTP range (http.ServeFile) — the player only buffers
  // what is played, so a long video never downloads in full. The poster is the
  // server ffmpeg frame.
  const videoUrl = apiClient.uploadDisplayUrl(file.file_id);
  const posterUrl = apiClient.resourceThumbnailUrl(file.file_id);

  return (
    <section className="viewer-video-reader" aria-label="Video player" role="region">
      <div className="viewer-stage-header viewer-pdf-header">
        <div className="viewer-stage-heading">
          <div className="viewer-pdf-title-row">
            <FileVideo aria-hidden="true" />
            <div className="viewer-stage-title viewer-pdf-title">{file.original_name}</div>
            <Badge variant="secondary" className="viewer-pdf-badge">
              Video
            </Badge>
          </div>
          <div className="viewer-stage-meta">{buildVideoStageMeta(file)}</div>
        </div>
        <div className="viewer-pdf-actions">
          <Button asChild type="button" size="sm" variant="outline">
            <a href={videoUrl} target="_blank" rel="noreferrer">
              <ExternalLink data-icon="inline-start" />
              Open in tab
            </a>
          </Button>
        </div>
      </div>

      <div className="viewer-video-shell">
        <video
          className="viewer-video-frame"
          src={videoUrl}
          poster={posterUrl}
          controls
          playsInline
          preload="metadata"
        />
      </div>
    </section>
  );
}

/**
 * Calm fallback shown when the image engine recognized the file but cannot decode it
 * (kind:"unsupported" / decodable:false — e.g. a Leica .lif). Replaces the broken 1×1
 * canvas + endless "Loading…" with a clear message and a download-original action.
 */
function UnsupportedFormatPanel({
  info,
  apiClient,
}: {
  info: UploadViewerInfo;
  apiClient: ApiClient;
}) {
  const downloadUrl = apiClient.resourceDownloadUrl(info.file_id);
  const message =
    info.message ?? "This file format can't be previewed by the image engine yet.";
  return (
    <section className="viewer-stage-document">
      <div className="viewer-empty flex flex-col items-center gap-3 text-center">
        <FileWarning className="size-8 opacity-60" aria-hidden />
        <div className="text-base font-medium">Preview unavailable</div>
        <p className="max-w-md text-sm opacity-70">{message}</p>
        <p className="text-xs opacity-50">{info.original_name}</p>
        <Button asChild size="sm" variant="secondary">
          <a href={downloadUrl} download>
            <Download className="mr-1.5 size-3.5" />
            Download original
          </a>
        </Button>
      </div>
    </section>
  );
}

export function UploadViewerWorkspace({
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
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
  const selectedFileIsVideo = isVideoUpload(selectedFile);
  // Text/data files (CSV/JSON/YAML/XML/Markdown/text) route to the dedicated text
  // viewer and must short-circuit the image-engine probe, exactly like PDF/video.
  const selectedTextKind = selectedFileIsPdf || selectedFileIsVideo ? null : classifyTextResource(selectedFile);
  const selectedFileIsText = selectedTextKind !== null;
  const selectedViewerInfo = selectedFileId ? viewerInfoById[selectedFileId] ?? null : null;
  const selectedViewerError = selectedFileId ? viewerErrorById[selectedFileId] ?? null : null;

  useEffect(() => {
    if (
      !active ||
      !selectedFileId ||
      selectedFileIsPdf ||
      selectedFileIsVideo ||
      selectedFileIsText ||
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
  }, [active, apiClient, selectedFileId, selectedFileIsPdf, selectedFileIsVideo, selectedFileIsText, selectedViewerError, selectedViewerInfo]);

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
        (selectedFileIsPdf || selectedFileIsVideo || selectedFileIsText) && "viewer-workspace-document",
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
          ) : selectedFileIsVideo && selectedFile ? (
            <VideoViewerPanel file={selectedFile} apiClient={apiClient} />
          ) : selectedFileIsText && selectedTextKind && selectedFile ? (
            <Suspense fallback={<ViewerPanelFallback />}>
              {/* key by file id so switching files fully remounts (resets view/state). */}
              <LazyTextResourceViewer
                key={selectedFile.file_id}
                file={selectedFile}
                kind={selectedTextKind}
                apiClient={apiClient}
              />
            </Suspense>
          ) : selectedViewerInfo?.decodable === false ? (
            <UnsupportedFormatPanel info={selectedViewerInfo} apiClient={apiClient} />
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
}: UploadViewerSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="viewer-sheet w-full p-0 sm:max-w-none">
        <SheetHeader className="viewer-sheet-header border-b px-4 pb-3">
          <SheetTitle className="viewer-sheet-title flex items-center gap-2">
            <Layers3 className="size-4" />
            Lens
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
          className="viewer-workspace-sheet"
        />
      </SheetContent>
    </Sheet>
  );
}
