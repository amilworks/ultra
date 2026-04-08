import { useCallback, useEffect, useMemo, useState } from "react";
import { ExternalLink, Layers3, RefreshCw } from "lucide-react";

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

import { ImageViewerShell } from "./viewer/ImageViewerShell";
import { Hdf5ViewerShell } from "./viewer/hdf5/Hdf5Overview";
import {
  buildInitialViewerIndices,
  clampViewerIndex,
  type ViewerIndices,
  type ViewerSurface,
} from "./viewer/shared";

type UploadViewerSheetProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
  onUseHdf5DatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
};

type BisqueViewerLink = {
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
  volume_channel: viewerInfo.display_defaults?.volume_channel ?? viewerInfo.selected_indices.C ?? 0,
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

export function UploadViewerSheet({
  open,
  onOpenChange,
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  onUseHdf5DatasetInChat,
}: UploadViewerSheetProps) {
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [viewerInfoById, setViewerInfoById] = useState<Record<string, UploadViewerInfo>>({});
  const [viewerErrorById, setViewerErrorById] = useState<Record<string, string>>({});
  const [viewerIndicesById, setViewerIndicesById] = useState<Record<string, ViewerIndices>>({});
  const [viewerSurfaceById, setViewerSurfaceById] = useState<Record<string, ViewerSurface>>({});
  const [viewerDisplayById, setViewerDisplayById] = useState<Record<string, ViewerDisplayState>>({});
  const [viewerSurfaceModeById, setViewerSurfaceModeById] = useState<Record<string, ViewerSurfaceMode>>({});
  const [viewerCaptionById, setViewerCaptionById] = useState<Record<string, string>>({});
  const [captionLoadingById, setCaptionLoadingById] = useState<Record<string, boolean>>({});
  const [viewerHdf5SelectedDatasetById, setViewerHdf5SelectedDatasetById] = useState<
    Record<string, string | null>
  >({});
  const [viewerHdf5DatasetSummaryByKey, setViewerHdf5DatasetSummaryByKey] = useState<
    Record<string, Hdf5DatasetSummary>
  >({});
  const [loadingFileId, setLoadingFileId] = useState<string | null>(null);

  useEffect(() => {
    const validIds = new Set(uploadedFiles.map((file) => file.file_id));

    if (uploadedFiles.length === 0) {
      setSelectedFileId(null);
      return;
    }
    if (!selectedFileId || !validIds.has(selectedFileId)) {
      setSelectedFileId(uploadedFiles[0].file_id);
    }

    const retainKeys = <T extends Record<string, unknown>>(record: T): T => {
      return Object.fromEntries(Object.entries(record).filter(([fileId]) => validIds.has(fileId))) as T;
    };
    const retainDatasetSummaries = (record: Record<string, Hdf5DatasetSummary>) =>
      Object.fromEntries(
        Object.entries(record).filter(([cacheKey]) => validIds.has(cacheKey.split(":", 1)[0] ?? ""))
      );

    setViewerInfoById((previous) => retainKeys(previous));
    setViewerErrorById((previous) => retainKeys(previous));
    setViewerIndicesById((previous) => retainKeys(previous));
    setViewerSurfaceById((previous) => retainKeys(previous));
    setViewerDisplayById((previous) => retainKeys(previous));
    setViewerSurfaceModeById((previous) => retainKeys(previous));
    setViewerCaptionById((previous) => retainKeys(previous));
    setCaptionLoadingById((previous) => retainKeys(previous));
    setViewerHdf5SelectedDatasetById((previous) => retainKeys(previous));
    setViewerHdf5DatasetSummaryByKey((previous) => retainDatasetSummaries(previous));
  }, [selectedFileId, uploadedFiles]);

  useEffect(() => {
    if (!selectedFileId) {
      return;
    }
    setViewerSurfaceModeById((previous) => {
      if (previous[selectedFileId]) {
        return previous;
      }
      return { ...previous, [selectedFileId]: "native" };
    });
  }, [selectedFileId]);

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
  const selectedViewerInfo = selectedFileId ? viewerInfoById[selectedFileId] ?? null : null;
  const selectedViewerError = selectedFileId ? viewerErrorById[selectedFileId] ?? null : null;
  const selectedCaption = selectedFileId ? viewerCaptionById[selectedFileId] : "";
  const selectedCaptionLoading = selectedFileId ? Boolean(captionLoadingById[selectedFileId]) : false;

  useEffect(() => {
    if (!open || !selectedFileId || selectedViewerInfo || selectedViewerError || loadingFileId === selectedFileId) {
      return;
    }
    let cancelled = false;
    const activeFileId = selectedFileId;
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
        setLoadingFileId((current) => (current === activeFileId ? null : current));
      });

    return () => {
      cancelled = true;
    };
  }, [apiClient, open, selectedFileId, selectedViewerError, selectedViewerInfo]);

  useEffect(() => {
    if (!open || !selectedFileId || !selectedViewerInfo) {
      return;
    }
    if (selectedViewerInfo.kind === "hdf5") {
      return;
    }
    if (selectedCaption || selectedCaptionLoading) {
      return;
    }
    let cancelled = false;
    const activeFileId = selectedFileId;
    setCaptionLoadingById((previous) => ({ ...previous, [activeFileId]: true }));
    void apiClient
      .getUploadCaption(activeFileId)
      .then((response) => {
        if (cancelled) {
          return;
        }
        const caption = (response.caption || "").trim();
        if (!caption) {
          return;
        }
        setViewerCaptionById((previous) => ({ ...previous, [activeFileId]: caption }));
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setCaptionLoadingById((previous) => {
          const next = { ...previous };
          delete next[activeFileId];
          return next;
        });
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient, open, selectedCaption, selectedFileId, selectedViewerInfo]);

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
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="viewer-sheet w-full p-0 sm:max-w-none">
        <SheetHeader className="border-b px-4 pb-3">
          <SheetTitle className="flex items-center gap-2">
            <Layers3 className="size-4" />
            Scientific Viewer
          </SheetTitle>
          <SheetDescription>
            Native BisQue-inspired viewer with deep zoom, orthogonal slices, and atlas-backed volume rendering. Use BisQue when you need the legacy fallback surface.
          </SheetDescription>
        </SheetHeader>

        <div className="viewer-body">
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
                    onClick={() => setSelectedFileId(file.file_id)}
                  >
                    {file.original_name}
                  </Button>
                );
              })
            )}
          </aside>

          <section className="viewer-stage">
            {!selectedFileId ? (
              <div className="viewer-empty">Select a file to start.</div>
            ) : showBisqueFrame ? (
              <>
                <div className="viewer-stage-header">
                  <div className="viewer-stage-title">{selectedFile?.original_name ?? "BisQue Viewer"}</div>
                  <div className="viewer-badges">
                    <Badge variant="outline">BisQue fallback</Badge>
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
                    <div className="viewer-stage-title">{selectedViewerInfo.original_name}</div>
                    <div className="viewer-badges">
                      <Badge variant="outline">{selectedViewerInfo.viewer.status}</Badge>
                      {selectedViewerInfo.modality ? (
                        <Badge variant="outline" className="capitalize">
                          {selectedViewerInfo.modality}
                        </Badge>
                      ) : null}
                      <Badge variant="outline">
                        {selectedViewerInfo.backend_mode ?? selectedViewerInfo.viewer.backend_mode ?? "native"}
                      </Badge>
                      <Badge variant="secondary">{selectedViewerInfo.dims_order}</Badge>
                      {selectedViewerInfo.is_volume ? (
                        <Badge variant="outline">{`${zAxisSize} z-slices`}</Badge>
                      ) : null}
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

                {isHdf5Viewer ? (
                  <Hdf5ViewerShell
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
                  <ImageViewerShell
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
                    selectedCaption={selectedCaption}
                    captionLoading={selectedCaptionLoading}
                  />
                )}
              </>
            ) : null}
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
