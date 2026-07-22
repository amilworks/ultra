import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";
import { HoverCard as HoverCardPrimitive } from "radix-ui";

import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ApiClient } from "@/lib/api";
import type { Hdf5DatasetSummary, Hdf5MaterialsDashboardResponse, UploadViewerInfo } from "@/types";

import { Hdf5Inspector } from "./Hdf5Inspector";
import { Hdf5OverlayContainerProvider } from "./Hdf5OverlayContainer";
import { MaterialsHdf5Dashboard } from "./MaterialsHdf5Dashboard";
import { Hdf5Navigator } from "./Hdf5Navigator";
import { PhaseMetadataSummary } from "./PhaseMetadataSummary";
import { formatCount, formatSummaryToken } from "./formatters";

import "./hdf5-viewer.css";

type Hdf5ViewerShellProps = {
  viewerInfo: UploadViewerInfo;
  apiClient: ApiClient;
  selectedDatasetPath: string | null;
  onSelectedDatasetPathChange: (path: string) => void;
  selectedDatasetSummary: Hdf5DatasetSummary | null;
  cacheDatasetSummary: (summary: Hdf5DatasetSummary) => void;
  onUseDatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
};

type MaterialsSection = "maps" | "grains" | "orientation" | "synthetic" | "explorer";

type DatasetLoadState = {
  key: string;
  error: string | null;
};

type MaterialsDashboardState = {
  key: string;
  status: "idle" | "loading" | "success" | "error";
  dashboard: Hdf5MaterialsDashboardResponse | null;
  error: string | null;
};

const currentFullscreenElement = (): Element | null =>
  document.fullscreenElement ??
  (document as unknown as { webkitFullscreenElement?: Element | null }).webkitFullscreenElement ??
  null;

const findFirstDatasetPath = (nodes: NonNullable<UploadViewerInfo["hdf5"]>["tree"]): string | null => {
  for (const node of nodes) {
    if (node.node_type === "dataset") {
      return node.path;
    }
    const nestedPath = findFirstDatasetPath(node.children ?? []);
    if (nestedPath) {
      return nestedPath;
    }
  }
  return null;
};

function Hdf5ExplorerPanel({
  apiClient,
  hdf5,
  activeDatasetPath,
  onSelectedDatasetPathChange,
  selectedDatasetSummary,
  loadingPath,
  loadError,
}: {
  apiClient: ApiClient;
  hdf5: NonNullable<UploadViewerInfo["hdf5"]>;
  activeDatasetPath: string | null;
  onSelectedDatasetPathChange: (path: string) => void;
  selectedDatasetSummary: Hdf5DatasetSummary | null;
  loadingPath: string | null;
  loadError: string | null;
}) {
  return (
    <div className="viewer-hdf-dashboard" data-hdf5-workspace="true">
      <Hdf5Inspector
        apiClient={apiClient}
        summary={selectedDatasetSummary}
        activeDatasetPath={activeDatasetPath}
        hdf5={hdf5}
        loading={loadingPath === activeDatasetPath}
        error={loadError}
      />

      <Hdf5Navigator
        tree={hdf5.tree}
        selectedPath={activeDatasetPath}
        onSelect={onSelectedDatasetPathChange}
        truncated={hdf5.summary.truncated}
      />
    </div>
  );
}

export function Hdf5ViewerShell({
  viewerInfo,
  apiClient,
  selectedDatasetPath,
  onSelectedDatasetPathChange,
  selectedDatasetSummary,
  cacheDatasetSummary,
  onUseDatasetInChat,
}: Hdf5ViewerShellProps) {
  const hdf5 = viewerInfo.hdf5;
  const preferredDatasetPath = useMemo(() => {
    if (!hdf5) {
      return null;
    }
    return hdf5.default_dataset_path ?? findFirstDatasetPath(hdf5.tree);
  }, [hdf5]);
  const activeDatasetPath = selectedDatasetPath ?? preferredDatasetPath;
  const [datasetLoadState, setDatasetLoadState] = useState<DatasetLoadState>({
    key: "",
    error: null,
  });
  const [materialsDashboardState, setMaterialsDashboardState] = useState<MaterialsDashboardState>({
    key: "",
    status: "idle",
    dashboard: null,
    error: null,
  });
  const materialsRequestRef = useRef<{
    key: string;
    request: Promise<Hdf5MaterialsDashboardResponse>;
  } | null>(null);
  const mountedRef = useRef(true);

  const materials = hdf5?.materials;
  const hasMaterialsDashboard = Boolean(materials?.detected);
  const [activeSection, setActiveSection] = useState<MaterialsSection>(
    hasMaterialsDashboard && materials?.recommended_view === "materials" ? "maps" : "explorer"
  );
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [overlayContainer, setOverlayContainer] = useState<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const nativeFullscreenOwnedRef = useRef(false);
  const cssFullscreenRef = useRef(false);
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [cssFullscreen, setCssFullscreen] = useState(false);
  const isFullscreen = nativeFullscreen || cssFullscreen;

  useEffect(() => {
    mountedRef.current = true;
    setOverlayContainer(shellRef.current);
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    cssFullscreenRef.current = cssFullscreen;
  }, [cssFullscreen]);

  const restoreFullscreenFocus = useCallback(() => {
    restoreFocusRef.current?.focus();
    restoreFocusRef.current = null;
  }, []);

  const exitFullscreen = useCallback(() => {
    if (cssFullscreenRef.current) {
      setCssFullscreen(false);
      return;
    }
    const exit =
      document.exitFullscreen ??
      (document as unknown as { webkitExitFullscreen?: () => void }).webkitExitFullscreen;
    if (currentFullscreenElement() === shellRef.current) {
      exit?.call(document);
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
    const shell = shellRef.current as
      | (HTMLDivElement & { webkitRequestFullscreen?: () => void })
      | null;
    if (!shell) {
      return;
    }
    if (cssFullscreenRef.current || currentFullscreenElement() === shell) {
      exitFullscreen();
      return;
    }
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const request = shell.requestFullscreen ?? shell.webkitRequestFullscreen;
    if (typeof request !== "function") {
      setCssFullscreen(true);
      return;
    }
    try {
      const result = request.call(shell) as Promise<void> | undefined;
      result?.catch(() => setCssFullscreen(true));
    } catch {
      setCssFullscreen(true);
    }
  }, [exitFullscreen]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      const active = currentFullscreenElement() === shellRef.current;
      if (active) {
        nativeFullscreenOwnedRef.current = true;
        setNativeFullscreen(true);
        setCssFullscreen(false);
        shellRef.current?.focus();
      } else if (nativeFullscreenOwnedRef.current) {
        nativeFullscreenOwnedRef.current = false;
        setNativeFullscreen(false);
        restoreFullscreenFocus();
      }
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange as EventListener);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange as EventListener);
    };
  }, [restoreFullscreenFocus]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" && event.key !== "Esc") {
        return;
      }
      const ownsNativeFullscreen =
        nativeFullscreenOwnedRef.current || currentFullscreenElement() === shellRef.current;
      if (!cssFullscreenRef.current && !ownsNativeFullscreen) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      exitFullscreen();
    };
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [exitFullscreen]);

  useEffect(() => {
    if (!cssFullscreen) {
      return;
    }
    const root = document.documentElement;
    const body = document.body;
    const previousRootOverflow = root.style.overflow;
    const previousBodyOverflow = body.style.overflow;
    root.style.overflow = "hidden";
    body.style.overflow = "hidden";
    shellRef.current?.focus();
    return () => {
      root.style.overflow = previousRootOverflow;
      body.style.overflow = previousBodyOverflow;
      restoreFullscreenFocus();
    };
  }, [cssFullscreen, restoreFullscreenFocus]);

  useEffect(() => {
    if (!selectedDatasetPath && preferredDatasetPath) {
      onSelectedDatasetPathChange(preferredDatasetPath);
    }
  }, [onSelectedDatasetPathChange, preferredDatasetPath, selectedDatasetPath]);

  const datasetLoadKey =
    hdf5?.enabled && hdf5.supported && activeDatasetPath && !selectedDatasetSummary
      ? [viewerInfo.file_id, activeDatasetPath].join("\u0000")
      : "";
  const loadingPath =
    datasetLoadKey && datasetLoadState.key !== datasetLoadKey ? activeDatasetPath : null;
  const loadError = datasetLoadState.key === datasetLoadKey ? datasetLoadState.error : null;

  const materialsDashboardKey = hasMaterialsDashboard ? viewerInfo.file_id : "";
  const currentMaterialsDashboardState =
    materialsDashboardState.key === materialsDashboardKey
      ? materialsDashboardState
      : {
          key: materialsDashboardKey,
          status: "idle" as const,
          dashboard: null,
          error: null,
        };
  const materialsDashboard = currentMaterialsDashboardState.dashboard;
  const materialsDashboardError = currentMaterialsDashboardState.error;
  const materialsDashboardLoading =
    currentMaterialsDashboardState.status === "loading" ||
    (currentMaterialsDashboardState.status === "idle" &&
      Boolean(materialsDashboardKey) &&
      activeSection !== "explorer");

  useEffect(() => {
    if (!datasetLoadKey || datasetLoadState.key === datasetLoadKey) {
      return;
    }
    let cancelled = false;
    const [, requestedDatasetPath] = datasetLoadKey.split("\u0000");
    apiClient
      .getHdf5DatasetSummary(viewerInfo.file_id, requestedDatasetPath ?? "")
      .then((summary) => {
        if (cancelled) {
          return;
        }
        cacheDatasetSummary(summary);
        setDatasetLoadState({ key: datasetLoadKey, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setDatasetLoadState({
          key: datasetLoadKey,
          error: error instanceof Error ? error.message : "Failed to load HDF5 dataset summary.",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [
    apiClient,
    cacheDatasetSummary,
    datasetLoadKey,
    datasetLoadState.key,
    viewerInfo.file_id,
  ]);

  useEffect(() => {
    if (
      activeSection === "explorer" ||
      !materialsDashboardKey ||
      (materialsDashboardState.key === materialsDashboardKey &&
        materialsDashboardState.status !== "idle") ||
      materialsRequestRef.current?.key === materialsDashboardKey
    ) {
      return;
    }
    setMaterialsDashboardState({
      key: materialsDashboardKey,
      status: "loading",
      dashboard: null,
      error: null,
    });
    const request = apiClient.getHdf5MaterialsDashboard(viewerInfo.file_id);
    materialsRequestRef.current = { key: materialsDashboardKey, request };
    void request
      .then((payload) => {
        if (!mountedRef.current) {
          return;
        }
        setMaterialsDashboardState({
          key: materialsDashboardKey,
          status: "success",
          dashboard: payload,
          error: null,
        });
      })
      .catch((error: unknown) => {
        if (!mountedRef.current) {
          return;
        }
        setMaterialsDashboardState({
          key: materialsDashboardKey,
          status: "error",
          dashboard: null,
          error: error instanceof Error ? error.message : "Failed to load the materials dashboard.",
        });
      })
      .finally(() => {
        if (materialsRequestRef.current?.key === materialsDashboardKey) {
          materialsRequestRef.current = null;
        }
      });
  }, [
    activeSection,
    apiClient,
    materialsDashboardKey,
    materialsDashboardState.key,
    materialsDashboardState.status,
    viewerInfo.file_id,
  ]);

  if (!hdf5) {
    return <div className="viewer-empty">HDF5 metadata unavailable.</div>;
  }

  const geometry = hdf5.summary.geometry;
  const geometrySummary = geometry?.dimensions?.length
    ? geometry.dimensions.join(" x ")
    : geometry?.path ?? "No geometry metadata";
  const activeDatasetPathSegments = activeDatasetPath?.split("/") ?? [];
  const activeDatasetName =
    selectedDatasetSummary?.dataset_name ??
    activeDatasetPathSegments[activeDatasetPathSegments.length - 1] ??
    "No dataset selected";
  const datasetKinds = Object.entries(hdf5.summary.dataset_kinds ?? {})
    .filter(([, count]) => Number(count) > 0)
    .map(([label, count]) => `${formatSummaryToken(label)} (${count})`);
  const summaryMetrics = [
    `${formatCount(hdf5.summary.group_count)} groups`,
    `${formatCount(hdf5.summary.dataset_count)} datasets`,
    geometrySummary,
    selectedDatasetSummary?.preview_kind
      ? formatSummaryToken(selectedDatasetSummary.preview_kind)
      : activeDatasetPath
        ? "Loading preview"
        : "No preview",
  ];
  const openDatasetInExplorer = (path: string) => {
    onSelectedDatasetPathChange(path);
    setActiveSection("explorer");
  };
  const retryMaterialsDashboard = () => {
    setMaterialsDashboardState({
      key: "",
      status: "idle",
      dashboard: null,
      error: null,
    });
  };
  const renderMaterialsSection = (section: Exclude<MaterialsSection, "explorer">) => {
    if (materialsDashboardLoading) {
      return <div className="viewer-empty">Loading the materials dashboard…</div>;
    }
    if (materialsDashboardError) {
      return (
        <div className="viewer-metadata-note viewer-hdf-material-error">
          <strong>Materials dashboard unavailable</strong>
          <span>{materialsDashboardError}</span>
          <Button type="button" variant="outline" size="sm" onClick={retryMaterialsDashboard}>
            Retry
          </Button>
        </div>
      );
    }
    if (!materialsDashboard) {
      return <div className="viewer-empty">Materials dashboard unavailable.</div>;
    }
    return (
      <MaterialsHdf5Dashboard
        apiClient={apiClient}
        dashboard={materialsDashboard}
        section={section}
        selectedDatasetPath={activeDatasetPath}
        onSelectedDatasetPathChange={onSelectedDatasetPathChange}
        onOpenDatasetInExplorer={openDatasetInExplorer}
        selectedDatasetSummary={selectedDatasetSummary}
        onUseDatasetInChat={onUseDatasetInChat}
      />
    );
  };

  return (
    <div
      ref={shellRef}
      className="viewer-hdf-shell"
      data-hdf5-workspace-root="true"
      data-hdf5-fullscreen={isFullscreen ? "true" : "false"}
      tabIndex={-1}
    >
      <Hdf5OverlayContainerProvider container={overlayContainer}>
      <div className="viewer-hdf-shell-top">
        <section className="viewer-hdf-hero">
          <div className="viewer-hdf-hero-copy">
            <div className="viewer-hdf-hero-heading">
              <div className="viewer-hdf-hero-title-stack">
                <h3>{viewerInfo.original_name}</h3>
                {activeDatasetPath ? (
                  <p className="viewer-hdf-current-inline" data-hdf5-active-dataset="true">
                    <span>Current dataset</span>
                    <strong>{activeDatasetName}</strong>
                  </p>
                ) : null}
              </div>
              <div className="viewer-hdf-hero-badges">
                <HoverCard openDelay={120} closeDelay={80}>
                  <HoverCardTrigger asChild>
                    <Button type="button" variant="outline" size="sm" className="viewer-hdf-context-trigger">
                      Context
                    </Button>
                  </HoverCardTrigger>
                  <HoverCardPrimitive.Portal container={overlayContainer}>
                    <HoverCardPrimitive.Content
                      align="end"
                      sideOffset={4}
                      className="viewer-hdf-context-card"
                      data-hdf5-overlay="context"
                    >
                    <div className="viewer-hdf-context-card-grid">
                      <div className="viewer-hdf-context-card-section">
                        <strong>Viewer</strong>
                        <p>{hdf5.supported ? "Native preview" : "Metadata only"}</p>
                        {hdf5.summary.truncated ? <p>Fast-open summary</p> : null}
                        {materials?.schema ? <p>{materials.schema}</p> : null}
                      </div>
                      {activeDatasetPath ? (
                        <div className="viewer-hdf-context-card-section">
                          <strong>Current dataset</strong>
                          <code className="viewer-hdf-context-code">{activeDatasetPath}</code>
                        </div>
                      ) : null}
                      {summaryMetrics.map((metric) => (
                        <div key={metric} className="viewer-hdf-context-card-row">
                          <span>{metric}</span>
                        </div>
                      ))}
                      {datasetKinds.length > 0 ? (
                        <div className="viewer-hdf-context-card-section">
                          <strong>Dataset families</strong>
                          <p>{datasetKinds.join(" • ")}</p>
                        </div>
                      ) : null}
                      <PhaseMetadataSummary
                        phaseNames={materials?.phase_names ?? []}
                        source={materials?.phase_names_source}
                        provenance={materials?.phase_names_provenance}
                      />
                      {materials?.capabilities?.length ? (
                        <div className="viewer-hdf-context-card-section">
                          <strong>Capabilities</strong>
                          <p>{materials.capabilities.map((capability) => formatSummaryToken(capability)).join(" • ")}</p>
                        </div>
                      ) : null}
                    </div>
                    </HoverCardPrimitive.Content>
                  </HoverCardPrimitive.Portal>
                </HoverCard>
                <Button
                  type="button"
                  variant="outline"
                  size="icon-xs"
                  className="viewer-hdf-fullscreen-toggle"
                  aria-label={isFullscreen ? "Exit HDF5 fullscreen" : "Enter HDF5 fullscreen"}
                  aria-pressed={isFullscreen}
                  onClick={toggleFullscreen}
                >
                  {isFullscreen ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}
                </Button>
              </div>
            </div>
          </div>
        </section>

        {hdf5.error ? (
          <div className="viewer-metadata-note">
            <strong>Viewer status</strong>
            <span>{hdf5.error}</span>
          </div>
        ) : null}
      </div>

      {hasMaterialsDashboard ? (
        <Tabs
          value={activeSection}
          onValueChange={(value) => setActiveSection(value as MaterialsSection)}
          className="viewer-hdf-workspace-tabs"
        >
          <TabsList className="viewer-hdf-workspace-tabs-list">
            <TabsTrigger value="maps">Maps</TabsTrigger>
            {materials?.capabilities.includes("grain_metrics") ? (
              <TabsTrigger value="grains">Grains</TabsTrigger>
            ) : null}
            {materials?.capabilities.includes("orientation") ? (
              <TabsTrigger value="orientation">Orientation</TabsTrigger>
            ) : null}
            {materials?.capabilities.includes("synthetic_stats") ? (
              <TabsTrigger value="synthetic">Synthetic Stats</TabsTrigger>
            ) : null}
            <TabsTrigger value="explorer">Explorer</TabsTrigger>
          </TabsList>

          <TabsContent value="maps" className="viewer-hdf-workspace-tab">
            {renderMaterialsSection("maps")}
          </TabsContent>

          <TabsContent value="grains" className="viewer-hdf-workspace-tab">
            {renderMaterialsSection("grains")}
          </TabsContent>

          <TabsContent value="orientation" className="viewer-hdf-workspace-tab">
            {renderMaterialsSection("orientation")}
          </TabsContent>

          <TabsContent value="synthetic" className="viewer-hdf-workspace-tab">
            {renderMaterialsSection("synthetic")}
          </TabsContent>

          <TabsContent value="explorer" className="viewer-hdf-workspace-tab">
            <Hdf5ExplorerPanel
              apiClient={apiClient}
              hdf5={hdf5}
              activeDatasetPath={activeDatasetPath}
              onSelectedDatasetPathChange={onSelectedDatasetPathChange}
              selectedDatasetSummary={selectedDatasetSummary}
              loadingPath={loadingPath}
              loadError={loadError}
            />
          </TabsContent>
        </Tabs>
      ) : (
        <Hdf5ExplorerPanel
          apiClient={apiClient}
          hdf5={hdf5}
          activeDatasetPath={activeDatasetPath}
          onSelectedDatasetPathChange={onSelectedDatasetPathChange}
          selectedDatasetSummary={selectedDatasetSummary}
          loadingPath={loadingPath}
          loadError={loadError}
        />
      )}
      </Hdf5OverlayContainerProvider>
    </div>
  );
}
