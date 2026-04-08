import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ApiClient } from "@/lib/api";
import type { Hdf5DatasetSummary, Hdf5MaterialsDashboardResponse, UploadViewerInfo } from "@/types";

import { Hdf5Inspector } from "./Hdf5Inspector";
import { MaterialsHdf5Dashboard } from "./MaterialsHdf5Dashboard";
import { Hdf5Navigator } from "./Hdf5Navigator";
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
      <Hdf5Navigator
        tree={hdf5.tree}
        selectedPath={activeDatasetPath}
        onSelect={onSelectedDatasetPathChange}
        truncated={hdf5.summary.truncated}
      />

      <Hdf5Inspector
        apiClient={apiClient}
        summary={selectedDatasetSummary}
        activeDatasetPath={activeDatasetPath}
        hdf5={hdf5}
        loading={loadingPath === activeDatasetPath}
        error={loadError}
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
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [materialsDashboard, setMaterialsDashboard] = useState<Hdf5MaterialsDashboardResponse | null>(null);
  const [materialsDashboardError, setMaterialsDashboardError] = useState<string | null>(null);
  const [materialsDashboardLoading, setMaterialsDashboardLoading] = useState(false);

  const materials = hdf5?.materials;
  const hasMaterialsDashboard = Boolean(materials?.detected);
  const [activeSection, setActiveSection] = useState<MaterialsSection>(
    hasMaterialsDashboard && materials?.recommended_view === "materials" ? "maps" : "explorer"
  );

  useEffect(() => {
    setActiveSection(hasMaterialsDashboard && materials?.recommended_view === "materials" ? "maps" : "explorer");
  }, [hasMaterialsDashboard, materials?.recommended_view, viewerInfo.file_id]);

  useEffect(() => {
    if (!selectedDatasetPath && preferredDatasetPath) {
      onSelectedDatasetPathChange(preferredDatasetPath);
    }
  }, [onSelectedDatasetPathChange, preferredDatasetPath, selectedDatasetPath]);

  useEffect(() => {
    if (!hdf5?.enabled || !hdf5.supported || !activeDatasetPath || selectedDatasetSummary) {
      return;
    }
    let cancelled = false;
    setLoadingPath(activeDatasetPath);
    setLoadError(null);
    apiClient
      .getHdf5DatasetSummary(viewerInfo.file_id, activeDatasetPath)
      .then((summary) => {
        if (cancelled) {
          return;
        }
        cacheDatasetSummary(summary);
        setLoadingPath((current) => (current === activeDatasetPath ? null : current));
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setLoadingPath((current) => (current === activeDatasetPath ? null : current));
        setLoadError(error instanceof Error ? error.message : "Failed to load HDF5 dataset summary.");
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeDatasetPath,
    apiClient,
    cacheDatasetSummary,
    hdf5?.enabled,
    hdf5?.supported,
    selectedDatasetSummary,
    viewerInfo.file_id,
  ]);

  useEffect(() => {
    if (!hasMaterialsDashboard || materialsDashboard || materialsDashboardLoading) {
      return;
    }
    let cancelled = false;
    setMaterialsDashboardLoading(true);
    setMaterialsDashboardError(null);
    apiClient
      .getHdf5MaterialsDashboard(viewerInfo.file_id)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setMaterialsDashboard(payload);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setMaterialsDashboardError(
          error instanceof Error ? error.message : "Failed to load the materials dashboard."
        );
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setMaterialsDashboardLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient, hasMaterialsDashboard, materialsDashboard, materialsDashboardLoading, viewerInfo.file_id]);

  useEffect(() => {
    setLoadingPath(null);
    setLoadError(null);
    setMaterialsDashboard(null);
    setMaterialsDashboardError(null);
    setMaterialsDashboardLoading(false);
  }, [viewerInfo.file_id]);

  if (!hdf5) {
    return <div className="viewer-empty">HDF5 metadata unavailable.</div>;
  }

  const geometry = hdf5.summary.geometry;
  const geometrySummary = geometry?.dimensions?.length
    ? geometry.dimensions.join(" x ")
    : geometry?.path ?? "No geometry metadata";
  const activeDatasetName =
    selectedDatasetSummary?.dataset_name ?? activeDatasetPath?.split("/").at(-1) ?? "No dataset selected";
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

  return (
    <div className="viewer-hdf-shell" data-hdf5-workspace-root="true">
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
                <HoverCardContent align="end" className="viewer-hdf-context-card">
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
                    {materials?.phase_names?.length ? (
                      <div className="viewer-hdf-context-card-section" data-hdf5-material-phases="true">
                        <strong>Detected phases</strong>
                        <p>{materials.phase_names.join(" • ")}</p>
                      </div>
                    ) : null}
                    {materials?.capabilities?.length ? (
                      <div className="viewer-hdf-context-card-section">
                        <strong>Capabilities</strong>
                        <p>{materials.capabilities.map((capability) => formatSummaryToken(capability)).join(" • ")}</p>
                      </div>
                    ) : null}
                  </div>
                </HoverCardContent>
              </HoverCard>
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
            {materialsDashboardLoading ? (
              <div className="viewer-empty">Loading the materials dashboard…</div>
            ) : materialsDashboardError ? (
              <div className="viewer-metadata-note">
                <strong>Materials dashboard unavailable</strong>
                <span>{materialsDashboardError}</span>
              </div>
            ) : materialsDashboard ? (
              <MaterialsHdf5Dashboard
                apiClient={apiClient}
                dashboard={materialsDashboard}
                section="maps"
                selectedDatasetPath={activeDatasetPath}
                onSelectedDatasetPathChange={onSelectedDatasetPathChange}
                selectedDatasetSummary={selectedDatasetSummary}
                onUseDatasetInChat={onUseDatasetInChat}
              />
            ) : null}
          </TabsContent>

          <TabsContent value="grains" className="viewer-hdf-workspace-tab">
            {materialsDashboard ? (
              <MaterialsHdf5Dashboard
                apiClient={apiClient}
                dashboard={materialsDashboard}
                section="grains"
                selectedDatasetPath={activeDatasetPath}
                onSelectedDatasetPathChange={onSelectedDatasetPathChange}
                selectedDatasetSummary={selectedDatasetSummary}
                onUseDatasetInChat={onUseDatasetInChat}
              />
            ) : (
              <div className="viewer-empty">Loading the materials dashboard…</div>
            )}
          </TabsContent>

          <TabsContent value="orientation" className="viewer-hdf-workspace-tab">
            {materialsDashboard ? (
              <MaterialsHdf5Dashboard
                apiClient={apiClient}
                dashboard={materialsDashboard}
                section="orientation"
                selectedDatasetPath={activeDatasetPath}
                onSelectedDatasetPathChange={onSelectedDatasetPathChange}
                selectedDatasetSummary={selectedDatasetSummary}
                onUseDatasetInChat={onUseDatasetInChat}
              />
            ) : (
              <div className="viewer-empty">Loading the materials dashboard…</div>
            )}
          </TabsContent>

          <TabsContent value="synthetic" className="viewer-hdf-workspace-tab">
            {materialsDashboard ? (
              <MaterialsHdf5Dashboard
                apiClient={apiClient}
                dashboard={materialsDashboard}
                section="synthetic"
                selectedDatasetPath={activeDatasetPath}
                onSelectedDatasetPathChange={onSelectedDatasetPathChange}
                selectedDatasetSummary={selectedDatasetSummary}
                onUseDatasetInChat={onUseDatasetInChat}
              />
            ) : (
              <div className="viewer-empty">Loading the materials dashboard…</div>
            )}
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
    </div>
  );
}
