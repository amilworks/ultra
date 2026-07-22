import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Scatter, ScatterChart, XAxis, YAxis } from "recharts";
import { HoverCard as HoverCardPrimitive } from "radix-ui";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { HoverCard, HoverCardTrigger } from "@/components/ui/hover-card";
import type { ApiClient } from "@/lib/api";
import type {
  Hdf5DatasetSummary,
  Hdf5MaterialsChartResponse,
  Hdf5MaterialsDashboardResponse,
} from "@/types";

import {
  Hdf5DatasetPreview,
  type Hdf5FeatureSelectionState,
} from "./Hdf5DatasetPreview";
import { useHdf5OverlayContainer } from "./Hdf5OverlayContainer";
import { formatSummaryToken } from "./formatters";

type MaterialsSection = "maps" | "grains" | "orientation" | "synthetic";

type MaterialsHdf5DashboardProps = {
  apiClient: ApiClient;
  dashboard: Hdf5MaterialsDashboardResponse;
  section: MaterialsSection;
  selectedDatasetPath: string | null;
  onSelectedDatasetPathChange: (path: string) => void;
  onOpenDatasetInExplorer: (path: string) => void;
  selectedDatasetSummary: Hdf5DatasetSummary | null;
  onUseDatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
};

const BAR_CHART_CONFIG = {
  value: { label: "Value", color: "var(--chart-1)" },
  count: { label: "Count", color: "var(--chart-2)" },
  percent: { label: "Percent", color: "var(--chart-3)" },
  average: { label: "Average", color: "var(--chart-1)" },
};

const SCATTER_CHART_CONFIG = {
  value: { label: "Value", color: "var(--chart-1)" },
};

const SECTION_COPY: Record<
  MaterialsSection,
  {
    title: string;
  }
> = {
  maps: {
    title: "Canonical spatial datasets",
  },
  grains: {
    title: "Grain-scale distributions",
  },
  orientation: {
    title: "Orientation quality and distributions",
  },
  synthetic: {
    title: "Targeted synthetic-statistics panels",
  },
};

const formatAxisTick = (value: string | number): string => {
  const text = String(value);
  if (text.length <= 14) {
    return text;
  }
  return `${text.slice(0, 12)}…`;
};

function MaterialsPanelHeader({
  section,
}: {
  section: MaterialsSection;
}) {
  const copy = SECTION_COPY[section];

  return (
    <header className="viewer-hdf-material-panel-header">
      <h4>{copy.title}</h4>
    </header>
  );
}

function MaterialsChartCard({
  chart,
  fileId,
  onOpenExplorer,
  onUseDatasetInChat,
}: {
  chart: Hdf5MaterialsChartResponse;
  fileId: string;
  onOpenExplorer: (datasetPath: string) => void;
  onUseDatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
}) {
  const overlayContainer = useHdf5OverlayContainer();
  const primaryPath = chart.source_paths[0] ?? null;
  const yKey = chart.y_key;
  const xKey = chart.x_key;
  const chartConfig = chart.kind === "scatter" ? SCATTER_CHART_CONFIG : BAR_CHART_CONFIG;

  return (
    <Card className="viewer-hdf-material-card" data-hdf5-material-chart-kind={chart.kind}>
      <CardHeader className="viewer-hdf-material-card-header">
        <div className="viewer-hdf-material-card-title-row">
          <div className="viewer-hdf-material-card-copy">
            <CardTitle>{chart.title}</CardTitle>
            {chart.description ? <CardDescription>{chart.description}</CardDescription> : null}
          </div>
          <div className="viewer-hdf-material-card-badges">
            {chart.units_hint ? <Badge variant="outline">{chart.units_hint}</Badge> : null}
            {chart.source_paths.length > 0 ? (
              <HoverCard openDelay={120} closeDelay={80}>
                <HoverCardTrigger asChild>
                  <Button type="button" variant="outline" size="sm">
                    Source
                  </Button>
                </HoverCardTrigger>
                <HoverCardPrimitive.Portal container={overlayContainer}>
                  <HoverCardPrimitive.Content
                    align="end"
                    sideOffset={4}
                    className="viewer-hdf-material-source-card"
                    data-hdf5-overlay="material-source"
                  >
                    <div className="viewer-hdf-material-source-card-grid">
                      <div className="viewer-hdf-material-source-card-section">
                        <strong>Source datasets</strong>
                        <div className="viewer-hdf-material-source-list">
                          {chart.source_paths.map((sourcePath) => (
                            <code key={sourcePath} className="viewer-hdf-path-chip viewer-hdf-material-path-chip">
                              {sourcePath}
                            </code>
                          ))}
                        </div>
                      </div>
                      {chart.provenance ? (
                        <div className="viewer-hdf-material-source-card-section">
                          <strong>Sampling</strong>
                          <p>{chart.provenance}</p>
                        </div>
                      ) : null}
                    </div>
                  </HoverCardPrimitive.Content>
                </HoverCardPrimitive.Portal>
              </HoverCard>
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent className="viewer-hdf-material-card-content">
        <ChartContainer
          config={chartConfig}
          className="viewer-hdf-chart-canvas viewer-hdf-material-chart-canvas"
        >
          {chart.kind === "scatter" ? (
            <ScatterChart data={chart.data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey={xKey}
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 11 }}
                name={xKey}
              />
              <YAxis
                type="number"
                dataKey={yKey}
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 11 }}
                width={42}
                name={yKey}
              />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideIndicator />} />
              <Scatter dataKey={yKey} fill="var(--color-value)" />
            </ScatterChart>
          ) : (
            <BarChart data={chart.data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey={xKey}
                tickLine={false}
                axisLine={false}
                minTickGap={18}
                tick={{ fontSize: 11 }}
                tickFormatter={formatAxisTick}
              />
              <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={42} />
              <ChartTooltip content={<ChartTooltipContent labelKey={xKey} />} />
              <Bar dataKey={yKey} fill="var(--color-value)" radius={[6, 6, 0, 0]} />
            </BarChart>
          )}
        </ChartContainer>

        <div className="viewer-hdf-material-card-footer">
          <span />
          <div className="viewer-hdf-material-actions">
            {primaryPath ? (
              <Button type="button" variant="outline" size="sm" onClick={() => onOpenExplorer(primaryPath)}>
                Open in Explorer
              </Button>
            ) : null}
            {chart.source_paths.length > 0 && onUseDatasetInChat ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onUseDatasetInChat(fileId, chart.source_paths)}
              >
                Use in chat
              </Button>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function MaterialsMapsPanel({
  apiClient,
  dashboard,
  selectedDatasetPath,
  onSelectedDatasetPathChange,
  onOpenDatasetInExplorer,
  selectedDatasetSummary,
  onUseDatasetInChat,
  featureSelection,
  onFeatureSelectionChange,
}: {
  apiClient: ApiClient;
  dashboard: Hdf5MaterialsDashboardResponse;
  selectedDatasetPath: string | null;
  onSelectedDatasetPathChange: (path: string) => void;
  onOpenDatasetInExplorer: (path: string) => void;
  selectedDatasetSummary: Hdf5DatasetSummary | null;
  onUseDatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
  featureSelection: Hdf5FeatureSelectionState | null;
  onFeatureSelectionChange: (selection: Hdf5FeatureSelectionState) => void;
}) {
  const recommendedPath =
    dashboard.overview.recommended_map_dataset_path ?? dashboard.maps[0]?.dataset_path ?? null;
  const selectionFacts = selectedDatasetSummary
    ? [
        selectedDatasetSummary.preview_kind ? formatSummaryToken(selectedDatasetSummary.preview_kind) : null,
        selectedDatasetSummary.units_hint ?? null,
      ].filter(Boolean)
    : [];

  useEffect(() => {
    if (!selectedDatasetPath && recommendedPath) {
      onSelectedDatasetPathChange(recommendedPath);
    }
  }, [onSelectedDatasetPathChange, recommendedPath, selectedDatasetPath]);

  return (
    <section className="viewer-hdf-material-panel">
      <div className="viewer-hdf-material-map-grid">
        {selectedDatasetSummary ? (
          <section className="viewer-hdf-material-map-stage" aria-label="Selected materials map">
            <header className="viewer-hdf-material-map-stage-header">
              <div className="viewer-hdf-material-card-copy">
                <p className="viewer-hdf-material-section-label">Selected map</p>
                <h3>{selectedDatasetSummary.dataset_name}</h3>
                {selectionFacts.length > 0 ? (
                  <p className="viewer-hdf-material-note">{selectionFacts.join(" • ")}</p>
                ) : null}
              </div>
              <div className="viewer-hdf-material-actions">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => onOpenDatasetInExplorer(selectedDatasetSummary.dataset_path)}
                >
                  Open in Explorer
                </Button>
                {onUseDatasetInChat ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      onUseDatasetInChat(selectedDatasetSummary.file_id, [selectedDatasetSummary.dataset_path])
                    }
                  >
                    Use in chat
                  </Button>
                ) : null}
              </div>
            </header>
            <div className="viewer-hdf-material-preview-content">
              <Hdf5DatasetPreview
                apiClient={apiClient}
                summary={selectedDatasetSummary}
                compactLayout
                featureSelection={featureSelection}
                onFeatureSelectionChange={onFeatureSelectionChange}
              />
            </div>
          </section>
        ) : (
          <div className="viewer-empty viewer-hdf-material-map-stage">Loading the selected materials map…</div>
        )}

        <aside className="viewer-hdf-material-map-rail" aria-label="Available materials maps">
          <header className="viewer-hdf-material-map-rail-header">
            <strong>Available maps</strong>
            <span>{dashboard.maps.length.toLocaleString()}</span>
          </header>
          <div className="viewer-hdf-material-map-actions">
            {dashboard.maps.map((mapEntry) => {
              const active = selectedDatasetPath === mapEntry.dataset_path;
              return (
                <Button
                  key={mapEntry.dataset_path}
                  type="button"
                  variant={active ? "secondary" : "outline"}
                  size="sm"
                  className="viewer-hdf-material-map-button"
                  onClick={() => onSelectedDatasetPathChange(mapEntry.dataset_path)}
                  aria-pressed={active}
                  data-hdf5-material-map={mapEntry.semantic_role}
                >
                  <span className="viewer-hdf-material-map-button-copy">
                    <strong>{mapEntry.title}</strong>
                    <small>
                      {mapEntry.description ?? formatSummaryToken(mapEntry.semantic_role)}
                    </small>
                  </span>
                </Button>
              );
            })}
          </div>
        </aside>
      </div>
    </section>
  );
}

function MaterialsChartsSection({
  dashboard,
  section,
  onOpenDatasetInExplorer,
  onUseDatasetInChat,
}: {
  dashboard: Hdf5MaterialsDashboardResponse;
  section: Exclude<MaterialsSection, "maps">;
  onOpenDatasetInExplorer: (path: string) => void;
  onUseDatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
}) {
  const charts =
    section === "grains"
      ? dashboard.grain_charts
      : section === "orientation"
        ? dashboard.orientation_charts
        : dashboard.synthetic_stats;

  return (
    <section className="viewer-hdf-material-panel">
      <MaterialsPanelHeader section={section} />

      {charts.length > 0 ? (
        <div className="viewer-hdf-material-chart-grid">
          {charts.map((chart) => (
            <MaterialsChartCard
              key={`${section}:${chart.title}:${chart.source_paths.join("|")}`}
              chart={chart}
              fileId={dashboard.file_id}
              onOpenExplorer={onOpenDatasetInExplorer}
              onUseDatasetInChat={onUseDatasetInChat}
            />
          ))}
        </div>
      ) : (
        <div className="viewer-empty">
          No {section === "synthetic" ? "synthetic statistics" : section} previews are available for this file.
        </div>
      )}
    </section>
  );
}

export function MaterialsHdf5Dashboard({
  apiClient,
  dashboard,
  section,
  selectedDatasetPath,
  onSelectedDatasetPathChange,
  onOpenDatasetInExplorer,
  selectedDatasetSummary,
  onUseDatasetInChat,
}: MaterialsHdf5DashboardProps) {
  const [featureSelection, setFeatureSelection] = useState<Hdf5FeatureSelectionState | null>(null);

  return (
    <div
      className="viewer-hdf-material-shell"
      data-hdf5-materials-dashboard="true"
      data-hdf5-material-section={section}
    >
      <div className="viewer-hdf-material-body">
        {section === "maps" ? (
          <MaterialsMapsPanel
            apiClient={apiClient}
            dashboard={dashboard}
            selectedDatasetPath={selectedDatasetPath}
            onSelectedDatasetPathChange={onSelectedDatasetPathChange}
            onOpenDatasetInExplorer={onOpenDatasetInExplorer}
            selectedDatasetSummary={selectedDatasetSummary}
            onUseDatasetInChat={onUseDatasetInChat}
            featureSelection={featureSelection}
            onFeatureSelectionChange={setFeatureSelection}
          />
        ) : (
          <MaterialsChartsSection
            dashboard={dashboard}
            section={section}
            onOpenDatasetInExplorer={onOpenDatasetInExplorer}
            onUseDatasetInChat={onUseDatasetInChat}
          />
        )}
      </div>
    </div>
  );
}
