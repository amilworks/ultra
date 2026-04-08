import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Scatter, ScatterChart, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type {
  Hdf5DatasetHistogramResponse,
  Hdf5DatasetSummary,
  Hdf5DatasetTablePreviewResponse,
} from "@/types";
import type { ApiClient } from "@/lib/api";

import { SlicePlaneCanvas } from "../SlicePlaneCanvas";
import { SliceStackVolumeCanvas } from "../SliceStackVolumeCanvas";

type Hdf5DatasetPreviewProps = {
  apiClient: ApiClient;
  summary: Hdf5DatasetSummary;
  compactLayout?: boolean;
};

const HISTOGRAM_CHART_CONFIG = {
  count: { label: "Count", color: "var(--chart-2)" },
};

const SCATTER_CHART_CONFIG = {
  value: { label: "Value", color: "var(--chart-1)" },
};

const VOLUME_PREVIEW_KINDS = new Set(["scalar_volume", "label_volume", "rgb_volume", "vector_volume"]);

const formatRangeValue = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Not available";
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
    return value.toExponential(2);
  }
  return value.toFixed(3).replace(/\.?0+$/, "");
};

const formatCount = (value: number): string => Math.max(0, Math.round(value)).toLocaleString();

const buildSampleSummary = (sampleCount: number | null | undefined, total: number): string => {
  if (typeof sampleCount !== "number" || !Number.isFinite(sampleCount)) {
    return "Bounded preview sample";
  }
  const ratio = total > 0 ? (sampleCount / total) * 100 : null;
  const ratioText = ratio != null && Number.isFinite(ratio) ? ` (${ratio.toFixed(ratio >= 10 ? 1 : 2)}%)` : "";
  return `${formatCount(sampleCount)} sampled values of ${formatCount(total)}${ratioText}`;
};

const axisLabel = (axis: "z" | "y" | "x"): string => {
  if (axis === "z") {
    return "XY";
  }
  if (axis === "y") {
    return "XZ";
  }
  return "YZ";
};

const axisSize = (summary: Hdf5DatasetSummary, axis: "z" | "y" | "x"): number => {
  const size = Number(summary.dimension_summary?.[axis] ?? 1);
  return Math.max(1, Number.isFinite(size) ? size : 1);
};

const buildPreviewNotice = (summary: Hdf5DatasetSummary, componentLabel: string | null): string | null => {
  if (summary.preview_kind === "vector_volume") {
    return `Component-aware scientific slice for ${componentLabel ?? "the selected component"}. This is not an RGB rendering.`;
  }
  if (summary.preview_kind === "rgb_volume") {
    return "Display-oriented RGB slice preview. The viewer keeps color interpretation explicit for this dataset.";
  }
  if (summary.preview_kind === "label_volume") {
    return "Categorical slice preview with a deterministic label palette. Colors identify regions, not intensity magnitude.";
  }
  return null;
};

function Hdf5VolumePreview({ apiClient, summary, compactLayout = false }: Hdf5DatasetPreviewProps) {
  const availableAxes = summary.slice_axes.length > 0 ? summary.slice_axes : (["z"] as Array<"z" | "y" | "x">);
  const [selectedAxis, setSelectedAxis] = useState<"z" | "y" | "x">(availableAxes[0] ?? "z");
  const [selectedComponent, setSelectedComponent] = useState(0);
  const canRenderVolume = Boolean(
    summary.volume_eligible &&
      summary.capabilities.includes("volume") &&
      summary.axis_sizes &&
      summary.preview_planes.z &&
      (summary.render_policy === "scalar" || summary.atlas_scheme)
  );
  const [selectedTab, setSelectedTab] = useState<"volume" | "visual" | "distribution">(
    canRenderVolume ? "volume" : "visual"
  );
  const [histogram, setHistogram] = useState<Hdf5DatasetHistogramResponse | null>(null);
  const [histogramError, setHistogramError] = useState<string | null>(null);
  const [histogramLoading, setHistogramLoading] = useState(false);

  useEffect(() => {
    setSelectedAxis(availableAxes[0] ?? "z");
    setSelectedComponent(0);
  }, [summary.dataset_path, availableAxes]);

  useEffect(() => {
    setSelectedTab(canRenderVolume ? "volume" : "visual");
  }, [canRenderVolume, summary.dataset_path]);

  const maxIndex = axisSize(summary, selectedAxis);
  const [selectedIndex, setSelectedIndex] = useState(Math.max(0, Math.floor(maxIndex / 2)));

  useEffect(() => {
    setSelectedIndex(Math.max(0, Math.floor(axisSize(summary, selectedAxis) / 2)));
  }, [selectedAxis, summary]);

  const componentCount = Math.max(1, Number(summary.component_count || 1));
  const componentLabels =
    summary.component_labels.length > 0
      ? summary.component_labels
      : Array.from({ length: componentCount }, (_, index) => `component_${index + 1}`);
  const activeComponent = Math.max(0, Math.min(selectedComponent, componentCount - 1));
  const activePlane = summary.preview_planes[selectedAxis];
  const previewUrl = apiClient.hdf5SlicePreviewUrl(summary.file_id, {
    datasetPath: summary.dataset_path,
    axis: selectedAxis,
    index: selectedIndex,
    component: summary.preview_kind === "vector_volume" ? activeComponent : undefined,
  });
  const volumeFallbackUrl = apiClient.hdf5SlicePreviewUrl(summary.file_id, {
    datasetPath: summary.dataset_path,
    axis: "z",
    index: Math.max(0, Math.floor(axisSize(summary, "z") / 2)),
  });
  const previewTabCount = (canRenderVolume ? 1 : 0) + 1 + (summary.capabilities.includes("histogram") ? 1 : 0);

  const renderPreviewTabsList = () =>
    previewTabCount > 1 ? (
      <TabsList className="viewer-hdf-preview-tabs-list">
        {canRenderVolume ? <TabsTrigger value="volume">Volume</TabsTrigger> : null}
        <TabsTrigger value="visual">Slice</TabsTrigger>
        {summary.capabilities.includes("histogram") ? <TabsTrigger value="distribution">Distribution</TabsTrigger> : null}
      </TabsList>
    ) : null;

  const renderComponentField = () =>
    summary.preview_kind === "vector_volume" ? (
      <label className="viewer-hdf-inline-field viewer-hdf-inline-field-compact">
        <span>Component</span>
        <Select value={String(activeComponent)} onValueChange={(value) => setSelectedComponent(Number(value) || 0)}>
          <SelectTrigger className="viewer-hdf-select">
            <SelectValue placeholder="Select component" />
          </SelectTrigger>
          <SelectContent>
            {componentLabels.map((label, index) => (
              <SelectItem key={`${label}:${index}`} value={String(index)}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
    ) : null;

  const renderCompactToolbar = () =>
    compactLayout && (previewTabCount > 1 || summary.preview_kind === "vector_volume") ? (
      <div className="viewer-hdf-preview-compact-toolbar">
        {renderPreviewTabsList()}
        {renderComponentField()}
      </div>
    ) : null;

  useEffect(() => {
    if (!summary.capabilities.includes("histogram")) {
      setHistogram(null);
      setHistogramError(null);
      return;
    }
    let cancelled = false;
    setHistogramLoading(true);
    setHistogramError(null);
    apiClient
      .getHdf5DatasetHistogram(summary.file_id, summary.dataset_path, {
        component: summary.preview_kind === "vector_volume" ? activeComponent : undefined,
        bins: 24,
      })
      .then((response) => {
        if (cancelled) {
          return;
        }
        setHistogram(response);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setHistogramError(error instanceof Error ? error.message : "Failed to load histogram preview.");
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setHistogramLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeComponent, apiClient, summary]);

  return (
    <div className="viewer-hdf-preview-body" data-hdf5-preview-kind={summary.preview_kind ?? "unknown"}>
      <Tabs
        value={selectedTab}
        onValueChange={(value) => setSelectedTab(value as "volume" | "visual" | "distribution")}
        className={`viewer-hdf-preview-tabs${compactLayout ? " viewer-hdf-preview-tabs-compact" : ""}`}
      >
        {!compactLayout ? (
          <div className="viewer-hdf-preview-toolbar">
            <div className="viewer-hdf-preview-toolbar-copy">
              <span>
                {canRenderVolume
                  ? "Use Volume for native 3D inspection, Slice for orthogonal spot checks, and Distribution when you want bounded numeric context."
                  : "Use Slice to inspect the selected dataset, then switch to Distribution when you want bounded numeric context."}
              </span>
            </div>
            {renderComponentField()}
            {renderPreviewTabsList()}
          </div>
        ) : null}

        {canRenderVolume ? (
          <TabsContent value="volume" className="viewer-hdf-preview-tab">
            {renderCompactToolbar()}
            <div className="viewer-hdf-preview-note">
              <strong>
                {summary.preview_kind === "label_volume" ? "Categorical volume" : "Scalar volume"}
              </strong>
              <span>
                {summary.render_policy === "scalar"
                  ? "Native 3D preserves scalar intensity ordering from the HDF5 dataset and scales the volume from stored geometry metadata."
                  : "Native 3D uses the categorical/display volume path and scales the volume from stored geometry metadata."}
              </span>
            </div>
            <div className="viewer-hdf-slice-shell" data-hdf5-volume-preview="true">
              <SliceStackVolumeCanvas
                volumeSource={
                  summary.render_policy === "scalar"
                    ? {
                        kind: "scalar",
                        loadScalarVolume: () =>
                          apiClient.getHdf5ScalarVolume(summary.file_id, {
                            datasetPath: summary.dataset_path,
                          }),
                        fallbackImageUrl: volumeFallbackUrl,
                        axisSizes: summary.axis_sizes!,
                        plane: summary.preview_planes.z,
                        physicalSpacing: summary.physical_spacing ?? null,
                        renderPolicy: summary.render_policy,
                        texturePolicy: summary.texture_policy,
                      }
                    : {
                        kind: "atlas",
                        atlasUrl: apiClient.hdf5AtlasPreviewUrl(summary.file_id, {
                          datasetPath: summary.dataset_path,
                        }),
                        fallbackImageUrl: volumeFallbackUrl,
                        atlasScheme: summary.atlas_scheme!,
                        axisSizes: summary.axis_sizes!,
                        plane: summary.preview_planes.z,
                        physicalSpacing: summary.physical_spacing ?? null,
                        renderPolicy: summary.render_policy,
                        texturePolicy: summary.texture_policy,
                      }
                }
                className="viewer-canvas-root viewer-hdf-slice-canvas"
              />
            </div>
          </TabsContent>
        ) : null}

        <TabsContent value="visual" className="viewer-hdf-preview-tab">
          <div className="viewer-hdf-slice-layout">
            <div className="viewer-hdf-slice-sidebar">
              {renderCompactToolbar()}
              <div className="viewer-hdf-preview-controls">
                <div className="viewer-hdf-axis-toggle" role="tablist" aria-label="Slice orientation">
                  {availableAxes.map((axis) => (
                    <Button
                      key={axis}
                      type="button"
                      size="sm"
                      variant={selectedAxis === axis ? "secondary" : "outline"}
                      onClick={() => setSelectedAxis(axis)}
                    >
                      {axisLabel(axis)}
                    </Button>
                  ))}
                </div>
              </div>

              {!canRenderVolume && summary.volume_reason ? (
                <div className="viewer-hdf-preview-note">
                  <strong>Slice-only</strong>
                  <span>{summary.volume_reason}</span>
                </div>
              ) : null}

              {buildPreviewNotice(summary, componentLabels[activeComponent] ?? null) ? (
                <div className="viewer-hdf-preview-note">
                  <strong>
                    {summary.preview_kind === "vector_volume"
                      ? "Component-aware"
                      : summary.preview_kind === "rgb_volume"
                        ? "RGB"
                        : "Slice"}
                  </strong>
                  <span>{buildPreviewNotice(summary, componentLabels[activeComponent] ?? null)}</span>
                </div>
              ) : null}

              <Card className="viewer-hdf-slider-panel">
                <CardContent className="viewer-hdf-slider-panel-content">
                  <div className="viewer-hdf-inline-field viewer-hdf-slider-field">
                    <div className="viewer-hdf-slider-header">
                      <span>{activePlane?.label ?? `${axisLabel(selectedAxis)} plane`}</span>
                      <span>
                        slice {selectedIndex + 1} / {maxIndex}
                      </span>
                    </div>
                    <Slider
                      className="viewer-hdf-slider"
                      min={0}
                      max={Math.max(0, maxIndex - 1)}
                      step={1}
                      value={[selectedIndex]}
                      onValueChange={(value) => setSelectedIndex(value[0] ?? 0)}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {activePlane ? (
              <div className="viewer-hdf-slice-shell" data-hdf5-slice-preview="true">
                <SlicePlaneCanvas
                  imageUrl={previewUrl}
                  descriptor={activePlane}
                  title={`${summary.dataset_name}-${selectedAxis}`}
                  className="viewer-canvas-root viewer-hdf-slice-canvas"
                />
              </div>
            ) : (
              <div className="viewer-empty">Slice descriptor unavailable for this dataset.</div>
            )}
          </div>
        </TabsContent>

        {summary.capabilities.includes("histogram") ? (
          <TabsContent value="distribution" className="viewer-hdf-preview-tab">
            {renderCompactToolbar()}
            <section className="viewer-hdf-chart-card" data-hdf5-histogram="true">
              <div className="viewer-hdf-tree-header">
                <strong>Sampled distribution</strong>
                <span>
                  {histogram?.component_label ? `${histogram.component_label} • ` : ""}
                  {buildSampleSummary(histogram?.sample_count, summary.element_count)}
                </span>
              </div>
              {histogramLoading ? (
                <div className="viewer-empty">Loading histogram preview...</div>
              ) : histogramError ? (
                <div className="viewer-metadata-note">
                  <strong>Histogram unavailable</strong>
                  <span>{histogramError}</span>
                </div>
              ) : histogram && histogram.bins.length > 0 ? (
                <>
                  <div className="viewer-hdf-histogram-summary">
                    <span>Min {formatRangeValue(histogram.min)}</span>
                    <span>Max {formatRangeValue(histogram.max)}</span>
                  </div>
                  <ChartContainer config={HISTOGRAM_CHART_CONFIG} className="viewer-hdf-chart-canvas h-[260px] w-full">
                    <BarChart data={histogram.bins}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={18} />
                      <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                      <ChartTooltip content={<ChartTooltipContent labelKey="label" />} />
                      <Bar dataKey="count" fill="var(--color-count)" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ChartContainer>
                </>
              ) : (
                <div className="viewer-empty">No histogram data available for this dataset.</div>
              )}
            </section>
          </TabsContent>
        ) : null}
      </Tabs>
    </div>
  );
}

function Hdf5TablePreview({ apiClient, summary }: Hdf5DatasetPreviewProps) {
  const [offset, setOffset] = useState(0);
  const [tablePreview, setTablePreview] = useState<Hdf5DatasetTablePreviewResponse | null>(null);
  const [tableError, setTableError] = useState<string | null>(null);
  const [tableLoading, setTableLoading] = useState(false);

  useEffect(() => {
    setOffset(0);
  }, [summary.dataset_path]);

  useEffect(() => {
    let cancelled = false;
    setTableLoading(true);
    setTableError(null);
    apiClient
      .getHdf5DatasetTablePreview(summary.file_id, summary.dataset_path, { offset, limit: 12 })
      .then((response) => {
        if (cancelled) {
          return;
        }
        setTablePreview(response);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setTableError(error instanceof Error ? error.message : "Failed to load table preview.");
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setTableLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient, offset, summary]);

  const canGoBack = offset > 0;
  const canGoForward = tablePreview ? offset + tablePreview.rows.length < tablePreview.total_rows : false;
  const defaultTab = tablePreview?.charts.length ? "charts" : "rows";

  return (
    <div className="viewer-hdf-preview-body" data-hdf5-preview-kind={summary.preview_kind ?? "table"}>
      <div className="viewer-hdf-preview-note">
        <strong>Table preview</strong>
        <span>
          Use charts for a quick read on the sampled distribution, then move to rows when you need exact values from the bounded preview window.
        </span>
      </div>

      {tableLoading ? (
        <div className="viewer-empty">Loading table preview...</div>
      ) : tableError ? (
        <div className="viewer-metadata-note">
          <strong>Table preview unavailable</strong>
          <span>{tableError}</span>
        </div>
      ) : tablePreview ? (
        <Tabs key={summary.dataset_path} defaultValue={defaultTab} className="viewer-hdf-preview-tabs">
          <TabsList className="viewer-hdf-preview-tabs-list">
            {tablePreview.charts.length > 0 ? <TabsTrigger value="charts">Charts</TabsTrigger> : null}
            <TabsTrigger value="rows">Rows</TabsTrigger>
          </TabsList>

          {tablePreview.charts.length > 0 ? (
            <TabsContent value="charts" className="viewer-hdf-preview-tab">
              <div className="viewer-hdf-chart-grid">
                {tablePreview.charts.map((chart) => (
                  <section
                    key={`${chart.kind}:${chart.title}`}
                    className="viewer-hdf-chart-card"
                    data-hdf5-chart-kind={chart.kind}
                  >
                    <div className="viewer-hdf-tree-header">
                      <strong>{chart.title}</strong>
                      <span>
                        {chart.description ? `${chart.description} ` : ""}
                        {`${formatCount(chart.data.length)} sampled row${chart.data.length === 1 ? "" : "s"}`}
                      </span>
                    </div>
                    <ChartContainer
                      config={chart.kind === "histogram" ? HISTOGRAM_CHART_CONFIG : SCATTER_CHART_CONFIG}
                      className="viewer-hdf-chart-canvas h-[260px] w-full"
                    >
                      {chart.kind === "histogram" ? (
                        <BarChart data={chart.data}>
                          <CartesianGrid vertical={false} />
                          <XAxis dataKey={chart.x_key} tickLine={false} axisLine={false} minTickGap={18} />
                          <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                          <ChartTooltip content={<ChartTooltipContent labelKey={chart.x_key} />} />
                          <Bar dataKey={chart.y_key} fill="var(--color-count)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      ) : (
                        <ScatterChart data={chart.data}>
                          <CartesianGrid />
                          <XAxis type="number" dataKey={chart.x_key} tickLine={false} axisLine={false} name={chart.x_key} />
                          <YAxis type="number" dataKey={chart.y_key} tickLine={false} axisLine={false} name={chart.y_key} />
                          <ChartTooltip cursor={false} content={<ChartTooltipContent hideIndicator />} />
                          <Scatter dataKey={chart.y_key} fill="var(--color-value)" />
                        </ScatterChart>
                      )}
                    </ChartContainer>
                  </section>
                ))}
              </div>
            </TabsContent>
          ) : null}

          <TabsContent value="rows" className="viewer-hdf-preview-tab viewer-hdf-preview-tab-rows">
            <div className="viewer-hdf-pagination">
              <span>
                Rows {formatCount(tablePreview.offset + 1)}-{formatCount(tablePreview.offset + tablePreview.rows.length)} of {formatCount(tablePreview.total_rows)}
              </span>
              <div className="viewer-hdf-pagination-actions">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={!canGoBack}
                  onClick={() => setOffset((current) => Math.max(0, current - tablePreview.limit))}
                >
                  Previous
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={!canGoForward}
                  onClick={() => setOffset((current) => current + tablePreview.limit)}
                >
                  Next
                </Button>
              </div>
            </div>

            <div className="viewer-hdf-table-shell" data-hdf5-table-preview="true">
              <table className="viewer-hdf-table">
                <thead>
                  <tr>
                    <th>Row</th>
                    {tablePreview.columns.map((column) => (
                      <th key={column.key}>{column.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tablePreview.rows.map((row, index) => (
                    <tr key={`${tablePreview.offset}:${index}`}>
                      <td>{formatCount(Number(row.row_index ?? tablePreview.offset + index))}</td>
                      {tablePreview.columns.map((column) => (
                        <td key={`${tablePreview.offset}:${index}:${column.key}`}>{String(row[column.key] ?? "—")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </TabsContent>
        </Tabs>
      ) : (
        <div className="viewer-empty">Table preview unavailable.</div>
      )}
    </div>
  );
}

export function Hdf5DatasetPreview({ apiClient, summary, compactLayout = false }: Hdf5DatasetPreviewProps) {
  const previewKind = summary.preview_kind ?? "unknown";

  if (VOLUME_PREVIEW_KINDS.has(previewKind)) {
    return <Hdf5VolumePreview apiClient={apiClient} summary={summary} compactLayout={compactLayout} />;
  }

  if (previewKind === "table" || previewKind === "series") {
    return <Hdf5TablePreview apiClient={apiClient} summary={summary} compactLayout={compactLayout} />;
  }

  return (
    <div className="viewer-empty">
      This dataset is currently best represented as metadata. Structured preview surfaces remain intentionally disabled.
    </div>
  );
}
