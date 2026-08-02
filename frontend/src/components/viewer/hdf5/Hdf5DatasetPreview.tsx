import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { Select as SelectPrimitive } from "radix-ui";
import { Bar, BarChart, CartesianGrid, Scatter, ScatterChart, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Select,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type {
  Hdf5DatasetHistogramResponse,
  Hdf5DatasetSummary,
  Hdf5DatasetTablePreviewResponse,
} from "@/types";
import { canonicalizeHdf5FeatureIds, type ApiClient } from "@/lib/api";

import { SlicePlaneCanvas } from "../SlicePlaneCanvas";
import { SliceStackVolumeCanvas } from "../SliceStackVolumeCanvas";
import { useHdf5OverlayContainer } from "./Hdf5OverlayContainer";

type Hdf5DatasetPreviewProps = {
  apiClient: ApiClient;
  summary: Hdf5DatasetSummary;
  compactLayout?: boolean;
  featureSelection?: Hdf5FeatureSelectionState | null;
  onFeatureSelectionChange?: (selection: Hdf5FeatureSelectionState) => void;
};

export type Hdf5FeatureSelectionState = {
  fileId: string;
  registrationKey: string;
  appliedFeatureIds: string[];
  draftFeatureIds: string;
  error: string | null;
};

type Hdf5FeatureSelectionProps = {
  selectedFeatureIds: string[];
  manualFeatureId: string;
  featureSelectionError: string | null;
  onManualFeatureIdChange: (value: string) => void;
  onApplyFeatureIds: (values: readonly string[]) => void;
  onRemoveFeatureId: (value: string) => void;
  onClearFeatureIds: () => void;
};

type Hdf5VolumeSource = NonNullable<Parameters<typeof SliceStackVolumeCanvas>[0]["volumeSource"]>;

type HistogramPreviewState = {
  key: string;
  status: "idle" | "loading" | "success" | "error";
  histogram: Hdf5DatasetHistogramResponse | null;
  error: string | null;
};

type TablePreviewState = {
  key: string;
  preview: Hdf5DatasetTablePreviewResponse | null;
  error: string | null;
};

const HISTOGRAM_CHART_CONFIG = {
  count: { label: "Count", color: "var(--chart-2)" },
};

const SCATTER_CHART_CONFIG = {
  value: { label: "Value", color: "var(--chart-1)" },
};

const VOLUME_PREVIEW_KINDS = new Set(["scalar_volume", "label_volume", "rgb_volume", "vector_volume"]);
const NO_FEATURE_IDS: string[] = [];

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

const canRenderNativeVolume = (summary: Hdf5DatasetSummary): boolean =>
  Boolean(
    summary.volume_eligible &&
      summary.capabilities.includes("volume") &&
      summary.axis_sizes &&
      summary.preview_planes.z &&
      (summary.render_policy === "scalar" || summary.atlas_scheme)
  );

function Hdf5SelectContent({ children }: { children: ReactNode }) {
  const overlayContainer = useHdf5OverlayContainer();
  return (
    <SelectPrimitive.Portal container={overlayContainer}>
      <SelectPrimitive.Content
        data-slot="select-content"
        data-hdf5-overlay="select"
        position="item-aligned"
        className="viewer-hdf-select-content"
      >
        <SelectPrimitive.Viewport className="viewer-hdf-select-viewport">
          {children}
        </SelectPrimitive.Viewport>
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  );
}

function Hdf5VolumePreview({
  apiClient,
  summary,
  compactLayout = false,
  selectedFeatureIds,
  manualFeatureId,
  featureSelectionError,
  onManualFeatureIdChange,
  onApplyFeatureIds,
  onRemoveFeatureId,
  onClearFeatureIds,
}: Hdf5DatasetPreviewProps & Hdf5FeatureSelectionProps) {
  const availableAxes = useMemo(
    () => (summary.slice_axes.length > 0 ? summary.slice_axes : (["z"] as Array<"z" | "y" | "x">)),
    [summary.slice_axes]
  );
  const [selectedAxis, setSelectedAxis] = useState<"z" | "y" | "x">(availableAxes[0] ?? "z");
  const [selectedComponent, setSelectedComponent] = useState(0);
  const canRenderVolume = canRenderNativeVolume(summary);
  const isCategoricalVolume = summary.preview_kind === "label_volume";
  const categoricalDepth = axisSize(summary, "z");
  const [categoricalMode, setCategoricalMode] = useState<"surface" | "xray">("surface");
  const [categoricalCutaway, setCategoricalCutaway] = useState(false);
  const [categoricalDepthIndex, setCategoricalDepthIndex] = useState(
    Math.floor((categoricalDepth - 1) / 2)
  );
  const [featureFilterOpen, setFeatureFilterOpen] = useState(Boolean(featureSelectionError));
  const featureFilterDescriptionId = useId();
  const featureFilterErrorId = useId();
  const [selectedTab, setSelectedTab] = useState<"volume" | "visual" | "distribution">("visual");
  const [histogramState, setHistogramState] = useState<HistogramPreviewState>({
    key: "",
    status: "idle",
    histogram: null,
    error: null,
  });
  const [histogramCache, setHistogramCache] = useState(
    new Map<string, { histogram: Hdf5DatasetHistogramResponse | null; error: string | null }>()
  );
  const histogramRequestsRef = useRef(new Map<string, Promise<Hdf5DatasetHistogramResponse>>());
  const histogramKeyRef = useRef("");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const maxIndex = axisSize(summary, selectedAxis);
  const [selectedIndex, setSelectedIndex] = useState(Math.max(0, Math.floor(maxIndex / 2)));
  const handleSelectedAxisChange = (axis: "z" | "y" | "x") => {
    setSelectedAxis(axis);
    setSelectedIndex(Math.max(0, Math.floor(axisSize(summary, axis) / 2)));
  };

  const componentCount = Math.max(1, Number(summary.component_count || 1));
  const componentLabels =
    summary.component_labels.length > 0
      ? summary.component_labels
      : Array.from({ length: componentCount }, (_, index) => `component_${index + 1}`);
  const activeComponent = Math.max(0, Math.min(selectedComponent, componentCount - 1));
  const histogramRequestKey = summary.capabilities.includes("histogram")
    ? [
        summary.file_id,
        summary.dataset_path,
        summary.preview_kind === "vector_volume" ? activeComponent : "scalar",
      ].join("\u0000")
    : "";
  const cachedHistogramState = histogramCache.get(histogramRequestKey);
  const currentHistogramState =
    histogramState.key === histogramRequestKey
      ? histogramState
      : cachedHistogramState
        ? {
            key: histogramRequestKey,
            status: cachedHistogramState.error ? ("error" as const) : ("success" as const),
            ...cachedHistogramState,
          }
        : {
            key: histogramRequestKey,
            status: "idle" as const,
            histogram: null,
            error: null,
          };
  const histogram = currentHistogramState.histogram;
  const histogramError = currentHistogramState.error;
  const histogramLoading = Boolean(
    selectedTab === "distribution" &&
      histogramRequestKey &&
      (currentHistogramState.status === "idle" || currentHistogramState.status === "loading")
  );
  const activePlane = summary.preview_planes[selectedAxis];
  const previewUrl = useMemo(
    () =>
      apiClient.hdf5SlicePreviewUrl(summary.file_id, {
        datasetPath: summary.dataset_path,
        axis: selectedAxis,
        index: selectedIndex,
        component: summary.preview_kind === "vector_volume" ? activeComponent : undefined,
        featureIds: selectedFeatureIds,
      }),
    [apiClient, activeComponent, selectedAxis, selectedFeatureIds, selectedIndex, summary]
  );
  const volumeFallbackUrl = useMemo(
    () =>
      apiClient.hdf5SlicePreviewUrl(summary.file_id, {
        datasetPath: summary.dataset_path,
        axis: "z",
        index: Math.max(0, Math.floor(axisSize(summary, "z") / 2)),
        featureIds: selectedFeatureIds,
      }),
    [apiClient, selectedFeatureIds, summary]
  );
  const hdf5VolumeSource = useMemo<Hdf5VolumeSource | null>(() => {
    if (!canRenderVolume || !summary.axis_sizes || !summary.preview_planes.z) {
      return null;
    }
    if (summary.render_policy === "scalar") {
      return {
        kind: "scalar",
        loadScalarVolume: (signal?: AbortSignal) =>
          apiClient.getHdf5ScalarVolume(summary.file_id, {
            datasetPath: summary.dataset_path,
            signal,
          }),
        fallbackImageUrl: volumeFallbackUrl,
        axisSizes: summary.axis_sizes,
        plane: summary.preview_planes.z,
        physicalSpacing: summary.physical_spacing ?? null,
        renderPolicy: summary.render_policy,
        texturePolicy: summary.texture_policy,
      };
    }
    if (!summary.atlas_scheme) {
      return null;
    }
    return {
      kind: "atlas",
      atlasUrl: apiClient.hdf5AtlasPreviewUrl(summary.file_id, {
        datasetPath: summary.dataset_path,
        component: summary.preview_kind === "vector_volume" ? activeComponent : undefined,
        featureIds: selectedFeatureIds,
      }),
      fallbackImageUrl: volumeFallbackUrl,
      atlasScheme: summary.atlas_scheme,
      axisSizes: summary.axis_sizes,
      plane: summary.preview_planes.z,
      physicalSpacing: summary.physical_spacing ?? null,
      renderPolicy: summary.render_policy,
      texturePolicy: selectedFeatureIds.length > 0 ? "nearest" : summary.texture_policy,
    };
  }, [
    apiClient,
    activeComponent,
    canRenderVolume,
    summary.atlas_scheme,
    summary.axis_sizes,
    summary.dataset_path,
    summary.file_id,
    summary.physical_spacing,
    summary.preview_planes.z,
    summary.preview_kind,
    summary.render_policy,
    summary.texture_policy,
    selectedFeatureIds,
    volumeFallbackUrl,
  ]);
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
          <Hdf5SelectContent>
            {componentLabels.map((label, index) => (
              <SelectItem key={`${label}:${index}`} value={String(index)}>
                {label}
              </SelectItem>
            ))}
          </Hdf5SelectContent>
        </Select>
      </label>
    ) : null;

  useEffect(() => {
    histogramKeyRef.current = histogramRequestKey;
  }, [histogramRequestKey]);

  useEffect(() => {
    if (selectedTab !== "distribution" || !histogramRequestKey) {
      return;
    }
    const cached = histogramCache.get(histogramRequestKey);
    if (cached) {
      return;
    }
    if (histogramRequestsRef.current.has(histogramRequestKey)) {
      return;
    }
    const request = apiClient.getHdf5DatasetHistogram(summary.file_id, summary.dataset_path, {
      component: summary.preview_kind === "vector_volume" ? activeComponent : undefined,
      bins: 24,
    });
    histogramRequestsRef.current.set(histogramRequestKey, request);
    void request
      .then((response) => {
        const cachedResponse = { histogram: response, error: null };
        setHistogramCache((current) => new Map(current).set(histogramRequestKey, cachedResponse));
        if (mountedRef.current && histogramKeyRef.current === histogramRequestKey) {
          setHistogramState({
            key: histogramRequestKey,
            status: "success",
            ...cachedResponse,
          });
        }
      })
      .catch((error: unknown) => {
        const cachedError = {
          histogram: null,
          error: error instanceof Error ? error.message : "Failed to load histogram preview.",
        };
        setHistogramCache((current) => new Map(current).set(histogramRequestKey, cachedError));
        if (mountedRef.current && histogramKeyRef.current === histogramRequestKey) {
          setHistogramState({
            key: histogramRequestKey,
            status: "error",
            ...cachedError,
          });
        }
      })
      .finally(() => {
        histogramRequestsRef.current.delete(histogramRequestKey);
      });
  }, [
    activeComponent,
    apiClient,
    histogramCache,
    histogramRequestKey,
    selectedTab,
    summary.dataset_path,
    summary.file_id,
    summary.preview_kind,
  ]);

  const renderFeatureFilter = () =>
    summary.feature_filter ? (
      <Collapsible
        className="viewer-hdf-feature-filter"
        data-hdf5-feature-filter="true"
        open={featureFilterOpen || Boolean(featureSelectionError)}
        onOpenChange={(open) => {
          if (!open && featureSelectionError) {
            return;
          }
          setFeatureFilterOpen(open);
        }}
      >
        <div className="viewer-hdf-feature-filter-heading">
          <CollapsibleTrigger asChild>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="viewer-hdf-feature-filter-trigger"
              aria-label={
                selectedFeatureIds.length > 0
                  ? `Filter grains, ${selectedFeatureIds.length.toLocaleString()} selected`
                  : "Filter grains"
              }
            >
              Filter grains
              <ChevronDown data-icon="inline-end" aria-hidden="true" />
            </Button>
          </CollapsibleTrigger>
          {selectedFeatureIds.length > 0 ? (
            <span className="viewer-hdf-feature-filter-count" aria-live="polite">
              {selectedFeatureIds.length.toLocaleString()} selected
            </span>
          ) : null}
          {selectedFeatureIds.length > 0 ? (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              aria-label="Clear grain filter"
              onClick={onClearFeatureIds}
            >
              Clear
            </Button>
          ) : null}
        </div>
        <CollapsibleContent className="viewer-hdf-feature-filter-content">
          <p id={featureFilterDescriptionId} className="viewer-hdf-feature-filter-helper">
            Raw Feature IDs; background 0 excluded.
          </p>
          <form
            className="viewer-hdf-feature-filter-form"
            onSubmit={(event) => {
              event.preventDefault();
              onApplyFeatureIds(manualFeatureId.split(","));
            }}
          >
            <input
              type="text"
              inputMode="text"
              value={manualFeatureId}
              aria-label="Feature IDs"
              aria-invalid={featureSelectionError ? "true" : undefined}
              aria-describedby={
                featureSelectionError
                  ? `${featureFilterDescriptionId} ${featureFilterErrorId}`
                  : featureFilterDescriptionId
              }
              placeholder="e.g. 7, 25"
              onChange={(event) => {
                setFeatureFilterOpen(true);
                onManualFeatureIdChange(event.currentTarget.value);
              }}
            />
            <Button type="submit" size="sm" variant="outline" disabled={!manualFeatureId.trim()}>
              Apply
            </Button>
          </form>
          {featureSelectionError ? (
            <p id={featureFilterErrorId} className="viewer-hdf-feature-filter-error" role="alert">
              {featureSelectionError}
            </p>
          ) : null}
          {selectedFeatureIds.length > 0 ? (
            <div className="viewer-hdf-feature-id-list" aria-label="Selected Feature IDs">
              {selectedFeatureIds.map((featureId) => (
                <button
                  key={featureId}
                  type="button"
                  aria-label={`Remove Feature ID ${featureId}`}
                  onClick={() => onRemoveFeatureId(featureId)}
                >
                  ID {featureId}<span aria-hidden="true">×</span>
                </button>
              ))}
            </div>
          ) : null}
        </CollapsibleContent>
      </Collapsible>
    ) : null;

  const renderPreviewToolbarActions = () => (
    <div className="viewer-hdf-preview-toolbar-actions">
      {renderComponentField()}
      {renderFeatureFilter()}
    </div>
  );

  const hasToolbarContent =
    previewTabCount > 1 || summary.preview_kind === "vector_volume" || Boolean(summary.feature_filter);

  const renderCompactToolbar = () =>
    compactLayout && hasToolbarContent ? (
      <div className="viewer-hdf-preview-compact-toolbar">
        {renderPreviewTabsList()}
        {renderPreviewToolbarActions()}
      </div>
    ) : null;

  return (
    <div className="viewer-hdf-preview-body" data-hdf5-preview-kind={summary.preview_kind ?? "unknown"}>
      <Tabs
        value={selectedTab}
        onValueChange={(value) => setSelectedTab(value as "volume" | "visual" | "distribution")}
        className={`viewer-hdf-preview-tabs${compactLayout ? " viewer-hdf-preview-tabs-compact" : ""}`}
      >
        {compactLayout
          ? renderCompactToolbar()
          : hasToolbarContent ? (
              <div className="viewer-hdf-preview-toolbar">
                {renderPreviewTabsList()}
                {renderPreviewToolbarActions()}
              </div>
            ) : null}

        {canRenderVolume && hdf5VolumeSource ? (
          <TabsContent value="volume" className="viewer-hdf-preview-tab">
            {isCategoricalVolume ? (
              <div
                className="viewer-hdf-categorical-controls"
                role="group"
                aria-label="Categorical volume rendering"
              >
                <div className="viewer-hdf-categorical-modes">
                  <Button
                    type="button"
                    size="sm"
                    variant={categoricalMode === "surface" ? "secondary" : "outline"}
                    aria-pressed={categoricalMode === "surface"}
                    onClick={() => setCategoricalMode("surface")}
                  >
                    Surface
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={categoricalMode === "xray" ? "secondary" : "outline"}
                    aria-pressed={categoricalMode === "xray"}
                    onClick={() => setCategoricalMode("xray")}
                  >
                    X-ray
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={categoricalCutaway ? "secondary" : "outline"}
                    aria-pressed={categoricalCutaway}
                    onClick={() => setCategoricalCutaway((active) => !active)}
                  >
                    Cutaway
                  </Button>
                </div>
                {categoricalMode === "xray" ? (
                  <p className="viewer-hdf-categorical-caveat" role="note">
                    X-ray blends labels for depth context; blended colors do not represent feature IDs.
                  </p>
                ) : null}
                {categoricalCutaway ? (
                  <label className="viewer-hdf-categorical-depth">
                    <span className="viewer-hdf-categorical-depth-label">
                      <strong>Z depth</strong>
                      <span>
                        Preview-grid Z {categoricalDepthIndex + 1} of {categoricalDepth}
                      </span>
                    </span>
                    <input
                      type="range"
                      className="viewer-hdf-slider"
                      aria-label="Cutaway Z depth"
                      aria-valuetext={`Preview-grid Z ${categoricalDepthIndex + 1} of ${categoricalDepth}`}
                      min={0}
                      max={Math.max(0, categoricalDepth - 1)}
                      step={1}
                      value={categoricalDepthIndex}
                      onChange={(event) => setCategoricalDepthIndex(Number(event.currentTarget.value))}
                    />
                    <small>Depth follows the bounded preview grid, not native source topology.</small>
                  </label>
                ) : null}
              </div>
            ) : null}
            <div className="viewer-hdf-slice-shell" data-hdf5-volume-preview="true">
              <SliceStackVolumeCanvas
                volumeSource={hdf5VolumeSource}
                categoricalMode={isCategoricalVolume ? categoricalMode : undefined}
                volumeCutaway={isCategoricalVolume ? categoricalCutaway : undefined}
                zIndex={isCategoricalVolume ? categoricalDepthIndex : undefined}
                featureMask={selectedFeatureIds.length > 0}
                cameraPersistenceKey={`${summary.file_id}:${summary.dataset_path}`}
                className="viewer-canvas-root viewer-hdf-slice-canvas"
              />
            </div>
          </TabsContent>
        ) : null}

        <TabsContent value="visual" className="viewer-hdf-preview-tab">
          <div className="viewer-hdf-slice-layout">
            <div className="viewer-hdf-slice-sidebar">
              <div className="viewer-hdf-preview-controls">
                <div className="viewer-hdf-axis-toggle" role="group" aria-label="Slice orientation">
                  {availableAxes.map((axis) => (
                    <Button
                      key={axis}
                      type="button"
                      size="sm"
                      variant={selectedAxis === axis ? "secondary" : "outline"}
                      aria-pressed={selectedAxis === axis}
                      onClick={() => handleSelectedAxisChange(axis)}
                    >
                      {axisLabel(axis)}
                    </Button>
                  ))}
                </div>
              </div>

              {!canRenderVolume && summary.volume_reason ? (
                <p className="viewer-hdf-detail-caption" data-hdf5-slice-only="true">
                  {summary.volume_reason}
                </p>
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
                    <input
                      type="range"
                      className="viewer-hdf-slider"
                      aria-label={`${activePlane?.label ?? `${axisLabel(selectedAxis)} plane`} slice`}
                      aria-valuetext={`Slice ${selectedIndex + 1} of ${maxIndex}`}
                      min={0}
                      max={Math.max(0, maxIndex - 1)}
                      step={1}
                      value={selectedIndex}
                      onChange={(event) => setSelectedIndex(Number(event.currentTarget.value))}
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
            <section className="viewer-hdf-chart-card" data-hdf5-histogram="true">
              <div className="viewer-hdf-tree-header">
                <strong>Sampled distribution</strong>
                <span>
                  {histogram?.component_label ? `${histogram.component_label} • ` : ""}
                  {buildSampleSummary(histogram?.sample_count, summary.element_count)}
                </span>
              </div>
              {selectedFeatureIds.length > 0 ? (
                <p className="viewer-hdf-histogram-filter-note" role="note">
                  Histogram values remain an unfiltered bounded sample of the full dataset.
                </p>
              ) : null}
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
  const [tableState, setTableState] = useState<TablePreviewState>({
    key: "",
    preview: null,
    error: null,
  });
  const tableRequestKey = [summary.file_id, summary.dataset_path, offset].join("\u0000");
  const currentTableState =
    tableState.key === tableRequestKey
      ? tableState
      : { key: tableRequestKey, preview: null, error: null };
  const tablePreview = currentTableState.preview;
  const tableError = currentTableState.error;
  const tableLoading = tableState.key !== tableRequestKey;

  useEffect(() => {
    let cancelled = false;
    apiClient
      .getHdf5DatasetTablePreview(summary.file_id, summary.dataset_path, { offset, limit: 12 })
      .then((response) => {
        if (cancelled) {
          return;
        }
        setTableState({ key: tableRequestKey, preview: response, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setTableState({
          key: tableRequestKey,
          preview: null,
          error: error instanceof Error ? error.message : "Failed to load table preview.",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient, offset, summary.dataset_path, summary.file_id, tableRequestKey]);

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

export function Hdf5DatasetPreview({
  apiClient,
  summary,
  compactLayout = false,
  featureSelection,
  onFeatureSelectionChange,
}: Hdf5DatasetPreviewProps) {
  const [localFeatureSelection, setLocalFeatureSelection] = useState<Hdf5FeatureSelectionState | null>(null);
  const storedSelection = featureSelection === undefined ? localFeatureSelection : featureSelection;
  const registrationKey = summary.feature_filter?.registration_key ?? "";
  const selectionMatches = Boolean(
    storedSelection &&
      storedSelection.fileId === summary.file_id &&
      storedSelection.registrationKey === registrationKey &&
      registrationKey
  );
  const activeSelection: Hdf5FeatureSelectionState = selectionMatches && storedSelection
    ? storedSelection
    : {
        fileId: summary.file_id,
        registrationKey,
        appliedFeatureIds: NO_FEATURE_IDS,
        draftFeatureIds: "",
        error: null,
      };
  const commitSelection = (next: Hdf5FeatureSelectionState): void => {
    if (onFeatureSelectionChange) {
      onFeatureSelectionChange(next);
    } else {
      setLocalFeatureSelection(next);
    }
  };
  const applyFeatureIds = (values: readonly string[]) => {
    try {
      const tokens = values.map((value) => value.trim());
      if (tokens.length === 0 || tokens.some((value) => !value)) {
        throw new RangeError("Enter one or more comma-separated Feature IDs.");
      }
      const nextIds = canonicalizeHdf5FeatureIds(
        [...activeSelection.appliedFeatureIds, ...tokens],
        summary.feature_filter?.max_ids ?? 64
      );
      const currentIds = activeSelection.appliedFeatureIds;
      const unchanged =
        currentIds.length === nextIds.length && currentIds.every((value, index) => value === nextIds[index]);
      if (unchanged && !activeSelection.error) {
        if (activeSelection.draftFeatureIds) {
          commitSelection({ ...activeSelection, appliedFeatureIds: currentIds, draftFeatureIds: "" });
        }
        return;
      }
      commitSelection({
        ...activeSelection,
        appliedFeatureIds: unchanged ? currentIds : nextIds,
        draftFeatureIds: "",
        error: null,
      });
    } catch (error) {
      commitSelection({
        ...activeSelection,
        error: error instanceof Error ? error.message : "Invalid Feature ID.",
      });
    }
  };
  const featureSelectionProps: Hdf5FeatureSelectionProps = {
    selectedFeatureIds: activeSelection.appliedFeatureIds,
    manualFeatureId: activeSelection.draftFeatureIds,
    featureSelectionError: activeSelection.error,
    onManualFeatureIdChange: (value) => {
      commitSelection({ ...activeSelection, draftFeatureIds: value, error: null });
    },
    onApplyFeatureIds: applyFeatureIds,
    onRemoveFeatureId: (value) => {
      commitSelection({
        ...activeSelection,
        appliedFeatureIds: activeSelection.appliedFeatureIds.filter((featureId) => featureId !== value),
        error: null,
      });
    },
    onClearFeatureIds: () => {
      if (activeSelection.appliedFeatureIds.length === 0 && !activeSelection.error) {
        return;
      }
      commitSelection({ ...activeSelection, appliedFeatureIds: NO_FEATURE_IDS, error: null });
    },
  };
  const previewKind = summary.preview_kind ?? "unknown";
  const previewKey = [
    summary.file_id,
    summary.dataset_path,
    registrationKey || "unregistered",
    previewKind,
    summary.slice_axes.join(","),
    canRenderNativeVolume(summary) ? "volume" : "slice",
    compactLayout ? "compact" : "full",
  ].join(":");

  if (VOLUME_PREVIEW_KINDS.has(previewKind)) {
    return (
      <Hdf5VolumePreview
        key={previewKey}
        apiClient={apiClient}
        summary={summary}
        compactLayout={compactLayout}
        {...featureSelectionProps}
      />
    );
  }

  if (previewKind === "table" || previewKind === "series") {
    return <Hdf5TablePreview key={previewKey} apiClient={apiClient} summary={summary} compactLayout={compactLayout} />;
  }

  return (
    <div className="viewer-empty">
      This dataset is currently best represented as metadata. Structured preview surfaces remain intentionally disabled.
    </div>
  );
}
