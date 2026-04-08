import { type ReactNode, useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ApiClient } from "@/lib/api";
import { writeClipboardText } from "@/lib/clipboard";
import type { Hdf5DatasetSummary, UploadViewerInfo } from "@/types";

import {
  buildSampleCoverage,
  describeGeometryProvenance,
  formatByteEstimate,
  formatCount,
  formatPathValue,
  formatSampleValue,
  formatSummaryToken,
} from "./formatters";
import { Hdf5DatasetPreview } from "./Hdf5DatasetPreview";

type Hdf5InspectorProps = {
  apiClient: ApiClient;
  summary: Hdf5DatasetSummary | null;
  activeDatasetPath: string | null;
  hdf5: NonNullable<UploadViewerInfo["hdf5"]>;
  loading: boolean;
  error: string | null;
};

function DetailSection({
  title,
  children,
  dataId,
}: {
  title: string;
  children: ReactNode;
  dataId?: string;
}) {
  return (
    <section className="viewer-hdf-detail-section" data-hdf5-detail-section={dataId}>
      <div className="viewer-hdf-detail-section-heading">
        <strong>{title}</strong>
      </div>
      {children}
    </section>
  );
}

export function Hdf5Inspector({
  apiClient,
  summary,
  activeDatasetPath,
  hdf5,
  loading,
  error,
}: Hdf5InspectorProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const timeout = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timeout);
  }, [copied]);

  return (
    <Card
      className="viewer-hdf-dashboard-card viewer-hdf-inspector-card gap-0 py-0 shadow-none"
      data-hdf5-inspector="true"
    >
      <CardHeader className="viewer-hdf-dashboard-header viewer-hdf-dashboard-header-split">
        <div>
          <CardTitle>Dataset details</CardTitle>
          <CardDescription>
            {summary
              ? "Keep the selected dataset context, sampling limits, and metadata readable while you explore."
              : "Choose a dataset to inspect structure, geometry, and sampling context."}
          </CardDescription>
        </div>
        {activeDatasetPath ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              void writeClipboardText(activeDatasetPath).then(() => setCopied(true));
            }}
          >
            {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
            {copied ? "Copied" : "Copy path"}
          </Button>
        ) : null}
      </CardHeader>

      <CardContent className="viewer-hdf-dashboard-content viewer-hdf-inspector-content">
        {!activeDatasetPath ? (
          <div className="viewer-empty viewer-hdf-empty-state">
            Select a dataset to inspect its structure, sampling context, and file metadata.
          </div>
        ) : loading && !summary ? (
          <div className="viewer-empty viewer-hdf-empty-state">Loading dataset details…</div>
        ) : error && !summary ? (
          <div className="viewer-metadata-note">
            <strong>Dataset load failed</strong>
            <span>{error}</span>
          </div>
        ) : summary ? (
          <>
            {summary.volume_reason ? (
              <div className="viewer-hdf-provenance-note">
                <strong>{summary.volume_eligible ? "Native 3D policy" : "Slice-only reason"}</strong>
                <span>{summary.volume_reason}</span>
              </div>
            ) : null}

            <Tabs key={summary.dataset_path} defaultValue="preview" className="viewer-hdf-detail-tabs">
              <TabsList className="viewer-hdf-detail-tabs-list">
                <TabsTrigger value="preview">Preview</TabsTrigger>
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="sampling">Sampling</TabsTrigger>
                <TabsTrigger value="metadata">Metadata</TabsTrigger>
                <TabsTrigger value="file">File</TabsTrigger>
              </TabsList>

              <TabsContent
                value="preview"
                className="viewer-hdf-detail-tab viewer-hdf-detail-tab-preview"
                data-hdf5-preview-workspace="true"
              >
                <Hdf5DatasetPreview apiClient={apiClient} summary={summary} />
              </TabsContent>

              <TabsContent value="overview" className="viewer-hdf-detail-tab">
                <ScrollArea className="viewer-hdf-scroll-area viewer-hdf-detail-scroll" data-hdf5-scroll-region="inspector">
                  <div className="viewer-hdf-detail-stack">
                    <DetailSection title="Dataset overview" dataId="overview">
                      <dl className="viewer-metadata-list">
                        <div className="viewer-metadata-row">
                          <dt>Dataset path</dt>
                          <dd>{summary.dataset_path}</dd>
                        </div>
                        <div className="viewer-metadata-row">
                          <dt>Shape</dt>
                          <dd>{summary.shape.length > 0 ? summary.shape.join(" x ") : "Scalar"}</dd>
                        </div>
                        <div className="viewer-metadata-row">
                          <dt>Total values</dt>
                          <dd>{formatCount(summary.element_count)}</dd>
                        </div>
                        <div className="viewer-metadata-row">
                          <dt>Estimated size</dt>
                          <dd>{formatByteEstimate(summary.estimated_bytes)}</dd>
                        </div>
                        {summary.capabilities.length > 0 ? (
                          <div className="viewer-metadata-row">
                            <dt>Preview capabilities</dt>
                            <dd>{summary.capabilities.map((capability) => formatSummaryToken(capability)).join(" • ")}</dd>
                          </div>
                        ) : null}
                        <div className="viewer-metadata-row">
                          <dt>Native 3D</dt>
                          <dd>{summary.volume_eligible ? "Ready" : summary.volume_reason ?? "Slice-only"}</dd>
                        </div>
                        {summary.dimension_summary && Object.keys(summary.dimension_summary).length > 0 ? (
                          <div className="viewer-metadata-row">
                            <dt>Dimension summary</dt>
                            <dd>
                              {Object.entries(summary.dimension_summary)
                                .map(([key, value]) => `${key}=${formatCount(value)}`)
                                .join(" • ")}
                            </dd>
                          </div>
                        ) : null}
                      </dl>
                    </DetailSection>

                    {summary.structured_fields.length > 0 ? (
                      <DetailSection title="Structured fields" dataId="structured-fields">
                        <div className="viewer-hdf-chip-grid">
                          {summary.structured_fields.map((field) => (
                            <div key={`${field.name}:${field.dtype}`} className="viewer-hdf-chip-card">
                              <strong>{field.name}</strong>
                              <span>{field.dtype}</span>
                            </div>
                          ))}
                        </div>
                      </DetailSection>
                    ) : null}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="sampling" className="viewer-hdf-detail-tab">
                <ScrollArea className="viewer-hdf-scroll-area viewer-hdf-detail-scroll">
                  <div className="viewer-hdf-detail-stack">
                    <div className="viewer-hdf-provenance-note" data-hdf5-sample-provenance="true">
                      <strong>Bounded preview sample</strong>
                      <span>
                        These values and statistics are preview-sized so scientists can browse quickly before running a full analysis.
                      </span>
                    </div>

                    <DetailSection title="Sampling summary" dataId="sampling">
                      {summary.sample_statistics ? (
                        <dl className="viewer-metadata-list">
                          <div className="viewer-metadata-row">
                            <dt>Sample coverage</dt>
                            <dd>{buildSampleCoverage(summary) ?? formatCount(summary.sample_statistics.sample_count)}</dd>
                          </div>
                          <div className="viewer-metadata-row">
                            <dt>Minimum</dt>
                            <dd>{formatPathValue(summary.sample_statistics.min)}</dd>
                          </div>
                          <div className="viewer-metadata-row">
                            <dt>Maximum</dt>
                            <dd>{formatPathValue(summary.sample_statistics.max)}</dd>
                          </div>
                          <div className="viewer-metadata-row">
                            <dt>Mean</dt>
                            <dd>{formatPathValue(summary.sample_statistics.mean)}</dd>
                          </div>
                          {summary.sample_statistics.unique_values != null ? (
                            <div className="viewer-metadata-row">
                              <dt>Unique values in sample</dt>
                              <dd>{formatCount(summary.sample_statistics.unique_values)}</dd>
                            </div>
                          ) : null}
                        </dl>
                      ) : (
                        <div className="viewer-metadata-note">
                          <strong>Sample statistics unavailable</strong>
                          <span>This dataset does not expose numeric sample statistics in the current preview contract.</span>
                        </div>
                      )}
                    </DetailSection>

                    <DetailSection title="Sample values" dataId="sample-values">
                      <div className="viewer-hdf-sample-block">
                        <div className="viewer-hdf-sample-header">
                          <strong>Sample excerpt</strong>
                          <span>
                            {summary.sample_shape.length > 0
                              ? `sample shape ${summary.sample_shape.join(" x ")}`
                              : "scalar sample"}
                          </span>
                        </div>
                        <pre className="viewer-hdf-sample-value">{formatSampleValue(summary.sample_values)}</pre>
                      </div>
                    </DetailSection>
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="metadata" className="viewer-hdf-detail-tab">
                <ScrollArea className="viewer-hdf-scroll-area viewer-hdf-detail-scroll">
                  <div className="viewer-hdf-detail-stack">
                    {summary.geometry ? (
                      <DetailSection title="Geometry" dataId="geometry">
                        {describeGeometryProvenance(summary) ? (
                          <div className="viewer-hdf-provenance-note" data-hdf5-provenance-panel="true">
                            <strong>{describeGeometryProvenance(summary)?.label}</strong>
                            <span>{describeGeometryProvenance(summary)?.detail}</span>
                          </div>
                        ) : null}
                        <dl className="viewer-metadata-list">
                          {summary.geometry.path ? (
                            <div className="viewer-metadata-row">
                              <dt>Geometry path</dt>
                              <dd>{summary.geometry.path}</dd>
                            </div>
                          ) : null}
                          {summary.geometry.dimensions?.length ? (
                            <div className="viewer-metadata-row">
                              <dt>Dimensions</dt>
                              <dd>{summary.geometry.dimensions.join(" x ")}</dd>
                            </div>
                          ) : null}
                          {summary.geometry.spacing?.length ? (
                            <div className="viewer-metadata-row">
                              <dt>Spacing</dt>
                              <dd>{summary.geometry.spacing.join(" x ")}</dd>
                            </div>
                          ) : null}
                          {summary.geometry.origin?.length ? (
                            <div className="viewer-metadata-row">
                              <dt>Origin</dt>
                              <dd>{summary.geometry.origin.join(" x ")}</dd>
                            </div>
                          ) : null}
                        </dl>
                      </DetailSection>
                    ) : null}

                    {Object.keys(summary.attributes ?? {}).length > 0 ? (
                      <DetailSection title="Attributes" dataId="attributes">
                        <dl className="viewer-metadata-list">
                          {Object.entries(summary.attributes).map(([key, value]) => (
                            <div key={key} className="viewer-metadata-row">
                              <dt>{key}</dt>
                              <dd>{formatPathValue(value)}</dd>
                            </div>
                          ))}
                        </dl>
                      </DetailSection>
                    ) : (
                      <div className="viewer-metadata-note">
                        <strong>No dataset attributes</strong>
                        <span>This dataset does not expose additional attributes in the current preview contract.</span>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="file" className="viewer-hdf-detail-tab">
                <ScrollArea className="viewer-hdf-scroll-area viewer-hdf-detail-scroll">
                  <div className="viewer-hdf-detail-stack">
                    <DetailSection title="File context" dataId="file-context">
                      <dl className="viewer-metadata-list">
                        {hdf5.default_dataset_path ? (
                          <div className="viewer-metadata-row">
                            <dt>Suggested dataset</dt>
                            <dd>{hdf5.default_dataset_path}</dd>
                          </div>
                        ) : null}
                        <div className="viewer-metadata-row">
                          <dt>Groups</dt>
                          <dd>{formatCount(hdf5.summary.group_count)}</dd>
                        </div>
                        <div className="viewer-metadata-row">
                          <dt>Datasets</dt>
                          <dd>{formatCount(hdf5.summary.dataset_count)}</dd>
                        </div>
                        {Object.keys(hdf5.summary.dataset_kinds ?? {}).length > 0 ? (
                          <div className="viewer-metadata-row">
                            <dt>Detected dataset kinds</dt>
                            <dd>
                              {Object.entries(hdf5.summary.dataset_kinds)
                                .filter(([, count]) => Number(count) > 0)
                                .map(([label, count]) => `${label} (${count})`)
                                .join(" • ")}
                            </dd>
                          </div>
                        ) : null}
                        {Object.keys(hdf5.root_attributes ?? {}).length > 0 ? (
                          <div className="viewer-metadata-row">
                            <dt>Root attributes</dt>
                            <dd>
                              {Object.entries(hdf5.root_attributes)
                                .slice(0, 8)
                                .map(([key, value]) => `${key}=${formatPathValue(value)}`)
                                .join(" • ")}
                            </dd>
                          </div>
                        ) : null}
                        {hdf5.limitations.length > 0 ? (
                          <div className="viewer-metadata-row">
                            <dt>Current limitations</dt>
                            <dd>{hdf5.limitations.join(" ")}</dd>
                          </div>
                        ) : null}
                      </dl>
                    </DetailSection>
                  </div>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </>
        ) : (
          <div className="viewer-empty viewer-hdf-empty-state">
            Select a dataset to inspect its structure, sampling context, and file metadata.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
