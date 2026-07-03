import { Suspense, lazy, type ComponentType } from "react";
import { Download, ImageIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { openFigureLightbox, type LightboxFigure } from "@/lib/figureLightbox";

type LazyModule = Record<string, unknown>;

const lazyNamed = <TModule extends LazyModule>(
  loader: () => Promise<TModule>,
  exportName: keyof TModule
) =>
  lazy(async () => {
    const module = await loader();
    return {
      default: module[exportName] as ComponentType<any>,
    };
  });

const LazyToolResultQuickPreview = lazyNamed(
  () => import("./ToolResultQuickPreview"),
  "ToolResultQuickPreview"
);
const LazyToolImageCarousel = lazyNamed(
  () => import("./ToolImageCarousel"),
  "ToolImageCarousel"
);

export type ToolCardMetric = {
  label: string;
  value: string;
};

export type PrairieImageAnalysis = {
  rawFile: string;
  fileLabel?: string;
  prairieDogCount?: number | null;
  burrowCount?: number | null;
  boxCount?: number | null;
  nearestBurrowDistancePxMean?: number | null;
  nearestBurrowDistancePxMin?: number | null;
  nearestBurrowDistancePxMedian?: number | null;
  nearestBurrowDistancePxMax?: number | null;
  overlappingBurrowCount?: number | null;
  capturedAt?: string | null;
  latitude?: number | null;
  longitude?: number | null;
};

export type PrairieDetectionInsights = {
  summary?: string;
  inferenceBackend?: string | null;
  tileSize?: number | null;
  tileOverlap?: number | null;
  tileCount?: number | null;
  conf?: number | null;
  iou?: number | null;
  mergeIou?: number | null;
  prairieDogCount: number;
  burrowCount: number;
  avgConfidence?: number | null;
  nearestBurrowDistancePxMean?: number | null;
  nearestBurrowDistancePxMin?: number | null;
  overlapCount?: number | null;
  metadataSummary?: {
    capturedAt?: string | null;
    latitude?: number | null;
    longitude?: number | null;
  };
  perImage: PrairieImageAnalysis[];
};

export type ToolDetectionBox = {
  className: string;
  confidence?: number | null;
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
};

export type ToolImageHoverDetails = {
  fileLabel?: string;
  masksGenerated?: number | null;
  avgPointsPerWindow?: number | null;
  minPoints?: number | null;
  maxPoints?: number | null;
  detectionBoxes?: ToolDetectionBox[];
  prairieImageAnalysis?: PrairieImageAnalysis;
};

export type ToolCardImage = {
  path: string;
  url: string;
  title: string;
  sourceName: string;
  sourcePath?: string;
  previewable: boolean;
  downloadUrl?: string;
  linkedFileId?: string | null;
  resultGroupId?: string | null;
  hoverDetails?: ToolImageHoverDetails;
};

export type YoloFigureClassCount = {
  name: string;
  count: number;
};

export type YoloFigureCard = {
  key: string;
  title: string;
  subtitle?: string;
  previewUrl: string;
  downloadUrl?: string;
  originalUrl?: string;
  previewKind?: string;
  sourceName?: string;
  rawSourceName?: string;
  sourcePath?: string;
  rawSourcePath?: string;
  imageWidth?: number | null;
  imageHeight?: number | null;
  boxCount?: number | null;
  classCounts: YoloFigureClassCount[];
  previewable: boolean;
};

export type YoloFigureAvailability = {
  missingAnnotatedFigure: boolean;
};

export type ToolResourceRow = {
  name: string;
  owner?: string;
  created?: string;
  resourceType?: string;
  uri?: string;
  resourceUri?: string;
  clientViewUrl?: string;
  imageServiceUrl?: string;
};

export type ToolDownloadRow = {
  status: string;
  outputPath?: string;
  resourceUri?: string;
  clientViewUrl?: string;
  imageServiceUrl?: string;
  error?: string;
};

export type ToolResultCard = {
  id: string;
  tool:
    | "yolo_detect"
    | "estimate_depth_pro"
    | "upload_to_bisque"
    | "load_bisque_resource"
    | "bisque_download_resource"
    | "bisque_download_dataset"
    | "bisque_create_dataset"
    | "bisque_add_to_dataset"
    | "bisque_add_gobjects"
    | "add_tags_to_resource"
    | "bisque_fetch_xml"
    | "delete_bisque_resource"
    | "run_bisque_module"
    | "search_bisque_resources";
  title: string;
  subtitle?: string;
  metrics: ToolCardMetric[];
  classes: Array<{ name: string; count: number }>;
  images: ToolCardImage[];
  resourceRows: ToolResourceRow[];
  downloadRows: ToolDownloadRow[];
  variant?: "prairie_detection";
  narrative?: string;
  prairieInsights?: PrairieDetectionInsights | null;
  yoloFigures?: YoloFigureCard[];
  yoloFigureAvailability?: YoloFigureAvailability | null;
  placement?: "before_text" | "after_text";
};

type ToolResultCardSectionProps = {
  cards: ToolResultCard[];
  messageId: string;
  onImportBisqueResourcesIntoConversation: (
    resourcesToImport: string[],
    options?: {
      materialize?: boolean;
      persistSelectionContext?: boolean;
      source?: string;
      originatingMessageId?: string | null;
    }
  ) => Promise<unknown>;
  onCopyBisqueResourceUri: (resourceUri: string) => Promise<void>;
};

type BisqueResourceHeader = {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  summary?: string;
  hideMetricBadges?: boolean;
};

const resourceBackedBisqueCardTools = new Set<ToolResultCard["tool"]>([
  "upload_to_bisque",
  "load_bisque_resource",
  "search_bisque_resources",
  "bisque_create_dataset",
  "bisque_add_to_dataset",
  "bisque_add_gobjects",
  "add_tags_to_resource",
  "bisque_fetch_xml",
  "run_bisque_module",
]);

const pluralizeCount = (count: number, singular: string, plural?: string): string =>
  `${count} ${count === 1 ? singular : plural ?? `${singular}s`}`;

const parseLeadingMetricCount = (
  metrics: Array<{ label: string; value: string }>,
  label: string
): number | null => {
  const raw = metrics.find((metric) => metric.label === label)?.value ?? "";
  const match = raw.match(/^\s*(\d+)/);
  if (!match) {
    return null;
  }
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) ? value : null;
};

const parseFractionMetric = (
  metrics: Array<{ label: string; value: string }>,
  label: string
): { numerator: number; denominator: number } | null => {
  const raw = metrics.find((metric) => metric.label === label)?.value ?? "";
  const match = raw.match(/^\s*(\d+)\s*\/\s*(\d+)/);
  if (!match) {
    return null;
  }
  const numerator = Number.parseInt(match[1], 10);
  const denominator = Number.parseInt(match[2], 10);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) {
    return null;
  }
  return { numerator, denominator };
};

const normalizeBisqueServiceKind = (
  value: string | null | undefined
): "image" | "table" | "dataset" | "resource" => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (
    normalized === "image" ||
    normalized === "image_service" ||
    normalized === "file"
  ) {
    return "image";
  }
  if (normalized === "table") {
    return "table";
  }
  if (normalized === "dataset") {
    return "dataset";
  }
  return "resource";
};

const bisqueServiceTitleForKind = (
  kind: "image" | "table" | "dataset" | "resource"
): string => {
  switch (kind) {
    case "image":
      return "Image Service";
    case "table":
      return "Table Service";
    case "dataset":
      return "Dataset Catalog";
    default:
      return "Resource Catalog";
  }
};

const inferBisqueServiceKindFromCard = (
  card: ToolResultCard
): "image" | "table" | "dataset" | "resource" => {
  const preferredRowKind = card.resourceRows
    .map((row) => normalizeBisqueServiceKind(row.resourceType))
    .find((kind) => kind !== "resource");
  if (preferredRowKind) {
    return preferredRowKind;
  }
  if (card.images.length > 0) {
    return "image";
  }
  return "resource";
};

const buildBisqueResourceHeader = (card: ToolResultCard): BisqueResourceHeader | null => {
  const kind = inferBisqueServiceKindFromCard(card);
  const serviceTitle = bisqueServiceTitleForKind(kind);

  if (card.tool === "search_bisque_resources") {
    const matches = parseLeadingMetricCount(card.metrics, "Matches") ?? card.resourceRows.length;
    const metadataCount = parseLeadingMetricCount(card.metrics, "Metadata");
    const downloadFraction = parseFractionMetric(card.metrics, "Downloads");
    const summaryParts = [`${pluralizeCount(matches, "result")} returned from the current query.`];
    if (metadataCount !== null && metadataCount > 0) {
      summaryParts.push(`${pluralizeCount(metadataCount, "record")} enriched with metadata.`);
    }
    if (downloadFraction && downloadFraction.denominator > 0) {
      summaryParts.push(
        `${downloadFraction.numerator} of ${downloadFraction.denominator} requested downloads prepared.`
      );
    }
    return {
      eyebrow: "BisQue",
      title: serviceTitle,
      summary: summaryParts.join(" "),
      hideMetricBadges: true,
    };
  }

  if (card.tool === "run_bisque_module") {
    const outputCount = parseLeadingMetricCount(card.metrics, "Outputs") ?? card.resourceRows.length;
    const summaryParts = [`${pluralizeCount(outputCount, "output")} recorded from the module run.`];
    return {
      eyebrow: "BisQue Module",
      title: card.title,
      subtitle: card.subtitle,
      summary: summaryParts.join(" "),
      hideMetricBadges: true,
    };
  }

  if (
    card.tool === "bisque_download_resource" ||
    card.tool === "bisque_download_dataset" ||
    card.tool === "bisque_create_dataset" ||
    card.tool === "bisque_add_to_dataset" ||
    card.tool === "bisque_add_gobjects" ||
    card.tool === "add_tags_to_resource" ||
    card.tool === "bisque_fetch_xml" ||
    card.tool === "delete_bisque_resource"
  ) {
    const resourceCount = card.resourceRows.length || parseLeadingMetricCount(card.metrics, "Resources") || 1;
    const summaryParts = [`${pluralizeCount(resourceCount, "resource")} affected.`];
    return {
      eyebrow: "BisQue",
      title: serviceTitle,
      summary: summaryParts.join(" "),
      hideMetricBadges: true,
    };
  }

  if (card.tool === "load_bisque_resource") {
    const tagCount = parseLeadingMetricCount(card.metrics, "Tags");
    const dimensions =
      card.metrics.find((metric) => metric.label === "Dimensions")?.value ?? "n/a";
    const summaryParts: string[] = [];
    if (tagCount !== null) {
      summaryParts.push(`${pluralizeCount(tagCount, "tag")} recorded.`);
    }
    if (dimensions && dimensions !== "n/a") {
      summaryParts.push(`Dimensions ${dimensions}.`);
    }
    return {
      eyebrow: "BisQue",
      title: `${serviceTitle} Record`,
      subtitle: card.subtitle,
      summary: summaryParts.join(" "),
      hideMetricBadges: true,
    };
  }

  if (card.tool === "upload_to_bisque") {
    const uploadValue = card.metrics.find((metric) => metric.label === "Uploaded")?.value ?? "";
    const datasetAction =
      card.metrics.find((metric) => metric.label === "Dataset")?.value ?? "none";
    const addedCount = parseLeadingMetricCount(card.metrics, "Added");
    const summaryParts: string[] = [];
    if (uploadValue) {
      summaryParts.push(`Uploaded ${uploadValue}.`);
    }
    if (datasetAction && datasetAction !== "none") {
      summaryParts.push(`Dataset action: ${datasetAction}.`);
    }
    if (addedCount !== null && addedCount > 0) {
      summaryParts.push(`${pluralizeCount(addedCount, "resource")} added to the dataset.`);
    }
    return {
      eyebrow: "BisQue",
      title: "Ingest Service",
      subtitle: card.subtitle ? `Target dataset: ${card.subtitle}` : undefined,
      summary: summaryParts.join(" "),
      hideMetricBadges: true,
    };
  }

  return null;
};

// Open the lightbox over a YOLO/prairie figure stack, focused on one figure.
const openYoloFigureLightbox = (figures: YoloFigureCard[], focusUrl: string): void => {
  const lightboxFigures: LightboxFigure[] = figures
    .filter((figure) => figure.previewable && figure.previewUrl)
    .map((figure) => ({ url: figure.previewUrl, downloadUrl: figure.downloadUrl, title: figure.title }));
  openFigureLightbox(
    lightboxFigures,
    lightboxFigures.findIndex((figure) => figure.url === focusUrl)
  );
};

function YoloFigureStack({
  figures,
  variant = "default",
}: {
  figures: YoloFigureCard[];
  variant?: "default" | "prairie";
}) {
  if (figures.length === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        "chat-tool-figure-stack",
        variant === "prairie" && "chat-tool-figure-stack--prairie"
      )}
      data-testid={variant === "prairie" ? "prairie-figure-stack" : "yolo-figure-stack"}
    >
      {figures.map((figure, index) => {
        const classSummary = figure.classCounts
          .map((item) => `${item.name} ${item.count}`)
          .join(" · ");
        const details = [
          figure.boxCount !== null && figure.boxCount !== undefined
            ? `${Math.round(figure.boxCount)} box${Math.round(figure.boxCount) === 1 ? "" : "es"}`
            : null,
          classSummary || null,
        ].filter((value): value is string => value !== null);
        return (
          <figure
            key={figure.key}
            className={cn(
              "chat-tool-figure-card",
              variant === "prairie" && "chat-tool-figure-card--prairie"
            )}
            data-testid={variant === "prairie" ? "prairie-figure-card" : "yolo-figure-card"}
          >
            <div className="chat-tool-figure-media-wrap">
              {figure.previewable ? (
                <button
                  type="button"
                  className="chat-tool-figure-imagebtn"
                  aria-label={`View ${figure.title}`}
                  onClick={() => openYoloFigureLightbox(figures, figure.previewUrl)}
                >
                  <img
                    src={figure.previewUrl}
                    alt={figure.title}
                    loading={index === 0 ? "eager" : "lazy"}
                    className="chat-tool-figure-image"
                    data-testid={variant === "prairie" ? "prairie-figure-image" : "yolo-figure-image"}
                  />
                </button>
              ) : (
                <div className="chat-tool-figure-placeholder chat-tool-image-placeholder">
                  <ImageIcon className="size-5" />
                  <span>Preview unavailable</span>
                </div>
              )}
            </div>
            <figcaption className="chat-tool-figure-caption">
              <div className="chat-tool-figure-meta">
                <div>
                  <p className="chat-tool-figure-title">{figure.title}</p>
                  {figure.subtitle ? (
                    <p className="chat-tool-figure-subtitle">{figure.subtitle}</p>
                  ) : null}
                </div>
                {details.length > 0 ? (
                  <p className="chat-tool-figure-summary">{details.join(" · ")}</p>
                ) : null}
              </div>
              <div className="chat-tool-figure-actions">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => openYoloFigureLightbox(figures, figure.previewUrl)}
                >
                  View annotated
                </Button>
                {figure.originalUrl ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      openFigureLightbox(
                        [
                          {
                            url: figure.originalUrl as string,
                            downloadUrl: figure.originalUrl,
                            title: `${figure.title} (original)`,
                          },
                        ],
                        0
                      )
                    }
                  >
                    View original
                  </Button>
                ) : null}
                <Button asChild variant="ghost" size="sm">
                  <a href={figure.downloadUrl ?? figure.previewUrl} download target="_blank" rel="noreferrer">
                    <Download className="size-4" />
                    Download
                  </a>
                </Button>
              </div>
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}

function YoloFigureUnavailable({
  variant = "default",
}: {
  variant?: "default" | "prairie";
}) {
  return (
    <p
      className={cn(
        "chat-tool-figure-unavailable",
        variant === "prairie" && "chat-tool-figure-unavailable--prairie"
      )}
    >
      Annotated figure unavailable for this run. If this is a restored result, the current
      session may not have access to the stored artifacts yet.
    </p>
  );
}

function PrairieDetectionCardBody({ card }: { card: ToolResultCard }) {
  if (!card.prairieInsights) {
    return null;
  }
  const figures = card.yoloFigures ?? [];

  return (
    <div className="chat-tool-prairie-shell" data-testid="prairie-detection-card">
      {card.metrics.length > 0 ? (
        <div className="chat-tool-prairie-stats">
          {card.metrics.map((metric) => (
            <div key={`${card.id}-${metric.label}`} className="chat-tool-prairie-stat">
              <span className="chat-tool-prairie-stat-label">{metric.label}</span>
              <strong className="chat-tool-prairie-stat-value">{metric.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {figures.length > 0 ? (
        <YoloFigureStack figures={figures} variant="prairie" />
      ) : card.yoloFigureAvailability?.missingAnnotatedFigure ? (
        <YoloFigureUnavailable variant="prairie" />
      ) : null}
    </div>
  );
}

export function ToolResultCardSection({
  cards,
  messageId,
  onImportBisqueResourcesIntoConversation,
  onCopyBisqueResourceUri,
}: ToolResultCardSectionProps) {
  return (
    <div className="chat-tool-cards">
      {cards.map((card) => {
        const usesResourceQuickPreview =
          card.images.length > 0 && resourceBackedBisqueCardTools.has(card.tool);
        const bisqueResourceHeader =
          usesResourceQuickPreview && card.tool !== "run_bisque_module"
            ? buildBisqueResourceHeader(card)
            : null;
        const showResourceTable =
          card.resourceRows.length > 0 &&
          !(usesResourceQuickPreview && card.resourceRows.length === 1);
        const isPrairieCard =
          card.variant === "prairie_detection" && Boolean(card.prairieInsights);

        return (
          <Card
            key={card.id}
            className={cn(
              "chat-tool-card",
              isPrairieCard && "chat-tool-card--prairie"
            )}
          >
            <CardHeader className="chat-tool-card-header">
              {isPrairieCard ? (
                <p className="chat-tool-card-eyebrow">Wildlife Detection</p>
              ) : bisqueResourceHeader?.eyebrow ? (
                <p className="chat-tool-card-eyebrow">{bisqueResourceHeader.eyebrow}</p>
              ) : null}
              <CardTitle className="chat-tool-card-title">
                {bisqueResourceHeader?.title ?? card.title}
              </CardTitle>
              {(bisqueResourceHeader?.subtitle ?? card.subtitle) ? (
                <p className="chat-tool-card-subtitle">
                  {bisqueResourceHeader?.subtitle ?? card.subtitle}
                </p>
              ) : null}
              {bisqueResourceHeader?.summary ? (
                <p className="chat-tool-card-summary">{bisqueResourceHeader.summary}</p>
              ) : null}
              {!isPrairieCard && !bisqueResourceHeader?.hideMetricBadges ? (
                <div className="chat-tool-metrics">
                  {card.metrics.map((metric) => (
                    <Badge key={`${card.id}-${metric.label}`} variant="secondary">
                      {metric.label}: {metric.value}
                    </Badge>
                  ))}
                </div>
              ) : null}
            </CardHeader>
            <CardContent className="chat-tool-card-content">
              {isPrairieCard ? <PrairieDetectionCardBody card={card} /> : null}
              {!isPrairieCard && card.classes.length > 0 ? (
                <div className="chat-tool-classes">
                  {card.classes.map((cls) => (
                    <Badge key={`${card.id}-${cls.name}`} variant="outline">
                      {cls.name} ({cls.count})
                    </Badge>
                  ))}
                </div>
              ) : null}
              {!isPrairieCard && showResourceTable ? (
                <div className="chat-tool-resource-table-wrap">
                  <table className="chat-tool-resource-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Created</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {card.resourceRows.map((row, rowIndex) => (
                        <tr key={`${card.id}-resource-${rowIndex}`}>
                          <td className="chat-tool-resource-name-cell">
                            <div className="chat-tool-resource-name" title={row.name}>
                              {row.name}
                            </div>
                          </td>
                          <td>
                            <span className="chat-tool-resource-date">
                              {row.created ?? "-"}
                            </span>
                          </td>
                          <td className="chat-tool-resource-actions-cell">
                            <div className="chat-tool-resource-actions">
                              {row.clientViewUrl || row.uri ? (
                                <a
                                  href={row.clientViewUrl || row.uri}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="chat-tool-resource-link"
                                >
                                  Open in BisQue
                                </a>
                              ) : null}
                              {row.resourceUri ? (
                                <button
                                  type="button"
                                  className="chat-tool-resource-link"
                                  onClick={() => {
                                    void onImportBisqueResourcesIntoConversation(
                                      [row.resourceUri as string],
                                      {
                                        materialize: false,
                                        persistSelectionContext: true,
                                        source: "tool_result_use_in_chat",
                                        originatingMessageId: messageId,
                                      }
                                    );
                                  }}
                                >
                                  Use in chat
                                </button>
                              ) : null}
                              {row.resourceUri ? (
                                <button
                                  type="button"
                                  className="chat-tool-resource-link"
                                  onClick={() => {
                                    void onCopyBisqueResourceUri(
                                      (row.clientViewUrl || row.resourceUri) as string
                                    );
                                  }}
                                >
                                  Copy link
                                </button>
                              ) : null}
                              {!row.clientViewUrl && !row.uri && !row.resourceUri
                                ? "-"
                                : null}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
              {!isPrairieCard && card.downloadRows.length > 0 ? (
                <div className="chat-tool-resource-table-wrap">
                  <p className="chat-tool-card-subtitle">Download activity</p>
                  <table className="chat-tool-resource-table">
                    <thead>
                      <tr>
                        <th>Status</th>
                        <th>Saved to</th>
                        <th>Resource</th>
                      </tr>
                    </thead>
                    <tbody>
                      {card.downloadRows.map((row, rowIndex) => (
                        <tr key={`${card.id}-download-${rowIndex}`}>
                          <td>{row.status}</td>
                          <td>{row.outputPath ?? "-"}</td>
                          <td>
                            {row.clientViewUrl ? (
                              <a
                                href={row.clientViewUrl}
                                target="_blank"
                                rel="noreferrer"
                                className="chat-tool-resource-link"
                              >
                                {row.clientViewUrl}
                              </a>
                            ) : (
                              row.resourceUri ?? "-"
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
              {!isPrairieCard && card.tool === "yolo_detect" && card.yoloFigures?.length ? (
                <YoloFigureStack figures={card.yoloFigures} variant="default" />
              ) : !isPrairieCard &&
                card.tool === "yolo_detect" &&
                card.yoloFigureAvailability?.missingAnnotatedFigure ? (
                <YoloFigureUnavailable variant="default" />
              ) : !isPrairieCard && card.images.length > 0 ? (
                usesResourceQuickPreview ? (
                  <Suspense fallback={null}>
                    <LazyToolResultQuickPreview
                      images={card.images}
                      resourceRows={card.resourceRows}
                      onUseInChat={(resourceUri: string) => {
                        void onImportBisqueResourcesIntoConversation([resourceUri], {
                          materialize: false,
                          persistSelectionContext: true,
                          source: "tool_result_use_in_chat",
                          originatingMessageId: messageId,
                        });
                      }}
                    />
                  </Suspense>
                ) : (
                  <Suspense fallback={null}>
                    <LazyToolImageCarousel
                      key={card.images.map((image) => image.path).join("|")}
                      images={card.images}
                    />
                  </Suspense>
                )
              ) : null}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
