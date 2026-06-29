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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

export type ScientificFigureCard = {
  key: string;
  kind?: string;
  title: string;
  file?: string;
  subtitle?: string;
  summary?: string;
  previewUrl?: string;
  url?: string;
  downloadUrl?: string;
  previewable: boolean;
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

export type MegasegFileInsight = {
  file: string;
  coveragePercent?: number | null;
  objectCount?: number | null;
  activeSliceCount?: number | null;
  zSliceCount?: number | null;
  largestComponentVoxels?: number | null;
  technicalSummary?: string | null;
};

export type MegasegInsights = {
  resultGroupId: string;
  heroFigure: ScientificFigureCard | null;
  secondaryFigures: ScientificFigureCard[];
  fileRows: MegasegFileInsight[];
  reportPath?: string | null;
  summaryCsvPath?: string | null;
  reportDownloadUrl?: string | null;
  summaryCsvDownloadUrl?: string | null;
  technicalSummary?: string | null;
  collectionLabel?: string;
  device?: string | null;
  structureChannel?: number | null;
  nucleusChannel?: number | null;
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
    | "segment_image_megaseg"
    | "yolo_detect"
    | "estimate_depth_pro"
    | "quantify_segmentation_masks"
    | "plot_quantified_detections"
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
    | "search_bisque_resources"
    | "bisque_advanced_search"
    | "bisque_find_assets";
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
  scientificFigures?: ScientificFigureCard[];
  megasegInsights?: MegasegInsights | null;
};

export type ResearchDigestEvidenceRow = {
  source: string;
  summary?: string;
  artifact?: string;
  runId?: string | null;
};

export type ResearchDigestMeasurementRow = {
  name: string;
  valueLabel: string;
};

export type ResearchDigestStatisticRow = {
  label: string;
  summary: string;
};

export type ResearchDigestData = {
  result: string;
  confidenceLevel?: "low" | "medium" | "high";
  confidenceWhy: string[];
  evidence: ResearchDigestEvidenceRow[];
  measurements: ResearchDigestMeasurementRow[];
  statisticalAnalysis: ResearchDigestStatisticRow[];
  qcWarnings: string[];
  limitations: string[];
  nextSteps: string[];
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

const artifactPreviewPrefixPatterns = [
  /^[a-f0-9]{32}__[a-f0-9]{12}__/i,
  /^[a-f0-9-]{36}__[a-f0-9]{12}__/i,
  /^[a-f0-9]{32}__/i,
  /^[a-f0-9-]{36}__/i,
  /^[a-f0-9]{12}__/i,
];

const resourceBackedBisqueCardTools = new Set<ToolResultCard["tool"]>([
  "upload_to_bisque",
  "load_bisque_resource",
  "search_bisque_resources",
  "bisque_advanced_search",
  "bisque_find_assets",
  "bisque_create_dataset",
  "bisque_add_to_dataset",
  "bisque_add_gobjects",
  "add_tags_to_resource",
  "bisque_fetch_xml",
  "run_bisque_module",
]);

const extractFilename = (value: string): string => {
  const normalized = String(value || "").replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? normalized;
};

const stripArtifactFilenamePrefixes = (value: string): string => {
  let normalized = extractFilename(value).trim();
  if (!normalized) {
    return "";
  }
  let changed = true;
  while (changed && normalized) {
    changed = false;
    for (const pattern of artifactPreviewPrefixPatterns) {
      const next = normalized.replace(pattern, "");
      if (next !== normalized) {
        normalized = next;
        changed = true;
      }
    }
  }
  return normalized;
};

const toDisplayFileLabel = (value: string): string => {
  const dehashed = stripArtifactFilenamePrefixes(value);
  if (dehashed.length <= 64) {
    return dehashed;
  }
  return `${dehashed.slice(0, 28)}...${dehashed.slice(-22)}`;
};

const formatPercentMetric = (value: number | null | undefined, digits = 2): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${Number(value).toFixed(digits)}%`;
};

const formatIntegerMetric = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Math.round(Number(value)).toLocaleString();
};

const scientificFigureKind = (figure: ScientificFigureCard): string =>
  String(figure.kind || figure.key || figure.title || "").trim().toLowerCase();

const scientificFigureFileLabel = (figure: ScientificFigureCard): string | undefined =>
  figure.file ?? figure.subtitle;

const scientificFigureUrl = (figure: ScientificFigureCard): string =>
  figure.previewUrl ?? figure.url ?? "";

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

  if (
    card.tool === "search_bisque_resources" ||
    card.tool === "bisque_advanced_search" ||
    card.tool === "bisque_find_assets"
  ) {
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

const scientificFigureTabLabel = (figure: ScientificFigureCard): string => {
  const kind = scientificFigureKind(figure);
  if (kind === "overlay_mid_z" || kind.includes("overlay_mid_z")) {
    return "Mid-Z";
  }
  if (kind === "overlay_mip" || kind.includes("overlay_mip")) {
    return "MIP";
  }
  if (kind === "mask_preview" || kind.includes("mask_preview")) {
    return "Mask";
  }
  if (kind === "probability_preview" || kind.includes("probability_preview")) {
    return "Probability";
  }
  return figure.title;
};

// Open the lightbox over a set of scientific (MegaSeg) figures, focused on one.
const openScientificFigureLightbox = (
  figures: Array<ScientificFigureCard | null>,
  focusUrl: string
): void => {
  const lightboxFigures: LightboxFigure[] = figures
    .filter((figure): figure is ScientificFigureCard => Boolean(figure) && Boolean(figure?.previewable))
    .map((figure) => ({ url: scientificFigureUrl(figure), downloadUrl: figure.downloadUrl, title: figure.title }))
    .filter((figure) => figure.url.length > 0);
  openFigureLightbox(
    lightboxFigures,
    lightboxFigures.findIndex((figure) => figure.url === focusUrl)
  );
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

function MegasegCardBody({ card }: { card: ToolResultCard }) {
  const insights = card.megasegInsights;
  if (!insights) {
    return null;
  }

  const heroFigure = insights.heroFigure;
  const heroFigureUrl = heroFigure ? scientificFigureUrl(heroFigure) : "";
  const visibleSecondaryFigures = insights.secondaryFigures.filter(
    (figure) => scientificFigureKind(figure) !== "mask_preview"
  );
  const deferredFigures = insights.secondaryFigures.filter((figure) =>
    scientificFigureKind(figure) === "mask_preview"
  );
  const defaultSecondaryFigure =
    visibleSecondaryFigures[0] ?? deferredFigures[0] ?? null;
  const scientificFigureSet: Array<ScientificFigureCard | null> = [
    heroFigure,
    ...visibleSecondaryFigures,
    ...deferredFigures,
  ];

  return (
    <div className="chat-scientific-result-shell" data-testid="megaseg-card">
      {heroFigure ? (
        <figure className="chat-scientific-hero">
          <div className="chat-scientific-hero-media">
            {heroFigure.previewable ? (
              <button
                type="button"
                className="chat-tool-figure-imagebtn"
                aria-label={`View ${heroFigure.title}`}
                onClick={() => openScientificFigureLightbox(scientificFigureSet, heroFigureUrl)}
              >
                <img
                  src={heroFigureUrl}
                  alt={heroFigure.title}
                  loading="eager"
                  className="chat-scientific-hero-image"
                />
              </button>
            ) : (
              <div className="chat-tool-figure-placeholder chat-tool-image-placeholder">
                <ImageIcon className="size-5" />
                <span>Preview unavailable</span>
              </div>
            )}
          </div>
          <figcaption className="chat-scientific-caption">
            <div className="chat-scientific-caption-copy">
              <p className="chat-tool-figure-title">{heroFigure.title}</p>
              {scientificFigureFileLabel(heroFigure) ? (
                <p className="chat-tool-figure-subtitle">
                  {toDisplayFileLabel(scientificFigureFileLabel(heroFigure) ?? "")}
                </p>
              ) : null}
              {heroFigure.summary ? (
                <p className="chat-tool-figure-summary">{heroFigure.summary}</p>
              ) : null}
            </div>
            <div className="chat-scientific-caption-actions">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => openScientificFigureLightbox(scientificFigureSet, heroFigureUrl)}
              >
                View figure
              </Button>
              <Button asChild variant="ghost" size="sm">
                <a
                  href={heroFigure.downloadUrl ?? heroFigureUrl}
                  download
                  target="_blank"
                  rel="noreferrer"
                >
                  Download
                </a>
              </Button>
            </div>
          </figcaption>
        </figure>
      ) : null}
      {card.narrative ? (
        <div className="chat-scientific-takeaway">
          <p className="chat-scientific-section-label">Takeaway</p>
          <p className="chat-tool-insight-body">{card.narrative}</p>
        </div>
      ) : null}
      {(insights.reportDownloadUrl || insights.summaryCsvDownloadUrl) ? (
        <div className="chat-scientific-downloads">
          {insights.reportDownloadUrl ? (
            <Button asChild variant="outline" size="sm">
              <a href={insights.reportDownloadUrl} target="_blank" rel="noreferrer">
                Open report
              </a>
            </Button>
          ) : null}
          {insights.summaryCsvDownloadUrl ? (
            <Button asChild variant="ghost" size="sm">
              <a href={insights.summaryCsvDownloadUrl} download target="_blank" rel="noreferrer">
                Download CSV
              </a>
            </Button>
          ) : null}
        </div>
      ) : null}
      {defaultSecondaryFigure ? (
        <Tabs
          defaultValue={defaultSecondaryFigure.key}
          className="chat-scientific-secondary"
        >
          <TabsList className="chat-scientific-tabs-list">
            {[...visibleSecondaryFigures, ...deferredFigures].map((figure) => (
              <TabsTrigger key={figure.key} value={figure.key}>
                {scientificFigureTabLabel(figure)}
              </TabsTrigger>
            ))}
          </TabsList>
          {[...visibleSecondaryFigures, ...deferredFigures].map((figure) => (
            <TabsContent
              key={figure.key}
              value={figure.key}
              className="chat-scientific-tab-panel"
            >
              <figure className="chat-scientific-secondary-figure">
                <div className="chat-scientific-secondary-media">
                  {figure.previewable ? (
                    <button
                      type="button"
                      className="chat-tool-figure-imagebtn"
                      aria-label={`View ${figure.title}`}
                      onClick={() => openScientificFigureLightbox(scientificFigureSet, scientificFigureUrl(figure))}
                    >
                      <img
                        src={scientificFigureUrl(figure)}
                        alt={figure.title}
                        loading="lazy"
                        className="chat-scientific-secondary-image"
                      />
                    </button>
                  ) : (
                    <div className="chat-tool-figure-placeholder chat-tool-image-placeholder">
                      <ImageIcon className="size-5" />
                      <span>Preview unavailable</span>
                    </div>
                  )}
                </div>
                <figcaption className="chat-scientific-secondary-caption">
                  <div>
                    <p className="chat-tool-figure-title">{figure.title}</p>
                    {scientificFigureFileLabel(figure) ? (
                      <p className="chat-tool-figure-subtitle">
                        {toDisplayFileLabel(scientificFigureFileLabel(figure) ?? "")}
                      </p>
                    ) : null}
                  </div>
                  <Button asChild variant="ghost" size="sm">
                    <a
                      href={figure.downloadUrl ?? scientificFigureUrl(figure)}
                      download
                      target="_blank"
                      rel="noreferrer"
                    >
                      Download
                    </a>
                  </Button>
                </figcaption>
              </figure>
            </TabsContent>
          ))}
        </Tabs>
      ) : null}
      {insights.fileRows.length > 0 ? (
        <div className="chat-tool-resource-table-wrap">
          <div className="chat-tool-megaseg-table-head">
            <div>
              <p className="chat-scientific-section-label">Quantitative summary</p>
              {insights.collectionLabel ? (
                <p className="chat-tool-card-summary">{insights.collectionLabel}</p>
              ) : null}
            </div>
            {(insights.device ||
              insights.structureChannel !== null ||
              insights.structureChannel !== undefined) ? (
              <div className="chat-tool-megaseg-meta">
                {insights.device ? <span>Device {insights.device}</span> : null}
                {insights.structureChannel !== null &&
                insights.structureChannel !== undefined ? (
                  <span>Structure ch {Math.round(insights.structureChannel)}</span>
                ) : null}
                {insights.nucleusChannel !== null &&
                insights.nucleusChannel !== undefined ? (
                  <span>Nucleus ch {Math.round(insights.nucleusChannel)}</span>
                ) : null}
              </div>
            ) : null}
          </div>
          <table className="chat-tool-resource-table">
            <thead>
              <tr>
                <th>Image</th>
                <th>Coverage</th>
                <th>Objects</th>
                <th>Active z-slices</th>
                <th>Largest component</th>
              </tr>
            </thead>
            <tbody>
              {insights.fileRows.map((row, rowIndex) => (
                <tr key={`${card.id}-megaseg-row-${rowIndex}`}>
                  <td className="chat-tool-resource-name-cell">
                    <div className="chat-tool-resource-name" title={row.file}>
                      {toDisplayFileLabel(row.file)}
                    </div>
                  </td>
                  <td>{formatPercentMetric(row.coveragePercent)}</td>
                  <td>{formatIntegerMetric(row.objectCount)}</td>
                  <td>
                    {row.activeSliceCount !== null &&
                    row.activeSliceCount !== undefined &&
                    row.zSliceCount !== null &&
                    row.zSliceCount !== undefined
                      ? `${formatIntegerMetric(row.activeSliceCount)}/${formatIntegerMetric(row.zSliceCount)}`
                      : "n/a"}
                  </td>
                  <td>{formatIntegerMetric(row.largestComponentVoxels)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {(insights.technicalSummary ||
        deferredFigures.length > 0 ||
        insights.reportPath ||
        insights.summaryCsvPath) ? (
        <details className="chat-scientific-appendix">
          <summary>Methods & evidence</summary>
          <div className="chat-scientific-appendix-body">
            {insights.technicalSummary ? (
              <p className="chat-tool-card-summary">{insights.technicalSummary}</p>
            ) : null}
            {deferredFigures.length > 0 ? (
              <ul className="chat-research-list">
                {deferredFigures.map((figure) => (
                  <li key={figure.key}>
                    {figure.title}
                    {scientificFigureFileLabel(figure)
                      ? ` (${toDisplayFileLabel(scientificFigureFileLabel(figure) ?? "")})`
                      : ""}
                  </li>
                ))}
              </ul>
            ) : null}
            {(insights.reportPath || insights.summaryCsvPath) ? (
              <ul className="chat-research-list">
                {insights.reportPath ? (
                  <li>
                    <strong>Report</strong>: {toDisplayFileLabel(insights.reportPath)}
                  </li>
                ) : null}
                {insights.summaryCsvPath ? (
                  <li>
                    <strong>CSV</strong>: {toDisplayFileLabel(insights.summaryCsvPath)}
                  </li>
                ) : null}
              </ul>
            ) : null}
          </div>
        </details>
      ) : null}
    </div>
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
        const isMegasegCard =
          card.tool === "segment_image_megaseg" && Boolean(card.megasegInsights);

        return (
          <Card
            key={card.id}
            className={cn(
              "chat-tool-card",
              isPrairieCard && "chat-tool-card--prairie",
              isMegasegCard && "chat-tool-card--scientific"
            )}
          >
            <CardHeader className="chat-tool-card-header">
              {isPrairieCard ? (
                <p className="chat-tool-card-eyebrow">Wildlife Detection</p>
              ) : isMegasegCard ? (
                <p className="chat-tool-card-eyebrow">Microscopy Segmentation</p>
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
              {isMegasegCard ? <MegasegCardBody card={card} /> : null}
              {isPrairieCard ? <PrairieDetectionCardBody card={card} /> : null}
              {!isMegasegCard && !isPrairieCard && card.classes.length > 0 ? (
                <div className="chat-tool-classes">
                  {card.classes.map((cls) => (
                    <Badge key={`${card.id}-${cls.name}`} variant="outline">
                      {cls.name} ({cls.count})
                    </Badge>
                  ))}
                </div>
              ) : null}
              {!isMegasegCard && !isPrairieCard && showResourceTable ? (
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
              {!isMegasegCard && !isPrairieCard && card.downloadRows.length > 0 ? (
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
              {!isMegasegCard && !isPrairieCard && card.tool === "yolo_detect" && card.yoloFigures?.length ? (
                <YoloFigureStack figures={card.yoloFigures} variant="default" />
              ) : !isMegasegCard &&
                !isPrairieCard &&
                card.tool === "yolo_detect" &&
                card.yoloFigureAvailability?.missingAnnotatedFigure ? (
                <YoloFigureUnavailable variant="default" />
              ) : !isMegasegCard && !isPrairieCard && card.images.length > 0 ? (
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

export function ResearchDigestCard({
  digest,
  followsVisuals = false,
}: {
  digest: ResearchDigestData;
  followsVisuals?: boolean;
}) {
  const summaryBits = [
    digest.measurements.length > 0
      ? `${digest.measurements.length} measurement${digest.measurements.length === 1 ? "" : "s"}`
      : null,
    digest.statisticalAnalysis.length > 0
      ? `${digest.statisticalAnalysis.length} statistical check${digest.statisticalAnalysis.length === 1 ? "" : "s"}`
      : null,
    digest.evidence.length > 0
      ? `${digest.evidence.length} supporting artifact${digest.evidence.length === 1 ? "" : "s"}`
      : null,
  ].filter((item): item is string => item !== null);
  const showCautionNote =
    String(digest.confidenceLevel || "").trim().toLowerCase() === "low" &&
    digest.confidenceWhy.length > 0;
  const showNextSteps =
    digest.nextSteps.length > 0 &&
    (showCautionNote || digest.qcWarnings.length > 0 || digest.limitations.length > 0);

  return (
    <section className="chat-research-digest" data-testid="research-digest-card">
      <div className="chat-research-digest-header">
        <p className="chat-research-digest-label">
          {followsVisuals ? "Evidence from this run" : "Measured evidence"}
        </p>
        {summaryBits.length > 0 ? (
          <p className="chat-research-digest-meta">{summaryBits.join(" · ")}</p>
        ) : null}
      </div>
      {showCautionNote ? (
        <p className="chat-research-digest-note">{digest.confidenceWhy.join(" ")}</p>
      ) : null}
      <div className="chat-research-digest-body">
        {digest.measurements.length > 0 ? (
          <section className="chat-research-digest-section">
            <div className="chat-research-digest-section-header">
              <p className="chat-tool-card-subtitle">Key measurements</p>
            </div>
            <div className="chat-tool-resource-table-wrap">
              <table className="chat-tool-resource-table">
                <thead>
                  <tr>
                    <th>Measurement</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {digest.measurements.map((row) => (
                    <tr key={`measurement-${row.name}`}>
                      <td>{row.name}</td>
                      <td>{row.valueLabel}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}
        {digest.evidence.length > 0 ? (
          <section className="chat-research-digest-section">
            <div className="chat-research-digest-section-header">
              <p className="chat-tool-card-subtitle">Evidence</p>
            </div>
            <ul className="chat-research-list">
              {digest.evidence.map((item, index) => (
                <li key={`evidence-${index}`}>
                  <strong>{item.source}</strong>
                  {item.summary ? `: ${item.summary}` : ""}
                  {item.artifact ? ` (${toDisplayFileLabel(item.artifact)})` : ""}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
        {digest.statisticalAnalysis.length > 0 ? (
          <section className="chat-research-digest-section">
            <div className="chat-research-digest-section-header">
              <p className="chat-tool-card-subtitle">Statistical analysis</p>
            </div>
            <ul className="chat-research-list">
              {digest.statisticalAnalysis.map((item, index) => (
                <li key={`stat-${index}`}>
                  <strong>{item.label}</strong>: {item.summary}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
        {digest.qcWarnings.length > 0 || digest.limitations.length > 0 ? (
          <div className="chat-research-digest-grid">
            {digest.qcWarnings.length > 0 ? (
              <section className="chat-research-digest-section">
                <div className="chat-research-digest-section-header">
                  <p className="chat-tool-card-subtitle">QC notes</p>
                </div>
                <ul className="chat-research-list">
                  {digest.qcWarnings.map((item, index) => (
                    <li key={`qc-${index}`}>{item}</li>
                  ))}
                </ul>
              </section>
            ) : null}
            {digest.limitations.length > 0 ? (
              <section className="chat-research-digest-section">
                <div className="chat-research-digest-section-header">
                  <p className="chat-tool-card-subtitle">Limits</p>
                </div>
                <ul className="chat-research-list">
                  {digest.limitations.map((item, index) => (
                    <li key={`limit-${index}`}>{item}</li>
                  ))}
                </ul>
              </section>
            ) : null}
          </div>
        ) : null}
        {showNextSteps ? (
          <section className="chat-research-digest-section">
            <div className="chat-research-digest-section-header">
              <p className="chat-tool-card-subtitle">Recommended next steps</p>
            </div>
            <ol className="chat-research-list chat-research-list--ordered">
              {digest.nextSteps.map((item, index) => (
                <li key={`step-${index}`}>{item}</li>
              ))}
            </ol>
          </section>
        ) : null}
      </div>
    </section>
  );
}
