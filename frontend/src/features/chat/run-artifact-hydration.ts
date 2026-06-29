type HydrationRunArtifact = {
  path?: string;
  url?: string;
  downloadUrl?: string;
};

type HydrationArtifactRecord = {
  path?: unknown;
  mime_type?: unknown;
  mimeType?: unknown;
};

type HydrationProgressEvent = {
  tool?: string;
};

type HydrationRunEvent = {
  event_type?: string;
  event_kind?: string;
  payload?: {
    event_kind?: unknown;
  };
};

type HydrationCardImage = {
  url?: string;
  downloadUrl?: string;
};

type HydrationCardFigure = {
  url?: string;
  downloadUrl?: string;
};

type HydrationToolCard = {
  tool?: string;
  images?: HydrationCardImage[];
  yoloFigureAvailability?: { missingAnnotatedFigure?: boolean } | null;
  megasegInsights?:
    | {
        heroFigure?: HydrationCardFigure | null;
        secondaryFigures?: HydrationCardFigure[];
      }
    | null;
};

type HydrationMessage = {
  role?: string;
  runId?: string | null;
  content?: string;
  runArtifacts?: HydrationRunArtifact[];
  progressEvents?: HydrationProgressEvent[];
  runEvents?: HydrationRunEvent[];
  responseMetadata?: Record<string, unknown> | null;
};

const VISUAL_TOOL_NAMES = new Set<string>([
  "segment_image_megaseg",
  "estimate_depth_pro",
  "yolo_detect",
]);

const looksLikeLocalFilesystemPath = (value: string): boolean => {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return false;
  }
  if (normalized.startsWith("file://")) {
    return true;
  }
  if (/^[A-Za-z]:[\\/]/.test(normalized)) {
    return true;
  }
  return /^\/(?:srv|mnt|private|Users)\//.test(normalized);
};

const HYDRATABLE_VISUAL_ARTIFACT_EXTENSIONS = [
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".svg",
  ".avif",
  ".tif",
  ".tiff",
  ".nii",
  ".nii.gz",
  ".nrrd",
  ".mha",
  ".mhd",
];

export const isHydratableRunArtifactVisual = (
  artifact: HydrationArtifactRecord
): boolean => {
  const mimeType = String(artifact?.mime_type ?? artifact?.mimeType ?? "")
    .trim()
    .toLowerCase();
  if (mimeType.startsWith("image/")) {
    return true;
  }
  const path = String(artifact?.path ?? "").trim().toLowerCase();
  if (!path) {
    return false;
  }
  return HYDRATABLE_VISUAL_ARTIFACT_EXTENSIONS.some((extension) =>
    path.endsWith(extension)
  );
};

const visualToolName = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase();

const messageHasVisualToolSignal = (message: HydrationMessage): boolean => {
  const progressEvents = Array.isArray(message.progressEvents)
    ? message.progressEvents
    : [];
  if (
    progressEvents.some((event) => VISUAL_TOOL_NAMES.has(visualToolName(event?.tool)))
  ) {
    return true;
  }
  const responseMetadata =
    message.responseMetadata &&
    typeof message.responseMetadata === "object" &&
    !Array.isArray(message.responseMetadata)
      ? message.responseMetadata
      : {};
  const toolInvocations = Array.isArray(responseMetadata.tool_invocations)
    ? responseMetadata.tool_invocations
    : [];
  return toolInvocations.some((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return false;
    }
    return VISUAL_TOOL_NAMES.has(visualToolName((entry as { tool?: unknown }).tool));
  });
};

const messageHasCreatedArtifactEvent = (message: HydrationMessage): boolean => {
  const runEvents = Array.isArray(message.runEvents) ? message.runEvents : [];
  return runEvents.some((event) => {
    const eventKind =
      String(event?.event_kind || "").trim() ||
      String(event?.event_type || "").trim() ||
      String(event?.payload?.event_kind || "").trim();
    return eventKind === "artifact.created";
  });
};

const messageMentionsOutputImagePath = (message: HydrationMessage): boolean => {
  const content = String(message.content || "");
  if (!content) {
    return false;
  }
  return /(?:\/outputs\/|(?:^|[\s`(["'])outputs\/|(?:^|[\s`(["'])paper_pages\/)[^\s`)"']+\.(?:png|jpe?g|gif|webp|svg)/i.test(
    content
  );
};

const decodePathSegment = (value: string): string => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const normalizeArtifactReferencePath = (value: string): string => {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return "";
  }
  let path = trimmed;
  try {
    const parsed = new URL(path, "http://ultra.local");
    path = parsed.pathname;
  } catch {
    path = path.split(/[?#]/, 1)[0] ?? path;
  }
  path = decodePathSegment(path).replace(/\\/g, "/").replace(/^\/+/, "");
  const outputsIndex = path.lastIndexOf("outputs/");
  if (outputsIndex >= 0) {
    path = path.slice(outputsIndex + "outputs/".length);
  }
  return path.replace(/^\/+/, "");
};

const artifactUrlForMarkdownPath = (
  rawPath: string,
  artifacts: HydrationRunArtifact[]
): string | null => {
  const normalizedPath = normalizeArtifactReferencePath(rawPath);
  if (!normalizedPath) {
    return null;
  }
  for (const artifact of artifacts) {
    const artifactPath = normalizeArtifactReferencePath(String(artifact?.path || ""));
    if (!artifactPath) {
      continue;
    }
    if (
      normalizedPath === artifactPath ||
      normalizedPath.endsWith(`/${artifactPath}`) ||
      artifactPath.endsWith(`/${normalizedPath}`)
    ) {
      return String(artifact.url || artifact.downloadUrl || "").trim() || null;
    }
  }
  return null;
};

export const rewriteArtifactMarkdownImageUrls = (
  content: string,
  artifacts: HydrationRunArtifact[]
): string => {
  const markdown = String(content || "");
  if (!markdown || !Array.isArray(artifacts) || artifacts.length === 0) {
    return markdown;
  }
  return markdown.replace(
    /(!\[[^\]]*]\(\s*)([^)\s]+)((?:\s+["'][^"']*["'])?\s*\))/g,
    (match, prefix: string, rawUrl: string, suffix: string) => {
      const resolvedUrl = artifactUrlForMarkdownPath(rawUrl, artifacts);
      return resolvedUrl ? `${prefix}${resolvedUrl}${suffix}` : match;
    }
  );
};

export const shouldHydrateRunArtifacts = (
  message: HydrationMessage,
  toolCards: HydrationToolCard[]
): boolean => {
  if (String(message.role || "").trim().toLowerCase() !== "assistant") {
    return false;
  }
  if (!String(message.runId || "").trim()) {
    return false;
  }
  const runArtifacts = Array.isArray(message.runArtifacts) ? message.runArtifacts : [];
  if (runArtifacts.length === 0) {
    return true;
  }
  if (
    toolCards.some((card) => Boolean(card.yoloFigureAvailability?.missingAnnotatedFigure))
  ) {
    return true;
  }
  if (messageHasCreatedArtifactEvent(message)) {
    return true;
  }
  if (messageMentionsOutputImagePath(message)) {
    return true;
  }

  const hasVisualToolCard = toolCards.some((card) => {
    if (VISUAL_TOOL_NAMES.has(visualToolName(card.tool))) {
      return true;
    }
    if ((card.images?.length ?? 0) > 0) {
      return true;
    }
    return Boolean(card.megasegInsights);
  });
  const hasVisualSignal = hasVisualToolCard || messageHasVisualToolSignal(message);
  if (!hasVisualSignal) {
    return false;
  }

  const candidateUrls = new Set<string>();
  runArtifacts.forEach((artifact) => {
    [artifact?.url, artifact?.downloadUrl, artifact?.path].forEach((value) => {
      const token = String(value || "").trim();
      if (token) {
        candidateUrls.add(token);
      }
    });
  });
  toolCards.forEach((card) => {
    (card.images ?? []).forEach((image) => {
      [image?.url, image?.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    });
    const heroFigure = card.megasegInsights?.heroFigure;
    if (heroFigure) {
      [heroFigure.url, heroFigure.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    }
    (card.megasegInsights?.secondaryFigures ?? []).forEach((figure) => {
      [figure.url, figure.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    });
  });

  return Array.from(candidateUrls).some(looksLikeLocalFilesystemPath);
};

// Non-visual durable run outputs the chat surfaces as readable/downloadable
// documents. Images keep flowing through the figure strip; this covers the
// markdown report, supporting code, and data tables a run produces.
export type RunDocumentKind = "report" | "code" | "data" | "document";

const REPORT_DOCUMENT_EXTENSIONS = [".md", ".markdown", ".mdx"];
const CODE_DOCUMENT_EXTENSIONS = [
  ".py",
  ".ipynb",
  ".r",
  ".jl",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".sh",
  ".bash",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".java",
  ".go",
  ".rs",
  ".m",
  ".sql",
];
const DATA_DOCUMENT_EXTENSIONS = [
  ".csv",
  ".tsv",
  ".json",
  ".jsonl",
  ".ndjson",
  ".geojson",
  ".yaml",
  ".yml",
  ".xml",
  ".parquet",
  ".feather",
  ".npy",
  ".npz",
  ".h5",
  ".hdf5",
  ".nc",
];
const PLAIN_DOCUMENT_EXTENSIONS = [".txt", ".text", ".log", ".tex", ".bib", ".rst", ".pdf"];

const pathHasExtension = (path: string, extensions: string[]): boolean => {
  const lower = String(path || "").trim().toLowerCase();
  return extensions.some((extension) => lower.endsWith(extension));
};

// Only intentional, agent-produced OUTPUTS are surfaced — never the user's own
// uploads, staged inputs, or intermediate tile artifacts. The outputs collector
// registers produced files under `outputs/`.
const isRunOutputArtifactPath = (path: string): boolean => {
  const normalized = String(path || "")
    .trim()
    .toLowerCase()
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
  if (!normalized) {
    return false;
  }
  return normalized.startsWith("outputs/") || normalized.includes("/outputs/");
};

export const classifyRunDocumentKind = (
  path: string,
  mimeType?: string | null
): RunDocumentKind | null => {
  const mime = String(mimeType ?? "").trim().toLowerCase();
  if (pathHasExtension(path, REPORT_DOCUMENT_EXTENSIONS) || mime === "text/markdown") {
    return "report";
  }
  if (pathHasExtension(path, CODE_DOCUMENT_EXTENSIONS) || mime.startsWith("text/x-")) {
    return "code";
  }
  if (
    pathHasExtension(path, DATA_DOCUMENT_EXTENSIONS) ||
    mime === "application/json" ||
    mime.endsWith("+json") ||
    mime === "text/csv" ||
    mime === "application/x-yaml" ||
    mime === "application/yaml" ||
    mime === "application/xml" ||
    mime.endsWith("+xml")
  ) {
    return "data";
  }
  if (pathHasExtension(path, PLAIN_DOCUMENT_EXTENSIONS) || mime.startsWith("text/")) {
    return "document";
  }
  return null;
};

export const isReportRunDocument = (path: string, mimeType?: string | null): boolean =>
  classifyRunDocumentKind(path, mimeType) === "report";

// A durable run artifact the chat should surface as a downloadable document or
// readable report: an intentional output, NOT a visual (images take the figure
// strip), with a recognized document/code/data type.
export const isHydratableRunArtifactDocument = (
  artifact: HydrationArtifactRecord
): boolean => {
  if (isHydratableRunArtifactVisual(artifact)) {
    return false;
  }
  const path = String(artifact?.path ?? "").trim();
  if (!path || !isRunOutputArtifactPath(path)) {
    return false;
  }
  const mimeType = String(artifact?.mime_type ?? artifact?.mimeType ?? "");
  return classifyRunDocumentKind(path, mimeType) !== null;
};
