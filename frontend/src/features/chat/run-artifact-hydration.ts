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

type HydrationToolCard = {
  tool?: string;
  images?: HydrationCardImage[];
  yoloFigureAvailability?: { missingAnnotatedFigure?: boolean } | null;
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

// Resolve a path the report references (markdown image, HTML <img src>) to the
// served URL of the matching run artifact. Exported for the report canvas's
// HTML pipeline, which rewrites <img> references the same way the markdown
// rewriter below does.
export const resolveRunOutputArtifactUrl = (
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
      const resolvedUrl = resolveRunOutputArtifactUrl(rawUrl, artifacts);
      return resolvedUrl ? `${prefix}${resolvedUrl}${suffix}` : match;
    }
  );
};

// Stable identity for a report within a conversation. Re-registrations of the
// same logical output (turn 3 regenerates outputs/report.html) land under new
// runs but keep this key, which is what groups them into one version chain
// behind one card instead of a trail of duplicate cards.
export const runReportPathKey = (path: string): string =>
  normalizeArtifactReferencePath(path).toLowerCase();

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
    return (card.images?.length ?? 0) > 0;
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
  });

  return Array.from(candidateUrls).some(looksLikeLocalFilesystemPath);
};

// Non-visual durable run outputs the chat surfaces as readable/downloadable
// documents. Images keep flowing through the figure strip; this covers the
// markdown report, supporting code, and data tables a run produces.
export type RunDocumentKind = "report" | "code" | "data" | "document";

// Reports come in two source formats with two different rendering paths: the
// markdown pipeline renders natively (prompt-kit Markdown + KaTeX), while HTML
// renders inside a sandboxed iframe. Both are the same "report" kind to every
// other consumer (cards, hydration gates, download chips).
export type RunReportFormat = "markdown" | "html";
export type RunDocumentPreviewFormat = RunReportFormat | "source" | "pdf";

const REPORT_DOCUMENT_EXTENSIONS = [".md", ".markdown", ".mdx"];
const HTML_REPORT_EXTENSIONS = [".html", ".htm"];
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
const SOURCE_DATA_DOCUMENT_EXTENSIONS = [
  ".csv",
  ".tsv",
  ".json",
  ".jsonl",
  ".ndjson",
  ".geojson",
  ".yaml",
  ".yml",
  ".xml",
];
const SOURCE_PLAIN_DOCUMENT_EXTENSIONS = [".txt", ".text", ".log", ".tex", ".bib", ".rst"];

const pathHasExtension = (path: string, extensions: string[]): boolean => {
  const lower = String(path || "").trim().toLowerCase();
  return extensions.some((extension) => lower.endsWith(extension));
};

// Only intentional, agent-produced OUTPUTS are surfaced — never the user's own
// uploads, staged inputs, or intermediate tile artifacts.
//
// CONTRACT (verified against the live control_artifacts table, 2026-08-01):
// registered paths are RELATIVE to the run's outputs root — "report.html",
// "toeplitz_figs/fig1.png" — with NO "outputs/" prefix. A prefix appears only
// when the agent happens to nest a directory literally named outputs/. The
// old gate here REQUIRED the prefix, which silently dark-holed every
// prefixless report ever registered (traced live on a real run: backend
// registered kind=report/text/html perfectly; this predicate rejected it).
// The table is written exclusively by the outputs collector — zero uploads/,
// staged_artifacts/, tool_outputs/ or diagnostics/ rows exist in it — so the
// gate ACCEPTS by default and denies the roots that would mark non-output
// content if registration ever widened.
const NON_OUTPUT_PATH_ROOTS = new Set([
  "uploads",
  "staged_artifacts",
  "staged_inputs",
  "tool_outputs",
  "diagnostics",
  "workspace",
  ".deepagents",
]);

const isRunOutputArtifactPath = (path: string): boolean => {
  const normalized = String(path || "")
    .trim()
    .toLowerCase()
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
  if (!normalized) {
    return false;
  }
  const rootSegment = normalized.split("/", 1)[0] ?? "";
  return !NON_OUTPUT_PATH_ROOTS.has(rootSegment);
};

// Which report renderer a document takes: HTML wins over the generic
// text/* fallbacks and must be checked before them — text/html would
// otherwise classify as a plain "document" and lose its reading surface.
export const runReportDocumentFormat = (
  path: string,
  mimeType?: string | null
): RunReportFormat | null => {
  const mime = String(mimeType ?? "").trim().toLowerCase();
  if (
    pathHasExtension(path, HTML_REPORT_EXTENSIONS) ||
    mime === "text/html" ||
    mime === "application/xhtml+xml"
  ) {
    return "html";
  }
  if (pathHasExtension(path, REPORT_DOCUMENT_EXTENSIONS) || mime === "text/markdown") {
    return "markdown";
  }
  return null;
};

// The canvas may only decode formats whose bytes have an explicit readable
// representation. Binary scientific containers still open the canvas, but
// land on a metadata/download state instead of being coerced through
// Response.text() and rendered as plausible-looking replacement characters.
export const runDocumentPreviewFormat = (
  path: string,
  mimeType?: string | null
): RunDocumentPreviewFormat | null => {
  const reportFormat = runReportDocumentFormat(path, mimeType);
  if (reportFormat) {
    return reportFormat;
  }
  const mime = String(mimeType ?? "").trim().toLowerCase();
  if (pathHasExtension(path, [".pdf"]) || mime === "application/pdf") {
    return "pdf";
  }
  if (
    pathHasExtension(path, [
      ...CODE_DOCUMENT_EXTENSIONS,
      ...SOURCE_DATA_DOCUMENT_EXTENSIONS,
      ...SOURCE_PLAIN_DOCUMENT_EXTENSIONS,
    ]) ||
    mime.startsWith("text/") ||
    mime === "application/json" ||
    mime.endsWith("+json") ||
    mime === "application/javascript" ||
    mime === "application/x-javascript" ||
    mime === "application/x-yaml" ||
    mime === "application/yaml" ||
    mime === "application/xml" ||
    mime.endsWith("+xml")
  ) {
    return "source";
  }
  return null;
};

const SOURCE_LANGUAGE_BY_EXTENSION: Record<string, string> = {
  bash: "bash",
  bib: "bibtex",
  c: "c",
  cpp: "cpp",
  csv: "csv",
  geojson: "json",
  go: "go",
  h: "c",
  hpp: "cpp",
  ipynb: "json",
  java: "java",
  jl: "julia",
  js: "javascript",
  json: "json",
  jsonl: "json",
  jsx: "jsx",
  log: "text",
  m: "matlab",
  ndjson: "json",
  py: "python",
  r: "r",
  rs: "rust",
  rst: "rst",
  sh: "bash",
  sql: "sql",
  text: "text",
  tex: "latex",
  ts: "typescript",
  tsv: "csv",
  tsx: "tsx",
  txt: "text",
  xml: "xml",
  yaml: "yaml",
  yml: "yaml",
};

export const runDocumentCodeLanguage = (
  path: string,
  mimeType?: string | null
): string => {
  const normalizedPath = String(path || "").trim().toLowerCase().split(/[?#]/, 1)[0] ?? "";
  const filename = normalizedPath.split("/").pop() ?? normalizedPath;
  const extension = filename.includes(".") ? filename.split(".").pop() ?? "" : "";
  if (SOURCE_LANGUAGE_BY_EXTENSION[extension]) {
    return SOURCE_LANGUAGE_BY_EXTENSION[extension];
  }
  const mime = String(mimeType ?? "").trim().toLowerCase();
  if (mime === "application/json" || mime.endsWith("+json")) {
    return "json";
  }
  if (mime === "application/xml" || mime.endsWith("+xml")) {
    return "xml";
  }
  if (mime.includes("yaml")) {
    return "yaml";
  }
  if (mime.includes("javascript")) {
    return "javascript";
  }
  return "text";
};

export const classifyRunDocumentKind = (
  path: string,
  mimeType?: string | null
): RunDocumentKind | null => {
  const mime = String(mimeType ?? "").trim().toLowerCase();
  if (runReportDocumentFormat(path, mimeType) !== null) {
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
