import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  Check,
  ChevronDown,
  Download,
  Loader2,
  X,
} from "lucide-react";
import { Markdown } from "@/components/prompt-kit/markdown";
import { formatBytes } from "@/lib/format";
import {
  resolveRunOutputArtifactUrl,
  rewriteArtifactMarkdownImageUrls,
  runReportDocumentFormat,
} from "@/features/chat/run-artifact-hydration";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// One registration of the report: the document as it exists on a specific
// run, plus that run's image artifacts so the report's own figure references
// can be resolved to served URLs. Ordered oldest → newest by the caller.
export type ReportCanvasVersion = {
  messageId: string;
  runId: string | null;
  document: {
    path: string;
    title: string;
    downloadUrl: string;
    mimeType?: string;
    sizeBytes?: number;
  };
  imageArtifacts: Array<{ path: string; url?: string; downloadUrl?: string }>;
};

// split: a fixed column in the stage grid (desktop).
// overlay: floats over the transcript when the stage is too narrow to split.
// sheet: full-screen on the phone regime, entered only from the card.
export type ReportCanvasMode = "split" | "overlay" | "sheet";

export type ReportCanvasProps = {
  versions: ReportCanvasVersion[];
  mode: ReportCanvasMode;
  closing?: boolean;
  onClose: () => void;
  loadDocumentText: (downloadUrl: string) => Promise<string>;
};

type CanvasBodyStatus = "loading" | "ready" | "error" | "oversize";

// The sandbox is the security boundary for model-generated HTML: an opaque
// origin (NO allow-same-origin, ever — it would hand the report the user's
// authenticated origin) plus a CSP that keeps the document self-contained.
// Figures work because the HOST fetches them with credentials and rewrites
// <img> to data: URIs before the document enters the frame. data:, not
// blob:, and that is load-bearing — a blob URL is scoped to the origin that
// minted it, and the frame's opaque origin cannot read the host's blobs
// (verified live: parent-minted blob = broken image, data: renders).
// blob: stays in the CSP for images the report's OWN scripts mint inside
// the frame, where minter and reader share the opaque origin.
const REPORT_FRAME_SANDBOX = "allow-scripts";
const REPORT_FRAME_CSP =
  "default-src 'none'; img-src data: blob:; media-src data: blob:; " +
  "style-src 'unsafe-inline'; font-src data:; script-src 'unsafe-inline'";

// Reports beyond this size are not pulled into memory and re-serialized into
// a srcdoc (data-URI figure inlining can inflate them further); the canvas
// offers the download instead. Generous on purpose — a self-contained
// benchmark page with embedded figures runs single-digit MB.
export const MAX_INLINE_REPORT_BYTES = 25 * 1024 * 1024;

const stripMarkdownInline = (value: string): string =>
  value
    .replace(/!\[[^\]]*]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim();

// The report names itself better than its filename does: prefer the HTML
// <title> / first markdown H1 for the chrome, fall back to the artifact name.
const markdownHeadingTitle = (content: string): string | null => {
  for (const line of String(content || "").split(/\r?\n/)) {
    const match = /^#\s+(.+)$/.exec(line.trim());
    if (match) {
      const title = stripMarkdownInline(match[1]);
      if (title) {
        return title;
      }
    }
  }
  return null;
};

type PreparedHtmlReport = {
  srcdoc: string;
  title: string | null;
};

// Prepare model-generated HTML for the sandboxed frame:
// - drop <base> so the document cannot re-point relative resolution,
// - resolve <img> references to run artifacts and inline them as data: URIs
//   (the frame's opaque origin can fetch neither /v2 with the user's cookies
//   nor the host's blob store — the host fetches and embeds instead),
// - inject the CSP as the first meta so everything else loads under it.
// Exported for tests.
export const prepareHtmlReportDocument = async (
  rawHtml: string,
  imageArtifacts: ReportCanvasVersion["imageArtifacts"],
  fetchImageDataUrl: (url: string) => Promise<string | null>
): Promise<PreparedHtmlReport> => {
  const parsed = new DOMParser().parseFromString(String(rawHtml || ""), "text/html");
  parsed.querySelectorAll("base").forEach((element) => element.remove());

  const images = Array.from(parsed.querySelectorAll("img"));
  await Promise.all(
    images.map(async (image) => {
      const src = String(image.getAttribute("src") || "").trim();
      if (!src || /^(data|blob):/i.test(src)) {
        return;
      }
      const artifactUrl = resolveRunOutputArtifactUrl(src, imageArtifacts);
      if (!artifactUrl) {
        // Unknown reference (external image, missing artifact): leave it for
        // the CSP to block rather than guessing. The alt text still renders.
        return;
      }
      try {
        const dataUrl = await fetchImageDataUrl(artifactUrl);
        if (dataUrl) {
          image.setAttribute("src", dataUrl);
        }
      } catch {
        // Non-blocking: a missing figure must not take down the report.
      }
    })
  );

  const csp = parsed.createElement("meta");
  csp.setAttribute("http-equiv", "Content-Security-Policy");
  csp.setAttribute("content", REPORT_FRAME_CSP);
  parsed.head.insertBefore(csp, parsed.head.firstChild);

  return {
    srcdoc: `<!doctype html>${parsed.documentElement.outerHTML}`,
    title: String(parsed.title || "").trim() || null,
  };
};

const fetchArtifactImageDataUrl = async (url: string): Promise<string | null> => {
  const response = await fetch(url, { method: "GET", credentials: "include" });
  if (!response.ok) {
    return null;
  }
  const blob = await response.blob();
  return await new Promise<string | null>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(blob);
  });
};

function CanvasStatus({
  status,
  error,
  downloadUrl,
  sizeBytes,
}: {
  status: CanvasBodyStatus;
  error: string;
  downloadUrl: string;
  sizeBytes?: number;
}) {
  if (status === "loading") {
    return (
      <div className="report-canvas-status" role="status">
        <Loader2 className="size-4 animate-spin" aria-hidden="true" />
        <span>Loading report…</span>
      </div>
    );
  }
  if (status === "oversize") {
    /* Not an error — a deliberate boundary, stated in the muted voice. */
    return (
      <div className="report-canvas-status">
        <span>
          This report is {formatBytes(sizeBytes ?? 0)} — too large to render inline.{" "}
          <a href={downloadUrl} download className="report-canvas-inline-link">
            Download it
          </a>{" "}
          to read locally.
        </span>
      </div>
    );
  }
  return (
    <div className="report-canvas-status report-canvas-status-error">
      <AlertCircle className="size-4" aria-hidden="true" />
      <span>
        {error} —{" "}
        <a href={downloadUrl} download className="report-canvas-inline-link">
          download instead
        </a>
        .
      </span>
    </div>
  );
}

export function ReportCanvas({
  versions,
  mode,
  closing = false,
  onClose,
  loadDocumentText,
}: ReportCanvasProps) {
  // null = follow the latest registration (new versions replace the view);
  // a number = the reader pinned an older registration from the chip.
  const [pinnedVersionIndex, setPinnedVersionIndex] = useState<number | null>(null);
  const latestIndex = versions.length - 1;
  const activeIndex =
    pinnedVersionIndex !== null && pinnedVersionIndex >= 0 && pinnedVersionIndex <= latestIndex
      ? pinnedVersionIndex
      : latestIndex;
  const active = versions[activeIndex];
  const format = active
    ? runReportDocumentFormat(active.document.path, active.document.mimeType)
    : null;

  /* Loaded content carries the key of the registration it belongs to; a key
     mismatch at render time IS the loading state. No synchronous setState in
     the effect, no stale content flashing while a new version fetches. */
  type LoadedReport =
    | { key: string; state: "markdown"; text: string }
    | { key: string; state: "html"; report: PreparedHtmlReport }
    | { key: string; state: "error"; message: string };
  const [loadedReport, setLoadedReport] = useState<LoadedReport | null>(null);

  // Fetch + prepare whenever the active registration changes. `status` stays
  // out of the dependencies (the ReportReader lesson: depending on it cancels
  // the in-flight read the moment it starts). The image-artifact list arrives
  // with a fresh identity on every message update, so the effect keys on a
  // VALUE fingerprint — identity alone would refetch the report on each
  // streaming delta of an unrelated later turn.
  const activeDownloadUrl = active?.document.downloadUrl ?? "";
  const activeImageArtifacts = active?.imageArtifacts;
  const imageArtifactsFingerprint = useMemo(
    () =>
      (activeImageArtifacts ?? [])
        .map((artifact) => `${artifact.path}|${artifact.url ?? artifact.downloadUrl ?? ""}`)
        .join("\n"),
    [activeImageArtifacts]
  );
  const imageArtifactsRef = useRef(activeImageArtifacts ?? []);
  useEffect(() => {
    imageArtifactsRef.current = activeImageArtifacts ?? [];
  }, [activeImageArtifacts]);
  const contentKey = `${activeDownloadUrl} ${imageArtifactsFingerprint}`;
  /* Derived at render, guarded in the effect: an oversized report is never
     pulled into memory, and the status below states the boundary quietly. */
  const activeOversized =
    (active?.document.sizeBytes ?? 0) > MAX_INLINE_REPORT_BYTES;
  useEffect(() => {
    if (!activeDownloadUrl || activeOversized) {
      return undefined;
    }
    let cancelled = false;
    const run = async () => {
      try {
        const text = await loadDocumentText(activeDownloadUrl);
        if (cancelled) {
          return;
        }
        if (format === "html") {
          const prepared = await prepareHtmlReportDocument(
            text,
            imageArtifactsRef.current,
            fetchArtifactImageDataUrl
          );
          if (cancelled) {
            return;
          }
          setLoadedReport({ key: contentKey, state: "html", report: prepared });
        } else {
          setLoadedReport({ key: contentKey, state: "markdown", text });
        }
      } catch (cause: unknown) {
        if (!cancelled) {
          setLoadedReport({
            key: contentKey,
            state: "error",
            message: cause instanceof Error ? cause.message : "Unable to load the report",
          });
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
    /* contentKey already embeds activeDownloadUrl, so the extra dep can never
       fire alone — it is here to keep exhaustive-deps literal. */
  }, [activeDownloadUrl, activeOversized, contentKey, format, loadDocumentText]);

  const current = loadedReport && loadedReport.key === contentKey ? loadedReport : null;
  const status: CanvasBodyStatus = activeOversized
    ? "oversize"
    : current === null
      ? "loading"
      : current.state === "error"
        ? "error"
        : "ready";
  const errorMessage = current?.state === "error" ? current.message : "";
  const markdownContent = current?.state === "markdown" ? current.text : "";
  const htmlReport = current?.state === "html" ? current.report : null;

  const renderedMarkdown = useMemo(
    () =>
      format === "markdown"
        ? rewriteArtifactMarkdownImageUrls(markdownContent, activeImageArtifacts ?? [])
        : "",
    [format, markdownContent, activeImageArtifacts]
  );

  const displayTitle = useMemo(() => {
    if (format === "html" && htmlReport?.title) {
      return htmlReport.title;
    }
    if (format === "markdown" && markdownContent) {
      return markdownHeadingTitle(markdownContent) ?? active?.document.title ?? "Report";
    }
    return active?.document.title ?? "Report";
  }, [format, htmlReport, markdownContent, active]);

  if (!active) {
    return null;
  }

  const versionNumber = activeIndex + 1;
  const provenance = [
    "Report",
    active.runId ? active.runId : null,
    versions.length > 1 || versionNumber > 1 ? `v${versionNumber}` : null,
    activeIndex !== latestIndex ? "superseded" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <aside
      id="report-canvas"
      className="report-canvas"
      data-mode={mode}
      data-closing={closing ? "true" : undefined}
      role={mode === "sheet" ? "dialog" : "complementary"}
      aria-modal={mode === "sheet" ? true : undefined}
      aria-label={`Report: ${displayTitle}`}
    >
      <div className="report-canvas-frame">
        <header className="report-canvas-head">
          {mode === "sheet" ? (
            <button
              type="button"
              className="report-canvas-button report-canvas-back"
              onClick={onClose}
              aria-label="Back to chat"
            >
              <ArrowLeft className="size-4" aria-hidden="true" />
            </button>
          ) : null}
          <div className="report-canvas-heading">
            <span className="report-canvas-title" title={displayTitle}>
              {displayTitle}
            </span>
            <span className="report-canvas-meta">{provenance}</span>
          </div>
          <div className="report-canvas-actions">
            {versions.length > 1 ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    className="report-canvas-version-chip"
                    aria-label={`Version ${versionNumber} of ${versions.length}`}
                  >
                    v{versionNumber}
                    <ChevronDown className="size-3" aria-hidden="true" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="report-canvas-version-menu">
                  {versions
                    .map((version, index) => ({ version, index }))
                    .reverse()
                    .map(({ version, index }) => (
                      <DropdownMenuItem
                        key={`${version.messageId}-${index}`}
                        onSelect={() =>
                          setPinnedVersionIndex(index === latestIndex ? null : index)
                        }
                      >
                        <span className="report-canvas-version-row">
                          <span>
                            v{index + 1}
                            {index === latestIndex ? " · latest" : ""}
                          </span>
                          {version.runId ? (
                            <span className="report-canvas-version-run">{version.runId}</span>
                          ) : null}
                        </span>
                        {index === activeIndex ? (
                          <Check className="size-3.5" aria-hidden="true" />
                        ) : null}
                      </DropdownMenuItem>
                    ))}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null}
            <a
              className="report-canvas-button"
              href={active.document.downloadUrl}
              download
              aria-label={`Download ${displayTitle}`}
              title="Download"
            >
              <Download className="size-4" aria-hidden="true" />
            </a>
            {mode !== "sheet" ? (
              <>
                <span className="report-canvas-separator" aria-hidden="true" />
                <button
                  type="button"
                  className="report-canvas-button"
                  onClick={onClose}
                  aria-label="Close report canvas"
                  title="Close"
                >
                  <X className="size-4" aria-hidden="true" />
                </button>
              </>
            ) : null}
          </div>
        </header>
        <div className="report-canvas-body">
          {status !== "ready" ? (
            <CanvasStatus
              status={status}
              error={errorMessage}
              downloadUrl={active.document.downloadUrl}
              sizeBytes={active.document.sizeBytes}
            />
          ) : format === "html" && htmlReport ? (
            <iframe
              className="report-canvas-html-frame"
              title={`Report: ${displayTitle}`}
              sandbox={REPORT_FRAME_SANDBOX}
              referrerPolicy="no-referrer"
              srcDoc={htmlReport.srcdoc}
            />
          ) : (
            <article className="report-canvas-article">
              <Markdown className="report-canvas-markdown">{renderedMarkdown}</Markdown>
            </article>
          )}
        </div>
      </div>
    </aside>
  );
}

export default ReportCanvas;
