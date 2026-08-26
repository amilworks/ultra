import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
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
  runDocumentCodeLanguage,
  runDocumentPreviewFormat,
  rewriteArtifactMarkdownImageUrls,
  type RunDocumentKind,
} from "@/features/chat/run-artifact-hydration";
import { CodeBlockCode } from "@/components/prompt-kit/code-block";
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
    kind: RunDocumentKind;
    mimeType?: string;
    sizeBytes?: number;
  };
  imageArtifacts: Array<{ path: string; url?: string; downloadUrl?: string }>;
};

// split: a user-resizable column in the stage grid — reader sees chat AND
//        report. Engages whenever the stage affords both minimum widths.
// sheet: full-screen immersion everywhere else. There is deliberately no
//        floating middle state: a panel half-covering prose serves neither
//        reading nor chat, and it turns the sidebar into a competing overlay.
export type ReportCanvasMode = "split" | "sheet";

export type ReportCanvasSplitBounds = { min: number; max: number };

export type ReportCanvasProps = {
  versions: ReportCanvasVersion[];
  mode: ReportCanvasMode;
  closing?: boolean;
  onClose: () => void;
  loadDocumentText: (downloadUrl: string) => Promise<string>;
  /* Split-regime resize. Live drag writes the width variable imperatively on
     the stage grid so the App tree never renders at pointer speed; the host
     receives one commit per gesture (and per keystroke) to persist. */
  splitWidth?: number;
  splitWidthBounds?: ReportCanvasSplitBounds;
  onSplitWidthCommit?: (width: number) => void;
  onSplitWidthReset?: () => void;
};

const DEFAULT_SPLIT_BOUNDS: ReportCanvasSplitBounds = { min: 320, max: 896 };
const SPLIT_KEYBOARD_STEP = 16;

type CanvasBodyStatus = "loading" | "ready" | "error" | "oversize" | "unsupported";

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

// Fragment links (#section — every report's table of contents) cannot
// navigate inside a sandboxed srcdoc frame: the browser treats the click as
// a document navigation to about:srcdoc, which cannot be re-fetched, so the
// frame either swallows the click or replaces the report with an error page
// (traced live on a delivered report whose TOC chips blanked the canvas).
// The author meant same-document scrolling, so this injected shim does
// exactly that: capture-phase, preventDefault always (a missing target must
// no-op, never blank the page), honoring reduced motion.
const FRAGMENT_NAV_SHIM =
  "(function () {" +
  '  var reduce = false;' +
  '  try { reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) {}' +
  '  document.addEventListener("click", function (event) {' +
  "    if (event.defaultPrevented || event.button !== 0) return;" +
  "    var origin = event.target;" +
  "    var anchor = origin && origin.closest ? origin.closest('a[href^=\"#\"]') : null;" +
  "    if (!anchor) return;" +
  "    event.preventDefault();" +
  '    var raw = anchor.getAttribute("href").slice(1);' +
  "    if (!raw) return;" +
  "    var id = raw;" +
  "    try { id = decodeURIComponent(raw); } catch (e) {}" +
  "    var target = document.getElementById(id);" +
  "    if (!target) {" +
  "      var named = document.getElementsByName(id);" +
  "      target = named && named.length ? named[0] : null;" +
  "    }" +
  "    if (target && target.scrollIntoView) {" +
  '      target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });' +
  "    }" +
  "  }, true);" +
  "})();";

// Reports beyond this size are not pulled into memory and re-serialized into
// a srcdoc (data-URI figure inlining can inflate them further); the canvas
// offers the download instead. Generous on purpose — a self-contained
// benchmark page with embedded figures runs single-digit MB.
export const MAX_INLINE_REPORT_BYTES = 25 * 1024 * 1024;
// Source artifacts favor fast inspection over exhaustive loading. This is the
// same desktop head budget as the Resources text viewer; larger outputs keep
// their download path without making a click allocate an entire generated log
// or table in the chat process.
export const MAX_INLINE_SOURCE_BYTES = 2 * 1024 * 1024;

const DOCUMENT_KIND_LABEL: Record<RunDocumentKind, string> = {
  report: "Report",
  code: "Code",
  data: "Data",
  document: "Document",
};

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

  // Last child of <body>, capture-phase listener: wins over page handlers and
  // covers content scripts append later. CSP allows it (script-src
  // 'unsafe-inline'); the sandbox allows it (allow-scripts).
  const navShim = parsed.createElement("script");
  navShim.textContent = FRAGMENT_NAV_SHIM;
  parsed.body.appendChild(navShim);

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
  kindLabel,
  sizeBytes,
}: {
  status: CanvasBodyStatus;
  error: string;
  downloadUrl: string;
  kindLabel: string;
  sizeBytes?: number;
}) {
  const noun = kindLabel.toLowerCase();
  if (status === "loading") {
    return (
      <div className="report-canvas-status" role="status">
        <Loader2 className="size-4 animate-spin" aria-hidden="true" />
        <span>Loading {noun}…</span>
      </div>
    );
  }
  if (status === "oversize") {
    /* Not an error — a deliberate boundary, stated in the muted voice. */
    return (
      <div className="report-canvas-status">
        <span>
          This {noun} is {formatBytes(sizeBytes ?? 0)} — too large to render inline.{" "}
          <a href={downloadUrl} download className="report-canvas-inline-link">
            Download it
          </a>{" "}
          to read locally.
        </span>
      </div>
    );
  }
  if (status === "unsupported") {
    return (
      <div className="report-canvas-status">
        <span>
          Preview is not available for this {noun} format yet —{" "}
          <a href={downloadUrl} download className="report-canvas-inline-link">
            download it
          </a>{" "}
          to inspect locally.
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
  splitWidth,
  splitWidthBounds,
  onSplitWidthCommit,
  onSplitWidthReset,
}: ReportCanvasProps) {
  // null = follow the latest registration (new versions replace the view);
  // a number = the reader pinned an older registration from the chip.
  const [pinnedVersionIndex, setPinnedVersionIndex] = useState<number | null>(null);
  const rootRef = useRef<HTMLElement | null>(null);
  /* Non-null only mid-gesture; the committed width lives with the host. */
  const [liveSplitWidth, setLiveSplitWidth] = useState<number | null>(null);
  const splitDragRef = useRef<{ pointerId: number; frameRight: number } | null>(null);
  const splitBounds = splitWidthBounds ?? DEFAULT_SPLIT_BOUNDS;
  const resolvedSplitWidth = liveSplitWidth ?? splitWidth;

  const clampSplitWidth = useCallback(
    (width: number) =>
      Math.round(Math.min(splitBounds.max, Math.max(splitBounds.min, width))),
    [splitBounds.max, splitBounds.min]
  );

  const stageShellElement = useCallback(
    () => rootRef.current?.closest(".app-main-shell") as HTMLElement | null,
    []
  );

  /* Pointer capture keeps the move stream on the handle even while the
     cursor crosses the report iframe — without it the frame swallows the
     gesture. The width variable is written straight onto the stage grid so
     dragging never renders the App tree. */
  const applyLiveSplitWidth = useCallback(
    (width: number) => {
      const clamped = clampSplitWidth(width);
      stageShellElement()?.style.setProperty("--report-canvas-col", `${clamped}px`);
      setLiveSplitWidth(clamped);
      return clamped;
    },
    [clampSplitWidth, stageShellElement]
  );

  const endSplitDrag = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, commit: boolean) => {
      const drag = splitDragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) {
        return;
      }
      splitDragRef.current = null;
      event.currentTarget.releasePointerCapture?.(event.pointerId);
      stageShellElement()?.removeAttribute("data-canvas-resizing");
      if (commit) {
        onSplitWidthCommit?.(applyLiveSplitWidth(drag.frameRight - event.clientX));
      }
      setLiveSplitWidth(null);
    },
    [applyLiveSplitWidth, onSplitWidthCommit, stageShellElement]
  );
  const latestIndex = versions.length - 1;
  const activeIndex =
    pinnedVersionIndex !== null && pinnedVersionIndex >= 0 && pinnedVersionIndex <= latestIndex
      ? pinnedVersionIndex
      : latestIndex;
  const active = versions[activeIndex];
  const activeKind = active?.document.kind ?? "document";
  const kindLabel = DOCUMENT_KIND_LABEL[activeKind];
  const format = active
    ? runDocumentPreviewFormat(active.document.path, active.document.mimeType)
    : null;
  const sourceLanguage = active
    ? runDocumentCodeLanguage(active.document.path, active.document.mimeType)
    : "text";

  /* Loaded content carries the key of the registration it belongs to; a key
     mismatch at render time IS the loading state. No synchronous setState in
     the effect, no stale content flashing while a new version fetches. */
  type LoadedReport =
    | { key: string; state: "markdown"; text: string }
    | { key: string; state: "html"; report: PreparedHtmlReport }
    | { key: string; state: "source"; text: string }
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
      format === "html" || format === "markdown"
        ? (activeImageArtifacts ?? [])
            .map((artifact) => `${artifact.path}|${artifact.url ?? artifact.downloadUrl ?? ""}`)
            .join("\n")
        : "",
    [activeImageArtifacts, format]
  );
  const imageArtifactsRef = useRef(activeImageArtifacts ?? []);
  useEffect(() => {
    imageArtifactsRef.current = activeImageArtifacts ?? [];
  }, [activeImageArtifacts]);
  const contentKey = `${activeDownloadUrl} ${format ?? "unsupported"} ${imageArtifactsFingerprint}`;
  const textByteLimit = activeKind === "report"
    ? MAX_INLINE_REPORT_BYTES
    : MAX_INLINE_SOURCE_BYTES;
  const loadsText = format === "html" || format === "markdown" || format === "source";
  /* Derived at render, guarded in the effect: an oversized readable artifact
     is never pulled into memory, and binary/PDF formats never enter this path. */
  const activeOversized =
    loadsText && (active?.document.sizeBytes ?? 0) > textByteLimit;
  useEffect(() => {
    if (!activeDownloadUrl || activeOversized || !loadsText) {
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
        } else if (format === "markdown") {
          setLoadedReport({ key: contentKey, state: "markdown", text });
        } else {
          setLoadedReport({ key: contentKey, state: "source", text });
        }
      } catch (cause: unknown) {
        if (!cancelled) {
          setLoadedReport({
            key: contentKey,
            state: "error",
            message: cause instanceof Error ? cause.message : "Unable to load the artifact",
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
  }, [activeDownloadUrl, activeOversized, contentKey, format, loadDocumentText, loadsText]);

  const current = loadedReport && loadedReport.key === contentKey ? loadedReport : null;
  const status: CanvasBodyStatus = format === null
    ? "unsupported"
    : format === "pdf"
      ? "ready"
      : activeOversized
        ? "oversize"
        : current === null
          ? "loading"
          : current.state === "error"
            ? "error"
            : "ready";
  const errorMessage = current?.state === "error" ? current.message : "";
  const markdownContent = current?.state === "markdown" ? current.text : "";
  const sourceContent = current?.state === "source" ? current.text : "";
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
      return markdownHeadingTitle(markdownContent) ?? active?.document.title ?? kindLabel;
    }
    return active?.document.title ?? kindLabel;
  }, [format, htmlReport, markdownContent, active, kindLabel]);

  if (!active) {
    return null;
  }

  const versionNumber = activeIndex + 1;
  const provenance = [
    kindLabel,
    active.runId ? active.runId : null,
    versions.length > 1 || versionNumber > 1 ? `v${versionNumber}` : null,
    activeIndex !== latestIndex ? "superseded" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <aside
      id="report-canvas"
      ref={rootRef}
      className="report-canvas"
      data-mode={mode}
      data-closing={closing ? "true" : undefined}
      role={mode === "sheet" ? "dialog" : "complementary"}
      aria-modal={mode === "sheet" ? true : undefined}
      aria-label={`Preview: ${displayTitle}`}
    >
      {mode === "split" ? (
        <div
          className="report-canvas-resize"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize artifact preview"
          aria-valuemin={splitBounds.min}
          aria-valuemax={splitBounds.max}
          aria-valuenow={resolvedSplitWidth}
          tabIndex={0}
          onPointerDown={(event) => {
            if (event.button !== 0) {
              return;
            }
            const frame = rootRef.current?.querySelector(".report-canvas-frame");
            if (!(frame instanceof HTMLElement)) {
              return;
            }
            event.preventDefault();
            /* preventDefault (needed so a drag never selects transcript
               text) also suppresses pointerdown's default focus — restore
               it so the keyboard path is one click away. */
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture?.(event.pointerId);
            /* The panel's right edge is anchored to the stage padding, so it
               is a stable reference for the whole gesture. */
            splitDragRef.current = {
              pointerId: event.pointerId,
              frameRight: frame.getBoundingClientRect().right,
            };
            stageShellElement()?.setAttribute("data-canvas-resizing", "true");
          }}
          onPointerMove={(event) => {
            const drag = splitDragRef.current;
            if (!drag || event.pointerId !== drag.pointerId) {
              return;
            }
            applyLiveSplitWidth(drag.frameRight - event.clientX);
          }}
          onPointerUp={(event) => endSplitDrag(event, true)}
          onPointerCancel={(event) => endSplitDrag(event, false)}
          onDoubleClick={() => onSplitWidthReset?.()}
          onKeyDown={(event) => {
            const current = resolvedSplitWidth;
            if (current === undefined) {
              return;
            }
            let next: number | null = null;
            if (event.key === "ArrowLeft") {
              next = clampSplitWidth(current + SPLIT_KEYBOARD_STEP);
            } else if (event.key === "ArrowRight") {
              next = clampSplitWidth(current - SPLIT_KEYBOARD_STEP);
            } else if (event.key === "Home") {
              next = splitBounds.min;
            } else if (event.key === "End") {
              next = splitBounds.max;
            }
            if (next === null || next === current) {
              if (next !== null) {
                event.preventDefault();
              }
              return;
            }
            event.preventDefault();
            stageShellElement()?.style.setProperty("--report-canvas-col", `${next}px`);
            onSplitWidthCommit?.(next);
          }}
        />
      ) : null}
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
                  aria-label="Close artifact preview"
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
              kindLabel={kindLabel}
              sizeBytes={active.document.sizeBytes}
            />
          ) : format === "html" && htmlReport ? (
            <iframe
              className="report-canvas-html-frame"
              title={`Preview: ${displayTitle}`}
              sandbox={REPORT_FRAME_SANDBOX}
              referrerPolicy="no-referrer"
              srcDoc={htmlReport.srcdoc}
            />
          ) : format === "pdf" ? (
            <iframe
              className="report-canvas-pdf-frame"
              title={`PDF preview: ${displayTitle}`}
              src={active.document.downloadUrl}
              referrerPolicy="no-referrer"
            />
          ) : format === "source" ? (
            <div className="report-canvas-source">
              <CodeBlockCode
                code={sourceContent}
                language={sourceLanguage}
                showToolbar={false}
                showLanguage={false}
                showCopyButton={false}
                className="report-canvas-source-code"
              />
            </div>
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
