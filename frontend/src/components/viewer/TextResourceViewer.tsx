import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type UIEvent,
} from "react";
import { Copy, Download, FileCode2, Info, Loader2, Table2, WrapText } from "lucide-react";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { LazyMarkdown } from "@/components/prompt-kit/lazy-markdown";
import { useBreakpoint } from "@/hooks/use-breakpoint";
import { looksNumeric, parseCsv, sniffDelimiter } from "@/lib/csv";
import {
  delimiterLabel,
  eolLabel,
  extensionOf,
  formatBytes,
  formatChipLabel,
  isGzipName,
  type TextResourceKind,
} from "@/lib/textFormat";
import type { ApiClient } from "@/lib/api";
import type { ResourceCsvRows, ResourceTextHead, UploadedFileRecord } from "@/types";

import { highlightLine } from "./text-highlight";

const HEAD_BYTES_DESKTOP = 2_000_000;
const HEAD_BYTES_PHONE = 1_000_000;
const CSV_PAGE = 200;
const ROW_HEIGHT = 22; // code lines (pinned in CSS)
const CSV_ROW_HEIGHT = 33; // table rows (pinned in CSS to match this model)
const OVERSCAN = 12;
// Wrap mode lays out a single <pre>; cap it so enabling wrap on a multi-MB head
// can never trigger a multi-second synchronous reflow. The unwrapped (windowed)
// view always shows the whole head.
const WRAP_CHAR_CAP = 200_000;

type TextView = "table" | "raw" | "json" | "yaml" | "rendered" | "source";

type Props = {
  file: UploadedFileRecord;
  kind: TextResourceKind;
  apiClient: ApiClient;
};

// useElementSize tracks a container's pixel height so the windowed surfaces render
// only the rows in view.
function useElementHeight<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [height, setHeight] = useState(520);
  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return;
    }
    // Functional setState + an empty dep list: one observer per mount (no churn),
    // and a fallback measurement when ResizeObserver is unavailable.
    const apply = (next: number | undefined) => {
      if (next) {
        setHeight((prev) => (Math.abs(next - prev) > 1 ? next : prev));
      }
    };
    if (typeof ResizeObserver === "undefined") {
      apply(node.clientHeight);
      return;
    }
    const observer = new ResizeObserver((entries) => apply(entries[0]?.contentRect.height));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return { ref, height };
}

async function decompressGzipHead(
  bytes: Uint8Array,
  maxBytes: number
): Promise<{ text: string; ok: boolean; truncated: boolean }> {
  const Decompression = (globalThis as { DecompressionStream?: typeof DecompressionStream }).DecompressionStream;
  if (!Decompression) {
    return { text: "", ok: false, truncated: false };
  }
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new Decompression("gzip"));
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let out = "";
  let decodedBytes = 0; // bound by decompressed BYTES, not UTF-16 code units
  let truncated = false;
  try {
    while (decodedBytes < maxBytes) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        decodedBytes += value.byteLength;
        // TextDecoder stream-mode holds back partial multibyte sequences, so the
        // accumulated string is always valid UTF-8 (no surrogate splitting).
        out += decoder.decode(value, { stream: true });
      }
    }
    if (decodedBytes >= maxBytes) {
      truncated = true;
    }
  } catch {
    // A Range head of a gzip stream ends mid-member; keep what decoded cleanly.
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* ignore */
    }
  }
  return { text: out, ok: out.length > 0, truncated };
}

function toggleOptions(kind: TextResourceKind): { value: TextView; label: string }[] {
  switch (kind) {
    case "csv":
      return [
        { value: "table", label: "Table" },
        { value: "raw", label: "Raw" },
      ];
    case "json":
      return [
        { value: "json", label: "JSON" },
        { value: "yaml", label: "YAML" },
      ];
    case "yaml":
      return [
        { value: "yaml", label: "YAML" },
        { value: "json", label: "JSON" },
      ];
    case "markdown":
      return [
        { value: "rendered", label: "Rendered" },
        { value: "source", label: "Source" },
      ];
    default:
      return [];
  }
}

function defaultView(kind: TextResourceKind): TextView {
  switch (kind) {
    case "csv":
      return "table";
    case "json":
      return "json";
    case "yaml":
      return "yaml";
    case "markdown":
      return "rendered";
    default:
      return "source";
  }
}

export function TextResourceViewer({ file, kind, apiClient }: Props) {
  const isPhoneView = useBreakpoint(641);
  const headBytes = isPhoneView ? HEAD_BYTES_PHONE : HEAD_BYTES_DESKTOP;
  const gzip = isGzipName(file.original_name);
  const usesServerCsv = kind === "csv" && !gzip;

  const [view, setView] = useState<TextView>(defaultView(kind));
  const [wrap, setWrap] = useState(false);
  const [copied, setCopied] = useState(false);

  const [head, setHead] = useState<ResourceTextHead | null>(null);
  const [headLoading, setHeadLoading] = useState(false);
  const [headError, setHeadError] = useState<string | null>(null);

  const [csv, setCsv] = useState<ResourceCsvRows | null>(null);
  const [csvRows, setCsvRows] = useState<string[][]>([]);
  const [csvLoading, setCsvLoading] = useState(false);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvLoadingMore, setCsvLoadingMore] = useState(false);

  const [converted, setConverted] = useState<{ view: TextView; text: string } | null>(null);
  const [convertError, setConvertError] = useState(false);

  const csvLoadingMoreRef = useRef(false);

  const options = toggleOptions(kind);
  const downloadUrl = apiClient.resourceDownloadUrl(file.file_id);

  const needsHead =
    !usesServerCsv || view === "raw" || (kind === "csv" && gzip);

  // --- head fetch (text / structured / markdown / gzip) ---------------------
  useEffect(() => {
    // Guard on the result (head), not a loading flag: keeps the effect from
    // refetching once loaded AND stays StrictMode-safe (the dev double-mount
    // cancels the first run, and the second re-runs because head is still null).
    if (!needsHead || head) {
      return;
    }
    let cancelled = false;
    const run = async () => {
      setHeadLoading(true);
      setHeadError(null);
      try {
        if (gzip) {
          const range = await apiClient.fetchResourceRangeBytes(file.file_id, { maxBytes: headBytes });
          const { text, ok, truncated: outTruncated } = await decompressGzipHead(range.bytes, headBytes);
          if (cancelled) {
            return;
          }
          if (!ok) {
            setHeadError("This compressed file could not be decompressed in the browser. Download it to view.");
            return;
          }
          const innerExt = extensionOf(file.original_name);
          setHead({
            file_id: file.file_id,
            original_name: file.original_name,
            content_type: file.content_type ?? "",
            format: innerExt === "tsv" ? "csv" : kind,
            total_size_bytes: range.totalBytes ?? text.length,
            offset: 0,
            returned_bytes: text.length,
            next_offset: text.length,
            truncated: range.truncated || outTruncated,
            encoding: "utf-8",
            eol: text.includes("\r\n") ? "crlf" : text.includes("\n") ? "lf" : "none",
            line_count: text ? text.split("\n").length : 0,
            approx_total_lines: text ? text.split("\n").length : 0,
            text,
          });
        } else {
          const result = await apiClient.resourceTextHead(file.file_id, { maxBytes: headBytes });
          if (!cancelled) {
            setHead(result);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setHeadError(error instanceof Error ? error.message : "Unable to load this file.");
        }
      } finally {
        if (!cancelled) {
          setHeadLoading(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [needsHead, head, gzip, headBytes, file.file_id, file.original_name, file.content_type, kind, apiClient]);

  // --- CSV first page (server) ---------------------------------------------
  useEffect(() => {
    if (!usesServerCsv || csv) {
      return;
    }
    let cancelled = false;
    const run = async () => {
      setCsvLoading(true);
      setCsvError(null);
      try {
        const page = await apiClient.resourceCsvRows(file.file_id, { limit: CSV_PAGE });
        if (cancelled) {
          return;
        }
        setCsv(page);
        setCsvRows(page.rows);
      } catch (error) {
        if (!cancelled) {
          setCsvError(error instanceof Error ? error.message : "Unable to load this table.");
        }
      } finally {
        if (!cancelled) {
          setCsvLoading(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [usesServerCsv, csv, file.file_id, apiClient]);

  // --- gzip CSV: parse the decompressed head into a table -------------------
  const gzipCsv = useMemo(() => {
    if (!gzip || kind !== "csv" || !head) {
      return null;
    }
    const delimiter = extensionOf(file.original_name) === "tsv" ? "\t" : sniffDelimiter(head.text);
    const { rows } = parseCsv(head.text, delimiter, head.truncated);
    const columns = rows.length > 0 ? rows[0] : [];
    return { columns, rows: rows.slice(1), delimiter };
  }, [gzip, kind, head, file.original_name]);

  const loadMoreCsv = useCallback(() => {
    // Synchronous ref latch: scroll events can fire several times before the
    // csvLoadingMore state re-render commits, which would otherwise double-fetch
    // the same page and append duplicate rows.
    if (!csv || !csv.has_more || csvLoadingMoreRef.current) {
      return;
    }
    csvLoadingMoreRef.current = true;
    setCsvLoadingMore(true);
    apiClient
      .resourceCsvRows(file.file_id, {
        offsetBytes: csv.next_offset_bytes,
        limit: CSV_PAGE,
        delimiter: csv.delimiter,
      })
      .then((page) => {
        setCsv((prev) => (prev ? { ...page, columns: prev.columns } : page));
        setCsvRows((prev) => [...prev, ...page.rows]);
      })
      .catch(() => {
        /* keep what we have; the user can still download */
      })
      .finally(() => {
        csvLoadingMoreRef.current = false;
        setCsvLoadingMore(false);
      });
  }, [csv, file.file_id, apiClient]);

  // --- JSON <-> YAML conversion (lazy js-yaml), on the bounded head ----------
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setConvertError(false);
      setConverted(null);
      if (!head || (kind !== "json" && kind !== "yaml") || view === kind) {
        return;
      }
      try {
        const yaml = await import("js-yaml");
        if (kind === "json" && view === "yaml") {
          const parsed = JSON.parse(head.text);
          const dumped = yaml.dump(parsed, { indent: 2, lineWidth: 100, noRefs: true });
          if (!cancelled) {
            setConverted({ view, text: dumped });
          }
        } else if (kind === "yaml" && view === "json") {
          const parsed = yaml.load(head.text, { json: true });
          const dumped = JSON.stringify(parsed, null, 2);
          if (!cancelled) {
            setConverted({ view, text: dumped });
          }
        }
      } catch {
        if (!cancelled) {
          setConvertError(true);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [head, kind, view]);

  // The text + format the code surface should render for the current view.
  const display = useMemo<{ text: string; format: TextResourceKind }>(() => {
    if (!head) {
      return { text: "", format: kind };
    }
    if (kind === "json" && view === "json") {
      try {
        return { text: JSON.stringify(JSON.parse(head.text), null, 2), format: "json" };
      } catch {
        return { text: head.text, format: "json" };
      }
    }
    if ((kind === "json" || kind === "yaml") && view !== kind) {
      if (converted && converted.view === view) {
        return { text: converted.text, format: view as TextResourceKind };
      }
      return { text: head.text, format: kind };
    }
    return { text: head.text, format: kind };
  }, [head, kind, view, converted]);

  const handleCopy = useCallback(() => {
    let payload = display.text;
    if (view === "table") {
      const cols = csv?.columns ?? gzipCsv?.columns ?? [];
      const rows = usesServerCsv ? csvRows : gzipCsv?.rows ?? [];
      payload = [cols, ...rows].map((row) => row.join("\t")).join("\n");
    }
    if (!payload) {
      return;
    }
    const markCopied = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    };
    const run = async () => {
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(payload);
          markCopied();
          return;
        }
      } catch {
        // Permission denied / not focused — fall back below.
      }
      // Fallback for insecure contexts / older webviews where the async API is absent.
      try {
        const textarea = document.createElement("textarea");
        textarea.value = payload;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        markCopied();
      } catch {
        // Give up quietly; the Download action remains available.
      }
    };
    void run();
  }, [display.text, view, csv, gzipCsv, csvRows, usesServerCsv]);

  const isTableView = view === "table";
  const showWrapToggle = !isTableView && view !== "rendered";
  // While a JSON<->YAML conversion is resolving, show a loader instead of briefly
  // flashing the original document under the newly-selected tab.
  const converting =
    head !== null && (kind === "json" || kind === "yaml") && view !== kind && !converted && !convertError;

  const chipLabel = formatChipLabel(kind, file.original_name);
  const sizeLabel = formatBytes(file.size_bytes);

  return (
    <section className="text-viewer" aria-label={`Viewer for ${file.original_name}`}>
      <div className="text-viewer-bar">
        <FileCode2 className="text-viewer-bar-icon" aria-hidden="true" />
        <span className="text-viewer-name" title={file.original_name}>
          {file.original_name}
        </span>
        <span className="text-viewer-chip">{gzip ? `${chipLabel}·GZ` : chipLabel}</span>
        <span className="text-viewer-size">{sizeLabel}</span>
        <span className="text-viewer-bar-spacer" />
        {options.length > 0 ? (
          <ToggleGroup
            type="single"
            value={view}
            onValueChange={(next) => next && setView(next as TextView)}
            className="text-viewer-toggle"
            aria-label={`${chipLabel} view`}
          >
            {options.map((option) => {
              const isConvertTarget =
                (kind === "json" || kind === "yaml") && option.value !== kind;
              const disabled = isConvertTarget && convertError;
              return (
                <ToggleGroupItem
                  key={option.value}
                  value={option.value}
                  disabled={disabled}
                  title={disabled ? "Conversion needs a valid, complete document" : undefined}
                >
                  {option.label}
                </ToggleGroupItem>
              );
            })}
          </ToggleGroup>
        ) : null}
        {showWrapToggle ? (
          <button
            type="button"
            className="text-viewer-action"
            data-active={wrap ? "true" : undefined}
            aria-pressed={wrap}
            aria-label="Toggle line wrap"
            title="Toggle line wrap"
            onClick={() => setWrap((value) => !value)}
          >
            <WrapText aria-hidden="true" />
          </button>
        ) : null}
        <button
          type="button"
          className="text-viewer-action"
          aria-label={copied ? "Copied" : "Copy"}
          title={copied ? "Copied" : "Copy"}
          onClick={handleCopy}
        >
          <Copy aria-hidden="true" />
        </button>
        <a
          className="text-viewer-action"
          href={downloadUrl}
          aria-label="Download original"
          title="Download original"
        >
          <Download aria-hidden="true" />
        </a>
      </div>

      <TruncationBanner
        kind={kind}
        view={view}
        head={head}
        csv={csv}
        usesServerCsv={usesServerCsv}
        gzipPartial={Boolean(gzipCsv && head?.truncated)}
      />

      <div className="text-viewer-body">
        <ViewerBody
          kind={kind}
          view={view}
          wrap={wrap}
          display={display}
          head={head}
          headLoading={headLoading}
          headError={headError}
          downloadUrl={downloadUrl}
          isTableView={isTableView}
          converting={converting}
          usesServerCsv={usesServerCsv}
          csv={csv}
          csvRows={csvRows}
          csvLoading={csvLoading}
          csvError={csvError}
          csvLoadingMore={csvLoadingMore}
          onLoadMoreCsv={loadMoreCsv}
          gzipCsv={gzipCsv}
          fileId={file.file_id}
        />
      </div>

      <ViewerFooter
        kind={kind}
        view={view}
        head={head}
        csv={csv}
        csvRows={csvRows}
        usesServerCsv={usesServerCsv}
        gzipCsv={gzipCsv}
      />
    </section>
  );
}

function TruncationBanner({
  kind,
  view,
  head,
  csv,
  usesServerCsv,
  gzipPartial,
}: {
  kind: TextResourceKind;
  view: TextView;
  head: ResourceTextHead | null;
  csv: ResourceCsvRows | null;
  usesServerCsv: boolean;
  gzipPartial: boolean;
}) {
  if (kind === "csv" && view === "table") {
    if (usesServerCsv && csv?.has_more) {
      return (
        <div className="text-viewer-banner" role="note">
          <Info aria-hidden="true" />
          <span>
            Large table — about {csv.approx_total_rows.toLocaleString()} rows. Scroll to load more, or download for
            the full file.
          </span>
        </div>
      );
    }
    if (gzipPartial) {
      return (
        <div className="text-viewer-banner" role="note">
          <Info aria-hidden="true" />
          <span>Showing the first rows of this compressed file. Download for the full file.</span>
        </div>
      );
    }
    return null;
  }
  if (head?.truncated) {
    return (
      <div className="text-viewer-banner" role="note">
        <Info aria-hidden="true" />
        <span>
          Large file — showing the first {head.line_count.toLocaleString()} lines of about{" "}
          {head.approx_total_lines.toLocaleString()}. Download for the full file.
        </span>
      </div>
    );
  }
  return null;
}

type ViewerBodyProps = {
  kind: TextResourceKind;
  view: TextView;
  wrap: boolean;
  display: { text: string; format: TextResourceKind };
  head: ResourceTextHead | null;
  headLoading: boolean;
  headError: string | null;
  downloadUrl: string;
  isTableView: boolean;
  converting: boolean;
  usesServerCsv: boolean;
  csv: ResourceCsvRows | null;
  csvRows: string[][];
  csvLoading: boolean;
  csvError: string | null;
  csvLoadingMore: boolean;
  onLoadMoreCsv: () => void;
  gzipCsv: { columns: string[]; rows: string[][]; delimiter: string } | null;
  fileId: string;
};

function ViewerBody(props: ViewerBodyProps) {
  const { kind, view, wrap, display, head, headLoading, headError, downloadUrl } = props;

  if (props.isTableView) {
    if (props.usesServerCsv) {
      if (props.csvError) {
        return <ViewerError message={props.csvError} downloadUrl={downloadUrl} />;
      }
      if (!props.csv) {
        return <ViewerLoading />;
      }
      return (
        <CsvTable
          columns={props.csv.columns ?? []}
          rows={props.csvRows}
          hasMore={props.csv.has_more}
          loadingMore={props.csvLoadingMore}
          onLoadMore={props.onLoadMoreCsv}
        />
      );
    }
    if (headLoading) {
      return <ViewerLoading />;
    }
    if (headError) {
      return <ViewerError message={headError} downloadUrl={downloadUrl} />;
    }
    if (props.gzipCsv) {
      return (
        <CsvTable
          columns={props.gzipCsv.columns}
          rows={props.gzipCsv.rows}
          hasMore={false}
          loadingMore={false}
          onLoadMore={() => undefined}
        />
      );
    }
    return <ViewerLoading />;
  }

  if (kind === "markdown" && view === "rendered") {
    if (headLoading) {
      return <ViewerLoading />;
    }
    if (headError) {
      return <ViewerError message={headError} downloadUrl={downloadUrl} />;
    }
    return (
      <div className="text-viewer-markdown">
        <Suspense fallback={<ViewerLoading />}>
          <LazyMarkdown id={`md-${props.fileId}`}>{head?.text ?? ""}</LazyMarkdown>
        </Suspense>
      </div>
    );
  }

  if (headLoading || props.converting) {
    return <ViewerLoading />;
  }
  if (headError) {
    return <ViewerError message={headError} downloadUrl={downloadUrl} />;
  }
  return <CodeSurface text={display.text} format={display.format} wrap={wrap} />;
}

function ViewerLoading() {
  return (
    <div className="text-viewer-status" role="status">
      <Loader2 className="text-viewer-spin" aria-hidden="true" />
      <span>Loading…</span>
    </div>
  );
}

function ViewerError({ message, downloadUrl }: { message: string; downloadUrl: string }) {
  return (
    <div className="text-viewer-status text-viewer-status-error" role="alert">
      <span>{message}</span>
      <a className="text-viewer-download-link" href={downloadUrl}>
        Download original
      </a>
    </div>
  );
}

function CodeSurface({ text, format, wrap }: { text: string; format: TextResourceKind; wrap: boolean }) {
  const lines = useMemo(() => text.split("\n"), [text]);
  const { ref, height } = useElementHeight<HTMLDivElement>();
  const [scrollTop, setScrollTop] = useState(0);

  const onScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  if (wrap) {
    // A single wrapped <pre> is laid out synchronously, so cap it: enabling wrap on
    // a multi-MB head must never freeze the main thread. The unwrapped view shows all.
    const capped = text.length > WRAP_CHAR_CAP;
    const shown = capped ? text.slice(0, WRAP_CHAR_CAP) : text;
    return (
      <div className="text-viewer-surface text-viewer-surface-wrap" ref={ref}>
        <pre className="text-viewer-pre-wrap">{shown}</pre>
        {capped ? (
          <p className="text-viewer-wrap-note">
            Wrapped preview limited for performance — turn off wrap to scroll the full head, or download the file.
          </p>
        ) : null}
      </div>
    );
  }

  const total = lines.length;
  const viewportRows = Math.ceil(height / ROW_HEIGHT);
  const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const end = Math.min(total, start + viewportRows + OVERSCAN * 2);
  const visible = lines.slice(start, end);
  const gutterWidth = `${Math.max(2, String(total).length)}ch`;

  return (
    <div className="text-viewer-surface" ref={ref} onScroll={onScroll}>
      <div style={{ height: start * ROW_HEIGHT }} aria-hidden="true" />
      {visible.map((line, index) => {
        const lineNumber = start + index + 1;
        return (
          <div className="text-viewer-row" key={lineNumber} style={{ height: ROW_HEIGHT }}>
            <span className="text-viewer-lineno" style={{ width: gutterWidth }}>
              {lineNumber}
            </span>
            <span className="text-viewer-line">{highlightLine(line, format)}</span>
          </div>
        );
      })}
      <div style={{ height: Math.max(0, (total - end) * ROW_HEIGHT) }} aria-hidden="true" />
    </div>
  );
}

function CsvTable({
  columns,
  rows,
  hasMore,
  loadingMore,
  onLoadMore,
}: {
  columns: string[];
  rows: string[][];
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
}) {
  const { ref, height } = useElementHeight<HTMLDivElement>();
  const [scrollTop, setScrollTop] = useState(0);

  const onScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const node = event.currentTarget;
      setScrollTop(node.scrollTop);
      if (hasMore && !loadingMore && node.scrollHeight - node.scrollTop - node.clientHeight < CSV_ROW_HEIGHT * 16) {
        onLoadMore();
      }
    },
    [hasMore, loadingMore, onLoadMore]
  );

  const columnCount = Math.max(columns.length, rows[0]?.length ?? 0);
  const headerCells = columns.length > 0 ? columns : Array.from({ length: columnCount }, (_, i) => `col_${i + 1}`);
  const total = rows.length;
  const viewportRows = Math.ceil(height / CSV_ROW_HEIGHT);
  const start = Math.max(0, Math.floor(scrollTop / CSV_ROW_HEIGHT) - OVERSCAN);
  const end = Math.min(total, start + viewportRows + OVERSCAN * 2);
  const visible = rows.slice(start, end);
  const spanAll = columnCount + 1;

  return (
    <div className="text-viewer-table-scroll" ref={ref} onScroll={onScroll}>
      <table className="text-viewer-table">
        <thead>
          <tr>
            <th className="text-viewer-rownum" scope="col">
              #
            </th>
            {headerCells.map((cell, index) => (
              <th key={index} scope="col" title={cell}>
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {start > 0 ? (
            <tr aria-hidden="true" className="text-viewer-spacer-row" style={{ height: start * CSV_ROW_HEIGHT }}>
              <td colSpan={spanAll} />
            </tr>
          ) : null}
          {visible.map((row, index) => {
            const rowNumber = start + index + 1;
            return (
              <tr key={rowNumber} className="text-viewer-data-row">
                <td className="text-viewer-rownum">{rowNumber}</td>
                {Array.from({ length: columnCount }, (_, columnIndex) => {
                  const value = row[columnIndex] ?? "";
                  return (
                    <td key={columnIndex} className={looksNumeric(value) ? "text-viewer-cell-num" : undefined}>
                      {value}
                    </td>
                  );
                })}
              </tr>
            );
          })}
          {end < total ? (
            <tr aria-hidden="true" className="text-viewer-spacer-row" style={{ height: (total - end) * CSV_ROW_HEIGHT }}>
              <td colSpan={spanAll} />
            </tr>
          ) : null}
        </tbody>
      </table>
      {loadingMore ? (
        <div className="text-viewer-table-more" role="status">
          <Loader2 className="text-viewer-spin" aria-hidden="true" />
          <span>Loading more rows…</span>
        </div>
      ) : null}
    </div>
  );
}

function ViewerFooter({
  kind,
  view,
  head,
  csv,
  csvRows,
  usesServerCsv,
  gzipCsv,
}: {
  kind: TextResourceKind;
  view: TextView;
  head: ResourceTextHead | null;
  csv: ResourceCsvRows | null;
  csvRows: string[][];
  usesServerCsv: boolean;
  gzipCsv: { columns: string[]; rows: string[][]; delimiter: string } | null;
}) {
  if (kind === "csv" && view === "table") {
    const columns = (usesServerCsv ? csv?.columns : gzipCsv?.columns) ?? [];
    const shown = usesServerCsv ? csvRows.length : gzipCsv?.rows.length ?? 0;
    const delimiter = usesServerCsv ? csv?.delimiter ?? "," : gzipCsv?.delimiter ?? ",";
    const approx = usesServerCsv ? csv?.approx_total_rows ?? shown : shown;
    const parts = [
      `${columns.length || (gzipCsv?.columns.length ?? 0)} columns`,
      approx > shown ? `${shown.toLocaleString()} of ~${approx.toLocaleString()} rows` : `${shown.toLocaleString()} rows`,
      delimiterLabel(delimiter),
    ];
    return (
      <div className="text-viewer-foot">
        <Table2 aria-hidden="true" />
        <span>{parts.filter(Boolean).join(" · ")}</span>
      </div>
    );
  }
  if (!head) {
    return null;
  }
  const parts = [
    head.line_count > 0
      ? head.truncated
        ? `${head.line_count.toLocaleString()} of ~${head.approx_total_lines.toLocaleString()} lines`
        : `${head.line_count.toLocaleString()} lines`
      : "",
    head.encoding ? head.encoding.toUpperCase() : "",
    eolLabel(head.eol),
  ];
  return (
    <div className="text-viewer-foot">
      <FileCode2 aria-hidden="true" />
      <span>{parts.filter(Boolean).join(" · ")}</span>
    </div>
  );
}
