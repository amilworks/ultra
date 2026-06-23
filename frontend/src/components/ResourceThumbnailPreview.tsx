import { useEffect, useMemo, useRef, useState, type ComponentType } from "react";

import { getCachedResourcePreview, getResourcePreview } from "@/lib/resourcePreviewCache";
import { parseCsv, sniffDelimiter } from "@/lib/csv";
import { extensionOf, formatChipLabel, type TextResourceKind } from "@/lib/textFormat";
import { highlightLine } from "@/components/viewer/text-highlight";
import type { ResourceRecord, ResourceTextHead } from "@/types";

const SNIPPET_LINES = 8;
const GRID_ROWS = 4;
const GRID_COLS = 4;

type Props = {
  resource: ResourceRecord;
  kind: TextResourceKind;
  fetchHead: (fileId: string, maxBytes: number) => Promise<ResourceTextHead>;
  fallbackIcon: ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  fallbackLabel: string;
};

// A calm, content-aware card thumbnail for text/data resources. It fetches a tiny
// head only once the card scrolls into view (IntersectionObserver), funneled
// through the shared concurrency-bounded cache, then renders a format-tailored
// preview. Until then (and on failure) it shows the muted kind icon, so the grid
// never blocks on the network and degrades gracefully.
export function ResourceThumbnailPreview({ resource, kind, fetchHead, fallbackIcon: FallbackIcon, fallbackLabel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);
  const [head, setHead] = useState<ResourceTextHead | null>(() => getCachedResourcePreview(resource.file_id) ?? null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (head || failed) {
      return;
    }
    const node = containerRef.current;
    if (!node) {
      return;
    }
    let cancelled = false;
    const begin = () => {
      if (startedRef.current) {
        return;
      }
      startedRef.current = true;
      getResourcePreview(resource.file_id, fetchHead)
        .then((result) => {
          if (!cancelled) {
            setHead(result);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setFailed(true);
          }
        });
    };
    if (typeof IntersectionObserver === "undefined") {
      begin();
      return () => {
        cancelled = true;
      };
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          begin();
          observer.disconnect();
        }
      },
      { rootMargin: "300px" }
    );
    observer.observe(node);
    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, [resource.file_id, fetchHead, head, failed]);

  const chip = formatChipLabel(kind, resource.original_name);

  return (
    <div className="resource-thumb" ref={containerRef}>
      {head && !failed ? <span className="resource-thumb-chip">{chip}</span> : null}
      {head && !failed ? (
        <PreviewBody kind={kind} head={head} name={resource.original_name} />
      ) : (
        <div className="resource-thumb-fallback">
          <FallbackIcon className="size-6" aria-hidden={true} />
          <span>{fallbackLabel}</span>
        </div>
      )}
    </div>
  );
}

function PreviewBody({ kind, head, name }: { kind: TextResourceKind; head: ResourceTextHead; name: string }) {
  if (kind === "csv") {
    return <CsvGridPreview text={head.text} name={name} />;
  }
  if (kind === "markdown") {
    return <MarkdownMiniPreview text={head.text} />;
  }
  return <CodeSnippetPreview text={head.text} kind={kind} />;
}

function CodeSnippetPreview({ text, kind }: { text: string; kind: TextResourceKind }) {
  const lines = useMemo(() => text.split("\n").slice(0, SNIPPET_LINES), [text]);
  return (
    <pre className="resource-thumb-snippet" aria-hidden="true">
      {lines.map((line, index) => (
        <div key={index}>{highlightLine(line, kind) || " "}</div>
      ))}
    </pre>
  );
}

function CsvGridPreview({ text, name }: { text: string; name: string }) {
  const grid = useMemo(() => {
    const delimiter = extensionOf(name) === "tsv" ? "\t" : sniffDelimiter(text);
    const { rows } = parseCsv(text, delimiter, true);
    const header = rows.length > 0 ? rows[0].slice(0, GRID_COLS) : [];
    const body = rows.slice(1, GRID_ROWS + 1).map((row) => row.slice(0, GRID_COLS));
    return { header, body };
  }, [text, name]);

  if (grid.header.length === 0) {
    return <pre className="resource-thumb-snippet" aria-hidden="true">{text.slice(0, 240)}</pre>;
  }
  return (
    <table className="resource-thumb-grid" aria-hidden="true">
      <tbody>
        <tr className="resource-thumb-grid-head">
          {grid.header.map((cell, index) => (
            <td key={index}>{cell}</td>
          ))}
        </tr>
        {grid.body.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {grid.header.map((_, colIndex) => (
              <td key={colIndex}>{row[colIndex] ?? ""}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// A lightweight Markdown "rendered mini" — first heading + first lines, extracted
// with cheap regex (NOT the full markdown renderer, which would be far too heavy
// across a 50-card grid).
function MarkdownMiniPreview({ text }: { text: string }) {
  const mini = useMemo(() => {
    let title = "";
    const body: string[] = [];
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line) {
        continue;
      }
      const heading = /^#{1,6}\s+(.*)$/.exec(line);
      if (heading) {
        if (!title) {
          title = heading[1].trim();
        }
        continue;
      }
      if (line.startsWith("```") || line.startsWith("---")) {
        continue;
      }
      body.push(line.replace(/[*_`>]/g, "").replace(/^[-+]\s+/, "• ").trim());
      if (body.length >= 3) {
        break;
      }
    }
    return { title, body };
  }, [text]);

  return (
    <div className="resource-thumb-md" aria-hidden="true">
      {mini.title ? <div className="resource-thumb-md-title">{mini.title}</div> : null}
      {mini.body.map((line, index) => (
        <div key={index} className="resource-thumb-md-line">
          {line}
        </div>
      ))}
    </div>
  );
}
