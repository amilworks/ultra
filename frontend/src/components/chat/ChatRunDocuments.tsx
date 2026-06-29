import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Database,
  Download,
  FileCode2,
  FileText,
  Loader2,
} from "lucide-react";
import { Markdown } from "@/components/prompt-kit/markdown";
import {
  rewriteArtifactMarkdownImageUrls,
  type RunDocumentKind,
} from "@/features/chat/run-artifact-hydration";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

export type ChatRunDocument = {
  path: string;
  title: string;
  downloadUrl: string;
  kind: RunDocumentKind;
  mimeType?: string;
  sizeBytes?: number;
};

// Minimal shape of the run's image artifacts, used only to resolve the report's
// own `![](outputs/figN.png)` references to their served URLs so figures render
// inside the report rather than as broken sandbox paths.
type ChatRunDocumentImage = {
  path: string;
  url?: string;
  downloadUrl?: string;
};

export type ChatRunDocumentsProps = {
  documents: ChatRunDocument[];
  imageArtifacts?: ChatRunDocumentImage[];
  loadDocumentText: (downloadUrl: string) => Promise<string>;
};

const KIND_META: Record<RunDocumentKind, { label: string; Icon: typeof FileText }> = {
  report: { label: "Report", Icon: FileText },
  code: { label: "Code", Icon: FileCode2 },
  data: { label: "Data", Icon: Database },
  document: { label: "Document", Icon: FileText },
};

const metaLabel = (document: ChatRunDocument): string => {
  const { label } = KIND_META[document.kind] ?? KIND_META.document;
  const size =
    document.sizeBytes && document.sizeBytes > 0 ? ` · ${formatBytes(document.sizeBytes)}` : "";
  return `${label}${size}`;
};

type ReportStatus = "loading" | "ready" | "error";

function ReportReader({
  document,
  imageArtifacts,
  loadDocumentText,
}: {
  document: ChatRunDocument;
  imageArtifacts: ChatRunDocumentImage[];
  loadDocumentText: ChatRunDocumentsProps["loadDocumentText"];
}) {
  // Reports answer the "let me read it in chat" ask, so they open by default;
  // the toggle lets a long report be collapsed without losing the download.
  const [expanded, setExpanded] = useState(true);
  const [status, setStatus] = useState<ReportStatus>("loading");
  const [rawContent, setRawContent] = useState("");
  const [error, setError] = useState("");

  // Fetch once on mount (reports are small and open by default). Status is kept
  // OUT of the dependency array on purpose: depending on it would re-run the
  // effect the instant we set "loading", whose cleanup would cancel the in-flight
  // request and strand the reader on the loading state.
  useEffect(() => {
    let cancelled = false;
    loadDocumentText(document.downloadUrl)
      .then((text) => {
        if (cancelled) {
          return;
        }
        setRawContent(text);
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) {
          return;
        }
        setError(cause instanceof Error ? cause.message : "Unable to load the report");
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [document.downloadUrl, loadDocumentText]);

  // Resolve the report's own `outputs/figN.png` references to served URLs at
  // render time so figures appear inline; decoupled from the fetch effect so a
  // changing image list never re-triggers a network read.
  const content = useMemo(
    () => rewriteArtifactMarkdownImageUrls(rawContent, imageArtifacts),
    [rawContent, imageArtifacts]
  );

  return (
    <section className="chat-document-report" aria-label={`Report: ${document.title}`}>
      <header className="chat-document-report-header">
        <button
          type="button"
          className="chat-document-report-toggle"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
        >
          {expanded ? (
            <ChevronDown className="size-4 shrink-0" aria-hidden="true" />
          ) : (
            <ChevronRight className="size-4 shrink-0" aria-hidden="true" />
          )}
          <FileText className="size-4 shrink-0" aria-hidden="true" />
          <span className="chat-document-report-title">{document.title}</span>
        </button>
        <span className="chat-document-report-meta">{metaLabel(document)}</span>
        <a
          href={document.downloadUrl}
          download
          className="chat-document-report-download"
          aria-label={`Download ${document.title}`}
        >
          <Download className="size-4" aria-hidden="true" />
          <span>Download</span>
        </a>
      </header>
      {expanded ? (
        <div className="chat-document-report-body">
          {status === "loading" ? (
            <div className="chat-document-report-status">
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              <span>Loading report…</span>
            </div>
          ) : status === "error" ? (
            <div className="chat-document-report-status chat-document-report-error">
              <AlertCircle className="size-4" aria-hidden="true" />
              <span>
                {error} —{" "}
                <a href={document.downloadUrl} download className="chat-document-report-inline-link">
                  download instead
                </a>
                .
              </span>
            </div>
          ) : (
            <Markdown className="chat-document-report-markdown">{content}</Markdown>
          )}
        </div>
      ) : null}
    </section>
  );
}

export function ChatRunDocuments({
  documents,
  imageArtifacts = [],
  loadDocumentText,
}: ChatRunDocumentsProps) {
  if (!Array.isArray(documents) || documents.length === 0) {
    return null;
  }
  const reports = documents.filter((document) => document.kind === "report");
  const files = documents.filter((document) => document.kind !== "report");

  return (
    <div className="chat-document-list">
      {reports.map((document) => (
        <ReportReader
          key={document.path}
          document={document}
          imageArtifacts={imageArtifacts}
          loadDocumentText={loadDocumentText}
        />
      ))}
      {files.length > 0 ? (
        <div className="chat-document-files">
          {files.map((document) => {
            const { Icon } = KIND_META[document.kind] ?? KIND_META.document;
            return (
              <a
                key={document.path}
                href={document.downloadUrl}
                download
                className="chat-document-chip"
              >
                <Icon className={cn("size-4 shrink-0 chat-document-chip-icon")} aria-hidden="true" />
                <span className="chat-document-chip-text">
                  <span className="chat-document-chip-title">{document.title}</span>
                  <span className="chat-document-chip-meta">{metaLabel(document)}</span>
                </span>
                <Download className="size-4 shrink-0 chat-document-chip-download" aria-hidden="true" />
              </a>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export default ChatRunDocuments;
