import { ChevronRight, Database, Download, FileCode2, FileText } from "lucide-react";
import {
  runReportPathKey,
  type RunDocumentKind,
} from "@/features/chat/run-artifact-hydration";
import { formatBytes } from "@/lib/format";

export type ChatRunDocument = {
  path: string;
  title: string;
  downloadUrl: string;
  kind: RunDocumentKind;
  mimeType?: string;
  sizeBytes?: number;
};

export type ChatRunDocumentsProps = {
  documents: ChatRunDocument[];
  /* The card is the report's identity in the transcript — one object, opened
     in the canvas rather than expanded inline. Version counts and the open
     path key are conversation-level facts, so they arrive from above. */
  openReportPathKey?: string | null;
  reportVersionCounts?: Record<string, number>;
  onOpenReport?: (document: ChatRunDocument) => void;
};

const KIND_META: Record<RunDocumentKind, { label: string; Icon: typeof FileText }> = {
  report: { label: "Report", Icon: FileText },
  code: { label: "Code", Icon: FileCode2 },
  data: { label: "Data", Icon: Database },
  document: { label: "Document", Icon: FileText },
};

const fileMetaLabel = (document: ChatRunDocument, open = false): string => {
  const { label } = KIND_META[document.kind] ?? KIND_META.document;
  const parts = [label];
  if (document.sizeBytes && document.sizeBytes > 0) {
    parts.push(formatBytes(document.sizeBytes));
  }
  if (open) {
    parts.push("open in canvas");
  }
  return parts.join(" · ");
};

const reportMetaLabel = (
  document: ChatRunDocument,
  version: number,
  open: boolean
): string => {
  const parts = ["Report"];
  if (version > 1) {
    parts.push(`v${version}`);
  }
  if (document.sizeBytes && document.sizeBytes > 0) {
    parts.push(formatBytes(document.sizeBytes));
  }
  if (open) {
    parts.push("open in canvas");
  }
  return parts.join(" · ");
};

function ReportDocumentCard({
  document,
  version,
  open,
  onOpen,
}: {
  document: ChatRunDocument;
  version: number;
  open: boolean;
  onOpen?: (document: ChatRunDocument) => void;
}) {
  // Without an open handler (canvas unavailable) the card degrades to the one
  // thing that always works: downloading the artifact.
  if (!onOpen) {
    return (
      <a href={document.downloadUrl} download className="chat-report-card" data-fallback="true">
        <span className="chat-report-card-tile">
          <FileText className="size-4" aria-hidden="true" />
        </span>
        <span className="chat-report-card-body">
          <span className="chat-report-card-title">{document.title}</span>
          <span className="chat-report-card-meta">{reportMetaLabel(document, version, false)}</span>
        </span>
        <Download className="size-4 shrink-0 chat-report-card-chevron" aria-hidden="true" />
      </a>
    );
  }
  return (
    <button
      type="button"
      className="chat-report-card"
      data-open={open ? "true" : undefined}
      aria-expanded={open}
      aria-controls="report-canvas"
      onClick={() => onOpen(document)}
    >
      <span className="chat-report-card-tile">
        <FileText className="size-4" aria-hidden="true" />
      </span>
      <span className="chat-report-card-body">
        <span className="chat-report-card-title">{document.title}</span>
        <span className="chat-report-card-meta">{reportMetaLabel(document, version, open)}</span>
      </span>
      <ChevronRight className="size-4 shrink-0 chat-report-card-chevron" aria-hidden="true" />
    </button>
  );
}

function FileDocumentChip({
  document,
  open,
  onOpen,
}: {
  document: ChatRunDocument;
  open: boolean;
  onOpen?: (document: ChatRunDocument) => void;
}) {
  const { Icon } = KIND_META[document.kind] ?? KIND_META.document;
  const identity = (
    <>
      <Icon className="size-4 shrink-0 chat-document-chip-icon" aria-hidden="true" />
      <span className="chat-document-chip-text">
        <span className="chat-document-chip-title">{document.title}</span>
        <span className="chat-document-chip-meta">{fileMetaLabel(document, open)}</span>
      </span>
    </>
  );

  if (!onOpen) {
    return (
      <a
        href={document.downloadUrl}
        download
        className="chat-document-chip chat-document-chip-fallback"
      >
        {identity}
        <Download className="size-4 shrink-0" aria-hidden="true" />
      </a>
    );
  }

  return (
    <span className="chat-document-chip" data-open={open ? "true" : undefined}>
      <button
        type="button"
        className="chat-document-chip-open"
        aria-label={`Preview ${document.title}`}
        aria-expanded={open}
        aria-controls="report-canvas"
        onClick={() => onOpen(document)}
      >
        {identity}
      </button>
      <a
        href={document.downloadUrl}
        download
        className="chat-document-chip-download"
        aria-label={`Download ${document.title}`}
        title={`Download ${document.title}`}
      >
        <Download className="size-3.5" aria-hidden="true" />
      </a>
    </span>
  );
}

export function ChatRunDocuments({
  documents,
  openReportPathKey = null,
  reportVersionCounts,
  onOpenReport,
}: ChatRunDocumentsProps) {
  if (!Array.isArray(documents) || documents.length === 0) {
    return null;
  }
  const reports = documents.filter((document) => document.kind === "report");
  const files = documents.filter((document) => document.kind !== "report");

  return (
    <div className="chat-document-list">
      {reports.map((document) => {
        const pathKey = runReportPathKey(document.path);
        return (
          <ReportDocumentCard
            key={document.path}
            document={document}
            version={Math.max(1, reportVersionCounts?.[pathKey] ?? 1)}
            open={Boolean(pathKey) && openReportPathKey === pathKey}
            onOpen={onOpenReport}
          />
        );
      })}
      {files.length > 0 ? (
        <div className="chat-document-files">
          {files.map((document) => {
            const pathKey = runReportPathKey(document.path);
            return (
              <FileDocumentChip
                key={document.path}
                document={document}
                open={Boolean(pathKey) && openReportPathKey === pathKey}
                onOpen={onOpenReport}
              />
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export default ChatRunDocuments;
