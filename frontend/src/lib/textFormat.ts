import type { ResourceTextFormat } from "../types";

// Detection + display helpers for the text/data resource viewer. Format is
// inferred from the filename extension first (the content type is unreliable —
// chunked/bundle uploads persist "application/octet-stream"), then the
// resource_kind hint, then the content type, then a content sniff.

export type TextResourceKind = ResourceTextFormat;

type ClassifiableResource = {
  content_type?: string | null;
  original_name?: string | null;
  resource_kind?: string | null;
};

const EXTENSION_FORMATS: Record<string, TextResourceKind> = {
  csv: "csv",
  tsv: "csv",
  json: "json",
  jsonl: "json",
  ndjson: "json",
  geojson: "json",
  yaml: "yaml",
  yml: "yaml",
  xml: "xml",
  xsd: "xml",
  xslt: "xml",
  md: "markdown",
  markdown: "markdown",
  mdx: "markdown",
  txt: "text",
  text: "text",
  log: "text",
  ini: "text",
  toml: "text",
  cfg: "text",
  conf: "text",
  properties: "text",
  env: "text",
};

// extensionOf returns the lowercased final extension, transparently looking past
// a trailing ".gz" so "data.csv.gz" resolves to "csv".
export function extensionOf(name: string | null | undefined): string {
  const trimmed = String(name ?? "")
    .trim()
    .toLowerCase();
  const base = trimmed.endsWith(".gz") ? trimmed.slice(0, -3) : trimmed;
  const dot = base.lastIndexOf(".");
  return dot >= 0 ? base.slice(dot + 1) : "";
}

export function isGzipName(name: string | null | undefined): boolean {
  return String(name ?? "")
    .trim()
    .toLowerCase()
    .endsWith(".gz");
}

// classifyTextResource returns the viewer format for a resource, or null if it is
// not a text/data file the viewer should open. Never matches image/video.
export function classifyTextResource(file: ClassifiableResource | null | undefined): TextResourceKind | null {
  if (!file) {
    return null;
  }
  const ext = extensionOf(file.original_name);
  if (ext && EXTENSION_FORMATS[ext]) {
    return EXTENSION_FORMATS[ext];
  }

  const kind = String(file.resource_kind ?? "")
    .trim()
    .toLowerCase();
  if (kind === "table") {
    return "csv";
  }

  const contentType = String(file.content_type ?? "")
    .split(";")[0]
    .trim()
    .toLowerCase();
  if (contentType) {
    if (contentType.includes("csv") || contentType.includes("tab-separated")) {
      return "csv";
    }
    if (contentType === "application/json" || contentType.endsWith("+json")) {
      return "json";
    }
    if (contentType === "application/xml" || contentType === "text/xml" || contentType.endsWith("+xml")) {
      return "xml";
    }
    if (contentType.includes("yaml")) {
      return "yaml";
    }
    if (contentType === "text/markdown") {
      return "markdown";
    }
    if (contentType.startsWith("text/")) {
      return "text";
    }
  }

  if (kind === "document") {
    return "text";
  }
  return null;
}

const FORMAT_CHIP_LABELS: Record<TextResourceKind, string> = {
  csv: "CSV",
  json: "JSON",
  yaml: "YAML",
  xml: "XML",
  markdown: "MD",
  text: "TXT",
};

export function formatChipLabel(kind: TextResourceKind, name?: string | null): string {
  if (kind === "csv" && extensionOf(name) === "tsv") {
    return "TSV";
  }
  return FORMAT_CHIP_LABELS[kind] ?? "TXT";
}

// formatBytes renders a calm, rounded byte size (base 1000, like the rest of the app).
export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  if (unit === 0) {
    return `${Math.round(value)} ${units[unit]}`;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function delimiterLabel(delimiter: string): string {
  switch (delimiter) {
    case ",":
      return "comma";
    case "\t":
      return "tab";
    case ";":
      return "semicolon";
    case "|":
      return "pipe";
    case " ":
      return "space";
    default:
      return delimiter || "comma";
  }
}

export function eolLabel(eol: string): string {
  switch (eol) {
    case "lf":
      return "LF";
    case "crlf":
      return "CRLF";
    default:
      return "";
  }
}
