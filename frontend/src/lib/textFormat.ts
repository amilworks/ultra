import type { ResourceTextFormat } from "../types";

// Detection + display helpers for the text/data resource viewer. Format is
// inferred from the filename extension first (the content type is unreliable —
// chunked/bundle uploads persist "application/octet-stream"), then explicit
// table and textual MIME hints. Generic document kind is not evidence of text.

export type TextResourceKind = ResourceTextFormat;

type ClassifiableResource = {
  content_type?: string | null;
  original_name?: string | null;
  resource_kind?: string | null;
};

const SOURCE_CODE_EXTENSIONS = new Set([
  "bash",
  "c",
  "cc",
  "clj",
  "cpp",
  "cs",
  "css",
  "cxx",
  "dart",
  "erl",
  "ex",
  "exs",
  "fish",
  "fs",
  "fsx",
  "go",
  "h",
  "hpp",
  "hrl",
  "htm",
  "html",
  "ipynb",
  "java",
  "js",
  "jsx",
  "kt",
  "kts",
  "less",
  "lua",
  "m",
  "php",
  "pl",
  "py",
  "pyw",
  "r",
  "rb",
  "rs",
  "sass",
  "scala",
  "scss",
  "sh",
  "sql",
  "swift",
  "tex",
  "ts",
  "tsx",
  "zsh",
]);

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

export function isSourceCodeName(name: string | null | undefined): boolean {
  return SOURCE_CODE_EXTENSIONS.has(extensionOf(name));
}

const FILE_TYPE_LABELS: Record<string, string> = {
  bash: "Shell",
  bib: "BibTeX",
  c: "C",
  cc: "C++",
  cfg: "Config",
  clj: "Clojure",
  conf: "Config",
  cpp: "C++",
  cs: "C#",
  css: "CSS",
  csv: "CSV",
  cxx: "C++",
  dart: "Dart",
  doc: "Word",
  docx: "Word",
  env: "Environment",
  erl: "Erlang",
  ex: "Elixir",
  exs: "Elixir",
  fish: "Shell",
  fs: "F#",
  fsx: "F#",
  geojson: "GeoJSON",
  go: "Go",
  h: "C header",
  h5: "HDF5",
  hdf5: "HDF5",
  hpp: "C++ header",
  hrl: "Erlang header",
  htm: "HTML",
  html: "HTML",
  ini: "Config",
  ipynb: "Jupyter notebook",
  java: "Java",
  js: "JavaScript",
  json: "JSON",
  jsonl: "JSON Lines",
  jsx: "JavaScript",
  kt: "Kotlin",
  kts: "Kotlin",
  less: "Less",
  log: "Log",
  lua: "Lua",
  m: "Source code",
  markdown: "Markdown",
  md: "Markdown",
  mdx: "MDX",
  ndjson: "JSON Lines",
  nii: "NIfTI",
  pdf: "PDF",
  php: "PHP",
  pl: "Perl",
  ppt: "PowerPoint",
  pptx: "PowerPoint",
  properties: "Properties",
  py: "Python",
  pyw: "Python",
  r: "R",
  rb: "Ruby",
  rs: "Rust",
  sass: "Sass",
  scala: "Scala",
  scss: "Sass",
  sh: "Shell",
  sql: "SQL",
  swift: "Swift",
  tex: "TeX",
  text: "Text",
  toml: "TOML",
  ts: "TypeScript",
  tsv: "TSV",
  tsx: "TypeScript",
  txt: "Text",
  xls: "Excel",
  xlsx: "Excel",
  xml: "XML",
  xsd: "XML Schema",
  xslt: "XSLT",
  yaml: "YAML",
  yml: "YAML",
  zsh: "Shell",
};

// A filename is stronger identity evidence than the broad backend kind
// ("file"/"document"). Known formats get a reader-facing name; short unknown
// extensions remain honest and recognizable instead of collapsing to "File".
export function fileTypeLabel(name: string | null | undefined): string | null {
  const extension = extensionOf(name);
  if (!extension) {
    return null;
  }
  if (FILE_TYPE_LABELS[extension]) {
    return FILE_TYPE_LABELS[extension];
  }
  return /^[a-z0-9]{1,8}$/.test(extension) ? extension.toUpperCase() : null;
}

// classifyTextResource returns the viewer format for a resource, or null if it is
// not a text/data file the viewer should open. Never matches image/video.
export function classifyTextResource(file: ClassifiableResource | null | undefined): TextResourceKind | null {
  if (!file) {
    return null;
  }
  const kind = String(file.resource_kind ?? "")
    .trim()
    .toLowerCase();
  const contentType = String(file.content_type ?? "")
    .split(";")[0]
    .trim()
    .toLowerCase();
  if (
    kind === "image" ||
    kind === "video" ||
    contentType.startsWith("image/") ||
    contentType.startsWith("video/")
  ) {
    return null;
  }

  const ext = extensionOf(file.original_name);
  if (SOURCE_CODE_EXTENSIONS.has(ext)) {
    return ext === "ipynb" ? "json" : "text";
  }
  if (ext && EXTENSION_FORMATS[ext]) {
    return EXTENSION_FORMATS[ext];
  }

  if (kind === "table") {
    return "csv";
  }

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

const EXTENSION_CHIP_LABELS: Record<string, string> = {
  bash: "SH",
  cpp: "C++",
  cxx: "C++",
  fish: "SH",
  ipynb: "IPYNB",
  javascript: "JS",
  markdown: "MD",
  ndjson: "JSONL",
  properties: "PROP",
  pyw: "PY",
  shell: "SH",
  typescript: "TS",
  zsh: "SH",
};

export function formatChipLabel(kind: TextResourceKind, name?: string | null): string {
  const extension = extensionOf(name);
  if (kind === "csv" && extension === "tsv") {
    return "TSV";
  }
  if (extension && (SOURCE_CODE_EXTENSIONS.has(extension) || kind === "text")) {
    return EXTENSION_CHIP_LABELS[extension] ?? extension.toUpperCase().slice(0, 6);
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
