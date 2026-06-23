import type { ReactNode } from "react";

import type { TextResourceKind } from "@/lib/textFormat";

// A calm, low-saturation, per-line syntax pass. It runs only on the rows the
// windowed surface actually renders, so cost is independent of file size and the
// main thread never highlights a multi-megabyte string at once. Classes map to
// muted accents defined in styles.css (.tv-k/.tv-s/.tv-n/.tv-b/.tv-p/.tv-c).

type Segment = { text: string; cls?: string };

function jsonSegments(line: string): Segment[] {
  const segments: Segment[] = [];
  const pattern =
    /("(?:[^"\\]|\\.)*"\s*:)|("(?:[^"\\]|\\.)*")|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\btrue\b|\bfalse\b|\bnull\b)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(line)) !== null) {
    if (match.index > last) {
      segments.push({ text: line.slice(last, match.index), cls: "tv-p" });
    }
    if (match[1]) {
      segments.push({ text: match[1], cls: "tv-k" });
    } else if (match[2]) {
      segments.push({ text: match[2], cls: "tv-s" });
    } else if (match[3]) {
      segments.push({ text: match[3], cls: "tv-n" });
    } else if (match[4]) {
      segments.push({ text: match[4], cls: "tv-b" });
    }
    last = pattern.lastIndex;
  }
  if (last < line.length) {
    segments.push({ text: line.slice(last), cls: "tv-p" });
  }
  return segments;
}

function scalarSegments(value: string): Segment[] {
  const trimmed = value.trim();
  if (trimmed === "") {
    return [{ text: value }];
  }
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return [{ text: value, cls: "tv-n" }];
  }
  if (/^(true|false|null|yes|no)$/i.test(trimmed)) {
    return [{ text: value, cls: "tv-b" }];
  }
  if (/^".*"$/.test(trimmed) || /^'.*'$/.test(trimmed)) {
    return [{ text: value, cls: "tv-s" }];
  }
  return [{ text: value }];
}

function yamlSegments(line: string): Segment[] {
  if (line.trim().startsWith("#")) {
    return [{ text: line, cls: "tv-c" }];
  }
  const keyMatch = /^(\s*(?:- )?)([^:\s][^:]*?)(:)(\s.*|)$/.exec(line);
  if (keyMatch) {
    const [, indent, key, colon, rest] = keyMatch;
    return [
      { text: indent, cls: "tv-p" },
      { text: key, cls: "tv-k" },
      { text: colon, cls: "tv-p" },
      ...scalarSegments(rest),
    ];
  }
  const listMatch = /^(\s*- )(.*)$/.exec(line);
  if (listMatch) {
    return [{ text: listMatch[1], cls: "tv-p" }, ...scalarSegments(listMatch[2])];
  }
  return [{ text: line }];
}

function xmlSegments(line: string): Segment[] {
  const segments: Segment[] = [];
  const pattern = /<[^>]+>/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(line)) !== null) {
    if (match.index > last) {
      segments.push({ text: line.slice(last, match.index) });
    }
    segments.push({ text: match[0], cls: "tv-k" });
    last = pattern.lastIndex;
  }
  if (last < line.length) {
    segments.push({ text: line.slice(last) });
  }
  return segments;
}

function segmentsFor(line: string, format: TextResourceKind): Segment[] {
  switch (format) {
    case "json":
      return jsonSegments(line);
    case "yaml":
      return yamlSegments(line);
    case "xml":
      return xmlSegments(line);
    default:
      return [{ text: line }];
  }
}

const HIGHLIGHTED: ReadonlySet<TextResourceKind> = new Set(["json", "yaml", "xml"]);

export function isHighlightable(format: TextResourceKind): boolean {
  return HIGHLIGHTED.has(format);
}

export function highlightLine(line: string, format: TextResourceKind): ReactNode {
  if (!isHighlightable(format) || line.length === 0) {
    return line;
  }
  const segments = segmentsFor(line, format);
  return segments.map((segment, index) =>
    segment.cls ? (
      <span key={index} className={segment.cls}>
        {segment.text}
      </span>
    ) : (
      <span key={index}>{segment.text}</span>
    )
  );
}
