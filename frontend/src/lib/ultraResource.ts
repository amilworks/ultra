/* ultra://resource references — the portable way notes (and future surfaces)
 * point at cataloged files. The trailing segment carries the original
 * filename so renderers can tell video from image without a lookup; the
 * download URL is resolved at render time, never stored. */

export const ULTRA_RESOURCE_PATTERN = /^ultra:\/\/resource\/([^/?#]+)(?:\/([^?#]*))?$/;
export const VIDEO_EXTENSION_PATTERN = /\.(mp4|mov|webm|m4v|avi|mkv)$/i;
export const IMAGE_EXTENSION_PATTERN = /\.(png|jpe?g|gif|webp|avif|svg|bmp|tiff?)$/i;

export const ultraResourceRef = (fileId: string, name: string): string =>
  `ultra://resource/${fileId}/${encodeURIComponent(name)}`;

export type UltraResourceRef = { fileId: string; name: string };

/* Refs now reach this parser from model-authored chat markdown, not only from
 * app-minted notes. micromark leaves a '%' followed by two alphanumerics
 * untouched ('5%wt_Ni.tif'), and decodeURIComponent throws on it — which,
 * inside a markdown link renderer, would take down the whole message. The name
 * is presentational; navigation only needs the id, so fall back to the raw
 * segment instead of throwing. */
const decodeNameSegment = (segment: string): string => {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
};

export const parseUltraResourceRef = (value: string): UltraResourceRef | null => {
  const match = ULTRA_RESOURCE_PATTERN.exec(value);
  if (!match) {
    return null;
  }
  return { fileId: match[1], name: decodeNameSegment(match[2] ?? "") };
};

export const markdownForUpload = (record: {
  file_id: string;
  original_name: string;
  content_type?: string | null;
}): string => {
  const name = record.original_name;
  const ref = ultraResourceRef(record.file_id, name);
  const type = record.content_type ?? "";
  const isMedia =
    type.startsWith("image/") ||
    type.startsWith("video/") ||
    IMAGE_EXTENSION_PATTERN.test(name) ||
    VIDEO_EXTENSION_PATTERN.test(name);
  return isMedia ? `![${name}](${ref})` : `[${name}](${ref})`;
};
