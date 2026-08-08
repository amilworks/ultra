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

export const parseUltraResourceRef = (value: string): UltraResourceRef | null => {
  const match = ULTRA_RESOURCE_PATTERN.exec(value);
  if (!match) {
    return null;
  }
  return { fileId: match[1], name: decodeURIComponent(match[2] ?? "") };
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
