// Paste-to-attach for the composer: extract attachable files from a paste's
// clipboard data.
//
// Files win over text — the convention users know from other chat composers:
// a pasted screenshot or a Finder-copied file attaches; plain text pastes
// normally because it has no files. The one real conflict (rich office
// content that carries both a text table and an image rendering) resolves in
// favor of the file; paste-without-formatting remains the text escape hatch.

const GENERIC_PASTED_IMAGE_PATTERN = /^image\.(png|jpe?g|gif|webp|tiff?|bmp)$/i;

const extensionForType = (type: string, fallback: string): string => {
  const subtype = type.split("/")[1] ?? "";
  if (!subtype) {
    return fallback;
  }
  if (subtype === "jpeg") {
    return "jpg";
  }
  if (subtype === "svg+xml") {
    return "svg";
  }
  return subtype;
};

const timestampSlug = (now: Date): string => {
  const pad = (value: number, width = 2) => String(value).padStart(width, "0");
  return (
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}` +
    `${pad(now.getMilliseconds(), 3)}`
  );
};

/**
 * Clipboard screenshots are all named "image.png" (or have no name at all),
 * which collides across pastes and confuses anything keyed on file names —
 * chips, resume-by-name matching. Give those a timestamped name; files copied
 * from the OS keep their real names.
 */
export const withPastedFileName = (file: File, now: Date): File => {
  const name = file.name ?? "";
  if (name && !GENERIC_PASTED_IMAGE_PATTERN.test(name)) {
    return file;
  }
  const fallbackExtension = name.includes(".") ? name.split(".").pop() ?? "png" : "png";
  const extension = extensionForType(file.type, fallbackExtension);
  return new File([file], `pasted-${timestampSlug(now)}.${extension}`, {
    type: file.type,
    lastModified: file.lastModified,
  });
};

/**
 * Files to attach from a paste, or an empty array when the paste is not
 * file-bearing (normal text paste — leave the event alone).
 */
export const filesFromClipboard = (
  clipboardData: Pick<DataTransfer, "files"> | null | undefined,
  now: Date = new Date()
): File[] => {
  const files = Array.from(clipboardData?.files ?? []);
  return files.map((file) => withPastedFileName(file, now));
};
