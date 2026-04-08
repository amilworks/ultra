const DEFAULT_BISQUE_THUMBNAIL = "1280,1280,BC,,png";

export const buildBisqueThumbnailUrl = (
  imageServiceUrl: string | null | undefined,
  thumbnail: string = DEFAULT_BISQUE_THUMBNAIL
): string | null => {
  const trimmed = String(imageServiceUrl ?? "").trim();
  if (!trimmed) {
    return null;
  }

  try {
    const url = new URL(trimmed);
    if (!url.searchParams.has("thumbnail")) {
      url.searchParams.set("thumbnail", thumbnail);
    }
    return url.toString();
  } catch {
    if (/(^|[?&])thumbnail=/.test(trimmed)) {
      return trimmed;
    }
    const separator = trimmed.includes("?") ? "&" : "?";
    return `${trimmed}${separator}thumbnail=${encodeURIComponent(thumbnail)}`;
  }
};
