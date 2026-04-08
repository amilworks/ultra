import { useEffect, useMemo, useState } from "react";
import { Download, ExternalLink, FileImage } from "lucide-react";

import { Button } from "@/components/ui/button";
import { buildBisqueThumbnailUrl } from "@/lib/bisquePreview";
import { cn } from "@/lib/utils";

type ToolPreviewImage = {
  path: string;
  url: string;
  title: string;
  sourceName: string;
  previewable: boolean;
  downloadUrl?: string;
};

type ToolPreviewResourceRow = {
  name: string;
  created?: string;
  resourceType?: string;
  uri?: string;
  resourceUri?: string;
  clientViewUrl?: string;
  imageServiceUrl?: string;
};

type ToolResultQuickPreviewProps = {
  images: ToolPreviewImage[];
  resourceRows: ToolPreviewResourceRow[];
  onUseInChat?: (resourceUri: string) => void;
};

const normalizePreviewValue = (value: string | null | undefined): string =>
  String(value ?? "").trim().toLowerCase();

const resourceTypeLabel = (value: string | null | undefined): string => {
  const normalized = normalizePreviewValue(value);
  if (!normalized) {
    return "Preview ready";
  }
  if (
    normalized === "image" ||
    normalized === "image_service" ||
    normalized === "file"
  ) {
    return "Image Service";
  }
  if (normalized === "table") {
    return "Table Service";
  }
  if (normalized === "dataset") {
    return "Dataset Catalog";
  }
  return "Resource Catalog";
};

const formatCreatedLabel = (value: string | null | undefined): string | null => {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return null;
  }
  const parsed = new Date(raw.replace(" ", "T"));
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
};

const extractAccession = (row?: ToolPreviewResourceRow): string | null => {
  const raw =
    row?.resourceUri ||
    row?.clientViewUrl ||
    row?.uri ||
    "";
  const match = String(raw).match(/\/([^/?#]+)(?:[?#].*)?$/);
  const accession = match?.[1]?.trim();
  return accession ? accession : null;
};

const downloadLabelForTitle = (title: string): string => {
  const lowered = String(title).trim().toLowerCase();
  if (lowered.endsWith(".ome.tiff") || lowered.endsWith(".ome.tif")) {
    return "Download OME-TIFF";
  }
  if (lowered.endsWith(".tiff") || lowered.endsWith(".tif")) {
    return "Download TIFF";
  }
  if (lowered.endsWith(".png")) {
    return "Download PNG";
  }
  if (lowered.endsWith(".jpg") || lowered.endsWith(".jpeg")) {
    return "Download JPEG";
  }
  return "Download asset";
};

const resourceRowMatchesImage = (
  row: ToolPreviewResourceRow,
  image: ToolPreviewImage
): boolean => {
  const rowValues = new Set(
    [
      row.name,
      row.resourceUri,
      row.clientViewUrl,
      row.uri,
      row.imageServiceUrl,
      buildBisqueThumbnailUrl(row.imageServiceUrl),
    ]
      .map((value) => normalizePreviewValue(value))
      .filter(Boolean)
  );
  const imageValues = [
    image.title,
    image.sourceName,
    image.url,
    image.downloadUrl,
    image.path.replace(/#.*$/, ""),
  ]
    .map((value) => normalizePreviewValue(value))
    .filter(Boolean);
  return imageValues.some((value) => rowValues.has(value));
};

function ToolResultPreviewTile({
  image,
  row,
  onUseInChat,
}: {
  image: ToolPreviewImage;
  row?: ToolPreviewResourceRow;
  onUseInChat?: (resourceUri: string) => void;
}) {
  const previewCandidates = useMemo(
    () =>
      Array.from(
        new Set(
          [image.url, buildBisqueThumbnailUrl(row?.imageServiceUrl), row?.imageServiceUrl]
            .map((value) => String(value ?? "").trim())
            .filter(Boolean)
        )
      ),
    [image.url, row?.imageServiceUrl]
  );
  const [previewIndex, setPreviewIndex] = useState(
    previewCandidates.length > 0 ? 0 : -1
  );

  useEffect(() => {
    setPreviewIndex(previewCandidates.length > 0 ? 0 : -1);
  }, [previewCandidates]);

  const previewUrl =
    previewIndex >= 0 && previewIndex < previewCandidates.length
      ? previewCandidates[previewIndex]
      : null;
  const openUrl = row?.clientViewUrl || row?.uri;
  const downloadUrl = image.downloadUrl ?? row?.imageServiceUrl ?? openUrl ?? image.url;
  const title = row?.name || image.title || image.sourceName;
  const serviceLabel = resourceTypeLabel(row?.resourceType);
  const createdLabel = formatCreatedLabel(row?.created);
  const accession = extractAccession(row);
  const provenanceParts = [
    serviceLabel,
    row?.resourceUri ? "BisQue record" : null,
    createdLabel,
  ].filter((value): value is string => Boolean(value));

  return (
    <article className="overflow-hidden rounded-xl border border-border/70 bg-background/80 shadow-sm">
      <div className="aspect-[4/3] bg-muted/40">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={title}
            loading="lazy"
            className="h-full w-full object-cover"
            onError={() => {
              setPreviewIndex((current) => {
                if (current < 0) {
                  return current;
                }
                const next = current + 1;
                return next < previewCandidates.length ? next : -1;
              });
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <FileImage className="size-8" />
          </div>
        )}
      </div>
      <div className="space-y-3 border-t border-border/60 bg-background/90 p-4">
        <div className="space-y-1.5">
          <p className="line-clamp-2 text-sm font-medium leading-snug text-foreground">
            {title}
          </p>
          {provenanceParts.length > 0 ? (
            <p className="text-xs leading-5 text-muted-foreground">
              {provenanceParts.join(" · ")}
            </p>
          ) : null}
          {accession ? (
            <p className="text-[11px] font-medium tracking-[0.08em] text-muted-foreground/80 uppercase">
              Accession {accession}
            </p>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          {openUrl ? (
            <Button type="button" size="sm" variant="outline" asChild>
              <a href={openUrl} target="_blank" rel="noreferrer">
                <ExternalLink className="mr-2 size-3.5" />
                Open record
              </a>
            </Button>
          ) : null}
          {row?.resourceUri && onUseInChat ? (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => onUseInChat(row.resourceUri as string)}
            >
              Add to chat
            </Button>
          ) : null}
          {downloadUrl ? (
            <Button type="button" size="sm" variant="ghost" asChild>
              <a href={downloadUrl} download target="_blank" rel="noreferrer">
                <Download className="mr-2 size-3.5" />
                {downloadLabelForTitle(title)}
              </a>
            </Button>
          ) : null}
        </div>
      </div>
    </article>
  );
}

export function ToolResultQuickPreview({
  images,
  resourceRows,
  onUseInChat,
}: ToolResultQuickPreviewProps) {
  const items = useMemo(
    () =>
      images.map((image) => ({
        image,
        row:
          resourceRows.find((candidate) => resourceRowMatchesImage(candidate, image)) ??
          undefined,
      })),
    [images, resourceRows]
  );

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        "grid gap-4",
        items.length === 1 ? "grid-cols-1" : "grid-cols-1 xl:grid-cols-2"
      )}
    >
      {items.map(({ image, row }) => (
        <ToolResultPreviewTile
          key={image.path}
          image={image}
          row={row}
          onUseInChat={onUseInChat}
        />
      ))}
    </div>
  );
}
