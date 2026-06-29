import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { ChevronLeft, ChevronRight, Download, ImageIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  ContextMenu,
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { cn } from "@/lib/utils";
import { openFigureLightbox } from "@/lib/figureLightbox";

type PrairieImageAnalysis = {
  prairieDogCount?: number | null;
  burrowCount?: number | null;
  nearestBurrowDistancePxMean?: number | null;
  capturedAt?: string | null;
  latitude?: number | null;
  longitude?: number | null;
};

type ToolDetectionBox = {
  className: string;
  confidence?: number | null;
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
};

type ToolImageHoverDetails = {
  fileLabel?: string;
  masksGenerated?: number | null;
  avgPointsPerWindow?: number | null;
  minPoints?: number | null;
  maxPoints?: number | null;
  detectionBoxes?: ToolDetectionBox[];
  prairieImageAnalysis?: PrairieImageAnalysis;
};

export type ToolImageCarouselImage = {
  path: string;
  url: string;
  title: string;
  previewable: boolean;
  downloadUrl?: string;
  hoverDetails?: ToolImageHoverDetails;
};

export type ToolImageCarouselProps = {
  images: ToolImageCarouselImage[];
  variant?: "default" | "prairie";
};

const yoloOverlayPalette = [
  "#8b9a6d",
  "#bb8263",
  "#d2aa67",
  "#6f8f9f",
  "#b08ba0",
  "#7f745f",
  "#769c90",
  "#cc9278",
];

const clampToRange = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

const hashText = (value: string): number => {
  let hash = 0;
  for (let idx = 0; idx < value.length; idx += 1) {
    hash = (hash * 31 + value.charCodeAt(idx)) >>> 0;
  }
  return hash;
};

const yoloClassColor = (className: string): string => {
  const normalized = String(className || "class").trim().toLowerCase();
  if (normalized === "prairie_dog") {
    return "#bb8263";
  }
  if (normalized === "burrow") {
    return "#8b9a6d";
  }
  const bucket = hashText(normalized || "class") % yoloOverlayPalette.length;
  return yoloOverlayPalette[bucket];
};

const detectionBoxStyle = (
  detection: ToolDetectionBox,
  imageSize: { width: number; height: number },
  classColor: string
): CSSProperties | null => {
  const width = Number(imageSize.width);
  const height = Number(imageSize.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  const xMin = clampToRange(detection.xMin, 0, width);
  const yMin = clampToRange(detection.yMin, 0, height);
  const xMax = clampToRange(detection.xMax, 0, width);
  const yMax = clampToRange(detection.yMax, 0, height);
  const left = (Math.min(xMin, xMax) / width) * 100;
  const top = (Math.min(yMin, yMax) / height) * 100;
  const boxWidth = ((Math.max(xMin, xMax) - Math.min(xMin, xMax)) / width) * 100;
  const boxHeight = ((Math.max(yMin, yMax) - Math.min(yMin, yMax)) / height) * 100;
  if (boxWidth <= 0 || boxHeight <= 0) {
    return null;
  }
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${boxWidth}%`,
    height: `${boxHeight}%`,
    borderColor: classColor,
    boxShadow: `inset 0 0 0 1px ${classColor}40`,
  };
};

export function ToolImageCarousel({
  images,
  variant = "default",
}: ToolImageCarouselProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [showDetectionOverlay, setShowDetectionOverlay] = useState(variant === "prairie");
  const [showImageInfo, setShowImageInfo] = useState(variant === "prairie");
  const [naturalSizeByPath, setNaturalSizeByPath] = useState<
    Record<string, { width: number; height: number }>
  >({});
  const stageRef = useRef<HTMLDivElement | null>(null);
  const longPressTimerRef = useRef<number | null>(null);
  const prefersPersistentControls = variant === "prairie";

  const clearLongPress = useCallback(() => {
    if (longPressTimerRef.current !== null) {
      window.clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  const triggerContextMenuAtPoint = useCallback((clientX: number, clientY: number) => {
    if (!stageRef.current) {
      return;
    }
    stageRef.current.dispatchEvent(
      new MouseEvent("contextmenu", {
        bubbles: true,
        cancelable: true,
        clientX,
        clientY,
        view: window,
      })
    );
  }, []);

  useEffect(() => clearLongPress, [clearLongPress]);

  if (images.length === 0) {
    return null;
  }

  const boundedIndex = Math.min(Math.max(activeIndex, 0), images.length - 1);
  const activeImage = images[boundedIndex];
  const activeDetections = activeImage.hoverDetails?.detectionBoxes ?? [];
  const activePrairieAnalysis = activeImage.hoverDetails?.prairieImageAnalysis;
  const hasDetectionData = activeImage.previewable && activeDetections.length > 0;
  const activeNaturalSize = naturalSizeByPath[activeImage.path];
  const hasDetectionOverlay = Boolean(hasDetectionData && activeNaturalSize);
  const showsPointHint = images.some(
    (image) => image.hoverDetails?.maxPoints !== null && image.hoverDetails?.maxPoints !== undefined
  );
  const hasMultipleImages = images.length > 1;
  const showDetectionLayer = Boolean(
    hasDetectionOverlay && prefersPersistentControls && showDetectionOverlay
  );
  const showInfoPanel = Boolean(
    activeImage.hoverDetails && (!prefersPersistentControls || showImageInfo)
  );

  const media = activeImage.previewable ? (
    <div
      className={cn(
        "chat-tool-carousel-media",
        hasDetectionData && "chat-tool-carousel-media--detection"
      )}
    >
      <img
        src={activeImage.url}
        alt={activeImage.title}
        loading="lazy"
        role="button"
        tabIndex={0}
        title="View full size"
        className={cn(
          "chat-tool-carousel-image chat-tool-carousel-image--zoomable",
          hasDetectionData && "chat-tool-carousel-image--detection"
        )}
        onClick={() =>
          openFigureLightbox(
            images
              .filter((image) => image.previewable !== false)
              .map((image) => ({ url: image.url, downloadUrl: image.downloadUrl, title: image.title })),
            boundedIndex
          )
        }
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openFigureLightbox(
              images
                .filter((image) => image.previewable !== false)
                .map((image) => ({ url: image.url, downloadUrl: image.downloadUrl, title: image.title })),
              boundedIndex
            );
          }
        }}
        onLoad={(event) => {
          const target = event.currentTarget;
          const width = target.naturalWidth;
          const height = target.naturalHeight;
          if (!width || !height) {
            return;
          }
          setNaturalSizeByPath((prev) => {
            const existing = prev[activeImage.path];
            if (existing && existing.width === width && existing.height === height) {
              return prev;
            }
            return {
              ...prev,
              [activeImage.path]: { width, height },
            };
          });
        }}
      />
      {hasDetectionOverlay ? (
        <div
          className={cn(
            "chat-tool-carousel-detection-layer",
            showDetectionLayer && "is-visible"
          )}
          data-testid={prefersPersistentControls ? "prairie-detection-overlay" : undefined}
        >
          {activeDetections.map((detection, detectionIndex) => {
            const classColor = yoloClassColor(detection.className);
            const style = detectionBoxStyle(detection, activeNaturalSize, classColor);
            if (!style) {
              return null;
            }
            return (
              <div
                key={`${activeImage.path}-${detection.className}-${detectionIndex}`}
                className="chat-tool-detection-box"
                style={style}
              >
                <span
                  className="chat-tool-detection-label"
                  style={{
                    backgroundColor: `${classColor}CC`,
                    borderColor: classColor,
                  }}
                >
                  {detection.className}
                  {detection.confidence !== null && detection.confidence !== undefined
                    ? ` ${(detection.confidence * 100).toFixed(0)}%`
                    : ""}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  ) : (
    <div className="chat-tool-carousel-image chat-tool-image-placeholder">
      <ImageIcon className="size-5" />
      <span>Preview unavailable</span>
    </div>
  );

  const stage = (
    <div
      ref={stageRef}
      className={cn(
        "chat-tool-carousel-stage",
        prefersPersistentControls && "chat-tool-carousel-stage--prairie"
      )}
      data-variant={variant}
      onTouchStart={(event) => {
        if (!prefersPersistentControls) {
          return;
        }
        const touch = event.touches[0];
        if (!touch) {
          return;
        }
        clearLongPress();
        const { clientX, clientY } = touch;
        longPressTimerRef.current = window.setTimeout(() => {
          triggerContextMenuAtPoint(clientX, clientY);
        }, 550);
      }}
      onTouchMove={() => {
        clearLongPress();
      }}
      onTouchEnd={() => {
        clearLongPress();
      }}
      onTouchCancel={() => {
        clearLongPress();
      }}
      data-testid={prefersPersistentControls ? "prairie-detection-stage" : undefined}
    >
      {media}
      {showInfoPanel ? (
        <div
          className={cn(
            "chat-tool-carousel-hover",
            prefersPersistentControls && "is-persistent"
          )}
        >
          {activeImage.hoverDetails?.fileLabel ? <p>{activeImage.hoverDetails.fileLabel}</p> : null}
          {activeImage.hoverDetails?.detectionBoxes &&
          activeImage.hoverDetails.detectionBoxes.length > 0 ? (
            <p>Detections: {activeImage.hoverDetails.detectionBoxes.length}</p>
          ) : null}
          {activePrairieAnalysis?.prairieDogCount !== null &&
          activePrairieAnalysis?.prairieDogCount !== undefined ? (
            <p>Prairie dogs: {Math.round(activePrairieAnalysis.prairieDogCount)}</p>
          ) : null}
          {activePrairieAnalysis?.burrowCount !== null &&
          activePrairieAnalysis?.burrowCount !== undefined ? (
            <p>Burrows: {Math.round(activePrairieAnalysis.burrowCount)}</p>
          ) : null}
          {activePrairieAnalysis?.nearestBurrowDistancePxMean !== null &&
          activePrairieAnalysis?.nearestBurrowDistancePxMean !== undefined ? (
            <p>
              Mean nearest burrow: {activePrairieAnalysis.nearestBurrowDistancePxMean.toFixed(1)} px
            </p>
          ) : null}
          {activePrairieAnalysis?.capturedAt ? <p>Captured: {activePrairieAnalysis.capturedAt}</p> : null}
          {activePrairieAnalysis?.latitude !== null &&
          activePrairieAnalysis?.latitude !== undefined &&
          activePrairieAnalysis?.longitude !== null &&
          activePrairieAnalysis?.longitude !== undefined ? (
            <p>
              GPS: {activePrairieAnalysis.latitude.toFixed(5)}, {activePrairieAnalysis.longitude.toFixed(5)}
            </p>
          ) : null}
          {activeImage.hoverDetails?.masksGenerated !== null &&
          activeImage.hoverDetails?.masksGenerated !== undefined ? (
            <p>Masks: {Math.round(activeImage.hoverDetails.masksGenerated)}</p>
          ) : null}
          {activeImage.hoverDetails?.avgPointsPerWindow !== null &&
          activeImage.hoverDetails?.avgPointsPerWindow !== undefined ? (
            <p>Avg points/window: {activeImage.hoverDetails.avgPointsPerWindow.toFixed(1)}</p>
          ) : null}
          {activeImage.hoverDetails?.minPoints !== null &&
          activeImage.hoverDetails?.maxPoints !== null &&
          activeImage.hoverDetails?.minPoints !== undefined &&
          activeImage.hoverDetails?.maxPoints !== undefined ? (
            <p>
              Point range: {Math.round(activeImage.hoverDetails.minPoints)}-
              {Math.round(activeImage.hoverDetails.maxPoints)}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );

  return (
    <div className="chat-tool-carousel">
      {prefersPersistentControls ? (
        <ContextMenu>
          <ContextMenuTrigger asChild>{stage}</ContextMenuTrigger>
          <ContextMenuContent className="w-56">
            <ContextMenuLabel>Prairie overlay controls</ContextMenuLabel>
            <ContextMenuCheckboxItem
              checked={showDetectionOverlay}
              onCheckedChange={(checked) => {
                setShowDetectionOverlay(Boolean(checked));
              }}
              disabled={!hasDetectionData}
            >
              Show bounding boxes
            </ContextMenuCheckboxItem>
            <ContextMenuCheckboxItem
              checked={showImageInfo}
              onCheckedChange={(checked) => {
                setShowImageInfo(Boolean(checked));
              }}
              disabled={!activeImage.hoverDetails}
            >
              Show image context
            </ContextMenuCheckboxItem>
            <ContextMenuSeparator />
            <ContextMenuLabel className="text-xs font-normal text-muted-foreground">
              Right-click or press and hold on the image to change the overlay.
            </ContextMenuLabel>
          </ContextMenuContent>
        </ContextMenu>
      ) : (
        stage
      )}
      {hasMultipleImages ? (
        <>
          <div className="chat-tool-carousel-controls">
            <div className="chat-tool-carousel-nav">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setActiveIndex((prev) => (prev - 1 + images.length) % images.length)}
                aria-label="Previous image"
              >
                <ChevronLeft />
              </Button>
              <span className="chat-tool-carousel-index">
                {boundedIndex + 1} / {images.length}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setActiveIndex((prev) => (prev + 1) % images.length)}
                aria-label="Next image"
              >
                <ChevronRight />
              </Button>
            </div>
            <Button asChild variant="outline" size="sm">
              <a
                href={activeImage.downloadUrl ?? activeImage.url}
                download
                target="_blank"
                rel="noreferrer"
              >
                <Download className="size-4" />
                Download
              </a>
            </Button>
          </div>
          <div className="chat-tool-carousel-thumbs">
            {images.map((image, imageIndex) => (
              <button
                key={image.path}
                type="button"
                className={cn(
                  "chat-tool-carousel-thumb",
                  imageIndex === boundedIndex && "is-active"
                )}
                onClick={() => setActiveIndex(imageIndex)}
                aria-label={`Select ${image.title}`}
              >
                {image.previewable ? (
                  <img src={image.url} alt={image.title} loading="lazy" />
                ) : (
                  <div className="chat-tool-image-placeholder">
                    <ImageIcon className="size-4" />
                  </div>
                )}
              </button>
            ))}
          </div>
        </>
      ) : null}
      {prefersPersistentControls ? (
        <p className="chat-tool-carousel-menu-hint">
          Right-click or press and hold on the image to hide boxes or context.
        </p>
      ) : null}
      {showsPointHint ? (
        <p className="chat-tool-carousel-tip">
          Tip: ask "rerun with 256 points" to increase prompt density.
        </p>
      ) : null}
    </div>
  );
}
