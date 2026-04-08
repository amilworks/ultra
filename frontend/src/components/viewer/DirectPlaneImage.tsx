import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";

import type { UploadViewerInfo } from "@/types";

type DirectPlaneImageProps = {
  imageUrl: string;
  placeholderUrl?: string | null;
  descriptor: UploadViewerInfo["viewer"]["default_plane"];
  title: string;
  className?: string;
  interactive?: boolean;
  orientationLabels?: {
    top: string;
    bottom: string;
    left: string;
    right: string;
  };
};

type PanOffset = {
  x: number;
  y: number;
};

const clampScale = (value: number) => Math.min(12, Math.max(1, value));

export function DirectPlaneImage({
  imageUrl,
  placeholderUrl,
  descriptor,
  title,
  className,
  interactive = true,
  orientationLabels,
}: DirectPlaneImageProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    origin: PanOffset;
  } | null>(null);
  const loadedUrlsRef = useRef<Set<string>>(new Set());

  const [visibleUrl, setVisibleUrl] = useState<string>(placeholderUrl || imageUrl);
  const [loadState, setLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState<PanOffset>({ x: 0, y: 0 });

  const aspectRatio = useMemo(() => `${Math.max(1e-6, descriptor.aspect_ratio)}`, [descriptor.aspect_ratio]);

  useEffect(() => {
    if (!visibleUrl) {
      setVisibleUrl(placeholderUrl || imageUrl);
    }
  }, [imageUrl, placeholderUrl, visibleUrl]);

  useEffect(() => {
    if (loadedUrlsRef.current.has(imageUrl)) {
      if (visibleUrl !== imageUrl) {
        setVisibleUrl(imageUrl);
      }
      setLoadState("ready");
      setLoadError(null);
      return;
    }
    if (visibleUrl === imageUrl) {
      setLoadState("loading");
      return;
    }

    let active = true;
    setLoadState("loading");
    setLoadError(null);
    const preload = new Image();
    preload.decoding = "async";
    preload.onload = async () => {
      try {
        if (typeof preload.decode === "function") {
          await preload.decode();
        }
      } catch {
        // Browser decode hints are advisory; keep the loaded image.
      }
      if (!active) {
        return;
      }
      loadedUrlsRef.current.add(imageUrl);
      setVisibleUrl(imageUrl);
      setLoadState("ready");
    };
    preload.onerror = () => {
      if (!active) {
        return;
      }
      setLoadError("Failed to load the next image frame.");
      setLoadState(visibleUrl ? "ready" : "error");
    };
    preload.src = imageUrl;
    return () => {
      active = false;
      preload.onload = null;
      preload.onerror = null;
    };
  }, [imageUrl, visibleUrl]);

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!interactive) {
      return;
    }
    event.preventDefault();
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const rect = container.getBoundingClientRect();
    const pointerX = event.clientX - rect.left - rect.width / 2;
    const pointerY = event.clientY - rect.top - rect.height / 2;
    setScale((previousScale) => {
      const nextScale = clampScale(previousScale * (event.deltaY < 0 ? 1.14 : 1 / 1.14));
      if (Math.abs(nextScale - previousScale) < 1e-6) {
        return previousScale;
      }
      const zoomFactor = nextScale / previousScale;
      setOffset((previousOffset) => {
        if (nextScale === 1) {
          return { x: 0, y: 0 };
        }
        return {
          x: pointerX - (pointerX - previousOffset.x) * zoomFactor,
          y: pointerY - (pointerY - previousOffset.y) * zoomFactor,
        };
      });
      return nextScale;
    });
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!interactive || scale <= 1) {
      return;
    }
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: offset,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!interactive || !drag || drag.pointerId !== event.pointerId) {
      return;
    }
    setOffset({
      x: drag.origin.x + (event.clientX - drag.startX),
      y: drag.origin.y + (event.clientY - drag.startY),
    });
  };

  const finishPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  return (
    <div
      ref={containerRef}
      className={className ?? "viewer-canvas-root"}
      data-viewer-title={title}
      data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
      data-viewer-renderer="image"
      data-viewer-loading-state={loadState}
      data-viewer-active-src={visibleUrl}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishPointer}
      onPointerCancel={finishPointer}
      onDoubleClick={resetView}
    >
      <div className="viewer-direct-image-shell" style={{ aspectRatio }}>
        <div
          className="viewer-direct-image-stage"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            cursor: interactive ? (dragRef.current ? "grabbing" : scale > 1 ? "grab" : "zoom-in") : "default",
          }}
        >
          <img
            src={visibleUrl}
            alt={title}
            className="viewer-direct-image-media"
            draggable={false}
            onLoad={(event) => {
              const currentSrc = event.currentTarget.currentSrc || visibleUrl;
              if (currentSrc) {
                loadedUrlsRef.current.add(currentSrc);
              }
              if (visibleUrl === imageUrl) {
                setLoadState("ready");
                setLoadError(null);
              }
            }}
            onError={() => {
              if (visibleUrl === imageUrl) {
                setLoadState("error");
                setLoadError("Failed to load image.");
              }
            }}
          />
        </div>
        {orientationLabels ? (
          <div className="viewer-orientation-overlay" aria-hidden="true">
            <span className="viewer-orientation-label viewer-orientation-label-top">{orientationLabels.top}</span>
            <span className="viewer-orientation-label viewer-orientation-label-bottom">{orientationLabels.bottom}</span>
            <span className="viewer-orientation-label viewer-orientation-label-left">{orientationLabels.left}</span>
            <span className="viewer-orientation-label viewer-orientation-label-right">{orientationLabels.right}</span>
          </div>
        ) : null}
        {loadState === "loading" ? (
          <div className="viewer-direct-image-status" aria-live="polite">
            <span>{visibleUrl === imageUrl ? "Loading image…" : "Loading next image…"}</span>
          </div>
        ) : null}
        {loadError ? (
          <div className="viewer-direct-image-error" role="status">
            <span>{loadError}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
