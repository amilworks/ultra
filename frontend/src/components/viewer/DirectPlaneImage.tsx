import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type WheelEvent,
} from "react";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import * as THREE from "three";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { UploadViewerInfo } from "@/types";

import {
  loadScalarSliceBitmap,
  loadSliceBitmap,
  sliceBitmapToTexture,
  type ScalarSliceSource,
} from "./sliceImageCache";

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
  /**
   * When provided, the slice is rendered from the in-memory volume (instant, no
   * network). Falls back to {@link imageUrl} when absent.
   */
  scalarSlice?: ScalarSliceSource | null;
};

export type ImageViewportSize = {
  width: number;
  height: number;
};

export type ImageViewportPoint = {
  x: number;
  y: number;
};

export type ImageViewport = {
  centerX: number;
  centerY: number;
  scale: number;
};

type DragState = {
  pointerId: number;
  startX: number;
  startY: number;
  origin: ImageViewport;
};

const DEFAULT_VIEWPORT: ImageViewport = { centerX: 0, centerY: 0, scale: 1 };
const DEVICE_PIXEL_RATIO_LIMIT = 2;
const VIEW_EPSILON = 1e-6;
const ZOOM_STEP = 1.22;
const WHEEL_LINE_HEIGHT_PX = 16;
const WHEEL_ZOOM_DELTA_LIMIT = 240;
const WHEEL_ZOOM_SENSITIVITY = 0.00115;
const PINCH_ZOOM_SENSITIVITY = 0.0027;

const isUsableSize = (size: ImageViewportSize | null | undefined) =>
  Boolean(size && Number.isFinite(size.width) && Number.isFinite(size.height) && size.width > 0 && size.height > 0);

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const viewsEqual = (left: ImageViewport, right: ImageViewport) =>
  Math.abs(left.centerX - right.centerX) < VIEW_EPSILON &&
  Math.abs(left.centerY - right.centerY) < VIEW_EPSILON &&
  Math.abs(left.scale - right.scale) < VIEW_EPSILON;

export function computeImageFitScale(imageSize: ImageViewportSize, viewportSize: ImageViewportSize) {
  if (!isUsableSize(imageSize) || !isUsableSize(viewportSize)) {
    return 1;
  }
  return Math.max(
    VIEW_EPSILON,
    Math.min(viewportSize.width / Math.max(1, imageSize.width), viewportSize.height / Math.max(1, imageSize.height))
  );
}

export function computeImageScaleLimits(imageSize: ImageViewportSize, viewportSize: ImageViewportSize) {
  const fitScale = computeImageFitScale(imageSize, viewportSize);
  return {
    fitScale,
    minScale: Math.max(VIEW_EPSILON, fitScale * 0.5),
    maxScale: Math.max(4, 16, fitScale * 64),
  };
}

export function clampImageViewport(
  viewport: ImageViewport,
  imageSize: ImageViewportSize,
  viewportSize: ImageViewportSize,
  minScale: number,
  maxScale: number
): ImageViewport {
  const scale = clamp(Number.isFinite(viewport.scale) ? viewport.scale : 1, minScale, maxScale);
  if (!isUsableSize(imageSize) || !isUsableSize(viewportSize)) {
    return { centerX: 0, centerY: 0, scale };
  }

  const halfImageWidth = imageSize.width / 2;
  const halfImageHeight = imageSize.height / 2;
  const halfViewWidth = viewportSize.width / (2 * scale);
  const halfViewHeight = viewportSize.height / (2 * scale);
  const centerX =
    halfViewWidth >= halfImageWidth
      ? 0
      : clamp(viewport.centerX, -halfImageWidth + halfViewWidth, halfImageWidth - halfViewWidth);
  const centerY =
    halfViewHeight >= halfImageHeight
      ? 0
      : clamp(viewport.centerY, -halfImageHeight + halfViewHeight, halfImageHeight - halfViewHeight);

  return { centerX, centerY, scale };
}

export function zoomImageViewportAtPoint(
  viewport: ImageViewport,
  point: ImageViewportPoint,
  nextScale: number,
  imageSize: ImageViewportSize,
  viewportSize: ImageViewportSize,
  minScale: number,
  maxScale: number
): ImageViewport {
  const previousScale = Math.max(VIEW_EPSILON, viewport.scale);
  const scale = clamp(nextScale, minScale, maxScale);
  const offsetX = point.x - viewportSize.width / 2;
  const offsetY = point.y - viewportSize.height / 2;
  const worldX = viewport.centerX + offsetX / previousScale;
  const worldY = viewport.centerY - offsetY / previousScale;

  return clampImageViewport(
    {
      centerX: worldX - offsetX / scale,
      centerY: worldY + offsetY / scale,
      scale,
    },
    imageSize,
    viewportSize,
    minScale,
    maxScale
  );
}

export function panImageViewport(
  viewport: ImageViewport,
  delta: ImageViewportPoint,
  imageSize: ImageViewportSize,
  viewportSize: ImageViewportSize,
  minScale: number,
  maxScale: number
): ImageViewport {
  const scale = Math.max(VIEW_EPSILON, viewport.scale);
  return clampImageViewport(
    {
      centerX: viewport.centerX - delta.x / scale,
      centerY: viewport.centerY + delta.y / scale,
      scale,
    },
    imageSize,
    viewportSize,
    minScale,
    maxScale
  );
}

export function wheelDeltaToZoomFactor(deltaY: number, deltaMode: number, ctrlKey: boolean, viewportHeight: number) {
  if (!Number.isFinite(deltaY) || deltaY === 0) {
    return 1;
  }
  const normalizedDelta =
    deltaMode === 1
      ? deltaY * WHEEL_LINE_HEIGHT_PX
      : deltaMode === 2
        ? deltaY * Math.max(1, viewportHeight)
        : deltaY;
  const boundedDelta = clamp(normalizedDelta, -WHEEL_ZOOM_DELTA_LIMIT, WHEEL_ZOOM_DELTA_LIMIT);
  const sensitivity = ctrlKey ? PINCH_ZOOM_SENSITIVITY : WHEEL_ZOOM_SENSITIVITY;
  return Math.exp(-boundedDelta * sensitivity);
}

function getLoadedImageSize(texture: THREE.Texture, descriptor: UploadViewerInfo["viewer"]["default_plane"]) {
  const image = texture.image as { naturalWidth?: number; naturalHeight?: number; width?: number; height?: number };
  const width = Math.max(1, image.naturalWidth || image.width || descriptor.pixel_size?.width || 1);
  const height = Math.max(1, image.naturalHeight || image.height || descriptor.pixel_size?.height || 1);
  return { width, height };
}

function formatZoomLabel(scale: number, fitScale: number) {
  if (!Number.isFinite(scale) || Math.abs(scale - fitScale) / Math.max(VIEW_EPSILON, fitScale) < 0.025) {
    return "Fit";
  }
  if (scale >= 1) {
    return `${scale >= 10 ? scale.toFixed(0) : scale.toFixed(2)}x`;
  }
  return `${Math.max(1, Math.round(scale * 100))}%`;
}

export function DirectPlaneImage({
  imageUrl,
  placeholderUrl,
  descriptor,
  title,
  className,
  interactive = true,
  orientationLabels,
  scalarSlice,
}: DirectPlaneImageProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasHostRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const meshRef = useRef<THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> | null>(null);
  const textureRef = useRef<THREE.Texture | null>(null);
  const loadTokenRef = useRef(0);
  const dragRef = useRef<DragState | null>(null);

  const [viewportSize, setViewportSize] = useState<ImageViewportSize>({ width: 0, height: 0 });
  const [imageSize, setImageSize] = useState<ImageViewportSize | null>(null);
  const [viewport, setViewport] = useState<ImageViewport>(DEFAULT_VIEWPORT);
  const [isFitViewport, setIsFitViewport] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [hasTexture, setHasTexture] = useState(false);
  const [loadState, setLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [loadError, setLoadError] = useState<string | null>(null);

  const descriptorAspect = Math.max(VIEW_EPSILON, descriptor.aspect_ratio);
  const naturalAspect = imageSize ? imageSize.width / Math.max(1, imageSize.height) : descriptorAspect;
  const scaleLimits = useMemo(
    () => computeImageScaleLimits(imageSize ?? { width: 1, height: 1 }, viewportSize),
    [imageSize, viewportSize]
  );
  const zoomLabel = formatZoomLabel(viewport.scale, scaleLimits.fitScale);
  const dimensionLabel = imageSize ? `${imageSize.width} x ${imageSize.height}` : "Loading";
  const primaryImageUrl = imageUrl || placeholderUrl || "";

  useEffect(() => {
    const canvasHost = canvasHostRef.current;
    if (!canvasHost) {
      return;
    }

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        depth: false,
        stencil: false,
        powerPreference: "high-performance",
      });
    } catch {
      queueMicrotask(() => {
        setLoadError("WebGL is unavailable. Showing the image fallback.");
        setLoadState("error");
      });
      return;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, DEVICE_PIXEL_RATIO_LIMIT));
    renderer.setClearColor(0x05070a, 1);
    renderer.domElement.className = "viewer-direct-image-canvas";
    renderer.domElement.setAttribute("aria-hidden", "true");
    canvasHost.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);
    camera.position.z = 1;
    const geometry = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.MeshBasicMaterial({
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    meshRef.current = mesh;

    return () => {
      textureRef.current?.dispose();
      textureRef.current = null;
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentElement === canvasHost) {
        canvasHost.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      meshRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvasHost = canvasHostRef.current;
    if (!canvasHost) {
      return;
    }

    const updateSize = () => {
      const rect = canvasHost.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      rendererRef.current?.setPixelRatio(Math.min(window.devicePixelRatio || 1, DEVICE_PIXEL_RATIO_LIMIT));
      rendererRef.current?.setSize(width, height, false);
      setViewportSize((previous) => (previous.width === width && previous.height === height ? previous : { width, height }));
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(canvasHost);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if ((!scalarSlice && !primaryImageUrl) || !meshRef.current) {
      return;
    }

    const token = loadTokenRef.current + 1;
    loadTokenRef.current = token;
    setLoadState("loading");
    setLoadError(null);

    const loadPromise = scalarSlice ? loadScalarSliceBitmap(scalarSlice) : loadSliceBitmap(primaryImageUrl);
    loadPromise
      .then((bitmap) => {
        if (loadTokenRef.current !== token || !meshRef.current) {
          return;
        }

        const texture = sliceBitmapToTexture(bitmap);
        const nextImageSize = getLoadedImageSize(texture, descriptor);
        const previousTexture = textureRef.current;
        textureRef.current = texture;
        meshRef.current.material.map = texture;
        meshRef.current.material.needsUpdate = true;
        meshRef.current.scale.set(nextImageSize.width, nextImageSize.height, 1);
        previousTexture?.dispose();
        setImageSize(nextImageSize);
        setIsFitViewport(true);
        setHasTexture(true);
        setLoadState("ready");
      })
      .catch(() => {
        if (loadTokenRef.current !== token) {
          return;
        }
        const hasPreviousTexture = Boolean(textureRef.current);
        setLoadError(hasPreviousTexture ? "Failed to load the next image frame." : "Failed to load image.");
        setLoadState(hasPreviousTexture ? "ready" : "error");
      });
  }, [descriptor, primaryImageUrl, scalarSlice]);

  useEffect(() => {
    if (!imageSize || !isUsableSize(viewportSize)) {
      return;
    }

    let cancelled = false;
    const nextLimits = computeImageScaleLimits(imageSize, viewportSize);
    queueMicrotask(() => {
      if (cancelled) {
        return;
      }
      setViewport((previous) => {
        const next = isFitViewport
          ? { centerX: 0, centerY: 0, scale: nextLimits.fitScale }
          : clampImageViewport(previous, imageSize, viewportSize, nextLimits.minScale, nextLimits.maxScale);
        return viewsEqual(previous, next) ? previous : next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [imageSize, isFitViewport, viewportSize]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (!renderer || !scene || !camera || !imageSize || !isUsableSize(viewportSize)) {
      return;
    }

    const scale = Math.max(VIEW_EPSILON, viewport.scale);
    const halfViewWidth = viewportSize.width / (2 * scale);
    const halfViewHeight = viewportSize.height / (2 * scale);
    camera.left = viewport.centerX - halfViewWidth;
    camera.right = viewport.centerX + halfViewWidth;
    camera.top = viewport.centerY + halfViewHeight;
    camera.bottom = viewport.centerY - halfViewHeight;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  }, [imageSize, viewport, viewportSize]);

  const resetViewport = useCallback(() => {
    if (!imageSize || !isUsableSize(viewportSize)) {
      return;
    }
    const nextLimits = computeImageScaleLimits(imageSize, viewportSize);
    setIsFitViewport(true);
    setViewport({ centerX: 0, centerY: 0, scale: nextLimits.fitScale });
  }, [imageSize, viewportSize]);

  const zoomBy = useCallback(
    (factor: number, point?: ImageViewportPoint) => {
      if (!imageSize || !isUsableSize(viewportSize)) {
        return;
      }
      const nextLimits = computeImageScaleLimits(imageSize, viewportSize);
      const anchor = point ?? { x: viewportSize.width / 2, y: viewportSize.height / 2 };
      setIsFitViewport(false);
      setViewport((previous) =>
        zoomImageViewportAtPoint(
          previous,
          anchor,
          previous.scale * factor,
          imageSize,
          viewportSize,
          nextLimits.minScale,
          nextLimits.maxScale
        )
      );
    },
    [imageSize, viewportSize]
  );

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!interactive) {
      return;
    }
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    const rect = event.currentTarget.getBoundingClientRect();
    const factor = wheelDeltaToZoomFactor(event.deltaY, event.deltaMode, event.ctrlKey, viewportSize.height || rect.height);
    if (Math.abs(factor - 1) < VIEW_EPSILON) {
      return;
    }
    zoomBy(factor, {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!interactive || !imageSize || !isUsableSize(viewportSize)) {
      return;
    }
    const target = event.target as HTMLElement | null;
    if (target?.closest("[data-viewer-image-control]")) {
      return;
    }
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: viewport,
    };
    setIsDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!interactive || !drag || drag.pointerId !== event.pointerId || !imageSize || !isUsableSize(viewportSize)) {
      return;
    }
    const nextLimits = computeImageScaleLimits(imageSize, viewportSize);
    setIsFitViewport(false);
    setViewport(
      panImageViewport(
        drag.origin,
        { x: event.clientX - drag.startX, y: event.clientY - drag.startY },
        imageSize,
        viewportSize,
        nextLimits.minScale,
        nextLimits.maxScale
      )
    );
  };

  const finishPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    setIsDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!interactive) {
      return;
    }
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomBy(ZOOM_STEP);
    }
    if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoomBy(1 / ZOOM_STEP);
    }
    if (event.key === "0") {
      event.preventDefault();
      resetViewport();
    }
  };

  return (
    <div
      ref={containerRef}
      className={className ?? "viewer-canvas-root"}
      role="group"
      aria-label={`${title} image viewer`}
      tabIndex={interactive ? 0 : undefined}
      data-viewer-title={title}
      data-viewer-aspect={descriptorAspect.toFixed(4)}
      data-viewer-renderer="threejs-image"
      data-viewer-loading-state={loadState}
      data-viewer-active-src={primaryImageUrl}
      data-image-natural-width={imageSize?.width ?? ""}
      data-image-natural-height={imageSize?.height ?? ""}
      data-image-natural-aspect={naturalAspect.toFixed(6)}
      data-image-descriptor-aspect={descriptorAspect.toFixed(6)}
      data-image-fit-scale={scaleLimits.fitScale.toFixed(6)}
      data-image-view-scale={viewport.scale.toFixed(6)}
      data-image-screen-aspect={naturalAspect.toFixed(6)}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishPointer}
      onPointerCancel={finishPointer}
      onDoubleClick={resetViewport}
      onKeyDown={handleKeyDown}
    >
      <div
        ref={canvasHostRef}
        className="viewer-direct-image-shell"
        style={{ cursor: interactive ? (isDragging ? "grabbing" : "grab") : "default" }}
      >
        {!hasTexture && loadState === "error" && primaryImageUrl ? (
          <img src={primaryImageUrl} alt={title} className="viewer-direct-image-fallback-media" draggable={false} />
        ) : null}
        {orientationLabels ? (
          <div className="viewer-orientation-overlay" aria-hidden="true">
            <span className="viewer-orientation-label viewer-orientation-label-top">{orientationLabels.top}</span>
            <span className="viewer-orientation-label viewer-orientation-label-bottom">{orientationLabels.bottom}</span>
            <span className="viewer-orientation-label viewer-orientation-label-left">{orientationLabels.left}</span>
            <span className="viewer-orientation-label viewer-orientation-label-right">{orientationLabels.right}</span>
          </div>
        ) : null}
        <div className="viewer-direct-image-toolbar" data-viewer-image-control="true">
          <Badge variant="outline" className="viewer-direct-image-readout">
            {dimensionLabel}
          </Badge>
          <Badge variant="secondary" className="viewer-direct-image-readout">
            {zoomLabel}
          </Badge>
          <Button type="button" variant="outline" size="icon-xs" aria-label="Zoom out" onClick={() => zoomBy(1 / ZOOM_STEP)}>
            <ZoomOut data-icon="inline-start" />
          </Button>
          <Button type="button" variant="outline" size="icon-xs" aria-label="Zoom in" onClick={() => zoomBy(ZOOM_STEP)}>
            <ZoomIn data-icon="inline-start" />
          </Button>
          <Button type="button" variant="outline" size="icon-xs" aria-label="Reset image view" onClick={resetViewport}>
            <Maximize2 data-icon="inline-start" />
          </Button>
        </div>
        {loadState === "loading" ? (
          <div className="viewer-direct-image-status" aria-live="polite">
            <span>{hasTexture ? "Loading next image..." : "Loading image..."}</span>
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
