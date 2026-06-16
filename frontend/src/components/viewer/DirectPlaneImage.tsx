import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import * as THREE from "three";

import type { UploadViewerInfo } from "@/types";

import {
  loadScalarSliceBitmap,
  loadSliceBitmap,
  sliceBitmapToTexture,
  type ScalarSliceSource,
} from "./sliceImageCache";
import {
  createChromeFadeController,
  prefersReducedMotionSafe,
  type ChromeFadeController,
} from "./chromeVisibility";
import { composeViewBlob, type ViewerCanvasHandle } from "./captureView";
import { ViewerZoomToolbar } from "./ViewerZoomToolbar";
import { ZOOM_STEP, formatZoomReadout, isAtFitScale, nativeZoomLevel } from "./viewerZoom";
import { classifyWheelGesture } from "./wheelGesture";

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

// getPlaneWorldSize returns the plane's physical extent (pixels x spacing) so the
// mesh and viewport keep anatomically correct proportions for anisotropic voxels
// (e.g. 0.5x0.5x5mm). Falls back to the pixel grid when spacing is absent.
function getPlaneWorldSize(
  descriptor: UploadViewerInfo["viewer"]["default_plane"],
  pixelSize: ImageViewportSize
): ImageViewportSize {
  const worldWidth = Number(descriptor.world_size?.width);
  const worldHeight = Number(descriptor.world_size?.height);
  if (Number.isFinite(worldWidth) && worldWidth > 0 && Number.isFinite(worldHeight) && worldHeight > 0) {
    return { width: worldWidth, height: worldHeight };
  }
  return pixelSize;
}

export const DirectPlaneImage = forwardRef<ViewerCanvasHandle, DirectPlaneImageProps>(function DirectPlaneImage(
  { imageUrl, placeholderUrl, descriptor, title, className, interactive = true, orientationLabels, scalarSlice },
  ref
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasHostRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const meshRef = useRef<THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> | null>(null);
  const textureRef = useRef<THREE.Texture | null>(null);
  const loadTokenRef = useRef(0);
  const dragRef = useRef<DragState | null>(null);
  // Active touch/pen pointers for pinch-zoom (phones emit no wheel events, so pinch
  // is the only way to zoom on touch). Keyed by pointerId; pinchRef holds the last
  // two-finger distance so each move applies an incremental zoom factor.
  const pointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinchRef = useRef<{ dist: number } | null>(null);
  const chromeFadeRef = useRef<ChromeFadeController | null>(null);

  const [viewportSize, setViewportSize] = useState<ImageViewportSize>({ width: 0, height: 0 });
  // imageSize is the physical world extent (drives fit/zoom + mesh aspect);
  // pixelSize is the voxel grid (drives the dimension readout only).
  const [imageSize, setImageSize] = useState<ImageViewportSize | null>(null);
  const [pixelSize, setPixelSize] = useState<ImageViewportSize | null>(null);
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
  // viewport.scale is screen-px per WORLD unit; convert to screen-px per SOURCE px so
  // the readout reads true "% of native" — identical semantics to the deep-zoom
  // surface, even when an image's physical spacing differs from its pixel grid.
  const nativeZoom =
    imageSize && pixelSize
      ? nativeZoomLevel(viewport.scale, imageSize.width, pixelSize.width)
      : viewport.scale;
  const zoomReadout = formatZoomReadout(nativeZoom, isAtFitScale(viewport.scale, scaleLimits.fitScale));
  const dimensionLabel = pixelSize ? `${pixelSize.width} x ${pixelSize.height}` : "Loading";
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

  // Receding chrome: reveal on pointer activity / focus, fade after idle. Disabled
  // (chrome stays fully visible) when the panel is non-interactive, since there is
  // then no pointer affordance to bring it back.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !interactive) {
      return;
    }
    const controller = createChromeFadeController(container, {
      reducedMotion: prefersReducedMotionSafe(),
    });
    chromeFadeRef.current = controller;
    controller.reveal();
    return () => {
      controller.dispose();
      chromeFadeRef.current = null;
      delete container.dataset.chromeFaded;
    };
  }, [interactive]);

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
        const nextPixelSize = getLoadedImageSize(texture, descriptor);
        const nextWorldSize = getPlaneWorldSize(descriptor, nextPixelSize);
        const previousTexture = textureRef.current;
        textureRef.current = texture;
        meshRef.current.material.map = texture;
        meshRef.current.material.needsUpdate = true;
        // Scale the textured plane to physical extent so anisotropic voxels are
        // not squashed; the texture (UV 0..1) stretches to match.
        meshRef.current.scale.set(nextWorldSize.width, nextWorldSize.height, 1);
        previousTexture?.dispose();
        setImageSize(nextWorldSize);
        setPixelSize(nextPixelSize);
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

  // Expose fit + "export the current view" to the host shell (context menu). The
  // renderer has no preserveDrawingBuffer, so we re-render the current frustum and
  // hand composeViewBlob the fresh frame, which snapshots it synchronously before the
  // unpreserved backbuffer clears. Orientation labels are composited in for slides.
  useImperativeHandle(
    ref,
    (): ViewerCanvasHandle => ({
      fitView: resetViewport,
      captureViewToBlob: () => {
        const renderer = rendererRef.current;
        const scene = sceneRef.current;
        const camera = cameraRef.current;
        const host = canvasHostRef.current;
        if (!renderer || !scene || !camera || !host || !imageSize || !isUsableSize(viewportSize)) {
          return Promise.resolve(null);
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
        return composeViewBlob(host, renderer.domElement, [".viewer-orientation-label"], "#05070a");
      },
    }),
    [resetViewport, imageSize, viewport, viewportSize]
  );

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

  // Distance + midpoint of the first two active pointers, for pinch-zoom.
  const readPinchGeometry = () => {
    const points = Array.from(pointersRef.current.values());
    if (points.length < 2) {
      return null;
    }
    const [a, b] = points;
    return {
      dist: Math.hypot(b.x - a.x, b.y - a.y),
      midX: (a.x + b.x) / 2,
      midY: (a.y + b.y) / 2,
    };
  };

  // React attaches onWheel as a PASSIVE listener, so its preventDefault() is a no-op —
  // a Mac trackpad pinch (which fires ctrl+wheel) would zoom the whole PAGE instead of
  // the image, and a mouse wheel would scroll the page. We attach a NATIVE non-passive
  // wheel listener (matching the deep-zoom canvas) so preventDefault works and the
  // gesture reliably zooms/pans the image. The handler is kept in a ref so it always
  // sees the latest state without re-attaching the listener.
  const wheelHandlerRef = useRef<(event: WheelEvent) => void>(() => {});
  useEffect(() => {
    wheelHandlerRef.current = (event: WheelEvent) => {
      chromeFadeRef.current?.reveal();
      if (!interactive || !imageSize || !isUsableSize(viewportSize)) {
        return;
      }
      event.preventDefault();
      const target = (event.currentTarget as HTMLElement | null) ?? containerRef.current;
      target?.focus({ preventScroll: true });
      const rect = target?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      // Two-finger trackpad scroll pans; pinch / ⌘-scroll / mouse wheel zooms toward the
      // cursor — consistent with the deep-zoom canvas (shared classifyWheelGesture).
      if (classifyWheelGesture(event) === "pan") {
        const nextLimits = computeImageScaleLimits(imageSize, viewportSize);
        setIsFitViewport(false);
        setViewport((previous) =>
          panImageViewport(
            previous,
            { x: -event.deltaX, y: -event.deltaY },
            imageSize,
            viewportSize,
            nextLimits.minScale,
            nextLimits.maxScale
          )
        );
        return;
      }
      const factor = wheelDeltaToZoomFactor(event.deltaY, event.deltaMode, event.ctrlKey, viewportSize.height || rect.height);
      if (Math.abs(factor - 1) < VIEW_EPSILON) {
        return;
      }
      zoomBy(factor, { x: event.clientX - rect.left, y: event.clientY - rect.top });
    };
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }
    const onWheel = (event: WheelEvent) => wheelHandlerRef.current(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    chromeFadeRef.current?.reveal();
    if (!interactive || !imageSize || !isUsableSize(viewportSize)) {
      return;
    }
    const target = event.target as HTMLElement | null;
    if (target?.closest("[data-viewer-image-control]")) {
      return;
    }
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
    if (pointersRef.current.size >= 2) {
      // Second finger down → switch from pan to pinch-zoom. Drop the in-flight
      // single-finger drag so the image doesn't lurch, and seed the pinch distance.
      dragRef.current = null;
      setIsDragging(false);
      const pinch = readPinchGeometry();
      pinchRef.current = pinch ? { dist: pinch.dist } : null;
      return;
    }
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: viewport,
    };
    setIsDragging(true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    chromeFadeRef.current?.reveal();
    if (!interactive || !imageSize || !isUsableSize(viewportSize)) {
      return;
    }
    if (pointersRef.current.has(event.pointerId)) {
      pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }
    // Two fingers down → pinch-zoom toward the midpoint, never pan.
    if (pointersRef.current.size >= 2) {
      const pinch = readPinchGeometry();
      const previous = pinchRef.current;
      if (pinch && previous && previous.dist > 0) {
        const factor = pinch.dist / previous.dist;
        if (Math.abs(factor - 1) > VIEW_EPSILON) {
          const rect = event.currentTarget.getBoundingClientRect();
          zoomBy(factor, { x: pinch.midX - rect.left, y: pinch.midY - rect.top });
        }
        pinchRef.current = { dist: pinch.dist };
      }
      return;
    }
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
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
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) {
      // Dropped below two fingers → pinch is over. A lingering finger won't resume
      // panning until it lifts and taps again (avoids a jump from the stale origin).
      pinchRef.current = null;
    }
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      setIsDragging(false);
    }
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
    if (event.key === "0" || event.key === "r" || event.key === "R") {
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
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishPointer}
      onPointerCancel={finishPointer}
      onDoubleClick={resetViewport}
      onKeyDown={handleKeyDown}
      onFocus={() => chromeFadeRef.current?.reveal()}
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
        <ViewerZoomToolbar
          readout={zoomReadout}
          extraReadout={dimensionLabel}
          onZoomOut={() => zoomBy(1 / ZOOM_STEP)}
          onZoomIn={() => zoomBy(ZOOM_STEP)}
          onFit={resetViewport}
        />
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
});
