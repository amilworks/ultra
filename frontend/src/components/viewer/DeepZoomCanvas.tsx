import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import * as THREE from "three";

import type { ApiClient } from "@/lib/api";
import { supportsSliceChannelColor } from "@/lib/viewerSliceContract";
import type { UploadViewerInfo } from "@/types";

import {
  clampImageViewport,
  computeImageScaleLimits,
  panImageViewport,
  wheelDeltaToZoomFactor,
  zoomImageViewportAtPoint,
  type ImageViewport,
  type ImageViewportSize,
} from "./DirectPlaneImage";
import { composeViewBlob, type ViewerCanvasHandle } from "./captureView";
import { ViewerZoomToolbar } from "./ViewerZoomToolbar";
import { ZOOM_STEP, formatZoomReadout, nativeZoomLevel } from "./viewerZoom";
import { createChromeFadeController, prefersReducedMotionSafe } from "./chromeVisibility";
import { classifyWheelGesture } from "./wheelGesture";

type DeepZoomCanvasProps = {
  apiClient: ApiClient;
  fileId: string;
  viewerInfo: UploadViewerInfo;
  axis?: "z" | "y" | "x";
  zIndex: number;
  tIndex: number;
  channels?: number[];
  /** Source-channel-indexed LUT palette. */
  channelColors?: string[];
  cacheKey?: string;
  className?: string;
};

type TileKey = `${number}:${number}:${number}`;
type TileLevel = UploadViewerInfo["viewer"]["tile_scheme"]["levels"][number];

// Transient tile-load failures (network blip, momentary engine load) are retried
// with exponential backoff before a tile is dropped, so deep zoom self-heals
// instead of leaving a permanent blank gap in the image.
const MAX_TILE_LOAD_RETRIES = 3;

// Pick the pyramid level whose pixels best match the screen at the current zoom — the
// one nearest 1 screen-px per level-px (Photoshop/Preview/Maps convention). Used for
// EVERY zoom including the fit view: forcing the coarsest overview level when the whole
// image is visible (the old shortcut) left huge images blurry until the user zoomed in.
// Generic over the level shape (only `downsample` is needed) so it's unit-testable.
export const chooseTileLevel = <T extends { downsample: number }>(
  levels: T[],
  viewportWidth: number,
  visibleWorldWidth: number,
  worldWidth: number,
  fullWidth: number,
): T => {
  const screenPixelsPerWorldUnit = Math.max(1, viewportWidth) / Math.max(1e-9, visibleWorldWidth);
  const sourcePixelsPerWorldUnit = fullWidth / Math.max(1e-9, worldWidth);
  const screenPixelsPerSourcePixel = screenPixelsPerWorldUnit / Math.max(1e-9, sourcePixelsPerWorldUnit);
  let bestLevel = levels[levels.length - 1];
  let bestScore = Number.POSITIVE_INFINITY;
  for (const level of levels) {
    const screenPixelsPerLevelPixel = screenPixelsPerSourcePixel * level.downsample;
    const score = Math.abs(Math.log2(Math.max(screenPixelsPerLevelPixel, 1e-6)));
    if (score < bestScore) {
      bestScore = score;
      bestLevel = level;
    }
  }
  return bestLevel;
};

export const orderTileLevelsForRendering = (levels: TileLevel[]): TileLevel[] =>
  [...levels].sort((left, right) => {
    const downsampleDelta = right.downsample - left.downsample;
    if (downsampleDelta !== 0) {
      return downsampleDelta;
    }
    return left.width * left.height - right.width * right.height;
  });

// niceScaleBar picks a round physical length whose on-screen width is ~targetPx, with
// unit promotion (µm→mm→m, m→km) for readability. screenPxPerUnit is screen px per one
// world/physical unit; unit is the metadata's pixel unit. Returns null when the unit is
// unusable, so the viewer shows the zoom % alone. Pure + exported for tests.
export const niceScaleBar = (
  screenPxPerUnit: number,
  unit: string,
  targetPx = 120
): { widthPx: number; label: string } | null => {
  if (!Number.isFinite(screenPxPerUnit) || screenPxPerUnit <= 0) {
    return null;
  }
  const base = (unit || "").trim().toLowerCase();
  // ladder of (unit label, factor relative to the base unit) for promotion
  const ladders: Record<string, Array<[string, number]>> = {
    um: [["µm", 1], ["mm", 1000], ["m", 1e6], ["km", 1e9]],
    "µm": [["µm", 1], ["mm", 1000], ["m", 1e6], ["km", 1e9]],
    micron: [["µm", 1], ["mm", 1000], ["m", 1e6], ["km", 1e9]],
    mm: [["mm", 1], ["m", 1000], ["km", 1e6]],
    m: [["m", 1], ["km", 1000]],
    nm: [["nm", 1], ["µm", 1000], ["mm", 1e6]],
  };
  const ladder = ladders[base] ?? [[unit || "px", 1]];
  const rawUnits = targetPx / screenPxPerUnit; // base units that span ~targetPx
  // choose the ladder rung that keeps the number in a readable range
  let label = ladder[0][0];
  let factor = ladder[0][1];
  for (const [lbl, f] of ladder) {
    if (rawUnits / f < 1000) {
      label = lbl;
      factor = f;
      break;
    }
    label = lbl;
    factor = f;
  }
  const inUnit = rawUnits / factor;
  // round to a nice 1/2/5 × 10^n
  const pow = Math.pow(10, Math.floor(Math.log10(inUnit)));
  const frac = inUnit / pow;
  const niceFrac = frac >= 5 ? 5 : frac >= 2 ? 2 : 1;
  const nice = niceFrac * pow;
  const widthPx = nice * factor * screenPxPerUnit;
  if (!Number.isFinite(widthPx) || widthPx <= 0) {
    return null;
  }
  const numText = nice >= 1 ? String(nice) : String(Number(nice.toPrecision(2)));
  return { widthPx, label: `${numText} ${label}` };
};

type ViewReadout = { zoom: string; atFit: boolean; scaleBar: { widthPx: number; label: string } | null };

export const DeepZoomCanvas = forwardRef<ViewerCanvasHandle, DeepZoomCanvasProps>(function DeepZoomCanvas(
  {
    apiClient,
    fileId,
    viewerInfo,
    axis = "z",
    zIndex,
    tIndex,
    channels,
    channelColors,
    cacheKey,
    className,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [readout, setReadout] = useState<ViewReadout>({ zoom: "Fit", atFit: true, scaleBar: null });
  // Imperative handles set by the WebGL effect so the toolbar/keyboard/context-menu can drive it.
  const controllerRef = useRef<{
    fit: () => void;
    zoomByAtCenter: (factor: number) => void;
    captureViewToBlob: () => Promise<Blob | null>;
  } | null>(null);

  // Expose fit + capture to the host shell (context menu). Delegates to the live
  // controllerRef the WebGL effect populates (null until the effect runs / on fallback).
  useImperativeHandle(
    ref,
    (): ViewerCanvasHandle => ({
      fitView: () => controllerRef.current?.fit(),
      captureViewToBlob: () => controllerRef.current?.captureViewToBlob() ?? Promise.resolve(null),
    }),
    []
  );

  const descriptor = useMemo(
    () => viewerInfo.viewer.planes[axis] ?? viewerInfo.viewer.default_plane,
    [axis, viewerInfo.viewer.default_plane, viewerInfo.viewer.planes]
  );
  const tileLevels = useMemo(
    () => orderTileLevelsForRendering(viewerInfo.viewer.tile_scheme.levels),
    [viewerInfo.viewer.tile_scheme.levels]
  );
  const physicalUnit = useMemo(() => {
    const units = viewerInfo.phys?.pixel_units;
    return Array.isArray(units) && units[0] ? String(units[0]) : "px";
  }, [viewerInfo.phys]);
  const effectiveChannelColors = supportsSliceChannelColor(viewerInfo)
    ? channelColors
    : undefined;
  const fallbackImageUrl = useMemo(
    () =>
      apiClient.uploadSliceUrl(fileId, {
        axis,
        z: zIndex,
        t: tIndex,
        channels,
        channelColors: effectiveChannelColors,
        cacheKey,
      }),
    [apiClient, axis, cacheKey, channels, effectiveChannelColors, fileId, tIndex, zIndex]
  );

  // Stable callback the imperative layer calls to publish the current view to the toolbar.
  const publishReadout = useCallback((next: ViewReadout) => {
    setReadout((prev) =>
      prev.zoom === next.zoom &&
      prev.atFit === next.atFit &&
      prev.scaleBar?.label === next.scaleBar?.label &&
      Math.round(prev.scaleBar?.widthPx ?? -1) === Math.round(next.scaleBar?.widthPx ?? -1)
        ? prev
        : next
    );
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    const commitRenderError = (message: string | null) => {
      window.setTimeout(() => {
        if (!disposed) {
          setRenderError(message);
        }
      }, 0);
    };
    try {
      // preserveDrawingBuffer keeps the backbuffer readable after compositing so the
      // context-menu "Export / Copy current view" can capture it (negligible cost for a
      // 2D image plane). Without it canvas.toBlob/toDataURL returns a blank frame.
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      });
      if (renderError) {
        commitRenderError(null);
      }
    } catch (error) {
      commitRenderError(error instanceof Error ? error.message : "WebGL unavailable");
      return () => {
        disposed = true;
      };
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xf5f2eb, 1);
    renderer.domElement.className = "viewer-webgl-canvas";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // zoom=1 always; the visible region is driven entirely by the frustum (viewport).
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    camera.position.set(0, 0, 10);

    const group = new THREE.Group();
    scene.add(group);

    const loader = new THREE.TextureLoader();
    const tileMeshes = new Map<TileKey, THREE.Mesh>();
    const textureCache = new Map<TileKey, THREE.Texture>();
    const materialCache = new Map<TileKey, THREE.MeshBasicMaterial>();
    const inflight = new Set<TileKey>();

    const worldWidth = Math.max(1, Number(descriptor.world_size.width) || 1);
    const worldHeight = Math.max(1, Number(descriptor.world_size.height) || 1);
    const fullWidth = Math.max(1, Number(descriptor.pixel_size.width) || 1);
    const imageSize: ImageViewportSize = { width: worldWidth, height: worldHeight };

    let viewportSize: ImageViewportSize = { width: 1, height: 1 };
    let viewport: ImageViewport = { centerX: 0, centerY: 0, scale: 1 };
    let limits = computeImageScaleLimits(imageSize, viewportSize);
    let hasFitted = false; // first layout always fits the whole image

    const render = () => renderer.render(scene, camera);

    const applyCamera = () => {
      const scale = Math.max(1e-9, viewport.scale);
      const halfW = viewportSize.width / (2 * scale);
      const halfH = viewportSize.height / (2 * scale);
      camera.left = viewport.centerX - halfW;
      camera.right = viewport.centerX + halfW;
      camera.top = viewport.centerY + halfH;
      camera.bottom = viewport.centerY - halfH;
      camera.updateProjectionMatrix();
    };

    const reportView = () => {
      const screenPxPerSourcePx = nativeZoomLevel(viewport.scale, worldWidth, fullWidth); // scale is px/world-unit
      const atFit = Math.abs(viewport.scale - limits.fitScale) / Math.max(1e-9, limits.fitScale) < 0.02;
      // scale bar: screen px per one physical unit == viewport.scale (world unit == physical unit)
      const scaleBar = niceScaleBar(viewport.scale, physicalUnit);
      publishReadout({ zoom: formatZoomReadout(screenPxPerSourcePx, atFit), atFit, scaleBar });
    };

    // Always pick the resolution-matched level (no coarsest-overview shortcut at fit),
    // so a huge image is sharp at the fit view instead of blurry until the user zooms.
    const chooseLevel = (): TileLevel =>
      chooseTileLevel(tileLevels, viewportSize.width, camera.right - camera.left, worldWidth, fullWidth);

    const clearTiles = (keysToKeep: Set<TileKey>) => {
      tileMeshes.forEach((mesh, key) => {
        if (keysToKeep.has(key)) {
          return;
        }
        group.remove(mesh);
        mesh.geometry.dispose();
        const material = materialCache.get(key);
        if (material) {
          material.dispose();
          materialCache.delete(key);
        }
        tileMeshes.delete(key);
      });
    };

    const ensureTile = (level: TileLevel, tileX: number, tileY: number, keepKeys: Set<TileKey>) => {
      const key: TileKey = `${level.level}:${tileX}:${tileY}`;
      keepKeys.add(key);
      if (tileMeshes.has(key) || inflight.has(key)) {
        return;
      }
      const tileSize = Math.max(64, viewerInfo.viewer.tile_scheme.tile_size || 256);
      const levelPixelWidth = Number(level.width);
      const levelPixelHeight = Number(level.height);
      const startX = tileX * tileSize;
      const startY = tileY * tileSize;
      const pixelWidth = Math.min(tileSize, levelPixelWidth - startX);
      const pixelHeight = Math.min(tileSize, levelPixelHeight - startY);
      if (pixelWidth <= 0 || pixelHeight <= 0) {
        return;
      }
      const worldTileWidth = (pixelWidth / levelPixelWidth) * worldWidth;
      const worldTileHeight = (pixelHeight / levelPixelHeight) * worldHeight;
      const centerX = -worldWidth / 2 + ((startX + pixelWidth / 2) / levelPixelWidth) * worldWidth;
      const centerY = worldHeight / 2 - ((startY + pixelHeight / 2) / levelPixelHeight) * worldHeight;

      const geometry = new THREE.PlaneGeometry(worldTileWidth, worldTileHeight);
      const material = new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, depthWrite: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(centerX, centerY, 0);
      mesh.visible = false;
      group.add(mesh);
      tileMeshes.set(key, mesh);
      materialCache.set(key, material);

      const reveal = (texture: THREE.Texture) => {
        material.map = texture;
        material.needsUpdate = true;
        mesh.visible = true;
        render();
      };
      const cached = textureCache.get(key);
      if (cached) {
        reveal(cached);
        return;
      }
      inflight.add(key);
      const url = apiClient.uploadTileUrl(fileId, {
        axis,
        level: level.level,
        tileX,
        tileY,
        size: tileSize,
        z: zIndex,
        t: tIndex,
        channels,
        channelColors: effectiveChannelColors,
        cacheKey,
      });
      // Permanently drop a tile only after retries are exhausted (a genuinely
      // unservable tile), removing its scene resources.
      const dropTile = () => {
        inflight.delete(key);
        group.remove(mesh);
        mesh.geometry.dispose();
        material.dispose();
        materialCache.delete(key);
        tileMeshes.delete(key);
        render();
      };
      // Self-heal transient tile failures (network blip, momentary engine load):
      // retry with exponential backoff before giving up, so deep zoom never shows a
      // permanent blank gap from a recoverable error. The tile stays "inflight"
      // across retries so updateTiles never starts a duplicate load for it.
      const attemptLoad = (attempt: number) => {
        loader.load(
          url,
          (texture) => {
            inflight.delete(key);
            if (disposed) {
              texture.dispose();
              return;
            }
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.generateMipmaps = false;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            textureCache.set(key, texture);
            reveal(texture);
          },
          undefined,
          () => {
            if (disposed) {
              inflight.delete(key);
              return;
            }
            if (attempt < MAX_TILE_LOAD_RETRIES) {
              const backoffMs = 300 * 2 ** attempt; // 300ms, 600ms, 1200ms
              window.setTimeout(() => {
                // Retry only if the tile is still wanted and the view is alive;
                // otherwise release the inflight slot so it can reload later.
                if (!disposed && tileMeshes.get(key) === mesh) {
                  attemptLoad(attempt + 1);
                } else {
                  inflight.delete(key);
                }
              }, backoffMs);
              return;
            }
            dropTile();
            commitRenderError("Tile delivery failed after retries.");
          }
        );
      };
      attemptLoad(0);
    };

    const updateTiles = () => {
      const level = chooseLevel();
      container.dataset.viewerLevel = String(level.level);
      const minX = camera.left;
      const maxX = camera.right;
      const minY = camera.bottom;
      const maxY = camera.top;
      const leftPx = ((minX + worldWidth / 2) / worldWidth) * level.width;
      const rightPx = ((maxX + worldWidth / 2) / worldWidth) * level.width;
      const topPx = ((worldHeight / 2 - maxY) / worldHeight) * level.height;
      const bottomPx = ((worldHeight / 2 - minY) / worldHeight) * level.height;
      const tileSize = Math.max(64, viewerInfo.viewer.tile_scheme.tile_size || 256);
      const startTileX = Math.max(0, Math.floor(leftPx / tileSize) - 1);
      const endTileX = Math.min(level.columns - 1, Math.ceil(rightPx / tileSize));
      const startTileY = Math.max(0, Math.floor(topPx / tileSize) - 1);
      const endTileY = Math.min(level.rows - 1, Math.ceil(bottomPx / tileSize));
      const keepKeys = new Set<TileKey>();
      for (let tileY = startTileY; tileY <= endTileY; tileY += 1) {
        for (let tileX = startTileX; tileX <= endTileX; tileX += 1) {
          ensureTile(level, tileX, tileY, keepKeys);
        }
      }
      clearTiles(keepKeys);
      render();
    };

    // setView applies a new viewport (clamped), then re-renders camera + tiles + readout.
    const setView = (next: ImageViewport) => {
      viewport = clampImageViewport(next, imageSize, viewportSize, limits.minScale, limits.maxScale);
      applyCamera();
      updateTiles();
      reportView();
    };

    const fit = () => {
      limits = computeImageScaleLimits(imageSize, viewportSize);
      setView({ centerX: 0, centerY: 0, scale: limits.fitScale });
    };

    const zoomByAtCenter = (factor: number) => {
      setView(
        zoomImageViewportAtPoint(
          viewport,
          { x: viewportSize.width / 2, y: viewportSize.height / 2 },
          viewport.scale * factor,
          imageSize,
          viewportSize,
          limits.minScale,
          limits.maxScale
        )
      );
    };
    controllerRef.current = {
      fit,
      zoomByAtCenter,
      // Flush the latest committed tiles, then composite the GL frame + scale bar.
      // preserveDrawingBuffer (set above) makes the async toBlob in composeViewBlob valid.
      captureViewToBlob: () => {
        render();
        return composeViewBlob(
          container,
          renderer.domElement,
          [".viewer-scale-bar-line", ".viewer-scale-bar-label"],
          "#f5f2eb"
        );
      },
    };

    const resize = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      // First layout fits the whole image; later resizes preserve a fitted view but
      // keep an explicit zoom/pan the user set.
      const wasFit = !hasFitted || Math.abs(viewport.scale - limits.fitScale) / Math.max(1e-9, limits.fitScale) < 0.02;
      hasFitted = true;
      viewportSize = { width, height };
      limits = computeImageScaleLimits(imageSize, viewportSize);
      setView(wasFit ? { centerX: 0, centerY: 0, scale: limits.fitScale } : viewport);
    };

    // --- pointer drag → pan, two-finger pinch → zoom ------------------------
    const drag = { active: false, pointerId: -1, lastX: 0, lastY: 0 };
    // Touch has no wheel, so pinch is the only zoom gesture on a phone. Track every
    // active pointer; with two down, zoom incrementally by the distance ratio.
    const pointers = new Map<number, { x: number; y: number }>();
    const pinch = { dist: 0 };
    const readPinch = () => {
      const points = Array.from(pointers.values());
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
    const el = renderer.domElement;
    el.style.touchAction = "none";
    el.style.cursor = "grab";

    // Receding chrome: reveal on pointer activity / keyboard focus, fade after idle.
    const chromeFade = createChromeFadeController(container, {
      reducedMotion: prefersReducedMotionSafe(),
    });
    const onFocusIn = () => chromeFade.reveal();
    container.addEventListener("focusin", onFocusIn);
    chromeFade.reveal();

    const onPointerDown = (e: PointerEvent) => {
      chromeFade.reveal();
      const target = e.target as HTMLElement | null;
      if (target?.closest("[data-viewer-image-control]")) {
        return;
      }
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      el.setPointerCapture(e.pointerId);
      if (pointers.size >= 2) {
        // Second finger → switch from pan to pinch-zoom; seed the pinch distance.
        drag.active = false;
        el.style.cursor = "grab";
        const geometry = readPinch();
        pinch.dist = geometry ? geometry.dist : 0;
        return;
      }
      drag.active = true;
      drag.pointerId = e.pointerId;
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;
      el.style.cursor = "grabbing";
    };
    const onPointerMove = (e: PointerEvent) => {
      chromeFade.reveal();
      if (pointers.has(e.pointerId)) {
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      }
      // Two fingers down → pinch-zoom toward the midpoint, never pan.
      if (pointers.size >= 2) {
        const geometry = readPinch();
        if (geometry && pinch.dist > 0 && geometry.dist > 0) {
          const factor = geometry.dist / pinch.dist;
          if (Math.abs(factor - 1) > 1e-6) {
            const rect = el.getBoundingClientRect();
            setView(
              zoomImageViewportAtPoint(
                viewport,
                { x: geometry.midX - rect.left, y: geometry.midY - rect.top },
                viewport.scale * factor,
                imageSize,
                viewportSize,
                limits.minScale,
                limits.maxScale
              )
            );
          }
          pinch.dist = geometry.dist;
        }
        return;
      }
      if (!drag.active || e.pointerId !== drag.pointerId) {
        return;
      }
      const dx = e.clientX - drag.lastX;
      const dy = e.clientY - drag.lastY;
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;
      setView(panImageViewport(viewport, { x: dx, y: dy }, imageSize, viewportSize, limits.minScale, limits.maxScale));
    };
    const onPointerUp = (e: PointerEvent) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) {
        pinch.dist = 0;
      }
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
      if (e.pointerId !== drag.pointerId) {
        return;
      }
      drag.active = false;
      el.style.cursor = "grab";
    };

    // --- wheel → pinch/⌘ zooms toward cursor; two-finger scroll pans --------
    const onWheel = (e: WheelEvent) => {
      chromeFade.reveal();
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      if (classifyWheelGesture(e) === "pan") {
        setView(
          panImageViewport(viewport, { x: -e.deltaX, y: -e.deltaY }, imageSize, viewportSize, limits.minScale, limits.maxScale)
        );
        return;
      }
      const factor = wheelDeltaToZoomFactor(e.deltaY, e.deltaMode, e.ctrlKey, viewportSize.height || rect.height);
      if (Math.abs(factor - 1) < 1e-6) {
        return;
      }
      setView(
        zoomImageViewportAtPoint(
          viewport,
          { x: e.clientX - rect.left, y: e.clientY - rect.top },
          viewport.scale * factor,
          imageSize,
          viewportSize,
          limits.minScale,
          limits.maxScale
        )
      );
    };

    const onDoubleClick = (e: MouseEvent) => {
      e.preventDefault();
      fit();
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("dblclick", onDoubleClick);

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    resize(); // initial fit

    return () => {
      disposed = true;
      controllerRef.current = null;
      observer.disconnect();
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("dblclick", onDoubleClick);
      container.removeEventListener("focusin", onFocusIn);
      chromeFade.dispose();
      tileMeshes.forEach((mesh) => {
        group.remove(mesh);
        mesh.geometry.dispose();
      });
      materialCache.forEach((material) => material.dispose());
      textureCache.forEach((texture) => texture.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    apiClient,
    axis,
    cacheKey,
    channels,
    descriptor.pixel_size.width,
    descriptor.world_size.height,
    descriptor.world_size.width,
    fileId,
    effectiveChannelColors,
    physicalUnit,
    publishReadout,
    renderError,
    tileLevels,
    tIndex,
    viewerInfo.viewer.tile_scheme.tile_size,
    zIndex,
  ]);

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      controllerRef.current?.zoomByAtCenter(ZOOM_STEP);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      controllerRef.current?.zoomByAtCenter(1 / ZOOM_STEP);
    } else if (event.key === "0" || event.key === "r" || event.key === "R") {
      event.preventDefault();
      controllerRef.current?.fit();
    }
  };

  if (renderError) {
    const tileDeliveryFailed = renderError.startsWith("Tile delivery failed");
    return (
      <div
        className={className ?? "viewer-canvas-root"}
        data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
        data-viewer-surface="2d"
        data-viewer-renderer="fallback"
      >
        <div className="viewer-image-fallback" style={{ aspectRatio: `${Math.max(1e-6, descriptor.aspect_ratio)}` }}>
          <img src={fallbackImageUrl} alt={`${descriptor.label} fallback`} className="viewer-image-fallback-media" />
        </div>
        <p className="viewer-fallback-note">
          {tileDeliveryFailed
            ? "Tile delivery failed after retries. Showing the selector-aware static plane instead."
            : "WebGL unavailable. Showing static plane preview instead of tiled deep zoom."}
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className ?? "viewer-canvas-root"}
      data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
      data-viewer-surface="2d"
      data-viewer-zoom={readout.zoom}
      role="group"
      aria-label="Deep-zoom image viewer (drag to pan, scroll or pinch to zoom)"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {readout.scaleBar ? (
        <div className="viewer-scale-bar viewer-chrome-fade" aria-hidden="true">
          <div className="viewer-scale-bar-line" style={{ width: `${Math.round(readout.scaleBar.widthPx)}px` }} />
          <span className="viewer-scale-bar-label">{readout.scaleBar.label}</span>
        </div>
      ) : null}
      <ViewerZoomToolbar
        readout={readout.zoom}
        onZoomOut={() => controllerRef.current?.zoomByAtCenter(1 / ZOOM_STEP)}
        onZoomIn={() => controllerRef.current?.zoomByAtCenter(ZOOM_STEP)}
        onFit={() => controllerRef.current?.fit()}
      />
    </div>
  );
});
