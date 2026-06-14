import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { Maximize2 } from "lucide-react";
import * as THREE from "three";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ApiClient } from "@/lib/api";
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

type DeepZoomCanvasProps = {
  apiClient: ApiClient;
  fileId: string;
  viewerInfo: UploadViewerInfo;
  axis?: "z" | "y" | "x";
  zIndex: number;
  tIndex: number;
  className?: string;
};

type TileKey = `${number}:${number}:${number}`;
type TileLevel = UploadViewerInfo["viewer"]["tile_scheme"]["levels"][number];

export const orderTileLevelsForRendering = (levels: TileLevel[]): TileLevel[] =>
  [...levels].sort((left, right) => {
    const downsampleDelta = right.downsample - left.downsample;
    if (downsampleDelta !== 0) {
      return downsampleDelta;
    }
    return left.width * left.height - right.width * right.height;
  });

// classifyWheelGesture maps a wheel event to pan vs zoom, the modern image-viewer
// convention (Figma/Maps/macOS): a trackpad PINCH and ctrl/⌘-scroll zoom; a
// trackpad two-finger SCROLL pans; a discrete mouse wheel zooms. Trackpad scroll is
// distinguished from a mouse wheel heuristically — it carries a horizontal component
// or fine/fractional pixel deltas, whereas a mouse wheel sends coarse vertical steps.
// Pure + exported for tests.
export const classifyWheelGesture = (event: {
  deltaX: number;
  deltaY: number;
  deltaMode: number;
  ctrlKey: boolean;
  metaKey: boolean;
}): "zoom" | "pan" => {
  if (event.ctrlKey || event.metaKey) {
    return "zoom"; // pinch (macOS fires ctrlKey) or ⌘/ctrl-scroll
  }
  if (event.deltaMode !== 0) {
    return "zoom"; // line/page deltas come from a classic mouse wheel
  }
  if (event.deltaX !== 0) {
    return "pan"; // horizontal component → trackpad two-finger scroll
  }
  // Pure-vertical pixel scroll: fine/fractional steps are a trackpad pan; coarse
  // integer steps (typical mouse wheel notch ~100px) are a zoom.
  if (!Number.isInteger(event.deltaY) || Math.abs(event.deltaY) < 50) {
    return "pan";
  }
  return "zoom";
};

// formatZoomReadout shows zoom as % of native (1 source px == 1 screen px == 100%),
// the convention scientists expect (Photoshop/Preview). "Fit" when at the fit scale.
const formatZoomReadout = (screenPxPerSourcePx: number, atFit: boolean): string => {
  if (atFit) {
    return "Fit";
  }
  const pct = screenPxPerSourcePx * 100;
  if (pct >= 1000) return `${Math.round(pct / 100) * 100}%`;
  if (pct >= 100) return `${Math.round(pct)}%`;
  if (pct >= 10) return `${Math.round(pct)}%`;
  return `${pct.toFixed(1)}%`;
};

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

export function DeepZoomCanvas({
  apiClient,
  fileId,
  viewerInfo,
  axis = "z",
  zIndex,
  tIndex,
  className,
}: DeepZoomCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [readout, setReadout] = useState<ViewReadout>({ zoom: "Fit", atFit: true, scaleBar: null });
  // Imperative handles set by the WebGL effect so the toolbar/keyboard can drive it.
  const controllerRef = useRef<{ fit: () => void; zoomByAtCenter: (factor: number) => void } | null>(null);

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
  const fallbackImageUrl = useMemo(
    () => apiClient.uploadSliceUrl(fileId, { axis, z: zIndex, t: tIndex }),
    [apiClient, axis, fileId, tIndex, zIndex]
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
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
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
    // world units per source pixel (drives the scale bar + native-zoom readout).
    const unitsPerSourcePx = worldWidth / fullWidth;

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
      const screenPxPerSourcePx = viewport.scale * unitsPerSourcePx; // scale is px/world-unit
      const atFit = Math.abs(viewport.scale - limits.fitScale) / Math.max(1e-9, limits.fitScale) < 0.02;
      // scale bar: screen px per one physical unit == viewport.scale (world unit == physical unit)
      const scaleBar = niceScaleBar(viewport.scale, physicalUnit);
      publishReadout({ zoom: formatZoomReadout(screenPxPerSourcePx, atFit), atFit, scaleBar });
    };

    const chooseLevel = (): TileLevel => {
      const visibleWorldWidth = camera.right - camera.left;
      const visibleWorldHeight = camera.top - camera.bottom;
      if (
        tileLevels.length > 0 &&
        visibleWorldWidth >= worldWidth * 0.98 &&
        visibleWorldHeight >= worldHeight * 0.98
      ) {
        return tileLevels[0];
      }
      const screenPixelsPerWorldUnit = Math.max(1, viewportSize.width) / Math.max(1e-9, visibleWorldWidth);
      const sourcePixelsPerWorldUnit = fullWidth / worldWidth;
      const screenPixelsPerSourcePixel = screenPixelsPerWorldUnit / Math.max(1e-9, sourcePixelsPerWorldUnit);
      let bestLevel = tileLevels[tileLevels.length - 1];
      let bestScore = Number.POSITIVE_INFINITY;
      tileLevels.forEach((level) => {
        const screenPixelsPerLevelPixel = screenPixelsPerSourcePixel * level.downsample;
        const score = Math.abs(Math.log2(Math.max(screenPixelsPerLevelPixel, 1e-6)));
        if (score < bestScore) {
          bestScore = score;
          bestLevel = level;
        }
      });
      return bestLevel;
    };

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
      });
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
          inflight.delete(key);
          group.remove(mesh);
          mesh.geometry.dispose();
          material.dispose();
          materialCache.delete(key);
          tileMeshes.delete(key);
          render();
        }
      );
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
    controllerRef.current = { fit, zoomByAtCenter };

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

    // --- pointer drag → pan -------------------------------------------------
    const drag = { active: false, pointerId: -1, lastX: 0, lastY: 0 };
    const el = renderer.domElement;
    el.style.touchAction = "none";
    el.style.cursor = "grab";

    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest("[data-viewer-image-control]")) {
        return;
      }
      drag.active = true;
      drag.pointerId = e.pointerId;
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    };
    const onPointerMove = (e: PointerEvent) => {
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
      if (e.pointerId !== drag.pointerId) {
        return;
      }
      drag.active = false;
      el.style.cursor = "grab";
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
    };

    // --- wheel → pinch/⌘ zooms toward cursor; two-finger scroll pans --------
    const onWheel = (e: WheelEvent) => {
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
    descriptor.pixel_size.width,
    descriptor.world_size.height,
    descriptor.world_size.width,
    fileId,
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
      controllerRef.current?.zoomByAtCenter(1.22);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      controllerRef.current?.zoomByAtCenter(1 / 1.22);
    } else if (event.key === "0") {
      event.preventDefault();
      controllerRef.current?.fit();
    }
  };

  if (renderError) {
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
        <p className="viewer-fallback-note">WebGL unavailable. Showing static plane preview instead of tiled deep zoom.</p>
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
        <div className="viewer-scale-bar" aria-hidden="true">
          <div className="viewer-scale-bar-line" style={{ width: `${Math.round(readout.scaleBar.widthPx)}px` }} />
          <span className="viewer-scale-bar-label">{readout.scaleBar.label}</span>
        </div>
      ) : null}
      <div className="viewer-direct-image-toolbar" data-viewer-image-control="true">
        <Badge variant="secondary" className="viewer-direct-image-readout" data-testid="deepzoom-zoom-readout">
          {readout.zoom}
        </Badge>
        <Button
          type="button"
          variant="outline"
          size="icon-xs"
          aria-label="Fit image to view"
          onClick={() => controllerRef.current?.fit()}
        >
          <Maximize2 data-icon="inline-start" />
        </Button>
      </div>
    </div>
  );
}
