import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { UploadViewerInfo } from "@/types";

import {
  loadScalarSliceBitmap,
  loadSliceBitmap,
  sliceBitmapToTexture,
  type ScalarSliceSource,
} from "./sliceImageCache";

export type PlanePoint = {
  row: number;
  col: number;
};

type PlaneMeasurement = {
  start: PlanePoint;
  end: PlanePoint;
};

export type PlaneCoordinateGrid = {
  width: number;
  height: number;
};

type SlicePlaneCanvasProps = {
  imageUrl: string;
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
  crosshair?: PlanePoint | null;
  measurement?: PlaneMeasurement | null;
  onSelectPoint?: (point: PlanePoint) => void;
  onMeasurePoint?: (point: PlanePoint) => void;
  measureMode?: boolean;
  /**
   * Pixel grid represented by the displayed texture. When this differs from the
   * source descriptor, overlays use delivery-texel centres and clicks return
   * delivery indices. The caller remains responsible for mapping those indices
   * back to canonical source coordinates.
   */
  coordinateGrid?: PlaneCoordinateGrid | null;
  /**
   * When provided, the slice is rendered from the in-memory volume (instant, no
   * network). Falls back to {@link imageUrl} when absent.
   */
  scalarSlice?: ScalarSliceSource | null;
};

type OverlayState = {
  crosshairX: number | null;
  crosshairY: number | null;
  measurement: {
    startX: number;
    startY: number;
    endX: number;
    endY: number;
    labelX: number;
    labelY: number;
  } | null;
};

const fitOrthographicCamera = (
  camera: THREE.OrthographicCamera,
  width: number,
  height: number,
  worldWidth: number,
  worldHeight: number
): void => {
  const viewportWidth = Math.max(1, width);
  const viewportHeight = Math.max(1, height);
  const viewportAspect = viewportWidth / viewportHeight;
  const imageAspect = worldWidth / Math.max(1e-9, worldHeight);

  let visibleWidth = worldWidth;
  let visibleHeight = worldHeight;
  if (viewportAspect > imageAspect) {
    visibleWidth = worldHeight * viewportAspect;
  } else {
    visibleHeight = worldWidth / viewportAspect;
  }

  camera.left = -visibleWidth / 2;
  camera.right = visibleWidth / 2;
  camera.top = visibleHeight / 2;
  camera.bottom = -visibleHeight / 2;
  camera.near = 0.01;
  camera.far = 100;
  camera.position.set(0, 0, 10);
  camera.zoom = 1;
  camera.updateProjectionMatrix();
};

const clampPlanePoint = (
  point: PlanePoint,
  descriptor: UploadViewerInfo["viewer"]["default_plane"],
  coordinateGrid?: PlaneCoordinateGrid | null
): PlanePoint => ({
  row: Math.max(
    0,
    Math.min(
      Math.round(point.row),
      Math.max(
        0,
        Math.floor(
          coordinateGrid?.height ?? descriptor.pixel_size.height
        ) - 1
      )
    )
  ),
  col: Math.max(
    0,
    Math.min(
      Math.round(point.col),
      Math.max(
        0,
        Math.floor(
          coordinateGrid?.width ?? descriptor.pixel_size.width
        ) - 1
      )
    )
  ),
});

export const planePointToTextureRatios = (
  point: PlanePoint,
  coordinateGrid: PlaneCoordinateGrid
): { x: number; y: number } => {
  const width = Math.max(1, Math.floor(coordinateGrid.width));
  const height = Math.max(1, Math.floor(coordinateGrid.height));
  return {
    x: (Math.max(0, Math.min(width - 1, Math.round(point.col))) + 0.5) / width,
    y: (Math.max(0, Math.min(height - 1, Math.round(point.row))) + 0.5) / height,
  };
};

export const textureRatiosToPlanePoint = (
  ratios: { x: number; y: number },
  coordinateGrid: PlaneCoordinateGrid
): PlanePoint => {
  const width = Math.max(1, Math.floor(coordinateGrid.width));
  const height = Math.max(1, Math.floor(coordinateGrid.height));
  return {
    row: Math.max(0, Math.min(height - 1, Math.floor(ratios.y * height))),
    col: Math.max(0, Math.min(width - 1, Math.floor(ratios.x * width))),
  };
};

const planePointToWorld = (
  point: PlanePoint,
  descriptor: UploadViewerInfo["viewer"]["default_plane"],
  coordinateGrid?: PlaneCoordinateGrid | null
): THREE.Vector3 => {
  const pixelWidth = Math.max(
    1,
    Math.floor(coordinateGrid?.width ?? descriptor.pixel_size.width)
  );
  const pixelHeight = Math.max(
    1,
    Math.floor(coordinateGrid?.height ?? descriptor.pixel_size.height)
  );
  const worldWidth = Math.max(1, descriptor.world_size.width);
  const worldHeight = Math.max(1, descriptor.world_size.height);
  const clamped = clampPlanePoint(point, descriptor, coordinateGrid);
  const textureRatios = coordinateGrid
    ? planePointToTextureRatios(clamped, coordinateGrid)
    : null;
  const xRatio = coordinateGrid
    ? textureRatios!.x
    : pixelWidth <= 1
      ? 0.5
      : clamped.col / (pixelWidth - 1);
  const yRatio = coordinateGrid
    ? textureRatios!.y
    : pixelHeight <= 1
      ? 0.5
      : clamped.row / (pixelHeight - 1);
  return new THREE.Vector3(
    (xRatio - 0.5) * worldWidth,
    (0.5 - yRatio) * worldHeight,
    0
  );
};

const worldToScreen = (
  point: THREE.Vector3,
  camera: THREE.OrthographicCamera,
  width: number,
  height: number
): { x: number; y: number } => {
  const projected = point.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * width,
    y: (-projected.y * 0.5 + 0.5) * height,
  };
};

const eventToPlanePoint = (
  event: MouseEvent,
  element: HTMLElement,
  descriptor: UploadViewerInfo["viewer"]["default_plane"],
  coordinateGrid?: PlaneCoordinateGrid | null
): PlanePoint => {
  const rect = element.getBoundingClientRect();
  const xRatio = rect.width <= 0 ? 0.5 : (event.clientX - rect.left) / rect.width;
  const yRatio = rect.height <= 0 ? 0.5 : (event.clientY - rect.top) / rect.height;
  if (coordinateGrid) {
    return clampPlanePoint(
      textureRatiosToPlanePoint({ x: xRatio, y: yRatio }, coordinateGrid),
      descriptor,
      coordinateGrid
    );
  }
  return clampPlanePoint(
    {
      row: yRatio * Math.max(0, descriptor.pixel_size.height - 1),
      col: xRatio * Math.max(0, descriptor.pixel_size.width - 1),
    },
    descriptor
  );
};

export function SlicePlaneCanvas({
  imageUrl,
  descriptor,
  title,
  className,
  interactive = true,
  orientationLabels,
  crosshair,
  measurement,
  onSelectPoint,
  onMeasurePoint,
  measureMode = false,
  coordinateGrid,
  scalarSlice,
}: SlicePlaneCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [overlayState, setOverlayState] = useState<OverlayState>({
    crosshairX: null,
    crosshairY: null,
    measurement: null,
  });
  const worldSize = useMemo(
    () => ({
      width: Math.max(1, Number(descriptor.world_size.width) || 1),
      height: Math.max(1, Number(descriptor.world_size.height) || 1),
    }),
    [descriptor.world_size.height, descriptor.world_size.width]
  );
  const planeAspect = useMemo(() => {
    const descriptorAspect = Number(descriptor.aspect_ratio);
    const aspect = Number.isFinite(descriptorAspect) && descriptorAspect > 0
      ? descriptorAspect
      : worldSize.width / Math.max(1e-9, worldSize.height);
    return Math.max(0.125, Math.min(8, aspect));
  }, [descriptor.aspect_ratio, worldSize.height, worldSize.width]);
  const planeStyle = { "--viewer-plane-aspect": String(planeAspect) } as CSSProperties;

  // The renderer is built once (below) and reused as the user scrubs slices and
  // clicks. These per-render inputs are read through a ref so changing them never
  // tears down and rebuilds the WebGL context.
  const latestRef = useRef({
    crosshair,
    measurement,
    descriptor,
    onSelectPoint,
    onMeasurePoint,
    measureMode,
    interactive,
    coordinateGrid,
  });
  useEffect(() => {
    latestRef.current = {
      crosshair,
      measurement,
      descriptor,
      onSelectPoint,
      onMeasurePoint,
      measureMode,
      interactive,
      coordinateGrid,
    };
  });

  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const renderRef = useRef<(() => void) | null>(null);
  const activeTextureRef = useRef<THREE.Texture | null>(null);

  // Build the renderer once per plane geometry. Crucially this does NOT depend on
  // imageUrl/crosshair/measurement/callbacks, so moving through slices only swaps
  // a texture instead of recreating three WebGL contexts (one per MPR plane).
  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    const commitRenderError = (message: string) => {
      window.setTimeout(() => {
        if (!disposed) {
          setRenderError(message);
        }
      }, 0);
    };
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch (error) {
      commitRenderError(error instanceof Error ? error.message : "WebGL unavailable");
      return () => {
        disposed = true;
      };
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x05070a, 1);
    renderer.domElement.className = "viewer-webgl-canvas";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableRotate = false;
    controls.enablePan = interactive;
    controls.enableZoom = interactive;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 0.9;

    const geometry = new THREE.PlaneGeometry(worldSize.width, worldSize.height);
    const material = new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const raycaster = new THREE.Raycaster();
    cameraRef.current = camera;
    materialRef.current = material;

    const updateOverlay = () => {
      const {
        crosshair: activeCrosshair,
        measurement: activeMeasurement,
        descriptor: activeDescriptor,
        coordinateGrid: activeCoordinateGrid,
      } =
        latestRef.current;
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      const next: OverlayState = {
        crosshairX: null,
        crosshairY: null,
        measurement: null,
      };
      if (activeCrosshair) {
        const screen = worldToScreen(
          planePointToWorld(
            activeCrosshair,
            activeDescriptor,
            activeCoordinateGrid
          ),
          camera,
          width,
          height
        );
        next.crosshairX = screen.x;
        next.crosshairY = screen.y;
      }
      if (activeMeasurement) {
        const start = worldToScreen(
          planePointToWorld(
            activeMeasurement.start,
            activeDescriptor,
            activeCoordinateGrid
          ),
          camera,
          width,
          height
        );
        const end = worldToScreen(
          planePointToWorld(
            activeMeasurement.end,
            activeDescriptor,
            activeCoordinateGrid
          ),
          camera,
          width,
          height
        );
        next.measurement = {
          startX: start.x,
          startY: start.y,
          endX: end.x,
          endY: end.y,
          labelX: (start.x + end.x) / 2,
          labelY: (start.y + end.y) / 2,
        };
      }
      setOverlayState(next);
    };

    const render = () => {
      renderer.render(scene, camera);
      updateOverlay();
    };
    renderRef.current = render;

    const resize = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      fitOrthographicCamera(camera, width, height, worldSize.width, worldSize.height);
      controls.target.set(0, 0, 0);
      controls.update();
      render();
    };

    const handleClick = (event: MouseEvent) => {
      const {
        interactive: activeInteractive,
        descriptor: activeDescriptor,
        onSelectPoint: selectPoint,
        onMeasurePoint: measurePoint,
        measureMode: activeMeasureMode,
        coordinateGrid: activeCoordinateGrid,
      } = latestRef.current;
      if (!activeInteractive) {
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1,
        -(((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1)
      );
      raycaster.setFromCamera(mouse, camera);
      const hit = raycaster.intersectObject(mesh, false)[0];
      if (!hit?.point) {
        return;
      }
      const worldX = hit.point.x;
      const worldY = hit.point.y;
      const xRatio =
        worldX / Math.max(1e-6, activeDescriptor.world_size.width) + 0.5;
      const yRatio =
        0.5 - worldY / Math.max(1e-6, activeDescriptor.world_size.height);
      const planePoint = clampPlanePoint(
        activeCoordinateGrid
          ? {
              row: Math.floor(
                yRatio * Math.max(1, activeCoordinateGrid.height)
              ),
              col: Math.floor(
                xRatio * Math.max(1, activeCoordinateGrid.width)
              ),
            }
          : {
              row:
                yRatio *
                Math.max(0, activeDescriptor.pixel_size.height - 1),
              col:
                xRatio *
                Math.max(0, activeDescriptor.pixel_size.width - 1),
            },
        activeDescriptor,
        activeCoordinateGrid
      );
      selectPoint?.(planePoint);
      if (activeMeasureMode) {
        measurePoint?.(planePoint);
      }
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    controls.addEventListener("change", render);
    renderer.domElement.addEventListener("click", handleClick);

    resize();

    return () => {
      disposed = true;
      controls.removeEventListener("change", render);
      renderer.domElement.removeEventListener("click", handleClick);
      observer.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      activeTextureRef.current?.dispose();
      activeTextureRef.current = null;
      renderer.dispose();
      renderer.domElement.remove();
      cameraRef.current = null;
      materialRef.current = null;
      renderRef.current = null;
    };
  }, [interactive, worldSize.width, worldSize.height]);

  // Swap the slice texture as imageUrl changes — served from the shared cache, so
  // revisited / prefetched slices appear with no network round-trip.
  useEffect(() => {
    const material = materialRef.current;
    if (!material || (!scalarSlice && !imageUrl)) {
      return;
    }
    let cancelled = false;
    const loadPromise = scalarSlice ? loadScalarSliceBitmap(scalarSlice) : loadSliceBitmap(imageUrl);
    loadPromise
      .then((bitmap) => {
        if (cancelled || materialRef.current !== material) {
          return;
        }
        const texture = sliceBitmapToTexture(
          bitmap,
          scalarSlice?.payload.sampling === "nearest" ? "nearest" : "linear"
        );
        const previous = activeTextureRef.current;
        activeTextureRef.current = texture;
        material.map = texture;
        material.needsUpdate = true;
        previous?.dispose();
        renderRef.current?.();
      })
      .catch(() => {
        renderRef.current?.();
      });
    return () => {
      cancelled = true;
    };
  }, [imageUrl, scalarSlice]);

  // Reposition crosshair/measurement overlays when they move, without rebuilding
  // the scene.
  useEffect(() => {
    renderRef.current?.();
  }, [
    coordinateGrid?.height,
    coordinateGrid?.width,
    crosshair,
    measurement,
  ]);

  const renderOverlay = () => (
    <>
      {orientationLabels ? (
        <div className="viewer-orientation-overlay" aria-hidden="true">
          <span className="viewer-orientation-label viewer-orientation-label-top">{orientationLabels.top}</span>
          <span className="viewer-orientation-label viewer-orientation-label-bottom">{orientationLabels.bottom}</span>
          <span className="viewer-orientation-label viewer-orientation-label-left">{orientationLabels.left}</span>
          <span className="viewer-orientation-label viewer-orientation-label-right">{orientationLabels.right}</span>
        </div>
      ) : null}
      {overlayState.crosshairX != null && overlayState.crosshairY != null ? (
        <div className="viewer-crosshair-overlay" aria-hidden="true">
          <span
            className="viewer-crosshair-line viewer-crosshair-line-vertical"
            style={{ left: `${overlayState.crosshairX}px` }}
          />
          <span
            className="viewer-crosshair-line viewer-crosshair-line-horizontal"
            style={{ top: `${overlayState.crosshairY}px` }}
          />
        </div>
      ) : null}
      {overlayState.measurement ? (
        <svg className="viewer-measurement-overlay" aria-hidden="true">
          <line
            x1={overlayState.measurement.startX}
            y1={overlayState.measurement.startY}
            x2={overlayState.measurement.endX}
            y2={overlayState.measurement.endY}
          />
          <circle cx={overlayState.measurement.startX} cy={overlayState.measurement.startY} r="4" />
          <circle cx={overlayState.measurement.endX} cy={overlayState.measurement.endY} r="4" />
        </svg>
      ) : null}
    </>
  );

  if (renderError) {
    return (
      <div
        className={className ?? "viewer-canvas-root"}
        style={planeStyle}
        data-viewer-title={title}
        data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
        data-viewer-renderer="fallback-slice-plane"
        data-viewer-world-width={worldSize.width.toFixed(4)}
        data-viewer-world-height={worldSize.height.toFixed(4)}
        data-viewer-pixel-width={String(descriptor.pixel_size.width)}
        data-viewer-pixel-height={String(descriptor.pixel_size.height)}
        data-viewer-coordinate-grid-width={coordinateGrid?.width}
        data-viewer-coordinate-grid-height={coordinateGrid?.height}
        data-crosshair-row={crosshair ? String(Math.round(crosshair.row)) : undefined}
        data-crosshair-col={crosshair ? String(Math.round(crosshair.col)) : undefined}
      >
        <div
          className="viewer-image-fallback"
          style={{ aspectRatio: `${Math.max(1e-6, descriptor.aspect_ratio)}` }}
          onClick={(event) => {
            const point = eventToPlanePoint(
              event.nativeEvent,
              event.currentTarget,
              descriptor,
              coordinateGrid
            );
            onSelectPoint?.(point);
            if (measureMode) {
              onMeasurePoint?.(point);
            }
          }}
        >
          <img src={imageUrl} alt={title} className="viewer-image-fallback-media" />
          {renderOverlay()}
        </div>
        <p className="viewer-fallback-note">WebGL unavailable. Showing static slice preview.</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className ?? "viewer-canvas-root"}
      style={planeStyle}
      data-viewer-title={title}
      data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
      data-viewer-renderer="threejs-slice-plane"
      data-viewer-world-width={worldSize.width.toFixed(4)}
      data-viewer-world-height={worldSize.height.toFixed(4)}
      data-viewer-pixel-width={String(descriptor.pixel_size.width)}
      data-viewer-pixel-height={String(descriptor.pixel_size.height)}
      data-viewer-coordinate-grid-width={coordinateGrid?.width}
      data-viewer-coordinate-grid-height={coordinateGrid?.height}
      data-crosshair-row={crosshair ? String(Math.round(crosshair.row)) : undefined}
      data-crosshair-col={crosshair ? String(Math.round(crosshair.col)) : undefined}
    >
      {renderOverlay()}
    </div>
  );
}
