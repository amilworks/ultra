import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { UploadViewerInfo } from "@/types";

type PlanePoint = {
  row: number;
  col: number;
};

type PlaneMeasurement = {
  start: PlanePoint;
  end: PlanePoint;
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
  descriptor: UploadViewerInfo["viewer"]["default_plane"]
): PlanePoint => ({
  row: Math.max(0, Math.min(Math.round(point.row), Math.max(0, descriptor.pixel_size.height - 1))),
  col: Math.max(0, Math.min(Math.round(point.col), Math.max(0, descriptor.pixel_size.width - 1))),
});

const planePointToWorld = (
  point: PlanePoint,
  descriptor: UploadViewerInfo["viewer"]["default_plane"]
): THREE.Vector3 => {
  const pixelWidth = Math.max(1, descriptor.pixel_size.width);
  const pixelHeight = Math.max(1, descriptor.pixel_size.height);
  const worldWidth = Math.max(1, descriptor.world_size.width);
  const worldHeight = Math.max(1, descriptor.world_size.height);
  const clamped = clampPlanePoint(point, descriptor);
  const xRatio = pixelWidth <= 1 ? 0.5 : clamped.col / (pixelWidth - 1);
  const yRatio = pixelHeight <= 1 ? 0.5 : clamped.row / (pixelHeight - 1);
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
  descriptor: UploadViewerInfo["viewer"]["default_plane"]
): PlanePoint => {
  const rect = element.getBoundingClientRect();
  const xRatio = rect.width <= 0 ? 0.5 : (event.clientX - rect.left) / rect.width;
  const yRatio = rect.height <= 0 ? 0.5 : (event.clientY - rect.top) / rect.height;
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

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      if (renderError) {
        setRenderError(null);
      }
    } catch (error) {
      setRenderError(error instanceof Error ? error.message : "WebGL unavailable");
      return;
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xf5f2eb, 1);
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
    const loader = new THREE.TextureLoader();
    let activeTexture: THREE.Texture | null = null;

    const updateOverlay = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      const next: OverlayState = {
        crosshairX: null,
        crosshairY: null,
        measurement: null,
      };
      if (crosshair) {
        const screen = worldToScreen(planePointToWorld(crosshair, descriptor), camera, width, height);
        next.crosshairX = screen.x;
        next.crosshairY = screen.y;
      }
      if (measurement) {
        const start = worldToScreen(planePointToWorld(measurement.start, descriptor), camera, width, height);
        const end = worldToScreen(planePointToWorld(measurement.end, descriptor), camera, width, height);
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
      if (!interactive) {
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
      const planePoint = clampPlanePoint(
        {
          row:
            (0.5 - worldY / Math.max(1e-6, descriptor.world_size.height)) *
            Math.max(0, descriptor.pixel_size.height - 1),
          col:
            (worldX / Math.max(1e-6, descriptor.world_size.width) + 0.5) *
            Math.max(0, descriptor.pixel_size.width - 1),
        },
        descriptor
      );
      onSelectPoint?.(planePoint);
      if (measureMode) {
        onMeasurePoint?.(planePoint);
      }
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    controls.addEventListener("change", render);
    renderer.domElement.addEventListener("click", handleClick);

    loader.load(
      imageUrl,
      (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        if (activeTexture) {
          activeTexture.dispose();
        }
        activeTexture = texture;
        material.map = texture;
        material.needsUpdate = true;
        resize();
      },
      undefined,
      () => {
        resize();
      }
    );

    resize();

    return () => {
      controls.removeEventListener("change", render);
      renderer.domElement.removeEventListener("click", handleClick);
      observer.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      if (activeTexture) {
        activeTexture.dispose();
      }
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    crosshair,
    descriptor,
    imageUrl,
    interactive,
    measureMode,
    measurement,
    onMeasurePoint,
    onSelectPoint,
    renderError,
    worldSize.height,
    worldSize.width,
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
        data-viewer-title={title}
        data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
        data-viewer-renderer="fallback"
        data-crosshair-row={crosshair ? String(Math.round(crosshair.row)) : undefined}
        data-crosshair-col={crosshair ? String(Math.round(crosshair.col)) : undefined}
      >
        <div
          className="viewer-image-fallback"
          style={{ aspectRatio: `${Math.max(1e-6, descriptor.aspect_ratio)}` }}
          onClick={(event) => {
            const point = eventToPlanePoint(event.nativeEvent, event.currentTarget, descriptor);
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
      data-viewer-title={title}
      data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
      data-crosshair-row={crosshair ? String(Math.round(crosshair.row)) : undefined}
      data-crosshair-col={crosshair ? String(Math.round(crosshair.col)) : undefined}
    >
      {renderOverlay()}
    </div>
  );
}
