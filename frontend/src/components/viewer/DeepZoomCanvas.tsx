import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { ApiClient } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

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

// How far past native (1 screen px per source px) the user may zoom for pixel
// inspection. The zoom cap is derived per-image from this so a 95k-px pyramid can
// actually reach native detail while a small image isn't over-zoomed absurdly.
const NATIVE_OVERZOOM = 2;

// Derive the OrbitControls zoom cap from the fit so the user can reach (and slightly
// exceed) native 1:1 regardless of image size. At fit (zoom 1) the whole image fills
// the viewport, so a 95174px-wide image shows ~0.01 screen px per source px and a
// fixed cap (was 64) never reaches detail (needs ~95x). Returns at least 8 so small
// images still zoom in a little. Pure + exported for tests.
export const computeAdaptiveMaxZoom = (
  viewportWidth: number,
  visibleWidthAtFit: number,
  worldWidth: number,
  fullWidth: number,
  overzoom: number = NATIVE_OVERZOOM
): number => {
  const screenPxPerSourcePxAtFit =
    (Math.max(1, viewportWidth) * Math.max(1e-9, worldWidth)) /
    (Math.max(1e-9, visibleWidthAtFit) * Math.max(1e-9, fullWidth));
  return Math.max(8, overzoom / Math.max(screenPxPerSourcePxAtFit, 1e-9));
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
  const descriptor = useMemo(
    () => viewerInfo.viewer.planes[axis] ?? viewerInfo.viewer.default_plane,
    [axis, viewerInfo.viewer.default_plane, viewerInfo.viewer.planes]
  );
  const tileLevels = useMemo(
    () => orderTileLevelsForRendering(viewerInfo.viewer.tile_scheme.levels),
    [viewerInfo.viewer.tile_scheme.levels]
  );
  const fallbackImageUrl = useMemo(
    () =>
      apiClient.uploadSliceUrl(fileId, {
        axis,
        z: zIndex,
        t: tIndex,
      }),
    [apiClient, axis, fileId, tIndex, zIndex]
  );

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
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
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
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableRotate = false;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 0.95;
    controls.minZoom = 0.8;
    controls.maxZoom = 64;

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

    const render = () => {
      renderer.render(scene, camera);
    };

    const chooseLevel = (): (typeof tileLevels)[number] => {
      const viewportWidth = Math.max(1, container.clientWidth || 1);
      const visibleWorldWidth = (camera.right - camera.left) / Math.max(camera.zoom, 1e-6);
      const visibleWorldHeight = (camera.top - camera.bottom) / Math.max(camera.zoom, 1e-6);
      if (
        tileLevels.length > 0 &&
        visibleWorldWidth >= worldWidth * 0.98 &&
        visibleWorldHeight >= worldHeight * 0.98
      ) {
        return tileLevels[0];
      }
      const screenPixelsPerWorldUnit = viewportWidth / Math.max(1e-9, visibleWorldWidth);
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

    const ensureTile = (
      level: (typeof tileLevels)[number],
      tileX: number,
      tileY: number,
      keepKeys: Set<TileKey>
    ) => {
      const key: TileKey = `${level.level}:${tileX}:${tileY}`;
      keepKeys.add(key);
      if (tileMeshes.has(key)) {
        return;
      }
      if (inflight.has(key)) {
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
      // transparent: honor the tile's alpha channel — geospatial orthomosaics
      // (and any RGBA raster) encode no-data regions as alpha=0, which must blend
      // to the page background, not composite as opaque black. depthWrite:false
      // avoids z-fighting between adjacent transparent tile quads.
      const material = new THREE.MeshBasicMaterial({
        toneMapped: false,
        transparent: true,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(centerX, centerY, 0);
      // Hidden until its texture resolves, so a not-yet-loaded (or failed) tile
      // shows the clear background rather than a flat white/colored placeholder quad.
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

      const cachedTexture = textureCache.get(key);
      if (cachedTexture) {
        reveal(cachedTexture);
        return;
      }

      inflight.add(key);
      const url = apiClient.uploadTileUrl(fileId, {
        axis,
        level: level.level,
        tileX,
        tileY,
        // Retile at the SAME size as the grid math (tile_scheme.tile_size). The served
        // file may be a derived pyramid with a different native tile size (e.g. 512 vs
        // the source's 256); without this the engine retiles at its default and the
        // canvas's (col,row) lands on the wrong region — half the tiles 500.
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
          // Drop the failed placeholder entirely so it is re-fetched on the next
          // pan/zoom instead of leaving a permanent untextured (white) quad.
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
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      camera.updateProjectionMatrix();
      const level = chooseLevel();
      container.dataset.viewerLevel = String(level.level);

      const visibleWorldWidth = (camera.right - camera.left) / Math.max(camera.zoom, 1e-6);
      const visibleWorldHeight = (camera.top - camera.bottom) / Math.max(camera.zoom, 1e-6);
      const minX = camera.position.x - visibleWorldWidth / 2;
      const maxX = camera.position.x + visibleWorldWidth / 2;
      const minY = camera.position.y - visibleWorldHeight / 2;
      const maxY = camera.position.y + visibleWorldHeight / 2;

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

    const resize = () => {
      const width = Math.max(1, container.clientWidth || 1);
      const height = Math.max(1, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      fitOrthographicCamera(camera, width, height, worldWidth, worldHeight);
      // Derive the zoom cap from the fit so the user can reach native resolution.
      // At fit (zoom 1) the whole image fills the viewport, so a 95174px-wide image
      // is ~0.01 screen px per source px — a fixed maxZoom (was 64) never reaches
      // 1:1 (needs ~95x). Cap at NATIVE_OVERZOOM past native, scaled per image.
      const visibleWidthAtFit = camera.right - camera.left;
      controls.maxZoom = computeAdaptiveMaxZoom(width, visibleWidthAtFit, worldWidth, fullWidth);
      controls.target.set(0, 0, 0);
      controls.update();
      updateTiles();
    };

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    controls.addEventListener("change", updateTiles);
    resize();

    return () => {
      disposed = true;
      controls.removeEventListener("change", updateTiles);
      observer.disconnect();
      controls.dispose();
      tileMeshes.forEach((mesh) => {
        group.remove(mesh);
        mesh.geometry.dispose();
      });
      materialCache.forEach((material) => material.dispose());
      textureCache.forEach((texture) => texture.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [apiClient, axis, descriptor.pixel_size.height, descriptor.pixel_size.width, descriptor.world_size.height, descriptor.world_size.width, fileId, renderError, tileLevels, tIndex, viewerInfo.viewer.tile_scheme.tile_size, zIndex]);

  if (renderError) {
    return (
      <div
        className={className ?? "viewer-canvas-root"}
        data-viewer-aspect={descriptor.aspect_ratio.toFixed(4)}
        data-viewer-surface="2d"
        data-viewer-renderer="fallback"
      >
        <div
          className="viewer-image-fallback"
          style={{ aspectRatio: `${Math.max(1e-6, descriptor.aspect_ratio)}` }}
        >
          <img
            src={fallbackImageUrl}
            alt={`${descriptor.label} fallback`}
            className="viewer-image-fallback-media"
          />
        </div>
        <p className="viewer-fallback-note">
          WebGL unavailable. Showing static plane preview instead of tiled deep zoom.
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
    />
  );
}
