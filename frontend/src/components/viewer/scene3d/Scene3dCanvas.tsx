import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import type { ApiClient } from "@/lib/api";
import type { Scene3dLayer, Scene3dManifest } from "@/types";

import { resolveVolumeScaleBar } from "../volumeScaleBar";
import { describeDecimation, maxElementsFor, resolveScenePixelRatio } from "./sceneBudget";
import {
  UPC1_FLAG_ALPHA,
  parseChunkHeader,
  pointViews,
  selectTier,
  splatViews,
  type ChunkHeader,
} from "./sceneChunks";
import { srgbBytesToLinearFloat } from "./sceneColor";
import { applyMat3, cameraBasisFromColmap, cameraCentreFromColmap } from "./sceneFrame";
import { projectionMatrixFor, type ColmapCamera } from "./sceneIntrinsics";

export type Scene3dSpecies = "points" | "splats" | "cameras";

export type Scene3dLayerVisibility = Record<Scene3dSpecies, boolean>;

type Props = {
  fileId: string;
  manifest: Scene3dManifest;
  apiClient: ApiClient;
  visibility: Scene3dLayerVisibility;
  /** Screen-space point sprite size, in pixels. */
  pointSize: number;
  /** Bumped by the shell's "Reset view" button to re-frame the camera. */
  resetToken: number;
};

/** What the readout reports per species: uploaded so far, and what the source holds. */
type LayerProgress = {
  type: Scene3dSpecies;
  loaded: number;
  total: number;
};

type ScaleBarState = {
  label: string;
  /** Bar width in CSS pixels, derived from the live camera — never a nominal guess. */
  barPx: number;
};

/**
 * A posed COLMAP camera. The frozen contract declares the cameras layer as
 * `encoding: "json"` but does not fix the JSON body, so this reader is deliberately
 * tolerant: it accepts either a bare array or `{cameras: [...]}`, and either a nested
 * `camera` object or the intrinsics inlined on the pose. `qvec` is COLMAP's
 * world-to-camera quaternion in **wxyz** order.
 */
type Scene3dCameraPose = {
  qvec: number[];
  tvec: number[];
  camera: ColmapCamera;
};

/** Frustum ray length as a fraction of the scene radius — long enough to read the pose. */
const FRUSTUM_SCALE = 0.06;

const UP_AXIS_VECTORS: Record<string, THREE.Vector3> = {
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

const SPECIES_LABEL: Record<Scene3dSpecies, string> = {
  points: "points",
  splats: "splats",
  cameras: "cameras",
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const numbers = (value: unknown): number[] =>
  Array.isArray(value) ? value.map((entry) => Number(entry)) : [];

// `navigator.deviceMemory` is not in the DOM lib and is absent on Safari; the budget
// treats absent as "assume the baseline", so an undefined read is the correct input.
const deviceMemoryGb = (): number | undefined => {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return typeof nav.deviceMemory === "number" ? nav.deviceMemory : undefined;
};

const isMobileDevice = (): boolean =>
  (navigator.maxTouchPoints ?? 0) > 1 && window.innerWidth <= 900;

const layerOfType = (manifest: Scene3dManifest, type: string): Scene3dLayer | undefined =>
  manifest.layers.find((layer) => layer.type === type);

/**
 * Scene centre and radius for the camera; a degenerate bbox still frames.
 *
 * Prefers `world.bbox_robust` (the derive's 1st..99th percentile box) over `world.bbox`.
 * Dense reconstructions routinely carry a few far-field outliers: on the measured COLMAP
 * corridor fixture the full bbox spans 3453 units while the middle 99% of its points
 * occupies 33 — 0.9% of the diagonal. Framing on the full bbox renders the actual scene
 * as a speck, which reads as "the viewer is broken" rather than "this file has outliers".
 */
const frameOf = (bbox: number[]): { centre: THREE.Vector3; radius: number } => {
  const box =
    Array.isArray(bbox) && bbox.length >= 6
      ? bbox.map((value) => (Number.isFinite(Number(value)) ? Number(value) : 0))
      : [-1, -1, -1, 1, 1, 1];
  const centre = new THREE.Vector3(
    (box[0] + box[3]) / 2,
    (box[1] + box[4]) / 2,
    (box[2] + box[5]) / 2
  );
  const span = Math.hypot(box[3] - box[0], box[4] - box[1], box[5] - box[2]) / 2;
  return { centre, radius: Number.isFinite(span) && span > 0 ? span : 1 };
};

const readCameraPoses = (payload: unknown): Scene3dCameraPose[] => {
  const rows = Array.isArray(payload)
    ? payload
    : isRecord(payload) && Array.isArray(payload.cameras)
      ? payload.cameras
      : [];
  const poses: Scene3dCameraPose[] = [];
  for (const row of rows) {
    if (!isRecord(row)) {
      continue;
    }
    const intrinsics = isRecord(row.camera) ? row.camera : row;
    const qvec = numbers(row.qvec ?? row.qvec_wxyz);
    const tvec = numbers(row.tvec);
    const width = Number(intrinsics.width);
    const height = Number(intrinsics.height);
    if (qvec.length !== 4 || tvec.length !== 3 || !(width > 0) || !(height > 0)) {
      continue;
    }
    poses.push({
      qvec,
      tvec,
      camera: {
        model: String(intrinsics.model ?? ""),
        width,
        height,
        params: numbers(intrinsics.params),
      },
    });
  }
  return poses;
};

/**
 * Frustum wireframe for one posed camera, in world coordinates.
 *
 * The corner rays come from the camera's own projection matrix — unprojecting the four
 * NDC corners of the near plane — so a principal-point offset shows up as the asymmetric
 * frustum it is. `THREE.PerspectiveCamera(fov, aspect)` is structurally symmetric and
 * cannot represent that, which is why it is not used here (contract §9).
 *
 * `near`/`far` are 1 and 2 only to give a well-formed matrix: the unprojected corners sit
 * at z = −1 in camera space and are then scaled to the display length, and ray direction
 * is invariant to that choice.
 */
const frustumSegments = (pose: Scene3dCameraPose, length: number): number[] => {
  const inverse = new THREE.Matrix4()
    .fromArray(projectionMatrixFor(pose.camera, 1, 2))
    .invert();
  const centre = cameraCentreFromColmap(pose.qvec, pose.tvec);
  // Columns of this RUB basis are the camera's right / up / backward axes in world
  // space. It is the ONE place the RDF→RUB flip is applied (contract §2).
  const basis = cameraBasisFromColmap(pose.qvec, pose.tvec);
  const apex = new THREE.Vector3(centre[0], centre[1], centre[2]);

  const corners = (
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ] as const
  ).map(([ndcX, ndcY]) => {
    const local = new THREE.Vector3(ndcX, ndcY, -1).applyMatrix4(inverse).multiplyScalar(length);
    const world = applyMat3(basis, [local.x, local.y, local.z]);
    return new THREE.Vector3(apex.x + world[0], apex.y + world[1], apex.z + world[2]);
  });

  const vertices: number[] = [];
  const push = (a: THREE.Vector3, b: THREE.Vector3) => {
    vertices.push(a.x, a.y, a.z, b.x, b.y, b.z);
  };
  for (let index = 0; index < 4; index += 1) {
    push(apex, corners[index]);
    push(corners[index], corners[(index + 1) % 4]);
  }
  return vertices;
};

/**
 * The WebGL surface. Points are `THREE.Points`, splats are Spark `SplatMesh`es fed
 * `ExtSplats` built straight from the wire arrays, cameras are `LineSegments`.
 *
 * Nothing here rotates a node that holds splats: spherical-harmonic coefficients live in
 * the asset's own frame, and rotating the node without a Wigner-D rotation of the bands
 * sign-flips every odd band — invisible in a screenshot, wrong while orbiting
 * (contract §2). Chunk origins are applied as translation only.
 */
export function Scene3dCanvas({
  fileId,
  manifest,
  apiClient,
  visibility,
  pointSize,
  resetToken,
}: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [gateMessage, setGateMessage] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [progress, setProgress] = useState<LayerProgress[]>([]);
  const [scaleBar, setScaleBar] = useState<ScaleBarState | null>(null);

  // Chunks keep arriving long after the user has touched the controls, so the loader
  // reads display state through a ref rather than a stale closure over the first render.
  const pointSizeRef = useRef(pointSize);

  // The scene graph stays inside the effect that owns it; the incremental effects below
  // reach it only through these setters, never by mutating objects held in a ref.
  const rigRef = useRef<{
    setVisibility: (next: Scene3dLayerVisibility) => void;
    setPointSize: (next: number) => void;
    reframe: () => void;
  } | null>(null);

  const unitLabel = manifest.world.units === "meters" ? "m" : "unit";

  useEffect(() => {
    const stage = stageRef.current;
    const host = hostRef.current;
    if (!stage || !host) {
      return;
    }

    let disposed = false;
    const abort = new AbortController();

    // Commit the gate out of the effect body, the same way SliceStackVolumeCanvas
    // commits its render errors: a synchronous setState here cascades a second render
    // before the context is even wired up.
    const commitGate = (message: string | null) => {
      window.setTimeout(() => {
        if (!disposed) {
          setGateMessage(message);
        }
      }, 0);
    };

    // WebGL2 gate, mirroring SliceStackVolumeCanvas: constructing the renderer is the
    // only reliable probe, and a WebGL1-only context cannot run the splat pass.
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    } catch (error) {
      commitGate(error instanceof Error ? error.message : "WebGL unavailable");
      return () => {
        disposed = true;
      };
    }
    if (!renderer.capabilities.isWebGL2) {
      renderer.dispose();
      commitGate("WebGL2 unavailable");
      return () => {
        disposed = true;
      };
    }
    commitGate(null);

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(resolveScenePixelRatio(window.devicePixelRatio || 1, isMobileDevice()));
    renderer.setClearColor(0x0b0d10, 1);
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const frame = frameOf(manifest.world.bbox_robust ?? manifest.world.bbox);
    // Depth planes fitted to the framed scene. A near/far ratio in the tens of thousands
    // (radius/1000 .. radius*40) exhausts depth precision on a kilometre-scale scan: every
    // fragment lands on NDC z = 1, points occlude each other, and the scene renders as a
    // few hundred lit pixels. Measured on the COLMAP corridor fixture.
    const fitDistance = frame.radius / Math.sin((50 * Math.PI) / 360);
    const camera = new THREE.PerspectiveCamera(
      50,
      1,
      Math.max(fitDistance - frame.radius * 1.25, frame.radius / 100),
      fitDistance + frame.radius * 4
    );
    // "Up" is a VIEW HINT (contract §2): it steers the controls and is never baked into
    // geometry. An unknown up axis keeps three's default rather than guessing one.
    camera.up.copy(UP_AXIS_VECTORS[manifest.world.up_axis] ?? new THREE.Vector3(0, 1, 0));

    const controls = new TrackballControls(camera, renderer.domElement);
    controls.staticMoving = true;
    controls.rotateSpeed = 4;
    controls.zoomSpeed = 1.4;
    controls.panSpeed = 0.8;

    const groups: Record<Scene3dSpecies, THREE.Group> = {
      points: new THREE.Group(),
      splats: new THREE.Group(),
      cameras: new THREE.Group(),
    };
    groups.points.visible = visibility.points;
    groups.splats.visible = visibility.splats;
    groups.cameras.visible = visibility.cameras;
    scene.add(groups.points, groups.splats, groups.cameras);

    const reframe = () => {
      const direction = new THREE.Vector3(1, 0.55, 1).normalize();
      camera.position.copy(frame.centre).addScaledVector(direction, frame.radius * 2.4);
      controls.target.copy(frame.centre);
      camera.lookAt(frame.centre);
      controls.update();
    };
    reframe();

    let lastScaleBarKey = "";
    const publishScaleBar = () => {
      const widthPx = Math.max(1, stage.clientWidth || 1);
      const distance = camera.position.distanceTo(controls.target);
      const worldHeight = 2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const worldWidth = worldHeight * (camera.aspect || 1);
      const bar = resolveVolumeScaleBar({ worldWidth, unit: unitLabel });
      if (!bar.visible || disposed) {
        return;
      }
      // Bar pixels come from the SAME world width the label states, so the drawn length
      // always means what it says.
      const barPx = Math.max(1, Math.round((bar.length / worldWidth) * widthPx));
      const key = `${bar.label}|${barPx}`;
      if (key === lastScaleBarKey) {
        return;
      }
      lastScaleBarKey = key;
      setScaleBar({ label: bar.label, barPx });
    };

    // On-demand rAF: a settled splat scene should not burn a GPU frame every 16 ms.
    let animationFrame = 0;
    const renderFrame = () => {
      animationFrame = 0;
      if (disposed) {
        return;
      }
      controls.update();
      renderer.render(scene, camera);
      publishScaleBar();
    };
    const requestRender = () => {
      if (disposed || animationFrame) {
        return;
      }
      animationFrame = window.requestAnimationFrame(renderFrame);
    };

    const resize = () => {
      const width = Math.max(1, stage.clientWidth || 1);
      const height = Math.max(1, stage.clientHeight || 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      controls.handleResize();
      requestRender();
    };
    const observer = new ResizeObserver(() => resize());
    observer.observe(stage);
    resize();
    publishScaleBar();

    controls.addEventListener("change", requestRender);
    controls.addEventListener("start", requestRender);
    controls.addEventListener("end", requestRender);

    // ---- loading ------------------------------------------------------------

    const splatLayer = layerOfType(manifest, "splats");
    const pointLayer = layerOfType(manifest, "points");
    const cameraLayer = layerOfType(manifest, "cameras");
    const species: Scene3dSpecies[] = [];
    if (splatLayer) {
      species.push("splats");
    }
    if (pointLayer) {
      species.push("points");
    }
    if (cameraLayer) {
      species.push("cameras");
    }

    const ceilings = maxElementsFor({
      isMobile: isMobileDevice(),
      deviceMemoryGb: deviceMemoryGb(),
    });
    const loaded: Record<Scene3dSpecies, number> = { points: 0, splats: 0, cameras: 0 };
    const disposables: Array<{ dispose: () => void }> = [];
    const pointMaterials: THREE.PointsMaterial[] = [];
    let sparkRenderer: { dispose: () => void } | null = null;

    const publishProgress = () => {
      if (disposed) {
        return;
      }
      setProgress(
        species.map((type) => ({
          type,
          loaded: loaded[type],
          total: layerOfType(manifest, type)?.total ?? 0,
        }))
      );
    };
    publishProgress();

    const addPointChunk = (buffer: ArrayBuffer, header: ChunkHeader, take: number) => {
      const { positions, colors } = pointViews(buffer, header);
      const geometry = new THREE.BufferGeometry();
      // Zero-copy: the attribute wraps the fetched bytes; three copies once, on upload.
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions.subarray(0, take * 3), 3)
      );

      // UPC1 colour is sRGB, source-faithful. three assumes a vertex-colour attribute is
      // already in the linear working space and would otherwise double-encode it
      // (measured: sRGB 0.2 renders at ~0.48). This is the one place that conversion
      // happens. Alpha is not sRGB-encoded, so it is carried across untouched.
      const withAlpha = (header.flags & UPC1_FLAG_ALPHA) !== 0;
      const rgbBytes = new Uint8Array(take * 3);
      for (let index = 0; index < take; index += 1) {
        rgbBytes[index * 3] = colors[index * 4];
        rgbBytes[index * 3 + 1] = colors[index * 4 + 1];
        rgbBytes[index * 3 + 2] = colors[index * 4 + 2];
      }
      const linear = srgbBytesToLinearFloat(rgbBytes);
      if (withAlpha) {
        const rgba = new Float32Array(take * 4);
        for (let index = 0; index < take; index += 1) {
          rgba[index * 4] = linear[index * 3];
          rgba[index * 4 + 1] = linear[index * 3 + 1];
          rgba[index * 4 + 2] = linear[index * 3 + 2];
          rgba[index * 4 + 3] = colors[index * 4 + 3] / 255;
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(rgba, 4));
      } else {
        geometry.setAttribute("color", new THREE.BufferAttribute(linear, 3));
      }

      // The header already carries the chunk bbox, so frustum culling gets its bounding
      // sphere without a second pass over a two-million-point buffer.
      const min = new THREE.Vector3(header.bboxMin[0], header.bboxMin[1], header.bboxMin[2]);
      const max = new THREE.Vector3(header.bboxMax[0], header.bboxMax[1], header.bboxMax[2]);
      geometry.boundingSphere = new THREE.Sphere(
        min.clone().add(max).multiplyScalar(0.5),
        Math.max(min.distanceTo(max) / 2, Number.EPSILON)
      );

      const material = new THREE.PointsMaterial({
        size: pointSizeRef.current,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: withAlpha,
      });
      const points = new THREE.Points(geometry, material);
      points.position.set(header.origin[0], header.origin[1], header.origin[2]);
      groups.points.add(points);
      pointMaterials.push(material);
      disposables.push(geometry, material);
    };

    const loadChunkedLayer = async (
      layer: Scene3dLayer,
      type: "points" | "splats",
      onChunk: (buffer: ArrayBuffer, header: ChunkHeader, take: number) => void | Promise<void>
    ) => {
      const ceiling = type === "splats" ? ceilings.splats : ceilings.points;
      const tierCount = Math.max(layer.tiers.length, 1);
      const seen = new Set<number>();
      for (let level = 0; level < tierCount; level += 1) {
        // selectTier returns the cumulative union in load order, so tier 0 (complete
        // spatial coverage at reduced density) always lands before any refinement.
        const order =
          layer.tiers.length > 0
            ? selectTier(layer.tiers, level)
            : layer.chunks.map((chunk) => chunk.index);
        for (const index of order) {
          if (disposed || seen.has(index)) {
            continue;
          }
          seen.add(index);
          if (loaded[type] >= ceiling) {
            // Budget reached. We stop here, and the readout states how many of the
            // source's elements are on screen — the ceiling is never applied silently.
            return;
          }
          const buffer = await apiClient.fetchScene3dChunk(fileId, index, {
            signal: abort.signal,
          });
          if (disposed) {
            return;
          }
          const header = parseChunkHeader(buffer);
          // Elements within a chunk are ordered by descending importance (contract §5),
          // so a prefix is a valid decimation of that chunk rather than a spatial hole.
          const take = Math.min(header.count, Math.max(0, ceiling - loaded[type]));
          if (take > 0) {
            await onChunk(buffer, header, take);
            loaded[type] += take;
          }
          publishProgress();
          requestRender();
        }
      }
    };

    const loadSplats = async (layer: Scene3dLayer) => {
      // Dynamic import inside the effect keeps Spark in its own lazy vendor-spark chunk:
      // a static import would pull the whole splat runtime into the viewer bundle for
      // every user who only ever opens a point cloud.
      const spark = await import("@sparkjsdev/spark");
      if (disposed) {
        return;
      }
      // One SparkRenderer for the whole scene. `sortRadial: false` selects view-space z
      // as the sort key; radial (Euclidean) distance diverges off-axis and produces
      // stable, orientation-dependent seams near the frustum edges (contract §9).
      const spark3d = new spark.SparkRenderer({
        renderer,
        sortRadial: false,
        onDirty: requestRender,
      });
      // depthTest against the point and frustum passes, but never depthWrite: splats are
      // a blended accumulation, and writing depth would occlude the splats behind them.
      spark3d.material.depthTest = true;
      spark3d.material.depthWrite = false;
      scene.add(spark3d);
      sparkRenderer = spark3d;

      await loadChunkedLayer(layer, "splats", async (buffer, header, take) => {
        const { extA, extB } = splatViews(buffer, header);
        const ext = new spark.ExtSplats({
          extArrays: [extA.subarray(0, take * 4), extB.subarray(0, take * 4)],
          numSplats: take,
        });
        const mesh = new spark.SplatMesh({ splats: ext });
        // Translation only. Chunk-local coordinates plus the world origin, with the
        // asset's own frame — and therefore its SH bands — left completely alone.
        mesh.position.set(header.origin[0], header.origin[1], header.origin[2]);
        groups.splats.add(mesh);
        disposables.push(ext, mesh);
        await mesh.initialized;
      });
    };

    const loadCameras = async (layer: Scene3dLayer) => {
      const poses: Scene3dCameraPose[] = [];
      for (const chunk of layer.chunks) {
        const buffer = await apiClient.fetchScene3dChunk(fileId, chunk.index, {
          signal: abort.signal,
        });
        if (disposed) {
          return;
        }
        poses.push(...readCameraPoses(JSON.parse(new TextDecoder().decode(buffer)) as unknown));
      }
      if (disposed || poses.length === 0) {
        return;
      }
      const length = frame.radius * FRUSTUM_SCALE;
      const vertices: number[] = [];
      for (const pose of poses) {
        try {
          vertices.push(...frustumSegments(pose, length));
          loaded.cameras += 1;
        } catch {
          // A camera model we cannot build an exact projection for is skipped, never
          // approximated with a symmetric frustum. The readout's count reflects it.
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      const material = new THREE.LineBasicMaterial({
        color: 0x8fa3bf,
        transparent: true,
        opacity: 0.75,
      });
      groups.cameras.add(new THREE.LineSegments(geometry, material));
      disposables.push(geometry, material);
      publishProgress();
      requestRender();
    };

    rigRef.current = {
      setVisibility: (next) => {
        groups.points.visible = next.points;
        groups.splats.visible = next.splats;
        groups.cameras.visible = next.cameras;
        requestRender();
      },
      setPointSize: (next) => {
        for (const material of pointMaterials) {
          material.size = next;
          material.needsUpdate = true;
        }
        requestRender();
      },
      reframe: () => {
        reframe();
        requestRender();
      },
    };

    const run = async () => {
      try {
        if (splatLayer) {
          await loadSplats(splatLayer);
        }
        if (pointLayer) {
          await loadChunkedLayer(pointLayer, "points", addPointChunk);
        }
        if (cameraLayer) {
          await loadCameras(cameraLayer);
        }
      } catch (error) {
        if (disposed || abort.signal.aborted) {
          return;
        }
        setLoadError(error instanceof Error ? error.message : "Could not load this scene.");
      }
    };
    void run();

    return () => {
      disposed = true;
      abort.abort();
      rigRef.current = null;
      observer.disconnect();
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      controls.removeEventListener("change", requestRender);
      controls.removeEventListener("start", requestRender);
      controls.removeEventListener("end", requestRender);
      controls.dispose();
      sparkRenderer?.dispose();
      for (const disposable of disposables) {
        disposable.dispose();
      }
      renderer.dispose();
      renderer.domElement.parentNode?.removeChild(renderer.domElement);
    };
    // Rebuilding the GL context is expensive, so only scene identity is a dependency.
    // Visibility, point size and reset are applied incrementally by the effects below;
    // the initial values are read once here to seed the scene graph.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiClient, fileId, manifest, unitLabel]);

  useEffect(() => {
    rigRef.current?.setVisibility(visibility);
  }, [visibility]);

  useEffect(() => {
    // The ref is what chunks arriving later read, so it is updated here rather than
    // during render.
    pointSizeRef.current = pointSize;
    rigRef.current?.setPointSize(pointSize);
  }, [pointSize]);

  useEffect(() => {
    rigRef.current?.reframe();
  }, [resetToken]);

  const readout = useMemo(
    () =>
      progress
        .filter((entry) => visibility[entry.type])
        .map((entry) => ({
          type: entry.type,
          text: `${SPECIES_LABEL[entry.type]} ${describeDecimation(entry.loaded, entry.total)}`,
        })),
    [progress, visibility]
  );

  return (
    <>
      <div className="scene3d-stage" ref={stageRef}>
        <div className="scene3d-canvas-host" ref={hostRef} />
        {gateMessage ? (
          <p className="scene3d-gate">
            <b>This scene needs WebGL2.</b>
            <span>
              {gateMessage}. The provenance panel still describes the scene exactly —
              download the original to open it in a desktop tool.
            </span>
          </p>
        ) : null}
      </div>
      <div className="scene3d-readout" data-testid="scene3d-readout">
        {loadError ? <b>{loadError}</b> : null}
        {readout.map((entry) => (
          <span key={entry.type}>{entry.text}</span>
        ))}
        {scaleBar ? (
          <span className="scene3d-scalebar">
            <span
              className="scene3d-scalebar-bar"
              style={{ width: `${scaleBar.barPx}px` }}
              aria-hidden="true"
            />
            <b>{scaleBar.label}</b>
          </span>
        ) : null}
      </div>
    </>
  );
}
