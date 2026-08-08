/**
 * Dev-only harness that renders a derived scene3d fixture end to end.
 *
 * It exists to verify the part unit tests cannot reach: that the bytes the Python
 * derive writes are consumed correctly by Spark and three.js on a real GPU. It uses
 * the SAME production modules as the Lens viewer — `sceneChunks` for the header and
 * the zero-copy views, and Spark's `ExtSplats`/`SplatMesh` — so a mismatch between the
 * writer and the reader shows up here rather than in front of a user.
 *
 * Not part of the app bundle: reached only via /scene3d-harness.html in dev.
 *
 *   ?scene=splats|points   which fixture           (default splats)
 *   ?budget=80             MB of chunk bytes to load
 *   ?tier=0                highest tier to include
 */
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import {
  parseChunkHeader,
  pointViews,
  selectTier,
  splatViews,
} from "@/components/viewer/scene3d/sceneChunks";
import { srgbBytesToLinearFloat } from "@/components/viewer/scene3d/sceneColor";
import {
  applyMat3,
  cameraBasisFromColmap,
  cameraCentreFromColmap,
} from "@/components/viewer/scene3d/sceneFrame";
import { focalOf } from "@/components/viewer/scene3d/sceneIntrinsics";
import { resolveScenePixelRatio } from "@/components/viewer/scene3d/sceneBudget";

type ChunkInfo = { index: number; count: number; bytes: number; origin: number[]; bbox: number[] };
type Manifest = {
  scene_kind: string;
  world: { bbox: number[]; bbox_robust?: number[]; units: string; up_axis: string };
  source: Record<string, unknown>;
  layers: { type: string; encoding: string; total: number; chunks: ChunkInfo[]; tiers: number[][] }[];
  limitations: string[];
};

const params = new URLSearchParams(location.search);
const scene = params.get("scene") ?? "splats";
const budgetBytes = Number(params.get("budget") ?? 80) * 1e6;
const tier = Number(params.get("tier") ?? 0);
const base = `/scene3d-fixture/${scene}`;

const hud = document.getElementById("hud") as HTMLDivElement;
const stage = document.getElementById("stage") as HTMLDivElement;
const log = (line: string) => {
  hud.textContent = `${hud.textContent}\n${line}`;
};

// Surfaced so the screenshotting driver can assert on real state instead of pixels alone.
declare global {
  interface Window {
    harness: Record<string, unknown>;
  }
}
window.harness = { ready: false, error: null };

async function main() {
  hud.textContent = `loading ${scene}…`;
  const manifest = (await (await fetch(`${base}/manifest.json`)).json()) as Manifest;
  const layer = manifest.layers[0];
  const byIndex = new Map(layer.chunks.map((c) => [c.index, c]));

  // preserveDrawingBuffer is required here and only here: without it the drawing buffer
  // is undefined after the frame is presented, so both gl.readPixels and Playwright's
  // screenshot come back fully black even when the scene rendered correctly. The product
  // renderers deliberately omit it (it costs a copy every frame) — see captureView.ts.
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  if (!renderer.capabilities.isWebGL2) throw new Error("WebGL2 unavailable");
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(resolveScenePixelRatio(window.devicePixelRatio || 1, false));
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  renderer.setClearColor(0x0b0f12, 1);
  stage.appendChild(renderer.domElement);

  const three = new THREE.Scene();
  // Frame on the robust box, exactly as Scene3dCanvas does.
  const bbox = manifest.world.bbox_robust ?? manifest.world.bbox;
  const centre = new THREE.Vector3(
    (bbox[0] + bbox[3]) / 2,
    (bbox[1] + bbox[4]) / 2,
    (bbox[2] + bbox[5]) / 2
  );
  // Bounding-SPHERE radius, then fit it to the vertical FOV. Using half the longest edge
  // and a hand-picked multiplier badly under-frames an elongated scene: the corridor
  // fixture is 1255 x 354 x 3198, and a "1.5x the longest half-edge" camera puts it far
  // enough away that its points land sub-pixel.
  const halfExtent = new THREE.Vector3(
    (bbox[3] - bbox[0]) / 2,
    (bbox[4] - bbox[1]) / 2,
    (bbox[5] - bbox[2]) / 2
  );
  const radius = halfExtent.length() || 1;
  const fovDeg = 50;

  const fitDistance = radius / Math.sin((fovDeg * Math.PI) / 360);
  // Fit the depth range to the scene. A near/far ratio in the tens of thousands (which
  // radius/1000 .. radius*40 gives on the 3.4 km corridor fixture) destroys depth
  // precision: every fragment lands on NDC z = 1 and the points occlude each other into
  // a few hundred lit pixels, which looks exactly like "the data is wrong".
  const camera = new THREE.PerspectiveCamera(
    fovDeg,
    stage.clientWidth / stage.clientHeight,
    Math.max(fitDistance - radius * 1.25, radius / 100),
    // Generous far so outliers beyond the robust box still draw when in view.
    fitDistance + radius * 4
  );
  camera.position
    .copy(centre)
    .add(new THREE.Vector3(0.7, 0.45, 0.7).normalize().multiplyScalar(fitDistance));
  camera.lookAt(centre);
  const controls = new TrackballControls(camera, renderer.domElement);
  controls.target.copy(centre);

  // Load a spatially spread prefix of the requested tier, under a byte budget.
  const wanted = selectTier(layer.tiers, tier);
  const picked: number[] = [];
  let acc = 0;
  for (const index of wanted) {
    const info = byIndex.get(index);
    if (!info) continue;
    if (acc + info.bytes > budgetBytes) continue;
    picked.push(index);
    acc += info.bytes;
  }

  let loaded = 0;
  let frustaDrawn = 0;
  let SparkRendererCtor: unknown = null;

  // Camera frusta, built from the SAME pure modules the Lens viewer uses. This is the
  // only check that the COLMAP world-to-camera convention survives the whole trip: the
  // derive emits qvec/tvec verbatim, and every inversion happens here.
  const cameraLayer = manifest.layers.find((entry) => entry.type === "cameras");
  if (cameraLayer && cameraLayer.chunks.length > 0) {
    const index = cameraLayer.chunks[0].index;
    const raw = await (await fetch(`${base}/chunk_${String(index).padStart(5, "0")}.bin`)).text();
    const poses = (JSON.parse(raw).cameras ?? []) as {
      qvec: number[];
      tvec: number[];
      camera: { model: string; width: number; height: number; params: number[] };
    }[];
    const rayLength = radius * 0.09;
    const vertices: number[] = [];
    for (const pose of poses) {
      const centre = cameraCentreFromColmap(pose.qvec, pose.tvec);
      const basis = cameraBasisFromColmap(pose.qvec, pose.tvec);
      const { fx, fy, cx, cy } = focalOf(pose.camera);
      // Corner rays in camera space from the REAL intrinsics, so an ignored fy or an
      // off-centre principal point shows up as a visibly skewed frustum.
      const { width, height } = pose.camera;
      for (const [px, py] of [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
      ]) {
        const dir = applyMat3(basis, [
          ((px - cx) / fx) * rayLength,
          -((py - cy) / fy) * rayLength,
          -rayLength,
        ]);
        vertices.push(centre[0], centre[1], centre[2]);
        vertices.push(centre[0] + dir[0], centre[1] + dir[1], centre[2] + dir[2]);
      }
      frustaDrawn += 1;
    }
    if (vertices.length > 0) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      three.add(
        new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xde8b34 }))
      );
    }
  }

  if (layer.type === "splats") {
    const spark = await import("@sparkjsdev/spark");
    // `product` mirrors Scene3dCanvas exactly, so this harness can bisect a difference
    // between the two rather than only proving the wire format.
    const productMode = params.get("mode") === "product";
    const sparkRenderer = productMode
      ? new spark.SparkRenderer({ renderer, sortRadial: false, onDirty: () => {} })
      : new spark.SparkRenderer({ renderer });
    if (productMode) {
      sparkRenderer.material.depthTest = true;
      sparkRenderer.material.depthWrite = false;
    }
    three.add(sparkRenderer);
    SparkRendererCtor = sparkRenderer;

    const splatGroup = new THREE.Group();
    if (productMode) three.add(splatGroup);

    for (const index of picked) {
      const buf = await (await fetch(`${base}/chunk_${String(index).padStart(5, "0")}.bin`)).arrayBuffer();
      const header = parseChunkHeader(buf);
      const { extA, extB } = splatViews(buf, header);
      const take = header.count;
      const ext = productMode
        ? new spark.ExtSplats({
            extArrays: [extA.subarray(0, take * 4), extB.subarray(0, take * 4)],
            numSplats: take,
          })
        : new spark.ExtSplats({ extArrays: [extA, extB], numSplats: header.count });
      const mesh = new spark.SplatMesh({ splats: ext });
      mesh.position.set(header.origin[0], header.origin[1], header.origin[2]);
      (productMode ? splatGroup : three).add(mesh);
      if (productMode) await mesh.initialized;
      loaded += header.count;
    }
  } else {
    for (const index of picked) {
      const buf = await (await fetch(`${base}/chunk_${String(index).padStart(5, "0")}.bin`)).arrayBuffer();
      const header = parseChunkHeader(buf);
      const { positions, colors } = pointViews(buf, header);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      // De-interleave RGB out of RGBA, then the single documented sRGB -> linear
      // conversion: three assumes vertex colours are already in the working space.
      const rgb = new Uint8Array(header.count * 3);
      for (let i = 0; i < header.count; i += 1) {
        rgb[i * 3] = colors[i * 4];
        rgb[i * 3 + 1] = colors[i * 4 + 1];
        rgb[i * 3 + 2] = colors[i * 4 + 2];
      }
      geometry.setAttribute("color", new THREE.BufferAttribute(srgbBytesToLinearFloat(rgb), 3));
      // Positions are chunk-LOCAL and the object is translated by the chunk origin, so the
      // bounding sphere must be computed from the local attribute (three would otherwise
      // cull most chunks against a sphere it derived before the translation was applied).
      geometry.computeBoundingSphere();
      // Screen-space point size. World-space sizing with attenuation makes a
      // kilometre-scale corridor scan invisible from a fitted camera, which would read as
      // "the data is wrong" when only the presentation was.
      const points = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: 1.6, vertexColors: true, sizeAttenuation: false })
      );
      points.position.set(header.origin[0], header.origin[1], header.origin[2]);
      three.add(points);
      loaded += header.count;
    }
  }

  hud.textContent =
    `${scene}  ${manifest.scene_kind}\n` +
    `chunks ${picked.length}/${layer.chunks.length}   ${(acc / 1e6).toFixed(1)} MB\n` +
    `showing ${loaded.toLocaleString("en-US")} of ${layer.total.toLocaleString("en-US")}\n` +
    (frustaDrawn ? `cameras ${frustaDrawn}\n` : "") +
    `bbox ${bbox.map((v) => v.toFixed(1)).join(", ")}\n` +
    `units ${manifest.world.units}`;

  const tick = () => {
    controls.update();
    renderer.render(three, camera);
    requestAnimationFrame(tick);
  };
  tick();

  // Give Spark a few frames to build its sort order before a screenshot is taken.
  await new Promise((resolve) => setTimeout(resolve, 1500));
  window.harness = {
    ready: true,
    error: null,
    scene,
    sceneKind: manifest.scene_kind,
    chunksLoaded: picked.length,
    chunksTotal: layer.chunks.length,
    elementsLoaded: loaded,
    elementsTotal: layer.total,
    bytes: acc,
    hasSpark: Boolean(SparkRendererCtor),
    frustaDrawn,
    objectsInScene: three.children.length,
    cameraPos: camera.position.toArray().map((v) => Math.round(v)),
    cameraTarget: controls.target.toArray().map((v) => Math.round(v)),
    // Where the world bbox actually lands in NDC: the decisive check when everything
    // draws but nothing is visible.
    bboxNdc: (() => {
      const pts = [];
      for (const x of [bbox[0], bbox[3]]) for (const y of [bbox[1], bbox[4]]) for (const z of [bbox[2], bbox[5]]) {
        const v = new THREE.Vector3(x, y, z).project(camera);
        pts.push([+v.x.toFixed(2), +v.y.toFixed(2), +v.z.toFixed(2)]);
      }
      return pts;
    })(),
    drawCalls: renderer.info.render.calls,
    pointsRendered: renderer.info.render.points,
    limitations: manifest.limitations,
    // Non-black pixel fraction: the cheap, decisive check that something rendered.
    nonBlack: (() => {
      const gl = renderer.getContext();
      const w = gl.drawingBufferWidth;
      const h = gl.drawingBufferHeight;
      const px = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let lit = 0;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] > 24 || px[i + 1] > 24 || px[i + 2] > 24) lit += 1;
      }
      return lit / (w * h);
    })(),
  };
  log("READY");
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  hud.textContent = `FAILED: ${message}`;
  window.harness = { ready: false, error: message };
});
