import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type * as ThreeTypes from "three";

import type { ApiClient } from "@/lib/api";
import type { Scene3dManifestResponse, UploadViewerInfo } from "@/types";

import { Scene3dViewerShell } from "./Scene3dViewerShell";

const sparkMocks = vi.hoisted(() => ({
  rendererOptions: vi.fn(),
}));

// jsdom has no WebGL, so the renderer is stubbed rather than the component: everything
// else in Scene3dCanvas — the budget, the chunk parse, the readout, the scale bar — is
// real, and only the GPU calls are hollow.
vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof ThreeTypes>();
  class StubWebGLRenderer {
    capabilities = { isWebGL2: true };
    domElement = document.createElement("canvas");
    outputColorSpace = "";
    setPixelRatio(): void {}
    setClearColor(): void {}
    setSize(): void {}
    render(): void {}
    dispose(): void {}
  }
  return { ...actual, WebGLRenderer: StubWebGLRenderer };
});

vi.mock("three/examples/jsm/controls/OrbitControls.js", async () => {
  const THREE = await import("three");
  class StubOrbitControls {
    target = new THREE.Vector3();
    enableDamping = false;
    screenSpacePanning = false;
    rotateSpeed = 0;
    zoomSpeed = 0;
    panSpeed = 0;
    update(): void {}
    dispose(): void {}
    addEventListener(): void {}
    removeEventListener(): void {}
  }
  return { OrbitControls: StubOrbitControls };
});

// GPU work stays hollow under jsdom, while the paged-loader contract remains real
// enough to catch a normalized RAD manifest being dropped before it reaches Spark.
vi.mock("@sparkjsdev/spark", async () => {
  const THREE = await import("three");
  class ExtSplats {
    dispose(): void {}
  }
  class PagedSplats {
    constructor(options: unknown) { void options; }
    setMaxSh(maxSh: number): void { void maxSh; }
    dispose(): void {}
  }
  class SplatMesh extends THREE.Object3D {
    initialized = Promise.resolve();
    dispose(): void {}
  }
  class SparkRenderer extends THREE.Object3D {
    material = { depthTest: false, depthWrite: true };
    lodInstances = new Map<SplatMesh, { numSplats: number }>();
    constructor(options: unknown) {
      super();
      sparkMocks.rendererOptions(options);
    }
    setDirty(): void {}
    update(): Promise<void> { return Promise.resolve(); }
    dispose(): void {}
  }
  return { ExtSplats, PagedSplats, SplatMesh, SparkRenderer };
});

const VIEWER_INFO = {
  kind: "scene3d",
  file_id: "file-1",
  original_name: "fused_model1_superpoint.ply",
  dims_order: "TCZYX",
  axis_sizes: { T: 1, C: 1, Z: 1, Y: 1, X: 1 },
  selected_indices: { T: 0, C: 0, Z: 0 },
  is_volume: false,
  is_timeseries: false,
  is_multichannel: false,
  scene3d: {
    status: "ready",
    scene_kind: "pointcloud",
    element_count: 2_068_089,
    service_urls: {
      manifest: "/v2/uploads/file-1/scene3d/manifest",
      chunk: "/v2/uploads/file-1/scene3d/chunk",
      download: "/v2/resources/file-1/download",
    },
  },
} as unknown as UploadViewerInfo;

/** A real `UPC1` chunk, built to the frozen header layout (contract §4.1 / §4.3). */
const upc1Chunk = (count: number): ArrayBuffer => {
  const buffer = new ArrayBuffer(64 + count * 12 + count * 4);
  const view = new DataView(buffer);
  const magic = "UPC1";
  for (let index = 0; index < 4; index += 1) {
    view.setUint8(index, magic.charCodeAt(index));
  }
  view.setUint16(4, 1, true); // version
  view.setUint16(6, 0, true); // flags: alpha not meaningful
  view.setUint32(8, count, true);
  view.setUint32(12, 0, true); // measured sh degree
  for (let axis = 0; axis < 3; axis += 1) {
    view.setFloat32(16 + axis * 4, -1, true); // bbox min
    view.setFloat32(28 + axis * 4, 1, true); // bbox max
    view.setFloat32(40 + axis * 4, 0, true); // origin
  }
  const positions = new Float32Array(buffer, 64, count * 3);
  const colors = new Uint8Array(buffer, 64 + count * 12, count * 4);
  for (let index = 0; index < count; index += 1) {
    positions[index * 3] = index / count;
    colors[index * 4] = 51; // sRGB 0.2 — the double-encoding case sceneColor guards
    colors[index * 4 + 3] = 255;
  }
  return buffer;
};

const READY_MANIFEST: Scene3dManifestResponse = {
  schema: "ultra.scene3d.v1",
  status: "ready",
  scene_kind: "pointcloud",
  source: {
    format: "ply",
    writer: "colmap",
    vertex_count: 2_068_089,
    bytes: 55_838_651,
    declared_sh_degree: 0,
    measured_sh_degree: 0,
    stride_bytes: 27,
    property_provenance: {
      preserved: ["x", "y", "z", "red", "green", "blue"],
      synthesized: ["alpha=255"],
      omitted: ["nx", "ny", "nz"],
      omitted_elements: [],
    },
  },
  world: {
    units: "arbitrary",
    up_axis: "unknown",
    up_axis_basis: "unknown",
    frame: "source",
    bbox: [-10, -10, -10, 10, 10, 10],
  },
  layers: [
    {
      type: "points",
      encoding: "upc-v1",
      total: 2_068_089,
      chunks: [{ index: 0, count: 1000, bytes: 16_064, origin: [0, 0, 0], bbox: [] }],
      tiers: [[0]],
      activation_domain: "post",
      source_frame: "source",
      quantization: {
        center: "f32-exact",
        color: "srgb-u8",
        out_of_range_color_fraction: 0.0031,
      },
    },
  ],
  limitations: [
    "Normals in the source are not rendered.",
    "This is a NeRF export; the radiance field itself is not rendered, only its point export.",
  ],
  service_urls: {
    chunk: "/v2/uploads/file-1/scene3d/chunk",
    download: "/v2/resources/file-1/download",
  },
};

const clientWith = (
  manifests: Scene3dManifestResponse[],
  chunk: ArrayBuffer = upc1Chunk(1000)
) => {
  let call = 0;
  const getScene3dManifest = vi.fn(async () => {
    const index = Math.min(call, manifests.length - 1);
    call += 1;
    return manifests[index];
  });
  const fetchScene3dChunk = vi.fn(async () => chunk);
  const getScene3dPagedLodSource = vi.fn(() => ({
    url: "/v2/uploads/file-1/scene3d/lod/scene-lod.rad",
    requestHeader: { "X-API-Key": "test" },
    withCredentials: true as const,
  }));
  return { getScene3dManifest, fetchScene3dChunk, getScene3dPagedLodSource } as unknown as ApiClient & {
    getScene3dManifest: ReturnType<typeof vi.fn>;
    fetchScene3dChunk: ReturnType<typeof vi.fn>;
    getScene3dPagedLodSource: ReturnType<typeof vi.fn>;
  };
};

afterEach(() => {
  vi.useRealTimers();
  sparkMocks.rendererOptions.mockReset();
});

describe("Scene3dViewerShell", () => {
  it("shows a calm deriving state and polls until the derive lands", async () => {
    vi.useFakeTimers();
    const apiClient = clientWith([{ status: "deriving" }, READY_MANIFEST]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText(/Preparing this scene/i)).toBeInTheDocument();
    expect(apiClient.getScene3dManifest).toHaveBeenCalledTimes(1);

    // First backoff step is 1s; after it the shell asks again and the scene resolves.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200);
    });
    expect(apiClient.getScene3dManifest).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/Preparing this scene/i)).not.toBeInTheDocument();
    expect(screen.getByText("Points")).toBeInTheDocument();
  });

  it("renders every manifest limitation verbatim", async () => {
    const apiClient = clientWith([READY_MANIFEST]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    for (const limitation of READY_MANIFEST.limitations ?? []) {
      await waitFor(() => expect(screen.getByText(limitation)).toBeInTheDocument());
    }
    // And the quantization block beside them.
    expect(screen.getByText("f32-exact")).toBeInTheDocument();
    expect(screen.getByText("0.31% of colours")).toBeInTheDocument();
    expect(screen.getByText("x, y, z, red, green, blue")).toBeInTheDocument();
    expect(screen.getByText("alpha=255")).toBeInTheDocument();
    expect(screen.getByText("nx, ny, nz")).toBeInTheDocument();
  });

  it("preserves the authenticated paged RAD contract through wire normalization", async () => {
    const radManifest: Scene3dManifestResponse = {
      ...READY_MANIFEST,
      generator_revision: "scene3d-rad-v4",
      scene_kind: "splat",
      source: {
        ...READY_MANIFEST.source!,
        vertex_count: 14_469_103,
        declared_sh_degree: 3,
        measured_sh_degree: 0,
      },
      layers: [{
        type: "splats",
        encoding: "spark-rad-v1",
        total: 14_469_103,
        chunks: [],
        tiers: [],
        activation_domain: "post",
        source_frame: "source",
        quantization: {},
        lod: {
          format: "spark-rad-v1",
          method: "bhatt-lod-quality",
          builder_revision: "spark-build-lod-2.1.0-f22236f",
          paged: true,
          source_elements: 14_469_103,
          max_sh_degree: 3,
          header: { name: "scene-lod.rad", bytes: 1024 },
          chunks: [{ name: "scene-lod-0.radc", bytes: 4096 }],
        },
      }],
      service_urls: {
        lod: "/v2/uploads/file-1/scene3d/lod/scene-lod.rad",
        download: "/v2/resources/file-1/download",
      },
    };
    const apiClient = clientWith([radManifest]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    await waitFor(() => expect(apiClient.getScene3dPagedLodSource).toHaveBeenCalledWith("file-1"));
    await waitFor(() => expect(sparkMocks.rendererOptions).toHaveBeenCalled());
    expect(sparkMocks.rendererOptions.mock.lastCall?.[0]).not.toHaveProperty("focalAdjustment");
    expect(screen.queryByText(/manifest is incomplete/i)).not.toBeInTheDocument();
    expect(screen.getByText("Gaussian splats")).toBeInTheDocument();
  });

  it("says how many of the source's elements are actually on screen", async () => {
    const apiClient = clientWith([READY_MANIFEST]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    const readout = await screen.findByTestId("scene3d-readout");
    await waitFor(() =>
      expect(readout).toHaveTextContent("points showing 1,000 of 2,068,089")
    );
  });

  it("labels the scale bar in units, never millimetres, for an arbitrary world", async () => {
    const apiClient = clientWith([READY_MANIFEST]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    const readout = await screen.findByTestId("scene3d-readout");
    await waitFor(() => expect(readout.textContent ?? "").toMatch(/\d+(\.\d+)?\s+unit\b/));
    expect(readout.textContent ?? "").not.toMatch(/mm/);
  });

  it("offers the original download when the derive failed", async () => {
    const apiClient = clientWith([{ status: "failed", error: "unsupported PLY property layout" }]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    await screen.findByText(/could not be prepared for the viewer/i);
    expect(screen.getByText("unsupported PLY property layout")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /original/i })).toHaveAttribute(
      "href",
      "/v2/resources/file-1/download"
    );
  });

  it("keeps scene controls compact and details opt-in", async () => {
    const apiClient = clientWith([READY_MANIFEST]);

    render(<Scene3dViewerShell viewerInfo={VIEWER_INFO} apiClient={apiClient} />);

    const points = await screen.findByRole("button", { name: /points/i });
    expect(points).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Scene details").closest("details")).not.toHaveAttribute("open");

    fireEvent.click(points);
    expect(points).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(screen.getByRole("button", { name: /reset view/i }));
  });
});
