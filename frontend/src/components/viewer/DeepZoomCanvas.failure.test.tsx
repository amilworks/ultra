import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { UploadViewerInfo } from "@/types";

const mockState = vi.hoisted(() => ({ textureLoads: vi.fn() }));

vi.mock("three", () => {
  class Disposable {
    dispose(): void {}
  }
  class WebGLRenderer extends Disposable {
    domElement = document.createElement("canvas");
    outputColorSpace = "";
    setPixelRatio(): void {}
    setClearColor(): void {}
    setSize(): void {}
    render(): void {}
  }
  class Scene {
    add(): void {}
  }
  class OrthographicCamera {
    position = { set: () => undefined };
    left = -1;
    right = 1;
    top = 1;
    bottom = -1;
    updateProjectionMatrix(): void {}
  }
  class Group {
    add(): void {}
    remove(): void {}
  }
  class PlaneGeometry extends Disposable {}
  class MeshBasicMaterial extends Disposable {
    map: unknown;
    needsUpdate = false;
  }
  class Mesh {
    position = { set: () => undefined };
    visible = false;
    constructor(
      public geometry: PlaneGeometry,
      public material: MeshBasicMaterial,
    ) {}
  }
  class TextureLoader {
    load(
      _url: string,
      _onLoad: (texture: unknown) => void,
      _onProgress: unknown,
      onError: () => void,
    ): void {
      mockState.textureLoads();
      onError();
    }
  }
  return {
    Group,
    LinearFilter: "linear",
    Mesh,
    MeshBasicMaterial,
    OrthographicCamera,
    PlaneGeometry,
    Scene,
    SRGBColorSpace: "srgb",
    TextureLoader,
    WebGLRenderer,
  };
});

import { DeepZoomCanvas } from "./DeepZoomCanvas";

const plane = {
  axis: "z" as const,
  label: "XY",
  axes: ["Y", "X"],
  pixel_size: { width: 512, height: 512 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 512, height: 512 },
  aspect_ratio: 1,
};

const viewerInfo = {
  file_id: "file-tiles",
  phys: { pixel_units: ["px"] },
  viewer: {
    default_plane: plane,
    planes: { z: plane },
    tile_scheme: {
      tile_size: 512,
      format: "png",
      levels: [{ level: 0, width: 512, height: 512, columns: 1, rows: 1, downsample: 1 }],
    },
  },
} as unknown as UploadViewerInfo;

afterEach(() => {
  vi.useRealTimers();
});

describe("DeepZoomCanvas tile failure", () => {
  it("shows the selector-aware static slice after HTTP tile retries are exhausted", async () => {
    vi.useFakeTimers();
    const uploadTileUrl = vi.fn(() => "https://ultra.example.org/v2/uploads/file-tiles/tiles/z/0/0/0");
    const uploadSliceUrl = vi.fn(
      () => "https://ultra.example.org/v2/uploads/file-tiles/slice?axis=z&z=0&t=2&channels=1",
    );
    const apiClient = { uploadTileUrl, uploadSliceUrl } as unknown as ApiClient;

    render(
      <DeepZoomCanvas
        apiClient={apiClient}
        fileId="file-tiles"
        viewerInfo={viewerInfo}
        zIndex={0}
        tIndex={2}
        channels={[1]}
      />,
    );

    act(() => {
      vi.advanceTimersByTime(300);
      vi.advanceTimersByTime(600);
      vi.advanceTimersByTime(1200);
      vi.runOnlyPendingTimers();
    });

    expect(uploadTileUrl).toHaveBeenCalledTimes(1);
    expect(mockState.textureLoads).toHaveBeenCalledTimes(4);
    const fallback = screen.getByRole("img", { name: "XY fallback" });
    expect(fallback).toHaveAttribute("src", expect.stringContaining("/slice?axis=z&z=0&t=2&channels=1"));
    expect(screen.getByText(/tile delivery failed/i)).toBeInTheDocument();
  });
});
