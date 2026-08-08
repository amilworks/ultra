import { describe, expect, it } from "vitest";

import type { Scene3dManifest } from "@/types";

import { describeSceneUpDirection, resolveSceneUpVector } from "./sceneOrientation";

const manifestFor = (
  format: string,
  upAxis: string,
  upAxisBasis = "heuristic"
): Scene3dManifest =>
  ({
    schema: "ultra.scene3d.v1",
    scene_kind: format === "colmap" ? "colmap" : "pointcloud",
    source: {
      format,
      writer: null,
      vertex_count: 1,
      bytes: 1,
      declared_sh_degree: 0,
      measured_sh_degree: 0,
      stride_bytes: 12,
    },
    world: {
      units: "arbitrary",
      up_axis: upAxis,
      up_axis_basis: upAxisBasis,
      frame: "source",
      bbox: [-1, -1, -1, 1, 1, 1],
    },
    layers: [],
    limitations: [],
    service_urls: {},
  }) satisfies Scene3dManifest;

describe("scene orientation", () => {
  it("uses the signed RDF convention for a Y-axis PLY scene", () => {
    const manifest = manifestFor("ply", "y");

    expect(resolveSceneUpVector(manifest)).toEqual([0, -1, 0]);
    expect(describeSceneUpDirection(manifest)).toBe("−Y");
  });

  it("keeps a Z-up PLY in its source frame", () => {
    const manifest = manifestFor("ply", "z");

    expect(resolveSceneUpVector(manifest)).toEqual([0, 0, 1]);
    expect(describeSceneUpDirection(manifest)).toBe("+Z");
  });

  it("does not impose the PLY sign convention on another Y-up format", () => {
    const manifest = manifestFor("gltf", "y");

    expect(resolveSceneUpVector(manifest)).toEqual([0, 1, 0]);
    expect(describeSceneUpDirection(manifest)).toBe("+Y");
  });

  it("lets declared source metadata override the heuristic PLY convention", () => {
    const manifest = manifestFor("ply", "y", "declared");

    expect(resolveSceneUpVector(manifest)).toEqual([0, 1, 0]);
    expect(describeSceneUpDirection(manifest)).toBe("+Y");
  });

  it("keeps the neutral Three.js fallback when the source has no up-axis evidence", () => {
    const manifest = manifestFor("ply", "unknown");

    expect(resolveSceneUpVector(manifest)).toEqual([0, 1, 0]);
    expect(describeSceneUpDirection(manifest)).toBe("unknown");
  });
});
