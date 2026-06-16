import { readFileSync } from "node:fs";
import path from "node:path";

import { DataUtils } from "three";
import { describe, expect, it } from "vitest";

import {
  advanceProgressiveVolumeSteps,
  computeVolumeInteriorCameraFrame,
  computePhysicalVolumeGeometry,
  computeVolumeCameraFit,
  computeVolumeSampleBudget,
  isVolumeInteriorInspectionActive,
  shouldShowVolumeContextEdges,
  shouldShowVolumeSliceCursorPlanes,
  resolveScalarVolumeColorMap,
  resolveScalarVolumeLighting,
  resolveScalarVolumeTransferFunction,
  resolveVolumeClipCue,
  resolveVolumeCameraMode,
  resolveVolumeSliceCursorCue,
  resolveVolumeViewPreset,
  resolveVolumeAxisCue,
  resolveVolumeScaleBar,
  resolveVolumeProjectionMode,
  resolveVolumeCutawayCutZ,
  resolveVolumeCutawayClip,
  resolveVolumeOrientationCue,
  scalarVolumePayloadValueAt,
  scalarVolumePayloadToTextureBytes,
  scalarVolumePayloadToHalfFloat,
} from "./SliceStackVolumeCanvas";

describe("volume sampling budget", () => {
  it("starts scalar volumes at interactive quality and ramps toward a settled budget", () => {
    const budget = computeVolumeSampleBudget({
      sourceKind: "scalar",
      volumeDepth: 256,
      projectionMode: "mip",
    });

    expect(budget.settledSteps).toBe(512);
    expect(budget.interactiveSteps).toBe(64);
    expect(advanceProgressiveVolumeSteps(budget.interactiveSteps, budget)).toBe(96);
    expect(advanceProgressiveVolumeSteps(480, budget)).toBe(512);
  });

  it("keeps atlas volumes cheaper while preserving a high-quality settled pass", () => {
    const budget = computeVolumeSampleBudget({
      sourceKind: "atlas",
      volumeDepth: 80,
      projectionMode: "composite",
    });

    expect(budget.settledSteps).toBe(160);
    expect(budget.interactiveSteps).toBe(32);
  });
});

describe("volume projection mode", () => {
  it("maps scalar volume fusion state to MIP or composite projection", () => {
    expect(
      resolveVolumeProjectionMode({
        renderPolicy: "scalar",
        modality: "medical",
        fusionMethod: "m",
      })
    ).toBe("mip");

    expect(
      resolveVolumeProjectionMode({
        renderPolicy: "scalar",
        modality: "medical",
        fusionMethod: "a",
      })
    ).toBe("composite");
  });

  it("keeps microscopy scalar volumes in MIP by default when no projection is selected", () => {
    expect(
      resolveVolumeProjectionMode({
        renderPolicy: "scalar",
        modality: "microscopy",
      })
    ).toBe("mip");
  });
});

describe("volume view presets", () => {
  it("maps named scientific planes to repeatable camera directions", () => {
    expect(resolveVolumeViewPreset("xy")).toEqual({
      id: "xy",
      label: "XY",
      direction: { x: 0, y: 0, z: 1 },
      up: { x: 0, y: 1, z: 0 },
    });

    expect(resolveVolumeViewPreset("unknown")).toEqual({
      id: "iso",
      label: "3D",
      direction: { x: 1.8, y: 1.4, z: 1.8 },
      up: { x: 0, y: 1, z: 0 },
    });
  });
});

describe("volume camera mode", () => {
  it("normalizes perspective and orthographic camera modes for scientific volume views", () => {
    expect(resolveVolumeCameraMode("orthographic")).toEqual({
      id: "orthographic",
      label: "Orthographic",
      isOrthographic: true,
    });

    expect(resolveVolumeCameraMode("unknown")).toEqual({
      id: "perspective",
      label: "Perspective",
      isOrthographic: false,
    });
  });
});

describe("volume canvas lifecycle", () => {
  it("mounts WebGL into a dedicated host so React overlays cannot race canvas cleanup", () => {
    const source = readFileSync(
      path.join(process.cwd(), "src/components/viewer/SliceStackVolumeCanvas.tsx"),
      "utf-8"
    );

    expect(source).toContain("canvasHostRef");
    expect(source).toContain("canvasHost.appendChild(renderer.domElement)");
    expect(source).toContain("renderer.domElement.parentNode?.removeChild(renderer.domElement)");
  });
});

describe("volume camera fit", () => {
  it("reserves HUD-safe margins while keeping the full volume in frame", () => {
    const relaxed = computeVolumeCameraFit({
      volumeRadius: 1,
      aspect: 16 / 9,
      safeInsets: { top: 0.04, right: 0.04, bottom: 0.08, left: 0.04 },
    });
    const hudSafe = computeVolumeCameraFit({
      volumeRadius: 1,
      aspect: 16 / 9,
      safeInsets: { top: 0.08, right: 0.06, bottom: 0.22, left: 0.06 },
    });

    expect(hudSafe.distance).toBeGreaterThan(relaxed.distance);
    expect(hudSafe.minDistance).toBeGreaterThan(2.7);
    // Re-baselined for the tighter VOLUME_CAMERA_FIT_MARGIN (1.35 -> 1.15): at
    // safeHeight ~0.70 the ortho height is (2 * 1.15) / 0.70 ~= 3.28.
    expect(hudSafe.orthographicFrustumHeight).toBeGreaterThan(3.2);
    expect(hudSafe.maxDistance).toBeGreaterThan(hudSafe.distance);
  });

  it("exposes close camera limits for interior volume inspection", () => {
    const fit = computeVolumeCameraFit({
      volumeRadius: 1,
      aspect: 1,
    });

    expect(fit.minDistance).toBeGreaterThan(2);
    expect(fit.insideMinDistance).toBeGreaterThan(0.02);
    expect(fit.insideMinDistance).toBeLessThan(0.2);
    expect(fit.inspectMaxZoom).toBeGreaterThan(6);
  });
});

describe("volume interior inspection camera", () => {
  it("places perspective cutaway inspection at the physical center of the active clip", () => {
    const frame = computeVolumeInteriorCameraFrame({
      clipBounds: {
        min: { x: 0.2, y: 0.3, z: 0.4 },
        max: { x: 0.8, y: 0.9, z: 1 },
      },
      normalizedScale: { x: 1, y: 0.5, z: 0.25 },
      preset: resolveVolumeViewPreset("xy"),
      volumeRadius: 1,
    });

    expect(frame.position).toEqual({ x: 0, y: 0.05, z: 0.05 });
    expect(frame.target.x).toBeCloseTo(0, 5);
    expect(frame.target.y).toBeCloseTo(0.05, 5);
    expect(frame.target.z).toBeLessThan(frame.position.z);
    expect(frame.lookDistance).toBeGreaterThan(0.1);
  });

  it("activates interior inspection only for clipped perspective volumes", () => {
    expect(
      isVolumeInteriorInspectionActive({
        clipActive: true,
        cameraMode: resolveVolumeCameraMode("perspective"),
      })
    ).toBe(true);
    expect(
      isVolumeInteriorInspectionActive({
        clipActive: true,
        cameraMode: resolveVolumeCameraMode("orthographic"),
      })
    ).toBe(false);
    expect(
      isVolumeInteriorInspectionActive({
        clipActive: false,
        cameraMode: resolveVolumeCameraMode("perspective"),
      })
    ).toBe(false);
  });

  it("removes in-canvas slice planes while the camera is inside the volume", () => {
    expect(shouldShowVolumeSliceCursorPlanes({ cueVisible: true, interiorInspectionActive: false })).toBe(true);
    expect(shouldShowVolumeSliceCursorPlanes({ cueVisible: true, interiorInspectionActive: true })).toBe(false);
  });

  it("removes the translucent cursor planes in the Z-cursor cutaway (the crisp cut face replaces them)", () => {
    // Cutaway keeps the overview camera (interiorInspectionActive false), but the
    // colored X/Y/Z cursor quads would still tint/occlude the cut face — hide them.
    expect(
      shouldShowVolumeSliceCursorPlanes({ cueVisible: true, interiorInspectionActive: false, cutawayActive: true })
    ).toBe(false);
    expect(
      shouldShowVolumeSliceCursorPlanes({ cueVisible: true, interiorInspectionActive: false, cutawayActive: false })
    ).toBe(true);
    // Context edges stay — they are thin extent lines, not footprint-filling quads.
    expect(shouldShowVolumeContextEdges({ cueVisible: true, interiorInspectionActive: false })).toBe(true);
  });

  it("keeps wireframe context out of the center-inside inspection view", () => {
    expect(shouldShowVolumeContextEdges({ cueVisible: true, interiorInspectionActive: false })).toBe(true);
    expect(shouldShowVolumeContextEdges({ cueVisible: true, interiorInspectionActive: true })).toBe(false);
  });
});

describe("volume Z-cursor cutaway", () => {
  it("slices at voxel centers so the cut face matches the 2D slice at the same index", () => {
    const depth = 10;
    // (index + 0.5) / depth — identical to the backend slice extraction.
    expect(resolveVolumeCutawayCutZ(0, depth)).toBeCloseTo(0.05, 5);
    expect(resolveVolumeCutawayCutZ(4, depth)).toBeCloseTo(0.45, 5);
    expect(resolveVolumeCutawayCutZ(9, depth)).toBeCloseTo(0.95, 5);
  });

  it("centers the cut when the Z index is missing and clamps out-of-range indices", () => {
    expect(resolveVolumeCutawayCutZ(null, 10)).toBeCloseTo(0.45, 5);
    expect(resolveVolumeCutawayCutZ(undefined, 10)).toBeCloseTo(0.45, 5);
    // Below the first / past the last slice still resolves to an in-volume cut.
    expect(resolveVolumeCutawayCutZ(-5, 10)).toBeCloseTo(0.05, 5);
    expect(resolveVolumeCutawayCutZ(99, 10)).toBeCloseTo(0.95, 5);
    // Degenerate single-slice and empty volumes never divide by zero.
    expect(resolveVolumeCutawayCutZ(0, 1)).toBeCloseTo(0.5, 5);
    expect(resolveVolumeCutawayCutZ(0, 0)).toBeCloseTo(0.5, 5);
  });

  it("keeps the volume from z=0 up to the cut face so the cross-section faces the camera", () => {
    expect(resolveVolumeCutawayClip(0.45)).toEqual({
      min: { x: 0, y: 0, z: 0 },
      max: { x: 1, y: 1, z: 0.45 },
    });
  });
});

describe("scalar volume color maps", () => {
  it("normalizes scalar volume color maps for shader uniforms and UI labels", () => {
    expect(resolveScalarVolumeColorMap("viridis")).toEqual({
      id: "viridis",
      label: "Viridis",
      shaderValue: 1,
    });

    expect(resolveScalarVolumeColorMap("unknown-palette")).toEqual({
      id: "grayscale",
      label: "Grayscale",
      shaderValue: 0,
    });
  });
});

describe("scalar volume transfer function", () => {
  it("normalizes signal floor and density controls for shader uniforms", () => {
    expect(
      resolveScalarVolumeTransferFunction({
        volume_signal_floor: 0.35,
        volume_density: 1.4,
      })
    ).toEqual({
      signalFloor: 0.35,
      densityScale: 1.4,
      signalFloorLabel: "35%",
      densityLabel: "1.40x",
    });

    expect(
      resolveScalarVolumeTransferFunction({
        volume_signal_floor: 2,
        volume_density: -1,
      })
    ).toEqual({
      signalFloor: 0.95,
      densityScale: 0.1,
      signalFloorLabel: "95%",
      densityLabel: "0.10x",
    });
  });
});

describe("scalar volume depth lighting", () => {
  it("normalizes lighting controls for shader uniforms and UI labels", () => {
    expect(
      resolveScalarVolumeLighting({
        volume_lighting: true,
        volume_lighting_strength: 0.72,
      })
    ).toEqual({
      enabled: true,
      strength: 0.72,
      strengthLabel: "72%",
    });

    expect(
      resolveScalarVolumeLighting({
        volume_lighting: false,
        volume_lighting_strength: 2,
      })
    ).toEqual({
      enabled: false,
      strength: 1,
      strengthLabel: "100%",
    });
  });

  it("uses the orthographic camera ray direction for scalar rim lighting", () => {
    const source = readFileSync(
      path.join(process.cwd(), "src/components/viewer/SliceStackVolumeCanvas.tsx"),
      "utf-8"
    );

    expect(source).toContain("vec3 viewDir = uOrthographicCamera");
    expect(source).toContain("? -normalize(uCameraDirectionLocal)");
    expect(source).toContain(": normalize(uCameraPositionLocal - (location - vec3(0.5)))");
  });

  it("uses step-length corrected opacity so interior raymarching does not become a flat wall", () => {
    const source = readFileSync(
      path.join(process.cwd(), "src/components/viewer/SliceStackVolumeCanvas.tsx"),
      "utf-8"
    );

    expect(source).toContain("float alphaFromOpacity(float opacityValue, float stepLength)");
    expect(source).toContain("length(delta)");
    expect(source).toContain("1.0 - pow(1.0 - baseAlpha");
  });
});

describe("physical volume geometry", () => {
  it("normalizes anisotropic voxel spacing into a physically faithful render scale", () => {
    const geometry = computePhysicalVolumeGeometry({
      planePixelSize: { width: 64, height: 32 },
      volumeDepth: 12,
      physicalSpacing: { x: 0.5, y: 0.5, z: 2 },
    });

    expect(geometry.worldWidth).toBe(32);
    expect(geometry.worldHeight).toBe(16);
    expect(geometry.worldDepth).toBe(24);
    expect(geometry.normalizedScale).toEqual({ x: 1, y: 0.5, z: 0.75 });
    expect(geometry.aspectLabel).toBe("32 x 16 x 24");
    expect(geometry.spacingLabel).toBe("0.50 x 0.50 x 2.00");
    expect(geometry.isAnisotropic).toBe(true);
  });
});

describe("physical volume scale bar", () => {
  it("chooses a readable physical scale bar length from the volume extent", () => {
    expect(resolveVolumeScaleBar({ worldWidth: 32, unit: "um" })).toEqual({
      visible: true,
      length: 10,
      label: "10 um",
      fraction: 0.3125,
    });

    expect(resolveVolumeScaleBar({ worldWidth: 0, unit: "um" })).toEqual({
      visible: false,
      length: 0,
      label: "",
      fraction: 0,
    });
  });
});

describe("physical volume axis cue", () => {
  it("labels the physical X/Y/Z volume extents with a shared unit", () => {
    expect(
      resolveVolumeAxisCue({
        worldWidth: 32,
        worldHeight: 16,
        worldDepth: 24,
        unit: "vox",
      })
    ).toEqual({
      visible: true,
      unit: "vox",
      summary: "X 32 vox / Y 16 vox / Z 24 vox",
      x: { axis: "X", length: 32, label: "X 32 vox" },
      y: { axis: "Y", length: 16, label: "Y 16 vox" },
      z: { axis: "Z", length: 24, label: "Z 24 vox" },
    });

    expect(
      resolveVolumeAxisCue({
        worldWidth: 32,
        worldHeight: 0,
        worldDepth: 24,
        unit: "vox",
      })
    ).toEqual({
      visible: false,
      unit: "vox",
      summary: "",
      x: { axis: "X", length: 0, label: "" },
      y: { axis: "Y", length: 0, label: "" },
      z: { axis: "Z", length: 0, label: "" },
    });
  });
});

describe("physical volume clip cue", () => {
  it("summarizes active cutaway bounds in physical units", () => {
    expect(
      resolveVolumeClipCue({
        worldWidth: 32,
        worldHeight: 16,
        worldDepth: 24,
        clipMin: { x: 0.2, y: 0, z: 0.25 },
        clipMax: { x: 0.8, y: 1, z: 0.75 },
        unit: "vox",
      })
    ).toEqual({
      active: true,
      unit: "vox",
      summary: "Clip X 6.40-25.60 vox / Y 0-16 vox / Z 6.00-18.00 vox",
      x: { axis: "X", start: 6.4, end: 25.6, length: 19.2, label: "X 6.40-25.60 vox" },
      y: { axis: "Y", start: 0, end: 16, length: 16, label: "Y 0-16 vox" },
      z: { axis: "Z", start: 6, end: 18, length: 12, label: "Z 6.00-18.00 vox" },
    });

    expect(
      resolveVolumeClipCue({
        worldWidth: 32,
        worldHeight: 16,
        worldDepth: 24,
        clipMin: { x: 0, y: 0, z: 0 },
        clipMax: { x: 1, y: 1, z: 1 },
        unit: "vox",
      }).active
    ).toBe(false);
  });
});

describe("volume slice cursor cue", () => {
  it("maps selected voxel indices to centered 3D slice cursor positions", () => {
    const cue = resolveVolumeSliceCursorCue({
      axisSizes: { X: 64, Y: 32, Z: 12 },
      indices: { x: 31, y: 15, z: 6 },
      worldWidth: 32,
      worldHeight: 16,
      worldDepth: 24,
      unit: "vox",
    });

    expect(cue.visible).toBe(true);
    expect(cue.summary).toBe("X 32/64 / Y 16/32 / Z 7/12");
    expect(cue.x.label).toBe("X 32/64");
    expect(cue.x.positionLabel).toBe("15.75 vox");
    expect(cue.x.normalized).toBeCloseTo(0.4921875, 6);
    expect(cue.y.positionLabel).toBe("7.75 vox");
    expect(cue.y.normalized).toBeCloseTo(0.484375, 6);
    expect(cue.z.positionLabel).toBe("13 vox");
    expect(cue.z.normalized).toBeCloseTo(0.5416667, 6);
  });

  it("clamps out-of-range slice cursor indices before computing labels", () => {
    const cue = resolveVolumeSliceCursorCue({
      axisSizes: { X: 4, Y: 1, Z: 10 },
      indices: { x: -2, y: 9, z: 99 },
      worldWidth: 4,
      worldHeight: 1,
      worldDepth: 10,
      unit: "vox",
    });

    expect(cue.summary).toBe("X 1/4 / Y 1/1 / Z 10/10");
    expect(cue.x.index).toBe(0);
    expect(cue.x.normalized).toBeCloseTo(0.125, 6);
    expect(cue.y.index).toBe(0);
    expect(cue.y.normalized).toBeCloseTo(0.5, 6);
    expect(cue.z.index).toBe(9);
    expect(cue.z.normalized).toBeCloseTo(0.95, 6);
  });
});

describe("volume orientation cue", () => {
  it("turns patient axis labels into a visible 3D orientation triad", () => {
    const cue = resolveVolumeOrientationCue({
      frame: "patient",
      row_axis: "y",
      col_axis: "x",
      slice_axis: "z",
      axis_labels: {
        x: { negative: "L", positive: "R" },
        y: { negative: "P", positive: "A" },
        z: { negative: "I", positive: "S" },
      },
    });

    expect(cue.frame).toBe("patient");
    expect(cue.x).toEqual({ negative: "L", positive: "R", label: "X L->R" });
    expect(cue.y).toEqual({ negative: "P", positive: "A", label: "Y P->A" });
    expect(cue.z).toEqual({ negative: "I", positive: "S", label: "Z I->S" });
  });
});

describe("scalar volume texture conversion", () => {
  it("reads uint16 source voxel values by x/y/z index", () => {
    const source = new Uint16Array([10, 20, 30, 40, 50, 60, 70, 80]);

    expect(
      scalarVolumePayloadValueAt(
        {
          data: source.buffer,
          width: 2,
          height: 2,
          depth: 2,
          dtype: "uint16",
          bytesPerVoxel: 2,
          rawMin: 10,
          rawMax: 80,
          channel: 0,
        },
        { x: 1, y: 0, z: 1 }
      )
    ).toBe(60);
  });

  it("returns null for out-of-bounds scalar voxel indices", () => {
    const source = new Uint8Array([1, 2, 3, 4]);
    const payload = {
      data: source.buffer,
      width: 2,
      height: 2,
      depth: 1,
      dtype: "uint8",
      bytesPerVoxel: 1,
      rawMin: 1,
      rawMax: 4,
      channel: 0,
    };

    expect(scalarVolumePayloadValueAt(payload, { x: -1, y: 0, z: 0 })).toBeNull();
    expect(scalarVolumePayloadValueAt(payload, { x: 2, y: 0, z: 0 })).toBeNull();
  });

  it("normalizes uint16 payloads by source range before uploading an 8-bit texture", () => {
    const source = new Uint16Array([10, 65, 120]);
    const textureBytes = scalarVolumePayloadToTextureBytes({
      data: source.buffer,
      width: 3,
      height: 1,
      depth: 1,
      dtype: "uint16",
      bytesPerVoxel: 2,
      rawMin: 10,
      rawMax: 120,
      channel: 0,
    });

    expect(Array.from(textureBytes)).toEqual([0, 128, 255]);
  });

  it("reads and normalizes signed int16 scalar volume payloads", () => {
    const source = new Int16Array([-1024, 0, 1024]);
    const payload = {
      data: source.buffer,
      width: 3,
      height: 1,
      depth: 1,
      dtype: "int16",
      bytesPerVoxel: 2,
      rawMin: -1024,
      rawMax: 1024,
      channel: 0,
    };

    expect(scalarVolumePayloadValueAt(payload, { x: 0, y: 0, z: 0 })).toBe(-1024);
    expect(Array.from(scalarVolumePayloadToTextureBytes(payload))).toEqual([0, 128, 255]);
  });

  it("reads and normalizes float32 scalar volume payloads", () => {
    const source = new Float32Array([-1.5, 0.5, 2.5]);
    const payload = {
      data: source.buffer,
      width: 3,
      height: 1,
      depth: 1,
      dtype: "float32",
      bytesPerVoxel: 4,
      rawMin: -1.5,
      rawMax: 2.5,
      channel: 0,
    };

    expect(scalarVolumePayloadValueAt(payload, { x: 1, y: 0, z: 0 })).toBe(0.5);
    expect(Array.from(scalarVolumePayloadToTextureBytes(payload))).toEqual([0, 128, 255]);
  });
});

describe("scalar volume half-float precision", () => {
  it("maps normalized range endpoints to canonical half-float bit patterns", () => {
    const source = new Uint16Array([10, 65, 120]);
    const half = scalarVolumePayloadToHalfFloat({
      data: source.buffer,
      width: 3,
      height: 1,
      depth: 1,
      dtype: "uint16",
      bytesPerVoxel: 2,
      rawMin: 10,
      rawMax: 120,
      channel: 0,
    });

    // 0.0 -> 0x0000, 0.5 -> 0x3800, 1.0 -> 0x3c00
    expect(Array.from(half)).toEqual([0x0000, 0x3800, 0x3c00]);
  });

  it("keeps a sub-window contrast step that 8-bit quantization collapses", () => {
    // A CT-scale range (air to bone) with two voxels 2 HU apart, the kind of
    // difference that separates CSF from adjacent tissue.
    const source = new Int16Array([10, 12]);
    const payload = {
      data: source.buffer,
      width: 2,
      height: 1,
      depth: 1,
      dtype: "int16",
      bytesPerVoxel: 2,
      rawMin: 0,
      rawMax: 4096,
      channel: 0,
    };

    const bytes = scalarVolumePayloadToTextureBytes(payload);
    expect(bytes[0]).toBe(bytes[1]); // 8-bit upload erases the boundary

    const half = scalarVolumePayloadToHalfFloat(payload);
    expect(half[0]).not.toBe(half[1]); // half-float keeps the two tissues distinct
    expect(DataUtils.fromHalfFloat(half[0])).toBeCloseTo(10 / 4096, 5);
    expect(DataUtils.fromHalfFloat(half[1])).toBeCloseTo(12 / 4096, 5);
  });
});

describe("scalar volume boundary-emphasis transfer function", () => {
  const source = readFileSync(
    path.join(process.cwd(), "src/components/viewer/SliceStackVolumeCanvas.tsx"),
    "utf-8"
  );

  it("uploads medical scalar volumes as 16-bit half-float to preserve soft-tissue contrast", () => {
    expect(source).toContain("scalarVolumePayloadToHalfFloat(payload)");
    expect(source).toContain("texture.type = THREE.HalfFloatType;");
  });

  it("modulates opacity by local gradient so tissue interfaces become visible surfaces", () => {
    expect(source).toContain("vec3 scalarGradient(vec3 location)");
    expect(source).toContain("float structuredOpacity(float opacityValue, vec3 gradient)");
    expect(source).toContain("mix(uInteriorOpacity, 1.0, edge)");
    expect(source).toContain("float opacity = structuredOpacity(opacityValue, gradient);");
  });

  it("corrects the shading normal by voxel spacing, not full-axis extent", () => {
    // A central difference spans one voxel, so the world gradient divides by the
    // voxel SPACING ratio (uVoxelSpacing), not the box extent (uVolumeScale,
    // which includes voxel count and previously tilted normals toward thick axes).
    expect(source).toContain("uniform vec3 uVoxelSpacing;");
    expect(source).toContain("gradient / max(vec3(0.06), uVoxelSpacing)");
    expect(source).not.toContain("gradient / max(vec3(0.06), uVolumeScale)");
    // The spacing-ratio uniform is wired from the physical geometry.
    expect(source).toContain("uVoxelSpacing: { value: new THREE.Vector3(voxelSpacingRatio.x");
  });

  it("accumulates optical depth in physical (world) space on anisotropic volumes", () => {
    // The ray marches in normalized cube space; scaling the step by the box
    // extents makes accumulated opacity invariant to viewing direction.
    expect(source).toContain("length(delta * uVolumeScale)");
  });
});
