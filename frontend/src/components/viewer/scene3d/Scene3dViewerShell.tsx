import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import type { ApiClient } from "@/lib/api";
import type {
  Scene3dCalibration,
  Scene3dChunkInfo,
  Scene3dLayer,
  Scene3dManifest,
  Scene3dManifestResponse,
  UploadViewerInfo,
} from "@/types";
import { Download, RotateCcw } from "lucide-react";

import {
  Scene3dCanvas,
  type Scene3dCameraCatalogEntry,
  type Scene3dLayerVisibility,
  type Scene3dSpecies,
} from "./Scene3dCanvas";
import { describeSceneUpDirection, inferredSignedUpAxis } from "./sceneOrientation";
import "./scene3d-viewer.css";

type Props = {
  viewerInfo: UploadViewerInfo;
  apiClient: ApiClient;
};

type SceneStatus = "loading" | "deriving" | "ready" | "failed";

type SceneCalibrationDraft = Pick<
  Scene3dCalibration,
  "signed_up_axis" | "handedness" | "units" | "units_per_source_unit"
>;

type ResolvedScene = {
  status: SceneStatus;
  manifest: Scene3dManifest | null;
  /** 0..1 when the derive worker reports it; null when it does not. */
  progress: number | null;
  failure: string | null;
};

const SPECIES_LABEL: Record<Scene3dSpecies, string> = {
  splats: "Gaussian splats",
  points: "Points",
  cameras: "Cameras",
};

const SPECIES_ORDER: Scene3dSpecies[] = ["splats", "points", "cameras"];

const QUANTIZATION_LABEL: Record<string, string> = {
  center: "centre",
  scale: "scale",
  rotation: "rotation",
  color: "colour",
};

// Explicit locale, like sceneBudget's counter: these numbers are provenance and must not
// change shape with the viewer's locale.
const count = (value: number): string =>
  Math.max(0, Math.floor(Number.isFinite(value) ? value : 0)).toLocaleString("en-US");

const percent = (value: number): string => `${(value * 100).toFixed(2)}%`;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const toFiniteNumber = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const toNonNegativeInt = (value: unknown, fallback: number): number =>
  Math.max(0, Math.round(toFiniteNumber(value, fallback)));

const toNumberArray = (value: unknown): number[] =>
  Array.isArray(value) ? value.map((entry) => toFiniteNumber(entry, 0)) : [];

const toStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.map((entry) => String(entry)) : [];

const summarizeProperties = (values: string[], limit = 8): string => {
  if (values.length <= limit) return values.join(", ");
  return `${values.slice(0, limit).join(", ")}, +${values.length - limit} more`;
};

const toTriple = (value: unknown): [number, number, number] => {
  const parsed = toNumberArray(value);
  return [parsed[0] ?? 0, parsed[1] ?? 0, parsed[2] ?? 0];
};

const normalizeChunk = (raw: unknown, fallbackIndex: number): Scene3dChunkInfo => {
  const source: Record<string, unknown> = isRecord(raw) ? raw : {};
  return {
    index: toNonNegativeInt(source.index, fallbackIndex),
    count: toNonNegativeInt(source.count, 0),
    bytes: toNonNegativeInt(source.bytes, 0),
    origin: toTriple(source.origin),
    bbox: toNumberArray(source.bbox),
  };
};

const normalizeLodArtifact = (raw: unknown): { name: string; bytes: number } => {
  const source: Record<string, unknown> = isRecord(raw) ? raw : {};
  return {
    name: String(source.name ?? ""),
    bytes: toNonNegativeInt(source.bytes, 0),
  };
};

const normalizeLayer = (raw: unknown): Scene3dLayer => {
  const source: Record<string, unknown> = isRecord(raw) ? raw : {};
  const quantization: Record<string, unknown> = isRecord(source.quantization)
    ? source.quantization
    : {};
  const lod: Record<string, unknown> | null = isRecord(source.lod) ? source.lod : null;
  return {
    type: String(source.type ?? ""),
    encoding: String(source.encoding ?? ""),
    total: toNonNegativeInt(source.total, 0),
    chunks: Array.isArray(source.chunks)
      ? source.chunks.map((chunk, index) => normalizeChunk(chunk, index))
      : [],
    tiers: Array.isArray(source.tiers)
      ? source.tiers.map((tier) =>
        Array.isArray(tier) ? tier.map((index) => toNonNegativeInt(index, 0)) : []
      )
      : [],
    activation_domain: String(source.activation_domain ?? "unknown"),
    source_frame: String(source.source_frame ?? "source"),
    quantization: {
      center: quantization.center == null ? undefined : String(quantization.center),
      scale: quantization.scale == null ? undefined : String(quantization.scale),
      rotation: quantization.rotation == null ? undefined : String(quantization.rotation),
      color: quantization.color == null ? undefined : String(quantization.color),
      out_of_range_color_fraction:
        quantization.out_of_range_color_fraction == null &&
        quantization.clamped_color_fraction == null
          ? undefined
          : toFiniteNumber(
            quantization.out_of_range_color_fraction ?? quantization.clamped_color_fraction,
            0
          ),
    },
    lod: lod
      ? {
        format: String(lod.format ?? ""),
        method: String(lod.method ?? ""),
        builder_revision: String(lod.builder_revision ?? ""),
        paged: lod.paged === true,
        source_elements: toNonNegativeInt(lod.source_elements, 0),
        max_sh_degree: Math.min(3, toNonNegativeInt(lod.max_sh_degree, 0)),
        header: normalizeLodArtifact(lod.header),
        chunks: Array.isArray(lod.chunks) ? lod.chunks.map(normalizeLodArtifact) : [],
      }
      : undefined,
  };
};

/**
 * Turn whatever the endpoint returned into a status and, when it is ready, a manifest
 * every field of which is a value this component can render. Same defensiveness as
 * `normalizeCiftiViewerInfo`: the viewer never trusts the wire, because a half-typed
 * manifest that crashes the panel loses the provenance the panel exists to show.
 */
const resolveScene = (raw: Scene3dManifestResponse | null): ResolvedScene => {
  if (!raw) {
    return { status: "loading", manifest: null, progress: null, failure: null };
  }
  const declared = String(raw.status ?? "").trim().toLowerCase();
  const layers = Array.isArray(raw.layers) ? raw.layers.map(normalizeLayer) : [];
  const status: SceneStatus =
    declared === "deriving" || declared === "failed" || declared === "ready"
      ? (declared as SceneStatus)
      : layers.length > 0
        ? "ready"
        : "deriving";
  const progress =
    typeof raw.progress === "number" && Number.isFinite(raw.progress)
      ? Math.min(1, Math.max(0, raw.progress))
      : null;
  const failure = raw.error == null ? null : String(raw.error);
  if (status !== "ready") {
    return { status, manifest: null, progress, failure };
  }

  const source: Record<string, unknown> = isRecord(raw.source) ? raw.source : {};
  const propertyProvenance: Record<string, unknown> | null = isRecord(
    source.property_provenance
  )
    ? source.property_provenance
    : null;
  const world: Record<string, unknown> = isRecord(raw.world) ? raw.world : {};
  const reconstruction: Record<string, unknown> | null = isRecord(raw.reconstruction)
    ? raw.reconstruction
    : null;
  const urls: Record<string, unknown> = isRecord(raw.service_urls) ? raw.service_urls : {};
  return {
    status,
    progress,
    failure,
    manifest: {
      schema: String(raw.schema ?? "ultra.scene3d.v1"),
      generator_revision:
        raw.generator_revision == null ? undefined : String(raw.generator_revision),
      scene_kind: String(raw.scene_kind ?? "pointcloud"),
      source: {
        format: String(source.format ?? "unknown"),
        writer: source.writer == null ? null : String(source.writer),
        sha256: source.sha256 == null ? undefined : String(source.sha256),
        vertex_count: toNonNegativeInt(source.vertex_count, 0),
        bytes: toNonNegativeInt(source.bytes, 0),
        declared_sh_degree: toNonNegativeInt(source.declared_sh_degree, 0),
        measured_sh_degree: toNonNegativeInt(source.measured_sh_degree, 0),
        stride_bytes: toNonNegativeInt(source.stride_bytes, 0),
        geometry_member:
          source.geometry_member == null ? undefined : String(source.geometry_member),
        geometry_bytes:
          source.geometry_bytes == null ? undefined : toNonNegativeInt(source.geometry_bytes, 0),
        colmap_model_path:
          source.colmap_model_path == null ? undefined : String(source.colmap_model_path),
        container_bytes:
          source.container_bytes == null
            ? undefined
            : toNonNegativeInt(source.container_bytes, 0),
        property_provenance: propertyProvenance
          ? {
            preserved: toStringArray(propertyProvenance.preserved),
            synthesized: toStringArray(propertyProvenance.synthesized),
            omitted: toStringArray(propertyProvenance.omitted),
            omitted_elements: Array.isArray(propertyProvenance.omitted_elements)
              ? propertyProvenance.omitted_elements
                .filter(isRecord)
                .map((element) => ({
                  name: String(element.name ?? "element"),
                  count: toNonNegativeInt(element.count, 0),
                }))
              : [],
          }
          : undefined,
      },
      world: {
        units: String(world.units ?? "arbitrary"),
        up_axis: String(world.up_axis ?? "unknown"),
        up_axis_basis: String(world.up_axis_basis ?? "unknown"),
        frame: String(world.frame ?? "source"),
        bbox: toNumberArray(world.bbox),
        // The camera frames on this, not on `bbox` — see frameOf in Scene3dCanvas.
        // Omitted when the manifest predates the field, in which case frameOf falls
        // back to `bbox`. This normalizer rebuilds `world` field by field, so a new
        // manifest key that is not listed here is silently dropped.
        bbox_robust:
          world.bbox_robust === undefined ? undefined : toNumberArray(world.bbox_robust),
      },
      layers,
      limitations: Array.isArray(raw.limitations) ? raw.limitations.map((item) => String(item)) : [],
      reconstruction: reconstruction
        ? {
          registered_images: toNonNegativeInt(reconstruction.registered_images, 0),
          matched_images: toNonNegativeInt(reconstruction.matched_images, 0),
          preview_images: toNonNegativeInt(reconstruction.preview_images, 0),
          preview_limit: toNonNegativeInt(reconstruction.preview_limit, 0),
          ambiguous_images: toNonNegativeInt(reconstruction.ambiguous_images, 0),
          unreadable_images: toNonNegativeInt(reconstruction.unreadable_images, 0),
        }
        : undefined,
      service_urls: {
        chunk: urls.chunk == null ? undefined : String(urls.chunk),
        lod: urls.lod == null ? undefined : String(urls.lod),
        camera_image:
          urls.camera_image == null ? undefined : String(urls.camera_image),
        download: urls.download == null ? undefined : String(urls.download),
      },
    },
  };
};

const layerTotals = (manifest: Scene3dManifest | null): Record<Scene3dSpecies, number> => {
  const totals: Record<Scene3dSpecies, number> = { splats: 0, points: 0, cameras: 0 };
  for (const layer of manifest?.layers ?? []) {
    if (layer.type === "splats" || layer.type === "points" || layer.type === "cameras") {
      totals[layer.type] = layer.total;
    }
  }
  return totals;
};

// 1s, 2s, 4s, 8s, then a steady 15s. A derive over a 3.4 GB splat file runs for minutes,
// so the tail is deliberately slow rather than a hot poll against the edge node.
const pollDelayMs = (attempt: number): number => Math.min(1000 * 2 ** attempt, 15000);

/**
 * The scene3d Lens shell: manifest fetch and derive-state handling, the layer panel,
 * the display controls, and the provenance panel that states — verbatim — what this
 * viewer is not doing. The GL surface itself is `Scene3dCanvas`.
 */
export function Scene3dViewerShell({ viewerInfo, apiClient }: Props) {
  const fileId = viewerInfo.file_id;
  const [response, setResponse] = useState<Scene3dManifestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [visibilityOverride, setVisibilityOverride] =
    useState<Partial<Scene3dLayerVisibility> | null>(null);
  const [pointSize, setPointSize] = useState(1.6);
  const [resetToken, setResetToken] = useState(0);
  const [calibration, setCalibration] = useState<Scene3dCalibration | null>(null);
  const [calibrationDraft, setCalibrationDraft] = useState<SceneCalibrationDraft>({
    signed_up_axis: "+y",
    handedness: "right",
    units: "arbitrary",
    units_per_source_unit: 1,
  });
  const calibrationIdentityRef = useRef<string | null>(null);
  const [calibrationSaving, setCalibrationSaving] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [cameraCatalog, setCameraCatalog] = useState<Scene3dCameraCatalogEntry[]>([]);
  const [selectedCameraIndex, setSelectedCameraIndex] = useState(0);
  const [cameraViewRequest, setCameraViewRequest] = useState<{
    index: number;
    token: number;
  } | null>(null);
  const [cameraPreviewState, setCameraPreviewState] = useState<{
    key: string;
    url: string | null;
    error: string | null;
  }>({ key: "", url: null, error: null });

  const scene = useMemo(() => resolveScene(response), [response]);
  const settled = scene.status === "ready" || scene.status === "failed";

  // Fetch the manifest once and cache it in state. The guard is the CACHED RESULT
  // (`settled`), not a "requested" ref: React StrictMode's mount/unmount/mount in dev
  // cancels the throwaway mount's fetch, and the real mount must still fetch — a
  // "requested" ref would have been set by the throwaway and would strand us forever.
  // A deriving manifest is not terminal, so `attempt` re-enters this effect on the
  // poll timer below rather than the effect looping on its own result.
  useEffect(() => {
    if (settled) {
      return;
    }
    let cancelled = false;
    // Fetch inside an async closure (not the effect body) so the loading/error setState
    // calls aren't flagged as synchronous-in-effect.
    const run = async () => {
      setError(null);
      try {
        const manifest = await apiClient.getScene3dManifest(fileId);
        if (!cancelled) {
          setResponse(manifest);
          const ready = resolveScene(manifest).manifest;
          if (ready) {
            const sourceSHA = ready.source.sha256;
            const identity = `${fileId}:${sourceSHA ?? "unbound"}`;
            if (calibrationIdentityRef.current !== identity) {
              const saved = viewerInfo.scene3d?.calibration;
              const current =
                saved && sourceSHA && saved.source_sha256 === sourceSHA ? saved : null;
              calibrationIdentityRef.current = identity;
              setCalibration(current);
              setCalibrationDraft(
                current ?? {
                  signed_up_axis: inferredSignedUpAxis(ready),
                  handedness: "right",
                  units: "arbitrary",
                  units_per_source_unit: 1,
                }
              );
            }
          }
        }
      } catch (cause: unknown) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : "Could not load this scene.");
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [apiClient, fileId, attempt, settled, viewerInfo.scene3d?.calibration]);

  // Poll while the derive job runs. Backoff, not a hot loop.
  useEffect(() => {
    if (scene.status !== "deriving") {
      return;
    }
    const timer = window.setTimeout(() => setAttempt((value) => value + 1), pollDelayMs(attempt));
    return () => window.clearTimeout(timer);
  }, [scene.status, attempt]);

  const retry = useCallback(() => {
    setError(null);
    setResponse(null);
    setAttempt((value) => value + 1);
  }, []);

  const updateCameraCatalog = useCallback((entries: Scene3dCameraCatalogEntry[]) => {
    setCameraCatalog(entries);
    setSelectedCameraIndex((current) =>
      entries.some((entry) => entry.index === current) ? current : (entries[0]?.index ?? 0)
    );
  }, []);

  const selectedCamera = useMemo(
    () => cameraCatalog.find((entry) => entry.index === selectedCameraIndex) ?? null,
    [cameraCatalog, selectedCameraIndex]
  );
  const previewCount = useMemo(
    () => cameraCatalog.filter((entry) => entry.sourceImage !== undefined).length,
    [cameraCatalog]
  );
  const registeredCameraCount =
    scene.manifest?.reconstruction?.registered_images ?? cameraCatalog.length;
  const cameraPreviewKey = selectedCamera?.sourceImage
    ? `${selectedCamera.index}:${selectedCamera.sourceImage.artifact_index}`
    : "";
  const cameraPreviewUrl =
    cameraPreviewState.key === cameraPreviewKey ? cameraPreviewState.url : null;
  const cameraPreviewError =
    cameraPreviewState.key === cameraPreviewKey ? cameraPreviewState.error : null;

  useEffect(() => {
    const preview = selectedCamera?.sourceImage;
    if (!preview) return;
    const key = `${selectedCamera.index}:${preview.artifact_index}`;
    const abort = new AbortController();
    let objectUrl: string | null = null;
    void apiClient
      .fetchScene3dCameraImage(fileId, preview.artifact_index, { signal: abort.signal })
      .then((blob) => {
        if (abort.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setCameraPreviewState({ key, url: objectUrl, error: null });
      })
      .catch((previewError: unknown) => {
        if (abort.signal.aborted) return;
        setCameraPreviewState({
          key,
          url: null,
          error:
            previewError instanceof Error ? previewError.message : "Could not load this preview.",
        });
      });
    return () => {
      abort.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [apiClient, fileId, selectedCamera]);

  const totals = useMemo(() => layerTotals(scene.manifest), [scene.manifest]);
  const present = useMemo(
    () => SPECIES_ORDER.filter((species) => (scene.manifest?.layers ?? []).some((l) => l.type === species)),
    [scene.manifest]
  );

  // Contract §9: points and splats are mutually exclusive BY DEFAULT. 3DGS initialises
  // its Gaussians at the sparse points, so opaque depth-writing point sprites punch
  // holes through the splat cloud at exactly the densest geometry.
  const visibility = useMemo<Scene3dLayerVisibility>(() => {
    const hasSplats = present.includes("splats");
    return {
      splats: hasSplats,
      points: present.includes("points") && !hasSplats,
      cameras: present.includes("cameras"),
      ...(visibilityOverride ?? {}),
    };
  }, [present, visibilityOverride]);

  const toggle = useCallback((species: Scene3dSpecies, next: boolean) => {
    setVisibilityOverride((previous) => ({ ...(previous ?? {}), [species]: next }));
  }, []);

  const overlapWarning = visibility.points && visibility.splats;
  const downloadUrl =
    scene.manifest?.service_urls?.download ?? viewerInfo.scene3d?.service_urls?.download;
  const measuredSh = scene.manifest?.source.measured_sh_degree;
  const declaredSh = scene.manifest?.source.declared_sh_degree;
  const upDirection = scene.manifest
    ? describeSceneUpDirection(scene.manifest, calibration)
    : "unknown";

  const saveCalibration = useCallback(async () => {
    const sourceSHA = scene.manifest?.source.sha256;
    const scale = calibrationDraft.units_per_source_unit;
    if (!sourceSHA || !Number.isFinite(scale) || scale < 1e-12 || scale > 1e12) {
      setCalibrationError("Enter a positive scale and wait for source identity to resolve.");
      return;
    }
    setCalibrationSaving(true);
    setCalibrationError(null);
    try {
      await apiClient.patchResourceMetadata(fileId, {
        ultra_scene3d_calibration_v1: {
          version: 1,
          source_sha256: sourceSHA,
          expected_revision: calibration?.revision ?? 0,
          ...calibrationDraft,
        },
      });
      setCalibration({
        version: 1,
        source_sha256: sourceSHA,
        revision: (calibration?.revision ?? 0) + 1,
        ...calibrationDraft,
      });
      setResetToken((value) => value + 1);
    } catch (saveError) {
      setCalibrationError(
        saveError instanceof Error ? saveError.message : "Could not save this calibration."
      );
    } finally {
      setCalibrationSaving(false);
    }
  }, [apiClient, calibration, calibrationDraft, fileId, scene.manifest?.source.sha256]);

  const downloadButton = downloadUrl ? (
    <Button asChild variant="outline" size="sm">
      <a href={downloadUrl}>
        <Download className="mr-1.5 size-3.5" />
        Original
      </a>
    </Button>
  ) : null;

  return (
    <div className="scene3d-shell">
      <div className="scene3d-header">
        <div className="scene3d-heading">
          <div className="scene3d-title" title={viewerInfo.original_name}>
            {viewerInfo.original_name}
          </div>
          <div className="scene3d-meta">
            <span className="scene3d-badge">3D scene</span>
            <span>{scene.manifest?.scene_kind ?? viewerInfo.scene3d?.scene_kind ?? "scene"}</span>
            {scene.manifest?.source.writer ? <span>{scene.manifest.source.writer}</span> : null}
            {scene.manifest?.source.vertex_count ? (
              <span className="scene3d-count">
                {count(scene.manifest.source.vertex_count)} vertices
              </span>
            ) : null}
          </div>
        </div>
        <div className="scene3d-header-actions">{downloadButton}</div>
      </div>

      {error ? (
        <div className="scene3d-status scene3d-status-error">
          <p>{error}</p>
          <Button type="button" variant="outline" size="sm" onClick={retry}>
            Retry
          </Button>
        </div>
      ) : scene.status === "failed" ? (
        <div className="scene3d-status scene3d-status-error">
          <p>This scene could not be prepared for the viewer.</p>
          {scene.failure ? <small>{scene.failure}</small> : null}
          <p className="scene3d-note">
            {viewerInfo.scene3d?.message ??
              "The original file is untouched. Download it to open the scene in a desktop tool."}
          </p>
        </div>
      ) : scene.status === "loading" || scene.status === "deriving" ? (
        <div className="scene3d-status">
          <p>
            {scene.status === "loading" ? "Reading the scene manifest…" : "Preparing this scene…"}
          </p>
          <div
            className={
              scene.progress === null
                ? "scene3d-progress scene3d-progress-indeterminate"
                : "scene3d-progress"
            }
          >
            <div
              className="scene3d-progress-fill"
              style={scene.progress === null ? undefined : { width: percent(scene.progress) }}
            />
          </div>
          <small>
            Splats and point maps are chunked once, on the worker, so the viewer never
            parses the source file. Large scenes take a few minutes; this page checks
            again on its own.
          </small>
        </div>
      ) : scene.manifest ? (
        <div className="scene3d-body">
          <div className="scene3d-toolbar" aria-label="3D scene controls">
            <div className="scene3d-toolgroup">
              <span className="scene3d-toollabel">Layers</span>
              <div className="scene3d-layer-toggles">
                {present.map((species) => (
                  <button
                    type="button"
                    key={species}
                    aria-pressed={visibility[species]}
                    title={`Show or hide ${SPECIES_LABEL[species].toLowerCase()}`}
                    onClick={() => toggle(species, !visibility[species])}
                  >
                    <span>{SPECIES_LABEL[species]}</span>
                    <span className="scene3d-layer-count">{count(totals[species])}</span>
                  </button>
                ))}
              </div>
            </div>
            {present.includes("points") ? (
              <div className="scene3d-toolgroup scene3d-point-control">
                <label htmlFor="scene3d-point-size">
                  Point size <span>{pointSize.toFixed(1)} px</span>
                </label>
                <input
                  id="scene3d-point-size"
                  type="range"
                  min={0.5}
                  max={6}
                  step={0.1}
                  value={pointSize}
                  title="Adjust point size on screen"
                  onChange={(event) => setPointSize(Number(event.target.value))}
                />
              </div>
            ) : null}
            <details className="scene3d-calibration-control">
              <summary
                title="Calibrate view-up and physical scale without changing source coordinates"
              >
                Frame &amp; scale
              </summary>
              <div className="scene3d-calibration-panel">
                <div className="scene3d-calibration-grid">
                  <label>
                    <span>Up axis</span>
                    <select
                      value={calibrationDraft.signed_up_axis}
                      onChange={(event) => setCalibrationDraft((current) => ({
                        ...current,
                        signed_up_axis: event.target.value as Scene3dCalibration["signed_up_axis"],
                      }))}
                    >
                      <option value="+x">+X</option><option value="-x">−X</option>
                      <option value="+y">+Y</option><option value="-y">−Y</option>
                      <option value="+z">+Z</option><option value="-z">−Z</option>
                    </select>
                  </label>
                  <label>
                    <span>Handedness</span>
                    <select
                      value={calibrationDraft.handedness}
                      title="Documents the source coordinate frame; it does not mirror geometry"
                      onChange={(event) => setCalibrationDraft((current) => ({
                        ...current,
                        handedness: event.target.value as Scene3dCalibration["handedness"],
                      }))}
                    >
                      <option value="right">Right-handed</option>
                      <option value="left">Left-handed</option>
                    </select>
                  </label>
                  <label>
                    <span>Unit</span>
                    <select
                      value={calibrationDraft.units}
                      onChange={(event) => setCalibrationDraft((current) => ({
                        ...current,
                        units: event.target.value as Scene3dCalibration["units"],
                      }))}
                    >
                      <option value="arbitrary">Arbitrary</option>
                      <option value="m">m</option><option value="cm">cm</option>
                      <option value="mm">mm</option><option value="um">µm</option>
                      <option value="nm">nm</option>
                    </select>
                  </label>
                  <label>
                    <span>Per source unit</span>
                    <input
                      type="number"
                      min="0.000000000001"
                      max="1000000000000"
                      step="any"
                      value={calibrationDraft.units_per_source_unit}
                      onChange={(event) => setCalibrationDraft((current) => ({
                        ...current,
                        units_per_source_unit: Number(event.target.value),
                      }))}
                    />
                  </label>
                </div>
                <p>
                  View-up and the scale bar change. Geometry, splat orientation, and
                  COLMAP poses stay in the source frame.
                </p>
                {calibrationError ? <small role="alert">{calibrationError}</small> : null}
                <button
                  type="button"
                  className="scene3d-btn"
                  disabled={calibrationSaving || !scene.manifest.source.sha256}
                  onClick={() => void saveCalibration()}
                >
                  {calibrationSaving ? "Saving…" : calibration ? "Update" : "Save"}
                </button>
              </div>
            </details>
            <button
              type="button"
              className="scene3d-btn scene3d-reset"
              title="Return to the initial scientific framing"
              onClick={() => setResetToken((value) => value + 1)}
            >
              <RotateCcw aria-hidden="true" />
              Reset view
            </button>
          </div>

          {overlapWarning ? (
            <p className="scene3d-warn">
              Points and splats occupy the same reconstruction sites. Showing both can
              make opaque points punch through the splat surface.
            </p>
          ) : null}

          <div className="scene3d-main">
            <Scene3dCanvas
              fileId={fileId}
              manifest={scene.manifest}
              apiClient={apiClient}
              visibility={visibility}
              pointSize={pointSize}
              resetToken={resetToken}
              calibration={calibration}
              onCameraCatalog={updateCameraCatalog}
              cameraViewRequest={cameraViewRequest}
            />
          </div>

          {cameraCatalog.length > 0 ? (
            <details className="scene3d-camera-validation">
              <summary>
                <span>Camera validation</span>
                <span>
                  {count(registeredCameraCount)} registered · {count(cameraCatalog.length)} viewable ·{" "}
                  {count(previewCount)} previews
                </span>
              </summary>
              <div className="scene3d-camera-panel">
                <div className="scene3d-camera-controls">
                  <div className="scene3d-section-title">Registered image</div>
                  <div className="scene3d-camera-stepper">
                    <button
                      type="button"
                      className="scene3d-btn"
                      aria-label="Previous registered camera"
                      disabled={selectedCameraIndex <= 0}
                      onClick={() => setSelectedCameraIndex((value) => Math.max(0, value - 1))}
                    >
                      Previous
                    </button>
                    <label>
                      <span className="sr-only">Registered camera number</span>
                      <input
                        type="number"
                        min={1}
                        max={cameraCatalog.length}
                        value={selectedCameraIndex + 1}
                        onChange={(event) => {
                          const next = Math.floor(Number(event.target.value));
                          if (Number.isFinite(next)) {
                            setSelectedCameraIndex(
                              Math.min(cameraCatalog.length - 1, Math.max(0, next - 1))
                            );
                          }
                        }}
                      />
                      <span>of {count(cameraCatalog.length)}</span>
                    </label>
                    <button
                      type="button"
                      className="scene3d-btn"
                      aria-label="Next registered camera"
                      disabled={selectedCameraIndex >= cameraCatalog.length - 1}
                      onClick={() => setSelectedCameraIndex((value) =>
                        Math.min(cameraCatalog.length - 1, value + 1)
                      )}
                    >
                      Next
                    </button>
                  </div>
                  <div className="scene3d-camera-name" title={selectedCamera?.name}>
                    {selectedCamera?.name}
                  </div>
                  <button
                    type="button"
                    className="scene3d-btn"
                    title="Place the viewer at this exact COLMAP pose and intrinsic projection"
                    onClick={() => setCameraViewRequest((current) => ({
                      index: selectedCameraIndex,
                      token: (current?.token ?? 0) + 1,
                    }))}
                  >
                    View from camera
                  </button>
                  <p className="scene3d-note">
                    The pose and intrinsic projection are applied exactly. The source scene
                    remains in its original coordinate frame.
                  </p>
                </div>
                <div className="scene3d-camera-preview">
                  {cameraPreviewUrl ? (
                    <img src={cameraPreviewUrl} alt={`Source preview for ${selectedCamera?.name}`} />
                  ) : cameraPreviewError ? (
                    <p role="alert">{cameraPreviewError}</p>
                  ) : selectedCamera?.sourceImage ? (
                    <p>Loading source preview…</p>
                  ) : (
                    <p>No uniquely matched source image was published for this camera.</p>
                  )}
                  {selectedCamera?.sourceImage ? (
                    <small>
                      Source {count(selectedCamera.sourceImage.source_width)} ×{" "}
                      {count(selectedCamera.sourceImage.source_height)} px · bounded preview
                    </small>
                  ) : null}
                </div>
              </div>
            </details>
          ) : null}

          <details className="scene3d-details">
            <summary>
              <span>Scene details</span>
              <span>
                {scene.manifest.world.frame} frame · {scene.manifest.world.units}
              </span>
            </summary>
            <div className="scene3d-details-grid">
              <section className="scene3d-section">
                <div className="scene3d-section-title">Provenance</div>
                {scene.manifest.limitations.length > 0 ? (
                  <ul className="scene3d-prov-list">
                    {scene.manifest.limitations.map((limitation) => (
                      <li key={limitation}>{limitation}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="scene3d-note">The derive reported no limitations for this scene.</p>
                )}

                <dl className="scene3d-quant">
                  <dt>frame</dt>
                  <dd>
                    {scene.manifest.world.frame} · up {upDirection} (
                    {calibration ? "user calibration" : scene.manifest.world.up_axis_basis})
                  </dd>
                  <dt>units</dt>
                  <dd>
                    {calibration
                      ? `${calibration.units_per_source_unit.toLocaleString("en-US", {
                        maximumSignificantDigits: 6,
                      })} ${calibration.units} / source unit`
                      : scene.manifest.world.units}
                  </dd>
                  {calibration ? (
                    <>
                      <dt>handedness</dt>
                      <dd>{calibration.handedness}-handed · documented, not mirrored</dd>
                    </>
                  ) : null}
                  {typeof measuredSh === "number" && typeof declaredSh === "number" ? (
                    <>
                      <dt>SH degree</dt>
                      <dd>
                        {measuredSh} measured
                        {declaredSh === measuredSh ? "" : ` · ${declaredSh} declared`}
                      </dd>
                    </>
                  ) : null}
                  {scene.manifest.source.stride_bytes > 0 ? (
                    <>
                      <dt>stride</dt>
                      <dd>{count(scene.manifest.source.stride_bytes)} B</dd>
                    </>
                  ) : null}
                  {scene.manifest.source.property_provenance?.preserved.length ? (
                    <>
                      <dt>preserved</dt>
                      <dd title={scene.manifest.source.property_provenance.preserved.join(", ")}>
                        {summarizeProperties(scene.manifest.source.property_provenance.preserved)}
                      </dd>
                    </>
                  ) : null}
                  {scene.manifest.source.property_provenance?.synthesized.length ? (
                    <>
                      <dt>synthesized</dt>
                      <dd title={scene.manifest.source.property_provenance.synthesized.join(", ")}>
                        {summarizeProperties(scene.manifest.source.property_provenance.synthesized)}
                      </dd>
                    </>
                  ) : null}
                  {scene.manifest.source.property_provenance?.omitted.length ? (
                    <>
                      <dt>omitted</dt>
                      <dd title={scene.manifest.source.property_provenance.omitted.join(", ")}>
                        {summarizeProperties(scene.manifest.source.property_provenance.omitted)}
                      </dd>
                    </>
                  ) : null}
                  {scene.manifest.source.property_provenance?.omitted_elements.length ? (
                    <>
                      <dt>omitted elements</dt>
                      <dd>
                        {scene.manifest.source.property_provenance.omitted_elements
                          .map((element) => `${element.name} (${count(element.count)})`)
                          .join(", ")}
                      </dd>
                    </>
                  ) : null}
                </dl>
              </section>

              <section className="scene3d-section">
                <div className="scene3d-section-title">Encoding</div>
                {scene.manifest.layers.map((layer) => (
                  <div className="scene3d-encoding" key={`${layer.type}-${layer.encoding}`}>
                    <div className="scene3d-encoding-title">
                      {layer.type} · {layer.encoding}
                    </div>
                    <dl className="scene3d-quant">
                      <dt>domain</dt>
                      <dd>
                        {layer.activation_domain} · {layer.source_frame}
                      </dd>
                      {Object.entries(QUANTIZATION_LABEL).flatMap(([key, label]) => {
                        const value = layer.quantization[key as keyof typeof layer.quantization];
                        return typeof value === "string" && value.length > 0
                          ? [<dt key={`${key}-term`}>{label}</dt>, <dd key={`${key}-def`}>{value}</dd>]
                          : [];
                      })}
                      {typeof layer.quantization.out_of_range_color_fraction === "number" ? (
                        <>
                          <dt>outside gamut</dt>
                          <dd>
                            {percent(layer.quantization.out_of_range_color_fraction)} of colours
                          </dd>
                        </>
                      ) : null}
                    </dl>
                  </div>
                ))}
              </section>
            </div>
          </details>
        </div>
      ) : (
        <div className="scene3d-status">Preparing view…</div>
      )}
    </div>
  );
}
