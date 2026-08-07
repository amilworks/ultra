import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import type { ApiClient } from "@/lib/api";
import type {
  Scene3dChunkInfo,
  Scene3dLayer,
  Scene3dManifest,
  Scene3dManifestResponse,
  UploadViewerInfo,
} from "@/types";
import { Download } from "lucide-react";

import { Scene3dCanvas, type Scene3dLayerVisibility, type Scene3dSpecies } from "./Scene3dCanvas";
import "./scene3d-viewer.css";

type Props = {
  viewerInfo: UploadViewerInfo;
  apiClient: ApiClient;
};

type SceneStatus = "loading" | "deriving" | "ready" | "failed";

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

const normalizeLayer = (raw: unknown): Scene3dLayer => {
  const source: Record<string, unknown> = isRecord(raw) ? raw : {};
  const quantization: Record<string, unknown> = isRecord(source.quantization)
    ? source.quantization
    : {};
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
      clamped_color_fraction:
        quantization.clamped_color_fraction == null
          ? undefined
          : toFiniteNumber(quantization.clamped_color_fraction, 0),
    },
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
  const world: Record<string, unknown> = isRecord(raw.world) ? raw.world : {};
  const urls: Record<string, unknown> = isRecord(raw.service_urls) ? raw.service_urls : {};
  return {
    status,
    progress,
    failure,
    manifest: {
      schema: String(raw.schema ?? "ultra.scene3d.v1"),
      scene_kind: String(raw.scene_kind ?? "pointcloud"),
      source: {
        format: String(source.format ?? "unknown"),
        writer: source.writer == null ? null : String(source.writer),
        vertex_count: toNonNegativeInt(source.vertex_count, 0),
        bytes: toNonNegativeInt(source.bytes, 0),
        declared_sh_degree: toNonNegativeInt(source.declared_sh_degree, 0),
        measured_sh_degree: toNonNegativeInt(source.measured_sh_degree, 0),
        stride_bytes: toNonNegativeInt(source.stride_bytes, 0),
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
      service_urls: {
        chunk: urls.chunk == null ? undefined : String(urls.chunk),
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
  }, [apiClient, fileId, attempt, settled]);

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
            The original file is untouched. Download it to open the scene in a desktop tool.
          </p>
          {downloadButton}
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
          {downloadButton}
        </div>
      ) : scene.manifest ? (
        <div className="scene3d-body">
          <div className="scene3d-main">
            <Scene3dCanvas
              fileId={fileId}
              manifest={scene.manifest}
              apiClient={apiClient}
              visibility={visibility}
              pointSize={pointSize}
              resetToken={resetToken}
            />
          </div>

          <aside className="scene3d-panel">
            <section className="scene3d-section">
              <div className="scene3d-section-title">Layers</div>
              {present.map((species) => (
                <label className="scene3d-layer" key={species}>
                  <input
                    type="checkbox"
                    checked={visibility[species]}
                    onChange={(event) => toggle(species, event.target.checked)}
                  />
                  <span className="scene3d-layer-name">{SPECIES_LABEL[species]}</span>
                  <span className="scene3d-layer-count">
                    <b>{count(totals[species])}</b> in source
                  </span>
                </label>
              ))}
              {overlapWarning ? (
                <p className="scene3d-warn">
                  Points and splats are drawn together. 3D Gaussian splatting initialises
                  its Gaussians at the sparse points, so the opaque point sprites will
                  punch through the splat cloud wherever the geometry is densest.
                </p>
              ) : null}
            </section>

            <section className="scene3d-section">
              <div className="scene3d-section-title">Display</div>
              {present.includes("points") ? (
                <div className="scene3d-control">
                  <label className="scene3d-control-label" htmlFor="scene3d-point-size">
                    <span>Point size</span>
                    <span>{pointSize.toFixed(1)} px</span>
                  </label>
                  <input
                    id="scene3d-point-size"
                    type="range"
                    min={0.5}
                    max={6}
                    step={0.1}
                    value={pointSize}
                    onChange={(event) => setPointSize(Number(event.target.value))}
                  />
                </div>
              ) : null}
              <button
                type="button"
                className="scene3d-btn"
                onClick={() => setResetToken((value) => value + 1)}
              >
                Reset view
              </button>
            </section>

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
                  {scene.manifest.world.frame} · up {scene.manifest.world.up_axis} (
                  {scene.manifest.world.up_axis_basis})
                </dd>
                <dt>units</dt>
                <dd>{scene.manifest.world.units}</dd>
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
              </dl>

              {scene.manifest.layers.map((layer) => (
                <div key={`${layer.type}-${layer.encoding}`}>
                  <div className="scene3d-section-title">
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
                    {typeof layer.quantization.clamped_color_fraction === "number" ? (
                      <>
                        <dt>clamped</dt>
                        <dd>{percent(layer.quantization.clamped_color_fraction)} of colours</dd>
                      </>
                    ) : null}
                  </dl>
                </div>
              ))}
            </section>
          </aside>
        </div>
      ) : (
        <div className="scene3d-status">Preparing view…</div>
      )}

      <p className="scene3d-footer">
        Rendered in the source world frame — nothing is re-oriented, so points, cameras and
        splats co-register with the model they came from and the spherical-harmonic bands
        stay valid. "Up" is a view hint applied to the controls only.
      </p>
    </div>
  );
}
