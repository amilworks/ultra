import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import type { ApiClient } from "@/lib/api";
import type { CiftiCarpetResponse, CiftiConnectivityResponse, UploadViewerInfo } from "@/types";
import { Download } from "lucide-react";

import { CiftiCarpet } from "./CiftiCarpet";
import { CiftiConnectivity } from "./CiftiConnectivity";
import "./cifti-viewer.css";

type CiftiView = "carpet" | "connectivity";

type Props = {
  viewerInfo: UploadViewerInfo;
  apiClient: ApiClient;
};

const VIEW_LABEL: Record<CiftiView, string> = {
  carpet: "Carpet plot",
  connectivity: "Connectivity",
};

// The CIFTI grayordinate viewer: renders the data MATRIX (no brain surfaces). A
// timeseries offers the carpet plot + a computed connectivity matrix; a *conn
// file offers the stored connectivity matrix; scalars/labels offer the carpet.
export function CiftiViewerShell({ viewerInfo, apiClient }: Props) {
  const cifti = viewerInfo.cifti;
  const fileId = viewerInfo.file_id;
  const views = useMemo<CiftiView[]>(() => {
    const v = (cifti?.views ?? []).filter((x): x is CiftiView => x === "carpet" || x === "connectivity");
    return v.length > 0 ? v : ["carpet"];
  }, [cifti]);

  const [rawView, setView] = useState<CiftiView>(views[0]);
  // The shell is keyed by file id, so `views` never changes under a mounted
  // instance — but guard anyway so a bad state never requests a missing view.
  const view = views.includes(rawView) ? rawView : views[0];
  const [carpet, setCarpet] = useState<CiftiCarpetResponse | null>(null);
  const [conn, setConn] = useState<CiftiConnectivityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  // Fetch the active view's payload once and cache it in state. We guard on the
  // cached value (not a "requested" set) so React StrictMode's mount/unmount/mount
  // in dev can't strand us: the throwaway mount's fetch is cancelled, and the real
  // mount still fetches because the cache is empty. Switching tabs reuses state.
  useEffect(() => {
    // Already cached (e.g. toggled back to this tab): don't refetch. This guard
    // also prevents an infinite loop, since carpet/conn are effect deps. The
    // render keys its loader on `!hasData`, so a stale `loading` flag is inert.
    const already = view === "carpet" ? carpet : conn;
    if (already) return;
    let cancelled = false;
    // Fetch inside an async closure (not the effect body) so the loading/error
    // setState calls aren't flagged as synchronous-in-effect.
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        if (view === "carpet") {
          const r = await apiClient.getCiftiCarpet(fileId);
          if (!cancelled) setCarpet(r);
        } else {
          const r = await apiClient.getCiftiConnectivity(fileId);
          if (!cancelled) setConn(r);
        }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load this view.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [view, fileId, apiClient, carpet, conn, reloadKey]);

  const retry = useCallback(() => {
    setError(null);
    setReloadKey((k) => k + 1);
  }, []);

  const downloadUrl = cifti?.service_urls?.download;
  const hasData = view === "carpet" ? carpet : conn;

  return (
    <div className="cifti-shell">
      <div className="cifti-header">
        <div className="cifti-heading">
          <div className="cifti-title" title={viewerInfo.original_name}>
            {viewerInfo.original_name}
          </div>
          <div className="cifti-meta">
            <span className="cifti-badge">CIFTI-2</span>
            <span>{cifti?.cifti_type ?? "connectivity"}</span>
            {cifti?.rows && cifti?.cols ? (
              <span className="cifti-dims">
                {cifti.rows.toLocaleString()} × {cifti.cols.toLocaleString()}
              </span>
            ) : null}
          </div>
        </div>
        <div className="cifti-header-actions">
          {views.length > 1 ? (
            <div className="cifti-seg cifti-viewtabs" role="tablist" aria-label="CIFTI view">
              {views.map((v) => (
                <button
                  key={v}
                  type="button"
                  role="tab"
                  aria-selected={view === v}
                  aria-pressed={view === v}
                  onClick={() => setView(v)}
                >
                  {VIEW_LABEL[v]}
                </button>
              ))}
            </div>
          ) : null}
          {downloadUrl ? (
            <Button asChild variant="outline" size="sm">
              <a href={downloadUrl}>
                <Download className="mr-1.5 size-3.5" />
                Original
              </a>
            </Button>
          ) : null}
        </div>
      </div>

      <div className="cifti-body">
        {error ? (
          <div className="cifti-status cifti-status-error">
            <p>{error}</p>
            <Button type="button" variant="outline" size="sm" onClick={retry}>
              Retry
            </Button>
          </div>
        ) : loading && !hasData ? (
          <div className="cifti-status">Loading {VIEW_LABEL[view].toLowerCase()}…</div>
        ) : view === "carpet" && carpet ? (
          <CiftiCarpet carpet={carpet} />
        ) : view === "connectivity" && conn ? (
          <CiftiConnectivity conn={conn} />
        ) : (
          <div className="cifti-status">Preparing view…</div>
        )}
      </div>

      <p className="cifti-footer">
        Showing the CIFTI data matrix. The anatomical surface render (grayordinates on the cortical
        surface) needs the matching meshes — open the original in Connectome Workbench for that.
      </p>
    </div>
  );
}
