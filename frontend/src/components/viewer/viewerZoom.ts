// Shared zoom constants + readout formatting for the viewer's 2D surfaces, so the
// direct-plane and deep-zoom canvases use the SAME zoom step and the SAME readout
// semantics. The readout is "% of native" (1 source px == 1 screen px == 100%, the
// Photoshop/Preview convention scientists expect). Pure + unit-tested.

// Multiplicative zoom step shared by the +/- keys and the toolbar zoom buttons on
// every 2D surface, so they never drift apart.
export const ZOOM_STEP = 1.22;

// formatZoomReadout shows zoom as % of native. "Fit" when sitting at the fit scale.
export const formatZoomReadout = (screenPxPerSourcePx: number, atFit: boolean): string => {
  if (atFit) {
    return "Fit";
  }
  const pct = screenPxPerSourcePx * 100;
  if (pct >= 1000) return `${Math.round(pct / 100) * 100}%`;
  if (pct >= 100) return `${Math.round(pct)}%`;
  if (pct >= 10) return `${Math.round(pct)}%`;
  return `${pct.toFixed(1)}%`;
};

// Whether a viewport scale is effectively at the fit scale (within a small tolerance),
// shared so both surfaces decide "Fit" identically.
export const isAtFitScale = (scale: number, fitScale: number): boolean =>
  !Number.isFinite(scale) || Math.abs(scale - fitScale) / Math.max(1e-9, fitScale) < 0.025;

// Convert a viewport scale (screen-px per WORLD unit — the shared ImageViewport
// convention used by both 2D surfaces) into screen-px per SOURCE pixel, the value
// formatZoomReadout expects. The ratio worldWidth/pixelWidth is world units per source
// pixel; it is 1 when an image's physical spacing equals its pixel grid, but differs
// for anisotropic/physical spacing. Both surfaces route through this so their
// "% of native" readouts are computed identically and can never drift.
export const nativeZoomLevel = (viewportScale: number, worldWidth: number, pixelWidth: number): number => {
  const unitsPerSourcePx = pixelWidth > 0 ? worldWidth / pixelWidth : 1;
  return viewportScale * unitsPerSourcePx;
};
