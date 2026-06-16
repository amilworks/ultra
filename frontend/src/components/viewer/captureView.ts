// Dependency-free "export the current view" support for the 2D viewer canvases.
//
// Each canvas exposes a ViewerCanvasHandle (fit + capture) to the host shell. Capture
// composites the live WebGL frame with the scientifically meaningful DOM overlays
// (orientation labels, scale bar) onto an offscreen 2D canvas at the backing-store
// (devicePixelRatio) resolution, then encodes a slide-ready PNG. Interactive chrome
// (zoom buttons, status banners, anything faded out) is deliberately excluded.
//
// WebGL note: the 2D renderers are created without preserveDrawingBuffer, so the
// drawing buffer is cleared after compositing. The caller MUST render a fresh frame
// immediately before calling composeViewBlob; composeViewBlob snapshots that frame
// synchronously via drawImage (so the unpreserved buffer is read in the same tick)
// and only the offscreen 2D canvas is encoded asynchronously.

export type ViewerCanvasHandle = {
  /** Reset the view to fit the image (same as the Fit button / double-click). */
  fitView: () => void;
  /** Capture the current composed view as a PNG blob, or null if unavailable
   *  (no WebGL context, or a cross-origin-tainted canvas). */
  captureViewToBlob: () => Promise<Blob | null>;
};

type OverlayDraw = {
  // Position relative to the GL canvas, in CSS pixels.
  rect: { left: number; top: number; width: number; height: number };
  kind: "pill" | "bar" | "text";
  text: string;
  computed: CSSStyleDeclaration;
  // Effective on-screen opacity (product of the node's ancestors up to the host) so
  // the chrome-fade — applied opacity-only on a wrapper, which doesn't show up in the
  // child's own computed opacity — is honored in the export instead of burned in solid.
  opacity: number;
};

const isPaintedColor = (color: string | null | undefined): boolean =>
  Boolean(color) && color !== "rgba(0, 0, 0, 0)" && color !== "transparent";

// CSS opacity is a non-inherited, compositing-time group effect: a parent's opacity
// is NOT reflected in a child's computed opacity. The chrome-fade sets opacity on a
// wrapper, so to know an overlay's true on-screen opacity we multiply every ancestor's
// opacity from the node up to (and including) the host.
const effectiveOpacity = (node: HTMLElement, host: HTMLElement): number => {
  let opacity = 1;
  let el: HTMLElement | null = node;
  while (el) {
    const value = Number(getComputedStyle(el).opacity);
    opacity *= Number.isFinite(value) ? value : 1;
    if (el === host) {
      break;
    }
    el = el.parentElement;
  }
  return opacity;
};

// Gather the visible overlay nodes under `host` matching `selectors`, with their
// geometry (relative to the GL canvas), resolved styles, and effective opacity. Hidden
// or faded-out nodes (display:none, visibility:hidden, effective opacity≈0 from the
// chrome-fade) are skipped so the export never burns in idle/invisible chrome.
const collectOverlays = (host: HTMLElement, glCanvas: HTMLCanvasElement, selectors: string[]): OverlayDraw[] => {
  const base = glCanvas.getBoundingClientRect();
  const out: OverlayDraw[] = [];
  for (const selector of selectors) {
    host.querySelectorAll<HTMLElement>(selector).forEach((node) => {
      if (node.offsetParent === null) {
        return;
      }
      const computed = getComputedStyle(node);
      const opacity = effectiveOpacity(node, host);
      if (computed.visibility === "hidden" || opacity < 0.05) {
        return;
      }
      const rect = node.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      out.push({
        rect: { left: rect.left - base.left, top: rect.top - base.top, width: rect.width, height: rect.height },
        kind: node.classList.contains("viewer-scale-bar-line")
          ? "bar"
          : node.classList.contains("viewer-orientation-label")
            ? "pill"
            : "text",
        text: node.textContent ?? "",
        computed,
        opacity,
      });
    });
  }
  return out;
};

const traceRoundRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void => {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
};

// Draw the GL frame + overlays onto an offscreen 2D canvas at backing-store
// resolution. Sourcing dimensions from glCanvas.width/height (already
// CSS_size × cappedPixelRatio) keeps the export crisp without double-scaling.
const compositeToCanvas = (
  glCanvas: HTMLCanvasElement,
  overlays: OverlayDraw[],
  clearColorCss: string,
): HTMLCanvasElement | null => {
  const width = Math.max(1, glCanvas.width);
  const height = Math.max(1, glCanvas.height);
  // CSS px -> backing px (the effective, capped devicePixelRatio the renderer used).
  const scale = glCanvas.clientWidth > 0 ? glCanvas.width / glCanvas.clientWidth : 1;

  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  if (!ctx) {
    // Fail closed: a blank canvas would encode to a transparent PNG, which callers
    // can't distinguish from a real frame. Returning null suppresses the export.
    return null;
  }
  ctx.imageSmoothingQuality = "high";
  // Opaque background first (the deep-zoom canvas uses alpha:true), then the GL frame.
  ctx.fillStyle = clearColorCss;
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(glCanvas, 0, 0, width, height);

  for (const overlay of overlays) {
    const x = overlay.rect.left * scale;
    const y = overlay.rect.top * scale;
    const boxWidth = overlay.rect.width * scale;
    const boxHeight = overlay.rect.height * scale;
    // Honor the overlay's effective on-screen opacity (e.g. a partially faded scale bar).
    ctx.globalAlpha = overlay.opacity;
    if (overlay.kind === "bar") {
      ctx.fillStyle = isPaintedColor(overlay.computed.backgroundColor) ? overlay.computed.backgroundColor : "#ffffff";
      ctx.fillRect(x, y, boxWidth, Math.max(1, boxHeight));
      ctx.globalAlpha = 1;
      continue;
    }
    const fontSize = (parseFloat(overlay.computed.fontSize) || 12) * scale;
    ctx.font = `${overlay.computed.fontWeight || "500"} ${fontSize}px ${overlay.computed.fontFamily || "sans-serif"}`;
    if (overlay.kind === "pill" && isPaintedColor(overlay.computed.backgroundColor)) {
      const radius = (parseFloat(overlay.computed.borderRadius) || 0) * scale;
      traceRoundRect(ctx, x, y, boxWidth, boxHeight, radius);
      ctx.fillStyle = overlay.computed.backgroundColor;
      ctx.fill();
    }
    ctx.fillStyle = isPaintedColor(overlay.computed.color) ? overlay.computed.color : "#ffffff";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(overlay.text, x + boxWidth / 2, y + boxHeight / 2);
    ctx.globalAlpha = 1;
  }
  return offscreen;
};

// Compose the already-rendered GL frame + overlays into a PNG blob. Returns null when
// capture is unavailable — no 2D context, or a cross-origin-tainted canvas (toBlob
// throws SecurityError) — instead of throwing, so callers can degrade gracefully.
export const composeViewBlob = (
  host: HTMLElement,
  glCanvas: HTMLCanvasElement,
  overlaySelectors: string[],
  clearColorCss: string,
): Promise<Blob | null> => {
  try {
    const overlays = collectOverlays(host, glCanvas, overlaySelectors);
    const offscreen = compositeToCanvas(glCanvas, overlays, clearColorCss);
    if (!offscreen) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      try {
        offscreen.toBlob((blob) => resolve(blob), "image/png");
      } catch {
        resolve(null);
      }
    });
  } catch {
    return Promise.resolve(null);
  }
};

// Trigger a browser download of a blob under `filename`.
export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoke after the download has had a chance to start.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

// Trigger a browser download of a same-origin URL (e.g. the original source file).
export const downloadFromUrl = (url: string, filename: string): void => {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
};

// Whether the browser can copy a raster image to the clipboard (secure context +
// async clipboard + ClipboardItem; Firefox historically lacks image write).
export const canCopyImageToClipboard = (): boolean =>
  typeof navigator !== "undefined" &&
  Boolean(navigator.clipboard?.write) &&
  typeof ClipboardItem !== "undefined";

// Copy a PNG blob to the clipboard. Returns false (rather than throwing) when
// unsupported or denied, so callers can fall back to a download.
export const copyBlobToClipboard = async (blob: Blob): Promise<boolean> => {
  if (!canCopyImageToClipboard()) {
    return false;
  }
  try {
    await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
    return true;
  } catch {
    return false;
  }
};

// Make a safe, readable PNG filename stem from a resource name.
export const exportFileStem = (name: string | undefined | null): string => {
  const base = (name ?? "image").replace(/\.[^./\\]+$/, "");
  const cleaned = base.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return cleaned.length > 0 ? cleaned : "image";
};
