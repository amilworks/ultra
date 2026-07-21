import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CiftiCarpetResponse } from "@/types";
import {
  COLORMAP_LABELS,
  COLORMAPS,
  type ColormapKey,
  decodeBase64Bytes,
  prettyStructure,
  sampleColor,
} from "./colormaps";

type Props = {
  carpet: CiftiCarpetResponse;
};

type Hover = { struct: string; gord: number; frame: number; time: string; z: string; color: string; x: number; y: number };

const PAD = { l: 122, r: 60, t: 14, b: 36 };
const CMAP_ORDER: ColormapKey[] = ["rdbu", "gray", "viridis"];

// The carpet plot: rows = brain grayordinates grouped by structure, columns =
// time (or the second matrix axis), colour = the per-row z-scored signal. One
// canvas draws the heatmap, axes, structure gutter, colourbar and time cursor
// together (a single coordinate space — no overlay-alignment drift). Scroll to
// zoom the time axis, drag to pan, hover to read a value, click to drop a cursor.
export function CiftiCarpet({ carpet }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const [cmap, setCmap] = useState<ColormapKey>("rdbu");
  const [hover, setHover] = useState<Hover | null>(null);
  const [zoomed, setZoomed] = useState(false);
  const viewRef = useRef({ f0: 0, f1: carpet.cols });
  const cursorRef = useRef<number | null>(null);
  const sizeRef = useRef({ w: 0, h: 0 });

  const mat = useMemo(() => decodeBase64Bytes(carpet.data), [carpet.data]);
  const { rows, cols, clip_z: clipZ } = carpet;
  const step = carpet.column_axis?.step ?? 0;
  const sourceRows = carpet.source_rows || rows;
  const structures = useMemo(
    () =>
      carpet.structures?.length
        ? carpet.structures
        : [{ name: "grayordinates", start: 0, end: rows }],
    [carpet.structures, rows]
  );

  const cssVar = (n: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";

  const plotRect = useCallback(() => {
    const { w, h } = sizeRef.current;
    return { x: PAD.l, y: PAD.t, w: w - PAD.l - PAD.r, h: h - PAD.t - PAD.b };
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const off = offRef.current;
    if (!canvas || !off) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h } = sizeRef.current;
    const R = plotRect();
    const view = viewRef.current;
    const line = cssVar("--line") || cssVar("--border");
    const ink = cssVar("--text-main") || cssVar("--foreground");
    const muted = cssVar("--text-muted") || cssVar("--muted-foreground");
    const panel = cssVar("--bg-panel") || cssVar("--background");
    const bg = cssVar("--bg-main") || panel;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = panel;
    ctx.fillRect(0, 0, w, h);

    ctx.imageSmoothingEnabled = view.f1 - view.f0 > R.w;
    ctx.drawImage(off, view.f0, 0, view.f1 - view.f0, rows, R.x, R.y, R.w, R.h);

    // Structure separators.
    for (const s of structures) {
      if (s.start > 0) {
        const y0 = R.y + (s.start / rows) * R.h;
        ctx.strokeStyle = bg;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(R.x, y0);
        ctx.lineTo(R.x + R.w, y0);
        ctx.stroke();
      }
    }
    // Gutter labels (clipped so they never spill onto the heatmap).
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, R.x - 8, h);
    ctx.clip();
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const s of structures) {
      const y0 = R.y + (s.start / rows) * R.h;
      const y1 = R.y + (s.end / rows) * R.h;
      ctx.fillStyle = ink;
      ctx.font = '600 11px "Inter", system-ui, sans-serif';
      ctx.fillText(prettyStructure(s.name), R.x - 12, (y0 + y1) / 2);
      const count = Math.round(((s.end - s.start) / rows) * sourceRows);
      ctx.fillStyle = muted;
      ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillText("~" + count.toLocaleString(), R.x - 12, (y0 + y1) / 2 + 15);
    }
    ctx.restore();

    // Axis frame + ticks.
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w, R.h);
    ctx.fillStyle = muted;
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const span = view.f1 - view.f0;
    for (let i = 0; i <= 6; i++) {
      const f = view.f0 + (span * i) / 6;
      const x = R.x + (R.w * i) / 6;
      ctx.strokeStyle = line;
      ctx.beginPath();
      ctx.moveTo(x, R.y + R.h);
      ctx.lineTo(x, R.y + R.h + 4);
      ctx.stroke();
      ctx.fillText(String(Math.round(f)), x, R.y + R.h + 7);
    }
    ctx.fillStyle = muted;
    ctx.fillText(step > 0 ? "frame" : "index", R.x + R.w / 2, R.y + R.h + 21);

    // Colourbar.
    const cbx = R.x + R.w + 18;
    const cbw = 12;
    const cbh = Math.min(R.h, 220);
    const cby = R.y + (R.h - cbh) / 2;
    for (let i = 0; i < cbh; i++) {
      ctx.fillStyle = sampleColor(cmap, 1 - i / cbh);
      ctx.fillRect(cbx, cby + i, cbw, 1);
    }
    ctx.strokeStyle = line;
    ctx.strokeRect(cbx + 0.5, cby + 0.5, cbw, cbh);
    ctx.fillStyle = muted;
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`+${clipZ}σ`, cbx + cbw + 5, cby + 4);
    ctx.fillText("0", cbx + cbw + 5, cby + cbh / 2);
    ctx.fillText(`−${clipZ}σ`, cbx + cbw + 5, cby + cbh - 4);

    // Time cursor.
    const cursor = cursorRef.current;
    if (cursor != null && cursor >= view.f0 && cursor <= view.f1) {
      const x = R.x + ((cursor - view.f0) / span) * R.w;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x, R.y);
      ctx.lineTo(x, R.y + R.h);
      ctx.stroke();
    }
  }, [rows, cmap, clipZ, step, structures, sourceRows, plotRect]);

  // Recolour the offscreen native-resolution image (cols × rows) on colormap /
  // data change, then repaint. Declared after `draw` so no temporal-dead-zone.
  const paintOffscreen = useCallback(() => {
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(cols, rows);
    const px = img.data;
    const lut = COLORMAPS[cmap];
    for (let i = 0; i < rows * cols; i++) {
      const k = mat[i] * 3;
      const o = i * 4;
      px[o] = lut[k];
      px[o + 1] = lut[k + 1];
      px[o + 2] = lut[k + 2];
      px[o + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    offRef.current = off;
  }, [mat, rows, cols, cmap]);

  useEffect(() => {
    paintOffscreen();
    draw();
  }, [paintOffscreen, draw]);

  // Size to container (single canvas, explicit CSS px to match device px / dpr).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      sizeRef.current = { w: rect.width, h: rect.height };
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  useEffect(() => {
    const mo = new MutationObserver(draw);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    return () => mo.disconnect();
  }, [draw]);

  const frameAtX = (clientX: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const R = plotRect();
    const rx = clientX - rect.left;
    if (rx < R.x || rx > R.x + R.w) return null;
    const view = viewRef.current;
    return view.f0 + ((rx - R.x) / R.w) * (view.f1 - view.f0);
  };
  const rowAtY = (clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const R = plotRect();
    const ry = clientY - rect.top;
    if (ry < R.y || ry > R.y + R.h) return null;
    return Math.floor(((ry - R.y) / R.h) * rows);
  };

  const onMove = (e: React.MouseEvent) => {
    const f = frameAtX(e.clientX);
    const r = rowAtY(e.clientY);
    if (f == null || r == null) {
      setHover(null);
      return;
    }
    const frame = Math.min(cols - 1, Math.max(0, Math.floor(f)));
    const u = mat[r * cols + frame];
    const z = (u / 255) * (2 * clipZ) - clipZ;
    const s = structures.find((x) => r >= x.start && r < x.end) ?? structures[0];
    const gord = Math.round((r / rows) * sourceRows);
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    setHover({
      struct: prettyStructure(s.name),
      gord,
      frame,
      time: step > 0 ? `${(frame * step).toFixed(2)}s` : String(frame),
      z: (z >= 0 ? "+" : "") + z.toFixed(2),
      color: sampleColor(cmap, u / 255),
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const onWheel = (e: React.WheelEvent) => {
    const f = frameAtX(e.clientX);
    if (f == null) return;
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const view = viewRef.current;
    let a = f - (f - view.f0) * factor;
    let b = f + (view.f1 - f) * factor;
    a = Math.max(0, a);
    b = Math.min(cols, b);
    if (b - a >= 8) {
      viewRef.current = { f0: a, f1: b };
      setZoomed(a > 0 || b < cols);
      draw();
    }
  };

  const dragRef = useRef<{ x: number; view: { f0: number; f1: number } } | null>(null);
  const onDown = (e: React.MouseEvent) => {
    dragRef.current = { x: e.clientX, view: { ...viewRef.current } };
  };
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const R = plotRect();
      const span = dragRef.current.view.f1 - dragRef.current.view.f0;
      const df = -((e.clientX - dragRef.current.x) / R.w) * span;
      let a = dragRef.current.view.f0 + df;
      let b = dragRef.current.view.f1 + df;
      if (a < 0) {
        b -= a;
        a = 0;
      }
      if (b > cols) {
        a -= b - cols;
        b = cols;
      }
      viewRef.current = { f0: Math.max(0, a), f1: Math.min(cols, b) };
      setZoomed(viewRef.current.f0 > 0 || viewRef.current.f1 < cols);
      draw();
    };
    const up = () => {
      dragRef.current = null;
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [cols, draw, plotRect]);

  const onClick = (e: React.MouseEvent) => {
    const f = frameAtX(e.clientX);
    if (f == null) return;
    cursorRef.current = Math.round(f);
    draw();
  };

  const resetZoom = () => {
    viewRef.current = { f0: 0, f1: cols };
    cursorRef.current = null;
    setZoomed(false);
    draw();
  };

  return (
    <div className="cifti-view">
      <div className="cifti-toolbar">
        <div className="cifti-toolgroup">
          <span className="cifti-toollabel">Colormap</span>
          <div className="cifti-seg" role="group" aria-label="Colormap">
            {CMAP_ORDER.map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={cmap === key}
                onClick={() => setCmap(key)}
              >
                {COLORMAP_LABELS[key]}
              </button>
            ))}
          </div>
        </div>
        <button type="button" className="cifti-btn" onClick={resetZoom} style={{ marginLeft: "auto" }}>
          Reset zoom
        </button>
      </div>

      <div className="cifti-stage">
        <canvas
          ref={canvasRef}
          className="cifti-canvas"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onWheel={onWheel}
          onMouseDown={onDown}
          onClick={onClick}
        />
        {hover ? (
          <div className="cifti-tip" style={{ left: hover.x, top: hover.y }}>
            <span className="cifti-tip-sw" style={{ background: hover.color }} />
            {hover.struct} · gord {hover.gord.toLocaleString()}
            <br />
            {step > 0 ? "frame" : "index"} {hover.frame} · {hover.time} · z {hover.z}
          </div>
        ) : null}
      </div>

      <div className="cifti-readout">
        <span>
          <span className="cifti-k">matrix</span>{" "}
          <b>
            {rows.toLocaleString()} × {cols.toLocaleString()}
          </b>
        </span>
        {sourceRows !== rows ? (
          <span>
            <span className="cifti-k">grayordinates</span> <b>{sourceRows.toLocaleString()}</b>{" "}
            <span className="cifti-k">(sampled to {rows})</span>
          </span>
        ) : null}
        {step > 0 ? (
          <span>
            <span className="cifti-k">TR</span> <b>{step}s</b>
          </span>
        ) : null}
        <span style={{ marginLeft: "auto" }}>
          <span className="cifti-k">view</span> <b>{zoomed ? "zoomed" : "all frames"}</b>
        </span>
      </div>
    </div>
  );
}
