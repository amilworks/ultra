import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CiftiConnectivityResponse } from "@/types";
import { COLORMAPS, prettyStructure, sampleColor } from "./colormaps";
import { scheduleFontsReadyRedraw } from "./fontReadyRedraw";

type Props = {
  conn: CiftiConnectivityResponse;
};

type Hover = { i: number; j: number; v: string; li: string; lj: string; color: string; x: number; y: number };

const PAD = { l: 132, r: 64, t: 14, b: 40 };

// A functional-connectivity matrix: an N×N symmetric heatmap. Correlation is
// centred at 0 on a diverging map (blue negative, red positive). For a *conn
// file this is the stored matrix; for a timeseries it is computed here from
// downsampled node signals. Hover reads the pair and its value.
export function CiftiConnectivity({ conn }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const sizeRef = useRef({ w: 0, h: 0 });

  const values = useMemo(() => {
    const raw = atob(conn.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new Float32Array(bytes.buffer, bytes.byteOffset, conn.n * conn.n);
  }, [conn.data, conn.n]);
  const n = conn.n;
  const magnitude = Math.max(Math.abs(conn.min), Math.abs(conn.max), 0.1);

  // Group nodes into contiguous structure bands for the separators + labels.
  // Only meaningful when a few anatomical structures each span many nodes — for
  // parcellated data every node is its own parcel, which would smear 100+ labels
  // down the gutter, so we suppress bands unless they're genuine groupings.
  const bands = useMemo(() => {
    if (!conn.labels?.length) return [] as { name: string; start: number; end: number }[];
    const out: { name: string; start: number; end: number }[] = [];
    let start = 0;
    for (let i = 1; i <= conn.labels.length; i++) {
      if (i === conn.labels.length || conn.labels[i] !== conn.labels[start]) {
        out.push({ name: conn.labels[start], start, end: i });
        start = i;
      }
    }
    const meaningful = out.length > 1 && out.length <= 16 && out.some((b) => b.end - b.start > 1);
    return meaningful ? out : [];
  }, [conn.labels]);

  const cssVar = (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";

  const plotRect = useCallback(() => {
    const { w, h } = sizeRef.current;
    // Square plot, centred in the available area.
    const side = Math.min(w - PAD.l - PAD.r, h - PAD.t - PAD.b);
    return { x: PAD.l, y: PAD.t, w: Math.max(side, 0), h: Math.max(side, 0) };
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const off = offRef.current;
    if (!canvas || !off) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h } = sizeRef.current;
    const R = plotRect();
    const line = cssVar("--line") || cssVar("--border");
    const ink = cssVar("--text-main") || cssVar("--foreground");
    const muted = cssVar("--text-muted") || cssVar("--muted-foreground");
    const panel = cssVar("--bg-panel") || cssVar("--background");
    const bg = cssVar("--bg-main") || panel;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = panel;
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, n, n, R.x, R.y, R.w, R.h);
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w, R.h);

    // Structure band separators (both axes) + left/bottom labels.
    ctx.font =
      '600 10px "Ultra Sans", "BisQue Inter Variable", system-ui, sans-serif';
    for (const b of bands) {
      const p0 = (b.start / n) * R.w;
      if (b.start > 0) {
        ctx.strokeStyle = bg;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(R.x + p0, R.y);
        ctx.lineTo(R.x + p0, R.y + R.h);
        ctx.moveTo(R.x, R.y + p0);
        ctx.lineTo(R.x + R.w, R.y + p0);
        ctx.stroke();
      }
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, R.x - 8, h);
    ctx.clip();
    ctx.fillStyle = ink;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const b of bands) {
      const mid = R.y + ((b.start + b.end) / 2 / n) * R.h;
      ctx.fillText(prettyStructure(b.name), R.x - 10, mid);
    }
    ctx.restore();

    // Colourbar.
    const cbx = R.x + R.w + 18;
    const cbw = 12;
    const cbh = Math.min(R.h, 220);
    const cby = R.y + (R.h - cbh) / 2;
    for (let i = 0; i < cbh; i++) {
      ctx.fillStyle = sampleColor("rdbu", 1 - i / cbh);
      ctx.fillRect(cbx, cby + i, cbw, 1);
    }
    ctx.strokeStyle = line;
    ctx.strokeRect(cbx + 0.5, cby + 0.5, cbw, cbh);
    ctx.fillStyle = muted;
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`+${magnitude.toFixed(2)}`, cbx + cbw + 5, cby + 4);
    ctx.fillText("0", cbx + cbw + 5, cby + cbh / 2);
    ctx.fillText(`−${magnitude.toFixed(2)}`, cbx + cbw + 5, cby + cbh - 4);
  }, [n, bands, magnitude, plotRect]);

  // Recolour the offscreen native-resolution matrix, then repaint. Declared after
  // `draw` to avoid a temporal-dead-zone reference.
  const paintOffscreen = useCallback(() => {
    const off = document.createElement("canvas");
    off.width = n;
    off.height = n;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(n, n);
    const px = img.data;
    const lut = COLORMAPS.rdbu;
    for (let i = 0; i < n * n; i++) {
      const t = (values[i] + magnitude) / (2 * magnitude);
      const k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
      const o = i * 4;
      px[o] = lut[k];
      px[o + 1] = lut[k + 1];
      px[o + 2] = lut[k + 2];
      px[o + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    offRef.current = off;
  }, [values, n, magnitude]);

  useEffect(() => {
    paintOffscreen();
    draw();
  }, [paintOffscreen, draw]);

  useEffect(
    () =>
      scheduleFontsReadyRedraw(
        document.fonts,
        [
          {
            query: '600 10px "Ultra Sans"',
            sample: "CORTEX_LEFT",
          },
          {
            query: '400 10px "JetBrains Mono"',
            sample: "+0.75 0 −0.75",
          },
        ],
        draw
      ),
    [draw]
  );

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

  const onMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const R = plotRect();
    const rx = e.clientX - rect.left;
    const ry = e.clientY - rect.top;
    if (rx < R.x || rx > R.x + R.w || ry < R.y || ry > R.y + R.h || R.w <= 0) {
      setHover(null);
      return;
    }
    const j = Math.min(n - 1, Math.floor(((rx - R.x) / R.w) * n));
    const i = Math.min(n - 1, Math.floor(((ry - R.y) / R.h) * n));
    const v = values[i * n + j];
    const t = (v + magnitude) / (2 * magnitude);
    setHover({
      i,
      j,
      v: (v >= 0 ? "+" : "") + v.toFixed(3),
      li: conn.labels ? prettyStructure(conn.labels[i] ?? `node ${i}`) : `node ${i}`,
      lj: conn.labels ? prettyStructure(conn.labels[j] ?? `node ${j}`) : `node ${j}`,
      color: sampleColor("rdbu", t),
      x: rx,
      y: ry,
    });
  };

  return (
    <div className="cifti-view">
      <div className="cifti-toolbar">
        <span className="cifti-toollabel">
          {conn.computed ? "Correlation (computed from timeseries)" : "Stored connectivity matrix"}
        </span>
        <span className="cifti-toollabel" style={{ marginLeft: "auto" }}>
          {n} × {n} nodes
        </span>
      </div>
      <div className="cifti-stage">
        <canvas ref={canvasRef} className="cifti-canvas" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        {hover ? (
          <div className="cifti-tip" style={{ left: hover.x, top: hover.y }}>
            <span className="cifti-tip-sw" style={{ background: hover.color }} />
            r = {hover.v}
            <br />
            {hover.li} × {hover.lj}
          </div>
        ) : null}
      </div>
      <div className="cifti-readout">
        <span>
          <span className="cifti-k">range</span>{" "}
          <b>
            {conn.min.toFixed(2)} … {conn.max.toFixed(2)}
          </b>
        </span>
        <span style={{ marginLeft: "auto" }}>
          <span className="cifti-k">source</span> <b>{conn.computed ? "computed" : "stored"}</b>
        </span>
      </div>
    </div>
  );
}
