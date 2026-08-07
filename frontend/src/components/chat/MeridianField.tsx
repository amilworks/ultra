import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

/* The one impossibility. Meridian rations wonder to a single surface — this
   field above the welcome prompt — and to a single warm point inside it.
   Night draws the deep field: silver magnitudes thinning toward the lower
   edge (atmospheric extinction), the meridian reference line, and one brass
   point with diffraction spikes. Day draws the PLATE — the same sky as a
   measured negative: réseau grid, dark specks, halation blooming the bright
   ones, and the brass ring marking the object under measurement.

   Static on purpose: a long exposure is a record of time, not an animation.
   Everything derives from the live theme tokens at draw time, and the drawing
   is deterministic (seeded), so theme flips and resizes repaint the identical
   sky. */

const SEED = 20260806;

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Parses #rrggbb or rgb()/rgba() strings; returns null for anything else. */
function parseColor(value: string): [number, number, number] | null {
  const hex = value.trim().match(/^#([0-9a-f]{6})$/i);
  if (hex) {
    const n = parseInt(hex[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const channels = value.match(/\d+(\.\d+)?/g);
  if (!channels || channels.length < 3) {
    return null;
  }
  return [Number(channels[0]), Number(channels[1]), Number(channels[2])];
}

function relativeLuminance([r, g, b]: [number, number, number]): number {
  const lin = (c: number): number => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

const rgba = ([r, g, b]: [number, number, number], a: number): string =>
  `rgba(${r}, ${g}, ${b}, ${a})`;

export function MeridianField({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const draw = (): void => {
      const context = canvas.getContext("2d");
      if (!context) {
        return; // jsdom and other non-rendering environments
      }
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width === 0 || height === 0) {
        return;
      }
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      const styles = getComputedStyle(canvas);
      const ink = parseColor(styles.color);
      const live = parseColor(styles.getPropertyValue("--accent-live"));
      const ground = parseColor(styles.getPropertyValue("--bg-main"));
      // --line is an rgba() string and is used verbatim as a strokeStyle.
      const line = styles.getPropertyValue("--line").trim();
      if (!ink || !live || !ground) {
        return;
      }
      const night = relativeLuminance(ground) < 0.5;
      const rand = createRng(SEED);
      const anchorX = width * 0.62;
      const anchorY = height * (night ? 0.4 : 0.44);

      /* Density scales with AREA, so a wide field never reads sparse and a
         phone-width one never reads crowded — calibrated to the original
         544x164 drawing (~110 stars). */
      const starCount = Math.round(Math.min(150, Math.max(46, (width * height) / 800)));
      const speckCount = Math.round(Math.min(160, Math.max(50, (width * height) / 740)));

      if (night) {
        /* The deep field: magnitudes, extinction, one point of light. */
        for (let i = 0; i < starCount; i++) {
          const x = rand() * width;
          const y = rand() * height;
          const magnitude = Math.pow(rand(), 3);
          const extinction = 1 - 0.35 * (y / height);
          context.globalAlpha = (0.18 + 0.72 * magnitude) * extinction;
          context.fillStyle = rgba(ink, 1);
          context.beginPath();
          context.arc(x, y, 0.4 + magnitude * 1.3, 0, Math.PI * 2);
          context.fill();
        }
        context.globalAlpha = 1;

        /* The meridian — the one definite line — with transit ticks. */
        context.strokeStyle = line;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(anchorX, 0);
        context.lineTo(anchorX, height);
        context.stroke();
        for (let y = 10, tick = 0; y < height; y += 14, tick++) {
          const half = tick % 2 === 0 ? 7 : 4;
          context.beginPath();
          context.moveTo(anchorX - half, y);
          context.lineTo(anchorX + half, y);
          context.stroke();
        }

        /* The point of light: halo, diffraction spikes, hot core. */
        const halo = context.createRadialGradient(
          anchorX,
          anchorY,
          0,
          anchorX,
          anchorY,
          26
        );
        halo.addColorStop(0, rgba(live, 0.32));
        halo.addColorStop(1, rgba(live, 0));
        context.fillStyle = halo;
        context.beginPath();
        context.arc(anchorX, anchorY, 26, 0, Math.PI * 2);
        context.fill();
        context.lineWidth = 0.9;
        for (const [dx, dy] of [
          [1, 0],
          [0, 1],
        ] as const) {
          const spike = context.createLinearGradient(
            anchorX - dx * 17,
            anchorY - dy * 17,
            anchorX + dx * 17,
            anchorY + dy * 17
          );
          spike.addColorStop(0, rgba(live, 0));
          spike.addColorStop(0.5, rgba(live, 0.75));
          spike.addColorStop(1, rgba(live, 0));
          context.strokeStyle = spike;
          context.beginPath();
          context.moveTo(anchorX - dx * 17, anchorY - dy * 17);
          context.lineTo(anchorX + dx * 17, anchorY + dy * 17);
          context.stroke();
        }
        context.fillStyle = rgba(live, 1);
        context.beginPath();
        context.arc(anchorX, anchorY, 2.4, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = rgba(ink, 1);
        context.beginPath();
        context.arc(anchorX, anchorY, 0.9, 0, Math.PI * 2);
        context.fill();
      } else {
        /* The plate: réseau, corner fiducials, dark specks, the marked object. */
        const pad = 10;
        context.strokeStyle = line;
        context.lineWidth = 1;
        for (let x = pad + 26; x < width - pad; x += 26) {
          context.beginPath();
          context.moveTo(x, pad);
          context.lineTo(x, height - pad);
          context.stroke();
        }
        for (let y = pad + 26; y < height - pad; y += 26) {
          context.beginPath();
          context.moveTo(pad, y);
          context.lineTo(width - pad, y);
          context.stroke();
        }
        context.strokeRect(pad, pad, width - pad * 2, height - pad * 2);
        context.strokeStyle = rgba(ink, 0.4);
        for (const [cx, cy] of [
          [pad, pad],
          [width - pad, pad],
          [pad, height - pad],
          [width - pad, height - pad],
        ] as const) {
          context.beginPath();
          context.moveTo(cx - 5, cy);
          context.lineTo(cx + 5, cy);
          context.moveTo(cx, cy - 5);
          context.lineTo(cx, cy + 5);
          context.stroke();
        }

        const softBlur = "filter" in context;
        for (let i = 0; i < speckCount; i++) {
          const x = pad + rand() * (width - pad * 2);
          const y = pad + rand() * (height - pad * 2);
          const magnitude = Math.pow(rand(), 3);
          if (magnitude > 0.55 && softBlur) {
            /* Halation: the emulsion blooms around the brightest objects. */
            context.save();
            context.filter = "blur(1.5px)";
            context.globalAlpha = 0.3;
            context.fillStyle = rgba(ink, 1);
            context.beginPath();
            context.arc(x, y, (0.4 + magnitude * 1.3) * 2.2, 0, Math.PI * 2);
            context.fill();
            context.restore();
          }
          context.globalAlpha = 0.25 + 0.65 * magnitude;
          context.fillStyle = rgba(ink, 1);
          context.beginPath();
          context.arc(x, y, 0.4 + magnitude * 1.3, 0, Math.PI * 2);
          context.fill();
        }
        context.globalAlpha = 1;

        /* The observer's annotation: this is the one being measured. */
        context.fillStyle = rgba(ink, 1);
        context.beginPath();
        context.arc(anchorX, anchorY, 2.2, 0, Math.PI * 2);
        context.fill();
        context.strokeStyle = rgba(live, 1);
        context.lineWidth = 1.3;
        context.beginPath();
        context.arc(anchorX, anchorY, 8.5, 0, Math.PI * 2);
        context.stroke();
        for (const [dx, dy] of [
          [0, -1],
          [0, 1],
          [-1, 0],
          [1, 0],
        ] as const) {
          context.beginPath();
          context.moveTo(anchorX + dx * 10.5, anchorY + dy * 10.5);
          context.lineTo(anchorX + dx * 14, anchorY + dy * 14);
          context.stroke();
        }
      }
    };

    draw();
    /* Redraw only when the LAYOUT size actually changed. Setting the canvas
       width attribute inside a ResizeObserver callback is a feedback loop the
       moment CSS fails to constrain the element (a dev-server hiccup did
       exactly this once: attribute width at 2x dpr became the natural width,
       which re-fired the observer, which doubled it again to full-bleed).
       The guard breaks the cycle; the inline max-width below caps the blast
       radius if it ever recurs. */
    let drawnWidth = canvas.clientWidth;
    let drawnHeight = canvas.clientHeight;
    const onResize = (): void => {
      if (canvas.clientWidth === drawnWidth && canvas.clientHeight === drawnHeight) {
        return;
      }
      drawnWidth = canvas.clientWidth;
      drawnHeight = canvas.clientHeight;
      draw();
    };
    const resizeObserver =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    resizeObserver?.observe(canvas);
    // Theme flips restamp the html class; the drawing re-reads tokens then.
    const themeObserver =
      typeof MutationObserver !== "undefined" ? new MutationObserver(draw) : null;
    themeObserver?.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => {
      resizeObserver?.disconnect();
      themeObserver?.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={cn("meridian-field", className)}
      // Belt for the observer guard's braces: even with zero stylesheet the
      // attribute-sized canvas can never outgrow its container.
      style={{ maxWidth: "100%" }}
      aria-hidden="true"
    />
  );
}
