import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

/* The one impossibility. Meridian rations wonder to a single surface — this
   registration map above the welcome prompt — and to one warm invariant.

   The story is mathematical and product-specific. An orthogonal source frame
   is carried through one smooth analytic transform into a registered frame.
   Every coordinate moves except the brass observation: T(a) = a. That fixed
   point stands for the scientific fact that must survive changes of modality,
   scale, representation, and analysis.

   Static on purpose: registration is a record, not an animation. Day and
   Night are two exposures of identical geometry, drawn only from live tokens.
   The construction is analytic rather than random, so every repaint is exact. */

export const MERIDIAN_INVARIANT = { u: 0.62, v: 0.44 } as const;
const ANCHOR_U = MERIDIAN_INVARIANT.u;
const ANCHOR_V = MERIDIAN_INVARIANT.v;
const MERIDIAN_STEP = 0.08;
const PARALLEL_STEP = 0.16;
const TWIST_RADIANS = 0.46;
const TWIST_FALLOFF = 1.35;
const RADIAL_EXPANSION = 0.045;

type Point = readonly [x: number, y: number];
type CoordinateAxis = "meridian" | "parallel";
type CoordinateValue = { offset: number; value: number };

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

const rgba = ([r, g, b]: [number, number, number], alpha: number): string =>
  `rgba(${r}, ${g}, ${b}, ${alpha})`;

/**
 * Smooth polar twist with a small positive radial term. The map tends to the
 * identity away from the anchor, remains one-to-one at this bounded strength,
 * and fixes the anchor exactly because radius zero always maps to radius zero.
 */
export function registrationMap(
  u: number,
  v: number,
  width: number,
  height: number
): Point {
  const aspect = width / height;
  const sourceX = (u - ANCHOR_U) * aspect;
  const sourceY = v - ANCHOR_V;
  const radiusSquared = sourceX * sourceX + sourceY * sourceY;
  const influence = Math.exp(-radiusSquared / TWIST_FALLOFF);
  const angle = TWIST_RADIANS * influence;
  const scale = 1 + RADIAL_EXPANSION * influence;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const registeredX = (sourceX * cosine - sourceY * sine) * scale;
  const registeredY = (sourceX * sine + sourceY * cosine) * scale;

  return [
    width * (ANCHOR_U + registeredX / aspect),
    height * (ANCHOR_V + registeredY),
  ];
}

function coordinateValues(
  anchor: number,
  step: number,
  minimum: number,
  maximum: number
): CoordinateValue[] {
  const firstOffset = Math.ceil((minimum - anchor) / step);
  const lastOffset = Math.floor((maximum - anchor) / step);
  const values: CoordinateValue[] = [];
  for (let offset = firstOffset; offset <= lastOffset; offset++) {
    values.push({ offset, value: anchor + offset * step });
  }
  return values;
}

function traceCoordinate(
  context: CanvasRenderingContext2D,
  axis: CoordinateAxis,
  fixed: number,
  minimum: number,
  maximum: number,
  step: number,
  width: number,
  height: number
): void {
  context.beginPath();
  let first = true;
  for (let parameter = minimum; parameter < maximum; parameter += step) {
    const u = axis === "meridian" ? fixed : parameter;
    const v = axis === "meridian" ? parameter : fixed;
    const [x, y] = registrationMap(u, v, width, height);
    if (first) {
      context.moveTo(x, y);
      first = false;
    } else {
      context.lineTo(x, y);
    }
  }
  const endU = axis === "meridian" ? fixed : maximum;
  const endV = axis === "meridian" ? maximum : fixed;
  const [endX, endY] = registrationMap(endU, endV, width, height);
  context.lineTo(endX, endY);
}

function drawRegistrationFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  line: string,
  meridians: CoordinateValue[],
  parallels: CoordinateValue[]
): void {
  const pad = 10;
  const cornerLength = 9;
  context.strokeStyle = line;
  context.lineWidth = 1;

  for (const [x, y, horizontal, vertical] of [
    [pad, pad, 1, 1],
    [width - pad, pad, -1, 1],
    [pad, height - pad, 1, -1],
    [width - pad, height - pad, -1, -1],
  ] as const) {
    context.beginPath();
    context.moveTo(x + horizontal * cornerLength, y);
    context.lineTo(x, y);
    context.lineTo(x, y + vertical * cornerLength);
    context.stroke();
  }

  for (const { offset, value } of meridians) {
    if (Math.abs(offset) % 2 !== 0) {
      continue;
    }
    const x = value * width;
    const tick = offset === 0 ? 4 : 2.5;
    context.beginPath();
    context.moveTo(x, pad - tick);
    context.lineTo(x, pad + tick);
    context.moveTo(x, height - pad - tick);
    context.lineTo(x, height - pad + tick);
    context.stroke();
  }

  for (const { offset, value } of parallels) {
    const y = value * height;
    const tick = offset === 0 ? 4 : 2.5;
    context.beginPath();
    context.moveTo(pad - tick, y);
    context.lineTo(pad + tick, y);
    context.moveTo(width - pad - tick, y);
    context.lineTo(width - pad + tick, y);
    context.stroke();
  }
}

function drawSourceAxes(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  line: string
): void {
  const pad = 10;
  const anchorX = width * ANCHOR_U;
  const anchorY = height * ANCHOR_V;
  context.save();
  context.strokeStyle = line;
  context.lineWidth = 0.75;
  context.setLineDash([1, 5]);
  context.beginPath();
  context.moveTo(anchorX, pad);
  context.lineTo(anchorX, height - pad);
  context.moveTo(pad, anchorY);
  context.lineTo(width - pad, anchorY);
  context.stroke();
  context.restore();
}

function drawCoordinateFamily(
  context: CanvasRenderingContext2D,
  axis: CoordinateAxis,
  coordinates: CoordinateValue[],
  minimum: number,
  maximum: number,
  curveStep: number,
  width: number,
  height: number,
  ink: [number, number, number],
  night: boolean
): void {
  for (const { offset, value } of coordinates) {
    const major = offset === 0;
    const secondary = Math.abs(offset) % 2 === 0;
    const baseAlpha = axis === "meridian" ? 0.17 : 0.12;
    const themeAdjustment = night ? 0.025 : 0.01;
    const hierarchy = secondary ? 0.045 : 0;
    const majorAlpha = axis === "meridian" ? 0.46 : 0.34;
    context.strokeStyle = rgba(
      ink,
      major ? majorAlpha : baseAlpha + themeAdjustment + hierarchy
    );
    context.lineWidth = major ? 1 : secondary ? 0.85 : 0.7;
    traceCoordinate(
      context,
      axis,
      value,
      minimum,
      maximum,
      curveStep,
      width,
      height
    );
    context.stroke();
  }
}

function drawInvariantTicks(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  meridians: CoordinateValue[],
  parallels: CoordinateValue[],
  ink: [number, number, number],
  night: boolean
): void {
  context.fillStyle = rgba(ink, night ? 0.5 : 0.56);
  for (const { offset, value } of meridians) {
    if (offset === 0) {
      continue;
    }
    const [x, y] = registrationMap(value, ANCHOR_V, width, height);
    context.fillRect(x - 0.6, y - 0.6, 1.2, 1.2);
  }
  for (const { offset, value } of parallels) {
    if (offset === 0) {
      continue;
    }
    const [x, y] = registrationMap(ANCHOR_U, value, width, height);
    context.fillRect(x - 0.6, y - 0.6, 1.2, 1.2);
  }
}

function drawInvariant(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  live: [number, number, number]
): void {
  const anchorX = width * ANCHOR_U;
  const anchorY = height * ANCHOR_V;
  const half = 6;
  context.strokeStyle = rgba(live, 1);
  context.fillStyle = rgba(live, 1);
  context.lineWidth = 1.2;
  context.strokeRect(anchorX - half, anchorY - half, half * 2, half * 2);
  context.fillRect(anchorX - 1.35, anchorY - 1.35, 2.7, 2.7);
}

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
      const line = styles.getPropertyValue("--line").trim();
      if (!ink || !live || !ground) {
        return;
      }
      const night = relativeLuminance(ground) < 0.5;
      const pad = 10;
      const minimumU = pad / width;
      const maximumU = 1 - minimumU;
      const minimumV = pad / height;
      const maximumV = 1 - minimumV;
      const meridians = coordinateValues(
        ANCHOR_U,
        MERIDIAN_STEP,
        minimumU,
        maximumU
      );
      const parallels = coordinateValues(
        ANCHOR_V,
        PARALLEL_STEP,
        minimumV,
        maximumV
      );
      const precisionScale = Math.max(
        0.72,
        Math.sqrt(Math.min(1, (width * height) / 117760))
      );
      const curveStep = 0.012 / precisionScale;

      drawRegistrationFrame(
        context,
        width,
        height,
        line,
        meridians,
        parallels
      );
      drawSourceAxes(context, width, height, line);

      context.save();
      context.beginPath();
      context.rect(pad, pad, width - pad * 2, height - pad * 2);
      context.clip();
      context.lineCap = "round";
      context.lineJoin = "round";
      drawCoordinateFamily(
        context,
        "parallel",
        parallels,
        minimumU,
        maximumU,
        curveStep,
        width,
        height,
        ink,
        night
      );
      drawCoordinateFamily(
        context,
        "meridian",
        meridians,
        minimumV,
        maximumV,
        curveStep,
        width,
        height,
        ink,
        night
      );
      context.restore();

      drawInvariantTicks(
        context,
        width,
        height,
        meridians,
        parallels,
        ink,
        night
      );
      drawInvariant(context, width, height, live);
    };

    draw();
    /* Redraw only when the LAYOUT size actually changed. Setting the canvas
       width attribute inside a ResizeObserver callback is a feedback loop the
       moment CSS fails to constrain the element. The guard breaks that cycle;
       the inline max-width below caps the blast radius if it ever recurs. */
    let drawnWidth = canvas.clientWidth;
    let drawnHeight = canvas.clientHeight;
    const onResize = (): void => {
      if (
        canvas.clientWidth === drawnWidth &&
        canvas.clientHeight === drawnHeight
      ) {
        return;
      }
      drawnWidth = canvas.clientWidth;
      drawnHeight = canvas.clientHeight;
      draw();
    };
    const resizeObserver =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    resizeObserver?.observe(canvas);
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
      style={{ maxWidth: "100%" }}
      aria-hidden="true"
    />
  );
}
