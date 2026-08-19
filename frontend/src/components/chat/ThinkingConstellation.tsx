import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

// The neural constellation beside the live "Thinking" label: thirty ink dots
// on a slowly precessing sphere, edges weighted by depth, and activation
// pulses that drag a fading tail along an edge and land with a brief ring on
// the target node. Owner-picked recipe (design mock variant C at 36px).
// Decorative chrome, so it obeys the calm law — currentColor only, a static
// frame under reduced motion — and it never renders React at animation
// speed: one mount, one canvas, refs all the way down.

type ConstellationNode = { x: number; y: number; z: number };
type ConstellationEdge = { a: number; b: number };

export type ConstellationGeometry = {
  nodes: ConstellationNode[];
  edges: ConstellationEdge[];
};

// Deterministic geometry (golden-angle sphere + nearest-neighbor edges with a
// few fixed cross-links) so every mount — and the test — sees the same shape.
export const buildConstellationGeometry = (count = 30): ConstellationGeometry => {
  const nodes: ConstellationNode[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = golden * i;
    nodes.push({ x: Math.cos(angle) * radius, y, z: Math.sin(angle) * radius });
  }

  const edgeKeys = new Set<string>();
  const edges: ConstellationEdge[] = [];
  const link = (a: number, b: number) => {
    if (a === b) {
      return;
    }
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    if (edgeKeys.has(key)) {
      return;
    }
    edgeKeys.add(key);
    edges.push(a < b ? { a, b } : { a: b, b: a });
  };

  nodes.forEach((node, index) => {
    const byDistance = nodes
      .map((other, otherIndex) => ({
        otherIndex,
        d:
          (node.x - other.x) ** 2 +
          (node.y - other.y) ** 2 +
          (node.z - other.z) ** 2,
      }))
      .filter((entry) => entry.otherIndex !== index)
      .sort((left, right) => left.d - right.d);
    link(index, byDistance[0].otherIndex);
    link(index, byDistance[1].otherIndex);
  });
  // A few long chords so it reads as a network, not a wireframe ball.
  for (let i = 0; i < count; i += 6) {
    link(i, (i + Math.floor(count / 2)) % count);
  }
  return { nodes, edges };
};

type Pulse = { edge: number; start: number; duration: number; forward: boolean };
type ArrivalRing = { node: number; start: number };

// Variant C, as approved on the design mock.
const ROTATION_RADIANS_PER_SECOND = 0.3;
const BASE_TILT = 0.5;
const PRECESSION_AMPLITUDE = 0.18;
const PRECESSION_PERIOD_MS = 9000;
const DEPTH_CONTRAST = 0.8;
const EDGE_ALPHA_MAX = 0.42;
const PULSE_EVERY_MS: [number, number] = [320, 720];
const PULSE_DURATION_MS: [number, number] = [360, 560];
const MAX_PULSES = 4;
const TRAIL_FRACTION = 0.35;
const RING_DURATION_MS = 280;
const GLOW_DECAY_MS = 520;
// The radius/linewidth math was tuned at 26px on the mock; scale from there.
const REFERENCE_SIZE = 26;

export type ThinkingConstellationProps = {
  size?: number;
  className?: string;
};

export function ThinkingConstellation({ size = 36, className }: ThinkingConstellationProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = size * dpr;
    canvas.height = size * dpr;

    const geometry = buildConstellationGeometry();
    const pulses: Pulse[] = [];
    const rings: ArrivalRing[] = [];
    const glow = new Float32Array(geometry.nodes.length);
    const sizeScale = size / REFERENCE_SIZE;
    let ink = getComputedStyle(canvas).color;
    let inkReadAt = performance.now();
    let nextPulseAt = 0;
    let frameHandle = 0;
    const reduceQuery = window.matchMedia?.("(prefers-reduced-motion: reduce)");

    const between = ([low, high]: [number, number]) => low + Math.random() * (high - low);

    const drawFrame = (now: number, animate: boolean) => {
      // currentColor follows the theme; re-read at most once a second.
      if (now - inkReadAt > 1000) {
        ink = getComputedStyle(canvas).color || ink;
        inkReadAt = now;
      }
      const theta = animate ? (now / 1000) * ROTATION_RADIANS_PER_SECOND : 0.9;
      // Slow precession so the sphere reads as drifting, not rigidly spun.
      const tilt =
        BASE_TILT +
        (animate
          ? Math.sin((now / PRECESSION_PERIOD_MS) * Math.PI * 2) * PRECESSION_AMPLITUDE
          : 0);
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      const sinX = Math.sin(tilt);
      const cosX = Math.cos(tilt);
      const half = (size * dpr) / 2;
      const scale = half * 0.82;

      const projected = geometry.nodes.map((node) => {
        const x = node.x * cosT + node.z * sinT;
        const z0 = -node.x * sinT + node.z * cosT;
        const y = node.y * cosX - z0 * sinX;
        const depth = node.y * sinX + z0 * cosX;
        return { px: half + x * scale, py: half + y * scale, depth };
      });

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = ink;
      context.fillStyle = ink;
      context.lineWidth = Math.max(0.6, 0.55 * dpr);

      geometry.edges.forEach((edge) => {
        const a = projected[edge.a];
        const b = projected[edge.b];
        const towardViewer = ((a.depth + b.depth) / 2 + 1) / 2;
        context.globalAlpha = 0.08 + (EDGE_ALPHA_MAX - 0.08) * towardViewer;
        context.beginPath();
        context.moveTo(a.px, a.py);
        context.lineTo(b.px, b.py);
        context.stroke();
      });

      projected.forEach((point, index) => {
        const towardViewer = (point.depth + 1) / 2;
        const nodeGlow = glow[index];
        const alphaFloor = 0.62 - 0.5 * DEPTH_CONTRAST;
        const alpha = alphaFloor + (0.98 - alphaFloor) * towardViewer + 0.4 * nodeGlow;
        const radius =
          (0.66 + (0.6 + 0.8 * DEPTH_CONTRAST) * towardViewer + 0.9 * nodeGlow) *
          dpr *
          sizeScale;
        context.globalAlpha = Math.min(1, alpha);
        context.beginPath();
        context.arc(point.px, point.py, radius, 0, Math.PI * 2);
        context.fill();
      });

      if (!animate) {
        context.globalAlpha = 1;
        return;
      }

      // Activation pulses: a bright signal runs an edge with a fading tail,
      // and the target node fires with a glow plus a brief expanding ring.
      if (now >= nextPulseAt && pulses.length < MAX_PULSES) {
        pulses.push({
          edge: Math.floor(Math.random() * geometry.edges.length),
          start: now,
          duration: between(PULSE_DURATION_MS),
          forward: Math.random() > 0.5,
        });
        nextPulseAt = now + between(PULSE_EVERY_MS);
      }
      for (let i = pulses.length - 1; i >= 0; i -= 1) {
        const pulse = pulses[i];
        const progress = (now - pulse.start) / pulse.duration;
        const edge = geometry.edges[pulse.edge];
        const targetNode = pulse.forward ? edge.b : edge.a;
        if (progress >= 1) {
          glow[targetNode] = 1;
          rings.push({ node: targetNode, start: now });
          pulses.splice(i, 1);
          continue;
        }
        const from = projected[pulse.forward ? edge.a : edge.b];
        const to = projected[targetNode];
        const headX = from.px + (to.px - from.px) * progress;
        const headY = from.py + (to.py - from.py) * progress;
        const tailProgress = Math.max(0, progress - TRAIL_FRACTION);
        const tailX = from.px + (to.px - from.px) * tailProgress;
        const tailY = from.py + (to.py - from.py) * tailProgress;
        const trail = context.createLinearGradient(tailX, tailY, headX, headY);
        trail.addColorStop(0, "rgba(0, 0, 0, 0)");
        trail.addColorStop(1, ink);
        context.save();
        context.strokeStyle = trail;
        context.globalAlpha = 0.55;
        context.lineWidth = Math.max(0.8, 0.8 * dpr);
        context.beginPath();
        context.moveTo(tailX, tailY);
        context.lineTo(headX, headY);
        context.stroke();
        context.restore();
        context.globalAlpha = 0.95;
        context.beginPath();
        context.arc(headX, headY, 1.05 * dpr * sizeScale, 0, Math.PI * 2);
        context.fill();
      }
      for (let i = rings.length - 1; i >= 0; i -= 1) {
        const ring = rings[i];
        const progress = (now - ring.start) / RING_DURATION_MS;
        if (progress >= 1) {
          rings.splice(i, 1);
          continue;
        }
        // The ring follows its node so rotation never strands it mid-air.
        const at = projected[ring.node];
        context.globalAlpha = 0.5 * (1 - progress);
        context.lineWidth = Math.max(0.6, 0.6 * dpr);
        context.beginPath();
        context.arc(at.px, at.py, (1 + 2.6 * progress) * dpr * sizeScale, 0, Math.PI * 2);
        context.stroke();
      }
      for (let i = 0; i < glow.length; i += 1) {
        if (glow[i] > 0) {
          glow[i] = Math.max(0, glow[i] - 16.7 / GLOW_DECAY_MS);
        }
      }
      context.globalAlpha = 1;
    };

    const loop = (now: number) => {
      drawFrame(now, true);
      frameHandle = window.requestAnimationFrame(loop);
    };

    const start = () => {
      window.cancelAnimationFrame(frameHandle);
      if (reduceQuery?.matches) {
        // Reduced motion: one legible static constellation, no loop.
        drawFrame(performance.now(), false);
        return;
      }
      frameHandle = window.requestAnimationFrame(loop);
    };

    const handleVisibility = () => {
      if (document.hidden) {
        window.cancelAnimationFrame(frameHandle);
      } else {
        start();
      }
    };

    start();
    reduceQuery?.addEventListener?.("change", start);
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      window.cancelAnimationFrame(frameHandle);
      reduceQuery?.removeEventListener?.("change", start);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [size]);

  return (
    <canvas
      ref={canvasRef}
      className={cn("thinking-constellation", className)}
      style={{ width: size, height: size }}
      aria-hidden="true"
    />
  );
}
