/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  ThinkingConstellation,
  buildConstellationGeometry,
} from "./ThinkingConstellation";

const componentSource = readFileSync(
  path.join(process.cwd(), "src/components/chat/ThinkingConstellation.tsx"),
  "utf8"
);

describe("buildConstellationGeometry", () => {
  it("is deterministic and shaped like a small network, not a wireframe ball", () => {
    const first = buildConstellationGeometry();
    const second = buildConstellationGeometry();
    expect(second).toEqual(first);
    /* Variant C (owner-approved on the design mock): 30 nodes. */
    expect(first.nodes).toHaveLength(30);
    /* Every node sits on the unit sphere (the projection math depends on it). */
    for (const node of first.nodes) {
      expect(node.x ** 2 + node.y ** 2 + node.z ** 2).toBeCloseTo(1, 6);
    }
    /* Nearest-neighbor links + a few chords: connected enough to read as a
       network, sparse enough to stay calm at header size. */
    expect(first.edges.length).toBeGreaterThanOrEqual(first.nodes.length);
    expect(first.edges.length).toBeLessThan(first.nodes.length * 3);
    const keys = first.edges.map((edge) => `${edge.a}:${edge.b}`);
    expect(new Set(keys).size).toBe(keys.length);
    for (const edge of first.edges) {
      expect(edge.a).toBeLessThan(edge.b);
    }
  });
});

describe("ThinkingConstellation", () => {
  it("renders a decorative canvas and survives environments without 2d context", () => {
    /* jsdom returns null from getContext — the component must no-op, not
       throw, and unmount must stay clean (the effect returns early). */
    const { container, unmount } = render(<ThinkingConstellation />);
    const canvas = container.querySelector("canvas");
    expect(canvas).not.toBeNull();
    expect(canvas).toHaveAttribute("aria-hidden", "true");
    expect(canvas?.className).toContain("thinking-constellation");
    unmount();
  });

  it("keeps the calm-law contracts in source", () => {
    /* Chrome draws in currentColor only — a hex literal here is a palette
       fork. Motion must be rAF-driven, honor reduced motion with a static
       frame, and always cancel on unmount. */
    expect(componentSource).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
    expect(componentSource).toMatch(/prefers-reduced-motion/);
    expect(componentSource).toMatch(/cancelAnimationFrame/);
    expect(componentSource).toMatch(/requestAnimationFrame/);
    expect(componentSource).not.toMatch(/setInterval/);
    expect(componentSource).toMatch(/aria-hidden="true"/);
    expect(componentSource).toMatch(/visibilitychange/);
  });

  it("ships the approved variant-C recipe at 36px", () => {
    /* The owner picked C at 36 on the design mock; these are its signature
       moves. Changing them is a design decision, not a refactor. */
    expect(componentSource).toMatch(/size = 36/);
    expect(componentSource).toMatch(/PRECESSION_AMPLITUDE = 0.18/);
    expect(componentSource).toMatch(/createLinearGradient/); // signal trails
    expect(componentSource).toMatch(/RING_DURATION_MS = 280/); // arrival rings
    expect(componentSource).toMatch(/MAX_PULSES = 4/);
    /* The trail gradient's transparent stop is alpha-only, not a hue. */
    expect(componentSource).toMatch(/rgba\(0, 0, 0, 0\)/);
  });
});
