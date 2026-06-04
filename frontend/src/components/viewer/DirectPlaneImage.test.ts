import { describe, expect, it } from "vitest";

import {
  clampImageViewport,
  computeImageFitScale,
  panImageViewport,
  wheelDeltaToZoomFactor,
  zoomImageViewportAtPoint,
} from "./DirectPlaneImage";

describe("DirectPlaneImage viewport math", () => {
  it("fits the natural bitmap aspect without stretching it to the container aspect", () => {
    const image = { width: 4000, height: 1000 };
    const viewport = { width: 1000, height: 500 };

    const fitScale = computeImageFitScale(image, viewport);

    expect(fitScale).toBe(0.25);
    expect((image.width * fitScale) / (image.height * fitScale)).toBe(4);
  });

  it("keeps cursor-centered zoom anchored to the same image point", () => {
    const image = { width: 1000, height: 800 };
    const viewport = { width: 500, height: 400 };
    const current = { centerX: 0, centerY: 0, scale: 0.5 };
    const pointer = { x: 375, y: 100 };
    const worldBefore = {
      x: current.centerX + (pointer.x - viewport.width / 2) / current.scale,
      y: current.centerY - (pointer.y - viewport.height / 2) / current.scale,
    };

    const next = zoomImageViewportAtPoint(current, pointer, 1, image, viewport, 0.25, 16);
    const worldAfter = {
      x: next.centerX + (pointer.x - viewport.width / 2) / next.scale,
      y: next.centerY - (pointer.y - viewport.height / 2) / next.scale,
    };

    expect(worldAfter.x).toBeCloseTo(worldBefore.x, 5);
    expect(worldAfter.y).toBeCloseTo(worldBefore.y, 5);
  });

  it("clamps pan so the image cannot drift outside the orthographic viewport", () => {
    const image = { width: 1000, height: 800 };
    const viewport = { width: 500, height: 400 };

    const next = panImageViewport(
      { centerX: 0, centerY: 0, scale: 1 },
      { x: -1000, y: -1000 },
      image,
      viewport,
      0.25,
      16
    );

    expect(next.centerX).toBe(250);
    expect(next.centerY).toBe(-200);
  });

  it("recenters an axis when the fitted view is larger than the image on that axis", () => {
    const next = clampImageViewport(
      { centerX: 300, centerY: 300, scale: 0.5 },
      { width: 1000, height: 800 },
      { width: 1000, height: 900 },
      0.25,
      16
    );

    expect(next.centerX).toBe(0);
    expect(next.centerY).toBe(0);
  });

  it("maps small macOS pinch deltas to smooth incremental zoom factors", () => {
    const zoomIn = wheelDeltaToZoomFactor(-5, 0, true, 600);
    const zoomOut = wheelDeltaToZoomFactor(5, 0, true, 600);

    expect(zoomIn).toBeGreaterThan(1);
    expect(zoomIn).toBeGreaterThan(1.01);
    expect(zoomIn).toBeLessThan(1.03);
    expect(zoomOut).toBeGreaterThan(0.98);
    expect(zoomOut).toBeLessThan(1);
  });

  it("normalizes coarse wheel deltas without jumping by a fixed zoom step", () => {
    const wheelZoomIn = wheelDeltaToZoomFactor(-120, 0, false, 600);
    const lineZoomOut = wheelDeltaToZoomFactor(3, 1, false, 600);

    expect(wheelZoomIn).toBeGreaterThan(1.1);
    expect(wheelZoomIn).toBeLessThan(1.2);
    expect(lineZoomOut).toBeGreaterThan(0.94);
    expect(lineZoomOut).toBeLessThan(1);
  });
});
