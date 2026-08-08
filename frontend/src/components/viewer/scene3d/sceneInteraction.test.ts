import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createSceneInteractionController,
  isContinuousSceneFrameDue,
  SCENE_CONTINUOUS_FRAME_INTERVAL_MS,
  SCENE_INTERACTION_SETTLE_MS,
} from "./sceneInteraction";

describe("isContinuousSceneFrameDue", () => {
  it("skips duplicate 120 Hz work while preserving the next ~60 Hz frame", () => {
    expect(isContinuousSceneFrameDue(Number.NEGATIVE_INFINITY, 0)).toBe(true);
    expect(isContinuousSceneFrameDue(0, 8.3)).toBe(false);
    expect(isContinuousSceneFrameDue(0, SCENE_CONTINUOUS_FRAME_INTERVAL_MS)).toBe(true);
  });

  it("fails open for invalid or reset clocks so a requested frame is never lost", () => {
    expect(isContinuousSceneFrameDue(100, 90)).toBe(true);
    expect(isContinuousSceneFrameDue(100, Number.NaN)).toBe(true);
  });
});

describe("createSceneInteractionController", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps wheel interaction active after OrbitControls emits synchronous start/end", () => {
    const changes: boolean[] = [];
    const controller = createSceneInteractionController((active) => changes.push(active));

    controller.start();
    controller.end();

    expect(changes).toEqual([true]);
    vi.advanceTimersByTime(SCENE_INTERACTION_SETTLE_MS - 1);
    expect(changes).toEqual([true]);
    vi.advanceTimersByTime(1);
    expect(changes).toEqual([true, false]);
  });

  it("extends the deadline across a burst without toggling full detail between wheel events", () => {
    const changes: boolean[] = [];
    const controller = createSceneInteractionController((active) => changes.push(active));

    controller.start();
    controller.end();
    vi.advanceTimersByTime(SCENE_INTERACTION_SETTLE_MS - 40);
    controller.start();
    controller.end();
    vi.advanceTimersByTime(41);

    expect(changes).toEqual([true]);
    vi.advanceTimersByTime(SCENE_INTERACTION_SETTLE_MS - 41);
    expect(changes).toEqual([true, false]);
  });

  it("cancels a pending transition when the canvas is disposed", () => {
    const changes: boolean[] = [];
    const controller = createSceneInteractionController((active) => changes.push(active));

    controller.start();
    controller.end();
    controller.dispose();
    vi.runAllTimers();

    expect(changes).toEqual([true]);
  });
});
