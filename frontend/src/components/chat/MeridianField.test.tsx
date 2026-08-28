import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MERIDIAN_INVARIANT,
  MERIDIAN_SOLVE_DURATION_MS,
  MeridianField,
  registrationMap,
  registrationMapAtPhase,
  registrationSolvePhase,
} from "./MeridianField";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("MeridianField", () => {
  it("renders an inert, hidden canvas and survives environments without 2d context", () => {
    // jsdom's getContext returns null; the component must treat a
    // non-rendering environment as a no-op, not a crash.
    const { container, unmount } = render(<MeridianField className="extra" />);
    const canvas = container.querySelector("canvas");
    expect(canvas).toBeTruthy();
    expect(canvas?.getAttribute("aria-hidden")).toBe("true");
    expect(canvas?.className).toContain("meridian-field");
    expect(canvas?.className).toContain("extra");
    unmount();
  });

  it("fixes its brass invariant and preserves orientation across the map", () => {
    const width = 736;
    const height = 115;
    for (const phase of [0, 0.25, 0.5, 0.75, 1]) {
      expect(
        registrationMapAtPhase(
          MERIDIAN_INVARIANT.u,
          MERIDIAN_INVARIANT.v,
          width,
          height,
          phase
        )
      ).toEqual([
        width * MERIDIAN_INVARIANT.u,
        height * MERIDIAN_INVARIANT.v,
      ]);
    }

    expect(registrationMapAtPhase(0.31, 0.72, width, height, 0)[0]).toBeCloseTo(
      width * 0.31
    );
    expect(registrationMapAtPhase(0.31, 0.72, width, height, 0)[1]).toBeCloseTo(
      height * 0.72
    );
    expect(registrationMapAtPhase(0.31, 0.72, width, height, 1)).toEqual(
      registrationMap(0.31, 0.72, width, height)
    );

    const epsilon = 0.0001;
    for (const phase of [0, 0.25, 0.5, 0.75, 1]) {
      for (const [u, v] of [
        [0.14, 0.12],
        [0.38, 0.44],
        [0.62, 0.44],
        [0.78, 0.6],
        [0.94, 0.76],
      ] as const) {
        const [xAfterU, yAfterU] = registrationMapAtPhase(
          u + epsilon,
          v,
          width,
          height,
          phase
        );
        const [xBeforeU, yBeforeU] = registrationMapAtPhase(
          u - epsilon,
          v,
          width,
          height,
          phase
        );
        const [xAfterV, yAfterV] = registrationMapAtPhase(
          u,
          v + epsilon,
          width,
          height,
          phase
        );
        const [xBeforeV, yBeforeV] = registrationMapAtPhase(
          u,
          v - epsilon,
          width,
          height,
          phase
        );
        const dxDu = (xAfterU - xBeforeU) / (2 * epsilon);
        const dyDu = (yAfterU - yBeforeU) / (2 * epsilon);
        const dxDv = (xAfterV - xBeforeV) / (2 * epsilon);
        const dyDv = (yAfterV - yBeforeV) / (2 * epsilon);
        expect(dxDu * dyDv - dxDv * dyDu).toBeGreaterThan(0);
      }
    }
  });

  it("solves once with bounded, endpoint-stationary phase timing", () => {
    expect(registrationSolvePhase(-100)).toBe(0);
    expect(registrationSolvePhase(0)).toBe(0);
    expect(registrationSolvePhase(MERIDIAN_SOLVE_DURATION_MS / 2)).toBeCloseTo(
      0.5
    );
    expect(registrationSolvePhase(MERIDIAN_SOLVE_DURATION_MS)).toBe(1);
    expect(registrationSolvePhase(MERIDIAN_SOLVE_DURATION_MS * 2)).toBe(1);

    const phases = [0, 0.2, 0.4, 0.6, 0.8, 1].map((fraction) =>
      registrationSolvePhase(MERIDIAN_SOLVE_DURATION_MS * fraction)
    );
    expect(phases).toEqual([...phases].sort((a, b) => a - b));
  });

  it("replays only when the new-chat phase key changes", () => {
    const context = {
      clearRect: vi.fn(),
      setTransform: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(736);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(115);
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({ matches: false })
    );

    let nextFrame = 1;
    const frames = new Map<number, FrameRequestCallback>();
    const requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      const frame = nextFrame++;
      frames.set(frame, callback);
      return frame;
    });
    const cancelAnimationFrame = vi.fn((frame: number) => frames.delete(frame));
    vi.stubGlobal("requestAnimationFrame", requestAnimationFrame);
    vi.stubGlobal("cancelAnimationFrame", cancelAnimationFrame);
    const performanceNow = vi.spyOn(performance, "now").mockReturnValue(100);

    const runNextFrame = (timestamp: number): void => {
      const entry = frames.entries().next().value as
        | [number, FrameRequestCallback]
        | undefined;
      expect(entry).toBeTruthy();
      if (!entry) {
        return;
      }
      frames.delete(entry[0]);
      entry[1](timestamp);
    };

    const { rerender, unmount } = render(
      <MeridianField phaseKey={4} solveStartedAtMs={100} />
    );
    expect(frames).toHaveLength(1);
    runNextFrame(100);
    runNextFrame(1_500);
    runNextFrame(2_900);
    expect(frames).toHaveLength(0);

    rerender(<MeridianField phaseKey={4} solveStartedAtMs={100} />);
    expect(frames).toHaveLength(0);

    rerender(<MeridianField phaseKey={5} solveStartedAtMs={3_000} />);
    expect(frames).toHaveLength(1);
    expect(requestAnimationFrame).toHaveBeenCalledTimes(4);

    // Conversation hydration can briefly unmount the welcome stage. The solve
    // origin lives above it, so a later remount settles instead of replaying.
    unmount();
    performanceNow.mockReturnValue(6_000);
    render(<MeridianField phaseKey={5} solveStartedAtMs={3_000} />);
    expect(frames).toHaveLength(0);
    expect(requestAnimationFrame).toHaveBeenCalledTimes(4);
  });

  it("draws the settled registration immediately for reduced motion", () => {
    const context = {
      clearRect: vi.fn(),
      setTransform: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(736);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(115);
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({ matches: true })
    );
    const requestAnimationFrame = vi.fn();
    vi.stubGlobal("requestAnimationFrame", requestAnimationFrame);

    render(<MeridianField phaseKey={1} solveStartedAtMs={0} />);

    expect(context.clearRect).toHaveBeenCalledOnce();
    expect(requestAnimationFrame).not.toHaveBeenCalled();
  });
});
