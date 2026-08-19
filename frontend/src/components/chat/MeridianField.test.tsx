import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  MERIDIAN_INVARIANT,
  MeridianField,
  registrationMap,
} from "./MeridianField";

afterEach(cleanup);

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
    expect(
      registrationMap(
        MERIDIAN_INVARIANT.u,
        MERIDIAN_INVARIANT.v,
        width,
        height
      )
    ).toEqual([
      width * MERIDIAN_INVARIANT.u,
      height * MERIDIAN_INVARIANT.v,
    ]);

    const epsilon = 0.0001;
    for (const [u, v] of [
      [0.14, 0.12],
      [0.38, 0.44],
      [0.62, 0.44],
      [0.78, 0.6],
      [0.94, 0.76],
    ] as const) {
      const [xAfterU, yAfterU] = registrationMap(
        u + epsilon,
        v,
        width,
        height
      );
      const [xBeforeU, yBeforeU] = registrationMap(
        u - epsilon,
        v,
        width,
        height
      );
      const [xAfterV, yAfterV] = registrationMap(
        u,
        v + epsilon,
        width,
        height
      );
      const [xBeforeV, yBeforeV] = registrationMap(
        u,
        v - epsilon,
        width,
        height
      );
      const dxDu = (xAfterU - xBeforeU) / (2 * epsilon);
      const dyDu = (yAfterU - yBeforeU) / (2 * epsilon);
      const dxDv = (xAfterV - xBeforeV) / (2 * epsilon);
      const dyDv = (yAfterV - yBeforeV) / (2 * epsilon);
      expect(dxDu * dyDv - dxDv * dyDu).toBeGreaterThan(0);
    }
  });
});
