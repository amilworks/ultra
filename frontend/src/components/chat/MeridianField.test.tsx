import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MeridianField } from "./MeridianField";

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
});
