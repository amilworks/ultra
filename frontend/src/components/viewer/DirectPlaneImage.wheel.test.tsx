import { render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { UploadViewerInfo } from "@/types";

import { DirectPlaneImage } from "./DirectPlaneImage";

// Minimal isotropic plane descriptor (world == pixel grid).
const descriptor: UploadViewerInfo["viewer"]["default_plane"] = {
  axis: "z",
  label: "XY",
  axes: ["Y", "X"],
  pixel_size: { width: 6000, height: 4000 },
  spacing: { row: 1, col: 1 },
  world_size: { width: 6000, height: 4000 },
  aspect_ratio: 1.5,
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("DirectPlaneImage wheel handling", () => {
  it("attaches a NON-passive native wheel listener so a Mac trackpad pinch zooms the image, not the page", () => {
    // React's onWheel is registered passively, so preventDefault() there is a no-op and
    // a ctrl+wheel pinch falls through to the browser's page zoom. The fix is a native
    // non-passive listener — assert it is actually attached that way.
    const addSpy = vi.spyOn(HTMLElement.prototype, "addEventListener");

    render(
      <DirectPlaneImage imageUrl="https://example.test/v2/uploads/file-123/display" descriptor={descriptor} title="2d-plane" />
    );

    const wheelCalls = addSpy.mock.calls.filter((call) => call[0] === "wheel");
    expect(wheelCalls.length).toBeGreaterThan(0);
    const hasNonPassiveWheel = wheelCalls.some((call) => {
      const options = call[2];
      return typeof options === "object" && options !== null && (options as AddEventListenerOptions).passive === false;
    });
    expect(hasNonPassiveWheel).toBe(true);
  });
});
