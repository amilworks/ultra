import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LensSidebarIcon } from "./LensSidebarIcon";

describe("LensSidebarIcon", () => {
  it("uses the original Layers3 icon when Lens is inactive", () => {
    const { container } = render(<LensSidebarIcon aria-hidden="true" />);

    expect(container.querySelector('svg[data-lens-icon="default"]')).not.toBeNull();
    expect(container.querySelector('[data-channel-color]')).toBeNull();
  });

  it("uses the original Layers3 icon when Lens is active", () => {
    const { container } = render(<LensSidebarIcon active aria-hidden="true" />);

    expect(container.querySelector('svg[data-lens-icon="active"]')).not.toBeNull();
    expect(container.querySelector('[data-channel-color]')).toBeNull();
  });
});
