import * as React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Sidebar,
  SidebarContent,
  SidebarProvider,
  SidebarTrigger,
} from "./sidebar";

const setViewportWidth = (width: number): void => {
  Object.defineProperty(window, "innerWidth", {
    value: width,
    writable: true,
    configurable: true,
  });
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => {
      const breakpoint = Number(query.match(/max-width:\s*(\d+)px/)?.[1] ?? width);
      return {
        matches: width <= breakpoint,
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      };
    })
  );
};

function MobileSidebarActionHarness() {
  const [panel, setPanel] = React.useState("chat");

  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarContent>
          <button
            type="button"
            data-sidebar-close-mobile="true"
            onClick={() => setPanel("resources")}
          >
            Resources
          </button>
        </SidebarContent>
      </Sidebar>
      <main>
        <SidebarTrigger aria-label="Open navigation" />
        <p>Panel: {panel}</p>
      </main>
    </SidebarProvider>
  );
}

describe("Sidebar", () => {
  beforeEach(() => {
    setViewportWidth(430);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("lets mobile close actions run before dismissing the sheet", async () => {
    render(<MobileSidebarActionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    fireEvent.click(await screen.findByRole("button", { name: "Resources" }));

    expect(screen.getByText("Panel: resources")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Resources" })).not.toBeInTheDocument();
    });
  });
});
