import { render, screen, fireEvent, within } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { FigureLightboxOverlay } from "./FigureLightboxOverlay";
import type { LightboxFigure } from "@/lib/figureLightbox";

beforeAll(() => {
  if (!window.matchMedia) {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
      }),
    });
  }
});

const FIGURES: LightboxFigure[] = [
  { url: "/fig-a.png", downloadUrl: "/fig-a.png?dl=1", title: "0210 Heatmap Panel" },
  { url: "/fig-b.png", title: "0250 Tile Uncertainty" },
  { url: "/fig-c.png", title: "Survey Uncertainty Heatmap" },
];

describe("FigureLightboxOverlay", () => {
  it("opens at the given index and steps through the set", () => {
    render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={vi.fn()} />);
    expect(screen.getByText("1 / 3")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Next figure"));
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Previous figure"));
    expect(screen.getByText("1 / 3")).toBeInTheDocument();
  });

  it("zoom-in increases the zoom readout", () => {
    render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={vi.fn()} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Zoom in"));
    expect(screen.getByText("125%")).toBeInTheDocument();
  });

  it("compare mode shows two figure panes side by side", () => {
    const { container } = render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={vi.fn()} />);
    // Radix portals to body; query the whole document.
    fireEvent.click(screen.getByRole("button", { name: /compare/i }));
    const panes = document.querySelectorAll(".figure-lightbox-pane");
    expect(panes.length).toBe(2);
    expect(container).toBeDefined();
  });

  it("download link points at the figure's download url", () => {
    render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={vi.fn()} />);
    const download = screen.getByLabelText("Download figure") as HTMLAnchorElement;
    expect(download.getAttribute("href")).toBe("/fig-a.png?dl=1");
  });

  it("calls onClose from the close button", () => {
    const onClose = vi.fn();
    render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText("Close"));
    expect(onClose).toHaveBeenCalled();
  });

  it("renders a filmstrip with one thumb per figure", () => {
    render(<FigureLightboxOverlay figures={FIGURES} initialIndex={0} onClose={vi.fn()} />);
    const strip = document.querySelector(".figure-lightbox-strip") as HTMLElement;
    expect(within(strip).getAllByRole("button")).toHaveLength(3);
  });
});
