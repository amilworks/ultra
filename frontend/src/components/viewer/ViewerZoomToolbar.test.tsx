import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ViewerZoomToolbar } from "./ViewerZoomToolbar";

describe("ViewerZoomToolbar", () => {
  it("renders the readout plus Zoom out / Zoom in / Fit controls shared by both 2D surfaces", () => {
    render(
      <ViewerZoomToolbar
        readout="73%"
        extraReadout="6000 x 4000"
        onZoomOut={() => {}}
        onZoomIn={() => {}}
        onFit={() => {}}
      />
    );
    expect(screen.getByTestId("viewer-zoom-readout")).toHaveTextContent("73%");
    expect(screen.getByText("6000 x 4000")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Zoom in" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Fit image to view" })).toBeInTheDocument();
  });

  it("invokes the matching handler for each control", () => {
    const onZoomOut = vi.fn();
    const onZoomIn = vi.fn();
    const onFit = vi.fn();
    render(<ViewerZoomToolbar readout="Fit" onZoomOut={onZoomOut} onZoomIn={onZoomIn} onFit={onFit} />);

    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    fireEvent.click(screen.getByRole("button", { name: "Fit image to view" }));

    expect(onZoomOut).toHaveBeenCalledTimes(1);
    expect(onZoomIn).toHaveBeenCalledTimes(1);
    expect(onFit).toHaveBeenCalledTimes(1);
  });

  it("marks itself as an image control so the canvas ignores pointerdowns on it", () => {
    const { container } = render(
      <ViewerZoomToolbar readout="Fit" onZoomOut={() => {}} onZoomIn={() => {}} onFit={() => {}} />
    );
    expect(container.querySelector('[data-viewer-image-control="true"]')).not.toBeNull();
  });
});
