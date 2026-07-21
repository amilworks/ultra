import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ZScrubThumbnail } from "./ZScrubThumbnail";

const sliceUrlFor = (z: number) => `/slice?z=${z}`;

describe("ZScrubThumbnail", () => {
  it("scrubs through z planes on wheel after hover loads the z count", async () => {
    const loadZCount = vi.fn().mockResolvedValue(10);
    const prefetch = vi.fn();
    render(
      <ZScrubThumbnail
        fileId="f1"
        alt="cells"
        staticThumbnailUrl="/thumb/f1"
        sliceUrlFor={sliceUrlFor}
        loadZCount={loadZCount}
        prefetch={prefetch}
      />,
    );
    const img = screen.getByAltText("cells");
    expect(img).toHaveAttribute("src", "/thumb/f1");

    fireEvent.mouseEnter(img);
    await waitFor(() => expect(loadZCount).toHaveBeenCalledWith("f1"));
    await waitFor(() => expect(img).toHaveAttribute("data-zscrub", "true"));

    fireEvent.wheel(img, { deltaY: 100, deltaMode: 0 });
    expect(img).toHaveAttribute("src", "/slice?z=2");
    expect(img).toHaveAttribute("data-z", "2");
    expect(prefetch).toHaveBeenCalledWith(["/slice?z=0", "/slice?z=1", "/slice?z=3", "/slice?z=4"]);

    fireEvent.wheel(img, { deltaY: -100, deltaMode: 0 });
    expect(img).toHaveAttribute("src", "/slice?z=0");
  });

  it("scrubs through z planes as the cursor moves down the thumbnail", async () => {
    const loadZCount = vi.fn().mockResolvedValue(10);
    const prefetch = vi.fn();
    render(
      <ZScrubThumbnail
        fileId="f1"
        alt="cells"
        staticThumbnailUrl="/thumb/f1"
        sliceUrlFor={sliceUrlFor}
        loadZCount={loadZCount}
        prefetch={prefetch}
      />,
    );
    const img = screen.getByAltText("cells");
    // jsdom returns an all-zero rect; give the thumbnail a real height so the
    // pointer offset maps to a plane.
    vi.spyOn(img, "getBoundingClientRect").mockReturnValue({
      top: 0, left: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect);

    fireEvent.mouseEnter(img);
    await waitFor(() => expect(img).toHaveAttribute("data-zscrub", "true"));

    // mousemove is rAF-coalesced (latest-position-wins), so flush a frame before
    // asserting the applied plane.
    fireEvent.mouseMove(img, { clientY: 100 }); // bottom edge -> last plane
    await waitFor(() => expect(img).toHaveAttribute("src", "/slice?z=9"));
    expect(img).toHaveAttribute("data-z", "9");

    fireEvent.mouseMove(img, { clientY: 50 }); // middle -> plane 5
    await waitFor(() => expect(img).toHaveAttribute("src", "/slice?z=5"));

    fireEvent.mouseMove(img, { clientY: 0 }); // top edge -> first plane
    await waitFor(() => expect(img).toHaveAttribute("data-z", "0"));
  });

  it("applies the pointer's entry plane after the lazy z count resolves", async () => {
    let resolveZCount: (count: number) => void = () => undefined;
    const loadZCount = vi.fn(
      () => new Promise<number>((resolve) => {
        resolveZCount = resolve;
      }),
    );
    render(
      <ZScrubThumbnail
        fileId="f1"
        alt="cells"
        staticThumbnailUrl="/thumb/f1"
        sliceUrlFor={sliceUrlFor}
        loadZCount={loadZCount}
        prefetch={vi.fn()}
      />,
    );
    const img = screen.getByAltText("cells");
    vi.spyOn(img, "getBoundingClientRect").mockReturnValue({
      top: 0, left: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect);

    // A real pointer enters once and can stop moving while viewer metadata is
    // still loading. The resolved stack must honor that original position.
    fireEvent.mouseEnter(img, { clientY: 75 });
    resolveZCount(10);

    await waitFor(() => expect(img).toHaveAttribute("src", "/slice?z=7"));
    expect(img).toHaveAttribute("data-z", "7");
  });

  it("reapplies the entry plane when returning with a cached z count", async () => {
    render(
      <ZScrubThumbnail
        fileId="f1"
        alt="cells"
        staticThumbnailUrl="/thumb/f1"
        sliceUrlFor={sliceUrlFor}
        loadZCount={vi.fn().mockResolvedValue(10)}
        prefetch={vi.fn()}
      />,
    );
    const img = screen.getByAltText("cells");
    vi.spyOn(img, "getBoundingClientRect").mockReturnValue({
      top: 0, left: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect);

    fireEvent.mouseEnter(img, { clientY: 75 });
    await waitFor(() => expect(img).toHaveAttribute("src", "/slice?z=7"));
    fireEvent.mouseLeave(img);
    expect(img).toHaveAttribute("src", "/thumb/f1");

    fireEvent.mouseEnter(img, { clientY: 25 });
    await waitFor(() => expect(img).toHaveAttribute("src", "/slice?z=2"));
  });

  it("does not scrub a flat (single-plane) image", async () => {
    const loadZCount = vi.fn().mockResolvedValue(1);
    render(
      <ZScrubThumbnail
        fileId="f2"
        alt="flat"
        staticThumbnailUrl="/thumb/f2"
        sliceUrlFor={sliceUrlFor}
        loadZCount={loadZCount}
        prefetch={vi.fn()}
      />,
    );
    const img = screen.getByAltText("flat");
    fireEvent.mouseEnter(img);
    await waitFor(() => expect(loadZCount).toHaveBeenCalled());
    await waitFor(() => expect(img).toHaveAttribute("data-zscrub", "false"));

    fireEvent.wheel(img, { deltaY: 100, deltaMode: 0 });
    expect(img).toHaveAttribute("src", "/thumb/f2");
  });

  it("resets to the static thumbnail when the pointer leaves", async () => {
    const loadZCount = vi.fn().mockResolvedValue(8);
    render(
      <ZScrubThumbnail
        fileId="f3"
        alt="stack"
        staticThumbnailUrl="/thumb/f3"
        sliceUrlFor={sliceUrlFor}
        loadZCount={loadZCount}
        prefetch={vi.fn()}
      />,
    );
    const img = screen.getByAltText("stack");
    fireEvent.mouseEnter(img);
    await waitFor(() => expect(img).toHaveAttribute("data-zscrub", "true"));
    fireEvent.wheel(img, { deltaY: 100, deltaMode: 0 });
    expect(img).toHaveAttribute("src", "/slice?z=2");

    fireEvent.mouseLeave(img);
    expect(img).toHaveAttribute("src", "/thumb/f3");
    expect(img).toHaveAttribute("data-z", "0");
  });
});
