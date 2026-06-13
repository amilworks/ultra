import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { VideoThumbnail } from "./VideoThumbnail";

describe("VideoThumbnail", () => {
  it("shows the poster image by default and only fetches video on hover", () => {
    const { container } = render(
      <VideoThumbnail posterUrl="/poster.png" videoUrl="/clip.mp4" alt="clip" className="wrap" />
    );
    const wrap = container.querySelector('[data-video-thumb]') as HTMLElement;

    // Default: a poster <img>, no <video> (no video bytes fetched).
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/poster.png");
    expect(container.querySelector("video")).toBeNull();

    // Hover: swap to an autoplaying, muted, looping <video> streamed from videoUrl,
    // with the poster bridging the buffer gap.
    fireEvent.mouseEnter(wrap);
    const video = container.querySelector("video") as HTMLVideoElement;
    expect(video).toBeTruthy();
    expect(video.getAttribute("src")).toBe("/clip.mp4");
    expect(video.getAttribute("poster")).toBe("/poster.png");
    expect(video.hasAttribute("loop")).toBe(true);
    expect(video.hasAttribute("autoplay")).toBe(true);
    expect(container.querySelector("img")).toBeNull();

    // Leave: back to the static poster (video unmounts → stream cancelled).
    fireEvent.mouseLeave(wrap);
    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/poster.png");
  });

  it("falls back to the poster (not the parent) when the video codec fails", () => {
    const onError = vi.fn();
    const { container } = render(
      <VideoThumbnail posterUrl="/poster.png" videoUrl="/clip.mp4" alt="clip" onError={onError} />
    );
    const wrap = container.querySelector('[data-video-thumb]') as HTMLElement;
    fireEvent.mouseEnter(wrap);
    fireEvent.error(container.querySelector("video") as HTMLVideoElement);
    // Reverts to the still poster; the tile is NOT marked failed.
    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/poster.png");
    expect(onError).not.toHaveBeenCalled();
  });

  it("reports poster load failure so the parent can fall back to an icon", () => {
    const onError = vi.fn();
    const { container } = render(
      <VideoThumbnail posterUrl="/bad.png" videoUrl="/clip.mp4" alt="clip" onError={onError} />
    );
    fireEvent.error(container.querySelector("img") as HTMLImageElement);
    expect(onError).toHaveBeenCalledTimes(1);
  });
});
