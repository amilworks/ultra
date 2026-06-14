import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Markdown } from "./markdown";

describe("Markdown BisQue links", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows a useful fallback card when the BisQue image preview cannot load", async () => {
    vi.useFakeTimers();
    const viewerUrl =
      "https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-TGMDi3uJpDHBjwxpBdrQcH";

    render(<Markdown>{`BisQue viewer link: [View in BisQue](${viewerUrl})`}</Markdown>);

    const primaryLink = screen.getByRole("link", { name: "View in BisQue" });
    const openViewerLink = screen.getByRole("link", { name: /Open viewer/i });
    expect(primaryLink).toHaveAttribute("href", viewerUrl);
    expect(openViewerLink).toHaveAttribute("href", viewerUrl);

    const trigger = primaryLink.closest(".bisque-link-wrap");
    expect(trigger).not.toBeNull();

    act(() => {
      fireEvent.pointerEnter(trigger as Element);
      vi.advanceTimersByTime(140);
    });

    const previewImage = screen.getByAltText("BisQue preview");

    act(() => {
      fireEvent.error(previewImage);
    });

    expect(screen.getByText("Open this resource in BisQue")).toBeInTheDocument();
    expect(screen.getByText(/Launch the BisQue viewer/i)).toBeInTheDocument();
    expect(screen.getByText("00-TGMDi3uJpDHBjwxpBdrQcH")).toBeInTheDocument();
    const fallbackImage = document.querySelector(".bisque-link-preview-image-fallback");
    expect(fallbackImage).toHaveAttribute("src", "/bq-bg8.png");
    const openViewerLinks = screen.getAllByRole("link", { name: /Open viewer/i });
    expect(openViewerLinks[openViewerLinks.length - 1]).toHaveAttribute("href", viewerUrl);
  });
});
