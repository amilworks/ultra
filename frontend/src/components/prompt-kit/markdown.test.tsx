import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { registerLensOpener } from "@/lib/lensNavigation";
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
    expect(fallbackImage).toHaveAttribute("src", "/bq-bg8.webp");
    const openViewerLinks = screen.getAllByRole("link", { name: /Open viewer/i });
    expect(openViewerLinks[openViewerLinks.length - 1]).toHaveAttribute("href", viewerUrl);
  });
});

describe("Markdown Lens links", () => {
  afterEach(() => {
    registerLensOpener(null);
  });

  const expectLensPill = (name: string, expectedHref: string) => {
    const primaryLink = screen.getByRole("link", { name });
    const pill = screen.getByRole("link", { name: "Open in Lens" });
    expect(primaryLink).toHaveAttribute("href", expectedHref);
    expect(pill).toHaveAttribute("href", expectedHref);
    expect(primaryLink).not.toHaveAttribute("target");
    expect(pill).not.toHaveAttribute("target");
    expect(primaryLink.closest(".ultra-link-wrap")).not.toBeNull();
    expect(pill).toHaveClass("ultra-link-open");
    expect(pill.querySelector("svg")).not.toBeNull();
    return { primaryLink, pill };
  };

  it("renders the pill for a relative deep link", () => {
    render(<Markdown>{"See [cells.tif](/?view=lens&resource=file-1)."}</Markdown>);
    expectLensPill("cells.tif", "/?view=lens&resource=file-1");
  });

  it("renders the pill for a same-origin absolute deep link, normalized to the relative form", () => {
    render(<Markdown>{"See [cells.tif](http://localhost:3000/?view=lens&resource=file-2)."}</Markdown>);
    expectLensPill("cells.tif", "/?view=lens&resource=file-2");
  });

  it("renders the pill for an ultra://resource reference", () => {
    render(<Markdown>{"See [cells.tif](ultra://resource/file-3/cells.tif)."}</Markdown>);
    expectLensPill("cells.tif", "/?view=lens&resource=file-3");
  });

  it("renders the pill for an ultra:// name with a raw percent sign instead of throwing", () => {
    // micromark leaves '%wt' unescaped; a naive decodeURIComponent would throw
    // inside the link renderer and blank the whole message.
    render(<Markdown>{"[5%wt sample](ultra://resource/file-3/5%wt_Ni.tif)"}</Markdown>);
    expect(screen.getByRole("link", { name: "Open in Lens" })).toHaveAttribute(
      "href",
      "/?view=lens&resource=file-3"
    );
  });

  it("plain left-click calls the registered opener with the file ids and prevents navigation", () => {
    const opener = vi.fn();
    registerLensOpener(opener);
    render(<Markdown>{"[stack](/?view=lens&resource=a,b)"}</Markdown>);
    // The raw-comma input resolves to the canonical (%2C) href the URL layer writes.
    const { primaryLink, pill } = expectLensPill("stack", "/?view=lens&resource=a%2Cb");

    const primaryEvent = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 });
    primaryLink.dispatchEvent(primaryEvent);
    expect(primaryEvent.defaultPrevented).toBe(true);
    expect(opener).toHaveBeenCalledTimes(1);
    expect(opener).toHaveBeenLastCalledWith(["a", "b"]);

    const pillEvent = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 });
    pill.dispatchEvent(pillEvent);
    expect(pillEvent.defaultPrevented).toBe(true);
    expect(opener).toHaveBeenCalledTimes(2);
  });

  it("modifier clicks are left to the browser", () => {
    const opener = vi.fn();
    registerLensOpener(opener);
    render(<Markdown>{"[stack](/?view=lens&resource=file-4)"}</Markdown>);
    const { primaryLink } = expectLensPill("stack", "/?view=lens&resource=file-4");

    for (const init of [{ metaKey: true }, { ctrlKey: true }, { shiftKey: true }, { altKey: true }, { button: 1 }]) {
      const event = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0, ...init });
      primaryLink.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(false);
    }
    expect(opener).not.toHaveBeenCalled();
  });

  it("without a registered opener the click is not prevented (href deep link still works)", () => {
    render(<Markdown>{"[stack](/?view=lens&resource=file-5)"}</Markdown>);
    const { pill } = expectLensPill("stack", "/?view=lens&resource=file-5");
    const event = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 });
    pill.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it("reads the opener at click time, not render time", () => {
    render(<Markdown>{"[stack](/?view=lens&resource=file-6)"}</Markdown>);
    const { primaryLink } = expectLensPill("stack", "/?view=lens&resource=file-6");
    const opener = vi.fn();
    registerLensOpener(opener);
    const event = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 });
    primaryLink.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(opener).toHaveBeenCalledWith(["file-6"]);
  });

  it("BisQue links still render their Open viewer pill", () => {
    const viewerUrl =
      "https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-abc";
    render(<Markdown>{`[View in BisQue](${viewerUrl})`}</Markdown>);
    expect(screen.getByRole("link", { name: /Open viewer/i })).toHaveAttribute("href", viewerUrl);
    expect(screen.queryByRole("link", { name: "Open in Lens" })).toBeNull();
  });

  it("a foreign-origin ?view=lens link renders as a plain external link", () => {
    render(<Markdown>{"[elsewhere](https://evil.example/?view=lens&resource=file-1)"}</Markdown>);
    const link = screen.getByRole("link", { name: "elsewhere" });
    expect(link).toHaveAttribute("href", "https://evil.example/?view=lens&resource=file-1");
    expect(link).toHaveAttribute("target", "_blank");
    expect(screen.queryByRole("link", { name: "Open in Lens" })).toBeNull();
    expect(document.querySelector(".ultra-link-wrap")).toBeNull();
  });
});
