import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ResourceBrowser } from "./ResourceBrowser";
import type { ResourceRecord } from "../types";

const imageResource: ResourceRecord = {
  file_id: "file_image",
  original_name: "prairie-cell-image.png",
  content_type: "image/png",
  size_bytes: 156_000,
  sha256: "sha-image",
  created_at: "2026-06-02T22:08:00Z",
  source_type: "bisque_import",
  resource_kind: "image",
  source_uri: "https://bisque.example.org/data_service/image/1",
  client_view_url: "https://bisque.example.org/client_service/view?resource=1",
  image_service_url: "https://bisque.example.org/image_service/1",
  has_thumbnail: true,
  sync_status: "bisque_sync_succeeded",
};

const fileResource: ResourceRecord = {
  file_id: "file_volume",
  original_name: "NPH_shunt_002_70yo.nii.gz",
  content_type: "application/gzip",
  size_bytes: 9_800_000,
  sha256: "sha-volume",
  created_at: "2026-06-02T22:10:00Z",
  source_type: "upload",
  resource_kind: "file",
  has_thumbnail: false,
  sync_status: "local_complete",
};

const baseProps = {
  resources: [imageResource, fileResource],
  totalCount: 2,
  loading: false,
  loadingMore: false,
  hasMore: false,
  error: null,
  query: "",
  kindFilter: "all" as const,
  sourceFilter: "all" as const,
  deletingFileIds: {},
  onQueryChange: vi.fn(),
  onKindFilterChange: vi.fn(),
  onSourceFilterChange: vi.fn(),
  onRefresh: vi.fn(),
  onLoadMore: vi.fn(),
  onOpenResource: vi.fn(),
  onUseInChat: vi.fn(),
  onDeleteResource: vi.fn(),
  thumbnailUrlFor: (resource: ResourceRecord) => `/thumb/${resource.file_id}`,
};

describe("ResourceBrowser", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn(() => ({
        matches: false,
        media: "(max-width: 720px)",
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }))
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders calm same-surface cards without redundant badges or external links", () => {
    render(<ResourceBrowser {...baseProps} />);

    expect(screen.getByText("prairie-cell-image.png")).toBeInTheDocument();
    expect(screen.getByText("NPH_shunt_002_70yo.nii.gz")).toBeInTheDocument();
    expect(screen.getByAltText("prairie-cell-image.png")).toHaveAttribute("src", "/thumb/file_image");
    expect(screen.getByText("File")).toBeInTheDocument();
    expect(screen.queryByText("BisQue viewer")).not.toBeInTheDocument();
    expect(screen.queryByText("Image service")).not.toBeInTheDocument();
    expect(screen.queryByText("Cataloged on BisQue")).not.toBeInTheDocument();
    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(2);
    expect(document.querySelectorAll(".resource-browser-badges")).toHaveLength(0);
  });

  it("keeps search input responsive through the parent callback", () => {
    const onQueryChange = vi.fn();
    render(<ResourceBrowser {...baseProps} onQueryChange={onQueryChange} />);

    fireEvent.change(screen.getByPlaceholderText("Search files, BisQue IDs, or URLs"), {
      target: { value: "nph" },
    });

    expect(onQueryChange).toHaveBeenCalledWith("nph");
  });

  it("shows fixed-card skeletons while loading", () => {
    render(<ResourceBrowser {...baseProps} resources={[]} totalCount={0} loading />);

    expect(screen.getByLabelText("Loading resources")).toBeInTheDocument();
    expect(document.querySelectorAll(".resource-browser-skeleton-card")).toHaveLength(8);
  });

  it("requests the next page when the load sentinel scrolls into view", () => {
    let observerCallback: IntersectionObserverCallback | null = null;
    class MockIntersectionObserver implements IntersectionObserver {
      readonly root = null;
      readonly rootMargin = "";
      readonly scrollMargin = "";
      readonly thresholds = [];
      disconnect = vi.fn();
      observe = vi.fn();
      takeRecords = vi.fn(() => []);
      unobserve = vi.fn();

      constructor(callback: IntersectionObserverCallback) {
        observerCallback = callback;
      }
    }
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
    const onLoadMore = vi.fn();

    render(
      <ResourceBrowser
        {...baseProps}
        totalCount={10}
        hasMore
        onLoadMore={onLoadMore}
      />
    );

    act(() => {
      observerCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver);
    });

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});
