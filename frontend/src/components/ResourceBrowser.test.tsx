import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ResourceBrowser } from "./ResourceBrowser";
import type {
  ResourceCollectionRecord,
  ResourceRecord,
  ResourceShareGrantRecord,
} from "../types";

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

const taggedFileResource: ResourceRecord = {
  ...fileResource,
  tags: ["NPH", "Under 70"],
};

const deletedFileResource: ResourceRecord = {
  ...fileResource,
  status: "deleted",
  original_name: "deleted-NPH_shunt_002_70yo.nii.gz",
};

const captionedResource: ResourceRecord = {
  ...imageResource,
  metadata: {
    data_agent: {
      caption_resources: {
        status: "succeeded",
        job_id: "data_agent_job_caption",
        summary_kind: "caption_generation",
        caption: "Prairie microscopy image with deterministic metadata caption.",
        caption_source: "deterministic_metadata",
        completed_at: "2026-06-02T22:18:00Z",
      },
      extract_metadata: {
        status: "succeeded",
        job_id: "data_agent_job_metadata",
        summary_kind: "metadata_extraction",
        completed_at: "2026-06-02T22:19:00Z",
      },
    },
  },
};

const nphFolder: ResourceCollectionRecord = {
  collection_id: "collection_nph",
  owner_user_id: "user_qa",
  name: "NPH review folder",
  collection_type: "folder",
  status: "active",
  resource_count: 7,
  created_at: "2026-06-02T22:12:00Z",
  updated_at: "2026-06-02T22:12:00Z",
  metadata: {},
};

const deletedNphFolder: ResourceCollectionRecord = {
  ...nphFolder,
  collection_id: "collection_deleted_nph",
  status: "deleted",
  updated_at: "2026-06-02T22:30:00Z",
};

const activeShareGrant: ResourceShareGrantRecord = {
  grant_id: "resource_grant_bob",
  resource_id: "file_image",
  owner_user_id: "user_qa",
  grantee_user_id: "bob",
  grantee_org_id: "org-b",
  role: "read",
  status: "active",
  created_at: "2026-06-02T22:20:00Z",
  updated_at: "2026-06-02T22:20:00Z",
  metadata: {},
};

const charlieShareGrant: ResourceShareGrantRecord = {
  grant_id: "resource_grant_charlie",
  resource_id: "file_image",
  owner_user_id: "user_qa",
  grantee_user_id: "charlie",
  grantee_org_id: "org-c",
  role: "read",
  status: "active",
  created_at: "2026-06-02T22:21:00Z",
  updated_at: "2026-06-02T22:21:00Z",
  metadata: {},
};

const sharedByMeResource: ResourceRecord = {
  ...imageResource,
  share_summary: {
    share_status: "shared_by_me",
    active_grant_count: 2,
    shared_by_me: true,
    shared_with_me: false,
  },
};

const sharedWithMeResource: ResourceRecord = {
  ...fileResource,
  share_summary: {
    share_status: "shared_with_me",
    active_grant_count: 1,
    shared_by_me: false,
    shared_with_me: true,
  },
};

const publicResource: ResourceRecord = {
  ...fileResource,
  file_id: "file_public",
  original_name: "public-supplement.nii.gz",
  share_summary: {
    share_status: "public",
    active_grant_count: 1,
    shared_by_me: true,
    shared_with_me: true,
  },
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
  onUploadFiles: vi.fn(),
  onOpenResource: vi.fn(),
  onUseInChat: vi.fn(),
  onDeleteResource: vi.fn(),
  thumbnailUrlFor: (resource: ResourceRecord) => `/thumb/${resource.file_id}`,
  downloadUrlFor: (resource: ResourceRecord) => `/download/${resource.file_id}`,
};

const createDragDataTransfer = () => {
  const dragData = new Map<string, string>();
  return {
    dropEffect: "",
    effectAllowed: "",
    setData: vi.fn((type: string, value: string) => {
      dragData.set(type, value);
    }),
    getData: vi.fn((type: string) => dragData.get(type) ?? ""),
  };
};

const createFileDropDataTransfer = (files: File[]) => ({
  dropEffect: "",
  effectAllowed: "",
  files,
  items: files.map((file) => ({ kind: "file", type: file.type })),
  types: ["Files"],
  setData: vi.fn(),
  getData: vi.fn(() => ""),
});

const clearStoredResourceViewMode = (): void => {
  try {
    globalThis.localStorage?.removeItem("bisque.resources.view_mode");
  } catch {
    // Some test runners disable localStorage; the component treats that as non-fatal too.
  }
};

const openResourceFilters = (): void => {
  fireEvent.click(screen.getByRole("button", { name: /More filters/ }));
};

describe("ResourceBrowser", () => {
  beforeEach(() => {
    clearStoredResourceViewMode();
    Element.prototype.scrollIntoView = vi.fn();
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
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    clearStoredResourceViewMode();
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

  it("uses compact cards for files without thumbnails", () => {
    render(<ResourceBrowser {...baseProps} />);

    expect(screen.getByLabelText("Resource prairie-cell-image.png")).toHaveAttribute(
      "data-preview",
      "true"
    );
    expect(screen.getByLabelText("Resource NPH_shunt_002_70yo.nii.gz")).toHaveAttribute(
      "data-preview",
      "false"
    );
  });

  it("opens resources from a plain card click while preserving explicit selection controls", () => {
    const onOpenResource = vi.fn();
    const onDeleteSelectedResources = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onOpenResource={onOpenResource}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByLabelText("Resource prairie-cell-image.png"));

    expect(onOpenResource).toHaveBeenCalledWith(imageResource);
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).not.toBeChecked();

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));

    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).toBeChecked();

    fireEvent.click(screen.getByLabelText("Resource NPH_shunt_002_70yo.nii.gz"), {
      metaKey: true,
    });

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" })).toBeChecked();
    expect(onOpenResource).toHaveBeenCalledTimes(1);
  });

  it("selects a contiguous resource range with Shift-click from the card surface", () => {
    const onOpenResource = vi.fn();
    const onDeleteSelectedResources = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[imageResource, fileResource, publicResource]}
        totalCount={3}
        onOpenResource={onOpenResource}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByLabelText("Resource public-supplement.nii.gz"), {
      shiftKey: true,
    });

    expect(screen.getByText("3 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Select public-supplement.nii.gz" })).toBeChecked();
    expect(onOpenResource).not.toHaveBeenCalled();
  });

  it("keeps card actions in the context menu without a visible resource three-dot button", async () => {
    const onDeleteSelectedResources = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    expect(
      screen.queryByRole("button", { name: "More actions for prairie-cell-image.png" })
    ).not.toBeInTheDocument();

    fireEvent.contextMenu(screen.getByLabelText("Resource prairie-cell-image.png"));
    expect(await screen.findByRole("menuitem", { name: "View" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Download" })).toBeInTheDocument();
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument();
  });

  it("supports select-all, clear, and move-to-trash keyboard shortcuts from the grid", async () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    const resourceCard = screen.getByLabelText("Resource prairie-cell-image.png");

    fireEvent.keyDown(resourceCard, { key: "a", metaKey: true });

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" })).toBeChecked();

    fireEvent.keyDown(resourceCard, { key: "Escape" });

    expect(screen.queryByText("2 selected")).not.toBeInTheDocument();

    fireEvent.keyDown(resourceCard, { key: "a", ctrlKey: true });
    expect(screen.getByText("2 selected")).toBeInTheDocument();

    fireEvent.keyDown(resourceCard, { key: "Delete" });

    await waitFor(() => {
      expect(onDeleteSelectedResources).toHaveBeenCalledWith([imageResource, fileResource]);
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
  });

  it("keeps resource keyboard shortcuts out of the search field", () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    const onQueryChange = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
        onQueryChange={onQueryChange}
      />
    );

    const searchInput = screen.getByPlaceholderText("Search resources");

    fireEvent.keyDown(searchInput, { key: "a", metaKey: true });
    fireEvent.keyDown(searchInput, { key: "Delete" });

    expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    expect(onDeleteSelectedResources).not.toHaveBeenCalled();
  });

  it("keeps the default Resources controls and filter sheet focused", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        onSharingFilterChange={vi.fn()}
        onStatusFilterChange={vi.fn()}
        onCreateCollectionFromSelection={vi.fn().mockResolvedValue(undefined)}
      />
    );

    expect(screen.getByRole("button", { name: "New" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Upload resources" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Upload folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "New folder" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Upload resource files")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Upload resource folder")).not.toBeInTheDocument();
    expect(screen.getByTestId("resource-upload-files-input")).toHaveAttribute("hidden");
    expect(screen.getByTestId("resource-upload-folder-input")).toHaveAttribute("hidden");
    expect(screen.getByPlaceholderText("Search resources")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "More filters" })).toBeInTheDocument();

    fireEvent.pointerDown(screen.getByRole("button", { name: "New" }), {
      button: 0,
      ctrlKey: false,
      pointerType: "mouse",
    });
    expect(screen.getByRole("menuitem", { name: "New folder" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Upload files" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Upload folder" })).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("menu", { name: "New" }), { key: "Escape" });
    expect(screen.queryByRole("menu", { name: "New" })).not.toBeInTheDocument();

    expect(screen.queryByRole("radio", { name: "Table view" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by metadata")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by processing status")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources created after")).not.toBeInTheDocument();

    openResourceFilters();

    expect(screen.getByRole("dialog", { name: "Resource filters and view" })).toBeInTheDocument();
    expect(
      screen.getByText("Choose a view, then narrow by type, source, sharing, or lifecycle.")
    ).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Table view" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Images" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Uploads" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Private" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Deleted" })).toBeInTheDocument();
    expect(screen.queryByText("Processing")).not.toBeInTheDocument();
    expect(screen.queryByText("Details")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by metadata")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by processing status")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by scientific descriptors")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by tags")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources created after")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources created before")).not.toBeInTheDocument();
  });

  it("switches between preview cards and a dense table with row context-menu actions", async () => {
    const onOpenResource = vi.fn();
    const onUseInChat = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[sharedByMeResource, fileResource]}
        onOpenResource={onOpenResource}
        onUseInChat={onUseInChat}
      />
    );

    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(2);

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Table view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    const table = screen.getByRole("table", { name: "Resources table" });
    expect(table).toBeInTheDocument();
    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(0);
    expect(within(table).getByText("prairie-cell-image.png")).toBeInTheDocument();
    expect(within(table).getByText("NPH_shunt_002_70yo.nii.gz")).toBeInTheDocument();
    expect(within(table).getByText("2 active grants")).toBeInTheDocument();
    expect(within(table).queryByText("Processing")).not.toBeInTheDocument();
    expect(within(table).queryByText("Caption ready")).not.toBeInTheDocument();
    expect(within(table).queryByText("Metadata ready")).not.toBeInTheDocument();
    expect(within(table).queryByText("Actions")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "View prairie-cell-image.png" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Use prairie-cell-image.png in chat" })
    ).not.toBeInTheDocument();
    expect(
      within(table).queryByRole("button", { name: "More actions for prairie-cell-image.png" })
    ).not.toBeInTheDocument();

    const prairieRow = within(table).getByText("prairie-cell-image.png").closest("tr");
    expect(prairieRow).toBeInstanceOf(HTMLElement);

    fireEvent.contextMenu(prairieRow as HTMLElement);
    fireEvent.click(screen.getByRole("menuitem", { name: "View" }));

    fireEvent.contextMenu(prairieRow as HTMLElement);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Use in chat" }));

    expect(onOpenResource).toHaveBeenCalledWith(sharedByMeResource);
    expect(onUseInChat).toHaveBeenCalledWith(sharedByMeResource);

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Card view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("table", { name: "Resources table" })).not.toBeInTheDocument();
    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(2);
  });

  it("opens resources from plain dense table row clicks while preserving explicit selection controls", () => {
    const onOpenResource = vi.fn();
    const onDeleteSelectedResources = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onOpenResource={onOpenResource}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Table view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    const table = screen.getByRole("table", { name: "Resources table" });
    const prairieRow = within(table).getByText("prairie-cell-image.png").closest("tr");
    const nphRow = within(table).getByText("NPH_shunt_002_70yo.nii.gz").closest("tr");
    expect(prairieRow).toBeInstanceOf(HTMLElement);
    expect(nphRow).toBeInstanceOf(HTMLElement);

    fireEvent.click(prairieRow as HTMLElement);

    expect(onOpenResource).toHaveBeenCalledWith(imageResource);
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).not.toBeChecked();

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));

    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).toBeChecked();

    fireEvent.click(nphRow as HTMLElement, { ctrlKey: true });

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" })).toBeChecked();
    expect(onOpenResource).toHaveBeenCalledTimes(1);
  });

  it("selects a contiguous resource range with Shift-click from dense table rows", () => {
    const onOpenResource = vi.fn();
    const onDeleteSelectedResources = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[imageResource, fileResource, publicResource]}
        totalCount={3}
        onOpenResource={onOpenResource}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Table view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    const table = screen.getByRole("table", { name: "Resources table" });
    const firstRow = within(table).getByText("prairie-cell-image.png").closest("tr");
    const lastRow = within(table).getByText("public-supplement.nii.gz").closest("tr");
    expect(firstRow).toBeInstanceOf(HTMLElement);
    expect(lastRow).toBeInstanceOf(HTMLElement);

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(lastRow as HTMLElement, { shiftKey: true });

    expect(screen.getByText("3 selected")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Select public-supplement.nii.gz" })).toBeChecked();
    expect(onOpenResource).not.toHaveBeenCalled();
  });

  it("remembers the preferred resource view mode locally", () => {
    const storedValues: Record<string, string> = {
      "bisque.resources.view_mode": "table",
    };
    const storage: Storage = {
      get length() {
        return Object.keys(storedValues).length;
      },
      clear: vi.fn(() => {
        for (const key of Object.keys(storedValues)) {
          delete storedValues[key];
        }
      }),
      getItem: vi.fn((key: string) => storedValues[key] ?? null),
      key: vi.fn((index: number) => Object.keys(storedValues)[index] ?? null),
      removeItem: vi.fn((key: string) => {
        delete storedValues[key];
      }),
      setItem: vi.fn((key: string, value: string) => {
        storedValues[key] = value;
      }),
    };
    vi.stubGlobal("localStorage", storage);

    render(<ResourceBrowser {...baseProps} />);

    expect(screen.getByRole("table", { name: "Resources table" })).toBeInTheDocument();
    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(0);

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Card view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(2);
    expect(storedValues["bisque.resources.view_mode"]).toBe("cards");
  });

  it("keeps compact Resources in card view even when desktop table mode was stored", () => {
    const originalInnerWidth = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      value: 430,
      writable: true,
      configurable: true,
    });
    const storedValues: Record<string, string> = {
      "bisque.resources.view_mode": "table",
    };
    const storage: Storage = {
      get length() {
        return Object.keys(storedValues).length;
      },
      clear: vi.fn(() => {
        for (const key of Object.keys(storedValues)) {
          delete storedValues[key];
        }
      }),
      getItem: vi.fn((key: string) => storedValues[key] ?? null),
      key: vi.fn((index: number) => Object.keys(storedValues)[index] ?? null),
      removeItem: vi.fn((key: string) => {
        delete storedValues[key];
      }),
      setItem: vi.fn((key: string, value: string) => {
        storedValues[key] = value;
      }),
    };
    vi.stubGlobal("localStorage", storage);

    try {
      render(<ResourceBrowser {...baseProps} />);

      expect(screen.queryByRole("table", { name: "Resources table" })).not.toBeInTheDocument();
      expect(document.querySelectorAll(".resource-browser-card")).toHaveLength(2);
      expect(storedValues["bisque.resources.view_mode"]).toBe("table");

      openResourceFilters();
      expect(screen.getByRole("radio", { name: "Card view" })).toBeInTheDocument();
      expect(screen.queryByRole("radio", { name: "Table view" })).not.toBeInTheDocument();
    } finally {
      Object.defineProperty(window, "innerWidth", {
        value: originalInnerWidth,
        writable: true,
        configurable: true,
      });
    }
  });

  it("virtualizes dense table rows for large resource catalogs", () => {
    const largeResources = Array.from({ length: 250 }, (_value, index): ResourceRecord => ({
      ...fileResource,
      file_id: `file_volume_${index}`,
      original_name: `NPH_shunt_${String(index).padStart(3, "0")}.nii.gz`,
      sha256: `sha-volume-${index}`,
      created_at: `2026-06-02T22:${String(index % 60).padStart(2, "0")}:00Z`,
    }));
    render(
      <ResourceBrowser
        {...baseProps}
        resources={largeResources}
        totalCount={largeResources.length}
      />
    );

    openResourceFilters();
    fireEvent.click(screen.getByRole("radio", { name: "Table view" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    const table = screen.getByRole("table", { name: "Resources table" });
    const tableShell = document.querySelector(".resource-browser-table-shell");
    if (!(tableShell instanceof HTMLElement)) {
      throw new Error("Resource table shell was not rendered.");
    }
    expect(within(table).getByText("NPH_shunt_000.nii.gz")).toBeInTheDocument();
    expect(within(table).queryByText("NPH_shunt_249.nii.gz")).not.toBeInTheDocument();
    expect(within(table).getAllByRole("row").length).toBeLessThan(80);

    Object.defineProperty(tableShell, "clientHeight", {
      value: 420,
      configurable: true,
    });
    Object.defineProperty(tableShell, "scrollTop", {
      value: 250 * 68,
      configurable: true,
    });
    fireEvent.scroll(tableShell);

    expect(within(table).getByText("NPH_shunt_249.nii.gz")).toBeInTheDocument();
    expect(within(table).queryByText("NPH_shunt_000.nii.gz")).not.toBeInTheDocument();
  });

  it("keeps search input responsive through the parent callback", () => {
    const onQueryChange = vi.fn();
    render(<ResourceBrowser {...baseProps} onQueryChange={onQueryChange} />);

    fireEvent.change(screen.getByPlaceholderText("Search resources"), {
      target: { value: "nph" },
    });

    expect(onQueryChange).toHaveBeenCalledWith("nph");
  });

  it("keeps metadata-specific filters out of the default filter sheet", () => {
    render(<ResourceBrowser {...baseProps} />);

    openResourceFilters();

    expect(screen.queryByLabelText("Filter resources by metadata")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Metadata filters")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by processing status")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources by scientific descriptors")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources created after")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Filter resources created before")).not.toBeInTheDocument();
  });

  it("surfaces an active tag filter as a clearable chip", () => {
    const onTagFilterChange = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        tagFilter="Under 70"
        onTagFilterChange={onTagFilterChange}
      />
    );

    expect(screen.getByRole("button", { name: /More filters/ })).toHaveTextContent("1");
    const chip = screen.getByRole("button", { name: "Clear tag filter Under 70" });
    expect(chip).toHaveTextContent("Tag: Under 70");

    fireEvent.click(chip);

    expect(onTagFilterChange).toHaveBeenCalledWith("");
  });

  it("resets the simple filters from the filter sheet", () => {
    const onKindFilterChange = vi.fn();
    const onSourceFilterChange = vi.fn();
    const onSharingFilterChange = vi.fn();
    const onStatusFilterChange = vi.fn();
    const onTagFilterChange = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        kindFilter="all"
        sourceFilter="all"
        sharingFilter="all"
        statusFilter="active"
        tagFilter="Under 70"
        onKindFilterChange={onKindFilterChange}
        onSourceFilterChange={onSourceFilterChange}
        onSharingFilterChange={onSharingFilterChange}
        onStatusFilterChange={onStatusFilterChange}
        onTagFilterChange={onTagFilterChange}
      />
    );

    openResourceFilters();
    expect(screen.queryByText("Processing")).not.toBeInTheDocument();
    expect(screen.queryByText("Details")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));

    expect(onKindFilterChange).toHaveBeenCalledWith("all");
    expect(onSourceFilterChange).toHaveBeenCalledWith("all");
    expect(onSharingFilterChange).toHaveBeenCalledWith("all");
    expect(onStatusFilterChange).toHaveBeenCalledWith("active");
    expect(onTagFilterChange).toHaveBeenCalledWith("");
  });

  it("passes selected files through the Resources upload control", () => {
    const onUploadFiles = vi.fn();
    render(<ResourceBrowser {...baseProps} onUploadFiles={onUploadFiles} />);

    const file = new File(["hello"], "cells.ome.tiff", { type: "image/tiff" });
    fireEvent.change(screen.getByTestId("resource-upload-files-input"), {
      target: { files: [file] },
    });

    expect(onUploadFiles).toHaveBeenCalledTimes(1);
    expect(onUploadFiles.mock.calls[0][0]).toEqual([file]);
  });

  it("passes folder-selected files through the Resources folder upload control", () => {
    const onUploadFiles = vi.fn();
    render(<ResourceBrowser {...baseProps} onUploadFiles={onUploadFiles} />);

    const file = new File(["tile"], "cells.ome.tiff", { type: "image/tiff" });
    Object.defineProperty(file, "webkitRelativePath", {
      value: "experiment-a/day-1/cells.ome.tiff",
      configurable: true,
    });
    const folderInput = screen.getByTestId("resource-upload-folder-input");
    expect(folderInput).toHaveAttribute("webkitdirectory");

    fireEvent.change(folderInput, {
      target: { files: [file] },
    });

    expect(onUploadFiles).toHaveBeenCalledTimes(1);
    expect(onUploadFiles.mock.calls[0][0]).toEqual([file]);
  });

  it("uploads dropped desktop files into the active folder", () => {
    const onUploadFiles = vi.fn();
    const file = new File(["nifti"], "dropped-brain.nii.gz", {
      type: "application/gzip",
    });
    const dataTransfer = createFileDropDataTransfer([file]);
    const { container } = render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onUploadFiles={onUploadFiles}
        onClearActiveCollection={vi.fn()}
      />
    );
    const content = container.querySelector(".resource-browser-content");
    expect(content).toBeInstanceOf(HTMLElement);

    fireEvent.dragOver(content as HTMLElement, { dataTransfer });
    expect(dataTransfer.dropEffect).toBe("copy");
    fireEvent.drop(content as HTMLElement, { dataTransfer });

    expect(onUploadFiles).toHaveBeenCalledTimes(1);
    expect(onUploadFiles.mock.calls[0][0]).toEqual([file]);
    expect(onUploadFiles.mock.calls[0][1]).toMatchObject({
      uploadTargetCollection: nphFolder,
    });
  });

  it("uploads dropped desktop files directly into a folder tile", () => {
    const onUploadFiles = vi.fn();
    const file = new File(["table"], "dropped-table.csv", {
      type: "text/csv",
    });
    const dataTransfer = createFileDropDataTransfer([file]);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onUploadFiles={onUploadFiles}
        onOpenCollection={vi.fn()}
      />
    );

    const folderButton = screen.getByRole("button", { name: "Open folder NPH review folder" });
    fireEvent.dragOver(folderButton, { dataTransfer });
    expect(dataTransfer.dropEffect).toBe("copy");
    fireEvent.drop(folderButton, { dataTransfer });

    expect(onUploadFiles).toHaveBeenCalledTimes(1);
    expect(onUploadFiles.mock.calls[0][0]).toEqual([file]);
    expect(onUploadFiles.mock.calls[0][1]).toMatchObject({
      uploadTargetCollection: nphFolder,
    });
  });

  it("shows verified-byte progress for resumable resource uploads", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-cells",
            fileName: "cells.ome.tiff",
            status: "uploading",
            totalBytes: 1_000,
            bytesVerified: 512,
          },
        ]}
      />
    );

    expect(screen.getByText("cells.ome.tiff")).toBeInTheDocument();
    expect(screen.getByText("Uploading")).toBeInTheDocument();
    const progress = screen.getByRole("progressbar", { name: "cells.ome.tiff upload progress" });
    expect(progress).toHaveAttribute("aria-valuenow", "51");
    expect(screen.getByText(/512 B verified/i)).toBeInTheDocument();
  });

  it("labels interrupted persisted uploads as ready to resume", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-brain",
            fileName: "brain.nii",
            status: "needs_file",
            totalBytes: 1_000,
            bytesVerified: 400,
            error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
          },
        ]}
      />
    );

    expect(screen.getByText("brain.nii")).toBeInTheDocument();
    expect(screen.getByText("Ready to resume")).toBeInTheDocument();
    expect(screen.getByText(/Select the same file or folder/i)).toBeInTheDocument();
  });

  it("labels paused persisted uploads and offers resume", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-paused",
            fileName: "paused-brain.nii",
            status: "paused",
            totalBytes: 1_000,
            bytesVerified: 512,
          },
        ]}
      />
    );

    expect(screen.getByText("paused-brain.nii")).toBeInTheDocument();
    expect(screen.getByText("Paused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume paused-brain.nii upload" })).toBeInTheDocument();
  });

  it("offers pause for active upload-session rows", () => {
    const onPauseUploadProgress = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-active",
            sessionId: "upload_session_active",
            fileName: "active-brain.nii",
            status: "uploading",
            totalBytes: 1_000,
            bytesVerified: 512,
          },
        ]}
        onPauseUploadProgress={onPauseUploadProgress}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Pause active-brain.nii upload" }));

    expect(onPauseUploadProgress).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "upload-active",
        sessionId: "upload_session_active",
        fileName: "active-brain.nii",
      })
    );
    expect(screen.queryByRole("button", { name: "Resume active-brain.nii upload" })).not.toBeInTheDocument();
  });

  it("offers server cancel for recoverable upload-session rows", () => {
    const onCancelUploadProgress = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-paused",
            sessionId: "upload_session_paused",
            fileName: "paused-brain.nii",
            status: "paused",
            totalBytes: 1_000,
            bytesVerified: 512,
          },
        ]}
        onCancelUploadProgress={onCancelUploadProgress}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel paused-brain.nii upload" }));

    expect(onCancelUploadProgress).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "upload-paused",
        sessionId: "upload_session_paused",
        fileName: "paused-brain.nii",
      })
    );
  });

  it("hides completed and canceled upload rows after refresh reconciliation", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-completed",
            fileName: "completed.nii",
            status: "completed",
            totalBytes: 1_000,
            bytesVerified: 1_000,
          },
          {
            id: "upload-canceled",
            fileName: "canceled.nii",
            status: "canceled",
            totalBytes: 1_000,
            bytesVerified: 256,
          },
          {
            id: "upload-failed",
            fileName: "failed.nii",
            status: "failed",
            totalBytes: 1_000,
            bytesVerified: 256,
            error: "Virus scan rejected upload.",
          },
        ]}
      />
    );

    expect(screen.queryByText("completed.nii")).not.toBeInTheDocument();
    expect(screen.queryByText("canceled.nii")).not.toBeInTheDocument();
    expect(screen.getByText("failed.nii")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("renders expired upload sessions without raw HTTP errors", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-expired",
            fileName: "expired-brain.nii",
            status: "failed",
            totalBytes: 1_000,
            bytesVerified: 256,
            error: "Request failed with status 404",
          },
        ]}
      />
    );

    expect(screen.getByText("expired-brain.nii")).toBeInTheDocument();
    expect(screen.getByText("Upload session expired. Start this upload again.")).toBeInTheDocument();
    expect(screen.queryByText("Request failed with status 404")).not.toBeInTheDocument();
  });

  it("collapses expired upload history into one quiet recovery summary", async () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-expired-a",
            fileName: "expired-a.nii",
            status: "failed",
            totalBytes: 1_000,
            bytesVerified: 256,
            error: "Request failed with status 404",
          },
          {
            id: "upload-expired-b",
            fileName: "expired-b.nii",
            status: "failed",
            totalBytes: 2_000,
            bytesVerified: 0,
            error: "Request failed with status 410",
          },
        ]}
      />
    );

    expect(screen.getByText("2 uploads need attention")).toBeInTheDocument();
    expect(screen.getByText("Expired sessions can be restarted when needed.")).toBeInTheDocument();
    expect(screen.queryByText("expired-a.nii")).not.toBeInTheDocument();
    expect(screen.queryByText("expired-b.nii")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show upload recovery details" }));

    expect(await screen.findByText("expired-a.nii")).toBeInTheDocument();
    expect(screen.getByText("expired-b.nii")).toBeInTheDocument();
    expect(screen.queryByText("Request failed with status 404")).not.toBeInTheDocument();
    expect(screen.queryByText("Request failed with status 410")).not.toBeInTheDocument();
  });

  it("offers clear retry and resume actions for unstable upload rows", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-cells",
            fileName: "cells.ome.tiff",
            status: "failed",
            totalBytes: 2_000,
            bytesVerified: 768,
            error: "Connection dropped while uploading.",
          },
          {
            id: "upload-brain-folder",
            fileName: "brain.nii",
            relativePath: "field-study/session-1/brain.nii",
            status: "needs_file",
            totalBytes: 1_000,
            bytesVerified: 400,
            error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
          },
        ]}
      />
    );

    const fileInput = screen.getByTestId("resource-upload-files-input") as HTMLInputElement;
    const folderInput = screen.getByTestId("resource-upload-folder-input") as HTMLInputElement;
    const fileClick = vi.spyOn(fileInput, "click").mockImplementation(() => undefined);
    const folderClick = vi.spyOn(folderInput, "click").mockImplementation(() => undefined);

    fireEvent.click(screen.getByRole("button", { name: "Retry cells.ome.tiff upload" }));
    fireEvent.click(screen.getByRole("button", { name: "Resume brain.nii upload" }));

    expect(fileClick).toHaveBeenCalledTimes(1);
    expect(folderClick).toHaveBeenCalledTimes(1);
  });

  it("blocks a resume reselection when the selected file does not match the interrupted upload", () => {
    const onUploadFiles = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onUploadFiles={onUploadFiles}
        uploadProgress={[
          {
            id: "upload-brain",
            fileName: "brain.nii",
            status: "needs_file",
            totalBytes: 1_000,
            bytesVerified: 400,
            error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
          },
        ]}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Resume brain.nii upload" }));
    fireEvent.change(screen.getByTestId("resource-upload-files-input"), {
      target: {
        files: [
          new File(["wrong"], "different.nii", {
            type: "application/x-nifti",
            lastModified: 1_780_915_200_000,
          }),
        ],
      },
    });

    expect(onUploadFiles).not.toHaveBeenCalled();
    expect(screen.getByText(/Select brain\.nii to resume this upload/i)).toBeInTheDocument();
  });

  it("blocks a resume reselection when the selected file identity differs from the interrupted upload", () => {
    const onUploadFiles = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onUploadFiles={onUploadFiles}
        uploadProgress={[
          {
            id: "upload-brain",
            fingerprint: `${"brain.nii"}:4:1780915200000:application/x-nifti:${"a".repeat(64)}`,
            fileName: "brain.nii",
            status: "needs_file",
            totalBytes: 4,
            bytesVerified: 2,
            error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
          },
        ]}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Resume brain.nii upload" }));
    fireEvent.change(screen.getByTestId("resource-upload-files-input"), {
      target: {
        files: [
          new File(["data"], "brain.nii", {
            type: "application/x-nifti",
            lastModified: 1_780_915_201_000,
          }),
        ],
      },
    });

    expect(onUploadFiles).not.toHaveBeenCalled();
    expect(screen.getByText(/Select brain\.nii to resume this upload/i)).toBeInTheDocument();
  });

  it("passes matching resume reselections with the original upload session context", () => {
    const onUploadFiles = vi.fn();
    const file = new File(["data"], "brain.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });
    render(
      <ResourceBrowser
        {...baseProps}
        onUploadFiles={onUploadFiles}
        uploadProgress={[
          {
            id: "upload-brain",
            fingerprint: `${"brain.nii"}:4:1780915200000:application/x-nifti:${"a".repeat(64)}`,
            sessionId: "upload_session_reselect",
            fileToken: "file-original-token",
            fileName: "brain.nii",
            status: "needs_file",
            totalBytes: 4,
            bytesVerified: 2,
            error: "Upload interrupted. Select the same file or folder to resume from verified chunks.",
          },
        ]}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Resume brain.nii upload" }));
    fireEvent.change(screen.getByTestId("resource-upload-files-input"), {
      target: {
        files: [file],
      },
    });

    expect(onUploadFiles).toHaveBeenCalledTimes(1);
    expect(onUploadFiles.mock.calls[0][0]).toEqual([file]);
    expect(onUploadFiles.mock.calls[0][1]).toMatchObject({
      resumeFrom: {
        id: "upload-brain",
        sessionId: "upload_session_reselect",
        fileToken: "file-original-token",
      },
    });
  });

  it("dismisses stale failed upload rows from the progress list", () => {
    const onDismissUploadProgress = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        uploadProgress={[
          {
            id: "upload-cells",
            fileName: "cells.ome.tiff",
            status: "failed",
            totalBytes: 2_000,
            bytesVerified: 768,
            error: "Connection dropped while uploading.",
          },
        ]}
        onDismissUploadProgress={onDismissUploadProgress}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Dismiss cells.ome.tiff upload" }));

    expect(onDismissUploadProgress).toHaveBeenCalledWith(
      expect.objectContaining({ id: "upload-cells", fileName: "cells.ome.tiff" })
    );
  });

  it("keeps Data Agent job controls out of the core resource browser", () => {
    render(<ResourceBrowser {...baseProps} />);

    expect(screen.queryByLabelText("Data Agent jobs")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh Data Agent jobs" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel Data Agent job Extract metadata" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry Data Agent job Quality check" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Inspect Data Agent job Deduplicate resources" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Resource prairie-cell-image.png")).toBeInTheDocument();
    expect(screen.getByLabelText("Resource NPH_shunt_002_70yo.nii.gz")).toBeInTheDocument();
  });

  it("keeps metadata processing details out of the default resource browser", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[captionedResource, fileResource]}
      />
    );

    expect(screen.queryByLabelText("Data Agent status for prairie-cell-image.png")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Processing status for prairie-cell-image.png")).not.toBeInTheDocument();
    expect(screen.queryByText("Caption ready")).not.toBeInTheDocument();
    expect(screen.queryByText("Metadata ready")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Prairie microscopy image with deterministic metadata caption.")
    ).not.toBeInTheDocument();
    expect(document.querySelectorAll('[class*="agent"]')).toHaveLength(0);
  });

  it("routes browse, use-in-chat, and move-to-trash actions through the resource context menu", async () => {
    const onOpenResource = vi.fn();
    const onUseInChat = vi.fn();
    const onDeleteResource = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onOpenResource={onOpenResource}
        onUseInChat={onUseInChat}
        onDeleteResource={onDeleteResource}
      />
    );

    const resourceCard = document.querySelector(".resource-browser-card");
    expect(resourceCard).toBeInstanceOf(HTMLElement);

    fireEvent.contextMenu(resourceCard as HTMLElement);
    fireEvent.click(await screen.findByRole("menuitem", { name: "View" }));

    fireEvent.contextMenu(resourceCard as HTMLElement);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Use in chat" }));

    fireEvent.contextMenu(resourceCard as HTMLElement);
    expect(await screen.findByRole("menuitem", { name: "Move to trash" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Move to trash" }));

    expect(onOpenResource).toHaveBeenCalledWith(imageResource);
    expect(onUseInChat).toHaveBeenCalledWith(imageResource);
    expect(onDeleteResource).toHaveBeenCalledWith(imageResource);
  });

  it("keeps core resource actions in the context menu without visible resource action buttons", async () => {
    const onOpenResource = vi.fn();
    const onUseInChat = vi.fn();
    const onDeleteResource = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        onOpenResource={onOpenResource}
        onUseInChat={onUseInChat}
        onDeleteResource={onDeleteResource}
      />
    );

    expect(
      screen.queryByRole("button", { name: "More actions for prairie-cell-image.png" })
    ).not.toBeInTheDocument();

    fireEvent.contextMenu(screen.getByLabelText("Resource prairie-cell-image.png"));
    fireEvent.click(await screen.findByRole("menuitem", { name: "View" }));

    fireEvent.contextMenu(screen.getByLabelText("Resource prairie-cell-image.png"));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Use in chat" }));

    fireEvent.contextMenu(screen.getByLabelText("Resource prairie-cell-image.png"));
    expect(await screen.findByRole("menuitem", { name: "Move to trash" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Move to trash" }));

    expect(onOpenResource).toHaveBeenCalledWith(imageResource);
    expect(onUseInChat).toHaveBeenCalledWith(imageResource);
    expect(onDeleteResource).toHaveBeenCalledWith(imageResource);
  });

  it("opens the resource context menu from a touch long press", async () => {
    const onOpenResource = vi.fn();
    render(<ResourceBrowser {...baseProps} onOpenResource={onOpenResource} />);

    const resourceCard = document.querySelector(".resource-browser-card");
    expect(resourceCard).toBeInstanceOf(HTMLElement);
    expect(screen.queryByRole("menuitem", { name: "View" })).not.toBeInTheDocument();

    fireEvent.pointerDown(resourceCard as HTMLElement, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 850));
    });
    fireEvent.pointerUp(resourceCard as HTMLElement, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });

    fireEvent.click(await screen.findByRole("menuitem", { name: "View" }));

    expect(onOpenResource).toHaveBeenCalledWith(imageResource);
  });

  it("offers download from the resource context menu without adding toolbar clutter", async () => {
    render(<ResourceBrowser {...baseProps} />);

    expect(screen.queryByRole("button", { name: "Download" })).not.toBeInTheDocument();

    const resourceCard = document.querySelector(".resource-browser-card");
    expect(resourceCard).toBeInstanceOf(HTMLElement);

    fireEvent.contextMenu(resourceCard as HTMLElement);
    const downloadItem = await screen.findByRole("menuitem", { name: "Download" });

    expect(downloadItem).toHaveAttribute("href", "/download/file_image");
    expect(downloadItem).toHaveAttribute("download", "prairie-cell-image.png");
  });

  it("renames a resource through the resource context menu", async () => {
    const onRenameResource = vi.fn().mockResolvedValue(undefined);
    render(<ResourceBrowser {...baseProps} onRenameResource={onRenameResource} />);

    const resourceCard = document.querySelector(".resource-browser-card");
    expect(resourceCard).toBeInstanceOf(HTMLElement);

    fireEvent.contextMenu(resourceCard as HTMLElement);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }));

    const dialog = await screen.findByRole("dialog", { name: "Rename resource" });
    const input = within(dialog).getByLabelText("Name");
    expect(input).toHaveValue("prairie-cell-image.png");

    fireEvent.change(input, { target: { value: "prairie-cell-reviewed.png" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    await waitFor(() => {
      expect(onRenameResource).toHaveBeenCalledWith(imageResource, "prairie-cell-reviewed.png");
    });
  });

  it("renames a focused resource with the F2 file-manager shortcut", async () => {
    const onOpenResource = vi.fn();
    const onRenameResource = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onOpenResource={onOpenResource}
        onRenameResource={onRenameResource}
      />
    );

    fireEvent.keyDown(screen.getByLabelText("Resource prairie-cell-image.png"), {
      key: "F2",
    });

    const dialog = await screen.findByRole("dialog", { name: "Rename resource" });
    const input = within(dialog).getByLabelText("Name");
    expect(input).toHaveValue("prairie-cell-image.png");
    expect(onOpenResource).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: "prairie-cell-reviewed.png" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    await waitFor(() => {
      expect(onRenameResource).toHaveBeenCalledWith(imageResource, "prairie-cell-reviewed.png");
    });
  });

  it("removes a resource from the active folder through the resource context menu", async () => {
    const onRemoveResourceFromCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={vi.fn()}
        onRemoveResourceFromCollection={onRemoveResourceFromCollection}
      />
    );

    const resourceCard = document.querySelector(".resource-browser-card");
    expect(resourceCard).toBeInstanceOf(HTMLElement);

    fireEvent.contextMenu(resourceCard as HTMLElement);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Remove from folder" }));

    await waitFor(() => {
      expect(onRemoveResourceFromCollection).toHaveBeenCalledWith(imageResource);
    });
  });

  it("creates an empty folder from the Resources header", async () => {
    const onCreateCollectionFromSelection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onCreateCollectionFromSelection={onCreateCollectionFromSelection}
      />
    );

    expect(screen.queryByText(/selected$/i)).not.toBeInTheDocument();
    fireEvent.pointerDown(screen.getByRole("button", { name: "New" }), {
      button: 0,
      ctrlKey: false,
      pointerType: "mouse",
    });
    fireEvent.click(await screen.findByRole("menuitem", { name: "New folder" }));

    const dialog = await screen.findByRole("dialog", { name: "New folder" });
    fireEvent.change(within(dialog).getByLabelText("Folder name"), {
      target: { value: "NPH review cohort" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => {
      expect(onCreateCollectionFromSelection).toHaveBeenCalledWith({
        collectionType: "folder",
        name: "NPH review cohort",
        resourceIds: [],
      });
    });
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "New folder" })).not.toBeInTheDocument();
    });
  });

  it("keeps selected-resource toolbar focused on core file-manager actions", async () => {
    const onCreateCollectionFromSelection = vi.fn().mockResolvedValue(undefined);
    const onAddSelectionToCollection = vi.fn().mockResolvedValue(undefined);
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    const onCreateBulkResourceShareGrants = vi.fn().mockResolvedValue([]);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onCreateCollectionFromSelection={onCreateCollectionFromSelection}
        onAddSelectionToCollection={onAddSelectionToCollection}
        onDeleteSelectedResources={onDeleteSelectedResources}
        onCreateBulkResourceShareGrants={onCreateBulkResourceShareGrants}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Share" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Move to" })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Target folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Move" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Move to trash" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Bulk resource actions"))
        .getAllByRole("button")
        .map((button) => button.getAttribute("aria-label") || button.textContent?.trim())
    ).toEqual(["Share", "Move to", "Move to trash", "Clear"]);
    expect(
      within(screen.getByLabelText("Bulk resource actions"))
        .getAllByRole("button")
        .every((button) => Boolean(button.getAttribute("aria-label")))
    ).toBe(true);
    expect(document.querySelectorAll(".resource-browser-bulk-action-label")).toHaveLength(4);
    expect(screen.queryByRole("button", { name: "Add to chat" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Tag" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add to folder" })).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Folder name")).not.toBeInTheDocument();
    expect(onCreateCollectionFromSelection).not.toHaveBeenCalled();
  });

  it("adds selected resources to an existing folder", async () => {
    const onAddSelectionToCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onAddSelectionToCollection={onAddSelectionToCollection}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByText("Chat, folders, tags, sharing, and delete actions.")).not.toBeInTheDocument();
    fireEvent.pointerDown(screen.getByRole("button", { name: "Move to" }), {
      button: 0,
      ctrlKey: false,
      pointerType: "mouse",
    });
    fireEvent.click(await screen.findByRole("menuitem", { name: /NPH review folder/ }));

    await waitFor(() => {
      expect(onAddSelectionToCollection).toHaveBeenCalledWith({
        collectionId: "collection_nph",
        resourceIds: ["file_image", "file_volume"],
      });
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
  });

  it("adds a dragged resource to a folder", async () => {
    const onAddSelectionToCollection = vi.fn().mockResolvedValue(undefined);
    const dataTransfer = createDragDataTransfer();
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onAddSelectionToCollection={onAddSelectionToCollection}
        onOpenCollection={vi.fn()}
      />
    );

    fireEvent.dragStart(screen.getByLabelText("Resource prairie-cell-image.png"), {
      dataTransfer,
    });
    fireEvent.dragOver(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      dataTransfer,
    });
    fireEvent.drop(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      dataTransfer,
    });

    await waitFor(() => {
      expect(onAddSelectionToCollection).toHaveBeenCalledWith({
        collectionId: "collection_nph",
        resourceIds: ["file_image"],
      });
    });
  });

  it("adds selected dragged resources to a folder", async () => {
    const onAddSelectionToCollection = vi.fn().mockResolvedValue(undefined);
    const dataTransfer = createDragDataTransfer();
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onAddSelectionToCollection={onAddSelectionToCollection}
        onOpenCollection={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    fireEvent.dragStart(screen.getByLabelText("Resource prairie-cell-image.png"), {
      dataTransfer,
    });
    fireEvent.dragOver(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      dataTransfer,
    });
    fireEvent.drop(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      dataTransfer,
    });

    await waitFor(() => {
      expect(onAddSelectionToCollection).toHaveBeenCalledWith({
        collectionId: "collection_nph",
        resourceIds: ["file_image", "file_volume"],
      });
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
  });

  it("moves a dragged resource out of the active folder through All resources", async () => {
    const onRemoveResourceFromCollection = vi.fn().mockResolvedValue(undefined);
    const dataTransfer = createDragDataTransfer();
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={vi.fn()}
        onRemoveResourceFromCollection={onRemoveResourceFromCollection}
      />
    );

    fireEvent.dragStart(screen.getByLabelText("Resource prairie-cell-image.png"), {
      dataTransfer,
    });
    fireEvent.dragOver(screen.getByRole("button", { name: "All resources" }), {
      dataTransfer,
    });
    fireEvent.drop(screen.getByRole("button", { name: "All resources" }), {
      dataTransfer,
    });

    await waitFor(() => {
      expect(onRemoveResourceFromCollection).toHaveBeenCalledWith(imageResource);
    });
  });

  it("moves selected dragged resources out of the active folder through All resources", async () => {
    const onAddSelectionToCollection = vi.fn().mockResolvedValue(undefined);
    const onRemoveResourceFromCollection = vi.fn().mockResolvedValue(undefined);
    const dataTransfer = createDragDataTransfer();
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onAddSelectionToCollection={onAddSelectionToCollection}
        onClearActiveCollection={vi.fn()}
        onRemoveResourceFromCollection={onRemoveResourceFromCollection}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    fireEvent.dragStart(screen.getByLabelText("Resource prairie-cell-image.png"), {
      dataTransfer,
    });
    fireEvent.dragOver(screen.getByRole("button", { name: "All resources" }), {
      dataTransfer,
    });
    fireEvent.drop(screen.getByRole("button", { name: "All resources" }), {
      dataTransfer,
    });

    await waitFor(() => {
      expect(onRemoveResourceFromCollection).toHaveBeenNthCalledWith(1, imageResource);
      expect(onRemoveResourceFromCollection).toHaveBeenNthCalledWith(2, fileResource);
    });
  });

  it("requests deletion for selected resources from the bulk toolbar", async () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    fireEvent.click(screen.getByRole("button", { name: "Move to trash" }));

    await waitFor(() => {
      expect(onDeleteSelectedResources).toHaveBeenCalledWith([imageResource, fileResource]);
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
  });

  it("keeps chat handoff out of the selected-resource toolbar", async () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add to chat" })).not.toBeInTheDocument();
  });

  it("shows deleted resources with lifecycle restore controls", async () => {
    const onStatusFilterChange = vi.fn();
    const onRestoreResource = vi.fn();
    const onDeleteResource = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[deletedFileResource]}
        totalCount={1}
        statusFilter="deleted"
        onStatusFilterChange={onStatusFilterChange}
        onRestoreResource={onRestoreResource}
        onDeleteResource={onDeleteResource}
      />
    );

    expect(screen.getByText("deleted-NPH_shunt_002_70yo.nii.gz")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Lifecycle status for deleted-NPH_shunt_002_70yo.nii.gz")
    ).toHaveTextContent("Deleted");
    openResourceFilters();
    fireEvent.click(screen.getByRole("button", { name: "Active" }));
    expect(onStatusFilterChange).toHaveBeenCalledWith("active");
    fireEvent.click(screen.getByRole("button", { name: "Done" }));

    const deletedCard = screen.getByLabelText("Resource deleted-NPH_shunt_002_70yo.nii.gz");
    fireEvent.contextMenu(deletedCard);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Restore" }));

    expect(onRestoreResource).toHaveBeenCalledWith(deletedFileResource);
    expect(onDeleteResource).not.toHaveBeenCalled();
  });

  it("restores selected deleted resources from the bulk toolbar", async () => {
    const secondDeletedResource: ResourceRecord = {
      ...deletedFileResource,
      file_id: "file_deleted_b",
      original_name: "deleted-followup-NPH.nii.gz",
    };
    const onRestoreSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[deletedFileResource, secondDeletedResource]}
        totalCount={2}
        statusFilter="deleted"
        onRestoreSelectedResources={onRestoreSelectedResources}
      />
    );

    fireEvent.click(
      screen.getByRole("checkbox", { name: "Select deleted-NPH_shunt_002_70yo.nii.gz" })
    );
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Select deleted-followup-NPH.nii.gz" })
    );

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    await waitFor(() => {
      expect(onRestoreSelectedResources).toHaveBeenCalledWith([
        deletedFileResource,
        secondDeletedResource,
      ]);
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
  });

  it("keeps deleted-resource selection focused on restore and clear", () => {
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[deletedFileResource]}
        totalCount={1}
        statusFilter="deleted"
        resourceCollections={[nphFolder]}
        onAddSelectionToCollection={vi.fn().mockResolvedValue(undefined)}
        onCreateBulkResourceShareGrants={vi.fn().mockResolvedValue([])}
        onDeleteSelectedResources={vi.fn().mockResolvedValue(undefined)}
        onRestoreSelectedResources={vi.fn().mockResolvedValue(undefined)}
      />
    );

    fireEvent.click(
      screen.getByRole("checkbox", { name: "Select deleted-NPH_shunt_002_70yo.nii.gz" })
    );

    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Share" })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Target folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Move to" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Move" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Move to trash" })).not.toBeInTheDocument();
  });

  it("keeps dataset snapshot controls out of the core resource browser", () => {
    const onCreateCollectionFromSelection = vi.fn().mockResolvedValue(undefined);
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onCreateCollectionFromSelection={onCreateCollectionFromSelection}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Folder name")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add to folder" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Dataset snapshots")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Dataset snapshot name")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Dataset name from current results")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create dataset" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create from results" })).not.toBeInTheDocument();
  });

  it("keeps Data Agent launchers out of selected and filtered resource views", () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        query="NPH"
        kindFilter="file"
        sourceFilter="upload"
        sharingFilter="private"
        tagFilter="Under 70"
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Tag" })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Data Agent job type" })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Current results Data Agent job type" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Run Data Agent on current results")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start Data Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run on results" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Queue agent" })).not.toBeInTheDocument();
  });

  it("manages resource sharing from the resource context menu", async () => {
    const onLoadResourceShareGrants = vi.fn().mockResolvedValue([activeShareGrant]);
    const onCreateResourceShareGrant = vi.fn().mockResolvedValue(charlieShareGrant);
    const onRevokeResourceShareGrant = vi.fn().mockResolvedValue({
      ...activeShareGrant,
      status: "revoked",
      revoked_at: "2026-06-02T22:22:00Z",
      updated_at: "2026-06-02T22:22:00Z",
    });
    render(
      <ResourceBrowser
        {...baseProps}
        onLoadResourceShareGrants={onLoadResourceShareGrants}
        onCreateResourceShareGrant={onCreateResourceShareGrant}
        onRevokeResourceShareGrant={onRevokeResourceShareGrant}
      />
    );

    fireEvent.contextMenu(screen.getByLabelText("Resource prairie-cell-image.png"));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Share" }));

    const dialog = await screen.findByRole("dialog", { name: "Share prairie-cell-image.png" });
    expect(onLoadResourceShareGrants).toHaveBeenCalledWith(imageResource);
    expect(dialog).toHaveTextContent("bob");
    expect(dialog).toHaveTextContent("org-b");
    expect(dialog).toHaveTextContent("Active");
    expect(dialog).toHaveTextContent("People with access");
    expect(dialog).toHaveTextContent("Add a person or team that can read this resource.");
    expect(screen.queryByText("Read grants and audit state.")).not.toBeInTheDocument();
    expect(screen.queryByText("Access grants")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Person or user ID"), {
      target: { value: " charlie " },
    });
    fireEvent.change(screen.getByLabelText("Team or organization ID"), {
      target: { value: " org-c " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() => {
      expect(onCreateResourceShareGrant).toHaveBeenCalledWith(imageResource, {
        granteeUserId: "charlie",
        granteeOrgId: "org-c",
        role: "read",
      });
    });
    expect(await screen.findByText("charlie")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Revoke read access for bob" }));

    await waitFor(() => {
      expect(onRevokeResourceShareGrant).toHaveBeenCalledWith(imageResource, activeShareGrant);
    });
    expect(await screen.findByText("Revoked")).toBeInTheDocument();
  });

  it("grants read access to selected resources from the bulk toolbar", async () => {
    const volumeShareGrant: ResourceShareGrantRecord = {
      ...charlieShareGrant,
      grant_id: "resource_grant_charlie_volume",
      resource_id: "file_volume",
    };
    const onCreateBulkResourceShareGrants = vi
      .fn()
      .mockResolvedValue([charlieShareGrant, volumeShareGrant]);
    render(
      <ResourceBrowser
        {...baseProps}
        onCreateBulkResourceShareGrants={onCreateBulkResourceShareGrants}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    fireEvent.click(screen.getByRole("button", { name: "Share" }));
    const dialog = await screen.findByRole("dialog", { name: "Share 2 selected resources" });
    expect(dialog).toHaveTextContent("2 resources");
    expect(dialog).toHaveTextContent("Add a person or team that can read these resources.");
    expect(screen.queryByText("Grant read access to selection")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Person or user ID"), {
      target: { value: " charlie " },
    });
    fireEvent.change(screen.getByLabelText("Team or organization ID"), {
      target: { value: " org-c " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Share selected" }));

    await waitFor(() => {
      expect(onCreateBulkResourceShareGrants).toHaveBeenCalledTimes(1);
    });
    expect(onCreateBulkResourceShareGrants).toHaveBeenCalledWith([imageResource, fileResource], {
      granteeUserId: "charlie",
      granteeOrgId: "org-c",
      role: "read",
    });
    await waitFor(() => {
      expect(screen.queryByText("2 selected")).not.toBeInTheDocument();
    });
    expect(screen.queryByRole("dialog", { name: "Share 2 selected resources" })).not.toBeInTheDocument();
  });

  it("shows sharing status chips and exposes sharing filters", () => {
    const onSharingFilterChange = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[sharedByMeResource, sharedWithMeResource, publicResource]}
        sharingFilter="all"
        onSharingFilterChange={onSharingFilterChange}
      />
    );

    expect(
      within(screen.getByLabelText("Sharing status for prairie-cell-image.png")).getByText(
        "2 active grants"
      )
    ).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Sharing status for NPH_shunt_002_70yo.nii.gz")).getByText(
        "Shared with me"
      )
    ).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Sharing status for public-supplement.nii.gz")).getByText(
        "Public"
      )
    ).toBeInTheDocument();

    openResourceFilters();
    fireEvent.click(screen.getByRole("button", { name: "Shared with me" }));
    expect(onSharingFilterChange).toHaveBeenCalledWith("shared_with_me");

    fireEvent.click(screen.getByRole("button", { name: "Public" }));
    expect(onSharingFilterChange).toHaveBeenCalledWith("public");
  });

  it("shows resource tags and exposes tag filters", () => {
    const onTagFilterChange = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[taggedFileResource]}
        totalCount={1}
        tagFilter=""
        onTagFilterChange={onTagFilterChange}
      />
    );

    const tagRegion = screen.getByLabelText("Tags for NPH_shunt_002_70yo.nii.gz");
    expect(within(tagRegion).getByRole("button", { name: "Filter by tag NPH" })).toBeInTheDocument();
    expect(within(tagRegion).getByRole("button", { name: "Filter by tag Under 70" })).toBeInTheDocument();

    fireEvent.click(within(tagRegion).getByRole("button", { name: "Filter by tag NPH" }));
    expect(onTagFilterChange).toHaveBeenCalledWith("NPH");
  });

  it("keeps bulk tagging out of the selected-resource toolbar", async () => {
    const onDeleteSelectedResources = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        onDeleteSelectedResources={onDeleteSelectedResources}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Select prairie-cell-image.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select NPH_shunt_002_70yo.nii.gz" }));

    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Tag" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Tag 2 selected resources" })).not.toBeInTheDocument();
  });

  it("opens an existing folder from the Resources folder strip", () => {
    const onOpenCollection = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
      />
    );

    expect(screen.getByLabelText("Resource folders")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open folder NPH review folder" }));

    expect(onOpenCollection).toHaveBeenCalledWith(nphFolder);
  });

  it("renames and moves a folder to trash through the folder context menu", async () => {
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    const onDeleteCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={vi.fn()}
        onRenameCollection={onRenameCollection}
        onDeleteCollection={onDeleteCollection}
      />
    );

    const folderButton = screen.getByRole("button", { name: "Open folder NPH review folder" });
    fireEvent.contextMenu(folderButton);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }));

    const dialog = await screen.findByRole("dialog", { name: "Rename folder" });
    const input = within(dialog).getByLabelText("Name");
    expect(input).toHaveValue("NPH review folder");
    fireEvent.change(input, { target: { value: "NPH reviewed" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    await waitFor(() => {
      expect(onRenameCollection).toHaveBeenCalledWith(nphFolder, "NPH reviewed");
    });

    fireEvent.contextMenu(folderButton);
    expect(await screen.findByRole("menuitem", { name: "Move folder to trash" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete folder" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Move folder to trash" }));

    await waitFor(() => {
      expect(onDeleteCollection).toHaveBeenCalledWith(nphFolder);
    });
  });

  it("renames a focused folder with the F2 file-manager shortcut", async () => {
    const onOpenCollection = vi.fn();
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
        onRenameCollection={onRenameCollection}
      />
    );

    const folderButton = screen.getByRole("button", { name: "Open folder NPH review folder" });
    fireEvent.keyDown(folderButton, { key: "F2" });

    const dialog = await screen.findByRole("dialog", { name: "Rename folder" });
    const input = within(dialog).getByLabelText("Name");
    expect(input).toHaveValue("NPH review folder");
    expect(onOpenCollection).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: "NPH reviewed" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    await waitFor(() => {
      expect(onRenameCollection).toHaveBeenCalledWith(nphFolder, "NPH reviewed");
    });
  });

  it("keeps folder actions in the context menu without a visible folder three-dot button", async () => {
    const onOpenCollection = vi.fn();
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    const onDeleteCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
        onRenameCollection={onRenameCollection}
        onDeleteCollection={onDeleteCollection}
      />
    );

    expect(
      screen.queryByRole("button", { name: "More actions for folder NPH review folder" })
    ).not.toBeInTheDocument();

    fireEvent.contextMenu(screen.getByRole("button", { name: "Open folder NPH review folder" }));
    expect(await screen.findByRole("menuitem", { name: "Open" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Rename" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete folder" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Move folder to trash" }));

    await waitFor(() => {
      expect(onDeleteCollection).toHaveBeenCalledWith(nphFolder);
    });
    expect(onOpenCollection).not.toHaveBeenCalled();
  });

  it("opens folder actions from a touch long press without opening the folder", async () => {
    const onOpenCollection = vi.fn();
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
        onRenameCollection={onRenameCollection}
      />
    );

    const folderButton = screen.getByRole("button", { name: "Open folder NPH review folder" });
    expect(screen.queryByRole("menuitem", { name: "Open" })).not.toBeInTheDocument();

    fireEvent.pointerDown(folderButton, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 850));
    });
    fireEvent.pointerUp(folderButton, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });

    expect(await screen.findByRole("menuitem", { name: "Open" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Rename" }));

    expect(await screen.findByRole("dialog", { name: "Rename folder" })).toBeInTheDocument();
    expect(onOpenCollection).not.toHaveBeenCalled();
  });

  it("keeps focused folder rename on F2 as the keyboard action path", async () => {
    const onOpenCollection = vi.fn();
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
        onRenameCollection={onRenameCollection}
      />
    );

    expect(
      screen.queryByRole("button", { name: "More actions for folder NPH review folder" })
    ).not.toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      key: "F2",
    });

    expect(await screen.findByRole("dialog", { name: "Rename folder" })).toBeInTheDocument();
    expect(onOpenCollection).not.toHaveBeenCalled();
    expect(onRenameCollection).not.toHaveBeenCalled();
  });

  it("moves a focused folder to trash with the Delete file-manager shortcut", async () => {
    const onOpenCollection = vi.fn();
    const onDeleteCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={onOpenCollection}
        onDeleteCollection={onDeleteCollection}
      />
    );

    fireEvent.keyDown(screen.getByRole("button", { name: "Open folder NPH review folder" }), {
      key: "Delete",
    });

    await waitFor(() => {
      expect(onDeleteCollection).toHaveBeenCalledWith(nphFolder);
    });
    expect(onOpenCollection).not.toHaveBeenCalled();
  });

  it("surfaces deleted folders with restore lifecycle controls", () => {
    const onOpenCollection = vi.fn();
    const onRestoreCollection = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        resources={[deletedFileResource]}
        totalCount={1}
        statusFilter="deleted"
        resourceCollections={[deletedNphFolder]}
        onOpenCollection={onOpenCollection}
        onRestoreCollection={onRestoreCollection}
      />
    );

    const deletedFolders = screen.getByLabelText("Deleted resource folders");
    expect(deletedFolders).toHaveTextContent("NPH review folder");
    expect(deletedFolders).toHaveTextContent("7 resources");
    expect(
      screen.queryByRole("button", { name: "Open folder NPH review folder" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Restore folder NPH review folder" }));

    expect(onRestoreCollection).toHaveBeenCalledWith(deletedNphFolder);
    expect(onOpenCollection).not.toHaveBeenCalled();
  });

  it("shows active folder context and returns to all resources", () => {
    const onClearActiveCollection = vi.fn();
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={onClearActiveCollection}
      />
    );

    expect(screen.getByLabelText("Active folder")).toHaveTextContent("NPH review folder");
    expect(screen.getByLabelText("Folder path")).toHaveTextContent("Resources>NPH review folder");
    expect(screen.getByText("2 of 2 resources in NPH review folder")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Folder actions" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Snapshot folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start folder Data Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Share folder" })).not.toBeInTheDocument();

    const activeFolderButton = screen.getByRole("button", {
      name: "Leave folder NPH review folder",
    });
    expect(activeFolderButton).toHaveTextContent("NPH review folder");

    fireEvent.click(activeFolderButton);

    expect(onClearActiveCollection).toHaveBeenCalledTimes(1);
    onClearActiveCollection.mockClear();

    fireEvent.click(screen.getByRole("button", { name: "All resources" }));

    expect(onClearActiveCollection).toHaveBeenCalledTimes(1);
  });

  it("moves up to the parent folder from the active folder crumb when nested", () => {
    const parentFolder: ResourceCollectionRecord = {
      ...nphFolder,
      collection_id: "collection_parent",
      name: "Parent folder",
      resource_count: 3,
    };
    const nestedFolder: ResourceCollectionRecord = {
      ...nphFolder,
      collection_id: "collection_nested",
      name: "Nested folder",
      parent_collection_id: parentFolder.collection_id,
      resource_count: 2,
    };
    const onClearActiveCollection = vi.fn();
    const onOpenCollection = vi.fn();

    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nestedFolder}
        resourceCollections={[parentFolder, nestedFolder]}
        onClearActiveCollection={onClearActiveCollection}
        onOpenCollection={onOpenCollection}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Go up from folder Nested folder" }));

    expect(onOpenCollection).toHaveBeenCalledWith(parentFolder);
    expect(onClearActiveCollection).not.toHaveBeenCalled();
  });

  it("keeps active-folder actions in the context menu without a visible folder three-dot button", async () => {
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    const onDeleteCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={vi.fn()}
        onRenameCollection={onRenameCollection}
        onDeleteCollection={onDeleteCollection}
      />
    );

    expect(
      screen.queryByRole("button", { name: "More actions for folder NPH review folder" })
    ).not.toBeInTheDocument();

    fireEvent.contextMenu(screen.getByLabelText("Active folder"));
    expect(await screen.findByRole("menuitem", { name: "Rename" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Open" })).not.toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete folder" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Move folder to trash" }));

    await waitFor(() => {
      expect(onDeleteCollection).toHaveBeenCalledWith(nphFolder);
    });
  });

  it("moves the active folder to trash with the Backspace file-manager shortcut", async () => {
    const onDeleteCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={vi.fn()}
        onDeleteCollection={onDeleteCollection}
      />
    );

    const activeFolder = screen.getByLabelText("Active folder");
    expect(activeFolder).toHaveAttribute("tabindex", "0");

    fireEvent.keyDown(activeFolder, { key: "Backspace" });

    await waitFor(() => {
      expect(onDeleteCollection).toHaveBeenCalledWith(nphFolder);
    });
  });

  it("opens active-folder actions from a touch long press", async () => {
    const onRenameCollection = vi.fn().mockResolvedValue(undefined);
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onClearActiveCollection={vi.fn()}
        onRenameCollection={onRenameCollection}
      />
    );

    const activeFolder = screen.getByLabelText("Active folder");
    expect(screen.queryByRole("menuitem", { name: "Rename" })).not.toBeInTheDocument();

    fireEvent.pointerDown(activeFolder, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 850));
    });
    fireEvent.pointerUp(activeFolder, {
      button: 0,
      ctrlKey: false,
      pointerType: "touch",
    });

    expect(await screen.findByRole("menuitem", { name: "Rename" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Open" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Rename" }));

    expect(await screen.findByRole("dialog", { name: "Rename folder" })).toBeInTheDocument();
  });

  it("keeps active folder advanced workflows out of the default folder view", () => {
    const onCreateResourceCollectionShareGrants = vi.fn().mockResolvedValue([
      { ...charlieShareGrant, resource_id: "file_image" },
      { ...charlieShareGrant, grant_id: "grant_folder_volume", resource_id: "file_volume" },
    ]);
    render(
      <ResourceBrowser
        {...baseProps}
        activeResourceCollection={nphFolder}
        onCreateResourceCollectionShareGrants={onCreateResourceCollectionShareGrants}
      />
    );

    expect(screen.getByLabelText("Active folder")).toHaveTextContent("NPH review folder");
    expect(screen.queryByRole("button", { name: "Folder actions" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Folder actions" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Share NPH review folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Share folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Snapshot folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start folder Data Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Folder Data Agent job type" })).not.toBeInTheDocument();
    expect(onCreateResourceCollectionShareGrants).not.toHaveBeenCalled();
  });

  it("shares a folder from the folder context menu without adding a visible folder toolbar", async () => {
    const onCreateResourceCollectionShareGrants = vi.fn().mockResolvedValue([
      { ...charlieShareGrant, resource_id: "file_image" },
      { ...charlieShareGrant, grant_id: "grant_folder_volume", resource_id: "file_volume" },
    ]);
    render(
      <ResourceBrowser
        {...baseProps}
        resourceCollections={[nphFolder]}
        onOpenCollection={vi.fn()}
        onCreateResourceCollectionShareGrants={onCreateResourceCollectionShareGrants}
      />
    );

    expect(screen.queryByRole("button", { name: "Share folder" })).not.toBeInTheDocument();

    fireEvent.contextMenu(screen.getByRole("button", { name: "Open folder NPH review folder" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Share" }));

    const dialog = await screen.findByRole("dialog", { name: "Share NPH review folder" });
    expect(dialog).toHaveTextContent("Add a person or team that can read every resource in this folder.");

    fireEvent.change(within(dialog).getByLabelText("Person or user ID"), {
      target: { value: " charlie " },
    });
    fireEvent.change(within(dialog).getByLabelText("Team or organization ID"), {
      target: { value: " org-c " },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Share folder" }));

    await waitFor(() => {
      expect(onCreateResourceCollectionShareGrants).toHaveBeenCalledWith(nphFolder, {
        granteeUserId: "charlie",
        granteeOrgId: "org-c",
        role: "read",
      });
    });
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Share NPH review folder" })).not.toBeInTheDocument();
    });
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
