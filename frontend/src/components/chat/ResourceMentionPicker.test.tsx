import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ResourceMentionPicker } from "./ResourceMentionPicker";
import { resourceMentionKindLabel, resourceMentionOptionId } from "@/features/chat/resource-mention";
import type { ResourceRecord } from "@/types";

const resource = (overrides: Partial<ResourceRecord>): ResourceRecord =>
  ({
    file_id: "f1",
    original_name: "fused.ply",
    content_type: "application/octet-stream",
    size_bytes: 226_000_000,
    created_at: "2026-08-20T00:25:00Z",
    resource_kind: "file",
    source_type: "upload",
    ...overrides,
  }) as ResourceRecord;

const baseProps = {
  variant: "popover" as const,
  anchor: { left: 40 },
  listboxId: "mention-list",
  query: "fus",
  loading: false,
  activeFileId: "f1",
  onActivate: vi.fn(),
  onPick: vi.fn(),
};

describe("ResourceMentionPicker", () => {
  it("lists results as a listbox with the active row selected and the match emphasized", () => {
    const results = [resource({}), resource({ file_id: "f2", original_name: "fused_model1.ply" })];
    const { container } = render(<ResourceMentionPicker {...baseProps} results={results} />);
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(2);
    expect(options[0].getAttribute("aria-selected")).toBe("true");
    expect(options[0].id).toBe(resourceMentionOptionId("mention-list", "f1"));
    expect(container.querySelector(".composer-mention-match")?.textContent).toBe("fus");
    expect(container.querySelector(".composer-mention-kind")?.textContent).toBe("PLY");
    // Anchored at the caret on desktop.
    const picker = container.querySelector(".composer-mention-picker") as HTMLElement;
    expect(picker.style.left).toBe("40px");
  });

  it("reports hover and pointer picks, and never steals focus on mousedown", () => {
    const onActivate = vi.fn();
    const onPick = vi.fn();
    const results = [resource({})];
    render(
      <ResourceMentionPicker {...baseProps} results={results} onActivate={onActivate} onPick={onPick} />
    );
    const option = screen.getByRole("option");
    fireEvent.mouseEnter(option);
    expect(onActivate).toHaveBeenCalledWith("f1");
    const mouseDown = fireEvent.mouseDown(option);
    // mousedown was cancelled: the textarea keeps focus and its caret.
    expect(mouseDown).toBe(false);
    fireEvent.click(option);
    expect(onPick).toHaveBeenCalledWith(results[0]);
  });

  it("explains an empty list in the right voice for each cause", () => {
    const { rerender } = render(
      <ResourceMentionPicker {...baseProps} results={[]} loading />
    );
    expect(screen.getByText("Searching your library…")).toBeInTheDocument();
    rerender(<ResourceMentionPicker {...baseProps} results={[]} loading={false} />);
    expect(screen.getByText("Nothing in your library matches “fus”.")).toBeInTheDocument();
    rerender(<ResourceMentionPicker {...baseProps} results={[]} loading={false} query="" />);
    expect(screen.getByText("Your library is empty.")).toBeInTheDocument();
    rerender(
      <ResourceMentionPicker {...baseProps} results={[]} loading={false} error="boom" />
    );
    expect(screen.getByText("Your library could not be searched right now.")).toBeInTheDocument();
  });

  it("offers the upload fallback in the footer when given one", () => {
    const onUploadInstead = vi.fn();
    render(
      <ResourceMentionPicker {...baseProps} results={[]} onUploadInstead={onUploadInstead} />
    );
    fireEvent.click(screen.getByText("Upload instead…"));
    expect(onUploadInstead).toHaveBeenCalledTimes(1);
  });

  it("derives a short mono kind label from the extension, else the kind", () => {
    expect(resourceMentionKindLabel(resource({ original_name: "3percentEBSD.h5" }))).toBe("H5");
    expect(resourceMentionKindLabel(resource({ original_name: "a.ome.zarr" }))).toBe("ZARR");
    expect(
      resourceMentionKindLabel(resource({ original_name: "noext", resource_kind: "image" }))
    ).toBe("IMAGE");
    expect(
      resourceMentionKindLabel(resource({ original_name: "x.verylongext", resource_kind: "table" }))
    ).toBe("TABLE");
  });
});
