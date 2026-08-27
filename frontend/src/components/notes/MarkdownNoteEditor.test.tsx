import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { MarkdownNoteEditor } from "@/components/notes/MarkdownNoteEditor";

describe("MarkdownNoteEditor task items", () => {
  const originalGetClientRects = Object.getOwnPropertyDescriptor(
    Range.prototype,
    "getClientRects"
  );
  const originalGetBoundingClientRect = Object.getOwnPropertyDescriptor(
    Range.prototype,
    "getBoundingClientRect"
  );

  beforeAll(() => {
    const rect = new DOMRect(0, 0, 0, 0);
    Object.defineProperty(Range.prototype, "getClientRects", {
      configurable: true,
      value: () => Object.assign([rect], { item: (index: number) => (index === 0 ? rect : null) }),
    });
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => rect,
    });
  });

  afterAll(() => {
    if (originalGetClientRects) {
      Object.defineProperty(Range.prototype, "getClientRects", originalGetClientRects);
    } else {
      delete (Range.prototype as Partial<Range>).getClientRects;
    }
    if (originalGetBoundingClientRect) {
      Object.defineProperty(
        Range.prototype,
        "getBoundingClientRect",
        originalGetBoundingClientRect
      );
    } else {
      delete (Range.prototype as Partial<Range>).getBoundingClientRect;
    }
  });

  it("exposes a keyboard-operable checkbox and serializes its checked state", async () => {
    const onMarkdownChange = vi.fn();

    render(
      <MarkdownNoteEditor
        defaultMarkdown="- [ ] Calibrate sample"
        resourceUrl={(fileId) => `/v2/files/${fileId}`}
        onMarkdownChange={onMarkdownChange}
        onBlur={() => undefined}
        onActiveStatesChange={() => undefined}
        onSelectionAnchorChange={() => undefined}
        onCaretAnchorChange={() => undefined}
        onMenuRequest={() => undefined}
        onMenuKeyDown={() => false}
        bindApi={() => undefined}
      />
    );

    const task = await screen.findByRole("checkbox", { name: "Calibrate sample" });
    expect(task).toHaveAttribute("aria-checked", "false");
    expect(task).toHaveAttribute("tabindex", "0");

    task.focus();
    expect(task).toHaveFocus();
    fireEvent.keyDown(task, { key: " ", code: "Space" });

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: "Calibrate sample" })).toHaveAttribute(
        "aria-checked",
        "true"
      );
      expect(onMarkdownChange).toHaveBeenCalledWith(
        expect.stringContaining("- [x] Calibrate sample")
      );
    });

    fireEvent.keyDown(screen.getByRole("checkbox", { name: "Calibrate sample" }), {
      key: "Enter",
    });
    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: "Calibrate sample" })).toHaveAttribute(
        "aria-checked",
        "false"
      );
      expect(onMarkdownChange).toHaveBeenLastCalledWith(
        expect.stringContaining("- [ ] Calibrate sample")
      );
    });
  });
});
