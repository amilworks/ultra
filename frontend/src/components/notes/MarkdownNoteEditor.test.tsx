import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { TextSelection } from "@milkdown/kit/prose/state";
import { CellSelection } from "@milkdown/kit/prose/tables";
import { undo, redo } from "@milkdown/kit/prose/history";
import type { EditorView } from "@milkdown/kit/prose/view";
import { MarkdownNoteEditor, type NotesEditorHandle } from "@/components/notes/MarkdownNoteEditor";

const TABLE_NOTE = [
  "## Transects",
  "- Spacing 40 m",
  "- [x] Flag GSD < 1.2 cm",
  "- [ ] Re-fly plot 7",
  "",
  "| Plot | GSD | Pass |",
  "| --- | --- | --- |",
  "| 3 | 1.1 cm | yes |",
  "| 7 | 1.4 cm | no |",
].join("\n");

describe("MarkdownNoteEditor", () => {
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

  it("publishes a handle only when the formatted body can receive focus", async () => {
    const focusedOnBind: boolean[] = [];
    render(
      <MarkdownNoteEditor
        defaultMarkdown="Calibrate sample"
        resourceUrl={(fileId) => `/v2/files/${fileId}`}
        onMarkdownChange={() => undefined}
        onBlur={() => undefined}
        onActiveStatesChange={() => undefined}
        onSelectionAnchorChange={() => undefined}
        onCaretAnchorChange={() => undefined}
        onMenuRequest={() => undefined}
        onMenuKeyDown={() => false}
        bindApi={(api) => {
          if (!api) return;
          api.focus();
          focusedOnBind.push(document.activeElement?.getAttribute("aria-label") === "Note body");
        }}
      />
    );
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Note body" })).toHaveFocus());
    expect(focusedOnBind.length).toBeGreaterThan(0);
    expect(focusedOnBind.every(Boolean)).toBe(true);
  });

  it("reports the native blur destination without discarding the toolbar selection anchor", async () => {
    const onBlur = vi.fn();
    const onSelectionAnchorChange = vi.fn();
    const bindApi = vi.fn();
    render(
      <>
        <MarkdownNoteEditor
          defaultMarkdown="Select these words"
          resourceUrl={(fileId) => `/v2/files/${fileId}`}
          onMarkdownChange={() => undefined}
          onBlur={onBlur}
          onActiveStatesChange={() => undefined}
          onSelectionAnchorChange={onSelectionAnchorChange}
          onCaretAnchorChange={() => undefined}
          onMenuRequest={() => undefined}
          onMenuKeyDown={() => false}
          bindApi={bindApi}
        />
        <button type="button">Formatting action</button>
      </>
    );
    const body = await screen.findByRole("textbox", { name: "Note body" });
    const action = screen.getByRole("button", { name: "Formatting action" });
    act(() => body.focus());
    onSelectionAnchorChange.mockClear();
    act(() => action.focus());
    expect(onBlur).toHaveBeenCalledTimes(1);
    expect(onBlur.mock.calls[0]?.[0]).toBeInstanceOf(FocusEvent);
    expect(onBlur.mock.calls[0]?.[0].relatedTarget).toBe(action);
    expect(onSelectionAnchorChange).not.toHaveBeenCalled();
    expect(action).toHaveFocus();
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

  const renderLinkSelection = async (markdown: string, from = 1) => {
    const onActiveStatesChange = vi.fn();
    const handle: { current: NotesEditorHandle | null } = { current: null };
    render(
      <MarkdownNoteEditor
        defaultMarkdown={markdown}
        resourceUrl={(fileId) => `/v2/files/${fileId}`}
        onMarkdownChange={() => undefined}
        onBlur={() => undefined}
        onActiveStatesChange={onActiveStatesChange}
        onSelectionAnchorChange={() => undefined}
        onCaretAnchorChange={() => undefined}
        onMenuRequest={() => undefined}
        onMenuKeyDown={() => false}
        bindApi={(api) => { handle.current = api; }}
      />
    );
    // Use the existing development harness to select real ProseMirror text;
    // jsdom does not implement native contenteditable selection movement.
    const getView = () => (window as unknown as {
      __notesEditorView?: () => EditorView | undefined;
    }).__notesEditorView?.();
    await waitFor(() => expect(getView()).toBeDefined());
    const view = getView()!;
    act(() => {
      view.dispatch(view.state.tr.setSelection(TextSelection.create(view.state.doc, from, from + 5)));
    });
    await waitFor(() => expect(onActiveStatesChange).toHaveBeenCalled());
    return { view, handle, onActiveStatesChange };
  };

  it("reads the URL from a link selected at its left boundary", async () => {
    const { onActiveStatesChange } = await renderLinkSelection(
      "Read [liana](https://example.org/paper) notebook", 6
    );
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({
      link: true, linkHref: "https://example.org/paper",
    }));
  });

  it("publishes an applied link without requiring the selection to move", async () => {
    const { view, handle, onActiveStatesChange } = await renderLinkSelection("liana notebook");
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({ link: false }));
    const selection = view.state.selection;

    act(() => handle.current!.applyLink("https://example.org/paper"));

    expect(view.dom.querySelector("a")).toHaveAttribute("href", "https://example.org/paper");
    expect(view.state.selection.eq(selection)).toBe(true);
    // Opening the link controls immediately must read the new URL and offer Remove.
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({
      link: true, linkHref: "https://example.org/paper",
    }));
  });

  it("refreshes link controls after removal, undo, and redo with the same selection", async () => {
    const { view, handle, onActiveStatesChange } = await renderLinkSelection(
      "[liana](https://example.org/paper) notebook"
    );
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({ link: true }));
    const selection = view.state.selection;

    act(() => handle.current!.removeLink());
    expect(view.dom.querySelector("a")).toBeNull();
    expect(view.state.selection.eq(selection)).toBe(true);
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({
      link: false, linkHref: null,
    }));

    act(() => { undo(view.state, view.dispatch); });
    expect(view.dom.querySelector("a")).toHaveAttribute("href", "https://example.org/paper");
    expect(view.state.selection.eq(selection)).toBe(true);
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({
      link: true, linkHref: "https://example.org/paper",
    }));

    act(() => { redo(view.state, view.dispatch); });
    expect(view.dom.querySelector("a")).toBeNull();
    expect(view.state.selection.eq(selection)).toBe(true);
    expect(onActiveStatesChange).toHaveBeenLastCalledWith(expect.objectContaining({
      link: false, linkHref: null,
    }));
  });



  const renderLinkedNote = async (markdown: string, from: number, to = from) => {
    const handle: { current: NotesEditorHandle | null } = { current: null };
    const onMarkdownChange = vi.fn();
    render(
      <MarkdownNoteEditor
        defaultMarkdown={markdown}
        resourceUrl={(fileId) => `/v2/files/${fileId}`}
        onMarkdownChange={onMarkdownChange}
        onBlur={() => undefined}
        onActiveStatesChange={() => undefined}
        onSelectionAnchorChange={() => undefined}
        onCaretAnchorChange={() => undefined}
        onMenuRequest={() => undefined}
        onMenuKeyDown={() => false}
        bindApi={(api) => { handle.current = api; }}
      />
    );
    // jsdom has no native contenteditable caret movement; use the existing
    // development hook to select text in the real editor document.
    const getView = () => (window as unknown as {
      __notesEditorView?: () => EditorView | undefined;
    }).__notesEditorView?.();
    await waitFor(() => expect(getView()).toBeDefined());
    const view = getView()!;
    act(() => {
      view.dispatch(view.state.tr.setSelection(TextSelection.create(view.state.doc, from, to)));
    });
    return { view, handle, onMarkdownChange };
  };

  it("unlinks the whole link at a caret across formatting while preserving its neighbors and undo", async () => {
    const { view, handle, onMarkdownChange } = await renderLinkedNote(
      'Before [one **bold** end](https://example.org/target "target")[next](https://example.org/target "neighbor") after\n\n[other](https://example.org/target "target")',
      14
    );
    const text = view.state.doc.textContent;
    const selection = view.state.selection;
    const before = view.dom.innerHTML;

    act(() => handle.current!.removeLink());

    expect(view.dom.querySelector('a[title="target"]')?.textContent).toBe("other");
    expect(Array.from(view.dom.querySelectorAll("a"), (a) => a.textContent)).toEqual(["next", "other"]);
    expect(view.dom.querySelector("strong")).toHaveTextContent("bold");
    expect(view.state.doc.textContent).toBe(text);
    expect(view.state.selection.eq(selection)).toBe(true);
    expect(view.hasFocus()).toBe(true);
    await waitFor(() => expect(onMarkdownChange).toHaveBeenLastCalledWith(
      expect.stringContaining('Before one **bold** end[next](https://example.org/target "neighbor") after')
    ));

    act(() => { undo(view.state, view.dispatch); });
    expect(view.dom.innerHTML).toBe(before);
    expect(view.state.selection.eq(selection)).toBe(true);
    act(() => { redo(view.state, view.dispatch); });
    expect(Array.from(view.dom.querySelectorAll("a"), (a) => a.textContent)).toEqual(["next", "other"]);
    expect(view.state.selection.eq(selection)).toBe(true);
  });

  it.each([1, 6])("unlinks at the active link boundary %s without moving the caret", async (position) => {
    const { view, handle } = await renderLinkedNote("[liana](https://example.org/paper) notebook", position);
    const selection = view.state.selection;
    act(() => handle.current!.removeLink());
    expect(view.dom.querySelector("a")).toBeNull();
    expect(view.state.selection.eq(selection)).toBe(true);
    act(() => { view.dispatch(view.state.tr.insertText("X")); });
    expect(view.dom.querySelector("a")).toBeNull();
  });

  it("keeps nonempty removal limited to the selected text", async () => {
    const { view, handle } = await renderLinkedNote("[liana](https://example.org/paper) notebook", 2, 5);
    const selection = view.state.selection;
    act(() => handle.current!.removeLink());
    expect(Array.from(view.dom.querySelectorAll("a"), (a) => a.textContent)).toEqual(["l", "a"]);
    expect(view.state.doc.textContent).toBe("liana notebook");
    expect(view.state.selection.eq(selection)).toBe(true);
  });

  it("unlinks every selected table cell while preserving unselected cells", async () => {
    const link = (text: string) => `[${text}](https://example.org/${text})`;
    const { view, handle } = await renderLinkedNote(
      `| Left | Right |\n| --- | --- |\n| ${link("A")} | ${link("B")} |\n| ${link("C")} | ${link("D")} |`,
      4
    );
    const cells: number[] = [];
    view.state.doc.descendants((node, position) => {
      if (node.type.name === "table_cell") cells.push(position);
    });
    expect(cells).toHaveLength(4);
    act(() => {
      view.dispatch(view.state.tr.setSelection(CellSelection.create(view.state.doc, cells[0], cells[2])));
    });
    const selection = view.state.selection;
    act(() => handle.current!.removeLink());
    expect(Array.from(view.dom.querySelectorAll("a"), (a) => a.textContent)).toEqual(["B", "D"]);
    expect(view.state.selection.eq(selection)).toBe(true);
  });

  it.each([false, true])("does not create a link when removal targets plain text (range=%s)", async (range) => {
    const { view, handle } = await renderLinkedNote("plain [link](https://example.org/paper)", 2, range ? 4 : 2);
    const before = view.state.doc;
    act(() => handle.current!.removeLink());
    expect(view.state.doc.eq(before)).toBe(true);
    expect(view.dom.querySelectorAll("a")).toHaveLength(1);
  });
  it("does not report trailing-paragraph normalization as an edit", async () => {
    const onMarkdownChange = vi.fn();
    let editor: NotesEditorHandle | null = null;

    render(
      <MarkdownNoteEditor
        defaultMarkdown={TABLE_NOTE}
        resourceUrl={(fileId) => `/resources/${fileId}`}
        onMarkdownChange={onMarkdownChange}
        onBlur={() => undefined}
        onActiveStatesChange={() => undefined}
        onSelectionAnchorChange={() => undefined}
        onCaretAnchorChange={() => undefined}
        onMenuRequest={() => undefined}
        onMenuKeyDown={() => false}
        bindApi={(nextEditor) => {
          editor = nextEditor;
        }}
      />
    );

    await screen.findByLabelText("Note body");
    await waitFor(() => expect(editor).not.toBeNull());
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 350));
    });

    expect(onMarkdownChange).not.toHaveBeenCalled();

    act(() => {
      editor?.insertMarkdown("User edit");
    });
    await waitFor(() => expect(onMarkdownChange).toHaveBeenCalledTimes(1));
    expect(onMarkdownChange.mock.calls[0]?.[0]).toContain("User edit");
  });
});
