import { act, render } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { ComposerEditor } from "./ComposerEditor";
import type { ComposerEditorProps, ComposerHandle } from "./composerHandle";

/* The editor under jsdom: no layout, so nothing here measures — but the
   document, the commands, the emitted events and the handle are all real. */

const registry = [{ label: "scan.tif", fileId: "f1" }];

const setup = (overrides: Partial<ComposerEditorProps> = {}) => {
  const ref = createRef<ComposerHandle>();
  const props: ComposerEditorProps = {
    value: "",
    tokens: registry,
    goneFileIds: [],
    disabled: false,
    placeholder: "Ask Ultra",
    ariaLabel: "Ask Ultra",
    mentionOpen: false,
    tokenDetails: (fileId) => (fileId === "f1" ? { title: "scan.tif · TIF · 2 MB", kind: "TIF" } : null),
    onValueChange: vi.fn(),
    onTokensChange: vi.fn(),
    onMentionChange: vi.fn(),
    onFocusChange: vi.fn(),
    onEnter: vi.fn(() => true),
    ...overrides,
  };
  const utils = render(<ComposerEditor ref={ref} {...props} />);
  return { ref, props, ...utils };
};

describe("ComposerEditor", () => {
  it("mounts a labelled textbox that names its placeholder but never draws it", () => {
    // A placeholder widget before ProseMirror's trailing <br> made an empty
    // editor two lines tall; the layout draws the hint instead.
    const { container } = setup();
    const editor = container.querySelector(".ProseMirror");
    expect(editor?.getAttribute("role")).toBe("textbox");
    expect(editor?.getAttribute("aria-label")).toBe("Ask Ultra");
    expect(editor?.getAttribute("aria-placeholder")).toBe("Ask Ultra");
    expect(container.querySelector(".composer-placeholder")).toBeNull();
    expect(container.querySelectorAll(".ProseMirror p").length).toBe(1);
  });

  it("emits the serialised text and the tokens it holds, and reports the @ mention", () => {
    const { ref, props } = setup();
    act(() => ref.current?.insertText("Register @sc"));
    expect(props.onValueChange).toHaveBeenLastCalledWith("Register @sc");
    expect(props.onMentionChange).toHaveBeenLastCalledWith({ query: "sc" });
    act(() => ref.current?.acceptMention(registry[0]));
    expect(props.onValueChange).toHaveBeenLastCalledWith("Register @scan.tif ");
    expect(props.onTokensChange).toHaveBeenLastCalledWith(registry);
    expect(props.onMentionChange).toHaveBeenLastCalledWith(null);
    expect(ref.current?.value).toBe("Register @scan.tif ");
  });

  it("draws a token as a real node with its kind, name, and a labelled remove", () => {
    const { ref, container } = setup();
    act(() => ref.current?.insertText("see @sc"));
    act(() => ref.current?.acceptMention(registry[0]));
    const token = container.querySelector(".composer-token");
    expect(token?.getAttribute("contenteditable")).toBe("false");
    expect(token?.querySelector(".composer-token-kind")?.textContent).toBe("TIF");
    expect(token?.querySelector(".composer-token-name")?.textContent).toBe("scan.tif");
    expect(token?.querySelector(".composer-token-remove")?.getAttribute("aria-label")).toBe("Remove scan.tif");
    expect(token?.getAttribute("title")).toBe("scan.tif · TIF · 2 MB");
  });

  it("refreshes a token's details when the app learns more about its file", () => {
    let known = false;
    const tokenDetails = (fileId: string) =>
      fileId === "f1"
        ? known
          ? { title: "scan.tif · TIF · 2 MB", kind: "TIF" }
          : { title: "scan.tif — no longer in your library", kind: "", gone: true }
        : null;
    const { ref, container, rerender, props } = setup({ tokenDetails });
    act(() => ref.current?.appendToken(registry[0]));
    expect(container.querySelector(".composer-token")?.classList.contains("composer-token-gone")).toBe(true);
    known = true;
    rerender(<ComposerEditor ref={ref} {...props} tokenDetails={(id) => tokenDetails(id)} />);
    expect(container.querySelector(".composer-token")?.classList.contains("composer-token-gone")).toBe(false);
    expect(container.querySelector(".composer-token-kind")?.textContent).toBe("TIF");
  });

  it("appends a token once, at the end when unfocused, and removes it whole", () => {
    const { ref, props } = setup({ value: "one two" });
    act(() => ref.current?.appendToken(registry[0]));
    expect(ref.current?.value).toBe("one two @scan.tif ");
    act(() => ref.current?.appendToken(registry[0]));
    expect(ref.current?.value).toBe("one two @scan.tif ");
    act(() => ref.current?.removeToken("f1"));
    expect(ref.current?.value).toBe("one two");
    expect(props.onTokensChange).toHaveBeenLastCalledWith([]);
  });

  it("follows an external value without echoing it back", () => {
    const { ref, props, rerender } = setup({ value: "first" });
    rerender(<ComposerEditor ref={ref} {...props} value="second @scan.tif" />);
    expect(ref.current?.value).toBe("second @scan.tif");
    expect(props.onValueChange).not.toHaveBeenCalled();
    expect(props.onTokensChange).not.toHaveBeenCalled();
  });

  it("recovers a pill when a label is registered after its text was written", () => {
    const { ref, props, rerender, container } = setup({ value: "use @scan.tif here", tokens: [] });
    expect(container.querySelector(".composer-token")).toBeNull();
    rerender(<ComposerEditor ref={ref} {...props} tokens={registry} />);
    expect(container.querySelector(".composer-token")).not.toBeNull();
    expect(ref.current?.value).toBe("use @scan.tif here");
  });

  it("turns a gone token back into an open mention", () => {
    const { ref, props } = setup({ value: "fix @scan.tif now" });
    act(() => ref.current?.reopenMentionFor("f1"));
    expect(ref.current?.value).toBe("fix @ now");
    expect(props.onMentionChange).toHaveBeenLastCalledWith({ query: "" });
  });
});
