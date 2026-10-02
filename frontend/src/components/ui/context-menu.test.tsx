import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "./context-menu";

function Menu({ children = "Resource", ...props }: ComponentProps<typeof ContextMenuTrigger>) {
  return (
    <ContextMenu>
      <ContextMenuTrigger tabIndex={0} {...props}>{children}</ContextMenuTrigger>
      <ContextMenuContent><ContextMenuItem>View</ContextMenuItem></ContextMenuContent>
    </ContextMenu>
  );
}

describe("ContextMenu keyboard opening", () => {
  it("opens at the focused trigger with Shift+F10 and returns focus on Escape", async () => {
    const onContextMenu = vi.fn();
    render(<Menu onContextMenu={onContextMenu} />);
    const trigger = screen.getByText("Resource");
    vi.spyOn(trigger, "getBoundingClientRect").mockReturnValue(new DOMRect(80, 120, 200, 90));
    trigger.focus();
    expect(fireEvent.keyDown(trigger, { key: "F10", shiftKey: true })).toBe(false);
    expect(onContextMenu).toHaveBeenCalledTimes(1);
    expect(onContextMenu.mock.calls[0][0]).toMatchObject({ clientX: 88, clientY: 128 });
    const item = await screen.findByRole("menuitem", { name: "View" });
    await waitFor(() => expect(item).toHaveFocus());
    fireEvent.keyDown(item, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it.each(["disabled", "caller", "child"])("respects %s cancellation", (source) => {
    const cancel = vi.fn((event) => event.preventDefault());
    render(source === "child"
      ? <Menu asChild><button onKeyDown={cancel}>Resource</button></Menu>
      : <Menu disabled={source === "disabled"} onKeyDown={source === "caller" ? cancel : undefined} />);
    fireEvent.keyDown(screen.getByText("Resource"), { key: "F10", shiftKey: true });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    if (source !== "disabled") expect(cancel).toHaveBeenCalledOnce();
  });

  it("preserves a descendant control's keyboard context menu", () => {
    const onContextMenu = vi.fn();
    render(<Menu onContextMenu={onContextMenu}><input aria-label="Rename" /></Menu>);
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "F10", shiftKey: true });
    expect(onContextMenu).not.toHaveBeenCalled();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it.each([
    { key: "F10" },
    { key: "F10", shiftKey: true, altKey: true },
    { key: "F10", shiftKey: true, ctrlKey: true },
    { key: "F10", shiftKey: true, metaKey: true },
    { key: "F10", shiftKey: true, isComposing: true },
  ])("leaves unrelated or composing keystrokes alone: %j", (key) => {
    render(<Menu />);
    expect(fireEvent.keyDown(screen.getByText("Resource"), key)).toBe(true);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
