import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Sidebar,
  SidebarContent,
  SidebarMenu,
  SidebarProvider,
} from "@/components/ui/sidebar";
import type { HistoryItem } from "@/features/chat/history";
import { ConversationHistoryRow } from "./ConversationHistoryRow";

const conversation: HistoryItem = {
  id: "conversation-1",
  title: "Stratified Clinical Trial Analysis",
  preview: "",
  period: "Today",
  running: false,
  messageCount: 2,
};

const setDesktopViewport = (): void => {
  Object.defineProperty(window, "innerWidth", {
    value: 1280,
    writable: true,
    configurable: true,
  });
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
  );
};

const renderHistoryRow = () => {
  const callbacks = {
    onOpen: vi.fn(),
    onCopyLink: vi.fn(async () => undefined),
    onCopyId: vi.fn(async () => undefined),
    onRename: vi.fn(),
    onDelete: vi.fn(),
  };

  render(
    <SidebarProvider>
      <Sidebar collapsible="none">
        <SidebarContent>
          <SidebarMenu>
            <ConversationHistoryRow
              conversation={conversation}
              active={false}
              deleting={false}
              renaming={false}
              {...callbacks}
            />
          </SidebarMenu>
        </SidebarContent>
      </Sidebar>
    </SidebarProvider>
  );

  return callbacks;
};

describe("ConversationHistoryRow", () => {
  beforeEach(() => {
    setDesktopViewport();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("opens the chat actions from a context click without rendering a hover ellipsis", async () => {
    const callbacks = renderHistoryRow();
    const row = screen.getByRole("button", { name: conversation.title });

    expect(
      screen.queryByRole("button", { name: `Conversation actions for ${conversation.title}` })
    ).not.toBeInTheDocument();

    fireEvent.contextMenu(row, { clientX: 80, clientY: 120 });

    const renameItem = await screen.findByRole("menuitem", { name: "Rename chat" });
    expect(screen.getByRole("menuitem", { name: "Copy chat link" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Copy chat ID" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Delete chat" })).toBeInTheDocument();
    const menuIcons = screen.getByRole("menu").querySelectorAll(".conversation-history-menu-icon");
    expect(menuIcons).toHaveLength(4);
    for (const icon of menuIcons) {
      expect(icon).toHaveAttribute("aria-hidden", "true");
    }

    fireEvent.click(renameItem);

    expect(callbacks.onRename).toHaveBeenCalledWith(conversation.id, conversation.title);
    expect(callbacks.onOpen).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.queryByRole("menuitem", { name: "Rename chat" })).not.toBeInTheDocument();
    });
  });

  it("supports the standard keyboard context-menu shortcut", async () => {
    renderHistoryRow();
    const row = screen.getByRole("button", { name: conversation.title });

    row.focus();
    fireEvent.keyDown(row, { key: "F10", shiftKey: true });

    expect(await screen.findByRole("menuitem", { name: "Rename chat" })).toBeInTheDocument();
    expect(row).toHaveAttribute("aria-keyshortcuts", "Shift+F10");

    fireEvent.keyDown(screen.getByRole("menu"), { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByRole("menuitem", { name: "Rename chat" })).not.toBeInTheDocument();
    });
  });

  it("opens the same menu after a touch long-press", async () => {
    const callbacks = renderHistoryRow();
    const row = screen.getByRole("button", { name: conversation.title });

    fireEvent.pointerDown(row, {
      pointerType: "touch",
      pointerId: 1,
      button: 0,
      buttons: 1,
      clientX: 80,
      clientY: 120,
    });

    expect(
      await screen.findByRole("menuitem", { name: "Rename chat" }, { timeout: 1_200 })
    ).toBeInTheDocument();

    fireEvent.pointerUp(row, {
      pointerType: "touch",
      pointerId: 1,
      button: 0,
      clientX: 80,
      clientY: 120,
    });
    expect(callbacks.onOpen).not.toHaveBeenCalled();
  });
});
