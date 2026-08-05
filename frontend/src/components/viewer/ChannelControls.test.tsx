import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { ChannelControls } from "./ChannelControls";

const colors = (count: number) =>
  Array.from({ length: count }, (_value, index) =>
    ["#3b82f6", "#22c55e", "#ef4444"][index % 3],
  );

describe("ChannelControls", () => {
  it("renders all 12 inline choices and disables only inactive choices at the composite cap", async () => {
    const channelNames = Array.from({ length: 12 }, (_value, index) =>
      index < 2 ? "Duplicate" : `Band ${index + 1}`,
    );

    function Harness() {
      const [selected, setSelected] = useState([5, 1, 3, 0, 2, 4, 6, 7]);
      return (
        <ChannelControls
          channelNames={channelNames}
          channelColors={colors(channelNames.length)}
          selectedIndices={selected}
          canEditColor={false}
          singleChannelMode={false}
          onToggleChannel={(index, active) => {
            setSelected((current) =>
              active ? current.filter((value) => value !== index) : [...current, index],
            );
          }}
          onSetChannelColor={() => {}}
        />
      );
    }

    const { container } = render(<Harness />);
    const group = screen.getByRole("group", { name: "Channels" });
    expect(within(group).getAllByRole("button")).toHaveLength(12);
    expect(screen.queryByRole("button", { name: /Choose channels/ })).toBeNull();
    expect(container.querySelectorAll('[data-viewer-channel-chip="true"]')).toHaveLength(12);

    const firstDuplicate = screen.getByRole("button", {
      name: "Duplicate, source channel 0",
    });
    const secondDuplicate = screen.getByRole("button", {
      name: "Duplicate, source channel 1",
    });
    expect(firstDuplicate).toBeEnabled();
    expect(secondDuplicate).toBeEnabled();
    expect(screen.getByRole("button", { name: "Band 9, source channel 8" })).toBeDisabled();
    expect(screen.getByText(/Maximum 8 channels selected/)).toBeInTheDocument();

    fireEvent.click(firstDuplicate);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Band 9, source channel 8" })).toBeEnabled();
    });
  });

  it("switches at 13 channels to a 320px virtual dialog and resets focus, query, and scroll", async () => {
    const channelNames = Array.from({ length: 13 }, () => "Signal");
    const portalContainer = document.createElement("section");
    document.body.append(portalContainer);

    const { container } = render(
      <ChannelControls
        channelNames={channelNames}
        channelColors={colors(channelNames.length)}
        selectedIndices={[12, 5, 1, 3]}
        canEditColor={false}
        singleChannelMode={false}
        portalContainer={portalContainer}
        onToggleChannel={() => {}}
        onSetChannelColor={() => {}}
      />,
    );

    expect(container.querySelectorAll('[data-viewer-channel-chip="true"]')).toHaveLength(4);
    expect(
      Array.from(container.querySelectorAll(".viewer-channel-toggle")).map((element) =>
        element.getAttribute("aria-label"),
      ),
    ).toEqual([
      "Signal, source channel 12",
      "Signal, source channel 5",
      "Signal, source channel 1",
      "Signal, source channel 3",
    ]);

    const trigger = screen.getByRole("button", {
      name: "Choose channels, 4 selected of 13",
    });
    fireEvent.click(trigger);
    const dialog = await screen.findByRole("dialog", { name: "Channels" });
    expect(portalContainer.contains(dialog)).toBe(true);

    const search = screen.getByRole("textbox", { name: "Search channels" });
    expect(search).toHaveFocus();
    const viewport = screen.getByLabelText("Channel catalog");
    expect(viewport).toHaveStyle({ height: "320px" });
    expect(viewport.style.getPropertyValue("--viewer-channel-row-height")).toBe("42px");
    expect(dialog.querySelectorAll('[data-viewer-channel-row="true"]')).toHaveLength(12);

    viewport.scrollTop = 420;
    fireEvent.scroll(viewport);
    await waitFor(() => {
      expect(dialog.querySelectorAll('[data-viewer-channel-row="true"]').length).toBeLessThanOrEqual(12);
    });

    fireEvent.change(search, { target: { value: "not-a-channel" } });
    expect(await screen.findByText("No matching channels")).toHaveAttribute("role", "status");
    expect(screen.getByText(/0 matching, 4 selected/)).toBeInTheDocument();

    fireEvent.keyDown(search, { key: "Escape", code: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Channels" })).toBeNull());
    expect(trigger).toHaveFocus();

    fireEvent.click(trigger);
    const reopenedSearch = await screen.findByRole("textbox", { name: "Search channels" });
    expect(reopenedSearch).toHaveValue("");
    expect(reopenedSearch).toHaveFocus();
    expect(screen.getByLabelText("Channel catalog").scrollTop).toBe(0);

    fireEvent.keyDown(reopenedSearch, { key: "Escape", code: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Channels" })).toBeNull());
    portalContainer.remove();
  });

  it("keeps sparse high source indices and semantic order while capping normalized selections", async () => {
    const channelNames = Array.from({ length: 260 }, (_value, index) => `Band ${index + 1}`);
    const onToggleChannel = vi.fn();
    const { container } = render(
      <ChannelControls
        channelNames={channelNames}
        channelColors={colors(channelNames.length)}
        selectedIndices={[259, 5, 1, 3, 5, -1, 1.5, Number.NaN, 260, 7, 8, 9, 10, 11]}
        canEditColor={false}
        singleChannelMode={false}
        onToggleChannel={onToggleChannel}
        onSetChannelColor={() => {}}
      />,
    );

    expect(
      Array.from(container.querySelectorAll(".viewer-channel-toggle")).map((element) =>
        element.getAttribute("aria-label"),
      ),
    ).toEqual([
      "Band 260, source channel 259",
      "Band 6, source channel 5",
      "Band 2, source channel 1",
      "Band 4, source channel 3",
      "Band 8, source channel 7",
      "Band 9, source channel 8",
      "Band 10, source channel 9",
      "Band 11, source channel 10",
    ]);

    fireEvent.click(screen.getByRole("button", { name: /Choose channels, 8 selected/ }));
    const dialog = await screen.findByRole("dialog", { name: "Channels" });
    const search = await screen.findByRole("textbox", { name: "Search channels" });
    const viewport = screen.getByLabelText("Channel catalog");
    viewport.scrollTop = 260 * 42 - 320;
    fireEvent.scroll(viewport);
    await waitFor(() => {
      expect(
        within(dialog).getByRole("button", { name: "Band 260, source channel 259" }),
      ).toBeInTheDocument();
    });
    expect(within(dialog).queryByRole("button", { name: "Band 1, source channel 0" })).toBeNull();
    expect(dialog.querySelectorAll('[data-viewer-channel-row="true"]')).toHaveLength(12);
    expect(
      within(dialog)
        .getByRole("button", { name: "Band 260, source channel 259" })
        .closest('[data-viewer-channel-row="true"]'),
    ).toHaveStyle({ transform: "translateY(10878px)" });

    fireEvent.change(search, { target: { value: "C258" } });
    expect(viewport.scrollTop).toBe(0);
    const inactiveHighChannel = await screen.findByRole("button", {
      name: "Band 259, source channel 258",
    });
    expect(inactiveHighChannel).toBeDisabled();
    expect(screen.getByText("Remove a channel to choose another.")).toBeInTheDocument();

    fireEvent.change(search, { target: { value: "C259" } });
    const selectedHighChannel = await screen.findByRole("button", {
      name: "Band 260, source channel 259",
    });
    expect(selectedHighChannel).toBeEnabled();
    fireEvent.click(selectedHighChannel);
    expect(onToggleChannel).toHaveBeenCalledWith(259, true);

    fireEvent.keyDown(selectedHighChannel, { key: "Escape", code: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Channels" })).toBeNull());
    fireEvent.click(screen.getByRole("button", { name: /Choose channels, 8 selected/ }));
    const reopenedDialog = await screen.findByRole("dialog", { name: "Channels" });
    expect(screen.getByLabelText("Channel catalog").scrollTop).toBe(0);
    expect(
      reopenedDialog.querySelector('[data-viewer-channel-row="true"]'),
    ).toHaveStyle({ transform: "translateY(0px)" });
  });

  it("keeps every channel enabled as a replacement in single-channel mode", async () => {
    const onToggleChannel = vi.fn();
    render(
      <ChannelControls
        channelNames={Array.from({ length: 13 }, (_value, index) => `Band ${index + 1}`)}
        channelColors={colors(13)}
        selectedIndices={[12]}
        canEditColor={false}
        singleChannelMode
        onToggleChannel={onToggleChannel}
        onSetChannelColor={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Choose channels/ }));
    const search = await screen.findByRole("textbox", { name: "Search channels" });
    fireEvent.change(search, { target: { value: "C5" } });
    const replacement = await screen.findByRole("button", {
      name: "Band 6, source channel 5",
    });
    expect(replacement).toBeEnabled();
    fireEvent.click(replacement);
    expect(onToggleChannel).toHaveBeenCalledWith(5, false);
  });
});
