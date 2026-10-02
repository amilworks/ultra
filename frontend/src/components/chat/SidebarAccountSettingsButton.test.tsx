import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SidebarAccountSettingsButton } from "./SidebarAccountSettingsButton";

describe("SidebarAccountSettingsButton", () => {
  it("opens usage directly from the account menu", async () => {
    const onOpenUsage = vi.fn();
    render(
      <SidebarAccountSettingsButton
        authUser="researcher@example.com"
        authMode="workos"
        authIsAdmin={false}
        themePreference="system"
        onThemePreferenceChange={vi.fn()}
        onOpenUsage={onOpenUsage}
        onOpenSettings={vi.fn()}
        onLogout={vi.fn().mockResolvedValue(undefined)}
      />
    );

    fireEvent.pointerDown(screen.getByRole("button", { name: "Open account menu" }), {
      button: 0,
      ctrlKey: false,
    });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Usage" }));

    expect(onOpenUsage).toHaveBeenCalledOnce();
  });
  it("exposes the current theme as a checked radio menu item", async () => {
    const onThemePreferenceChange = vi.fn();
    render(
      <SidebarAccountSettingsButton
        authUser="researcher@example.com"
        authMode="workos"
        authIsAdmin={false}
        themePreference="system"
        onThemePreferenceChange={onThemePreferenceChange}
        onOpenUsage={vi.fn()}
        onOpenSettings={vi.fn()}
        onLogout={vi.fn().mockResolvedValue(undefined)}
      />
    );

    fireEvent.pointerDown(screen.getByRole("button", { name: "Open account menu" }), {
      button: 0,
      ctrlKey: false,
    });

    const system = await screen.findByRole("menuitemradio", { name: "System" });
    const light = screen.getByRole("menuitemradio", { name: "Light" });
    const dark = screen.getByRole("menuitemradio", { name: "Dark" });

    expect(system).toHaveAttribute("aria-checked", "true");
    expect(light).toHaveAttribute("aria-checked", "false");
    expect(dark).toHaveAttribute("aria-checked", "false");

    fireEvent.click(dark);
    expect(onThemePreferenceChange).toHaveBeenCalledOnce();
    expect(onThemePreferenceChange).toHaveBeenCalledWith("dark");
  });
});
