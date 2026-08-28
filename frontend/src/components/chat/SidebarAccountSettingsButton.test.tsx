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
});
