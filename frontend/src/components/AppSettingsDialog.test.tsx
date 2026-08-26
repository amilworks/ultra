import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { CurrentUserResponse } from "@/types";

import { AppSettingsDialog } from "./AppSettingsDialog";

const currentUser: CurrentUserResponse = {
  user: {
    user_id: "user_ada",
    email: "ada@example.com",
    display_name: "Dr. Ada Lovelace",
    role: "user",
  },
  profile: {
    display_name: "Dr. Ada Lovelace",
    title: "Principal Investigator",
    institution: "UCSB Vision Research Lab",
    research_interests: "Cell segmentation",
    bio: "I run a microscopy lab.",
  },
};

describe("AppSettingsDialog Profile save", () => {
  it("keeps the save control focusable while a failed request can be retried", async () => {
    let rejectSave: (reason?: unknown) => void = () => undefined;
    const saveProfile = vi.fn(
      () =>
        new Promise<CurrentUserResponse>((_resolve, reject) => {
          rejectSave = reject;
        })
    );

    render(
      <AppSettingsDialog
        open
        onOpenChange={vi.fn()}
        initialTab="profile"
        authUser="ada@example.com"
        authMode="workos"
        authIsAdmin={false}
        bisqueCredentialsLinked={false}
        themePreference="system"
        resolvedTheme="light"
        bisqueNavLinks={null}
        onThemePreferenceChange={vi.fn()}
        onOpenAdmin={vi.fn()}
        onLogout={vi.fn(async () => undefined)}
        onUnlinkBisqueAccount={vi.fn(async () => undefined)}
        onLinkBisqueAccount={vi.fn(async () => ({ imageCount: 0 }))}
        loadProfile={vi.fn(async () => currentUser)}
        saveProfile={saveProfile}
        formatError={(error) =>
          error instanceof Error ? error.message : "Unable to save profile"
        }
      />
    );

    const saveButton = await screen.findByRole("button", { name: "Save profile" });
    saveButton.focus();
    fireEvent.click(saveButton);

    await waitFor(() => expect(saveButton).toHaveAccessibleName("Saving..."));
    expect(saveButton).toHaveFocus();
    expect(saveButton).not.toBeDisabled();
    expect(saveButton).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(saveButton);
    expect(saveProfile).toHaveBeenCalledTimes(1);

    await act(async () => {
      rejectSave(new Error("Profile service unavailable"));
    });

    await waitFor(() => expect(saveButton).toHaveAccessibleName("Save profile"));
    expect(saveButton).toHaveFocus();
    expect(saveButton).not.toHaveAttribute("aria-disabled");
    expect(screen.getByText("Profile service unavailable")).toBeInTheDocument();

    saveProfile.mockResolvedValueOnce(currentUser);
    fireEvent.click(saveButton);

    await waitFor(() => expect(saveProfile).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(saveButton).toHaveAccessibleName("Saved"));
    expect(saveButton).toHaveFocus();
  });
});
