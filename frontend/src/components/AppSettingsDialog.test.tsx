import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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

const horizontalRect = (left: number, right: number): DOMRect =>
  ({
    x: left,
    y: 0,
    top: 0,
    right,
    bottom: 40,
    left,
    width: right - left,
    height: 40,
    toJSON: () => ({}),
  }) as DOMRect;

afterEach(() => {
  vi.unstubAllGlobals();
});

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

describe("AppSettingsDialog BisQue link", () => {
  it("keeps pending controls focusable and leaves a failed attempt ready to retry", async () => {
    let rejectLink: (reason?: unknown) => void = () => undefined;
    const onLinkBisqueAccount = vi.fn(
      () =>
        new Promise<{ imageCount: number }>((_resolve, reject) => {
          rejectLink = reject;
        })
    );

    render(
      <AppSettingsDialog
        open
        onOpenChange={vi.fn()}
        initialTab="bisque"
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
        onLinkBisqueAccount={onLinkBisqueAccount}
        formatError={(error) =>
          error instanceof Error ? error.message : "Unable to link BisQue"
        }
      />
    );

    const usernameInput = screen.getByLabelText("Username");
    const passwordInput = screen.getByLabelText("Password");
    const linkForm = passwordInput.closest("form") as HTMLFormElement;
    fireEvent.change(usernameInput, { target: { value: "ada" } });
    fireEvent.change(passwordInput, { target: { value: "secret" } });

    const linkButton = screen.getByRole("button", { name: "Link account" });
    linkButton.focus();
    fireEvent.click(linkButton);

    await waitFor(() => expect(linkButton).toHaveAccessibleName("Testing account..."));
    expect(linkButton).toHaveFocus();
    expect(linkButton).not.toBeDisabled();
    expect(linkButton).toHaveAttribute("aria-disabled", "true");
    expect(usernameInput).not.toBeDisabled();
    expect(usernameInput).toHaveAttribute("readonly");
    expect(usernameInput).toHaveAttribute("aria-disabled", "true");
    expect(passwordInput).not.toBeDisabled();
    expect(passwordInput).toHaveAttribute("readonly");
    expect(passwordInput).toHaveAttribute("aria-disabled", "true");

    fireEvent.submit(linkForm);
    expect(onLinkBisqueAccount).toHaveBeenCalledTimes(1);

    await act(async () => {
      rejectLink(new Error("BisQue authentication unavailable"));
    });

    await waitFor(() => expect(linkButton).toHaveAccessibleName("Link account"));
    expect(linkButton).toHaveFocus();
    expect(linkButton).not.toBeDisabled();
    expect(linkButton).not.toHaveAttribute("aria-disabled");
    expect(passwordInput).toHaveValue("secret");
    expect(passwordInput).not.toHaveAttribute("readonly");
    expect(screen.getByText("BisQue authentication unavailable")).toBeVisible();

    onLinkBisqueAccount.mockResolvedValueOnce({ imageCount: 3 });
    passwordInput.focus();
    fireEvent.submit(linkForm);

    await waitFor(() => expect(onLinkBisqueAccount).toHaveBeenCalledTimes(2));
  });
});

describe("AppSettingsDialog BisQue unlink", () => {
  it("reflects a parent-owned unlink continuation and blocks duplicate requests", () => {
    const onUnlinkBisqueAccount = vi.fn(async () => undefined);
    render(
      <AppSettingsDialog
        open
        onOpenChange={vi.fn()}
        initialTab="bisque"
        authUser="ada@example.com"
        authMode="workos"
        authIsAdmin={false}
        bisqueCredentialsLinked
        bisqueUnlinkPending
        themePreference="system"
        resolvedTheme="light"
        bisqueNavLinks={null}
        onThemePreferenceChange={vi.fn()}
        onOpenAdmin={vi.fn()}
        onLogout={vi.fn(async () => undefined)}
        onUnlinkBisqueAccount={onUnlinkBisqueAccount}
        onLinkBisqueAccount={vi.fn(async () => ({ imageCount: 0 }))}
        formatError={(error) =>
          error instanceof Error ? error.message : "Unable to unlink BisQue"
        }
      />
    );

    const unlinkButton = screen.getByRole("button", { name: "Unlinking..." });
    expect(unlinkButton).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(unlinkButton);
    expect(onUnlinkBisqueAccount).not.toHaveBeenCalled();
  });

  it("focuses the credential form after a successful unlink", async () => {
    const onUnlinkBisqueAccount = vi.fn(async () => undefined);
    const sharedProps = {
      open: true,
      onOpenChange: vi.fn(),
      initialTab: "bisque" as const,
      authUser: "ada@example.com",
      authMode: "workos" as const,
      authIsAdmin: false,
      themePreference: "system" as const,
      resolvedTheme: "light" as const,
      bisqueNavLinks: null,
      onThemePreferenceChange: vi.fn(),
      onOpenAdmin: vi.fn(),
      onLogout: vi.fn(async () => undefined),
      onUnlinkBisqueAccount,
      onLinkBisqueAccount: vi.fn(async () => ({ imageCount: 0 })),
      formatError: (error: unknown) =>
        error instanceof Error ? error.message : "Unable to unlink BisQue",
    };
    const { rerender } = render(
      <AppSettingsDialog {...sharedProps} bisqueCredentialsLinked />
    );

    const unlinkButton = screen.getByRole("button", { name: "Unlink account" });
    unlinkButton.focus();
    fireEvent.click(unlinkButton);
    await waitFor(() => expect(onUnlinkBisqueAccount).toHaveBeenCalledTimes(1));

    rerender(<AppSettingsDialog {...sharedProps} bisqueCredentialsLinked={false} />);

    await waitFor(() => expect(screen.getByLabelText("Username")).toHaveFocus());
  });

  it("keeps the unlink control focusable while a failed request can be retried", async () => {
    let rejectUnlink: (reason?: unknown) => void = () => undefined;
    const onUnlinkBisqueAccount = vi.fn(
      () =>
        new Promise<void>((_resolve, reject) => {
          rejectUnlink = reject;
        })
    );

    render(
      <AppSettingsDialog
        open
        onOpenChange={vi.fn()}
        initialTab="bisque"
        authUser="ada@example.com"
        authMode="workos"
        authIsAdmin={false}
        bisqueCredentialsLinked
        themePreference="system"
        resolvedTheme="light"
        bisqueNavLinks={null}
        onThemePreferenceChange={vi.fn()}
        onOpenAdmin={vi.fn()}
        onLogout={vi.fn(async () => undefined)}
        onUnlinkBisqueAccount={onUnlinkBisqueAccount}
        onLinkBisqueAccount={vi.fn(async () => ({ imageCount: 0 }))}
        formatError={(error) =>
          error instanceof Error ? error.message : "Unable to unlink BisQue"
        }
      />
    );

    const unlinkButton = screen.getByRole("button", { name: "Unlink account" });
    unlinkButton.focus();
    fireEvent.click(unlinkButton);

    await waitFor(() => expect(unlinkButton).toHaveAccessibleName("Unlinking..."));
    expect(unlinkButton).toHaveFocus();
    expect(unlinkButton).not.toBeDisabled();
    expect(unlinkButton).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(unlinkButton);
    expect(onUnlinkBisqueAccount).toHaveBeenCalledTimes(1);

    await act(async () => {
      rejectUnlink(new Error("BisQue service unavailable"));
    });

    await waitFor(() => expect(unlinkButton).toHaveAccessibleName("Unlink account"));
    expect(unlinkButton).toHaveFocus();
    expect(unlinkButton).not.toHaveAttribute("aria-disabled");
    expect(screen.getByText("BisQue account linked")).toBeInTheDocument();
    const errorMessage = screen.getByText("BisQue service unavailable");
    expect(errorMessage).toBeVisible();

    onUnlinkBisqueAccount.mockResolvedValueOnce(undefined);
    fireEvent.click(unlinkButton);

    await waitFor(() => expect(onUnlinkBisqueAccount).toHaveBeenCalledTimes(2));
  });
});

describe("AppSettingsDialog responsive section navigation", () => {
  it("keeps the selected section visible when the tab strip narrows", () => {
    let notifyResize: ResizeObserverCallback | null = null;
    const observeList = vi.fn();
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(callback: ResizeObserverCallback) {
          notifyResize = callback;
        }

        observe = observeList;
        unobserve = vi.fn();
        disconnect = vi.fn();
      }
    );

    render(
      <AppSettingsDialog
        open
        onOpenChange={vi.fn()}
        initialTab="about"
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
        formatError={(error) =>
          error instanceof Error ? error.message : "Unable to update settings"
        }
      />
    );

    const tabList = screen.getByRole("tablist", { name: "Settings sections" });
    const aboutTab = screen.getByRole("tab", { name: "About" });
    vi.spyOn(tabList, "getBoundingClientRect").mockReturnValue(horizontalRect(20, 370));
    const aboutBounds = vi
      .spyOn(aboutTab, "getBoundingClientRect")
      .mockReturnValue(horizontalRect(510, 600.2));
    Object.defineProperty(tabList, "scrollLeft", {
      configurable: true,
      writable: true,
      value: 0,
    });
    act(() => {
      aboutTab.focus();
    });

    expect(observeList).toHaveBeenCalledWith(tabList);
    expect(notifyResize).not.toBeNull();
    act(() => {
      notifyResize?.([], {} as ResizeObserver);
    });

    expect(aboutTab).toHaveFocus();
    expect(aboutTab).toHaveAttribute("aria-selected", "true");
    expect(tabList.scrollLeft).toBe(231);

    aboutBounds.mockReturnValue(horizontalRect(40, 130));
    tabList.scrollLeft = 17;
    act(() => {
      notifyResize?.([], {} as ResizeObserver);
    });
    expect(tabList.scrollLeft).toBe(17);
  });
});
