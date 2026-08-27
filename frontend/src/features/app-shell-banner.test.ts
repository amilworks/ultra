import { createElement } from "react";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SystemMessage } from "@/components/prompt-kit/system-message";
import {
  appShellBannerVariant,
  MISSING_REQUESTED_CONVERSATION_MESSAGE,
  MISSING_REQUESTED_CONVERSATION_NEW_MESSAGE,
  selectLatestAvailableConversation,
  shouldShowAppShellBanner,
} from "./app-shell-banner";

const missingChatMessage = MISSING_REQUESTED_CONVERSATION_MESSAGE;

describe("app shell banner visibility", () => {
  it("keeps stale chat recovery copy out of non-chat panels", () => {
    expect(shouldShowAppShellBanner("chat", missingChatMessage)).toBe(true);
    expect(shouldShowAppShellBanner("resources", missingChatMessage)).toBe(false);
    expect(shouldShowAppShellBanner("scientific-viewer", missingChatMessage)).toBe(false);
    expect(
      shouldShowAppShellBanner("resources", MISSING_REQUESTED_CONVERSATION_NEW_MESSAGE)
    ).toBe(false);
  });

  it("still allows real global errors outside chat", () => {
    expect(shouldShowAppShellBanner("resources", "Clipboard access is unavailable.")).toBe(true);
    expect(shouldShowAppShellBanner("chat", null)).toBe(false);
  });

  it("renders successful stale-chat recovery neutrally and keeps actual failures red", () => {
    const recovered = render(
      createElement(
        SystemMessage,
        { variant: appShellBannerVariant(missingChatMessage), fill: true },
        missingChatMessage
      )
    );
    expect(recovered.container.firstElementChild).toHaveClass("bg-zinc-100");
    expect(recovered.container.firstElementChild).not.toHaveClass("bg-red-100");
    recovered.unmount();

    const failed = render(
      createElement(
        SystemMessage,
        { variant: appShellBannerVariant("Failed to load chat"), fill: true },
        "Failed to load chat"
      )
    );
    expect(failed.container.firstElementChild).toHaveClass("bg-red-100");
  });

  it("selects the latest durable fallback independent of array order", () => {
    const conversations = [
      { id: "older", hydrated: true, updatedAt: 10, available: true },
      { id: "missing", hydrated: true, updatedAt: 40, available: true },
      { id: "blank", hydrated: true, updatedAt: 50, available: false },
      { id: "newer", hydrated: true, updatedAt: 30, available: true },
      { id: "loading", hydrated: false, updatedAt: 60, available: true },
    ];

    expect(
      selectLatestAvailableConversation(
        conversations,
        "missing",
        (conversation) => conversation.available
      )?.id
    ).toBe("newer");
  });
});
