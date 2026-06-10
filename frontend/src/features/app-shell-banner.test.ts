import { describe, expect, it } from "vitest";

import { shouldShowAppShellBanner } from "../App";

const missingChatMessage =
  "Requested chat was not found. Opened the latest available conversation instead.";

describe("app shell banner visibility", () => {
  it("keeps stale chat recovery copy out of non-chat panels", () => {
    expect(shouldShowAppShellBanner("chat", missingChatMessage)).toBe(true);
    expect(shouldShowAppShellBanner("resources", missingChatMessage)).toBe(false);
    expect(shouldShowAppShellBanner("scientific-viewer", missingChatMessage)).toBe(false);
  });

  it("still allows real global errors outside chat", () => {
    expect(shouldShowAppShellBanner("resources", "Clipboard access is unavailable.")).toBe(true);
    expect(shouldShowAppShellBanner("chat", null)).toBe(false);
  });
});
