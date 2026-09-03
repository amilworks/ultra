import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const settingsSource = readFileSync(
  path.join(process.cwd(), "src/components/AppSettingsDialog.tsx"),
  "utf8"
);
const accountMenuSource = readFileSync(
  path.join(process.cwd(), "src/components/chat/SidebarAccountSettingsButton.tsx"),
  "utf8"
);

describe("task-first blank chat", () => {
  it("keeps usage analytics out of the empty chat branch", () => {
    const emptyBranchStart = appSource.indexOf("messages.length === 0 ? (");
    expect(emptyBranchStart).toBeGreaterThan(-1);
    const messagesBranchStart = appSource.indexOf("messages.map((message", emptyBranchStart);
    expect(messagesBranchStart).toBeGreaterThan(emptyBranchStart);

    const emptyBranch = appSource.slice(emptyBranchStart, messagesBranchStart);
    expect(emptyBranch).not.toContain("<UserTokenUsagePanel");
    expect(emptyBranch).not.toContain("mobile-usage-disclosure");
  });

  it("does not prefetch account analytics just because a New Chat is open", () => {
    expect(appSource).not.toContain("const shouldShowBlankChatUsage");
    expect(appSource).not.toContain("useBlankChatTokenUsage");
    expect(appSource).not.toContain("blankChatTokenUsage={blankChatTokenUsage}");
  });

  it("keeps usage available in account settings with a direct account-menu route", () => {
    expect(settingsSource).toContain('<TabsTrigger value="usage"');
    expect(settingsSource).toContain("<UserTokenUsagePanel");
    expect(accountMenuSource).toContain("onOpenUsage");
    expect(accountMenuSource).toContain("<span>Usage</span>");
  });
});
