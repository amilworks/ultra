import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("post-authentication landing", () => {
  it("opens a clean New Chat when the URL does not name a conversation", () => {
    const start = appSource.indexOf("const targetConversationId = readConversationIdFromLocation();");
    const end = appSource.indexOf("setConversationsHydrated(true);", start);
    const bootstrap = appSource.slice(start, end);

    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    expect(bootstrap).toContain("const shouldOpenNewChat = !targetConversationId;");
    expect(bootstrap).toContain(
      "const landingConversation = shouldOpenNewChat ? createConversationState() : null;"
    );
    expect(bootstrap).toMatch(/if\s*\(\s*targetConversationId\s*&&/);
    expect(bootstrap).toContain("if (landingConversation)");
  });

  it("keeps deep links authoritative and never publishes the blank landing id", () => {
    expect(appSource).toContain("shouldExposeConversationInUrl(urlConversation)");
    expect(appSource).toContain("readConversationIdFromLocation()");
    expect(appSource).toContain("prependResolvedConversation(");
  });

  it("starts a new investigation in High mode so Pro remains an explicit choice", () => {
    const start = appSource.indexOf("const createConversationState = ()");
    const end = appSource.indexOf("const toMillis =", start);
    const factory = appSource.slice(start, end);

    expect(factory).toContain("composerWorkflowPreset: null");
    expect(factory).not.toContain("PRO_MODE_COMPOSER_WORKFLOW_PRESET");
  });
});
