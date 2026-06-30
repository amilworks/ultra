import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const packageSource = readFileSync(path.join(process.cwd(), "package.json"), "utf8");

describe("chat transcript virtualization wiring", () => {
  it("installs react-virtuoso as the full transcript virtualization dependency", () => {
    expect(JSON.parse(packageSource).dependencies).toHaveProperty("react-virtuoso");
  });

  it("uses Virtuoso for long hydrated transcripts while retaining the short-list path", () => {
    expect(appSource).toContain('import("react-virtuoso")');
    expect(appSource).toContain('const LazyVirtuoso = lazyNamed(loadVirtuosoModule, "Virtuoso");');
    expect(appSource).toContain("const shouldVirtualizeMessages = messages.length > MESSAGE_WINDOW_SIZE");
    expect(appSource).toContain("<LazyVirtuoso");
    expect(appSource).toContain("data={messages}");
    expect(appSource).toContain("computeItemKey={(_: number, message: UiMessage) => message.id}");
    expect(appSource).toContain('followOutput="auto"');
    expect(appSource).toContain("initialTopMostItemIndex={messages.length - 1}");

    // Virtuoso mounts only once the StickToBottom scroll element is captured (else customScrollParent
    // would be undefined and Virtuoso would create a broken nested scroller for the first paint).
    const virtualizedBranchStart = appSource.indexOf(
      "shouldVirtualizeMessages && virtualizedScrollParent ? ("
    );
    expect(virtualizedBranchStart).toBeGreaterThan(-1);
    const shortBranchStart = appSource.indexOf(": (", virtualizedBranchStart);
    expect(shortBranchStart).toBeGreaterThan(virtualizedBranchStart);
    const virtualizedBranch = appSource.slice(virtualizedBranchStart, shortBranchStart);
    expect(virtualizedBranch).not.toContain("Show earlier messages");

    const shortBranch = appSource.slice(shortBranchStart, appSource.indexOf("</ChatContainerContent>", shortBranchStart));
    expect(shortBranch).toContain("visibleMessages.map((message, index)");
  });
});
