import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("chat request autonomy defaults", () => {
  it("does not serialize frontend runtime or tool-call caps by default", () => {
    const chatRequestStart = appSource.indexOf("const chatRequest = {");
    expect(chatRequestStart).toBeGreaterThanOrEqual(0);

    const chatRequestEnd = appSource.indexOf("};", chatRequestStart);
    expect(chatRequestEnd).toBeGreaterThan(chatRequestStart);

    const chatRequestSource = appSource.slice(chatRequestStart, chatRequestEnd);

    expect(chatRequestSource).not.toContain("budgets:");
    expect(chatRequestSource).not.toContain("max_tool_calls");
    expect(chatRequestSource).not.toContain("max_runtime_seconds");
  });
});
