import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("chat submit terminal state", () => {
  it("clears the active streaming message after a successful chat response", () => {
    const successStart = appSource.indexOf(
      "const assistantText =\n        response.response_text?.trim() || streamedText.trim()"
    );
    expect(successStart).toBeGreaterThan(-1);

    const catchStart = appSource.indexOf("} catch (error)", successStart);
    expect(catchStart).toBeGreaterThan(successStart);

    const successHandler = appSource.slice(successStart, catchStart);
    expect(successHandler).toContain("sending: false");
    expect(successHandler).toContain("streamingMessageId:");
    expect(successHandler).toContain("current.streamingMessageId === messageId");
    expect(successHandler).toContain("? null");
  });
});
