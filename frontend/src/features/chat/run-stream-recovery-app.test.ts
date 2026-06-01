import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("active run stream recovery wiring", () => {
  it("does not let effect cleanup suppress a detached resumed stream terminal result", () => {
    const resumeStart = appSource.indexOf(".resumeRunStream(target.runId");
    expect(resumeStart).toBeGreaterThan(-1);

    const thenStart = appSource.indexOf(".then((response)", resumeStart);
    const catchStart = appSource.indexOf(".catch((error)", resumeStart);
    expect(thenStart).toBeGreaterThan(resumeStart);
    expect(catchStart).toBeGreaterThan(thenStart);

    const terminalHandler = appSource.slice(thenStart, catchStart);
    expect(terminalHandler).not.toContain("cancelled");
  });
});
