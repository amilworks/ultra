import { describe, expect, it } from "vitest";

import { canCopyImageToClipboard, exportFileStem } from "./captureView";

describe("captureView helpers", () => {
  it("derives a safe export filename stem from a resource name", () => {
    expect(exportFileStem("EnrNE_Day2_Run1_0210.JPG")).toBe("EnrNE_Day2_Run1_0210");
    expect(exportFileStem("cell stack.ome.tiff")).toBe("cell-stack.ome");
    expect(exportFileStem("brain.nii.gz")).toBe("brain.nii");
    expect(exportFileStem("weird name (1)!.png")).toBe("weird-name-1");
    expect(exportFileStem("")).toBe("image");
    expect(exportFileStem(undefined)).toBe("image");
    expect(exportFileStem(null)).toBe("image");
  });

  it("reports clipboard-image support without throwing", () => {
    // jsdom has no ClipboardItem, so support is false here; the call must be safe.
    expect(typeof canCopyImageToClipboard()).toBe("boolean");
  });
});
