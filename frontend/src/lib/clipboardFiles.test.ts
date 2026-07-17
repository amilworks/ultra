import { describe, expect, it } from "vitest";

import { filesFromClipboard, withPastedFileName } from "./clipboardFiles";

const NOW = new Date(2026, 6, 15, 8, 45, 12, 345);

describe("withPastedFileName", () => {
  it("renames a generic clipboard screenshot with a timestamp", () => {
    const pasted = new File(["png-bytes"], "image.png", { type: "image/png" });
    const renamed = withPastedFileName(pasted, NOW);
    expect(renamed.name).toBe("pasted-2026-07-15-084512345.png");
    expect(renamed.type).toBe("image/png");
    expect(renamed.size).toBe(pasted.size);
  });

  it("maps jpeg to a .jpg extension", () => {
    const pasted = new File(["jpg"], "image.jpeg", { type: "image/jpeg" });
    expect(withPastedFileName(pasted, NOW).name).toBe("pasted-2026-07-15-084512345.jpg");
  });

  it("keeps the real name of a file copied from the OS", () => {
    const copied = new File(["tif"], "scan_0042.ome.tif", { type: "image/tiff" });
    expect(withPastedFileName(copied, NOW)).toBe(copied);
  });

  it("names a nameless clipboard file from its MIME type", () => {
    const nameless = new File(["png"], "", { type: "image/png" });
    expect(withPastedFileName(nameless, NOW).name).toBe(
      "pasted-2026-07-15-084512345.png"
    );
  });
});

describe("filesFromClipboard", () => {
  it("returns an empty array for a text-only paste", () => {
    expect(filesFromClipboard({ files: [] as unknown as FileList }, NOW)).toEqual([]);
    expect(filesFromClipboard(null, NOW)).toEqual([]);
  });

  it("returns renamed files for a file-bearing paste", () => {
    const screenshot = new File(["x"], "image.png", { type: "image/png" });
    const named = new File(["y"], "report.pdf", { type: "application/pdf" });
    const result = filesFromClipboard(
      { files: [screenshot, named] as unknown as FileList },
      NOW
    );
    expect(result.map((file) => file.name)).toEqual([
      "pasted-2026-07-15-084512345.png",
      "report.pdf",
    ]);
  });
});
