import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { DIRECTORY_BUNDLE_EXTENSIONS, firstBundleSegmentIndex, isDirectoryBundleName } from "./specialFormats";

// Same contract as the Go and Python parity tests: the frontend mirror must
// match backend/shared/special_formats.json exactly, so a format added to the
// canonical spec fails here until the frontend supports it.
describe("special-format parity with backend/shared/special_formats.json", () => {
  it("mirrors every directory-shaped format's extensions", () => {
    const specPath = resolve(__dirname, "../../../backend/shared/special_formats.json");
    const spec = JSON.parse(readFileSync(specPath, "utf8")) as {
      formats: Array<{ id: string; shape: string; extensions: string[] }>;
    };
    const directoryExtensions = spec.formats
      .filter((format) => format.shape === "directory")
      .flatMap((format) => format.extensions)
      .sort();
    expect([...DIRECTORY_BUNDLE_EXTENSIONS].sort()).toEqual(directoryExtensions);
  });

  it("detects bundle roots at any depth, never file segments", () => {
    expect(isDirectoryBundleName("scan.ome.zarr")).toBe(true);
    expect(isDirectoryBundleName("SCAN.ZARR")).toBe(true);
    expect(isDirectoryBundleName("scan.tif")).toBe(false);
    expect(firstBundleSegmentIndex("scan.zarr/.zgroup")).toBe(0);
    expect(firstBundleSegmentIndex("run7/scan.ome.zarr/0/0.0")).toBe(1);
    expect(firstBundleSegmentIndex("run7/notes.zarr")).toBe(-1); // file, not dir
    expect(firstBundleSegmentIndex("plain/folder/data.csv")).toBe(-1);
  });
});
