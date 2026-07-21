import { describe, expect, it } from "vitest";
import { bundleRootForRelativePath, groupPendingUploads } from "./pendingBundles";

describe("bundleRootForRelativePath", () => {
  it("detects .ome.zarr / .zarr directory roots", () => {
    expect(bundleRootForRelativePath("scan.ome.zarr/.zattrs")).toBe("scan.ome.zarr");
    expect(bundleRootForRelativePath("scan.zarr/0/0/0")).toBe("scan.zarr");
    expect(bundleRootForRelativePath("SCAN.OME.ZARR/0")).toBe("SCAN.OME.ZARR");
  });
  it("returns null for non-bundle paths", () => {
    expect(bundleRootForRelativePath("photo.png")).toBeNull();
    expect(bundleRootForRelativePath("")).toBeNull();
    expect(bundleRootForRelativePath(undefined)).toBeNull();
    expect(bundleRootForRelativePath("folder/file.tif")).toBeNull();
  });
});

describe("groupPendingUploads", () => {
  it("collapses all members of one zarr folder into a single bundle group", () => {
    const files = [
      { name: ".zattrs", size: 100, webkitRelativePath: "scan.ome.zarr/.zattrs" },
      { name: ".zgroup", size: 50, webkitRelativePath: "scan.ome.zarr/.zgroup" },
      { name: "0", size: 1000, webkitRelativePath: "scan.ome.zarr/0/0/0" },
    ];
    const groups = groupPendingUploads(files);
    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({ name: "scan.ome.zarr", isBundle: true, indices: [0, 1, 2], totalBytes: 1150 });
  });

  it("keeps non-bundle files as individual groups and preserves their indices", () => {
    const files = [
      { name: "a.png", size: 10, webkitRelativePath: "" },
      { name: ".zattrs", size: 5, webkitRelativePath: "v.zarr/.zattrs" },
      { name: "b.tif", size: 20, webkitRelativePath: "" },
      { name: "0", size: 30, webkitRelativePath: "v.zarr/0" },
    ];
    const groups = groupPendingUploads(files);
    expect(groups).toHaveLength(3); // a.png, the v.zarr bundle, b.tif
    expect(groups[0]).toMatchObject({ name: "a.png", isBundle: false, indices: [0] });
    expect(groups[1]).toMatchObject({ name: "v.zarr", isBundle: true, indices: [1, 3], totalBytes: 35 });
    expect(groups[2]).toMatchObject({ name: "b.tif", isBundle: false, indices: [2] });
  });

  it("groups two distinct zarr folders separately", () => {
    const files = [
      { name: "0", size: 1, webkitRelativePath: "a.zarr/0" },
      { name: "0", size: 1, webkitRelativePath: "b.zarr/0" },
      { name: "1", size: 1, webkitRelativePath: "a.zarr/1" },
    ];
    const groups = groupPendingUploads(files);
    expect(groups.map((g) => g.name)).toEqual(["a.zarr", "b.zarr"]);
    expect(groups[0].indices).toEqual([0, 2]);
    expect(groups[1].indices).toEqual([1]);
  });
});

describe("plain folder grouping (display-only)", () => {
  it("groups a generic folder's files into one folder chip", () => {
    const groups = groupPendingUploads([
      { name: "a.csv", size: 10, webkitRelativePath: "exp/a.csv" },
      { name: "b.csv", size: 20, webkitRelativePath: "exp/b.csv" },
      { name: "loose.txt", size: 5 },
    ]);
    expect(groups).toHaveLength(2);
    expect(groups[0]).toMatchObject({ name: "exp", isFolder: true, isBundle: false, fileCount: 2, totalBytes: 30 });
    expect(groups[1]).toMatchObject({ name: "loose.txt", isBundle: false });
    expect(groups[0].indices).toEqual([0, 1]);
  });

  it("keeps zarr groups as bundles, never folder chips", () => {
    const groups = groupPendingUploads([
      { name: ".zattrs", size: 1, webkitRelativePath: "scan.zarr/.zattrs" },
    ]);
    expect(groups[0].isBundle).toBe(true);
    expect(groups[0].isFolder).toBeFalsy();
  });
});
