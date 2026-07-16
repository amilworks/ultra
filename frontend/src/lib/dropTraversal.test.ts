import { describe, expect, it } from "vitest";

import {
  collectDroppedFiles,
  isOsFileDrag,
  MAX_DROPPED_FILES,
  type DirectoryEntryLike,
  type DropPayload,
  type EntryLike,
  type FileEntryLike,
} from "./dropTraversal";

const fileEntry = (
  fullPath: string,
  options: { failWith?: string } = {}
): FileEntryLike => {
  const name = fullPath.split("/").filter(Boolean).pop() ?? "";
  return {
    isFile: true,
    isDirectory: false,
    name,
    fullPath,
    file: (onSuccess, onError) => {
      if (options.failWith) {
        onError(new Error(options.failWith));
        return;
      }
      onSuccess(new File(["x"], name, { type: "application/octet-stream" }));
    },
  };
};

// batchSize models Chromium's 100-entry readEntries pagination.
const directoryEntry = (
  fullPath: string,
  children: EntryLike[],
  options: { batchSize?: number; failWith?: string } = {}
): DirectoryEntryLike => {
  const name = fullPath.split("/").filter(Boolean).pop() ?? "";
  return {
    isFile: false,
    isDirectory: true,
    name,
    fullPath,
    createReader: () => {
      let cursor = 0;
      return {
        readEntries: (onSuccess, onError) => {
          if (options.failWith) {
            onError(new Error(options.failWith));
            return;
          }
          const batch = children.slice(cursor, cursor + (options.batchSize ?? 100));
          cursor += batch.length;
          onSuccess(batch);
        },
      };
    },
  };
};

const payload = (entries: EntryLike[], fallbackFiles: File[] = []): DropPayload => ({
  entries,
  fallbackFiles,
});

describe("collectDroppedFiles", () => {
  it("synthesizes webkitRelativePath for files under a dropped directory", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/experiment_A", [
          fileEntry("/experiment_A/image1.tif"),
          directoryEntry("/experiment_A/masks", [fileEntry("/experiment_A/masks/m1.png")]),
        ]),
      ])
    );
    expect(result.files.map((file) => file.webkitRelativePath)).toEqual([
      "experiment_A/image1.tif",
      "experiment_A/masks/m1.png",
    ]);
    expect(result.topLevelDirectories).toEqual(["experiment_A"]);
    expect(result.issues).toEqual([]);
  });

  it("does not synthesize webkitRelativePath for loose dropped files (picker parity)", async () => {
    const result = await collectDroppedFiles(payload([fileEntry("/photo.jpg")]));
    expect(result.files).toHaveLength(1);
    // jsdom Files lack the property entirely; real browsers report "". Either
    // way the walker must not have defined an own property on a loose file.
    expect(
      Object.getOwnPropertyDescriptor(result.files[0], "webkitRelativePath")
    ).toBeUndefined();
    expect(result.topLevelDirectories).toEqual([]);
  });

  it("drains readEntries past Chromium's 100-entry batches", async () => {
    const children = Array.from({ length: 250 }, (_, index) =>
      fileEntry(`/big/chunk_${index}.bin`)
    );
    const result = await collectDroppedFiles(
      payload([directoryEntry("/big", children, { batchSize: 100 })])
    );
    expect(result.files).toHaveLength(250);
  });

  it("keeps zarr dotfiles but skips OS junk, counting it", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/store.zarr", [
          fileEntry("/store.zarr/.zattrs"),
          fileEntry("/store.zarr/.zgroup"),
          fileEntry("/store.zarr/.DS_Store"),
          fileEntry("/store.zarr/._shadow"),
          fileEntry("/store.zarr/Thumbs.db"),
          fileEntry("/store.zarr/0.0"),
        ]),
      ])
    );
    expect(result.files.map((file) => file.name)).toEqual([".zattrs", ".zgroup", "0.0"]);
    expect(result.skippedJunkCount).toBe(3);
  });

  it("continues past unreadable entries and reports them as issues", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/mixed", [
          fileEntry("/mixed/ok1.dat"),
          fileEntry("/mixed/locked.dat", { failWith: "NotReadableError" }),
          fileEntry("/mixed/ok2.dat"),
        ]),
      ])
    );
    expect(result.files.map((file) => file.name)).toEqual(["ok1.dat", "ok2.dat"]);
    expect(result.issues).toEqual([
      { kind: "entry-error", path: "mixed/locked.dat", message: "NotReadableError" },
    ]);
  });

  it("reports an unreadable directory without sinking sibling entries", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/broken", [], { failWith: "SecurityError" }),
        fileEntry("/fine.txt"),
      ])
    );
    expect(result.files.map((file) => file.name)).toEqual(["fine.txt"]);
    expect(result.issues).toContainEqual({
      kind: "entry-error",
      path: "broken",
      message: "SecurityError",
    });
  });

  it("stops at MAX_DROPPED_FILES with a too-many-files issue", async () => {
    const children = Array.from({ length: MAX_DROPPED_FILES + 5 }, (_, index) =>
      fileEntry(`/huge/f${index}.bin`)
    );
    const result = await collectDroppedFiles(payload([directoryEntry("/huge", children)]));
    expect(result.files).toHaveLength(MAX_DROPPED_FILES);
    expect(result.issues).toContainEqual({
      kind: "too-many-files",
      limit: MAX_DROPPED_FILES,
    });
  });

  it("flags a directory that yields no uploadable files", async () => {
    const result = await collectDroppedFiles(
      payload([directoryEntry("/empty", [fileEntry("/empty/.DS_Store")])])
    );
    expect(result.files).toEqual([]);
    expect(result.issues).toContainEqual({ kind: "empty-directory", name: "empty" });
  });

  it("falls back to the flat file list when no entries exist, minus junk", async () => {
    const keep = new File(["x"], "data.csv");
    const junk = new File(["x"], ".DS_Store");
    const result = await collectDroppedFiles(payload([], [keep, junk]));
    expect(result.files).toEqual([keep]);
    expect(result.skippedJunkCount).toBe(1);
  });

  it("re-roots a zarr store nested inside a dropped folder at the store", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/run7", [
          fileEntry("/run7/notes.txt"),
          directoryEntry("/run7/scan.ome.zarr", [
            fileEntry("/run7/scan.ome.zarr/.zattrs"),
            directoryEntry("/run7/scan.ome.zarr/0", [
              fileEntry("/run7/scan.ome.zarr/0/0.0"),
            ]),
          ]),
        ]),
      ])
    );
    expect(result.files.map((file) => file.webkitRelativePath)).toEqual([
      "run7/notes.txt",
      "scan.ome.zarr/.zattrs",
      "scan.ome.zarr/0/0.0",
    ]);
  });

  it("keeps a directly-dropped zarr rooted at itself and never re-roots within it", async () => {
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/outer.zarr", [
          fileEntry("/outer.zarr/.zgroup"),
          directoryEntry("/outer.zarr/inner.zarr", [
            fileEntry("/outer.zarr/inner.zarr/.zattrs"),
          ]),
        ]),
      ])
    );
    expect(result.files.map((file) => file.webkitRelativePath)).toEqual([
      "outer.zarr/.zgroup",
      "outer.zarr/inner.zarr/.zattrs",
    ]);
  });

  it("withholds a zarr store that the file cap truncated mid-walk", async () => {
    const storeChildren = Array.from({ length: 50 }, (_, index) =>
      fileEntry(`/big.zarr/chunk_${index}`)
    );
    const looseChildren = Array.from({ length: MAX_DROPPED_FILES - 20 }, (_, index) =>
      fileEntry(`/loose/f${index}.bin`)
    );
    const result = await collectDroppedFiles(
      payload([
        directoryEntry("/loose", looseChildren),
        directoryEntry("/big.zarr", storeChildren),
      ])
    );
    // The store hit the cap mid-walk: none of its files may upload.
    expect(
      result.files.some((file) => (file.webkitRelativePath ?? "").startsWith("big.zarr/"))
    ).toBe(false);
    expect(result.issues).toContainEqual({ kind: "truncated-bundle", name: "big.zarr" });
    expect(result.issues).toContainEqual({
      kind: "too-many-files",
      limit: MAX_DROPPED_FILES,
    });
  });
});

describe("isOsFileDrag", () => {
  it("accepts pure OS file drags and rejects in-page image/link drags", () => {
    expect(isOsFileDrag({ types: ["Files"] })).toBe(true);
    expect(isOsFileDrag({ types: ["Files", "text/html", "text/uri-list"] })).toBe(false);
    expect(isOsFileDrag({ types: ["text/plain"] })).toBe(false);
    expect(isOsFileDrag(null)).toBe(false);
  });
});
