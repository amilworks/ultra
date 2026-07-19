// Frontend mirror of backend/shared/special_formats.json — the canonical
// registry of directory-shaped special formats (today: OME-Zarr). Go and
// Python mirror the same file with parity tests; specialFormats.parity.test.ts
// does the same for this module, so adding a format to the JSON fails loudly
// here until the frontend catches up.
//
// "Directory bundle" = a folder that is ONE logical dataset: the upload
// pipeline must keep its relative structure and commit it as a single bundle
// resource, never as loose per-file resources.

export const DIRECTORY_BUNDLE_EXTENSIONS = [".ome.zarr", ".zarr"] as const;

/** True when a path segment names a directory-bundle root (e.g. `scan.zarr`). */
export const isDirectoryBundleName = (name: string): boolean => {
  const lower = name.trim().toLowerCase();
  return DIRECTORY_BUNDLE_EXTENSIONS.some((extension) => lower.endsWith(extension));
};

/**
 * Index of the first directory-bundle segment in a relative path, or -1.
 * `run7/scan.zarr/0/.zattrs` → 1; `scan.zarr/.zgroup` → 0; `a/b.txt` → -1.
 */
export const firstBundleSegmentIndex = (relativePath: string): number => {
  const segments = relativePath.split("/");
  // The last segment is a file name, never a bundle root.
  for (let index = 0; index < segments.length - 1; index += 1) {
    if (isDirectoryBundleName(segments[index])) {
      return index;
    }
  }
  return -1;
};
