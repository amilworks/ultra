import { isDirectoryBundleName } from "./specialFormats";

// Group composer pending uploads so a folder-picked directory format (OME-Zarr) shows as
// ONE chip instead of dozens of member-file chips. A directory bundle = files whose
// webkitRelativePath top segment is a directory special format (mirrors the backend
// special-format registry: shape "directory", extensions .ome.zarr / .zarr). The backend
// commits those members as a single bundle resource, so the composer should preview them
// as a single unit too.

export type PendingUploadLike = {
  name: string;
  size: number;
  webkitRelativePath?: string;
};

export type PendingUploadGroup = {
  /** Display name: the bundle/folder root dir for a group, else the file name. */
  name: string;
  isBundle: boolean;
  /** True for a plain (non-bundle) folder grouped for display only — its files
      still upload individually; the chip just stops a 500-file folder from
      becoming 500 chips. */
  isFolder?: boolean;
  /** Number of member files in a folder/bundle group. */
  fileCount?: number;
  /** Indices into the original files array this group represents (for removal). */
  indices: number[];
  totalBytes: number;
};

// The bundle root (top path segment) for a file, or null if it isn't a directory bundle.
// `.ome.zarr` ends with `.zarr`, so a single suffix check covers both registry extensions.
export const bundleRootForRelativePath = (relativePath: string | undefined): string | null => {
  const top = (relativePath ?? "").split("/")[0] ?? "";
  return top.length > 0 && isDirectoryBundleName(top) ? top : null;
};

export const groupPendingUploads = (files: readonly PendingUploadLike[]): PendingUploadGroup[] => {
  const groups: PendingUploadGroup[] = [];
  const byRoot = new Map<string, PendingUploadGroup>();
  files.forEach((file, index) => {
    const root = bundleRootForRelativePath(file.webkitRelativePath);
    if (root) {
      const existing = byRoot.get(root);
      if (existing) {
        existing.indices.push(index);
        existing.totalBytes += file.size;
        existing.fileCount = (existing.fileCount ?? 1) + 1;
        return;
      }
      const group: PendingUploadGroup = { name: root, isBundle: true, fileCount: 1, indices: [index], totalBytes: file.size };
      byRoot.set(root, group);
      groups.push(group);
      return;
    }
    const folderRoot = (file.webkitRelativePath ?? "").includes("/")
      ? ((file.webkitRelativePath ?? "").split("/")[0] ?? "")
      : "";
    if (folderRoot) {
      const key = `folder:${folderRoot}`;
      const existing = byRoot.get(key);
      if (existing) {
        existing.indices.push(index);
        existing.totalBytes += file.size;
        existing.fileCount = (existing.fileCount ?? 1) + 1;
        return;
      }
      const group: PendingUploadGroup = {
        name: folderRoot,
        isBundle: false,
        isFolder: true,
        fileCount: 1,
        indices: [index],
        totalBytes: file.size,
      };
      byRoot.set(key, group);
      groups.push(group);
      return;
    }
    groups.push({ name: file.name, isBundle: false, indices: [index], totalBytes: file.size });
  });
  return groups;
};
