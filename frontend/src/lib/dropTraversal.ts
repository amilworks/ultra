// Drag-and-drop directory traversal for upload drops.
//
// Browsers expose dropped directories only through the (webkit-prefixed but
// universally shipped) FileSystem Entry API on DataTransferItem. Reading
// `event.dataTransfer.files` flat — what this app did before — hands Chrome's
// zero-byte directory pseudo-File to the upload pipeline, which is why dropped
// folders "uploaded" as one broken file.
//
// Two hard rules of that API shape everything here:
// 1. `webkitGetAsEntry()` is only valid DURING the drop event dispatch — the
//    entries must be snapshotted synchronously (`snapshotDropPayload`) before
//    any `await`; the async walk (`collectDroppedFiles`) then runs on the
//    snapshot.
// 2. `DirectoryReader.readEntries` returns AT MOST 100 entries per call in
//    Chromium and must be re-called until it yields an empty batch; a single
//    call silently truncates big scientific folders.
//
// Files found under a dropped directory get `webkitRelativePath` synthesized
// (Object.defineProperty — the property is a read-only getter) so the entire
// existing pipeline — upload-session `relative_path`, OME-Zarr bundle
// detection, composer bundle chips — treats a dropped folder exactly like a
// folder chosen via the `webkitdirectory` picker. Files dropped loose keep
// their native empty `webkitRelativePath`, matching picker parity.

export type DropIssue =
  | { kind: "entry-error"; path: string; message: string }
  | { kind: "too-many-files"; limit: number }
  | { kind: "empty-directory"; name: string }
  | { kind: "truncated-bundle"; name: string };

/**
 * True only for drags that originate outside the page (OS file managers).
 * Browsers also put "Files" in types for in-page image/link drags, but those
 * carry text/html or text/uri-list alongside — a pure OS file drag does not.
 * Without this, dragging a chat figure summons the drop overlay.
 */
export const isOsFileDrag = (
  dataTransfer: { types?: readonly string[] } | null | undefined
): boolean => {
  const types = Array.from(dataTransfer?.types ?? []);
  return (
    types.includes("Files") &&
    !types.includes("text/html") &&
    !types.includes("text/uri-list")
  );
};

export type DropCollection = {
  files: File[];
  /** Top-level directory names, in drop order. */
  topLevelDirectories: string[];
  /** OS metadata junk skipped during the walk (.DS_Store and friends). */
  skippedJunkCount: number;
  issues: DropIssue[];
};

/** Mirrors the server's per-upload-session file cap (uploadSessionMaxFilesPerBatch). */
export const MAX_DROPPED_FILES = 10_000;

// OS metadata files that users never mean to upload. Deliberately NOT a
// blanket dotfile filter: OME-Zarr stores depend on dotfiles (.zattrs,
// .zgroup, .zarray) and stripping them would corrupt every dropped store.
const JUNK_FILE_NAMES = new Set([".ds_store", "thumbs.db", "desktop.ini"]);

const isJunkFileName = (name: string): boolean => {
  const lower = name.toLowerCase();
  // "._*" are AppleDouble resource forks shadowing real files.
  return JUNK_FILE_NAMES.has(lower) || lower.startsWith("._");
};

// Minimal structural types for the Entry API — lib.dom's typings for it are
// incomplete, and structural types keep the walker unit-testable with fakes.
export type FileEntryLike = {
  isFile: true;
  isDirectory: false;
  name: string;
  fullPath: string;
  file: (onSuccess: (file: File) => void, onError: (error: unknown) => void) => void;
};

export type DirectoryEntryLike = {
  isFile: false;
  isDirectory: true;
  name: string;
  fullPath: string;
  createReader: () => {
    readEntries: (
      onSuccess: (entries: Array<FileEntryLike | DirectoryEntryLike>) => void,
      onError: (error: unknown) => void
    ) => void;
  };
};

export type EntryLike = FileEntryLike | DirectoryEntryLike;

export type DropPayload = {
  entries: EntryLike[];
  /** Flat fallback when the browser exposes no entries (no directory support). */
  fallbackFiles: File[];
};

/**
 * Synchronous snapshot of a drop's DataTransfer. MUST be called inside the
 * drop event handler, before any await — the items list is neutered as soon
 * as the event dispatch ends.
 */
export function snapshotDropPayload(dataTransfer: DataTransfer): DropPayload {
  const entries: EntryLike[] = [];
  const items = dataTransfer.items;
  if (items) {
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      if (item.kind !== "file") {
        continue;
      }
      const getEntry = (
        item as DataTransferItem & { webkitGetAsEntry?: () => unknown }
      ).webkitGetAsEntry;
      const entry = typeof getEntry === "function" ? getEntry.call(item) : null;
      if (entry) {
        entries.push(entry as EntryLike);
      }
    }
  }
  return { entries, fallbackFiles: Array.from(dataTransfer.files ?? []) };
}

const entryFile = (entry: FileEntryLike): Promise<File> =>
  new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });

const readAllDirectoryEntries = async (
  directory: DirectoryEntryLike
): Promise<EntryLike[]> => {
  const reader = directory.createReader();
  const all: EntryLike[] = [];
  // Chromium batches at 100 entries per readEntries call; drain until empty.
  for (;;) {
    const batch = await new Promise<EntryLike[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) {
      return all;
    }
    // Loop, not push(...batch): Firefox returns entire directories in ONE
    // batch, and spreading ~125k+ arguments overflows V8's argument limit.
    for (const entry of batch) {
      all.push(entry);
    }
  }
};

const relativePathForEntry = (entry: EntryLike, pathBase: number): string =>
  entry.fullPath.slice(pathBase).replace(/^\/+/, "");

const isZarrDirectoryName = (name: string): boolean =>
  name.toLowerCase().endsWith(".zarr");

const defineRelativePath = (file: File, relativePath: string): File => {
  try {
    Object.defineProperty(file, "webkitRelativePath", {
      value: relativePath,
      configurable: true,
    });
  } catch {
    // Non-configurable in some exotic environment: the file still uploads,
    // just without folder placement — strictly better than dropping it.
  }
  return file;
};

const describeError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  const name = (error as { name?: string } | null)?.name;
  return name ? String(name) : "could not be read";
};

/**
 * Walk a snapshotted drop payload into upload-ready File objects.
 *
 * - Files inside dropped directories get `webkitRelativePath` synthesized
 *   from the entry's fullPath (picker parity); loose files keep it empty.
 * - OS junk is skipped and counted; unreadable entries become issues and the
 *   walk continues — one locked file must not sink a 5,000-file drop.
 * - The walk stops at MAX_DROPPED_FILES with a `too-many-files` issue so the
 *   user hears about the server's session cap before hashing gigabytes.
 */
export async function collectDroppedFiles(payload: DropPayload): Promise<DropCollection> {
  const collection: DropCollection = {
    files: [],
    topLevelDirectories: [],
    skippedJunkCount: 0,
    issues: [],
  };

  if (payload.entries.length === 0) {
    // No Entry API: keep the flat file list (pre-existing behavior), minus junk.
    for (const file of payload.fallbackFiles) {
      if (isJunkFileName(file.name)) {
        collection.skippedJunkCount += 1;
        continue;
      }
      collection.files.push(file);
    }
    return collection;
  }

  let capped = false;
  // Bundle roots whose walk was cut off by the cap: a partially-walked zarr
  // must not upload, or the server commits a silently incomplete bundle.
  const truncatedBundleRoots = new Set<string>();

  type WalkContext = {
    underDirectory: boolean;
    /** Chars to strip from entry.fullPath when synthesizing relative paths. */
    pathBase: number;
    /** Top path segment of the enclosing zarr store, when inside one. */
    zarrRoot: string | null;
  };

  const walk = async (entry: EntryLike, context: WalkContext): Promise<void> => {
    if (capped) {
      return;
    }
    if (entry.isFile) {
      if (isJunkFileName(entry.name)) {
        collection.skippedJunkCount += 1;
        return;
      }
      if (collection.files.length >= MAX_DROPPED_FILES) {
        capped = true;
        collection.issues.push({ kind: "too-many-files", limit: MAX_DROPPED_FILES });
        if (context.zarrRoot) {
          truncatedBundleRoots.add(context.zarrRoot);
        }
        return;
      }
      try {
        const file = await entryFile(entry);
        collection.files.push(
          context.underDirectory
            ? defineRelativePath(file, relativePathForEntry(entry, context.pathBase))
            : file
        );
      } catch (error) {
        collection.issues.push({
          kind: "entry-error",
          path: relativePathForEntry(entry, context.pathBase),
          message: describeError(error),
        });
      }
      return;
    }
    // Re-root at the first .zarr segment: the server detects bundles by the
    // TOP relative-path segment only, so a store nested inside a dropped
    // folder (run7/scan.zarr/…) would otherwise explode into thousands of
    // loose resources. Rooting its subtree at the store (scan.zarr/…) makes a
    // nested store upload identically to a directly-dropped one. First .zarr
    // wins — zarr-within-zarr is one store.
    let childContext: WalkContext = { ...context, underDirectory: true };
    if (!context.zarrRoot && isZarrDirectoryName(entry.name)) {
      childContext = {
        underDirectory: true,
        pathBase: entry.fullPath.length - entry.name.length,
        zarrRoot: entry.name,
      };
    }
    try {
      const children = await readAllDirectoryEntries(entry);
      for (const child of children) {
        await walk(child, childContext);
        if (capped) {
          return;
        }
      }
    } catch (error) {
      collection.issues.push({
        kind: "entry-error",
        path: relativePathForEntry(entry, context.pathBase),
        message: describeError(error),
      });
    }
  };

  const rootContext: WalkContext = { underDirectory: false, pathBase: 0, zarrRoot: null };
  for (const entry of payload.entries) {
    if (entry.isDirectory) {
      collection.topLevelDirectories.push(entry.name);
      const before = collection.files.length;
      await walk(entry, rootContext);
      if (!capped && collection.files.length === before) {
        collection.issues.push({ kind: "empty-directory", name: entry.name });
      }
    } else {
      await walk(entry, rootContext);
    }
    if (capped) {
      break;
    }
  }

  if (truncatedBundleRoots.size > 0) {
    collection.files = collection.files.filter((file) => {
      const topSegment = (file.webkitRelativePath ?? "").split("/")[0] ?? "";
      return !truncatedBundleRoots.has(topSegment);
    });
    for (const name of truncatedBundleRoots) {
      collection.issues.push({ kind: "truncated-bundle", name });
    }
  }

  return collection;
}

/**
 * One-call convenience for drop handlers: synchronous snapshot + async walk.
 * Call synchronously inside the drop event handler and await the result.
 */
export function collectFilesFromDataTransfer(
  dataTransfer: DataTransfer
): Promise<DropCollection> {
  return collectDroppedFiles(snapshotDropPayload(dataTransfer));
}

/**
 * Human-readable summary of a drop's problems, or null when there is nothing
 * the user needs to hear. Junk skipping is deliberately silent — stripping
 * .DS_Store is the expected behavior, not news.
 */
export function summarizeDropIssues(collection: DropCollection): string | null {
  const parts: string[] = [];
  const capIssue = collection.issues.find((issue) => issue.kind === "too-many-files");
  if (capIssue && capIssue.kind === "too-many-files") {
    parts.push(
      `Drops are limited to ${capIssue.limit.toLocaleString()} files — the first ${capIssue.limit.toLocaleString()} were kept. Split larger folders into separate drops.`
    );
  }
  const emptyDirectories = collection.issues.filter(
    (issue): issue is Extract<DropIssue, { kind: "empty-directory" }> =>
      issue.kind === "empty-directory"
  );
  if (emptyDirectories.length === 1) {
    parts.push(`"${emptyDirectories[0].name}" contained no uploadable files.`);
  } else if (emptyDirectories.length > 1) {
    parts.push(`${emptyDirectories.length} dropped folders contained no uploadable files.`);
  }
  const readErrors = collection.issues.filter((issue) => issue.kind === "entry-error");
  if (readErrors.length === 1 && readErrors[0].kind === "entry-error") {
    parts.push(`"${readErrors[0].path}" could not be read and was skipped.`);
  } else if (readErrors.length > 1) {
    parts.push(`${readErrors.length} items could not be read and were skipped.`);
  }
  for (const issue of collection.issues) {
    if (issue.kind === "truncated-bundle") {
      parts.push(
        `"${issue.name}" was skipped: the store did not fit in this drop's file limit, and a partial store must not upload. Drop it on its own.`
      );
    }
  }
  return parts.length > 0 ? parts.join(" ") : null;
}
