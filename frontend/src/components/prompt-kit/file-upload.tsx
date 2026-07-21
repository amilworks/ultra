import React from "react";
import { cn } from "../../lib/cn";
import { Button } from "@/components/ui/button";
import { Slot } from "radix-ui";
import {
  collectDroppedFiles,
  isOsFileDrag,
  normalizePickedFiles,
  snapshotDropPayload,
  type DropCollection,
} from "@/lib/dropTraversal";

type FileUploadContextValue = {
  openFilePicker: () => void;
  openFolderPicker: () => void;
  allowDirectories: boolean;
  dragActive: boolean;
};

const FileUploadContext = React.createContext<FileUploadContextValue | null>(null);

export function useFileUploadContext(): FileUploadContextValue {
  const context = React.useContext(FileUploadContext);
  if (!context) {
    throw new Error("File upload components must be used inside FileUpload.");
  }
  return context;
}

type FileUploadProps = {
  onFilesAdded: (files: File[]) => void;
  children: React.ReactNode;
  multiple?: boolean;
  accept?: string;
  className?: string;
  /** Enable a folder picker (webkitdirectory) for directory-format uploads (OME-Zarr). */
  allowDirectories?: boolean;
  /**
   * Drop outcome beyond the files themselves — skipped OS junk, unreadable
   * entries, empty/oversized folders. Fires after every drop traversal, even
   * when no files survive it (an empty drop is a signal the user needs).
   */
  onDropCollected?: (collection: DropCollection) => void;
};

// Only external OS file drags should light the drop affordance or be handled
// on drop — in-page drags (text selections, images, resource cards) also
// dispatch drag events over the composer, and image drags even carry "Files".
const isExternalFileDrag = (event: React.DragEvent): boolean =>
  isOsFileDrag(event.dataTransfer);

export function FileUpload({
  onFilesAdded,
  children,
  multiple = true,
  accept,
  className,
  allowDirectories = false,
  onDropCollected,
}: FileUploadProps) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const folderInputRef = React.useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = React.useState(false);

  // webkitdirectory/directory aren't in React's input typings — set them on the DOM node.
  React.useEffect(() => {
    if (allowDirectories && folderInputRef.current) {
      folderInputRef.current.setAttribute("webkitdirectory", "");
      folderInputRef.current.setAttribute("directory", "");
    }
  }, [allowDirectories]);

  // Picker selections flow through the SAME funnel as drops: junk filtering,
  // nested-bundle re-rooting, caps, and issue reporting. One affordance, one
  // set of guarantees — the intelligence lives here, not in which button the
  // user happened to press.
  const addFiles = React.useCallback(
    (fileList: FileList | null) => {
      const files = Array.from(fileList ?? []);
      if (files.length === 0) {
        return;
      }
      const collection = normalizePickedFiles(files);
      if (collection.files.length > 0) {
        onFilesAdded(collection.files);
      }
      onDropCollected?.(collection);
    },
    [onFilesAdded, onDropCollected]
  );

  return (
    <FileUploadContext.Provider
      value={{
        openFilePicker: () => inputRef.current?.click(),
        openFolderPicker: () => folderInputRef.current?.click(),
        allowDirectories,
        dragActive,
      }}
    >
      <div
        className={cn("pk-file-upload", dragActive && "pk-file-upload-drag", className)}
        onDragEnter={(event) => {
          if (!isExternalFileDrag(event)) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          setDragActive(true);
        }}
        onDragOver={(event) => {
          if (!isExternalFileDrag(event)) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          setDragActive(true);
        }}
        onDragLeave={(event) => {
          if (!isExternalFileDrag(event)) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          const nextTarget = event.relatedTarget as Node | null;
          if (!nextTarget || !event.currentTarget.contains(nextTarget)) {
            setDragActive(false);
          }
        }}
        onDrop={(event) => {
          if (!isExternalFileDrag(event)) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          setDragActive(false);
          // Entries are only alive during this dispatch: snapshot NOW, walk async.
          const payload = snapshotDropPayload(event.dataTransfer);
          void collectDroppedFiles(payload).then((collection) => {
            if (collection.files.length > 0) {
              onFilesAdded(collection.files);
            }
            onDropCollected?.(collection);
          });
        }}
      >
        <input
          ref={inputRef}
          type="file"
          className="pk-file-upload-input"
          multiple={multiple}
          accept={accept}
          onChange={(event) => {
            addFiles(event.target.files);
            event.currentTarget.value = "";
          }}
        />
        {allowDirectories ? (
          <input
            ref={folderInputRef}
            type="file"
            className="pk-file-upload-input"
            multiple
            onChange={(event) => {
              addFiles(event.target.files);
              event.currentTarget.value = "";
            }}
          />
        ) : null}
        {children}
      </div>
    </FileUploadContext.Provider>
  );
}

type FileUploadTriggerProps = React.ComponentPropsWithoutRef<"button"> & {
  asChild?: boolean;
};

export const FileUploadTrigger = React.forwardRef<HTMLButtonElement, FileUploadTriggerProps>(
  function FileUploadTrigger(
    {
      asChild = false,
      className,
      children,
      onClick,
      ...props
    }: FileUploadTriggerProps,
    ref
  ) {
    const { openFilePicker } = useFileUploadContext();

    if (asChild) {
      return (
        <Slot.Root
          {...props}
          ref={ref as React.Ref<HTMLElement>}
          role="button"
          className={cn(className)}
          onClick={(event) => {
            event.stopPropagation();
            onClick?.(event as React.MouseEvent<HTMLButtonElement>);
            if (!event.defaultPrevented) {
              openFilePicker();
            }
          }}
        >
          {children}
        </Slot.Root>
      );
    }

    return (
      <Button
        {...props}
        ref={ref}
        type={props.type ?? "button"}
        variant="outline"
        size="sm"
        className={cn("pk-file-upload-trigger", className)}
        onClick={(event) => {
          event.stopPropagation();
          onClick?.(event);
          if (!event.defaultPrevented) {
            openFilePicker();
          }
        }}
      >
        {children}
      </Button>
    );
  }
);

FileUploadTrigger.displayName = "FileUploadTrigger";

// Opens the folder picker (webkitdirectory) for directory-format uploads (OME-Zarr).
// Renders nothing unless the parent FileUpload has allowDirectories.
export const FileUploadFolderTrigger = React.forwardRef<HTMLButtonElement, FileUploadTriggerProps>(
  function FileUploadFolderTrigger({ asChild = false, className, children, onClick, ...props }, ref) {
    const { openFolderPicker, allowDirectories } = useFileUploadContext();
    if (!allowDirectories) {
      return null;
    }
    if (asChild) {
      return (
        <Slot.Root
          {...props}
          ref={ref as React.Ref<HTMLElement>}
          role="button"
          className={cn(className)}
          onClick={(event) => {
            event.stopPropagation();
            onClick?.(event as React.MouseEvent<HTMLButtonElement>);
            if (!event.defaultPrevented) {
              openFolderPicker();
            }
          }}
        >
          {children}
        </Slot.Root>
      );
    }
    return (
      <Button
        {...props}
        ref={ref}
        type={props.type ?? "button"}
        variant="outline"
        size="sm"
        className={cn("pk-file-upload-trigger", className)}
        onClick={(event) => {
          event.stopPropagation();
          onClick?.(event);
          if (!event.defaultPrevented) {
            openFolderPicker();
          }
        }}
      >
        {children}
      </Button>
    );
  }
);

FileUploadFolderTrigger.displayName = "FileUploadFolderTrigger";
