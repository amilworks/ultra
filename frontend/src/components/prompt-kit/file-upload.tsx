import React from "react";
import { cn } from "../../lib/cn";
import { Button } from "@/components/ui/button";
import { Slot } from "radix-ui";

type FileUploadContextValue = {
  openFilePicker: () => void;
  openFolderPicker: () => void;
  allowDirectories: boolean;
  dragActive: boolean;
};

const FileUploadContext = React.createContext<FileUploadContextValue | null>(null);

function useFileUploadContext(): FileUploadContextValue {
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
};

export function FileUpload({
  onFilesAdded,
  children,
  multiple = true,
  accept,
  className,
  allowDirectories = false,
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

  const addFiles = React.useCallback(
    (fileList: FileList | null) => {
      const files = Array.from(fileList ?? []);
      if (files.length > 0) {
        onFilesAdded(files);
      }
    },
    [onFilesAdded]
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
          event.preventDefault();
          event.stopPropagation();
          setDragActive(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setDragActive(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          event.stopPropagation();
          const nextTarget = event.relatedTarget as Node | null;
          if (!nextTarget || !event.currentTarget.contains(nextTarget)) {
            setDragActive(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setDragActive(false);
          addFiles(event.dataTransfer.files);
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

export function FileUploadContent({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const { dragActive } = useFileUploadContext();
  return (
    <div
      {...props}
      className={cn("pk-file-upload-content", dragActive && "pk-file-upload-content-drag", className)}
    >
      {children}
    </div>
  );
}
