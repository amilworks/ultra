import { FileUp, FolderUp, Library, NotebookPen, Plus, Slash } from "lucide-react";

import { useFileUploadContext } from "@/components/prompt-kit/file-upload";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { ComposerTooltip } from "./ComposerTooltip";

/* ONE attach affordance. The browser forces two hidden inputs (webkitdirectory
   is exclusive), but the user sees a single +: format intelligence (zarr
   re-rooting, junk, caps) lives in the shared upload funnel, not in which
   button was pressed. Library and workflow lead the list because they are the
   two things the + used to hide. */
export function ComposerAttachMenu({
  disabled,
  onCloseAutoFocus,
  onOpenNotes,
  onOpenResources,
  onStartWorkflow,
}: {
  disabled: boolean;
  onCloseAutoFocus?: (event: Event) => void;
  onOpenNotes?: () => void;
  onOpenResources: () => void;
  onStartWorkflow: () => void;
}) {
  const { openFilePicker, openFolderPicker, allowDirectories } = useFileUploadContext();
  const addLabel = onOpenNotes ? "Add files, folders, or a Note" : "Add files or a folder";
  return (
    <DropdownMenu>
      <ComposerTooltip label={addLabel} disabled={disabled}>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={addLabel}
            data-testid="composer-attach-menu"
            className="composer-control composer-attach"
            disabled={disabled}
            onMouseDown={(event) => event.preventDefault()}
          >
            <Plus size={18} />
          </Button>
        </DropdownMenuTrigger>
      </ComposerTooltip>
      <DropdownMenuContent
        align="start"
        sideOffset={8}
        className="app-composer-attach-menu"
        onCloseAutoFocus={onCloseAutoFocus}
      >
        <DropdownMenuItem onSelect={onOpenResources} data-testid="composer-attach-resources">
          <Library data-icon="inline-start" aria-hidden="true" />
          <div className="app-composer-attach-menu-item">
            <span>From your Resources</span>
            <span className="app-composer-attach-menu-detail">Files already in your library</span>
          </div>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onStartWorkflow} data-testid="composer-attach-workflow">
          <Slash data-icon="inline-start" aria-hidden="true" />
          <div className="app-composer-attach-menu-item">
            <span>Start from a workflow</span>
            <span className="app-composer-attach-menu-detail">The same list as typing /</span>
          </div>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => openFilePicker()}>
          <FileUp data-icon="inline-start" aria-hidden="true" />
          <div className="app-composer-attach-menu-item">
            <span>Files</span>
            <span className="app-composer-attach-menu-detail">Images, tables, documents</span>
          </div>
        </DropdownMenuItem>
        {allowDirectories ? (
          <DropdownMenuItem onSelect={() => openFolderPicker()}>
            <FolderUp data-icon="inline-start" aria-hidden="true" />
            <div className="app-composer-attach-menu-item">
              <span>Folder</span>
              <span className="app-composer-attach-menu-detail">OME-Zarr uploads as one dataset</span>
            </div>
          </DropdownMenuItem>
        ) : null}
        {onOpenNotes ? (
          <DropdownMenuItem onSelect={onOpenNotes}>
            <NotebookPen data-icon="inline-start" aria-hidden="true" />
            <div className="app-composer-attach-menu-item">
              <span>Use a note</span>
              <span className="app-composer-attach-menu-detail">Use for this message</span>
            </div>
          </DropdownMenuItem>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
