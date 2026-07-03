import { memo } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  mobileSidebarCloseProps,
  mobileSidebarKeepOpenProps,
  useSidebar,
} from "@/components/ui/sidebar";
import { Check, Copy, Link2, MoreHorizontal, Pencil, Trash, X } from "lucide-react";
import { RunningStatusPill } from "./RunningStatusPill";
import type { HistoryItem } from "@/features/chat/history";

type ConversationHistoryActionsProps = {
  conversationId: string;
  conversationTitle: string;
  deleting: boolean;
  renaming: boolean;
  onCopyLink: (conversationId: string) => Promise<void>;
  onCopyId: (conversationId: string) => Promise<void>;
  onRename: (conversationId: string, conversationTitle: string) => void;
  onDelete: (conversationId: string) => void;
};

const ConversationHistoryActions = ({
  conversationId,
  conversationTitle,
  deleting,
  renaming,
  onCopyLink,
  onCopyId,
  onRename,
  onDelete,
}: ConversationHistoryActionsProps) => {
  const { isMobile } = useSidebar();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <SidebarMenuAction asChild showOnHover {...mobileSidebarKeepOpenProps}>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={`Conversation actions for ${conversationTitle}`}
            disabled={deleting}
            className="app-history-action-button size-7 rounded-md border border-transparent bg-transparent p-0 text-muted-foreground shadow-none hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
          >
            <MoreHorizontal />
            <span className="sr-only">Conversation actions</span>
          </Button>
        </SidebarMenuAction>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        className="w-52 rounded-lg"
        side={isMobile ? "bottom" : "right"}
        align={isMobile ? "end" : "start"}
        sideOffset={8}
      >
        <DropdownMenuItem
          disabled={deleting || renaming}
          onClick={() => {
            if (deleting || renaming) {
              return;
            }
            onRename(conversationId, conversationTitle);
          }}
        >
          <Pencil className="text-muted-foreground" />
          <span>Rename chat</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void onCopyLink(conversationId)}>
          <Link2 className="text-muted-foreground" />
          <span>Copy chat link</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void onCopyId(conversationId)}>
          <Copy className="text-muted-foreground" />
          <span>Copy chat ID</span>
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          disabled={deleting}
          onClick={() => {
            if (deleting) {
              return;
            }
            onDelete(conversationId);
          }}
        >
          <Trash />
          <span>Delete chat</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

type ConversationRenameEditorProps = {
  conversation: HistoryItem;
  value: string;
  disabled: boolean;
  onTitleChange: (conversationId: string, title: string) => void;
  onSubmit: () => Promise<void>;
  onCancel: () => void;
};

export const ConversationRenameEditor = memo(function ConversationRenameEditor({
  conversation,
  value,
  disabled,
  onTitleChange,
  onSubmit,
  onCancel,
}: ConversationRenameEditorProps) {
  return (
    <div className="app-history-rename-shell">
      <Input
        value={value}
        onChange={(event) => {
          onTitleChange(conversation.id, event.target.value);
        }}
        onFocus={(event) => {
          event.currentTarget.select();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void onSubmit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
          }
        }}
        autoFocus
        maxLength={120}
        aria-label={`Rename ${conversation.title}`}
        data-testid="conversation-rename-input"
        className="app-history-rename-input"
        disabled={disabled}
      />
      <div className="app-history-rename-actions">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-history-rename-button"
          aria-label="Save chat name"
          onClick={() => {
            void onSubmit();
          }}
          disabled={disabled}
        >
          <Check className="size-4" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-history-rename-button"
          aria-label="Cancel renaming chat"
          onClick={onCancel}
          disabled={disabled}
        >
          <X className="size-4" />
        </Button>
      </div>
    </div>
  );
});

type ConversationHistoryRowProps = {
  conversation: HistoryItem;
  active: boolean;
  deleting: boolean;
  renaming: boolean;
  onOpen: (conversation: HistoryItem) => void;
  onCopyLink: (conversationId: string) => Promise<void>;
  onCopyId: (conversationId: string) => Promise<void>;
  onRename: (conversationId: string, conversationTitle: string) => void;
  onDelete: (conversationId: string) => void;
};

export const ConversationHistoryRow = memo(function ConversationHistoryRow({
  conversation,
  active,
  deleting,
  renaming,
  onOpen,
  onCopyLink,
  onCopyId,
  onRename,
  onDelete,
}: ConversationHistoryRowProps) {
  return (
    <SidebarMenuItem className="app-history-item">
      <SidebarMenuButton
        isActive={active}
        className="app-history-button group/history h-auto py-2"
        onClick={() => onOpen(conversation)}
        {...mobileSidebarCloseProps}
      >
        <div className="flex min-w-0 w-full items-center gap-2">
          <span className="truncate">{conversation.title}</span>
          <div className="ml-auto flex items-center gap-1.5">
            {conversation.running ? <RunningStatusPill size="compact" /> : null}
          </div>
        </div>
      </SidebarMenuButton>
      <ConversationHistoryActions
        conversationId={conversation.id}
        conversationTitle={conversation.title}
        deleting={deleting}
        renaming={renaming}
        onCopyLink={onCopyLink}
        onCopyId={onCopyId}
        onRename={onRename}
        onDelete={onDelete}
      />
    </SidebarMenuItem>
  );
});
