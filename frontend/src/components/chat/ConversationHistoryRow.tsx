import { memo, type KeyboardEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { Input } from "@/components/ui/input";
import {
  SidebarMenuButton,
  SidebarMenuItem,
  mobileSidebarCloseProps,
} from "@/components/ui/sidebar";
import { Check, Copy, Link2, Pencil, Trash, X } from "lucide-react";
import { RunningStatusPill } from "./RunningStatusPill";
import type { HistoryItem } from "@/features/chat/history";

type ConversationHistoryMenuProps = {
  children: ReactNode;
  conversationId: string;
  conversationTitle: string;
  deleting: boolean;
  renaming: boolean;
  onCopyLink: (conversationId: string) => Promise<void>;
  onCopyId: (conversationId: string) => Promise<void>;
  onRename: (conversationId: string, conversationTitle: string) => void;
  onDelete: (conversationId: string) => void;
};

const ConversationHistoryMenu = ({
  children,
  conversationId,
  conversationTitle,
  deleting,
  renaming,
  onCopyLink,
  onCopyId,
  onRename,
  onDelete,
}: ConversationHistoryMenuProps) => (
  <ContextMenu>
    <ContextMenuTrigger asChild disabled={deleting}>
      {children}
    </ContextMenuTrigger>
    <ContextMenuContent className="w-52 rounded-lg">
      <ContextMenuItem
        disabled={deleting || renaming}
        onSelect={() => {
          if (deleting || renaming) {
            return;
          }
          onRename(conversationId, conversationTitle);
        }}
      >
        <Pencil
          aria-hidden="true"
          className="conversation-history-menu-icon text-muted-foreground"
        />
        <span>Rename chat</span>
      </ContextMenuItem>
      <ContextMenuItem onSelect={() => void onCopyLink(conversationId)}>
        <Link2
          aria-hidden="true"
          className="conversation-history-menu-icon text-muted-foreground"
        />
        <span>Copy chat link</span>
      </ContextMenuItem>
      <ContextMenuItem onSelect={() => void onCopyId(conversationId)}>
        <Copy
          aria-hidden="true"
          className="conversation-history-menu-icon text-muted-foreground"
        />
        <span>Copy chat ID</span>
      </ContextMenuItem>
      <ContextMenuItem
        variant="destructive"
        disabled={deleting}
        onSelect={() => {
          if (deleting) {
            return;
          }
          onDelete(conversationId);
        }}
      >
        <Trash aria-hidden="true" className="conversation-history-menu-icon" />
        <span>Delete chat</span>
      </ContextMenuItem>
    </ContextMenuContent>
  </ContextMenu>
);

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

const openConversationMenuFromKeyboard = (event: KeyboardEvent<HTMLButtonElement>): void => {
  if (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10")) {
    return;
  }

  event.preventDefault();
  const bounds = event.currentTarget.getBoundingClientRect();
  event.currentTarget.dispatchEvent(
    new MouseEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      clientX: bounds.left + Math.min(24, bounds.width / 2),
      clientY: bounds.top + bounds.height / 2,
    })
  );
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
      <ConversationHistoryMenu
        conversationId={conversation.id}
        conversationTitle={conversation.title}
        deleting={deleting}
        renaming={renaming}
        onCopyLink={onCopyLink}
        onCopyId={onCopyId}
        onRename={onRename}
        onDelete={onDelete}
      >
        <SidebarMenuButton
          isActive={active}
          className="app-history-button h-auto py-2"
          onClick={() => onOpen(conversation)}
          onKeyDown={openConversationMenuFromKeyboard}
          aria-haspopup="menu"
          aria-keyshortcuts="Shift+F10"
          {...mobileSidebarCloseProps}
        >
          <div className="flex min-w-0 w-full items-center gap-2">
            <span className="truncate">{conversation.title}</span>
            <div className="ml-auto flex items-center gap-1.5">
              {conversation.running ? <RunningStatusPill size="compact" /> : null}
            </div>
          </div>
        </SidebarMenuButton>
      </ConversationHistoryMenu>
    </SidebarMenuItem>
  );
});
