import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { FolderOpen, MessageCircle, PlusIcon } from "lucide-react";
import { RunningStatusPill } from "./RunningStatusPill";
import type { HistoryItem } from "@/features/chat/history";

type CollapsedSidebarRailProps = {
  recentItems: HistoryItem[];
  activeConversationId: string | null;
  resourcesActive: boolean;
  onCreateConversation: () => void;
  onOpenResources: () => void;
  onOpenRecent: (conversation: HistoryItem) => void;
};

export function CollapsedSidebarRail({
  recentItems,
  activeConversationId,
  resourcesActive,
  onCreateConversation,
  onOpenResources,
  onOpenRecent,
}: CollapsedSidebarRailProps) {
  const [recentsOpen, setRecentsOpen] = useState(false);

  return (
    <nav className="app-collapsed-sidebar-rail" aria-label="Collapsed navigation">
      <SidebarTrigger
        className="app-collapsed-sidebar-toggle"
        aria-label="Expand sidebar"
        title="Expand sidebar"
      />

      <div className="app-collapsed-sidebar-actions" role="list">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-collapsed-sidebar-action"
          aria-label="New chat"
          title="New chat"
          onClick={onCreateConversation}
        >
          <PlusIcon data-icon="inline-start" />
        </Button>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-collapsed-sidebar-action"
          aria-label="Resources"
          title="Resources"
          data-active={resourcesActive ? "true" : undefined}
          onClick={onOpenResources}
        >
          <FolderOpen data-icon="inline-start" />
        </Button>

        <Popover open={recentsOpen} onOpenChange={setRecentsOpen}>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="app-collapsed-sidebar-action"
              aria-label="Recents"
              title="Recents"
              data-active={recentsOpen ? "true" : undefined}
            >
              <MessageCircle data-icon="inline-start" />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            side="right"
            align="center"
            sideOffset={8}
            className="app-collapsed-recents-popover"
          >
            <div className="app-collapsed-recents-header">Recents</div>
            <div className="app-collapsed-recents-list">
              {recentItems.length > 0 ? (
                recentItems.map((conversation) => (
                  <button
                    key={conversation.id}
                    type="button"
                    className="app-collapsed-recent-item"
                    data-active={conversation.id === activeConversationId ? "true" : undefined}
                    onClick={() => {
                      onOpenRecent(conversation);
                      setRecentsOpen(false);
                    }}
                  >
                    <span className="truncate">{conversation.title}</span>
                    {conversation.running ? <RunningStatusPill size="compact" /> : null}
                  </button>
                ))
              ) : (
                <div className="app-collapsed-recents-empty">No recent chats yet</div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </nav>
  );
}
