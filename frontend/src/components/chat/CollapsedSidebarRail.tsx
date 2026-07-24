import type { ReactNode } from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { LensSidebarIcon } from "@/components/icons/LensSidebarIcon";
import { Database, FolderOpen, MessageCircle, PlusIcon, Shield } from "lucide-react";
import { RunningStatusPill } from "./RunningStatusPill";
import type { HistoryItem } from "@/features/chat/history";

type CollapsedSidebarRailProps = {
  recentItems: HistoryItem[];
  activeConversationId: string | null;
  resourcesActive: boolean;
  trainingActive: boolean;
  lensActive: boolean;
  adminActive: boolean;
  isAdmin: boolean;
  onCreateConversation: () => void;
  onOpenResources: () => void;
  onOpenTraining: () => void;
  onOpenLens: () => void;
  onOpenAdmin: () => void;
  onOpenRecent: (conversation: HistoryItem) => void;
};

/**
 * Collapsed-rail tooltip: the icon is the only label a collapsed control has, so
 * every rail control gets one. Sided right (the rail hugs the left edge) and
 * carrying the same keyboard shortcut the expanded row shows on hover, so
 * collapsing the sidebar costs no discoverability.
 *
 * These replace the previous native `title=` attributes rather than joining
 * them — keeping both would show two tooltips, the browser's and this one.
 * `aria-label` stays on each control; it, not this, is what assistive tech reads.
 */
function RailTooltip({
  label,
  shortcut,
  children,
}: {
  label: string;
  shortcut?: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side="right" sideOffset={8} className="app-collapsed-sidebar-tooltip">
        <span>{label}</span>
        {shortcut ? (
          <span className="app-collapsed-sidebar-tooltip-key" aria-hidden="true">
            {shortcut}
          </span>
        ) : null}
      </TooltipContent>
    </Tooltip>
  );
}

export function CollapsedSidebarRail({
  recentItems,
  activeConversationId,
  resourcesActive,
  trainingActive,
  lensActive,
  adminActive,
  isAdmin,
  onCreateConversation,
  onOpenResources,
  onOpenTraining,
  onOpenLens,
  onOpenAdmin,
  onOpenRecent,
}: CollapsedSidebarRailProps) {
  const [recentsOpen, setRecentsOpen] = useState(false);

  return (
    <nav className="app-collapsed-sidebar-rail" aria-label="Collapsed navigation">
      <RailTooltip label="Expand sidebar">
        <SidebarTrigger className="app-collapsed-sidebar-toggle" aria-label="Expand sidebar" />
      </RailTooltip>

      {/* Mirrors the expanded sidebar's primary nav, in the same order, so the
          two states are the same map at two zoom levels — then Recents last,
          standing in for the history list that sits below the nav when open. */}
      <div className="app-collapsed-sidebar-actions" role="list">
        <RailTooltip label="New chat" shortcut="⌘⇧K">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="app-collapsed-sidebar-action"
            aria-label="New chat"
            aria-keyshortcuts="Control+Shift+K Meta+Shift+K"
            onClick={onCreateConversation}
          >
            <PlusIcon data-icon="inline-start" />
          </Button>
        </RailTooltip>

        <RailTooltip label="Resources" shortcut="⌘⇧E">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="app-collapsed-sidebar-action"
            aria-label="Resources"
            aria-keyshortcuts="Control+Shift+E Meta+Shift+E"
            data-active={resourcesActive ? "true" : undefined}
            onClick={onOpenResources}
          >
            <FolderOpen data-icon="inline-start" />
          </Button>
        </RailTooltip>

        <RailTooltip label="Training" shortcut="⌘⇧T">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="app-collapsed-sidebar-action"
            aria-label="Training dashboard"
            aria-keyshortcuts="Control+Shift+T Meta+Shift+T"
            data-active={trainingActive ? "true" : undefined}
            onClick={onOpenTraining}
          >
            <Database data-icon="inline-start" />
          </Button>
        </RailTooltip>

        {isAdmin ? (
          <RailTooltip label="Admin">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="app-collapsed-sidebar-action"
              aria-label="Admin"
              data-active={adminActive ? "true" : undefined}
              onClick={onOpenAdmin}
            >
              <Shield data-icon="inline-start" />
            </Button>
          </RailTooltip>
        ) : null}

        <RailTooltip label="Lens — scientific image viewer">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="app-collapsed-sidebar-action"
            aria-label="Lens — scientific image viewer"
            data-active={lensActive ? "true" : undefined}
            onClick={onOpenLens}
          >
            <LensSidebarIcon active={lensActive} data-icon="inline-start" aria-hidden="true" />
          </Button>
        </RailTooltip>

        <Popover open={recentsOpen} onOpenChange={setRecentsOpen}>
          <RailTooltip label="Recents">
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="app-collapsed-sidebar-action"
                aria-label="Recents"
                data-active={recentsOpen ? "true" : undefined}
              >
                <MessageCircle data-icon="inline-start" />
              </Button>
            </PopoverTrigger>
          </RailTooltip>
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
