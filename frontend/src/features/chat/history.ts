// Sidebar conversation-history grouping types shared by the history list and
// the collapsed-rail recents popover.
export type HistoryPeriod = "Today" | "Yesterday" | "Last 7 days" | "Older";

export type HistoryItem = {
  id: string;
  title: string;
  preview: string;
  period: HistoryPeriod;
  running: boolean;
  messageCount: number;
};

// Module constant so the history grouping keeps a stable array reference (it is
// re-derived on every App render, including each keystroke).
export const HISTORY_PERIOD_ORDER: HistoryPeriod[] = ["Today", "Yesterday", "Last 7 days", "Older"];
