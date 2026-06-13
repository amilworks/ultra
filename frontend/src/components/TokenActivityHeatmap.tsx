import { useMemo, useState } from "react";

import { formatTokens } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TokenUsageDailyPoint } from "@/types";

type HeatmapMode = "daily" | "weekly" | "cumulative";

const WEEKS = 26;
const MODES: { value: HeatmapMode; label: string }[] = [
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "cumulative", label: "Cumulative" },
];

type HeatmapCell = {
  key: string;
  value: number;
  level: number;
  future: boolean;
};

const utcDayKey = (date: Date): string => date.toISOString().slice(0, 10);

const levelForRatio = (value: number, max: number): number => {
  if (value <= 0 || max <= 0) {
    return 0;
  }
  // sqrt eases the heavily skewed token distribution into a calmer gradient.
  const ratio = Math.sqrt(value / max);
  return Math.min(4, Math.max(1, Math.ceil(ratio * 4)));
};

const formatDayLabel = (key: string): string => {
  const date = new Date(`${key}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) {
    return key;
  }
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
};

export function TokenActivityHeatmap({
  daily,
}: {
  daily: TokenUsageDailyPoint[];
}) {
  const [mode, setMode] = useState<HeatmapMode>("daily");

  const totalsByDay = useMemo(() => {
    const map = new Map<string, number>();
    for (const point of daily) {
      if (point?.day) {
        map.set(point.day, Math.max(0, Number(point.total_tokens) || 0));
      }
    }
    return map;
  }, [daily]);

  const { cells, maxValue } = useMemo(() => {
    const now = new Date();
    const today = new Date(
      Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
    );
    const endDow = today.getUTCDay();
    const start = new Date(today);
    start.setUTCDate(start.getUTCDate() - ((WEEKS - 1) * 7 + endDow));

    const todayKey = utcDayKey(today);
    const days: { key: string; daily: number; future: boolean }[] = [];
    const cursor = new Date(start);
    for (let i = 0; i < WEEKS * 7; i += 1) {
      const key = utcDayKey(cursor);
      days.push({
        key,
        daily: totalsByDay.get(key) ?? 0,
        future: key > todayKey,
      });
      cursor.setUTCDate(cursor.getUTCDate() + 1);
    }

    // Per-mode cell value.
    let running = 0;
    const weeklyTotals: number[] = [];
    for (let week = 0; week < WEEKS; week += 1) {
      let sum = 0;
      for (let row = 0; row < 7; row += 1) {
        sum += days[week * 7 + row]?.daily ?? 0;
      }
      weeklyTotals[week] = sum;
    }

    const computed: HeatmapCell[] = days.map((day, index) => {
      let value = day.daily;
      if (mode === "cumulative") {
        running += day.daily;
        value = running;
      } else if (mode === "weekly") {
        value = weeklyTotals[Math.floor(index / 7)] ?? 0;
      }
      return { key: day.key, value, level: 0, future: day.future };
    });

    const max = computed.reduce(
      (acc, cell) => (cell.future ? acc : Math.max(acc, cell.value)),
      0
    );
    for (const cell of computed) {
      cell.level = cell.future ? 0 : levelForRatio(cell.value, max);
    }
    return { cells: computed, maxValue: max };
  }, [totalsByDay, mode]);

  const unitLabel = mode === "cumulative" ? "cumulative" : mode === "weekly" ? "this week" : "";

  return (
    <div className="token-heatmap">
      <div className="token-heatmap-toolbar">
        <div
          className="token-heatmap-modes"
          role="tablist"
          aria-label="Activity view"
        >
          {MODES.map((option) => (
            <button
              key={option.value}
              type="button"
              role="tab"
              aria-selected={mode === option.value}
              className={cn(
                "token-heatmap-mode",
                mode === option.value && "is-active"
              )}
              onClick={() => setMode(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <span className="token-heatmap-peak">
          {maxValue > 0
            ? `Peak ${formatTokens(maxValue)} tokens`
            : "No activity yet"}
        </span>
      </div>
      <div className="token-heatmap-grid" role="img" aria-label="Token activity heatmap">
        {cells.map((cell) =>
          cell.future ? (
            <span key={cell.key} className="token-heatmap-cell is-future" aria-hidden="true" />
          ) : (
            <span
              key={cell.key}
              className="token-heatmap-cell"
              data-level={cell.level}
              title={`${formatDayLabel(cell.key)} — ${formatTokens(cell.value)} tokens${
                unitLabel ? ` (${unitLabel})` : ""
              }`}
            />
          )
        )}
      </div>
      <div className="token-heatmap-legend">
        <span>Less</span>
        <span className="token-heatmap-cell" data-level={0} aria-hidden="true" />
        <span className="token-heatmap-cell" data-level={1} aria-hidden="true" />
        <span className="token-heatmap-cell" data-level={2} aria-hidden="true" />
        <span className="token-heatmap-cell" data-level={3} aria-hidden="true" />
        <span className="token-heatmap-cell" data-level={4} aria-hidden="true" />
        <span>More</span>
      </div>
    </div>
  );
}
