import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDurationSeconds, formatTokens } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TokenUsageResponse } from "@/types";
import { TokenActivityHeatmap } from "./TokenActivityHeatmap";
import { SystemMessage } from "./prompt-kit/system-message";

type UsagePanelDensity = "default" | "compact";

export type UserTokenUsagePanelProps = {
  tokenUsage: TokenUsageResponse | null;
  loading?: boolean;
  error?: string | null;
  className?: string;
  density?: UsagePanelDensity;
};

function UsageStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="user-token-usage-stat">
      <div className="user-token-usage-stat-value">{value}</div>
      <div className="user-token-usage-stat-label">{label}</div>
    </div>
  );
}

export function UserTokenUsagePanel({
  tokenUsage,
  loading = false,
  error = null,
  className,
  density = "default",
}: UserTokenUsagePanelProps) {
  const usageSummary = tokenUsage?.summary;
  return (
    <section className={cn("user-token-usage-panel", className)} data-density={density}>
      <div className="user-token-usage-heading">
        <h2>Usage</h2>
        <p>Token activity across all of your runs.</p>
      </div>
      <Separator />
      {loading ? (
        <div className="user-token-usage-loading">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-44 w-full" />
        </div>
      ) : error ? (
        <SystemMessage variant="error" fill>
          {error}
        </SystemMessage>
      ) : usageSummary ? (
        <>
          <div className="user-token-usage-stats">
            <UsageStat
              label="Lifetime tokens"
              value={formatTokens(usageSummary.lifetime_total_tokens)}
            />
            <UsageStat
              label="Peak day"
              value={formatTokens(usageSummary.peak_daily_total)}
            />
            <UsageStat
              label="Longest task"
              value={formatDurationSeconds(usageSummary.longest_task_seconds)}
            />
            <UsageStat
              label="Current streak"
              value={`${usageSummary.current_streak_days} day${
                usageSummary.current_streak_days === 1 ? "" : "s"
              }`}
            />
            <UsageStat
              label="Longest streak"
              value={`${usageSummary.longest_streak_days} day${
                usageSummary.longest_streak_days === 1 ? "" : "s"
              }`}
            />
          </div>
          <Separator />
          <TokenActivityHeatmap daily={tokenUsage.daily} />
        </>
      ) : (
        <div className="user-token-usage-empty">
          No token usage recorded yet. Run a task to start tracking.
        </div>
      )}
    </section>
  );
}
