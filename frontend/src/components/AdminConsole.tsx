import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, RefreshCw, Shield, StopCircle, Trash2, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/format";
import type {
  AdminIssueRecord,
  AdminOverviewResponse,
  AdminRunRecord,
  AdminUserSummary,
} from "../types";

type AdminConsoleProps = {
  overview: AdminOverviewResponse | null;
  users: AdminUserSummary[];
  runs: AdminRunRecord[];
  issues: AdminIssueRecord[];
  loadingOverview: boolean;
  loadingUsers: boolean;
  loadingRuns: boolean;
  loadingIssues: boolean;
  error: string | null;
  runCancellingById: Record<string, boolean>;
  deletingConversationKey: string | null;
  runStatusFilter: string;
  runQuery: string;
  userQuery: string;
  onRunStatusFilterChange: (value: string) => void;
  onRunQueryChange: (value: string) => void;
  onUserQueryChange: (value: string) => void;
  onRefreshAll: () => void;
  onRefreshUsers: () => void;
  onRefreshRuns: () => void;
  onRefreshIssues: () => void;
  onCancelRun: (runId: string) => void;
  onDeleteConversation: (conversationId: string, userId: string) => void;
};

const statusOptions = [
  { value: "", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "pending", label: "Pending" },
  { value: "failed", label: "Failed" },
  { value: "succeeded", label: "Succeeded" },
  { value: "canceled", label: "Canceled" },
];

const usageChartConfig = {
  runs_total: { label: "Runs", color: "var(--chart-1)" },
  runs_failed: { label: "Failed", color: "var(--chart-5)" },
  uploads: { label: "Uploads", color: "var(--chart-2)" },
};

const toolChartConfig = {
  count: { label: "Runs", color: "var(--chart-1)" },
  failed: { label: "Failed", color: "var(--chart-5)" },
};

const countFormatter = new Intl.NumberFormat();

const formatClock = (iso: string): string => {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.valueOf())) {
    return iso;
  }
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

const formatCount = (value: number | null | undefined): string =>
  countFormatter.format(Number.isFinite(Number(value)) ? Number(value) : 0);

const formatDateTime = (iso: string | null | undefined): string => {
  if (!iso) {
    return "—";
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.valueOf())) {
    return String(iso);
  }
  return parsed.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
};

const formatDuration = (seconds: number | null | undefined): string => {
  const value = Number(seconds ?? 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "—";
  }
  if (value < 60) {
    return `${value.toFixed(0)}s`;
  }
  if (value < 3600) {
    return `${(value / 60).toFixed(1)}m`;
  }
  return `${(value / 3600).toFixed(1)}h`;
};

const runStatusBadgeClass = (status: string): string => {
  const normalized = status.toLowerCase();
  if (normalized === "running" || normalized === "pending") {
    return "border-emerald-500/45 bg-emerald-500/15 text-emerald-700 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-300";
  }
  if (normalized === "failed") {
    return "border-destructive/45 bg-destructive/10 text-destructive";
  }
  if (normalized === "succeeded") {
    return "border-sky-500/45 bg-sky-500/15 text-sky-700 dark:border-sky-400/45 dark:bg-sky-500/20 dark:text-sky-300";
  }
  return "border-border bg-muted/60 text-muted-foreground";
};

const issueSeverityBadgeClass = (severity: string): string => {
  const normalized = severity.toLowerCase();
  if (normalized === "high") {
    return "border-destructive/45 bg-destructive/10 text-destructive";
  }
  if (normalized === "medium") {
    return "border-amber-500/45 bg-amber-500/15 text-amber-700 dark:border-amber-400/45 dark:bg-amber-500/20 dark:text-amber-300";
  }
  return "border-border bg-muted/60 text-muted-foreground";
};

export function AdminConsole({
  overview,
  users,
  runs,
  issues,
  loadingOverview,
  loadingUsers,
  loadingRuns,
  loadingIssues,
  error,
  runCancellingById,
  deletingConversationKey,
  runStatusFilter,
  runQuery,
  userQuery,
  onRunStatusFilterChange,
  onRunQueryChange,
  onUserQueryChange,
  onRefreshAll,
  onRefreshUsers,
  onRefreshRuns,
  onRefreshIssues,
  onCancelRun,
  onDeleteConversation,
}: AdminConsoleProps) {
  const usageSeries = useMemo(
    () =>
      (overview?.usage_last_24h ?? []).map((row) => ({
        ...row,
        bucket_label: formatClock(row.bucket_start),
      })),
    [overview]
  );
  const toolSeries = useMemo(
    () =>
      (overview?.tool_usage_7d ?? []).map((row) => ({
        ...row,
        tool_label:
          row.tool_name.length > 18 ? `${row.tool_name.slice(0, 18)}…` : row.tool_name,
      })),
    [overview]
  );
  const kpis = overview?.kpis;
  const activeUserShare =
    kpis && kpis.total_users > 0 ? (kpis.active_users_24h / kpis.total_users) * 100 : 0;
  const messagesPerActiveUser =
    kpis && kpis.active_users_24h > 0 ? kpis.messages_last_24h / kpis.active_users_24h : 0;

  return (
    <section className="admin-console mx-auto flex-1 overflow-y-auto px-3 py-6 sm:px-6 sm:py-8">
      <div className="admin-shell">
        <div className="admin-header-row">
          <div>
            <h2 className="admin-title">Admin Console</h2>
            <p className="admin-subtitle">
              Platform health, user activity, run control, and operational triage.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="admin-refresh-button"
            onClick={onRefreshAll}
            disabled={loadingOverview || loadingUsers || loadingRuns || loadingIssues}
          >
            <RefreshCw
              className={cn(
                "size-4",
                (loadingOverview || loadingUsers || loadingRuns || loadingIssues) && "animate-spin"
              )}
            />
            Refresh
          </Button>
        </div>

        {error ? <p className="admin-error-banner">{error}</p> : null}

        <div className="admin-kpi-grid">
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Users</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.total_users ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Active (24h): {loadingOverview ? "…" : formatCount(kpis?.active_users_24h ?? 0)}
              </p>
              <p className="admin-kpi-footnote">
                Active share:{" "}
                {loadingOverview ? "…" : `${activeUserShare.toFixed(activeUserShare >= 10 ? 0 : 1)}%`}
              </p>
            </CardContent>
          </Card>
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Engaged Users (24h)</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.active_users_24h ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Messages / active user:{" "}
                {loadingOverview ? "…" : messagesPerActiveUser.toFixed(messagesPerActiveUser >= 10 ? 0 : 1)}
              </p>
              <p className="admin-kpi-footnote">
                Researchers with recent platform activity.
              </p>
            </CardContent>
          </Card>
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Conversations</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.total_conversations ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Started (24h):{" "}
                {loadingOverview ? "…" : formatCount(kpis?.conversations_started_24h ?? 0)}
              </p>
              <p className="admin-kpi-footnote">
                Avg messages:{" "}
                {loadingOverview ? "…" : (kpis?.avg_messages_per_conversation ?? 0).toFixed(1)}
              </p>
            </CardContent>
          </Card>
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Messages</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.total_messages ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Last 24h: {loadingOverview ? "…" : formatCount(kpis?.messages_last_24h ?? 0)}
              </p>
              <p className="admin-kpi-footnote">
                Prompts / replies:{" "}
                {loadingOverview
                  ? "…"
                  : `${formatCount(kpis?.user_messages_last_24h ?? 0)} / ${formatCount(kpis?.assistant_messages_last_24h ?? 0)}`}
              </p>
            </CardContent>
          </Card>
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Runs (24h)</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.runs_last_24h ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Success rate:{" "}
                {loadingOverview ? "…" : `${(kpis?.success_rate_last_24h ?? 0).toFixed(1)}%`}
              </p>
              <p className="admin-kpi-footnote">
                Running / failed:{" "}
                {loadingOverview
                  ? "…"
                  : `${formatCount(kpis?.running_runs ?? 0)} / ${formatCount(kpis?.failed_runs_24h ?? 0)}`}
              </p>
            </CardContent>
          </Card>
          <Card className="admin-kpi-card">
            <CardHeader>
              <CardDescription>Uploads</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.total_uploads ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Storage: {loadingOverview ? "…" : formatBytes(kpis?.total_storage_bytes ?? 0)}
              </p>
              <p className="admin-kpi-footnote">
                Soft deleted: {loadingOverview ? "…" : formatCount(kpis?.soft_deleted_uploads ?? 0)}
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="admin-chart-grid">
          <Card>
            <CardHeader>
              <CardTitle>Usage Last 24h</CardTitle>
              <CardDescription>Runs, failures, and uploads by hour.</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={usageChartConfig}
                className="admin-chart-canvas h-[260px] w-full"
              >
                <LineChart data={usageSeries}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="bucket_label" tickLine={false} axisLine={false} minTickGap={18} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="line" />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line type="monotone" dataKey="runs_total" stroke="var(--color-runs_total)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="runs_failed" stroke="var(--color-runs_failed)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="uploads" stroke="var(--color-uploads)" strokeWidth={2} dot={false} />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Tool Activity (7d)</CardTitle>
              <CardDescription>Top tools and failure load.</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={toolChartConfig}
                className="admin-chart-canvas h-[260px] w-full"
              >
                <BarChart data={toolSeries}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="tool_label" tickLine={false} axisLine={false} minTickGap={12} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="var(--color-count)" radius={6} />
                  <Bar dataKey="failed" fill="var(--color-failed)" radius={6} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>

        <div className="admin-data-grid">
          <Card className="admin-users-card">
            <CardHeader>
              <div className="admin-card-title-row">
                <div className="flex items-center gap-2">
                  <Users className="size-4" />
                  <CardTitle>Users</CardTitle>
                </div>
                <Button type="button" variant="ghost" size="sm" onClick={onRefreshUsers}>
                  <RefreshCw className={cn("size-4", loadingUsers && "animate-spin")} />
                </Button>
              </div>
              <CardDescription>Per-user utilization and latest activity.</CardDescription>
              <Input
                value={userQuery}
                onChange={(event) => onUserQueryChange(event.target.value)}
                placeholder="Filter by user id"
                className="admin-filter-input"
              />
            </CardHeader>
            <CardContent className="admin-table-wrap">
              {loadingUsers ? <p className="admin-empty">Loading users…</p> : null}
              {!loadingUsers && users.length === 0 ? (
                <p className="admin-empty">No users matched the current filter.</p>
              ) : null}
              {!loadingUsers && users.length > 0 ? (
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>User</th>
                      <th>Runs</th>
                      <th>Failed</th>
                      <th>Uploads</th>
                      <th>Storage</th>
                      <th>Last Activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((row) => (
                      <tr key={row.user_id}>
                        <td className="font-mono text-xs">{row.user_id}</td>
                        <td>{row.runs_total}</td>
                        <td>{row.runs_failed}</td>
                        <td>{row.uploads}</td>
                        <td>{formatBytes(row.storage_bytes)}</td>
                        <td>{formatDateTime(row.last_activity_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
            </CardContent>
          </Card>

          <Card className="admin-issues-card">
            <CardHeader>
              <div className="admin-card-title-row">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="size-4" />
                  <CardTitle>Recent Issues</CardTitle>
                </div>
                <Button type="button" variant="ghost" size="sm" onClick={onRefreshIssues}>
                  <RefreshCw className={cn("size-4", loadingIssues && "animate-spin")} />
                </Button>
              </div>
              <CardDescription>Failures and stalled processing signals.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {loadingIssues ? <p className="admin-empty">Loading issues…</p> : null}
              {!loadingIssues && issues.length === 0 ? (
                <p className="admin-empty">No issues found.</p>
              ) : null}
              {!loadingIssues && issues.length > 0 ? (
                <div className="space-y-2">
                  {issues.map((issue, index) => (
                    <article key={`${issue.issue_type}-${issue.run_id || issue.upload_id || index}`} className="admin-issue-item">
                      <div className="flex items-center justify-between gap-2">
                        <Badge variant="outline" className={issueSeverityBadgeClass(issue.severity)}>
                          {issue.severity}
                        </Badge>
                        <span className="admin-issue-time">{formatDateTime(issue.occurred_at)}</span>
                      </div>
                      <p className="admin-issue-message">{issue.message}</p>
                      <p className="admin-issue-meta">
                        {issue.user_id ? `User ${issue.user_id}` : "User unknown"}
                        {issue.run_id ? ` • Run ${issue.run_id.slice(0, 8)}` : ""}
                        {issue.upload_id ? ` • Upload ${issue.upload_id.slice(0, 8)}` : ""}
                      </p>
                    </article>
                  ))}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>

        <Card className="admin-runs-card">
          <CardHeader>
            <div className="admin-card-title-row">
              <div className="flex items-center gap-2">
                <Shield className="size-4" />
                <CardTitle>Runs</CardTitle>
              </div>
              <Button type="button" variant="ghost" size="sm" onClick={onRefreshRuns}>
                <RefreshCw className={cn("size-4", loadingRuns && "animate-spin")} />
              </Button>
            </div>
            <CardDescription>
              Cancel stuck runs and remove problematic conversation state.
            </CardDescription>
            <div className="admin-run-filters">
              <Input
                value={runQuery}
                onChange={(event) => onRunQueryChange(event.target.value)}
                placeholder="Filter by run id, goal, user, or conversation"
                className="admin-filter-input"
              />
              <div className="admin-status-filter">
                {statusOptions.map((option) => (
                  <Button
                    key={option.value || "all"}
                    type="button"
                    variant={runStatusFilter === option.value ? "secondary" : "ghost"}
                    size="sm"
                    onClick={() => onRunStatusFilterChange(option.value)}
                  >
                    {option.label}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent className="admin-table-wrap">
            {loadingRuns ? <p className="admin-empty">Loading runs…</p> : null}
            {!loadingRuns && runs.length === 0 ? (
              <p className="admin-empty">No runs matched the current filter.</p>
            ) : null}
            {!loadingRuns && runs.length > 0 ? (
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Status</th>
                    <th>User</th>
                    <th>Updated</th>
                    <th>Duration</th>
                    <th>Tools</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((row) => {
                    const isRunning =
                      row.status.toLowerCase() === "running" ||
                      row.status.toLowerCase() === "pending";
                    const conversationKey = `${row.user_id || ""}:${row.conversation_id || ""}`;
                    return (
                      <tr key={row.run_id}>
                        <td>
                          <div className="admin-run-id-cell">
                            <span className="font-mono text-xs">{row.run_id}</span>
                            <span className="admin-goal-preview" title={row.goal}>
                              {row.goal}
                            </span>
                          </div>
                        </td>
                        <td>
                          <Badge variant="outline" className={runStatusBadgeClass(row.status)}>
                            {row.status}
                          </Badge>
                        </td>
                        <td className="font-mono text-xs">{row.user_id || "—"}</td>
                        <td>{formatDateTime(row.updated_at)}</td>
                        <td>{formatDuration(row.duration_seconds)}</td>
                        <td>
                          <div className="admin-run-tools">
                            {row.tool_names.length > 0
                              ? row.tool_names.map((tool) => (
                                  <Badge key={`${row.run_id}-${tool}`} variant="outline">
                                    {tool}
                                  </Badge>
                                ))
                              : "—"}
                          </div>
                        </td>
                        <td>
                          <div className="admin-run-actions">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              disabled={!isRunning || Boolean(runCancellingById[row.run_id])}
                              onClick={() => onCancelRun(row.run_id)}
                            >
                              <StopCircle className="size-4" />
                              {runCancellingById[row.run_id] ? "Canceling…" : "Cancel"}
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              disabled={
                                !row.user_id ||
                                !row.conversation_id ||
                                deletingConversationKey === conversationKey
                              }
                              onClick={() => {
                                if (!row.user_id || !row.conversation_id) {
                                  return;
                                }
                                onDeleteConversation(row.conversation_id, row.user_id);
                              }}
                            >
                              <Trash2 className="size-4" />
                              {deletingConversationKey === conversationKey
                                ? "Deleting…"
                                : "Delete chat"}
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
