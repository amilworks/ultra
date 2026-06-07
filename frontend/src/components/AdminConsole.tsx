import { Fragment, type FormEvent, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  Building2,
  CheckCircle2,
  LayoutDashboard,
  RefreshCw,
  RotateCcw,
  Shield,
  StopCircle,
  Trash2,
  UserPlus,
  Users,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  AdminCreateOrganizationRequest,
  AdminCreateUserRequest,
  AdminActivityPeriod,
  AdminIssueRecord,
  AdminOrganization,
  AdminOverviewResponse,
  AdminRunRecord,
  AdminUserStatus,
  AdminUserSummary,
  RunEvent,
} from "../types";

type AdminConsoleProps = {
  overview: AdminOverviewResponse | null;
  organizations: AdminOrganization[];
  users: AdminUserSummary[];
  runs: AdminRunRecord[];
  issues: AdminIssueRecord[];
  loadingOverview: boolean;
  loadingOrganizations: boolean;
  loadingUsers: boolean;
  loadingRuns: boolean;
  loadingIssues: boolean;
  error: string | null;
  runCancellingById: Record<string, boolean>;
  runRequeueingById?: Record<string, boolean>;
  userDeletingById?: Record<string, boolean>;
  userUpdatingById?: Record<string, boolean>;
  deletingConversationKey: string | null;
  activeRunEventRunId: string | null;
  runEventsById: Record<string, RunEvent[]>;
  runEventsLoadingById: Record<string, boolean>;
  runStatusFilter: string;
  runQuery: string;
  userQuery: string;
  onRunStatusFilterChange: (value: string) => void;
  onRunQueryChange: (value: string) => void;
  onUserQueryChange: (value: string) => void;
  onRefreshAll: () => void;
  onRefreshOrganizations: () => void;
  onRefreshUsers: () => void;
  onRefreshRuns: () => void;
  onRefreshIssues: () => void;
  onCreateOrganization: (payload: AdminCreateOrganizationRequest) => Promise<void> | void;
  onCreateUser: (payload: AdminCreateUserRequest) => Promise<void> | void;
  onDeleteUser: (userId: string) => Promise<void> | void;
  onUpdateUserStatus?: (userId: string, status: AdminUserStatus) => Promise<void> | void;
  onCancelRun: (runId: string) => void;
  onRequeueRun?: (runId: string) => void;
  onDeleteConversation: (conversationId: string, userId: string) => void;
  onInspectRunEvents: (runId: string) => void;
};

type AdminConsolePage = "overview" | "users";

const statusOptions = [
  { value: "", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "queued", label: "Queued" },
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

const activityPeriodOrder = ["Daily", "Weekly", "Monthly", "Total"];

const activityMetricRows: Array<{
  label: string;
  value: (period: AdminActivityPeriod) => number;
}> = [
  { label: "Messages", value: (period) => period.messages },
  { label: "Tool calls", value: (period) => period.tool_calls },
  { label: "Active users", value: (period) => period.active_users },
  { label: "Runs", value: (period) => period.runs },
  { label: "Failed runs", value: (period) => period.failed_runs },
  { label: "Artifacts", value: (period) => period.artifacts },
];

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

const formatSilentAge = (seconds: number | null | undefined): string => {
  const value = Number(seconds ?? 0);
  if (!Number.isFinite(value) || value < 0) {
    return "—";
  }
  if (value < 60) {
    return `${value.toFixed(0)}s silent`;
  }
  if (value < 3600) {
    return `${(value / 60).toFixed(1)}m silent`;
  }
  return `${(value / 3600).toFixed(1)}h silent`;
};

const pluralUnit = (count: number, singular: string): string =>
  `${formatCount(count)} ${singular}${count === 1 ? "" : "s"}`;

const formatRunEventSummary = (run: AdminRunRecord): string => {
  const deltaCount = Math.max(0, Number(run.message_delta_count ?? 0));
  const toolCount = Math.max(0, Number(run.tool_call_count ?? 0));
  const artifactCount = Math.max(0, Number(run.artifact_count ?? 0));
  const heartbeatCount = Math.max(0, Number(run.heartbeat_count ?? 0));
  return [
    pluralUnit(deltaCount, "delta"),
    pluralUnit(toolCount, "tool"),
    pluralUnit(artifactCount, "artifact"),
    pluralUnit(heartbeatCount, "heartbeat"),
  ].join(" • ");
};

const formatRunLeaseSummary = (run: AdminRunRecord): string | null => {
  const workerId = String(run.lease_worker_id ?? "").trim();
  if (!workerId) {
    return null;
  }
  if (run.lease_expired) {
    return `Lease expired: ${workerId}`;
  }
  if (run.lease_active && run.lease_seconds_remaining !== null && run.lease_seconds_remaining !== undefined) {
    return `Lease: ${workerId} (${formatDuration(run.lease_seconds_remaining)} left)`;
  }
  return `Lease: ${workerId}`;
};

const isRecoverableRunStatus = (status: string): boolean => {
  const normalized = status.toLowerCase();
  return (
    normalized === "queued" ||
    normalized === "pending" ||
    normalized === "running" ||
    normalized === "waiting_for_input" ||
    normalized === "waiting_for_task"
  );
};

const formatQueueConsumerSummary = (
  pending: number | null | undefined,
  inFlight: number | null | undefined,
  redelivered: number | null | undefined
): string =>
  [
    `${formatCount(pending)} pending`,
    `${formatCount(inFlight)} in-flight`,
    `${formatCount(redelivered)} redelivered`,
  ].join(" • ");

const runEventPayload = (event: RunEvent): Record<string, unknown> =>
  event.payload && typeof event.payload === "object" ? event.payload : {};

const runEventSequence = (event: RunEvent): number | null => {
  const value = Number(runEventPayload(event).sequence);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : null;
};

const runEventLabel = (event: RunEvent): string => {
  const kind = String(runEventPayload(event).event_kind ?? event.event_type ?? "run_event");
  const sequence = runEventSequence(event);
  return sequence ? `${kind} #${sequence}` : kind;
};

const runEventDetail = (event: RunEvent): string => {
  const payload = runEventPayload(event);
  for (const key of ["message", "delta", "stage", "tool_name", "artifact_id"]) {
    const value = String(payload[key] ?? "").trim();
    if (value) {
      return value;
    }
  }
  return "No event detail";
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

const userStatusBadgeClass = (status: string): string => {
  const normalized = status.toLowerCase();
  if (normalized === "active") {
    return "admin-runtime-good";
  }
  if (normalized === "disabled") {
    return "border-destructive/45 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted/60 text-muted-foreground";
};

export function AdminConsole({
  overview,
  organizations,
  users,
  runs,
  issues,
  loadingOverview,
  loadingOrganizations,
  loadingUsers,
  loadingRuns,
  loadingIssues,
  error,
  runCancellingById,
  runRequeueingById = {},
  userDeletingById = {},
  userUpdatingById = {},
  deletingConversationKey,
  activeRunEventRunId,
  runEventsById,
  runEventsLoadingById,
  runStatusFilter,
  runQuery,
  userQuery,
  onRunStatusFilterChange,
  onRunQueryChange,
  onUserQueryChange,
  onRefreshAll,
  onRefreshOrganizations,
  onRefreshUsers,
  onRefreshRuns,
  onRefreshIssues,
  onCreateOrganization,
  onCreateUser,
  onDeleteUser,
  onUpdateUserStatus,
  onCancelRun,
  onRequeueRun,
  onDeleteConversation,
  onInspectRunEvents,
}: AdminConsoleProps) {
  const [activePage, setActivePage] = useState<AdminConsolePage>("overview");
  const [newUserEmail, setNewUserEmail] = useState("");
  const [newUserName, setNewUserName] = useState("");
  const [newUserRole, setNewUserRole] = useState("researcher");
  const [newUserOrg, setNewUserOrg] = useState("local-org");
  const [newOrgID, setNewOrgID] = useState("");
  const [newOrgName, setNewOrgName] = useState("");
  const [creatingOrganization, setCreatingOrganization] = useState(false);
  const [createOrganizationError, setCreateOrganizationError] = useState<string | null>(null);
  const [creatingUser, setCreatingUser] = useState(false);
  const [createUserError, setCreateUserError] = useState<string | null>(null);
  const usageSeries = useMemo(
    () =>
      (overview?.usage_last_24h ?? []).map((row) => ({
        ...row,
        bucket_label: formatClock(row.bucket_start),
      })),
    [overview]
  );

  const submitNewUser = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const email = newUserEmail.trim();
    const displayName = newUserName.trim();
    if (!email) {
      setCreateUserError("Email is required.");
      return;
    }
    setCreatingUser(true);
    setCreateUserError(null);
    try {
      await onCreateUser({
        email,
        display_name: displayName,
        role: newUserRole.trim() || "researcher",
        org_id: newUserOrg.trim() || "local-org",
      });
      setNewUserEmail("");
      setNewUserName("");
      setNewUserRole("researcher");
      setNewUserOrg("local-org");
    } catch (error) {
      setCreateUserError(error instanceof Error ? error.message : "Unable to create user.");
    } finally {
      setCreatingUser(false);
    }
  };
  const submitNewOrganization = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const orgID = newOrgID.trim();
    const name = newOrgName.trim();
    if (!orgID && !name) {
      setCreateOrganizationError("Organization name or ID is required.");
      return;
    }
    setCreatingOrganization(true);
    setCreateOrganizationError(null);
    try {
      await onCreateOrganization({
        org_id: orgID,
        name,
        status: "active",
      });
      setNewOrgID("");
      setNewOrgName("");
    } catch (error) {
      setCreateOrganizationError(
        error instanceof Error ? error.message : "Unable to create organization."
      );
    } finally {
      setCreatingOrganization(false);
    }
  };
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
  const queue = overview?.queue;
  const activeUserShare =
    kpis && kpis.total_users > 0 ? (kpis.active_users_24h / kpis.total_users) * 100 : 0;
  const messagesPerActiveUser =
    kpis && kpis.active_users_24h > 0 ? kpis.messages_last_24h / kpis.active_users_24h : 0;
  const runtime = overview?.runtime;
  const activityByLabel = useMemo(() => {
    const periods = new Map<string, AdminActivityPeriod>();
    for (const period of overview?.activity ?? []) {
      periods.set(period.label, period);
    }
    return periods;
  }, [overview]);
  const monthlyActivity = activityByLabel.get("Monthly");
  const organizationOptions = useMemo(() => {
    const seen = new Set<string>();
    const records = organizations.filter((org) => {
      const id = org.org_id.trim();
      if (!id || seen.has(id)) {
        return false;
      }
      seen.add(id);
      return true;
    });
    if (!seen.has("local-org")) {
      records.unshift({
        org_id: "local-org",
        name: "Local Organization",
        status: "active",
        created_at: "",
        updated_at: "",
        metadata: {},
      });
    }
    return records;
  }, [organizations]);
  const userDirectoryStats = useMemo(() => {
    const orgIDs = new Set<string>();
    let active = 0;
    let disabled = 0;
    let pending = 0;
    let admin = 0;
    for (const org of organizationOptions) {
      if (org.org_id) {
        orgIDs.add(org.org_id);
      }
    }
    for (const user of users) {
      const status = String(user.status || "observed").toLowerCase();
      const role = String(user.role || "observed").toLowerCase();
      if (status === "active" || status === "observed") {
        active += 1;
      }
      if (status === "disabled") {
        disabled += 1;
      }
      if (status === "pending") {
        pending += 1;
      }
      if (role === "admin") {
        admin += 1;
      }
      if (user.org_id) {
        orgIDs.add(user.org_id);
      }
    }
    return {
      active,
      admin,
      disabled,
      pending,
      organizations: orgIDs.size,
      visibleUsers: users.length,
    };
  }, [organizationOptions, users]);

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

        <Tabs
          value={activePage}
          onValueChange={(value) => setActivePage(value as AdminConsolePage)}
          className="admin-page-tabs"
        >
          <TabsList className="admin-page-tabs-list">
            <TabsTrigger
              value="overview"
              className="admin-page-tab"
              onClick={() => setActivePage("overview")}
            >
              <LayoutDashboard className="size-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="users"
              className="admin-page-tab"
              onClick={() => setActivePage("users")}
            >
              <Users className="size-4" />
              Users Management
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="admin-page-panel">
        <Card className="admin-runtime-card">
          <CardHeader>
            <div className="admin-card-title-row">
              <div>
                <CardTitle>Runtime</CardTitle>
                <CardDescription>Control-plane storage, worker dispatch, and NATS subjects.</CardDescription>
              </div>
              <Badge variant="outline" className={runtime?.nats_configured ? "admin-runtime-good" : "admin-runtime-warn"}>
                {runtime?.nats_configured ? "NATS JetStream" : "Local stub"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="admin-runtime-grid">
            <div>
              <span>Dispatch</span>
              <strong>{loadingOverview ? "…" : runtime?.dispatch_mode ?? "unknown"}</strong>
            </div>
            <div>
              <span>Worker</span>
              <strong>
                {loadingOverview
                  ? "…"
                  : runtime?.stub_worker_enabled
                    ? "stub worker"
                    : "external workers"}
              </strong>
            </div>
            <div>
              <span>Store</span>
              <strong>{loadingOverview ? "…" : runtime?.store_backend ?? "unknown"}</strong>
            </div>
            <div>
              <span>Jobs</span>
              <strong>{loadingOverview ? "…" : runtime?.nats_jobs_subject ?? runtime?.job_transport ?? "unknown"}</strong>
            </div>
            <div>
              <span>Events</span>
              <strong>{loadingOverview ? "…" : runtime?.nats_events_subject ?? runtime?.event_transport ?? "unknown"}</strong>
            </div>
            <div>
              <span>Cancel</span>
              <strong>{loadingOverview ? "…" : runtime?.nats_cancel_subject ?? "not configured"}</strong>
            </div>
            <div>
              <span>Recovery</span>
              <strong>
                {loadingOverview
                  ? "…"
                  : runtime?.run_recovery_enabled
                    ? `auto every ${formatDuration(runtime.run_recovery_interval_seconds)}`
                    : "disabled"}
              </strong>
            </div>
          </CardContent>
        </Card>

        <Card className="admin-runtime-card admin-activity-card">
          <CardHeader>
            <div className="admin-card-title-row">
              <div>
                <CardTitle>Activity</CardTitle>
                <CardDescription>Usage across prompts, tool work, users, runs, and outputs.</CardDescription>
              </div>
              <Badge variant="outline" className="admin-runtime-good">
                {loadingOverview ? "…" : `${formatCount(monthlyActivity?.active_users ?? 0)} monthly active`}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="admin-activity-panel">
            <div className="admin-activity-grid" role="table" aria-label="Platform activity metrics">
              <div className="admin-activity-header admin-activity-metric" role="columnheader">
                Metric
              </div>
              {activityPeriodOrder.map((label) => {
                const period = activityByLabel.get(label);
                return (
                  <div key={label} className="admin-activity-header" role="columnheader">
                    <strong>{label}</strong>
                    <span>{period?.window ?? "—"}</span>
                  </div>
                );
              })}
              {activityMetricRows.map((row) => (
                <Fragment key={row.label}>
                  <div className="admin-activity-metric" role="rowheader">
                    {row.label}
                  </div>
                  {activityPeriodOrder.map((label) => {
                    const period = activityByLabel.get(label);
                    return (
                      <div key={`${row.label}-${label}`} className="admin-activity-value" role="cell">
                        {loadingOverview || !period ? "…" : formatCount(row.value(period))}
                      </div>
                    );
                  })}
                </Fragment>
              ))}
            </div>
            <p className="admin-kpi-footnote">
              Tool calls count started invocations; failed runs and artifacts help track research throughput.
            </p>
          </CardContent>
        </Card>

        <Card className="admin-runtime-card">
          <CardHeader>
            <div className="admin-card-title-row">
              <div>
                <CardTitle>Queue Health</CardTitle>
                <CardDescription>JetStream stream depth, worker consumers, and redelivery pressure.</CardDescription>
              </div>
              <Badge variant="outline" className={queue?.available ? "admin-runtime-good" : "admin-runtime-warn"}>
                {queue?.available ? "Connected" : "Unavailable"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="admin-queue-panel">
            <div className="admin-runtime-grid">
              <div>
                <span>Stream</span>
                <strong>{loadingOverview ? "…" : queue?.stream ?? "not configured"}</strong>
              </div>
              <div>
                <span>Messages</span>
                <strong>{loadingOverview ? "…" : formatCount(queue?.stream_messages ?? 0)}</strong>
              </div>
              <div>
                <span>Consumers</span>
                <strong>{loadingOverview ? "…" : formatCount(queue?.consumer_count ?? 0)}</strong>
              </div>
              <div>
                <span>Last sequence</span>
                <strong>{loadingOverview ? "…" : formatCount(queue?.last_sequence ?? 0)}</strong>
              </div>
            </div>
            {!loadingOverview && queue?.error ? (
              <p className="admin-queue-error">{queue.error}</p>
            ) : null}
            {!loadingOverview && queue?.consumers?.length ? (
              <div className="admin-queue-consumer-list">
                {queue.consumers.map((consumer) => (
                  <article key={`${consumer.role || "consumer"}-${consumer.name}`} className="admin-queue-consumer">
                    <div className="admin-queue-consumer-heading">
                      <div>
                        <strong>{consumer.name}</strong>
                        <span>{consumer.role || "consumer"} • {consumer.subject || "subject unknown"}</span>
                      </div>
                      <Badge variant="outline" className={consumer.active ? "admin-runtime-good" : "admin-runtime-warn"}>
                        {consumer.active ? "active" : "missing"}
                      </Badge>
                    </div>
                    <p>
                      {formatQueueConsumerSummary(
                        consumer.pending_messages,
                        consumer.in_flight_messages,
                        consumer.redelivered_messages
                      )}
                    </p>
                    {consumer.error ? <p className="admin-queue-error">{consumer.error}</p> : null}
                  </article>
                ))}
              </div>
            ) : null}
          </CardContent>
        </Card>

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
              <CardDescription>Stale runs</CardDescription>
              <CardTitle className="admin-kpi-value">
                {loadingOverview ? "…" : formatCount(kpis?.stale_running_runs ?? 0)}
              </CardTitle>
            </CardHeader>
            <CardContent className="admin-kpi-stack">
              <p className="admin-kpi-footnote">
                Running with no worker event for 5m+.
              </p>
              <p className="admin-kpi-footnote">
                Use run events before canceling compute.
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

        <div className="admin-data-grid admin-data-grid-single">
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
                    const isRecoverable = isRecoverableRunStatus(row.status);
                    const canRequeue = row.stale && isRecoverable && Boolean(onRequeueRun);
                    const conversationKey = `${row.user_id || ""}:${row.conversation_id || ""}`;
                    const eventsOpen = activeRunEventRunId === row.run_id;
                    const runEvents = runEventsById[row.run_id] ?? [];
                    const runEventsLoading = Boolean(runEventsLoadingById[row.run_id]);
                    const runLeaseSummary = formatRunLeaseSummary(row);
                    return (
                      <Fragment key={row.run_id}>
                        <tr>
                          <td>
                            <div className="admin-run-id-cell">
                              <span className="font-mono text-xs">{row.run_id}</span>
                              <span className="admin-goal-preview" title={row.goal}>
                                {row.goal}
                              </span>
                              {row.stale && row.stale_reason ? (
                                <span className="admin-run-stale-reason">{row.stale_reason}</span>
                              ) : null}
                            </div>
                          </td>
                          <td>
                            <div className="admin-run-status-cell">
                              <div className="admin-run-status-badges">
                                <Badge variant="outline" className={runStatusBadgeClass(row.status)}>
                                  {row.status}
                                </Badge>
                                {row.stale ? (
                                  <Badge variant="outline" className="admin-stale-badge">
                                    Stale
                                  </Badge>
                                ) : null}
                              </div>
                              <div className="admin-run-event-meta">
                                <span>
                                  {row.last_event_kind
                                    ? `${row.last_event_kind}${row.last_event_sequence ? ` #${row.last_event_sequence}` : ""}`
                                    : row.event_count > 0
                                      ? `${row.event_count} events`
                                      : "No worker events"}
                                </span>
                                {row.last_activity_age_seconds !== null &&
                                row.last_activity_age_seconds !== undefined ? (
                                  <span>{formatSilentAge(row.last_activity_age_seconds)}</span>
                                ) : null}
                                <span>{formatRunEventSummary(row)}</span>
                                {runLeaseSummary ? <span>{runLeaseSummary}</span> : null}
                                {row.last_tool_name ? <span>Last tool: {row.last_tool_name}</span> : null}
                              </div>
                            </div>
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
                                onClick={() => onInspectRunEvents(row.run_id)}
                              >
                                Events
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                disabled={!isRecoverable || Boolean(runCancellingById[row.run_id])}
                                onClick={() => onCancelRun(row.run_id)}
                              >
                                <StopCircle className="size-4" />
                                {runCancellingById[row.run_id] ? "Canceling…" : "Cancel"}
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                disabled={!canRequeue || Boolean(runRequeueingById[row.run_id])}
                                onClick={() => onRequeueRun?.(row.run_id)}
                              >
                                <RotateCcw className="size-4" />
                                {runRequeueingById[row.run_id] ? "Requeueing…" : "Requeue"}
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
                        {eventsOpen ? (
                          <tr className="admin-run-events-row">
                            <td colSpan={7}>
                              <div className="admin-run-events-panel">
                                <div className="admin-run-events-header">
                                  <span>Run Event Timeline</span>
                                  <span>{runEventsLoading ? "Loading events…" : `${runEvents.length} events`}</span>
                                </div>
                                {runEventsLoading ? <p className="admin-empty">Loading run events…</p> : null}
                                {!runEventsLoading && runEvents.length === 0 ? (
                                  <p className="admin-empty">No run events recorded yet.</p>
                                ) : null}
                                {!runEventsLoading && runEvents.length > 0 ? (
                                  <ol className="admin-run-events-list">
                                    {runEvents.slice(-12).map((event, index) => (
                                      <li key={`${row.run_id}-${runEventLabel(event)}-${index}`}>
                                        <div className="admin-run-event-line">
                                          <span className="admin-run-event-kind">{runEventLabel(event)}</span>
                                          <span className="admin-run-event-time">{formatDateTime(event.ts)}</span>
                                        </div>
                                        <p>{runEventDetail(event)}</p>
                                      </li>
                                    ))}
                                  </ol>
                                ) : null}
                              </div>
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
          </CardContent>
        </Card>
          </TabsContent>

          <TabsContent value="users" className="admin-page-panel">
            <Card className="admin-users-hero-card">
              <CardHeader>
                <div className="admin-card-title-row">
                  <div>
                    <p className="admin-section-label">Admin / WorkOS</p>
                    <h3 className="admin-subpage-title">Users Management</h3>
                    <CardDescription>
                      WorkOS AuthKit signs users in. This page maintains Ultra roles,
                      organization mapping, and disabled-state controls for authenticated identities.
                    </CardDescription>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="admin-refresh-button"
                    onClick={() => {
                      onRefreshOrganizations();
                      onRefreshUsers();
                    }}
                    disabled={loadingOrganizations || loadingUsers}
                  >
                    <RefreshCw
                      className={cn(
                        "size-4",
                        (loadingOrganizations || loadingUsers) && "animate-spin"
                      )}
                    />
                    Refresh directory
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="admin-user-stat-grid">
                <div className="admin-user-stat">
                  <span>Visible users</span>
                  <strong>{loadingUsers ? "…" : formatCount(userDirectoryStats.visibleUsers)}</strong>
                </div>
                <div className="admin-user-stat">
                  <span>Active or observed</span>
                  <strong>{loadingUsers ? "…" : formatCount(userDirectoryStats.active)}</strong>
                </div>
                <div className="admin-user-stat">
                  <span>Admins</span>
                  <strong>{loadingUsers ? "…" : formatCount(userDirectoryStats.admin)}</strong>
                </div>
                <div className="admin-user-stat">
                  <span>Pending review</span>
                  <strong>{loadingUsers ? "…" : formatCount(userDirectoryStats.pending)}</strong>
                </div>
                <div className="admin-user-stat">
                  <span>Organizations</span>
                  <strong>{loadingOrganizations ? "…" : formatCount(userDirectoryStats.organizations)}</strong>
                </div>
              </CardContent>
            </Card>

            <div className="admin-users-management-grid">
              <Card className="admin-users-directory-card">
                <CardHeader>
                  <div className="admin-card-title-row">
                    <div className="flex items-center gap-2">
                      <Users className="size-4" />
                      <CardTitle>Account Directory</CardTitle>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      aria-label="Refresh users"
                      onClick={onRefreshUsers}
                    >
                      <RefreshCw className={cn("size-4", loadingUsers && "animate-spin")} />
                    </Button>
                  </div>
                  <CardDescription>
                    Search users by identity, role, status, organization, or local account id.
                  </CardDescription>
                  <Input
                    value={userQuery}
                    onChange={(event) => onUserQueryChange(event.target.value)}
                    placeholder="Filter by user, email, role, or org"
                    aria-label="Filter users"
                    className="admin-filter-input"
                  />
                </CardHeader>
                <CardContent className="admin-table-wrap">
                  {loadingUsers ? <p className="admin-empty">Loading users…</p> : null}
                  {!loadingUsers && users.length === 0 ? (
                    <p className="admin-empty">No users matched the current filter.</p>
                  ) : null}
                  {!loadingUsers && users.length > 0 ? (
                    <table className="admin-table admin-users-table">
                      <thead>
                        <tr>
                          <th>User</th>
                          <th>WorkOS identity</th>
                          <th>Role</th>
                          <th>Organization</th>
                          <th>Status</th>
                          <th>Activity</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {users.map((row) => {
                          const normalizedStatus = String(row.status || "observed").toLowerCase();
                          const pending = normalizedStatus === "pending";
                          const disabled = normalizedStatus === "disabled";
                          const rejected = normalizedStatus === "rejected";
                          const deleting = Boolean(userDeletingById[row.user_id]);
                          const updating = Boolean(userUpdatingById[row.user_id]);
                          return (
                            <tr key={row.user_id}>
                              <td>
                                <div className="admin-user-cell">
                                  <span className="admin-user-name">
                                    {row.display_name || row.user_id}
                                  </span>
                                  <span className="font-mono text-xs">{row.user_id}</span>
                                </div>
                              </td>
                              <td>
                                <span className="admin-user-email">
                                  {row.email || "Email unavailable"}
                                </span>
                              </td>
                              <td>
                                <Badge variant="outline">{row.role || "observed"}</Badge>
                              </td>
                              <td className="font-mono text-xs">{row.org_id || "local-org"}</td>
                              <td>
                                <Badge
                                  variant="outline"
                                  className={userStatusBadgeClass(row.status || "observed")}
                                >
                                  {row.status || "observed"}
                                </Badge>
                              </td>
                              <td>
                                <div className="admin-user-activity">
                                  <span>{formatCount(row.runs_total)} runs</span>
                                  <span>{formatCount(row.runs_failed)} failed</span>
                                  <span>{formatCount(row.uploads)} uploads</span>
                                  <span>{formatBytes(row.storage_bytes)}</span>
                                  <span>{formatDateTime(row.last_activity_at)}</span>
                                </div>
                              </td>
                              <td>
                                <div className="admin-user-actions">
                                  {pending && onUpdateUserStatus ? (
                                    <>
                                      <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        aria-label={`Approve user ${row.user_id}`}
                                        disabled={updating}
                                        onClick={() => onUpdateUserStatus(row.user_id, "active")}
                                      >
                                        <CheckCircle2 className="size-4" />
                                        {updating ? "Updating…" : "Approve"}
                                      </Button>
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        aria-label={`Reject user ${row.user_id}`}
                                        disabled={updating}
                                        onClick={() => onUpdateUserStatus(row.user_id, "rejected")}
                                      >
                                        <StopCircle className="size-4" />
                                        Reject
                                      </Button>
                                    </>
                                  ) : null}
                                  {(disabled || rejected) && onUpdateUserStatus ? (
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      aria-label={`Re-enable user ${row.user_id}`}
                                      disabled={updating}
                                      onClick={() => onUpdateUserStatus(row.user_id, "active")}
                                    >
                                      <CheckCircle2 className="size-4" />
                                      {updating ? "Updating…" : "Re-enable"}
                                    </Button>
                                  ) : null}
                                  {!pending && !disabled && !rejected ? (
                                    <Button
                                      type="button"
                                      variant="ghost"
                                      size="sm"
                                      aria-label={`Deactivate user ${row.user_id}`}
                                      disabled={deleting || updating}
                                      onClick={() => onDeleteUser(row.user_id)}
                                    >
                                      <Trash2 className="size-4" />
                                      {deleting ? "Deactivating…" : "Deactivate"}
                                    </Button>
                                  ) : null}
                                  {pending && !onUpdateUserStatus ? (
                                    <span className="admin-user-action-note">Awaiting review</span>
                                  ) : null}
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

              <div className="admin-users-side-panel">
                <Card className="admin-management-form-card">
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <UserPlus className="size-4" />
                      <CardTitle>Create User</CardTitle>
                    </div>
                    <CardDescription>
                      Add an Ultra account record for a WorkOS-authenticated identity.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form className="admin-create-user-form" onSubmit={submitNewUser}>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-user-email">Email</Label>
                        <Input
                          id="admin-new-user-email"
                          value={newUserEmail}
                          onChange={(event) => setNewUserEmail(event.target.value)}
                          placeholder="email@institution.edu"
                          aria-label="New user email"
                        />
                      </div>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-user-name">Display name</Label>
                        <Input
                          id="admin-new-user-name"
                          value={newUserName}
                          onChange={(event) => setNewUserName(event.target.value)}
                          placeholder="Jane Researcher"
                          aria-label="New user display name"
                        />
                      </div>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-user-role">Role</Label>
                        <Select
                          value={newUserRole}
                          onValueChange={(value) => setNewUserRole(value)}
                        >
                          <SelectTrigger
                            id="admin-new-user-role"
                            aria-label="New user role"
                            className="admin-select-trigger"
                          >
                            <SelectValue placeholder="Select role" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectItem value="researcher">Researcher</SelectItem>
                              <SelectItem value="admin">Admin</SelectItem>
                              <SelectItem value="observer">Observer</SelectItem>
                              <SelectItem value="observed">Observed</SelectItem>
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-user-org">Organization</Label>
                        <Select
                          value={newUserOrg}
                          onValueChange={(value) => setNewUserOrg(value)}
                        >
                          <SelectTrigger
                            id="admin-new-user-org"
                            aria-label="New user org"
                            className="admin-select-trigger"
                          >
                            <SelectValue placeholder="Select organization" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              {organizationOptions.map((org) => (
                                <SelectItem key={org.org_id} value={org.org_id}>
                                  {org.name ? `${org.name} (${org.org_id})` : org.org_id}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                      <Button type="submit" size="sm" disabled={creatingUser || loadingUsers}>
                        <UserPlus className="size-4" />
                        {creatingUser ? "Creating…" : "Create user"}
                      </Button>
                    </form>
                    {createUserError ? <p className="admin-error-text">{createUserError}</p> : null}
                  </CardContent>
                </Card>

                <Card className="admin-management-form-card">
                  <CardHeader>
                    <div className="admin-card-title-row">
                      <div className="flex items-center gap-2">
                        <Building2 className="size-4" />
                        <CardTitle>Organizations</CardTitle>
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        aria-label="Refresh organizations"
                        onClick={onRefreshOrganizations}
                      >
                        <RefreshCw className={cn("size-4", loadingOrganizations && "animate-spin")} />
                      </Button>
                    </div>
                    <CardDescription>
                      Maintain the organization ids used for WorkOS-backed role scoping.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="admin-org-management-content">
                    <form className="admin-create-org-form" onSubmit={submitNewOrganization}>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-org-id">Organization id</Label>
                        <Input
                          id="admin-new-org-id"
                          value={newOrgID}
                          onChange={(event) => setNewOrgID(event.target.value)}
                          placeholder="org-id"
                          aria-label="New org id"
                        />
                      </div>
                      <div className="admin-form-field">
                        <Label htmlFor="admin-new-org-name">Name</Label>
                        <Input
                          id="admin-new-org-name"
                          value={newOrgName}
                          onChange={(event) => setNewOrgName(event.target.value)}
                          placeholder="Organization name"
                          aria-label="New org name"
                        />
                      </div>
                      <Button
                        type="submit"
                        size="sm"
                        disabled={creatingOrganization || loadingOrganizations}
                      >
                        <Building2 className="size-4" />
                        {creatingOrganization ? "Creating…" : "Create org"}
                      </Button>
                    </form>
                    {createOrganizationError ? (
                      <p className="admin-error-text">{createOrganizationError}</p>
                    ) : null}
                    <div className="admin-org-list" aria-label="Organizations">
                      {organizationOptions.map((org) => (
                        <article key={org.org_id} className="admin-org-row">
                          <div>
                            <strong>{org.name || org.org_id}</strong>
                            <span>{org.org_id}</span>
                            <span>
                              {formatCount(org.uploads ?? 0)} uploads · {formatBytes(org.storage_bytes ?? 0)}
                            </span>
                          </div>
                          <Badge variant="outline" className={userStatusBadgeClass(org.status || "active")}>
                            {org.status || "active"}
                          </Badge>
                        </article>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
