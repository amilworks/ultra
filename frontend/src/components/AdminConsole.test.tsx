import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AdminConsole } from "./AdminConsole";
import type { AdminOverviewResponse, AdminRunRecord, RunEvent } from "../types";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock as typeof ResizeObserver;

const emptyOverview: AdminOverviewResponse = {
  generated_at: "2026-06-01T00:00:00Z",
  runtime: {
    app_version: "test-version",
    store_backend: "memory",
    dispatch_mode: "nats_jetstream",
    job_transport: "nats_jetstream",
    event_transport: "nats_jetstream_to_local_fanout",
    stub_worker_enabled: false,
    nats_configured: true,
    nats_stream: "ULTRA_RUNS",
    nats_jobs_subject: "ultra.runs.jobs",
    nats_rarespot_jobs_subject: "ultra.runs.rarespot.jobs",
    nats_events_subject: "ultra.runs.events",
    nats_cancel_subject: "ultra.runs.cancel",
    nats_event_consumer: "ultra-control-event-ingest",
    artifact_root: "data/artifacts",
    upload_root: "data/uploads",
    run_recovery_enabled: true,
    run_recovery_interval_seconds: 30,
    run_recovery_batch_limit: 1000,
  },
  queue: {
    available: true,
    mode: "nats_jetstream",
    stream: "ULTRA_RUNS",
    stream_subjects: ["ultra.runs.jobs", "ultra.runs.events", "ultra.runs.cancel"],
    stream_messages: 42,
    stream_bytes: 4096,
    first_sequence: 1,
    last_sequence: 42,
    consumer_count: 2,
    consumers: [
      {
        name: "ultra-deepagents-worker",
        role: "deepagents",
        subject: "ultra.runs.jobs",
        active: true,
        ack_wait_seconds: 600,
        max_deliver: 5,
        pending_messages: 3,
        in_flight_messages: 1,
        redelivered_messages: 2,
        waiting_pull_requests: 1,
      },
    ],
  },
  kpis: {
    total_users: 1,
    active_users_24h: 1,
    total_conversations: 1,
    conversations_started_24h: 1,
    total_messages: 0,
    messages_last_24h: 0,
    user_messages_last_24h: 0,
    assistant_messages_last_24h: 0,
    total_runs: 1,
    runs_last_24h: 1,
    success_rate_last_24h: 0,
    running_runs: 1,
    stale_running_runs: 1,
    failed_runs_24h: 0,
    total_uploads: 0,
    soft_deleted_uploads: 0,
    total_storage_bytes: 0,
    avg_messages_per_conversation: 0,
  },
  usage_last_24h: [],
  tool_usage_7d: [],
  workers: [],
  top_users: [],
  recent_issues: [],
};

describe("AdminConsole", () => {
  it("surfaces stale worker-event diagnostics for long autonomous runs", () => {
    const staleRun: AdminRunRecord = {
      run_id: "run_stale",
      user_id: "local-user",
      conversation_id: "thread_stale",
      goal: "Train a UNet model and return weights",
      status: "running",
      created_at: "2026-06-01T00:00:00Z",
      updated_at: "2026-06-01T00:10:00Z",
      duration_seconds: null,
      tool_names: ["execute"],
      last_event_kind: "run.heartbeat",
      last_event_at: "2026-06-01T00:10:00Z",
      last_event_sequence: 42,
      last_activity_age_seconds: 1210,
      event_count: 42,
      message_delta_count: 1,
      tool_call_count: 1,
      artifact_count: 1,
      heartbeat_count: 1,
      last_tool_name: "execute",
      last_tool_at: "2026-06-01T00:09:30Z",
      first_delta_latency_seconds: 15,
      first_tool_latency_seconds: 9,
      first_artifact_latency_seconds: 22,
      lease_worker_id: "ultra-deepagents-worker",
      lease_expires_at: "2026-06-01T00:05:00Z",
      lease_active: false,
      lease_expired: true,
      lease_seconds_remaining: 0,
      lease_last_renewed_at: "2026-06-01T00:00:00Z",
      lease_last_renewed_age_seconds: 1800,
      stale: true,
      stale_reason: "No worker event for 20.2m; latest event was run.heartbeat.",
    };
    const events: RunEvent[] = [
      {
        event_type: "tool_call.started",
        ts: "2026-06-01T00:09:30Z",
        payload: {
          sequence: 41,
          tool_name: "execute",
          message: "Execute training script",
        },
      },
      {
        event_type: "run.heartbeat",
        ts: "2026-06-01T00:10:00Z",
        payload: {
          sequence: 42,
          stage: "silent_compute",
        },
      },
    ];
    const onInspectRunEvents = vi.fn();
    const onRequeueRun = vi.fn();

    render(
      <AdminConsole
        overview={emptyOverview}
        organizations={[]}
        users={[]}
        runs={[staleRun]}
        issues={[]}
        loadingOverview={false}
        loadingOrganizations={false}
        loadingUsers={false}
        loadingRuns={false}
        loadingIssues={false}
        error={null}
        runCancellingById={{}}
        runRequeueingById={{}}
        deletingConversationKey={null}
        runStatusFilter="running"
        runQuery=""
        userQuery=""
        activeRunEventRunId="run_stale"
        runEventsById={{ run_stale: events }}
        runEventsLoadingById={{}}
        onRunStatusFilterChange={vi.fn()}
        onRunQueryChange={vi.fn()}
        onUserQueryChange={vi.fn()}
        onRefreshAll={vi.fn()}
        onRefreshOrganizations={vi.fn()}
        onRefreshUsers={vi.fn()}
        onRefreshRuns={vi.fn()}
        onRefreshIssues={vi.fn()}
        onCreateOrganization={vi.fn()}
        onCreateUser={vi.fn()}
        onDeleteUser={vi.fn()}
        onCancelRun={vi.fn()}
        onRequeueRun={onRequeueRun}
        onDeleteConversation={vi.fn()}
        onInspectRunEvents={onInspectRunEvents}
      />
    );

    const staleRunsCard = screen.getByText("Stale runs").closest("[data-slot='card']");
    expect(staleRunsCard).not.toBeNull();
    expect(within(staleRunsCard as HTMLElement).getByText("1")).toBeInTheDocument();
    expect(screen.getByText("Runtime")).toBeInTheDocument();
    expect(screen.getByText("Recovery")).toBeInTheDocument();
    expect(screen.getByText("auto every 30s")).toBeInTheDocument();
    expect(screen.getByText("NATS JetStream")).toBeInTheDocument();
    expect(screen.getByText("ultra.runs.jobs")).toBeInTheDocument();
    expect(screen.getByText("external workers")).toBeInTheDocument();
    expect(screen.getByText("Workers")).toBeInTheDocument();
    expect(
      screen.getByText(
        "No app-level worker heartbeats yet. 1 active NATS worker consumer is ready to pull jobs."
      )
    ).toBeInTheDocument();
    expect(screen.getByText("Queue Health")).toBeInTheDocument();
    expect(screen.getByText("ULTRA_RUNS")).toBeInTheDocument();
    expect(screen.getByText("ultra-deepagents-worker")).toBeInTheDocument();
    expect(screen.getByText("3 pending • 1 in-flight • 2 redelivered")).toBeInTheDocument();
    expect(screen.getByText("Stale")).toBeInTheDocument();
    expect(screen.getByText("No worker event for 20.2m; latest event was run.heartbeat.")).toBeInTheDocument();
    expect(screen.getAllByText("run.heartbeat #42").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("20.2m silent")).toBeInTheDocument();
    expect(screen.getByText("1 delta • 1 tool • 1 artifact • 1 heartbeat")).toBeInTheDocument();
    expect(screen.getByText("Lease expired: ultra-deepagents-worker")).toBeInTheDocument();
    expect(screen.getByText("Last tool: execute")).toBeInTheDocument();
    const timeline = screen.getByText("Run Event Timeline").closest(".admin-run-events-panel");
    expect(timeline).not.toBeNull();
    expect(within(timeline as HTMLElement).getByText("tool_call.started #41")).toBeInTheDocument();
    expect(within(timeline as HTMLElement).getByText("run.heartbeat #42")).toBeInTheDocument();
    expect(within(timeline as HTMLElement).getByText("Execute training script")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Events" }));
    expect(onInspectRunEvents).toHaveBeenCalledWith("run_stale");
    fireEvent.click(screen.getByRole("button", { name: /Requeue/i }));
    expect(onRequeueRun).toHaveBeenCalledWith("run_stale");
  });

  it("renders durable worker heartbeat records", () => {
    render(
      <AdminConsole
        overview={{
          ...emptyOverview,
          workers: [
            {
              worker_id: "ultra-deepagents-worker@host:123",
              worker_kind: "deepagents",
              status: "busy",
              current_run_id: "run_active",
              hostname: "host",
              version: "0.1.0",
              started_at: "2026-06-01T00:00:00Z",
              last_heartbeat_at: "2026-06-01T00:02:00Z",
              updated_at: "2026-06-01T00:02:00Z",
              heartbeat_age_seconds: 12,
              active: true,
              stale: false,
              metadata: { active_tasks: 1 },
            },
          ],
        }}
        organizations={[]}
        users={[]}
        runs={[]}
        issues={[]}
        loadingOverview={false}
        loadingOrganizations={false}
        loadingUsers={false}
        loadingRuns={false}
        loadingIssues={false}
        error={null}
        runCancellingById={{}}
        deletingConversationKey={null}
        runStatusFilter="running"
        runQuery=""
        userQuery=""
        activeRunEventRunId={null}
        runEventsById={{}}
        runEventsLoadingById={{}}
        onRunStatusFilterChange={vi.fn()}
        onRunQueryChange={vi.fn()}
        onUserQueryChange={vi.fn()}
        onRefreshAll={vi.fn()}
        onRefreshOrganizations={vi.fn()}
        onRefreshUsers={vi.fn()}
        onRefreshRuns={vi.fn()}
        onRefreshIssues={vi.fn()}
        onCreateOrganization={vi.fn()}
        onCreateUser={vi.fn()}
        onDeleteUser={vi.fn()}
        onCancelRun={vi.fn()}
        onDeleteConversation={vi.fn()}
        onInspectRunEvents={vi.fn()}
      />
    );

    expect(screen.getByText("ultra-deepagents-worker@host:123")).toBeInTheDocument();
    expect(screen.getByText("deepagents • host")).toBeInTheDocument();
    expect(screen.getByText("busy")).toBeInTheDocument();
    expect(screen.getByText("Running run_active • heartbeat 12s ago")).toBeInTheDocument();
    expect(screen.getByText("Version: 0.1.0")).toBeInTheDocument();
  });

  it("submits admin-created user accounts", async () => {
    const onCreateUser = vi.fn().mockResolvedValue(undefined);

    render(
      <AdminConsole
        overview={emptyOverview}
        organizations={[
          {
            org_id: "local-org",
            name: "Local Organization",
            status: "active",
            created_at: "2026-06-01T00:00:00Z",
            updated_at: "2026-06-01T00:00:00Z",
            metadata: {},
          },
          {
            org_id: "smithsonian",
            name: "Smithsonian",
            status: "active",
            created_at: "2026-06-01T00:00:00Z",
            updated_at: "2026-06-01T00:00:00Z",
            metadata: {},
          },
        ]}
        users={[
          {
            user_id: "user_existing",
            email: "existing@example.org",
            display_name: "Existing Researcher",
            role: "researcher",
            status: "active",
            org_id: "local-org",
            conversations: 0,
            messages: 0,
            runs_total: 0,
            runs_running: 0,
            runs_failed: 0,
            runs_succeeded: 0,
            uploads: 0,
            storage_bytes: 0,
          },
        ]}
        runs={[]}
        issues={[]}
        loadingOverview={false}
        loadingOrganizations={false}
        loadingUsers={false}
        loadingRuns={false}
        loadingIssues={false}
        error={null}
        runCancellingById={{}}
        deletingConversationKey={null}
        runStatusFilter="running"
        runQuery=""
        userQuery=""
        activeRunEventRunId={null}
        runEventsById={{}}
        runEventsLoadingById={{}}
        onRunStatusFilterChange={vi.fn()}
        onRunQueryChange={vi.fn()}
        onUserQueryChange={vi.fn()}
        onRefreshAll={vi.fn()}
        onRefreshOrganizations={vi.fn()}
        onRefreshUsers={vi.fn()}
        onRefreshRuns={vi.fn()}
        onRefreshIssues={vi.fn()}
        onCreateOrganization={vi.fn()}
        onCreateUser={onCreateUser}
        onDeleteUser={vi.fn()}
        onCancelRun={vi.fn()}
        onDeleteConversation={vi.fn()}
        onInspectRunEvents={vi.fn()}
      />
    );

    fireEvent.change(screen.getByLabelText("New user email"), {
      target: { value: "grace@example.org" },
    });
    fireEvent.change(screen.getByLabelText("New user display name"), {
      target: { value: "Grace Hopper" },
    });
    fireEvent.change(screen.getByLabelText("New user role"), {
      target: { value: "admin" },
    });
    fireEvent.change(screen.getByLabelText("New user org"), {
      target: { value: "smithsonian" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create user" }));

    expect(onCreateUser).toHaveBeenCalledWith({
      email: "grace@example.org",
      display_name: "Grace Hopper",
      role: "admin",
      org_id: "smithsonian",
    });
  });

  it("soft-removes admin-managed user accounts", () => {
    const onDeleteUser = vi.fn();

    render(
      <AdminConsole
        overview={emptyOverview}
        organizations={[]}
        users={[
          {
            user_id: "user_existing",
            email: "existing@example.org",
            display_name: "Existing Researcher",
            role: "researcher",
            status: "active",
            org_id: "local-org",
            conversations: 0,
            messages: 0,
            runs_total: 1,
            runs_running: 0,
            runs_failed: 0,
            runs_succeeded: 1,
            uploads: 0,
            storage_bytes: 0,
          },
        ]}
        runs={[]}
        issues={[]}
        loadingOverview={false}
        loadingOrganizations={false}
        loadingUsers={false}
        loadingRuns={false}
        loadingIssues={false}
        error={null}
        runCancellingById={{}}
        deletingConversationKey={null}
        userDeletingById={{}}
        runStatusFilter="running"
        runQuery=""
        userQuery=""
        activeRunEventRunId={null}
        runEventsById={{}}
        runEventsLoadingById={{}}
        onRunStatusFilterChange={vi.fn()}
        onRunQueryChange={vi.fn()}
        onUserQueryChange={vi.fn()}
        onRefreshAll={vi.fn()}
        onRefreshOrganizations={vi.fn()}
        onRefreshUsers={vi.fn()}
        onRefreshRuns={vi.fn()}
        onRefreshIssues={vi.fn()}
        onCreateOrganization={vi.fn()}
        onCreateUser={vi.fn()}
        onDeleteUser={onDeleteUser}
        onCancelRun={vi.fn()}
        onDeleteConversation={vi.fn()}
        onInspectRunEvents={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Deactivate user user_existing" }));

    expect(onDeleteUser).toHaveBeenCalledWith("user_existing");
  });

  it("submits admin-created organizations", async () => {
    const onCreateOrganization = vi.fn().mockResolvedValue(undefined);

    render(
      <AdminConsole
        overview={emptyOverview}
        organizations={[
          {
            org_id: "local-org",
            name: "Local Organization",
            status: "active",
            created_at: "2026-06-01T00:00:00Z",
            updated_at: "2026-06-01T00:00:00Z",
            metadata: {},
          },
        ]}
        users={[]}
        runs={[]}
        issues={[]}
        loadingOverview={false}
        loadingOrganizations={false}
        loadingUsers={false}
        loadingRuns={false}
        loadingIssues={false}
        error={null}
        runCancellingById={{}}
        deletingConversationKey={null}
        runStatusFilter="running"
        runQuery=""
        userQuery=""
        activeRunEventRunId={null}
        runEventsById={{}}
        runEventsLoadingById={{}}
        onRunStatusFilterChange={vi.fn()}
        onRunQueryChange={vi.fn()}
        onUserQueryChange={vi.fn()}
        onRefreshAll={vi.fn()}
        onRefreshOrganizations={vi.fn()}
        onRefreshUsers={vi.fn()}
        onRefreshRuns={vi.fn()}
        onRefreshIssues={vi.fn()}
        onCreateOrganization={onCreateOrganization}
        onCreateUser={vi.fn()}
        onDeleteUser={vi.fn()}
        onCancelRun={vi.fn()}
        onDeleteConversation={vi.fn()}
        onInspectRunEvents={vi.fn()}
      />
    );

    fireEvent.change(screen.getByLabelText("New org id"), {
      target: { value: "allen-institute" },
    });
    fireEvent.change(screen.getByLabelText("New org name"), {
      target: { value: "Allen Institute" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create org" }));

    expect(onCreateOrganization).toHaveBeenCalledWith({
      org_id: "allen-institute",
      name: "Allen Institute",
      status: "active",
    });
  });
});
