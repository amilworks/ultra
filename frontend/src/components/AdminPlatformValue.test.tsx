import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AdminPlatformValue } from "./AdminPlatformValue";
import type { AdminMetricsResponse } from "../types";

const sampleMetrics: AdminMetricsResponse = {
  generated_at: "2026-06-26T12:00:00Z",
  available: true,
  range_days: 90,
  north_star: {
    label: "Weekly active researchers completing a useful run",
    definition: "A run is useful when it succeeds and produces an output.",
    current_week: 142,
    previous_week: 120,
    delta_pct: 18.3,
    weekly: [
      { week_start: "2026-06-15", value: 120 },
      { week_start: "2026-06-22", value: 142 },
    ],
  },
  kpis: {
    wau: 210,
    mau: 553,
    stickiness_pct: 38,
    new_users: 141,
    activation_rate_pct: 61,
    activation_window_days: 7,
    week4_retention_pct: 47,
    useful_run_rate_pct: 73,
    useful_runs: 1964,
    total_runs: 2690,
  },
  retention_cohorts: {
    unit: "week",
    max_periods: 6,
    cohorts: [
      {
        cohort_start: "2026-05-25",
        size: 41,
        values_pct: [100, 71, null, null, null, null],
        retained: [41, 29, 0, 0, 0, 0],
      },
    ],
  },
  power_user_curve: {
    window_days: 28,
    total_users: 132,
    power_user_threshold: 21,
    power_users: 40,
    power_user_share_pct: 30.3,
    buckets: [
      { days_active: 1, users: 58 },
      { days_active: 21, users: 20 },
      { days_active: 28, users: 20 },
    ],
  },
  activation_funnel: [
    { stage: "Signed up", users: 141, of_previous_pct: null, of_top_pct: 100 },
    { stage: "Started a run", users: 121, of_previous_pct: 85.8, of_top_pct: 85.8 },
    { stage: "Produced an output", users: 98, of_previous_pct: 81, of_top_pct: 69.5 },
    { stage: "Returned and ran again", users: 67, of_previous_pct: 68.4, of_top_pct: 47.5 },
  ],
  cost: {
    currency: "USD",
    priced: false,
    total_tokens: 18_000_000,
    total_cost: null,
    cost_per_useful_run: null,
    tokens_per_useful_run: 9165,
    useful_runs: 1964,
    unpriced_models: ["qwen3.6-27b"],
    by_model: [
      { model: "deepseek_v4", input_tokens: 10_000_000, output_tokens: 2_000_000, total_tokens: 12_000_000, runs: 100, cost: null, priced: false },
    ],
    daily: [{ day: "2026-06-25", total_tokens: 1_800_000, cost: null }],
  },
};

describe("AdminPlatformValue", () => {
  it("renders the value-proving surface from metrics", () => {
    render(
      <AdminPlatformValue
        metrics={sampleMetrics}
        loading={false}
        rangeDays={90}
        onRangeDaysChange={vi.fn()}
      />
    );

    // North star headline value and delta — a meaningful percentage once the
    // baseline is large enough (120 → 142).
    expect(screen.getByText("142")).toBeInTheDocument();
    expect(screen.getByText(/\+18% vs last week/)).toBeInTheDocument();

    // KPI tiles.
    expect(screen.getByText("Activation rate")).toBeInTheDocument();
    expect(screen.getByText("61%")).toBeInTheDocument();
    expect(screen.getByText("Stickiness")).toBeInTheDocument();

    // Retention cohort heatmap, including the not-yet-measurable null cells.
    expect(screen.getByText("Weekly retention by signup cohort")).toBeInTheDocument();
    expect(screen.getAllByText("·").length).toBeGreaterThanOrEqual(1);

    // Power-user curve + funnel.
    expect(screen.getByText("Power-user curve")).toBeInTheDocument();
    expect(screen.getByText("Signed up")).toBeInTheDocument();
    expect(screen.getByText("Returned and ran again")).toBeInTheDocument();

    // Total-tokens KPI (replaces the per-run efficiency tile) shows the
    // aggregate across all users.
    expect(screen.getAllByText("Total tokens").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("18M").length).toBeGreaterThanOrEqual(1);

    // Cost reports tokens (no prices configured) and flags the unpriced model.
    expect(screen.getByText("AI usage and cost")).toBeInTheDocument();
    expect(screen.getByText(/No price configured for: qwen3\.6-27b/)).toBeInTheDocument();
  });

  it("shows an absolute delta (not a noisy percentage) on tiny counts", () => {
    const tiny: AdminMetricsResponse = {
      ...sampleMetrics,
      north_star: {
        ...sampleMetrics.north_star,
        current_week: 2,
        previous_week: 1,
        delta_pct: 100,
      },
      kpis: { ...sampleMetrics.kpis, week4_retention_pct: null },
    };
    render(
      <AdminPlatformValue metrics={tiny} loading={false} rangeDays={90} onRangeDaysChange={vi.fn()} />
    );
    expect(screen.getByText(/\+1 vs last week/)).toBeInTheDocument();
    expect(screen.queryByText(/100% vs/)).not.toBeInTheDocument();
    // Null KPI reads as a calm phrase, not a cryptic dash.
    expect(screen.getByText("No data yet")).toBeInTheDocument();
  });

  it("shows a clear state when value metrics are unavailable", () => {
    const unavailable: AdminMetricsResponse = {
      ...sampleMetrics,
      available: false,
    };
    render(
      <AdminPlatformValue
        metrics={unavailable}
        loading={false}
        rangeDays={90}
        onRangeDaysChange={vi.fn()}
      />
    );
    expect(screen.getByText(/Value metrics need the Postgres store/i)).toBeInTheDocument();
  });
});
