import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { UserTokenUsagePanel } from "./UserTokenUsagePanel";
import type { TokenUsageResponse } from "@/types";

const usage: TokenUsageResponse = {
  days: 365,
  summary: {
    lifetime_input_tokens: 400_000,
    lifetime_output_tokens: 445_000,
    lifetime_total_tokens: 845_000,
    peak_daily_total: 845_000,
    longest_task_seconds: 5700,
    current_streak_days: 1,
    longest_streak_days: 1,
    active_days: 1,
    last_active_day: "2026-06-17",
  },
  daily: [
    {
      day: "2026-06-17",
      input_tokens: 400_000,
      output_tokens: 445_000,
      total_tokens: 845_000,
      run_count: 1,
    },
  ],
};

describe("UserTokenUsagePanel", () => {
  it("renders the GitHub-style usage summary and heatmap", () => {
    render(<UserTokenUsagePanel tokenUsage={usage} />);

    expect(screen.getByRole("heading", { name: "Usage" })).toBeInTheDocument();
    expect(screen.getByText("Token activity across all of your runs.")).toBeInTheDocument();
    expect(screen.getByText("Lifetime tokens")).toBeInTheDocument();
    expect(screen.getAllByText("845K")).toHaveLength(2);
    expect(screen.getByText("1h 35m")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Token activity heatmap" })).toBeInTheDocument();
  });
});
