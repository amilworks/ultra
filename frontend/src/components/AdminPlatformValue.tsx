import { Fragment, type ReactNode, useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  Activity,
  Coins,
  Cpu,
  Repeat,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Users,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { AdminMetricCohort, AdminMetricsResponse } from "../types";

type AdminPlatformValueProps = {
  metrics: AdminMetricsResponse | null;
  loading: boolean;
  rangeDays: number;
  onRangeDaysChange: (days: number) => void;
};

const rangeOptions = [
  { value: "30", label: "Last 30 days" },
  { value: "90", label: "Last 90 days" },
  { value: "180", label: "Last 180 days" },
  { value: "365", label: "Last 365 days" },
];

const integerFormatter = new Intl.NumberFormat("en-US");
const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const formatInt = (value: number | null | undefined): string =>
  value == null ? "—" : integerFormatter.format(Math.round(value));

const formatCompact = (value: number | null | undefined): string =>
  value == null ? "—" : compactFormatter.format(value);

const formatPct = (value: number | null | undefined, digits = 0): string =>
  value == null ? "—" : `${value.toFixed(digits)}%`;

const formatCurrency = (value: number | null | undefined, currency: string): string => {
  if (value == null) {
    return "—";
  }
  const digits = Math.abs(value) >= 100 ? 0 : value < 1 ? 3 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency || "USD",
    maximumFractionDigits: digits,
  }).format(value);
};

const formatWeekLabel = (iso: string): string => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
};

const cohortLabel = (iso: string): string => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
};

// cohortCellStyle maps a retention percentage to a fixed blue ramp (mode-stable
// so the heatmap reads the same in light and dark) with a legible text color.
const cohortCellStyle = (pct: number | null): { background: string; color: string } => {
  if (pct == null) {
    return { background: "transparent", color: "var(--muted-foreground)" };
  }
  if (pct >= 90) {
    return { background: "#0c447c", color: "#ffffff" };
  }
  if (pct >= 65) {
    return { background: "#185fa5", color: "#ffffff" };
  }
  if (pct >= 50) {
    return { background: "#378add", color: "#ffffff" };
  }
  if (pct >= 35) {
    return { background: "#85b7eb", color: "#042c53" };
  }
  if (pct >= 20) {
    return { background: "#b5d4f4", color: "#042c53" };
  }
  return { background: "#e6f1fb", color: "#042c53" };
};

const northStarConfig = {
  value: { label: "Researchers", color: "var(--chart-1)" },
};

const powerCurveConfig = {
  users: { label: "Users", color: "var(--chart-1)" },
};

const dailyCostConfig = {
  total_tokens: { label: "Tokens", color: "var(--chart-2)" },
};

type DeltaBadge = { text: string; positive: boolean };

// northStarDelta keeps the headline change readable for non-technical viewers:
// percentages are noisy on tiny counts (1 → 2 reads as "+100%"), so we show the
// absolute change until the baseline is big enough for a percentage to mean
// something.
function northStarDelta(current: number, previous: number, deltaPct: number | null): DeltaBadge | null {
  if (deltaPct == null) {
    return null;
  }
  const diff = current - previous;
  if (diff === 0) {
    return { text: "No change vs last week", positive: true };
  }
  const sign = diff > 0 ? "+" : "−";
  const text =
    previous < 10
      ? `${sign}${Math.abs(diff)} vs last week`
      : `${sign}${Math.abs(Math.round(deltaPct))}% vs last week`;
  return { text, positive: diff > 0 };
}

// pctKpi renders a percentage KPI, falling back to a calm "No data yet" instead
// of a cryptic dash when the metric can't be computed yet.
function pctKpi(value: number | null | undefined): { value: string; muted: boolean } {
  return value == null ? { value: "No data yet", muted: true } : { value: `${Math.round(value)}%`, muted: false };
}

export function AdminPlatformValue({
  metrics,
  loading,
  rangeDays,
  onRangeDaysChange,
}: AdminPlatformValueProps) {
  const northStarSeries = useMemo(
    () =>
      (metrics?.north_star.weekly ?? []).map((point) => ({
        week: formatWeekLabel(point.week_start),
        value: point.value,
      })),
    [metrics]
  );

  const powerSeries = useMemo(() => {
    const curve = metrics?.power_user_curve;
    if (!curve) {
      return [] as Array<{ days: number; users: number; power: boolean }>;
    }
    return curve.buckets.map((bucket) => ({
      days: bucket.days_active,
      users: bucket.users,
      power: bucket.days_active >= curve.power_user_threshold,
    }));
  }, [metrics]);

  const dailyCostSeries = useMemo(
    () =>
      (metrics?.cost.daily ?? []).map((row) => ({
        day: formatWeekLabel(row.day),
        total_tokens: row.total_tokens,
        cost: row.cost,
      })),
    [metrics]
  );

  const rangeSelector = (
    <Select value={String(rangeDays)} onValueChange={(value) => onRangeDaysChange(Number(value))}>
      <SelectTrigger className="admin-value-range" aria-label="Metrics time range">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {rangeOptions.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  if (loading && !metrics) {
    return (
      <div className="admin-value-surface">
        <div className="admin-value-toolbar">
          <p className="admin-value-caption">Loading platform value metrics…</p>
          {rangeSelector}
        </div>
        <Card className="admin-value-empty-card">
          <CardContent>
            <p className="admin-empty">Crunching runs, cohorts, and token usage…</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (metrics && !metrics.available) {
    return (
      <div className="admin-value-surface">
        <div className="admin-value-toolbar">
          <p className="admin-value-caption">Platform value metrics</p>
          {rangeSelector}
        </div>
        <Card className="admin-value-empty-card">
          <CardHeader>
            <CardTitle>Value metrics need the Postgres store</CardTitle>
            <CardDescription>
              Retention, activation, and cohort analytics are computed from grouped Postgres
              queries. This deployment is running an in-memory store, so only operational metrics
              are available.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  if (!metrics) {
    return null;
  }

  const { north_star: northStar, kpis, retention_cohorts: retention, power_user_curve: power, activation_funnel: funnel, cost } = metrics;
  const delta = northStarDelta(northStar.current_week, northStar.previous_week, northStar.delta_pct);
  const activation = pctKpi(kpis.activation_rate_pct);
  const stickiness = pctKpi(kpis.stickiness_pct);
  const week4 = pctKpi(kpis.week4_retention_pct);
  const usefulRate = kpis.useful_run_rate_pct;
  const periods = Array.from({ length: retention.max_periods }, (_, index) => index);

  return (
    <div className="admin-value-surface">
      <div className="admin-value-toolbar">
        <p className="admin-value-caption">
          How much value the platform is delivering — and whether real usage is growing.
        </p>
        {rangeSelector}
      </div>

      <Card className="admin-northstar-card">
        <CardHeader>
          <div className="admin-card-title-row">
            <div className="flex items-center gap-2">
              <Sparkles className="size-4" />
              <CardTitle>North star · {northStar.label}</CardTitle>
            </div>
            {delta ? (
              <Badge variant="outline" className={delta.positive ? "admin-runtime-good" : "admin-runtime-warn"}>
                {delta.positive ? <TrendingUp className="size-3.5" /> : <TrendingDown className="size-3.5" />}
                {delta.text}
              </Badge>
            ) : null}
          </div>
          <CardDescription>{northStar.definition}</CardDescription>
        </CardHeader>
        <CardContent className="admin-northstar-body">
          <div className="admin-northstar-figure">
            <span className="admin-northstar-value">{formatInt(northStar.current_week)}</span>
            <span className="admin-northstar-sub">
              this week · {formatInt(northStar.previous_week)} last week
            </span>
          </div>
          <ChartContainer config={northStarConfig} className="admin-northstar-chart h-[120px] w-full">
            <LineChart data={northStarSeries} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="week" tickLine={false} axisLine={false} minTickGap={24} />
              <YAxis hide allowDecimals={false} />
              <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
              <Line
                type="monotone"
                dataKey="value"
                stroke="var(--color-value)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <div className="admin-value-kpi-grid">
        <ValueKpi
          icon={<Target className="size-4" />}
          label="Activation rate"
          value={activation.value}
          muted={activation.muted}
          footnote={`New users with a useful run within ${kpis.activation_window_days} days`}
        />
        <ValueKpi
          icon={<Activity className="size-4" />}
          label="Stickiness"
          value={stickiness.value}
          muted={stickiness.muted}
          footnote={`${formatInt(kpis.wau)} weekly of ${formatInt(kpis.mau)} monthly active`}
        />
        <ValueKpi
          icon={<Repeat className="size-4" />}
          label="Week-4 retention"
          value={week4.value}
          muted={week4.muted}
          footnote="Still active 4 weeks after signup"
        />
        <ValueKpi
          icon={<Users className="size-4" />}
          label="New researchers"
          value={formatInt(kpis.new_users)}
          footnote="Signed up in this period"
        />
        <ValueKpi
          icon={<Sparkles className="size-4" />}
          label="Useful runs"
          value={formatInt(kpis.useful_runs)}
          footnote={`of ${formatInt(kpis.total_runs)} runs${usefulRate != null ? ` · ${Math.round(usefulRate)}% useful` : ""}`}
        />
        <ValueKpi
          icon={<Cpu className="size-4" />}
          label="Total tokens"
          value={formatCompact(cost.total_tokens)}
          footnote="AI tokens used across all users"
        />
      </div>

      <Card className="admin-cohort-card">
        <CardHeader>
          <CardTitle>Weekly retention by signup cohort</CardTitle>
          <CardDescription>
            Each row is a cohort; a curve that flattens to a plateau is the strongest proof users
            keep coming back.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {retention.cohorts.length === 0 ? (
            <p className="admin-empty">No cohorts in this range yet.</p>
          ) : (
            <div
              className="admin-cohort-grid"
              style={{ gridTemplateColumns: `minmax(72px, 1fr) 44px repeat(${retention.max_periods}, minmax(40px, 1fr))` }}
              role="table"
              aria-label="Weekly retention by signup cohort"
            >
              <div className="admin-cohort-head" role="columnheader">cohort</div>
              <div className="admin-cohort-head admin-cohort-num" role="columnheader">n</div>
              {periods.map((period) => (
                <div key={`head-${period}`} className="admin-cohort-head admin-cohort-num" role="columnheader">
                  {`W${period}`}
                </div>
              ))}
              {retention.cohorts.map((cohort: AdminMetricCohort) => (
                <Fragment key={cohort.cohort_start}>
                  <div className="admin-cohort-label" role="rowheader">{cohortLabel(cohort.cohort_start)}</div>
                  <div className="admin-cohort-num admin-cohort-size">{formatInt(cohort.size)}</div>
                  {periods.map((period) => {
                    const pct = cohort.values_pct[period] ?? null;
                    const style = cohortCellStyle(pct);
                    return (
                      <div
                        key={`${cohort.cohort_start}-${period}`}
                        className="admin-cohort-cell"
                        style={style}
                        role="cell"
                      >
                        {pct == null ? "·" : Math.round(pct)}
                      </div>
                    );
                  })}
                </Fragment>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="admin-data-grid">
        <Card className="admin-power-card">
          <CardHeader>
            <div className="admin-card-title-row">
              <CardTitle>Power-user curve</CardTitle>
              <Badge variant="outline" className="admin-runtime-good">
                {`${formatPct(power.power_user_share_pct)} near-daily`}
              </Badge>
            </div>
            <CardDescription>
              Days active in the last {power.window_days}; a rising right tail (≥{power.power_user_threshold}{" "}
              days) is a committed core.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {power.total_users === 0 ? (
              <p className="admin-empty">No active users in this window yet.</p>
            ) : (
              <ChartContainer config={powerCurveConfig} className="admin-chart-canvas h-[220px] w-full">
                <BarChart data={powerSeries}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="days" tickLine={false} axisLine={false} minTickGap={8} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} domain={[0, "dataMax"]} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="users" radius={3}>
                    {powerSeries.map((entry) => (
                      <Cell key={entry.days} fill={entry.power ? "#2a78d6" : "var(--muted-foreground)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>

        <Card className="admin-funnel-card">
          <CardHeader>
            <CardTitle>New-user activation funnel</CardTitle>
            <CardDescription>Where new researchers drop off on the way to value.</CardDescription>
          </CardHeader>
          <CardContent className="admin-funnel-body">
            {funnel.length === 0 ? (
              <p className="admin-empty">No new users in this range yet.</p>
            ) : (
              funnel.map((stage) => {
                const width = stage.of_top_pct ?? 0;
                return (
                  <div key={stage.stage} className="admin-funnel-row">
                    <div className="admin-funnel-meta">
                      <span className="admin-funnel-stage">{stage.stage}</span>
                      <span className="admin-funnel-count">
                        {formatInt(stage.users)}
                        {stage.of_previous_pct != null ? (
                          <span className="admin-funnel-step">{` ${stage.of_previous_pct.toFixed(0)}%`}</span>
                        ) : null}
                      </span>
                    </div>
                    <div className="admin-funnel-track">
                      <div className="admin-funnel-fill" style={{ width: `${Math.max(2, width)}%` }} />
                    </div>
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="admin-cost-card">
        <CardHeader>
          <div className="admin-card-title-row">
            <div className="flex items-center gap-2">
              <Coins className="size-4" />
              <CardTitle>AI usage and cost</CardTitle>
            </div>
            <Badge variant="outline" className={cost.priced ? "admin-runtime-good" : "admin-runtime-warn"}>
              {cost.priced ? formatCurrency(cost.total_cost, cost.currency) : `${formatCompact(cost.total_tokens)} tokens`}
            </Badge>
          </div>
          <CardDescription>
            {cost.priced
              ? `AI token usage over this period, priced at ${formatCurrency(cost.total_cost, cost.currency)} total.`
              : "AI token usage over this period. Add model prices to see dollar cost."}
          </CardDescription>
        </CardHeader>
        <CardContent className="admin-cost-body">
          <div className="admin-runtime-grid">
            <div>
              <span>Total tokens</span>
              <strong>{formatCompact(cost.total_tokens)}</strong>
            </div>
            <div>
              <span>Tokens / useful run</span>
              <strong>{formatCompact(cost.tokens_per_useful_run)}</strong>
            </div>
            <div>
              <span>Useful runs</span>
              <strong>{formatInt(cost.useful_runs)}</strong>
            </div>
            <div>
              <span>{cost.priced ? "Cost / useful run" : "Models"}</span>
              <strong>
                {cost.priced
                  ? formatCurrency(cost.cost_per_useful_run, cost.currency)
                  : formatInt(cost.by_model.length)}
              </strong>
            </div>
          </div>
          {dailyCostSeries.length > 0 ? (
            <ChartContainer config={dailyCostConfig} className="admin-chart-canvas h-[160px] w-full">
              <BarChart data={dailyCostSeries}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="day" tickLine={false} axisLine={false} minTickGap={16} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => formatCompact(Number(value))} width={44} />
                <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                <Bar dataKey="total_tokens" fill="var(--color-total_tokens)" radius={3} />
              </BarChart>
            </ChartContainer>
          ) : (
            <p className="admin-empty">No token usage recorded in this range.</p>
          )}

          {cost.by_model.length > 0 ? (
            <div className="admin-cost-models">
              {cost.by_model.slice(0, 6).map((model) => (
                <div key={model.model} className="admin-cost-model-row">
                  <span className="admin-cost-model-name">{model.model}</span>
                  <span className="admin-cost-model-tokens">{formatCompact(model.total_tokens)} tok</span>
                  <span className="admin-cost-model-runs">{formatInt(model.runs)} runs</span>
                  <span className="admin-cost-model-cost">
                    {model.priced ? formatCurrency(model.cost, cost.currency) : "—"}
                  </span>
                </div>
              ))}
            </div>
          ) : null}

          {cost.unpriced_models.length > 0 ? (
            <p className="admin-kpi-footnote">
              No price configured for: {cost.unpriced_models.join(", ")}.
            </p>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

type ValueKpiProps = {
  icon: ReactNode;
  label: string;
  value: string;
  footnote: string;
  muted?: boolean;
};

function ValueKpi({ icon, label, value, footnote, muted = false }: ValueKpiProps) {
  return (
    <Card className="admin-kpi-card admin-value-kpi-card">
      <CardHeader className="admin-value-kpi-head">
        <CardDescription className="admin-value-kpi-label">
          {icon}
          {label}
        </CardDescription>
        <CardTitle className={muted ? "admin-kpi-value admin-kpi-value-muted" : "admin-kpi-value"}>
          {value}
        </CardTitle>
      </CardHeader>
      <CardContent className="admin-value-kpi-foot">
        <p className="admin-kpi-footnote">{footnote}</p>
      </CardContent>
    </Card>
  );
}
