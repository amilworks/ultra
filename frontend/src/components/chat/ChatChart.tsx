import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { CodeBlock, CodeBlockCode } from "@/components/prompt-kit/code-block";
import {
  chartTokenVar,
  parseChartSpec,
  type ChartSpec,
} from "@/features/chat/chart-spec";

// Distinct marker shapes give scatter a secondary (non-color) encoding, so
// series stay distinguishable for colorblind viewers — see the palette rules.
const SCATTER_SHAPES = [
  "circle",
  "triangle",
  "square",
  "diamond",
  "cross",
  "star",
  "wye",
  "circle",
] as const;

function buildConfig(spec: ChartSpec): ChartConfig {
  const config: ChartConfig = {};
  for (const s of spec.series) {
    config[s.key] = { label: s.label, color: chartTokenVar(s.colorIndex) };
  }
  return config;
}

function ChartInner({ spec }: { spec: ChartSpec }) {
  const multi = spec.series.length > 1;
  const legend = multi ? (
    <ChartLegend content={<ChartLegendContent />} />
  ) : null;

  if (spec.type === "pie") {
    const valueKey = spec.series[0].key;
    return (
      <PieChart>
        <ChartTooltip content={<ChartTooltipContent nameKey={spec.x} hideLabel />} />
        <Pie data={spec.data} dataKey={valueKey} nameKey={spec.x} innerRadius={48} strokeWidth={2}>
          {spec.data.map((_row, i) => (
            <Cell key={i} fill={chartTokenVar((i % 8) + 1)} />
          ))}
        </Pie>
      </PieChart>
    );
  }

  if (spec.type === "scatter") {
    return (
      <ScatterChart margin={{ left: 4, right: 12, top: 8, bottom: 4 }}>
        <CartesianGrid vertical={false} />
        <XAxis type="number" dataKey={spec.x} tickLine={false} axisLine={false} tickMargin={8} />
        <YAxis type="number" tickLine={false} axisLine={false} width={40} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
        {spec.series.map((s, i) => (
          <Scatter
            key={s.key}
            name={s.label}
            data={spec.data.map((row) => ({ [spec.x]: row[spec.x], [s.key]: row[s.key] }))}
            dataKey={s.key}
            fill={`var(--color-${s.key})`}
            shape={SCATTER_SHAPES[i % SCATTER_SHAPES.length]}
          />
        ))}
        {legend}
      </ScatterChart>
    );
  }

  const axes = (
    <>
      <CartesianGrid vertical={false} />
      <XAxis dataKey={spec.x} tickLine={false} axisLine={false} tickMargin={8} />
      <YAxis tickLine={false} axisLine={false} width={40} />
      <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
    </>
  );

  if (spec.type === "bar") {
    return (
      <BarChart data={spec.data} margin={{ left: 4, right: 12, top: 8 }}>
        {axes}
        {spec.series.map((s) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            fill={`var(--color-${s.key})`}
            radius={spec.stacked ? 0 : 3}
            stackId={spec.stacked ? "a" : undefined}
          />
        ))}
        {legend}
      </BarChart>
    );
  }

  if (spec.type === "area") {
    return (
      <AreaChart data={spec.data} margin={{ left: 4, right: 12, top: 8 }}>
        {axes}
        {spec.series.map((s) => (
          <Area
            key={s.key}
            dataKey={s.key}
            stroke={`var(--color-${s.key})`}
            fill={`var(--color-${s.key})`}
            fillOpacity={0.12}
            strokeWidth={2}
            stackId={spec.stacked ? "a" : undefined}
          />
        ))}
        {legend}
      </AreaChart>
    );
  }

  // line (default)
  return (
    <LineChart data={spec.data} margin={{ left: 4, right: 12, top: 8 }}>
      {axes}
      {spec.series.map((s) => (
        <Line
          key={s.key}
          dataKey={s.key}
          stroke={`var(--color-${s.key})`}
          strokeWidth={2}
          dot={false}
        />
      ))}
      {legend}
    </LineChart>
  );
}

/**
 * Renders a validated agent ```chart spec. If the source is incomplete (mid
 * stream) or fails validation, it falls back to showing the raw JSON as a code
 * block — never an execution path, never a scary error.
 */
export default function ChatChart({ source }: { source: string }) {
  const parsed = useMemo(() => {
    try {
      return { spec: parseChartSpec(source), error: null as string | null };
    } catch (e) {
      return { spec: null, error: e instanceof Error ? e.message : "invalid chart" };
    }
  }, [source]);

  if (!parsed.spec) {
    return (
      <CodeBlock className="language-json">
        <CodeBlockCode code={source} language="json" />
      </CodeBlock>
    );
  }

  const spec = parsed.spec;
  const config = buildConfig(spec);
  return (
    <figure className="pk-chart" role="group" aria-label={spec.title ?? "chart"}>
      {spec.title ? <figcaption className="pk-chart-title">{spec.title}</figcaption> : null}
      <ChartContainer config={config} className="h-[260px] w-full">
        <ChartInner spec={spec} />
      </ChartContainer>
    </figure>
  );
}
