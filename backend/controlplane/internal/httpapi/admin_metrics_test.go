package httpapi

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func mustPtr(t *testing.T, p *float64) float64 {
	t.Helper()
	if p == nil {
		t.Fatalf("expected non-nil pointer")
	}
	return *p
}

func TestBuildAdminMetricsResponse(t *testing.T) {
	now := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	thisWeek := isoWeekStartUTC(now)
	prevWeek := thisWeek.AddDate(0, 0, -7)

	m := domain.AdminMetrics{
		RangeDays: 90,
		NorthStar: domain.AdminNorthStar{Weekly: []domain.AdminWeekPoint{
			{WeekStart: prevWeek, Value: 120},
			{WeekStart: thisWeek, Value: 142},
		}},
		WAU:                  210,
		MAU:                  553,
		NewUsers:             141,
		ActivationActivated:  86,
		ActivationCohort:     141,
		ActivationWindowDays: 7,
		UsefulRuns:           1964,
		TotalRuns:            2690,
		RetentionCohorts: []domain.AdminRetentionCohort{
			{CohortStart: thisWeek.AddDate(0, 0, -35), Size: 34, Retained: []int{34, 24, 19, 17, 16, 16}},
			{CohortStart: thisWeek.AddDate(0, 0, -7), Size: 41, Retained: []int{40, 28, 0, 0, 0, 0}},
		},
		PowerUserCurve: domain.AdminPowerUserCurve{
			WindowDays:         28,
			PowerUserThreshold: 21,
			Buckets: []domain.AdminPowerUserBucket{
				{DaysActive: 1, Users: 58}, {DaysActive: 2, Users: 34},
				{DaysActive: 21, Users: 12}, {DaysActive: 27, Users: 8}, {DaysActive: 28, Users: 20},
			},
		},
		Funnel: []domain.AdminFunnelStage{
			{Stage: "Signed up", Users: 141},
			{Stage: "Started a run", Users: 121},
			{Stage: "Produced an output", Users: 98},
			{Stage: "Returned and ran again", Users: 67},
		},
		TokensByModel: []domain.AdminModelTokens{
			{Model: "deepseek_v4", InputTokens: 10_000_000, OutputTokens: 2_000_000, TotalTokens: 12_000_000, Runs: 100},
			{Model: "Qwen3.6-27B", InputTokens: 5_000_000, OutputTokens: 1_000_000, TotalTokens: 6_000_000, Runs: 50},
		},
		TokensDaily: []domain.AdminDayModelTokens{
			{Day: now.AddDate(0, 0, -2), Model: "deepseek_v4", InputTokens: 1_000_000, OutputTokens: 200_000, TotalTokens: 1_200_000},
			{Day: now.AddDate(0, 0, -2), Model: "Qwen3.6-27B", InputTokens: 500_000, OutputTokens: 100_000, TotalTokens: 600_000},
			{Day: now.AddDate(0, 0, -1), Model: "deepseek_v4", InputTokens: 2_000_000, OutputTokens: 400_000, TotalTokens: 2_400_000},
		},
	}
	prices := map[string]ModelPrice{"deepseek_v4": {InputPerMTok: 0.27, OutputPerMTok: 1.10}}

	resp := buildAdminMetricsResponse(m, now, 90, prices)

	if !resp.Available {
		t.Fatalf("expected available=true")
	}
	if resp.NorthStar.CurrentWeek != 142 || resp.NorthStar.PreviousWeek != 120 {
		t.Fatalf("north star current/previous = %d/%d", resp.NorthStar.CurrentWeek, resp.NorthStar.PreviousWeek)
	}
	if got := mustPtr(t, resp.NorthStar.DeltaPct); got != 18.3 {
		t.Fatalf("north star delta = %v, want 18.3", got)
	}
	if got := mustPtr(t, resp.KPIs.StickinessPct); got != 38 {
		t.Fatalf("stickiness = %v, want 38", got)
	}
	if got := mustPtr(t, resp.KPIs.ActivationRatePct); got != 61 {
		t.Fatalf("activation = %v, want 61", got)
	}
	if got := mustPtr(t, resp.KPIs.UsefulRunRatePct); got != 73 {
		t.Fatalf("useful run rate = %v, want 73", got)
	}

	// Mature cohort: all six periods measurable.
	mature := resp.Retention.Cohorts[0]
	if mature.ValuesPct[4] == nil {
		t.Fatalf("mature cohort week-4 should be measurable")
	}
	if got := *mature.ValuesPct[4]; got != 47.1 {
		t.Fatalf("mature cohort week-4 pct = %v, want 47.1", got)
	}
	// Young cohort: only periods 0 and 1 measurable, the rest must be null
	// (not yet measurable) — never a misleading 0%.
	young := resp.Retention.Cohorts[1]
	if young.ValuesPct[0] == nil || young.ValuesPct[1] == nil {
		t.Fatalf("young cohort weeks 0-1 should be measurable")
	}
	if young.ValuesPct[2] != nil {
		t.Fatalf("young cohort week-2 should be null (not yet measurable), got %v", *young.ValuesPct[2])
	}
	if got := mustPtr(t, resp.KPIs.Week4RetentionPct); got != 47.1 {
		t.Fatalf("week4 retention KPI = %v, want 47.1", got)
	}

	// Power-user share counts only buckets at/above the threshold.
	if resp.PowerUserCurve.TotalUsers != 132 || resp.PowerUserCurve.PowerUsers != 40 {
		t.Fatalf("power curve total/power = %d/%d, want 132/40", resp.PowerUserCurve.TotalUsers, resp.PowerUserCurve.PowerUsers)
	}

	if len(resp.ActivationFunnel) != 4 {
		t.Fatalf("funnel stages = %d, want 4", len(resp.ActivationFunnel))
	}
	if resp.ActivationFunnel[0].OfPreviousPct != nil {
		t.Fatalf("first funnel stage should have no previous pct")
	}
	if got := mustPtr(t, resp.ActivationFunnel[1].OfPreviousPct); got != 85.8 {
		t.Fatalf("funnel stage 2 of-previous = %v, want 85.8", got)
	}

	// Cost: deepseek priced (2.70 input + 2.20 output = 4.90); qwen unpriced.
	if !resp.Cost.Priced {
		t.Fatalf("cost should be priced")
	}
	if got := mustPtr(t, resp.Cost.TotalCost); got != 4.90 {
		t.Fatalf("total cost = %v, want 4.90", got)
	}
	if resp.Cost.TotalTokens != 18_000_000 {
		t.Fatalf("total tokens = %d, want 18000000", resp.Cost.TotalTokens)
	}
	if len(resp.Cost.UnpricedModels) != 1 || resp.Cost.UnpricedModels[0] != "Qwen3.6-27B" {
		t.Fatalf("unpriced models = %v, want [Qwen3.6-27B]", resp.Cost.UnpricedModels)
	}
	if got := mustPtr(t, resp.Cost.TokensPerUsefulRun); got != 9165 {
		t.Fatalf("tokens per useful run = %v, want 9165", got)
	}
	if len(resp.Cost.Daily) != 2 {
		t.Fatalf("daily cost points = %d, want 2", len(resp.Cost.Daily))
	}
	if resp.Cost.Daily[0].TotalTokens != 1_800_000 {
		t.Fatalf("day-1 tokens = %d, want 1800000", resp.Cost.Daily[0].TotalTokens)
	}
	if resp.Cost.Daily[0].Cost == nil {
		t.Fatalf("day-1 cost should be priced")
	}
}

func TestBuildAdminMetricsResponseNoPrices(t *testing.T) {
	now := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	m := domain.AdminMetrics{
		UsefulRuns:    100,
		TokensByModel: []domain.AdminModelTokens{{Model: "deepseek_v4", TotalTokens: 1_000_000, Runs: 10}},
	}
	resp := buildAdminMetricsResponse(m, now, 90, map[string]ModelPrice{})
	if resp.Cost.Priced {
		t.Fatalf("cost must not be priced without a price map")
	}
	if resp.Cost.TotalCost != nil {
		t.Fatalf("total cost must be nil without prices")
	}
	if resp.Cost.TotalTokens != 1_000_000 {
		t.Fatalf("token volume should still be reported, got %d", resp.Cost.TotalTokens)
	}
	if len(resp.Cost.UnpricedModels) != 1 {
		t.Fatalf("unpriced models = %v", resp.Cost.UnpricedModels)
	}
}

func TestLoadModelPricesFromEnv(t *testing.T) {
	t.Setenv(modelPricesEnv, `{"DeepSeek_V4": {"input_per_mtok": 0.27, "output_per_mtok": 1.1}}`)
	prices := loadModelPricesFromEnv()
	price, ok := priceFor(prices, "deepseek_v4")
	if !ok {
		t.Fatalf("expected normalized lookup to find deepseek_v4")
	}
	if price.InputPerMTok != 0.27 || price.OutputPerMTok != 1.1 {
		t.Fatalf("price = %+v", price)
	}

	t.Setenv(modelPricesEnv, "not json")
	if len(loadModelPricesFromEnv()) != 0 {
		t.Fatalf("invalid JSON must yield no prices (never fabricate cost)")
	}
}

func TestParseRangeDaysParam(t *testing.T) {
	cases := map[string]int{
		"":     defaultMetricsRangeDays,
		"7":    minMetricsRangeDays,
		"45":   45,
		"9999": maxMetricsRangeDays,
		"abc":  defaultMetricsRangeDays,
	}
	for raw, want := range cases {
		req := httptest.NewRequest("GET", "/admin/metrics?range_days="+raw, nil)
		if got := parseRangeDaysParam(req); got != want {
			t.Fatalf("parseRangeDaysParam(%q) = %d, want %d", raw, got, want)
		}
	}
}

func TestIsoWeekStartUTC(t *testing.T) {
	// 2026-06-26 is a Friday; the ISO week starts Monday 2026-06-22.
	got := isoWeekStartUTC(time.Date(2026, 6, 26, 23, 0, 0, 0, time.UTC))
	want := time.Date(2026, 6, 22, 0, 0, 0, 0, time.UTC)
	if !got.Equal(want) {
		t.Fatalf("isoWeekStartUTC = %v, want %v", got, want)
	}
}
