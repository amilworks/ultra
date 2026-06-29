package httpapi

import (
	"context"
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// northStarLabel and northStarDefinition describe the headline value metric.
// They live here so the API is self-documenting for the dashboard.
const (
	northStarLabel      = "Weekly active researchers completing a useful run"
	northStarDefinition = "A run is \"useful\" when it succeeds and produces an output (artifact). Counts distinct users per ISO week."
)

// Defaults for the value-metric computation. Overridable per request where it
// makes sense; otherwise tuned to weekly-cadence research usage.
const (
	defaultMetricsRangeDays         = 90
	minMetricsRangeDays             = 14
	maxMetricsRangeDays             = 365
	cohortMaxPeriods                = 6
	activationWindowDays            = 7
	powerUserWindowDays             = 28
	powerUserThreshold              = 21
	costDailyWindowDays             = 30
	defaultCostCurrency             = "USD"
	modelPricesEnv                  = "ULTRA_CONTROL_MODEL_PRICES"
	tokensPerMillion        float64 = 1_000_000
)

// ModelPrice is the per-million-token price for one model, used to turn the
// token ledger into currency. Prices are operator-configured (no hardcoded
// vendor prices) so the cost panel never fabricates numbers.
type ModelPrice struct {
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

// loadModelPricesFromEnv parses ULTRA_CONTROL_MODEL_PRICES, a JSON object of
// {"model": {"input_per_mtok": 0.27, "output_per_mtok": 1.10}}. Keys are
// normalized (lower-cased, trimmed). A parse error logs and yields no prices,
// which makes the cost panel report token volume only — never a wrong dollar.
func loadModelPricesFromEnv() map[string]ModelPrice {
	raw := strings.TrimSpace(os.Getenv(modelPricesEnv))
	if raw == "" {
		return map[string]ModelPrice{}
	}
	parsed := map[string]ModelPrice{}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		log.Printf("admin metrics: ignoring invalid %s: %v", modelPricesEnv, err)
		return map[string]ModelPrice{}
	}
	prices := make(map[string]ModelPrice, len(parsed))
	for model, price := range parsed {
		key := normalizeModelKey(model)
		if key == "" {
			continue
		}
		prices[key] = price
	}
	return prices
}

func normalizeModelKey(model string) string {
	return strings.ToLower(strings.TrimSpace(model))
}

// adminMetricsStore is the optional store capability that computes the
// value-proving metric set from grouped queries. Stores without it (the
// in-memory dev store, test fakes) make the endpoint report available=false.
type adminMetricsStore interface {
	AdminMetrics(ctx context.Context, params domain.AdminMetricsParams) (domain.AdminMetrics, error)
}

func (deps ServerDeps) handleAdminMetrics(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	now := time.Now().UTC()
	rangeDays := parseRangeDaysParam(r)

	store, ok := deps.Store.(adminMetricsStore)
	if !ok {
		writeJSON(w, http.StatusOK, adminMetricsResponse{
			GeneratedAt: now.Format(time.RFC3339Nano),
			Available:   false,
			RangeDays:   rangeDays,
			NorthStar:   adminNorthStarResponse{Label: northStarLabel, Definition: northStarDefinition},
		})
		return
	}

	params := domain.AdminMetricsParams{
		Now:                  now,
		RangeStart:           now.AddDate(0, 0, -rangeDays),
		CohortMaxPeriods:     cohortMaxPeriods,
		ActivationWindowDays: activationWindowDays,
		PowerUserWindowDays:  powerUserWindowDays,
		PowerUserThreshold:   powerUserThreshold,
		CostSince:            now.AddDate(0, 0, -costDailyWindowDays),
	}
	metrics, err := store.AdminMetrics(r.Context(), params)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	resp := buildAdminMetricsResponse(metrics, now, rangeDays, deps.ModelPrices)
	writeJSON(w, http.StatusOK, resp)
}

func parseRangeDaysParam(r *http.Request) int {
	raw := strings.TrimSpace(r.URL.Query().Get("range_days"))
	if raw == "" {
		return defaultMetricsRangeDays
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return defaultMetricsRangeDays
	}
	if value < minMetricsRangeDays {
		return minMetricsRangeDays
	}
	if value > maxMetricsRangeDays {
		return maxMetricsRangeDays
	}
	return value
}

type adminMetricsResponse struct {
	GeneratedAt      string                     `json:"generated_at"`
	Available        bool                       `json:"available"`
	RangeDays        int                        `json:"range_days"`
	NorthStar        adminNorthStarResponse     `json:"north_star"`
	KPIs             adminMetricKPIsResponse    `json:"kpis"`
	Retention        adminRetentionResponse     `json:"retention_cohorts"`
	PowerUserCurve   adminPowerCurveResponse    `json:"power_user_curve"`
	ActivationFunnel []adminFunnelStageResponse `json:"activation_funnel"`
	Cost             adminCostResponse          `json:"cost"`
}

type adminWeekPointResponse struct {
	WeekStart string `json:"week_start"`
	Value     int    `json:"value"`
}

type adminNorthStarResponse struct {
	Label        string                   `json:"label"`
	Definition   string                   `json:"definition"`
	CurrentWeek  int                      `json:"current_week"`
	PreviousWeek int                      `json:"previous_week"`
	DeltaPct     *float64                 `json:"delta_pct"`
	Weekly       []adminWeekPointResponse `json:"weekly"`
}

type adminMetricKPIsResponse struct {
	WAU                  int      `json:"wau"`
	MAU                  int      `json:"mau"`
	StickinessPct        *float64 `json:"stickiness_pct"`
	NewUsers             int      `json:"new_users"`
	ActivationRatePct    *float64 `json:"activation_rate_pct"`
	ActivationWindowDays int      `json:"activation_window_days"`
	Week4RetentionPct    *float64 `json:"week4_retention_pct"`
	UsefulRunRatePct     *float64 `json:"useful_run_rate_pct"`
	UsefulRuns           int      `json:"useful_runs"`
	TotalRuns            int      `json:"total_runs"`
}

type adminCohortResponse struct {
	CohortStart string     `json:"cohort_start"`
	Size        int        `json:"size"`
	ValuesPct   []*float64 `json:"values_pct"`
	Retained    []int      `json:"retained"`
}

type adminRetentionResponse struct {
	Unit       string                `json:"unit"`
	MaxPeriods int                   `json:"max_periods"`
	Cohorts    []adminCohortResponse `json:"cohorts"`
}

type adminPowerBucketResponse struct {
	DaysActive int `json:"days_active"`
	Users      int `json:"users"`
}

type adminPowerCurveResponse struct {
	WindowDays         int                        `json:"window_days"`
	TotalUsers         int                        `json:"total_users"`
	PowerUserThreshold int                        `json:"power_user_threshold"`
	PowerUsers         int                        `json:"power_users"`
	PowerUserSharePct  *float64                   `json:"power_user_share_pct"`
	Buckets            []adminPowerBucketResponse `json:"buckets"`
}

type adminFunnelStageResponse struct {
	Stage         string   `json:"stage"`
	Users         int      `json:"users"`
	OfPreviousPct *float64 `json:"of_previous_pct"`
	OfTopPct      *float64 `json:"of_top_pct"`
}

type adminModelCostResponse struct {
	Model        string   `json:"model"`
	InputTokens  int64    `json:"input_tokens"`
	OutputTokens int64    `json:"output_tokens"`
	TotalTokens  int64    `json:"total_tokens"`
	Runs         int      `json:"runs"`
	Cost         *float64 `json:"cost"`
	Priced       bool     `json:"priced"`
}

type adminCostDayResponse struct {
	Day         string   `json:"day"`
	TotalTokens int64    `json:"total_tokens"`
	Cost        *float64 `json:"cost"`
}

type adminCostResponse struct {
	Currency           string                   `json:"currency"`
	Priced             bool                     `json:"priced"`
	TotalTokens        int64                    `json:"total_tokens"`
	TotalCost          *float64                 `json:"total_cost"`
	CostPerUsefulRun   *float64                 `json:"cost_per_useful_run"`
	TokensPerUsefulRun *float64                 `json:"tokens_per_useful_run"`
	UsefulRuns         int                      `json:"useful_runs"`
	UnpricedModels     []string                 `json:"unpriced_models"`
	ByModel            []adminModelCostResponse `json:"by_model"`
	Daily              []adminCostDayResponse   `json:"daily"`
}

func buildAdminMetricsResponse(m domain.AdminMetrics, now time.Time, rangeDays int, prices map[string]ModelPrice) adminMetricsResponse {
	thisWeek := isoWeekStartUTC(now)
	prevWeek := thisWeek.AddDate(0, 0, -7)

	weekly := make([]adminWeekPointResponse, 0, len(m.NorthStar.Weekly))
	weeklyByKey := make(map[string]int, len(m.NorthStar.Weekly))
	for _, pt := range m.NorthStar.Weekly {
		key := pt.WeekStart.UTC().Format(dateLayout)
		weeklyByKey[key] = pt.Value
		weekly = append(weekly, adminWeekPointResponse{WeekStart: key, Value: pt.Value})
	}
	current := weeklyByKey[thisWeek.Format(dateLayout)]
	previous := weeklyByKey[prevWeek.Format(dateLayout)]

	resp := adminMetricsResponse{
		GeneratedAt: now.Format(time.RFC3339Nano),
		Available:   true,
		RangeDays:   rangeDays,
		NorthStar: adminNorthStarResponse{
			Label:        northStarLabel,
			Definition:   northStarDefinition,
			CurrentWeek:  current,
			PreviousWeek: previous,
			DeltaPct:     deltaPct(current, previous),
			Weekly:       weekly,
		},
		KPIs: adminMetricKPIsResponse{
			WAU:                  m.WAU,
			MAU:                  m.MAU,
			StickinessPct:        pctPtr(m.WAU, m.MAU),
			NewUsers:             m.NewUsers,
			ActivationRatePct:    pctPtr(m.ActivationActivated, m.ActivationCohort),
			ActivationWindowDays: m.ActivationWindowDays,
			UsefulRunRatePct:     pctPtr(m.UsefulRuns, m.TotalRuns),
			UsefulRuns:           m.UsefulRuns,
			TotalRuns:            m.TotalRuns,
		},
	}

	resp.Retention, resp.KPIs.Week4RetentionPct = buildRetention(m.RetentionCohorts, thisWeek)
	resp.PowerUserCurve = buildPowerCurve(m.PowerUserCurve)
	resp.ActivationFunnel = buildFunnel(m.Funnel)
	resp.Cost = buildCost(m, prices)
	return resp
}

const dateLayout = "2006-01-02"

// isoWeekStartUTC returns the Monday 00:00 UTC of the week containing t, to
// match Postgres date_trunc('week', ...) which is ISO (Monday-anchored).
func isoWeekStartUTC(t time.Time) time.Time {
	t = t.UTC()
	offset := (int(t.Weekday()) + 6) % 7
	day := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	return day.AddDate(0, 0, -offset)
}

func buildRetention(cohorts []domain.AdminRetentionCohort, thisWeek time.Time) (adminRetentionResponse, *float64) {
	out := adminRetentionResponse{Unit: "week", MaxPeriods: cohortMaxPeriods, Cohorts: []adminCohortResponse{}}
	var week4Sum float64
	var week4Count int
	for _, cohort := range cohorts {
		measurable := weeksBetween(cohort.CohortStart, thisWeek) + 1
		if measurable > cohortMaxPeriods {
			measurable = cohortMaxPeriods
		}
		values := make([]*float64, cohortMaxPeriods)
		retained := make([]int, cohortMaxPeriods)
		for p := 0; p < cohortMaxPeriods; p++ {
			if p < len(cohort.Retained) {
				retained[p] = cohort.Retained[p]
			}
			if p < measurable && cohort.Size > 0 {
				pct := round1(float64(retained[p]) / float64(cohort.Size) * 100)
				values[p] = &pct
			}
		}
		if measurable > 4 && cohort.Size > 0 {
			week4Sum += float64(retained[4]) / float64(cohort.Size) * 100
			week4Count++
		}
		out.Cohorts = append(out.Cohorts, adminCohortResponse{
			CohortStart: cohort.CohortStart.UTC().Format(dateLayout),
			Size:        cohort.Size,
			ValuesPct:   values,
			Retained:    retained,
		})
	}
	var week4 *float64
	if week4Count > 0 {
		v := round1(week4Sum / float64(week4Count))
		week4 = &v
	}
	return out, week4
}

func weeksBetween(start, end time.Time) int {
	days := int(end.UTC().Sub(start.UTC()).Hours() / 24)
	if days < 0 {
		return 0
	}
	return days / 7
}

func buildPowerCurve(curve domain.AdminPowerUserCurve) adminPowerCurveResponse {
	out := adminPowerCurveResponse{
		WindowDays:         curve.WindowDays,
		PowerUserThreshold: curve.PowerUserThreshold,
		Buckets:            make([]adminPowerBucketResponse, 0, len(curve.Buckets)),
	}
	total := 0
	power := 0
	for _, bucket := range curve.Buckets {
		total += bucket.Users
		if bucket.DaysActive >= curve.PowerUserThreshold {
			power += bucket.Users
		}
		out.Buckets = append(out.Buckets, adminPowerBucketResponse{DaysActive: bucket.DaysActive, Users: bucket.Users})
	}
	out.TotalUsers = total
	out.PowerUsers = power
	out.PowerUserSharePct = pctPtr(power, total)
	return out
}

func buildFunnel(stages []domain.AdminFunnelStage) []adminFunnelStageResponse {
	out := make([]adminFunnelStageResponse, 0, len(stages))
	if len(stages) == 0 {
		return out
	}
	top := stages[0].Users
	for i, stage := range stages {
		entry := adminFunnelStageResponse{Stage: stage.Stage, Users: stage.Users, OfTopPct: pctPtr(stage.Users, top)}
		if i > 0 {
			entry.OfPreviousPct = pctPtr(stage.Users, stages[i-1].Users)
		}
		out = append(out, entry)
	}
	return out
}

func buildCost(m domain.AdminMetrics, prices map[string]ModelPrice) adminCostResponse {
	out := adminCostResponse{
		Currency:       defaultCostCurrency,
		UsefulRuns:     m.UsefulRuns,
		UnpricedModels: []string{},
		ByModel:        []adminModelCostResponse{},
		Daily:          []adminCostDayResponse{},
	}
	havePrices := len(prices) > 0
	var totalCost float64
	var totalTokens int64
	anyPriced := false
	unpriced := map[string]bool{}

	for _, model := range m.TokensByModel {
		totalTokens += model.TotalTokens
		entry := adminModelCostResponse{
			Model:        model.Model,
			InputTokens:  model.InputTokens,
			OutputTokens: model.OutputTokens,
			TotalTokens:  model.TotalTokens,
			Runs:         model.Runs,
		}
		if price, ok := priceFor(prices, model.Model); ok {
			cost := round2(tokenCost(model.InputTokens, model.OutputTokens, price))
			entry.Cost = &cost
			entry.Priced = true
			totalCost += cost
			anyPriced = true
		} else if model.TotalTokens > 0 {
			unpriced[model.Model] = true
		}
		out.ByModel = append(out.ByModel, entry)
	}

	out.TotalTokens = totalTokens
	out.Priced = havePrices && anyPriced
	if out.Priced {
		tc := round2(totalCost)
		out.TotalCost = &tc
		if m.UsefulRuns > 0 {
			cpr := round4(totalCost / float64(m.UsefulRuns))
			out.CostPerUsefulRun = &cpr
		}
	}
	if m.UsefulRuns > 0 && totalTokens > 0 {
		tpr := round1(float64(totalTokens) / float64(m.UsefulRuns))
		out.TokensPerUsefulRun = &tpr
	}
	for model := range unpriced {
		out.UnpricedModels = append(out.UnpricedModels, model)
	}
	sort.Strings(out.UnpricedModels)

	out.Daily = buildCostDaily(m.TokensDaily, prices, out.Priced)
	return out
}

func buildCostDaily(daily []domain.AdminDayModelTokens, prices map[string]ModelPrice, priced bool) []adminCostDayResponse {
	type dayAgg struct {
		tokens int64
		cost   float64
	}
	order := []string{}
	byDay := map[string]*dayAgg{}
	for _, row := range daily {
		key := row.Day.UTC().Format(dateLayout)
		agg, ok := byDay[key]
		if !ok {
			agg = &dayAgg{}
			byDay[key] = agg
			order = append(order, key)
		}
		agg.tokens += row.TotalTokens
		if price, ok := priceFor(prices, row.Model); ok {
			agg.cost += tokenCost(row.InputTokens, row.OutputTokens, price)
		}
	}
	sort.Strings(order)
	out := make([]adminCostDayResponse, 0, len(order))
	for _, key := range order {
		agg := byDay[key]
		entry := adminCostDayResponse{Day: key, TotalTokens: agg.tokens}
		if priced {
			c := round2(agg.cost)
			entry.Cost = &c
		}
		out = append(out, entry)
	}
	return out
}

func priceFor(prices map[string]ModelPrice, model string) (ModelPrice, bool) {
	price, ok := prices[normalizeModelKey(model)]
	return price, ok
}

func tokenCost(input, output int64, price ModelPrice) float64 {
	return float64(input)/tokensPerMillion*price.InputPerMTok + float64(output)/tokensPerMillion*price.OutputPerMTok
}

func deltaPct(current, previous int) *float64 {
	if previous <= 0 {
		return nil
	}
	v := round1(float64(current-previous) / float64(previous) * 100)
	return &v
}

func pctPtr(num, den int) *float64 {
	if den <= 0 {
		return nil
	}
	v := round1(float64(num) / float64(den) * 100)
	return &v
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }
func round2(v float64) float64 { return math.Round(v*100) / 100 }
func round4(v float64) float64 { return math.Round(v*10000) / 10000 }
