package gateengine

import (
	"strings"
	"testing"
)

// The M2 gate proofs (plan section 11-M2): the engine is trusted only because
// these tests exist. Proof 4 (the seeded RareSpot rows reproduce the five v1
// clauses) lives with the store seed; everything else is here.

const testSchema = "detection.v1"
const testKernel = "yolov5_two_pass/v1"

func run(metrics Blob) Run {
	return Run{Metrics: metrics, MetricSchema: testSchema, KernelVersion: testKernel, GoldHash: "hash-1"}
}

func detectionClauses() []Clause {
	return []Clause{
		{ClauseKey: "agg_map50", MetricPath: "aggregate.map50", Comparator: "max_drop_vs_active", Value: 0.005, Enabled: true, Required: true},
		{ClauseKey: "class_recall_abs", MetricPath: "per_class.*.recall_at_op", Comparator: "abs_floor", Value: 0.50, Enabled: true, Required: true},
		{ClauseKey: "slice_prior_map50", MetricPath: "per_slice.prior_train.map50", Comparator: "max_drop_vs_active", Value: 0.02, Slice: "prior_train", Params: map[string]any{"min_label_count": 10}, Enabled: true, Required: true},
		{ClauseKey: "slice_held_map50", MetricPath: "per_slice.held_out_test.map50", Comparator: "max_drop_vs_active", Value: 0.005, Slice: "held_out_test", Params: map[string]any{"min_label_count": 10}, Enabled: true, Required: true},
		{ClauseKey: "fp_empty_ceiling", MetricPath: "aggregate.fp_per_empty_frame", Comparator: "max_rise_vs_active", Value: 0.10, Enabled: true, Required: true},
	}
}

func healthyBlob(map50 float64, recall float64) Blob {
	return Blob{
		"aggregate": map[string]any{
			"map50":              map50,
			"fp_per_empty_frame": 0.28,
		},
		"per_class": map[string]any{
			"prairie_dog": map[string]any{"recall_at_op": recall, "predicted_count": float64(40), "label_count": float64(60)},
			"burrow":      map[string]any{"recall_at_op": 0.71, "predicted_count": float64(220), "label_count": float64(240)},
		},
		"per_slice": map[string]any{
			"prior_train":   map[string]any{"map50": map50 - 0.01, "label_count": float64(300)},
			"held_out_test": map[string]any{"map50": nil, "label_count": float64(0)},
		},
	}
}

// Proof 1: the baseline must PASS against itself.
func TestBaselinePassesAgainstItself(t *testing.T) {
	t.Parallel()
	blob := healthyBlob(0.83, 0.62)
	result := Evaluate(detectionClauses(), run(blob), run(blob), []string{"prairie_dog", "burrow"})
	if !result.Passed {
		t.Fatalf("baseline vs itself must pass; reasons: %v", result.Reasons)
	}
	// The empty held-out slice is recorded-excluded, never failed (proof 7).
	foundExclusion := false
	for _, clause := range result.Clauses {
		if clause.ClauseKey == "slice_held_map50" {
			if clause.Outcome != "excluded" {
				t.Fatalf("empty held-out slice outcome = %s, want excluded", clause.Outcome)
			}
			foundExclusion = true
		}
	}
	if !foundExclusion {
		t.Fatal("held-out clause missing from outcomes")
	}
}

// Proof 2: a deliberately-degraded candidate FAILS with the correct clauses
// and numbers in the reasons.
func TestDegradedCandidateFailsWithReasons(t *testing.T) {
	t.Parallel()
	baseline := healthyBlob(0.83, 0.62)
	degraded := healthyBlob(0.74, 0.41) // map50 crater + prairie_dog recall below the floor
	result := Evaluate(detectionClauses(), run(degraded), run(baseline), []string{"prairie_dog", "burrow"})
	if result.Passed {
		t.Fatal("degraded candidate must fail")
	}
	joined := strings.Join(result.Reasons, " | ")
	if !strings.Contains(joined, "aggregate.map50") {
		t.Fatalf("reasons missing the aggregate regression: %s", joined)
	}
	if !strings.Contains(joined, "per_class.prairie_dog.recall_at_op") || !strings.Contains(joined, "0.41") {
		t.Fatalf("reasons missing the per-class recall floor failure with its number: %s", joined)
	}
	// Per-class wildcard expanded per class: burrow passed, prairie_dog failed.
	var burrowOutcome, dogOutcome string
	for _, clause := range result.Clauses {
		if clause.ClauseKey == "class_recall_abs" {
			if strings.Contains(clause.MetricPath, "burrow") {
				burrowOutcome = clause.Outcome
			}
			if strings.Contains(clause.MetricPath, "prairie_dog") {
				dogOutcome = clause.Outcome
			}
		}
	}
	if burrowOutcome != "passed" || dogOutcome != "failed" {
		t.Fatalf("wildcard expansion outcomes: burrow=%s dog=%s", burrowOutcome, dogOutcome)
	}
}

// Proof 3: an empty/disabled clause set must FAIL - the vacuous-pass hole.
func TestEmptyClauseSetFailsClosed(t *testing.T) {
	t.Parallel()
	blob := healthyBlob(0.83, 0.62)
	for _, clauses := range [][]Clause{{}, {{ClauseKey: "x", MetricPath: "aggregate.map50", Comparator: "abs_floor", Value: 0.1, Enabled: false}}} {
		result := Evaluate(clauses, run(blob), run(blob), nil)
		if result.Passed {
			t.Fatal("empty/disabled clause set must fail closed")
		}
		if !strings.Contains(strings.Join(result.Reasons, " "), "no enabled guardrail clauses") {
			t.Fatalf("wrong reason: %v", result.Reasons)
		}
	}
}

func TestWildcardClauseWithoutManifestClassesFailsClosed(t *testing.T) {
	t.Parallel()
	clauses := []Clause{
		{ClauseKey: "class_recall_abs", MetricPath: "per_class.*.recall_at_op", Comparator: "abs_floor", Value: 0.50, Enabled: true},
		{ClauseKey: "agg_map50", MetricPath: "aggregate.map50", Comparator: "abs_floor", Value: 0.80, Enabled: true},
	}
	blob := healthyBlob(0.83, 0.62)
	result := Evaluate(clauses, run(blob), run(blob), []string{})

	if result.Passed {
		t.Fatal("wildcard clause without authoritative manifest class names must fail closed")
	}
	if len(result.Clauses) != 2 {
		t.Fatalf("clause outcomes = %d, want aggregate plus failed wildcard", len(result.Clauses))
	}
	if aggregate := result.Clauses[1]; aggregate.ClauseKey != "agg_map50" || aggregate.Outcome != "passed" {
		t.Fatalf("aggregate outcome = %+v, want retained passed outcome", aggregate)
	}

	wildcard := result.Clauses[0]
	wantReason := "clause class_recall_abs (per_class.*.recall_at_op) cannot be evaluated: no authoritative manifest class names"
	if wildcard.ClauseKey != "class_recall_abs" || wildcard.MetricPath != "per_class.*.recall_at_op" || wildcard.Outcome != "failed" {
		t.Fatalf("wildcard outcome = %+v, want failed outcome retaining source clause key and path", wildcard)
	}
	if wildcard.Reason != wantReason {
		t.Fatalf("wildcard reason = %q, want %q", wildcard.Reason, wantReason)
	}
	if len(result.Reasons) != 1 || result.Reasons[0] != wantReason {
		t.Fatalf("top-level reasons = %v, want [%q]", result.Reasons, wantReason)
	}
}

// Proof 5: a synthetic SEGMENTATION clause set evaluates with zero engine
// changes - Gate B is config, not code.
func TestSyntheticSegmentationClauseSet(t *testing.T) {
	t.Parallel()
	clauses := []Clause{
		{ClauseKey: "agg_miou", MetricPath: "aggregate.miou", Comparator: "max_drop_vs_active", Value: 0.01, Enabled: true, Required: true},
		{ClauseKey: "class_dice_abs", MetricPath: "per_class.*.dice", Comparator: "abs_floor", Value: 0.60, Enabled: true},
		{ClauseKey: "slice_prior_miou", MetricPath: "per_slice.prior_train.miou", Comparator: "max_drop_vs_active", Value: 0.02, Slice: "prior_train", Params: map[string]any{"min_label_count": 10}, Enabled: true, Required: true},
	}
	segBlob := func(miou float64, dice float64) Blob {
		return Blob{
			"aggregate": map[string]any{"miou": miou},
			"per_class": map[string]any{
				"nucleus": map[string]any{"dice": dice, "predicted_count": float64(9_000_000), "label_count": float64(10_000_000)},
			},
			"per_slice": map[string]any{
				"prior_train": map[string]any{"miou": miou - 0.005, "label_count": float64(12)},
			},
		}
	}
	segRun := func(blob Blob) Run {
		return Run{Metrics: blob, MetricSchema: "segmentation.v1", KernelVersion: "megaseg_iou/v1", GoldHash: "seg-hash"}
	}
	pass := Evaluate(clauses, segRun(segBlob(0.81, 0.72)), segRun(segBlob(0.80, 0.70)), []string{"nucleus"})
	if !pass.Passed {
		t.Fatalf("healthy segmentation candidate must pass; reasons: %v", pass.Reasons)
	}
	fail := Evaluate(clauses, segRun(segBlob(0.70, 0.31)), segRun(segBlob(0.80, 0.70)), []string{"nucleus"})
	if fail.Passed {
		t.Fatal("regressed segmentation candidate must fail")
	}
}

// Proof 6: a scope with 9 gold boxes under min_label_count 10 produces a
// RECORDED exclusion - not a pass, not a fail.
func TestMinSupportExclusionRecorded(t *testing.T) {
	t.Parallel()
	baseline := healthyBlob(0.83, 0.62)
	candidate := healthyBlob(0.83, 0.62)
	(candidate["per_slice"].(map[string]any))["prior_train"] = map[string]any{"map50": 0.10, "label_count": float64(9)}
	(baseline["per_slice"].(map[string]any))["prior_train"] = map[string]any{"map50": 0.81, "label_count": float64(9)}
	result := Evaluate(detectionClauses(), run(candidate), run(baseline), []string{"prairie_dog", "burrow"})
	// The catastrophic-looking 0.10 on a 9-box slice must NOT fail the gate -
	// and must NOT silently pass either: recorded exclusion.
	if !result.Passed {
		t.Fatalf("under-supported slice must not fail the gate; reasons: %v", result.Reasons)
	}
	joined := strings.Join(result.Reasons, " | ")
	if !strings.Contains(joined, "excluded") || !strings.Contains(joined, "under 10 gold labels") {
		t.Fatalf("exclusion not recorded: %s", joined)
	}
}

// Proof 7 (v1.2): an EMPTY held_out_test slice produces recorded exclusions
// and a computable overall gate - invariant 7 precedes invariant 2, so
// pending_new_survey never renders the gate permanently red.
func TestEmptyHeldOutSliceExcludedBeforeNoBaseline(t *testing.T) {
	t.Parallel()
	blob := healthyBlob(0.83, 0.62) // held_out: label_count 0, map50 null
	result := Evaluate(detectionClauses(), run(blob), run(blob), []string{"prairie_dog", "burrow"})
	if !result.Passed {
		t.Fatalf("gate must be computable with an empty held-out slice; reasons: %v", result.Reasons)
	}
	for _, clause := range result.Clauses {
		if clause.ClauseKey == "slice_held_map50" && clause.Outcome != "excluded" {
			t.Fatalf("empty slice outcome = %s (reason %s), want excluded", clause.Outcome, clause.Reason)
		}
	}
}

// Invariant 2: a metric missing from a POPULATED scope fails closed and
// directs to re-baselining.
func TestMissingMetricFailsClosed(t *testing.T) {
	t.Parallel()
	baseline := healthyBlob(0.83, 0.62)
	candidate := healthyBlob(0.83, 0.62)
	delete(candidate["aggregate"].(map[string]any), "map50")
	result := Evaluate(detectionClauses(), run(candidate), run(baseline), []string{"prairie_dog", "burrow"})
	if result.Passed {
		t.Fatal("missing metric must fail closed")
	}
	if !strings.Contains(strings.Join(result.Reasons, " "), "no comparable baseline") {
		t.Fatalf("wrong reason: %v", result.Reasons)
	}
}

// Invariant 3: kernel/schema skew refuses to compare.
func TestMixedVersionComparisonRefused(t *testing.T) {
	t.Parallel()
	blob := healthyBlob(0.83, 0.62)
	newer := Run{Metrics: blob, MetricSchema: testSchema, KernelVersion: "yolov5_two_pass/v2", GoldHash: "hash-1"}
	result := Evaluate(detectionClauses(), newer, run(blob), []string{"prairie_dog", "burrow"})
	if result.Passed {
		t.Fatal("mixed kernel versions must refuse")
	}
	if !strings.Contains(strings.Join(result.Reasons, " "), "mixed-version") {
		t.Fatalf("wrong reason: %v", result.Reasons)
	}
}

// Invariant 6: a class the candidate stopped predicting entirely FAILS as
// forgetting even when its metric would be absent.
func TestPredictedZeroFailsAsForgetting(t *testing.T) {
	t.Parallel()
	baseline := healthyBlob(0.83, 0.62)
	candidate := healthyBlob(0.83, 0.62)
	(candidate["per_class"].(map[string]any))["prairie_dog"] = map[string]any{
		"recall_at_op": 0.0, "predicted_count": float64(0), "label_count": float64(60),
	}
	result := Evaluate(detectionClauses(), run(candidate), run(baseline), []string{"prairie_dog", "burrow"})
	if result.Passed {
		t.Fatal("predicted_count 0 must fail")
	}
	if !strings.Contains(strings.Join(result.Reasons, " "), "predicted zero times") {
		t.Fatalf("wrong reason: %v", result.Reasons)
	}
}

func TestComparatorTable(t *testing.T) {
	t.Parallel()
	cases := []struct {
		comparator string
		candidate  float64
		baseline   float64
		value      float64
		strict     bool
		want       string
	}{
		{"max_drop_vs_active", 0.829, 0.83, 0.005, false, "passed"},
		{"max_drop_vs_active", 0.80, 0.83, 0.005, false, "failed"},
		{"max_rise_vs_active", 0.35, 0.28, 0.10, false, "passed"},
		{"max_rise_vs_active", 0.45, 0.28, 0.10, false, "failed"},
		{"abs_floor", 0.50, 0, 0.50, false, "passed"},
		{"abs_floor", 0.50, 0, 0.50, true, "failed"},
		{"abs_ceiling", 0.10, 0, 0.10, false, "passed"},
		{"abs_ceiling", 0.10, 0, 0.10, true, "failed"},
	}
	for _, testCase := range cases {
		clauses := []Clause{{
			ClauseKey: "case", MetricPath: "aggregate.metric", Comparator: testCase.comparator,
			Value: testCase.value, Params: map[string]any{"strict": testCase.strict}, Enabled: true,
		}}
		candidate := run(Blob{"aggregate": map[string]any{"metric": testCase.candidate}})
		baseline := run(Blob{"aggregate": map[string]any{"metric": testCase.baseline}})
		result := Evaluate(clauses, candidate, baseline, nil)
		got := result.Clauses[0].Outcome
		if got != testCase.want {
			t.Fatalf("%s cand=%v base=%v strict=%v: outcome %s, want %s",
				testCase.comparator, testCase.candidate, testCase.baseline, testCase.strict, got, testCase.want)
		}
	}
}

func TestGateA(t *testing.T) {
	t.Parallel()
	policy := Policy{MinReviewed: 50, MinNewObjects: 200, MinPerClassObjects: map[string]int64{"prairie_dog": 20, "burrow": 20}, MinDays: 3}
	passing := true
	ready := EvaluateGateA(policy, GateACounts{
		ReviewedImagesSinceActive: 62, NewTotalObjects: 214,
		NewPerClassObjects: map[string]int64{"prairie_dog": 21, "burrow": 25},
		DaysSinceLastTrain: 10, ActivePassesGold: &passing,
	})
	if !ready.Ready {
		t.Fatalf("gate should be ready; reasons: %v", ready.Reasons)
	}
	blocked := EvaluateGateA(policy, GateACounts{
		ReviewedImagesSinceActive: 62, NewTotalObjects: 214,
		NewPerClassObjects: map[string]int64{"prairie_dog": 7, "burrow": 25},
		DaysSinceLastTrain: 10, ActivePassesGold: nil,
	})
	if blocked.Ready {
		t.Fatal("gate must be blocked")
	}
	joined := strings.Join(blocked.Reasons, " | ")
	if !strings.Contains(joined, "prairie_dog labels: 7 of 20") {
		t.Fatalf("missing per-class reason with numbers: %s", joined)
	}
	if !strings.Contains(joined, "cannot check the gold precondition") {
		t.Fatalf("missing the fixed precondition reason: %s", joined)
	}
	notPassing := false
	broken := EvaluateGateA(policy, GateACounts{
		ReviewedImagesSinceActive: 100, NewTotalObjects: 300,
		NewPerClassObjects: map[string]int64{"prairie_dog": 30, "burrow": 30},
		DaysSinceLastTrain: 10, ActivePassesGold: &notPassing,
	})
	if broken.Ready {
		t.Fatal("never retrain off a broken base")
	}
}
