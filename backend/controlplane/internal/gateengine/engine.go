// Package gateengine is the generic, fail-closed Gate B evaluator of the
// GoldGate plan (section 7.6): Gate B = AND over a model's ENABLED guardrail
// clause rows, candidate vs pinned-active metrics on the identical gold
// content hash, emitting reasons-with-numbers for every failed clause and a
// recorded exclusion for every under-supported scope. "Config not code" can
// never mean "vacuously passable": the engine's invariants make an empty
// clause set, a missing metric, or a kernel/schema skew a FAILURE, never a
// silent pass. The package is deliberately store-free and model-free — a
// segmentation clause set is a different input, zero new code (proven by
// test).
package gateengine

import (
	"fmt"
	"sort"
	"strings"
)

type Clause struct {
	ClauseKey  string
	MetricPath string
	Comparator string // max_drop_vs_active | abs_floor | max_rise_vs_active | abs_ceiling
	Value      float64
	Slice      string
	Params     map[string]any // {min_label_count, strict, ...}
	Enabled    bool
	Required   bool
}

// Blob is a benchmark metrics document (detection.v1 / segmentation.v1 / ...).
type Blob = map[string]any

type Run struct {
	Metrics       Blob
	MetricSchema  string
	KernelVersion string
	GoldHash      string
}

// ClauseOutcome is the PINNED metadata.guardrails.clauses[] element shape the
// UI renders data-driven (plan section 7.6, v1.3) — one element per evaluated
// clause expansion (per-class wildcards emit one per class).
type ClauseOutcome struct {
	ClauseKey      string   `json:"clause_key"`
	MetricPath     string   `json:"metric_path"`
	Slice          string   `json:"slice,omitempty"`
	Comparator     string   `json:"comparator"`
	Value          float64  `json:"value"`
	CandidateValue *float64 `json:"candidate_value"`
	BaselineValue  *float64 `json:"baseline_value"`
	Outcome        string   `json:"outcome"` // passed | failed | excluded
	Reason         string   `json:"reason,omitempty"`
}

type Result struct {
	Passed  bool
	Reasons []string
	Clauses []ClauseOutcome
}

// Evaluate runs Gate B. classNames drives per-class wildcard expansion (the
// manifest's class names, plan section 7.6 — never inferred from the blob so
// a class the candidate silently DROPPED still gets evaluated and fails).
func Evaluate(clauses []Clause, candidate Run, baseline Run, classNames []string) Result {
	result := Result{Reasons: []string{}, Clauses: []ClauseOutcome{}}

	// Invariant 3: no mixed-version comparison — kernel/schema skew is a
	// re-baselining event, not a silent comparison.
	if candidate.MetricSchema != baseline.MetricSchema || candidate.KernelVersion != baseline.KernelVersion {
		result.Passed = false
		result.Reasons = append(result.Reasons, fmt.Sprintf(
			"refusing mixed-version comparison: candidate %s/%s vs baseline %s/%s - re-baseline the active model",
			candidate.MetricSchema, candidate.KernelVersion, baseline.MetricSchema, baseline.KernelVersion))
		return result
	}
	// Comparing across different gold hashes is meaningless; the caller pins
	// both to the current frozen hash, and the engine refuses drift.
	if candidate.GoldHash != "" && baseline.GoldHash != "" && candidate.GoldHash != baseline.GoldHash {
		result.Passed = false
		result.Reasons = append(result.Reasons,
			"no comparable baseline: candidate and baseline were benchmarked on different gold content hashes")
		return result
	}

	enabled := make([]Clause, 0, len(clauses))
	for _, clause := range clauses {
		if clause.Enabled {
			enabled = append(enabled, clause)
		}
	}
	// Invariant 1: the empty-set rule — a mis-keyed model, a wiped seed, or a
	// fully-disabled set can never vacuously pass.
	if len(enabled) == 0 {
		result.Passed = false
		result.Reasons = append(result.Reasons, "no enabled guardrail clauses for model_key")
		return result
	}

	anyFailed := false
	for _, clause := range enabled {
		expansions := expandClause(clause, classNames)
		if len(expansions) == 0 {
			reason := fmt.Sprintf("clause %s (%s) cannot be evaluated: no authoritative manifest class names", clause.ClauseKey, clause.MetricPath)
			result.Clauses = append(result.Clauses, ClauseOutcome{
				ClauseKey:  clause.ClauseKey,
				MetricPath: clause.MetricPath,
				Slice:      clause.Slice,
				Comparator: clause.Comparator,
				Value:      clause.Value,
				Outcome:    "failed",
				Reason:     reason,
			})
			anyFailed = true
			result.Reasons = append(result.Reasons, reason)
			continue
		}
		for _, expansion := range expansions {
			outcome := evaluateExpansion(expansion, candidate.Metrics, baseline.Metrics)
			result.Clauses = append(result.Clauses, outcome)
			switch outcome.Outcome {
			case "failed":
				anyFailed = true
				result.Reasons = append(result.Reasons, outcome.Reason)
			case "excluded":
				result.Reasons = append(result.Reasons, outcome.Reason)
			}
		}
	}
	result.Passed = !anyFailed
	return result
}

type expansion struct {
	Clause
	className string
}

func expandClause(clause Clause, classNames []string) []expansion {
	if !strings.Contains(clause.MetricPath, "per_class.*") {
		return []expansion{{Clause: clause}}
	}
	names := append([]string(nil), classNames...)
	sort.Strings(names)
	expansions := make([]expansion, 0, len(names))
	for _, name := range names {
		expanded := clause
		expanded.MetricPath = strings.Replace(clause.MetricPath, "per_class.*", "per_class."+name, 1)
		expansions = append(expansions, expansion{Clause: expanded, className: name})
	}
	return expansions
}

func evaluateExpansion(clause expansion, candidate Blob, baseline Blob) ClauseOutcome {
	outcome := ClauseOutcome{
		ClauseKey:  clause.ClauseKey,
		MetricPath: clause.MetricPath,
		Slice:      clause.Slice,
		Comparator: clause.Comparator,
		Value:      clause.Value,
	}
	scope, scopeLabel := clauseScope(clause.MetricPath)

	// Invariant 7 (evaluated BEFORE the no-comparable-baseline rule): an
	// under-supported or EMPTY scope is recorded-excluded, never failed — a
	// pending_new_survey gold set must not turn the gate permanently red.
	labelCount, labelCountPresent := lookupNumber(candidate, scope+".label_count")
	minLabelCount := clauseMinLabelCount(clause.Params)
	if scope != "" && labelCountPresent {
		if labelCount == 0 {
			outcome.Outcome = "excluded"
			outcome.Reason = fmt.Sprintf("%s excluded - no gold labels in scope (label_count 0)", scopeLabel)
			return outcome
		}
		if minLabelCount > 0 && labelCount < float64(minLabelCount) {
			outcome.Outcome = "excluded"
			outcome.Reason = fmt.Sprintf("%s excluded - under %d gold labels (label_count %.0f)", scopeLabel, minLabelCount, labelCount)
			return outcome
		}
	}

	// Invariant 6: a class predicted ZERO times is forgetting - FAIL; the
	// absent-labels side was handled above as EXCLUDE.
	if clause.className != "" {
		if predicted, ok := lookupNumber(candidate, scope+".predicted_count"); ok && predicted == 0 {
			outcome.Outcome = "failed"
			outcome.Reason = fmt.Sprintf("%s predicted zero times by the candidate (forgetting)", clause.className)
			return outcome
		}
	}

	candidateValue, candidateOK := lookupNumber(candidate, clause.MetricPath)
	baselineValue, baselineOK := lookupNumber(baseline, clause.MetricPath)
	if candidateOK {
		outcome.CandidateValue = &candidateValue
	}
	if baselineOK {
		outcome.BaselineValue = &baselineValue
	}
	needsBaseline := clause.Comparator == "max_drop_vs_active" || clause.Comparator == "max_rise_vs_active"
	// Invariant 2: a missing metric on either side fails closed and directs
	// to re-baselining (plan section 7.5).
	if !candidateOK || (needsBaseline && !baselineOK) {
		outcome.Outcome = "failed"
		outcome.Reason = fmt.Sprintf("no comparable baseline for %s - run the benchmark on the active model first (section 7.5)", clause.MetricPath)
		return outcome
	}

	strict := clauseStrict(clause.Params)
	passed := false
	var reason string
	switch clause.Comparator {
	case "max_drop_vs_active":
		passed = candidateValue >= baselineValue-clause.Value
		if !passed {
			reason = fmt.Sprintf("regressed on %s: %.4g vs %.4g active (drop %.4g > %.4g tolerance)",
				clause.MetricPath, candidateValue, baselineValue, baselineValue-candidateValue, clause.Value)
		}
	case "max_rise_vs_active":
		passed = candidateValue <= baselineValue+clause.Value
		if !passed {
			reason = fmt.Sprintf("rose on %s: %.4g vs %.4g active (rise %.4g > %.4g tolerance)",
				clause.MetricPath, candidateValue, baselineValue, candidateValue-baselineValue, clause.Value)
		}
	case "abs_floor":
		if strict {
			passed = candidateValue > clause.Value
		} else {
			passed = candidateValue >= clause.Value
		}
		if !passed {
			reason = fmt.Sprintf("%s %.4g below absolute floor %.4g", clause.MetricPath, candidateValue, clause.Value)
		}
	case "abs_ceiling":
		if strict {
			passed = candidateValue < clause.Value
		} else {
			passed = candidateValue <= clause.Value
		}
		if !passed {
			reason = fmt.Sprintf("%s %.4g above absolute ceiling %.4g", clause.MetricPath, candidateValue, clause.Value)
		}
	default:
		outcome.Outcome = "failed"
		outcome.Reason = fmt.Sprintf("unknown comparator %q on clause %s (fail closed)", clause.Comparator, clause.ClauseKey)
		return outcome
	}

	if passed {
		outcome.Outcome = "passed"
	} else {
		outcome.Outcome = "failed"
		outcome.Reason = reason
	}
	return outcome
}

// clauseScope returns the blob prefix whose label_count guards this clause's
// support (per_class.<name> or per_slice.<name>), and a human label for the
// exclusion reason. Aggregate clauses have no support scope.
func clauseScope(metricPath string) (string, string) {
	parts := strings.Split(metricPath, ".")
	if len(parts) >= 3 && (parts[0] == "per_class" || parts[0] == "per_slice") {
		return parts[0] + "." + parts[1], parts[1]
	}
	return "", ""
}

func clauseMinLabelCount(params map[string]any) int {
	if params == nil {
		return 0
	}
	switch value := params["min_label_count"].(type) {
	case float64:
		return int(value)
	case int:
		return value
	case int64:
		return int(value)
	default:
		return 0
	}
}

func clauseStrict(params map[string]any) bool {
	if params == nil {
		return false
	}
	strict, _ := params["strict"].(bool)
	return strict
}

// lookupNumber resolves a dotted path into nested JSON maps; a JSON null (or
// any non-number) at the leaf is "absent" — the empty-slice emission contract
// (section 7.3) makes null legal only where label_count==0, which invariant 7
// already excluded before this lookup can fail a clause.
func lookupNumber(blob Blob, path string) (float64, bool) {
	if blob == nil {
		return 0, false
	}
	parts := strings.Split(path, ".")
	var current any = map[string]any(blob)
	for _, part := range parts {
		node, ok := current.(map[string]any)
		if !ok {
			return 0, false
		}
		current, ok = node[part]
		if !ok {
			return 0, false
		}
	}
	switch value := current.(type) {
	case float64:
		return value, true
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	default:
		return 0, false
	}
}

// --- Gate A (data-readiness trigger, section 7.1) ----------------------------

type Policy struct {
	MinReviewed        int64
	MinNewObjects      int64
	MinPerClassObjects map[string]int64
	MinDays            int64
}

type GateACounts struct {
	ReviewedImagesSinceActive int64
	NewTotalObjects           int64
	NewPerClassObjects        map[string]int64
	DaysSinceLastTrain        float64
	// ActivePassesGold is the FIXED ENGINE PRECONDITION (never a policy
	// column): the active version has a passing benchmark on the CURRENT
	// gold content hash. Nil = gold does not exist yet ("cannot check").
	ActivePassesGold *bool
}

type GateAResult struct {
	Ready   bool
	Reasons []string
}

func EvaluateGateA(policy Policy, counts GateACounts) GateAResult {
	result := GateAResult{Reasons: []string{}}
	if counts.ReviewedImagesSinceActive < policy.MinReviewed {
		result.Reasons = append(result.Reasons, fmt.Sprintf(
			"reviewed images since the last train: %d of %d needed", counts.ReviewedImagesSinceActive, policy.MinReviewed))
	}
	if counts.NewTotalObjects < policy.MinNewObjects {
		result.Reasons = append(result.Reasons, fmt.Sprintf(
			"new labeled objects: %d of %d needed", counts.NewTotalObjects, policy.MinNewObjects))
	}
	classNames := make([]string, 0, len(policy.MinPerClassObjects))
	for name := range policy.MinPerClassObjects {
		classNames = append(classNames, name)
	}
	sort.Strings(classNames)
	for _, name := range classNames {
		needed := policy.MinPerClassObjects[name]
		have := counts.NewPerClassObjects[name]
		if have < needed {
			result.Reasons = append(result.Reasons, fmt.Sprintf(
				"new %s labels: %d of %d needed", name, have, needed))
		}
	}
	if counts.DaysSinceLastTrain >= 0 && counts.DaysSinceLastTrain < float64(policy.MinDays) {
		result.Reasons = append(result.Reasons, fmt.Sprintf(
			"last train was %.1f days ago - the debounce requires %d", counts.DaysSinceLastTrain, policy.MinDays))
	}
	switch {
	case counts.ActivePassesGold == nil:
		result.Reasons = append(result.Reasons, "cannot check the gold precondition - no gold set frozen yet")
	case !*counts.ActivePassesGold:
		result.Reasons = append(result.Reasons, "the active model does not pass the current gold set - never retrain off a broken base")
	}
	result.Ready = len(result.Reasons) == 0
	return result
}
