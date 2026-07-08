package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// The GoldGate lifecycle, end to end over the real router + MemoryStore +
// MemoryBus: dispatch (capability-gated), the async gold freeze, benchmark
// stamping through the gate engine (the worker never self-grades), the
// server-vetoed promote, the held-out override, and the ungated rollback.

func newTrainingTestRouter(t *testing.T) (http.Handler, *store.MemoryStore, *eventbus.MemoryBus) {
	t.Helper()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test", Runs: service, Store: mem, TrainingJobs: bus})
	return router, mem, bus
}

func doJSON(t *testing.T, router http.Handler, method, path string, body any) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	var reader *bytes.Reader
	if body != nil {
		payload, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		reader = bytes.NewReader(payload)
	} else {
		reader = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, reader)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	decoded := map[string]any{}
	if len(rec.Body.Bytes()) > 0 {
		_ = json.Unmarshal(rec.Body.Bytes(), &decoded)
	}
	return rec, decoded
}

func passingDetectionMetrics(map50 float64) domain.JSONMap {
	return domain.JSONMap{
		"aggregate": map[string]any{"map50": map50, "map50_95": map50 - 0.2, "fp_per_empty_frame": 0.28, "precision_at_op": 0.8, "recall_at_op": 0.65},
		"per_class": map[string]any{
			"prairie_dog": map[string]any{"recall_at_op": 0.62, "ap50": map50 - 0.05, "predicted_count": float64(40), "label_count": float64(60)},
			"burrow":      map[string]any{"recall_at_op": 0.71, "ap50": map50, "predicted_count": float64(220), "label_count": float64(240)},
		},
		"per_slice": map[string]any{
			"prior_train":   map[string]any{"map50": map50 - 0.01, "label_count": float64(300)},
			"held_out_test": map[string]any{"map50": nil, "label_count": float64(0)},
		},
	}
}

func postBenchmarkResult(t *testing.T, router http.Handler, modelKey, versionID, goldSetID, goldHash string, metrics domain.JSONMap) map[string]any {
	t.Helper()
	// Dispatch a benchmark job first (the worker reports against its job id).
	rec, dispatch := doJSON(t, router, http.MethodPost, "/v2/training/models/"+modelKey+"/benchmark/run",
		map[string]any{"version_id": versionID})
	if rec.Code != http.StatusAccepted {
		t.Fatalf("benchmark dispatch = %d body=%s", rec.Code, rec.Body.String())
	}
	jobID, _ := dispatch["job_id"].(string)
	rec, result := doJSON(t, router, http.MethodPost, "/v2/training/jobs/"+jobID+"/benchmark-result", map[string]any{
		"model_version_id":      versionID,
		"gold_set_id":           goldSetID,
		"gold_set_content_hash": goldHash,
		"metric_schema":         "detection.v1",
		"kernel_version":        "yolov5_two_pass/v1",
		"metrics":               metrics,
		"report_uri":            "file:///tmp/report-" + versionID + ".json",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("benchmark-result = %d body=%s", rec.Code, rec.Body.String())
	}
	return result
}

// freezeGoldViaAPI walks draft -> freeze dispatch -> worker gold-result, the
// full async freeze (section 4.6), and returns (goldSetID, contentHash).
func freezeGoldViaAPI(t *testing.T, router http.Handler, modelKey string, heldOutState string) (string, string) {
	t.Helper()
	rec, draft := doJSON(t, router, http.MethodPost, "/v2/training/models/"+modelKey+"/gold-sets", map[string]any{})
	if rec.Code != http.StatusCreated {
		t.Fatalf("gold draft = %d body=%s", rec.Code, rec.Body.String())
	}
	goldSetID, _ := draft["gold_set_id"].(string)
	rec, dispatch := doJSON(t, router, http.MethodPost, "/v2/training/models/"+modelKey+"/gold-sets/"+goldSetID+"/freeze", map[string]any{"confirm": true})
	if rec.Code != http.StatusAccepted {
		t.Fatalf("freeze dispatch = %d body=%s", rec.Code, rec.Body.String())
	}
	jobID, _ := dispatch["job_id"].(string)
	contentHash := "hash-" + goldSetID
	items := []map[string]any{}
	for index := 0; index < 3; index++ {
		items = append(items, map[string]any{
			"item_id":         fmt.Sprintf("%s-frame-%d", goldSetID, index),
			"source_ref":      map[string]any{"bisque_image_id": fmt.Sprintf("00-%s-%d", modelKey, index)},
			"slice":           "prior_train",
			"label_kind":      "boxes",
			"content_sha256":  fmt.Sprintf("sha-%s-%d", goldSetID, index),
			"phash":           "00000000000000aa",
			"gt_label_sha256": fmt.Sprintf("label-sha-%d", index),
			"gt_label_uri":    "file:///tmp/labels.txt",
			"strata_tags":     map[string]any{"class_presence": "both"},
		})
	}
	rec, _ = doJSON(t, router, http.MethodPost, "/v2/training/jobs/"+jobID+"/gold-result", map[string]any{
		"gold_set_id":        goldSetID,
		"status":             "frozen",
		"content_hash":       contentHash,
		"label_stats":        map[string]any{"box_count": 42},
		"strata_summary":     map[string]any{"prior_train": 3},
		"provenance":         map[string]any{"held_out_state": heldOutState, "prior_slice_provenance": "recovered_train_v0"},
		"split_manifest_uri": "file:///tmp/gold-manifest.json",
		"items":              items,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("gold-result = %d body=%s", rec.Code, rec.Body.String())
	}
	return goldSetID, contentHash
}

func TestTrainingDispatchCapabilityGateAndPublish(t *testing.T) {
	t.Parallel()
	router, _, bus := newTrainingTestRouter(t)

	rec, body := doJSON(t, router, http.MethodPost, "/v2/training/models/yolov5_rarespot/sync", nil)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("sync dispatch = %d body=%s", rec.Code, rec.Body.String())
	}
	if body["job_type"] != "training.sync" || body["model_key"] != "yolov5_rarespot" {
		t.Fatalf("dispatch envelope wrong: %#v", body)
	}
	jobs := bus.TrainingJobs()
	if len(jobs) != 1 || jobs[0].JobType != "training.sync" {
		t.Fatalf("published jobs = %#v", jobs)
	}

	// Unknown model 404s BEFORE publish; a capability the manifest does not
	// declare 409s BEFORE publish.
	rec, _ = doJSON(t, router, http.MethodPost, "/v2/training/models/nope/sync", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown model = %d", rec.Code)
	}
	if len(bus.TrainingJobs()) != 1 {
		t.Fatal("nothing may publish for an unknown model")
	}
}

func TestTrainingGoldFreezeLifecycleAndBaseline(t *testing.T) {
	t.Parallel()
	router, mem, _ := newTrainingTestRouter(t)
	goldSetID, goldHash := freezeGoldViaAPI(t, router, "yolov5_rarespot", "pending_new_survey")

	gold, err := mem.GetCurrentTrainingGoldSet(t.Context(), "yolov5_rarespot")
	if err != nil || gold.GoldSetID != goldSetID || gold.ContentHash != goldHash {
		t.Fatalf("frozen gold not current: %+v err=%v", gold, err)
	}
	if gold.ItemCount != 3 || gold.FrozenAt == nil {
		t.Fatalf("gold facts not sealed: %+v", gold)
	}

	// Baseline pin (section 7.5): benchmark the ACTIVE v0 - it compares
	// against itself and must pass; readiness lights up on the status read.
	result := postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v0", goldSetID, goldHash, passingDetectionMetrics(0.83))
	if passed, _ := result["guardrails_passed"].(bool); !passed {
		t.Fatalf("baseline must pass vs itself: %#v", result)
	}
	rec, status := doJSON(t, router, http.MethodGet, "/v2/training/models/yolov5_rarespot/status", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if ready, _ := status["canonical_benchmark_ready"].(bool); !ready {
		t.Fatalf("benchmark readiness must derive from the stamped guardrails: %#v", status["canonical_benchmark_ready"])
	}
	if status["last_benchmark_at"] == nil {
		t.Fatal("last_benchmark_at must come from guardrails.benchmarked_at")
	}
}

func TestTrainingPromoteVetoOverrideAndRollback(t *testing.T) {
	t.Parallel()
	router, mem, _ := newTrainingTestRouter(t)
	goldSetID, goldHash := freezeGoldViaAPI(t, router, "yolov5_rarespot", "pending_new_survey")
	// Pin the baseline.
	postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v0", goldSetID, goldHash, passingDetectionMetrics(0.83))

	// Register an externally-trained candidate (section 3.6) - guilty until
	// benchmarked: promote must 409 gate_failed.
	rec, _ := doJSON(t, router, http.MethodPost, "/v2/training/models/yolov5_rarespot/versions", map[string]any{
		"version_id":  "yolov5_rarespot-v1",
		"weights_uri": "barrel:/mnt/barrel-data/ultra/training/weights/yolov5_rarespot/v1/weights.pt",
		"provenance":  map[string]any{"trained_by": "offline run", "hyp_sha256": "abc"},
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("register version = %d body=%s", rec.Code, rec.Body.String())
	}
	rec, veto := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote", nil)
	if rec.Code != http.StatusConflict || veto["error"] != "gate_failed" {
		t.Fatalf("unbenchmarked promote must 409 gate_failed, got %d %#v", rec.Code, veto)
	}

	// A FAILING benchmark stamps a failed verdict; promote still 409s.
	failing := passingDetectionMetrics(0.60)
	postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v1", goldSetID, goldHash, failing)
	rec, _ = doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote", nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("failed-gate promote must 409, got %d", rec.Code)
	}

	// A PASSING benchmark promotes to canary; the second promote (canary ->
	// active) hits the held-out override guard (section 8.2), succeeds with an
	// audited override, and stamps activated_at + the audit event.
	postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v1", goldSetID, goldHash, passingDetectionMetrics(0.84))
	rec, promoted := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote", nil)
	if rec.Code != http.StatusOK || promoted["status"] != "canary" {
		t.Fatalf("promote to canary = %d %#v", rec.Code, promoted)
	}
	rec, veto = doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote", nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("canary->active while held-out pending must 409 without override, got %d", rec.Code)
	}
	if !strings.Contains(fmt.Sprint(veto["reasons"]), "override_reason") {
		t.Fatalf("override reason missing from veto: %#v", veto)
	}
	rec, promoted = doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote",
		map[string]any{"override_reason": "field season deadline - drift watched manually"})
	if rec.Code != http.StatusOK || promoted["status"] != "active" {
		t.Fatalf("override promote = %d %#v", rec.Code, promoted)
	}
	if promoted["activated_at"] == nil {
		t.Fatal("activated_at must be stamped in the graduation transaction")
	}
	events, err := mem.ListTrainingModelVersionEvents(t.Context(), "yolov5_rarespot-v1", 10)
	if err != nil {
		t.Fatalf("events: %v", err)
	}
	foundOverride := false
	for _, event := range events {
		if event.EventType == "promote_with_pending_held_out" {
			foundOverride = true
		}
	}
	if !foundOverride {
		t.Fatalf("override must write the distinct audit event; got %#v", events)
	}
	// The lineage pointer + status row moved with the promotion.
	rec, status := doJSON(t, router, http.MethodGet, "/v2/training/models/yolov5_rarespot/status", nil)
	if rec.Code != http.StatusOK || status["active_model_version"] != "yolov5_rarespot-v1" {
		t.Fatalf("status active pointer = %#v", status["active_model_version"])
	}

	// Rollback: instant, UNGATED, back to v0.
	rec, restored := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/rollback", nil)
	if rec.Code != http.StatusOK || restored["version_id"] != "yolov5_rarespot-v0" || restored["status"] != "active" {
		t.Fatalf("rollback = %d %#v", rec.Code, restored)
	}
	rec, status = doJSON(t, router, http.MethodGet, "/v2/training/models/yolov5_rarespot/status", nil)
	if status["active_model_version"] != "yolov5_rarespot-v0" {
		t.Fatalf("rollback did not repoint status: %#v", status["active_model_version"])
	}
}

// TestTrainingStaleGoldHashCannotPromote: re-checking the HASH inside the
// transaction defends against a stale-passing candidate after gold changed.
func TestTrainingStaleGoldHashCannotPromote(t *testing.T) {
	t.Parallel()
	router, mem, _ := newTrainingTestRouter(t)
	goldSetID, goldHash := freezeGoldViaAPI(t, router, "yolov5_rarespot", "populated")
	postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v0", goldSetID, goldHash, passingDetectionMetrics(0.83))
	rec, _ := doJSON(t, router, http.MethodPost, "/v2/training/models/yolov5_rarespot/versions", map[string]any{
		"version_id": "yolov5_rarespot-v1", "weights_uri": "barrel:/x.pt", "provenance": map[string]any{"src": "test"},
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("register = %d", rec.Code)
	}
	postBenchmarkResult(t, router, "yolov5_rarespot", "yolov5_rarespot-v1", goldSetID, goldHash, passingDetectionMetrics(0.84))

	// Gold v2 freezes AFTER the candidate's passing benchmark: retire v1 gold
	// and freeze a new one with a different hash.
	if _, err := mem.UpdateTrainingGoldSetState(t.Context(), domain.UpdateTrainingGoldSetStateInput{GoldSetID: goldSetID, Status: "retired"}); err != nil {
		t.Fatalf("retire gold: %v", err)
	}
	_, _ = freezeGoldViaAPI(t, router, "yolov5_rarespot", "populated")

	rec, veto := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/yolov5_rarespot-v1/promote", nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("stale-hash promote must 409, got %d %#v", rec.Code, veto)
	}
	if !strings.Contains(fmt.Sprint(veto["reasons"]), "current gold content hash") {
		t.Fatalf("wrong veto reasons: %#v", veto)
	}
}

// TestTrainingSeedReproducesV1Clauses is M2 proof 4: the seeded clause rows
// reproduce the five v1 clauses to the number (values, comparators, required).
func TestTrainingSeedReproducesV1Clauses(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	clauses, err := mem.ListTrainingGuardrailClauses(t.Context(), "yolov5_rarespot")
	if err != nil {
		t.Fatalf("clauses: %v", err)
	}
	expected := map[string]struct {
		path       string
		comparator string
		value      float64
	}{
		"agg_map50":           {"aggregate.map50", "max_drop_vs_active", 0.005},
		"agg_map50_95":        {"aggregate.map50_95", "max_drop_vs_active", 0.005},
		"class_recall_delta":  {"per_class.*.recall_at_op", "max_drop_vs_active", 0.02},
		"class_recall_abs":    {"per_class.*.recall_at_op", "abs_floor", 0.50},
		"slice_prior_map50":   {"per_slice.prior_train.map50", "max_drop_vs_active", 0.02},
		"slice_held_map50":    {"per_slice.held_out_test.map50", "max_drop_vs_active", 0.005},
		"class_ap50_collapse": {"per_class.*.ap50", "max_drop_vs_active", 0.05},
		"class_ap50_abs":      {"per_class.*.ap50", "abs_floor", 0.10},
		"fp_empty_ceiling":    {"aggregate.fp_per_empty_frame", "max_rise_vs_active", 0.10},
		"precision_delta":     {"aggregate.precision_at_op", "max_drop_vs_active", 0.03},
	}
	if len(clauses) != len(expected) {
		t.Fatalf("clause count = %d, want %d", len(clauses), len(expected))
	}
	for _, clause := range clauses {
		want, ok := expected[clause.ClauseKey]
		if !ok {
			t.Fatalf("unexpected clause %q", clause.ClauseKey)
		}
		if clause.MetricPath != want.path || clause.Comparator != want.comparator || clause.Value != want.value {
			t.Fatalf("clause %s = {%s %s %v}, want {%s %s %v}",
				clause.ClauseKey, clause.MetricPath, clause.Comparator, clause.Value, want.path, want.comparator, want.value)
		}
		if !clause.Required || !clause.Enabled {
			t.Fatalf("clause %s must be required+enabled (v1's 'PASS requires ALL FIVE')", clause.ClauseKey)
		}
	}
}

// TestMegaSegAcceptanceWalk is the M5 acceptance test (plan section 13): a
// SECOND model onboards with the documented seed only - zero ALTERs, zero
// route additions, zero enum edits, zero out-of-band inserts - and walks
// gold-freeze -> external registration -> benchmark -> canary -> active ->
// rollback through the SAME generic surface, with segmentation clause rows.
func TestMegaSegAcceptanceWalk(t *testing.T) {
	t.Parallel()
	router, mem, _ := newTrainingTestRouter(t)
	now := time.Now().UTC()

	// The documented seed migration (the only permitted direct writes).
	mem.SeedTrainingModel(
		domain.TrainingModelRecord{
			ModelKey: "megaseg", TaskType: "segmentation", DisplayName: "MegaSeg",
			DatasetFormat: "volume_mask_pairs", MetricSchema: "segmentation.v1",
			Capabilities: []string{"BENCHMARK"},
			Executor:     domain.JSONMap{"gpu_pool": "titan"},
			Classes:      domain.JSONMap{"0": "nucleus"},
			Metadata:     domain.JSONMap{"framework": "PyTorch"},
			CreatedAt:    now,
		},
		domain.TrainingDomainRecord{DomainID: "cellbio", Name: "Cell biology", Metadata: domain.JSONMap{}, CreatedAt: now, UpdatedAt: now},
		domain.TrainingLineageRecord{LineageID: "megaseg-shared", DomainID: "cellbio", ModelKey: "megaseg", Scope: "shared", ActiveVersionID: "megaseg-v0", Metadata: domain.JSONMap{}, CreatedAt: now, UpdatedAt: now},
		domain.TrainingModelVersionRecord{VersionID: "megaseg-v0", LineageID: "megaseg-shared", ModelKey: "megaseg", Status: "active", IsFrozen: true, Metrics: domain.JSONMap{}, Metadata: domain.JSONMap{}, CreatedAt: now, UpdatedAt: now},
		domain.TrainingModelStatusRecord{ModelKey: "megaseg", DatasetName: "curated volumes", ActiveModelVersion: "megaseg-v0", ClassCounts: domain.JSONMap{}, PerClassNewObjects: domain.JSONMap{}, UnsupportedClassCounts: domain.JSONMap{}, RetrainGateReasons: []string{}, RetrainGateCounts: domain.JSONMap{}, RetrainGateThresholds: domain.JSONMap{}},
		domain.TrainingGatePolicyRecord{ModelKey: "megaseg", MinReviewed: 1, MinNewObjects: 1, MinPerClassObjects: domain.JSONMap{}, MinDays: 0},
		[]domain.TrainingGuardrailClauseRecord{
			{ModelKey: "megaseg", ClauseKey: "agg_miou", MetricPath: "aggregate.miou", Comparator: "max_drop_vs_active", Value: 0.01, Enabled: true, Required: true},
			{ModelKey: "megaseg", ClauseKey: "class_dice_abs", MetricPath: "per_class.*.dice", Comparator: "abs_floor", Value: 0.60, Enabled: true, Required: true},
		},
	)

	// FINETUNE is not declared: retrain/sync dispatch must 409 BEFORE publish.
	rec, _ := doJSON(t, router, http.MethodPost, "/v2/training/models/megaseg/sync", nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("benchmark-only sync must 409 unsupported_capability, got %d", rec.Code)
	}

	// Gold materializes THROUGH the gold_freeze job (BENCHMARK capability).
	goldSetID, goldHash := freezeGoldViaAPI(t, router, "megaseg", "populated")

	segMetrics := func(miou float64, dice float64) domain.JSONMap {
		return domain.JSONMap{
			"aggregate": map[string]any{"miou": miou},
			"per_class": map[string]any{"nucleus": map[string]any{"dice": dice, "predicted_count": float64(9_000_000), "label_count": float64(10_000_000)}},
			"per_slice": map[string]any{"prior_train": map[string]any{"miou": miou - 0.005, "label_count": float64(12)}},
		}
	}
	// Baseline pin for megaseg-v0.
	baseline := postBenchmarkResult(t, router, "megaseg", "megaseg-v0", goldSetID, goldHash, segMetrics(0.80, 0.70))
	if passed, _ := baseline["guardrails_passed"].(bool); !passed {
		t.Fatalf("megaseg baseline must pass vs itself: %#v", baseline)
	}

	// Externally-trained candidate via the SAME generic route.
	rec, _ = doJSON(t, router, http.MethodPost, "/v2/training/models/megaseg/versions", map[string]any{
		"version_id": "megaseg-v1", "weights_uri": "barrel:/mnt/barrel-data/ultra/training/weights/megaseg/v1.pt",
		"provenance": map[string]any{"trained_by": "external lab run"},
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("register megaseg candidate = %d body=%s", rec.Code, rec.Body.String())
	}
	postBenchmarkResult(t, router, "megaseg", "megaseg-v1", goldSetID, goldHash, segMetrics(0.81, 0.72))

	// candidate -> canary -> active (held-out populated: no override needed).
	rec, promoted := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/megaseg-v1/promote", nil)
	if rec.Code != http.StatusOK || promoted["status"] != "canary" {
		t.Fatalf("megaseg promote canary = %d %#v", rec.Code, promoted)
	}
	rec, promoted = doJSON(t, router, http.MethodPost, "/v2/training/model-versions/megaseg-v1/promote", nil)
	if rec.Code != http.StatusOK || promoted["status"] != "active" {
		t.Fatalf("megaseg promote active = %d %#v", rec.Code, promoted)
	}
	// Rollback closes the walk.
	rec, restored := doJSON(t, router, http.MethodPost, "/v2/training/model-versions/megaseg-v1/rollback", nil)
	if rec.Code != http.StatusOK || restored["version_id"] != "megaseg-v0" {
		t.Fatalf("megaseg rollback = %d %#v", rec.Code, restored)
	}
	// And the registry lists both models - the switcher's multi-model seam.
	models, err := mem.ListTrainingModels(t.Context())
	if err != nil || len(models) != 2 {
		t.Fatalf("registry should list 2 models, got %d err=%v", len(models), err)
	}
}

func TestTrainingLeaseAcceptsPythonWireFloats(t *testing.T) {
	t.Parallel()
	router, _, _ := newTrainingTestRouter(t)

	rec, dispatch := doJSON(t, router, http.MethodPost, "/v2/training/models/yolov5_rarespot/sync", nil)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("sync dispatch = %d body=%s", rec.Code, rec.Body.String())
	}
	jobID, _ := dispatch["job_id"].(string)

	// EXACT wire bytes: the Python client json.dumps()-es its float settings,
	// so ttl_seconds arrives as "3600.0". An int-typed request struct 400s on
	// that and silently wedges the whole training queue (observed live) - the
	// raw string literals here are the contract, do not switch to doJSON.
	post := func(method, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, "/v2/training/jobs/"+jobID+"/lease", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	acquire := post(http.MethodPost, `{"worker_id": "ultra-training-worker@test:1", "ttl_seconds": 3600.0}`)
	if acquire.Code != http.StatusOK {
		t.Fatalf("acquire with float ttl = %d body=%s", acquire.Code, acquire.Body.String())
	}
	var lease struct {
		LeaseToken string `json:"lease_token"`
	}
	if err := json.Unmarshal(acquire.Body.Bytes(), &lease); err != nil || lease.LeaseToken == "" {
		t.Fatalf("acquire response lease_token missing: %s", acquire.Body.String())
	}

	renew := post(http.MethodPatch, fmt.Sprintf(`{"lease_token": %q, "ttl_seconds": 3600.0}`, lease.LeaseToken))
	if renew.Code != http.StatusOK {
		t.Fatalf("renew with float ttl = %d body=%s", renew.Code, renew.Body.String())
	}

	release := post(http.MethodDelete, fmt.Sprintf(`{"lease_token": %q}`, lease.LeaseToken))
	if release.Code != http.StatusOK {
		t.Fatalf("release = %d body=%s", release.Code, release.Body.String())
	}
}
