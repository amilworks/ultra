package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// The M0 key-match + coherence gate: the one canonical model-key spelling and
// the seeded rows must agree with the shared cross-language fixture, and the
// lineage walk the frontend performs must resolve without silent nulls
// (active_model_version is a VERSION id that exists, never the model key).

type trainingContractFixture struct {
	JobPhases []string `json:"job_phases"`
	Manifest  struct {
		ModelKey             string            `json:"model_key"`
		TaskType             string            `json:"task_type"`
		DisplayName          string            `json:"display_name"`
		DatasetFormat        string            `json:"dataset_format"`
		MetricSchema         string            `json:"metric_schema"`
		RequiresPhash        bool              `json:"requires_phash"`
		Capabilities         []string          `json:"capabilities"`
		Classes              map[string]string `json:"classes"`
		LeakageDefensesExtra []string          `json:"leakage_defenses_extra"`
		GTLayerPriority      []string          `json:"gt_layer_priority"`
		Executor             map[string]any    `json:"executor"`
	} `json:"manifest"`
	Seed struct {
		VersionZero struct {
			VersionID string `json:"version_id"`
			Status    string `json:"status"`
			IsFrozen  bool   `json:"is_frozen"`
		} `json:"version_zero"`
		Lineage struct {
			LineageID       string `json:"lineage_id"`
			DomainID        string `json:"domain_id"`
			Scope           string `json:"scope"`
			ActiveVersionID string `json:"active_version_id"`
		} `json:"lineage"`
		GatePolicy struct {
			MinReviewed        int            `json:"min_reviewed"`
			MinNewObjects      int            `json:"min_new_objects"`
			MinPerClassObjects map[string]int `json:"min_per_class_objects"`
			MinDays            int            `json:"min_days"`
		} `json:"gate_policy"`
		GuardrailClauseKeys []string `json:"guardrail_clause_keys"`
	} `json:"seed"`
}

func loadTrainingContractFixture(t *testing.T) trainingContractFixture {
	t.Helper()
	path := filepath.Join("..", "..", "..", "contracts", "training", "yolov5_rarespot.manifest.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read shared training contract fixture: %v", err)
	}
	var fixture trainingContractFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("parse shared training contract fixture: %v", err)
	}
	return fixture
}

func TestTrainingSeedMatchesSharedContractFixture(t *testing.T) {
	t.Parallel()
	fixture := loadTrainingContractFixture(t)
	model := trainingSeedModel(domain.Now())

	if model.ModelKey != fixture.Manifest.ModelKey {
		t.Fatalf("model_key: seed %q != fixture %q", model.ModelKey, fixture.Manifest.ModelKey)
	}
	if TrainingSeedModelKey != fixture.Manifest.ModelKey {
		t.Fatalf("TrainingSeedModelKey %q != fixture %q", TrainingSeedModelKey, fixture.Manifest.ModelKey)
	}
	if model.TaskType != fixture.Manifest.TaskType {
		t.Fatalf("task_type: seed %q != fixture %q", model.TaskType, fixture.Manifest.TaskType)
	}
	if model.DisplayName != fixture.Manifest.DisplayName {
		t.Fatalf("display_name: seed %q != fixture %q", model.DisplayName, fixture.Manifest.DisplayName)
	}
	if model.DatasetFormat != fixture.Manifest.DatasetFormat {
		t.Fatalf("dataset_format: seed %q != fixture %q", model.DatasetFormat, fixture.Manifest.DatasetFormat)
	}
	if model.MetricSchema != fixture.Manifest.MetricSchema {
		t.Fatalf("metric_schema: seed %q != fixture %q", model.MetricSchema, fixture.Manifest.MetricSchema)
	}
	if model.RequiresPhash != fixture.Manifest.RequiresPhash {
		t.Fatalf("requires_phash: seed %v != fixture %v", model.RequiresPhash, fixture.Manifest.RequiresPhash)
	}
	assertStringSliceEqual(t, "capabilities", model.Capabilities, fixture.Manifest.Capabilities)
	assertStringSliceEqual(t, "leakage_defenses_extra", model.LeakageDefensesExtra, fixture.Manifest.LeakageDefensesExtra)
	for id, name := range fixture.Manifest.Classes {
		if got, _ := model.Classes[id].(string); got != name {
			t.Fatalf("classes[%s]: seed %q != fixture %q", id, got, name)
		}
	}
	if len(model.Classes) != len(fixture.Manifest.Classes) {
		t.Fatalf("classes count: seed %d != fixture %d", len(model.Classes), len(fixture.Manifest.Classes))
	}
	layerPriority := metadataAnySliceToStrings(model.Metadata["gt_layer_priority"])
	assertStringSliceEqual(t, "gt_layer_priority", layerPriority, fixture.Manifest.GTLayerPriority)
	if len(fixture.Manifest.Executor) == 0 {
		t.Fatal("fixture executor is empty - the parity gate would be vacuous")
	}
	if len(model.Executor) != len(fixture.Manifest.Executor) {
		t.Fatalf("executor key count: seed %d != fixture %d (parity must be bidirectional)", len(model.Executor), len(fixture.Manifest.Executor))
	}
	for key, want := range fixture.Manifest.Executor {
		got := model.Executor[key]
		wantJSON, _ := json.Marshal(want)
		gotJSON, _ := json.Marshal(got)
		if string(wantJSON) != string(gotJSON) {
			t.Fatalf("executor[%s]: seed %s != fixture %s", key, gotJSON, wantJSON)
		}
	}

	// The Gate A threshold values exist in three seed copies (gate_policies
	// INSERT, the status row's thresholds mirror, trainingSeedStatus). Pin the
	// Go-side mirror to the fixture's canonical gate_policy here; the schema
	// test below pins the SQL literals.
	status := trainingSeedStatus()
	thresholds := status.RetrainGateThresholds
	if int(thresholds["min_reviewed"].(float64)) != fixture.Seed.GatePolicy.MinReviewed ||
		int(thresholds["min_new_objects"].(float64)) != fixture.Seed.GatePolicy.MinNewObjects ||
		int(thresholds["min_days"].(float64)) != fixture.Seed.GatePolicy.MinDays {
		t.Fatalf("retrain_gate_thresholds %v does not match fixture gate_policy %+v", thresholds, fixture.Seed.GatePolicy)
	}
	perClass, _ := thresholds["min_per_class_objects"].(map[string]any)
	if len(perClass) != len(fixture.Seed.GatePolicy.MinPerClassObjects) {
		t.Fatalf("min_per_class_objects key count: seed %d != fixture %d", len(perClass), len(fixture.Seed.GatePolicy.MinPerClassObjects))
	}
	for name, want := range fixture.Seed.GatePolicy.MinPerClassObjects {
		if got, _ := perClass[name].(float64); int(got) != want {
			t.Fatalf("min_per_class_objects[%s]: seed %v != fixture %d", name, perClass[name], want)
		}
	}

	version := trainingSeedVersion(domain.Now())
	if version.VersionID != fixture.Seed.VersionZero.VersionID {
		t.Fatalf("version_zero id: seed %q != fixture %q", version.VersionID, fixture.Seed.VersionZero.VersionID)
	}
	if version.Status != fixture.Seed.VersionZero.Status {
		t.Fatalf("version_zero status: seed %q != fixture %q", version.Status, fixture.Seed.VersionZero.Status)
	}
	if version.IsFrozen != fixture.Seed.VersionZero.IsFrozen {
		t.Fatalf("version_zero is_frozen: seed %v != fixture %v", version.IsFrozen, fixture.Seed.VersionZero.IsFrozen)
	}

	lineage := trainingSeedLineage(domain.Now())
	if lineage.LineageID != fixture.Seed.Lineage.LineageID ||
		lineage.DomainID != fixture.Seed.Lineage.DomainID ||
		lineage.Scope != fixture.Seed.Lineage.Scope ||
		lineage.ActiveVersionID != fixture.Seed.Lineage.ActiveVersionID {
		t.Fatalf("lineage seed %+v does not match fixture %+v", lineage, fixture.Seed.Lineage)
	}
}

// TestTrainingJobPhasesMatchSharedContractFixture is the Go half of the "one
// enum PR" gate: the five phase literals exist once in Go
// (domain.TrainingJobPhases) and once in Python (TRAINING_JOB_PHASES), both
// pinned to the same fixture. Adding a sixth literal anywhere must fail here.
func TestTrainingJobPhasesMatchSharedContractFixture(t *testing.T) {
	t.Parallel()
	fixture := loadTrainingContractFixture(t)
	if len(fixture.JobPhases) == 0 {
		t.Fatal("fixture job_phases is empty - the enum gate would be vacuous")
	}
	assertStringSliceEqual(t, "job_phases", domain.TrainingJobPhases, fixture.JobPhases)
	for _, phase := range fixture.JobPhases {
		if !domain.IsTrainingJobPhase(phase) {
			t.Fatalf("IsTrainingJobPhase(%q) = false for a fixture phase", phase)
		}
	}
	if domain.IsTrainingJobPhase("training.rarespot.finetune") {
		t.Fatal("model-named job types must never validate - phases are generic, model_key travels in the envelope")
	}
}

// TestTrainingSeedCoherence walks the exact path the frontend dashboard walks
// (models -> status -> domains -> lineages -> versions) against the seeded
// MemoryStore and asserts the silent-null bug class is dead: every pointer
// resolves, and active_model_version is an existing version id.
func TestTrainingSeedCoherence(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	models, err := memory.ListTrainingModels(ctx)
	if err != nil || len(models) != 1 {
		t.Fatalf("expected exactly one seeded model, got %d (err %v)", len(models), err)
	}
	modelKey := models[0].ModelKey

	status, err := memory.GetTrainingModelStatus(ctx, modelKey)
	if err != nil {
		t.Fatalf("status for seeded model: %v", err)
	}
	if status.ActiveModelVersion == "" || status.ActiveModelVersion == modelKey {
		t.Fatalf("active_model_version must be a version id, got %q", status.ActiveModelVersion)
	}

	if _, err := memory.GetTrainingModelVersion(ctx, status.ActiveModelVersion); err != nil {
		t.Fatalf("status.active_model_version %q does not resolve to a version row: %v", status.ActiveModelVersion, err)
	}

	domains, err := memory.ListTrainingDomains(ctx, 500, 0)
	if err != nil || len(domains) == 0 {
		t.Fatalf("expected seeded training domain, got %d (err %v)", len(domains), err)
	}
	var lineage domain.TrainingLineageRecord
	found := false
	for _, trainingDomain := range domains {
		lineages, err := memory.ListTrainingLineages(ctx, trainingDomain.DomainID, 500, 0)
		if err != nil {
			t.Fatalf("list lineages: %v", err)
		}
		for _, row := range lineages {
			if row.ModelKey == modelKey && row.Scope == "shared" {
				lineage = row
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("no shared lineage found for model %q - the dashboard walk would silently null", modelKey)
	}
	if lineage.ActiveVersionID != status.ActiveModelVersion {
		t.Fatalf("lineage.active_version_id %q != status.active_model_version %q", lineage.ActiveVersionID, status.ActiveModelVersion)
	}

	versions, err := memory.ListTrainingModelVersions(ctx, lineage.LineageID, 200, 0)
	if err != nil || len(versions) == 0 {
		t.Fatalf("expected seeded versions for lineage %q, got %d (err %v)", lineage.LineageID, len(versions), err)
	}
	foundActive := false
	for _, version := range versions {
		if version.VersionID == status.ActiveModelVersion && version.Status == "active" {
			foundActive = true
		}
	}
	if !foundActive {
		t.Fatalf("versions list for %q does not contain active version %q", lineage.LineageID, status.ActiveModelVersion)
	}
}

// TestTrainingSchemaSeedMatchesFixture pins the schema.sql seed literals to the
// fixture so the SQL seed cannot drift from the Go/Python constants unnoticed.
func TestTrainingSchemaSeedMatchesFixture(t *testing.T) {
	t.Parallel()
	fixture := loadTrainingContractFixture(t)
	raw, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}
	schema := string(raw)
	for _, marker := range []string{
		"'" + fixture.Manifest.ModelKey + "'",
		"'" + fixture.Seed.VersionZero.VersionID + "'",
		"'" + fixture.Seed.Lineage.LineageID + "'",
		"'" + fixture.Seed.Lineage.DomainID + "'",
	} {
		if !containsString(schema, marker) {
			t.Fatalf("schema.sql seed is missing marker %s", marker)
		}
	}
	if len(fixture.Seed.GuardrailClauseKeys) == 0 {
		t.Fatal("fixture guardrail_clause_keys is empty - the clause-drift gate would be vacuous")
	}
	for _, clauseKey := range fixture.Seed.GuardrailClauseKeys {
		if !containsString(schema, "'"+clauseKey+"'") {
			t.Fatalf("schema.sql guardrail clause seed is missing %q", clauseKey)
		}
	}
	// Pin the gate-policy VALUES in the SQL seed (both copies: the policy row
	// and the status thresholds mirror) to the fixture, so a threshold tune in
	// one copy cannot ship silently while prod keeps the other.
	policy := fixture.Seed.GatePolicy
	policyMarker := fmt.Sprintf("VALUES ('%s', %d, %d, '{\"prairie_dog\":%d,\"burrow\":%d}'::jsonb, %d)",
		fixture.Manifest.ModelKey, policy.MinReviewed, policy.MinNewObjects,
		policy.MinPerClassObjects["prairie_dog"], policy.MinPerClassObjects["burrow"], policy.MinDays)
	if !containsString(schema, policyMarker) {
		t.Fatalf("schema.sql gate_policies seed does not match the fixture values (want marker %q)", policyMarker)
	}
	thresholdsMarker := fmt.Sprintf("\"min_reviewed\":%d,\"min_new_objects\":%d,\"min_per_class_objects\":{\"prairie_dog\":%d,\"burrow\":%d},\"min_days\":%d",
		policy.MinReviewed, policy.MinNewObjects,
		policy.MinPerClassObjects["prairie_dog"], policy.MinPerClassObjects["burrow"], policy.MinDays)
	if !containsString(schema, thresholdsMarker) {
		t.Fatalf("schema.sql model_status thresholds mirror does not match the fixture gate_policy (want marker %q)", thresholdsMarker)
	}
	for _, stale := range []string{"rarespot-prairie-yolo", "prairie_yolo"} {
		if containsString(schema, stale) {
			t.Fatalf("schema.sql contains a stale model-key spelling %q", stale)
		}
	}
}

// TestRequiredTablesExistInSchemaSQL cross-checks the boot-blocking table list
// against schema.sql's CREATE TABLE names, so a typo in either is caught here
// instead of at prod boot (VerifyPostgresSchema fails App construction).
func TestRequiredTablesExistInSchemaSQL(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}
	created := map[string]bool{}
	for _, match := range regexp.MustCompile(`CREATE TABLE IF NOT EXISTS (\w+)`).FindAllStringSubmatch(string(raw), -1) {
		created[match[1]] = true
	}
	if len(created) == 0 {
		t.Fatal("no CREATE TABLE statements found in schema.sql")
	}
	for _, table := range requiredPostgresControlTables {
		if !created[table] {
			t.Fatalf("requiredPostgresControlTables lists %q but schema.sql never creates it - the control plane would refuse to boot", table)
		}
	}
}

func assertStringSliceEqual(t *testing.T, name string, got []string, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: seed %v != fixture %v", name, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: seed %q != fixture %q", name, i, got[i], want[i])
		}
	}
}

func metadataAnySliceToStrings(value any) []string {
	raw, ok := value.([]any)
	if !ok {
		return nil
	}
	values := make([]string, 0, len(raw))
	for _, item := range raw {
		if text, ok := item.(string); ok {
			values = append(values, text)
		}
	}
	return values
}

func containsString(haystack string, needle string) bool {
	return needle != "" && strings.Contains(haystack, needle)
}
