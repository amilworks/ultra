package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

type adminEquivalenceFixture struct {
	pool  *pgxpool.Pool
	store *store.PostgresStore
	deps  ServerDeps
}

func newAdminEquivalenceFixture(t testing.TB) *adminEquivalenceFixture {
	t.Helper()
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	pgStore := store.NewPostgresStore(pool)
	return &adminEquivalenceFixture{
		pool:  pool,
		store: pgStore,
		deps:  ServerDeps{Store: pgStore},
	}
}

// seed creates a spread of users, threads, messages, runs, events, leases,
// and resources across all admin activity windows, including a stalled run
// with an expired lease.
func (f *adminEquivalenceFixture) seed(t testing.TB) {
	t.Helper()
	ctx := context.Background()
	now := domain.Now()
	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	userA := "admin-eq-a-" + suffix
	userB := "admin-eq-b-" + suffix

	threadA, err := f.store.CreateThread(ctx, domain.CreateThreadInput{UserID: userA, Title: "equivalence A"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	threadB, err := f.store.CreateThread(ctx, domain.CreateThreadInput{UserID: userB, Title: "equivalence B"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	messageTimes := []time.Time{
		now.Add(-time.Hour),
		now.Add(-3 * 24 * time.Hour),
		now.Add(-10 * 24 * time.Hour),
		now.Add(-40 * 24 * time.Hour),
	}
	for index, ts := range messageTimes {
		role := "user"
		if index%2 == 1 {
			role = "assistant"
		}
		if _, err := f.store.AppendThreadMessage(ctx, domain.ThreadMessage{
			ThreadID:  threadA.ThreadID,
			Role:      role,
			Content:   fmt.Sprintf("message %d", index),
			CreatedAt: ts,
		}); err != nil {
			t.Fatalf("AppendThreadMessage: %v", err)
		}
	}
	if _, err := f.store.AppendThreadMessage(ctx, domain.ThreadMessage{
		ThreadID:  threadB.ThreadID,
		Role:      "user",
		Content:   "recent message",
		CreatedAt: now.Add(-2 * time.Hour),
	}); err != nil {
		t.Fatalf("AppendThreadMessage: %v", err)
	}

	newRun := func(threadID, userID string) domain.RunRecord {
		run, err := f.store.CreateRun(ctx, domain.CreateRunInput{
			ThreadID: threadID,
			UserID:   userID,
			Goal:     "equivalence run",
			Messages: []domain.ThreadMessage{{Role: "user", Content: "go"}},
			Metadata: domain.JSONMap{"selected_tool_names": []string{"rarespot_ecology"}},
		})
		if err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		return run
	}
	succeeded := newRun(threadA.ThreadID, userA)
	failed := newRun(threadA.ThreadID, userA)
	stalled := newRun(threadB.ThreadID, userB)

	eventTimes := []time.Time{now.Add(-30 * time.Minute), now.Add(-5 * 24 * time.Hour), now.Add(-20 * 24 * time.Hour)}
	for index, ts := range eventTimes {
		if _, err := f.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     succeeded.RunID,
			ThreadID:  threadA.ThreadID,
			EventKind: "tool_call.started",
			Message:   fmt.Sprintf("tool %d", index),
			Payload:   domain.JSONMap{"tool_name": "rarespot_ecology"},
			TS:        ts,
		}); err != nil {
			t.Fatalf("AppendRunEvent tool: %v", err)
		}
	}
	if _, err := f.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     succeeded.RunID,
		ThreadID:  threadA.ThreadID,
		EventKind: "artifact.created",
		Message:   "artifact",
		Payload:   domain.JSONMap{"artifact_id": "artifact-" + suffix},
		TS:        now.Add(-45 * time.Minute),
	}); err != nil {
		t.Fatalf("AppendRunEvent artifact: %v", err)
	}
	if _, err := f.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     stalled.RunID,
		ThreadID:  threadB.ThreadID,
		EventKind: "message.delta",
		Message:   "stale progress",
		TS:        now.Add(-2 * time.Hour),
	}); err != nil {
		t.Fatalf("AppendRunEvent delta: %v", err)
	}

	if _, err := f.store.CompleteRun(ctx, domain.CompleteRunInput{RunID: succeeded.RunID, ResponseText: "done"}); err != nil {
		t.Fatalf("CompleteRun: %v", err)
	}
	if _, err := f.store.UpdateRunStatus(ctx, failed.RunID, domain.RunStatusFailed, "", "exploded"); err != nil {
		t.Fatalf("UpdateRunStatus failed: %v", err)
	}
	if _, err := f.store.UpdateRunStatus(ctx, stalled.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus running: %v", err)
	}
	if _, err := f.store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    stalled.RunID,
		WorkerID: "worker-" + suffix,
		TTL:      time.Minute,
	}); err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	if _, err := f.pool.Exec(ctx, `UPDATE control_run_leases SET lease_expires_at = $1 WHERE run_id = $2`,
		now.Add(-10*time.Minute), stalled.RunID); err != nil {
		t.Fatalf("expire lease: %v", err)
	}

	resources := []domain.UpsertResourceInput{
		{ResourceID: "res-eq-1-" + suffix, OriginalName: "a.tif", SizeBytes: 1000, OwnerUserID: userA, ProjectID: "proj-eq-" + suffix, Status: "active"},
		{ResourceID: "res-eq-2-" + suffix, OriginalName: "b.tif", SizeBytes: 2000, OwnerUserID: userA, OwnerOrgID: "org-eq-" + suffix, Status: "active"},
		{ResourceID: "res-eq-3-" + suffix, OriginalName: "c.tif", SizeBytes: 4000, OwnerUserID: userB, Status: "active"},
		{ResourceID: "res-eq-4-" + suffix, OriginalName: "d.tif", SizeBytes: 8000, OwnerUserID: userB, Status: "deleted"},
	}
	for _, input := range resources {
		if _, err := f.store.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource: %v", err)
		}
	}
}

// normalizeAdminSnapshot strips fields that legitimately differ between two
// computations moments apart (generation timestamps, age-in-seconds values)
// and re-sorts collections whose sort keys can tie, so the comparison only
// sees data-derived content.
func normalizeAdminSnapshot(t testing.TB, snapshot adminSnapshot) map[string]any {
	t.Helper()
	data, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	var tree map[string]any
	if err := json.Unmarshal(data, &tree); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	// Both snapshots are computed with the same injected clock, so every
	// field must match exactly; only collections whose sort keys can tie
	// (ties ordered by map iteration) are re-sorted deterministically.
	for _, key := range []string{"Issues", "Users", "ToolUsage7d", "ResourceProjects", "Workers"} {
		items, ok := tree[key].([]any)
		if !ok {
			continue
		}
		sort.Slice(items, func(i, j int) bool {
			left, _ := json.Marshal(items[i])
			right, _ := json.Marshal(items[j])
			return string(left) < string(right)
		})
	}
	return tree
}

// TestAdminSnapshotAggregateMatchesLegacy is the equivalence proof for the
// aggregate admin snapshot: both implementations run against the same live
// Postgres data and must produce identical content.
func TestAdminSnapshotAggregateMatchesLegacy(t *testing.T) {
	fixture := newAdminEquivalenceFixture(t)
	fixture.seed(t)
	ctx := context.Background()

	compare := func() (map[string]any, map[string]any, error) {
		now := domain.Now()
		legacy, err := fixture.deps.loadAdminSnapshotLegacy(ctx, now)
		if err != nil {
			return nil, nil, fmt.Errorf("legacy snapshot: %w", err)
		}
		aggregate, err := fixture.deps.loadAdminSnapshotAggregate(ctx, fixture.store, now)
		if err != nil {
			return nil, nil, fmt.Errorf("aggregate snapshot: %w", err)
		}
		return normalizeAdminSnapshot(t, legacy), normalizeAdminSnapshot(t, aggregate), nil
	}
	legacyTree, aggregateTree, err := compare()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(legacyTree, aggregateTree) {
		// One retry tolerates pre-existing rows in a shared test database
		// drifting across a window boundary between the two computations.
		legacyTree, aggregateTree, err = compare()
		if err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(legacyTree, aggregateTree) {
		differences := []string{}
		diffJSONTrees("", legacyTree, aggregateTree, &differences)
		if len(differences) > 60 {
			differences = differences[:60]
		}
		t.Fatalf("aggregate snapshot diverges from legacy in %d places:\n%s", len(differences), strings.Join(differences, "\n"))
	}
}

func diffJSONTrees(path string, left any, right any, out *[]string) {
	switch typedLeft := left.(type) {
	case map[string]any:
		typedRight, ok := right.(map[string]any)
		if !ok {
			*out = append(*out, fmt.Sprintf("%s: type mismatch", path))
			return
		}
		keys := map[string]bool{}
		for key := range typedLeft {
			keys[key] = true
		}
		for key := range typedRight {
			keys[key] = true
		}
		for key := range keys {
			diffJSONTrees(path+"."+key, typedLeft[key], typedRight[key], out)
		}
	case []any:
		typedRight, ok := right.([]any)
		if !ok {
			*out = append(*out, fmt.Sprintf("%s: type mismatch", path))
			return
		}
		if len(typedLeft) != len(typedRight) {
			*out = append(*out, fmt.Sprintf("%s: list length %d (legacy) vs %d (aggregate)", path, len(typedLeft), len(typedRight)))
			return
		}
		for index := range typedLeft {
			diffJSONTrees(fmt.Sprintf("%s[%d]", path, index), typedLeft[index], typedRight[index], out)
		}
	default:
		if !reflect.DeepEqual(left, right) {
			*out = append(*out, fmt.Sprintf("%s: %v (legacy) vs %v (aggregate)", path, left, right))
		}
	}
}

func BenchmarkAdminSnapshot(b *testing.B) {
	fixture := newAdminEquivalenceFixture(b)
	fixture.seed(b)
	ctx := context.Background()
	b.Run("legacy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := fixture.deps.loadAdminSnapshotLegacy(ctx, domain.Now()); err != nil {
				b.Fatalf("legacy snapshot: %v", err)
			}
		}
	})
	b.Run("aggregate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := fixture.deps.loadAdminSnapshotAggregate(ctx, fixture.store, domain.Now()); err != nil {
				b.Fatalf("aggregate snapshot: %v", err)
			}
		}
	})
}
