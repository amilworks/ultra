package store

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresViewerCalibrationCASAllowsExactlyOneConcurrentWriter(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	resourceID := "pg-viewer-cas-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	ownerUserID := "pg-viewer-owner-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "mask.ome.tiff",
		SHA256:       "source-sha",
		OwnerUserID:  ownerUserID,
		OwnerOrgID:   "org-viewer-cas",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}

	start := make(chan struct{})
	results := make(chan error, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	for _, threshold := range []int{120, 220} {
		threshold := threshold
		go func() {
			ready.Done()
			<-start
			_, err := store.MergeResourceMetadataForUser(
				ctx,
				domain.MergeResourceMetadataInput{
					ResourceID:                 resourceID,
					UserID:                     ownerUserID,
					OrgID:                      "org-viewer-cas",
					Patch:                      viewerCalibrationPatch("source-sha", 1, threshold),
					ExpectedSourceSHA256:       "source-sha",
					SelectionExpectedRevisions: map[string]int{"c0:t0": 0},
				},
			)
			results <- err
		}()
	}
	ready.Wait()
	close(start)

	successes := 0
	conflicts := 0
	for range 2 {
		switch err := <-results; {
		case err == nil:
			successes++
		case errors.Is(err, ErrConflict):
			conflicts++
		default:
			t.Fatalf("concurrent calibration write: %v", err)
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("concurrent results successes=%d conflicts=%d, want 1/1", successes, conflicts)
	}

	resource, err := store.GetResourceForUser(ctx, resourceID, ownerUserID, "org-viewer-cas")
	if err != nil {
		t.Fatalf("GetResourceForUser: %v", err)
	}
	calibration, _ := resourceMetadataMap(resource.Metadata["ultra_viewer_calibration_v1"])
	selections, _ := resourceMetadataMap(calibration["selections"])
	selection, _ := resourceMetadataMap(selections["c0:t0"])
	revision, valid := resourceMetadataInteger(selection["revision"])
	threshold, thresholdValid := resourceMetadataInteger(selection["threshold_value"])
	if !valid || revision != 1 || !thresholdValid || (threshold != 120 && threshold != 220) {
		t.Fatalf("persisted selection = %#v, want revision 1 from one winner", selection)
	}
}

func TestPostgresStoreThreadRunEventArtifactFlow(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "pg-user",
		Title:  "Postgres flow",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "pg-user",
		Goal:     "persist run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "persist run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.accepted",
		Message:   "accepted",
		Payload:   domain.JSONMap{"ok": true},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.Sequence < 1 {
		t.Fatalf("event sequence = %d, want >= 1", event.Sequence)
	}
}

func TestPostgresStoreUpdateUserProfile(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)

	userID := "pg-prof-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	if _, err := store.CreateUser(ctx, domain.CreateUserInput{
		UserID:   userID,
		Role:     "researcher",
		Status:   "active",
		Metadata: domain.JSONMap{"existing": "kept"},
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}

	account, err := store.UpdateUserProfile(ctx, domain.UpdateUserProfileInput{
		UserID: userID,
		Profile: domain.UserProfile{
			DisplayName:       "Grace Hopper",
			Title:             "Rear Admiral",
			Institution:       "US Navy",
			ResearchInterests: "compilers",
			Bio:               "Invented the first compiler.",
		},
	})
	if err != nil {
		t.Fatalf("UpdateUserProfile: %v", err)
	}
	if account.DisplayName != "Grace Hopper" {
		t.Fatalf("display_name synced = %q, want Grace Hopper", account.DisplayName)
	}

	got, found, err := store.GetUserByID(ctx, userID)
	if err != nil || !found {
		t.Fatalf("GetUserByID found=%v err=%v", found, err)
	}
	if got.Metadata["existing"] != "kept" {
		t.Fatalf("metadata.existing = %v, want preserved across jsonb_set", got.Metadata["existing"])
	}
	profile, ok := got.Metadata["profile"].(map[string]any)
	if !ok {
		t.Fatalf("metadata.profile = %T, want map", got.Metadata["profile"])
	}
	if profile["title"] != "Rear Admiral" || profile["institution"] != "US Navy" {
		t.Fatalf("profile = %+v, want Rear Admiral / US Navy", profile)
	}
	if profile["research_interests"] != "compilers" {
		t.Fatalf("profile.research_interests = %v, want compilers", profile["research_interests"])
	}
}

func TestPostgresStoreRecordsAndReadsTokenUsage(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)

	userID := "pg-tok-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	day1 := time.Date(2026, 6, 9, 0, 0, 0, 0, time.UTC)
	day2 := time.Date(2026, 6, 10, 0, 0, 0, 0, time.UTC)

	for _, in := range []domain.RecordUserTokenUsageInput{
		{UserID: userID, Day: day1, InputTokens: 100, OutputTokens: 20, TotalTokens: 120},
		{UserID: userID, Day: day2, InputTokens: 200, OutputTokens: 40, TotalTokens: 240},
		{UserID: userID, Day: day2, InputTokens: 50, OutputTokens: 10, TotalTokens: 60},
	} {
		if err := store.RecordUserTokenUsage(ctx, in); err != nil {
			t.Fatalf("RecordUserTokenUsage: %v", err)
		}
	}

	stats, err := store.GetUserTokenUsageStats(ctx, userID)
	if err != nil {
		t.Fatalf("GetUserTokenUsageStats: %v", err)
	}
	if stats.InputTokens != 350 || stats.OutputTokens != 70 || stats.TotalTokens != 420 {
		t.Fatalf("lifetime = %+v, want 350/70/420", stats)
	}
	if stats.PeakDailyTotal != 300 {
		t.Fatalf("peak daily = %d, want 300 (day2 240+60)", stats.PeakDailyTotal)
	}
	if stats.LastActiveDay == nil || !stats.LastActiveDay.UTC().Equal(day2) {
		t.Fatalf("last active day = %v, want %v", stats.LastActiveDay, day2)
	}

	daily, err := store.ListUserTokenUsageDaily(ctx, userID, day1)
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily: %v", err)
	}
	if len(daily) != 2 {
		t.Fatalf("daily rows = %d, want 2: %+v", len(daily), daily)
	}
	if !daily[0].Day.UTC().Equal(day1) || daily[0].TotalTokens != 120 || daily[0].RunCount != 1 {
		t.Fatalf("daily[0] = %+v, want day1 total 120 run_count 1", daily[0])
	}
	if !daily[1].Day.UTC().Equal(day2) || daily[1].TotalTokens != 300 || daily[1].RunCount != 2 {
		t.Fatalf("daily[1] = %+v, want day2 total 300 run_count 2", daily[1])
	}

	recent, err := store.ListUserTokenUsageDaily(ctx, userID, day2)
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily recent: %v", err)
	}
	if len(recent) != 1 || !recent[0].Day.UTC().Equal(day2) {
		t.Fatalf("recent = %+v, want only day2", recent)
	}

	longest, err := store.GetUserLongestRunSeconds(ctx, userID)
	if err != nil {
		t.Fatalf("GetUserLongestRunSeconds: %v", err)
	}
	if longest != 0 {
		t.Fatalf("longest run seconds = %d, want 0 (no completed runs for user)", longest)
	}
}

func TestPostgresStoreRejectsDuplicateRunIdempotencyKey(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-idempotency-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	metadata := domain.JSONMap{"idempotency_key": "same-submit-key"}
	_, err = store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "first run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "first run"}},
		Metadata: metadata,
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	_, err = store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "duplicate run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "duplicate run"}},
		Metadata: metadata,
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateRun duplicate err = %v, want ErrConflict", err)
	}
}

func TestPostgresStoreTenantScopedQueriesFilterByUser(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := domain.NewID("tenant")
	aliceID := "alice-" + suffix
	bobID := "bob-" + suffix
	aliceThread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          aliceID,
		Title:           "Alice Postgres thread",
		InitialMessages: []domain.ThreadMessage{{Role: "user", Content: "alice message"}},
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}
	aliceRun, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: aliceThread.ThreadID,
		UserID:   aliceID,
		Goal:     "alice postgres run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "alice postgres run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun alice: %v", err)
	}
	if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     aliceRun.RunID,
		ThreadID:  aliceThread.ThreadID,
		EventKind: "message.delta",
		Message:   "alice trace",
	}); err != nil {
		t.Fatalf("AppendRunEvent alice: %v", err)
	}
	aliceArtifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    aliceRun.RunID,
		ThreadID: aliceThread.ThreadID,
		Kind:     "report",
		Path:     "alice.md",
	})
	if err != nil {
		t.Fatalf("CreateArtifact alice: %v", err)
	}
	bobThread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: bobID, Title: "Bob Postgres thread"})
	if err != nil {
		t.Fatalf("CreateThread bob: %v", err)
	}
	if _, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID,
		UserID:   bobID,
		Goal:     "bob postgres run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bob postgres run"}},
	}); err != nil {
		t.Fatalf("CreateRun bob: %v", err)
	}

	alicePage, err := store.ListThreadsForUser(ctx, aliceID, 10, 0, "")
	if err != nil {
		t.Fatalf("ListThreadsForUser alice: %v", err)
	}
	if alicePage.TotalCount != 1 || len(alicePage.Threads) != 1 || alicePage.Threads[0].ThreadID != aliceThread.ThreadID {
		t.Fatalf("alice threads = %+v, want only alice thread", alicePage)
	}
	if _, err := store.GetThreadForUser(ctx, aliceThread.ThreadID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetThreadForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListThreadMessagesForUser(ctx, aliceThread.ThreadID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListThreadMessagesForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetRunForUser(ctx, aliceRun.RunID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRunForUser bob err = %v, want ErrNotFound", err)
	}
	bobRuns, err := store.ListRunsForUser(ctx, bobID, "", "", 10, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser bob: %v", err)
	}
	if len(bobRuns) != 1 || bobRuns[0].UserID != bobID {
		t.Fatalf("bob runs = %+v, want only bob run", bobRuns)
	}
	if _, err := store.ListRunEventsForUser(ctx, aliceRun.RunID, bobID, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunArtifactsForUser(ctx, aliceRun.RunID, bobID, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunArtifactsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetArtifactForUser(ctx, aliceArtifact.ArtifactID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetArtifactForUser bob err = %v, want ErrNotFound", err)
	}
}

func TestPostgresStoreResourceCatalogFiltersSoftDeletesAndRestores(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := domain.NewID("resource")
	resourceID := "file_alice_" + suffix
	bobResourceID := "file_bob_" + suffix
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  "alice-" + suffix,
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		OriginalName: "cells.ome.tiff",
		ContentType:  "image/tiff",
		SizeBytes:    128,
		SHA256:       "abc123",
		StorageURI:   "file:///srv/ultra/shared/uploads/" + resourceID + "__cells.ome.tiff",
		StoragePath:  resourceID + "__cells.ome.tiff",
		SourceType:   "upload",
		ResourceKind: "image",
		ProjectID:    "project-ct",
		Status:       "active",
		Metadata: domain.JSONMap{
			"label": "NPH",
			"data_agent": domain.JSONMap{
				"caption_resources": domain.JSONMap{
					"caption": "Postgres deterministic metadata caption.",
					"status":  "succeeded",
				},
			},
		},
	}); err != nil {
		t.Fatalf("UpsertResource alice: %v", err)
	}
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   bobResourceID,
		OwnerUserID:  "bob-" + suffix,
		OwnerOrgID:   "org-a",
		OriginalName: "other.csv",
		SizeBytes:    64,
		SourceType:   "upload",
		ResourceKind: "table",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource bob: %v", err)
	}

	page, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:    "alice-" + suffix,
		OrgID:     "org-a",
		Query:     "ome",
		Kind:      "image",
		Source:    "upload",
		ProjectID: "project-ct",
		Limit:     20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser alice: %v", err)
	}
	if page.TotalCount != 1 || len(page.Resources) != 1 || page.Resources[0].ResourceID != resourceID {
		t.Fatalf("alice resources = %+v, want only alice image", page)
	}
	if page.Resources[0].ProjectID != "project-ct" {
		t.Fatalf("alice project_id = %q, want project-ct", page.Resources[0].ProjectID)
	}
	metadataPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice-" + suffix,
		OrgID:  "org-a",
		Query:  "deterministic metadata caption",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser metadata query: %v", err)
	}
	if metadataPage.TotalCount != 1 || len(metadataPage.Resources) != 1 || metadataPage.Resources[0].ResourceID != resourceID {
		t.Fatalf("metadata resources = %+v, want only captioned resource", metadataPage)
	}
	wrongProject, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice-" + suffix, OrgID: "org-a", ProjectID: "project-other", Limit: 20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser wrong project: %v", err)
	}
	if wrongProject.TotalCount != 0 || len(wrongProject.Resources) != 0 {
		t.Fatalf("wrong project resources = %+v, want none", wrongProject)
	}
	if _, err := store.GetResourceForUser(ctx, resourceID, "bob-"+suffix, "org-a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser bob err = %v, want ErrNotFound", err)
	}
	deleted, err := store.SoftDeleteResourceForUser(ctx, resourceID, "alice-"+suffix, "org-a", time.Now())
	if err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}
	if deleted.Status != "deleted" || deleted.DeletedAt.IsZero() || deleted.RetentionExpiresAt.IsZero() {
		t.Fatalf("deleted = %+v, want deleted status with retention expiry", deleted)
	}
	deletedPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{UserID: "alice-" + suffix, OrgID: "org-a", Limit: 20})
	if err != nil {
		t.Fatalf("ListResourcesForUser after delete: %v", err)
	}
	if deletedPage.TotalCount != 0 || len(deletedPage.Resources) != 0 {
		t.Fatalf("deleted resources = %+v, want no active rows", deletedPage)
	}
	restored, err := store.RestoreResourceForUser(ctx, resourceID, "alice-"+suffix, "org-a", time.Now())
	if err != nil {
		t.Fatalf("RestoreResourceForUser: %v", err)
	}
	if restored.Status != "active" || !restored.DeletedAt.IsZero() || !restored.RetentionExpiresAt.IsZero() {
		t.Fatalf("restored = %+v, want active with empty retention fields", restored)
	}
	if _, err := store.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
		ResourceID:  resourceID,
		ActorUserID: "alice-" + suffix,
		ActorOrgID:  "org-a",
		EventType:   "resource.restored",
	}); err != nil {
		t.Fatalf("CreateResourceEvent: %v", err)
	}
}

func TestPostgresStoreListResourcesPastRetentionIncludesStorageURI(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := domain.NewID("resource")
	resourceID := "file_expired_" + suffix
	storageURI := "file:///srv/ultra/shared/uploads/" + resourceID + "__cells.ome.tiff"
	storagePath := resourceID + "__cells.ome.tiff"
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  "alice-" + suffix,
		OwnerOrgID:   "org-a",
		OriginalName: "cells.ome.tiff",
		ContentType:  "image/tiff",
		SizeBytes:    128,
		StorageURI:   storageURI,
		StoragePath:  storagePath,
		SourceType:   "upload",
		ResourceKind: "image",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	if _, err := store.SoftDeleteResourceForUser(ctx, resourceID, "alice-"+suffix, "org-a", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}

	expired, err := store.ListResourcesPastRetention(ctx, time.Now(), 10000)
	if err != nil {
		t.Fatalf("ListResourcesPastRetention: %v", err)
	}
	for _, resource := range expired {
		if resource.ResourceID != resourceID {
			continue
		}
		if resource.StorageURI != storageURI {
			t.Fatalf("expired StorageURI = %q, want %q", resource.StorageURI, storageURI)
		}
		if resource.StoragePath != storagePath {
			t.Fatalf("expired StoragePath = %q, want %q", resource.StoragePath, storagePath)
		}
		return
	}
	t.Fatalf("expired resources missing %s: %+v", resourceID, expired)
}

func TestPostgresStoreListResourceEventsForUserScopesAndFilters(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := domain.NewID("audit")
	base := time.Date(2026, 6, 8, 12, 0, 0, 0, time.UTC)
	alice := "alice-" + suffix
	bob := "bob-" + suffix
	charlie := "charlie-" + suffix
	activeID := "file_audit_active_" + suffix
	deletedID := "file_audit_deleted_" + suffix
	bobPrivateID := "file_audit_bob_private_" + suffix
	inputs := []domain.UpsertResourceInput{
		{
			ResourceID:   activeID,
			OwnerUserID:  alice,
			OwnerOrgID:   "org-a",
			OriginalName: "audit-active.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base,
		},
		{
			ResourceID:   deletedID,
			OwnerUserID:  alice,
			OwnerOrgID:   "org-a",
			OriginalName: "audit-deleted.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "deleted",
			CreatedAt:    base.Add(time.Minute),
			DeletedAt:    base.Add(5 * time.Minute),
		},
		{
			ResourceID:   bobPrivateID,
			OwnerUserID:  bob,
			OwnerOrgID:   "org-b",
			OriginalName: "bob-private.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base.Add(2 * time.Minute),
		},
	}
	for _, input := range inputs {
		if _, err := store.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	if _, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID:      activeID,
		OwnerUserID:     alice,
		OwnerOrgID:      "org-a",
		GranteeUserID:   bob,
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: alice,
		CreatedAt:       base.Add(3 * time.Minute),
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	events := []domain.AppendResourceEventInput{
		{ResourceID: activeID, ActorUserID: alice, ActorOrgID: "org-a", EventType: "resource.tagged", TS: base.Add(4 * time.Minute), Metadata: domain.JSONMap{"tag": "NPH"}},
		{ResourceID: deletedID, ActorUserID: alice, ActorOrgID: "org-a", EventType: "resource.deleted", TS: base.Add(5 * time.Minute)},
		{ResourceID: bobPrivateID, ActorUserID: bob, ActorOrgID: "org-b", EventType: "resource.tagged", TS: base.Add(6 * time.Minute), Metadata: domain.JSONMap{"tag": "private"}},
	}
	for _, event := range events {
		if _, err := store.CreateResourceEvent(ctx, event); err != nil {
			t.Fatalf("CreateResourceEvent(%s): %v", event.ResourceID, err)
		}
	}

	aliceEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID: alice,
		OrgID:  "org-a",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser alice: %v", err)
	}
	if aliceEvents.TotalCount != 2 || len(aliceEvents.Events) != 2 {
		t.Fatalf("alice events = %+v, want active and deleted owned events", aliceEvents)
	}
	if aliceEvents.Events[0].ResourceID != deletedID || aliceEvents.Events[1].ResourceID != activeID {
		t.Fatalf("alice event order = %+v, want deleted then active", aliceEvents.Events)
	}

	deletedEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID:    alice,
		OrgID:     "org-a",
		EventType: "resource.deleted",
		Limit:     10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser deleted: %v", err)
	}
	if deletedEvents.TotalCount != 1 || len(deletedEvents.Events) != 1 || deletedEvents.Events[0].ResourceID != deletedID {
		t.Fatalf("deleted events = %+v, want only deleted resource event", deletedEvents)
	}

	bobEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID:     bob,
		OrgID:      "org-b",
		ResourceID: activeID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser bob shared resource: %v", err)
	}
	if bobEvents.TotalCount != 1 || len(bobEvents.Events) != 1 || bobEvents.Events[0].ResourceID != activeID {
		t.Fatalf("bob shared events = %+v, want only shared active resource event", bobEvents)
	}

	foreignEvents, err := store.ListResourceEventsForUser(ctx, domain.ResourceEventListInput{
		UserID: charlie,
		OrgID:  "org-c",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceEventsForUser charlie: %v", err)
	}
	if foreignEvents.TotalCount != 0 || len(foreignEvents.Events) != 0 {
		t.Fatalf("charlie events = %+v, want no leaked audit events", foreignEvents)
	}
}

func TestPostgresStoreListResourceIDsForOwner(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}

	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("owner_lookup"), "-", "_")
	alice := "alice-" + suffix
	bob := "bob-" + suffix
	org := "org-a-" + suffix
	ownerOrgID := "file_owner_org_" + suffix
	ownerNoOrgID := "file_owner_no_org_" + suffix
	otherOwnerID := "file_other_owner_" + suffix
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID:   ownerOrgID,
			OwnerUserID:  alice,
			OwnerOrgID:   org,
			OriginalName: "owner-org.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
		{
			ResourceID:   ownerNoOrgID,
			OwnerUserID:  alice,
			OriginalName: "owner-no-org.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
		{
			ResourceID:   otherOwnerID,
			OwnerUserID:  bob,
			OwnerOrgID:   org,
			OriginalName: "other-owner.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
		},
	} {
		if _, err := store.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}

	existing, err := store.ListResourceIDsForOwner(ctx, " "+alice+" ", " "+org+" ", []string{
		" " + ownerOrgID + " ",
		ownerOrgID,
		ownerNoOrgID,
		otherOwnerID,
		"file_missing_" + suffix,
		"",
	})
	if err != nil {
		t.Fatalf("ListResourceIDsForOwner: %v", err)
	}
	if !existing[ownerOrgID] || !existing[ownerNoOrgID] {
		t.Fatalf("existing = %+v, want owned org and no-org resources", existing)
	}
	if existing[otherOwnerID] || existing["file_missing_"+suffix] || len(existing) != 2 {
		t.Fatalf("existing = %+v, want only resources visible to owner", existing)
	}
}

func TestPostgresStoreResourceReadGrantMakesResourceVisibleToGrantee(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := domain.NewID("share")
	resourceID := "file_shared_read_" + suffix
	aliceID := "alice-" + suffix
	bobID := "bob-" + suffix
	now := time.Date(2026, 6, 8, 16, 0, 0, 0, time.UTC)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  aliceID,
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		OriginalName: "shared-nph-study.nii.gz",
		ContentType:  "application/gzip",
		SizeBytes:    512,
		SHA256:       "sha-shared-read-" + suffix,
		SourceType:   "upload",
		ResourceKind: "file",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	before, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: bobID,
		OrgID:  "org-b",
		Query:  "shared-nph",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser before grant: %v", err)
	}
	if before.TotalCount != 0 || len(before.Resources) != 0 {
		t.Fatalf("bob resources before grant = %+v, want none", before)
	}
	if _, err := store.GetResourceForUser(ctx, resourceID, bobID, "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser before grant err = %v, want ErrNotFound", err)
	}

	grant, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		GrantID:         "resource_grant_shared_read_" + suffix,
		ResourceID:      resourceID,
		OwnerUserID:     aliceID,
		OwnerOrgID:      "org-a",
		GranteeUserID:   bobID,
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: aliceID,
		CreatedAt:       now.Add(time.Second),
		Metadata:        domain.JSONMap{"reason": "collaborative review"},
	})
	if err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if grant.Status != "active" || grant.ResourceID != resourceID || grant.GranteeUserID != bobID || grant.Role != "read" {
		t.Fatalf("grant = %+v, want active bob read grant", grant)
	}

	after, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: bobID,
		OrgID:  "org-b",
		Query:  "shared-nph",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser after grant: %v", err)
	}
	if after.TotalCount != 1 || len(after.Resources) != 1 || after.Resources[0].ResourceID != resourceID {
		t.Fatalf("bob resources after grant = %+v, want shared resource", after)
	}
	if after.Resources[0].OwnerUserID != aliceID {
		t.Fatalf("shared resource owner = %q, want %s", after.Resources[0].OwnerUserID, aliceID)
	}
	loaded, err := store.GetResourceForUser(ctx, resourceID, bobID, "org-b")
	if err != nil {
		t.Fatalf("GetResourceForUser after grant: %v", err)
	}
	if loaded.ResourceID != resourceID || loaded.OwnerUserID != aliceID {
		t.Fatalf("loaded shared resource = %+v, want alice-owned resource", loaded)
	}
}

func TestPostgresStoreGetResourceForUserHonorsActivePublicGrant(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := domain.NewID("public-share")
	resourceID := "file_public_calpha_" + suffix
	ownerID := "calphad-owner-" + suffix
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  ownerID,
		OwnerOrgID:   "org-owner",
		OwnerRole:    "researcher",
		OriginalName: "public-al-co-w.tdb",
		ContentType:  "application/x-thermocalc-tdb",
		SizeBytes:    21274,
		SHA256:       "sha-public-calphad-" + suffix,
		SourceType:   "upload",
		ResourceKind: "document",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	grant, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		GrantID:         "resource_grant_public_calpha_" + suffix,
		ResourceID:      resourceID,
		OwnerUserID:     ownerID,
		OwnerOrgID:      "org-owner",
		Public:          true,
		Role:            "read",
		CreatedByUserID: ownerID,
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if grant.GranteeUserID != domain.PublicResourceGranteeUserID || grant.Status != "active" {
		t.Fatalf("public grant = %+v", grant)
	}

	loaded, err := store.GetResourceForUser(ctx, resourceID, "unrelated-reader-"+suffix, "org-unrelated")
	if err != nil {
		t.Fatalf("GetResourceForUser(public): %v", err)
	}
	if loaded.ResourceID != resourceID || loaded.OwnerUserID != ownerID {
		t.Fatalf("publicly resolved resource = %+v", loaded)
	}
}

func TestPostgresStoreResourceShareGrantRevocationRemovesCollaboratorAccess(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := domain.NewID("share")
	resourceID := "file_revoked_share_" + suffix
	aliceID := "alice-" + suffix
	bobID := "bob-" + suffix
	now := time.Date(2026, 6, 8, 17, 0, 0, 0, time.UTC)
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  aliceID,
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		OriginalName: "revoked-nph-study.nii.gz",
		ContentType:  "application/gzip",
		SizeBytes:    512,
		SHA256:       "sha-revoked-share-" + suffix,
		SourceType:   "upload",
		ResourceKind: "file",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	grant, err := store.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		GrantID:         "resource_grant_revoked_share_" + suffix,
		ResourceID:      resourceID,
		OwnerUserID:     aliceID,
		OwnerOrgID:      "org-a",
		GranteeUserID:   bobID,
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: aliceID,
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if _, err := store.GetResourceForUser(ctx, resourceID, bobID, "org-b"); err != nil {
		t.Fatalf("GetResourceForUser before revoke: %v", err)
	}
	grants, err := store.ListResourceShareGrantsForResource(ctx, domain.ListResourceShareGrantsInput{
		ResourceID:  resourceID,
		OwnerUserID: aliceID,
		OwnerOrgID:  "org-a",
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("ListResourceShareGrantsForResource: %v", err)
	}
	if len(grants) != 1 || grants[0].GrantID != grant.GrantID || grants[0].Status != "active" {
		t.Fatalf("grants before revoke = %+v, want active grant", grants)
	}

	revokedAt := now.Add(2 * time.Second)
	revoked, err := store.RevokeResourceShareGrant(ctx, domain.RevokeResourceShareGrantInput{
		ResourceID:  resourceID,
		GrantID:     grant.GrantID,
		OwnerUserID: aliceID,
		OwnerOrgID:  "org-a",
		RevokedAt:   revokedAt,
	})
	if err != nil {
		t.Fatalf("RevokeResourceShareGrant: %v", err)
	}
	if revoked.Status != "revoked" || !revoked.RevokedAt.Equal(revokedAt) {
		t.Fatalf("revoked grant = %+v, want revoked at %s", revoked, revokedAt)
	}
	if _, err := store.GetResourceForUser(ctx, resourceID, bobID, "org-b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser after revoke err = %v, want ErrNotFound", err)
	}
	allGrants, err := store.ListResourceShareGrantsForResource(ctx, domain.ListResourceShareGrantsInput{
		ResourceID:  resourceID,
		OwnerUserID: aliceID,
		OwnerOrgID:  "org-a",
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("ListResourceShareGrantsForResource after revoke: %v", err)
	}
	if len(allGrants) != 1 || allGrants[0].Status != "revoked" {
		t.Fatalf("grants after revoke = %+v, want revoked grant retained for audit", allGrants)
	}
}

func TestPostgresStoreUploadSessionLifecycle(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("test"), "-", "_")
	sessionID := "upload_session_" + suffix
	userID := "upload-user-" + suffix

	session, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:      sessionID,
		OwnerUserID:    userID,
		OwnerOrgID:     "org-a",
		Status:         "active",
		TotalBytes:     5,
		IdempotencyKey: "idem-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if session.SessionID != sessionID {
		t.Fatalf("session id = %q", session.SessionID)
	}

	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:    sessionID,
		FileToken:    "file-a",
		OriginalName: "cells.tif",
		SizeBytes:    5,
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile: %v", err)
	}
	if _, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  sessionID,
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  5,
		SHA256:     "chunk-sha",
		Status:     "verified",
	}); err != nil {
		t.Fatalf("UpsertUploadChunk: %v", err)
	}
	chunks, err := store.ListUploadChunks(ctx, sessionID, "file-a")
	if err != nil {
		t.Fatalf("ListUploadChunks: %v", err)
	}
	if len(chunks) != 1 || chunks[0].SHA256 != "chunk-sha" {
		t.Fatalf("chunks = %+v", chunks)
	}
	sessionChunks, err := store.ListUploadSessionChunks(ctx, sessionID)
	if err != nil {
		t.Fatalf("ListUploadSessionChunks: %v", err)
	}
	if len(sessionChunks) != 1 || sessionChunks[0].FileToken != "file-a" {
		t.Fatalf("session chunks = %+v", sessionChunks)
	}
	totals, err := store.GetUploadSessionTotals(ctx, sessionID)
	if err != nil {
		t.Fatalf("GetUploadSessionTotals: %v", err)
	}
	if totals.BytesReceived != 5 || totals.BytesVerified != 5 || totals.BytesCommitted != 0 || totals.AllComplete {
		t.Fatalf("totals = %+v, want received/verified bytes before completion", totals)
	}
}

func TestPostgresStoreUploadSessionCountersUpdateIncrementally(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("upload_counters"), "-", "_")
	sessionID := "upload_session_counters_" + suffix
	userID := "counter-user-" + suffix
	now := domain.Now()

	if _, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:   sessionID,
		OwnerUserID: userID,
		OwnerOrgID:  "org-counters",
		Status:      "active",
		TotalBytes:  12,
		CreatedAt:   now,
		UpdatedAt:   now,
	}); err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:    sessionID,
		FileToken:    "file-a",
		OriginalName: "file-a.bin",
		SizeBytes:    5,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile file-a: %v", err)
	}
	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:    sessionID,
		FileToken:    "file-b",
		OriginalName: "file-b.bin",
		SizeBytes:    7,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile file-b: %v", err)
	}
	if _, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  sessionID,
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  5,
		SHA256:     "chunk-a",
		Status:     "received",
		ReceivedAt: now,
	}); err != nil {
		t.Fatalf("UpsertUploadChunk received: %v", err)
	}
	loaded, err := store.GetUploadSessionForUser(ctx, sessionID, userID, "org-counters")
	if err != nil {
		t.Fatalf("GetUploadSessionForUser after received: %v", err)
	}
	if loaded.BytesReceived != 5 || loaded.BytesVerified != 0 || loaded.BytesCommitted != 0 {
		t.Fatalf("session counters after received chunk = %+v, want received=5 verified=0 committed=0", loaded)
	}
	if _, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  sessionID,
		FileToken:  "file-a",
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  5,
		SHA256:     "chunk-a",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	}); err != nil {
		t.Fatalf("UpsertUploadChunk verified replay: %v", err)
	}
	loaded, err = store.GetUploadSessionForUser(ctx, sessionID, userID, "org-counters")
	if err != nil {
		t.Fatalf("GetUploadSessionForUser after verified: %v", err)
	}
	if loaded.BytesReceived != 5 || loaded.BytesVerified != 5 || loaded.BytesCommitted != 0 {
		t.Fatalf("session counters after verified chunk = %+v, want received=5 verified=5 committed=0", loaded)
	}
	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:      sessionID,
		FileToken:      "file-a",
		OriginalName:   "file-a.bin",
		SizeBytes:      5,
		ComputedSHA256: "file-a-sha",
		Status:         "completed",
		CreatedAt:      now,
		UpdatedAt:      now,
		CompletedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile completed file-a: %v", err)
	}
	loaded, err = store.GetUploadSessionForUser(ctx, sessionID, userID, "org-counters")
	if err != nil {
		t.Fatalf("GetUploadSessionForUser after completed file: %v", err)
	}
	if loaded.BytesReceived != 5 || loaded.BytesVerified != 5 || loaded.BytesCommitted != 5 {
		t.Fatalf("session counters after completed file = %+v, want received=5 verified=5 committed=5", loaded)
	}
	totals, err := store.GetUploadSessionTotals(ctx, sessionID)
	if err != nil {
		t.Fatalf("GetUploadSessionTotals: %v", err)
	}
	if totals.BytesReceived != 5 || totals.BytesVerified != 5 || totals.BytesCommitted != 5 || totals.AllComplete {
		t.Fatalf("session totals after one completed file = %+v, want stored counters and not all complete", totals)
	}
}

func TestPostgresStoreUploadChunkDoesNotReplaceVerifiedBytes(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("verified_conflict"), "-", "_")
	sessionID := "upload_session_" + suffix
	fileToken := "file_" + suffix
	now := domain.Now()

	if _, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:   sessionID,
		OwnerUserID: "upload-user-" + suffix,
		OwnerOrgID:  "org-a",
		Status:      "active",
		TotalBytes:  6,
		CreatedAt:   now,
		UpdatedAt:   now,
	}); err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if _, err := store.UpsertUploadSessionFile(ctx, domain.UpsertUploadSessionFileInput{
		SessionID:    sessionID,
		FileToken:    fileToken,
		OriginalName: "cells.tif",
		SizeBytes:    6,
		Status:       "uploading",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertUploadSessionFile: %v", err)
	}
	verified, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  sessionID,
		FileToken:  fileToken,
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  6,
		SHA256:     "abcdef",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	})
	if err != nil {
		t.Fatalf("UpsertUploadChunk verified: %v", err)
	}
	if _, err := store.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  sessionID,
		FileToken:  fileToken,
		ChunkIndex: 0,
		Offset:     0,
		SizeBytes:  6,
		SHA256:     "different",
		Status:     "verified",
		ReceivedAt: now,
		VerifiedAt: now,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting verified UpsertUploadChunk err = %v, want ErrConflict", err)
	}
	chunks, err := store.ListUploadChunks(ctx, sessionID, fileToken)
	if err != nil {
		t.Fatalf("ListUploadChunks: %v", err)
	}
	if len(chunks) != 1 || chunks[0].SHA256 != verified.SHA256 {
		t.Fatalf("chunks after conflict = %+v, want original verified manifest", chunks)
	}
}

func TestPostgresStoreUploadSessionOperationalMetrics(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	before, err := store.UploadSessionOperationalMetrics(ctx)
	if err != nil {
		t.Fatalf("UploadSessionOperationalMetrics before: %v", err)
	}

	suffix := strings.ReplaceAll(domain.NewID("test"), "-", "_")
	for _, input := range []domain.CreateUploadSessionInput{
		{
			SessionID:     "upload_session_metrics_active_" + suffix,
			OwnerUserID:   "metrics-user-" + suffix,
			Status:        "active",
			TotalBytes:    100,
			BytesReceived: 64,
			BytesVerified: 32,
		},
		{
			SessionID:     "upload_session_metrics_paused_" + suffix,
			OwnerUserID:   "metrics-user-" + suffix,
			Status:        "paused",
			TotalBytes:    200,
			BytesReceived: 128,
			BytesVerified: 128,
		},
		{
			SessionID:      "upload_session_metrics_completed_" + suffix,
			OwnerUserID:    "metrics-user-" + suffix,
			Status:         "completed",
			TotalBytes:     300,
			BytesReceived:  300,
			BytesVerified:  300,
			BytesCommitted: 300,
		},
	} {
		if _, err := store.CreateUploadSession(ctx, input); err != nil {
			t.Fatalf("CreateUploadSession(%s): %v", input.SessionID, err)
		}
	}

	after, err := store.UploadSessionOperationalMetrics(ctx)
	if err != nil {
		t.Fatalf("UploadSessionOperationalMetrics after: %v", err)
	}
	delta := domain.UploadSessionOperationalMetrics{
		Total:          after.Total - before.Total,
		Active:         after.Active - before.Active,
		Paused:         after.Paused - before.Paused,
		Completed:      after.Completed - before.Completed,
		Failed:         after.Failed - before.Failed,
		Canceled:       after.Canceled - before.Canceled,
		Other:          after.Other - before.Other,
		BytesTotal:     after.BytesTotal - before.BytesTotal,
		BytesReceived:  after.BytesReceived - before.BytesReceived,
		BytesVerified:  after.BytesVerified - before.BytesVerified,
		BytesCommitted: after.BytesCommitted - before.BytesCommitted,
	}
	if delta.Total != 3 || delta.Active != 1 || delta.Paused != 1 || delta.Completed != 1 || delta.Failed != 0 || delta.Canceled != 0 || delta.Other != 0 {
		t.Fatalf("upload session metrics delta = %+v, want active/paused/completed session counts", delta)
	}
	if delta.BytesTotal != 600 || delta.BytesReceived != 492 || delta.BytesVerified != 460 || delta.BytesCommitted != 300 {
		t.Fatalf("upload session metrics byte delta = %+v, want summed bytes", delta)
	}
}

func TestPostgresStoreUploadSessionEvents(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("test"), "-", "_")
	session, err := store.CreateUploadSession(ctx, domain.CreateUploadSessionInput{
		SessionID:   "upload_session_events_" + suffix,
		OwnerUserID: "events-user-" + suffix,
		OwnerOrgID:  "org-events",
		Status:      "active",
		TotalBytes:  12,
	})
	if err != nil {
		t.Fatalf("CreateUploadSession: %v", err)
	}
	if _, err := store.AppendUploadSessionEvent(ctx, domain.AppendUploadSessionEventInput{
		SessionID:   session.SessionID,
		ActorUserID: "events-user-" + suffix,
		ActorOrgID:  "org-events",
		EventType:   "upload_session.created",
		Metadata: domain.JSONMap{
			"status":      "active",
			"total_bytes": int64(12),
		},
	}); err != nil {
		t.Fatalf("AppendUploadSessionEvent created: %v", err)
	}
	if _, err := store.AppendUploadSessionEvent(ctx, domain.AppendUploadSessionEventInput{
		SessionID:   session.SessionID,
		ActorUserID: "events-user-" + suffix,
		ActorOrgID:  "org-events",
		EventType:   "upload_session.paused",
		Metadata: domain.JSONMap{
			"status": "paused",
		},
	}); err != nil {
		t.Fatalf("AppendUploadSessionEvent paused: %v", err)
	}

	events, err := store.ListUploadSessionEvents(ctx, session.SessionID, 10)
	if err != nil {
		t.Fatalf("ListUploadSessionEvents: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("events = %+v, want two upload-session audit events", events)
	}
	if events[0].ActorUserID != "events-user-"+suffix || events[0].ActorOrgID != "org-events" {
		t.Fatalf("events = %+v, want actor persisted", events)
	}
	if !uploadSessionEventRecordsContain(events, "upload_session.created") || !uploadSessionEventRecordsContain(events, "upload_session.paused") {
		t.Fatalf("events = %+v, want created and paused events", events)
	}
}

func TestPostgresStoreResourceCatalogFiltersByTags(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("tags"), "-", "_")
	userID := "tag-user-" + suffix
	orgID := "tag-org-" + suffix
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_tag_nph_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "tagged-nph.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Tags:         []string{"NPH", "Under 70", "MRI"},
		},
		{
			ResourceID:   "file_tag_control_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "tagged-control.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    64,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Tags:         []string{"control", "MRI"},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Tags:   []string{"nph", "under 70"},
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser tag filter: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_tag_nph_"+suffix {
		t.Fatalf("tag-filtered resources = %+v, want only NPH under-70 resource", matches)
	}
	if got := matches.Resources[0].Tags; !reflect.DeepEqual(got, []string{"NPH", "Under 70", "MRI"}) {
		t.Fatalf("resource tags = %#v, want persisted display tags", got)
	}

	result, err := store.BulkTagResourcesForUser(ctx, domain.BulkTagResourcesInput{
		OwnerUserID: userID,
		OwnerOrgID:  orgID,
		ActorUserID: userID,
		ActorOrgID:  orgID,
		ResourceIDs: []string{"file_tag_nph_" + suffix, "file_tag_control_" + suffix},
		Tags:        []string{"reviewed", "NPH"},
	})
	if err != nil {
		t.Fatalf("BulkTagResourcesForUser: %v", err)
	}
	if result.UpdatedCount != 2 || len(result.Events) != 2 {
		t.Fatalf("bulk tag result = %+v, want two resources and audit events", result)
	}
	reviewed, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Tags:   []string{"reviewed"},
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser reviewed tag: %v", err)
	}
	if reviewed.TotalCount != 2 || len(reviewed.Resources) != 2 {
		t.Fatalf("reviewed resources = %+v, want both tagged resources", reviewed)
	}
}

func TestPostgresStoreResourceCatalogFiltersScientificMetadata(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("metadata"), "-", "_")
	userID := "metadata-user-" + suffix
	orgID := "metadata-org-" + suffix
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_metadata_nph_under_70_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "sub-001.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(68),
			},
		},
		{
			ResourceID:   "file_metadata_nph_over_70_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "sub-002.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(74),
			},
		},
		{
			ResourceID:   "file_metadata_control_under_70_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "sub-003.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			Metadata: domain.JSONMap{
				"label":       "control",
				"format":      "nifti",
				"subject_age": float64(64),
			},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	matches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		MetadataFilters: []domain.ResourceMetadataFilter{
			{Path: "label", Operator: "eq", Value: "NPH"},
			{Path: "format", Operator: "eq", Value: "nifti"},
			{Path: "subject_age", Operator: "lt", Value: "70"},
		},
		Limit: 20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser metadata filters: %v", err)
	}
	if matches.TotalCount != 1 || len(matches.Resources) != 1 || matches.Resources[0].ResourceID != "file_metadata_nph_under_70_"+suffix {
		t.Fatalf("metadata-filtered resources = %+v, want only NPH under-70 NIfTI", matches)
	}
}

func TestPostgresStoreResourceSearchParsesScientificPredicatesAndFilePatterns(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	userID := "pg-search-" + suffix
	orgID := "org-search-" + suffix
	now := time.Date(2026, 6, 27, 9, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_old_64_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "Norm_old_004_64yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now,
			Metadata: domain.JSONMap{
				"label": "NPH",
				"image_header": domain.JSONMap{
					"reader":      "nifti-1",
					"array_dtype": "float32",
					"width":       float64(256),
				},
			},
		},
		{
			ResourceID:   "file_old_81_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "Norm_old_001_81yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			Metadata: domain.JSONMap{
				"label": "control",
				"image_header": domain.JSONMap{
					"reader":      "nifti-1",
					"array_dtype": "uint16",
					"width":       float64(512),
				},
			},
		},
		{
			ResourceID:   "file_young_40_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "Norm_young_005_40yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
		{
			ResourceID:   "file_old_plain_72_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "Norm_old_008_72.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    now.Add(-time.Second),
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
		{
			ResourceID:   "file_photo_" + suffix,
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			OriginalName: "prairie-camera.jpg",
			ContentType:  "image/jpeg",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    now.Add(3 * time.Second),
			Metadata: domain.JSONMap{
				"exif": domain.JSONMap{
					"camera_model":    "Sony A1",
					"focal_length_mm": float64(35),
					"iso":             float64(800),
				},
				"image_header": domain.JSONMap{
					"format": "jpeg",
					"width":  float64(2048),
					"height": float64(1024),
				},
			},
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	ageMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser age query: %v", err)
	}
	if got := resourceIDs(ageMatches.Resources); !reflect.DeepEqual(got, []string{"file_old_81_" + suffix, "file_old_64_" + suffix, "file_old_plain_72_" + suffix}) {
		t.Fatalf("age query resources = %v, want filename-derived old subjects", got)
	}

	combined, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "NPH age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser combined query: %v", err)
	}
	if got := resourceIDs(combined.Resources); !reflect.DeepEqual(got, []string{"file_old_64_" + suffix, "file_old_plain_72_" + suffix}) {
		t.Fatalf("combined query resources = %v, want only NPH subject over 60", got)
	}

	nifti, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "*.nii.gz",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser glob query: %v", err)
	}
	if got := resourceIDs(nifti.Resources); !reflect.DeepEqual(got, []string{"file_young_40_" + suffix, "file_old_81_" + suffix, "file_old_64_" + suffix, "file_old_plain_72_" + suffix}) {
		t.Fatalf("glob query resources = %v, want NIfTI gz resources", got)
	}

	niftiFamily, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "*.nii",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser NIfTI-family glob query: %v", err)
	}
	if got := resourceIDs(niftiFamily.Resources); !reflect.DeepEqual(got, []string{"file_young_40_" + suffix, "file_old_81_" + suffix, "file_old_64_" + suffix, "file_old_plain_72_" + suffix}) {
		t.Fatalf("*.nii query resources = %v, want all NIfTI resources including .nii.gz", got)
	}

	headerMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "width > 1000",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser header query: %v", err)
	}
	if got := resourceIDs(headerMatches.Resources); !reflect.DeepEqual(got, []string{"file_photo_" + suffix}) {
		t.Fatalf("header query resources = %v, want image header width match", got)
	}

	exifMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "focal_length > 30",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser exif query: %v", err)
	}
	if got := resourceIDs(exifMatches.Resources); !reflect.DeepEqual(got, []string{"file_photo_" + suffix}) {
		t.Fatalf("EXIF query resources = %v, want focal-length match", got)
	}
}

func TestPostgresStoreBackfillsResourceSearchFactsForExistingResources(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	userID := "pg-search-backfill-" + suffix
	orgID := "org-search-backfill-" + suffix
	now := time.Date(2026, 6, 27, 11, 0, 0, 0, time.UTC)
	for _, resource := range []struct {
		id       string
		name     string
		metadata string
		created  time.Time
	}{
		{
			id:       "legacy_old_72_" + suffix,
			name:     "NPH_shunt_013_72yo.nii.gz",
			metadata: `{"label":"NPH"}`,
			created:  now,
		},
		{
			id:       "legacy_old_40_" + suffix,
			name:     "Norm_young_005_40yo.nii.gz",
			metadata: `{"label":"control"}`,
			created:  now.Add(time.Second),
		},
	} {
		_, err := pool.Exec(ctx, `
INSERT INTO control_resources (
  resource_id, owner_user_id, owner_org_id, original_name, content_type, size_bytes, source_type,
  resource_kind, status, created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, 'application/gzip', 128, 'upload', 'file', 'active', $5, $5, $6::jsonb)`,
			resource.id,
			userID,
			orgID,
			resource.name,
			resource.created,
			resource.metadata,
		)
		if err != nil {
			t.Fatalf("insert legacy resource %s: %v", resource.id, err)
		}
	}

	if err := BackfillPostgresResourceSearchIndexes(ctx, pool); err != nil {
		t.Fatalf("BackfillPostgresResourceSearchIndexes: %v", err)
	}

	ageMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser age query: %v", err)
	}
	if got := resourceIDs(ageMatches.Resources); !reflect.DeepEqual(got, []string{"legacy_old_72_" + suffix}) {
		t.Fatalf("backfilled age query resources = %v, want legacy old subject", got)
	}

	niftiMatches, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "*.nii",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser NIfTI query: %v", err)
	}
	if got := resourceIDs(niftiMatches.Resources); !reflect.DeepEqual(got, []string{"legacy_old_40_" + suffix, "legacy_old_72_" + suffix}) {
		t.Fatalf("backfilled *.nii query resources = %v, want all legacy NIfTI resources", got)
	}
}

func TestPostgresStoreResourceCollectionBulkMembership(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("test"), "-", "_")
	userID := "collection-user-" + suffix
	orgID := "collection-org-" + suffix
	now := domain.Now()
	resourceIDs := []string{"file_collection_a_" + suffix, "file_collection_b_" + suffix}
	for index, resourceID := range resourceIDs {
		if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   resourceID,
			OriginalName: "collection-" + strconv.Itoa(index) + ".nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    int64(10 + index),
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			ProjectID:    "collection-project-" + suffix,
			Status:       "active",
			CreatedAt:    now.Add(time.Duration(index) * time.Second),
			UpdatedAt:    now.Add(time.Duration(index) * time.Second),
			Tags:         []string{[]string{"NPH", "MRI"}[index]},
			Metadata: domain.JSONMap{
				"data_agent": domain.JSONMap{
					"caption_resources": domain.JSONMap{
						"caption": "Folder searchable shunt imaging metadata.",
						"status":  "succeeded",
					},
				},
			},
		}); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resourceID, err)
		}
	}

	collection, err := store.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_" + suffix,
		OwnerUserID:    userID,
		OwnerOrgID:     orgID,
		ProjectID:      "collection-project-" + suffix,
		Name:           "Postgres NPH folder",
		CollectionType: "folder",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	added, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   userID,
		OwnerOrgID:    orgID,
		ResourceIDs:   resourceIDs,
		AddedByUserID: userID,
		AddedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}
	if added.AddedCount != 2 || added.Collection.ResourceCount != 2 {
		t.Fatalf("added = %+v, want two resource memberships", added)
	}
	page, err := store.ListResourceCollectionsForUser(ctx, domain.ResourceCollectionListInput{
		UserID: userID,
		OrgID:  orgID,
		Type:   "folder",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceCollectionsForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Collections) != 1 || page.Collections[0].ResourceCount != 2 {
		t.Fatalf("collection page = %+v, want one folder with two members", page)
	}
	members, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       userID,
		OrgID:        orgID,
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser: %v", err)
	}
	if members.TotalCount != 2 || len(members.Resources) != 2 || members.Resources[0].ResourceID != resourceIDs[0] || members.Resources[1].ResourceID != resourceIDs[1] {
		t.Fatalf("members = %+v, want inserted resources in collection order", members)
	}
	metadataMembers, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       userID,
		OrgID:        orgID,
		Query:        "shunt imaging metadata",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser metadata query: %v", err)
	}
	if metadataMembers.TotalCount != 2 || len(metadataMembers.Resources) != 2 {
		t.Fatalf("metadata members = %+v, want both captioned collection resources", metadataMembers)
	}
	taggedMembers, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       userID,
		OrgID:        orgID,
		Tags:         []string{"nph"},
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser tag filter: %v", err)
	}
	if taggedMembers.TotalCount != 1 || len(taggedMembers.Resources) != 1 || taggedMembers.Resources[0].ResourceID != resourceIDs[0] {
		t.Fatalf("tagged members = %+v, want only NPH-tagged collection resource", taggedMembers)
	}
	if got := taggedMembers.Resources[0].Tags; !reflect.DeepEqual(got, []string{"NPH"}) {
		t.Fatalf("tagged member tags = %#v, want display tag from resource metadata", got)
	}
}

func TestPostgresStoreResourceCollectionShareGrantAppliesToFutureMembers(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("folder_acl"), "-", "_")
	aliceID := "alice-" + suffix
	bobID := "bob-" + suffix
	orgID := "org-a-" + suffix
	now := time.Date(2026, 6, 8, 20, 0, 0, 0, time.UTC)
	initialResourceID := "file_folder_acl_initial_" + suffix
	futureResourceID := "file_folder_acl_future_" + suffix
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   initialResourceID,
			OriginalName: "initial-folder-acl.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    12,
			OwnerUserID:  aliceID,
			OwnerOrgID:   orgID,
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   futureResourceID,
			OriginalName: "future-folder-acl.nii.gz",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    15,
			OwnerUserID:  aliceID,
			OwnerOrgID:   orgID,
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := store.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	collection, err := store.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_folder_acl_" + suffix,
		OwnerUserID:    aliceID,
		OwnerOrgID:     orgID,
		Name:           "Postgres inherited ACL folder",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   aliceID,
		OwnerOrgID:    orgID,
		ResourceIDs:   []string{initialResourceID},
		AddedByUserID: aliceID,
		AddedAt:       now,
	}); err != nil {
		t.Fatalf("AddResourcesToCollection initial: %v", err)
	}
	shareResult, err := store.CreateResourceCollectionShareGrant(ctx, domain.CreateResourceCollectionShareGrantInput{
		GrantID:         "collection_grant_folder_acl_" + suffix,
		CollectionID:    collection.CollectionID,
		OwnerUserID:     aliceID,
		OwnerOrgID:      orgID,
		GranteeUserID:   bobID,
		GranteeOrgID:    "org-b",
		Role:            "read",
		CreatedByUserID: aliceID,
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"reason": "folder review"},
	})
	if err != nil {
		t.Fatalf("CreateResourceCollectionShareGrant: %v", err)
	}
	if shareResult.Grant.CollectionID != collection.CollectionID || len(shareResult.ResourceGrants) != 1 {
		t.Fatalf("shareResult = %+v, want collection grant and initial resource grant", shareResult)
	}
	added, err := store.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   aliceID,
		OwnerOrgID:    orgID,
		ResourceIDs:   []string{futureResourceID},
		AddedByUserID: aliceID,
		AddedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("AddResourcesToCollection future: %v", err)
	}
	if len(added.InheritedShareGrants) != 1 {
		t.Fatalf("future inherited grants = %+v, want one inherited grant", added.InheritedShareGrants)
	}
	futureGrant := added.InheritedShareGrants[0]
	if futureGrant.ResourceID != futureResourceID || futureGrant.GranteeUserID != bobID || futureGrant.Role != "read" || futureGrant.Status != "active" {
		t.Fatalf("future inherited grant = %+v, want active Bob read grant", futureGrant)
	}
	if futureGrant.Metadata["collection_share_grant_id"] != shareResult.Grant.GrantID || futureGrant.Metadata["source"] != "resource_collection_share_inherited" {
		t.Fatalf("future inherited metadata = %+v, want collection-share provenance", futureGrant.Metadata)
	}
	bobCollections, err := store.ListResourceCollectionsForUser(ctx, domain.ResourceCollectionListInput{
		UserID: bobID,
		OrgID:  "org-b",
		Query:  "inherited ACL",
		Type:   "folder",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourceCollectionsForUser as bob: %v", err)
	}
	if bobCollections.TotalCount != 1 || len(bobCollections.Collections) != 1 || bobCollections.Collections[0].CollectionID != collection.CollectionID {
		t.Fatalf("bob collections = %+v, want shared folder", bobCollections)
	}
	bobCollection, err := store.GetResourceCollectionForUser(ctx, collection.CollectionID, bobID, "org-b")
	if err != nil {
		t.Fatalf("GetResourceCollectionForUser as bob: %v", err)
	}
	if bobCollection.ResourceCount != 2 {
		t.Fatalf("bob collection = %+v, want two visible members", bobCollection)
	}
	bobMembers, err := store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       bobID,
		OrgID:        "org-b",
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForCollectionForUser as bob: %v", err)
	}
	if bobMembers.TotalCount != 2 || len(bobMembers.Resources) != 2 {
		t.Fatalf("bob folder resources = %+v, want shared current and future folder members", bobMembers)
	}
	if !bobMembers.Resources[0].ShareSummary.SharedWithMe || !bobMembers.Resources[1].ShareSummary.SharedWithMe {
		t.Fatalf("bob folder share summaries = %+v, %+v, want shared_with_me", bobMembers.Resources[0].ShareSummary, bobMembers.Resources[1].ShareSummary)
	}
	bobPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: bobID,
		OrgID:  "org-b",
		Query:  "folder-acl",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser as bob: %v", err)
	}
	if bobPage.TotalCount != 2 || len(bobPage.Resources) != 2 {
		t.Fatalf("bob resources = %+v, want current and future folder members", bobPage)
	}
}

func TestPostgresStoreDatasetSnapshotFreezesResourceManifest(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("snapshot"), "-", "_")
	userID := "snapshot-user-" + suffix
	orgID := "snapshot-org-" + suffix
	now := domain.Now()
	resourceID := "file_snapshot_" + suffix
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "nph-postgres.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    64,
		SHA256:       "sha-before-" + suffix,
		OwnerUserID:  userID,
		OwnerOrgID:   orgID,
		ProjectID:    "snapshot-project-" + suffix,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	snapshot, entries, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_" + suffix,
		OwnerUserID:     userID,
		OwnerOrgID:      orgID,
		ProjectID:       "snapshot-project-" + suffix,
		Name:            "Postgres frozen dataset",
		ResourceIDs:     []string{resourceID},
		CreatedByUserID: userID,
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}
	if snapshot.ResourceCount != 1 || snapshot.TotalBytes != 64 || len(entries) != 1 || entries[0].SHA256 != "sha-before-"+suffix {
		t.Fatalf("snapshot = %+v entries=%+v, want frozen resource manifest", snapshot, entries)
	}
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "nph-postgres-renamed.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    128,
		SHA256:       "sha-after-" + suffix,
		OwnerUserID:  userID,
		OwnerOrgID:   orgID,
		ProjectID:    "snapshot-project-" + suffix,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now.Add(2 * time.Second),
	}); err != nil {
		t.Fatalf("mutate resource: %v", err)
	}
	_, loadedEntries, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, userID, orgID)
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser: %v", err)
	}
	if len(loadedEntries) != 1 || loadedEntries[0].OriginalName != "nph-postgres.nii.gz" || loadedEntries[0].SizeBytes != 64 || loadedEntries[0].SHA256 != "sha-before-"+suffix {
		t.Fatalf("loaded entries = %+v, want immutable manifest", loadedEntries)
	}
}

func TestPostgresStoreDatasetSnapshotShareGrantAllowsCollaboratorRead(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}

	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("snapshot_share"), "-", "_")
	ownerUserID := "snapshot-owner-" + suffix
	ownerOrgID := "snapshot-org-" + suffix
	granteeUserID := "snapshot-grantee-" + suffix
	granteeOrgID := "snapshot-grantee-org-" + suffix
	resourceID := "file_snapshot_share_" + suffix
	now := domain.Now()
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "shared-postgres.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    72,
		SHA256:       "sha-share-" + suffix,
		OwnerUserID:  ownerUserID,
		OwnerOrgID:   ownerOrgID,
		ProjectID:    "snapshot-share-project-" + suffix,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	snapshot, _, err := store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_share_" + suffix,
		OwnerUserID:     ownerUserID,
		OwnerOrgID:      ownerOrgID,
		ProjectID:       "snapshot-share-project-" + suffix,
		Name:            "Shared Postgres dataset",
		ResourceIDs:     []string{resourceID},
		CreatedByUserID: ownerUserID,
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}
	if _, _, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, granteeUserID, granteeOrgID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDatasetSnapshotForUser before grant err = %v, want ErrNotFound", err)
	}
	ownerEvents, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     ownerUserID,
		OrgID:      ownerOrgID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser owner after create: %v", err)
	}
	if ownerEvents.TotalCount != 1 || len(ownerEvents.Events) != 1 || ownerEvents.Events[0].EventType != "dataset_snapshot.created" {
		t.Fatalf("owner events after create = %+v, want created event", ownerEvents)
	}
	grant, err := store.CreateDatasetSnapshotShareGrant(ctx, domain.CreateDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshot.SnapshotID,
		OwnerUserID:     ownerUserID,
		OwnerOrgID:      ownerOrgID,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            "read",
		CreatedByUserID: ownerUserID,
		CreatedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshotShareGrant: %v", err)
	}
	granteeEvents, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     granteeUserID,
		OrgID:      granteeOrgID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser grantee after grant: %v", err)
	}
	if granteeEvents.TotalCount != 2 || len(granteeEvents.Events) != 2 || granteeEvents.Events[0].EventType != "dataset_snapshot.shared" {
		t.Fatalf("grantee events after grant = %+v, want shared then created", granteeEvents)
	}
	loaded, loadedEntries, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, granteeUserID, granteeOrgID)
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser after grant: %v", err)
	}
	if loaded.SnapshotID != snapshot.SnapshotID || loaded.OwnerUserID != ownerUserID || len(loadedEntries) != 1 || loadedEntries[0].SHA256 != "sha-share-"+suffix {
		t.Fatalf("loaded shared snapshot = %+v entries=%+v, want frozen collaborator manifest", loaded, loadedEntries)
	}
	page, err := store.ListDatasetSnapshotsForUser(ctx, domain.DatasetSnapshotListInput{
		UserID: granteeUserID,
		OrgID:  granteeOrgID,
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotsForUser after grant: %v", err)
	}
	if page.TotalCount < 1 {
		t.Fatalf("shared snapshot page = %+v, want at least the granted snapshot", page)
	}
	revoked, err := store.RevokeDatasetSnapshotShareGrant(ctx, domain.RevokeDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshot.SnapshotID,
		GrantID:         grant.GrantID,
		OwnerUserID:     ownerUserID,
		OwnerOrgID:      ownerOrgID,
		RevokedByUserID: ownerUserID,
		RevokedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("RevokeDatasetSnapshotShareGrant: %v", err)
	}
	if revoked.Status != "revoked" || revoked.RevokedAt.IsZero() {
		t.Fatalf("revoked grant = %+v, want revoked lifecycle", revoked)
	}
	ownerEvents, err = store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     ownerUserID,
		OrgID:      ownerOrgID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDatasetSnapshotEventsForUser owner after revoke: %v", err)
	}
	gotEventTypes := []string{}
	for _, event := range ownerEvents.Events {
		gotEventTypes = append(gotEventTypes, event.EventType)
	}
	if !reflect.DeepEqual(gotEventTypes, []string{"dataset_snapshot.share_revoked", "dataset_snapshot.shared", "dataset_snapshot.created"}) {
		t.Fatalf("owner event types after revoke = %v, want revoke/share/create", gotEventTypes)
	}
	if _, _, err := store.GetDatasetSnapshotForUser(ctx, snapshot.SnapshotID, granteeUserID, granteeOrgID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDatasetSnapshotForUser after revoke err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListDatasetSnapshotEventsForUser(ctx, domain.DatasetSnapshotEventListInput{
		SnapshotID: snapshot.SnapshotID,
		UserID:     granteeUserID,
		OrgID:      granteeOrgID,
		Limit:      10,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListDatasetSnapshotEventsForUser grantee after revoke err = %v, want ErrNotFound", err)
	}
}

func TestPostgresStoreDataAgentJobLifecycleRecordsEvents(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("data_agent"), "-", "_")
	userID := "agent-user-" + suffix
	orgID := "agent-org-" + suffix
	now := domain.Now()
	resourceIDs := []string{"file_agent_a_" + suffix, "file_agent_b_" + suffix}
	for index, resourceID := range resourceIDs {
		if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   resourceID,
			OriginalName: "agent-" + strconv.Itoa(index) + ".nii.gz",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    int64(20 + index),
			OwnerUserID:  userID,
			OwnerOrgID:   orgID,
			ProjectID:    "agent-project-" + suffix,
			Status:       "active",
			CreatedAt:    now.Add(time.Duration(index) * time.Second),
			UpdatedAt:    now.Add(time.Duration(index) * time.Second),
		}); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resourceID, err)
		}
	}

	job, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_" + suffix,
		OwnerUserID:     userID,
		OwnerOrgID:      orgID,
		ProjectID:       "agent-project-" + suffix,
		JobType:         "caption_resources",
		ResourceIDs:     resourceIDs,
		InputSelector:   domain.JSONMap{"label": "NPH"},
		CreatedByUserID: userID,
		CreatedAt:       now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	if job.Status != "queued" || job.ResourceCount != 2 || job.ProgressTotal != 2 {
		t.Fatalf("job = %+v, want queued two-resource job", job)
	}
	running, runningEvent, err := store.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID:             job.JobID,
		OwnerUserID:       userID,
		OwnerOrgID:        orgID,
		Status:            "running",
		ProgressCompleted: 1,
		ProgressTotal:     2,
		ActorUserID:       userID,
		ActorOrgID:        orgID,
		Message:           "Captioned first resource",
		UpdatedAt:         now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("UpdateDataAgentJob: %v", err)
	}
	if running.Status != "running" || running.ProgressCompleted != 1 || runningEvent.EventType != "data_agent.job.progressed" {
		t.Fatalf("running job = %+v event=%+v, want progressed job", running, runningEvent)
	}
	canceled, canceledEvent, err := store.ControlDataAgentJob(ctx, domain.ControlDataAgentJobInput{
		JobID:       job.JobID,
		OwnerUserID: userID,
		OwnerOrgID:  orgID,
		Action:      "cancel",
		Reason:      "Paused before retry.",
		ActorUserID: userID,
		ActorOrgID:  orgID,
		TS:          now.Add(4 * time.Second),
	})
	if err != nil {
		t.Fatalf("ControlDataAgentJob cancel: %v", err)
	}
	if canceled.Status != "canceled" || canceledEvent.EventType != "data_agent.job.canceled" {
		t.Fatalf("canceled job = %+v event=%+v, want canceled job", canceled, canceledEvent)
	}
	retried, retriedEvent, err := store.ControlDataAgentJob(ctx, domain.ControlDataAgentJobInput{
		JobID:       job.JobID,
		OwnerUserID: userID,
		OwnerOrgID:  orgID,
		Action:      "retry",
		Reason:      "Retry from Postgres test.",
		ActorUserID: userID,
		ActorOrgID:  orgID,
		TS:          now.Add(5 * time.Second),
	})
	if err != nil {
		t.Fatalf("ControlDataAgentJob retry: %v", err)
	}
	if retried.Status != "queued" || retried.ProgressCompleted != 0 || retried.Error != "" || retriedEvent.EventType != "data_agent.job.retried" {
		t.Fatalf("retried job = %+v event=%+v, want reset queued job", retried, retriedEvent)
	}
	loaded, err := store.GetDataAgentJobForUser(ctx, job.JobID, userID, orgID)
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.JobID != job.JobID || loaded.InputSelector["label"] != "NPH" {
		t.Fatalf("loaded job = %+v, want persisted selector", loaded)
	}
	page, err := store.ListDataAgentJobsForUser(ctx, domain.DataAgentJobListInput{
		UserID:  userID,
		OrgID:   orgID,
		Status:  "queued",
		JobType: "caption_resources",
		Limit:   10,
	})
	if err != nil {
		t.Fatalf("ListDataAgentJobsForUser: %v", err)
	}
	if page.TotalCount != 1 || len(page.Jobs) != 1 || page.Jobs[0].JobID != job.JobID {
		t.Fatalf("job page = %+v, want created job", page)
	}
	events, err := store.ListDataAgentJobEvents(ctx, job.JobID, userID, orgID, 10)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	if len(events) != 4 || events[0].EventType != "data_agent.job.created" || events[1].EventType != "data_agent.job.progressed" || events[3].EventType != "data_agent.job.retried" {
		t.Fatalf("events = %+v, want ordered lifecycle events", events)
	}
}

func TestPostgresStoreRecoversDispatchFailedDataAgentJobWithoutLease(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}

	store := NewPostgresStore(pool)
	suffix := strings.ReplaceAll(domain.NewID("data_agent_retry"), "-", "_")
	userID := "agent-user-" + suffix
	orgID := "agent-org-" + suffix
	now := domain.Now()
	resourceID := "file_agent_retry_" + suffix
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "retry.nii.gz",
		ResourceKind: "file",
		SourceType:   "upload",
		SizeBytes:    42,
		OwnerUserID:  userID,
		OwnerOrgID:   orgID,
		ProjectID:    "agent-project-" + suffix,
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	job, err := store.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_retry_" + suffix,
		OwnerUserID:     userID,
		OwnerOrgID:      orgID,
		ProjectID:       "agent-project-" + suffix,
		JobType:         "extract_metadata",
		ResourceIDs:     []string{resourceID},
		InputSelector:   domain.JSONMap{"resource_ids": []any{resourceID}},
		CreatedByUserID: userID,
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	if _, err := store.AppendDataAgentJobEvent(ctx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.dispatch_failed",
		ActorUserID: userID,
		ActorOrgID:  orgID,
		TS:          now.Add(2 * time.Second),
		Message:     "Data Agent job dispatch failed.",
		Metadata:    domain.JSONMap{"error": "nats publish unavailable"},
	}); err != nil {
		t.Fatalf("AppendDataAgentJobEvent dispatch_failed: %v", err)
	}

	result, err := store.RecoverExpiredDataAgentJobLeases(ctx, domain.RecoverExpiredDataAgentJobLeasesInput{
		Now:    now.Add(3 * time.Second),
		Reason: "automatic expired data-agent lease recovery",
		Limit:  1000,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredDataAgentJobLeases: %v", err)
	}
	found := false
	for _, recovered := range result.RequeuedJobs {
		if recovered.JobID != job.JobID {
			continue
		}
		found = true
		if recovered.Status != "queued" || recovered.OwnerUserID != userID || recovered.OwnerOrgID != orgID {
			t.Fatalf("recovered retry job = %+v, want queued owner-scoped job", recovered)
		}
	}
	if !found {
		t.Fatalf("recovery result = %+v, want dispatch-failed job %s", result, job.JobID)
	}
	events, err := store.ListDataAgentJobEvents(ctx, job.JobID, userID, orgID, 10)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	if got := events[len(events)-1].EventType; got != "data_agent.job.dispatch_failed" {
		t.Fatalf("last event after store retry lookup = %s, want dispatch_failed until app publishes", got)
	}

	if _, err := store.AppendDataAgentJobEvent(ctx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.dispatched",
		ActorUserID: userID,
		ActorOrgID:  orgID,
		TS:          now.Add(4 * time.Second),
		Message:     "Data Agent job dispatched after retry.",
		Metadata:    domain.JSONMap{"dispatch_id": "dispatch_retry_success"},
	}); err != nil {
		t.Fatalf("AppendDataAgentJobEvent dispatched: %v", err)
	}
	result, err = store.RecoverExpiredDataAgentJobLeases(ctx, domain.RecoverExpiredDataAgentJobLeasesInput{
		Now:    now.Add(5 * time.Second),
		Reason: "automatic expired data-agent lease recovery",
		Limit:  1000,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredDataAgentJobLeases after dispatched: %v", err)
	}
	for _, recovered := range result.RequeuedJobs {
		if recovered.JobID == job.JobID {
			t.Fatalf("recovery result after dispatched = %+v, want job excluded once latest event is dispatched", result)
		}
	}
}

func TestPostgresStoreCompleteRunRepairsSucceededRunMissingResponseText(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-terminal-repair-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres terminal repair",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "repair terminal response",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "repair terminal response"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus succeeded with empty response: %v", err)
	}

	repaired, err := store.CompleteRun(ctx, domain.CompleteRunInput{
		RunID:        run.RunID,
		ResponseText: "Recovered Postgres final answer.",
	})
	if err != nil {
		t.Fatalf("CompleteRun repair: %v", err)
	}
	if repaired.Status != domain.RunStatusSucceeded || repaired.ResponseText != "Recovered Postgres final answer." {
		t.Fatalf("repaired run = %+v, want succeeded with recovered response text", repaired)
	}
	messages, err := store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("messages = %d, want %d user+assistant messages: %+v", got, want, messages)
	}
	if messages[1].Role != "assistant" || messages[1].Content != "Recovered Postgres final answer." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want recovered response owned by run", messages[1])
	}
}

func TestPostgresStoreClearRunLeaseEvictsAnyActiveToken(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-clear-lease-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres clear lease",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "recover leased run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "recover leased run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	lease, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Hour,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	cleared, ok, err := store.ClearRunLease(ctx, run.RunID)
	if err != nil {
		t.Fatalf("ClearRunLease: %v", err)
	}
	if !ok || cleared.LeaseToken != lease.LeaseToken || cleared.WorkerID != "worker-a" {
		t.Fatalf("cleared lease = %+v ok=%v, want worker-a lease", cleared, ok)
	}
	if _, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
		Now:        now.Add(time.Minute),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewRunLease after clear err = %v, want ErrConflict", err)
	}
	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Hour,
		Now:      now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease replacement: %v", err)
	}
}

func TestPostgresStoreCreateAndListOrganizations(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	orgID := "pg-org-" + domain.NewID("test")
	org, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{
		OrgID:    orgID,
		Name:     "Postgres Organization",
		Status:   "active",
		Metadata: domain.JSONMap{"source": "postgres_test"},
	})
	if err != nil {
		t.Fatalf("CreateOrganization: %v", err)
	}
	if org.OrgID != orgID || org.Name != "Postgres Organization" {
		t.Fatalf("organization = %+v, want persisted org", org)
	}

	orgs, err := store.ListOrganizations(ctx, 10, orgID)
	if err != nil {
		t.Fatalf("ListOrganizations: %v", err)
	}
	if len(orgs) != 1 || orgs[0].OrgID != orgID {
		t.Fatalf("organizations = %+v, want created org", orgs)
	}
	fetched, found, err := store.GetOrganization(ctx, " "+orgID+" ")
	if err != nil {
		t.Fatalf("GetOrganization: %v", err)
	}
	if !found || fetched.OrgID != orgID || fetched.Metadata["source"] != "postgres_test" {
		t.Fatalf("GetOrganization = %+v found=%t, want created org", fetched, found)
	}
	if _, found, err := store.GetOrganization(ctx, "missing-"+orgID); err != nil || found {
		t.Fatalf("GetOrganization missing found=%t err=%v, want not found without error", found, err)
	}
}

func TestPostgresStoreUpsertsAndListsWorkerHeartbeats(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	workerID := "pg-worker-" + domain.NewID("test")
	started := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	firstBeat := started.Add(10 * time.Second)
	secondBeat := started.Add(70 * time.Second)
	if _, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        workerID,
		WorkerKind:      "deepagents",
		Status:          "idle",
		Hostname:        "pg-host",
		Version:         "v1",
		StartedAt:       started,
		LastHeartbeatAt: firstBeat,
		Metadata:        domain.JSONMap{"durable": "ultra-deepagents-worker"},
	}); err != nil {
		t.Fatalf("UpsertWorkerHeartbeat first: %v", err)
	}
	updated, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        workerID,
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    "run_123",
		Hostname:        "pg-host",
		Version:         "v2",
		LastHeartbeatAt: secondBeat,
		Metadata:        domain.JSONMap{"active_tasks": float64(1)},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat second: %v", err)
	}
	if updated.Status != "busy" || updated.CurrentRunID != "run_123" || updated.Version != "v2" {
		t.Fatalf("updated worker = %+v, want busy worker", updated)
	}
	if !updated.StartedAt.Equal(started) {
		t.Fatalf("started_at = %s, want original %s", updated.StartedAt, started)
	}
	fetched, heartbeatFound, err := store.GetWorkerHeartbeat(ctx, workerID)
	if err != nil {
		t.Fatalf("GetWorkerHeartbeat: %v", err)
	}
	if !heartbeatFound || fetched.WorkerID != workerID || fetched.CurrentRunID != "run_123" || fetched.Metadata["active_tasks"] != float64(1) {
		t.Fatalf("GetWorkerHeartbeat = %+v found=%t, want updated worker", fetched, heartbeatFound)
	}
	if _, found, err := store.GetWorkerHeartbeat(ctx, "missing-"+workerID); err != nil || found {
		t.Fatalf("GetWorkerHeartbeat missing found=%t err=%v, want not found without error", found, err)
	}
	workers, err := store.ListWorkerHeartbeats(ctx, 100)
	if err != nil {
		t.Fatalf("ListWorkerHeartbeats: %v", err)
	}
	found := false
	for _, worker := range workers {
		if worker.WorkerID != workerID {
			continue
		}
		found = true
		if worker.Status != "busy" || worker.CurrentRunID != "run_123" || worker.Metadata["active_tasks"] != float64(1) {
			t.Fatalf("listed worker = %+v, want updated worker", worker)
		}
	}
	if !found {
		t.Fatalf("workers = %+v, want %s", workers, workerID)
	}
}

func uploadSessionEventRecordsContain(events []domain.UploadSessionEventRecord, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}
