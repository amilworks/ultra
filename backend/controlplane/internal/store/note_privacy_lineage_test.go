package store

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestMemoryRunNotePrivacyLineagePropagatesAndStaysThreadScoped(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	history := NewMemoryStore()
	userID := "memory-note-lineage-owner"

	privateThread, err := history.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "Private Note thread"})
	if err != nil {
		t.Fatalf("CreateThread private: %v", err)
	}
	noteRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: privateThread.ThreadID,
		UserID:   userID,
		Goal:     "Use my Note",
		Metadata: domain.JSONMap{"selection_context": domain.JSONMap{
			domain.NoteAccessSelectionKey: domain.JSONMap{"mode": "search", "notes": []any{}},
		}},
	})
	if err != nil {
		t.Fatalf("CreateRun Note: %v", err)
	}
	if noteRun.Metadata[domain.NotePrivacyLineageMetadataKey] != true {
		t.Fatalf("Note run metadata = %+v, want server-authored privacy marker", noteRun.Metadata)
	}
	noteRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: noteRun.RunID, ResponseText: "private-note-lineage-source",
	})
	if err != nil {
		t.Fatalf("CompleteRun Note: %v", err)
	}

	followupRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: privateThread.ThreadID,
		UserID:   userID,
		Goal:     "Explain that answer",
	})
	if err != nil {
		t.Fatalf("CreateRun follow-up: %v", err)
	}
	if !domain.RunHasNotePrivacyLineage(followupRun) || domain.RunHasNoteAccessSelection(followupRun) {
		t.Fatalf("follow-up metadata = %+v, want inherited lineage without Notes authority", followupRun.Metadata)
	}
	followupRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: followupRun.RunID, ResponseText: "private-note-lineage-followup",
	})
	if err != nil {
		t.Fatalf("CompleteRun follow-up: %v", err)
	}
	// Simulate rows written before lineage propagation was deployed: the direct
	// run retains only its legacy note_access marker and its descendant has no
	// marker at all. Cross-conversation reads must classify the whole thread.
	history.mu.Lock()
	legacyNoteRun := history.runs[noteRun.RunID]
	delete(legacyNoteRun.Metadata, domain.NotePrivacyLineageMetadataKey)
	history.runs[noteRun.RunID] = legacyNoteRun
	unmarkedFollowup := history.runs[followupRun.RunID]
	delete(unmarkedFollowup.Metadata, domain.NotePrivacyLineageMetadataKey)
	history.runs[followupRun.RunID] = unmarkedFollowup
	history.mu.Unlock()

	ordinaryThread, err := history.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "Ordinary thread"})
	if err != nil {
		t.Fatalf("CreateThread ordinary: %v", err)
	}
	spoofedMetadata := domain.JSONMap{domain.NotePrivacyLineageMetadataKey: true}
	ordinaryRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: ordinaryThread.ThreadID,
		UserID:   userID,
		Goal:     "Independent ordinary work",
		Metadata: spoofedMetadata,
	})
	if err != nil {
		t.Fatalf("CreateRun ordinary: %v", err)
	}
	if domain.RunHasNotePrivacyLineage(ordinaryRun) {
		t.Fatalf("caller spoof minted privacy lineage: %+v", ordinaryRun.Metadata)
	}
	if spoofedMetadata[domain.NotePrivacyLineageMetadataKey] != true {
		t.Fatalf("CreateRun mutated caller metadata: %+v", spoofedMetadata)
	}
	ordinaryRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: ordinaryRun.RunID, ResponseText: "ordinary-lineage-control",
	})
	if err != nil {
		t.Fatalf("CompleteRun ordinary: %v", err)
	}

	for _, query := range []string{"private-note-lineage-source", "private-note-lineage-followup"} {
		hits, err := history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: query})
		if err != nil || len(hits) != 0 {
			t.Fatalf("episodic search %q = %+v err=%v, want no private lineage", query, hits, err)
		}
	}
	hits, err := history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: "ordinary-lineage-control"})
	if err != nil || len(hits) != 1 || hits[0].RunID != ordinaryRun.RunID {
		t.Fatalf("ordinary episodic search = %+v err=%v", hits, err)
	}

	unscoped, err := history.ListRunsForUser(ctx, userID, "", "", 20, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser unscoped: %v", err)
	}
	responses := make(map[string]string, len(unscoped))
	followupOffset := -1
	for index, run := range unscoped {
		responses[run.RunID] = run.ResponseText
		if run.RunID == followupRun.RunID {
			followupOffset = index
		}
	}
	if responses[noteRun.RunID] != "" || responses[followupRun.RunID] != "" {
		t.Fatalf("unscoped responses leaked private lineage: %+v", responses)
	}
	if responses[ordinaryRun.RunID] != ordinaryRun.ResponseText {
		t.Fatalf("unscoped ordinary response = %q, want %q", responses[ordinaryRun.RunID], ordinaryRun.ResponseText)
	}
	page, err := history.ListRunsForUser(ctx, userID, "", "", 1, followupOffset)
	if err != nil || len(page) != 1 || page[0].RunID != followupRun.RunID || page[0].ResponseText != "" {
		t.Fatalf("descendant-only unscoped page = %+v err=%v", page, err)
	}
	allRuns, err := history.ListRuns(ctx, "", "", 20, 0)
	if err != nil {
		t.Fatalf("ListRuns unscoped: %v", err)
	}
	for _, run := range allRuns {
		if (run.RunID == noteRun.RunID || run.RunID == followupRun.RunID) && run.ResponseText != "" {
			t.Fatalf("unscoped internal list leaked private lineage: %+v", run)
		}
	}

	scoped, err := history.ListRunsForUser(ctx, userID, privateThread.ThreadID, "", 20, 0)
	if err != nil || len(scoped) != 2 {
		t.Fatalf("ListRunsForUser source thread = %+v err=%v", scoped, err)
	}
	for _, run := range scoped {
		if run.ResponseText == "" {
			t.Fatalf("source-thread response was blanked: %+v", run)
		}
	}
}

func TestPostgresRunNotePrivacyLineagePropagatesAndStaysThreadScoped(t *testing.T) {
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
	history := NewPostgresStore(pool)
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	userID := "pg-note-lineage-owner-" + suffix

	privateThread, err := history.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "Private lineage " + suffix})
	if err != nil {
		t.Fatalf("CreateThread private: %v", err)
	}
	noteRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: privateThread.ThreadID,
		UserID:   userID,
		Goal:     "Use my Note " + suffix,
		Metadata: domain.JSONMap{"selection_context": domain.JSONMap{
			domain.NoteAccessSelectionKey: domain.JSONMap{"mode": "search", "notes": []any{}},
		}},
	})
	if err != nil {
		t.Fatalf("CreateRun Note: %v", err)
	}
	if noteRun.Metadata[domain.NotePrivacyLineageMetadataKey] != true {
		t.Fatalf("Note run metadata = %+v, want server-authored privacy marker", noteRun.Metadata)
	}
	noteRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: noteRun.RunID, ResponseText: "pg-private-source-" + suffix,
	})
	if err != nil {
		t.Fatalf("CompleteRun Note: %v", err)
	}

	// Simulate a run written before the dedicated lineage marker shipped. The
	// legacy note_access key must still taint the next run during a rolling deploy.
	if _, err := pool.Exec(ctx, `
UPDATE control_runs
SET metadata = metadata - $2
WHERE run_id = $1`, noteRun.RunID, domain.NotePrivacyLineageMetadataKey); err != nil {
		t.Fatalf("remove marker to simulate legacy row: %v", err)
	}
	followupRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: privateThread.ThreadID,
		UserID:   userID,
		Goal:     "Explain that private answer " + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun follow-up: %v", err)
	}
	if !domain.RunHasNotePrivacyLineage(followupRun) || domain.RunHasNoteAccessSelection(followupRun) {
		t.Fatalf("follow-up metadata = %+v, want inherited lineage without Notes authority", followupRun.Metadata)
	}
	followupRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: followupRun.RunID, ResponseText: "pg-private-followup-" + suffix,
	})
	if err != nil {
		t.Fatalf("CompleteRun follow-up: %v", err)
	}
	// Simulate the descendant being inserted by an older replica after the
	// legacy direct run: it has neither a direct scope nor the new lineage key.
	if _, err := pool.Exec(ctx, `
UPDATE control_runs
SET metadata = metadata - $2
WHERE run_id = $1`, followupRun.RunID, domain.NotePrivacyLineageMetadataKey); err != nil {
		t.Fatalf("remove descendant marker to simulate rolling upgrade: %v", err)
	}

	ordinaryThread, err := history.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "Ordinary lineage " + suffix})
	if err != nil {
		t.Fatalf("CreateThread ordinary: %v", err)
	}
	ordinaryRun, err := history.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: ordinaryThread.ThreadID,
		UserID:   userID,
		Goal:     "Independent ordinary work " + suffix,
		Metadata: domain.JSONMap{domain.NotePrivacyLineageMetadataKey: true},
	})
	if err != nil {
		t.Fatalf("CreateRun ordinary: %v", err)
	}
	if domain.RunHasNotePrivacyLineage(ordinaryRun) {
		t.Fatalf("caller spoof minted Postgres privacy lineage: %+v", ordinaryRun.Metadata)
	}
	ordinaryRun, err = history.CompleteRun(ctx, domain.CompleteRunInput{
		RunID: ordinaryRun.RunID, ResponseText: "pg-ordinary-control-" + suffix,
	})
	if err != nil {
		t.Fatalf("CompleteRun ordinary: %v", err)
	}

	for _, query := range []string{noteRun.ResponseText, followupRun.ResponseText} {
		hits, err := history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: query})
		if err != nil || len(hits) != 0 {
			t.Fatalf("Postgres episodic search %q = %+v err=%v, want no private lineage", query, hits, err)
		}
	}
	hits, err := history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: ordinaryRun.ResponseText})
	if err != nil || len(hits) != 1 || hits[0].RunID != ordinaryRun.RunID {
		t.Fatalf("ordinary Postgres episodic search = %+v err=%v", hits, err)
	}

	unscoped, err := history.ListRunsForUser(ctx, userID, "", "", 20, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser unscoped: %v", err)
	}
	responses := make(map[string]string, len(unscoped))
	followupOffset := -1
	for index, run := range unscoped {
		responses[run.RunID] = run.ResponseText
		if run.RunID == followupRun.RunID {
			followupOffset = index
		}
	}
	if responses[noteRun.RunID] != "" || responses[followupRun.RunID] != "" {
		t.Fatalf("unscoped Postgres responses leaked private lineage: %+v", responses)
	}
	if responses[ordinaryRun.RunID] != ordinaryRun.ResponseText {
		t.Fatalf("unscoped Postgres ordinary response = %q, want %q", responses[ordinaryRun.RunID], ordinaryRun.ResponseText)
	}
	page, err := history.ListRunsForUser(ctx, userID, "", "", 1, followupOffset)
	if err != nil || len(page) != 1 || page[0].RunID != followupRun.RunID || page[0].ResponseText != "" {
		t.Fatalf("Postgres descendant-only unscoped page = %+v err=%v", page, err)
	}
	allRuns, err := history.ListRuns(ctx, "", "", 500, 0)
	if err != nil {
		t.Fatalf("ListRuns Postgres unscoped: %v", err)
	}
	for _, run := range allRuns {
		if (run.RunID == noteRun.RunID || run.RunID == followupRun.RunID) && run.ResponseText != "" {
			t.Fatalf("unscoped Postgres internal list leaked private lineage: %+v", run)
		}
	}

	scoped, err := history.ListRunsForUser(ctx, userID, privateThread.ThreadID, "", 20, 0)
	if err != nil || len(scoped) != 2 {
		t.Fatalf("ListRunsForUser Postgres source thread = %+v err=%v", scoped, err)
	}
	for _, run := range scoped {
		if run.ResponseText == "" {
			t.Fatalf("Postgres source-thread response was blanked: %+v", run)
		}
	}
}
