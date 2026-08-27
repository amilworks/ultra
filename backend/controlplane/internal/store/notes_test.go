package store

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestNoteMatchSnippetPreservesUnicodeRuneBoundaries(t *testing.T) {
	t.Parallel()
	body := strings.Repeat("α", 130) + " ẞ-value follows"
	snippet := noteMatchSnippet(body, "ß-value", 500)
	if !utf8.ValidString(snippet) {
		t.Fatalf("snippet is invalid UTF-8: %q", snippet)
	}
	if !strings.Contains(snippet, "ẞ-value") {
		t.Fatalf("snippet %q does not contain the case-folded match", snippet)
	}
}

func TestMemoryNoteOrderingUsesContentRecencyAndDeterministicTies(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	base := domain.Now().Add(-4 * time.Hour)
	oldPinned, err := notes.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_old_pinned", UserID: "ada", Title: "Archive",
		BodyMarkdown: "old body matchword", Pinned: false,
		EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: base,
	})
	if err != nil {
		t.Fatalf("CreateNote old: %v", err)
	}
	newer, err := notes.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_newer", UserID: "ada", Title: "Current",
		BodyMarkdown: "new body matchword", EditorMode: domain.NoteEditorModeMarkdown,
		CreatedAt: base.Add(time.Hour),
	})
	if err != nil {
		t.Fatalf("CreateNote newer: %v", err)
	}
	pinned := true
	oldPinned, err = notes.UpdateNoteForUser(ctx, oldPinned.NoteID, "ada", domain.NoteUpdateInput{
		ExpectedRevision: oldPinned.Revision, Pinned: &pinned,
	})
	if err != nil {
		t.Fatalf("pin old Note: %v", err)
	}
	if !oldPinned.ContentUpdatedAt.Equal(base) || !oldPinned.UpdatedAt.After(oldPinned.ContentUpdatedAt) {
		t.Fatalf("pin changed content recency: %+v", oldPinned)
	}

	browse, err := notes.ListNotesForUser(ctx, domain.NoteListInput{UserID: "ada", Sort: domain.NoteListSortBrowse})
	if err != nil || len(browse.Notes) != 2 || browse.Notes[0].NoteID != oldPinned.NoteID {
		t.Fatalf("browse order = %+v err=%v", browse.Notes, err)
	}
	recent, err := notes.ListNotesForUser(ctx, domain.NoteListInput{UserID: "ada", Sort: domain.NoteListSortRecent})
	if err != nil || len(recent.Notes) != 2 || recent.Notes[0].NoteID != newer.NoteID {
		t.Fatalf("recent list order = %+v err=%v", recent.Notes, err)
	}

	seeds := []domain.NoteRecord{
		{NoteID: "note_exact", UserID: "ada", Title: "matchword", BodyMarkdown: "old", Pinned: false, EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: base.Add(-3 * time.Hour)},
		{NoteID: "note_title", UserID: "ada", Title: "matchword details", BodyMarkdown: "middle", Pinned: false, EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: base.Add(-2 * time.Hour)},
		{NoteID: "note_body", UserID: "ada", Title: "Other", BodyMarkdown: "newest matchword", Pinned: true, EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: base.Add(2 * time.Hour)},
	}
	for _, seed := range seeds {
		if _, err := notes.CreateNote(ctx, seed); err != nil {
			t.Fatalf("CreateNote %s: %v", seed.NoteID, err)
		}
	}
	relevance, err := notes.SearchNotesForUser(ctx, domain.NoteSearchInput{
		UserID: "ada", Query: "matchword", Sort: domain.NoteSearchSortRelevance, Limit: 10,
	})
	if err != nil || len(relevance.Notes) != 5 || relevance.Notes[0].NoteID != "note_exact" ||
		relevance.Notes[1].NoteID != "note_title" || relevance.Notes[2].NoteID != "note_body" {
		t.Fatalf("relevance order = %+v err=%v", relevance.Notes, err)
	}
	filteredRecent, err := notes.SearchNotesForUser(ctx, domain.NoteSearchInput{
		UserID: "ada", Query: "matchword", Sort: domain.NoteSearchSortRecent, Limit: 2,
		SnapshotAt: base.Add(3 * time.Hour),
		After: &domain.NoteSearchPageAnchor{
			Rank: 0, ContentUpdatedAt: base.Add(2 * time.Hour), NoteID: "note_body",
		},
	})
	if err != nil || len(filteredRecent.Notes) != 2 || filteredRecent.Notes[0].NoteID != newer.NoteID || filteredRecent.Notes[1].NoteID != oldPinned.NoteID {
		t.Fatalf("filtered recent keyset page = %+v err=%v", filteredRecent.Notes, err)
	}
	allRecent, err := notes.SearchNotesForUser(ctx, domain.NoteSearchInput{
		UserID: "ada", Sort: domain.NoteSearchSortRecent, Limit: 1,
	})
	if err != nil || len(allRecent.Notes) != 1 || allRecent.Notes[0].NoteID != "note_body" {
		t.Fatalf("blank-query recent = %+v err=%v", allRecent.Notes, err)
	}
}

func TestMemoryCreateNoteIdempotencySurvivesLostResponseAndTombstonesDeletion(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	now := domain.Now()
	record := domain.NoteRecord{
		NoteID: "note_create_first", UserID: "ada", Title: "Low entropy secret",
		BodyMarkdown: "private captured text", Pinned: true,
		EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: now,
	}
	digest := domain.ComputeNoteCreateRequestDigest(
		record.UserID, record.OrgID, record.Title, record.BodyMarkdown, record.Pinned, record.EditorMode,
	)
	input := domain.CreateNoteIdempotentInput{
		Record: record, IdempotencyKey: "stable-random-draft-key", RequestDigest: digest,
	}
	created, first, err := notes.CreateNoteForUserIdempotent(ctx, input)
	if err != nil || !first || created.NoteID != record.NoteID {
		t.Fatalf("first idempotent create = %+v first=%t err=%v", created, first, err)
	}
	// A retry is free to carry a newly generated server candidate id; the
	// owner/key/request binding must still replay the committed Note.
	retry := input
	retry.Record.NoteID = "note_create_retry_candidate"
	replayed, first, err := notes.CreateNoteForUserIdempotent(ctx, retry)
	if err != nil || first || replayed.NoteID != created.NoteID || len(notes.notes) != 1 {
		t.Fatalf("lost-response replay = %+v first=%t notes=%d err=%v", replayed, first, len(notes.notes), err)
	}
	if foundReplay, found, err := notes.FindNoteCreateReplayForUser(ctx, record.UserID, input.IdempotencyKey, input.RequestDigest); err != nil || !found || foundReplay.NoteID != created.NoteID {
		t.Fatalf("read-only create replay lookup = %+v found=%t err=%v", foundReplay, found, err)
	}
	if _, found, err := notes.FindNoteCreateReplayForUser(ctx, "mallory", input.IdempotencyKey, input.RequestDigest); err != nil || found {
		t.Fatalf("foreign create replay lookup found=%t err=%v", found, err)
	}
	conflict := retry
	conflict.RequestDigest = domain.ComputeNoteCreateRequestDigest(
		record.UserID, record.OrgID, "Different", record.BodyMarkdown, record.Pinned, record.EditorMode,
	)
	if _, _, err := notes.CreateNoteForUserIdempotent(ctx, conflict); !errors.Is(err, ErrNoteCreateIdempotencyConflict) {
		t.Fatalf("different create request error = %v, want typed idempotency conflict", err)
	}
	if _, found, err := notes.FindNoteCreateReplayForUser(ctx, record.UserID, input.IdempotencyKey, conflict.RequestDigest); !found || !errors.Is(err, ErrNoteCreateIdempotencyConflict) {
		t.Fatalf("different create replay lookup found=%t err=%v", found, err)
	}
	if err := notes.DeleteNoteForUser(ctx, created.NoteID, "ada"); err != nil {
		t.Fatalf("DeleteNoteForUser: %v", err)
	}
	if _, _, err := notes.CreateNoteForUserIdempotent(ctx, retry); !errors.Is(err, ErrNoteCreateReplayDeleted) {
		t.Fatalf("deleted Note replay error = %v, want terminal replay-deleted error", err)
	}
	if _, found, err := notes.FindNoteCreateReplayForUser(ctx, record.UserID, input.IdempotencyKey, input.RequestDigest); !found || !errors.Is(err, ErrNoteCreateReplayDeleted) {
		t.Fatalf("deleted create lookup found=%t err=%v, want terminal replay-deleted error", found, err)
	}
	receipt := notes.noteCreateReceipts[record.UserID+"\x00"+input.IdempotencyKey]
	if receipt.NoteID != "" || receipt.RequestDigest != "" || len(notes.notes) != 0 {
		t.Fatalf("deletion tombstone retained sensitive binding or resurrected Note: %+v notes=%d", receipt, len(notes.notes))
	}
}

func TestMemoryDirectNoteAppendIsIdempotentAndConditionallyUndoable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	now := domain.Now()
	note, err := notes.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_direct", UserID: "ada", Title: "Capture target",
		BodyMarkdown: "baseline", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: now,
	})
	if err != nil {
		t.Fatalf("CreateNote: %v", err)
	}
	input := domain.DirectNoteAppendInput{
		OperationID: "ndop_first", UserID: "ada", NoteID: note.NoteID,
		ExpectedRevision: note.Revision, BodyMarkdown: "exact capture",
		IdempotencyKey: "capture-key", Now: now.Add(time.Second),
	}
	input.RequestDigest = domain.ComputeNoteDirectAppendRequestDigest(
		input.UserID, input.NoteID, input.ExpectedRevision, input.BodyMarkdown,
	)
	receipt, created, err := notes.DirectAppendNoteForUser(ctx, input)
	if err != nil || !created || receipt.OperationID != input.OperationID || receipt.AppendedBytes != len("\n\nexact capture") {
		t.Fatalf("DirectAppendNoteForUser = %+v created=%t err=%v", receipt, created, err)
	}
	current, err := notes.GetNoteForUser(ctx, note.NoteID, "ada")
	if err != nil || current.BodyMarkdown != "baseline\n\nexact capture" || current.Revision != note.Revision+1 {
		t.Fatalf("appended Note = %+v err=%v", current, err)
	}
	replayInput := input
	replayInput.OperationID = "ndop_replay_should_not_win"
	replayed, created, err := notes.DirectAppendNoteForUser(ctx, replayInput)
	if err != nil || created || replayed.OperationID != receipt.OperationID {
		t.Fatalf("direct append replay = %+v created=%t err=%v", replayed, created, err)
	}
	if foundReplay, found, err := notes.FindNoteDirectAppendReplayForUser(ctx, input.UserID, input.IdempotencyKey, input.RequestDigest); err != nil || !found || foundReplay.OperationID != receipt.OperationID {
		t.Fatalf("read-only direct replay lookup = %+v found=%t err=%v", foundReplay, found, err)
	}
	if _, found, err := notes.FindNoteDirectAppendReplayForUser(ctx, "mallory", input.IdempotencyKey, input.RequestDigest); err != nil || found {
		t.Fatalf("foreign direct replay lookup found=%t err=%v", found, err)
	}
	conflicting := replayInput
	conflicting.BodyMarkdown = "different capture"
	conflicting.RequestDigest = domain.ComputeNoteDirectAppendRequestDigest(
		conflicting.UserID, conflicting.NoteID, conflicting.ExpectedRevision, conflicting.BodyMarkdown,
	)
	if _, _, err := notes.DirectAppendNoteForUser(ctx, conflicting); !errors.Is(err, ErrNoteAppendIdempotencyConflict) {
		t.Fatalf("same key different request err = %v", err)
	}
	if _, found, err := notes.FindNoteDirectAppendReplayForUser(ctx, input.UserID, input.IdempotencyKey, conflicting.RequestDigest); !found || !errors.Is(err, ErrNoteAppendIdempotencyConflict) {
		t.Fatalf("different direct replay lookup found=%t err=%v", found, err)
	}
	if stored := notes.noteDirectAppendOps[receipt.OperationID]; stored.NoteTitle != "" {
		t.Fatalf("stored direct receipt retained title: %+v", stored)
	}

	undone, err := notes.UndoDirectNoteAppendForUser(ctx, domain.UndoDirectNoteAppendInput{
		OperationID: receipt.OperationID, UserID: "ada", Now: now.Add(2 * time.Second),
	})
	if err != nil || undone.UndoRevision != receipt.AfterRevision+1 {
		t.Fatalf("UndoDirectNoteAppendForUser = %+v err=%v", undone, err)
	}
	current, _ = notes.GetNoteForUser(ctx, note.NoteID, "ada")
	if current.BodyMarkdown != "baseline" {
		t.Fatalf("undone body = %q", current.BodyMarkdown)
	}
	if replayedUndo, err := notes.UndoDirectNoteAppendForUser(ctx, domain.UndoDirectNoteAppendInput{
		OperationID: receipt.OperationID, UserID: "ada", Now: now.Add(3 * time.Second),
	}); err != nil || replayedUndo.UndoRevision != undone.UndoRevision {
		t.Fatalf("undo replay = %+v err=%v", replayedUndo, err)
	}

	second := domain.DirectNoteAppendInput{
		OperationID: "ndop_second", UserID: "ada", NoteID: note.NoteID,
		ExpectedRevision: current.Revision, BodyMarkdown: "second capture",
		IdempotencyKey: "capture-key-2", Now: now.Add(4 * time.Second),
	}
	second.RequestDigest = domain.ComputeNoteDirectAppendRequestDigest(second.UserID, second.NoteID, second.ExpectedRevision, second.BodyMarkdown)
	secondReceipt, _, err := notes.DirectAppendNoteForUser(ctx, second)
	if err != nil {
		t.Fatalf("second direct append: %v", err)
	}
	pinned := true
	if _, err := notes.UpdateNoteForUser(ctx, note.NoteID, "ada", domain.NoteUpdateInput{
		ExpectedRevision: secondReceipt.AfterRevision, Pinned: &pinned,
	}); err != nil {
		t.Fatalf("later metadata mutation: %v", err)
	}
	if _, err := notes.UndoDirectNoteAppendForUser(ctx, domain.UndoDirectNoteAppendInput{
		OperationID: secondReceipt.OperationID, UserID: "ada", Now: now.Add(5 * time.Second),
	}); !errors.Is(err, ErrNoteUndoConflict) {
		t.Fatalf("unsafe direct undo err = %v", err)
	}
	if err := notes.DeleteNoteForUser(ctx, note.NoteID, "ada"); err != nil {
		t.Fatalf("DeleteNoteForUser: %v", err)
	}
	if len(notes.noteDirectAppendOps) != 0 {
		t.Fatalf("hard delete retained direct receipts: %+v", notes.noteDirectAppendOps)
	}
	if _, found, err := notes.FindNoteDirectAppendReplayForUser(ctx, input.UserID, input.IdempotencyKey, input.RequestDigest); err != nil || found {
		t.Fatalf("deleted direct replay lookup found=%t err=%v, want no surviving effect", found, err)
	}
}

func TestMemoryDirectNoteAppendCombinedSizeFailureIsDefinitivelyUncommitted(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	now := domain.Now()
	note, err := notes.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_direct_full", UserID: "ada", Title: "Full",
		BodyMarkdown: strings.Repeat("x", maxStoredNoteBodyBytes),
		EditorMode:   domain.NoteEditorModeMarkdown, CreatedAt: now,
	})
	if err != nil {
		t.Fatalf("CreateNote: %v", err)
	}
	input := domain.DirectNoteAppendInput{
		OperationID: "ndop_full", UserID: "ada", NoteID: note.NoteID,
		ExpectedRevision: note.Revision, BodyMarkdown: "capture",
		IdempotencyKey: "full-capture-key", Now: now.Add(time.Second),
	}
	input.RequestDigest = domain.ComputeNoteDirectAppendRequestDigest(
		input.UserID, input.NoteID, input.ExpectedRevision, input.BodyMarkdown,
	)
	if _, created, err := notes.DirectAppendNoteForUser(ctx, input); created || !errors.Is(err, ErrNoteAppendNotCommitted) {
		t.Fatalf("oversize direct append created=%t err=%v, want definitively uncommitted", created, err)
	}
	current, err := notes.GetNoteForUser(ctx, note.NoteID, note.UserID)
	if err != nil || current.Revision != note.Revision || current.BodyMarkdown != note.BodyMarkdown {
		t.Fatalf("oversize direct append mutated Note: %+v err=%v", current, err)
	}
	if len(notes.noteDirectAppendOps) != 0 {
		t.Fatalf("oversize direct append stored receipt: %+v", notes.noteDirectAppendOps)
	}
}

func TestMemoryThreadHardDeleteCascadesRunScopedNoteAuthority(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	thread, err := notes.CreateThread(ctx, domain.CreateThreadInput{UserID: "ada", Title: "private run"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := notes.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "ada", Goal: "use note"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := domain.Now()
	note, err := notes.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_memory_cascade", UserID: "ada", Title: "Private title",
		BodyMarkdown: "baseline", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: now,
	})
	if err != nil {
		t.Fatalf("CreateNote: %v", err)
	}
	readToken := "memory-cascade-read-token"
	if err := notes.CreateNoteReadGrant(ctx, domain.NoteReadGrantRecord{
		TokenHash: domain.NoteBodySHA256(readToken), RunID: run.RunID, UserID: "ada",
		NoteID: note.NoteID, Revision: note.Revision, CreatedAt: now, ExpiresAt: now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("CreateNoteReadGrant: %v", err)
	}
	proposal, err := notes.CreateNoteAppendProposal(ctx, domain.CreateNoteAppendProposalInput{
		ProposalID: "nprop_memory_cascade", RunID: run.RunID, UserID: "ada",
		NoteID: note.NoteID, ExpectedRevision: note.Revision, BodyMarkdown: "append",
		ReadTokenHash: domain.NoteBodySHA256(readToken), IdempotencyKey: "memory-cascade-tool",
		RequestDigest: domain.ComputeNoteContentDigest(note.NoteID+":1", "append"),
		Now:           now, ExpiresAt: now.Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("CreateNoteAppendProposal: %v", err)
	}
	receipt, err := notes.CommitNoteAppendProposalForUser(ctx, domain.CommitNoteAppendProposalInput{
		ProposalID: proposal.ProposalID, OperationID: "nop_memory_cascade", UserID: "ada", Now: now,
	})
	if err != nil {
		t.Fatalf("CommitNoteAppendProposalForUser: %v", err)
	}
	if err := notes.ConsumeNoteSearchBudget(ctx, run.RunID, "ada"); err != nil {
		t.Fatalf("ConsumeNoteSearchBudget: %v", err)
	}
	if stored := notes.noteAppendProposals[proposal.ProposalID]; stored.NoteTitle != "" || stored.BodyMarkdown != "" {
		t.Fatalf("stored proposal retained sensitive text: %+v", stored)
	}
	if stored := notes.noteAppendOperations[receipt.OperationID]; stored.NoteTitle != "" {
		t.Fatalf("stored operation retained Note title: %+v", stored)
	}

	if _, err := notes.HardDeleteThreadForUser(ctx, thread.ThreadID, "ada"); err != nil {
		t.Fatalf("HardDeleteThreadForUser: %v", err)
	}
	if len(notes.noteReadGrants) != 0 || len(notes.noteAppendProposals) != 0 ||
		len(notes.noteAppendOperations) != 0 || len(notes.noteRunUsage) != 0 {
		t.Fatalf("run-scoped Notes state survived hard delete: grants=%d proposals=%d operations=%d usage=%d",
			len(notes.noteReadGrants), len(notes.noteAppendProposals), len(notes.noteAppendOperations), len(notes.noteRunUsage))
	}
	if _, err := notes.GetNoteForUser(ctx, note.NoteID, "ada"); err != nil {
		t.Fatalf("independent Note was deleted with conversation: %v", err)
	}
}

func TestMemoryNoteRetrievalBudgetMatchesDurableLimits(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	notes := NewMemoryStore()
	for index := 0; index < maxNoteSearchCalls; index++ {
		if err := notes.ConsumeNoteSearchBudget(ctx, "run_search_budget", "ada"); err != nil {
			t.Fatalf("search budget call %d: %v", index, err)
		}
	}
	if err := notes.ConsumeNoteSearchBudget(ctx, "run_search_budget", "ada"); !errors.Is(err, ErrNoteRetrievalBudget) {
		t.Fatalf("search over budget err = %v", err)
	}
	if err := notes.ConsumeNoteSearchBudget(ctx, "run_search_budget", "mallory"); !errors.Is(err, ErrNoteRetrievalBudget) {
		t.Fatalf("search owner mismatch err = %v", err)
	}

	for returned := 0; returned < maxNoteReadBytes; returned += maxNoteReadCallBytes {
		if err := notes.ConsumeNoteReadBudget(ctx, "run_read_budget", "ada", maxNoteReadCallBytes); err != nil {
			t.Fatalf("read budget at %d bytes: %v", returned, err)
		}
	}
	if err := notes.ConsumeNoteReadBudget(ctx, "run_read_budget", "ada", 1); !errors.Is(err, ErrNoteRetrievalBudget) {
		t.Fatalf("cumulative read over budget err = %v", err)
	}
	if err := notes.ConsumeNoteReadBudget(ctx, "run_new_budget", "ada", maxNoteReadCallBytes+1); !errors.Is(err, ErrNoteRetrievalBudget) {
		t.Fatalf("per-call read over budget err = %v", err)
	}
}

func TestMemoryExpiredReadGrantSweepIsBounded(t *testing.T) {
	t.Parallel()
	now := domain.Now()
	notes := NewMemoryStore()
	notes.noteReadGrants["expired-one"] = domain.NoteReadGrantRecord{TokenHash: "expired-one", ExpiresAt: now.Add(-time.Minute)}
	notes.noteReadGrants["expired-two"] = domain.NoteReadGrantRecord{TokenHash: "expired-two", ExpiresAt: now}
	notes.noteReadGrants["active"] = domain.NoteReadGrantRecord{TokenHash: "active", ExpiresAt: now.Add(time.Minute)}
	if expired, err := notes.ExpireNoteReadGrants(context.Background(), now, 1); err != nil || expired != 1 {
		t.Fatalf("first grant sweep = %d err=%v", expired, err)
	}
	if got := len(notes.noteReadGrants); got != 2 {
		t.Fatalf("bounded grant sweep left %d rows, want 2", got)
	}
	if expired, err := notes.ExpireNoteReadGrants(context.Background(), now, 10); err != nil || expired != 1 {
		t.Fatalf("second grant sweep = %d err=%v", expired, err)
	}
	if _, active := notes.noteReadGrants["active"]; !active || len(notes.noteReadGrants) != 1 {
		t.Fatalf("grant sweep removed active authority: %+v", notes.noteReadGrants)
	}
}

func TestMemoryNoteDerivedRunStaysInSourceButNotCrossConversationHistory(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	history := NewMemoryStore()
	userID := "memory-note-history-owner"
	createSucceededRun := func(title, goal, response string, metadata domain.JSONMap) domain.RunRecord {
		thread, err := history.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: title})
		if err != nil {
			t.Fatalf("CreateThread(%q): %v", title, err)
		}
		run, err := history.CreateRun(ctx, domain.CreateRunInput{
			ThreadID: thread.ThreadID, UserID: userID, Goal: goal, Metadata: metadata,
		})
		if err != nil {
			t.Fatalf("CreateRun(%q): %v", goal, err)
		}
		completed, err := history.CompleteRun(ctx, domain.CompleteRunInput{RunID: run.RunID, ResponseText: response})
		if err != nil {
			t.Fatalf("CompleteRun(%q): %v", goal, err)
		}
		return completed
	}

	noteSentinel := "note-derived-history-sentinel-7b3e"
	noteRun := createSucceededRun("Private Note result", "Use my Note for "+noteSentinel, "Answer copied from "+noteSentinel,
		domain.JSONMap{"selection_context": domain.JSONMap{
			domain.NoteAccessSelectionKey: domain.JSONMap{"mode": "search", "notes": []any{}},
		}})
	ordinarySentinel := "ordinary-history-sentinel-2c91"
	ordinaryRun := createSucceededRun("Ordinary result", "Analyze "+ordinarySentinel, "Independent answer "+ordinarySentinel, nil)

	hits, err := history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: noteSentinel})
	if err != nil {
		t.Fatalf("SearchRunHistoryForUser(Note sentinel): %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("Note-derived response escaped through episodic history: %+v", hits)
	}
	hits, err = history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{Query: ordinarySentinel})
	if err != nil || len(hits) != 1 || hits[0].RunID != ordinaryRun.RunID {
		t.Fatalf("ordinary episodic history = %+v err=%v", hits, err)
	}
	hits, err = history.SearchRunHistoryForUser(ctx, userID, domain.RunHistorySearchOptions{})
	if err != nil || len(hits) != 1 || hits[0].RunID != ordinaryRun.RunID {
		t.Fatalf("recency episodic history = %+v err=%v, want only ordinary run", hits, err)
	}

	stored, err := history.GetRunForUser(ctx, noteRun.RunID, userID)
	if err != nil || stored.ResponseText != "Answer copied from "+noteSentinel {
		t.Fatalf("source Note run response = %q err=%v, want preserved", stored.ResponseText, err)
	}
	unscoped, err := history.ListRunsForUser(ctx, userID, "", "", 20, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser unscoped: %v", err)
	}
	seenNote, seenOrdinary := false, false
	for _, run := range unscoped {
		if run.RunID == noteRun.RunID {
			seenNote = true
			if run.ResponseText != "" {
				t.Fatalf("unscoped run list copied Note-derived response %q", run.ResponseText)
			}
		}
		if run.RunID == ordinaryRun.RunID {
			seenOrdinary = true
			if run.ResponseText == "" {
				t.Fatal("unscoped run list redacted an ordinary response")
			}
		}
	}
	if !seenNote || !seenOrdinary {
		t.Fatalf("unscoped run list omitted lifecycle records: note=%t ordinary=%t", seenNote, seenOrdinary)
	}
	allRuns, err := history.ListRuns(ctx, "", "", 20, 0)
	if err != nil {
		t.Fatalf("ListRuns unscoped: %v", err)
	}
	for _, run := range allRuns {
		if run.RunID == noteRun.RunID && run.ResponseText != "" {
			t.Fatalf("unscoped internal run list copied Note-derived response %q", run.ResponseText)
		}
	}
	scoped, err := history.ListRunsForUser(ctx, userID, noteRun.ThreadID, "", 20, 0)
	if err != nil || len(scoped) != 1 || scoped[0].ResponseText != stored.ResponseText {
		t.Fatalf("source-thread run list = %+v err=%v, want preserved response", scoped, err)
	}
}
