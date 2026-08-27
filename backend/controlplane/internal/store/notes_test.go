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
