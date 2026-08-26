package runcontrol

import "github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"

// storedNoteAccessMatches keeps an idempotency key from replaying a run under
// a different Notes capability. Generic selection-context changes retain their
// historical behavior; the reserved note_access sub-object is immutable.
func storedNoteAccessMatches(run domain.RunRecord, requestedSelection domain.JSONMap) bool {
	requested, requestedPresent, requestedValid := domain.ParseNoteAccessScope(requestedSelection)
	if !requestedValid {
		return false
	}
	stored, storedPresent := domain.NoteAccessScopeFromRun(run)
	if requestedPresent != storedPresent {
		return false
	}
	if !requestedPresent {
		return true
	}
	return domain.NoteAccessRequestMatchesStoredScope(requested, stored)
}
