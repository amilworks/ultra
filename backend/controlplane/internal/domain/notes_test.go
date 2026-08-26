package domain

import "testing"

func TestParseNoteAccessScopeFailsClosedAndCapsReferences(t *testing.T) {
	t.Parallel()
	if _, present, valid := ParseNoteAccessScope(nil); present || !valid {
		t.Fatalf("absent scope present=%t valid=%t, want false/true", present, valid)
	}
	if _, present, valid := ParseNoteAccessScope(JSONMap{
		NoteAccessSelectionKey: JSONMap{"mode": "selected", "notes": []any{}},
	}); !present || valid {
		t.Fatalf("empty selected scope present=%t valid=%t, want true/false", present, valid)
	}
	search, present, valid := ParseNoteAccessScope(JSONMap{
		NoteAccessSelectionKey: JSONMap{"mode": "search", "allow_append_proposal": true},
	})
	if !present || !valid || search.Mode != NoteAccessModeSearch || len(search.Notes) != 0 || !search.AllowAppendProposal {
		t.Fatalf("empty search scope = %+v present=%t valid=%t", search, present, valid)
	}
	for name, allow := range map[string]any{"null append flag": nil, "string append flag": "true", "numeric append flag": float64(1)} {
		if _, _, valid := ParseNoteAccessScope(JSONMap{
			NoteAccessSelectionKey: JSONMap{"mode": "search", "allow_append_proposal": allow},
		}); valid {
			t.Fatalf("%s was accepted", name)
		}
	}

	tooMany := make([]any, MaxRunNoteReferences+1)
	for index := range tooMany {
		tooMany[index] = JSONMap{"note_id": NewID("note")}
	}
	if _, _, valid := ParseNoteAccessScope(JSONMap{
		NoteAccessSelectionKey: JSONMap{"mode": "search", "notes": tooMany},
	}); valid {
		t.Fatal("scope exceeding the selected-reference cap was accepted")
	}
	for name, notes := range map[string][]any{
		"duplicate":           {JSONMap{"note_id": "note_one"}, JSONMap{"note_id": "note_one"}},
		"zero revision":       {JSONMap{"note_id": "note_one", "revision": float64(0)}},
		"fractional revision": {JSONMap{"note_id": "note_one", "revision": 1.5}},
	} {
		notes := notes
		t.Run(name, func(t *testing.T) {
			if _, _, valid := ParseNoteAccessScope(JSONMap{
				NoteAccessSelectionKey: JSONMap{"mode": "search", "notes": notes},
			}); valid {
				t.Fatalf("malformed notes %v were accepted", notes)
			}
		})
	}
}

func TestCanonicalNoteAccessSelectionPreservesUnrelatedContext(t *testing.T) {
	t.Parallel()
	selection := CanonicalNoteAccessSelection(JSONMap{
		"workflow":             "analysis",
		NoteAccessSelectionKey: JSONMap{"mode": "search", "notes": []any{}},
	}, NoteAccessScope{
		Mode:                NoteAccessModeSelected,
		Notes:               []NoteReference{{NoteID: "note_one", Revision: 7}},
		AllowAppendProposal: true,
	})
	if selection["workflow"] != "analysis" {
		t.Fatalf("unrelated context was discarded: %+v", selection)
	}
	scope, present, valid := ParseNoteAccessScope(selection)
	if !present || !valid || scope.Mode != NoteAccessModeSelected ||
		len(scope.Notes) != 1 || scope.Notes[0] != (NoteReference{NoteID: "note_one", Revision: 7}) ||
		!scope.AllowAppendProposal {
		t.Fatalf("canonical scope = %+v present=%t valid=%t", scope, present, valid)
	}
}

func TestNoteAccessRequestMatchesStoredScopeTreatsOnlyOmittedRevisionAsReplayWildcard(t *testing.T) {
	t.Parallel()
	stored := NoteAccessScope{
		Mode:                NoteAccessModeSelected,
		Notes:               []NoteReference{{NoteID: "note_one", Revision: 7}},
		AllowAppendProposal: true,
	}
	requested := stored
	requested.Notes = []NoteReference{{NoteID: "note_one"}}
	if !NoteAccessRequestMatchesStoredScope(requested, stored) {
		t.Fatal("raw scope with omitted revision did not match stored canonical scope")
	}
	for name, mutate := range map[string]func(*NoteAccessScope){
		"mode":              func(scope *NoteAccessScope) { scope.Mode = NoteAccessModeSearch },
		"note id":           func(scope *NoteAccessScope) { scope.Notes[0].NoteID = "note_two" },
		"explicit revision": func(scope *NoteAccessScope) { scope.Notes[0].Revision = 8 },
		"append flag":       func(scope *NoteAccessScope) { scope.AllowAppendProposal = false },
	} {
		candidate := requested
		candidate.Notes = append([]NoteReference(nil), requested.Notes...)
		mutate(&candidate)
		if NoteAccessRequestMatchesStoredScope(candidate, stored) {
			t.Fatalf("different %s matched stored scope", name)
		}
	}
}

func TestRunHasNoteAccessSelectionUsesReservedMarkerFailClosed(t *testing.T) {
	t.Parallel()
	for name, run := range map[string]RunRecord{
		"valid": {
			Metadata: JSONMap{"selection_context": JSONMap{
				NoteAccessSelectionKey: JSONMap{"mode": "search", "notes": []any{}},
			}},
		},
		"malformed reserved value": {
			Metadata: JSONMap{"selection_context": JSONMap{NoteAccessSelectionKey: "corrupt"}},
		},
		"null reserved value": {
			Metadata: JSONMap{"selection_context": JSONMap{NoteAccessSelectionKey: nil}},
		},
	} {
		run := run
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if !RunHasNoteAccessSelection(run) {
				t.Fatalf("run with reserved Note marker was treated as content-safe: %+v", run.Metadata)
			}
		})
	}
	if RunHasNoteAccessSelection(RunRecord{Metadata: JSONMap{
		"selection_context": JSONMap{"workflow": "ordinary"},
	}}) {
		t.Fatal("ordinary selection context was marked Note-derived")
	}
}

func TestRunHasNotePrivacyLineageSupportsServerMarkerAndLegacyRuns(t *testing.T) {
	t.Parallel()

	for name, run := range map[string]RunRecord{
		"server marker": {
			Metadata: JSONMap{NotePrivacyLineageMetadataKey: true},
		},
		"malformed server marker fails closed": {
			Metadata: JSONMap{NotePrivacyLineageMetadataKey: "corrupt"},
		},
		"legacy Note access": {
			Metadata: JSONMap{"selection_context": JSONMap{
				NoteAccessSelectionKey: JSONMap{"mode": "search", "notes": []any{}},
			}},
		},
	} {
		run := run
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if !RunHasNotePrivacyLineage(run) {
				t.Fatalf("privacy-bearing run was treated as safe: %+v", run.Metadata)
			}
		})
	}

	if RunHasNotePrivacyLineage(RunRecord{Metadata: JSONMap{"label": "ordinary"}}) {
		t.Fatal("ordinary run was assigned Note privacy lineage")
	}
}
