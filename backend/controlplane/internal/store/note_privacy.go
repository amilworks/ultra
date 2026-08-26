package store

import "github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"

// serverAuthoredNotePrivacyMetadata removes any caller-supplied lineage value
// and restores it only when the run itself has Notes access or its source
// thread already carries privacy lineage. The shallow copy is sufficient: this
// helper changes only the reserved top-level key and must not mutate the
// caller's metadata map.
func serverAuthoredNotePrivacyMetadata(metadata domain.JSONMap, inherited bool) domain.JSONMap {
	canonical := make(domain.JSONMap, len(metadata)+1)
	for key, value := range metadata {
		if key != domain.NotePrivacyLineageMetadataKey {
			canonical[key] = value
		}
	}
	if inherited || domain.RunHasNoteAccessSelection(domain.RunRecord{Metadata: canonical}) {
		canonical[domain.NotePrivacyLineageMetadataKey] = true
	}
	return canonical
}

func notePrivacyThreadIDsFromMemoryRuns(runs map[string]domain.RunRecord) map[string]struct{} {
	threadIDs := make(map[string]struct{})
	for _, run := range runs {
		if domain.RunHasNotePrivacyLineage(run) {
			threadIDs[run.ThreadID] = struct{}{}
		}
	}
	return threadIDs
}
