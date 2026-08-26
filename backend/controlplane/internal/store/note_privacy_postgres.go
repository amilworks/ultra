package store

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// notePrivacyThreadIDs returns the privacy classification for the complete
// source thread, not just the paginated run row being rendered. That dynamic
// check protects unmarked descendants written by an older replica during a
// rolling upgrade without requiring a blocking metadata backfill.
func (s *PostgresStore) notePrivacyThreadIDs(ctx context.Context, runs []domain.RunRecord) (map[string]struct{}, error) {
	threadIDs := make([]string, 0, len(runs))
	seen := make(map[string]struct{}, len(runs))
	for _, run := range runs {
		if _, exists := seen[run.ThreadID]; exists {
			continue
		}
		seen[run.ThreadID] = struct{}{}
		threadIDs = append(threadIDs, run.ThreadID)
	}
	private := make(map[string]struct{})
	if len(threadIDs) == 0 {
		return private, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT DISTINCT thread_id
FROM control_runs
WHERE thread_id = ANY($1::text[])
  AND (
    COALESCE(metadata, '{}'::jsonb) ? $2
    OR COALESCE(metadata->'selection_context', '{}'::jsonb) ? $3
  )`, threadIDs, domain.NotePrivacyLineageMetadataKey, domain.NoteAccessSelectionKey)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var threadID string
		if err := rows.Scan(&threadID); err != nil {
			return nil, err
		}
		private[threadID] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return private, nil
}
