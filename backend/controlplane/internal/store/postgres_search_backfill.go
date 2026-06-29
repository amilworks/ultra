package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

// BackfillPostgresResourceSearchIndexes rebuilds derived search documents and
// typed facts for resources that predate the derived search tables.
func BackfillPostgresResourceSearchIndexes(ctx context.Context, pool *pgxpool.Pool) error {
	if pool == nil {
		return errors.New("postgres pool is required")
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin resource search backfill: %w", err)
	}
	defer tx.Rollback(ctx)

	rows, err := tx.Query(ctx, `
SELECT
  r.resource_id,
  r.owner_user_id,
  r.owner_org_id,
  r.owner_role,
  r.original_name,
  r.content_type,
  r.size_bytes,
  r.sha256,
  r.storage_uri,
  r.storage_path,
  r.source_type,
  r.resource_kind,
  r.source_uri,
  r.project_id,
  r.status,
  r.created_at,
  r.updated_at,
  r.deleted_at,
  r.retention_expires_at,
  r.metadata
FROM control_resources r
WHERE NOT EXISTS (
    SELECT 1 FROM control_resource_search_documents sd WHERE sd.resource_id = r.resource_id
  )
  OR NOT EXISTS (
    SELECT 1 FROM control_resource_search_facts sf WHERE sf.resource_id = r.resource_id
  )
ORDER BY r.created_at ASC, r.resource_id ASC`)
	if err != nil {
		return fmt.Errorf("select resources for search backfill: %w", err)
	}
	resources := make([]domain.ResourceRecord, 0)
	for rows.Next() {
		row, err := scanControlResourceRow(rows)
		if err != nil {
			rows.Close()
			return fmt.Errorf("scan resource for search backfill: %w", err)
		}
		resources = append(resources, resourceFromRow(row))
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("iterate resources for search backfill: %w", err)
	}
	rows.Close()

	for _, resource := range resources {
		if err := upsertResourceSearchDocumentTx(ctx, tx, resource); err != nil {
			return fmt.Errorf("backfill resource search indexes for %s: %w", resource.ResourceID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit resource search backfill: %w", err)
	}
	return nil
}
