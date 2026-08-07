package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store/sqlc"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PostgresStore struct {
	pool    *pgxpool.Pool
	queries *sqlc.Queries
}

const resourceLifecycleAdvisorySeed int64 = 0x554c545241

func lockResourceLifecycleTx(ctx context.Context, tx pgx.Tx, resourceID string) error {
	_, err := tx.Exec(
		ctx,
		`SELECT pg_advisory_xact_lock(hashtextextended($1, $2))`,
		resourceID,
		resourceLifecycleAdvisorySeed,
	)
	return err
}

func upsertResourceSearchDocumentTx(ctx context.Context, tx pgx.Tx, resource domain.ResourceRecord) error {
	searchText := resourceSearchDocument(resource)
	updatedAt := resource.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	resourceID := strings.TrimSpace(resource.ResourceID)
	_, err := tx.Exec(ctx, `
INSERT INTO control_resource_search_documents (
  resource_id, owner_user_id, owner_org_id, project_id, status, search_text, search_vector, updated_at
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), $5, $6, to_tsvector('simple', $6), $7)
ON CONFLICT (resource_id) DO UPDATE SET
  owner_user_id = EXCLUDED.owner_user_id,
  owner_org_id = EXCLUDED.owner_org_id,
  project_id = EXCLUDED.project_id,
  status = EXCLUDED.status,
  search_text = EXCLUDED.search_text,
  search_vector = EXCLUDED.search_vector,
  updated_at = EXCLUDED.updated_at`,
		resourceID,
		strings.TrimSpace(resource.OwnerUserID),
		strings.TrimSpace(resource.OwnerOrgID),
		strings.TrimSpace(resource.ProjectID),
		strings.TrimSpace(resource.Status),
		searchText,
		timestamptz(updatedAt),
	)
	if err != nil {
		return err
	}
	batch := &pgx.Batch{}
	batch.Queue(`DELETE FROM control_resource_search_facts WHERE resource_id = $1`, resourceID)
	for _, fact := range resourceSearchFacts(resource) {
		var factNumber any
		if fact.Number != nil {
			factNumber = *fact.Number
		}
		batch.Queue(`
INSERT INTO control_resource_search_facts (
  resource_id, owner_user_id, owner_org_id, project_id, status, fact_key, fact_text, fact_number, fact_source, updated_at
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), $5, $6, $7, $8, $9, $10)`,
			resourceID,
			strings.TrimSpace(resource.OwnerUserID),
			strings.TrimSpace(resource.OwnerOrgID),
			strings.TrimSpace(resource.ProjectID),
			strings.TrimSpace(resource.Status),
			fact.Key,
			fact.Text,
			factNumber,
			fact.Source,
			timestamptz(updatedAt),
		)
	}
	results := tx.SendBatch(ctx, batch)
	defer func() {
		_ = results.Close()
	}()
	for index := 0; index < batch.Len(); index++ {
		if _, err := results.Exec(); err != nil {
			return err
		}
	}
	return nil
}

func NewPostgresStore(pool *pgxpool.Pool) *PostgresStore {
	return &PostgresStore{
		pool:    pool,
		queries: sqlc.New(pool),
	}
}

func postgresResourceDescriptorFilterSQL(placeholder string) string {
	param := strings.TrimSpace(placeholder)
	if param == "" {
		param = "$1"
	}
	return strings.ReplaceAll(`
  AND (
    cardinality(__PARAM__::text[]) = 0
    OR NOT EXISTS (
      SELECT 1
      FROM unnest(__PARAM__::text[]) AS descriptor_filters(filter)
      WHERE NOT (
        COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ? lower(descriptor_filters.filter)
        OR lower(COALESCE(r.metadata->>'label', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'scientific_descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'diagnosis', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'modality', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'organism', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata->>'species', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,label}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,scientific_descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'labels') = 'array' THEN r.metadata->'labels' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'descriptors') = 'array' THEN r.metadata->'descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'scientific_descriptors') = 'array' THEN r.metadata->'scientific_descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'diagnoses') = 'array' THEN r.metadata->'diagnoses' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,labels}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,labels}' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
          WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
        )
      )
    )
  )`, "__PARAM__", param)
}

func (s *PostgresStore) CreateUser(ctx context.Context, input domain.CreateUserInput) (domain.UserAccount, error) {
	now := domain.Now()
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = domain.NewID("user")
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "researcher"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_users (user_id, email, display_name, role, status, org_id, created_at, updated_at, metadata)
VALUES ($1, NULLIF($2, ''), NULLIF($3, ''), $4, $5, NULLIF($6, ''), $7, $8, $9)
RETURNING user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata`,
		userID,
		normalizeEmail(input.Email),
		strings.TrimSpace(input.DisplayName),
		role,
		status,
		strings.TrimSpace(input.OrgID),
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanUserAccount(row)
}

func (s *PostgresStore) CreateOrganization(ctx context.Context, input domain.CreateOrganizationInput) (domain.Organization, error) {
	now := domain.Now()
	orgID := normalizeOrgID(input.OrgID)
	if orgID == "" {
		orgID = domain.NewID("org")
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		name = orgID
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_organizations (org_id, name, status, created_at, updated_at, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
RETURNING org_id, name, status, created_at, updated_at, metadata`,
		orgID,
		name,
		status,
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanOrganization(row)
}

func (s *PostgresStore) GetOrganization(ctx context.Context, orgID string) (domain.Organization, bool, error) {
	orgID = normalizeOrgID(orgID)
	if orgID == "" {
		return domain.Organization{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT org_id, name, status, created_at, updated_at, metadata
FROM control_organizations
WHERE org_id = $1`, orgID)
	org, err := scanOrganization(row)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.Organization{}, false, nil
		}
		return domain.Organization{}, false, err
	}
	return org, true, nil
}

func (s *PostgresStore) ListOrganizations(ctx context.Context, limit int, query string) ([]domain.Organization, error) {
	query = strings.TrimSpace(query)
	rows, err := s.pool.Query(ctx, `
SELECT org_id, name, status, created_at, updated_at, metadata
FROM control_organizations
WHERE $1 = ''
   OR org_id ILIKE '%' || $1 || '%'
   OR name ILIKE '%' || $1 || '%'
   OR status ILIKE '%' || $1 || '%'
ORDER BY created_at DESC
LIMIT $2`, query, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	orgs := []domain.Organization{}
	for rows.Next() {
		org, err := scanOrganization(rows)
		if err != nil {
			return nil, err
		}
		orgs = append(orgs, org)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return orgs, nil
}

func (s *PostgresStore) ListUsers(ctx context.Context, limit int, query string) ([]domain.UserAccount, error) {
	query = strings.TrimSpace(query)
	rows, err := s.pool.Query(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE $1 = ''
   OR user_id ILIKE '%' || $1 || '%'
   OR COALESCE(email, '') ILIKE '%' || $1 || '%'
   OR COALESCE(display_name, '') ILIKE '%' || $1 || '%'
   OR role ILIKE '%' || $1 || '%'
   OR COALESCE(org_id, '') ILIKE '%' || $1 || '%'
ORDER BY created_at DESC
LIMIT $2`, query, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	users := []domain.UserAccount{}
	for rows.Next() {
		user, err := scanUserAccount(rows)
		if err != nil {
			return nil, err
		}
		users = append(users, user)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return users, nil
}

func (s *PostgresStore) GetUserByID(ctx context.Context, userID string) (domain.UserAccount, bool, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.UserAccount{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE user_id = $1`, userID)
	user, err := scanUserAccount(row)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.UserAccount{}, false, nil
		}
		return domain.UserAccount{}, false, err
	}
	return user, true, nil
}

func (s *PostgresStore) GetUserByEmail(ctx context.Context, email string) (domain.UserAccount, bool, error) {
	email = normalizeEmail(email)
	if email == "" {
		return domain.UserAccount{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE lower(email) = $1`, email)
	user, err := scanUserAccount(row)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.UserAccount{}, false, nil
		}
		return domain.UserAccount{}, false, err
	}
	return user, true, nil
}

func (s *PostgresStore) UpdateUserStatus(ctx context.Context, userID string, status string) (domain.UserAccount, error) {
	userID = strings.TrimSpace(userID)
	status = strings.TrimSpace(status)
	if status == "" {
		status = "disabled"
	}
	row := s.pool.QueryRow(ctx, `
UPDATE control_users
SET status = $2,
    updated_at = $3
WHERE user_id = $1
RETURNING user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata`,
		userID,
		status,
		domain.Now(),
	)
	return scanUserAccount(row)
}

func (s *PostgresStore) UpdateUserProfile(ctx context.Context, input domain.UpdateUserProfileInput) (domain.UserAccount, error) {
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		return domain.UserAccount{}, ErrNotFound
	}
	profileJSON, err := json.Marshal(input.Profile)
	if err != nil {
		return domain.UserAccount{}, err
	}
	displayName := strings.TrimSpace(input.Profile.DisplayName)
	row := s.pool.QueryRow(ctx, `
UPDATE control_users
SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{profile}', $2::jsonb, true),
    display_name = CASE WHEN $3 <> '' THEN $3 ELSE display_name END,
    updated_at = $4
WHERE user_id = $1
RETURNING user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata`,
		userID,
		profileJSON,
		displayName,
		domain.Now(),
	)
	return scanUserAccount(row)
}

func (s *PostgresStore) RecordUserTokenUsage(ctx context.Context, input domain.RecordUserTokenUsageInput) error {
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		return nil
	}
	now := input.OccurredAt
	if now.IsZero() {
		now = domain.Now()
	}
	day := tokenUsageBucketDay(input.Day, now)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Increment the daily bucket and read back its new running total so the
	// lifetime row can track the peak single-day spend correctly.
	var newDailyTotal int64
	if err := tx.QueryRow(ctx, `
INSERT INTO control_user_token_usage_daily (user_id, day, input_tokens, output_tokens, total_tokens, run_count, updated_at)
VALUES ($1, $2, $3, $4, $5, 1, $6)
ON CONFLICT (user_id, day) DO UPDATE SET
  input_tokens = control_user_token_usage_daily.input_tokens + EXCLUDED.input_tokens,
  output_tokens = control_user_token_usage_daily.output_tokens + EXCLUDED.output_tokens,
  total_tokens = control_user_token_usage_daily.total_tokens + EXCLUDED.total_tokens,
  run_count = control_user_token_usage_daily.run_count + 1,
  updated_at = EXCLUDED.updated_at
RETURNING total_tokens`,
		userID, day, input.InputTokens, input.OutputTokens, input.TotalTokens, now,
	).Scan(&newDailyTotal); err != nil {
		return err
	}

	if _, err := tx.Exec(ctx, `
INSERT INTO control_user_token_usage_lifetime (user_id, input_tokens, output_tokens, total_tokens, peak_daily_total, last_active_day, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (user_id) DO UPDATE SET
  input_tokens = control_user_token_usage_lifetime.input_tokens + EXCLUDED.input_tokens,
  output_tokens = control_user_token_usage_lifetime.output_tokens + EXCLUDED.output_tokens,
  total_tokens = control_user_token_usage_lifetime.total_tokens + EXCLUDED.total_tokens,
  peak_daily_total = GREATEST(control_user_token_usage_lifetime.peak_daily_total, EXCLUDED.peak_daily_total),
  last_active_day = EXCLUDED.last_active_day,
  updated_at = EXCLUDED.updated_at`,
		userID, input.InputTokens, input.OutputTokens, input.TotalTokens, newDailyTotal, day, now,
	); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (s *PostgresStore) RecordRunTokenUsage(ctx context.Context, input domain.RecordRunTokenUsageInput) (domain.RunTokenUsageRecord, bool, error) {
	runID := strings.TrimSpace(input.RunID)
	usageEventID := strings.TrimSpace(input.UsageEventID)
	userID := strings.TrimSpace(input.UserID)
	if runID == "" || usageEventID == "" || userID == "" {
		return domain.RunTokenUsageRecord{}, false, nil
	}
	totalTokens := input.TotalTokens
	if totalTokens <= 0 {
		totalTokens = input.InputTokens + input.OutputTokens
	}
	if totalTokens <= 0 && input.InputTokens <= 0 && input.OutputTokens <= 0 {
		return domain.RunTokenUsageRecord{}, false, nil
	}
	now := input.OccurredAt
	if now.IsZero() {
		now = domain.Now()
	}
	day := tokenUsageBucketDay(input.Day, now)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunTokenUsageRecord{}, false, err
	}
	defer tx.Rollback(ctx)

	record, inserted, err := insertRunTokenUsageTx(ctx, tx, domain.RecordRunTokenUsageInput{
		RunID:        runID,
		UsageEventID: usageEventID,
		UserID:       userID,
		Model:        strings.TrimSpace(input.Model),
		Day:          day,
		InputTokens:  input.InputTokens,
		OutputTokens: input.OutputTokens,
		TotalTokens:  totalTokens,
		OccurredAt:   now,
	})
	if err != nil {
		return domain.RunTokenUsageRecord{}, false, err
	}
	if !inserted {
		if err := tx.Commit(ctx); err != nil {
			return domain.RunTokenUsageRecord{}, false, err
		}
		return record, false, nil
	}
	if _, err := incrementUserTokenUsageTx(ctx, tx, userID, day, input.InputTokens, input.OutputTokens, totalTokens, 0, now); err != nil {
		return domain.RunTokenUsageRecord{}, false, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunTokenUsageRecord{}, false, err
	}
	return record, true, nil
}

func (s *PostgresStore) FinalizeRunTokenUsage(ctx context.Context, input domain.FinalizeRunTokenUsageInput) (domain.RunTokenUsageSummary, bool, error) {
	runID := strings.TrimSpace(input.RunID)
	if runID == "" {
		return domain.RunTokenUsageSummary{}, false, nil
	}
	completedAt := input.CompletedAt
	if completedAt.IsZero() {
		completedAt = domain.Now()
	}
	day := completedAt.UTC().Truncate(24 * time.Hour)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunTokenUsageSummary{}, false, err
	}
	defer tx.Rollback(ctx)

	summary := domain.RunTokenUsageSummary{RunID: runID, Day: day}
	err = tx.QueryRow(ctx, `
SELECT user_id,
       COALESCE(MAX(NULLIF(model, '')), '') AS model,
       COALESCE(SUM(input_tokens), 0) AS input_tokens,
       COALESCE(SUM(output_tokens), 0) AS output_tokens,
       COALESCE(SUM(total_tokens), 0) AS total_tokens
FROM control_run_token_usage
WHERE run_id = $1
GROUP BY user_id
ORDER BY user_id
LIMIT 1`, runID).Scan(
		&summary.UserID,
		&summary.Model,
		&summary.InputTokens,
		&summary.OutputTokens,
		&summary.TotalTokens,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return summary, false, nil
		}
		return domain.RunTokenUsageSummary{}, false, err
	}
	if summary.UserID == "" || (summary.TotalTokens <= 0 && summary.InputTokens <= 0 && summary.OutputTokens <= 0) {
		return summary, false, nil
	}
	tag, err := tx.Exec(ctx, `
INSERT INTO control_run_token_usage_finalized (run_id, user_id, day, finalized_at)
VALUES ($1, $2, $3, $4)
ON CONFLICT (run_id) DO NOTHING`, runID, summary.UserID, day, timestamptz(completedAt))
	if err != nil {
		return domain.RunTokenUsageSummary{}, false, err
	}
	if tag.RowsAffected() == 0 {
		summary.Finalized = true
		if err := tx.Commit(ctx); err != nil {
			return domain.RunTokenUsageSummary{}, false, err
		}
		return summary, false, nil
	}
	if _, err := incrementUserTokenUsageTx(ctx, tx, summary.UserID, day, 0, 0, 0, 1, completedAt); err != nil {
		return domain.RunTokenUsageSummary{}, false, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunTokenUsageSummary{}, false, err
	}
	summary.Finalized = true
	return summary, true, nil
}

func insertRunTokenUsageTx(ctx context.Context, tx pgx.Tx, input domain.RecordRunTokenUsageInput) (domain.RunTokenUsageRecord, bool, error) {
	var record domain.RunTokenUsageRecord
	err := tx.QueryRow(ctx, `
INSERT INTO control_run_token_usage (run_id, usage_event_id, user_id, model, day, input_tokens, output_tokens, total_tokens, occurred_at, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (run_id, usage_event_id) DO NOTHING
RETURNING run_id, usage_event_id, user_id, model, day, input_tokens, output_tokens, total_tokens, occurred_at, created_at`,
		input.RunID,
		input.UsageEventID,
		input.UserID,
		input.Model,
		input.Day,
		input.InputTokens,
		input.OutputTokens,
		input.TotalTokens,
		timestamptz(input.OccurredAt),
		timestamptz(input.OccurredAt),
	).Scan(
		&record.RunID,
		&record.UsageEventID,
		&record.UserID,
		&record.Model,
		&record.Day,
		&record.InputTokens,
		&record.OutputTokens,
		&record.TotalTokens,
		&record.OccurredAt,
		&record.CreatedAt,
	)
	if err == nil {
		return record, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return domain.RunTokenUsageRecord{}, false, err
	}
	if err := tx.QueryRow(ctx, `
SELECT run_id, usage_event_id, user_id, model, day, input_tokens, output_tokens, total_tokens, occurred_at, created_at
FROM control_run_token_usage
WHERE run_id = $1 AND usage_event_id = $2`, input.RunID, input.UsageEventID).Scan(
		&record.RunID,
		&record.UsageEventID,
		&record.UserID,
		&record.Model,
		&record.Day,
		&record.InputTokens,
		&record.OutputTokens,
		&record.TotalTokens,
		&record.OccurredAt,
		&record.CreatedAt,
	); err != nil {
		return domain.RunTokenUsageRecord{}, false, err
	}
	return record, false, nil
}

func incrementUserTokenUsageTx(ctx context.Context, tx pgx.Tx, userID string, day time.Time, inputTokens int64, outputTokens int64, totalTokens int64, runCount int64, now time.Time) (int64, error) {
	var newDailyTotal int64
	if err := tx.QueryRow(ctx, `
INSERT INTO control_user_token_usage_daily (user_id, day, input_tokens, output_tokens, total_tokens, run_count, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (user_id, day) DO UPDATE SET
  input_tokens = control_user_token_usage_daily.input_tokens + EXCLUDED.input_tokens,
  output_tokens = control_user_token_usage_daily.output_tokens + EXCLUDED.output_tokens,
  total_tokens = control_user_token_usage_daily.total_tokens + EXCLUDED.total_tokens,
  run_count = control_user_token_usage_daily.run_count + EXCLUDED.run_count,
  updated_at = EXCLUDED.updated_at
RETURNING total_tokens`,
		userID, day, inputTokens, outputTokens, totalTokens, runCount, now,
	).Scan(&newDailyTotal); err != nil {
		return 0, err
	}

	_, err := tx.Exec(ctx, `
INSERT INTO control_user_token_usage_lifetime (user_id, input_tokens, output_tokens, total_tokens, peak_daily_total, last_active_day, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (user_id) DO UPDATE SET
  input_tokens = control_user_token_usage_lifetime.input_tokens + EXCLUDED.input_tokens,
  output_tokens = control_user_token_usage_lifetime.output_tokens + EXCLUDED.output_tokens,
  total_tokens = control_user_token_usage_lifetime.total_tokens + EXCLUDED.total_tokens,
  peak_daily_total = GREATEST(control_user_token_usage_lifetime.peak_daily_total, EXCLUDED.peak_daily_total),
  last_active_day = EXCLUDED.last_active_day,
  updated_at = EXCLUDED.updated_at`,
		userID, inputTokens, outputTokens, totalTokens, newDailyTotal, day, now,
	)
	return newDailyTotal, err
}

func (s *PostgresStore) GetUserTokenUsageStats(ctx context.Context, userID string) (domain.UserTokenUsageStats, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.UserTokenUsageStats{}, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT user_id, input_tokens, output_tokens, total_tokens, peak_daily_total, last_active_day, updated_at
FROM control_user_token_usage_lifetime
WHERE user_id = $1`, userID)
	var stats domain.UserTokenUsageStats
	var lastActive *time.Time
	if err := row.Scan(
		&stats.UserID,
		&stats.InputTokens,
		&stats.OutputTokens,
		&stats.TotalTokens,
		&stats.PeakDailyTotal,
		&lastActive,
		&stats.UpdatedAt,
	); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.UserTokenUsageStats{UserID: userID}, nil
		}
		return domain.UserTokenUsageStats{}, err
	}
	stats.LastActiveDay = lastActive
	return stats, nil
}

func (s *PostgresStore) ListUserTokenUsageDaily(ctx context.Context, userID string, since time.Time) ([]domain.UserTokenUsageDaily, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return []domain.UserTokenUsageDaily{}, nil
	}
	sinceDay := since.UTC().Truncate(24 * time.Hour)
	rows, err := s.pool.Query(ctx, `
SELECT day, input_tokens, output_tokens, total_tokens, run_count
FROM control_user_token_usage_daily
WHERE user_id = $1 AND day >= $2
ORDER BY day ASC`, userID, sinceDay)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	daily := []domain.UserTokenUsageDaily{}
	for rows.Next() {
		var record domain.UserTokenUsageDaily
		if err := rows.Scan(
			&record.Day,
			&record.InputTokens,
			&record.OutputTokens,
			&record.TotalTokens,
			&record.RunCount,
		); err != nil {
			return nil, err
		}
		daily = append(daily, record)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return daily, nil
}

func (s *PostgresStore) GetUserLongestRunSeconds(ctx context.Context, userID string) (int64, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return 0, nil
	}
	var seconds float64
	err := s.pool.QueryRow(ctx, `
SELECT COALESCE(MAX(EXTRACT(EPOCH FROM (completed_at - started_at))), 0)
FROM control_runs
WHERE user_id = $1
  AND started_at IS NOT NULL
  AND completed_at IS NOT NULL
  AND completed_at >= started_at`, userID).Scan(&seconds)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return 0, nil
		}
		return 0, err
	}
	if seconds < 0 {
		seconds = 0
	}
	return int64(seconds), nil
}

func (s *PostgresStore) UpsertBisqueCredential(ctx context.Context, input domain.UpsertBisqueCredentialInput) (domain.BisqueCredentialRecord, error) {
	now := domain.Now()
	sessionID := strings.TrimSpace(input.SessionID)
	if sessionID == "" {
		sessionID = domain.NewID("bisque_session")
	}
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = "local-user"
	}
	orgID := strings.TrimSpace(input.OrgID)
	if orgID == "" {
		orgID = "local-org"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	lastVerifiedAt := pgtype.Timestamptz{}
	if !input.LastVerifiedAt.IsZero() {
		lastVerifiedAt = pgtype.Timestamptz{Time: input.LastVerifiedAt.UTC(), Valid: true}
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_bisque_credentials (
  session_id, user_id, org_id, root_url, username,
  password_ciphertext, password_nonce, password_key_id, password_algorithm,
  status, last_verified_at, created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (user_id, org_id, root_url) DO UPDATE
SET username = EXCLUDED.username,
    password_ciphertext = EXCLUDED.password_ciphertext,
    password_nonce = EXCLUDED.password_nonce,
    password_key_id = EXCLUDED.password_key_id,
    password_algorithm = EXCLUDED.password_algorithm,
    status = EXCLUDED.status,
    last_verified_at = EXCLUDED.last_verified_at,
    updated_at = EXCLUDED.updated_at,
    metadata = EXCLUDED.metadata
RETURNING session_id, user_id, COALESCE(org_id, ''), root_url, username,
          password_ciphertext, password_nonce, password_key_id, password_algorithm,
          status, last_verified_at, created_at, updated_at, metadata`,
		sessionID,
		userID,
		orgID,
		strings.TrimRight(strings.TrimSpace(input.RootURL), "/"),
		strings.TrimSpace(input.Username),
		strings.TrimSpace(input.PasswordCiphertext),
		strings.TrimSpace(input.PasswordNonce),
		strings.TrimSpace(input.PasswordKeyID),
		strings.TrimSpace(input.PasswordAlgorithm),
		status,
		lastVerifiedAt,
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanBisqueCredential(row)
}

func (s *PostgresStore) GetBisqueCredentialBySessionID(ctx context.Context, sessionID string) (domain.BisqueCredentialRecord, bool, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT session_id, user_id, COALESCE(org_id, ''), root_url, username,
       password_ciphertext, password_nonce, password_key_id, password_algorithm,
       status, last_verified_at, created_at, updated_at, metadata
FROM control_bisque_credentials
WHERE session_id = $1 AND status <> 'deleted'`,
		sessionID,
	)
	record, err := scanBisqueCredential(row)
	if errors.Is(err, ErrNotFound) {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	if err != nil {
		return domain.BisqueCredentialRecord{}, false, err
	}
	return record, true, nil
}

func (s *PostgresStore) DeleteBisqueCredentialBySessionID(ctx context.Context, sessionID string) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	_, err := s.pool.Exec(ctx, `
UPDATE control_bisque_credentials
SET status = 'deleted',
    updated_at = $2
WHERE session_id = $1`,
		sessionID,
		domain.Now(),
	)
	return err
}

// GetActiveBisqueCredentialForUser resolves a user's linked BisQue credential by
// account identity so linked detection no longer depends on the session cookie.
// Backed by the control_bisque_credentials_user_status_idx index.
func (s *PostgresStore) GetActiveBisqueCredentialForUser(ctx context.Context, userID string, orgID string) (domain.BisqueCredentialRecord, bool, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	orgID = strings.TrimSpace(orgID)
	row := s.pool.QueryRow(ctx, `
SELECT session_id, user_id, COALESCE(org_id, ''), root_url, username,
       password_ciphertext, password_nonce, password_key_id, password_algorithm,
       status, last_verified_at, created_at, updated_at, metadata
FROM control_bisque_credentials
WHERE user_id = $1 AND status = 'active'
  AND ($2 = '' OR COALESCE(org_id, '') = '' OR COALESCE(org_id, '') = $2)
ORDER BY updated_at DESC
LIMIT 1`,
		userID,
		orgID,
	)
	record, err := scanBisqueCredential(row)
	if errors.Is(err, ErrNotFound) {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	if err != nil {
		return domain.BisqueCredentialRecord{}, false, err
	}
	return record, true, nil
}

func (s *PostgresStore) UpsertWorkerHeartbeat(ctx context.Context, input domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error) {
	now := domain.Now()
	workerID := strings.TrimSpace(input.WorkerID)
	if workerID == "" {
		return domain.WorkerHeartbeatRecord{}, ErrConflict
	}
	workerKind := strings.TrimSpace(input.WorkerKind)
	if workerKind == "" {
		workerKind = "worker"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "alive"
	}
	heartbeatAt := input.LastHeartbeatAt
	if heartbeatAt.IsZero() {
		heartbeatAt = now
	}
	heartbeatAt = heartbeatAt.UTC()
	startedAt := input.StartedAt
	if startedAt.IsZero() {
		startedAt = heartbeatAt
	}
	startedAt = startedAt.UTC()
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_worker_heartbeats (
  worker_id, worker_kind, status, current_run_id, hostname, version,
  started_at, last_heartbeat_at, updated_at, metadata
)
VALUES ($1, $2, $3, NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''), $7, $8, $9, $10)
ON CONFLICT (worker_id) DO UPDATE
SET worker_kind = EXCLUDED.worker_kind,
    status = EXCLUDED.status,
    current_run_id = EXCLUDED.current_run_id,
    hostname = EXCLUDED.hostname,
    version = EXCLUDED.version,
    started_at = control_worker_heartbeats.started_at,
    last_heartbeat_at = EXCLUDED.last_heartbeat_at,
    updated_at = EXCLUDED.updated_at,
    metadata = EXCLUDED.metadata
RETURNING worker_id, worker_kind, status, COALESCE(current_run_id, ''), COALESCE(hostname, ''), COALESCE(version, ''),
          started_at, last_heartbeat_at, updated_at, metadata`,
		workerID,
		workerKind,
		status,
		strings.TrimSpace(input.CurrentRunID),
		strings.TrimSpace(input.Hostname),
		strings.TrimSpace(input.Version),
		startedAt,
		heartbeatAt,
		now,
		jsonBytes(input.Metadata),
	)
	return scanWorkerHeartbeat(row)
}

func (s *PostgresStore) ListWorkerHeartbeats(ctx context.Context, limit int) ([]domain.WorkerHeartbeatRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT worker_id, worker_kind, status, COALESCE(current_run_id, ''), COALESCE(hostname, ''), COALESCE(version, ''),
       started_at, last_heartbeat_at, updated_at, metadata
FROM control_worker_heartbeats
ORDER BY last_heartbeat_at DESC
LIMIT $1`, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	workers := []domain.WorkerHeartbeatRecord{}
	for rows.Next() {
		worker, err := scanWorkerHeartbeat(rows)
		if err != nil {
			return nil, err
		}
		workers = append(workers, worker)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return workers, nil
}

func (s *PostgresStore) GetWorkerHeartbeat(ctx context.Context, workerID string) (domain.WorkerHeartbeatRecord, bool, error) {
	worker, err := scanWorkerHeartbeat(s.pool.QueryRow(ctx, `
SELECT worker_id, worker_kind, status, COALESCE(current_run_id, ''), COALESCE(hostname, ''), COALESCE(version, ''),
       started_at, last_heartbeat_at, updated_at, metadata
FROM control_worker_heartbeats
WHERE worker_id = $1`, strings.TrimSpace(workerID)))
	if err == nil {
		return worker, true, nil
	}
	if errors.Is(err, ErrNotFound) {
		return domain.WorkerHeartbeatRecord{}, false, nil
	}
	return domain.WorkerHeartbeatRecord{}, false, err
}

func (s *PostgresStore) CreateThread(ctx context.Context, input domain.CreateThreadInput) (domain.ThreadRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	defer tx.Rollback(ctx)

	q := s.queries.WithTx(tx)
	now := domain.Now()
	row, err := q.CreateThread(ctx, sqlc.CreateThreadParams{
		ThreadID:     domain.NewID("thread"),
		UserID:       input.UserID,
		Title:        nullableText(input.Title),
		Status:       string(domain.ThreadStatusActive),
		CreatedAt:    timestamptz(now),
		UpdatedAt:    timestamptz(now),
		LatestRunID:  pgtype.Text{},
		CheckpointID: pgtype.Text{},
		Summary:      pgtype.Text{},
		Metadata:     jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	for _, msg := range input.InitialMessages {
		if _, err := q.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
			MessageID: domain.NewID("msg"),
			ThreadID:  row.ThreadID,
			Role:      msg.Role,
			Content:   msg.Content,
			CreatedAt: timestamptz(now),
			Metadata:  jsonBytes(msg.Metadata),
			RunID:     pgtype.Text{},
		}); err != nil {
			return domain.ThreadRecord{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ThreadRecord{}, err
	}
	return threadFromRow(row), nil
}

func (s *PostgresStore) GetThread(ctx context.Context, threadID string) (domain.ThreadRecord, error) {
	row, err := s.queries.GetThread(ctx, threadID)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(row), nil
}

func (s *PostgresStore) GetThreadForUser(ctx context.Context, threadID string, userID string) (domain.ThreadRecord, error) {
	row, err := s.queries.GetThreadForUser(ctx, sqlc.GetThreadForUserParams{
		ThreadID: threadID,
		UserID:   userID,
	})
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	thread := threadFromRow(row)
	if !threadVisibleForUserRead(thread) {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *PostgresStore) UpdateThreadForUser(ctx context.Context, input domain.UpdateThreadInput) (domain.ThreadRecord, error) {
	now := domain.Now()
	row, err := s.pool.Query(ctx, `
UPDATE control_threads
SET title = COALESCE(NULLIF($3, ''), title),
	    metadata = COALESCE(metadata, '{}'::jsonb) || $4::jsonb,
	    updated_at = $5
	WHERE thread_id = $1 AND user_id = $2
	  AND status <> 'deleted'
	RETURNING thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata`,
		input.ThreadID,
		input.UserID,
		normalizedThreadTitle(input.Title),
		jsonBytes(mapOrEmpty(input.Metadata)),
		now,
	)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	defer row.Close()
	thread, err := pgx.CollectOneRow(row, pgx.RowToStructByName[sqlc.ControlThread])
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(thread), nil
}

// HardDeleteThreadForUser permanently removes a conversation and everything
// derived from it. There is no undo.
//
// This exists because the product promised erasure and performed concealment:
// the old path ran `UPDATE control_threads SET status='deleted'`, so not one row
// was removed, the schema's ON DELETE CASCADE chains never fired, and the whole
// transcript stayed readable in control_threads.metadata.frontend_state.
//
// The cascades were already declared and correct — they were simply unreachable.
// Deleting the thread row now does most of the work:
//
//	control_threads
//	  └─ control_thread_messages   (schema.sql:85)
//	  └─ control_runs              (schema.sql:95)
//	       └─ run event sequences  (schema.sql:127)
//	       └─ control_run_events   (schema.sql:135)
//	       └─ run leases           (schema.sql:170)
//	       └─ control_artifacts    (schema.sql:193)
//
// Two things the cascades cannot reach, and why they are handled explicitly:
//
//  1. control_run_token_usage and control_run_token_usage_finalized key on
//     run_id with NO foreign key, so they would survive as orphans. They are
//     deleted here by run id, collected before the parent row goes.
//  2. Artifact blobs live outside Postgres behind control_artifacts.storage_uri.
//     The rows cascade; the bytes do not. The URIs are returned so the caller
//     can unlink them after the transaction commits — deliberately outside the
//     tx, because a blob-store failure must not roll back a deletion the user
//     has already been told is permanent.
//
// Deliberately NOT touched: control_resources. Uploaded files are independent of
// conversations by design (no thread FK) and have their own hard-delete path in
// PurgeResource. Deleting a conversation must never delete the user's data.
//
// deepagents_checkpoint_threads is owned by the Python runtime, which already
// deletes on terminal ack and GCs at 72h; it is not reachable from this pool's
// schema and is left to that owner.
func (s *PostgresStore) HardDeleteThreadForUser(ctx context.Context, threadID string, userID string) ([]string, error) {
	threadID = strings.TrimSpace(threadID)
	userID = strings.TrimSpace(userID)
	if threadID == "" || userID == "" {
		return nil, ErrNotFound
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// Ownership is asserted here, before anything is removed, so a mismatched
	// user can never cause a partial delete.
	var owned string
	if err := tx.QueryRow(ctx,
		`SELECT thread_id FROM control_threads WHERE thread_id = $1 AND user_id = $2 FOR UPDATE`,
		threadID, userID,
	).Scan(&owned); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, mapPgError(err)
	}

	// Collect before deleting: after the cascade these rows are gone.
	runRows, err := tx.Query(ctx, `SELECT run_id FROM control_runs WHERE thread_id = $1`, threadID)
	if err != nil {
		return nil, mapPgError(err)
	}
	runIDs := make([]string, 0, 8)
	for runRows.Next() {
		var id string
		if err := runRows.Scan(&id); err != nil {
			runRows.Close()
			return nil, mapPgError(err)
		}
		runIDs = append(runIDs, id)
	}
	runRows.Close()
	if err := runRows.Err(); err != nil {
		return nil, mapPgError(err)
	}

	blobRows, err := tx.Query(ctx,
		`SELECT COALESCE(storage_uri, '') FROM control_artifacts WHERE thread_id = $1 OR run_id = ANY($2::text[])`,
		threadID, runIDs,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	storageURIs := make([]string, 0, 8)
	for blobRows.Next() {
		var uri string
		if err := blobRows.Scan(&uri); err != nil {
			blobRows.Close()
			return nil, mapPgError(err)
		}
		if uri = strings.TrimSpace(uri); uri != "" {
			storageURIs = append(storageURIs, uri)
		}
	}
	blobRows.Close()
	if err := blobRows.Err(); err != nil {
		return nil, mapPgError(err)
	}

	// Tables with NO foreign key at all. Nothing in the catalog points at these,
	// so they cannot be discovered below — they must be listed. They key on
	// run_id and would simply be orphaned, invisibly.
	if len(runIDs) > 0 {
		for _, table := range []string{"control_run_token_usage", "control_run_token_usage_finalized"} {
			// Fixed identifiers from a literal slice, never user input.
			if _, err := tx.Exec(ctx, `DELETE FROM `+table+` WHERE run_id = ANY($1::text[])`, runIDs); err != nil {
				return nil, mapPgError(err)
			}
		}
	}

	// Tables that DO reference us but without ON DELETE CASCADE. These are worse
	// than orphans: a NO ACTION foreign key makes Postgres raise on the parent
	// delete, so the transaction aborts and the user's delete fails outright.
	//
	// Discovered from the catalog rather than hardcoded, because the set is not
	// knowable from the checked-in schema and differs per deployment. schema.sql
	// declares none of them; the local compose database has two
	// (control_run_specs, control_calphad_validation_events) while the real
	// application database has nine, including several control_ultra_admission_*
	// tables that reference threads by thread_id rather than runs by run_id. A
	// hardcoded list was wrong the moment it was written and would rot again.
	//
	// Identifiers come from pg_class/pg_attribute — the catalog, never user
	// input — and are quoted with quote_ident before interpolation.
	refRows, err := tx.Query(ctx, `
SELECT quote_ident(src.relname), quote_ident(a.attname), tgt.relname
FROM pg_constraint c
JOIN pg_class src ON src.oid = c.conrelid
JOIN pg_class tgt ON tgt.oid = c.confrelid
JOIN unnest(c.conkey) AS k(attnum) ON true
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
WHERE c.contype = 'f'
  AND tgt.relname IN ('control_runs', 'control_threads')
  -- Only NO ACTION ('a') and RESTRICT ('r') actually block the parent delete.
  -- CASCADE ('c') and SET NULL ('n') resolve themselves, and trying to DELETE
  -- from a SET NULL referencer would be actively wrong here: several of these
  -- tables are append-only ledgers whose triggers raise on DELETE, so a sweep
  -- that touched them would abort the whole transaction.
  AND c.confdeltype IN ('a', 'r')`)
	if err != nil {
		return nil, mapPgError(err)
	}
	type dependent struct{ table, column, parent string }
	dependents := make([]dependent, 0, 8)
	for refRows.Next() {
		var d dependent
		if err := refRows.Scan(&d.table, &d.column, &d.parent); err != nil {
			refRows.Close()
			return nil, mapPgError(err)
		}
		dependents = append(dependents, d)
	}
	refRows.Close()
	if err := refRows.Err(); err != nil {
		return nil, mapPgError(err)
	}

	for _, d := range dependents {
		var err error
		if d.parent == "control_threads" {
			_, err = tx.Exec(ctx, `DELETE FROM `+d.table+` WHERE `+d.column+` = $1`, threadID)
		} else {
			if len(runIDs) == 0 {
				continue
			}
			_, err = tx.Exec(ctx, `DELETE FROM `+d.table+` WHERE `+d.column+` = ANY($1::text[])`, runIDs)
		}
		if err != nil {
			return nil, mapPgError(err)
		}
	}

	// The parent row last: everything above depends on it still existing.
	tag, err := tx.Exec(ctx, `DELETE FROM control_threads WHERE thread_id = $1 AND user_id = $2`, threadID, userID)
	if err != nil {
		return nil, mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return nil, ErrNotFound
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, mapPgError(err)
	}
	return storageURIs, nil
}

func (s *PostgresStore) SoftDeleteThreadForUser(ctx context.Context, threadID string, userID string, deletedAt time.Time) (domain.ThreadRecord, error) {
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	rows, err := s.pool.Query(ctx, `
UPDATE control_threads
SET status = $3,
    updated_at = $4
WHERE thread_id = $1
  AND user_id = $2
  AND status <> $3
RETURNING thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata`,
		threadID,
		userID,
		string(domain.ThreadStatusDeleted),
		timestamptz(deletedAt.UTC()),
	)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	defer rows.Close()
	thread, err := pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlThread])
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(thread), nil
}

func (s *PostgresStore) ApplyGeneratedThreadTitle(ctx context.Context, input domain.ApplyGeneratedThreadTitleInput) (domain.ThreadRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	defer tx.Rollback(ctx)

	row, err := lockedControlThread(ctx, tx, input.ThreadID)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	thread := threadFromRow(row)
	title := normalizedThreadTitle(input.Title)
	if title == "" || !generatedThreadTitleEligible(thread) {
		if err := tx.Commit(ctx); err != nil {
			return domain.ThreadRecord{}, err
		}
		return thread, nil
	}
	now := domain.Now()
	metadata := generatedThreadTitleMetadata(thread.Metadata, input, thread.Title, now)
	updated, err := tx.Query(ctx, `
UPDATE control_threads
SET title = $2,
    metadata = $3,
    updated_at = $4
WHERE thread_id = $1
RETURNING thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata`,
		input.ThreadID,
		title,
		jsonBytes(metadata),
		now,
	)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	defer updated.Close()
	updatedThread, err := pgx.CollectOneRow(updated, pgx.RowToStructByName[sqlc.ControlThread])
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ThreadRecord{}, err
	}
	return threadFromRow(updatedThread), nil
}

func lockedControlThread(ctx context.Context, tx pgx.Tx, threadID string) (sqlc.ControlThread, error) {
	rows, err := tx.Query(ctx, `
SELECT thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata
FROM control_threads
WHERE thread_id = $1
FOR UPDATE`, threadID)
	if err != nil {
		return sqlc.ControlThread{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlThread])
}

func (s *PostgresStore) ListThreads(ctx context.Context, limit int, offset int, status string) (domain.ThreadListPage, error) {
	resolvedLimit := limit32(limit, 100)
	resolvedOffset := max(offset, 0)
	resolvedStatus := strings.TrimSpace(status)
	totalCount, err := s.queries.CountThreads(ctx, resolvedStatus)
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	rows, err := s.queries.ListThreads(ctx, sqlc.ListThreadsParams{
		Column1: resolvedStatus,
		Limit:   resolvedLimit,
		Offset:  int32(resolvedOffset),
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	threads := make([]domain.ThreadRecord, 0, len(rows))
	for _, row := range rows {
		threads = append(threads, threadFromRow(row))
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: int(totalCount),
		Limit:      int(resolvedLimit),
		Offset:     resolvedOffset,
	}, nil
}

func (s *PostgresStore) ListThreadsForUser(ctx context.Context, userID string, limit int, offset int, status string) (domain.ThreadListPage, error) {
	resolvedLimit := limit32(limit, 100)
	resolvedOffset := max(offset, 0)
	resolvedStatus := strings.TrimSpace(status)
	totalCount, err := s.queries.CountThreadsForUser(ctx, sqlc.CountThreadsForUserParams{
		UserID:  userID,
		Column2: resolvedStatus,
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	rows, err := s.queries.ListThreadsForUser(ctx, sqlc.ListThreadsForUserParams{
		UserID:  userID,
		Column2: resolvedStatus,
		Limit:   resolvedLimit,
		Offset:  int32(resolvedOffset),
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	threads := make([]domain.ThreadRecord, 0, len(rows))
	for _, row := range rows {
		threads = append(threads, threadFromRow(row))
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: int(totalCount),
		Limit:      int(resolvedLimit),
		Offset:     resolvedOffset,
	}, nil
}

func (s *PostgresStore) ListThreadMessages(ctx context.Context, threadID string) ([]domain.ThreadMessage, error) {
	rows, err := s.queries.ListThreadMessages(ctx, threadID)
	if err != nil {
		return nil, err
	}
	messages := make([]domain.ThreadMessage, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, threadMessageFromRow(row))
	}
	return messages, nil
}

// ListThreadMessagePageForUser returns a "load earlier" page of a thread's messages. It currently
// loads the (small) message set and pages in Go; the limit+before-cursor contract is keyset-friendly
// so this can become a single keyset query if thread sizes ever warrant it, without changing callers.
func (s *PostgresStore) ListThreadMessagePageForUser(
	ctx context.Context,
	threadID string,
	userID string,
	beforeMessageID string,
	limit int,
) ([]domain.ThreadMessage, bool, error) {
	all, err := s.ListThreadMessagesForUser(ctx, threadID, userID)
	if err != nil {
		return nil, false, err
	}
	page, hasMore := pageThreadMessagesTail(all, beforeMessageID, limit)
	return page, hasMore, nil
}

func (s *PostgresStore) ListThreadMessagesForUser(ctx context.Context, threadID string, userID string) ([]domain.ThreadMessage, error) {
	if _, err := s.GetThreadForUser(ctx, threadID, userID); err != nil {
		return nil, err
	}
	rows, err := s.queries.ListThreadMessagesForUser(ctx, sqlc.ListThreadMessagesForUserParams{
		ThreadID: threadID,
		UserID:   userID,
	})
	if err != nil {
		return nil, err
	}
	messages := make([]domain.ThreadMessage, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, threadMessageFromRow(row))
	}
	return messages, nil
}

func (s *PostgresStore) AppendThreadMessage(ctx context.Context, message domain.ThreadMessage) (domain.ThreadMessage, error) {
	if message.MessageID == "" {
		message.MessageID = domain.NewID("msg")
	}
	if message.CreatedAt.IsZero() {
		message.CreatedAt = domain.Now()
	}
	row, err := s.queries.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
		MessageID: message.MessageID,
		ThreadID:  message.ThreadID,
		Role:      message.Role,
		Content:   message.Content,
		CreatedAt: timestamptz(message.CreatedAt),
		Metadata:  jsonBytes(message.Metadata),
		RunID:     nullableText(message.RunID),
	})
	if err != nil {
		return domain.ThreadMessage{}, mapPgError(err)
	}
	return threadMessageFromRow(row), nil
}

func (s *PostgresStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunRecord{}, err
	}
	defer tx.Rollback(ctx)

	q := s.queries.WithTx(tx)
	now := domain.Now()
	workflowKind := input.WorkflowKind
	if workflowKind == "" {
		workflowKind = "deep_agents"
	}
	mode := input.Mode
	if mode == "" {
		mode = "durable"
	}
	row, err := q.CreateRun(ctx, sqlc.CreateRunParams{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       string(domain.RunStatusQueued),
		WorkflowKind: workflowKind,
		Mode:         nullableText(mode),
		CreatedAt:    timestamptz(now),
		UpdatedAt:    timestamptz(now),
		Metadata:     jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if !input.Internal {
		if err := q.SetThreadLatestRun(ctx, sqlc.SetThreadLatestRunParams{
			ThreadID:    input.ThreadID,
			LatestRunID: nullableText(row.RunID),
			UpdatedAt:   timestamptz(now),
		}); err != nil {
			return domain.RunRecord{}, mapPgError(err)
		}
		for _, msg := range input.Messages {
			if _, err := q.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
				MessageID: domain.NewID("msg"),
				ThreadID:  input.ThreadID,
				Role:      msg.Role,
				Content:   msg.Content,
				CreatedAt: timestamptz(now),
				Metadata:  jsonBytes(msg.Metadata),
				RunID:     nullableText(threadMessageRunID(msg, row.RunID)),
			}); err != nil {
				return domain.RunRecord{}, err
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunRecord{}, err
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) FindRunByIdempotencyKey(ctx context.Context, threadID string, userID string, idempotencyKey string) (domain.RunRecord, bool, error) {
	row, err := s.pool.Query(ctx, `
SELECT run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
       planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
       response_text, error, created_at, updated_at, started_at, completed_at, metadata
FROM control_runs
WHERE thread_id = $1 AND user_id = $2 AND metadata->>'idempotency_key' = $3
ORDER BY created_at ASC
LIMIT 1`, threadID, userID, idempotencyKey)
	if err != nil {
		return domain.RunRecord{}, false, err
	}
	defer row.Close()
	rows, err := pgx.CollectRows(row, pgx.RowToStructByName[sqlc.ControlRun])
	if err != nil {
		return domain.RunRecord{}, false, err
	}
	if len(rows) == 0 {
		return domain.RunRecord{}, false, nil
	}
	return runFromRow(rows[0]), true, nil
}

func (s *PostgresStore) MarkRunDispatched(ctx context.Context, runID string, dispatchedAt time.Time) (domain.RunRecord, error) {
	if dispatchedAt.IsZero() {
		dispatchedAt = domain.Now()
	}
	tag, err := s.pool.Exec(ctx, `
UPDATE control_runs
SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('job_dispatched_at', $2::text),
    updated_at = $3
WHERE run_id = $1`,
		runID,
		dispatchedAt.UTC().Format(time.RFC3339Nano),
		domain.Now(),
	)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return domain.RunRecord{}, ErrNotFound
	}
	return s.GetRun(ctx, runID)
}

func (s *PostgresStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	row, err := s.queries.GetRun(ctx, runID)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) GetRunForUser(ctx context.Context, runID string, userID string) (domain.RunRecord, error) {
	row, err := s.queries.GetRunForUser(ctx, sqlc.GetRunForUserParams{
		RunID:  runID,
		UserID: userID,
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) ListRuns(ctx context.Context, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	rows, err := s.queries.ListRuns(ctx, sqlc.ListRunsParams{
		Column1: threadID,
		Column2: status,
		Limit:   limit32(limit, 100),
		Offset:  int32(max(offset, 0)),
	})
	if err != nil {
		return nil, err
	}
	runs := make([]domain.RunRecord, 0, len(rows))
	for _, row := range rows {
		runs = append(runs, runFromRow(row))
	}
	return runs, nil
}

func (s *PostgresStore) ListRunsForUser(ctx context.Context, userID string, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	if strings.TrimSpace(threadID) != "" {
		if _, err := s.GetThreadForUser(ctx, threadID, userID); err != nil {
			return nil, err
		}
	}
	rows, err := s.queries.ListRunsForUser(ctx, sqlc.ListRunsForUserParams{
		UserID:  userID,
		Column2: strings.TrimSpace(threadID),
		Column3: strings.TrimSpace(status),
		Limit:   limit32(limit, 100),
		Offset:  int32(max(offset, 0)),
	})
	if err != nil {
		return nil, err
	}
	runs := make([]domain.RunRecord, 0, len(rows))
	for _, row := range rows {
		runs = append(runs, runFromRow(row))
	}
	return runs, nil
}

// runHistorySnippetChars bounds the response excerpt returned per episodic hit.
const runHistorySnippetChars = 600

// runHistorySearchTerms splits an episodic query into up to 10 lowercase terms.
// Empty result means "no keyword filter" (recency-only).
func runHistorySearchTerms(query string) []string {
	terms := make([]string, 0, 10)
	for _, field := range strings.Fields(strings.ToLower(query)) {
		term := strings.TrimSpace(field)
		if term == "" {
			continue
		}
		if len(term) > 64 {
			term = term[:64]
		}
		terms = append(terms, term)
		if len(terms) >= 10 {
			break
		}
	}
	return terms
}

// SearchRunHistoryForUser powers episodic memory: it returns the user's own past
// succeeded runs matching an optional keyword (over goal, final response, and
// thread title), most recent first. Scoped to user_id at the DB level so it can
// never surface another user's history. ILIKE over one user's bounded history on
// the (user_id, status, updated_at) index — no full-text index required.
func (s *PostgresStore) SearchRunHistoryForUser(ctx context.Context, userID string, opts domain.RunHistorySearchOptions) ([]domain.RunHistoryHit, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return []domain.RunHistoryHit{}, nil
	}
	terms := runHistorySearchTerms(opts.Query)
	limit := opts.Limit
	if limit <= 0 || limit > 20 {
		limit = 20
	}
	var since *time.Time
	if opts.Since != nil && !opts.Since.IsZero() {
		t := opts.Since.UTC()
		since = &t
	}
	// All-terms-match: every whitespace-separated query term must appear somewhere
	// in goal+response+title (case-insensitive). A natural multi-word query like
	// "ferret rna-seq interferon" matches even when the words are not contiguous.
	rows, err := s.pool.Query(ctx, `
SELECT r.run_id, r.thread_id, COALESCE(t.title, ''), r.goal,
       LEFT(COALESCE(r.response_text, ''), $5), r.completed_at
FROM control_runs r
LEFT JOIN control_threads t ON t.thread_id = r.thread_id
WHERE r.user_id = $1
  AND r.status = 'succeeded'
  -- This backs the agent's episodic-memory tool, so anything it returns the
  -- model may quote back at the user. Without this clause a deleted
  -- conversation could resurface in a later answer.
  --
  -- Hard delete removes these runs outright, so for anything deleted from now
  -- on the clause is redundant. It is here for the rows already soft-deleted
  -- before that shipped, which still carry their runs, and as a standing guard
  -- if any soft-delete path is ever reintroduced. IS NULL keeps runs whose
  -- thread row is genuinely absent (the LEFT JOIN case) visible.
  AND (t.thread_id IS NULL OR t.status <> 'deleted')
  AND (cardinality($2::text[]) = 0 OR NOT EXISTS (
        SELECT 1 FROM unnest($2::text[]) AS term
        WHERE (r.goal || ' ' || COALESCE(r.response_text, '') || ' ' || COALESCE(t.title, ''))
              NOT ILIKE '%' || term || '%'
      ))
  AND ($3::timestamptz IS NULL OR r.completed_at >= $3)
  AND ($4 = '' OR r.run_id <> $4)
ORDER BY r.completed_at DESC NULLS LAST
LIMIT $6`, userID, terms, since, strings.TrimSpace(opts.ExcludeRunID), runHistorySnippetChars, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	hits := []domain.RunHistoryHit{}
	for rows.Next() {
		var hit domain.RunHistoryHit
		var completedAt pgtype.Timestamptz
		if err := rows.Scan(&hit.RunID, &hit.ThreadID, &hit.Title, &hit.Goal, &hit.ResponseSnippet, &completedAt); err != nil {
			return nil, err
		}
		hit.CompletedAt = timePtr(completedAt)
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return hits, nil
}

func (s *PostgresStore) GetRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	lease, err := scanRunLease(s.pool.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1`, runID))
	if err == nil {
		return lease, true, nil
	}
	if errors.Is(err, ErrNotFound) {
		var exists bool
		if existsErr := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, runID).Scan(&exists); existsErr != nil {
			return domain.RunLeaseRecord{}, false, existsErr
		}
		if !exists {
			return domain.RunLeaseRecord{}, false, ErrNotFound
		}
		return domain.RunLeaseRecord{}, false, nil
	}
	return domain.RunLeaseRecord{}, false, err
}

func (s *PostgresStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	row, err := s.queries.UpdateRunStatus(ctx, sqlc.UpdateRunStatusParams{
		RunID:     runID,
		Status:    string(status),
		Column3:   responseText,
		Column4:   errorText,
		UpdatedAt: timestamptz(domain.Now()),
	})
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			existing, getErr := s.GetRun(ctx, runID)
			if getErr != nil {
				return domain.RunRecord{}, getErr
			}
			if isTerminalRunStatus(existing.Status) {
				return existing, nil
			}
		}
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) CompleteRun(ctx context.Context, input domain.CompleteRunInput) (domain.RunRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunRecord{}, err
	}
	defer tx.Rollback(ctx)

	row, err := lockedControlRun(ctx, tx, input.RunID)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	run := runFromRow(row)
	responseText := strings.TrimSpace(input.ResponseText)
	if run.Status == domain.RunStatusSucceeded {
		if strings.TrimSpace(run.ResponseText) == "" && responseText != "" {
			updated, err := repairSucceededRunResponseTextTx(ctx, tx, run.RunID, responseText, domain.Now())
			if err != nil {
				return domain.RunRecord{}, mapPgError(err)
			}
			run = runFromRow(updated)
		}
		if err := appendCompletedAssistantMessageTx(ctx, tx, run, responseText); err != nil {
			return domain.RunRecord{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return domain.RunRecord{}, err
		}
		return run, nil
	}
	if isTerminalRunStatus(run.Status) {
		if err := tx.Commit(ctx); err != nil {
			return domain.RunRecord{}, err
		}
		return run, nil
	}
	if err := appendCompletedAssistantMessageTx(ctx, tx, run, responseText); err != nil {
		return domain.RunRecord{}, err
	}
	updated, err := completeControlRunTx(ctx, tx, run.RunID, responseText, domain.Now())
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunRecord{}, err
	}
	return runFromRow(updated), nil
}

func lockedControlRun(ctx context.Context, tx pgx.Tx, runID string) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
SELECT run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
       planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
       response_text, error, created_at, updated_at, started_at, completed_at, metadata
FROM control_runs
WHERE run_id = $1
FOR UPDATE`, runID)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func repairSucceededRunResponseTextTx(ctx context.Context, tx pgx.Tx, runID string, responseText string, now time.Time) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
UPDATE control_runs
SET response_text = $2,
    updated_at = $3
WHERE run_id = $1
  AND status = 'succeeded'
  AND COALESCE(response_text, '') = ''
RETURNING run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
          planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
          response_text, error, created_at, updated_at, started_at, completed_at, metadata`,
		runID,
		responseText,
		timestamptz(now),
	)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func appendCompletedAssistantMessageTx(ctx context.Context, tx pgx.Tx, run domain.RunRecord, responseText string) error {
	if responseText == "" || isInternalRunMetadata(run.Metadata) {
		return nil
	}
	var exists bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS(
  SELECT 1
  FROM control_thread_messages
  WHERE thread_id = $1
    AND run_id = $2
    AND lower(btrim(role)) = 'assistant'
    AND content = $3
)`, run.ThreadID, run.RunID, responseText).Scan(&exists); err != nil {
		return err
	}
	if exists {
		return nil
	}
	now := domain.Now()
	_, err := tx.Exec(ctx, `
INSERT INTO control_thread_messages (message_id, thread_id, role, content, created_at, metadata, run_id)
VALUES ($1, $2, 'assistant', $3, $4, '{}'::jsonb, $5)`,
		domain.NewID("msg"),
		run.ThreadID,
		responseText,
		now,
		run.RunID,
	)
	return mapPgError(err)
}

func completeControlRunTx(ctx context.Context, tx pgx.Tx, runID string, responseText string, now time.Time) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
UPDATE control_runs
SET status = 'succeeded',
    response_text = NULLIF($2, ''),
    error = NULL,
    updated_at = $3,
    completed_at = $3
WHERE run_id = $1
  AND status NOT IN ('succeeded', 'failed', 'canceled')
RETURNING run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
          planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
          response_text, error, created_at, updated_at, started_at, completed_at, metadata`,
		runID,
		responseText,
		timestamptz(now),
	)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func (s *PostgresStore) AcquireRunLease(ctx context.Context, input domain.AcquireRunLeaseInput) (domain.RunLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx, `SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`, input.RunID).Scan(&status); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if isTerminalRunStatus(domain.RunStatus(status)) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	ttl := positiveLeaseTTL(input.TTL)
	existing, err := scanRunLease(tx.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1
FOR UPDATE`, input.RunID))
	if err == nil && existing.LeaseExpiresAt.After(now) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	if err != nil && !errors.Is(err, ErrNotFound) {
		return domain.RunLeaseRecord{}, err
	}
	lease := domain.RunLeaseRecord{
		RunID:          input.RunID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("lease"),
		LeaseExpiresAt: now.Add(ttl),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	row := tx.QueryRow(ctx, `
INSERT INTO control_run_leases (run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (run_id) DO UPDATE
SET worker_id = EXCLUDED.worker_id,
    lease_token = EXCLUDED.lease_token,
    lease_expires_at = EXCLUDED.lease_expires_at,
    updated_at = EXCLUDED.updated_at
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		lease.RunID,
		lease.WorkerID,
		lease.LeaseToken,
		lease.LeaseExpiresAt,
		lease.CreatedAt,
		lease.UpdatedAt,
	)
	lease, err = scanRunLease(row)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	if _, err := tx.Exec(ctx, `
UPDATE control_runs
SET status = 'running',
    updated_at = $2,
    started_at = COALESCE(started_at, $2)
WHERE run_id = $1
  AND status NOT IN ('succeeded', 'failed', 'canceled')`, input.RunID, now); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) RenewRunLease(ctx context.Context, input domain.RenewRunLeaseInput) (domain.RunLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx, `SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`, input.RunID).Scan(&status); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if isTerminalRunStatus(domain.RunStatus(status)) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	existing, err := scanRunLease(tx.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1
FOR UPDATE`, input.RunID))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.RunLeaseRecord{}, ErrConflict
		}
		return domain.RunLeaseRecord{}, err
	}
	// Token match alone authorizes renewal — even of an EXPIRED lease. A
	// worker that survived a control-plane outage must be able to revive its
	// lease and keep its (expensive, GPU-hours) computation. If recovery
	// already requeued the run, ClearRunLease removed this row, the lookup
	// above misses, and the worker gets the authoritative conflict instead.
	if existing.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	lease, err := scanRunLease(tx.QueryRow(ctx, `
UPDATE control_run_leases
SET lease_expires_at = $3,
    updated_at = $4
WHERE run_id = $1 AND lease_token = $2
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		input.RunID,
		strings.TrimSpace(input.LeaseToken),
		now.Add(positiveLeaseTTL(input.TTL)),
		now,
	))
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	if _, err := tx.Exec(ctx, `UPDATE control_runs SET updated_at = $2 WHERE run_id = $1`, input.RunID, now); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) ReleaseRunLease(ctx context.Context, input domain.ReleaseRunLeaseInput) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM control_run_leases WHERE run_id = $1 AND lease_token = $2`, input.RunID, strings.TrimSpace(input.LeaseToken))
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() > 0 {
		return nil
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, input.RunID).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return ErrNotFound
	}
	var activeToken string
	err = s.pool.QueryRow(ctx, `SELECT lease_token FROM control_run_leases WHERE run_id = $1`, input.RunID).Scan(&activeToken)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	if err != nil {
		return mapPgError(err)
	}
	return ErrConflict
}

func (s *PostgresStore) ClearRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	lease, err := scanRunLease(s.pool.QueryRow(ctx, `
DELETE FROM control_run_leases
WHERE run_id = $1
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`, runID))
	if err == nil {
		return lease, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return domain.RunLeaseRecord{}, false, err
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, runID).Scan(&exists); err != nil {
		return domain.RunLeaseRecord{}, false, mapPgError(err)
	}
	if !exists {
		return domain.RunLeaseRecord{}, false, ErrNotFound
	}
	return domain.RunLeaseRecord{}, false, nil
}

// appendRunEventSQL inserts one event in a single statement. The CTE chain
// allocates the next sequence from control_run_event_sequences: writers for
// the same run serialize on that row's lock, and the outer INSERT reads the
// CTE's output, so the allocation is ordered by data dependency. GREATEST
// against the events table's actual MAX makes the allocator self-healing if
// the counter is ever stale (missing backfill, mixed-version writers); the
// UNIQUE(run_id, sequence_number) constraint remains the final guard.
//
// $1 event_id, $2 run_id, $3 thread_id, $4 event_kind, $5 event_type,
// $6 node_name, $7 task_id, $8 checkpoint_id, $9 scope_id, $10 agent_role,
// $11 level, $12 ts, $13 message, $14 payload, $15 source_sequence,
// $16 no_source_sequence.
//
// $16 stores source_sequence as NULL: control-plane authored events (steer
// lifecycle) live OUTSIDE the worker's source_sequence space. Defaulting them
// to the new sequence_number would claim the worker's next stamp under the
// partial unique (run_id, source_sequence) index, and ingest would then DROP
// the worker event arriving with that stamp — one lost worker event (possibly
// the terminal one) per CP append on a live run.
const appendRunEventSQL = `
WITH next AS (
  INSERT INTO control_run_event_sequences AS s (run_id, last_sequence)
  VALUES ($2, COALESCE((SELECT MAX(sequence_number) FROM control_run_events WHERE run_id = $2), 0) + 1)
  ON CONFLICT (run_id) DO UPDATE
    SET last_sequence = GREATEST(
      s.last_sequence,
      COALESCE((SELECT MAX(e.sequence_number) FROM control_run_events e WHERE e.run_id = s.run_id), 0)
    ) + 1
  RETURNING last_sequence
)
INSERT INTO control_run_events (
  event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type,
  node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
)
SELECT $1, next.last_sequence, CASE WHEN $16::boolean THEN NULL ELSE COALESCE($15::bigint, next.last_sequence) END, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
FROM next
RETURNING event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
`

// appendRunEventIfRunActiveSQL is the single-statement ingest path. Outcome
// is encoded in the returned row's "appended" column:
//
//   - (row, appended=true): the event was inserted; the run was live.
//   - (row, appended=false): an event with this ID already exists; the stored
//     row is returned untouched so a redelivered (possibly mutated) duplicate
//     can never win over what was first ingested. Checked BEFORE the live-run
//     gate: duplicates of a terminal event must still be reported as
//     duplicates so their side effects and fanout can be replayed.
//   - no row: no duplicate and the run is missing or terminal; the event is
//     deliberately dropped and the sequence counter is left untouched.
//
// A concurrent insert of the same event ID on another replica can still
// surface as a primary-key ErrConflict; callers handle that by re-reading.
const appendRunEventIfRunActiveSQL = `
WITH live_run AS (
  SELECT run_id FROM control_runs
  WHERE run_id = $2 AND status NOT IN ('succeeded', 'failed', 'canceled')
    AND NOT EXISTS (SELECT 1 FROM control_run_events d WHERE d.event_id = $1)
), next AS (
  INSERT INTO control_run_event_sequences AS s (run_id, last_sequence)
  SELECT live_run.run_id, COALESCE((SELECT MAX(sequence_number) FROM control_run_events WHERE run_id = live_run.run_id), 0) + 1
  FROM live_run
  ON CONFLICT (run_id) DO UPDATE
    SET last_sequence = GREATEST(
      s.last_sequence,
      COALESCE((SELECT MAX(e.sequence_number) FROM control_run_events e WHERE e.run_id = s.run_id), 0)
    ) + 1
  RETURNING last_sequence
), inserted AS (
	  INSERT INTO control_run_events (
	    event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type,
	    node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
	  )
	  SELECT $1, next.last_sequence, CASE WHEN $16::boolean THEN NULL ELSE COALESCE($15::bigint, next.last_sequence) END, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
	  FROM next
	  RETURNING event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
)
SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload, true AS appended
FROM inserted
UNION ALL
SELECT e.event_id, e.sequence_number, e.source_sequence, e.run_id, e.thread_id, e.event_kind, e.event_type, e.node_name, e.task_id, e.checkpoint_id, e.scope_id, e.agent_role, e.level, e.ts, e.message, e.payload, false
FROM control_run_events e
WHERE e.event_id = $1 AND NOT EXISTS (SELECT 1 FROM inserted)
`

// RunEventAppendOutcome reports what a conditional event append did.
type RunEventAppendOutcome int

const (
	// RunEventAppendOutcomeDropped: no duplicate exists and the run is
	// missing or terminal; nothing was written.
	RunEventAppendOutcomeDropped RunEventAppendOutcome = iota
	// RunEventAppendOutcomeAppended: the event was inserted.
	RunEventAppendOutcomeAppended
	// RunEventAppendOutcomeDuplicate: an event with this ID already exists;
	// the stored record is returned and nothing was written.
	RunEventAppendOutcomeDuplicate
)

func (s *PostgresStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	// The sequence allocator makes collisions on (run_id, sequence_number)
	// impossible among writers that go through it; the bounded retry covers
	// writers that bypassed it (e.g. an older binary during a rolling deploy).
	const maxSequenceCollisionRetries = 3
	args := appendRunEventArgs(input)
	var lastErr error
	for attempt := 0; attempt < maxSequenceCollisionRetries; attempt++ {
		var row sqlc.ControlRunEvent
		err := s.pool.QueryRow(ctx, appendRunEventSQL, args...).Scan(runEventRowDestinations(&row)...)
		if err == nil {
			return runEventFromRow(row), nil
		}
		if isRunEventSequenceCollision(err) {
			lastErr = err
			continue
		}
		return domain.RunEventRecord{}, mapPgError(err)
	}
	return domain.RunEventRecord{}, mapPgError(lastErr)
}

// AppendRunEventIfRunActive performs ingest's dedupe-or-append-or-drop in a
// single statement; see appendRunEventIfRunActiveSQL for outcome semantics.
func (s *PostgresStore) AppendRunEventIfRunActive(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, RunEventAppendOutcome, error) {
	const maxSequenceCollisionRetries = 3
	args := appendRunEventArgs(input)
	var lastErr error
	for attempt := 0; attempt < maxSequenceCollisionRetries; attempt++ {
		var row sqlc.ControlRunEvent
		var appended bool
		err := s.pool.QueryRow(ctx, appendRunEventIfRunActiveSQL, args...).
			Scan(append(runEventRowDestinations(&row), &appended)...)
		if err == nil {
			if appended {
				return runEventFromRow(row), RunEventAppendOutcomeAppended, nil
			}
			return runEventFromRow(row), RunEventAppendOutcomeDuplicate, nil
		}
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.RunEventRecord{}, RunEventAppendOutcomeDropped, nil
		}
		if isRunEventSequenceCollision(err) {
			lastErr = err
			continue
		}
		return domain.RunEventRecord{}, RunEventAppendOutcomeDropped, mapPgError(err)
	}
	return domain.RunEventRecord{}, RunEventAppendOutcomeDropped, mapPgError(lastErr)
}

func appendRunEventArgs(input domain.AppendRunEventInput) []any {
	eventID := input.EventID
	if eventID == "" {
		eventID = domain.NewID("event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	return []any{
		eventID,
		input.RunID,
		nullableText(input.ThreadID),
		input.EventKind,
		nullableText(input.EventType),
		nullableText(input.NodeName),
		nullableText(input.TaskID),
		nullableText(input.CheckpointID),
		nullableText(input.ScopeID),
		nullableText(input.AgentRole),
		nullableText(input.Level),
		timestamptz(ts),
		nullableText(input.Message),
		jsonBytes(input.Payload),
		nullableInt8(input.SourceSequence),
		input.NoSourceSequence,
	}
}

func runEventRowDestinations(row *sqlc.ControlRunEvent) []any {
	return []any{
		&row.EventID,
		&row.SequenceNumber,
		&row.SourceSequence,
		&row.RunID,
		&row.ThreadID,
		&row.EventKind,
		&row.EventType,
		&row.NodeName,
		&row.TaskID,
		&row.CheckpointID,
		&row.ScopeID,
		&row.AgentRole,
		&row.Level,
		&row.Ts,
		&row.Message,
		&row.Payload,
	}
}

func isRunEventSequenceCollision(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) &&
		pgErr.Code == "23505" &&
		pgErr.ConstraintName == "control_run_events_run_id_sequence_number_key"
}

// AdminUserMessageStats aggregates per-user thread-message activity for the
// admin dashboard in one grouped query, replacing a ListThreadMessages call
// per thread.
func (s *PostgresStore) AdminUserMessageStats(ctx context.Context, since24h, since7d, since30d time.Time) ([]domain.AdminUserMessageStats, error) {
	rows, err := s.pool.Query(ctx, `
SELECT t.user_id,
  count(*)::int,
  count(*) FILTER (WHERE m.created_at >= $1)::int,
  count(*) FILTER (WHERE m.created_at >= $2)::int,
  count(*) FILTER (WHERE m.created_at >= $3)::int,
  count(*) FILTER (WHERE lower(m.role) = 'user')::int,
  count(*) FILTER (WHERE lower(m.role) = 'user' AND m.created_at >= $1)::int,
  count(*) FILTER (WHERE lower(m.role) = 'user' AND m.created_at >= $2)::int,
  count(*) FILTER (WHERE lower(m.role) = 'user' AND m.created_at >= $3)::int,
  count(*) FILTER (WHERE lower(m.role) = 'assistant')::int,
  count(*) FILTER (WHERE lower(m.role) = 'assistant' AND m.created_at >= $1)::int,
  count(*) FILTER (WHERE lower(m.role) = 'assistant' AND m.created_at >= $2)::int,
  count(*) FILTER (WHERE lower(m.role) = 'assistant' AND m.created_at >= $3)::int
FROM control_thread_messages m
JOIN control_threads t ON t.thread_id = m.thread_id
GROUP BY t.user_id
`, timestamptz(since24h), timestamptz(since7d), timestamptz(since30d))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	stats := []domain.AdminUserMessageStats{}
	for rows.Next() {
		var stat domain.AdminUserMessageStats
		if err := rows.Scan(
			&stat.UserID,
			&stat.Messages.Total, &stat.Messages.Last24h, &stat.Messages.Last7d, &stat.Messages.Last30d,
			&stat.UserMessages.Total, &stat.UserMessages.Last24h, &stat.UserMessages.Last7d, &stat.UserMessages.Last30d,
			&stat.AssistantMessages.Total, &stat.AssistantMessages.Last24h, &stat.AssistantMessages.Last7d, &stat.AssistantMessages.Last30d,
		); err != nil {
			return nil, err
		}
		stats = append(stats, stat)
	}
	return stats, rows.Err()
}

// AdminUserEventStats aggregates per-user tool-call/artifact event activity
// in one grouped query (backed by a partial index on those event kinds),
// replacing a full ListRunEvents scan per run.
func (s *PostgresStore) AdminUserEventStats(ctx context.Context, since24h, since7d, since30d time.Time) ([]domain.AdminUserEventStats, error) {
	rows, err := s.pool.Query(ctx, `
SELECT r.user_id,
  count(*) FILTER (WHERE e.event_kind = 'tool_call.started')::int,
  count(*) FILTER (WHERE e.event_kind = 'tool_call.started' AND e.ts >= $1)::int,
  count(*) FILTER (WHERE e.event_kind = 'tool_call.started' AND e.ts >= $2)::int,
  count(*) FILTER (WHERE e.event_kind = 'tool_call.started' AND e.ts >= $3)::int,
  count(*) FILTER (WHERE e.event_kind = 'artifact.created')::int,
  count(*) FILTER (WHERE e.event_kind = 'artifact.created' AND e.ts >= $1)::int,
  count(*) FILTER (WHERE e.event_kind = 'artifact.created' AND e.ts >= $2)::int,
  count(*) FILTER (WHERE e.event_kind = 'artifact.created' AND e.ts >= $3)::int
FROM control_run_events e
JOIN control_runs r ON r.run_id = e.run_id
WHERE e.event_kind IN ('tool_call.started', 'artifact.created')
GROUP BY r.user_id
`, timestamptz(since24h), timestamptz(since7d), timestamptz(since30d))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	stats := []domain.AdminUserEventStats{}
	for rows.Next() {
		var stat domain.AdminUserEventStats
		if err := rows.Scan(
			&stat.UserID,
			&stat.ToolCalls.Total, &stat.ToolCalls.Last24h, &stat.ToolCalls.Last7d, &stat.ToolCalls.Last30d,
			&stat.Artifacts.Total, &stat.Artifacts.Last24h, &stat.Artifacts.Last7d, &stat.Artifacts.Last30d,
		); err != nil {
			return nil, err
		}
		stats = append(stats, stat)
	}
	return stats, rows.Err()
}

// AdminResourceStats aggregates resource-catalog accounting in the store,
// replacing a 100k-row ListResources scan per admin request.
func (s *PostgresStore) AdminResourceStats(ctx context.Context) (domain.AdminResourceStats, error) {
	stats := domain.AdminResourceStats{}
	err := s.pool.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE status = 'active')::int,
  count(*) FILTER (WHERE status = 'deleted')::int,
  COALESCE(sum(size_bytes) FILTER (WHERE status = 'active'), 0)::bigint
FROM control_resources
`).Scan(&stats.ActiveResources, &stats.SoftDeletedResources, &stats.ActiveBytes)
	if err != nil {
		return domain.AdminResourceStats{}, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT 'user' AS dimension, COALESCE(NULLIF(trim(owner_user_id), ''), 'local-user') AS owner, count(*)::int, COALESCE(sum(size_bytes), 0)::bigint
FROM control_resources WHERE status = 'active' GROUP BY 2
UNION ALL
SELECT 'org', trim(owner_org_id), count(*)::int, COALESCE(sum(size_bytes), 0)::bigint
FROM control_resources WHERE status = 'active' AND trim(COALESCE(owner_org_id, '')) <> '' GROUP BY 2
UNION ALL
SELECT 'project', trim(project_id), count(*)::int, COALESCE(sum(size_bytes), 0)::bigint
FROM control_resources WHERE status = 'active' AND trim(COALESCE(project_id, '')) <> '' GROUP BY 2
`)
	if err != nil {
		return domain.AdminResourceStats{}, err
	}
	defer rows.Close()
	for rows.Next() {
		var dimension string
		var owner domain.AdminResourceOwnerStats
		if err := rows.Scan(&dimension, &owner.Owner, &owner.Uploads, &owner.StorageBytes); err != nil {
			return domain.AdminResourceStats{}, err
		}
		switch dimension {
		case "user":
			stats.Users = append(stats.Users, owner)
		case "org":
			stats.Orgs = append(stats.Orgs, owner)
		case "project":
			stats.Projects = append(stats.Projects, owner)
		}
	}
	return stats, rows.Err()
}

// AdminMetrics computes the value-proving ("platform value") metric set from
// existing run/message/token/artifact data. A "useful run" is a succeeded run
// that produced at least one artifact; user activity counts runs created plus
// user thread-messages, so engagement without a completed run still registers.
// Cost in currency is layered on by the handler from the configured price map.
func (s *PostgresStore) AdminMetrics(ctx context.Context, p domain.AdminMetricsParams) (domain.AdminMetrics, error) {
	rangeStart := timestamptz(p.RangeStart)
	metrics := domain.AdminMetrics{ActivationWindowDays: p.ActivationWindowDays}

	weekly, err := s.adminNorthStarWeekly(ctx, rangeStart)
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.NorthStar.Weekly = weekly

	if err := s.pool.QueryRow(ctx, `SELECT count(*)::int FROM control_users WHERE created_at >= $1`, rangeStart).Scan(&metrics.NewUsers); err != nil {
		return domain.AdminMetrics{}, err
	}

	wauStart := timestamptz(p.Now.Add(-7 * 24 * time.Hour))
	mauStart := timestamptz(p.Now.Add(-28 * 24 * time.Hour))
	if err := s.pool.QueryRow(ctx, `
WITH active AS (
  SELECT user_id, created_at AS ts FROM control_runs WHERE created_at >= $2
  UNION ALL
  SELECT t.user_id, m.created_at FROM control_thread_messages m
    JOIN control_threads t ON t.thread_id = m.thread_id
   WHERE m.created_at >= $2 AND lower(m.role) = 'user'
)
SELECT count(DISTINCT user_id) FILTER (WHERE ts >= $1)::int, count(DISTINCT user_id)::int FROM active
`, wauStart, mauStart).Scan(&metrics.WAU, &metrics.MAU); err != nil {
		return domain.AdminMetrics{}, err
	}

	activationCutoff := timestamptz(p.Now.Add(-time.Duration(p.ActivationWindowDays) * 24 * time.Hour))
	if err := s.pool.QueryRow(ctx, `
WITH cohort AS (
  SELECT user_id, created_at FROM control_users WHERE created_at >= $1 AND created_at < $2
),
useful AS (
  SELECT user_id, min(created_at) AS first_useful
  FROM control_runs r
  WHERE r.status = 'succeeded' AND EXISTS (SELECT 1 FROM control_artifacts a WHERE a.run_id = r.run_id)
  GROUP BY user_id
)
SELECT
  count(*) FILTER (WHERE u.first_useful IS NOT NULL AND u.first_useful <= c.created_at + make_interval(days => $3))::int,
  count(*)::int
FROM cohort c LEFT JOIN useful u ON u.user_id = c.user_id
`, rangeStart, activationCutoff, p.ActivationWindowDays).Scan(&metrics.ActivationActivated, &metrics.ActivationCohort); err != nil {
		return domain.AdminMetrics{}, err
	}

	cohorts, err := s.adminRetentionCohorts(ctx, rangeStart, p.CohortMaxPeriods)
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.RetentionCohorts = cohorts

	curve, err := s.adminPowerUserCurve(ctx, p.Now, p.PowerUserWindowDays, p.PowerUserThreshold)
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.PowerUserCurve = curve

	funnel, err := s.adminActivationFunnel(ctx, rangeStart)
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.Funnel = funnel

	if err := s.pool.QueryRow(ctx, `
SELECT
  count(*) FILTER (WHERE r.status = 'succeeded' AND EXISTS (SELECT 1 FROM control_artifacts a WHERE a.run_id = r.run_id))::int,
  count(*)::int
FROM control_runs r WHERE r.created_at >= $1
`, rangeStart).Scan(&metrics.UsefulRuns, &metrics.TotalRuns); err != nil {
		return domain.AdminMetrics{}, err
	}

	byModel, err := s.adminTokensByModel(ctx, rangeStart)
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.TokensByModel = byModel

	daily, err := s.adminTokensDaily(ctx, timestamptz(p.CostSince))
	if err != nil {
		return domain.AdminMetrics{}, err
	}
	metrics.TokensDaily = daily

	return metrics, nil
}

func (s *PostgresStore) adminNorthStarWeekly(ctx context.Context, rangeStart pgtype.Timestamptz) ([]domain.AdminWeekPoint, error) {
	rows, err := s.pool.Query(ctx, `
SELECT date_trunc('week', r.created_at AT TIME ZONE 'UTC')::date AS wk, count(DISTINCT r.user_id)::int
FROM control_runs r
WHERE r.created_at >= $1
  AND r.status = 'succeeded'
  AND EXISTS (SELECT 1 FROM control_artifacts a WHERE a.run_id = r.run_id)
GROUP BY 1 ORDER BY 1
`, rangeStart)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	points := []domain.AdminWeekPoint{}
	for rows.Next() {
		var pt domain.AdminWeekPoint
		if err := rows.Scan(&pt.WeekStart, &pt.Value); err != nil {
			return nil, err
		}
		points = append(points, pt)
	}
	return points, rows.Err()
}

func (s *PostgresStore) adminRetentionCohorts(ctx context.Context, rangeStart pgtype.Timestamptz, maxPeriods int) ([]domain.AdminRetentionCohort, error) {
	if maxPeriods < 1 {
		maxPeriods = 1
	}
	sizeRows, err := s.pool.Query(ctx, `
SELECT date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS cw, count(*)::int
FROM control_users WHERE created_at >= $1 GROUP BY 1 ORDER BY 1
`, rangeStart)
	if err != nil {
		return nil, err
	}
	defer sizeRows.Close()
	order := []time.Time{}
	cohortByWeek := map[time.Time]*domain.AdminRetentionCohort{}
	for sizeRows.Next() {
		var cw time.Time
		var size int
		if err := sizeRows.Scan(&cw, &size); err != nil {
			return nil, err
		}
		cohort := &domain.AdminRetentionCohort{CohortStart: cw, Size: size, Retained: make([]int, maxPeriods)}
		cohortByWeek[cw] = cohort
		order = append(order, cw)
	}
	if err := sizeRows.Err(); err != nil {
		return nil, err
	}

	retRows, err := s.pool.Query(ctx, `
WITH signup AS (
  SELECT user_id, date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS cw
  FROM control_users WHERE created_at >= $1
),
activity AS (
  SELECT user_id, date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS aw
    FROM control_runs WHERE created_at >= $1
  UNION
  SELECT t.user_id, date_trunc('week', m.created_at AT TIME ZONE 'UTC')::date
    FROM control_thread_messages m JOIN control_threads t ON t.thread_id = m.thread_id
   WHERE m.created_at >= $1 AND lower(m.role) = 'user'
)
SELECT s.cw, ((a.aw - s.cw) / 7)::int AS period, count(DISTINCT s.user_id)::int
FROM signup s JOIN activity a ON a.user_id = s.user_id AND a.aw >= s.cw
GROUP BY 1, 2 ORDER BY 1, 2
`, rangeStart)
	if err != nil {
		return nil, err
	}
	defer retRows.Close()
	for retRows.Next() {
		var cw time.Time
		var period, retained int
		if err := retRows.Scan(&cw, &period, &retained); err != nil {
			return nil, err
		}
		cohort, ok := cohortByWeek[cw]
		if !ok || period < 0 || period >= maxPeriods {
			continue
		}
		cohort.Retained[period] = retained
	}
	if err := retRows.Err(); err != nil {
		return nil, err
	}

	cohorts := make([]domain.AdminRetentionCohort, 0, len(order))
	for _, cw := range order {
		cohorts = append(cohorts, *cohortByWeek[cw])
	}
	return cohorts, nil
}

func (s *PostgresStore) adminPowerUserCurve(ctx context.Context, now time.Time, windowDays, threshold int) (domain.AdminPowerUserCurve, error) {
	if windowDays < 1 {
		windowDays = 28
	}
	curve := domain.AdminPowerUserCurve{WindowDays: windowDays, PowerUserThreshold: threshold}
	since := timestamptz(now.Add(-time.Duration(windowDays) * 24 * time.Hour))
	rows, err := s.pool.Query(ctx, `
WITH active_days AS (
  SELECT user_id, count(DISTINCT d)::int AS days FROM (
    SELECT user_id, date_trunc('day', created_at AT TIME ZONE 'UTC')::date AS d
      FROM control_runs WHERE created_at >= $1
    UNION
    SELECT t.user_id, date_trunc('day', m.created_at AT TIME ZONE 'UTC')::date
      FROM control_thread_messages m JOIN control_threads t ON t.thread_id = m.thread_id
     WHERE m.created_at >= $1 AND lower(m.role) = 'user'
  ) x GROUP BY user_id
)
SELECT days, count(*)::int FROM active_days GROUP BY days ORDER BY days
`, since)
	if err != nil {
		return domain.AdminPowerUserCurve{}, err
	}
	defer rows.Close()
	counts := make(map[int]int, windowDays)
	for rows.Next() {
		var days, users int
		if err := rows.Scan(&days, &users); err != nil {
			return domain.AdminPowerUserCurve{}, err
		}
		counts[days] = users
	}
	if err := rows.Err(); err != nil {
		return domain.AdminPowerUserCurve{}, err
	}
	curve.Buckets = make([]domain.AdminPowerUserBucket, 0, windowDays)
	for d := 1; d <= windowDays; d++ {
		curve.Buckets = append(curve.Buckets, domain.AdminPowerUserBucket{DaysActive: d, Users: counts[d]})
	}
	return curve, nil
}

func (s *PostgresStore) adminActivationFunnel(ctx context.Context, rangeStart pgtype.Timestamptz) ([]domain.AdminFunnelStage, error) {
	var signedUp, startedRun, producedOutput, returned int
	err := s.pool.QueryRow(ctx, `
WITH cohort AS (
  SELECT user_id, date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS cw
  FROM control_users WHERE created_at >= $1
),
runs AS (SELECT DISTINCT user_id FROM control_runs WHERE created_at >= $1),
useful AS (
  SELECT DISTINCT user_id FROM control_runs r
  WHERE r.created_at >= $1 AND r.status = 'succeeded'
    AND EXISTS (SELECT 1 FROM control_artifacts a WHERE a.run_id = r.run_id)
),
activity AS (
  SELECT user_id, date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS aw
    FROM control_runs WHERE created_at >= $1
  UNION
  SELECT t.user_id, date_trunc('week', m.created_at AT TIME ZONE 'UTC')::date
    FROM control_thread_messages m JOIN control_threads t ON t.thread_id = m.thread_id
   WHERE m.created_at >= $1 AND lower(m.role) = 'user'
),
returned AS (
  SELECT DISTINCT s.user_id FROM cohort s
  JOIN activity a ON a.user_id = s.user_id AND a.aw > s.cw
)
SELECT
  (SELECT count(*) FROM cohort)::int,
  (SELECT count(*) FROM cohort c WHERE c.user_id IN (SELECT user_id FROM runs))::int,
  (SELECT count(*) FROM cohort c WHERE c.user_id IN (SELECT user_id FROM useful))::int,
  (SELECT count(*) FROM returned)::int
`, rangeStart).Scan(&signedUp, &startedRun, &producedOutput, &returned)
	if err != nil {
		return nil, err
	}
	return []domain.AdminFunnelStage{
		{Stage: "Signed up", Users: signedUp},
		{Stage: "Started a run", Users: startedRun},
		{Stage: "Produced an output", Users: producedOutput},
		{Stage: "Returned and ran again", Users: returned},
	}, nil
}

func (s *PostgresStore) adminTokensByModel(ctx context.Context, rangeStart pgtype.Timestamptz) ([]domain.AdminModelTokens, error) {
	rows, err := s.pool.Query(ctx, `
SELECT model,
  COALESCE(sum(input_tokens), 0)::bigint,
  COALESCE(sum(output_tokens), 0)::bigint,
  COALESCE(sum(total_tokens), 0)::bigint,
  count(DISTINCT run_id)::int
FROM control_run_token_usage WHERE occurred_at >= $1 GROUP BY model ORDER BY 4 DESC
`, rangeStart)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.AdminModelTokens{}
	for rows.Next() {
		var m domain.AdminModelTokens
		if err := rows.Scan(&m.Model, &m.InputTokens, &m.OutputTokens, &m.TotalTokens, &m.Runs); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

func (s *PostgresStore) adminTokensDaily(ctx context.Context, since pgtype.Timestamptz) ([]domain.AdminDayModelTokens, error) {
	rows, err := s.pool.Query(ctx, `
SELECT day, model,
  COALESCE(sum(input_tokens), 0)::bigint,
  COALESCE(sum(output_tokens), 0)::bigint,
  COALESCE(sum(total_tokens), 0)::bigint
FROM control_run_token_usage WHERE day >= $1::date GROUP BY day, model ORDER BY day
`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.AdminDayModelTokens{}
	for rows.Next() {
		var d domain.AdminDayModelTokens
		if err := rows.Scan(&d.Day, &d.Model, &d.InputTokens, &d.OutputTokens, &d.TotalTokens); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *PostgresStore) GetRunEvent(ctx context.Context, eventID string) (domain.RunEventRecord, bool, error) {
	if eventID == "" {
		return domain.RunEventRecord{}, false, nil
	}
	row, err := s.queries.GetRunEvent(ctx, eventID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.RunEventRecord{}, false, nil
		}
		return domain.RunEventRecord{}, false, mapPgError(err)
	}
	return runEventFromRow(row), true, nil
}

func (s *PostgresStore) GetRunEventBySourceSequence(ctx context.Context, runID string, sourceSequence int64) (domain.RunEventRecord, bool, error) {
	if strings.TrimSpace(runID) == "" || sourceSequence <= 0 {
		return domain.RunEventRecord{}, false, nil
	}
	row := sqlc.ControlRunEvent{}
	err := s.pool.QueryRow(ctx, `
SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type,
       node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
FROM control_run_events
WHERE run_id = $1 AND source_sequence = $2
`, runID, sourceSequence).Scan(runEventRowDestinations(&row)...)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.RunEventRecord{}, false, nil
		}
		return domain.RunEventRecord{}, false, mapPgError(err)
	}
	return runEventFromRow(row), true, nil
}

const listRunEventsSQL = `
SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
FROM (
  SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload FROM control_run_events WHERE run_id = $1 ORDER BY sequence_number DESC LIMIT $2
) recent_events
ORDER BY sequence_number ASC
`

const listRunEventsForUserSQL = `
SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
FROM (
  SELECT e.event_id, e.sequence_number, e.source_sequence, e.run_id, e.thread_id, e.event_kind, e.event_type, e.node_name, e.task_id, e.checkpoint_id, e.scope_id, e.agent_role, e.level, e.ts, e.message, e.payload
  FROM control_run_events e
  JOIN control_runs r ON r.run_id = e.run_id
  WHERE e.run_id = $1
    AND r.user_id = $2
  ORDER BY e.sequence_number DESC
  LIMIT $3
) recent_events
ORDER BY sequence_number ASC
`

const listRunEventsAfterSQL = `
SELECT event_id, sequence_number, source_sequence, run_id, thread_id, event_kind, event_type, node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
FROM control_run_events
WHERE run_id = $1 AND sequence_number > $2
ORDER BY sequence_number ASC
LIMIT $3
`

const listRunEventsAfterForUserSQL = `
SELECT e.event_id, e.sequence_number, e.source_sequence, e.run_id, e.thread_id, e.event_kind, e.event_type, e.node_name, e.task_id, e.checkpoint_id, e.scope_id, e.agent_role, e.level, e.ts, e.message, e.payload
FROM control_run_events e
JOIN control_runs r ON r.run_id = e.run_id
WHERE e.run_id = $1
  AND r.user_id = $2
  AND e.sequence_number > $3
ORDER BY e.sequence_number ASC
LIMIT $4
`

// PruneRunEventDeltas deletes ephemeral token-delta events (the given event kinds) for runs that
// reached a TERMINAL status and completed before olderThan, up to limit rows per call. It NEVER
// touches an active run (completed_at IS NULL guards that even if a status were ever mislabeled),
// and only ever removes the listed delta kinds — structural, terminal, and token-usage events are
// preserved for the trace/observability. The durable answer already lives in control_thread_messages,
// so dropping the per-token deltas is lossless for the conversation. Batched (LIMIT) to keep the
// delete bounded and avoid long locks on a large table.
func (s *PostgresStore) PruneRunEventDeltas(ctx context.Context, olderThan time.Time, kinds []string, limit int) (int64, error) {
	if len(kinds) == 0 || limit <= 0 {
		return 0, nil
	}
	tag, err := s.pool.Exec(ctx, `
		DELETE FROM control_run_events
		WHERE event_id IN (
			SELECT e.event_id
			FROM control_run_events e
			JOIN control_runs r ON r.run_id = e.run_id
			WHERE r.status IN ('succeeded', 'failed', 'canceled')
			  AND r.completed_at IS NOT NULL
			  AND r.completed_at < $1
			  AND e.event_kind = ANY($2)
			LIMIT $3
		)`,
		timestamptz(olderThan), kinds, limit)
	if err != nil {
		return 0, mapPgError(err)
	}
	return tag.RowsAffected(), nil
}

func (s *PostgresStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	resolvedLimit := limit32(limit, 500)
	rows, err := s.pool.Query(ctx, listRunEventsSQL, runID, resolvedLimit)
	if err != nil {
		return nil, err
	}
	return scanRunEventRows(rows, runEventListCapacity(resolvedLimit, true))
}

func (s *PostgresStore) ListRunEventsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.RunEventRecord, error) {
	resolvedLimit := limit32(limit, 500)
	rows, err := s.pool.Query(ctx, listRunEventsForUserSQL, runID, userID, resolvedLimit)
	if err != nil {
		return nil, err
	}
	events, err := scanRunEventRows(rows, runEventListCapacity(resolvedLimit, true))
	if err != nil {
		return nil, err
	}
	if len(events) == 0 {
		if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
			return nil, err
		}
	}
	return events, nil
}

func (s *PostgresStore) ListRunEventsAfter(ctx context.Context, runID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	resolvedLimit := limit32(limit, 500)
	if afterSequence > 0 {
		rows, err := s.queries.ListRunEventsAfter(ctx, sqlc.ListRunEventsAfterParams{
			RunID:          runID,
			SequenceNumber: afterSequence,
			Limit:          resolvedLimit,
		})
		if err != nil {
			return nil, err
		}
		return runEventsFromRows(rows), nil
	}
	rows, err := s.pool.Query(ctx, listRunEventsAfterSQL, runID, afterSequence, resolvedLimit)
	if err != nil {
		return nil, err
	}
	return scanRunEventRows(rows, runEventListCapacity(resolvedLimit, true))
}

func (s *PostgresStore) ListRunEventsAfterForUser(ctx context.Context, runID string, userID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	resolvedLimit := limit32(limit, 500)
	if afterSequence > 0 {
		rows, err := s.queries.ListRunEventsAfterForUser(ctx, sqlc.ListRunEventsAfterForUserParams{
			RunID:          runID,
			UserID:         userID,
			SequenceNumber: afterSequence,
			Limit:          resolvedLimit,
		})
		if err != nil {
			return nil, err
		}
		if len(rows) == 0 {
			if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
				return nil, err
			}
		}
		return runEventsFromRows(rows), nil
	}
	rows, err := s.pool.Query(ctx, listRunEventsAfterForUserSQL, runID, userID, afterSequence, resolvedLimit)
	if err != nil {
		return nil, err
	}
	events, err := scanRunEventRows(rows, runEventListCapacity(resolvedLimit, afterSequence <= 0))
	if err != nil {
		return nil, err
	}
	if len(events) == 0 {
		if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
			return nil, err
		}
	}
	return events, nil
}

func scanRunEventRows(rows pgx.Rows, capacity int) ([]domain.RunEventRecord, error) {
	if capacity < 0 {
		capacity = 0
	}
	events := make([]domain.RunEventRecord, 0, capacity)
	var event domain.RunEventRecord
	var threadID pgtype.Text
	var eventType pgtype.Text
	var nodeName pgtype.Text
	var taskID pgtype.Text
	var checkpointID pgtype.Text
	var scopeID pgtype.Text
	var agentRole pgtype.Text
	var level pgtype.Text
	var ts pgtype.Timestamptz
	var message pgtype.Text
	var payload []byte
	scanTargets := []any{
		&event.EventID,
		&event.Sequence,
		&event.SourceSequence,
		&event.RunID,
		&threadID,
		&event.EventKind,
		&eventType,
		&nodeName,
		&taskID,
		&checkpointID,
		&scopeID,
		&agentRole,
		&level,
		&ts,
		&message,
		&payload,
	}
	_, err := pgx.ForEachRow(rows, scanTargets, func() error {
		event.ThreadID = textValue(threadID)
		event.EventType = textValue(eventType)
		event.NodeName = textValue(nodeName)
		event.TaskID = textValue(taskID)
		event.CheckpointID = textValue(checkpointID)
		event.ScopeID = textValue(scopeID)
		event.AgentRole = textValue(agentRole)
		event.Level = textValue(level)
		event.TS = timeValue(ts)
		event.Message = textValue(message)
		event.Payload = jsonMap(payload)
		events = append(events, event)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return events, nil
}

func runEventListCapacity(limit int32, preallocate bool) int {
	if !preallocate || limit <= 0 {
		return 0
	}
	const maxPreallocatedRunEvents = 500
	if limit > maxPreallocatedRunEvents {
		return maxPreallocatedRunEvents
	}
	return int(limit)
}

func runEventsFromRows(rows []sqlc.ControlRunEvent) []domain.RunEventRecord {
	events := make([]domain.RunEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, runEventFromRow(row))
	}
	return events
}

func (s *PostgresStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	artifactID := input.ArtifactID
	if artifactID == "" {
		artifactID = domain.NewID("artifact")
	}
	if input.ArtifactID != "" {
		existing, err := s.GetArtifact(ctx, input.ArtifactID)
		if err == nil {
			return existing, nil
		}
		if !errors.Is(err, ErrNotFound) {
			return domain.ArtifactRecord{}, err
		}
	}
	now := domain.Now()
	row, err := s.queries.CreateArtifact(ctx, sqlc.CreateArtifactParams{
		ArtifactID:    artifactID,
		RunID:         input.RunID,
		ThreadID:      nullableText(input.ThreadID),
		Kind:          input.Kind,
		Path:          nullableText(input.Path),
		SourcePath:    nullableText(input.SourcePath),
		PreviewPath:   nullableText(input.PreviewPath),
		Title:         nullableText(input.Title),
		ResultGroupID: nullableText(input.ResultGroupID),
		MimeType:      nullableText(input.MimeType),
		SizeBytes:     nullableInt8(input.SizeBytes),
		Sha256:        nullableText(input.SHA256),
		StorageUri:    nullableText(input.StorageURI),
		ToolName:      nullableText(input.ToolName),
		Category:      nullableText(input.Category),
		CreatedAt:     timestamptz(now),
		UpdatedAt:     timestamptz(now),
		Metadata:      jsonBytes(input.Metadata),
	})
	if err != nil {
		if input.ArtifactID != "" {
			if existing, getErr := s.GetArtifact(ctx, input.ArtifactID); getErr == nil {
				return existing, nil
			}
		}
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func (s *PostgresStore) ListRunArtifacts(ctx context.Context, runID string, limit int) ([]domain.ArtifactRecord, error) {
	rows, err := s.queries.ListRunArtifacts(ctx, sqlc.ListRunArtifactsParams{
		RunID: runID,
		Limit: limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	artifacts := make([]domain.ArtifactRecord, 0, len(rows))
	for _, row := range rows {
		artifacts = append(artifacts, artifactFromRow(row))
	}
	return artifacts, nil
}

func (s *PostgresStore) ListRunArtifactsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.ArtifactRecord, error) {
	if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
		return nil, err
	}
	rows, err := s.queries.ListRunArtifactsForUser(ctx, sqlc.ListRunArtifactsForUserParams{
		RunID:  runID,
		UserID: userID,
		Limit:  limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	artifacts := make([]domain.ArtifactRecord, 0, len(rows))
	for _, row := range rows {
		artifacts = append(artifacts, artifactFromRow(row))
	}
	return artifacts, nil
}

func (s *PostgresStore) GetArtifact(ctx context.Context, artifactID string) (domain.ArtifactRecord, error) {
	row, err := s.queries.GetArtifact(ctx, artifactID)
	if err != nil {
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func (s *PostgresStore) GetArtifactForUser(ctx context.Context, artifactID string, userID string) (domain.ArtifactRecord, error) {
	row, err := s.queries.GetArtifactForUser(ctx, sqlc.GetArtifactForUserParams{
		ArtifactID: artifactID,
		UserID:     userID,
	})
	if err != nil {
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func (s *PostgresStore) CreateUploadSession(ctx context.Context, input domain.CreateUploadSessionInput) (domain.UploadSessionRecord, error) {
	sessionID := strings.TrimSpace(input.SessionID)
	if sessionID == "" {
		sessionID = domain.NewID("upload_session")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	sourceType := strings.TrimSpace(input.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	row, err := s.queries.CreateUploadSession(ctx, sqlc.CreateUploadSessionParams{
		SessionID:          sessionID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         nullableText(input.OwnerOrgID),
		OwnerRole:          nullableText(input.OwnerRole),
		ProjectID:          nullableText(input.ProjectID),
		SourceType:         sourceType,
		Status:             status,
		TotalBytes:         input.TotalBytes,
		BytesReceived:      input.BytesReceived,
		BytesVerified:      input.BytesVerified,
		BytesCommitted:     input.BytesCommitted,
		IdempotencyKey:     nullableText(input.IdempotencyKey),
		BrowserFingerprint: nullableText(input.BrowserFingerprint),
		Error:              nullableText(input.Error),
		CreatedAt:          timestamptz(createdAt),
		UpdatedAt:          timestamptz(updatedAt),
		CompletedAt:        nullableTimestamptz(input.CompletedAt),
		Metadata:           jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.UploadSessionRecord{}, mapPgError(err)
	}
	return uploadSessionFromRow(row), nil
}

func (s *PostgresStore) GetUploadSessionForUser(ctx context.Context, sessionID string, userID string, orgID string) (domain.UploadSessionRecord, error) {
	row, err := s.queries.GetUploadSessionForUser(ctx, sqlc.GetUploadSessionForUserParams{
		SessionID:   strings.TrimSpace(sessionID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
	})
	if err != nil {
		return domain.UploadSessionRecord{}, mapPgError(err)
	}
	return uploadSessionFromRow(row), nil
}

func (s *PostgresStore) GetUploadSessionByIdempotencyKeyForUser(ctx context.Context, idempotencyKey string, userID string, orgID string) (domain.UploadSessionRecord, error) {
	row, err := s.queries.GetUploadSessionByIdempotencyKey(ctx, sqlc.GetUploadSessionByIdempotencyKeyParams{
		OwnerUserID:    strings.TrimSpace(userID),
		Column2:        strings.TrimSpace(orgID),
		IdempotencyKey: nullableText(idempotencyKey),
	})
	if err != nil {
		return domain.UploadSessionRecord{}, mapPgError(err)
	}
	return uploadSessionFromRow(row), nil
}

// RetentionBacklog reports reclaimable and operator-blocked retention storage.
func (s *PostgresStore) RetentionBacklog(ctx context.Context, now time.Time) (domain.ResourceRetentionBacklog, error) {
	var backlog domain.ResourceRetentionBacklog
	err := s.pool.QueryRow(ctx,
		`SELECT
		   sum(expired_resources), sum(reclaimable_bytes),
		   sum(blocked_resources), sum(blocked_bytes),
		   sum(purging_resources), sum(purging_bytes)
		 FROM (
		   SELECT count(*) AS expired_resources, COALESCE(sum(size_bytes), 0) AS reclaimable_bytes,
		     0::bigint AS blocked_resources, 0::bigint AS blocked_bytes,
		     0::bigint AS purging_resources, 0::bigint AS purging_bytes
		   FROM control_resources
		   WHERE status = 'deleted' AND retention_expires_at IS NOT NULL AND retention_expires_at < $1
		   UNION ALL
		   SELECT 0, 0, count(*), COALESCE(sum(size_bytes), 0), 0, 0
		   FROM control_resources WHERE status = 'retention_blocked'
		   UNION ALL
		   SELECT 0, 0, 0, 0, count(*), COALESCE(sum(size_bytes), 0)
		   FROM control_resources WHERE status = 'purging'
		 ) AS retention_states`,
		now.UTC(),
	).Scan(
		&backlog.Count,
		&backlog.Bytes,
		&backlog.BlockedCount,
		&backlog.BlockedBytes,
		&backlog.PurgingCount,
		&backlog.PurgingBytes,
	)
	if err != nil {
		return domain.ResourceRetentionBacklog{}, mapPgError(err)
	}
	return backlog, nil
}

// ListResourcesPastRetention returns soft-deleted resources whose undelete window has
// elapsed, oldest first, up to limit. Only the fields a reclaim needs are populated.
func (s *PostgresStore) ListResourcesPastRetention(ctx context.Context, now time.Time, limit int) ([]domain.ResourceRecord, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx,
		`SELECT resource_id, COALESCE(storage_uri, ''), COALESCE(storage_path, ''), COALESCE(original_name, ''), size_bytes
		 FROM control_resources
		 WHERE status = 'deleted' AND retention_expires_at IS NOT NULL AND retention_expires_at < $1
		 ORDER BY retention_expires_at ASC LIMIT $2`,
		now.UTC(), limit,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	out := []domain.ResourceRecord{}
	for rows.Next() {
		var r domain.ResourceRecord
		if err := rows.Scan(&r.ResourceID, &r.StorageURI, &r.StoragePath, &r.OriginalName, &r.SizeBytes); err != nil {
			return nil, mapPgError(err)
		}
		r.Status = "deleted"
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return out, nil
}

// ClaimResourcesPastRetention atomically leases eligible rows to one GC
// replica. PostgreSQL's clock is authoritative for both retention expiry and
// stale-lease recovery, avoiding application-host clock skew.
func (s *PostgresStore) ClaimResourcesPastRetention(ctx context.Context, lease time.Duration, limit int) ([]domain.ResourceRecord, error) {
	if lease <= 0 {
		lease = 15 * time.Minute
	}
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		WITH deleted_candidates AS MATERIALIZED (
			SELECT resource_id,
				COALESCE(storage_uri, '') AS storage_uri,
				COALESCE(storage_path, '') AS storage_path,
				status,
				retention_expires_at,
				updated_at
			FROM control_resources
			WHERE status = 'deleted'
				AND retention_expires_at IS NOT NULL
				AND retention_expires_at < statement_timestamp()
			ORDER BY CASE
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND btrim(COALESCE(storage_uri, '')) = ''
						AND btrim(COALESCE(storage_path, '')) <> '' THEN 0
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND lower(btrim(COALESCE(storage_uri, ''))) LIKE 'file://%' THEN 1
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..') THEN 2
					ELSE 3
				END,
				retention_expires_at ASC,
				resource_id ASC
			FOR UPDATE SKIP LOCKED
			LIMIT $2
		), stale_purging_candidates AS MATERIALIZED (
			SELECT resource_id,
				COALESCE(storage_uri, '') AS storage_uri,
				COALESCE(storage_path, '') AS storage_path,
				status,
				retention_expires_at,
				updated_at
			FROM control_resources
			WHERE status = 'purging'
				AND updated_at < statement_timestamp() - make_interval(secs => $1::double precision)
			ORDER BY CASE
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND btrim(COALESCE(storage_uri, '')) = ''
						AND btrim(COALESCE(storage_path, '')) <> '' THEN 0
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND lower(btrim(COALESCE(storage_uri, ''))) LIKE 'file://%' THEN 1
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..') THEN 2
					ELSE 3
				END,
				updated_at ASC,
				resource_id ASC
			FOR UPDATE SKIP LOCKED
			LIMIT $2
		), bounded_candidates AS (
			SELECT * FROM deleted_candidates
			UNION ALL
			SELECT * FROM stale_purging_candidates
		), candidates AS (
			SELECT resource_id
			FROM bounded_candidates
			ORDER BY CASE
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND btrim(storage_uri) = ''
						AND btrim(storage_path) <> '' THEN 0
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..')
						AND lower(btrim(storage_uri)) LIKE 'file://%' THEN 1
					WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$'
						AND resource_id NOT IN ('.', '..') THEN 2
					ELSE 3
				END,
				CASE WHEN status = 'deleted' THEN 0 ELSE 1 END,
				retention_expires_at ASC NULLS LAST,
				updated_at ASC,
				resource_id ASC
			LIMIT $2
		)
		UPDATE control_resources AS resource
		SET status = 'purging', updated_at = clock_timestamp()
		FROM candidates
		WHERE resource.resource_id = candidates.resource_id
		RETURNING resource.resource_id,
			COALESCE(resource.storage_uri, ''),
			COALESCE(resource.storage_path, ''),
			COALESCE(resource.original_name, ''),
			resource.size_bytes,
			resource.updated_at`, lease.Seconds(), limit)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	claimed := make([]domain.ResourceRecord, 0, limit)
	for rows.Next() {
		var resource domain.ResourceRecord
		if err := rows.Scan(
			&resource.ResourceID,
			&resource.StorageURI,
			&resource.StoragePath,
			&resource.OriginalName,
			&resource.SizeBytes,
			&resource.UpdatedAt,
		); err != nil {
			return nil, mapPgError(err)
		}
		resource.Status = "purging"
		claimed = append(claimed, resource)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return claimed, nil
}

// RenewResourceRetentionClaim extends an exact live claim and returns its new
// timestamp token. A worker must use the returned token for every later
// renewal, release, block, or purge operation.
func (s *PostgresStore) RenewResourceRetentionClaim(ctx context.Context, resourceID string, claimedAt time.Time) (time.Time, bool, error) {
	var renewedAt time.Time
	err := s.pool.QueryRow(ctx, `
		UPDATE control_resources
		SET updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
		WHERE resource_id = $1 AND status = 'purging' AND updated_at = $2
		RETURNING updated_at`, resourceID, claimedAt.UTC()).Scan(&renewedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, mapPgError(err)
	}
	return renewedAt, true, nil
}

// ReleaseResourceRetentionClaim makes an exact failed claim immediately
// reclaimable while preserving the terminal purging state. A stale takeover
// must never regress a resource to deleted after another replica may already
// have removed bytes or published the permanent filesystem tombstone.
func (s *PostgresStore) ReleaseResourceRetentionClaim(ctx context.Context, resourceID string, claimedAt time.Time) (bool, error) {
	result, err := s.pool.Exec(ctx, `
		UPDATE control_resources
		SET updated_at = to_timestamp(0)
		WHERE resource_id = $1 AND status = 'purging' AND updated_at = $2`,
		resourceID, claimedAt.UTC())
	if err != nil {
		return false, mapPgError(err)
	}
	return result.RowsAffected() == 1, nil
}

// BlockResourceRetentionClaim removes an exact terminal claim from automatic
// retry while preserving an operator-visible, non-reactivatable tombstone.
func (s *PostgresStore) BlockResourceRetentionClaim(ctx context.Context, resourceID string, claimedAt time.Time) (bool, error) {
	result, err := s.pool.Exec(ctx, `
		UPDATE control_resources
		SET status = 'retention_blocked', updated_at = clock_timestamp()
		WHERE resource_id = $1 AND status = 'purging' AND updated_at = $2`,
		resourceID, claimedAt.UTC())
	if err != nil {
		return false, mapPgError(err)
	}
	return result.RowsAffected() == 1, nil
}

// PurgeClaimedResource permanently deletes only the exact claim held by the
// caller. Filesystem cleanup must finish before this method is called.
func (s *PostgresStore) PurgeClaimedResource(ctx context.Context, resourceID string, claimedAt time.Time) (bool, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return false, err
	}
	defer tx.Rollback(ctx)
	if err := lockResourceLifecycleTx(ctx, tx, resourceID); err != nil {
		return false, mapPgError(err)
	}
	var purged bool
	err = tx.QueryRow(ctx, `
WITH deleted AS (
  DELETE FROM control_resources
  WHERE resource_id = $1 AND status = 'purging' AND updated_at = $2
  RETURNING resource_id
), tombstoned AS (
  INSERT INTO control_resource_purge_tombstones (resource_id, purged_at)
  SELECT resource_id, clock_timestamp() FROM deleted
  ON CONFLICT (resource_id) DO UPDATE SET purged_at = EXCLUDED.purged_at
  RETURNING resource_id
)
SELECT EXISTS (SELECT 1 FROM tombstoned)`, resourceID, claimedAt.UTC()).Scan(&purged)
	if err != nil {
		return false, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return false, err
	}
	return purged, nil
}

// resourceUsageWhere aggregates active-resource count + bytes for one ownership column.
// A server-side aggregate so a quota check costs one indexed scan instead of shipping the
// whole catalog to the app and capping it.
func (s *PostgresStore) resourceUsageWhere(ctx context.Context, column, value string) (int, int64, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, 0, nil
	}
	var count int64
	var bytes int64
	// column is a fixed identifier chosen by the typed callers below, never user input.
	query := `SELECT count(*), COALESCE(sum(size_bytes), 0) FROM control_resources WHERE status = 'active' AND ` + column + ` = $1`
	if err := s.pool.QueryRow(ctx, query, value).Scan(&count, &bytes); err != nil {
		return 0, 0, mapPgError(err)
	}
	return int(count), bytes, nil
}

func (s *PostgresStore) ResourceUsageForOwner(ctx context.Context, userID string) (int, int64, error) {
	return s.resourceUsageWhere(ctx, "owner_user_id", userID)
}

func (s *PostgresStore) ResourceUsageForOrg(ctx context.Context, orgID string) (int, int64, error) {
	return s.resourceUsageWhere(ctx, "owner_org_id", orgID)
}

func (s *PostgresStore) ResourceUsageForProject(ctx context.Context, projectID string) (int, int64, error) {
	return s.resourceUsageWhere(ctx, "project_id", projectID)
}

// ClearUploadSessionIdempotencyKey nulls a session's idempotency_key so a re-upload of
// the same content can claim a fresh session (the partial unique index excludes empty
// keys). A no-op if the session does not exist.
func (s *PostgresStore) ClearUploadSessionIdempotencyKey(ctx context.Context, sessionID string) error {
	_, err := s.pool.Exec(ctx,
		`UPDATE control_upload_sessions SET idempotency_key = NULL, updated_at = now() WHERE session_id = $1`,
		strings.TrimSpace(sessionID),
	)
	if err != nil {
		return mapPgError(err)
	}
	return nil
}

func (s *PostgresStore) UpdateUploadSession(ctx context.Context, input domain.UpdateUploadSessionInput) (domain.UploadSessionRecord, error) {
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	row, err := s.queries.UpdateUploadSession(ctx, sqlc.UpdateUploadSessionParams{
		SessionID:      strings.TrimSpace(input.SessionID),
		OwnerUserID:    strings.TrimSpace(input.OwnerUserID),
		OwnerOrgID:     nullableText(input.OwnerOrgID),
		Status:         strings.TrimSpace(input.Status),
		BytesReceived:  input.BytesReceived,
		BytesVerified:  input.BytesVerified,
		BytesCommitted: input.BytesCommitted,
		Error:          nullableText(input.Error),
		UpdatedAt:      timestamptz(updatedAt),
		CompletedAt:    nullableTimestamptz(input.CompletedAt),
		Metadata:       jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.UploadSessionRecord{}, mapPgError(err)
	}
	return uploadSessionFromRow(row), nil
}

func (s *PostgresStore) UpsertUploadSessionFile(ctx context.Context, input domain.UpsertUploadSessionFileInput) (domain.UploadSessionFileRecord, error) {
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.UploadSessionFileRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	var previousStatus string
	var previousSize int64
	err = tx.QueryRow(ctx, `
SELECT status, size_bytes
FROM control_upload_session_files
WHERE session_id = $1 AND file_token = $2
FOR UPDATE`,
		strings.TrimSpace(input.SessionID),
		strings.TrimSpace(input.FileToken),
	).Scan(&previousStatus, &previousSize)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return domain.UploadSessionFileRecord{}, mapPgError(err)
	}
	row, err := s.queries.WithTx(tx).UpsertUploadSessionFile(ctx, sqlc.UpsertUploadSessionFileParams{
		SessionID:      strings.TrimSpace(input.SessionID),
		FileToken:      strings.TrimSpace(input.FileToken),
		ResourceID:     nullableText(input.ResourceID),
		OriginalName:   strings.TrimSpace(input.OriginalName),
		RelativePath:   nullableText(input.RelativePath),
		ContentType:    nullableText(input.ContentType),
		SizeBytes:      input.SizeBytes,
		DeclaredSha256: nullableText(input.DeclaredSHA256),
		ComputedSha256: nullableText(input.ComputedSHA256),
		Status:         status,
		Error:          nullableText(input.Error),
		CreatedAt:      timestamptz(createdAt),
		UpdatedAt:      timestamptz(updatedAt),
		CompletedAt:    nullableTimestamptz(input.CompletedAt),
		Metadata:       jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.UploadSessionFileRecord{}, mapPgError(err)
	}
	file := uploadSessionFileFromRow(row)
	previousCommitted := uploadSessionFileCommittedContribution(previousStatus, previousSize)
	nextCommitted := uploadSessionFileCommittedContribution(file.Status, file.SizeBytes)
	if delta := nextCommitted - previousCommitted; delta != 0 {
		if _, err := tx.Exec(ctx, `
UPDATE control_upload_sessions
SET bytes_committed = GREATEST(0, bytes_committed + $2),
    updated_at = GREATEST(updated_at, $3)
WHERE session_id = $1`,
			file.SessionID,
			delta,
			timestamptz(updatedAt),
		); err != nil {
			return domain.UploadSessionFileRecord{}, mapPgError(err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.UploadSessionFileRecord{}, mapPgError(err)
	}
	return file, nil
}

const postgresUpsertUploadSessionFileSQL = `
INSERT INTO control_upload_session_files (
  session_id, file_token, resource_id, original_name, relative_path, content_type,
  size_bytes, declared_sha256, computed_sha256, status, error, created_at, updated_at, completed_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
ON CONFLICT (session_id, file_token) DO UPDATE SET
  resource_id = EXCLUDED.resource_id,
  original_name = EXCLUDED.original_name,
  relative_path = EXCLUDED.relative_path,
  content_type = EXCLUDED.content_type,
  size_bytes = EXCLUDED.size_bytes,
  declared_sha256 = EXCLUDED.declared_sha256,
  computed_sha256 = EXCLUDED.computed_sha256,
  status = EXCLUDED.status,
  error = EXCLUDED.error,
  updated_at = EXCLUDED.updated_at,
  completed_at = EXCLUDED.completed_at,
  metadata = EXCLUDED.metadata
RETURNING session_id, file_token, resource_id, original_name, relative_path, content_type,
          size_bytes, declared_sha256, computed_sha256, status, error, created_at, updated_at,
          completed_at, metadata`

func (s *PostgresStore) CreateUploadSessionFiles(ctx context.Context, inputs []domain.UpsertUploadSessionFileInput) ([]domain.UploadSessionFileRecord, error) {
	if len(inputs) == 0 {
		return []domain.UploadSessionFileRecord{}, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	batch := &pgx.Batch{}
	for _, input := range inputs {
		createdAt := input.CreatedAt
		if createdAt.IsZero() {
			createdAt = domain.Now()
		}
		updatedAt := input.UpdatedAt
		if updatedAt.IsZero() {
			updatedAt = createdAt
		}
		status := strings.TrimSpace(input.Status)
		if status == "" {
			status = "active"
		}
		batch.Queue(
			postgresUpsertUploadSessionFileSQL,
			strings.TrimSpace(input.SessionID),
			strings.TrimSpace(input.FileToken),
			nullableText(input.ResourceID),
			strings.TrimSpace(input.OriginalName),
			nullableText(input.RelativePath),
			nullableText(input.ContentType),
			input.SizeBytes,
			nullableText(input.DeclaredSHA256),
			nullableText(input.ComputedSHA256),
			status,
			nullableText(input.Error),
			timestamptz(createdAt),
			timestamptz(updatedAt),
			nullableTimestamptz(input.CompletedAt),
			jsonBytes(input.Metadata),
		)
	}
	results := tx.SendBatch(ctx, batch)
	files := make([]domain.UploadSessionFileRecord, 0, len(inputs))
	for range inputs {
		file, err := scanUploadSessionFileRow(results.QueryRow())
		if err != nil {
			_ = results.Close()
			return nil, mapPgError(err)
		}
		files = append(files, file)
	}
	if err := results.Close(); err != nil {
		return nil, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, mapPgError(err)
	}
	return files, nil
}

func uploadSessionFileCommittedContribution(status string, sizeBytes int64) int64 {
	if strings.TrimSpace(status) == "completed" {
		return sizeBytes
	}
	return 0
}

func (s *PostgresStore) ListUploadSessionFiles(ctx context.Context, sessionID string) ([]domain.UploadSessionFileRecord, error) {
	rows, err := s.queries.ListUploadSessionFiles(ctx, strings.TrimSpace(sessionID))
	if err != nil {
		return nil, err
	}
	files := make([]domain.UploadSessionFileRecord, 0, len(rows))
	for _, row := range rows {
		files = append(files, uploadSessionFileFromRow(row))
	}
	return files, nil
}

func (s *PostgresStore) GetUploadSessionFile(ctx context.Context, sessionID string, fileToken string) (domain.UploadSessionFileRecord, error) {
	row, err := s.queries.GetUploadSessionFile(ctx, sqlc.GetUploadSessionFileParams{
		SessionID: strings.TrimSpace(sessionID),
		FileToken: strings.TrimSpace(fileToken),
	})
	if err != nil {
		return domain.UploadSessionFileRecord{}, mapPgError(err)
	}
	return uploadSessionFileFromRow(row), nil
}

func (s *PostgresStore) UpsertUploadChunk(ctx context.Context, input domain.UpsertUploadChunkInput) (domain.UploadChunkRecord, error) {
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "received"
	}
	counterUpdatedAt := input.ReceivedAt
	if counterUpdatedAt.IsZero() {
		counterUpdatedAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.UploadChunkRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	var previousStatus string
	var previousSize int64
	err = tx.QueryRow(ctx, `
SELECT status, size_bytes
FROM control_upload_chunks
WHERE session_id = $1 AND file_token = $2 AND chunk_index = $3
FOR UPDATE`,
		strings.TrimSpace(input.SessionID),
		strings.TrimSpace(input.FileToken),
		int32(input.ChunkIndex),
	).Scan(&previousStatus, &previousSize)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return domain.UploadChunkRecord{}, mapPgError(err)
	}
	row, err := s.queries.WithTx(tx).UpsertUploadChunk(ctx, sqlc.UpsertUploadChunkParams{
		SessionID:  strings.TrimSpace(input.SessionID),
		FileToken:  strings.TrimSpace(input.FileToken),
		ChunkIndex: int32(input.ChunkIndex),
		ByteOffset: input.Offset,
		SizeBytes:  input.SizeBytes,
		Sha256:     strings.TrimSpace(input.SHA256),
		Status:     status,
		StorageUri: nullableText(input.StorageURI),
		ReceivedAt: nullableTimestamptz(input.ReceivedAt),
		VerifiedAt: nullableTimestamptz(input.VerifiedAt),
		Error:      nullableText(input.Error),
		Metadata:   jsonBytes(input.Metadata),
	})
	if err != nil {
		mapped := mapPgError(err)
		if errors.Is(mapped, ErrNotFound) {
			return domain.UploadChunkRecord{}, fmt.Errorf("%w: verified upload chunk cannot be replaced with different bytes", ErrConflict)
		}
		return domain.UploadChunkRecord{}, mapped
	}
	chunk := uploadChunkFromRow(row)
	previousReceived, previousVerified := uploadSessionChunkByteContribution(previousStatus, previousSize)
	nextReceived, nextVerified := uploadSessionChunkByteContribution(chunk.Status, chunk.SizeBytes)
	receivedDelta := nextReceived - previousReceived
	verifiedDelta := nextVerified - previousVerified
	if receivedDelta != 0 || verifiedDelta != 0 {
		if _, err := tx.Exec(ctx, `
UPDATE control_upload_sessions
SET bytes_received = GREATEST(0, bytes_received + $2),
    bytes_verified = GREATEST(0, bytes_verified + $3),
    updated_at = GREATEST(updated_at, $4)
WHERE session_id = $1`,
			chunk.SessionID,
			receivedDelta,
			verifiedDelta,
			timestamptz(counterUpdatedAt),
		); err != nil {
			return domain.UploadChunkRecord{}, mapPgError(err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.UploadChunkRecord{}, mapPgError(err)
	}
	return chunk, nil
}

func uploadSessionChunkByteContribution(status string, sizeBytes int64) (int64, int64) {
	switch strings.TrimSpace(status) {
	case "received":
		return sizeBytes, 0
	case "verified":
		return sizeBytes, sizeBytes
	default:
		return 0, 0
	}
}

func (s *PostgresStore) ListUploadChunks(ctx context.Context, sessionID string, fileToken string) ([]domain.UploadChunkRecord, error) {
	rows, err := s.queries.ListUploadChunks(ctx, sqlc.ListUploadChunksParams{
		SessionID: strings.TrimSpace(sessionID),
		FileToken: strings.TrimSpace(fileToken),
	})
	if err != nil {
		return nil, err
	}
	chunks := make([]domain.UploadChunkRecord, 0, len(rows))
	for _, row := range rows {
		chunks = append(chunks, uploadChunkFromRow(row))
	}
	return chunks, nil
}

func (s *PostgresStore) ListUploadSessionChunks(ctx context.Context, sessionID string) ([]domain.UploadChunkRecord, error) {
	rows, err := s.queries.ListUploadSessionChunks(ctx, strings.TrimSpace(sessionID))
	if err != nil {
		return nil, err
	}
	chunks := make([]domain.UploadChunkRecord, 0, len(rows))
	for _, row := range rows {
		chunks = append(chunks, uploadChunkFromRow(row))
	}
	return chunks, nil
}

func (s *PostgresStore) GetUploadSessionTotals(ctx context.Context, sessionID string) (domain.UploadSessionTotals, error) {
	row, err := s.queries.GetUploadSessionTotals(ctx, strings.TrimSpace(sessionID))
	if err != nil {
		return domain.UploadSessionTotals{}, mapPgError(err)
	}
	return domain.UploadSessionTotals{
		BytesReceived:  row.BytesReceived,
		BytesVerified:  row.BytesVerified,
		BytesCommitted: row.BytesCommitted,
		AllComplete:    row.AllComplete.Valid && row.AllComplete.Bool,
	}, nil
}

func (s *PostgresStore) UploadSessionOperationalMetrics(ctx context.Context) (domain.UploadSessionOperationalMetrics, error) {
	row, err := s.queries.UploadSessionOperationalMetrics(ctx)
	if err != nil {
		return domain.UploadSessionOperationalMetrics{}, mapPgError(err)
	}
	return domain.UploadSessionOperationalMetrics{
		Total:          int(row.Total),
		Active:         int(row.Active),
		Paused:         int(row.Paused),
		Completed:      int(row.Completed),
		Failed:         int(row.Failed),
		Canceled:       int(row.Canceled),
		Other:          int(row.Other),
		BytesTotal:     row.BytesTotal,
		BytesReceived:  row.BytesReceived,
		BytesVerified:  row.BytesVerified,
		BytesCommitted: row.BytesCommitted,
	}, nil
}

func (s *PostgresStore) AppendUploadSessionEvent(ctx context.Context, input domain.AppendUploadSessionEventInput) (domain.UploadSessionEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("upload_session_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	row, err := s.queries.AppendUploadSessionEvent(ctx, sqlc.AppendUploadSessionEventParams{
		EventID:     eventID,
		SessionID:   strings.TrimSpace(input.SessionID),
		ActorUserID: nullableText(input.ActorUserID),
		ActorOrgID:  nullableText(input.ActorOrgID),
		EventType:   strings.TrimSpace(input.EventType),
		Ts:          timestamptz(ts),
		Metadata:    jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.UploadSessionEventRecord{}, mapPgError(err)
	}
	return uploadSessionEventFromRow(row), nil
}

func (s *PostgresStore) ListUploadSessionEvents(ctx context.Context, sessionID string, limit int) ([]domain.UploadSessionEventRecord, error) {
	rows, err := s.queries.ListUploadSessionEvents(ctx, sqlc.ListUploadSessionEventsParams{
		SessionID: strings.TrimSpace(sessionID),
		Limit:     limit32(limit, 200),
	})
	if err != nil {
		return nil, mapPgError(err)
	}
	events := make([]domain.UploadSessionEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, uploadSessionEventFromRow(row))
	}
	return events, nil
}

func (s *PostgresStore) FindActiveResourceByShaForUser(ctx context.Context, userID string, orgID string, sha256 string, sizeBytes int64) (domain.ResourceRecord, error) {
	row, err := s.queries.FindActiveResourceByShaForUser(ctx, sqlc.FindActiveResourceByShaForUserParams{
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
		Sha256:      nullableText(sha256),
		SizeBytes:   sizeBytes,
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) UpsertResource(ctx context.Context, input domain.UpsertResourceInput) (domain.ResourceRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceRecord{}, err
	}
	defer tx.Rollback(ctx)
	resourceID := input.ResourceID
	if resourceID == "" {
		resourceID = domain.NewID("file")
	}
	if !domain.IsCanonicalResourceID(resourceID) {
		return domain.ResourceRecord{}, ErrConflict
	}
	if err := lockResourceLifecycleTx(ctx, tx, resourceID); err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	sourceType := strings.TrimSpace(input.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	resourceKind := strings.TrimSpace(input.ResourceKind)
	if resourceKind == "" {
		resourceKind = "file"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	tags := normalizeResourceTags(input.Tags)
	metadata := mapOrEmpty(input.Metadata)
	if len(tags) == 0 {
		tags = resourceTagsFromMetadata(metadata)
	}
	metadata = resourceMetadataWithTags(metadata, tags)
	row, err := s.queries.WithTx(tx).UpsertResource(ctx, sqlc.UpsertResourceParams{
		ResourceID:         resourceID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         nullableText(input.OwnerOrgID),
		OwnerRole:          nullableText(input.OwnerRole),
		OriginalName:       strings.TrimSpace(input.OriginalName),
		ContentType:        nullableText(input.ContentType),
		SizeBytes:          input.SizeBytes,
		Sha256:             nullableText(input.SHA256),
		StorageUri:         nullableText(input.StorageURI),
		StoragePath:        nullableText(input.StoragePath),
		SourceType:         sourceType,
		ResourceKind:       resourceKind,
		SourceUri:          nullableText(input.SourceURI),
		ProjectID:          nullableText(input.ProjectID),
		Status:             status,
		CreatedAt:          timestamptz(createdAt),
		UpdatedAt:          timestamptz(updatedAt),
		DeletedAt:          nullableTimestamptz(input.DeletedAt),
		RetentionExpiresAt: nullableTimestamptz(input.RetentionExpiresAt),
		Metadata:           jsonBytes(metadata),
	})
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.ResourceRecord{}, ErrConflict
		}
		return domain.ResourceRecord{}, mapPgError(err)
	}
	resource := resourceFromRow(row)
	if err := upsertResourceSearchDocumentTx(ctx, tx, resource); err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceRecord{}, err
	}
	return resource, nil
}

func (s *PostgresStore) MergeResourceMetadataForUser(ctx context.Context, input domain.MergeResourceMetadataInput) (domain.ResourceRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceRecord{}, err
	}
	defer tx.Rollback(ctx)
	resourceID := strings.TrimSpace(input.ResourceID)
	userID := strings.TrimSpace(input.UserID)
	orgID := strings.TrimSpace(input.OrgID)
	selected, err := scanControlResourceRow(tx.QueryRow(ctx, `
SELECT resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
       storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
       updated_at, deleted_at, retention_expires_at, metadata
FROM control_resources
WHERE resource_id = $1
  AND owner_user_id = $2
  AND COALESCE(owner_org_id, '') = $3
FOR UPDATE`, resourceID, userID, orgID))
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	resource := resourceFromRow(selected)
	if err := validateViewerCalibrationPrecondition(resource, input); err != nil {
		return domain.ResourceRecord{}, err
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	metadata := mergeResourceMetadata(resource.Metadata, input.Patch)
	updated, err := scanControlResourceRow(tx.QueryRow(ctx, `
UPDATE control_resources
SET metadata = $2,
    updated_at = $3
WHERE resource_id = $1
RETURNING resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
          storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
          updated_at, deleted_at, retention_expires_at, metadata`, resourceID, jsonBytes(metadata), timestamptz(updatedAt)))
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	resource = resourceFromRow(updated)
	if err := upsertResourceSearchDocumentTx(ctx, tx, resource); err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceRecord{}, err
	}
	return resource, nil
}

func (s *PostgresStore) RenameResourceForUser(ctx context.Context, input domain.RenameResourceInput) (domain.ResourceRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceRecord{}, err
	}
	defer tx.Rollback(ctx)
	resourceID := strings.TrimSpace(input.ResourceID)
	userID := strings.TrimSpace(input.UserID)
	orgID := strings.TrimSpace(input.OrgID)
	name := strings.TrimSpace(input.OriginalName)
	if name == "" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	updated, err := scanControlResourceRow(tx.QueryRow(ctx, `
UPDATE control_resources
SET original_name = $4,
    updated_at = $5
WHERE resource_id = $1
  AND owner_user_id = $2
  AND COALESCE(owner_org_id, '') = $3
  AND status = 'active'
RETURNING resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
          storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
          updated_at, deleted_at, retention_expires_at, metadata`, resourceID, userID, orgID, name, timestamptz(updatedAt)))
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	resource := resourceFromRow(updated)
	if err := upsertResourceSearchDocumentTx(ctx, tx, resource); err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceRecord{}, err
	}
	return resource, nil
}

func (s *PostgresStore) GetResourceForUser(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	row, err := s.queries.GetResourceForUser(ctx, sqlc.GetResourceForUserParams{
		ResourceID:  strings.TrimSpace(resourceID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) GetResourceForOwner(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	row, err := s.queries.GetResourceForOwner(ctx, sqlc.GetResourceForOwnerParams{
		ResourceID:  strings.TrimSpace(resourceID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) ListResourceIDsForOwner(ctx context.Context, userID string, orgID string, resourceIDs []string) (map[string]bool, error) {
	resourceIDs = uniqueTrimmedStrings(resourceIDs)
	existing := make(map[string]bool, len(resourceIDs))
	if len(resourceIDs) == 0 {
		return existing, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT resource_id
FROM control_resources
WHERE owner_user_id = $1
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $2)
  AND resource_id = ANY($3::text[])`,
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
		resourceIDs,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	for rows.Next() {
		var resourceID string
		if err := rows.Scan(&resourceID); err != nil {
			return nil, mapPgError(err)
		}
		existing[resourceID] = true
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return existing, nil
}

func (s *PostgresStore) resourceSearchCandidateIDs(ctx context.Context, parsed parsedResourceSearchQuery, input domain.ResourceListInput, status string) ([]string, bool, error) {
	if !parsed.hasFactPredicates() {
		return nil, false, nil
	}
	var candidates map[string]struct{}
	intersect := func(ids []string) {
		if candidates == nil {
			candidates = make(map[string]struct{}, len(ids))
			for _, id := range ids {
				candidates[id] = struct{}{}
			}
			return
		}
		seen := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			seen[id] = struct{}{}
		}
		for id := range candidates {
			if _, ok := seen[id]; !ok {
				delete(candidates, id)
			}
		}
	}
	for _, predicate := range parsed.NumericPredicates {
		ids, err := s.resourceSearchCandidateIDsForNumericPredicate(ctx, predicate, input, status)
		if err != nil {
			return nil, true, err
		}
		intersect(ids)
		if len(candidates) == 0 {
			return []string{}, true, nil
		}
	}
	for _, predicate := range parsed.TextPredicates {
		ids, err := s.resourceSearchCandidateIDsForTextPredicate(ctx, predicate, input, status)
		if err != nil {
			return nil, true, err
		}
		intersect(ids)
		if len(candidates) == 0 {
			return []string{}, true, nil
		}
	}
	out := make([]string, 0, len(candidates))
	for id := range candidates {
		out = append(out, id)
	}
	return out, true, nil
}

func (s *PostgresStore) resourceSearchCandidateIDsForNumericPredicate(ctx context.Context, predicate resourceSearchNumericPredicate, input domain.ResourceListInput, status string) ([]string, error) {
	factPredicate := `
sf.fact_key = $1
  AND sf.fact_number IS NOT NULL
  AND sf.fact_number = $2`
	switch predicate.Op {
	case "gt":
		factPredicate = `
sf.fact_key = $1
  AND sf.fact_number IS NOT NULL
  AND sf.fact_number > $2`
	case "gte":
		factPredicate = `
sf.fact_key = $1
  AND sf.fact_number IS NOT NULL
  AND sf.fact_number >= $2`
	case "lt":
		factPredicate = `
sf.fact_key = $1
  AND sf.fact_number IS NOT NULL
  AND sf.fact_number < $2`
	case "lte":
		factPredicate = `
sf.fact_key = $1
  AND sf.fact_number IS NOT NULL
  AND sf.fact_number <= $2`
	case "eq":
	default:
		return []string{}, nil
	}
	return s.resourceSearchCandidateIDsForQuery(ctx, postgresResourceSearchCandidateQuery(factPredicate), predicate.Key, predicate.Number, status, strings.TrimSpace(input.UserID), strings.TrimSpace(input.OrgID))
}

func (s *PostgresStore) resourceSearchCandidateIDsForTextPredicate(ctx context.Context, predicate resourceSearchTextPredicate, input domain.ResourceListInput, status string) ([]string, error) {
	return s.resourceSearchCandidateIDsForQuery(ctx, postgresResourceSearchCandidateQuery(`
sf.fact_key = $1
  AND sf.fact_text = $2`),
		predicate.Key,
		predicate.Text,
		status,
		strings.TrimSpace(input.UserID),
		strings.TrimSpace(input.OrgID),
	)
}

func postgresResourceSearchCandidateQuery(factPredicate string) string {
	return `
SELECT DISTINCT resource_id
FROM (
  SELECT sf.resource_id
  FROM control_resource_search_facts sf
  WHERE ` + factPredicate + `
    AND sf.status = $3
    AND sf.owner_user_id = $4
    AND (COALESCE(sf.owner_org_id, '') = '' OR sf.owner_org_id = $5)
  UNION
  SELECT sf.resource_id
  FROM control_resource_search_facts sf
  JOIN control_resources r ON r.resource_id = sf.resource_id
  WHERE ` + factPredicate + `
  ` + postgresResourceSearchSharedCandidateVisibilitySQL() + `
) AS candidate_resources`
}

func postgresResourceSearchSharedCandidateVisibilitySQL() string {
	return `
AND sf.status = $3
AND sf.status = 'active'
AND r.status = 'active'
AND EXISTS (
  SELECT 1
  FROM control_resource_share_grants g
  WHERE g.resource_id = r.resource_id
    AND g.status = 'active'
    AND (
      COALESCE(g.grantee_user_id, '') = '__public__'
      OR (
        COALESCE(g.grantee_user_id, '') <> ''
        AND g.grantee_user_id = $4
        AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $5)
      )
      OR (
        COALESCE(g.grantee_user_id, '') = ''
        AND COALESCE(g.grantee_org_id, '') <> ''
        AND g.grantee_org_id = $5
      )
    )
  )
`
}

func (s *PostgresStore) resourceSearchCandidateIDsForQuery(ctx context.Context, query string, args ...any) ([]string, error) {
	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var resourceID string
		if err := rows.Scan(&resourceID); err != nil {
			return nil, mapPgError(err)
		}
		ids = append(ids, resourceID)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return ids, nil
}

func (s *PostgresStore) ListResourcesForUser(ctx context.Context, input domain.ResourceListInput) (domain.ResourceListPage, error) {
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	parsedQuery := parseResourceSearchQuery(input.Query)
	candidateIDs, factFilterEnabled, err := s.resourceSearchCandidateIDs(ctx, parsedQuery, input, status)
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	if factFilterEnabled && len(candidateIDs) == 0 {
		return domain.ResourceListPage{
			Resources:  []domain.ResourceRecord{},
			TotalCount: 0,
			Limit:      int(limit),
			Offset:     int(offset),
		}, nil
	}
	metadataFilterSpecs := resourceMetadataFilterSpecs(input.MetadataFilters)
	descriptorFilters := normalizeResourceDescriptors(input.Descriptors)
	params := sqlc.ListResourcesForUserParams{
		OwnerUserID: strings.TrimSpace(input.UserID),
		OwnerOrgID:  nullableText(input.OrgID),
		Status:      status,
		Column4:     strings.TrimSpace(input.Kind),
		Column5:     strings.TrimSpace(input.Source),
		Column6:     strings.TrimSpace(input.ProjectID),
		Column7:     strings.TrimSpace(parsedQuery.ResidualText),
		Column8:     resourceTagKeys(input.Tags),
		Column9:     metadataFilterSpecs,
		Column10:    nullableTimestamptz(input.CreatedAfter),
		Column11:    nullableTimestamptz(input.CreatedBefore),
		Column12:    strings.ToLower(strings.TrimSpace(input.ProcessingStatus)),
		Column13:    strings.ToLower(strings.TrimSpace(input.Sharing)),
		Column14:    descriptorFilters,
		Column15:    factFilterEnabled,
		Column16:    candidateIDs,
		Limit:       limit,
		Offset:      offset,
	}
	rows, err := s.queries.ListResourcesForUser(ctx, params)
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	count, err := s.queries.CountResourcesForUser(ctx, sqlc.CountResourcesForUserParams{
		OwnerUserID: params.OwnerUserID,
		OwnerOrgID:  params.OwnerOrgID,
		Status:      params.Status,
		Column4:     params.Column4,
		Column5:     params.Column5,
		Column6:     params.Column6,
		Column7:     params.Column7,
		Column8:     params.Column8,
		Column9:     params.Column9,
		Column10:    params.Column10,
		Column11:    params.Column11,
		Column12:    params.Column12,
		Column13:    params.Column13,
		Column14:    params.Column14,
		Column15:    params.Column15,
		Column16:    params.Column16,
	})
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	resources := make([]domain.ResourceRecord, 0, len(rows))
	for _, row := range rows {
		resources = append(resources, resourceFromListResourcesForUserRow(row))
	}
	return domain.ResourceListPage{
		Resources:  resources,
		TotalCount: int(count),
		Limit:      int(params.Limit),
		Offset:     int(params.Offset),
	}, nil
}

func (s *PostgresStore) BulkTagResourcesForUser(ctx context.Context, input domain.BulkTagResourcesInput) (domain.BulkTagResourcesResult, error) {
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	tags := normalizeResourceTags(input.Tags)
	if len(resourceIDs) == 0 || len(tags) == 0 {
		return domain.BulkTagResourcesResult{}, ErrNotFound
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	actorUserID := strings.TrimSpace(input.ActorUserID)
	if actorUserID == "" {
		actorUserID = ownerUserID
	}
	actorOrgID := strings.TrimSpace(input.ActorOrgID)
	if actorOrgID == "" {
		actorOrgID = ownerOrgID
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.BulkTagResourcesResult{}, err
	}
	defer tx.Rollback(ctx)

	resources := make([]domain.ResourceRecord, 0, len(resourceIDs))
	events := make([]domain.ResourceEventRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		selected, err := scanControlResourceRow(tx.QueryRow(ctx, `
SELECT resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
       storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
       updated_at, deleted_at, retention_expires_at, metadata
FROM control_resources
WHERE resource_id = $1
  AND owner_user_id = $2
  AND COALESCE(owner_org_id, '') = $3
  AND status = 'active'
FOR UPDATE`, resourceID, ownerUserID, ownerOrgID))
		if err != nil {
			return domain.BulkTagResourcesResult{}, mapPgError(err)
		}
		resource := resourceFromRow(selected)
		resource.Tags = mergeResourceTags(tagsForResource(resource), tags)
		resource.Metadata = resourceMetadataWithTags(resource.Metadata, resource.Tags)
		updated, err := scanControlResourceRow(tx.QueryRow(ctx, `
UPDATE control_resources
SET metadata = $2,
    updated_at = $3
WHERE resource_id = $1
RETURNING resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
          storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
          updated_at, deleted_at, retention_expires_at, metadata`, resourceID, jsonBytes(resource.Metadata), timestamptz(ts)))
		if err != nil {
			return domain.BulkTagResourcesResult{}, mapPgError(err)
		}
		updatedResource := resourceFromRow(updated)
		if err := upsertResourceSearchDocumentTx(ctx, tx, updatedResource); err != nil {
			return domain.BulkTagResourcesResult{}, mapPgError(err)
		}
		eventMetadata := domain.JSONMap{
			"tags_added": append([]string(nil), tags...),
			"tags":       append([]string(nil), resource.Tags...),
		}
		if len(input.Metadata) > 0 {
			eventMetadata["request"] = cloneResourceMetadataValue(input.Metadata)
		}
		eventID := domain.NewID("resource_event")
		event, err := scanResourceEventRow(tx.QueryRow(ctx, `
INSERT INTO control_resource_events (event_id, resource_id, actor_user_id, actor_org_id, event_type, ts, metadata)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), 'resource.tagged', $5, $6)
RETURNING event_id, resource_id, COALESCE(actor_user_id, ''), COALESCE(actor_org_id, ''), event_type, ts, metadata`,
			eventID,
			resourceID,
			actorUserID,
			actorOrgID,
			ts.UTC(),
			jsonBytes(eventMetadata),
		))
		if err != nil {
			return domain.BulkTagResourcesResult{}, mapPgError(err)
		}
		resources = append(resources, updatedResource)
		events = append(events, event)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.BulkTagResourcesResult{}, err
	}
	return domain.BulkTagResourcesResult{
		UpdatedCount: len(resources),
		Resources:    resources,
		Events:       events,
	}, nil
}

func (s *PostgresStore) ListResources(ctx context.Context, limit int, offset int) ([]domain.ResourceRecord, error) {
	rows, err := s.queries.ListResources(ctx, sqlc.ListResourcesParams{
		Limit:  limit32(limit, 1000),
		Offset: offset32(offset),
	})
	if err != nil {
		return nil, err
	}
	resources := make([]domain.ResourceRecord, 0, len(rows))
	for _, row := range rows {
		resources = append(resources, resourceFromRow(row))
	}
	return resources, nil
}

// ListResourceLifecycleFenceCandidates returns deletion states in stable ID
// order so startup reconciliation cannot skip rows when unrelated resources
// are concurrently inserted or updated.
func (s *PostgresStore) ListResourceLifecycleFenceCandidates(ctx context.Context, afterResourceID string, limit int) ([]domain.ResourceRecord, error) {
	rows, err := s.queries.ListResourceLifecycleFenceCandidates(ctx, sqlc.ListResourceLifecycleFenceCandidatesParams{
		AfterResourceID: afterResourceID,
		PageLimit:       limit32(limit, 100),
	})
	if err != nil {
		return nil, mapPgError(err)
	}
	resources := make([]domain.ResourceRecord, 0, len(rows))
	for _, row := range rows {
		resources = append(resources, resourceFromRow(row))
	}
	return resources, nil
}

func (s *PostgresStore) GetResourceLifecycleStatus(ctx context.Context, resourceID string) (string, bool, error) {
	status, err := s.queries.GetResourceLifecycleStatus(ctx, resourceID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, mapPgError(err)
	}
	return strings.TrimSpace(status), true, nil
}

func (s *PostgresStore) CreateResourceCollection(ctx context.Context, input domain.CreateResourceCollectionInput) (domain.ResourceCollectionRecord, error) {
	collectionID := strings.TrimSpace(input.CollectionID)
	if collectionID == "" {
		collectionID = domain.NewID("collection")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	collectionType := strings.ToLower(strings.TrimSpace(input.CollectionType))
	if collectionType == "" {
		collectionType = "collection"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_resource_collections (
  collection_id, owner_user_id, owner_org_id, owner_role, project_id, parent_collection_id,
  name, description, collection_type, status, created_at, updated_at, metadata
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''),
        $7, NULLIF($8, ''), $9, $10, $11, $12, $13)
RETURNING collection_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), COALESCE(parent_collection_id, ''), name,
          COALESCE(description, ''), collection_type, status, 0::bigint,
          created_at, updated_at, metadata`,
		collectionID,
		ownerUserID,
		strings.TrimSpace(input.OwnerOrgID),
		strings.TrimSpace(input.OwnerRole),
		strings.TrimSpace(input.ProjectID),
		strings.TrimSpace(input.ParentCollectionID),
		strings.TrimSpace(input.Name),
		strings.TrimSpace(input.Description),
		collectionType,
		status,
		createdAt.UTC(),
		updatedAt.UTC(),
		jsonBytes(input.Metadata),
	)
	collection, err := scanResourceCollectionRow(row)
	if err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	return collection, nil
}

func (s *PostgresStore) GetResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string) (domain.ResourceCollectionRecord, error) {
	collection, err := scanResourceCollectionRow(s.pool.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.status = 'active'
  AND (
    (c.owner_user_id = $2 AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3))
    OR EXISTS (
      SELECT 1
      FROM control_resource_collection_share_grants g
      WHERE g.collection_id = c.collection_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    )
  )
GROUP BY c.collection_id`,
		strings.TrimSpace(collectionID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
	))
	if err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	return collection, nil
}

func (s *PostgresStore) RenameResourceCollectionForUser(ctx context.Context, input domain.RenameResourceCollectionInput) (domain.ResourceCollectionRecord, error) {
	name := strings.TrimSpace(input.Name)
	if name == "" {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	collection, err := scanResourceCollectionRow(s.pool.QueryRow(ctx, `
UPDATE control_resource_collections c
SET name = $4,
    updated_at = $5
WHERE c.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
  AND c.status = 'active'
RETURNING c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
          COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
          COALESCE(c.description, ''), c.collection_type, c.status,
          (
            SELECT COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint
            FROM control_resource_collection_members m
            LEFT JOIN control_resources r ON r.resource_id = m.resource_id
            WHERE m.collection_id = c.collection_id
          ) AS resource_count,
          c.created_at, c.updated_at, c.metadata`,
		strings.TrimSpace(input.CollectionID),
		strings.TrimSpace(input.UserID),
		strings.TrimSpace(input.OrgID),
		name,
		updatedAt.UTC(),
	))
	if err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	return collection, nil
}

func (s *PostgresStore) ListResourceCollectionsForUser(ctx context.Context, input domain.ResourceCollectionListInput) (domain.ResourceCollectionListPage, error) {
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	ownerUserID := strings.TrimSpace(input.UserID)
	ownerOrgID := strings.TrimSpace(input.OrgID)
	collectionType := strings.ToLower(strings.TrimSpace(input.Type))
	projectID := strings.TrimSpace(input.ProjectID)
	query := strings.TrimSpace(input.Query)
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	countRow := s.pool.QueryRow(ctx, `
SELECT COUNT(*)
FROM control_resource_collections c
WHERE c.status = $3
  AND (
    (c.owner_user_id = $1 AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $2))
    OR ($3::text = 'active' AND EXISTS (
      SELECT 1
      FROM control_resource_collection_share_grants g
      WHERE g.collection_id = c.collection_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $1 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $2)
        )
    ))
  )
  AND ($4::text = '' OR c.collection_type = $4)
  AND ($5::text = '' OR COALESCE(c.project_id, '') = $5)
  AND ($6::text = '' OR c.name ILIKE '%' || $6 || '%' OR COALESCE(c.description, '') ILIKE '%' || $6 || '%' OR c.collection_id ILIKE '%' || $6 || '%')`,
		ownerUserID,
		ownerOrgID,
		status,
		collectionType,
		projectID,
		query,
	)
	var total int
	if err := countRow.Scan(&total); err != nil {
		return domain.ResourceCollectionListPage{}, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.status = $3
  AND (
    (c.owner_user_id = $1 AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $2))
    OR ($3::text = 'active' AND EXISTS (
      SELECT 1
      FROM control_resource_collection_share_grants g
      WHERE g.collection_id = c.collection_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $1 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $2)
        )
    ))
  )
  AND ($4::text = '' OR c.collection_type = $4)
  AND ($5::text = '' OR COALESCE(c.project_id, '') = $5)
  AND ($6::text = '' OR c.name ILIKE '%' || $6 || '%' OR COALESCE(c.description, '') ILIKE '%' || $6 || '%' OR c.collection_id ILIKE '%' || $6 || '%')
GROUP BY c.collection_id
ORDER BY c.updated_at DESC, c.collection_id ASC
LIMIT $7 OFFSET $8`,
		ownerUserID,
		ownerOrgID,
		status,
		collectionType,
		projectID,
		query,
		limit,
		offset,
	)
	if err != nil {
		return domain.ResourceCollectionListPage{}, mapPgError(err)
	}
	defer rows.Close()
	collections, err := scanResourceCollectionRows(rows)
	if err != nil {
		return domain.ResourceCollectionListPage{}, err
	}
	return domain.ResourceCollectionListPage{
		Collections: collections,
		TotalCount:  total,
		Limit:       int(limit),
		Offset:      int(offset),
	}, nil
}

func (s *PostgresStore) SoftDeleteResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string, deletedAt time.Time) (domain.ResourceCollectionRecord, error) {
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	// Block deleting a folder that still has active subfolders: children keep
	// pointing at the deleted parent and become unreachable in nested browsing.
	var hasActiveChildren bool
	if err := s.pool.QueryRow(ctx, `
	SELECT EXISTS(
	  SELECT 1 FROM control_resource_collections
	  WHERE parent_collection_id = $1 AND status = 'active'
	)`, strings.TrimSpace(collectionID)).Scan(&hasActiveChildren); err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	if hasActiveChildren {
		return domain.ResourceCollectionRecord{}, fmt.Errorf("%w: collection has active subfolders", ErrConflict)
	}
	collection, err := scanResourceCollectionRow(s.pool.QueryRow(ctx, `
	UPDATE control_resource_collections c
	SET status = 'deleted',
	    updated_at = $4
	WHERE c.collection_id = $1
	  AND c.owner_user_id = $2
	  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
	  AND c.status <> 'deleted'
	RETURNING c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
	          COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
	          COALESCE(c.description, ''), c.collection_type, c.status,
	          (
	            SELECT COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint
	            FROM control_resource_collection_members m
	            LEFT JOIN control_resources r ON r.resource_id = m.resource_id
	            WHERE m.collection_id = c.collection_id
	          ) AS resource_count,
	          c.created_at, c.updated_at, c.metadata`,
		strings.TrimSpace(collectionID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
		deletedAt.UTC(),
	))
	if err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	return collection, nil
}

func (s *PostgresStore) RestoreResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string, restoredAt time.Time) (domain.ResourceCollectionRecord, error) {
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	collection, err := scanResourceCollectionRow(s.pool.QueryRow(ctx, `
	UPDATE control_resource_collections c
	SET status = 'active',
	    updated_at = $4
	WHERE c.collection_id = $1
	  AND c.owner_user_id = $2
	  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
	RETURNING c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
	          COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
	          COALESCE(c.description, ''), c.collection_type, c.status,
	          (
	            SELECT COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint
	            FROM control_resource_collection_members m
	            LEFT JOIN control_resources r ON r.resource_id = m.resource_id
	            WHERE m.collection_id = c.collection_id
	          ) AS resource_count,
	          c.created_at, c.updated_at, c.metadata`,
		strings.TrimSpace(collectionID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
		restoredAt.UTC(),
	))
	if err != nil {
		return domain.ResourceCollectionRecord{}, mapPgError(err)
	}
	return collection, nil
}

func (s *PostgresStore) AddResourcesToCollection(ctx context.Context, input domain.AddResourcesToCollectionInput) (domain.AddResourcesToCollectionResult, error) {
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 {
		return domain.AddResourcesToCollectionResult{}, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.AddResourcesToCollectionResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	collection, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
  AND c.status = 'active'
GROUP BY c.collection_id`,
		strings.TrimSpace(input.CollectionID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	))
	if err != nil {
		return domain.AddResourcesToCollectionResult{}, mapPgError(err)
	}
	resourceRows, err := tx.Query(ctx, `
SELECT resource_id
FROM control_resources
WHERE resource_id = ANY($1::text[])
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'`,
		resourceIDs,
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	)
	if err != nil {
		return domain.AddResourcesToCollectionResult{}, mapPgError(err)
	}
	found := map[string]struct{}{}
	for resourceRows.Next() {
		var resourceID string
		if err := resourceRows.Scan(&resourceID); err != nil {
			resourceRows.Close()
			return domain.AddResourcesToCollectionResult{}, err
		}
		found[resourceID] = struct{}{}
	}
	if err := resourceRows.Err(); err != nil {
		resourceRows.Close()
		return domain.AddResourcesToCollectionResult{}, err
	}
	resourceRows.Close()
	if len(found) != len(resourceIDs) {
		return domain.AddResourcesToCollectionResult{}, ErrNotFound
	}
	var nextPosition int64
	if err := tx.QueryRow(ctx, `SELECT COALESCE(MAX(position), -1) + 1 FROM control_resource_collection_members WHERE collection_id = $1`, collection.CollectionID).Scan(&nextPosition); err != nil {
		return domain.AddResourcesToCollectionResult{}, mapPgError(err)
	}
	addedAt := input.AddedAt
	if addedAt.IsZero() {
		addedAt = domain.Now()
	}
	collectionGrants, err := activeResourceCollectionShareGrantsTx(ctx, tx, collection.CollectionID)
	if err != nil {
		return domain.AddResourcesToCollectionResult{}, err
	}
	memberships := make([]domain.ResourceCollectionMembershipRecord, 0, len(resourceIDs))
	inheritedShareGrants := make([]domain.ResourceShareGrantRecord, 0)
	addedCount := 0
	for _, resourceID := range resourceIDs {
		member, inserted, err := upsertCollectionMemberTx(ctx, tx, collection.CollectionID, resourceID, nextPosition, strings.TrimSpace(input.AddedByUserID), addedAt, input.Metadata)
		if err != nil {
			return domain.AddResourcesToCollectionResult{}, err
		}
		if inserted {
			nextPosition++
			addedCount++
			for _, collectionGrant := range collectionGrants {
				grant, err := createInheritedResourceShareGrantTx(ctx, tx, resourceID, collectionGrant, addedAt, "resource_collection_share_inherited")
				if err != nil {
					return domain.AddResourcesToCollectionResult{}, err
				}
				inheritedShareGrants = append(inheritedShareGrants, grant)
			}
		}
		memberships = append(memberships, member)
	}
	if _, err := tx.Exec(ctx, `UPDATE control_resource_collections SET updated_at = $2 WHERE collection_id = $1`, collection.CollectionID, addedAt.UTC()); err != nil {
		return domain.AddResourcesToCollectionResult{}, mapPgError(err)
	}
	updated, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
GROUP BY c.collection_id`, collection.CollectionID))
	if err != nil {
		return domain.AddResourcesToCollectionResult{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.AddResourcesToCollectionResult{}, err
	}
	return domain.AddResourcesToCollectionResult{
		Collection:           updated,
		AddedCount:           addedCount,
		Memberships:          memberships,
		InheritedShareGrants: inheritedShareGrants,
	}, nil
}

func (s *PostgresStore) RemoveResourcesFromCollection(ctx context.Context, input domain.RemoveResourcesFromCollectionInput) (domain.RemoveResourcesFromCollectionResult, error) {
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 {
		return domain.RemoveResourcesFromCollectionResult{}, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	collection, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
  AND c.status = 'active'
GROUP BY c.collection_id`,
		strings.TrimSpace(input.CollectionID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	))
	if err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
	}
	resourceRows, err := tx.Query(ctx, `
SELECT resource_id
FROM control_resources
WHERE resource_id = ANY($1::text[])
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)`,
		resourceIDs,
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	)
	if err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
	}
	found := map[string]struct{}{}
	for resourceRows.Next() {
		var resourceID string
		if err := resourceRows.Scan(&resourceID); err != nil {
			resourceRows.Close()
			return domain.RemoveResourcesFromCollectionResult{}, err
		}
		found[resourceID] = struct{}{}
	}
	if err := resourceRows.Err(); err != nil {
		resourceRows.Close()
		return domain.RemoveResourcesFromCollectionResult{}, err
	}
	resourceRows.Close()
	if len(found) != len(resourceIDs) {
		return domain.RemoveResourcesFromCollectionResult{}, ErrNotFound
	}
	removedAt := input.RemovedAt
	if removedAt.IsZero() {
		removedAt = domain.Now()
	}
	rows, err := tx.Query(ctx, `
DELETE FROM control_resource_collection_members
WHERE collection_id = $1
  AND resource_id = ANY($2::text[])
RETURNING collection_id, resource_id, position, COALESCE(added_by_user_id, ''), added_at, metadata`,
		collection.CollectionID,
		resourceIDs,
	)
	if err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
	}
	memberships := make([]domain.ResourceCollectionMembershipRecord, 0, len(resourceIDs))
	removedResourceIDs := make([]string, 0, len(resourceIDs))
	for rows.Next() {
		var member domain.ResourceCollectionMembershipRecord
		var metadata []byte
		if err := rows.Scan(&member.CollectionID, &member.ResourceID, &member.Position, &member.AddedByUserID, &member.AddedAt, &metadata); err != nil {
			rows.Close()
			return domain.RemoveResourcesFromCollectionResult{}, err
		}
		member.AddedAt = member.AddedAt.UTC()
		member.Metadata = jsonMap(metadata)
		memberships = append(memberships, member)
		removedResourceIDs = append(removedResourceIDs, member.ResourceID)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return domain.RemoveResourcesFromCollectionResult{}, err
	}
	rows.Close()
	revokedShareGrants := make([]domain.ResourceShareGrantRecord, 0)
	if len(removedResourceIDs) > 0 {
		revokedRows, err := tx.Query(ctx, `
UPDATE control_resource_share_grants AS g
SET status = 'revoked',
    revoked_at = $3,
    updated_at = $3
FROM control_resource_collection_share_grants AS cg
WHERE g.resource_id = ANY($1::text[])
  AND cg.collection_id = $2
  AND g.status = 'active'
  AND g.metadata->>'collection_share_grant_id' = cg.grant_id
RETURNING g.grant_id, g.resource_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
          COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
          COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata`,
			removedResourceIDs,
			collection.CollectionID,
			timestamptz(removedAt),
		)
		if err != nil {
			return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
		}
		for revokedRows.Next() {
			grant, err := scanResourceShareGrantRow(revokedRows)
			if err != nil {
				revokedRows.Close()
				return domain.RemoveResourcesFromCollectionResult{}, err
			}
			revokedShareGrants = append(revokedShareGrants, grant)
		}
		if err := revokedRows.Err(); err != nil {
			revokedRows.Close()
			return domain.RemoveResourcesFromCollectionResult{}, err
		}
		revokedRows.Close()
		if _, err := tx.Exec(ctx, `UPDATE control_resource_collections SET updated_at = $2 WHERE collection_id = $1`, collection.CollectionID, removedAt.UTC()); err != nil {
			return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
		}
	}
	updated, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
GROUP BY c.collection_id`, collection.CollectionID))
	if err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RemoveResourcesFromCollectionResult{}, err
	}
	return domain.RemoveResourcesFromCollectionResult{
		Collection:                  updated,
		RemovedCount:                len(memberships),
		Memberships:                 memberships,
		RevokedInheritedShareGrants: revokedShareGrants,
	}, nil
}

func (s *PostgresStore) CreateResourceCollectionShareGrant(ctx context.Context, input domain.CreateResourceCollectionShareGrantInput) (domain.CreateResourceCollectionShareGrantResult, error) {
	collectionID := strings.TrimSpace(input.CollectionID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if input.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.CreateResourceCollectionShareGrantResult{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("collection_grant")
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.CreateResourceCollectionShareGrantResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	collection, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
  AND c.status = 'active'
GROUP BY c.collection_id`,
		collectionID,
		ownerUserID,
		ownerOrgID,
	))
	if err != nil {
		return domain.CreateResourceCollectionShareGrantResult{}, mapPgError(err)
	}
	grant, err := scanResourceCollectionShareGrantRow(tx.QueryRow(ctx, `
INSERT INTO control_resource_collection_share_grants (
  grant_id, collection_id, owner_user_id, owner_org_id, owner_role,
  grantee_user_id, grantee_org_id, role, status, created_by_user_id,
  created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING grant_id, collection_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(grantee_user_id, ''), COALESCE(grantee_org_id, ''), role, status,
          COALESCE(created_by_user_id, ''), created_at, updated_at, revoked_at, metadata`,
		grantID,
		collection.CollectionID,
		collection.OwnerUserID,
		nullableText(collection.OwnerOrgID),
		nullableText(collection.OwnerRole),
		nullableText(granteeUserID),
		nullableText(granteeOrgID),
		role,
		status,
		nullableText(createdByUserID),
		timestamptz(createdAt),
		timestamptz(updatedAt),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.CreateResourceCollectionShareGrantResult{}, mapPgError(err)
	}
	rows, err := tx.Query(ctx, `
SELECT r.resource_id
FROM control_resource_collection_members m
JOIN control_resources r ON r.resource_id = m.resource_id
LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
WHERE m.collection_id = $1
  AND r.owner_user_id = $2
  AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)
  AND r.status = 'active'
ORDER BY m.position ASC, m.added_at ASC, m.resource_id ASC`,
		collection.CollectionID,
		collection.OwnerUserID,
		collection.OwnerOrgID,
	)
	if err != nil {
		return domain.CreateResourceCollectionShareGrantResult{}, mapPgError(err)
	}
	resourceIDs := make([]string, 0)
	for rows.Next() {
		var resourceID string
		if err := rows.Scan(&resourceID); err != nil {
			rows.Close()
			return domain.CreateResourceCollectionShareGrantResult{}, err
		}
		resourceIDs = append(resourceIDs, resourceID)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return domain.CreateResourceCollectionShareGrantResult{}, err
	}
	rows.Close()
	resourceGrants := make([]domain.ResourceShareGrantRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		resourceGrant, err := createInheritedResourceShareGrantTx(ctx, tx, resourceID, grant, createdAt, "resource_collection_share")
		if err != nil {
			return domain.CreateResourceCollectionShareGrantResult{}, err
		}
		resourceGrants = append(resourceGrants, resourceGrant)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.CreateResourceCollectionShareGrantResult{}, err
	}
	return domain.CreateResourceCollectionShareGrantResult{
		Grant:          grant,
		ResourceGrants: resourceGrants,
	}, nil
}

func (s *PostgresStore) ListResourcesForCollectionForUser(ctx context.Context, input domain.ResourceCollectionResourceListInput) (domain.ResourceListPage, error) {
	collectionID := strings.TrimSpace(input.CollectionID)
	ownerUserID := strings.TrimSpace(input.UserID)
	ownerOrgID := strings.TrimSpace(input.OrgID)
	kind := strings.TrimSpace(input.Kind)
	source := strings.TrimSpace(input.Source)
	projectID := strings.TrimSpace(input.ProjectID)
	query := strings.TrimSpace(input.Query)
	sharing := strings.ToLower(strings.TrimSpace(input.Sharing))
	tagKeys := resourceTagKeys(input.Tags)
	descriptorFilters := normalizeResourceDescriptors(input.Descriptors)
	metadataFilterSpecs := resourceMetadataFilterSpecs(input.MetadataFilters)
	processingStatus := strings.ToLower(strings.TrimSpace(input.ProcessingStatus))
	if _, err := scanResourceCollectionRow(s.pool.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.status = 'active'
  AND (
    (c.owner_user_id = $2 AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3))
    OR EXISTS (
      SELECT 1
      FROM control_resource_collection_share_grants g
      WHERE g.collection_id = c.collection_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    )
  )
GROUP BY c.collection_id`, collectionID, ownerUserID, ownerOrgID)); err != nil {
		return domain.ResourceListPage{}, mapPgError(err)
	}
	var total int
	if err := s.pool.QueryRow(ctx, `
SELECT COUNT(*)
FROM control_resource_collection_members m
JOIN control_resources r ON r.resource_id = m.resource_id
LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
WHERE m.collection_id = $1
  AND (
    (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3))
    OR EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    )
  )
  AND r.status = 'active'
  AND ($4::text = '' OR r.resource_kind = $4)
  AND ($5::text = '' OR r.source_type = $5)
  AND ($6::text = '' OR r.project_id = $6)
  AND (
    $7::text = ''
    OR sd.search_vector @@ plainto_tsquery('simple', $7::text)
    OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($7::text) || '%'
  )
  AND (
    cardinality($8::text[]) = 0
    OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $8::text[]
  )
	AND (
		cardinality($10::text[]) = 0
		OR NOT EXISTS (
      SELECT 1
      FROM unnest($10::text[]) AS metadata_filters(filter)
      CROSS JOIN LATERAL (
        SELECT split_part(metadata_filters.filter, ':', 1) AS path,
               split_part(metadata_filters.filter, ':', 2) AS operator,
               substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
      ) mf
      CROSS JOIN LATERAL (
        SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
               r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
      ) mv
      WHERE NOT (
        (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
        OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
        OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
        OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
        OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
      )
		)
	)
  AND ($11::timestamptz IS NULL OR r.created_at >= $11)
  AND ($12::timestamptz IS NULL OR r.created_at <= $12)
  AND (
    $13::text = ''
    OR $13::text = 'all'
    OR ($13::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_ready' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
    ))
    OR ($13::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_failed' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
    ))
  )
`+postgresResourceDescriptorFilterSQL("$14")+`
  AND (
    $9::text = ''
    OR $9::text = 'all'
    OR ($9::text = 'private' AND NOT EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
    OR ($9::text = 'public' AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
    ))
    OR ($9::text = 'shared' AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
    OR ($9::text = 'shared_by_me' AND NOT EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
    ) AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
  )`, collectionID, ownerUserID, ownerOrgID, kind, source, projectID, query, tagKeys, sharing, metadataFilterSpecs, nullableTimestamptz(input.CreatedAfter), nullableTimestamptz(input.CreatedBefore), processingStatus, descriptorFilters).Scan(&total); err != nil {
		return domain.ResourceListPage{}, mapPgError(err)
	}
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	rows, err := s.pool.Query(ctx, `
SELECT r.resource_id, r.owner_user_id, COALESCE(r.owner_org_id, ''), COALESCE(r.owner_role, ''),
       r.original_name, COALESCE(r.content_type, ''), r.size_bytes, COALESCE(r.sha256, ''),
       COALESCE(r.storage_uri, ''), COALESCE(r.storage_path, ''), r.source_type, r.resource_kind,
       COALESCE(r.source_uri, ''), COALESCE(r.project_id, ''), r.status, r.created_at, r.updated_at,
       r.deleted_at, r.retention_expires_at, r.metadata,
       CASE
         WHEN EXISTS (
           SELECT 1
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
         ) THEN 'public'
         WHEN (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)) AND EXISTS (
           SELECT 1
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
         ) THEN 'shared_by_me'
         WHEN NOT (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)) AND EXISTS (
           SELECT 1
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND (
               COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
               OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
               OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
             )
         ) THEN 'shared_with_me'
         ELSE 'private'
       END AS share_status,
       CASE
         WHEN EXISTS (
           SELECT 1
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
         ) THEN (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
         )
         WHEN r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3) THEN (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
         )
         ELSE (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND (
               COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
               OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
               OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
             )
         )
       END AS active_grant_count,
       NOT EXISTS (
         SELECT 1
         FROM control_resource_share_grants g
         WHERE g.resource_id = r.resource_id
           AND g.status = 'active'
           AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
       ) AND (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)) AND EXISTS (
         SELECT 1
         FROM control_resource_share_grants g
         WHERE g.resource_id = r.resource_id
           AND g.status = 'active'
       ) AS shared_by_me,
       NOT (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)) AND EXISTS (
         SELECT 1
         FROM control_resource_share_grants g
         WHERE g.resource_id = r.resource_id
           AND g.status = 'active'
           AND (
             COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
             OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
             OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
           )
       ) AS shared_with_me
FROM control_resource_collection_members m
JOIN control_resources r ON r.resource_id = m.resource_id
LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
WHERE m.collection_id = $1
  AND (
    (r.owner_user_id = $2 AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3))
    OR EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
          OR (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    )
  )
  AND r.status = 'active'
  AND ($4::text = '' OR r.resource_kind = $4)
  AND ($5::text = '' OR r.source_type = $5)
  AND ($6::text = '' OR r.project_id = $6)
  AND (
    $7::text = ''
    OR sd.search_vector @@ plainto_tsquery('simple', $7::text)
    OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($7::text) || '%'
  )
  AND (
    cardinality($8::text[]) = 0
    OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $8::text[]
  )
	AND (
		cardinality($10::text[]) = 0
		OR NOT EXISTS (
      SELECT 1
      FROM unnest($10::text[]) AS metadata_filters(filter)
      CROSS JOIN LATERAL (
        SELECT split_part(metadata_filters.filter, ':', 1) AS path,
               split_part(metadata_filters.filter, ':', 2) AS operator,
               substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
      ) mf
      CROSS JOIN LATERAL (
        SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
               r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
      ) mv
      WHERE NOT (
        (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
        OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
        OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
        OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
        OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
      )
		)
	)
  AND ($11::timestamptz IS NULL OR r.created_at >= $11)
  AND ($12::timestamptz IS NULL OR r.created_at <= $12)
  AND (
    $13::text = ''
    OR $13::text = 'all'
    OR ($13::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_ready' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
    ))
    OR ($13::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_failed' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
    ))
  )
`+postgresResourceDescriptorFilterSQL("$14")+`
  AND (
    $9::text = ''
    OR $9::text = 'all'
    OR ($9::text = 'private' AND NOT EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
    OR ($9::text = 'public' AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
    ))
    OR ($9::text = 'shared' AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
    OR ($9::text = 'shared_by_me' AND NOT EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND COALESCE(g.grantee_user_id, '') = '`+domain.PublicResourceGranteeUserID+`'
    ) AND EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
    ))
  )
ORDER BY m.position ASC, m.added_at ASC, m.resource_id ASC
LIMIT $15 OFFSET $16`, collectionID, ownerUserID, ownerOrgID, kind, source, projectID, query, tagKeys, sharing, metadataFilterSpecs, nullableTimestamptz(input.CreatedAfter), nullableTimestamptz(input.CreatedBefore), processingStatus, descriptorFilters, limit, offset)
	if err != nil {
		return domain.ResourceListPage{}, mapPgError(err)
	}
	defer rows.Close()
	resources, err := scanResourceRowsWithShareSummary(rows)
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	return domain.ResourceListPage{
		Resources:  resources,
		TotalCount: total,
		Limit:      int(limit),
		Offset:     int(offset),
	}, nil
}

func (s *PostgresStore) CreateDatasetSnapshot(ctx context.Context, input domain.CreateDatasetSnapshotInput) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 && input.ResourceQuery == nil {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	snapshotID := strings.TrimSpace(input.SnapshotID)
	if snapshotID == "" {
		snapshotID = domain.NewID("dataset_snapshot")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		name = snapshotID
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if len(resourceIDs) == 0 && input.ResourceQuery != nil {
		resourceIDs, err = s.datasetSnapshotResourceIDsForQueryTx(ctx, tx, input)
		if err != nil {
			return domain.DatasetSnapshotRecord{}, nil, err
		}
		if len(resourceIDs) == 0 {
			return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
		}
	}

	rows, err := tx.Query(ctx, `
SELECT resource_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       original_name, COALESCE(content_type, ''), size_bytes, COALESCE(sha256, ''),
       COALESCE(storage_uri, ''), COALESCE(storage_path, ''), source_type, resource_kind,
       COALESCE(source_uri, ''), COALESCE(project_id, ''), status, created_at, updated_at,
       deleted_at, retention_expires_at, metadata
FROM control_resources
WHERE resource_id = ANY($1::text[])
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'`,
		resourceIDs,
		ownerUserID,
		ownerOrgID,
	)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	resources, err := scanResourceRows(rows)
	rows.Close()
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	resourcesByID := make(map[string]domain.ResourceRecord, len(resources))
	for _, resource := range resources {
		resourcesByID[resource.ResourceID] = resource
	}
	if len(resourcesByID) != len(resourceIDs) {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	var totalBytes int64
	for _, resourceID := range resourceIDs {
		totalBytes += resourcesByID[resourceID].SizeBytes
	}
	projectID := strings.TrimSpace(input.ProjectID)
	if projectID == "" && input.ResourceQuery != nil {
		projectID = strings.TrimSpace(input.ResourceQuery.ProjectID)
	}

	snapshot, err := scanDatasetSnapshotRow(tx.QueryRow(ctx, `
INSERT INTO control_dataset_snapshots (
  snapshot_id, owner_user_id, owner_org_id, owner_role, project_id, source_collection_id,
  name, description, status, resource_count, total_bytes, created_by_user_id, created_at, metadata
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''),
        $7, NULLIF($8, ''), 'active', $9, $10, NULLIF($11, ''), $12, $13)
RETURNING snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), COALESCE(source_collection_id, ''), name,
          COALESCE(description, ''), status, resource_count, total_bytes,
          COALESCE(created_by_user_id, ''), created_at, metadata`,
		snapshotID,
		ownerUserID,
		ownerOrgID,
		strings.TrimSpace(input.OwnerRole),
		projectID,
		strings.TrimSpace(input.SourceCollectionID),
		name,
		strings.TrimSpace(input.Description),
		int64(len(resourceIDs)),
		totalBytes,
		createdByUserID,
		createdAt.UTC(),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}

	entries := make([]domain.DatasetSnapshotResourceRecord, 0, len(resourceIDs))
	for position, resourceID := range resourceIDs {
		resource := resourcesByID[resourceID]
		entry, err := scanDatasetSnapshotResourceRow(tx.QueryRow(ctx, `
INSERT INTO control_dataset_snapshot_resources (
  snapshot_id, resource_id, position, original_name, content_type, size_bytes, sha256,
  source_type, resource_kind, storage_uri, source_uri, project_id, resource_created_at, metadata
)
VALUES ($1, $2, $3, $4, NULLIF($5, ''), $6, NULLIF($7, ''),
        $8, $9, NULLIF($10, ''), NULLIF($11, ''), NULLIF($12, ''), $13, $14)
RETURNING snapshot_id, resource_id, position, original_name, COALESCE(content_type, ''),
          size_bytes, COALESCE(sha256, ''), source_type, resource_kind, COALESCE(storage_uri, ''),
          COALESCE(source_uri, ''), COALESCE(project_id, ''), resource_created_at, metadata`,
			snapshot.SnapshotID,
			resource.ResourceID,
			int64(position),
			resource.OriginalName,
			resource.ContentType,
			resource.SizeBytes,
			resource.SHA256,
			resource.SourceType,
			resource.ResourceKind,
			resource.StorageURI,
			resource.SourceURI,
			resource.ProjectID,
			resource.CreatedAt.UTC(),
			jsonBytes(resource.Metadata),
		))
		if err != nil {
			return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
		}
		entries = append(entries, entry)
	}
	if _, err := insertDatasetSnapshotEventTx(ctx, tx, domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: createdByUserID,
		ActorOrgID:  ownerOrgID,
		EventType:   "dataset_snapshot.created",
		TS:          snapshot.CreatedAt,
		Metadata: domain.JSONMap{
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               snapshot.Metadata["source"],
		},
	}); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	return snapshot, entries, nil
}

func (s *PostgresStore) datasetSnapshotResourceIDsForQueryTx(ctx context.Context, tx pgx.Tx, input domain.CreateDatasetSnapshotInput) ([]string, error) {
	if input.ResourceQuery == nil {
		return nil, nil
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	querySelector := input.ResourceQuery
	query := strings.TrimSpace(querySelector.Query)
	kind := strings.ToLower(strings.TrimSpace(querySelector.Kind))
	source := strings.ToLower(strings.TrimSpace(querySelector.Source))
	projectID := strings.TrimSpace(querySelector.ProjectID)
	if projectID == "" {
		projectID = strings.TrimSpace(input.ProjectID)
	}
	sharing := strings.ToLower(strings.TrimSpace(querySelector.Sharing))
	tagKeys := resourceTagKeys(querySelector.Tags)
	descriptorFilters := normalizeResourceDescriptors(querySelector.Descriptors)
	metadataFilterSpecs := resourceMetadataFilterSpecs(querySelector.MetadataFilters)
	processingStatus := strings.ToLower(strings.TrimSpace(querySelector.ProcessingStatus))
	sourceCollectionID := strings.TrimSpace(input.SourceCollectionID)
	var rows pgx.Rows
	var err error
	if sourceCollectionID != "" {
		if _, err := scanResourceCollectionRow(tx.QueryRow(ctx, `
SELECT c.collection_id, c.owner_user_id, COALESCE(c.owner_org_id, ''), COALESCE(c.owner_role, ''),
       COALESCE(c.project_id, ''), COALESCE(c.parent_collection_id, ''), c.name,
       COALESCE(c.description, ''), c.collection_type, c.status,
       COUNT(m.resource_id) FILTER (WHERE r.status = 'active')::bigint AS resource_count,
       c.created_at, c.updated_at, c.metadata
FROM control_resource_collections c
LEFT JOIN control_resource_collection_members m ON m.collection_id = c.collection_id
LEFT JOIN control_resources r ON r.resource_id = m.resource_id
WHERE c.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
  AND c.status = 'active'
GROUP BY c.collection_id`, sourceCollectionID, ownerUserID, ownerOrgID)); err != nil {
			return nil, mapPgError(err)
		}
		rows, err = tx.Query(ctx, `
SELECT r.resource_id
FROM control_resource_collection_members m
JOIN control_resources r ON r.resource_id = m.resource_id
LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
WHERE m.collection_id = $1
  AND r.owner_user_id = $2
  AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)
  AND r.status = 'active'
  AND ($4::text = '' OR r.resource_kind = $4)
  AND ($5::text = '' OR r.source_type = $5)
  AND ($6::text = '' OR COALESCE(r.project_id, '') = $6)
  AND (
    $7::text = ''
    OR sd.search_vector @@ plainto_tsquery('simple', $7::text)
    OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($7::text) || '%'
  )
  AND (
    cardinality($8::text[]) = 0
    OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $8::text[]
  )
  AND (
    cardinality($10::text[]) = 0
    OR NOT EXISTS (
      SELECT 1
      FROM unnest($10::text[]) AS metadata_filters(filter)
      CROSS JOIN LATERAL (
        SELECT split_part(metadata_filters.filter, ':', 1) AS path,
               split_part(metadata_filters.filter, ':', 2) AS operator,
               substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
      ) mf
      CROSS JOIN LATERAL (
        SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
               r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
      ) mv
      WHERE NOT (
        (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
        OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
        OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
        OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
        OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
      )
    )
  )
  AND ($11::timestamptz IS NULL OR r.created_at >= $11)
  AND ($12::timestamptz IS NULL OR r.created_at <= $12)
  AND (
    $13::text = ''
    OR $13::text = 'all'
    OR ($13::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_ready' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
    ))
    OR ($13::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($13::text = 'data_agent_failed' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
    ))
  )
`+postgresResourceDescriptorFilterSQL("$14")+`
  AND (
    $9::text = ''
    OR $9::text = 'all'
    OR ($9::text = 'private' AND NOT EXISTS (
      SELECT 1 FROM control_resource_share_grants g WHERE g.resource_id = r.resource_id AND g.status = 'active'
    ))
    OR ($9::text IN ('shared', 'shared_by_me') AND EXISTS (
      SELECT 1 FROM control_resource_share_grants g WHERE g.resource_id = r.resource_id AND g.status = 'active'
    ))
  )
ORDER BY m.position ASC, m.added_at ASC, m.resource_id ASC`,
			sourceCollectionID,
			ownerUserID,
			ownerOrgID,
			kind,
			source,
			projectID,
			query,
			tagKeys,
			sharing,
			metadataFilterSpecs,
			nullableTimestamptz(querySelector.CreatedAfter),
			nullableTimestamptz(querySelector.CreatedBefore),
			processingStatus,
			descriptorFilters,
		)
	} else {
		rows, err = tx.Query(ctx, `
SELECT r.resource_id
FROM control_resources r
LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
WHERE r.owner_user_id = $1
  AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
  AND r.status = 'active'
  AND ($3::text = '' OR r.resource_kind = $3)
  AND ($4::text = '' OR r.source_type = $4)
  AND ($5::text = '' OR COALESCE(r.project_id, '') = $5)
  AND (
    $6::text = ''
    OR sd.search_vector @@ plainto_tsquery('simple', $6::text)
    OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($6::text) || '%'
  )
  AND (
    cardinality($7::text[]) = 0
    OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $7::text[]
  )
  AND (
    cardinality($9::text[]) = 0
    OR NOT EXISTS (
      SELECT 1
      FROM unnest($9::text[]) AS metadata_filters(filter)
      CROSS JOIN LATERAL (
        SELECT split_part(metadata_filters.filter, ':', 1) AS path,
               split_part(metadata_filters.filter, ':', 2) AS operator,
               substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
      ) mf
      CROSS JOIN LATERAL (
        SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
               r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
      ) mv
      WHERE NOT (
        (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
        OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
        OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
        OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
        OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
        OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
      )
    )
  )
  AND ($10::timestamptz IS NULL OR r.created_at >= $10)
  AND ($11::timestamptz IS NULL OR r.created_at <= $11)
  AND (
    $12::text = ''
    OR $12::text = 'all'
    OR ($12::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
    OR ($12::text = 'data_agent_ready' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
    ))
    OR ($12::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($12::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
    OR ($12::text = 'data_agent_failed' AND (
      lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
      OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
    ))
  )
`+postgresResourceDescriptorFilterSQL("$13")+`
  AND (
    $8::text = ''
    OR $8::text = 'all'
    OR ($8::text = 'private' AND NOT EXISTS (
      SELECT 1 FROM control_resource_share_grants g WHERE g.resource_id = r.resource_id AND g.status = 'active'
    ))
    OR ($8::text IN ('shared', 'shared_by_me') AND EXISTS (
      SELECT 1 FROM control_resource_share_grants g WHERE g.resource_id = r.resource_id AND g.status = 'active'
    ))
  )
ORDER BY r.created_at DESC, r.resource_id ASC`,
			ownerUserID,
			ownerOrgID,
			kind,
			source,
			projectID,
			query,
			tagKeys,
			sharing,
			metadataFilterSpecs,
			nullableTimestamptz(querySelector.CreatedAfter),
			nullableTimestamptz(querySelector.CreatedBefore),
			processingStatus,
			descriptorFilters,
		)
	}
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	resourceIDs := []string{}
	for rows.Next() {
		var resourceID string
		if err := rows.Scan(&resourceID); err != nil {
			return nil, err
		}
		resourceIDs = append(resourceIDs, resourceID)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return resourceIDs, nil
}

func (s *PostgresStore) GetDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	snapshot, err := scanDatasetSnapshotRow(s.pool.QueryRow(ctx, `
SELECT snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), COALESCE(source_collection_id, ''), name,
       COALESCE(description, ''), status, resource_count, total_bytes,
       COALESCE(created_by_user_id, ''), created_at, metadata
FROM control_dataset_snapshots s
WHERE s.snapshot_id = $1
  AND s.status = 'active'
  AND (
    (s.owner_user_id = $2 AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $3))
    OR EXISTS (
      SELECT 1
      FROM control_dataset_snapshot_share_grants g
      WHERE g.snapshot_id = s.snapshot_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    )
  )`,
		strings.TrimSpace(snapshotID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
	))
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT snapshot_id, resource_id, position, original_name, COALESCE(content_type, ''),
       size_bytes, COALESCE(sha256, ''), source_type, resource_kind, COALESCE(storage_uri, ''),
       COALESCE(source_uri, ''), COALESCE(project_id, ''), resource_created_at, metadata
FROM control_dataset_snapshot_resources
WHERE snapshot_id = $1
ORDER BY position ASC, resource_id ASC`, snapshot.SnapshotID)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	defer rows.Close()
	entries, err := scanDatasetSnapshotResourceRows(rows)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	return snapshot, entries, nil
}

func (s *PostgresStore) SoftDeleteDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string, deletedAt time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	snapshot, err := scanDatasetSnapshotRow(tx.QueryRow(ctx, `
UPDATE control_dataset_snapshots
SET status = 'deleted'
WHERE snapshot_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status <> 'deleted'
RETURNING snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), COALESCE(source_collection_id, ''), name,
          COALESCE(description, ''), status, resource_count, total_bytes,
          COALESCE(created_by_user_id, ''), created_at, metadata`,
		strings.TrimSpace(snapshotID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
	))
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	rows, err := tx.Query(ctx, `
SELECT snapshot_id, resource_id, position, original_name, COALESCE(content_type, ''),
       size_bytes, COALESCE(sha256, ''), source_type, resource_kind, COALESCE(storage_uri, ''),
       COALESCE(source_uri, ''), COALESCE(project_id, ''), resource_created_at, metadata
FROM control_dataset_snapshot_resources
WHERE snapshot_id = $1
ORDER BY position ASC, resource_id ASC`, snapshot.SnapshotID)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	entries, err := scanDatasetSnapshotResourceRows(rows)
	rows.Close()
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	if _, err := insertDatasetSnapshotEventTx(ctx, tx, domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: strings.TrimSpace(userID),
		ActorOrgID:  strings.TrimSpace(orgID),
		EventType:   "dataset_snapshot.deleted",
		TS:          afterEventTime(latestDatasetSnapshotEventTimeTx(ctx, tx, snapshot.SnapshotID, snapshot.CreatedAt), deletedAt),
		Metadata: domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               "dataset_snapshot_lifecycle",
			"deleted_at":           deletedAt.UTC().Format(time.RFC3339Nano),
		},
	}); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	return snapshot, entries, nil
}

func (s *PostgresStore) RestoreDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string, restoredAt time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	snapshot, err := scanDatasetSnapshotRow(tx.QueryRow(ctx, `
UPDATE control_dataset_snapshots
SET status = 'active'
WHERE snapshot_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
RETURNING snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), COALESCE(source_collection_id, ''), name,
          COALESCE(description, ''), status, resource_count, total_bytes,
          COALESCE(created_by_user_id, ''), created_at, metadata`,
		strings.TrimSpace(snapshotID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
	))
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	rows, err := tx.Query(ctx, `
SELECT snapshot_id, resource_id, position, original_name, COALESCE(content_type, ''),
       size_bytes, COALESCE(sha256, ''), source_type, resource_kind, COALESCE(storage_uri, ''),
       COALESCE(source_uri, ''), COALESCE(project_id, ''), resource_created_at, metadata
FROM control_dataset_snapshot_resources
WHERE snapshot_id = $1
ORDER BY position ASC, resource_id ASC`, snapshot.SnapshotID)
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	entries, err := scanDatasetSnapshotResourceRows(rows)
	rows.Close()
	if err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	if _, err := insertDatasetSnapshotEventTx(ctx, tx, domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: strings.TrimSpace(userID),
		ActorOrgID:  strings.TrimSpace(orgID),
		EventType:   "dataset_snapshot.restored",
		TS:          afterEventTime(latestDatasetSnapshotEventTimeTx(ctx, tx, snapshot.SnapshotID, snapshot.CreatedAt), restoredAt),
		Metadata: domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               "dataset_snapshot_lifecycle",
			"restored_at":          restoredAt.UTC().Format(time.RFC3339Nano),
		},
	}); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DatasetSnapshotRecord{}, nil, err
	}
	return snapshot, entries, nil
}

func (s *PostgresStore) ListDatasetSnapshotsForUser(ctx context.Context, input domain.DatasetSnapshotListInput) (domain.DatasetSnapshotListPage, error) {
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	ownerUserID := strings.TrimSpace(input.UserID)
	ownerOrgID := strings.TrimSpace(input.OrgID)
	projectID := strings.TrimSpace(input.ProjectID)
	sourceCollectionID := strings.TrimSpace(input.SourceCollectionID)
	query := strings.TrimSpace(input.Query)
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	countRow := s.pool.QueryRow(ctx, `
SELECT COUNT(*)
FROM control_dataset_snapshots s
WHERE s.status = $3
  AND (
    (s.owner_user_id = $1 AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $2))
    OR ($3::text = 'active' AND EXISTS (
      SELECT 1
      FROM control_dataset_snapshot_share_grants g
      WHERE g.snapshot_id = s.snapshot_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $1 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $2)
        )
    ))
  )
  AND ($4::text = '' OR COALESCE(s.project_id, '') = $4)
  AND ($5::text = '' OR COALESCE(s.source_collection_id, '') = $5)
  AND ($6::text = '' OR s.name ILIKE '%' || $6 || '%' OR COALESCE(s.description, '') ILIKE '%' || $6 || '%' OR s.snapshot_id ILIKE '%' || $6 || '%' OR s.metadata::text ILIKE '%' || $6 || '%')`,
		ownerUserID,
		ownerOrgID,
		status,
		projectID,
		sourceCollectionID,
		query,
	)
	var total int
	if err := countRow.Scan(&total); err != nil {
		return domain.DatasetSnapshotListPage{}, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), COALESCE(source_collection_id, ''), name,
       COALESCE(description, ''), status, resource_count, total_bytes,
       COALESCE(created_by_user_id, ''), created_at, metadata
FROM control_dataset_snapshots s
WHERE s.status = $3
  AND (
    (s.owner_user_id = $1 AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $2))
    OR ($3::text = 'active' AND EXISTS (
      SELECT 1
      FROM control_dataset_snapshot_share_grants g
      WHERE g.snapshot_id = s.snapshot_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $1 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $2)
        )
    ))
  )
  AND ($4::text = '' OR COALESCE(s.project_id, '') = $4)
  AND ($5::text = '' OR COALESCE(s.source_collection_id, '') = $5)
  AND ($6::text = '' OR s.name ILIKE '%' || $6 || '%' OR COALESCE(s.description, '') ILIKE '%' || $6 || '%' OR s.snapshot_id ILIKE '%' || $6 || '%' OR s.metadata::text ILIKE '%' || $6 || '%')
ORDER BY s.created_at DESC, s.snapshot_id ASC
LIMIT $7 OFFSET $8`,
		ownerUserID,
		ownerOrgID,
		status,
		projectID,
		sourceCollectionID,
		query,
		limit,
		offset,
	)
	if err != nil {
		return domain.DatasetSnapshotListPage{}, mapPgError(err)
	}
	defer rows.Close()
	snapshots, err := scanDatasetSnapshotRows(rows)
	if err != nil {
		return domain.DatasetSnapshotListPage{}, err
	}
	return domain.DatasetSnapshotListPage{
		Snapshots:  snapshots,
		TotalCount: total,
		Limit:      int(limit),
		Offset:     int(offset),
	}, nil
}

func (s *PostgresStore) CreateDatasetSnapshotShareGrant(ctx context.Context, input domain.CreateDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error) {
	snapshotID := strings.TrimSpace(input.SnapshotID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("dataset_snapshot_grant")
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var canonicalOwnerUserID string
	var canonicalOwnerOrgID string
	var canonicalOwnerRole string
	var canonicalSnapshotCreatedAt time.Time
	if err := tx.QueryRow(ctx, `
SELECT owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''), created_at
FROM control_dataset_snapshots
WHERE snapshot_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'
FOR UPDATE`, snapshotID, ownerUserID, ownerOrgID).Scan(&canonicalOwnerUserID, &canonicalOwnerOrgID, &canonicalOwnerRole, &canonicalSnapshotCreatedAt); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}

	grant, err := scanDatasetSnapshotShareGrantRow(tx.QueryRow(ctx, `
INSERT INTO control_dataset_snapshot_share_grants (
  grant_id, snapshot_id, owner_user_id, owner_org_id, owner_role,
  grantee_user_id, grantee_org_id, role, status, created_by_user_id,
  created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING grant_id, snapshot_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(grantee_user_id, ''), COALESCE(grantee_org_id, ''), role, status,
          COALESCE(created_by_user_id, ''), created_at, updated_at, revoked_at, metadata`,
		grantID,
		snapshotID,
		canonicalOwnerUserID,
		nullableText(canonicalOwnerOrgID),
		nullableText(canonicalOwnerRole),
		nullableText(granteeUserID),
		nullableText(granteeOrgID),
		role,
		status,
		nullableText(createdByUserID),
		timestamptz(createdAt),
		timestamptz(updatedAt),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}
	if _, err := insertDatasetSnapshotEventTx(ctx, tx, domain.DatasetSnapshotEventRecord{
		SnapshotID:  grant.SnapshotID,
		ActorUserID: createdByUserID,
		ActorOrgID:  grant.OwnerOrgID,
		EventType:   "dataset_snapshot.shared",
		TS:          afterEventTime(latestDatasetSnapshotEventTimeTx(ctx, tx, grant.SnapshotID, canonicalSnapshotCreatedAt), grant.CreatedAt),
		Metadata: domain.JSONMap{
			"grant_id":        grant.GrantID,
			"grantee_user_id": grant.GranteeUserID,
			"grantee_org_id":  grant.GranteeOrgID,
			"role":            grant.Role,
		},
	}); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, err
	}
	return grant, nil
}

func (s *PostgresStore) ListDatasetSnapshotShareGrants(ctx context.Context, input domain.ListDatasetSnapshotShareGrantsInput) ([]domain.DatasetSnapshotShareGrantRecord, error) {
	snapshotID := strings.TrimSpace(input.SnapshotID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	status := strings.TrimSpace(input.Status)
	var exists int
	if err := s.pool.QueryRow(ctx, `
SELECT 1
FROM control_dataset_snapshots
WHERE snapshot_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'`, snapshotID, ownerUserID, ownerOrgID).Scan(&exists); err != nil {
		return nil, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT g.grant_id, g.snapshot_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
       COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
       COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata
FROM control_dataset_snapshot_share_grants g
JOIN control_dataset_snapshots s ON s.snapshot_id = g.snapshot_id
WHERE g.snapshot_id = $1
  AND s.owner_user_id = $2
  AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $3)
  AND s.status = 'active'
  AND ($4::text = '' OR g.status = $4)
ORDER BY g.created_at DESC, g.grant_id ASC
LIMIT $5`, snapshotID, ownerUserID, ownerOrgID, status, limit32(input.Limit, 200))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	grants := make([]domain.DatasetSnapshotShareGrantRecord, 0)
	for rows.Next() {
		grant, err := scanDatasetSnapshotShareGrantRow(rows)
		if err != nil {
			return nil, err
		}
		grants = append(grants, grant)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return grants, nil
}

func (s *PostgresStore) RevokeDatasetSnapshotShareGrant(ctx context.Context, input domain.RevokeDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error) {
	snapshotID := strings.TrimSpace(input.SnapshotID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	grantID := strings.TrimSpace(input.GrantID)
	revokedAt := input.RevokedAt
	if revokedAt.IsZero() {
		revokedAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	grant, err := scanDatasetSnapshotShareGrantRow(tx.QueryRow(ctx, `
UPDATE control_dataset_snapshot_share_grants AS g
SET status = 'revoked',
    revoked_at = $5,
    updated_at = $5,
    metadata = CASE
      WHEN $6::text = '' AND $7::text = '' THEN g.metadata
      ELSE g.metadata
        || CASE WHEN $6::text = '' THEN '{}'::jsonb ELSE jsonb_build_object('revoked_by_user_id', $6::text) END
        || CASE WHEN $7::text = '' THEN '{}'::jsonb ELSE jsonb_build_object('revocation_reason', $7::text) END
    END
FROM control_dataset_snapshots s
WHERE g.snapshot_id = s.snapshot_id
  AND g.grant_id = $1
  AND g.snapshot_id = $2
  AND s.owner_user_id = $3
  AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $4)
  AND s.status = 'active'
  AND g.status = 'active'
RETURNING g.grant_id, g.snapshot_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
          COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
          COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata`,
		grantID,
		snapshotID,
		ownerUserID,
		ownerOrgID,
		timestamptz(revokedAt),
		strings.TrimSpace(input.RevokedByUserID),
		strings.TrimSpace(input.RevocationReason),
	))
	if err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}
	previousEventTS := latestDatasetSnapshotEventTimeTx(ctx, tx, grant.SnapshotID, grant.CreatedAt)
	actorUserID := strings.TrimSpace(input.RevokedByUserID)
	if actorUserID == "" {
		actorUserID = ownerUserID
	}
	if _, err := insertDatasetSnapshotEventTx(ctx, tx, domain.DatasetSnapshotEventRecord{
		SnapshotID:  grant.SnapshotID,
		ActorUserID: actorUserID,
		ActorOrgID:  grant.OwnerOrgID,
		EventType:   "dataset_snapshot.share_revoked",
		TS:          afterEventTime(previousEventTS, grant.RevokedAt),
		Metadata: domain.JSONMap{
			"grant_id":        grant.GrantID,
			"grantee_user_id": grant.GranteeUserID,
			"grantee_org_id":  grant.GranteeOrgID,
			"role":            grant.Role,
		},
	}); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, err
	}
	return grant, nil
}

func (s *PostgresStore) ListDatasetSnapshotEventsForUser(ctx context.Context, input domain.DatasetSnapshotEventListInput) (domain.DatasetSnapshotEventListPage, error) {
	snapshotID := strings.TrimSpace(input.SnapshotID)
	userID := strings.TrimSpace(input.UserID)
	orgID := strings.TrimSpace(input.OrgID)
	eventType := strings.TrimSpace(input.EventType)
	actorUserID := strings.TrimSpace(input.ActorUserID)
	var visible int
	if err := s.pool.QueryRow(ctx, `
SELECT 1
FROM control_dataset_snapshots s
WHERE s.snapshot_id = $1
  AND (
    (s.owner_user_id = $2 AND (COALESCE(s.owner_org_id, '') = '' OR s.owner_org_id = $3))
    OR (s.status = 'active' AND EXISTS (
      SELECT 1
      FROM control_dataset_snapshot_share_grants g
      WHERE g.snapshot_id = s.snapshot_id
        AND g.status = 'active'
        AND g.role = 'read'
        AND (
          (COALESCE(g.grantee_user_id, '') <> '' AND g.grantee_user_id = $2 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3))
          OR (COALESCE(g.grantee_user_id, '') = '' AND COALESCE(g.grantee_org_id, '') <> '' AND g.grantee_org_id = $3)
        )
    ))
  )`, snapshotID, userID, orgID).Scan(&visible); err != nil {
		return domain.DatasetSnapshotEventListPage{}, mapPgError(err)
	}
	var total int
	if err := s.pool.QueryRow(ctx, `
SELECT COUNT(*)
FROM control_dataset_snapshot_events
WHERE snapshot_id = $1
  AND ($2::text = '' OR event_type = $2)
  AND ($3::text = '' OR actor_user_id = $3)`, snapshotID, eventType, actorUserID).Scan(&total); err != nil {
		return domain.DatasetSnapshotEventListPage{}, mapPgError(err)
	}
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	rows, err := s.pool.Query(ctx, `
SELECT event_id, snapshot_id, COALESCE(actor_user_id, ''), COALESCE(actor_org_id, ''),
       event_type, ts, metadata
FROM control_dataset_snapshot_events
WHERE snapshot_id = $1
  AND ($2::text = '' OR event_type = $2)
  AND ($3::text = '' OR actor_user_id = $3)
ORDER BY ts DESC, event_id ASC
LIMIT $4 OFFSET $5`, snapshotID, eventType, actorUserID, limit, offset)
	if err != nil {
		return domain.DatasetSnapshotEventListPage{}, mapPgError(err)
	}
	defer rows.Close()
	events := make([]domain.DatasetSnapshotEventRecord, 0)
	for rows.Next() {
		event, err := scanDatasetSnapshotEventRow(rows)
		if err != nil {
			return domain.DatasetSnapshotEventListPage{}, err
		}
		events = append(events, event)
	}
	if err := rows.Err(); err != nil {
		return domain.DatasetSnapshotEventListPage{}, mapPgError(err)
	}
	return domain.DatasetSnapshotEventListPage{
		Events:     events,
		TotalCount: total,
		Limit:      int(limit),
		Offset:     int(offset),
	}, nil
}

func (s *PostgresStore) CreateDataAgentJob(ctx context.Context, input domain.CreateDataAgentJobInput) (domain.DataAgentJobRecord, error) {
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	jobID := strings.TrimSpace(input.JobID)
	if jobID == "" {
		jobID = domain.NewID("data_agent_job")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status == "" {
		status = "queued"
	}
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	resourceCount := input.ResourceCount
	if len(resourceIDs) > 0 || resourceCount == 0 {
		resourceCount = len(resourceIDs)
	}
	progressTotal := input.ProgressTotal
	if progressTotal == 0 {
		progressTotal = resourceCount
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	inputSelector := cloneJSONMap(input.InputSelector)
	if len(inputSelector) == 0 && len(resourceIDs) > 0 {
		inputSelector["resource_ids"] = append([]string(nil), resourceIDs...)
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DataAgentJobRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if len(resourceIDs) > 0 {
		rows, err := tx.Query(ctx, `
SELECT resource_id
FROM control_resources
WHERE resource_id = ANY($1::text[])
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'`,
			resourceIDs,
			ownerUserID,
			ownerOrgID,
		)
		if err != nil {
			return domain.DataAgentJobRecord{}, mapPgError(err)
		}
		visible := map[string]struct{}{}
		for rows.Next() {
			var resourceID string
			if err := rows.Scan(&resourceID); err != nil {
				rows.Close()
				return domain.DataAgentJobRecord{}, err
			}
			visible[resourceID] = struct{}{}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return domain.DataAgentJobRecord{}, err
		}
		rows.Close()
		if len(visible) != len(resourceIDs) {
			return domain.DataAgentJobRecord{}, ErrNotFound
		}
	}

	job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
INSERT INTO control_data_agent_jobs (
  job_id, owner_user_id, owner_org_id, owner_role, project_id, job_type, status,
  resource_count, progress_completed, progress_total, error, created_by_user_id,
  created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), $6, $7,
        $8, $9, $10, NULLIF($11, ''), NULLIF($12, ''), $13, $14, $15, $16, $17, $18, $19)
RETURNING job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
          progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
          created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata`,
		jobID,
		ownerUserID,
		ownerOrgID,
		strings.TrimSpace(input.OwnerRole),
		strings.TrimSpace(input.ProjectID),
		jobType,
		status,
		int64(resourceCount),
		int64(input.ProgressCompleted),
		int64(progressTotal),
		strings.TrimSpace(input.Error),
		createdByUserID,
		createdAt.UTC(),
		updatedAt.UTC(),
		nullableTimestamptz(input.StartedAt),
		nullableTimestamptz(input.CompletedAt),
		jsonBytes(inputSelector),
		jsonBytes(input.OutputSummary),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, mapPgError(err)
	}

	for position, resourceID := range resourceIDs {
		if _, err := tx.Exec(ctx, `
INSERT INTO control_data_agent_job_resources (job_id, resource_id, position, metadata)
VALUES ($1, $2, $3, '{}')
ON CONFLICT (job_id, resource_id) DO NOTHING`,
			job.JobID,
			resourceID,
			int64(position),
		); err != nil {
			return domain.DataAgentJobRecord{}, mapPgError(err)
		}
	}

	if _, err := scanDataAgentJobEventRow(tx.QueryRow(ctx, `
INSERT INTO control_data_agent_job_events (
  event_id, job_id, sequence, event_type, actor_user_id, actor_org_id, ts, message, metadata
)
VALUES ($1, $2, 1, 'data_agent.job.created', NULLIF($3, ''), NULLIF($4, ''), $5,
        'Data Agent job queued.', $6)
RETURNING event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
          COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata`,
		domain.NewID("data_agent_job_event"),
		job.JobID,
		createdByUserID,
		ownerOrgID,
		job.CreatedAt,
		jsonBytes(domain.JSONMap{
			"job_type":       job.JobType,
			"resource_count": job.ResourceCount,
		}),
	)); err != nil {
		return domain.DataAgentJobRecord{}, mapPgError(err)
	}

	if err := tx.Commit(ctx); err != nil {
		return domain.DataAgentJobRecord{}, err
	}
	return job, nil
}

// LinkDataAgentJobResource records a resource against a data-agent job. Used by the
// batch-analysis worker to attach the OUTPUT resources it produces (io_role='output')
// to the job, alongside the input images recorded at creation. Idempotent: re-running
// a job (resume after restart) upserts the same row instead of duplicating.
func (s *PostgresStore) LinkDataAgentJobResource(ctx context.Context, input domain.LinkDataAgentJobResourceInput) error {
	jobID := strings.TrimSpace(input.JobID)
	resourceID := strings.TrimSpace(input.ResourceID)
	if jobID == "" || resourceID == "" {
		return ErrNotFound
	}
	ioRole := strings.ToLower(strings.TrimSpace(input.IORole))
	if ioRole == "" {
		ioRole = "input"
	}
	if _, err := s.pool.Exec(ctx, `
INSERT INTO control_data_agent_job_resources (job_id, resource_id, position, io_role, metadata)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (job_id, resource_id) DO UPDATE SET io_role = EXCLUDED.io_role, metadata = EXCLUDED.metadata`,
		jobID,
		resourceID,
		int64(input.Position),
		ioRole,
		jsonBytes(mapOrEmpty(input.Metadata)),
	); err != nil {
		return mapPgError(err)
	}
	return nil
}

func (s *PostgresStore) GetDataAgentJobForUser(ctx context.Context, jobID string, userID string, orgID string) (domain.DataAgentJobRecord, error) {
	job, err := scanDataAgentJobRow(s.pool.QueryRow(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)`,
		strings.TrimSpace(jobID),
		strings.TrimSpace(userID),
		strings.TrimSpace(orgID),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, mapPgError(err)
	}
	return job, nil
}

func (s *PostgresStore) ListDataAgentJobsForUser(ctx context.Context, input domain.DataAgentJobListInput) (domain.DataAgentJobListPage, error) {
	ownerUserID := strings.TrimSpace(input.UserID)
	ownerOrgID := strings.TrimSpace(input.OrgID)
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	status := strings.ToLower(strings.TrimSpace(input.Status))
	projectID := strings.TrimSpace(input.ProjectID)
	limit := limit32(input.Limit, 200)
	offset := offset32(input.Offset)
	countRow := s.pool.QueryRow(ctx, `
SELECT COUNT(*)
FROM control_data_agent_jobs
WHERE owner_user_id = $1
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $2)
  AND ($3::text = '' OR status = $3)
  AND ($4::text = '' OR job_type = $4)
  AND ($5::text = '' OR COALESCE(project_id, '') = $5)`,
		ownerUserID,
		ownerOrgID,
		status,
		jobType,
		projectID,
	)
	var total int
	if err := countRow.Scan(&total); err != nil {
		return domain.DataAgentJobListPage{}, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE owner_user_id = $1
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $2)
  AND ($3::text = '' OR status = $3)
  AND ($4::text = '' OR job_type = $4)
  AND ($5::text = '' OR COALESCE(project_id, '') = $5)
ORDER BY updated_at DESC, job_id ASC
LIMIT $6 OFFSET $7`,
		ownerUserID,
		ownerOrgID,
		status,
		jobType,
		projectID,
		limit,
		offset,
	)
	if err != nil {
		return domain.DataAgentJobListPage{}, mapPgError(err)
	}
	defer rows.Close()
	jobs, err := scanDataAgentJobRows(rows)
	if err != nil {
		return domain.DataAgentJobListPage{}, err
	}
	return domain.DataAgentJobListPage{
		Jobs:       jobs,
		TotalCount: total,
		Limit:      int(limit),
		Offset:     int(offset),
	}, nil
}

func (s *PostgresStore) UpdateDataAgentJob(ctx context.Context, input domain.UpdateDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if !validDataAgentJobStatus(status) {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, errors.New("invalid data agent job status")
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	existing, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
FOR UPDATE`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	progressCompleted := clampDataAgentProgress(input.ProgressCompleted, input.ProgressTotal)
	progressTotal := input.ProgressTotal
	if progressTotal <= 0 {
		progressTotal = existing.ProgressTotal
	}
	if progressTotal < progressCompleted {
		progressTotal = progressCompleted
	}
	startedAt := existing.StartedAt
	if !input.StartedAt.IsZero() {
		startedAt = input.StartedAt.UTC()
	} else if status == "running" && startedAt.IsZero() {
		startedAt = updatedAt.UTC()
	}
	completedAt := existing.CompletedAt
	if !input.CompletedAt.IsZero() {
		completedAt = input.CompletedAt.UTC()
	} else if dataAgentJobStatusIsTerminal(status) {
		completedAt = updatedAt.UTC()
	} else if !dataAgentJobStatusIsTerminal(status) {
		completedAt = time.Time{}
	}
	outputSummary := existing.OutputSummary
	if input.OutputSummary != nil {
		outputSummary = cloneJSONMap(input.OutputSummary)
	}
	// Only replace metadata when the update carries some: status/progress updates send an
	// empty map, and clobbering would drop create-time metadata (e.g. results_collection_id)
	// that downstream output registration relies on.
	metadata := existing.Metadata
	if len(input.Metadata) > 0 {
		metadata = cloneJSONMap(input.Metadata)
	}
	job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
UPDATE control_data_agent_jobs
SET status = $2,
    progress_completed = $3,
    progress_total = $4,
    error = NULLIF($5, ''),
    updated_at = $6,
    started_at = $7,
    completed_at = $8,
    output_summary = $9,
    metadata = $10
WHERE job_id = $1
RETURNING job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
          progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
          created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata`,
		existing.JobID,
		status,
		int64(progressCompleted),
		int64(progressTotal),
		strings.TrimSpace(input.Error),
		updatedAt.UTC(),
		nullableTimestamptz(startedAt),
		nullableTimestamptz(completedAt),
		jsonBytes(outputSummary),
		jsonBytes(metadata),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	eventMetadata := cloneJSONMap(input.EventMetadata)
	if len(eventMetadata) == 0 {
		eventMetadata["status"] = status
	}
	event, err := appendDataAgentJobEventTx(ctx, tx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   dataAgentJobStatusEventType(status),
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          updatedAt,
		Message:     strings.TrimSpace(input.Message),
		Metadata:    eventMetadata,
	})
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return job, event, nil
}

func (s *PostgresStore) ControlDataAgentJob(ctx context.Context, input domain.ControlDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	existing, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
FOR UPDATE`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	action := strings.ToLower(strings.TrimSpace(input.Action))
	reason := strings.TrimSpace(input.Reason)
	status := existing.Status
	progressCompleted := existing.ProgressCompleted
	errorText := existing.Error
	startedAt := existing.StartedAt
	completedAt := existing.CompletedAt
	eventType := ""
	switch action {
	case "cancel":
		status = "canceled"
		errorText = reason
		completedAt = ts.UTC()
		eventType = "data_agent.job.canceled"
	case "retry":
		status = "queued"
		progressCompleted = 0
		errorText = ""
		startedAt = time.Time{}
		completedAt = time.Time{}
		eventType = "data_agent.job.retried"
	default:
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, errors.New("data agent job control action must be cancel or retry")
	}
	job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
UPDATE control_data_agent_jobs
SET status = $2,
    progress_completed = $3,
    error = NULLIF($4, ''),
    updated_at = $5,
    started_at = $6,
    completed_at = $7
WHERE job_id = $1
RETURNING job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
          progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
          created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata`,
		existing.JobID,
		status,
		int64(progressCompleted),
		errorText,
		ts.UTC(),
		nullableTimestamptz(startedAt),
		nullableTimestamptz(completedAt),
	))
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	event, err := appendDataAgentJobEventTx(ctx, tx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   eventType,
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          ts,
		Message:     reason,
		Metadata:    cloneJSONMap(input.Metadata),
	})
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return job, event, nil
}

func (s *PostgresStore) AcquireDataAgentJobLease(ctx context.Context, input domain.AcquireDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
FOR UPDATE`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	))
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	if dataAgentJobStatusIsTerminal(job.Status) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	existingLease, err := scanDataAgentJobLeaseRow(tx.QueryRow(ctx, `
SELECT job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_data_agent_job_leases
WHERE job_id = $1
FOR UPDATE`, job.JobID))
	if err == nil && existingLease.LeaseExpiresAt.After(now) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrConflict
	}
	if err != nil && !errors.Is(err, ErrNotFound) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	leaseInput := domain.DataAgentJobLeaseRecord{
		JobID:          job.JobID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("lease"),
		LeaseExpiresAt: now.Add(positiveLeaseTTL(input.TTL)),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	lease, err := scanDataAgentJobLeaseRow(tx.QueryRow(ctx, `
INSERT INTO control_data_agent_job_leases (job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (job_id) DO UPDATE
SET worker_id = EXCLUDED.worker_id,
    lease_token = EXCLUDED.lease_token,
    lease_expires_at = EXCLUDED.lease_expires_at,
    updated_at = EXCLUDED.updated_at
RETURNING job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		leaseInput.JobID,
		leaseInput.WorkerID,
		leaseInput.LeaseToken,
		leaseInput.LeaseExpiresAt,
		leaseInput.CreatedAt,
		leaseInput.UpdatedAt,
	))
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	job, err = scanDataAgentJobRow(tx.QueryRow(ctx, `
UPDATE control_data_agent_jobs
SET status = 'running',
    updated_at = $2,
    started_at = COALESCE(started_at, $2)
WHERE job_id = $1
RETURNING job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
          progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
          created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata`,
		job.JobID,
		now,
	))
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	event, err := appendDataAgentJobEventTx(ctx, tx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.leased",
		ActorUserID: strings.TrimSpace(input.OwnerUserID),
		ActorOrgID:  strings.TrimSpace(input.OwnerOrgID),
		TS:          now,
		Message:     "Data Agent job leased.",
		Metadata: domain.JSONMap{
			"worker_id":        lease.WorkerID,
			"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
		},
	})
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return lease, job, event, nil
}

func (s *PostgresStore) RenewDataAgentJobLease(ctx context.Context, input domain.RenewDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var status string
	if err := tx.QueryRow(ctx, `
SELECT status
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
FOR UPDATE`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	).Scan(&status); err != nil {
		return domain.DataAgentJobLeaseRecord{}, mapPgError(err)
	}
	if dataAgentJobStatusIsTerminal(status) {
		return domain.DataAgentJobLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	existing, err := scanDataAgentJobLeaseRow(tx.QueryRow(ctx, `
SELECT job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_data_agent_job_leases
WHERE job_id = $1
FOR UPDATE`, strings.TrimSpace(input.JobID)))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.DataAgentJobLeaseRecord{}, ErrConflict
		}
		return domain.DataAgentJobLeaseRecord{}, err
	}
	if existing.LeaseToken != strings.TrimSpace(input.LeaseToken) || !existing.LeaseExpiresAt.After(now) {
		return domain.DataAgentJobLeaseRecord{}, ErrConflict
	}
	lease, err := scanDataAgentJobLeaseRow(tx.QueryRow(ctx, `
UPDATE control_data_agent_job_leases
SET lease_expires_at = $3,
    updated_at = $4
WHERE job_id = $1 AND lease_token = $2
RETURNING job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.LeaseToken),
		now.Add(positiveLeaseTTL(input.TTL)),
		now,
	))
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, mapPgError(err)
	}
	if _, err := tx.Exec(ctx, `UPDATE control_data_agent_jobs SET updated_at = $2 WHERE job_id = $1`, strings.TrimSpace(input.JobID), now); err != nil {
		return domain.DataAgentJobLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.DataAgentJobLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) ReleaseDataAgentJobLease(ctx context.Context, input domain.ReleaseDataAgentJobLeaseInput) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	jobID := strings.TrimSpace(input.JobID)
	if err := tx.QueryRow(ctx, `
SELECT job_id
FROM control_data_agent_jobs
WHERE job_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)`,
		jobID,
		strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID),
	).Scan(&jobID); err != nil {
		return mapPgError(err)
	}
	tag, err := tx.Exec(ctx, `DELETE FROM control_data_agent_job_leases WHERE job_id = $1 AND lease_token = $2`, jobID, strings.TrimSpace(input.LeaseToken))
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() > 0 {
		return tx.Commit(ctx)
	}
	var activeToken string
	err = tx.QueryRow(ctx, `SELECT lease_token FROM control_data_agent_job_leases WHERE job_id = $1`, jobID).Scan(&activeToken)
	if errors.Is(err, pgx.ErrNoRows) {
		return tx.Commit(ctx)
	}
	if err != nil {
		return mapPgError(err)
	}
	return ErrConflict
}

func (s *PostgresStore) RecoverExpiredDataAgentJobLeases(ctx context.Context, input domain.RecoverExpiredDataAgentJobLeasesInput) (domain.RecoverExpiredDataAgentJobLeasesResult, error) {
	now := input.Now
	if now.IsZero() {
		now = domain.Now()
	}
	reason := strings.TrimSpace(input.Reason)
	if reason == "" {
		reason = "automatic expired data-agent lease recovery"
	}
	limit := input.Limit
	if limit <= 0 {
		limit = 1000
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RecoverExpiredDataAgentJobLeasesResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	rows, err := tx.Query(ctx, `
SELECT j.job_id
FROM control_data_agent_jobs j
JOIN control_data_agent_job_leases l ON l.job_id = j.job_id
WHERE j.status IN ('queued', 'running')
ORDER BY j.updated_at ASC, j.job_id ASC
LIMIT $1
FOR UPDATE OF j, l`, int32(limit))
	if err != nil {
		return domain.RecoverExpiredDataAgentJobLeasesResult{}, mapPgError(err)
	}
	jobIDs := []string{}
	for rows.Next() {
		var jobID string
		if err := rows.Scan(&jobID); err != nil {
			rows.Close()
			return domain.RecoverExpiredDataAgentJobLeasesResult{}, mapPgError(err)
		}
		jobIDs = append(jobIDs, jobID)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return domain.RecoverExpiredDataAgentJobLeasesResult{}, mapPgError(err)
	}
	rows.Close()
	result := domain.RecoverExpiredDataAgentJobLeasesResult{Checked: len(jobIDs)}
	for _, jobID := range jobIDs {
		job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
	SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1`, jobID))
		if err != nil {
			return result, mapPgError(err)
		}
		lease, err := scanDataAgentJobLeaseRow(tx.QueryRow(ctx, `
SELECT job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_data_agent_job_leases
WHERE job_id = $1`, job.JobID))
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				continue
			}
			return result, err
		}
		if lease.LeaseExpiresAt.After(now) {
			continue
		}
		if _, err := tx.Exec(ctx, `DELETE FROM control_data_agent_job_leases WHERE job_id = $1`, job.JobID); err != nil {
			return result, mapPgError(err)
		}
		progressTotal := job.ProgressTotal
		if progressTotal <= 0 {
			progressTotal = job.ResourceCount
		}
		requeued, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
UPDATE control_data_agent_jobs
SET status = 'queued',
    progress_completed = 0,
    progress_total = $2,
    error = NULL,
    updated_at = $3,
    started_at = NULL,
    completed_at = NULL
WHERE job_id = $1
RETURNING job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
          progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
          created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata`,
			job.JobID,
			int64(progressTotal),
			now.UTC(),
		))
		if err != nil {
			return result, mapPgError(err)
		}
		if _, err := appendDataAgentJobEventTx(ctx, tx, domain.AppendDataAgentJobEventInput{
			JobID:       requeued.JobID,
			EventType:   "data_agent.job.requeued",
			ActorUserID: requeued.OwnerUserID,
			ActorOrgID:  requeued.OwnerOrgID,
			TS:          now,
			Message:     reason,
			Metadata: domain.JSONMap{
				"reason":           reason,
				"recovery":         "expired_data_agent_job_lease",
				"lease_worker_id":  lease.WorkerID,
				"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
			},
		}); err != nil {
			return result, err
		}
		result.RequeuedJobs = append(result.RequeuedJobs, requeued)
	}
	// A job whose dispatch keeps failing must not be re-dispatched forever every
	// recovery pass: once it has failed to dispatch this many times, mark it
	// failed so it surfaces terminally instead of churning silently.
	if _, err := tx.Exec(ctx, `
UPDATE control_data_agent_jobs j
SET status = 'failed',
    error = 'dispatch failed repeatedly; giving up after '
            || $1::text || ' attempts',
    completed_at = now(),
    updated_at = now()
WHERE j.status = 'queued'
  AND NOT EXISTS (
    SELECT 1 FROM control_data_agent_job_leases l WHERE l.job_id = j.job_id
  )
  AND (
    SELECT count(*)
    FROM control_data_agent_job_events e
    WHERE e.job_id = j.job_id AND e.event_type = 'data_agent.job.dispatch_failed'
  ) >= $1`, int32(dataAgentJobMaxDispatchRetries)); err != nil {
		return result, mapPgError(err)
	}
	if remaining := limit - len(result.RequeuedJobs); remaining > 0 {
		rows, err := tx.Query(ctx, `
SELECT j.job_id
FROM control_data_agent_jobs j
WHERE j.status = 'queued'
  AND NOT EXISTS (
    SELECT 1 FROM control_data_agent_job_leases l WHERE l.job_id = j.job_id
  )
  AND (
    SELECT e.event_type
    FROM control_data_agent_job_events e
    WHERE e.job_id = j.job_id
    ORDER BY e.sequence DESC
    LIMIT 1
  ) = 'data_agent.job.dispatch_failed'
  AND (
    SELECT count(*)
    FROM control_data_agent_job_events e
    WHERE e.job_id = j.job_id AND e.event_type = 'data_agent.job.dispatch_failed'
  ) < $2
ORDER BY j.updated_at ASC, j.job_id ASC
LIMIT $1
FOR UPDATE OF j`, int32(remaining), int32(dataAgentJobMaxDispatchRetries))
		if err != nil {
			return result, mapPgError(err)
		}
		retryJobIDs := []string{}
		for rows.Next() {
			var jobID string
			if err := rows.Scan(&jobID); err != nil {
				rows.Close()
				return result, mapPgError(err)
			}
			retryJobIDs = append(retryJobIDs, jobID)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return result, mapPgError(err)
		}
		rows.Close()
		result.Checked += len(retryJobIDs)
		for _, jobID := range retryJobIDs {
			job, err := scanDataAgentJobRow(tx.QueryRow(ctx, `
SELECT job_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(project_id, ''), job_type, status, resource_count, progress_completed,
       progress_total, COALESCE(error, ''), COALESCE(created_by_user_id, ''),
       created_at, updated_at, started_at, completed_at, input_selector, output_summary, metadata
FROM control_data_agent_jobs
WHERE job_id = $1`, jobID))
			if err != nil {
				return result, mapPgError(err)
			}
			result.RequeuedJobs = append(result.RequeuedJobs, job)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RecoverExpiredDataAgentJobLeasesResult{}, err
	}
	return result, nil
}

func (s *PostgresStore) AppendDataAgentJobEvent(ctx context.Context, input domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("data_agent_job_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	sequence := input.Sequence
	event, err := scanDataAgentJobEventRow(s.pool.QueryRow(ctx, `
INSERT INTO control_data_agent_job_events (
  event_id, job_id, sequence, event_type, actor_user_id, actor_org_id, ts, message, metadata
)
VALUES (
  $1, $2,
  CASE WHEN $3::bigint > 0 THEN $3::bigint ELSE (SELECT COALESCE(MAX(sequence), 0) + 1 FROM control_data_agent_job_events WHERE job_id = $2) END,
  $4, NULLIF($5, ''), NULLIF($6, ''), $7, NULLIF($8, ''), $9
)
RETURNING event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
          COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata`,
		eventID,
		strings.TrimSpace(input.JobID),
		sequence,
		strings.TrimSpace(input.EventType),
		strings.TrimSpace(input.ActorUserID),
		strings.TrimSpace(input.ActorOrgID),
		ts.UTC(),
		strings.TrimSpace(input.Message),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	return event, nil
}

func (s *PostgresStore) ListDataAgentJobEvents(ctx context.Context, jobID string, userID string, orgID string, limit int) ([]domain.DataAgentJobEventRecord, error) {
	if _, err := s.GetDataAgentJobForUser(ctx, jobID, userID, orgID); err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
       COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata
FROM control_data_agent_job_events
WHERE job_id = $1
ORDER BY sequence ASC, event_id ASC
LIMIT $2`,
		strings.TrimSpace(jobID),
		limit32(limit, 200),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	return scanDataAgentJobEventRows(rows)
}

func appendDataAgentJobEventTx(ctx context.Context, tx pgx.Tx, input domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("data_agent_job_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	sequence := input.Sequence
	event, err := scanDataAgentJobEventRow(tx.QueryRow(ctx, `
INSERT INTO control_data_agent_job_events (
  event_id, job_id, sequence, event_type, actor_user_id, actor_org_id, ts, message, metadata
)
VALUES (
  $1, $2,
  CASE WHEN $3::bigint > 0 THEN $3::bigint ELSE (SELECT COALESCE(MAX(sequence), 0) + 1 FROM control_data_agent_job_events WHERE job_id = $2) END,
  $4, NULLIF($5, ''), NULLIF($6, ''), $7, NULLIF($8, ''), $9
)
RETURNING event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
          COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata`,
		eventID,
		strings.TrimSpace(input.JobID),
		sequence,
		strings.TrimSpace(input.EventType),
		strings.TrimSpace(input.ActorUserID),
		strings.TrimSpace(input.ActorOrgID),
		ts.UTC(),
		strings.TrimSpace(input.Message),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.DataAgentJobEventRecord{}, mapPgError(err)
	}
	return event, nil
}

func (s *PostgresStore) SoftDeleteResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, deletedAt time.Time) (domain.ResourceRecord, error) {
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	row, err := s.queries.SoftDeleteResourceForUser(ctx, sqlc.SoftDeleteResourceForUserParams{
		ResourceID:         strings.TrimSpace(resourceID),
		OwnerUserID:        strings.TrimSpace(userID),
		OwnerOrgID:         nullableText(orgID),
		DeletedAt:          timestamptz(deletedAt),
		RetentionExpiresAt: timestamptz(deletedAt.UTC().Add(defaultResourceRetention)),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) RestoreResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, restoredAt time.Time) (domain.ResourceRecord, error) {
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	row, err := s.queries.RestoreResourceForUser(ctx, sqlc.RestoreResourceForUserParams{
		ResourceID:  strings.TrimSpace(resourceID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
		UpdatedAt:   timestamptz(restoredAt),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) SoftDeleteResourceForUserWithEvent(
	ctx context.Context,
	input domain.ResourceLifecycleMutationInput,
) (domain.ResourceLifecycleMutationResult, error) {
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	ts = ts.UTC()
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	resourceID := strings.TrimSpace(input.ResourceID)
	if err := lockResourceLifecycleTx(ctx, tx, resourceID); err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	queries := s.queries.WithTx(tx)
	row, err := queries.SoftDeleteResourceForUser(ctx, sqlc.SoftDeleteResourceForUserParams{
		ResourceID:         resourceID,
		OwnerUserID:        strings.TrimSpace(input.OwnerUserID),
		OwnerOrgID:         nullableText(input.OwnerOrgID),
		DeletedAt:          timestamptz(ts),
		RetentionExpiresAt: timestamptz(ts.Add(defaultResourceRetention)),
	})
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, mapPgError(err)
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("resource_event")
	}
	eventRow, err := queries.CreateResourceEvent(ctx, sqlc.CreateResourceEventParams{
		EventID:     eventID,
		ResourceID:  resourceID,
		ActorUserID: nullableText(input.ActorUserID),
		ActorOrgID:  nullableText(input.ActorOrgID),
		EventType:   "resource.deleted",
		Ts:          timestamptz(ts),
		Metadata:    jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	return domain.ResourceLifecycleMutationResult{
		Resource: resourceFromRow(row),
		Event:    resourceEventFromRow(eventRow),
	}, nil
}

func (s *PostgresStore) RestoreResourceForUserWithEvent(
	ctx context.Context,
	input domain.ResourceLifecycleMutationInput,
) (domain.ResourceLifecycleMutationResult, error) {
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	ts = ts.UTC()
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	resourceID := strings.TrimSpace(input.ResourceID)
	if err := lockResourceLifecycleTx(ctx, tx, resourceID); err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	queries := s.queries.WithTx(tx)
	row, err := queries.RestoreResourceForUser(ctx, sqlc.RestoreResourceForUserParams{
		ResourceID:  resourceID,
		OwnerUserID: strings.TrimSpace(input.OwnerUserID),
		OwnerOrgID:  nullableText(input.OwnerOrgID),
		UpdatedAt:   timestamptz(ts),
	})
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, mapPgError(err)
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("resource_event")
	}
	eventRow, err := queries.CreateResourceEvent(ctx, sqlc.CreateResourceEventParams{
		EventID:     eventID,
		ResourceID:  resourceID,
		ActorUserID: nullableText(input.ActorUserID),
		ActorOrgID:  nullableText(input.ActorOrgID),
		EventType:   "resource.restored",
		Ts:          timestamptz(ts),
		Metadata:    jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ResourceLifecycleMutationResult{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceLifecycleMutationResult{}, err
	}
	return domain.ResourceLifecycleMutationResult{
		Resource: resourceFromRow(row),
		Event:    resourceEventFromRow(eventRow),
	}, nil
}

func (s *PostgresStore) ResourceStorageStats(ctx context.Context) (domain.ResourceStorageStats, error) {
	row, err := s.queries.ResourceStorageStats(ctx)
	if err != nil {
		return domain.ResourceStorageStats{}, err
	}
	return domain.ResourceStorageStats{
		TotalResources: int(row.TotalResources),
		TotalBytes:     row.TotalBytes,
	}, nil
}

func (s *PostgresStore) CreateResourceEvent(ctx context.Context, input domain.AppendResourceEventInput) (domain.ResourceEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("resource_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	row, err := s.queries.CreateResourceEvent(ctx, sqlc.CreateResourceEventParams{
		EventID:     eventID,
		ResourceID:  strings.TrimSpace(input.ResourceID),
		ActorUserID: nullableText(input.ActorUserID),
		ActorOrgID:  nullableText(input.ActorOrgID),
		EventType:   strings.TrimSpace(input.EventType),
		Ts:          timestamptz(ts),
		Metadata:    jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ResourceEventRecord{}, mapPgError(err)
	}
	return resourceEventFromRow(row), nil
}

func (s *PostgresStore) CreateResourceShareGrant(ctx context.Context, input domain.CreateResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceShareGrantRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	grant, err := createResourceShareGrantTx(ctx, tx, input)
	if err != nil {
		return domain.ResourceShareGrantRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceShareGrantRecord{}, err
	}
	return grant, nil
}

func createResourceShareGrantTx(ctx context.Context, tx pgx.Tx, input domain.CreateResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if input.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.ResourceShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("resource_grant")
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}

	var canonicalOwnerUserID string
	var canonicalOwnerOrgID string
	var canonicalOwnerRole string
	if err := tx.QueryRow(ctx, `
SELECT owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, '')
FROM control_resources
WHERE resource_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'
FOR UPDATE`, resourceID, ownerUserID, ownerOrgID).Scan(&canonicalOwnerUserID, &canonicalOwnerOrgID, &canonicalOwnerRole); err != nil {
		return domain.ResourceShareGrantRecord{}, mapPgError(err)
	}

	// Idempotent share: re-sharing with the same grantee returns the existing
	// active grant instead of stacking duplicates (the table has no unique
	// constraint, and duplicates make "People with access" lie).
	existing, existingErr := scanResourceShareGrantRow(tx.QueryRow(ctx, `
SELECT grant_id, resource_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(grantee_user_id, ''), COALESCE(grantee_org_id, ''), role, status,
       COALESCE(created_by_user_id, ''), created_at, updated_at, revoked_at, metadata
FROM control_resource_share_grants
WHERE resource_id = $1 AND status = 'active'
  AND COALESCE(grantee_user_id, '') = $2 AND COALESCE(grantee_org_id, '') = $3
ORDER BY created_at ASC
LIMIT 1`, resourceID, granteeUserID, granteeOrgID))
	if existingErr == nil {
		return existing, nil
	}
	if !errors.Is(mapPgError(existingErr), ErrNotFound) {
		return domain.ResourceShareGrantRecord{}, mapPgError(existingErr)
	}
	grant, err := scanResourceShareGrantRow(tx.QueryRow(ctx, `
INSERT INTO control_resource_share_grants (
  grant_id, resource_id, owner_user_id, owner_org_id, owner_role,
  grantee_user_id, grantee_org_id, role, status, created_by_user_id,
  created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING grant_id, resource_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
          COALESCE(grantee_user_id, ''), COALESCE(grantee_org_id, ''), role, status,
          COALESCE(created_by_user_id, ''), created_at, updated_at, revoked_at, metadata`,
		grantID,
		resourceID,
		canonicalOwnerUserID,
		nullableText(canonicalOwnerOrgID),
		nullableText(canonicalOwnerRole),
		nullableText(granteeUserID),
		nullableText(granteeOrgID),
		role,
		status,
		nullableText(createdByUserID),
		timestamptz(createdAt),
		timestamptz(updatedAt),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.ResourceShareGrantRecord{}, mapPgError(err)
	}
	return grant, nil
}

func (s *PostgresStore) ListResourceShareGrantsForResource(ctx context.Context, input domain.ListResourceShareGrantsInput) ([]domain.ResourceShareGrantRecord, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	status := strings.TrimSpace(input.Status)
	var exists int
	if err := s.pool.QueryRow(ctx, `
SELECT 1
FROM control_resources
WHERE resource_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status = 'active'`, resourceID, ownerUserID, ownerOrgID).Scan(&exists); err != nil {
		return nil, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT g.grant_id, g.resource_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
       COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
       COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata
FROM control_resource_share_grants g
JOIN control_resources r ON r.resource_id = g.resource_id
WHERE g.resource_id = $1
  AND r.owner_user_id = $2
  AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)
  AND r.status = 'active'
  AND ($4::text = '' OR g.status = $4)
ORDER BY g.created_at DESC, g.grant_id ASC
LIMIT $5`, resourceID, ownerUserID, ownerOrgID, status, limit32(input.Limit, 200))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	grants := make([]domain.ResourceShareGrantRecord, 0)
	for rows.Next() {
		grant, err := scanResourceShareGrantRow(rows)
		if err != nil {
			return nil, err
		}
		grants = append(grants, grant)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return grants, nil
}

func (s *PostgresStore) RevokeResourceShareGrant(ctx context.Context, input domain.RevokeResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	grantID := strings.TrimSpace(input.GrantID)
	revokedAt := input.RevokedAt
	if revokedAt.IsZero() {
		revokedAt = domain.Now()
	}
	grant, err := scanResourceShareGrantRow(s.pool.QueryRow(ctx, `
UPDATE control_resource_share_grants AS g
SET status = 'revoked',
    revoked_at = $5,
    updated_at = $5
FROM control_resources r
WHERE g.resource_id = r.resource_id
  AND g.grant_id = $1
  AND g.resource_id = $2
  AND r.owner_user_id = $3
  AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $4)
  AND r.status = 'active'
  AND g.status = 'active'
RETURNING g.grant_id, g.resource_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
          COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
          COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata`,
		grantID,
		resourceID,
		ownerUserID,
		ownerOrgID,
		timestamptz(revokedAt),
	))
	if err != nil {
		return domain.ResourceShareGrantRecord{}, mapPgError(err)
	}
	return grant, nil
}

func (s *PostgresStore) ListResourceEvents(ctx context.Context, resourceID string, limit int) ([]domain.ResourceEventRecord, error) {
	rows, err := s.queries.ListResourceEvents(ctx, sqlc.ListResourceEventsParams{
		ResourceID: strings.TrimSpace(resourceID),
		Limit:      limit32(limit, 200),
	})
	if err != nil {
		return nil, mapPgError(err)
	}
	events := make([]domain.ResourceEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, resourceEventFromRow(row))
	}
	return events, nil
}

func (s *PostgresStore) ListResourceEventsForUser(ctx context.Context, input domain.ResourceEventListInput) (domain.ResourceEventListPage, error) {
	params := sqlc.ListResourceEventsForUserParams{
		OwnerUserID: strings.TrimSpace(input.UserID),
		OwnerOrgID:  nullableText(input.OrgID),
		Column3:     strings.TrimSpace(input.ResourceID),
		Column4:     strings.TrimSpace(input.EventType),
		Column5:     strings.TrimSpace(input.ActorUserID),
		Limit:       limit32(input.Limit, 200),
		Offset:      offset32(input.Offset),
	}
	rows, err := s.queries.ListResourceEventsForUser(ctx, params)
	if err != nil {
		return domain.ResourceEventListPage{}, mapPgError(err)
	}
	count, err := s.queries.CountResourceEventsForUser(ctx, sqlc.CountResourceEventsForUserParams{
		OwnerUserID: params.OwnerUserID,
		OwnerOrgID:  params.OwnerOrgID,
		Column3:     params.Column3,
		Column4:     params.Column4,
		Column5:     params.Column5,
	})
	if err != nil {
		return domain.ResourceEventListPage{}, mapPgError(err)
	}
	events := make([]domain.ResourceEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, resourceEventFromRow(row))
	}
	return domain.ResourceEventListPage{
		Events:     events,
		TotalCount: int(count),
		Limit:      int(params.Limit),
		Offset:     int(params.Offset),
	}, nil
}

func threadFromRow(row sqlc.ControlThread) domain.ThreadRecord {
	return domain.ThreadRecord{
		ThreadID:     row.ThreadID,
		UserID:       row.UserID,
		Title:        textValue(row.Title),
		Status:       domain.ThreadStatus(row.Status),
		CreatedAt:    timeValue(row.CreatedAt),
		UpdatedAt:    timeValue(row.UpdatedAt),
		LatestRunID:  textValue(row.LatestRunID),
		CheckpointID: textValue(row.CheckpointID),
		Summary:      textValue(row.Summary),
		Metadata:     jsonMap(row.Metadata),
	}
}

func threadMessageFromRow(row sqlc.ControlThreadMessage) domain.ThreadMessage {
	return domain.ThreadMessage{
		MessageID: row.MessageID,
		ThreadID:  row.ThreadID,
		Role:      row.Role,
		Content:   row.Content,
		CreatedAt: timeValue(row.CreatedAt),
		Metadata:  jsonMap(row.Metadata),
		RunID:     textValue(row.RunID),
	}
}

func runFromRow(row sqlc.ControlRun) domain.RunRecord {
	return domain.RunRecord{
		RunID:           row.RunID,
		ThreadID:        row.ThreadID,
		UserID:          row.UserID,
		Goal:            row.Goal,
		Status:          domain.RunStatus(row.Status),
		WorkflowKind:    row.WorkflowKind,
		Mode:            textValue(row.Mode),
		CurrentNode:     textValue(row.CurrentNode),
		ParentRunID:     textValue(row.ParentRunID),
		PlannerVersion:  textValue(row.PlannerVersion),
		AgentRole:       textValue(row.AgentRole),
		TraceGroupID:    textValue(row.TraceGroupID),
		CheckpointID:    textValue(row.CheckpointID),
		CheckpointState: jsonMap(row.CheckpointState),
		BudgetState:     jsonMap(row.BudgetState),
		ResponseText:    textValue(row.ResponseText),
		Error:           textValue(row.Error),
		CreatedAt:       timeValue(row.CreatedAt),
		UpdatedAt:       timeValue(row.UpdatedAt),
		StartedAt:       timePtr(row.StartedAt),
		CompletedAt:     timePtr(row.CompletedAt),
		Metadata:        jsonMap(row.Metadata),
	}
}

func runEventFromRow(row sqlc.ControlRunEvent) domain.RunEventRecord {
	return domain.RunEventRecord{
		EventID:        row.EventID,
		Sequence:       row.SequenceNumber,
		SourceSequence: int8Value(row.SourceSequence),
		RunID:          row.RunID,
		ThreadID:       textValue(row.ThreadID),
		EventKind:      row.EventKind,
		EventType:      textValue(row.EventType),
		NodeName:       textValue(row.NodeName),
		TaskID:         textValue(row.TaskID),
		CheckpointID:   textValue(row.CheckpointID),
		ScopeID:        textValue(row.ScopeID),
		AgentRole:      textValue(row.AgentRole),
		Level:          textValue(row.Level),
		TS:             timeValue(row.Ts),
		Message:        textValue(row.Message),
		Payload:        jsonMap(row.Payload),
	}
}

func artifactFromRow(row sqlc.ControlArtifact) domain.ArtifactRecord {
	return domain.ArtifactRecord{
		ArtifactID:    row.ArtifactID,
		RunID:         row.RunID,
		ThreadID:      textValue(row.ThreadID),
		Kind:          row.Kind,
		Path:          textValue(row.Path),
		SourcePath:    textValue(row.SourcePath),
		PreviewPath:   textValue(row.PreviewPath),
		Title:         textValue(row.Title),
		ResultGroupID: textValue(row.ResultGroupID),
		MimeType:      textValue(row.MimeType),
		SizeBytes:     int8Value(row.SizeBytes),
		SHA256:        textValue(row.Sha256),
		StorageURI:    textValue(row.StorageUri),
		ToolName:      textValue(row.ToolName),
		Category:      textValue(row.Category),
		CreatedAt:     timeValue(row.CreatedAt),
		UpdatedAt:     timeValue(row.UpdatedAt),
		Metadata:      jsonMap(row.Metadata),
	}
}

func scanControlResourceRow(row pgx.Row) (sqlc.ControlResource, error) {
	var resource sqlc.ControlResource
	err := row.Scan(
		&resource.ResourceID,
		&resource.OwnerUserID,
		&resource.OwnerOrgID,
		&resource.OwnerRole,
		&resource.OriginalName,
		&resource.ContentType,
		&resource.SizeBytes,
		&resource.Sha256,
		&resource.StorageUri,
		&resource.StoragePath,
		&resource.SourceType,
		&resource.ResourceKind,
		&resource.SourceUri,
		&resource.ProjectID,
		&resource.Status,
		&resource.CreatedAt,
		&resource.UpdatedAt,
		&resource.DeletedAt,
		&resource.RetentionExpiresAt,
		&resource.Metadata,
	)
	return resource, err
}

func resourceFromRow(row sqlc.ControlResource) domain.ResourceRecord {
	resource := domain.ResourceRecord{
		ResourceID:         row.ResourceID,
		OriginalName:       row.OriginalName,
		ContentType:        textValue(row.ContentType),
		SizeBytes:          row.SizeBytes,
		SHA256:             textValue(row.Sha256),
		StorageURI:         textValue(row.StorageUri),
		StoragePath:        textValue(row.StoragePath),
		SourceType:         row.SourceType,
		ResourceKind:       row.ResourceKind,
		SourceURI:          textValue(row.SourceUri),
		ProjectID:          textValue(row.ProjectID),
		OwnerUserID:        row.OwnerUserID,
		OwnerOrgID:         textValue(row.OwnerOrgID),
		OwnerRole:          textValue(row.OwnerRole),
		Status:             row.Status,
		CreatedAt:          timeValue(row.CreatedAt),
		UpdatedAt:          timeValue(row.UpdatedAt),
		DeletedAt:          timeValue(row.DeletedAt),
		RetentionExpiresAt: timeValue(row.RetentionExpiresAt),
		Metadata:           jsonMap(row.Metadata),
	}
	return resourceWithNormalizedTags(resource)
}

func resourceFromListResourcesForUserRow(row sqlc.ListResourcesForUserRow) domain.ResourceRecord {
	resource := domain.ResourceRecord{
		ResourceID:         row.ResourceID,
		OriginalName:       row.OriginalName,
		ContentType:        textValue(row.ContentType),
		SizeBytes:          row.SizeBytes,
		SHA256:             textValue(row.Sha256),
		StorageURI:         textValue(row.StorageUri),
		StoragePath:        textValue(row.StoragePath),
		SourceType:         row.SourceType,
		ResourceKind:       row.ResourceKind,
		SourceURI:          textValue(row.SourceUri),
		ProjectID:          textValue(row.ProjectID),
		OwnerUserID:        row.OwnerUserID,
		OwnerOrgID:         textValue(row.OwnerOrgID),
		OwnerRole:          textValue(row.OwnerRole),
		Status:             row.Status,
		CreatedAt:          timeValue(row.CreatedAt),
		UpdatedAt:          timeValue(row.UpdatedAt),
		DeletedAt:          timeValue(row.DeletedAt),
		RetentionExpiresAt: timeValue(row.RetentionExpiresAt),
		Metadata:           jsonMap(row.Metadata),
		ShareSummary: domain.ResourceShareSummary{
			ShareStatus:      row.ShareStatus,
			ActiveGrantCount: int(row.ActiveGrantCount),
			SharedByMe:       row.SharedByMe,
			SharedWithMe:     row.SharedWithMe,
			Public:           row.ShareStatus == "public",
		},
	}
	return resourceWithNormalizedTags(resource)
}

func uploadSessionFromRow(row sqlc.ControlUploadSession) domain.UploadSessionRecord {
	return domain.UploadSessionRecord{
		SessionID:          row.SessionID,
		OwnerUserID:        row.OwnerUserID,
		OwnerOrgID:         textValue(row.OwnerOrgID),
		OwnerRole:          textValue(row.OwnerRole),
		ProjectID:          textValue(row.ProjectID),
		SourceType:         row.SourceType,
		Status:             row.Status,
		TotalBytes:         row.TotalBytes,
		BytesReceived:      row.BytesReceived,
		BytesVerified:      row.BytesVerified,
		BytesCommitted:     row.BytesCommitted,
		IdempotencyKey:     textValue(row.IdempotencyKey),
		BrowserFingerprint: textValue(row.BrowserFingerprint),
		Error:              textValue(row.Error),
		CreatedAt:          timeValue(row.CreatedAt),
		UpdatedAt:          timeValue(row.UpdatedAt),
		CompletedAt:        timeValue(row.CompletedAt),
		Metadata:           jsonMap(row.Metadata),
	}
}

func uploadSessionFileFromRow(row sqlc.ControlUploadSessionFile) domain.UploadSessionFileRecord {
	return domain.UploadSessionFileRecord{
		SessionID:      row.SessionID,
		FileToken:      row.FileToken,
		ResourceID:     textValue(row.ResourceID),
		OriginalName:   row.OriginalName,
		RelativePath:   textValue(row.RelativePath),
		ContentType:    textValue(row.ContentType),
		SizeBytes:      row.SizeBytes,
		DeclaredSHA256: textValue(row.DeclaredSha256),
		ComputedSHA256: textValue(row.ComputedSha256),
		Status:         row.Status,
		Error:          textValue(row.Error),
		CreatedAt:      timeValue(row.CreatedAt),
		UpdatedAt:      timeValue(row.UpdatedAt),
		CompletedAt:    timeValue(row.CompletedAt),
		Metadata:       jsonMap(row.Metadata),
	}
}

func scanUploadSessionFileRow(row pgx.Row) (domain.UploadSessionFileRecord, error) {
	var record sqlc.ControlUploadSessionFile
	if err := row.Scan(
		&record.SessionID,
		&record.FileToken,
		&record.ResourceID,
		&record.OriginalName,
		&record.RelativePath,
		&record.ContentType,
		&record.SizeBytes,
		&record.DeclaredSha256,
		&record.ComputedSha256,
		&record.Status,
		&record.Error,
		&record.CreatedAt,
		&record.UpdatedAt,
		&record.CompletedAt,
		&record.Metadata,
	); err != nil {
		return domain.UploadSessionFileRecord{}, err
	}
	return uploadSessionFileFromRow(record), nil
}

func uploadSessionEventFromRow(row sqlc.ControlUploadSessionEvent) domain.UploadSessionEventRecord {
	return domain.UploadSessionEventRecord{
		EventID:     row.EventID,
		SessionID:   row.SessionID,
		ActorUserID: textValue(row.ActorUserID),
		ActorOrgID:  textValue(row.ActorOrgID),
		EventType:   row.EventType,
		TS:          timeValue(row.Ts),
		Metadata:    jsonMap(row.Metadata),
	}
}

func uploadChunkFromRow(row sqlc.ControlUploadChunk) domain.UploadChunkRecord {
	return domain.UploadChunkRecord{
		SessionID:  row.SessionID,
		FileToken:  row.FileToken,
		ChunkIndex: int(row.ChunkIndex),
		Offset:     row.ByteOffset,
		SizeBytes:  row.SizeBytes,
		SHA256:     row.Sha256,
		Status:     row.Status,
		StorageURI: textValue(row.StorageUri),
		ReceivedAt: timeValue(row.ReceivedAt),
		VerifiedAt: timeValue(row.VerifiedAt),
		Error:      textValue(row.Error),
		Metadata:   jsonMap(row.Metadata),
	}
}

type resourceCollectionScanner interface {
	Scan(dest ...any) error
}

func scanResourceCollectionRow(row resourceCollectionScanner) (domain.ResourceCollectionRecord, error) {
	var collection domain.ResourceCollectionRecord
	var resourceCount int64
	var metadata []byte
	if err := row.Scan(
		&collection.CollectionID,
		&collection.OwnerUserID,
		&collection.OwnerOrgID,
		&collection.OwnerRole,
		&collection.ProjectID,
		&collection.ParentCollectionID,
		&collection.Name,
		&collection.Description,
		&collection.CollectionType,
		&collection.Status,
		&resourceCount,
		&collection.CreatedAt,
		&collection.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.ResourceCollectionRecord{}, err
	}
	collection.ResourceCount = int(resourceCount)
	collection.CreatedAt = collection.CreatedAt.UTC()
	collection.UpdatedAt = collection.UpdatedAt.UTC()
	collection.Metadata = jsonMap(metadata)
	return collection, nil
}

func scanResourceCollectionRows(rows pgx.Rows) ([]domain.ResourceCollectionRecord, error) {
	collections := []domain.ResourceCollectionRecord{}
	for rows.Next() {
		collection, err := scanResourceCollectionRow(rows)
		if err != nil {
			return nil, err
		}
		collections = append(collections, collection)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return collections, nil
}

func scanDatasetSnapshotRow(row scanner) (domain.DatasetSnapshotRecord, error) {
	var snapshot domain.DatasetSnapshotRecord
	var resourceCount int64
	var metadata []byte
	if err := row.Scan(
		&snapshot.SnapshotID,
		&snapshot.OwnerUserID,
		&snapshot.OwnerOrgID,
		&snapshot.OwnerRole,
		&snapshot.ProjectID,
		&snapshot.SourceCollectionID,
		&snapshot.Name,
		&snapshot.Description,
		&snapshot.Status,
		&resourceCount,
		&snapshot.TotalBytes,
		&snapshot.CreatedByUserID,
		&snapshot.CreatedAt,
		&metadata,
	); err != nil {
		return domain.DatasetSnapshotRecord{}, err
	}
	snapshot.ResourceCount = int(resourceCount)
	snapshot.CreatedAt = snapshot.CreatedAt.UTC()
	snapshot.Metadata = jsonMap(metadata)
	return snapshot, nil
}

func scanDatasetSnapshotRows(rows pgx.Rows) ([]domain.DatasetSnapshotRecord, error) {
	snapshots := []domain.DatasetSnapshotRecord{}
	for rows.Next() {
		snapshot, err := scanDatasetSnapshotRow(rows)
		if err != nil {
			return nil, err
		}
		snapshots = append(snapshots, snapshot)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return snapshots, nil
}

func scanDatasetSnapshotResourceRow(row scanner) (domain.DatasetSnapshotResourceRecord, error) {
	var entry domain.DatasetSnapshotResourceRecord
	var resourceCreatedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&entry.SnapshotID,
		&entry.ResourceID,
		&entry.Position,
		&entry.OriginalName,
		&entry.ContentType,
		&entry.SizeBytes,
		&entry.SHA256,
		&entry.SourceType,
		&entry.ResourceKind,
		&entry.StorageURI,
		&entry.SourceURI,
		&entry.ProjectID,
		&resourceCreatedAt,
		&metadata,
	); err != nil {
		return domain.DatasetSnapshotResourceRecord{}, err
	}
	entry.ResourceCreatedAt = timeValue(resourceCreatedAt)
	entry.Metadata = jsonMap(metadata)
	return entry, nil
}

func scanDatasetSnapshotResourceRows(rows pgx.Rows) ([]domain.DatasetSnapshotResourceRecord, error) {
	entries := []domain.DatasetSnapshotResourceRecord{}
	for rows.Next() {
		entry, err := scanDatasetSnapshotResourceRow(rows)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}

func scanDatasetSnapshotShareGrantRow(row scanner) (domain.DatasetSnapshotShareGrantRecord, error) {
	var grant domain.DatasetSnapshotShareGrantRecord
	var revokedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&grant.GrantID,
		&grant.SnapshotID,
		&grant.OwnerUserID,
		&grant.OwnerOrgID,
		&grant.OwnerRole,
		&grant.GranteeUserID,
		&grant.GranteeOrgID,
		&grant.Role,
		&grant.Status,
		&grant.CreatedByUserID,
		&grant.CreatedAt,
		&grant.UpdatedAt,
		&revokedAt,
		&metadata,
	); err != nil {
		return domain.DatasetSnapshotShareGrantRecord{}, mapPgError(err)
	}
	grant.CreatedAt = grant.CreatedAt.UTC()
	grant.UpdatedAt = grant.UpdatedAt.UTC()
	grant.RevokedAt = timeValue(revokedAt)
	grant.Metadata = jsonMap(metadata)
	return grant, nil
}

func insertDatasetSnapshotEventTx(ctx context.Context, tx pgx.Tx, event domain.DatasetSnapshotEventRecord) (domain.DatasetSnapshotEventRecord, error) {
	eventID := strings.TrimSpace(event.EventID)
	if eventID == "" {
		eventID = domain.NewID("dataset_snapshot_event")
	}
	ts := event.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	return scanDatasetSnapshotEventRow(tx.QueryRow(ctx, `
INSERT INTO control_dataset_snapshot_events (
  event_id, snapshot_id, actor_user_id, actor_org_id, event_type, ts, metadata
)
VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), $5, $6, $7)
RETURNING event_id, snapshot_id, COALESCE(actor_user_id, ''), COALESCE(actor_org_id, ''),
          event_type, ts, metadata`,
		eventID,
		strings.TrimSpace(event.SnapshotID),
		strings.TrimSpace(event.ActorUserID),
		strings.TrimSpace(event.ActorOrgID),
		strings.TrimSpace(event.EventType),
		timestamptz(ts.UTC()),
		jsonBytes(event.Metadata),
	))
}

func latestDatasetSnapshotEventTimeTx(ctx context.Context, tx pgx.Tx, snapshotID string, fallback time.Time) time.Time {
	var latest pgtype.Timestamptz
	if err := tx.QueryRow(ctx, `
SELECT MAX(ts)
FROM control_dataset_snapshot_events
WHERE snapshot_id = $1`, strings.TrimSpace(snapshotID)).Scan(&latest); err != nil {
		return fallback
	}
	value := timeValue(latest)
	if value.IsZero() {
		return fallback
	}
	return value
}

func scanDatasetSnapshotEventRow(row scanner) (domain.DatasetSnapshotEventRecord, error) {
	var event domain.DatasetSnapshotEventRecord
	var metadata []byte
	if err := row.Scan(
		&event.EventID,
		&event.SnapshotID,
		&event.ActorUserID,
		&event.ActorOrgID,
		&event.EventType,
		&event.TS,
		&metadata,
	); err != nil {
		return domain.DatasetSnapshotEventRecord{}, mapPgError(err)
	}
	event.TS = event.TS.UTC()
	event.Metadata = jsonMap(metadata)
	return event, nil
}

func scanDataAgentJobRow(row scanner) (domain.DataAgentJobRecord, error) {
	var job domain.DataAgentJobRecord
	var resourceCount int64
	var progressCompleted int64
	var progressTotal int64
	var startedAt pgtype.Timestamptz
	var completedAt pgtype.Timestamptz
	var inputSelector []byte
	var outputSummary []byte
	var metadata []byte
	if err := row.Scan(
		&job.JobID,
		&job.OwnerUserID,
		&job.OwnerOrgID,
		&job.OwnerRole,
		&job.ProjectID,
		&job.JobType,
		&job.Status,
		&resourceCount,
		&progressCompleted,
		&progressTotal,
		&job.Error,
		&job.CreatedByUserID,
		&job.CreatedAt,
		&job.UpdatedAt,
		&startedAt,
		&completedAt,
		&inputSelector,
		&outputSummary,
		&metadata,
	); err != nil {
		return domain.DataAgentJobRecord{}, err
	}
	job.ResourceCount = int(resourceCount)
	job.ProgressCompleted = int(progressCompleted)
	job.ProgressTotal = int(progressTotal)
	job.CreatedAt = job.CreatedAt.UTC()
	job.UpdatedAt = job.UpdatedAt.UTC()
	job.StartedAt = timeValue(startedAt)
	job.CompletedAt = timeValue(completedAt)
	job.InputSelector = jsonMap(inputSelector)
	job.OutputSummary = jsonMap(outputSummary)
	job.Metadata = jsonMap(metadata)
	return job, nil
}

func scanDataAgentJobRows(rows pgx.Rows) ([]domain.DataAgentJobRecord, error) {
	jobs := []domain.DataAgentJobRecord{}
	for rows.Next() {
		job, err := scanDataAgentJobRow(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, job)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return jobs, nil
}

func scanDataAgentJobEventRow(row scanner) (domain.DataAgentJobEventRecord, error) {
	var event domain.DataAgentJobEventRecord
	var metadata []byte
	if err := row.Scan(
		&event.EventID,
		&event.JobID,
		&event.Sequence,
		&event.EventType,
		&event.ActorUserID,
		&event.ActorOrgID,
		&event.TS,
		&event.Message,
		&metadata,
	); err != nil {
		return domain.DataAgentJobEventRecord{}, err
	}
	event.TS = event.TS.UTC()
	event.Metadata = jsonMap(metadata)
	return event, nil
}

func scanDataAgentJobEventRows(rows pgx.Rows) ([]domain.DataAgentJobEventRecord, error) {
	events := []domain.DataAgentJobEventRecord{}
	for rows.Next() {
		event, err := scanDataAgentJobEventRow(rows)
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return events, nil
}

func upsertCollectionMemberTx(ctx context.Context, tx pgx.Tx, collectionID string, resourceID string, position int64, addedByUserID string, addedAt time.Time, metadata domain.JSONMap) (domain.ResourceCollectionMembershipRecord, bool, error) {
	var member domain.ResourceCollectionMembershipRecord
	var metadataBytes []byte
	err := tx.QueryRow(ctx, `
INSERT INTO control_resource_collection_members (collection_id, resource_id, position, added_by_user_id, added_at, metadata)
VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6)
ON CONFLICT (collection_id, resource_id) DO NOTHING
RETURNING collection_id, resource_id, position, COALESCE(added_by_user_id, ''), added_at, metadata`,
		collectionID,
		resourceID,
		position,
		addedByUserID,
		addedAt.UTC(),
		jsonBytes(metadata),
	).Scan(&member.CollectionID, &member.ResourceID, &member.Position, &member.AddedByUserID, &member.AddedAt, &metadataBytes)
	if err == nil {
		member.AddedAt = member.AddedAt.UTC()
		member.Metadata = jsonMap(metadataBytes)
		return member, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return domain.ResourceCollectionMembershipRecord{}, false, mapPgError(err)
	}
	err = tx.QueryRow(ctx, `
SELECT collection_id, resource_id, position, COALESCE(added_by_user_id, ''), added_at, metadata
FROM control_resource_collection_members
WHERE collection_id = $1 AND resource_id = $2`, collectionID, resourceID).
		Scan(&member.CollectionID, &member.ResourceID, &member.Position, &member.AddedByUserID, &member.AddedAt, &metadataBytes)
	if err != nil {
		return domain.ResourceCollectionMembershipRecord{}, false, mapPgError(err)
	}
	member.AddedAt = member.AddedAt.UTC()
	member.Metadata = jsonMap(metadataBytes)
	return member, false, nil
}

func activeResourceCollectionShareGrantsTx(ctx context.Context, tx pgx.Tx, collectionID string) ([]domain.ResourceCollectionShareGrantRecord, error) {
	rows, err := tx.Query(ctx, `
SELECT grant_id, collection_id, owner_user_id, COALESCE(owner_org_id, ''), COALESCE(owner_role, ''),
       COALESCE(grantee_user_id, ''), COALESCE(grantee_org_id, ''), role, status,
       COALESCE(created_by_user_id, ''), created_at, updated_at, revoked_at, metadata
FROM control_resource_collection_share_grants
WHERE collection_id = $1
  AND status = 'active'
ORDER BY created_at ASC, grant_id ASC`, strings.TrimSpace(collectionID))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	grants := make([]domain.ResourceCollectionShareGrantRecord, 0)
	for rows.Next() {
		grant, err := scanResourceCollectionShareGrantRow(rows)
		if err != nil {
			return nil, err
		}
		grants = append(grants, grant)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return grants, nil
}

func createInheritedResourceShareGrantTx(ctx context.Context, tx pgx.Tx, resourceID string, collectionGrant domain.ResourceCollectionShareGrantRecord, createdAt time.Time, source string) (domain.ResourceShareGrantRecord, error) {
	metadata := cloneJSONMap(collectionGrant.Metadata)
	metadata["collection_id"] = collectionGrant.CollectionID
	metadata["collection_share_grant_id"] = collectionGrant.GrantID
	metadata["source"] = source
	return createResourceShareGrantTx(ctx, tx, domain.CreateResourceShareGrantInput{
		ResourceID:      resourceID,
		OwnerUserID:     collectionGrant.OwnerUserID,
		OwnerOrgID:      collectionGrant.OwnerOrgID,
		OwnerRole:       collectionGrant.OwnerRole,
		GranteeUserID:   collectionGrant.GranteeUserID,
		GranteeOrgID:    collectionGrant.GranteeOrgID,
		Role:            collectionGrant.Role,
		Status:          "active",
		CreatedByUserID: collectionGrant.CreatedByUserID,
		CreatedAt:       createdAt,
		Metadata:        metadata,
	})
}

func scanResourceRows(rows pgx.Rows) ([]domain.ResourceRecord, error) {
	resources := []domain.ResourceRecord{}
	for rows.Next() {
		var resource domain.ResourceRecord
		var deletedAt pgtype.Timestamptz
		var retentionExpiresAt pgtype.Timestamptz
		var metadata []byte
		if err := rows.Scan(
			&resource.ResourceID,
			&resource.OwnerUserID,
			&resource.OwnerOrgID,
			&resource.OwnerRole,
			&resource.OriginalName,
			&resource.ContentType,
			&resource.SizeBytes,
			&resource.SHA256,
			&resource.StorageURI,
			&resource.StoragePath,
			&resource.SourceType,
			&resource.ResourceKind,
			&resource.SourceURI,
			&resource.ProjectID,
			&resource.Status,
			&resource.CreatedAt,
			&resource.UpdatedAt,
			&deletedAt,
			&retentionExpiresAt,
			&metadata,
		); err != nil {
			return nil, err
		}
		resource.CreatedAt = resource.CreatedAt.UTC()
		resource.UpdatedAt = resource.UpdatedAt.UTC()
		resource.DeletedAt = timeValue(deletedAt)
		resource.RetentionExpiresAt = timeValue(retentionExpiresAt)
		resource.Metadata = jsonMap(metadata)
		resources = append(resources, resource)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return resources, nil
}

func scanResourceRowsWithShareSummary(rows pgx.Rows) ([]domain.ResourceRecord, error) {
	resources := []domain.ResourceRecord{}
	for rows.Next() {
		var resource domain.ResourceRecord
		var deletedAt pgtype.Timestamptz
		var retentionExpiresAt pgtype.Timestamptz
		var metadata []byte
		var activeGrantCount int64
		if err := rows.Scan(
			&resource.ResourceID,
			&resource.OwnerUserID,
			&resource.OwnerOrgID,
			&resource.OwnerRole,
			&resource.OriginalName,
			&resource.ContentType,
			&resource.SizeBytes,
			&resource.SHA256,
			&resource.StorageURI,
			&resource.StoragePath,
			&resource.SourceType,
			&resource.ResourceKind,
			&resource.SourceURI,
			&resource.ProjectID,
			&resource.Status,
			&resource.CreatedAt,
			&resource.UpdatedAt,
			&deletedAt,
			&retentionExpiresAt,
			&metadata,
			&resource.ShareSummary.ShareStatus,
			&activeGrantCount,
			&resource.ShareSummary.SharedByMe,
			&resource.ShareSummary.SharedWithMe,
		); err != nil {
			return nil, err
		}
		resource.CreatedAt = resource.CreatedAt.UTC()
		resource.UpdatedAt = resource.UpdatedAt.UTC()
		resource.DeletedAt = timeValue(deletedAt)
		resource.RetentionExpiresAt = timeValue(retentionExpiresAt)
		resource.Metadata = jsonMap(metadata)
		resource.ShareSummary.ActiveGrantCount = int(activeGrantCount)
		resource.ShareSummary.Public = resource.ShareSummary.ShareStatus == "public"
		resources = append(resources, resourceWithNormalizedTags(resource))
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return resources, nil
}

func resourceEventFromRow(row sqlc.ControlResourceEvent) domain.ResourceEventRecord {
	return domain.ResourceEventRecord{
		EventID:     row.EventID,
		ResourceID:  row.ResourceID,
		ActorUserID: textValue(row.ActorUserID),
		ActorOrgID:  textValue(row.ActorOrgID),
		EventType:   row.EventType,
		TS:          timeValue(row.Ts),
		Metadata:    jsonMap(row.Metadata),
	}
}

func scanResourceEventRow(row scanner) (domain.ResourceEventRecord, error) {
	var event domain.ResourceEventRecord
	var metadata []byte
	if err := row.Scan(
		&event.EventID,
		&event.ResourceID,
		&event.ActorUserID,
		&event.ActorOrgID,
		&event.EventType,
		&event.TS,
		&metadata,
	); err != nil {
		return domain.ResourceEventRecord{}, mapPgError(err)
	}
	event.Metadata = jsonMap(metadata)
	return event, nil
}

func scanResourceCollectionShareGrantRow(row scanner) (domain.ResourceCollectionShareGrantRecord, error) {
	var grant domain.ResourceCollectionShareGrantRecord
	var revokedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&grant.GrantID,
		&grant.CollectionID,
		&grant.OwnerUserID,
		&grant.OwnerOrgID,
		&grant.OwnerRole,
		&grant.GranteeUserID,
		&grant.GranteeOrgID,
		&grant.Role,
		&grant.Status,
		&grant.CreatedByUserID,
		&grant.CreatedAt,
		&grant.UpdatedAt,
		&revokedAt,
		&metadata,
	); err != nil {
		return domain.ResourceCollectionShareGrantRecord{}, mapPgError(err)
	}
	grant.CreatedAt = grant.CreatedAt.UTC()
	grant.UpdatedAt = grant.UpdatedAt.UTC()
	grant.RevokedAt = timeValue(revokedAt)
	grant.Metadata = jsonMap(metadata)
	return grant, nil
}

func scanResourceShareGrantRow(row scanner) (domain.ResourceShareGrantRecord, error) {
	var grant domain.ResourceShareGrantRecord
	var revokedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&grant.GrantID,
		&grant.ResourceID,
		&grant.OwnerUserID,
		&grant.OwnerOrgID,
		&grant.OwnerRole,
		&grant.GranteeUserID,
		&grant.GranteeOrgID,
		&grant.Role,
		&grant.Status,
		&grant.CreatedByUserID,
		&grant.CreatedAt,
		&grant.UpdatedAt,
		&revokedAt,
		&metadata,
	); err != nil {
		return domain.ResourceShareGrantRecord{}, mapPgError(err)
	}
	grant.CreatedAt = grant.CreatedAt.UTC()
	grant.UpdatedAt = grant.UpdatedAt.UTC()
	grant.RevokedAt = timeValue(revokedAt)
	grant.Metadata = jsonMap(metadata)
	return grant, nil
}

type scanner interface {
	Scan(dest ...any) error
}

func scanOrganization(row scanner) (domain.Organization, error) {
	var org domain.Organization
	var metadata []byte
	if err := row.Scan(
		&org.OrgID,
		&org.Name,
		&org.Status,
		&org.CreatedAt,
		&org.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.Organization{}, mapPgError(err)
	}
	org.CreatedAt = org.CreatedAt.UTC()
	org.UpdatedAt = org.UpdatedAt.UTC()
	org.Metadata = jsonMap(metadata)
	return org, nil
}

func scanUserAccount(row scanner) (domain.UserAccount, error) {
	var user domain.UserAccount
	var metadata []byte
	if err := row.Scan(
		&user.UserID,
		&user.Email,
		&user.DisplayName,
		&user.Role,
		&user.Status,
		&user.OrgID,
		&user.CreatedAt,
		&user.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.UserAccount{}, mapPgError(err)
	}
	user.CreatedAt = user.CreatedAt.UTC()
	user.UpdatedAt = user.UpdatedAt.UTC()
	user.Metadata = jsonMap(metadata)
	return user, nil
}

func scanBisqueCredential(row scanner) (domain.BisqueCredentialRecord, error) {
	var record domain.BisqueCredentialRecord
	var lastVerifiedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&record.SessionID,
		&record.UserID,
		&record.OrgID,
		&record.RootURL,
		&record.Username,
		&record.PasswordCiphertext,
		&record.PasswordNonce,
		&record.PasswordKeyID,
		&record.PasswordAlgorithm,
		&record.Status,
		&lastVerifiedAt,
		&record.CreatedAt,
		&record.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.BisqueCredentialRecord{}, mapPgError(err)
	}
	if lastVerifiedAt.Valid {
		record.LastVerifiedAt = lastVerifiedAt.Time.UTC()
	}
	record.CreatedAt = record.CreatedAt.UTC()
	record.UpdatedAt = record.UpdatedAt.UTC()
	record.Metadata = jsonMap(metadata)
	return record, nil
}

func scanRunLease(row scanner) (domain.RunLeaseRecord, error) {
	var lease domain.RunLeaseRecord
	if err := row.Scan(
		&lease.RunID,
		&lease.WorkerID,
		&lease.LeaseToken,
		&lease.LeaseExpiresAt,
		&lease.CreatedAt,
		&lease.UpdatedAt,
	); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	lease.LeaseExpiresAt = lease.LeaseExpiresAt.UTC()
	lease.CreatedAt = lease.CreatedAt.UTC()
	lease.UpdatedAt = lease.UpdatedAt.UTC()
	return lease, nil
}

func scanDataAgentJobLeaseRow(row scanner) (domain.DataAgentJobLeaseRecord, error) {
	var lease domain.DataAgentJobLeaseRecord
	if err := row.Scan(
		&lease.JobID,
		&lease.WorkerID,
		&lease.LeaseToken,
		&lease.LeaseExpiresAt,
		&lease.CreatedAt,
		&lease.UpdatedAt,
	); err != nil {
		return domain.DataAgentJobLeaseRecord{}, mapPgError(err)
	}
	lease.LeaseExpiresAt = lease.LeaseExpiresAt.UTC()
	lease.CreatedAt = lease.CreatedAt.UTC()
	lease.UpdatedAt = lease.UpdatedAt.UTC()
	return lease, nil
}

func scanWorkerHeartbeat(row scanner) (domain.WorkerHeartbeatRecord, error) {
	var worker domain.WorkerHeartbeatRecord
	var metadata []byte
	if err := row.Scan(
		&worker.WorkerID,
		&worker.WorkerKind,
		&worker.Status,
		&worker.CurrentRunID,
		&worker.Hostname,
		&worker.Version,
		&worker.StartedAt,
		&worker.LastHeartbeatAt,
		&worker.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.WorkerHeartbeatRecord{}, mapPgError(err)
	}
	worker.StartedAt = worker.StartedAt.UTC()
	worker.LastHeartbeatAt = worker.LastHeartbeatAt.UTC()
	worker.UpdatedAt = worker.UpdatedAt.UTC()
	worker.Metadata = jsonMap(metadata)
	return worker, nil
}

func mapPgError(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" {
		return ErrConflict
	}
	// Foreign-key violation (e.g. a collection parent that vanished between
	// validation and insert) is a caller problem, not a server fault.
	if errors.As(err, &pgErr) && pgErr.Code == "23503" {
		return ErrNotFound
	}
	return err
}

func jsonBytes(value domain.JSONMap) []byte {
	if value == nil {
		value = domain.JSONMap{}
	}
	data, _ := json.Marshal(value)
	return data
}

func jsonMap(data []byte) domain.JSONMap {
	if len(data) == 0 {
		return domain.JSONMap{}
	}
	var value domain.JSONMap
	if err := json.Unmarshal(data, &value); err != nil {
		return domain.JSONMap{}
	}
	return value
}

func nullableText(value string) pgtype.Text {
	if value == "" {
		return pgtype.Text{}
	}
	return pgtype.Text{String: value, Valid: true}
}

func textValue(value pgtype.Text) string {
	if !value.Valid {
		return ""
	}
	return value.String
}

func nullableInt8(value int64) pgtype.Int8 {
	if value == 0 {
		return pgtype.Int8{}
	}
	return pgtype.Int8{Int64: value, Valid: true}
}

func int8Value(value pgtype.Int8) int64 {
	if !value.Valid {
		return 0
	}
	return value.Int64
}

func timestamptz(value time.Time) pgtype.Timestamptz {
	return pgtype.Timestamptz{Time: value.UTC(), Valid: true}
}

func nullableTimestamptz(value time.Time) pgtype.Timestamptz {
	if value.IsZero() {
		return pgtype.Timestamptz{}
	}
	return timestamptz(value)
}

func timeValue(value pgtype.Timestamptz) time.Time {
	if !value.Valid {
		return time.Time{}
	}
	return value.Time.UTC()
}

func timePtr(value pgtype.Timestamptz) *time.Time {
	if !value.Valid {
		return nil
	}
	t := value.Time.UTC()
	return &t
}

func limit32(limit int, fallback int32) int32 {
	if limit <= 0 {
		return fallback
	}
	return int32(limit)
}

func offset32(offset int) int32 {
	if offset <= 0 {
		return 0
	}
	return int32(offset)
}

// ListResourceCollectionShareGrantsForCollection returns the owner's active
// and revoked collection-level grants — the missing half of folder sharing:
// without it a folder share could be created but never seen or undone.
func (s *PostgresStore) ListResourceCollectionShareGrantsForCollection(ctx context.Context, collectionID string, ownerUserID string, ownerOrgID string) ([]domain.ResourceCollectionShareGrantRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT g.grant_id, g.collection_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
       COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
       COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata
FROM control_resource_collection_share_grants g
JOIN control_resource_collections c ON c.collection_id = g.collection_id
WHERE g.collection_id = $1
  AND c.owner_user_id = $2
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $3)
ORDER BY g.created_at DESC`,
		strings.TrimSpace(collectionID),
		strings.TrimSpace(ownerUserID),
		strings.TrimSpace(ownerOrgID),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	grants := []domain.ResourceCollectionShareGrantRecord{}
	for rows.Next() {
		grant, err := scanResourceCollectionShareGrantRow(rows)
		if err != nil {
			return nil, err
		}
		grants = append(grants, grant)
	}
	return grants, rows.Err()
}

// RevokeResourceCollectionShareGrant flips a collection grant to revoked and
// cascades to every inherited per-resource grant carrying its back-pointer —
// one call fully un-shares a folder, however many members it fanned out to.
func (s *PostgresStore) RevokeResourceCollectionShareGrant(ctx context.Context, collectionID string, grantID string, ownerUserID string, ownerOrgID string, revokedAt time.Time) (domain.ResourceCollectionShareGrantRecord, error) {
	if revokedAt.IsZero() {
		revokedAt = domain.Now()
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ResourceCollectionShareGrantRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	grant, err := scanResourceCollectionShareGrantRow(tx.QueryRow(ctx, `
UPDATE control_resource_collection_share_grants g
SET status = 'revoked',
    revoked_at = $5,
    updated_at = $5
FROM control_resource_collections c
WHERE g.grant_id = $2
  AND g.collection_id = $1
  AND g.collection_id = c.collection_id
  AND c.owner_user_id = $3
  AND (COALESCE(c.owner_org_id, '') = '' OR c.owner_org_id = $4)
  AND g.status = 'active'
RETURNING g.grant_id, g.collection_id, g.owner_user_id, COALESCE(g.owner_org_id, ''), COALESCE(g.owner_role, ''),
          COALESCE(g.grantee_user_id, ''), COALESCE(g.grantee_org_id, ''), g.role, g.status,
          COALESCE(g.created_by_user_id, ''), g.created_at, g.updated_at, g.revoked_at, g.metadata`,
		strings.TrimSpace(collectionID),
		strings.TrimSpace(grantID),
		strings.TrimSpace(ownerUserID),
		strings.TrimSpace(ownerOrgID),
		timestamptz(revokedAt),
	))
	if err != nil {
		return domain.ResourceCollectionShareGrantRecord{}, mapPgError(err)
	}
	if _, err := tx.Exec(ctx, `
UPDATE control_resource_share_grants
SET status = 'revoked',
    revoked_at = $2,
    updated_at = $2
WHERE status = 'active'
  AND metadata->>'collection_share_grant_id' = $1`,
		grant.GrantID,
		timestamptz(revokedAt),
	); err != nil {
		return domain.ResourceCollectionShareGrantRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ResourceCollectionShareGrantRecord{}, err
	}
	return grant, nil
}
