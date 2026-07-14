package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSQLCThreadListContractIncludesPaginationAndCount(t *testing.T) {
	t.Parallel()

	source, err := os.ReadFile("queries.sql")
	if err != nil {
		t.Fatalf("read queries.sql: %v", err)
	}
	generated, err := os.ReadFile("sqlc/queries.sql.go")
	if err != nil {
		t.Fatalf("read generated sqlc queries: %v", err)
	}
	schema, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}

	sourceText := string(source)
	generatedText := string(generated)
	for _, expectation := range []struct {
		name   string
		source string
		gen    string
	}{
		{
			name:   "thread count query",
			source: "-- name: CountThreads :one",
			gen:    "func (q *Queries) CountThreads",
		},
		{
			name:   "status filter",
			source: "WHERE ($1::text = '' OR status = $1)",
			gen:    "WHERE ($1::text = '' OR status = $1)",
		},
		{
			name:   "paged list query",
			source: "LIMIT $2 OFFSET $3",
			gen:    "LIMIT $2 OFFSET $3",
		},
		{
			name:   "list params",
			source: "-- name: ListThreads :many",
			gen:    "type ListThreadsParams struct",
		},
		{
			name:   "tenant thread list query",
			source: "-- name: ListThreadsForUser :many",
			gen:    "func (q *Queries) ListThreadsForUser",
		},
		{
			name:   "tenant run list query",
			source: "-- name: ListRunsForUser :many",
			gen:    "func (q *Queries) ListRunsForUser",
		},
		{
			name:   "tenant artifact lookup query",
			source: "-- name: GetArtifactForUser :one",
			gen:    "func (q *Queries) GetArtifactForUser",
		},
		{
			name:   "resource upsert query",
			source: "-- name: UpsertResource :one",
			gen:    "func (q *Queries) UpsertResource",
		},
		{
			name:   "tenant resource list query",
			source: "-- name: ListResourcesForUser :many",
			gen:    "func (q *Queries) ListResourcesForUser",
		},
		{
			name:   "resource soft delete query",
			source: "-- name: SoftDeleteResourceForUser :one",
			gen:    "func (q *Queries) SoftDeleteResourceForUser",
		},
	} {
		if !strings.Contains(sourceText, expectation.source) {
			t.Fatalf("queries.sql missing %s marker %q", expectation.name, expectation.source)
		}
		if !strings.Contains(generatedText, expectation.gen) {
			t.Fatalf("generated sqlc missing %s marker %q", expectation.name, expectation.gen)
		}
	}
	if !strings.Contains(string(schema), "control_threads_user_status_updated_idx") {
		t.Fatalf("schema.sql missing tenant thread owner/status/update index")
	}
	if !strings.Contains(string(schema), "control_resources_owner_status_created_idx") {
		t.Fatalf("schema.sql missing tenant resource owner/status/create index")
	}
	if !strings.Contains(string(schema), "control_resources_project_status_idx") {
		t.Fatalf("schema.sql missing resource project/status index")
	}
}

func TestGetResourceForUserSQLCContractIncludesPublicShareGrants(t *testing.T) {
	t.Parallel()

	source, err := os.ReadFile("queries.sql")
	if err != nil {
		t.Fatalf("read queries.sql: %v", err)
	}
	generated, err := os.ReadFile("sqlc/queries.sql.go")
	if err != nil {
		t.Fatalf("read generated sqlc queries: %v", err)
	}
	publicPredicate := "COALESCE(g.grantee_user_id, '') = '__public__'"
	for _, contract := range []struct {
		name  string
		text  string
		start string
		end   string
	}{
		{name: "queries.sql", text: string(source), start: "-- name: GetResourceForUser :one", end: "-- name: GetResourceForOwner :one"},
		{name: "generated sqlc", text: string(generated), start: "const getResourceForUser = `", end: "type GetResourceForUserParams struct"},
	} {
		start := strings.Index(contract.text, contract.start)
		if start < 0 {
			t.Fatalf("%s missing %q", contract.name, contract.start)
		}
		endOffset := strings.Index(contract.text[start:], contract.end)
		if endOffset < 0 {
			t.Fatalf("%s missing %q after GetResourceForUser", contract.name, contract.end)
		}
		section := contract.text[start : start+endOffset]
		if !strings.Contains(section, publicPredicate) {
			t.Fatalf("%s GetResourceForUser omits active public grants", contract.name)
		}
	}
}

func TestResourceSearchQueriesUseMetadataValuesNotSerializedJSONKeys(t *testing.T) {
	t.Parallel()

	source, err := os.ReadFile("queries.sql")
	if err != nil {
		t.Fatalf("read queries.sql: %v", err)
	}
	generated, err := os.ReadFile("sqlc/queries.sql.go")
	if err != nil {
		t.Fatalf("read generated sqlc queries: %v", err)
	}
	postgres, err := os.ReadFile("postgres.go")
	if err != nil {
		t.Fatalf("read postgres.go: %v", err)
	}
	schema, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}
	migration, err := os.ReadFile("../../migrations/000001_run_control.up.sql")
	if err != nil {
		t.Fatalf("read migration: %v", err)
	}
	for _, file := range []struct {
		name string
		text string
	}{
		{name: "queries.sql", text: string(source)},
		{name: "sqlc/queries.sql.go", text: string(generated)},
		{name: "postgres.go", text: string(postgres)},
	} {
		if strings.Contains(file.text, "lower(COALESCE(r.metadata::text") {
			t.Fatalf("%s still searches serialized resource metadata JSON keys", file.name)
		}
		if strings.Contains(file.text, "jsonb_path_query(r.metadata") {
			t.Fatalf("%s still scans resource metadata JSON values at query time", file.name)
		}
		if !strings.Contains(file.text, "control_resource_search_documents") {
			t.Fatalf("%s missing durable resource search-document query wiring", file.name)
		}
		if !strings.Contains(file.text, "search_vector @@ plainto_tsquery('simple'") {
			t.Fatalf("%s missing indexable resource search-vector predicate", file.name)
		}
		if !strings.Contains(file.text, "lower(sd.search_text) LIKE") && !strings.Contains(file.text, "lower(COALESCE(sd.search_text") {
			t.Fatalf("%s missing phrase-preserving resource search-text predicate", file.name)
		}
	}
	if !strings.Contains(string(postgres), "control_resource_search_facts") {
		t.Fatalf("postgres.go missing durable resource search-fact candidate lookup")
	}
	for _, file := range []struct {
		name string
		text string
	}{
		{name: "queries.sql", text: string(source)},
		{name: "sqlc/queries.sql.go", text: string(generated)},
	} {
		if !strings.Contains(file.text, "r.resource_id = ANY($16::text[])") {
			t.Fatalf("%s missing indexed fact candidate filter", file.name)
		}
	}
	for _, ddl := range []struct {
		name string
		text string
	}{
		{name: "schema.sql", text: string(schema)},
		{name: "migration", text: string(migration)},
	} {
		for _, want := range []string{
			"CREATE TABLE IF NOT EXISTS control_resource_search_documents",
			"search_vector tsvector NOT NULL",
			"control_resource_search_documents_vector_idx",
			"control_resource_search_documents_owner_status_idx",
			"CREATE TABLE IF NOT EXISTS control_resource_search_facts",
			"fact_number double precision",
			"control_resource_search_facts_number_idx",
			"control_resource_search_facts_text_idx",
			"control_resource_search_facts_owner_number_idx",
			"control_resource_search_facts_owner_text_idx",
		} {
			if !strings.Contains(ddl.text, want) {
				t.Fatalf("%s missing resource search-document DDL %q", ddl.name, want)
			}
		}
	}
}

func TestSchemaDoesNotCarryRedundantSequenceIndexes(t *testing.T) {
	t.Parallel()

	schema, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}
	migration, err := os.ReadFile("../../migrations/000001_run_control.up.sql")
	if err != nil {
		t.Fatalf("read migration: %v", err)
	}
	for _, ddl := range []struct {
		name string
		text string
	}{
		{name: "schema.sql", text: string(schema)},
		{name: "migration", text: string(migration)},
	} {
		for _, redundant := range []string{
			"CREATE INDEX IF NOT EXISTS control_run_events_run_sequence_idx",
			"CREATE INDEX IF NOT EXISTS control_run_events_run_event_idx",
			"CREATE INDEX IF NOT EXISTS control_data_agent_job_events_job_sequence_idx",
		} {
			if strings.Contains(ddl.text, redundant) {
				t.Fatalf("%s still creates redundant hot-path index %q", ddl.name, redundant)
			}
		}
	}
}

func TestPostgresDeploymentEnablesQueryObservability(t *testing.T) {
	t.Parallel()

	repoRoot := filepath.Join("..", "..", "..", "..")
	files := []struct {
		name string
		path string
	}{
		{name: "docker-compose.yml", path: filepath.Join(repoRoot, "docker-compose.yml")},
		{name: "docker-compose.postgres.yml", path: filepath.Join(repoRoot, "docker-compose.postgres.yml")},
		{name: "ultra-postgres.service", path: filepath.Join(repoRoot, "deploy", "systemd", "ultra-postgres.service")},
	}
	for _, file := range files {
		raw, err := os.ReadFile(file.path)
		if err != nil {
			t.Fatalf("read %s: %v", file.name, err)
		}
		text := string(raw)
		for _, want := range []string{
			"shared_preload_libraries=pg_stat_statements",
			"track_io_timing=on",
			"track_wal_io_timing=on",
			"log_temp_files=",
		} {
			if !strings.Contains(text, want) {
				t.Fatalf("%s missing Postgres observability setting %q", file.name, want)
			}
		}
		if !strings.Contains(text, "wal_compression=lz4") && !strings.Contains(text, "ULTRA_PG_WAL_COMPRESSION:-lz4") {
			t.Fatalf("%s missing lz4 WAL compression default", file.name)
		}
		if !strings.Contains(text, "default_toast_compression=lz4") && !strings.Contains(text, "ULTRA_PG_DEFAULT_TOAST_COMPRESSION:-lz4") {
			t.Fatalf("%s missing lz4 TOAST compression default", file.name)
		}
	}

	schema, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("read schema.sql: %v", err)
	}
	if !strings.Contains(string(schema), "CREATE EXTENSION IF NOT EXISTS pg_stat_statements") {
		t.Fatalf("schema.sql missing pg_stat_statements extension creation")
	}
}
