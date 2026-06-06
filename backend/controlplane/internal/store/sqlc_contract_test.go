package store

import (
	"os"
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
}
