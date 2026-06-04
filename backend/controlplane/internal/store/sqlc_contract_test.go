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
	} {
		if !strings.Contains(sourceText, expectation.source) {
			t.Fatalf("queries.sql missing %s marker %q", expectation.name, expectation.source)
		}
		if !strings.Contains(generatedText, expectation.gen) {
			t.Fatalf("generated sqlc missing %s marker %q", expectation.name, expectation.gen)
		}
	}
}
