package store

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// The in-memory store has no foreign keys, so the hard-delete tests there can
// never catch a referencing table that Postgres would refuse to orphan. This
// asks the real database instead.
//
// Two failure shapes hide behind a missing entry in the sweep list:
//
//   - a table with NO foreign key (control_run_token_usage*) is silently
//     orphaned — the delete "succeeds" and leaves rows behind;
//   - a table whose foreign key is declared NO ACTION (control_run_specs,
//     control_calphad_validation_events) makes Postgres RAISE on the parent
//     delete, so the whole transaction aborts and the user's delete fails.
//
// The second kind is why this test exists. Neither of those tables appears in
// schema.sql, so no amount of reading the checked-in schema would have found
// them; they were discovered by querying a running database. Anything that
// references control_runs or control_threads with confdeltype <> 'c' must be
// handled explicitly in HardDeleteThreadForUser.
func TestHardDeleteSweepsEveryNonCascadingReferencer(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	// HardDeleteThreadForUser discovers these from the catalog at runtime, so
	// this test does not police a hand-maintained list. It asserts the weaker but
	// more useful property: every non-cascading referencer is reachable by the
	// column the sweep actually uses (thread_id for thread references, run_id for
	// run references). A composite or oddly-named key would slip past the sweep,
	// and that is what this catches.

	rows, err := pool.Query(ctx, `
SELECT src.relname, a.attname, tgt.relname
FROM pg_constraint c
JOIN pg_class src ON src.oid = c.conrelid
JOIN pg_class tgt ON tgt.oid = c.confrelid
JOIN unnest(c.conkey) AS k(attnum) ON true
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
WHERE c.contype = 'f'
  AND tgt.relname IN ('control_runs', 'control_threads')
  AND c.confdeltype IN ('a', 'r')
ORDER BY src.relname`)
	if err != nil {
		t.Fatalf("query foreign keys: %v", err)
	}
	defer rows.Close()

	seen := 0
	for rows.Next() {
		var child, column, parent string
		if err := rows.Scan(&child, &column, &parent); err != nil {
			t.Fatalf("scan: %v", err)
		}
		seen++
		want := "run_id"
		if parent == "control_threads" {
			want = "thread_id"
		}
		if column != want {
			t.Errorf(
				"%s references %s via %q, but the hard-delete sweep deletes by %q; "+
					"a conversation delete will abort with a foreign-key violation",
				child, parent, column, want,
			)
		}
	}
	t.Logf("checked %d non-cascading referencers of control_runs/control_threads", seen)
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate: %v", err)
	}
}
