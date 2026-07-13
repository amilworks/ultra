package store

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestApplyPostgresSchemaOnFreshDatabase applies the consolidated schema.sql
// to a brand-new, empty database. The routine store Postgres tests reuse an
// already-migrated database, which can hide ordering defects between dependent
// DDL statements. Only a truly fresh apply exercises the statement order end
// to end.
func TestApplyPostgresSchemaOnFreshDatabase(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if strings.TrimSpace(dsn) == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	admin, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New(admin): %v", err)
	}
	defer admin.Close()

	name := fmt.Sprintf("ultra_fresh_apply_%d", time.Now().UnixNano())
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Fatalf("create fresh database: %v", err)
	}
	defer func() {
		dropCtx, dropCancel := context.WithTimeout(context.Background(), time.Minute)
		defer dropCancel()
		if _, err := admin.Exec(dropCtx, "DROP DATABASE IF EXISTS "+name+" WITH (FORCE)"); err != nil {
			t.Errorf("drop fresh database %s: %v", name, err)
		}
	}()

	parsed, err := url.Parse(dsn)
	if err != nil {
		t.Fatalf("parse test DSN: %v", err)
	}
	parsed.Path = "/" + name
	fresh, err := pgxpool.New(ctx, parsed.String())
	if err != nil {
		t.Fatalf("pgxpool.New(fresh): %v", err)
	}
	defer fresh.Close()

	if err := ApplyPostgresSchema(ctx, fresh); err != nil {
		t.Fatalf("ApplyPostgresSchema on a fresh database: %v", err)
	}
	// Applying again must stay idempotent: the reapply bridges in schema.sql
	// are exercised with the schema now fully present.
	if err := ApplyPostgresSchema(ctx, fresh); err != nil {
		t.Fatalf("ApplyPostgresSchema reapply: %v", err)
	}
}
