package store

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestVerifyPostgresSchemaReportsMissingControlTables(t *testing.T) {
	t.Parallel()

	err := VerifyPostgresSchema(context.Background(), fakeSchemaQuerier{
		presentTables: []string{"control_threads", "control_runs"},
	})

	if err == nil {
		t.Fatalf("VerifyPostgresSchema() error = nil, want missing table error")
	}
	message := err.Error()
	for _, want := range []string{"control_organizations", "control_users", "control_run_events", "control_run_leases", "control_worker_heartbeats", "control_artifacts", "control_resources", "control_resource_search_documents", "control_resource_search_facts", "control_resource_events", "control_resource_collections", "control_resource_collection_share_grants", "control_resource_collection_members", "control_dataset_snapshots", "control_dataset_snapshot_resources", "control_dataset_snapshot_share_grants", "control_dataset_snapshot_events", "control_data_agent_jobs", "control_data_agent_job_resources", "control_data_agent_job_events", "control_data_agent_job_leases", "control_upload_sessions", "control_upload_session_files", "control_upload_session_events", "control_upload_chunks", "control_bisque_credentials"} {
		if !strings.Contains(message, want) {
			t.Fatalf("VerifyPostgresSchema() error = %q, want mention %s", message, want)
		}
	}
}

func TestVerifyPostgresSchemaAcceptsCurrentControlSchema(t *testing.T) {
	t.Parallel()

	err := VerifyPostgresSchema(context.Background(), fakeSchemaQuerier{
		presentTables:  requiredPostgresControlTables,
		calphadCatalog: validCalphadSchemaCatalogFixture(),
	})

	if err != nil {
		t.Fatalf("VerifyPostgresSchema() error = %v, want nil", err)
	}
}

func TestVerifyPostgresSchemaWrapsDatabaseErrors(t *testing.T) {
	t.Parallel()

	err := VerifyPostgresSchema(context.Background(), fakeSchemaQuerier{
		err: errors.New("catalog unavailable"),
	})

	if err == nil {
		t.Fatalf("VerifyPostgresSchema() error = nil, want catalog error")
	}
	if !strings.Contains(err.Error(), "verify postgres schema") {
		t.Fatalf("VerifyPostgresSchema() error = %q, want schema context", err)
	}
}

type fakeSchemaQuerier struct {
	presentTables  []string
	calphadCatalog calphadSchemaCatalog
	err            error
}

func (q fakeSchemaQuerier) QueryRow(_ context.Context, sql string, _ ...any) pgx.Row {
	return fakeSchemaRow{
		presentTables:  q.presentTables,
		calphadCatalog: q.calphadCatalog,
		catalogQuery:   strings.Contains(sql, "calphad_schema_catalog_v1"),
		err:            q.err,
	}
}

type fakeSchemaRow struct {
	presentTables  []string
	calphadCatalog calphadSchemaCatalog
	catalogQuery   bool
	err            error
}

func (r fakeSchemaRow) Scan(dest ...any) error {
	if r.err != nil {
		return r.err
	}
	if r.catalogQuery {
		if len(dest) != 7 {
			return errors.New("expected seven CALPHAD catalog destinations")
		}
		values := [][]string{
			r.calphadCatalog.Columns,
			r.calphadCatalog.Constraints,
			r.calphadCatalog.Indexes,
			r.calphadCatalog.Functions,
			r.calphadCatalog.Triggers,
		}
		for index, value := range values {
			target, ok := dest[index].(*[]string)
			if !ok {
				return errors.New("expected *[]string CALPHAD catalog destination")
			}
			*target = append([]string(nil), value...)
		}
		writerFunctionsExact, writerOK := dest[5].(*bool)
		capacityAccounting, capacityOK := dest[6].(*bool)
		if !writerOK || !capacityOK {
			return errors.New("expected *bool CALPHAD catalog destination")
		}
		*writerFunctionsExact = r.calphadCatalog.WriterFunctionsExact
		*capacityAccounting = r.calphadCatalog.CapacityAccountingValid
		return nil
	}
	target, ok := dest[0].(*[]string)
	if !ok {
		return errors.New("expected *[]string destination")
	}
	*target = append([]string(nil), r.presentTables...)
	return nil
}

func validCalphadSchemaCatalogFixture() calphadSchemaCatalog {
	records := func(fingerprints map[string][]string) []string {
		out := make([]string, 0, len(fingerprints))
		for name, fragments := range fingerprints {
			out = append(out, name+"\n"+strings.Join(fragments, " "))
		}
		return out
	}
	return calphadSchemaCatalog{
		Columns:                 append([]string(nil), requiredCalphadColumns...),
		Constraints:             records(requiredCalphadConstraintFingerprints),
		Indexes:                 records(requiredCalphadIndexFingerprints),
		Functions:               records(requiredCalphadFunctionFingerprints),
		Triggers:                records(requiredCalphadTriggerFingerprints),
		WriterFunctionsExact:    true,
		CapacityAccountingValid: true,
	}
}
