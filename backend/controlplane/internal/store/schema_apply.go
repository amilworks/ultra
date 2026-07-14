package store

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

//go:embed schema.sql
var postgresControlSchemaSQL string

type schemaExecer interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
}

// GrantPostgresServingPrivileges gives the non-owner application role normal
// DML access while making CALPHAD persistence SELECT plus exact writer
// EXECUTE only. The migration role remains the schema/function owner.
func GrantPostgresServingPrivileges(ctx context.Context, db schemaExecer, servingRole string) error {
	servingRole = strings.TrimSpace(servingRole)
	if servingRole == "" {
		return fmt.Errorf("grant postgres serving privileges: serving role is required")
	}
	role := pgx.Identifier{servingRole}.Sanitize()
	roleLiteral := "'" + strings.ReplaceAll(servingRole, "'", "''") + "'"
	script := fmt.Sprintf(`
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM %[1]s;
GRANT USAGE ON SCHEMA public TO %[1]s;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %[1]s;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO %[1]s;
REVOKE ALL PRIVILEGES ON TABLE
  control_calphad_revisions,
  control_calphad_input_blobs,
  control_calphad_evidence_blobs,
  control_calphad_validation_events,
  control_calphad_tenant_capacity
FROM PUBLIC, %[1]s;
GRANT SELECT ON TABLE
  control_calphad_revisions,
  control_calphad_input_blobs,
  control_calphad_evidence_blobs,
  control_calphad_validation_events,
  control_calphad_tenant_capacity
TO %[1]s;

DO $calphad_writer_acl$
DECLARE
  routine record;
BEGIN
  FOR routine IN
    SELECT namespace.nspname AS schema_name, procedure.proname AS function_name,
           pg_get_function_identity_arguments(procedure.oid) AS identity_arguments
    FROM pg_proc procedure
    JOIN pg_namespace namespace ON namespace.oid=procedure.pronamespace
    WHERE namespace.nspname='public'
      AND procedure.proname IN ('ultra_create_calphad_revision_v1',
                                'ultra_append_calphad_validation_v1')
  LOOP
    EXECUTE format(
      'REVOKE ALL ON FUNCTION %%I.%%I(%%s) FROM PUBLIC, %%I',
      routine.schema_name, routine.function_name, routine.identity_arguments, %[2]s
    );
  END LOOP;
END;
$calphad_writer_acl$;

REVOKE ALL ON FUNCTION public.ultra_create_calphad_revision_v1(
  text, text, text, text, text, bigint, text, double precision,
  double precision, bytea, jsonb
) FROM PUBLIC, %[1]s;
REVOKE ALL ON FUNCTION public.ultra_append_calphad_validation_v1(
  text, text, text, text, bigint, text, jsonb, double precision,
  double precision, text, text, text, text, text, text, text, text,
  text, bigint, bytea, text, text, text, text, text, text, jsonb
) FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_revision_parent() FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_validation_run_authority() FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_pressure_binding() FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_input_retention() FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage() FROM PUBLIC, %[1]s;
REVOKE EXECUTE ON FUNCTION public.ultra_reject_calphad_ledger_mutation() FROM PUBLIC, %[1]s;

GRANT EXECUTE ON FUNCTION public.ultra_create_calphad_revision_v1(
  text, text, text, text, text, bigint, text, double precision,
  double precision, bytea, jsonb
) TO %[1]s;
GRANT EXECUTE ON FUNCTION public.ultra_append_calphad_validation_v1(
  text, text, text, text, bigint, text, jsonb, double precision,
  double precision, text, text, text, text, text, text, text, text,
  text, bigint, bytea, text, text, text, text, text, text, jsonb
) TO %[1]s;
`, role, roleLiteral)
	if _, err := db.Exec(ctx, script); err != nil {
		return fmt.Errorf("grant postgres serving privileges: %w", err)
	}
	return nil
}

func ApplyPostgresSchema(ctx context.Context, db schemaExecer) error {
	// The whole schema runs as one implicit transaction. Concurrent appliers
	// (rolling deploys running `migrate` from several replicas, or parallel
	// test packages) would otherwise interleave DDL lock acquisition and can
	// deadlock; the advisory lock serializes them.
	script := "SELECT pg_advisory_xact_lock(hashtext('ultra_control_schema_apply')::bigint);\n" + postgresControlSchemaSQL
	if _, err := db.Exec(ctx, script); err != nil {
		return fmt.Errorf("apply postgres schema: %w", err)
	}
	return nil
}
