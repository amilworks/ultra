#!/usr/bin/env bash
set -euo pipefail

create_role_if_missing() {
  local role_name="$1"
  local role_password="$2"

  psql --username "$POSTGRES_USER" --dbname postgres <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${role_name}') THEN
    CREATE ROLE "${role_name}" LOGIN PASSWORD '${role_password}';
  END IF;
END
\$\$;
SQL
}

create_db_if_missing() {
  local db_name="$1"
  local db_owner="$2"
  local exists

  exists="$(psql --username "$POSTGRES_USER" --dbname postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${db_name}'")"
  if [ "$exists" != "1" ]; then
    psql --username "$POSTGRES_USER" --dbname postgres -c "CREATE DATABASE \"${db_name}\" OWNER \"${db_owner}\""
  fi
}

create_role_if_missing "${BISQUE_DB_USER:-bisque}" "${BISQUE_DB_PASSWORD:?BISQUE_DB_PASSWORD is required}"
create_role_if_missing "${KEYCLOAK_DB_USERNAME:-keycloak}" "${KEYCLOAK_DB_PASSWORD:?KEYCLOAK_DB_PASSWORD is required}"
create_role_if_missing "${ULTRA_DB_USERNAME:-ultra}" "${ULTRA_DB_PASSWORD:?ULTRA_DB_PASSWORD is required}"

create_db_if_missing "${BISQUE_DB_NAME:-bisque}" "${BISQUE_DB_USER:-bisque}"
create_db_if_missing "${KEYCLOAK_DB_DATABASE:-keycloak}" "${KEYCLOAK_DB_USERNAME:-keycloak}"
create_db_if_missing "${ULTRA_DB_DATABASE:-ultra}" "${ULTRA_DB_USERNAME:-ultra}"
