CREATE TABLE IF NOT EXISTS control_resource_purge_tombstones (
  resource_id text PRIMARY KEY,
  purged_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS control_resources_retention_expiry_idx
  ON control_resources(retention_expires_at, resource_id)
  WHERE status = 'deleted' AND retention_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS control_resources_purging_lease_idx
  ON control_resources(updated_at, resource_id)
  WHERE status = 'purging';

CREATE INDEX IF NOT EXISTS control_resources_retention_claim_idx
  ON control_resources(
    (CASE
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..')
        AND btrim(COALESCE(storage_uri, '')) = '' AND btrim(COALESCE(storage_path, '')) <> '' THEN 0
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..')
        AND lower(btrim(COALESCE(storage_uri, ''))) LIKE 'file://%' THEN 1
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..') THEN 2
      ELSE 3
    END),
    retention_expires_at,
    resource_id
  )
  WHERE status = 'deleted' AND retention_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS control_resources_purging_claim_idx
  ON control_resources(
    (CASE
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..')
        AND btrim(COALESCE(storage_uri, '')) = '' AND btrim(COALESCE(storage_path, '')) <> '' THEN 0
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..')
        AND lower(btrim(COALESCE(storage_uri, ''))) LIKE 'file://%' THEN 1
      WHEN resource_id ~ '^[A-Za-z0-9_.:-]+$' AND resource_id NOT IN ('.', '..') THEN 2
      ELSE 3
    END),
    updated_at,
    resource_id
  )
  WHERE status = 'purging';

CREATE INDEX IF NOT EXISTS control_resources_retention_blocked_idx
  ON control_resources(resource_id) INCLUDE (size_bytes)
  WHERE status = 'retention_blocked';

CREATE INDEX IF NOT EXISTS control_resources_lifecycle_fence_idx
  ON control_resources((resource_id COLLATE "C"))
  WHERE status IN ('deleted', 'purging', 'retention_blocked');
